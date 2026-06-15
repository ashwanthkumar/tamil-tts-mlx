"""Python SDK: non-AR Tamil TTS from the two ONNX graphs (CPU, all platforms, single forward).

    uv run python -m tamiltts.mlx.onnx_infer_ns -m models/tamil_ns --text "வணக்கம்" -o out.wav

Pipeline: enc_dur.onnx -> durations -> integer length-regulate (host) -> decoder.onnx -> Griffin-Lim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

BOS_ID, EOS_ID = 1, 2


class TamilNSTTS:
    def __init__(self, prefix: str, vocoder: str | None = "models/hifigan.onnx"):
        self.enc = ort.InferenceSession(prefix + ".enc_dur.onnx", providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(prefix + ".decoder.onnx", providers=["CPUExecutionProvider"])
        meta = json.loads(Path(prefix + ".tokenizer.json").read_text(encoding="utf-8"))
        self.vocab = meta["vocab"]
        self.mel_mean = np.array(meta["mel_mean"], np.float32)
        self.mel_std = np.array(meta["mel_std"], np.float32)
        self.a = meta["audio"]
        # neural vocoder (HiFi-GAN) if available -> natural audio; else Griffin-Lim
        self.voc = ort.InferenceSession(vocoder, providers=["CPUExecutionProvider"]) \
            if vocoder and Path(vocoder).exists() else None

    def encode_text(self, text):
        return np.array([[BOS_ID] + [self.vocab[c] for c in text if c in self.vocab] + [EOS_ID]], np.int64)

    def synth_mel(self, text, speed=1.0):
        tokens = self.encode_text(text)
        enc, log_dur = self.enc.run(None, {"tokens": tokens})
        Tt = tokens.shape[1]
        dur = np.maximum(np.round((np.exp(log_dur[0]) - 1.0) / speed), 0).astype(np.int64)
        if dur.sum() == 0:
            dur[:] = 1
        expand = np.repeat(np.arange(Tt, dtype=np.int64), dur)[None, :]
        mel = self.dec.run(None, {"enc": enc, "expand_idx": expand})[0][0]
        return mel * self.mel_std + self.mel_mean

    def synth(self, text, speed=1.0):
        a = self.a
        logmel = self.synth_mel(text, speed)               # (T, n_mels) denormalized log-mel
        if self.voc is not None:                            # HiFi-GAN: mel (1, n_mels, T) -> wav
            wav = self.voc.run(None, {"mel": logmel.T[None].astype(np.float32)})[0][0, 0]
        else:                                               # Griffin-Lim fallback
            import librosa
            mel = np.exp(logmel.T)
            S = librosa.feature.inverse.mel_to_stft(mel, sr=a["sr"], n_fft=a["n_fft"], power=1.0,
                                                     fmin=a["fmin"], fmax=a["fmax"])
            wav = librosa.griffinlim(S, n_iter=60, hop_length=a["hop"], win_length=a["win"]).astype(np.float32)
        p = np.abs(wav).max()
        return wav * (0.95 / p) if p > 1e-6 else wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="models/tamil_ns")
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--out", default="ns_onnx_out.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    tts = TamilNSTTS(args.model)
    wav = tts.synth(args.text, args.speed)
    sf.write(args.out, wav, tts.a["sr"])
    print(f"wrote {args.out} ({len(wav)/tts.a['sr']:.2f}s)")


if __name__ == "__main__":
    main()
