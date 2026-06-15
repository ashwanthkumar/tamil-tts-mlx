"""Python SDK: generate Tamil speech from the exported ONNX model (CPU, all platforms).

    uv run python -m tamiltts.mlx.onnx_infer -m models/tamil_mlx.onnx \
        --text "வணக்கம், இது தமிழ் பேச்சு." -o out.wav

Self-contained: needs only onnxruntime + numpy + librosa (Griffin-Lim). The ONNX graph is a
single forward (tokens, mel_in) -> (mel, mel_post, stop); we run the short autoregressive loop here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

BOS_ID, EOS_ID = 1, 2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TamilTTS:
    def __init__(self, onnx_path: str, tokenizer_path: str | None = None):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        tok = tokenizer_path or str(Path(onnx_path).with_suffix(".tokenizer.json"))
        meta = json.loads(Path(tok).read_text(encoding="utf-8"))
        self.vocab = meta["vocab"]
        self.mel_mean = np.array(meta["mel_mean"], dtype=np.float32)
        self.mel_std = np.array(meta["mel_std"], dtype=np.float32)
        self.a = meta["audio"]
        self.n_mels = self.a["n_mels"]

    def encode_text(self, text: str) -> np.ndarray:
        ids = [BOS_ID] + [self.vocab[c] for c in text if c in self.vocab] + [EOS_ID]
        return np.array([ids], dtype=np.int64)

    def synth_mel(self, text: str, max_frames: int = 800, stop_thresh: float = 0.5) -> np.ndarray:
        tokens = self.encode_text(text)
        mel_in = np.zeros((1, 1, self.n_mels), dtype=np.float32)  # go frame
        frames = []
        for _ in range(max_frames):
            _, mel_post, stop = self.sess.run(None, {"tokens": tokens, "mel_in": mel_in})
            last = mel_post[:, -1:, :]
            frames.append(last)
            mel_in = np.concatenate([mel_in, last], axis=1)
            if _sigmoid(stop[0, -1]) > stop_thresh:
                break
        mel_norm = np.concatenate(frames, axis=1)[0]
        return mel_norm * self.mel_std + self.mel_mean  # denormalized log-mel (T, n_mels)

    def synth(self, text: str, **kw) -> np.ndarray:
        import librosa
        a = self.a
        logmel = self.synth_mel(text, **kw)
        mel = np.exp(logmel.T)
        S = librosa.feature.inverse.mel_to_stft(mel, sr=a["sr"], n_fft=a["n_fft"], power=1.0,
                                                 fmin=a["fmin"], fmax=a["fmax"])
        return librosa.griffinlim(S, n_iter=60, hop_length=a["hop"], win_length=a["win"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="models/tamil_mlx.onnx")
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--out", default="mlx_onnx_out.wav")
    ap.add_argument("--max_frames", type=int, default=800)
    args = ap.parse_args()
    tts = TamilTTS(args.model)
    wav = tts.synth(args.text, max_frames=args.max_frames)
    sf.write(args.out, wav, tts.a["sr"])
    print(f"wrote {args.out} ({len(wav)/tts.a['sr']:.2f}s)")


if __name__ == "__main__":
    main()
