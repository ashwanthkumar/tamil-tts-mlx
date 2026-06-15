"""Inference for the non-AR FastTTS (single forward pass, no AR loop).

encode text -> predict durations -> length-regulate -> non-causal decode -> mel -> HiFi-GAN
(falls back to Griffin-Lim if models/hifigan.onnx is absent).

    uv run python -m tamiltts.mlx.infer_ns --run runs_mlx_ns/tamil_ns --text "வணக்கம்" -o out.wav
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from .audio import SR, mel_to_wav
from .dataset import TTSData
from .model import key_pad_mask
from .model_ns import NSConfig, FastTTS, gather_expand
from .normalize import normalize

# HiFi-GAN ONNX vocoder (natural audio); falls back to Griffin-Lim if absent. Loaded once.
_VOC = "unset"


def _vocoder(path: str = "models/hifigan.onnx"):
    global _VOC
    if _VOC == "unset":
        try:
            import onnxruntime as ort
            _VOC = ort.InferenceSession(path, providers=["CPUExecutionProvider"]) if Path(path).exists() else None
        except Exception:
            _VOC = None
    return _VOC


def vocode(logmel: np.ndarray) -> np.ndarray:
    """Denormalized (T, 80) log-mel -> waveform via HiFi-GAN ONNX, else Griffin-Lim."""
    voc = _vocoder()
    if voc is not None:
        return voc.run(None, {"mel": logmel.T[None].astype(np.float32)})[0][0, 0]
    return mel_to_wav(logmel)


def load_model(run_dir: Path) -> FastTTS:
    cfg = json.loads((run_dir / "config.json").read_text())["cfg"]
    model = FastTTS(NSConfig(**cfg))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(mx.load(str(run_dir / "latest.safetensors")).items())))
    mx.eval(model.parameters()); model.eval()
    return model


def predict_durations(model, enc, speed: float = 1.0) -> np.ndarray:
    logdur = np.array(model.dur(enc))[0]              # (Tt,)
    dur = np.maximum(np.round((np.exp(logdur) - 1.0) / speed), 0).astype(np.int32)
    return dur


def generate(model, data: TTSData, text: str, speed: float = 1.0, max_total: int = 4000):
    tok_ids = data.encode_text(normalize(text))
    tok = mx.array([tok_ids], dtype=mx.int32)
    Tt = tok.shape[1]
    src = key_pad_mask(mx.array([Tt]), Tt)
    enc = model.encode(tok, src)
    dur = predict_durations(model, enc, speed)
    if dur.sum() == 0:
        dur[:] = 1
    expand = np.repeat(np.arange(Tt, dtype=np.int32), dur)[None, :max_total]
    Tm = expand.shape[1]
    dec_mask = key_pad_mask(mx.array([Tm]), Tm)
    expanded = gather_expand(enc, mx.array(expand))
    _, mel_post = model.decode(expanded, dec_mask)
    mx.eval(mel_post)
    mel = np.array(mel_post[0]) * data.mel_std + data.mel_mean
    return vocode(mel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--out", type=Path, default=Path("ns_out.wav"))
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    data = TTSData(args.data)
    model = load_model(args.run)
    wav = generate(model, data, args.text, args.speed)
    sf.write(str(args.out), wav, SR)
    print(f"wrote {args.out} ({len(wav)/SR:.2f}s)")


if __name__ == "__main__":
    main()
