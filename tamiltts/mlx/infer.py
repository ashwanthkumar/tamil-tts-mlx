"""Generate audio from a trained MLX TransformerTTS checkpoint.

    uv run python -m tamiltts.mlx.infer --run runs_mlx/tamil_mlx --text "வணக்கம்" -o out.wav
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
from .model import TTSConfig, TransformerTTS, key_pad_mask, causal_mask


def load_model(run_dir: Path):
    cfg_d = json.loads((run_dir / "config.json").read_text())["cfg"]
    cfg = TTSConfig(**cfg_d)
    model = TransformerTTS(cfg)
    ckpt = run_dir / "latest.safetensors"
    weights = mx.load(str(ckpt))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model


def generate(model, data: TTSData, text: str, max_frames: int = 800, stop_thresh: float = 0.5):
    tok = mx.array([data.encode_text(text)], dtype=mx.int32)
    Tt = tok.shape[1]
    src_mask = key_pad_mask(mx.array([Tt]), Tt)
    mem = model.encode(tok, src_mask)

    mel_in = mx.zeros((1, 1, model.c.n_mels))  # go frame (normalized space)
    frames = []
    for _ in range(max_frames):
        Tm = mel_in.shape[1]
        self_mask = causal_mask(Tm)
        _, mel_post, stop = model.decode(mel_in, mem, self_mask, src_mask)
        last = mel_post[:, -1:, :]
        frames.append(last)
        mel_in = mx.concatenate([mel_in, last], axis=1)
        if float(mx.sigmoid(stop[0, -1])) > stop_thresh:
            break
    mx.eval(frames)
    mel_norm = mx.concatenate(frames, axis=1)[0]  # (T, n_mels) normalized
    mel = np.array(mel_norm) * data.mel_std + data.mel_mean
    return mel_to_wav(mel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--out", type=Path, default=Path("mlx_out.wav"))
    ap.add_argument("--max_frames", type=int, default=800)
    args = ap.parse_args()

    data = TTSData(args.data)
    model = load_model(args.run)
    wav = generate(model, data, args.text, args.max_frames)
    sf.write(str(args.out), wav, SR)
    print(f"wrote {args.out} ({len(wav)/SR:.2f}s)")


if __name__ == "__main__":
    main()
