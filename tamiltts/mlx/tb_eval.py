"""Live quality tracking in TensorBoard for the MLX run — audio + mel images.

Runs alongside training (does NOT touch it): polls the newest checkpoint, and for each new
step writes to the SAME TB logdir:
  - audio/sample_*  : generated speech you can play in the TB "Audio" tab
  - mel/pred,target : teacher-forced predicted vs ground-truth mel (TB "Images" tab)

    uv run python -m tamiltts.mlx.tb_eval --run runs_mlx/tamil_mlx --data data/mlx --interval 180
"""
from __future__ import annotations

import argparse
import glob
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .dataset import TTSData
from .infer import load_model, generate
from .model import key_pad_mask, causal_mask

SAMPLE_TEXTS = [
    "வணக்கம், இது தமிழ் பேச்சு.",
    "எப்படி இருக்கிறீர்கள்?",
]


def latest_step(run_dir: Path):
    ckpts = glob.glob(str(run_dir / "ckpt_*.safetensors"))
    if not ckpts:
        return None
    steps = [int(re.search(r"ckpt_(\d+)", c).group(1)) for c in ckpts]
    return max(steps)


def mel_image(mel: np.ndarray) -> np.ndarray:
    """(T, n_mels) -> (n_mels, T) in [0,1], low freq at bottom, for add_image dataformats='HW'."""
    m = mel.T
    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / (hi - lo + 1e-8)
    return m[::-1].copy()


def teacher_forced_mel(model, data: TTSData, item):
    b = data._collate([item])
    tok = mx.array(b["tok"]); mel_in = mx.array(b["mel_in"])
    Tt, Tm = tok.shape[1], mel_in.shape[1]
    src = key_pad_mask(mx.array(b["tlen"]), Tt)
    self_m = causal_mask(Tm) + key_pad_mask(mx.array(b["mlen"]), Tm)
    _, mel_post, _ = model(tok, src, mel_in, self_m, src)
    n = int(b["mlen"][0])
    return np.array(mel_post[0])[:n], b["mel"][0][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--max_frames", type=int, default=400)
    args = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter
    from .audio import SR

    data = TTSData(args.data)
    # IMPORTANT: separate run dir from training's tb/ — sharing it makes TensorBoard purge the
    # training scalars (eval logs lower step numbers -> looks like a restart -> step-regression purge).
    writer = SummaryWriter(str(args.run / "tb_eval"))
    print(f"[tb_eval] writing audio+mel to {args.run/'tb_eval'} every {args.interval}s", flush=True)

    last = -1
    while True:
        step = latest_step(args.run)
        if step is not None and step != last:
            try:
                model = load_model(args.run)
                legend = ["| # | text (audio/sample_N) |", "|---|---|"]
                for i, txt in enumerate(SAMPLE_TEXTS):
                    wav = generate(model, data, txt, max_frames=args.max_frames)
                    peak = np.abs(wav).max()
                    if peak > 1e-6:
                        wav = wav * (0.95 / peak)  # peak-normalize so quiet early samples are audible
                    writer.add_audio(f"audio/sample_{i}", wav, step, sample_rate=SR)
                    # show the sentence next to the audio (TB "Text" tab), same index
                    writer.add_text(f"text/sample_{i}", txt, step)
                    legend.append(f"| {i} | {txt} |")
                writer.add_text("text/legend", "\n".join(legend), step)
                pred, tgt = teacher_forced_mel(model, data, data.val[0])
                writer.add_image("mel/pred", mel_image(pred), step, dataformats="HW")
                writer.add_image("mel/target", mel_image(tgt), step, dataformats="HW")
                writer.flush()
                print(f"[tb_eval] logged audio+mel at step {step}", flush=True)
                last = step
            except Exception as e:
                print(f"[tb_eval] skip step {step}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
