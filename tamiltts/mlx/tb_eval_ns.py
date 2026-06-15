"""Live TB quality tracking for the non-AR run: generated audio + pred/target mel images + text.

Polls the latest non-AR checkpoint and logs to <run>/tb_eval (own run dir, so it never collides
with the training scalars in <run>/tb). Same pattern as tb_eval.py but for FastTTS.

    uv run python -m tamiltts.mlx.tb_eval_ns --run runs_mlx_ns/tamil_ns --interval 120
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
from .infer_ns import load_model, generate
from .model import key_pad_mask

SAMPLE_TEXTS = ["வணக்கம், இது தமிழ் பேச்சு.", "எப்படி இருக்கிறீர்கள்?"]


def latest_step(run_dir: Path):
    ck = glob.glob(str(run_dir / "ckpt_*.safetensors"))
    return max((int(re.search(r"ckpt_(\d+)", c).group(1)) for c in ck), default=None)


def mel_image(mel):
    m = mel.T
    lo, hi = float(m.min()), float(m.max())
    return ((m - lo) / (hi - lo + 1e-8))[::-1].copy()


def teacher_forced(model, data, item):
    b = data._collate_ns([item])
    tok = mx.array(b["tok"]); expand = mx.array(b["expand_idx"])
    Tt, Tm = tok.shape[1], expand.shape[1]
    src = key_pad_mask(mx.array(b["tlen"]), Tt); dec = key_pad_mask(mx.array(b["mlen"]), Tm)
    _, mel_post, _ = model(tok, src, expand, dec)
    n = int(b["mlen"][0])
    return np.array(mel_post[0])[:n], b["mel"][0][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    from .audio import SR

    data = TTSData(args.data)
    writer = SummaryWriter(str(args.run / "tb_eval"))
    print(f"[tb_eval_ns] -> {args.run/'tb_eval'} every {args.interval}s", flush=True)
    # pick a val item that has durations for the teacher-forced mel image
    tf_item = next((it for it in data.val if data.has_durations(it["id"])), None)

    last = -1
    while True:
        step = latest_step(args.run)
        if step is not None and step != last:
            try:
                model = load_model(args.run)
                for i, txt in enumerate(SAMPLE_TEXTS):
                    wav = generate(model, data, txt)
                    peak = np.abs(wav).max()
                    if peak > 1e-6:
                        wav = wav * (0.95 / peak)
                    writer.add_audio(f"audio/sample_{i}", wav, step, sample_rate=SR)
                    writer.add_text(f"text/sample_{i}", txt, step)
                if tf_item is not None:
                    pred, tgt = teacher_forced(model, data, tf_item)
                    writer.add_image("mel/pred", mel_image(pred), step, dataformats="HW")
                    writer.add_image("mel/target", mel_image(tgt), step, dataformats="HW")
                writer.flush()
                print(f"[tb_eval_ns] logged step {step}", flush=True)
                last = step
            except Exception as e:
                print(f"[tb_eval_ns] skip step {step}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
