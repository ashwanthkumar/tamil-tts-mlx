"""Pronunciation verification: synthesize the full Tamil consonant inventory and the உயிர்மெய் grid,
logged to TensorBoard so each can be checked by ear.

  - consonant/<c>  : each of the 18 consonants in isolation (base 'a' form)
  - row/<c>        : each consonant across all 12 vowels (க கா கி கீ ... — catches vowel-sign issues)

    uv run python -m tamiltts.mlx.verify_inventory --run runs_mlx_ns/tamil_ns2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .audio import SR
from .dataset import TTSData
from .infer_ns import load_model, generate

# the 18 Tamil consonants (மெய்), shown in base உயிர்மெய் form
CONS = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"]
# vowel signs: a, ā, i, ī, u, ū, e, ē, ai, o, ō, au
VSIGN = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]


def norm(w):
    p = float(np.abs(w).max())
    return w * (0.95 / p) if p > 1e-6 else w


def _latest_step(run: Path) -> int:
    sf = run / "latest_state.json"
    return int(json.loads(sf.read_text()).get("step", 0)) if sf.exists() else 0


def _log_inventory(writer, data, model, step):
    for c in CONS:
        writer.add_audio(f"consonant/{c}", norm(generate(model, data, c)), step, sample_rate=SR)
        row = " ".join(c + v for v in VSIGN)
        writer.add_audio(f"row/{c}", norm(generate(model, data, row)), step, sample_rate=SR)
        writer.add_text(f"text/row_{c}", row, step)
    writer.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs_mlx_ns/tamil_ns2"))
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--watch", action="store_true", help="poll the latest checkpoint and refresh each time it advances")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter

    data = TTSData(args.data)
    writer = SummaryWriter(str(args.run / "verify"))

    if not args.watch:
        step = _latest_step(args.run)
        _log_inventory(writer, data, load_model(args.run), step)
        print(f"[verify] logged {len(CONS)} consonants + {len(CONS)} vowel-rows at step {step}", flush=True)
        return

    print(f"[verify] watching {args.run} (refresh every checkpoint) -> {args.run/'verify'}", flush=True)
    last = -1
    while True:
        step = _latest_step(args.run)
        if step != last:
            try:
                _log_inventory(writer, data, load_model(args.run), step)
                print(f"[verify] logged inventory at step {step}", flush=True)
                last = step
            except Exception as e:
                print(f"[verify] skip step {step}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
