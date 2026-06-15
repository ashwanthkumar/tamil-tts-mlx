"""Pronunciation verification: synthesize the full Tamil consonant inventory and the உயிர்மெய் grid,
logged to TensorBoard so each can be checked by ear.

  - consonant/<c>  : each of the 18 consonants in isolation (base 'a' form)
  - row/<c>        : each consonant across all 12 vowels (க கா கி கீ ... — catches vowel-sign issues)

    uv run python -m tamiltts.mlx.verify_inventory --run runs_mlx_ns/tamil_ns2
"""
from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs_mlx_ns/tamil_ns2"))
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    args = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter

    data = TTSData(args.data)
    model = load_model(args.run)
    step = 0
    sf = args.run / "latest_state.json"
    if sf.exists():
        step = int(json.loads(sf.read_text()).get("step", 0))
    writer = SummaryWriter(str(args.run / "verify"))
    print(f"[verify] model step {step}; logging consonant inventory + vowel grid to {args.run/'verify'}", flush=True)

    for c in CONS:
        writer.add_audio(f"consonant/{c}", norm(generate(model, data, c)), step, sample_rate=SR)
        row = " ".join(c + v for v in VSIGN)
        writer.add_audio(f"row/{c}", norm(generate(model, data, row)), step, sample_rate=SR)
        writer.add_text(f"text/row_{c}", row, step)
    writer.flush()
    print(f"[verify] done: {len(CONS)} consonants + {len(CONS)} vowel-rows at step {step}", flush=True)


if __name__ == "__main__":
    main()
