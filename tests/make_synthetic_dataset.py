"""Generate a tiny synthetic dataset to validate TRAINING MECHANICS (not voice quality).

Real Tamil transcripts + synthetic audio (tones/noise). Lets us exercise the full Coqui VITS
pipeline — phonemization, dataloader, MPS forward/backward, checkpointing, ONNX export — without
downloading the 8.4GB corpus. The audio is meaningless; only the plumbing is being tested.

    uv run python tests/make_synthetic_dataset.py --out data_smoke --n 18
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050

# Short, varied real Tamil sentences so espeak-ng has genuine text to phonemize.
SENTENCES = [
    "வணக்கம்.",
    "இது ஒரு சோதனை.",
    "தமிழ் மொழி இனிமையானது.",
    "நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "எனக்கு தமிழ் தெரியும்.",
    "இன்று வானிலை நன்றாக உள்ளது.",
    "நான் புத்தகம் படிக்கிறேன்.",
    "அவன் பள்ளிக்கு செல்கிறான்.",
    "நாங்கள் நண்பர்கள்.",
    "மரத்தில் பறவைகள் உள்ளன.",
    "நன்றி, மீண்டும் வருகிறேன்.",
    "இந்த உணவு சுவையாக இருக்கிறது.",
    "காலை வணக்கம்.",
    "மாலை நேரம் அழகாக உள்ளது.",
    "தண்ணீர் குடிக்க வேண்டும்.",
    "புதிய திட்டம் தயாராக உள்ளது.",
    "கடல் அலைகள் ஓசை எழுப்புகின்றன.",
    "மலையின் உச்சியில் பனி உள்ளது.",
    "குழந்தைகள் விளையாடுகிறார்கள்.",
    "சந்திரன் வானத்தில் ஒளிர்கிறது.",
]


def _synth_clip(idx: int, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120.0 + 25.0 * (idx % 5)
    # A couple of harmonics + light noise + an envelope: enough signal for the spec/mel path.
    wav = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.05 * np.random.default_rng(idx).standard_normal(n)
    )
    env = np.minimum(1.0, np.minimum(t / 0.05, (seconds - t) / 0.05))
    wav = wav * np.clip(env, 0.0, 1.0)
    peak = float(np.max(np.abs(wav))) or 1.0
    return (0.9 * wav / peak).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data_smoke"))
    p.add_argument("--n", type=int, default=18)
    args = p.parse_args()

    out = args.out.resolve()
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for i in range(args.n):
        text = SENTENCES[i % len(SENTENCES)]
        seconds = 1.4 + 0.6 * math.sin(i)  # vary length 0.8..2.0s
        seconds = max(1.0, seconds)
        clip_id = f"smoke_{i:03d}"
        sf.write(wav_dir / f"{clip_id}.wav", _synth_clip(i, seconds), SR, subtype="PCM_16")
        rows.append((clip_id, text))

    n_val = max(2, args.n // 6)
    train, val = rows[:-n_val], rows[-n_val:]

    def _write(path: Path, items: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
            for cid, text in items:
                w.writerow([cid, text, text])

    _write(out / "metadata_train.csv", train)
    _write(out / "metadata_val.csv", val)
    print(f"wrote {args.n} synthetic clips to {wav_dir} (train={len(train)} val={len(val)})")


if __name__ == "__main__":
    main()
