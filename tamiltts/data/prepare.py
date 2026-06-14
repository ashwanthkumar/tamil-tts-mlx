"""Prepare the IndicTTS Tamil corpus for VITS training.

Downloads ``SPRINGLab/IndicTTS_Tamil`` (parquet, on the Hugging Face Hub), keeps a single
speaker (default: female), resamples every clip to 22.05 kHz mono, writes the wavs and an
LJSpeech-style ``metadata.csv`` (``id|text|text``) plus a train/val split that Coqui-TTS reads.

Usage:
    uv run python -m tamiltts.data.prepare --speaker female --out data
    uv run python -m tamiltts.data.prepare --speaker female --out data --limit 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DATASET_ID = "SPRINGLab/IndicTTS_Tamil"
TARGET_SR = 22050


def _find_gender_column(columns: list[str]) -> str | None:
    for cand in ("gender", "speaker", "sex", "speaker_gender"):
        if cand in columns:
            return cand
    return None


def _find_text_column(columns: list[str]) -> str:
    for cand in ("text", "transcription", "sentence", "normalized_text"):
        if cand in columns:
            return cand
    raise SystemExit(f"Could not find a text column in {columns!r}")


def prepare(
    out_dir: Path,
    speaker: str = "female",
    target_sr: int = TARGET_SR,
    val_size: int = 100,
    limit: int | None = None,
    split: str = "train",
    streaming: bool = False,
) -> None:
    # Heavy imports live inside the function so `--help` works without the `train` extras.
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise SystemExit(
            "Missing training dependencies. Install them with:\n"
            "    uv sync --extra train\n"
            f"(import error: {exc})"
        )

    out_dir = out_dir.resolve()
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_ID} (split={split}, streaming={streaming}) ...",
        file=sys.stderr,
    )
    # Streaming avoids downloading all 8.4GB up front; ideal for --limit smoke tests.
    ds = load_dataset(DATASET_ID, split=split, streaming=streaming)
    columns = list(ds.features.keys())
    print(f"Columns: {columns}", file=sys.stderr)

    gender_col = _find_gender_column(columns)
    text_col = _find_text_column(columns)
    if gender_col is None:
        print(
            "WARNING: no gender/speaker column found; keeping ALL rows "
            "(both speakers will be mixed).",
            file=sys.stderr,
        )

    rows: list[tuple[str, str]] = []
    kept = 0
    skipped = 0
    for i, ex in enumerate(ds):
        if gender_col is not None:
            value = str(ex[gender_col]).strip().lower()
            if speaker.lower() not in value and value not in speaker.lower():
                continue

        text = (ex[text_col] or "").strip()
        if not text:
            skipped += 1
            continue

        audio = ex["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

        # Light peak normalization to keep levels consistent across the corpus.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0:
            wav = 0.95 * wav / peak

        clip_id = f"ta_{speaker}_{kept:05d}"
        sf.write(wav_dir / f"{clip_id}.wav", wav, target_sr, subtype="PCM_16")
        rows.append((clip_id, text))
        kept += 1

        if kept % 200 == 0:
            print(f"  ... {kept} clips written", file=sys.stderr)
        if limit is not None and kept >= limit:
            break

    if not rows:
        raise SystemExit(
            "No clips matched. Check the --speaker value against the dataset's gender labels."
        )

    # Deterministic split: last `val_size` rows go to validation.
    val_size = min(val_size, max(1, len(rows) // 10))
    train_rows = rows[:-val_size]
    val_rows = rows[-val_size:]

    def _write(path: Path, items: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
            for clip_id, text in items:
                writer.writerow([clip_id, text, text])

    _write(out_dir / "metadata.csv", rows)
    _write(out_dir / "metadata_train.csv", train_rows)
    _write(out_dir / "metadata_val.csv", val_rows)

    print(
        f"\nDone. kept={kept} skipped_empty={skipped} "
        f"train={len(train_rows)} val={len(val_rows)}\n"
        f"  wavs:     {wav_dir}\n"
        f"  metadata: {out_dir / 'metadata.csv'}",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare IndicTTS Tamil for VITS training.")
    p.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    p.add_argument("--speaker", default="female", help="speaker/gender to keep (default: female)")
    p.add_argument("--sample-rate", type=int, default=TARGET_SR)
    p.add_argument("--val-size", type=int, default=100, help="number of clips held out for validation")
    p.add_argument("--limit", type=int, default=None, help="cap clips (for a quick smoke test)")
    p.add_argument("--split", default="train", help="HF dataset split to read (default: train)")
    p.add_argument(
        "--streaming",
        action="store_true",
        help="stream from the Hub instead of downloading all 8.4GB (use with --limit)",
    )
    args = p.parse_args()

    prepare(
        out_dir=args.out,
        speaker=args.speaker,
        target_sr=args.sample_rate,
        val_size=args.val_size,
        limit=args.limit,
        split=args.split,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
