"""Prepare the IndicTTS Tamil corpus for training.

Downloads ``SPRINGLab/IndicTTS_Tamil`` (parquet, on the Hugging Face Hub), keeps a single
speaker (default: female), resamples every clip to 22.05 kHz mono, writes the wavs and an
LJSpeech-style ``metadata.csv`` (``id|text|text``) plus a train/val split that
``tamiltts.mlx.preprocess`` reads.

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


def _match_speaker(value, speaker: str, names: list[str] | None) -> bool:
    """True if a row's gender `value` matches the wanted `speaker`.

    The IndicTTS gender column is a ClassLabel int (0='female', 1='male'); `names` is that
    mapping when known. Falls back to a substring match for string-valued columns.
    """
    if names is not None:
        try:
            return names[int(value)].lower() == speaker.lower()
        except (ValueError, IndexError, TypeError):
            return False
    sval = str(value).strip().lower()
    return speaker.lower() in sval or sval in speaker.lower()


def _classlabel_names_from_parquet(pf, column: str) -> list[str] | None:
    """Read a ClassLabel's `names` list from a datasets-written parquet's schema metadata."""
    import json

    md = pf.schema_arrow.metadata or {}
    for k, v in md.items():
        key = k.decode() if isinstance(k, bytes) else k
        if key.lower() != "huggingface":
            continue
        info = json.loads(v)
        feats = (info.get("info") or info).get("features") or info.get("features") or {}
        feat = feats.get(column)
        if isinstance(feat, dict) and feat.get("_type") == "ClassLabel":
            return feat.get("names")
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

    gender_names = None
    if gender_col is not None and hasattr(ds.features.get(gender_col), "names"):
        gender_names = ds.features[gender_col].names

    rows: list[tuple[str, str]] = []
    kept = 0
    skipped = 0
    for i, ex in enumerate(ds):
        if gender_col is not None:
            if not _match_speaker(ex[gender_col], speaker, gender_names):
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


def prepare_sharded(
    out_dir: Path,
    speaker: str = "female",
    target_sr: int = TARGET_SR,
    val_size: int = 100,
    limit: int | None = None,
    stop_after_empty: int = 0,
) -> None:
    """Low-disk prep: download ONE parquet shard at a time, extract the wanted speaker's wavs,
    then delete the shard cache before fetching the next. Peak disk stays ~1 shard + output wavs,
    so the full ~8.4GB corpus can be processed on a machine with only a few GB free.
    """
    try:
        import io
        import shutil

        import librosa
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing training dependencies. Install with `uv sync --extra train`.\n"
            f"(import error: {exc})"
        )

    out_dir = out_dir.resolve()
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = Path("~/.cache/huggingface/hub").expanduser()

    api = HfApi()
    info = api.repo_info(DATASET_ID, repo_type="dataset", files_metadata=True)
    shards = sorted(
        s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")
    )
    print(f"{len(shards)} parquet shards to process (speaker={speaker})", file=sys.stderr)

    rows: list[tuple[str, str]] = []
    kept = 0
    empty_streak = 0
    for si, shard in enumerate(shards):
        kept_before = kept
        local = hf_hub_download(DATASET_ID, shard, repo_type="dataset")
        pf = pq.ParquetFile(local)
        names = pf.schema_arrow.names
        gender_col = _find_gender_column(names)
        text_col = _find_text_column(names)
        gender_names = (
            _classlabel_names_from_parquet(pf, gender_col) if gender_col else None
        )

        for batch in pf.iter_batches(batch_size=32):
            d = batch.to_pydict()
            for j in range(len(d[text_col])):
                if gender_col is not None:
                    if not _match_speaker(d[gender_col][j], speaker, gender_names):
                        continue
                text = (d[text_col][j] or "").strip()
                if not text:
                    continue
                audio = d["audio"][j]
                raw = audio["bytes"] if isinstance(audio, dict) else audio
                wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                peak = float(np.max(np.abs(wav))) if wav.size else 0.0
                if peak > 0:
                    wav = 0.95 * wav / peak
                clip_id = f"ta_{speaker}_{kept:05d}"
                sf.write(wav_dir / f"{clip_id}.wav", wav, target_sr, subtype="PCM_16")
                rows.append((clip_id, text))
                kept += 1
                if limit is not None and kept >= limit:
                    break
            if limit is not None and kept >= limit:
                break

        # Reclaim disk before fetching the next shard.
        shutil.rmtree(hub_cache, ignore_errors=True)
        print(f"  shard {si + 1}/{len(shards)} done; kept={kept} clips so far", file=sys.stderr)
        if limit is not None and kept >= limit:
            break

        # This corpus stores each speaker contiguously, so once the wanted speaker's clips stop
        # appearing we can skip the remaining (other-speaker) shards instead of downloading them.
        empty_streak = empty_streak + 1 if kept == kept_before else 0
        if stop_after_empty > 0 and empty_streak >= stop_after_empty and kept > 0:
            print(
                f"  no new {speaker} clips for {empty_streak} shards; stopping early "
                f"(skipping {len(shards) - si - 1} remaining shards)",
                file=sys.stderr,
            )
            break

    if not rows:
        raise SystemExit("No clips matched; check --speaker against the dataset gender labels.")

    _write_metadata(out_dir, rows, val_size)


def _write_metadata(out_dir: Path, rows: list[tuple[str, str]], val_size: int) -> None:
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
        f"\nDone. total={len(rows)} train={len(train_rows)} val={len(val_rows)}\n"
        f"  wavs:     {out_dir / 'wavs'}\n  metadata: {out_dir / 'metadata.csv'}",
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
    p.add_argument(
        "--low-disk",
        action="store_true",
        help="process one parquet shard at a time, deleting each before the next "
        "(handles the full corpus with only a few GB free)",
    )
    p.add_argument(
        "--stop-after-empty",
        type=int,
        default=0,
        help="(low-disk) stop after N consecutive shards yield no wanted-speaker clips. "
        "Faster, but LOSSY on this corpus: most female clips are in the first ~6 shards, yet some "
        "are scattered later (e.g. shard 14/15/16), so early-stop drops them. Default 0 = scan all.",
    )
    args = p.parse_args()

    if args.low_disk:
        prepare_sharded(
            out_dir=args.out,
            speaker=args.speaker,
            target_sr=args.sample_rate,
            val_size=args.val_size,
            limit=args.limit,
            stop_after_empty=args.stop_after_empty,
        )
        return

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
