"""Precompute log-mels + char vocab + normalization stats for MLX TTS training.

    uv run python -m tamiltts.mlx.preprocess --data data --out data/mlx [--limit N]

Outputs under <out>/:
  mels/<id>.npy        log-mel (T, 80) float32
  vocab.json           {char: id} incl. <pad>,<bos>,<eos>
  stats.json           mel mean/std (global), max frames, items list
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audio import load_wav, wav_to_mel

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def read_metadata(csv_path: Path):
    items = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2 and row[0] and row[1]:
                items.append((row[0], row[1]))
    return items


def build_vocab(texts) -> dict:
    chars = set()
    for t in texts:
        chars |= set(t)
    vocab = {PAD: 0, BOS: 1, EOS: 2}
    for c in sorted(chars):
        vocab[c] = len(vocab)
    return vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/mlx"))
    ap.add_argument("--limit", type=int, default=0, help="cap items (for quick tests)")
    args = ap.parse_args()

    train = read_metadata(args.data / "metadata_train.csv")
    val = read_metadata(args.data / "metadata_val.csv")
    if args.limit:
        train = train[: args.limit]
        val = val[: max(2, args.limit // 10)]

    vocab = build_vocab([t for _, t in train])
    mel_dir = args.out / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)

    # running stats for normalization
    n = 0
    s = np.zeros(80, dtype=np.float64)
    ss = np.zeros(80, dtype=np.float64)
    max_frames = 0
    kept = {"train": [], "val": []}

    for split, items in (("train", train), ("val", val)):
        for i, (uid, text) in enumerate(items):
            wav_path = args.data / "wavs" / f"{uid}.wav"
            if not wav_path.exists():
                continue
            mel = wav_to_mel(load_wav(str(wav_path)))  # (T, 80)
            np.save(mel_dir / f"{uid}.npy", mel)
            kept[split].append({"id": uid, "text": text, "frames": int(mel.shape[0])})
            max_frames = max(max_frames, mel.shape[0])
            if split == "train":
                n += mel.shape[0]
                s += mel.sum(axis=0)
                ss += (mel.astype(np.float64) ** 2).sum(axis=0)
            if (i + 1) % 250 == 0:
                print(f"  {split}: {i+1}/{len(items)}", flush=True)

    mean = (s / n).astype(np.float32)
    var = (ss / n - (s / n) ** 2).clip(min=1e-8)
    std = np.sqrt(var).astype(np.float32)

    stats = {
        "mel_mean": mean.tolist(),
        "mel_std": std.tolist(),
        "max_frames": int(max_frames),
        "n_train": len(kept["train"]),
        "n_val": len(kept["val"]),
    }
    (args.out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=0), encoding="utf-8")
    (args.out / "train.json").write_text(json.dumps(kept["train"], ensure_ascii=False), encoding="utf-8")
    (args.out / "val.json").write_text(json.dumps(kept["val"], ensure_ascii=False), encoding="utf-8")

    print(f"done. train={len(kept['train'])} val={len(kept['val'])} vocab={len(vocab)} "
          f"max_frames={max_frames}")


if __name__ == "__main__":
    main()
