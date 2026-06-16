"""Extract per-token pitch (F0) + energy for FastSpeech-2 variance adaptors (v0.2).

Reads cached wavs + mels + aligner durations, computes per-frame F0 (pyworld) and energy
(STFT L2), interpolates F0 through unvoiced regions, pools to per-token means via the
duration vector, and writes:
  data/mlx/pitch/<id>.npy    per-token pitch (Hz, continuous)   float32 (n_tok,)
  data/mlx/energy/<id>.npy   per-token energy                   float32 (n_tok,)
  data/mlx/variance_stats.json   global pitch/energy mean+std (train split)

Run AFTER preprocess + aligner (it needs durations).

    uv run python -m tamiltts.mlx.extract_variance --data data --mlx data/mlx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import pyworld as pw

from .audio import load_wav, SR, N_FFT, HOP, WIN

FRAME_PERIOD = HOP / SR * 1000.0   # ms; aligns F0 frames to the mel hop


def _fit(x: np.ndarray, T: int) -> np.ndarray:
    """Trim/pad a per-frame vector to exactly T frames (match the cached mel)."""
    if len(x) >= T:
        return x[:T]
    pad = "edge" if len(x) else "constant"
    return np.pad(x, (0, T - len(x)), mode=pad)


def frame_energy(wav: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN, center=True))
    return np.sqrt((S ** 2).sum(axis=0)).astype(np.float32)   # (T,)


def frame_f0(wav: np.ndarray, T: int) -> np.ndarray:
    f0, t = pw.dio(wav.astype(np.float64), SR, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, SR).astype(np.float32)
    # interpolate through unvoiced (f0==0) so the pitch contour is continuous
    voiced = f0 > 0
    if voiced.sum() >= 2:
        idx = np.arange(len(f0))
        f0 = np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)
    elif voiced.sum() == 1:
        f0[:] = f0[voiced][0]
    return _fit(f0, T)


def pool_per_token(frame_vals: np.ndarray, dur: np.ndarray) -> np.ndarray:
    """Average per-frame values into per-token means using the duration vector."""
    n_tok = len(dur)
    idx = np.repeat(np.arange(n_tok), dur)            # frame -> token index
    n = min(len(idx), len(frame_vals))
    sums = np.bincount(idx[:n], weights=frame_vals[:n], minlength=n_tok)
    cnt = np.maximum(dur.astype(np.float64), 1.0)
    return (sums[:n_tok] / cnt).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--mlx", type=Path, default=Path("data/mlx"))
    ap.add_argument("--limit", type=int, default=0, help="cap items (for quick tests)")
    args = ap.parse_args()

    mel_dir = args.mlx / "mels"
    dur_dir = args.mlx / "durations"
    pitch_dir = args.mlx / "pitch"; pitch_dir.mkdir(parents=True, exist_ok=True)
    energy_dir = args.mlx / "energy"; energy_dir.mkdir(parents=True, exist_ok=True)

    train = json.loads((args.mlx / "train.json").read_text(encoding="utf-8"))
    val = json.loads((args.mlx / "val.json").read_text(encoding="utf-8"))
    items = train + val
    if args.limit:
        items = items[: args.limit]
    train_ids = {it["id"] for it in train}

    # running stats over TRAIN per-token values
    ps = ps2 = es = es2 = 0.0
    pn = en = 0
    done = skipped = 0
    for i, it in enumerate(items):
        uid = it["id"]
        durf = dur_dir / f"{uid}.npy"
        if not durf.exists():
            skipped += 1
            continue
        wav = load_wav(str(args.data / "wavs" / f"{uid}.wav"))   # same trim as mel extraction
        mel = np.load(mel_dir / f"{uid}.npy")
        T = mel.shape[0]
        energy = _fit(frame_energy(wav), T)
        pitch = frame_f0(wav, T)
        dur = np.load(durf).astype(np.int64)
        p_tok = pool_per_token(pitch, dur)
        e_tok = pool_per_token(energy, dur)
        np.save(pitch_dir / f"{uid}.npy", p_tok)
        np.save(energy_dir / f"{uid}.npy", e_tok)
        if uid in train_ids:
            ps += float(p_tok.sum()); ps2 += float((p_tok.astype(np.float64) ** 2).sum()); pn += len(p_tok)
            es += float(e_tok.sum()); es2 += float((e_tok.astype(np.float64) ** 2).sum()); en += len(e_tok)
        done += 1
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(items)} (done {done}, skipped {skipped})", flush=True)

    p_mean = ps / pn
    p_std = max((ps2 / pn - p_mean ** 2), 1e-10) ** 0.5
    e_mean = es / en
    e_std = max((es2 / en - e_mean ** 2), 1e-10) ** 0.5
    stats = {"pitch_mean": float(p_mean), "pitch_std": float(p_std),
             "energy_mean": float(e_mean), "energy_std": float(e_std)}
    (args.mlx / "variance_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"done. variance for {done} items, skipped {skipped} (no durations)")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
