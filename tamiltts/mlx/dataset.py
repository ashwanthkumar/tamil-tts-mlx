"""Dataset / batching for MLX TTS training. Loads cached mels + tokenizes text."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class TTSData:
    def __init__(self, out_dir: str | Path):
        out = Path(out_dir)
        self.vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
        stats = json.loads((out / "stats.json").read_text())
        self.mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
        self.mel_std = np.array(stats["mel_std"], dtype=np.float32)
        self.mel_dir = out / "mels"
        self.train = json.loads((out / "train.json").read_text(encoding="utf-8"))
        self.val = json.loads((out / "val.json").read_text(encoding="utf-8"))
        # variance adaptors (v0.2): per-token pitch/energy + normalization stats (optional)
        self.pitch_dir = out / "pitch"
        self.energy_dir = out / "energy"
        vp = out / "variance_stats.json"
        if vp.exists():
            vs = json.loads(vp.read_text())
            self.pitch_mean, self.pitch_std = vs["pitch_mean"], vs["pitch_std"]
            self.energy_mean, self.energy_std = vs["energy_mean"], vs["energy_std"]
        else:
            self.pitch_mean = self.energy_mean = 0.0
            self.pitch_std = self.energy_std = 1.0

    def encode_text(self, text: str) -> list[int]:
        ids = [BOS_ID]
        for ch in text:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
        ids.append(EOS_ID)
        return ids

    def _load_mel(self, uid: str) -> np.ndarray:
        mel = np.load(self.mel_dir / f"{uid}.npy")
        return (mel - self.mel_mean) / self.mel_std

    # ----- non-autoregressive (FastSpeech) batching: needs extracted durations -----
    @property
    def dur_dir(self) -> Path:
        return self.mel_dir.parent / "durations"

    def has_durations(self, uid: str) -> bool:
        return (self.dur_dir / f"{uid}.npy").exists()

    def _load_variance(self, uid: str, kind: str):
        """Load + z-normalize a per-token pitch/energy vector; None if absent."""
        d = self.pitch_dir if kind == "pitch" else self.energy_dir
        f = d / f"{uid}.npy"
        if not f.exists():
            return None
        v = np.load(f).astype(np.float32)
        mean, std = (self.pitch_mean, self.pitch_std) if kind == "pitch" else (self.energy_mean, self.energy_std)
        return (v - mean) / std

    def batches_ns(self, split: str = "train", batch_size: int = 16, shuffle: bool = True,
                   rng: np.random.Generator | None = None, max_frames: int = 1200):
        items = [it for it in (self.train if split == "train" else self.val)
                 if it["frames"] <= max_frames and self.has_durations(it["id"])]
        items.sort(key=lambda x: x["frames"])
        groups = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        order = np.arange(len(groups))
        if shuffle:
            (rng or np.random).shuffle(order)
        for gi in order:
            yield self._collate_ns(groups[int(gi)])

    def _collate_ns(self, group):
        toks = [self.encode_text(it["text"]) for it in group]
        durs = [np.load(self.dur_dir / f"{it['id']}.npy").astype(np.int32) for it in group]
        mels = [self._load_mel(it["id"]) for it in group]
        # align lengths: duration vector length must match token length
        for i in range(len(group)):
            t = len(toks[i])
            if len(durs[i]) != t:                       # safety pad/trim
                d = np.zeros(t, dtype=np.int32)
                d[: min(t, len(durs[i]))] = durs[i][: min(t, len(durs[i]))]
                durs[i] = d
        tlen = np.array([len(t) for t in toks], dtype=np.int32)
        mlen = np.array([int(d.sum()) for d in durs], dtype=np.int32)  # mel len implied by durations
        Tt, Tm = int(tlen.max()), int(mlen.max())
        B = len(group); n_mels = mels[0].shape[1]

        tok = np.zeros((B, Tt), dtype=np.int32)
        dur = np.zeros((B, Tt), dtype=np.float32)
        pitch = np.zeros((B, Tt), dtype=np.float32)    # per-token, z-normalized (0 = mean)
        energy = np.zeros((B, Tt), dtype=np.float32)
        expand_idx = np.zeros((B, Tm), dtype=np.int32)
        mel = np.zeros((B, Tm, n_mels), dtype=np.float32)
        for i, (t, d, m) in enumerate(zip(toks, durs, mels)):
            tok[i, : len(t)] = t
            dur[i, : len(d)] = d
            pv = self._load_variance(group[i]["id"], "pitch")
            ev = self._load_variance(group[i]["id"], "energy")
            if pv is not None:
                pitch[i, : min(len(pv), Tt)] = pv[:Tt]
            if ev is not None:
                energy[i, : min(len(ev), Tt)] = ev[:Tt]
            idx = np.repeat(np.arange(len(d), dtype=np.int32), d)  # token idx per mel frame
            n = min(len(idx), m.shape[0], Tm)
            expand_idx[i, :n] = idx[:n]
            mel[i, :n] = m[:n]
            mlen[i] = n
        return {"tok": tok, "tlen": tlen, "dur": dur, "pitch": pitch, "energy": energy,
                "expand_idx": expand_idx, "mel": mel, "mlen": mlen}

    def batches(self, split: str = "train", batch_size: int = 16, shuffle: bool = True,
                rng: np.random.Generator | None = None, max_frames: int = 1000):
        items = list(self.train if split == "train" else self.val)
        items = [it for it in items if it["frames"] <= max_frames]
        # bucket by frame length to minimise padding
        items.sort(key=lambda x: x["frames"])
        groups = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        order = np.arange(len(groups))
        if shuffle:
            (rng or np.random).shuffle(order)
        for gi in order:
            yield self._collate(groups[int(gi)])

    def _collate(self, group):
        toks = [self.encode_text(it["text"]) for it in group]
        mels = [self._load_mel(it["id"]) for it in group]
        tlen = np.array([len(t) for t in toks], dtype=np.int32)
        mlen = np.array([m.shape[0] for m in mels], dtype=np.int32)
        Tt, Tm = int(tlen.max()), int(mlen.max())
        B = len(group)
        n_mels = mels[0].shape[1]

        tok = np.full((B, Tt), PAD_ID, dtype=np.int32)
        mel = np.zeros((B, Tm, n_mels), dtype=np.float32)
        stop = np.zeros((B, Tm), dtype=np.float32)
        for i, (t, m) in enumerate(zip(toks, mels)):
            tok[i, : len(t)] = t
            mel[i, : m.shape[0]] = m
            stop[i, m.shape[0] - 1:] = 1.0  # stop fires from last valid frame on

        go = np.zeros((B, 1, n_mels), dtype=np.float32)
        mel_in = np.concatenate([go, mel[:, :-1]], axis=1)  # teacher-forced input
        return {
            "tok": tok, "tlen": tlen, "mel": mel, "mel_in": mel_in,
            "mlen": mlen, "stop": stop,
        }
