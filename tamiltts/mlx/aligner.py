"""Forward-sum (CTC) aligner -> hard durations for the non-AR TTS.

The AR teacher's attention had no alignment, so instead we train a small PyTorch alignment module
that learns a MONOTONIC text<->mel alignment via the forward-sum objective (RAD-TTS / "one TTS
alignment to rule them all", Badlani et al. 2021). Then Monotonic Alignment Search (MAS) extracts
per-token hard durations, saved to <data>/durations/<id>.npy for tamiltts.mlx.train_ns.

This module is training-prep only (never exported); PyTorch is convenient here for nn.CTCLoss.

    uv run python -m tamiltts.mlx.aligner --data data/mlx --steps 8000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# MPS lacks aten::_ctc_loss; let that one op fall back to CPU (convs/distance still run on MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import TTSData


# ----------------------------- model -----------------------------

def conv_block(cin, cout, k=3):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.ReLU(),
                         nn.Conv1d(cout, cout, k, padding=k // 2), nn.ReLU())


class ConvAligner(nn.Module):
    """Distance-based soft attention between text-key and mel-query embeddings."""
    def __init__(self, vocab, n_mels=80, d=256, d_a=80):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.key = nn.Sequential(conv_block(d, d), nn.Conv1d(d, d_a, 1))
        self.query = nn.Sequential(conv_block(n_mels, d), nn.Conv1d(d, d_a, 1))

    def forward(self, tokens, mel):
        # tokens (B,Tt) ; mel (B,Tm,n_mels)
        k = self.key(self.embed(tokens).transpose(1, 2))      # (B,d_a,Tt)
        q = self.query(mel.transpose(1, 2))                   # (B,d_a,Tm)
        # negative L2 distance via ||q||^2+||k||^2-2q.k (efficient; no (B,d,Tm,Tt) blowup)
        q2 = (q * q).sum(1).unsqueeze(2)                      # (B,Tm,1)
        k2 = (k * k).sum(1).unsqueeze(1)                      # (B,1,Tt)
        qk = torch.bmm(q.transpose(1, 2), k)                 # (B,Tm,Tt)
        dist = -(q2 + k2 - 2.0 * qk) / (k.shape[1] ** 0.5)
        return F.log_softmax(dist, dim=2)                     # log P(text | mel frame)


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1.0):
        super().__init__()
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.blank = blank_logprob

    def forward(self, attn_logprob, text_lens, mel_lens):
        # attn_logprob (B, Tm, Tt) -> pad blank class at text index 0
        ap = F.pad(attn_logprob, (1, 0), value=self.blank)    # (B,Tm,Tt+1)
        B = ap.shape[0]
        total = 0.0
        for b in range(B):
            tl, ml = int(text_lens[b]), int(mel_lens[b])
            tgt = torch.arange(1, tl + 1).unsqueeze(0)         # (1,tl) on CPU for ctc
            cur = ap[b, :ml, : tl + 1].unsqueeze(1)            # (ml,1,tl+1)
            cur = F.log_softmax(cur, dim=-1)
            total = total + self.ctc(cur, tgt,
                                     torch.tensor([ml]),       # input/target lengths on CPU
                                     torch.tensor([tl]))
        return total / B


# ----------------------------- MAS (hard durations) -----------------------------

def mas(attn_tt_tm: np.ndarray) -> np.ndarray:
    """Monotonic Alignment Search. value (Tt, Tm) log-prob -> durations (Tt,), each >=1, sum=Tm."""
    Tt, Tm = attn_tt_tm.shape
    neg = -1e9
    Q = np.full((Tt, Tm), neg, dtype=np.float64)
    back = np.zeros((Tt, Tm), dtype=np.int8)  # 0=stay(same token), 1=move(prev token)
    Q[0, 0] = attn_tt_tm[0, 0]
    for t in range(1, Tm):
        stay = Q[:, t - 1]
        move = np.concatenate([[neg], Q[:-1, t - 1]])
        take_move = move > stay
        Q[:, t] = attn_tt_tm[:, t] + np.where(take_move, move, stay)
        back[:, t] = take_move.astype(np.int8)
    # backtrack from (Tt-1, Tm-1)
    dur = np.zeros(Tt, dtype=np.int32)
    s = Tt - 1
    for t in range(Tm - 1, -1, -1):
        dur[s] += 1
        if t > 0 and back[s, t] == 1:
            s -= 1
    return dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # CPU: dot-product attention is cheap, and this avoids per-step MPS<->CPU CTC transfer thrash.
    dev = "cpu"
    data = TTSData(args.data)
    items = [it for it in data.train if it["frames"] <= args.max_frames]
    if args.limit:
        items = items[: args.limit]
    print(f"[aligner] device={dev} items={len(items)} vocab={len(data.vocab)}", flush=True)

    model = ConvAligner(len(data.vocab)).to(dev)
    loss_fn = ForwardSumLoss().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    def make_batch(group):
        toks = [data.encode_text(it["text"]) for it in group]
        mels = [data._load_mel(it["id"]) for it in group]
        tl = np.array([len(t) for t in toks]); ml = np.array([m.shape[0] for m in mels])
        Tt, Tm = int(tl.max()), int(ml.max())
        tok = np.zeros((len(group), Tt), np.int64)
        mel = np.zeros((len(group), Tm, mels[0].shape[1]), np.float32)
        for i, (t, m) in enumerate(zip(toks, mels)):
            tok[i, : len(t)] = t; mel[i, : m.shape[0]] = m
        return (torch.tensor(tok, device=dev), torch.tensor(mel, device=dev),
                torch.tensor(tl), torch.tensor(ml))

    items_sorted = sorted(items, key=lambda x: x["frames"])
    groups = [items_sorted[i:i + args.batch] for i in range(0, len(items_sorted), args.batch)]
    model.train(); step = 0; run = 0.0
    while step < args.steps:
        for gi in rng.permutation(len(groups)):
            tok, mel, tl, ml = make_batch(groups[gi])
            ap_lp = model(tok, mel)
            loss = loss_fn(ap_lp, tl, ml)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); step += 1
            if step % 100 == 0:
                print(f"step {step:5d} | fwd-sum loss {run/100:.4f}", flush=True); run = 0.0
            if step >= args.steps:
                break

    # ---- extract hard durations for ALL clips (train+val) ----
    out_dir = args.data / "durations"; out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    allit = [it for it in (data.train + data.val) if it["frames"] <= args.max_frames]
    with torch.no_grad():
        for i, it in enumerate(allit):
            tok, mel, tl, ml = make_batch([it])
            ap_lp = model(tok, mel)[0].cpu().numpy()      # (Tm, Tt)
            n, T = int(ml[0]), int(tl[0])
            dur = mas(ap_lp[:n, :T].T)                     # (Tt, Tm)->durations
            np.save(out_dir / f"{it['id']}.npy", dur)
            if (i + 1) % 250 == 0:
                print(f"  durations {i+1}/{len(allit)} (sum={int(dur.sum())} max={int(dur.max())} "
                      f"zeros={int((dur==0).sum())})", flush=True)
    print(f"done. wrote {len(allit)} durations to {out_dir}")


if __name__ == "__main__":
    main()
