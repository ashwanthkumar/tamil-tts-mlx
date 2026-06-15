"""MLX-native forward-sum aligner -> hard durations (runs on the Apple GPU, no PyTorch/CPU).

Learns a monotonic text<->mel alignment via a log-space forward-sum DP (the monotonic-path
likelihood, differentiable). Then Monotonic Alignment Search extracts per-token hard durations,
saved to <data>/durations/<id>.npy for tamiltts.mlx.train_ns.

    uv run python -m tamiltts.mlx.mlx_aligner --data data/mlx --steps 4000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .dataset import TTSData

NEG = -1e9


class Aligner(nn.Module):
    def __init__(self, vocab, n_mels=80, d=256, d_a=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.k1 = nn.Conv1d(d, d, 3, padding=1); self.k2 = nn.Conv1d(d, d_a, 1)
        self.q1 = nn.Conv1d(n_mels, d, 3, padding=1); self.q2 = nn.Conv1d(d, d_a, 1)
        self.scale = d_a ** -0.5

    def __call__(self, tok, mel):
        # channels-last conv (B,L,C)
        k = self.k2(nn.relu(self.k1(self.embed(tok))))     # (B,Tt,d_a)
        q = self.q2(nn.relu(self.q1(mel)))                 # (B,Tm,d_a)
        # negative L2 distance via ||q||^2 + ||k||^2 - 2 q.k (no (B,d,Tm,Tt) blowup)
        q2 = (q * q).sum(2)[:, :, None]                    # (B,Tm,1)
        k2 = (k * k).sum(2)[:, None, :]                    # (B,1,Tt)
        score = -(q2 + k2 - 2.0 * (q @ k.transpose(0, 2, 1))) * self.scale  # (B,Tm,Tt)
        return score - mx.logsumexp(score, axis=2, keepdims=True)   # log P(token|frame)


def forward_sum_loss(attn_logprob, tlen, mlen):
    """Monotonic forward-sum: -log P(monotonic path token0..Tt-1 over frames). attn (B,Tm,Tt)."""
    B, Tm, Tt = attn_logprob.shape
    a = mx.concatenate([attn_logprob[:, 0, :1], mx.full((B, Tt - 1), NEG)], axis=1)  # alpha at t=0
    alphas = [a]
    for t in range(1, Tm):
        shifted = mx.concatenate([mx.full((B, 1), NEG), a[:, :-1]], axis=1)
        a = attn_logprob[:, t, :] + mx.logaddexp(a, shifted)
        alphas.append(a)
    A = mx.stack(alphas, axis=1)                            # (B,Tm,Tt)
    bi = mx.arange(B)
    final = A[bi, mlen - 1, tlen - 1]                       # alpha at (mlen-1, tlen-1)
    return -(final / mlen.astype(mx.float32)).mean()


def guided_loss(attn_logprob, tlen, mlen, sigma=0.2):
    """Diagonal prior: penalize attention mass far from the t/Tt == s/Tm diagonal. (B,Tm,Tt)."""
    B, Tm, Tt = attn_logprob.shape
    ml = mlen.astype(mx.float32)[:, None, None]
    tl = tlen.astype(mx.float32)[:, None, None]
    tpos = mx.arange(Tm)[None, :, None] / ml          # (B,Tm,1)
    spos = mx.arange(Tt)[None, None, :] / tl          # (B,1,Tt)
    W = 1.0 - mx.exp(-((tpos - spos) ** 2) / (2 * sigma * sigma))
    mask = ((mx.arange(Tm)[None, :, None] < mlen[:, None, None]) &
            (mx.arange(Tt)[None, None, :] < tlen[:, None, None])).astype(mx.float32)
    attn = mx.exp(attn_logprob)
    return (attn * W * mask).sum() / mx.maximum(mask.sum(), 1.0)


def mas(logp_tt_tm: np.ndarray) -> np.ndarray:
    """Viterbi monotonic alignment. (Tt,Tm) -> durations (Tt,), each>=1, sum=Tm."""
    Tt, Tm = logp_tt_tm.shape
    Q = np.full((Tt, Tm), -1e9); back = np.zeros((Tt, Tm), np.int8)
    Q[0, 0] = logp_tt_tm[0, 0]
    for t in range(1, Tm):
        stay = Q[:, t - 1]
        move = np.concatenate([[-1e9], Q[:-1, t - 1]])
        mv = move > stay
        Q[:, t] = logp_tt_tm[:, t] + np.where(mv, move, stay); back[:, t] = mv
    dur = np.zeros(Tt, np.int32); s = Tt - 1
    for t in range(Tm - 1, -1, -1):
        dur[s] += 1
        if t > 0 and back[s, t] == 1:
            s -= 1
    return dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = TTSData(args.data)
    items = [it for it in data.train if it["frames"] <= args.max_frames]
    if args.limit:
        items = items[: args.limit]
    items.sort(key=lambda x: x["frames"])
    groups = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    print(f"[mlx-aligner] device={mx.default_device()} items={len(items)} vocab={len(data.vocab)}", flush=True)

    model = Aligner(len(data.vocab)); mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr)

    def collate(group):
        toks = [data.encode_text(it["text"]) for it in group]
        mels = [data._load_mel(it["id"]) for it in group]
        tl = np.array([len(t) for t in toks], np.int32); ml = np.array([m.shape[0] for m in mels], np.int32)
        Tt, Tm = int(tl.max()), int(ml.max())
        tok = np.zeros((len(group), Tt), np.int32); mel = np.zeros((len(group), Tm, mels[0].shape[1]), np.float32)
        for i, (t, m) in enumerate(zip(toks, mels)):
            tok[i, :len(t)] = t; mel[i, :m.shape[0]] = m
        return tok, mel, tl, ml

    def loss_fn(model, tok, mel, tl, ml):
        ap = model(mx.array(tok), mx.array(mel))
        tl, ml = mx.array(tl), mx.array(ml)
        return guided_loss(ap, tl, ml)   # DECISIVE TEST: pure diagonal prior must diagonalize

    lag = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(0); step = 0; run = 0.0; t0 = time.time()
    model.train()
    while step < args.steps:
        for gi in rng.permutation(len(groups)):
            tok, mel, tl, ml = collate(groups[int(gi)])
            loss, grads = lag(model, tok, mel, tl, ml)
            opt.update(model, grads); mx.eval(model.parameters(), opt.state, loss)
            run += float(loss); step += 1
            if step % 50 == 0:
                print(f"step {step:5d} | fwd-sum {run/50:.4f} | {(time.time()-t0)/50*1000:.0f} ms/step", flush=True)
                run = 0.0; t0 = time.time()
            if step >= args.steps:
                break

    # extract durations (full resolution, no grad) via MAS
    out = args.data / "durations"; out.mkdir(parents=True, exist_ok=True)
    model.eval()
    allit = [it for it in (data.train + data.val) if it["frames"] <= args.max_frames]
    for i, it in enumerate(allit):
        tok, mel, tl, ml = collate([it])
        lp = np.array(model(mx.array(tok), mx.array(mel)))[0]    # (Tm,Tt)
        if i == 0:
            n, T = int(ml[0]), int(tl[0])
            am = lp[:n, :T].argmax(axis=1)
            print(f"[debug] clip0 Tt={T} Tm={n} argmax first30={am[:30].tolist()}", flush=True)
            print(f"[debug] argmax distinct tokens used={len(np.unique(am))} "
                  f"maxprob_mean={np.exp(lp[:n,:T].max(axis=1)).mean():.3f}", flush=True)
        dur = mas(lp[:int(ml[0]), :int(tl[0])].T)
        np.save(out / f"{it['id']}.npy", dur)
        if (i + 1) % 400 == 0:
            print(f"  durations {i+1}/{len(allit)} (sum={int(dur.sum())} max={int(dur.max())} zeros={int((dur==0).sum())})", flush=True)
    print(f"done. wrote {len(allit)} durations to {out}")


if __name__ == "__main__":
    main()
