"""Distill per-token durations from the trained AR model's cross-attention (FastSpeech1-style).

For each clip, run the AR decoder teacher-forced, average the decoder cross-attention over all
layers + heads to get an alignment (T_mel x T_text), then assign each mel frame to its argmax text
token. duration[i] = #frames assigned to token i (sums to T_mel). Saved to <data>/durations/<id>.npy.

    uv run python -m tamiltts.mlx.extract_durations --ar-run runs_mlx/tamil_mlx --data data/mlx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .dataset import TTSData
from .infer import load_model
from .model import key_pad_mask, causal_mask


def cross_attn_scores(layer, x, mem, n_heads):
    """Replicate mlx MultiHeadAttention to return softmax scores (1, heads, L, S) for one layer."""
    ca = layer.cross_attn
    q = ca.query_proj(x)
    k = ca.key_proj(mem)
    B, L, D = q.shape
    S = k.shape[1]
    q = q.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)      # (B,h,L,dh)
    k = k.reshape(B, S, n_heads, -1).transpose(0, 2, 3, 1)      # (B,h,dh,S)
    scale = (1.0 / q.shape[-1]) ** 0.5
    scores = mx.softmax((q * scale) @ k, axis=-1)               # (B,h,L,S)
    return scores


def alignment(model, n_heads, tok, mel_in, src_mask, self_mask, cross_mask):
    """Average cross-attention over all decoder layers + heads -> (T_mel, T_text)."""
    mem = model.encode(tok, src_mask)
    x = mx.maximum(model.prenet1(mel_in), 0.0)
    x = mx.maximum(model.prenet2(x), 0.0)
    x = x + model._pe[: mel_in.shape[1]]
    acc = None
    for layer in model.dec_layers:
        x1 = layer.n1(x + layer.self_attn(x, x, x, self_mask))
        s = cross_attn_scores(layer, x1, mem, n_heads).mean(axis=1)[0]   # (L,S)
        acc = s if acc is None else acc + s
        x = layer.n2(x1 + layer.cross_attn(x1, mem, mem, cross_mask))
        x = layer.n3(x + layer.l2(nn.gelu(layer.l1(x))))
    return acc / len(model.dec_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-run", type=Path, default=Path("runs_mlx/tamil_mlx"))
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0, help="cap #clips (smoke test)")
    args = ap.parse_args()

    data = TTSData(args.data)
    cfg = json.loads((args.ar_run / "config.json").read_text())["cfg"]
    n_heads = cfg["n_heads"]
    model = load_model(args.ar_run)
    model.eval()

    out_dir = args.data / "durations"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [it for it in (data.train + data.val) if it["frames"] <= args.max_frames]
    if args.limit:
        items = items[: args.limit]
    n_ok = 0
    for i, it in enumerate(items):
        b = data._collate([it])
        tok = mx.array(b["tok"]); mel_in = mx.array(b["mel_in"])
        Tt, Tm = tok.shape[1], mel_in.shape[1]
        src = key_pad_mask(mx.array(b["tlen"]), Tt)
        self_m = causal_mask(Tm) + key_pad_mask(mx.array(b["mlen"]), Tm)
        A = np.array(alignment(model, n_heads, tok, mel_in, src, self_m, src))  # (Tm,Tt)
        n = int(b["mlen"][0]); A = A[:n]
        assigned = A.argmax(axis=1)                      # token idx per mel frame
        dur = np.bincount(assigned, minlength=Tt).astype(np.int32)
        # safety: durations must sum to mel length
        if dur.sum() != n:
            dur[-1] += (n - dur.sum())
        np.save(out_dir / f"{it['id']}.npy", dur)
        n_ok += 1
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(items)} (last: Tt={Tt} sum_dur={dur.sum()} max_dur={dur.max()})", flush=True)

    print(f"done. wrote {n_ok} duration files to {out_dir}")


if __name__ == "__main__":
    main()
