"""Train the MLX TransformerTTS on the prepared Tamil data — on the Apple GPU.

    uv run python -m tamiltts.mlx.train --data data/mlx --out runs_mlx [--steps N] [--batch 16]

Saves checkpoints + config under <out>/<run>/. Logs step time, loss, and peak GPU memory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .dataset import TTSData
from .model import TTSConfig, TransformerTTS, key_pad_mask, causal_mask


def peak_gb() -> float:
    fn = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return (fn() / 1e9) if fn else 0.0
    except Exception:
        return 0.0


# Std of Gaussian noise added to the teacher-forcing frames (set from --input_noise in main()).
# This teaches the decoder to recover from imperfect previous frames -> fixes autoregressive
# free-running collapse (exposure bias) while keeping inference deterministic (clean ONNX export).
_INPUT_NOISE = 0.0


def loss_fn(model, batch):
    tok = mx.array(batch["tok"])
    mel_in = mx.array(batch["mel_in"])
    if _INPUT_NOISE > 0.0:
        mel_in = mel_in + _INPUT_NOISE * mx.random.normal(mel_in.shape)
    mel_t = mx.array(batch["mel"])
    stop_t = mx.array(batch["stop"])
    tlen = mx.array(batch["tlen"])
    mlen = mx.array(batch["mlen"])
    Tt, Tm = tok.shape[1], mel_in.shape[1]

    src_mask = key_pad_mask(tlen, Tt)
    cross_mask = src_mask
    self_mask = causal_mask(Tm) + key_pad_mask(mlen, Tm)

    mel, mel_post, stop = model(tok, src_mask, mel_in, self_mask, cross_mask)

    fmask = (mx.arange(Tm)[None, :] < mlen[:, None]).astype(mx.float32)  # (B,T)
    m3 = fmask[:, :, None]
    denom = mx.maximum(fmask.sum() * mel_t.shape[2], 1.0)
    l_pre = (mx.abs(mel - mel_t) * m3).sum() / denom
    l_post = (mx.abs(mel_post - mel_t) * m3).sum() / denom
    bce = nn.losses.binary_cross_entropy(stop, stop_t, with_logits=True, reduction="none")
    l_stop = (bce * fmask).sum() / mx.maximum(fmask.sum(), 1.0)
    return l_pre + l_post + l_stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--out", type=Path, default=Path("runs_mlx"))
    ap.add_argument("--run", default="tamil_mlx")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", default="", help="run dir to resume from (loads weights+optimizer+step)")
    ap.add_argument("--input_noise", type=float, default=0.0,
                    help="std of Gaussian noise on teacher-forcing frames (fixes AR exposure bias)")
    args = ap.parse_args()

    global _INPUT_NOISE
    _INPUT_NOISE = args.input_noise

    from mlx.utils import tree_flatten
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    data = TTSData(args.data)
    cfg = TTSConfig(vocab_size=len(data.vocab), d_model=args.d_model,
                    enc_layers=args.layers, dec_layers=args.layers)
    model = TransformerTTS(cfg)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))

    opt = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    run_dir = args.out / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        from mlx.utils import tree_unflatten
        rd = Path(args.resume)
        w = mx.load(str(rd / "latest.safetensors"))
        model.update(tree_unflatten(list(w.items())))
        mx.eval(model.parameters())
        state_f = rd / "latest_state.json"
        if state_f.exists():
            start_step = int(json.loads(state_f.read_text()).get("step", 0))
        opt_f = rd / "latest_opt.safetensors"
        if opt_f.exists():
            # init optimizer state then overwrite with the saved moments for a seamless continue
            opt.init(model.trainable_parameters())
            try:
                opt.state = tree_unflatten(list(mx.load(str(opt_f)).items()))
                mx.eval(opt.state)
                print(f"[mlx-tts] resumed optimizer state from {opt_f.name}", flush=True)
            except Exception as e:
                print(f"[mlx-tts] WARN: optimizer state not restored ({e}); continuing with fresh moments", flush=True)
        print(f"[mlx-tts] RESUMED from {rd} at step {start_step}", flush=True)
    (run_dir / "config.json").write_text(json.dumps({
        "cfg": cfg.__dict__, "vocab_size": len(data.vocab), "lr": args.lr,
        "batch": args.batch, "n_params": int(nparams),
    }, indent=2))

    writer = SummaryWriter(str(run_dir / "tb")) if SummaryWriter else None

    def eval_val():
        model.eval()
        tot, nb = 0.0, 0
        for vb in data.batches("val", args.batch, shuffle=False, max_frames=args.max_frames):
            tot += float(loss_fn(model, vb)); nb += 1
        model.train()
        return tot / max(nb, 1)

    print(f"[mlx-tts] device={mx.default_device()} params={nparams/1e6:.2f}M "
          f"vocab={len(data.vocab)} train={len(data.train)} tb={'on' if writer else 'off'}", flush=True)

    rng = np.random.default_rng(0)
    step = start_step
    t0 = time.time()
    running = 0.0
    model.train()
    while step < args.steps:
        for batch in data.batches("train", args.batch, shuffle=True, rng=rng,
                                   max_frames=args.max_frames):
            loss, grads = loss_and_grad(model, batch)
            grads = optim.clip_grad_norm(grads, 1.0)[0]
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            running += float(loss)
            step += 1

            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every
                peak = peak_gb()
                avg = running / args.log_every
                print(f"step {step:6d} | loss {avg:.4f} | {dt*1000:6.0f} ms/step | "
                      f"peak {peak:.2f} GB", flush=True)
                if writer:
                    writer.add_scalar("train/loss", avg, step)
                    writer.add_scalar("perf/ms_per_step", dt * 1000, step)
                    writer.add_scalar("mem/peak_gb", peak, step)
                running = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                save_ckpt(model, opt, run_dir, step)
                vloss = eval_val()
                print(f"  val loss {vloss:.4f}", flush=True)
                if writer:
                    writer.add_scalar("val/loss", vloss, step)

            if step >= args.steps:
                break

    save_ckpt(model, opt, run_dir, step)
    print("done.")


def mx_tree_items(tree, prefix=""):
    """Flatten an MLX parameter pytree to (name, array) pairs."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from mx_tree_items(v, f"{prefix}.{k}")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from mx_tree_items(v, f"{prefix}.{i}")
    else:
        yield prefix, tree


def save_ckpt(model, opt, run_dir: Path, step: int):
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(model.parameters()))
    path = run_dir / f"ckpt_{step:06d}.safetensors"
    mx.save_safetensors(str(path), flat)
    # "latest" pointers + optimizer state + step, so --resume can continue seamlessly
    mx.save_safetensors(str(run_dir / "latest.safetensors"), flat)
    try:
        opt_flat = {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)}
        mx.save_safetensors(str(run_dir / "latest_opt.safetensors"), opt_flat)
    except Exception as e:
        print(f"  (opt state not saved: {e})", flush=True)
    (run_dir / "latest_state.json").write_text(json.dumps({"step": step}))
    print(f"  saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
