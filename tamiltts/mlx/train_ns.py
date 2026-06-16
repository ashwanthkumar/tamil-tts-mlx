"""Train the non-autoregressive FastTTS on Tamil data (MLX, Apple GPU).

Requires per-utterance durations from the forced aligner (tamiltts.mlx.aligner). No AR loop -> no collapse.

    uv run python -m tamiltts.mlx.train_ns --data data/mlx --out runs_mlx_ns --steps 80000
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
from .model_ns import NSConfig, FastTTS
from .model import key_pad_mask


def peak_gb() -> float:
    fn = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return (fn() / 1e9) if fn else 0.0
    except Exception:
        return 0.0


DUR_W = 1.0
PITCH_W = 0.1
ENERGY_W = 0.1


def loss_fn(model, batch):
    tok = mx.array(batch["tok"]); tlen = mx.array(batch["tlen"])
    mlen = mx.array(batch["mlen"]); expand = mx.array(batch["expand_idx"])
    mel_t = mx.array(batch["mel"]); dur_t = mx.array(batch["dur"])
    pitch_t = mx.array(batch["pitch"]); energy_t = mx.array(batch["energy"])
    Tt, Tm = tok.shape[1], mel_t.shape[1]

    src_mask = key_pad_mask(tlen, Tt)
    dec_mask = key_pad_mask(mlen, Tm)
    # condition the decoder on ground-truth pitch/energy during training (teacher forcing)
    mel, mel_post, logdur, pitch_p, energy_p = model(tok, src_mask, expand, dec_mask, pitch_t, energy_t)

    fmask = (mx.arange(Tm)[None, :] < mlen[:, None]).astype(mx.float32)[:, :, None]
    denom = mx.maximum(fmask.sum() * mel_t.shape[2], 1.0)
    l_pre = (mx.abs(mel - mel_t) * fmask).sum() / denom
    l_post = (mx.abs(mel_post - mel_t) * fmask).sum() / denom

    tmask = (mx.arange(Tt)[None, :] < tlen[:, None]).astype(mx.float32)
    tdenom = mx.maximum(tmask.sum(), 1.0)
    logdur_t = mx.log(dur_t + 1.0)
    l_dur = (((logdur - logdur_t) ** 2) * tmask).sum() / tdenom
    l_pitch = (((pitch_p - pitch_t) ** 2) * tmask).sum() / tdenom
    l_energy = (((energy_p - energy_t) ** 2) * tmask).sum() / tdenom
    return l_pre + l_post + DUR_W * l_dur + PITCH_W * l_pitch + ENERGY_W * l_energy


def save_ckpt(model, opt, run_dir: Path, step: int):
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(run_dir / f"ckpt_{step:06d}.safetensors"), flat)
    mx.save_safetensors(str(run_dir / "latest.safetensors"), flat)
    try:
        opt_flat = {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)}
        mx.save_safetensors(str(run_dir / "latest_opt.safetensors"), opt_flat)
    except Exception:
        pass
    (run_dir / "latest_state.json").write_text(json.dumps({"step": step}))
    print(f"  saved ckpt_{step:06d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--out", type=Path, default=Path("runs_mlx_ns"))
    ap.add_argument("--run", default="tamil_ns")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    ap.add_argument("--warmup", type=int, default=3000, help="LR warmup steps (avoids mean-collapse)")
    args = ap.parse_args()

    from mlx.utils import tree_flatten, tree_unflatten
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    data = TTSData(args.data)
    cfg = NSConfig(vocab_size=len(data.vocab), d_model=args.d_model,
                   enc_layers=args.layers, dec_layers=args.layers)
    model = FastTTS(cfg)
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    # LR warmup + cosine decay + low weight decay: avoids the mean-collapse basin that a constant
    # high LR fell into (encoder went constant -> decoder predicted the average spectrum -> flat loss).
    wu = min(args.warmup, max(args.steps // 4, 1))
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu),
         optim.cosine_decay(args.lr, max(args.steps - wu, 1), args.lr * 0.1)],
        [wu])
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-6)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    run_dir = args.out / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"cfg": cfg.__dict__, "vocab_size": len(data.vocab)}, indent=2))

    start_step = 0
    if args.resume:
        rd = Path(args.resume)
        model.update(tree_unflatten(list(mx.load(str(rd / "latest.safetensors")).items())))
        mx.eval(model.parameters())
        sf = rd / "latest_state.json"
        if sf.exists():
            start_step = int(json.loads(sf.read_text()).get("step", 0))
        of = rd / "latest_opt.safetensors"
        if of.exists():
            opt.init(model.trainable_parameters())
            try:
                opt.state = tree_unflatten(list(mx.load(str(of)).items())); mx.eval(opt.state)
            except Exception as e:
                print(f"  WARN opt not restored: {e}", flush=True)
        print(f"[ns] RESUMED at step {start_step}", flush=True)

    writer = SummaryWriter(str(run_dir / "tb")) if SummaryWriter else None
    n_train = sum(1 for it in data.train if it["frames"] <= args.max_frames and data.has_durations(it["id"]))
    print(f"[ns] device={mx.default_device()} params={nparams/1e6:.2f}M train={n_train} "
          f"tb={'on' if writer else 'off'}", flush=True)

    def eval_val():
        model.eval(); tot, nb = 0.0, 0
        for vb in data.batches_ns("val", args.batch, shuffle=False, max_frames=args.max_frames):
            tot += float(loss_fn(model, vb)); nb += 1
        model.train(); return tot / max(nb, 1)

    rng = np.random.default_rng(0)
    step = start_step; t0 = time.time(); running = 0.0
    model.train()
    while step < args.steps:
        for batch in data.batches_ns("train", args.batch, shuffle=True, rng=rng, max_frames=args.max_frames):
            loss, grads = loss_and_grad(model, batch)
            grads = optim.clip_grad_norm(grads, 1.0)[0]
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            running += float(loss); step += 1

            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every; peak = peak_gb(); avg = running / args.log_every
                print(f"step {step:6d} | loss {avg:.4f} | {dt*1000:6.0f} ms/step | peak {peak:.2f} GB", flush=True)
                if writer:
                    writer.add_scalar("train/loss", avg, step)
                    writer.add_scalar("perf/ms_per_step", dt * 1000, step)
                running = 0.0; t0 = time.time()
            if step % args.save_every == 0:
                save_ckpt(model, opt, run_dir, step)
                vl = eval_val(); print(f"  val loss {vl:.4f}", flush=True)
                if writer:
                    writer.add_scalar("val/loss", vl, step)
            if step >= args.steps:
                break

    save_ckpt(model, opt, run_dir, step)
    print("done.")


if __name__ == "__main__":
    main()
