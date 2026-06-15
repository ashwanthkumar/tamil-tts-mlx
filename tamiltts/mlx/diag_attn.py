"""Inspect the AR model's cross-attention per (layer, head): is any head a clean monotonic diagonal?

Metrics per head: coverage (#distinct attended tokens / #tokens), monotonicity (fraction of frames
whose argmax >= previous), focus (mean max attention prob). A usable alignment head has all three high.
"""
import argparse, json
from pathlib import Path
import mlx.core as mx, mlx.nn as nn, numpy as np
from .dataset import TTSData
from .model import TTSConfig, TransformerTTS, key_pad_mask, causal_mask


def load_weights(model, path):
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(mx.load(str(path)).items()))); mx.eval(model.parameters()); model.eval()


def per_head_scores(model, n_heads, tok, mel_in, src, self_m):
    mem = model.encode(tok, src)
    x = mx.maximum(model.prenet1(mel_in), 0.0); x = mx.maximum(model.prenet2(x), 0.0)
    x = x + model._pe[: mel_in.shape[1]]
    out = []
    for layer in model.dec_layers:
        x1 = layer.n1(x + layer.self_attn(x, x, x, self_m))
        ca = layer.cross_attn
        q = ca.query_proj(x1); k = ca.key_proj(mem)
        B, L, D = q.shape; S = k.shape[1]
        q = q.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_heads, -1).transpose(0, 2, 3, 1)
        sc = mx.softmax((q * (1.0 / q.shape[-1]) ** 0.5) @ k, axis=-1)[0]  # (h,L,S)
        out.append(np.array(sc))
        x = layer.n2(x1 + layer.cross_attn(x1, mem, mem, src))
        x = layer.n3(x + layer.l2(nn.gelu(layer.l1(x))))
    return out  # list[layer] of (h, L, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs_mlx/tamil_mlx"))
    ap.add_argument("--weights", default="", help="specific .safetensors (default: <run>/latest)")
    args = ap.parse_args()
    data = TTSData("data/mlx")
    cfg = json.loads((args.run / "config.json").read_text())["cfg"]
    h = cfg["n_heads"]
    model = TransformerTTS(TTSConfig(**cfg))
    load_weights(model, args.weights or (args.run / "latest.safetensors"))

    it = data.train[0]; b = data._collate([it])
    tok = mx.array(b["tok"]); mel_in = mx.array(b["mel_in"]); Tt, Tm = tok.shape[1], mel_in.shape[1]
    src = key_pad_mask(mx.array(b["tlen"]), Tt); self_m = causal_mask(Tm) + key_pad_mask(mx.array(b["mlen"]), Tm)
    n = int(b["mlen"][0])
    layers = per_head_scores(model, h, tok, mel_in, src, self_m)
    print(f"clip frames={n} tokens={Tt}")
    print("layer head | coverage monotonic focus")
    best = None
    for li, sc in enumerate(layers):
        for hi in range(h):
            A = sc[hi, :n, :Tt]
            am = A.argmax(axis=1)
            cov = len(np.unique(am)) / Tt
            mono = np.mean(np.diff(am) >= 0)
            foc = A.max(axis=1).mean()
            score = cov * mono
            print(f"  L{li} H{hi} | cov {cov:.2f}  mono {mono:.2f}  focus {foc:.2f}")
            if best is None or score > best[0]:
                best = (score, li, hi, cov, mono, foc)
    print(f"BEST: layer {best[1]} head {best[2]} | cov {best[3]:.2f} mono {best[4]:.2f} focus {best[5]:.2f}")


if __name__ == "__main__":
    main()
