"""Diagnostic: measure free-running AR collapse (frame-energy over time) for the current model."""
import numpy as np, mlx.core as mx
from pathlib import Path
from tamiltts.mlx.dataset import TTSData
from tamiltts.mlx.infer import load_model
from tamiltts.mlx.model import key_pad_mask, causal_mask

data = TTSData("data/mlx")
m = load_model(Path("runs_mlx/tamil_mlx")); m.eval()
it = data.val[0]
tok = mx.array([data.encode_text(it["text"])], dtype=mx.int32); T = tok.shape[1]
s = key_pad_mask(mx.array([T]), T); mem = m.encode(tok, s)
mi = mx.zeros((1, 1, 80)); fr = []; stops = []
for i in range(300):
    Tm = mi.shape[1]
    _, mpp, st = m.decode(mi, mem, causal_mask(Tm), s)
    last = mpp[:, -1:, :]
    fr.append(np.array(last[0, 0])); stops.append(float(mx.sigmoid(st[0, -1])))
    mi = mx.concatenate([mi, last], axis=1)
fs = np.array(fr).std(axis=1)
print("FREE-RUN (deterministic) frame-energy(std) over 300 frames:")
print(f"  first20={fs[:20].mean():.2f}  mid(100-150)={fs[100:150].mean():.2f}  last100={fs[-100:].mean():.2f}")
print("  BASELINE pre-fix(210k): first~0.42 -> last100 0.17 (collapsed)")
print(f"  stop-prob max: {max(stops):.2f} -> {'FIRES' if max(stops) > 0.5 else 'not firing'}")
