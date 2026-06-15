"""Collapse check for the non-AR model: is the encoder differentiating tokens and the decoder
producing time-varying mel (vs the mean-collapse where both go constant)?

    uv run python -m tamiltts.mlx.diag_ns --run runs_mlx_ns/tamil_ns2
"""
import argparse
from pathlib import Path
import numpy as np, mlx.core as mx
from .dataset import TTSData
from .infer_ns import load_model
from .model_ns import gather_expand
from .model import key_pad_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    args = ap.parse_args()
    data = TTSData(args.data); m = load_model(args.run)
    it = next(x for x in data.val if data.has_durations(x["id"]))
    b = data._collate_ns([it])
    tok = mx.array(b["tok"]); expand = mx.array(b["expand_idx"]); Tt, Tm = tok.shape[1], expand.shape[1]
    src = key_pad_mask(mx.array(b["tlen"]), Tt); dec = key_pad_mask(mx.array(b["mlen"]), Tm)
    enc = m.encode(tok, src)
    e = np.array(enc[0])[: int(b["tlen"][0])]
    mel, mel_post, _ = m(tok, src, expand, dec)
    n = int(b["mlen"][0]); pm = np.array(mel_post[0])[:n]; tg = b["mel"][0][:n]
    enc_d = float(np.abs(np.diff(e, axis=0)).mean())
    mel_d = float(np.abs(np.diff(pm, axis=0)).mean())
    print(f"encoder token-delta {enc_d:.3f} (collapse if ~0; healthy >0.3)")
    print(f"decoder mel frame-delta {mel_d:.3f} (collapse if ~0; target {float(np.abs(np.diff(tg,axis=0)).mean()):.3f})")
    print(f"pred std {pm.std():.2f} vs target std {tg.std():.2f}  | mel L1 {np.abs(pm-tg).mean():.3f}")
    print("VERDICT:", "HEALTHY (learning)" if (enc_d > 0.1 and mel_d > 0.05) else "MEAN-COLLAPSE")


if __name__ == "__main__":
    main()
