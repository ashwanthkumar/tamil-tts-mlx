"""Export the non-AR FastTTS (MLX) to portable ONNX — single forward, no AR loop.

Two graphs (length regulation is integer repeat, done host-side between them):
  enc_dur.onnx : tokens[1,Tt] int64 -> enc[1,Tt,d] float32, log_dur[1,Tt] float32
  decoder.onnx : enc[1,Tt,d], expand_idx[1,Tm] int64 -> mel_post[1,Tm,80]
SDK: run enc_dur -> dur=round(exp(log_dur)-1) -> expand_idx=repeat(arange(Tt),dur) -> decoder -> Griffin-Lim.

PyTorch twin mirrors model_ns.py exactly; weights ported 1:1 from MLX safetensors.

    uv run python -m tamiltts.mlx.export_onnx_ns --run runs_mlx_ns/tamil_ns2 --out models/tamil_ns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .export_onnx import MHA, EncLayer, Postnet, sinusoid  # reuse the verified torch twins


class DurPred(nn.Module):
    def __init__(self, d, k=3):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv1d(d, d, k, padding=p); self.c2 = nn.Conv1d(d, d, k, padding=p)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)

    def forward(self, x):  # (B,T,d)
        h = self.n1(F.relu(self.c1(x.transpose(1, 2)).transpose(1, 2)))
        h = self.n2(F.relu(self.c2(h.transpose(1, 2)).transpose(1, 2)))
        return self.proj(h)[..., 0]


class TorchFastTTS(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        d, h, ff = c["d_model"], c["n_heads"], c["d_ff"]
        self.n_mels = c["n_mels"]; self.scale = d ** 0.5
        self.embed = nn.Embedding(c["vocab_size"], d)
        self.enc_layers = nn.ModuleList([EncLayer(d, h, ff) for _ in range(c["enc_layers"])])
        self.dur = DurPred(d, c.get("dur_kernel", 3))
        self.dec_layers = nn.ModuleList([EncLayer(d, h, ff) for _ in range(c["dec_layers"])])
        self.mel_out = nn.Linear(d, c["n_mels"])
        self.postnet = Postnet(c["n_mels"], c["postnet_dim"], c["postnet_layers"])
        self.register_buffer("pe", sinusoid(c["max_len"], d), persistent=False)

    def encode(self, tokens):
        x = self.embed(tokens) * self.scale + self.pe[: tokens.shape[1]]
        for l in self.enc_layers:
            x = l(x, None)
        return x

    def enc_dur(self, tokens):
        enc = self.encode(tokens)
        return enc, self.dur(enc)

    def decode(self, enc, expand_idx):
        d = enc.shape[2]
        idx = expand_idx.unsqueeze(-1).expand(-1, -1, d)        # (B,Tm,d)
        expanded = torch.gather(enc, 1, idx)
        x = expanded + self.pe[: expanded.shape[1]]
        for l in self.dec_layers:
            x = l(x, None)
        return self.postnet(self.mel_out(x))


def port(model: TorchFastTTS, w: dict):
    sd = {}
    for k, v in w.items():
        t = torch.from_numpy(np.array(v))
        if (".convs." in k or k.startswith("dur.c")) and k.endswith(".weight") and t.ndim == 3:
            t = t.permute(0, 2, 1).contiguous()                # MLX Conv1d (O,K,I) -> torch (O,I,K)
        sd[k] = t
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/tamil_ns"))
    ap.add_argument("--data", type=Path, default=Path("data/mlx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())["cfg"]
    model = TorchFastTTS(cfg).eval()
    from safetensors.numpy import load_file
    miss, unexp = port(model, load_file(str(args.run / "latest.safetensors")))
    if miss: print("WARN missing:", miss[:6])
    if unexp: print("WARN unexpected:", unexp[:6])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc_path = str(args.out) + ".enc_dur.onnx"
    dec_path = str(args.out) + ".decoder.onnx"
    d = cfg["d_model"]

    tokens = torch.zeros(1, 12, dtype=torch.long)
    torch.onnx.export(model, (tokens,), enc_path,
                      input_names=["tokens"], output_names=["enc", "log_dur"],
                      dynamic_axes={"tokens": {1: "Tt"}, "enc": {1: "Tt"}, "log_dur": {1: "Tt"}},
                      opset_version=args.opset,
                      # export only enc_dur path
                      )
    print(f"exported {enc_path}")
    # decoder graph
    class Dec(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, enc, expand_idx): return self.m.decode(enc, expand_idx)
    enc_ex = torch.zeros(1, 12, d); idx_ex = torch.zeros(1, 30, dtype=torch.long)
    torch.onnx.export(Dec(model), (enc_ex, idx_ex), dec_path,
                      input_names=["enc", "expand_idx"], output_names=["mel_post"],
                      dynamic_axes={"enc": {1: "Tt"}, "expand_idx": {1: "Tm"}, "mel_post": {1: "Tm"}},
                      opset_version=args.opset)
    print(f"exported {dec_path}")

    # tokenizer + mel stats for the SDKs (vocab, normalization, audio params)
    stats = json.loads((args.data / "stats.json").read_text())
    payload = {
        "vocab": json.loads((args.data / "vocab.json").read_text(encoding="utf-8")),
        "mel_mean": stats["mel_mean"], "mel_std": stats["mel_std"],
        "audio": {"sr": 22050, "n_fft": 1024, "hop": 256, "win": 1024, "n_mels": cfg["n_mels"], "fmin": 0, "fmax": 8000},
    }
    tok_out = str(args.out) + ".tokenizer.json"
    Path(tok_out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {tok_out}")


# NOTE: model.forward must return the enc_dur outputs for the first export.
def _patch_forward():
    TorchFastTTS.forward = lambda self, tokens: self.enc_dur(tokens)


_patch_forward()

if __name__ == "__main__":
    main()
