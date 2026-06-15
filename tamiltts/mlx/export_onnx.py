"""Export an MLX-trained TransformerTTS to a portable ONNX graph.

MLX has no native ONNX export, so we mirror the model in PyTorch with identical math,
port the trained MLX weights 1:1, and torch.onnx.export a single forward graph:

    (tokens [1,Tt] int64, mel_in [1,Tm,80] float32) -> (mel_post [1,Tm,80], stop [1,Tm])

The SDKs run the short autoregressive loop host-side (encode is cheap; re-run each step),
then Griffin-Lim vocodes mel_post -> wav. Masking matches inference (batch=1, unpadded):
encoder/cross unmasked, decoder causal.

    uv run python -m tamiltts.mlx.export_onnx --run runs_mlx/tamil_mlx --out models/tamil_mlx.onnx
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- PyTorch twin (mirrors tamiltts/mlx/model.py) ---------------------------

def sinusoid(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    i = torch.arange(0, d, 2).unsqueeze(0).float()
    ang = pos / torch.pow(torch.tensor(10000.0), i / d)
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class MHA(nn.Module):
    """Matches mlx.nn.MultiHeadAttention (bias=False, scale=1/sqrt(head_dim))."""
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.query_proj = nn.Linear(d, d, bias=False)
        self.key_proj = nn.Linear(d, d, bias=False)
        self.value_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, q, k, v, mask=None):
        B, L, D = q.shape
        S = k.shape[1]
        q = self.query_proj(q).reshape(B, L, self.h, -1).transpose(1, 2)
        k = self.key_proj(k).reshape(B, S, self.h, -1).permute(0, 2, 3, 1)
        v = self.value_proj(v).reshape(B, S, self.h, -1).transpose(1, 2)
        scale = math.sqrt(1.0 / q.shape[-1])
        scores = (q * scale) @ k
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores, dim=-1)
        out = (scores @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class EncLayer(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.attn = MHA(d, h)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, ff); self.l2 = nn.Linear(ff, d)

    def forward(self, x, mask):
        x = self.n1(x + self.attn(x, x, x, mask))
        x = self.n2(x + self.l2(F.gelu(self.l1(x))))
        return x


class DecLayer(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.self_attn = MHA(d, h); self.cross_attn = MHA(d, h)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, ff); self.l2 = nn.Linear(ff, d)

    def forward(self, x, mem, self_mask, cross_mask):
        x = self.n1(x + self.self_attn(x, x, x, self_mask))
        x = self.n2(x + self.cross_attn(x, mem, mem, cross_mask))
        x = self.n3(x + self.l2(F.gelu(self.l1(x))))
        return x


class Postnet(nn.Module):
    def __init__(self, n_mels, dim, layers):
        super().__init__()
        dims = [n_mels] + [dim] * (layers - 1) + [n_mels]
        self.convs = nn.ModuleList([nn.Conv1d(dims[i], dims[i + 1], 5, padding=2) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(layers)])

    def forward(self, x):  # x: (B, T, n_mels), channels-last like MLX
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h.transpose(1, 2)).transpose(1, 2)  # (B,T,C)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = torch.tanh(h)
        return x + h


class TorchTTS(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        d, h, ff = c["d_model"], c["n_heads"], c["d_ff"]
        self.n_mels = c["n_mels"]
        self.scale = d ** 0.5
        self.embed = nn.Embedding(c["vocab_size"], d)
        self.enc_layers = nn.ModuleList([EncLayer(d, h, ff) for _ in range(c["enc_layers"])])
        self.prenet1 = nn.Linear(c["n_mels"], d)
        self.prenet2 = nn.Linear(d, d)
        self.dec_layers = nn.ModuleList([DecLayer(d, h, ff) for _ in range(c["dec_layers"])])
        self.mel_out = nn.Linear(d, c["n_mels"])
        self.stop_out = nn.Linear(d, 1)
        self.postnet = Postnet(c["n_mels"], c["postnet_dim"], c["postnet_layers"])
        self.register_buffer("pe", sinusoid(c["max_len"], d), persistent=False)

    def forward(self, tokens, mel_in):
        x = self.embed(tokens) * self.scale + self.pe[: tokens.shape[1]]
        for layer in self.enc_layers:
            x = layer(x, None)
        mem = x
        Tm = mel_in.shape[1]
        cm = torch.full((Tm, Tm), -1e9, device=mel_in.device)
        cm = torch.triu(cm, diagonal=1).view(1, 1, Tm, Tm)
        y = F.relu(self.prenet1(mel_in))
        y = F.relu(self.prenet2(y))
        y = y + self.pe[:Tm]
        for layer in self.dec_layers:
            y = layer(y, mem, cm, None)
        mel = self.mel_out(y)
        stop = self.stop_out(y)[..., 0]
        mel_post = self.postnet(mel)
        return mel, mel_post, stop


# --------------------------- weight porting ---------------------------

def load_mlx_weights(run_dir: Path) -> dict:
    from safetensors.numpy import load_file
    return load_file(str(run_dir / "latest.safetensors"))


def port_weights(model: TorchTTS, w: dict):
    """Copy MLX (numpy) weights into the torch twin. Conv1d needs (O,K,I)->(O,I,K)."""
    sd = {}
    for k, v in w.items():
        t = torch.from_numpy(np.array(v))
        if ".convs." in k and k.endswith(".weight") and t.ndim == 3:
            t = t.permute(0, 2, 1).contiguous()  # MLX (O,K,I) -> torch (O,I,K)
        sd[k] = t
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/tamil_mlx.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())["cfg"]
    model = TorchTTS(cfg).eval()
    w = load_mlx_weights(args.run)
    missing, unexpected = port_weights(model, w)
    if missing:
        print("WARNING missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("WARNING unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokens = torch.zeros(1, 12, dtype=torch.long)
    mel_in = torch.zeros(1, 20, cfg["n_mels"], dtype=torch.float32)
    torch.onnx.export(
        model, (tokens, mel_in), str(args.out),
        input_names=["tokens", "mel_in"],
        output_names=["mel", "mel_post", "stop"],
        dynamic_axes={"tokens": {1: "Tt"}, "mel_in": {1: "Tm"},
                      "mel": {1: "Tm"}, "mel_post": {1: "Tm"}, "stop": {1: "Tm"}},
        opset_version=args.opset,
    )
    print(f"exported {args.out}")

    # copy tokenizer (vocab) + mel stats next to the model for the SDKs
    import shutil
    data_dir = Path("data/mlx")
    tok_out = args.out.with_suffix(".tokenizer.json")
    from .audio import _mel_basis  # (n_mels, 1+n_fft/2)
    mel_inv = np.linalg.pinv(_mel_basis).astype(np.float32)  # (1+n_fft/2, n_mels) for mel->linear
    payload = {
        "vocab": json.loads((data_dir / "vocab.json").read_text(encoding="utf-8")),
        "mel_mean": json.loads((data_dir / "stats.json").read_text())["mel_mean"],
        "mel_std": json.loads((data_dir / "stats.json").read_text())["mel_std"],
        "audio": {"sr": 22050, "n_fft": 1024, "hop": 256, "win": 1024, "n_mels": cfg["n_mels"],
                   "fmin": 0, "fmax": 8000},
        "mel_inv": mel_inv.tolist(),  # pseudo-inverse mel filterbank for the Rust Griffin-Lim
    }
    tok_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {tok_out}")


if __name__ == "__main__":
    main()
