"""PyTorch twin building blocks that mirror the MLX model, used for ONNX export.

MLX has no native ONNX export, so we mirror the model in PyTorch with identical math
and port the trained MLX weights 1:1. These layers are the verified twins reused by the
non-AR exporter (`tamiltts.mlx.export_onnx_ns`); see it for the actual export entrypoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
