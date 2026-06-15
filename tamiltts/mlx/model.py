"""Shared MLX building blocks for the non-AR FastTTS (char -> 80-dim log-mel).

Holds the transformer encoder layer, sinusoidal positions, config, and masking reused
by `model_ns.py` (the FastTTS model) and the trainer/inference modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

NEG = -1e9


def sinusoid(max_len: int, d: int) -> mx.array:
    pos = mx.arange(max_len)[:, None]
    i = mx.arange(0, d, 2)[None, :]
    ang = pos / mx.power(10000.0, i / d)
    pe = mx.zeros((max_len, d))
    pe[:, 0::2] = mx.sin(ang)
    pe[:, 1::2] = mx.cos(ang)
    return pe


@dataclass
class TTSConfig:
    vocab_size: int = 64
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 4
    dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 2048
    postnet_dim: int = 256
    postnet_layers: int = 5


class EncLayer(nn.Module):
    def __init__(self, c: TTSConfig):
        super().__init__()
        self.attn = nn.MultiHeadAttention(c.d_model, c.n_heads)
        self.n1 = nn.LayerNorm(c.d_model)
        self.n2 = nn.LayerNorm(c.d_model)
        self.l1 = nn.Linear(c.d_model, c.d_ff)
        self.l2 = nn.Linear(c.d_ff, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x, mask):
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.n2(x + self.drop(self.l2(nn.gelu(self.l1(x)))))
        return x


def key_pad_mask(lengths, max_len):
    """Additive mask (B,1,1,max_len): 0 where valid, NEG where pad."""
    idx = mx.arange(max_len)[None, :]
    valid = idx < lengths[:, None]            # (B, max_len) bool
    return mx.where(valid, 0.0, NEG)[:, None, None, :]
