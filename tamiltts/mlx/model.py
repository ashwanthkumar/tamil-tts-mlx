"""Compact TransformerTTS in MLX (char -> log-mel, autoregressive w/ cross-attention).

Training is teacher-forced and fully parallel over time (causal mask); only inference
is sequential. Designed to fit comfortably on a 32GB Apple Silicon GPU.
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


class DecLayer(nn.Module):
    def __init__(self, c: TTSConfig):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(c.d_model, c.n_heads)
        self.cross_attn = nn.MultiHeadAttention(c.d_model, c.n_heads)
        self.n1 = nn.LayerNorm(c.d_model)
        self.n2 = nn.LayerNorm(c.d_model)
        self.n3 = nn.LayerNorm(c.d_model)
        self.l1 = nn.Linear(c.d_model, c.d_ff)
        self.l2 = nn.Linear(c.d_ff, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x, mem, self_mask, cross_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, self_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, mem, mem, cross_mask)))
        x = self.n3(x + self.drop(self.l2(nn.gelu(self.l1(x)))))
        return x


class Postnet(nn.Module):
    """Conv1d stack (channels-last) producing a residual refinement of the mel."""
    def __init__(self, c: TTSConfig):
        super().__init__()
        dims = [c.n_mels] + [c.postnet_dim] * (c.postnet_layers - 1) + [c.n_mels]
        self.convs = [nn.Conv1d(dims[i], dims[i + 1], 5, padding=2) for i in range(c.postnet_layers)]
        self.norms = [nn.LayerNorm(dims[i + 1]) for i in range(c.postnet_layers)]
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x):  # x: (B, T, n_mels)
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h)
            h = norm(h)
            if i < len(self.convs) - 1:
                h = self.drop(mx.tanh(h))
        return x + h


class TransformerTTS(nn.Module):
    def __init__(self, c: TTSConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.enc_layers = [EncLayer(c) for _ in range(c.enc_layers)]
        self.prenet1 = nn.Linear(c.n_mels, c.d_model)
        self.prenet2 = nn.Linear(c.d_model, c.d_model)
        self.dec_layers = [DecLayer(c) for _ in range(c.dec_layers)]
        self.mel_out = nn.Linear(c.d_model, c.n_mels)
        self.stop_out = nn.Linear(c.d_model, 1)
        self.postnet = Postnet(c)
        self.drop = nn.Dropout(c.dropout)
        self._pe = sinusoid(c.max_len, c.d_model)
        self._scale = c.d_model ** 0.5

    def encode(self, tokens, src_mask):
        x = self.embed(tokens) * self._scale + self._pe[: tokens.shape[1]]
        x = self.drop(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, mel_in, mem, self_mask, cross_mask):
        x = nn.relu(self.prenet1(mel_in))
        x = self.drop(nn.relu(self.prenet2(x)))
        x = x + self._pe[: mel_in.shape[1]]
        for layer in self.dec_layers:
            x = layer(x, mem, self_mask, cross_mask)
        mel = self.mel_out(x)
        stop = self.stop_out(x)[..., 0]
        mel_post = self.postnet(mel)
        return mel, mel_post, stop

    def __call__(self, tokens, src_mask, mel_in, self_mask, cross_mask):
        mem = self.encode(tokens, src_mask)
        return self.decode(mel_in, mem, self_mask, cross_mask)


def key_pad_mask(lengths, max_len):
    """Additive mask (B,1,1,max_len): 0 where valid, NEG where pad."""
    idx = mx.arange(max_len)[None, :]
    valid = idx < lengths[:, None]            # (B, max_len) bool
    return mx.where(valid, 0.0, NEG)[:, None, None, :]


def causal_mask(t):
    m = mx.triu(mx.full((t, t), NEG), k=1)
    return m[None, None, :, :]
