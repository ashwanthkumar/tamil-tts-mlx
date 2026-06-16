"""Non-autoregressive FastSpeech-2-style TTS in MLX (no AR loop -> cannot collapse).

text -> encoder -> {duration, pitch, energy} predictors -> add pitch/energy embeddings ->
length regulator (expand by durations) -> non-causal decoder -> mel (+postnet). Alignment/length
is explicit (durations), so there is no exposure bias and no stop token. Single forward -> clean ONNX.

Variance adaptors (v0.2): pitch + energy are predicted per token and embedded (Conv1d 1->d) into the
encoder states before length regulation. Training conditions on ground-truth pitch/energy; inference
uses the predictions, with optional scale knobs (FastPitch-style continuous conditioning).
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .model import EncLayer, sinusoid, key_pad_mask


@dataclass
class NSConfig:
    vocab_size: int = 64
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 4
    dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 4096
    postnet_dim: int = 256
    postnet_layers: int = 5
    dur_kernel: int = 3


class VariancePredictor(nn.Module):
    """2-conv predictor used for duration (log-frames), pitch, and energy. Returns (B,T)."""

    def __init__(self, c: NSConfig):
        super().__init__()
        p = c.dur_kernel // 2
        self.c1 = nn.Conv1d(c.d_model, c.d_model, c.dur_kernel, padding=p)
        self.c2 = nn.Conv1d(c.d_model, c.d_model, c.dur_kernel, padding=p)
        self.n1 = nn.LayerNorm(c.d_model)
        self.n2 = nn.LayerNorm(c.d_model)
        self.proj = nn.Linear(c.d_model, 1)
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x):  # x: (B,T,d) channels-last
        h = self.drop(self.n1(mx.maximum(self.c1(x), 0.0)))
        h = self.drop(self.n2(mx.maximum(self.c2(h), 0.0)))
        return self.proj(h)[..., 0]  # (B,T)


class Postnet(nn.Module):
    def __init__(self, c: NSConfig):
        super().__init__()
        dims = [c.n_mels] + [c.postnet_dim] * (c.postnet_layers - 1) + [c.n_mels]
        self.convs = [nn.Conv1d(dims[i], dims[i + 1], 5, padding=2) for i in range(c.postnet_layers)]
        self.norms = [nn.LayerNorm(dims[i + 1]) for i in range(c.postnet_layers)]
        self.drop = nn.Dropout(c.dropout)

    def __call__(self, x):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(h))
            if i < len(self.convs) - 1:
                h = self.drop(mx.tanh(h))
        return x + h


def gather_expand(enc, idx):
    """enc (B,T_text,d), idx (B,T_mel) int -> (B,T_mel,d) gathered along axis=1."""
    B, _, d = enc.shape
    Tm = idx.shape[1]
    idx_e = mx.broadcast_to(idx[:, :, None], (B, Tm, d))
    return mx.take_along_axis(enc, idx_e, axis=1)


class FastTTS(nn.Module):
    def __init__(self, c: NSConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.enc_layers = [EncLayer(_as_tts(c)) for _ in range(c.enc_layers)]
        self.dur = VariancePredictor(c)
        self.pitch_pred = VariancePredictor(c)
        self.energy_pred = VariancePredictor(c)
        # continuous conditioning: project the (normalized) scalar contour to d (FastPitch-style)
        self.pitch_emb = nn.Conv1d(1, c.d_model, c.dur_kernel, padding=c.dur_kernel // 2)
        self.energy_emb = nn.Conv1d(1, c.d_model, c.dur_kernel, padding=c.dur_kernel // 2)
        self.dec_layers = [EncLayer(_as_tts(c)) for _ in range(c.dec_layers)]
        self.mel_out = nn.Linear(c.d_model, c.n_mels)
        self.postnet = Postnet(c)
        self.drop = nn.Dropout(c.dropout)
        self._pe = sinusoid(c.max_len, c.d_model)
        self._scale = c.d_model ** 0.5

    def encode(self, tok, src_mask):
        x = self.embed(tok) * self._scale + self._pe[: tok.shape[1]]
        x = self.drop(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, expanded, dec_mask):
        x = expanded + self._pe[: expanded.shape[1]]
        for layer in self.dec_layers:
            x = layer(x, dec_mask)
        mel = self.mel_out(x)
        return mel, self.postnet(mel)

    def variance(self, enc, pitch_gt=None, energy_gt=None, p_scale=1.0, e_scale=1.0,
                 p_mean=0.0, p_std=1.0, e_mean=0.0, e_std=1.0):
        """Predict pitch/energy and add their embeddings to enc (before length regulation).

        Conditions on ground truth when given (training); else on the predictions (inference).
        The scale knobs are applied in REAL (denormalized) space so they're intuitive — p_scale=1.3
        means ~30% higher F0 — by denormalizing with (mean,std), multiplying, then renormalizing.
        With the default mean=0/std=1/scale=1 this is a no-op (training path is unaffected).
        Returns (conditioned_enc, pitch_pred, energy_pred). Inputs/targets are normalized scalars.
        """
        pitch_p = self.pitch_pred(enc)              # (B,T) normalized
        energy_p = self.energy_pred(enc)
        pv = pitch_gt if pitch_gt is not None else pitch_p
        ev = energy_gt if energy_gt is not None else energy_p
        pv = ((pv * p_std + p_mean) * p_scale - p_mean) / p_std
        ev = ((ev * e_std + e_mean) * e_scale - e_mean) / e_std
        cond = enc + self.pitch_emb(pv[..., None]) + self.energy_emb(ev[..., None])
        return cond, pitch_p, energy_p

    def __call__(self, tok, src_mask, expand_idx, dec_mask, pitch=None, energy=None):
        enc = self.encode(tok, src_mask)
        logdur = self.dur(enc)
        cond, pitch_p, energy_p = self.variance(enc, pitch, energy)
        expanded = gather_expand(cond, expand_idx)
        mel, mel_post = self.decode(expanded, dec_mask)
        return mel, mel_post, logdur, pitch_p, energy_p


# EncLayer expects a TTSConfig-like object with d_model/n_heads/d_ff/dropout.
def _as_tts(c: NSConfig):
    from .model import TTSConfig
    return TTSConfig(d_model=c.d_model, n_heads=c.n_heads, d_ff=c.d_ff, dropout=c.dropout)
