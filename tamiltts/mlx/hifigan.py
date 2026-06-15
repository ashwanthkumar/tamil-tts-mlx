"""HiFi-GAN V1 generator (PyTorch) — neural vocoder to replace Griffin-Lim (de-robotize).

Standard jik876 V1 architecture. hop = prod(upsample_rates) = 8*8*2*2 = 256, 80 mels — matches our
mel front-end exactly, so it consumes our model's denormalized log-mel directly. Used to (a) load
pretrained LJSpeech weights and (b) export a mel->wav ONNX graph for the SDKs.

Input:  mel (B, 80, T)   log-mel = log(clip(mel, 1e-5))  (same as tamiltts.mlx.audio)
Output: wav (B, 1, T*256) in [-1, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU = 0.1


class ResBlock(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=(k - 1) * d // 2)) for d in dilations])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1, padding=(k - 1) // 2)) for _ in dilations])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = x + xt
        return x

    def remove_wn(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4),
                 upsample_initial_channel=512, resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_ups = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, 3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(self.num_ups):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))

    def forward(self, x):  # x: (B, n_mels, T)
        x = self.conv_pre(x)
        for i in range(self.num_ups):
            x = self.ups[i](F.leaky_relu(x, LRELU))
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)

    def remove_wn(self):
        for u in self.ups:
            remove_weight_norm(u)
        for r in self.resblocks:
            r.remove_wn()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
