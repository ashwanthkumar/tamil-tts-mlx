"""Convert a pretrained LJSpeech HiFi-GAN checkpoint to a mel->wav ONNX vocoder.

Download the weights first (MIT-licensed, matches our 22.05kHz / hop256 / 80-mel front-end):
    curl -sL https://huggingface.co/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin \
        -o models/hifigan_lj.bin

Then:
    uv run python -m tamiltts.mlx.export_hifigan --weights models/hifigan_lj.bin --out models/hifigan.onnx
"""
from __future__ import annotations

import argparse

import torch

from .hifigan import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/hifigan_lj.bin")
    ap.add_argument("--out", default="models/hifigan.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    G = Generator()
    sd = torch.load(args.weights, map_location="cpu", weights_only=True)
    sd = sd if "conv_pre.bias" in sd else sd["generator"]
    missing, unexpected = G.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARN load: missing={len(missing)} unexpected={len(unexpected)}")
    G.remove_wn()
    G.eval()

    mel = torch.zeros(1, 80, 40)
    torch.onnx.export(G, (mel,), args.out, input_names=["mel"], output_names=["wav"],
                      dynamic_axes={"mel": {2: "T"}, "wav": {2: "S"}}, opset_version=args.opset)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
