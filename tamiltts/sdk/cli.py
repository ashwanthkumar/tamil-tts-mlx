"""Command-line Tamil TTS.

    tamil-tts "வணக்கம், இது தமிழ் பேச்சு தொகுப்பு." -o hello.wav
    tamil-tts "..." -m models/tamil_female.onnx --length-scale 1.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .synthesizer import SynthesisOptions, TamilTTS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tamil-tts", description="Synthesize Tamil speech from text.")
    p.add_argument("text", help="Tamil text to speak")
    p.add_argument("-o", "--out", default="out.wav", help="output WAV path (default: out.wav)")
    p.add_argument(
        "-m",
        "--model",
        default="models/tamil_female.onnx",
        help="path to the ONNX model (tokenizer.json is found alongside it)",
    )
    p.add_argument("--noise-scale", type=float, default=0.667)
    p.add_argument("--length-scale", type=float, default=1.0, help=">1 slower, <1 faster")
    p.add_argument("--noise-scale-w", type=float, default=0.8)
    args = p.parse_args(argv)

    if not Path(args.model).exists():
        print(
            f"error: model not found at {args.model}\n"
            "Train and export first (see README), or pass -m <path-to.onnx>.",
            file=sys.stderr,
        )
        return 2

    tts = TamilTTS(args.model)
    opts = SynthesisOptions(
        noise_scale=args.noise_scale,
        length_scale=args.length_scale,
        noise_scale_w=args.noise_scale_w,
    )
    out = tts.save(args.text, args.out, opts)
    print(f"wrote {out} ({tts.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
