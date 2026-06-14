"""ONNX inference for the Tamil VITS model.

Lightweight: depends only on numpy + onnxruntime + soundfile (and espeak-ng on PATH).

    from tamiltts.sdk import TamilTTS
    tts = TamilTTS("models/tamil_female.onnx")
    tts.save("வணக்கம்", "hello.wav")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..phonemize import Tokenizer


@dataclass
class SynthesisOptions:
    """VITS inference knobs (mapped to the model's `scales` input)."""

    noise_scale: float = 0.667     # voice variability
    length_scale: float = 1.0      # >1 slower, <1 faster
    noise_scale_w: float = 0.8     # duration variability


class TamilTTS:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path | None = None):
        import onnxruntime as ort

        model_path = Path(model_path)
        if tokenizer_path is None:
            tokenizer_path = model_path.with_suffix(".tokenizer.json")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.sample_rate = _read_sample_rate(tokenizer_path, default=22050)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # let ORT pick (uses all CPU cores)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def synthesize(self, text: str, options: SynthesisOptions | None = None) -> np.ndarray:
        """Return a float32 waveform in [-1, 1] at ``self.sample_rate``."""
        options = options or SynthesisOptions()
        ids = self.tokenizer.encode(text)
        if not ids:
            raise ValueError("Text produced no phonemes; check input and espeak-ng language.")

        x = np.asarray([ids], dtype=np.int64)
        x_len = np.asarray([x.shape[1]], dtype=np.int64)
        scales = np.asarray(
            [options.noise_scale, options.length_scale, options.noise_scale_w],
            dtype=np.float32,
        )

        feeds = {"input": x, "input_lengths": x_len, "scales": scales}
        # Some exports name the speaker input "sid"; single-speaker models accept None/0.
        if "sid" in self._input_names:
            feeds["sid"] = np.asarray([0], dtype=np.int64)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        out = self.session.run(None, feeds)[0]
        wav = np.asarray(out, dtype=np.float32).reshape(-1)
        # Guard against clipping from high noise scales.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        return wav

    def save(self, text: str, out_path: str | Path, options: SynthesisOptions | None = None) -> Path:
        import soundfile as sf

        wav = self.synthesize(text, options)
        out_path = Path(out_path)
        sf.write(str(out_path), wav, self.sample_rate, subtype="PCM_16")
        return out_path


def _read_sample_rate(tokenizer_path: str | Path, default: int) -> int:
    import json

    try:
        data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
        return int(data.get("sample_rate", default))
    except Exception:
        return default
