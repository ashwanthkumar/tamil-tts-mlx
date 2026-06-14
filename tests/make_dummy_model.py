"""Build a tiny dummy ONNX model that honors the tamil-tts I/O contract.

It ignores the phoneme ids and emits a fixed 0.5s sine tone, but it exercises the *exact*
input/output signature the real VITS export uses. This lets you smoke-test both SDKs
(Python and Rust) before a real model has finished training.

    uv run --with onnx python tests/make_dummy_model.py
    uv run tamil-tts "வணக்கம்" -m models/dummy.onnx -o /tmp/dummy.wav
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SR = 22050
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "dummy.onnx"


def main() -> None:
    t = np.linspace(0, 0.5, SR // 2, endpoint=False, dtype=np.float32)
    sine = (0.3 * np.sin(2 * np.pi * 220.0 * t)).reshape(1, 1, -1)
    wav_const = numpy_helper.from_array(sine, name="wav_const")

    # Declare the contract inputs even though the dummy ignores them.
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3]),
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, SR // 2])

    node = helper.make_node("Identity", ["wav_const"], ["output"])
    graph = helper.make_graph([node], "dummy_tamil_tts", inputs, [output], initializer=[wav_const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))

    # Minimal tokenizer.json so the SDK can load it.
    tok = {
        "language": "ta",
        "add_blank": True,
        "sample_rate": SR,
        "blank_id": 0,
        "pad_id": 0,
        "bos_id": None,
        "eos_id": None,
        "ignore": [" "],
        "id_map": {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzˈʌɳkmidʉtɻ")},
    }
    OUT.with_suffix(".tokenizer.json").write_text(
        json.dumps(tok, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT} and {OUT.with_suffix('.tokenizer.json')}")


if __name__ == "__main__":
    main()
