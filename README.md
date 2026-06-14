# tamil-tts

Small, fast, **CPU-friendly Tamil text-to-speech**. A single-speaker (female) [VITS]
model trained on the [IndicTTS Tamil] corpus and exported to **ONNX** for fast inference,
with both a **Python** and a **Rust** SDK.

Design goals: tiny model (~15–30 MB), real-time-or-faster on CPU, no GPU needed at inference.

> Status: **pipeline validated end-to-end on an M2** — install, a real VITS training step on MPS,
> ONNX export, and both SDKs were smoke-tested with `tests/make_synthetic_dataset.py`. What
> remains is the *real* training run to convergence (a long job you drive on the M1 Studio).

[VITS]: https://arxiv.org/abs/2106.06103
[IndicTTS Tamil]: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil

## Why this stack

| Requirement            | Choice                                   |
| ---------------------- | ---------------------------------------- |
| Small + fast on CPU    | VITS (single-speaker) → ONNX             |
| Tamil phonemization    | `espeak-ng` (`ta`)                       |
| Train on Apple Silicon | PyTorch **MPS** (M2 GPU / M1 Studio GPU) |
| Easy, fast inference   | `onnxruntime` (Python) + `ort` (Rust)    |

**On MLX:** PyTorch has no MLX backend — Metal GPU acceleration on Apple Silicon comes from the
**MPS** backend, which is what we use. MLX would forfeit clean ONNX export and a stable Rust SDK,
so it's documented as an optional future track only (`docs/mlx.md`), not the default.

## Hardware plan

- **Training:** Mac Studio (M1 Max/Ultra, 32 GB) via MPS — faster GPU, bigger batches.
- **Inference:** MacBook Air M2 (or any CPU) via ONNX — the model is portable.

## Layout

```
tamiltts/            Python package
  data/prepare.py    IndicTTS Tamil (parquet) -> 22.05kHz wavs + metadata (female speaker)
  phonemize.py       Tamil text -> espeak-ng phoneme ids
  train.py           VITS training wrapper (Coqui-TTS, MPS-aware)
  export_onnx.py     trained checkpoint -> models/tamil_female.onnx
  sdk/               onnxruntime inference SDK + CLI
configs/             training + model config
rust/                Rust SDK crate (ort + onnxruntime)
scripts/             01_prepare_data.sh, 02_train.sh, 03_export.sh
docs/                dataset license/attribution, model card, MLX notes
```

## Quickstart

### 0. Environment

```bash
uv sync --extra train          # full env for data prep + training (use on the Studio)
# or, inference-only (on the Air):
uv sync
brew install espeak-ng         # required for Tamil phonemization
```

### 1. Prepare data

```bash
uv run python -m tamiltts.data.prepare --speaker female --out data
```

### 2. Train (on the M1 Studio)

```bash
uv run python -m tamiltts.train --config configs/tamil_female_vits.json
```

### 3. Export to ONNX

```bash
uv run python -m tamiltts.export_onnx --run runs/tamil_female --out models/tamil_female.onnx
```

### 4. Synthesize

Python:
```bash
uv run tamil-tts "வணக்கம், இது தமிழ் பேச்சு தொகுப்பு." -o hello.wav
```

Rust:
```bash
cd rust && cargo run --release --example synthesize -- "வணக்கம்" hello.wav
```

## License

Code: MIT (`LICENSE`). Data & trained model: CC-BY-4.0 + IIT Madras Indic TTS EULA —
see [`docs/DATASET_LICENSE.md`](docs/DATASET_LICENSE.md). Attribution is required.
