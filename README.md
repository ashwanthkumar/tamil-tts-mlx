# tamil-tts (MLX)

Small, **CPU-friendly Tamil text-to-speech** — a single-speaker (female) voice trained on
**Apple Silicon with [MLX]**, exported to **ONNX**, with **Python** and **Rust** inference SDKs.
No GPU needed at inference; runs on any platform.

- **Acoustic model:** non-autoregressive, FastSpeech-style transformer (char-level Tamil → mel),
  trained from scratch in MLX on the Apple GPU.
- **Vocoder:** HiFi-GAN (mel → waveform) — natural, non-robotic audio.
- **~33 MB** acoustic model + 56 MB vocoder, 22.05 kHz.

## 🔊 Demo

https://github.com/ashwanthkumar/tamil-tts-mlx/raw/main/videos/intro.mp4

A short intro with sample Tamil sentences (on-screen text + synthesized voice). The clip is built
with Remotion — source in [`videos/intro/`](videos/intro/). Individual samples:
[demo1.mp4](samples/demo1.mp4) · [demo2.mp4](samples/demo2.mp4).

## Quick start

Download the model assets from the [latest release](https://github.com/ashwanthkumar/tamil-tts-mlx/releases/latest)
into `models/` (`tamil_ns.enc_dur.onnx`, `tamil_ns.decoder.onnx`, `hifigan.onnx`, `tamil_ns.tokenizer.json`), then:

**Python**
```bash
uv sync --extra train          # or: pip install onnxruntime numpy soundfile librosa
uv run python -m tamiltts.mlx.onnx_infer_ns -m models/tamil_ns --text "வணக்கம், இது தமிழ் பேச்சு." -o out.wav
```

**Rust**
```bash
cd rust
cargo run --release --example synthesize_ns -- "வணக்கம்" out.wav ../models/tamil_ns
```

The SDKs run `text → enc_dur → (length-regulate) → decoder → mel → hifigan → wav`, entirely on CPU.
(Griffin-Lim is a built-in fallback if `hifigan.onnx` is absent.)

## How it works / reproduce it

- **Architecture + step-by-step reproduction on any MLX Mac** (data → aligner → train → ONNX →
  vocoder → SDKs), plus TensorBoard/remote setup: [`docs/MLX_RUNBOOK.md`](docs/MLX_RUNBOOK.md).
- **Model card:** [`docs/model_card.md`](docs/model_card.md).

## Model & licenses

- Trained ~20k steps in MLX on a 32 GB Mac. Single-speaker female Tamil voice; not a speaker cloner.
- **Training data:** [IndicTTS Tamil] — CC-BY-4.0 + IIT Madras Indic TTS EULA, attribution required
  (see [`docs/DATASET_LICENSE.md`](docs/DATASET_LICENSE.md)).
- **Vocoder weights:** HiFi-GAN [`jaketae/hifigan-lj-v1`](https://huggingface.co/jaketae/hifigan-lj-v1) (MIT).
- Code: MIT (see `LICENSE`).

[MLX]: https://github.com/ml-explore/mlx
[IndicTTS Tamil]: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil
