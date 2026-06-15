# Model card: tamil-tts (female, non-AR FastTTS + HiFi-GAN)

- **Task:** Tamil text-to-speech (single-speaker, female)
- **Acoustic model:** non-autoregressive FastSpeech-2-style transformer (char-level Tamil → 80-dim
  log-mel), trained from scratch in **MLX on Apple Silicon**. ~7.9M params; duration-predictor only
  (no pitch/energy variance adaptor), single fixed speaker, no style/emotion conditioning.
- **Vocoder:** HiFi-GAN V1 (mel → waveform), pretrained on LJSpeech (MIT; matches our 22.05 kHz /
  hop 256 / 80-mel front-end). Replaces Griffin-Lim for natural (non-robotic) audio.
- **Sample rate:** 22.05 kHz mono
- **Inference:** CPU via onnxruntime (Python) / `ort` (Rust); no GPU required. Single forward pass
  per stage (no autoregressive loop).
- **Tokenization:** character-level over the Tamil text (no espeak); `<bos>`/`<eos>` + per-char ids.
  Vocab + mel normalization stats ship in `<model>.tokenizer.json`.

## Artifacts (3 ONNX + 1 tokenizer)

| File | Role |
|---|---|
| `tamil_ns.enc_dur.onnx` | tokens → encoder features + per-token log-durations |
| `tamil_ns.decoder.onnx` | length-regulated features → mel |
| `hifigan.onnx` | mel → waveform |
| `tamil_ns.tokenizer.json` | char vocab, mel mean/std, audio params, mel pseudo-inverse |

Pipeline: text → enc_dur → (host-side integer length-regulation by predicted durations) → decoder
→ mel → hifigan → wav. Griffin-Lim remains available as a dependency-free fallback vocoder.

## Training data

[IndicTTS Tamil](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil) — female speaker subset
(~7.8 h used, 22.05 kHz). License: CC-BY-4.0 + IIT Madras Indic TTS EULA. See
[`DATASET_LICENSE.md`](DATASET_LICENSE.md). Attribution required in redistributions.

## Intended use & limitations

- Tamil speech synthesis from clean text input. Single voice; not a speaker-cloning model.
- Acoustic model trained to ~20k steps (intelligible, natural-sounding with HiFi-GAN). Quality
  improves with further training (resume via `tamiltts.mlx.train_ns --resume`).
- Char-level tokenization; out-of-vocabulary characters are skipped. Code-mixed English / unusual
  punctuation may degrade output.
- HiFi-GAN vocoder is LJSpeech-pretrained (English data); it generalizes to our mels but a Tamil
  fine-tune could improve fidelity further.
- Not evaluated for safety-critical or biometric use.

## Inference knobs

| Param   | Default | Effect                                   |
| ------- | ------- | ---------------------------------------- |
| `speed` | 1.0     | speaking rate (>1 faster, <1 slower)     |

## Inference (CPU)

Python: `uv run python -m tamiltts.mlx.onnx_infer_ns -m models/tamil_ns --text "வணக்கம்" -o out.wav`
Rust:   `cargo run --release --example synthesize_ns -- "வணக்கம்" out.wav ../models/tamil_ns`

## License

The model is **free to use under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)**.
Note the upstream attribution that still applies: training data is IndicTTS Tamil (CC-BY-4.0 + IIT
Madras Indic TTS EULA — attribution required, see `docs/DATASET_LICENSE.md`); vocoder weights are MIT.

## Provenance

Acoustic model trained in this repo (MLX). Vocoder weights: HiFi-GAN `jaketae/hifigan-lj-v1` (MIT),
re-exported to ONNX. Reproduction steps: `docs/MLX_RUNBOOK.md`.
