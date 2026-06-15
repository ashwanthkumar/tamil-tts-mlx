# MLX Tamil-TTS — architecture & reproduction runbook

How to reproduce the shipped Tamil voice on any Apple-Silicon (MLX-enabled) Mac, end to end:
train on the GPU → export ONNX → generate with the Python/Rust SDKs. For *why* it's built this way
(and the dead-ends we avoided), see [`MLX_TTS_LEARNINGS.md`](MLX_TTS_LEARNINGS.md).

## Architecture

Two-stage, non-autoregressive (FastSpeech-style) acoustic model + neural vocoder. Non-AR means a
single forward pass with no previous-frame feedback loop — it cannot drift/collapse, and it exports
to a clean ONNX graph.

```
Tamil text
  │  char-level tokenize (<bos> + chars + <eos>)            tamiltts/mlx/dataset.py
  ▼
┌──────────────────────────── Acoustic model (MLX, GPU) ───────────────────────────┐
│ Encoder (transformer)  ──►  Duration predictor ──► per-token durations            │
│        │                                                                          │
│        ▼   Length Regulator: repeat each token's encoder state by its duration    │
│ Non-causal decoder (transformer) ──► mel (80-dim log-mel)  + Postnet              │
└───────────────────────────────────────────────────────────────────────────────────┘
  │  model_ns.py (FastTTS).  Durations for training come from a forward-sum CTC aligner.
  ▼
mel ──► HiFi-GAN V1 vocoder ──► waveform (22.05 kHz)         hifigan.py / hifigan.onnx
```

- **Tokenizer:** character-level over Tamil (no espeak/phonemes). Vocab + mel mean/std ship in
  `<model>.tokenizer.json`.
- **Mel front-end** (`audio.py`): sr 22050, n_fft 1024, hop 256, win 1024, 80 mels, fmin 0, fmax 8000,
  `log(clip(mel, 1e-5))`. This is the standard LJSpeech def — it matches the pretrained HiFi-GAN.
- **Aligner** (`aligner.py`): a small PyTorch model trained with a forward-sum (CTC, with blank)
  objective to learn a monotonic text↔mel alignment; MAS extracts per-token durations. One-time
  training-prep (never shipped). CTC isn't on MPS, so this single step runs on CPU.
- **Acoustic model** (`model_ns.py`, trained by `train_ns.py`): pure MLX, runs on the Apple GPU.
- **Vocoder** (`hifigan.py`): HiFi-GAN V1, pretrained LJSpeech weights (MIT), exported to ONNX.

### ONNX layout (3 graphs + tokenizer)
`enc_dur.onnx` (tokens→enc+log-durations) · `decoder.onnx` (length-regulated enc→mel) ·
`hifigan.onnx` (mel→wav) · `tamil_ns.tokenizer.json`. Length regulation (integer repeat by predicted
durations) is done host-side in the SDKs between `enc_dur` and `decoder`.

## Reproduce the model (any MLX Mac, 16–32 GB)

```bash
# 0. prereqs
brew install espeak-ng uv gh          # espeak-ng only needed for the legacy VITS path; harmless
uv sync --extra train                 # torch (<2.9), onnx, librosa, tensorboard, ...
uv pip install mlx mlx-audio          # MLX (Apple GPU). verify: python -c "import mlx.core as mx; print(mx.default_device())"

# 1. data: female speaker wavs + metadata, then cache mels + vocab + stats
uv run python -m tamiltts.data.prepare --low-disk --speaker female --out data
uv run python -m tamiltts.mlx.preprocess --data data --out data/mlx     # -> data/mlx/{mels,vocab.json,stats.json,*.json}

# 2. durations: forward-sum CTC aligner (one-time prep; CPU, ~10-20 min)
uv run python -m tamiltts.mlx.aligner --data data/mlx --steps 2000      # -> data/mlx/durations/*.npy
#    sanity: durations should be well-distributed (mean ~6-8 frames/token, no token hogging all frames)

# 3. train the acoustic model on the GPU (MLX). WARMUP IS REQUIRED (avoids mean-collapse).
uv run python -m tamiltts.mlx.train_ns --data data/mlx --out runs_mlx_ns --run tamil_ns \
    --steps 60000 --batch 16 --layers 4 --d_model 256 --max_frames 1200 \
    --lr 2e-4 --warmup 3000 --save_every 2000
#    ~250 ms/step, ~6 GB GPU. Intelligible by ~16-20k; stop when TB audio sounds good.
#    Resume/extend any time:  --resume runs_mlx_ns/tamil_ns --steps 120000

# 4. export the acoustic model to ONNX (2 graphs + tokenizer)
uv run python -m tamiltts.mlx.export_onnx_ns --run runs_mlx_ns/tamil_ns --out models/tamil_ns --data data/mlx

# 5. vocoder: pretrained HiFi-GAN -> ONNX (MIT weights, matches our mel def)
curl -sL https://huggingface.co/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin -o models/hifigan_lj.bin
uv run python -m tamiltts.mlx.export_hifigan --weights models/hifigan_lj.bin --out models/hifigan.onnx

# 6. generate (CPU; works on any platform)
uv run python -m tamiltts.mlx.onnx_infer_ns -m models/tamil_ns --text "வணக்கம், இது தமிழ் பேச்சு." -o out.wav
# Rust:
cd rust && cargo run --release --example synthesize_ns -- "வணக்கம்" out.wav ../models/tamil_ns
```

`onnx_infer_ns` uses `models/hifigan.onnx` automatically when present (Griffin-Lim fallback if not).
The Rust SDK looks for `hifigan.onnx` next to the model prefix.

## TensorBoard setup (progress + audio, incl. remote/tailnet)

`train_ns` logs scalars (`train/loss`, `val/loss`, `ms/step`) to `runs_mlx_ns/<run>/tb`.
`tb_eval_ns` is a side process that logs **generated audio + pred/target mel images** per checkpoint.

```bash
# serve (note: --host 0.0.0.0, NOT --bind_all which binds IPv6-only on macOS).
# tensorboard needs setuptools<81 (it imports the removed pkg_resources):
uv run --with "setuptools<81" tensorboard --logdir runs_mlx_ns --host 0.0.0.0 --port 16006

# live audio/mel eval poller (separate terminal; polls latest checkpoint):
uv run python -m tamiltts.mlx.tb_eval_ns --run runs_mlx_ns/<run> --data data/mlx --interval 120
```

In the UI, tick the `<run>/tb` (curves) and `<run>/tb_eval` (Audio/Images/Text) runs.

**Remote access (tailnet/LAN):** TensorBoard binds to all IPv4 interfaces, but the macOS Application
Firewall blocks inbound to an un-allowlisted binary. Allow the **actual** python binary (the resolved
`.../Python.app/Contents/MacOS/Python`, not the `bin/python3.12` symlink):

```bash
PYBIN=$(python3 -c "import sys; print(sys.executable)")   # resolve real venv python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PYBIN"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$PYBIN"
```
Then reach it at `http://<tailnet-ip>:16006/`. Gotcha: `tb_eval` must use its **own** run dir
(`<run>/tb_eval`), never the training `tb/` dir — sharing a run makes TB purge the scalar curves.

## Notes
- `train_ns --resume` restores weights + optimizer state + step, so you can stop/extend freely.
- For smaller artifacts, fp16-export the vocoder (~28 MB vs 56 MB). Acoustic model is ~33 MB.
- The vocoder is English-pretrained; a Tamil HiFi-GAN fine-tune would improve fidelity further.
