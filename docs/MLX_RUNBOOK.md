# MLX Tamil-TTS — architecture & runbook

How the model is built and how to reproduce it on any Apple-Silicon (MLX) Mac: train on the GPU →
export ONNX → generate with the Python/Rust SDKs.

## Architecture

Two-stage, non-autoregressive (FastSpeech-style) acoustic model + neural vocoder. A single forward
pass per stage (no autoregressive loop), which exports cleanly to ONNX.

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
  │  model_ns.py (FastTTS). Per-token durations for training come from a forward-sum aligner.
  ▼
mel ──► HiFi-GAN V1 vocoder ──► waveform (22.05 kHz)        hifigan.py / hifigan.onnx
```

- **Tokenizer:** character-level over Tamil. Vocab + mel mean/std ship in `<model>.tokenizer.json`.
- **Mel front-end** (`audio.py`): sr 22050, n_fft 1024, hop 256, win 1024, 80 mels, fmin 0, fmax 8000,
  `log(clip(mel, 1e-5))` — the standard LJSpeech definition, which matches the HiFi-GAN vocoder.
- **Aligner** (`aligner.py`): a small model trained with a forward-sum (CTC) objective to learn a
  monotonic text↔mel alignment; MAS extracts per-token durations. Training-prep only (not shipped);
  it runs on CPU (CTC is unavailable on MPS).
- **Acoustic model** (`model_ns.py`, trained by `train_ns.py`): MLX, runs on the Apple GPU.
- **Vocoder** (`hifigan.py`): HiFi-GAN V1, LJSpeech-pretrained weights (MIT), exported to ONNX.

### ONNX layout (3 graphs + tokenizer)
`tamil_ns.enc_dur.onnx` (tokens→enc+log-durations) · `tamil_ns.decoder.onnx` (length-regulated
enc→mel) · `hifigan.onnx` (mel→wav) · `tamil_ns.tokenizer.json`. Length regulation (integer repeat by
predicted durations) runs host-side in the SDKs between `enc_dur` and `decoder`.

## Reproduce (any MLX Mac, 16–32 GB)

```bash
# 0. prereqs
brew install uv gh
uv sync --extra train            # torch (<2.9), onnx, librosa, tensorboard, ...
uv pip install mlx mlx-audio     # verify: uv run python -c "import mlx.core as mx; print(mx.default_device())"

# 1. data: female-speaker wavs + metadata, then cache mels + vocab + stats
uv run python -m tamiltts.data.prepare --low-disk --speaker female --out data
uv run python -m tamiltts.mlx.preprocess --data data --out data/mlx     # -> data/mlx/{mels,vocab.json,stats.json,...}

# 2. durations: forward-sum aligner (training-prep, CPU)  -> data/mlx/durations/*.npy
uv run python -m tamiltts.mlx.aligner --data data/mlx --steps 2000
#    durations should be well-distributed (mean ~6-8 frames/token).

# 3. train the acoustic model on the GPU (MLX). Use LR warmup.
uv run python -m tamiltts.mlx.train_ns --data data/mlx --out runs_mlx_ns --run tamil_ns \
    --steps 60000 --batch 16 --layers 4 --d_model 256 --max_frames 1200 \
    --lr 2e-4 --warmup 3000 --save_every 2000
#    ~250 ms/step, ~6 GB GPU. Resume/extend: --resume runs_mlx_ns/tamil_ns --steps 120000

# 4. export the acoustic model to ONNX (2 graphs + tokenizer)
uv run python -m tamiltts.mlx.export_onnx_ns --run runs_mlx_ns/tamil_ns --out models/tamil_ns --data data/mlx

# 5. vocoder: pretrained HiFi-GAN -> ONNX
curl -sL https://huggingface.co/jaketae/hifigan-lj-v1/resolve/main/pytorch_model.bin -o models/hifigan_lj.bin
uv run python -m tamiltts.mlx.export_hifigan --weights models/hifigan_lj.bin --out models/hifigan.onnx

# 6. generate (CPU, any platform)
uv run python -m tamiltts.mlx.onnx_infer_ns -m models/tamil_ns --text "வணக்கம், இது தமிழ் பேச்சு." -o out.wav
cd rust && cargo run --release --example synthesize_ns -- "வணக்கம்" out.wav ../models/tamil_ns
```

`onnx_infer_ns` uses `models/hifigan.onnx` automatically when present (Griffin-Lim fallback if not).
The Rust SDK looks for `hifigan.onnx` next to the model prefix.

## TensorBoard (progress + audio)

`train_ns` logs scalars (`train/loss`, `val/loss`, `ms/step`) to `runs_mlx_ns/<run>/tb`. `tb_eval_ns`
logs generated audio + predicted/target mel images per checkpoint to `runs_mlx_ns/<run>/tb_eval`.

```bash
# serve (use --host 0.0.0.0, not --bind_all; tensorboard needs setuptools<81 for pkg_resources)
uv run --with "setuptools<81" tensorboard --logdir runs_mlx_ns --host 0.0.0.0 --port 16006
# audio/mel eval poller (separate terminal; polls the latest checkpoint)
uv run python -m tamiltts.mlx.tb_eval_ns --run runs_mlx_ns/<run> --data data/mlx --interval 120
```

In the UI, enable the `<run>/tb` (curves) and `<run>/tb_eval` (Audio/Images/Text) runs. `tb_eval` must
use its own run dir (`<run>/tb_eval`), separate from the training `tb/` dir.

**Remote access (tailnet/LAN):** the macOS Application Firewall must allow the actual python binary
(the resolved `.../Python.app/Contents/MacOS/Python`, not the `bin/python3.12` symlink):

```bash
PYBIN=$(uv run python -c "import sys; print(sys.executable)")
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PYBIN"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$PYBIN"
```

Then open `http://<tailnet-ip>:16006/`.

## Notes
- `train_ns --resume` restores weights + optimizer state + step, so training can stop/extend freely.
- fp16-exporting the vocoder roughly halves its size (~56 MB → ~28 MB). Acoustic model is ~33 MB.
