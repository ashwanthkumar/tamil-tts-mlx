# Runbook: train the Tamil voice on the M1 Studio

Concrete, copy-paste steps to take this repo from clone → trained Tamil female voice → ONNX →
working Python/Rust inference, on an Apple Silicon Mac (built for an M1 Studio, 32 GB).

> If you are **Claude Code** reading this: do the sections in order. Steps 0–2 are quick. Step 3
> (training) is a long background job — launch it, then poll the log; don't block on it. Each
> dependency pin in `pyproject.toml` is there for a reason (comments explain); don't "upgrade" them.

## What's already verified (on an M2 Air)

Install, real female-data prep, a real VITS training step on MPS, ONNX export, and both SDKs
synthesizing from the export all work. The only thing not yet done is the **full training run to
convergence** — that's this runbook. Expect **many hours to a few days** depending on target
quality; VITS single-speaker typically needs ~hundreds of thousands of steps.

## 0. Prerequisites

```bash
# Homebrew tools
brew install espeak-ng uv gh        # espeak-ng is REQUIRED (Tamil phonemization)

# Clone the private repo (gh must be logged in: `gh auth status`)
gh repo clone ashwanthkumar/tamil-tts
cd tamil-tts
```

## 1. Environment

```bash
uv sync --extra train               # installs coqui-tts, torch (pinned <2.9), onnx, etc.
uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"   # expect True
```

If `uv sync` resolves a different Python, force 3.12: `uv python install 3.12 && uv sync --extra train`.

## 2. Prepare the data (full female speaker)

The `gender` column is a ClassLabel (`0=female`, `1=male`). The `--low-disk` mode downloads one
parquet shard at a time and deletes it after extracting wavs, so peak disk stays ~1 shard + the
~1.6 GB of output wavs (works even with little free space):

```bash
uv run python -m tamiltts.data.prepare --low-disk --speaker female --out data
# -> data/wavs/*.wav (22.05kHz mono) + data/metadata_{train,val}.csv
```

This scans all 17 shards. The female speaker is *mostly* in the first ~6 shards (~3,243 clips) but
a few clips appear later (e.g. shard 14), so **scan all shards for completeness** — don't cut it
short. There is a `--stop-after-empty N` flag that stops once female clips stop appearing for N
shards; it's faster but **lossy on this corpus** (drops the scattered later clips), so only use it
when you explicitly accept missing a few hundred clips.

(On a Studio with plenty of disk you can also drop `--low-disk` for a one-shot full download.)

Sanity check:
```bash
ls data/wavs | wc -l                # thousands of clips
```

## 3. Train (the long run)

```bash
./scripts/02_train.sh               # reads configs/tamil_female_vits.json; auto-uses MPS
```

- **Bigger batches on 32 GB:** edit `configs/tamil_female_vits.json` → try `"batch_size": 32`.
- **Watch progress:** `tensorboard --logdir runs/` (loss_mel and the audio samples are what matter).
- **Resume** after a stop: `./scripts/02_train.sh --continue_path runs/<run-dir>`
- **Run in background + log:**
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  nohup uv run python -m tamiltts.train --config configs/tamil_female_vits.json > train.log 2>&1 &
  tail -f train.log
  ```
- **When is it "done"?** Listen to the eval samples in TensorBoard; stop when speech is clear and
  stable (often 100k–300k+ steps). There's no single magic number.

## 4. Export to ONNX

```bash
./scripts/03_export.sh runs/<your-run-dir>     # the dir containing config.json
# -> models/tamil_female.onnx + models/tamil_female.tokenizer.json
```

## 5. Synthesize (CPU, no GPU needed)

Python:
```bash
uv run tamil-tts "வணக்கம், இது தமிழ் பேச்சு தொகுப்பு." -o hello.wav
```

Rust:
```bash
cd rust && cargo run --release --example synthesize -- "வணக்கம்" hello.wav ../models/tamil_female.onnx
```

The exported `models/tamil_female.onnx` (+ `.tokenizer.json`) is portable — copy it to the M2 Air
(or anywhere) and the same `tamil-tts` CLI / Rust example will run it on CPU.

## 6. (Optional) Quick smoke of the whole pipeline without full training

Proves train→export→synth end-to-end in ~2 minutes on synthetic audio:
```bash
uv run python tests/make_synthetic_dataset.py --out data_smoke --n 18
uv run python -m tamiltts.train --config configs/smoke_test.json
uv run python -m tamiltts.export_onnx --run "$(ls -d runs_smoke/tamil_smoke-*/ | head -1)" --out models/smoke.onnx
uv run tamil-tts "வணக்கம்" -m models/smoke.onnx -o /tmp/smoke.wav
```

## Troubleshooting

- **`espeak-ng not found`** → `brew install espeak-ng`.
- **Torch wants torchcodec / FFmpeg errors** → keep torch `<2.9` (already pinned); don't upgrade.
- **`isin_mps_friendly` ImportError** → transformers must be `<5` (already pinned).
- **MPS op not implemented** → `export PYTORCH_ENABLE_MPS_FALLBACK=1` (scripts already set this).
- **0 clips after prep** → the gender column is an int ClassLabel; use `--speaker female` (not a number).
- **Out of memory** → lower `batch_size` in the config.
