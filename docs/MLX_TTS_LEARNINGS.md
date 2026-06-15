# MLX Tamil-TTS — learnings & dead-ends (so we don't repeat them)

Chronological, hard-won lessons from building a Tamil TTS that trains on Apple Silicon (MLX/GPU),
exports to ONNX, and serves via Python + Rust SDKs. Read this before touching the training pipeline.

## 1. The original blocker: Coqui's trainer is CUDA-only
- `tamiltts/train.py` (Coqui `trainer`) prints `device: mps` but the `trainer` library only moves the
  model to GPU under `if self.use_cuda`. **On a Mac it silently trains on CPU** (~54 s/step → months).
- **Lesson:** never trust a "device" print. Verify the GPU is actually busy (`sample <pid>` showed
  ~all `libtorch_cpu`, Metal idle). Use a stack sample / activity monitor to confirm.
- Fix: train natively in **MLX** (mlx 0.31.2). Real GPU use, ~250 ms/step for an ~8M-param model,
  ~6 GB peak for batch 16 — well within 32 GB.

## 2. AR TransformerTTS collapses at inference (exposure bias) — and low loss HID it
- A from-scratch autoregressive TransformerTTS reached teacher-forced L1 **0.06** (looked great) but
  **free-running generation collapsed after the first syllable** (energy decayed to a flat hum).
- Root cause: it minimized loss by **copying the previous frame** (self-attention), not by grounding
  on text. Cross-attention was **~uniform** (focus 0.02, ~3 of 66 tokens used) — no real alignment.
- **Lesson:** teacher-forced loss is a vanity metric. **Validate free-run generation AND alignment
  early**, not just loss. Check cross-attention is diagonal; check generated energy is sustained.
- What did NOT fix it: **input-noise fine-tune** (gaussian noise on TF frames). Gaussian noise doesn't
  penalize mean-regression drift. Inference-time pre-net dropout helped only partially (and breaks
  deterministic ONNX export). Scheduled sampling might help but AR stays fragile + slow.

## 3. The fix: non-autoregressive (FastSpeech-style)
- `model_ns.py` `FastTTS`: encoder → duration predictor → length regulator (gather/expand by
  durations) → **non-causal** decoder → mel + postnet. **No AR loop → structurally cannot collapse.**
  Also single forward pass → clean ONNX, no host-side decode loop.
- Validated: generated energy sustained across the whole utterance (16/16 windows nonzero).
- Duration predictor learns length: output scales with text (10 tok→0.6 s, 88 tok→5 s). Early on it
  slightly undershoots per-token (~5 vs aligner's ~6.5 frames); converges with the `l_dur` loss.

## 4. Alignment is the hard part — what worked and what didn't
Goal: per-token durations to drive the length regulator. Attempts:
- ❌ **Distill from the AR teacher's attention** — dead on arrival; the AR cross-attention had no
  alignment (see §2).
- ❌ **MLX-native forward-sum (monotonic, no blank)** — mode-collapses: attention confidently maps
  ~all frames to ONE token (maxprob ~0.8, 3/66 tokens used). Loss decreases to a degenerate solution.
- ❌ **Guided-attention (diagonal prior) alone / dominant** — too weak a gradient to escape the
  collapse basin (penalty margin collapse-vs-diagonal was only ~0.05 vs ~0.003).
- ✅ **PyTorch CTC aligner with BLANK** (`aligner.py`) — the blank symbol is what makes forward-sum
  spread properly. Loss 5.06→1.1 over 2000 steps; durations well-distributed (mean ~6.5 fr/token,
  p90 ~9-12, no degeneracy). This is the FastSpeech-1 recipe (distill durations from a CTC aligner).
- **Caveats:** MPS lacks `aten::_ctc_loss` → that op needs CPU fallback. The aligner is **one-time,
  throwaway PREP** (never ships), so a ~10-min CPU run is acceptable; the actual model trains on GPU.
  Use **efficient L2 distance** `||q||²+||k||²-2q·k` (NOT the (B,d,Tm,Tt) 4D tensor → 600 MB/step, and
  NOT dot-product → magnitude-collapse). `mlx_aligner.py` kept for reference (the GPU attempt).

## 5. Export & SDKs
- MLX has no native ONNX export. Bridge: mirror the model in **PyTorch with identical math**, port
  weights 1:1, `torch.onnx.export`. Verified **MLX↔ONNX parity 3.8e-6**.
- Weight-port gotchas: `Conv1d` weight is `(O,K,I)` in MLX vs `(O,I,K)` in torch (transpose);
  MLX `MultiHeadAttention` uses `bias=False` and `query/key/value/out_proj` submodule names.
- Non-AR ONNX is single-pass (text→mel); length regulator expansion is integer `repeat` done
  host-side from predicted durations (clean, deterministic). Vocoder = Griffin-Lim (ship `mel_inv`
  pseudo-inverse in the tokenizer JSON so Rust matches Python).
- Rust SDK uses `ort` + `rustfft` (Griffin-Lim). Python SDK uses `onnxruntime` + librosa.

## 6. TensorBoard / ops gotchas (Apple Silicon + remote)
- Serve with `--host 0.0.0.0` (IPv4). **`--bind_all` binds IPv6-only on this macOS** → breaks IPv4.
- macOS **Application Firewall** blocks non-loopback until the **exact running binary** is allowed —
  the resolved `.../Python.app/Contents/MacOS/Python`, NOT the `bin/python3.12` symlink.
- `tb_eval` MUST write to its **own run dir** (`<run>/tb_eval`), not the training `<run>/tb`. Sharing
  a run makes TB purge/freeze the scalars (eval logs lower step numbers → looks like a restart).
- TensorBoard needs `setuptools<81` (it imports the removed `pkg_resources`): `uv run --with
  "setuptools<81" tensorboard ...`. After restarting the TB *server*, the browser needs a hard-refresh.
- iMessage notifications over SSH are blocked by **TCC** (can't approve the Automation prompt
  headless) — don't try to bypass security controls; use TB for monitoring.

## 7. Sizes / footprint (final non-AR model)
- ~7.9M params → **~33 MB** ONNX (fp32) + 0.76 MB tokenizer. fp16 → ~16 MB, int8 → ~8 MB.
- Inference: single forward pass on CPU; **Griffin-Lim (60 iters) is the dominant per-clip cost**.
  Process RSS ~285 MB is mostly the Python runtime (model working set ~14 MB); Rust SDK is leaner.
- Training (separate): ~6 GB GPU.

## 8. Infra that carried across all the rewrites (reused, not wasted)
Data prep (4315 clips, cached mels, vocab, stats), the MLX training harness (loop, **resume/extend**
= weights+optimizer+step, checkpointing), TB + live audio/mel eval, the ONNX bridge, and the SDK
scaffolding. Only the AR *model weights* + AR training hours were a write-off.
