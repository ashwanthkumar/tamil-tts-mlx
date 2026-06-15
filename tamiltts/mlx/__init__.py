"""MLX-native Tamil TTS: GPU training on Apple Silicon.

A non-AR FastTTS (char-level Tamil -> 80-dim log-mel via an encoder, duration predictor,
length regulator, and non-causal decoder) trained from scratch with MLX so the Apple GPU
is actually used, then exported to portable ONNX for the Python/Rust SDKs.
"""
