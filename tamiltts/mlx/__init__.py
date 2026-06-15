"""MLX-native Tamil TTS: GPU training on Apple Silicon.

A compact TransformerTTS (char-level Tamil -> 80-dim log-mel, autoregressive with
cross-attention) trained from scratch with MLX so the Apple GPU is actually used
(unlike the Coqui/PyTorch path, whose trainer is CUDA-only and silently runs on CPU).
"""
