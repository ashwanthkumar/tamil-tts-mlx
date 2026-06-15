"""Audio front-end for the MLX TTS: wav <-> log-mel, plus Griffin-Lim vocoding.

Mel settings mirror the VITS config (sr 22050, n_fft 1024, hop 256, 80 mels) so the
features are comparable to the PyTorch path.
"""
from __future__ import annotations

import numpy as np
import librosa

SR = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000

_mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


def load_wav(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SR, mono=True)
    # trim leading/trailing silence to tighten alignment
    wav, _ = librosa.effects.trim(wav, top_db=30)
    return wav.astype(np.float32)


def wav_to_mel(wav: np.ndarray) -> np.ndarray:
    """-> log-mel of shape (T, N_MELS), float32."""
    S = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN, center=True))
    mel = _mel_basis @ S
    logmel = np.log(np.clip(mel, 1e-5, None))
    return logmel.T.astype(np.float32)


def mel_to_wav(logmel: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """Griffin-Lim inversion of a (T, N_MELS) log-mel back to a waveform."""
    mel = np.exp(np.asarray(logmel, dtype=np.float32).T)
    S = librosa.feature.inverse.mel_to_stft(mel, sr=SR, n_fft=N_FFT, power=1.0, fmin=FMIN, fmax=FMAX)
    wav = librosa.griffinlim(S, n_iter=n_iter, hop_length=HOP, win_length=WIN)
    return wav.astype(np.float32)
