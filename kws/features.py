"""Turn raw audio into a compact log-mel feature grid.

Keyword spotting does not work on raw samples - it works on a time/frequency
picture of the sound. The standard front-end is:

    audio -> frames -> |FFT|^2 -> mel filterbank -> log -> (pool)

This module implements exactly that with nothing but NumPy, so it runs anywhere
and is easy to read. The output is a small fixed-size grid (default 16 mel
bands x 16 time steps = 256 numbers) suitable for a tiny model and for export to
a microcontroller.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000     # Hz - the Speech Commands standard
FRAME_LEN = 400          # 25 ms window
FRAME_HOP = 160          # 10 ms step
N_FFT = 512
N_MELS = 16              # mel bands (kept small for TinyML)
N_FRAMES = 16            # time steps after pooling
FEATURE_DIM = N_MELS * N_FRAMES


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT,
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Build the triangular mel filterbank matrix of shape (n_mels, n_fft/2+1).

    Each row is one triangular filter that pools FFT bins into a mel band.
    """
    n_bins = n_fft // 2 + 1
    low_mel = _hz_to_mel(np.array(0.0))
    high_mel = _hz_to_mel(np.array(sample_rate / 2.0))
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, center):
            if center != left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


# Build the (constant) filterbank once at import time.
_FILTERBANK = mel_filterbank()


def _frame(signal: np.ndarray) -> np.ndarray:
    """Slice a 1-D signal into overlapping, windowed frames."""
    if len(signal) < FRAME_LEN:
        signal = np.pad(signal, (0, FRAME_LEN - len(signal)))
    n_frames = 1 + (len(signal) - FRAME_LEN) // FRAME_HOP
    idx = (np.arange(FRAME_LEN)[None, :]
           + FRAME_HOP * np.arange(n_frames)[:, None])
    frames = signal[idx]
    return frames * np.hamming(FRAME_LEN).astype(np.float32)


def _pool_time(spec: np.ndarray, n_out: int = N_FRAMES) -> np.ndarray:
    """Average-pool a (n_mels, n_frames) spectrogram down to n_out columns."""
    n_mels, n_frames = spec.shape
    if n_frames == n_out:
        return spec
    edges = np.linspace(0, n_frames, n_out + 1).astype(int)
    pooled = np.empty((n_mels, n_out), dtype=np.float32)
    for i in range(n_out):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        pooled[:, i] = spec[:, lo:hi].mean(axis=1)
    return pooled


def log_mel(signal: np.ndarray) -> np.ndarray:
    """Compute the pooled log-mel grid for one clip.

    Returns a ``(N_MELS, N_FRAMES)`` float32 array, normalised to roughly zero
    mean / unit scale so it is friendly to both training and quantisation.
    """
    signal = np.asarray(signal, dtype=np.float32)
    frames = _frame(signal)
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = power @ _FILTERBANK.T              # (n_frames, n_mels)
    log = np.log(mel.T + 1e-6)               # (n_mels, n_frames)
    pooled = _pool_time(log)
    pooled -= pooled.mean()
    std = pooled.std()
    if std > 1e-6:
        pooled /= std
    return pooled.astype(np.float32)


def extract(signal: np.ndarray) -> np.ndarray:
    """Flatten :func:`log_mel` into a 1-D feature vector of length FEATURE_DIM."""
    return log_mel(signal).reshape(-1)
