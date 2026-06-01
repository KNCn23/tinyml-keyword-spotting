"""A self-contained synthetic keyword dataset.

Real keyword spotting is trained on a corpus such as Google's *Speech Commands*.
To keep this project dependency-free and instantly runnable (no multi-gigabyte
download), we synthesise short utterances instead: each keyword is given a
distinct set of formant frequencies - like the resonances that distinguish
spoken vowels - plus a natural amplitude envelope and a dab of noise. An
``_unknown`` class of pure noise stands in for background/other words.

The synthetic clips are obviously not real speech, but they exercise the *exact*
same pipeline (features -> model -> quantise -> C inference) you would use on
real audio. To train on the real corpus instead, replace
:func:`make_dataset` with a loader that yields ``(waveform, label)`` pairs and
feed them through :func:`kws.features.extract` unchanged - see the README.
"""

from __future__ import annotations

import numpy as np

from .features import SAMPLE_RATE, extract

# Each keyword is defined by a few "formant" frequencies (Hz). The values are
# chosen to be acoustically distinct so a tiny model can separate them.
KEYWORD_FORMANTS: dict[str, list[float]] = {
    "yes":  [500, 1700, 2500],
    "no":   [400, 900, 2300],
    "up":   [350, 1100, 2600],
    "down": [550, 800, 2400],
}
UNKNOWN = "_unknown"
LABELS = list(KEYWORD_FORMANTS) + [UNKNOWN]
DURATION_S = 1.0


def _envelope(n: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth attack/sustain/decay amplitude envelope."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    attack = np.clip(t / rng.uniform(0.05, 0.15), 0, 1)
    decay = np.clip((1 - t) / rng.uniform(0.1, 0.3), 0, 1)
    return attack * decay


def synth_keyword(label: str, rng: np.random.Generator) -> np.ndarray:
    """Generate one ~1 s waveform for the given label."""
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    if label == UNKNOWN:
        # Broadband noise plus a random off-vocabulary tone.
        sig = rng.normal(0, 0.3, n).astype(np.float32)
        sig += 0.3 * np.sin(2 * np.pi * rng.uniform(3000, 6000) * t)
        return sig * _envelope(n, rng)

    sig = np.zeros(n, dtype=np.float32)
    for f in KEYWORD_FORMANTS[label]:
        jitter = rng.uniform(0.97, 1.03)        # small per-sample frequency drift
        amp = rng.uniform(0.6, 1.0)
        sig += amp * np.sin(2 * np.pi * f * jitter * t)
    sig += rng.normal(0, 0.05, n).astype(np.float32)   # mild noise
    return sig * _envelope(n, rng)


def make_dataset(n_per_class: int = 120, seed: int = 0):
    """Build a feature matrix and label vector.

    Returns:
        X: float32 array, shape ``(n_samples, FEATURE_DIM)``.
        y: int array of class indices.
        labels: the list of class names (index order matches ``y``).
    """
    rng = np.random.default_rng(seed)
    feats, ys = [], []
    for class_idx, label in enumerate(LABELS):
        for _ in range(n_per_class):
            wave = synth_keyword(label, rng)
            feats.append(extract(wave))
            ys.append(class_idx)
    X = np.stack(feats).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    # Shuffle so train/test splits are not class-ordered.
    perm = rng.permutation(len(y))
    return X[perm], y[perm], LABELS
