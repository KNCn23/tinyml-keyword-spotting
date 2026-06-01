"""Tests for the feature front-end, the model, and INT8 quantisation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kws.dataset import LABELS, make_dataset, synth_keyword
from kws.features import FEATURE_DIM, N_FRAMES, N_MELS, extract, log_mel
from kws.model import MLP
from kws.quantize import QuantMLP, quantize_tensor


# --- features -----------------------------------------------------------

def test_feature_shape_and_finiteness():
    rng = np.random.default_rng(0)
    wave = synth_keyword("yes", rng)
    grid = log_mel(wave)
    assert grid.shape == (N_MELS, N_FRAMES)
    assert np.all(np.isfinite(grid))
    assert extract(wave).shape == (FEATURE_DIM,)


def test_short_signal_is_padded_not_crashing():
    # Far shorter than one frame: must still produce a valid feature vector.
    feat = extract(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert feat.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(feat))


# --- quantisation -------------------------------------------------------

def test_quantize_tensor_is_close_to_original():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 5)).astype(np.float32)
    q, scale = quantize_tensor(w)
    assert q.dtype == np.int8
    assert np.abs(q).max() <= 127
    recon = q.astype(np.float32) * scale
    # Within one quantisation step of the original.
    assert np.max(np.abs(recon - w)) <= scale + 1e-6


# --- model + end-to-end -------------------------------------------------

def test_model_trains_and_quantises_without_losing_much():
    X, y, labels = make_dataset(n_per_class=60, seed=0)
    n_test = len(X) // 5
    Xtr, ytr, Xte, yte = X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]

    model = MLP(FEATURE_DIM, 32, len(labels), seed=2)
    model.fit(Xtr, ytr, epochs=30, verbose=False)

    _, float_acc = model.evaluate(Xte, yte)
    # The synthetic classes are well separated; a small MLP should ace them.
    assert float_acc > 0.9

    qmodel = QuantMLP.from_float(model)
    int8_acc = qmodel.evaluate(Xte, yte)
    # INT8 should track the float model closely.
    assert int8_acc > float_acc - 0.05


def test_label_set_is_consistent():
    _, y, labels = make_dataset(n_per_class=10, seed=0)
    assert labels == LABELS
    assert set(np.unique(y)) <= set(range(len(LABELS)))
