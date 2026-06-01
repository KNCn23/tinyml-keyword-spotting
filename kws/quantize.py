"""Post-training INT8 quantisation.

The trick that makes a model fit on a microcontroller is replacing 32-bit floats
with 8-bit integers. We use the same scheme as PyTorch / TFLite "dynamic"
quantisation of linear layers:

* **Weights** are quantised once, offline, with a symmetric per-tensor scale:
  ``q = round(w / scale)`` clipped to ``[-127, 127]`` where
  ``scale = max(|w|) / 127``.
* **Activations** are quantised on the fly at inference time with their own
  per-vector scale.
* The matrix multiply runs in **int32** accumulators; the result is turned back
  into a float with ``acc * scale_x * scale_w`` and the float bias is added.

This keeps every multiply an ``int8 x int8`` operation (cheap on an MCU) while
the accuracy stays within a hair of the float model. The integer arithmetic
here is byte-for-byte what the C reference in ``c/kws_infer.c`` performs.
"""

from __future__ import annotations

import numpy as np

from .model import MLP


def quantize_tensor(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantisation of a float tensor.

    Returns the int8 array and the float scale needed to reconstruct it.
    """
    max_abs = float(np.max(np.abs(w)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def _quantize_vec(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Dynamic per-vector quantisation of an activation row."""
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int32)
    return q, scale


class QuantMLP:
    """An INT8 version of :class:`kws.model.MLP`."""

    def __init__(self, qW1, sW1, b1, qW2, sW2, b2):
        self.qW1, self.sW1, self.b1 = qW1, sW1, b1.astype(np.float32)
        self.qW2, self.sW2, self.b2 = qW2, sW2, b2.astype(np.float32)

    @classmethod
    def from_float(cls, model: MLP) -> "QuantMLP":
        qW1, sW1 = quantize_tensor(model.W1)
        qW2, sW2 = quantize_tensor(model.W2)
        return cls(qW1, sW1, model.b1, qW2, sW2, model.b2)

    def _layer(self, x_q, sx, qW, sW, bias):
        """One quantised affine layer: int8 matmul -> dequantise -> + bias."""
        acc = x_q @ qW.astype(np.int32)          # int32 accumulation
        return acc * (sx * sW) + bias

    def forward_one(self, x: np.ndarray) -> np.ndarray:
        """Logits for a single feature vector, via the integer path."""
        x_q, sx = _quantize_vec(x)
        z1 = self._layer(x_q, sx, self.qW1, self.sW1, self.b1)
        a1 = np.maximum(z1, 0.0)                  # ReLU in float
        a1_q, sa = _quantize_vec(a1)
        return self._layer(a1_q, sa, self.qW2, self.sW2, self.b2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.forward_one(x)) for x in X])

    def evaluate(self, X, y) -> float:
        """Classification accuracy on a labelled set."""
        return float((self.predict(X) == y).mean())


def size_report(model: MLP, qmodel: QuantMLP) -> dict:
    """Compare the float and int8 footprints of the weight tensors."""
    float_bytes = (model.W1.size + model.W2.size) * 4
    int8_bytes = qmodel.qW1.size + qmodel.qW2.size
    return {
        "float32_weight_bytes": float_bytes,
        "int8_weight_bytes": int8_bytes,
        "compression_x": round(float_bytes / int8_bytes, 2),
    }
