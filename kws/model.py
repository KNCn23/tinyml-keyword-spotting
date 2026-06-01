"""A tiny two-layer perceptron, implemented and trained in pure NumPy.

    input (256) -> Dense + ReLU (hidden) -> Dense -> softmax (n_classes)

Two affine layers with a ReLU between them is enough to separate the keyword
classes and is small enough to run on a microcontroller once quantised. Training
uses mini-batch gradient descent with the Adam optimiser and a cross-entropy
loss. Everything is hand-written so you can see exactly what happens in the
forward and backward passes.
"""

from __future__ import annotations

import numpy as np


def _he_init(fan_in: int, fan_out: int, rng) -> np.ndarray:
    """He initialisation, appropriate for ReLU networks."""
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)


class MLP:
    """A 2-layer perceptron with a ReLU hidden layer."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = _he_init(in_dim, hidden, rng)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = _he_init(hidden, n_classes, rng)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    # --- inference --------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return class logits for a batch ``X`` of shape (N, in_dim)."""
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(self._z1, 0.0)        # ReLU
        return self._a1 @ self.W2 + self.b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    # --- training ---------------------------------------------------------
    def _adam_step(self, grads, lr, beta1, beta2, eps, t):
        for name, g in grads.items():
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (g * g)
            m_hat = self._m[name] / (1 - beta1 ** t)
            v_hat = self._v[name] / (1 - beta2 ** t)
            getattr(self, name)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y, *, epochs=40, batch_size=32, lr=1e-2, seed=0, verbose=True):
        """Train in place. Returns the history of (epoch, loss, accuracy)."""
        rng = np.random.default_rng(seed)
        n, n_classes = len(X), self.W2.shape[1]
        params = ["W1", "b1", "W2", "b2"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in params}
        step = 0
        history = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                logits = self.forward(xb)

                # softmax + cross-entropy
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / exp.sum(axis=1, keepdims=True)

                # gradient of loss w.r.t. logits
                dlogits = probs.copy()
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)

                # backprop
                dW2 = self._a1.T @ dlogits
                db2 = dlogits.sum(axis=0)
                da1 = dlogits @ self.W2.T
                dz1 = da1 * (self._z1 > 0)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)

                step += 1
                self._adam_step(
                    {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2},
                    lr, 0.9, 0.999, 1e-8, step,
                )

            loss, acc = self.evaluate(X, y)
            history.append((epoch, loss, acc))
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"epoch {epoch:3d}  loss {loss:.4f}  acc {acc:.3f}")
        return history

    def evaluate(self, X, y):
        """Return (mean cross-entropy loss, accuracy) on a labelled set."""
        logits = self.forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-9).mean())
        acc = float((probs.argmax(axis=1) == y).mean())
        return loss, acc

    # --- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "MLP":
        d = np.load(path)
        in_dim, hidden = d["W1"].shape
        n_classes = d["W2"].shape[1]
        model = cls(in_dim, hidden, n_classes)
        model.W1, model.b1 = d["W1"], d["b1"]
        model.W2, model.b2 = d["W2"], d["b2"]
        return model
