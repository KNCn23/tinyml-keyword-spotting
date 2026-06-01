#!/usr/bin/env python3
"""Train the keyword-spotting model on the synthetic dataset.

    python scripts/train.py [--out models/kws.npz] [--epochs 40]

Saves the trained float weights to a .npz file for the other scripts to use.
"""

import _bootstrap  # noqa: F401  (sets sys.path)

import argparse
from pathlib import Path

import numpy as np

from kws.dataset import make_dataset
from kws.features import FEATURE_DIM
from kws.model import MLP


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/kws.npz", help="where to save weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32, help="hidden layer size")
    ap.add_argument("--per-class", type=int, default=150,
                    help="samples synthesised per class")
    args = ap.parse_args()

    print("Synthesising dataset...")
    X, y, labels = make_dataset(n_per_class=args.per_class, seed=0)

    # 80/20 train/test split.
    n_test = len(X) // 5
    Xtr, ytr = X[:-n_test], y[:-n_test]
    Xte, yte = X[-n_test:], y[-n_test:]
    print(f"  {len(Xtr)} train / {len(Xte)} test, {len(labels)} classes, "
          f"feature dim {FEATURE_DIM}")

    model = MLP(FEATURE_DIM, args.hidden, len(labels), seed=1)
    print("Training...")
    model.fit(Xtr, ytr, epochs=args.epochs)

    _, test_acc = model.evaluate(Xte, yte)
    print(f"\nTest accuracy: {test_acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    # np.savez appends .npz; normalise the printed path.
    saved = out if out.suffix == ".npz" else out.with_suffix(".npz")
    print(f"Saved weights to {saved}")


if __name__ == "__main__":
    main()
