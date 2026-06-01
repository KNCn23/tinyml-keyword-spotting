#!/usr/bin/env python3
"""Compare the float and INT8 models: accuracy and model size.

    python scripts/evaluate.py [--model models/kws.npz]
"""

import _bootstrap  # noqa: F401

import argparse

from kws.dataset import make_dataset
from kws.model import MLP
from kws.quantize import QuantMLP, size_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/kws.npz")
    args = ap.parse_args()

    # A fresh test set with a different seed than training used.
    X, y, labels = make_dataset(n_per_class=80, seed=7)

    model = MLP.load(args.model)
    qmodel = QuantMLP.from_float(model)

    _, float_acc = model.evaluate(X, y)
    int8_acc = qmodel.evaluate(X, y)
    sizes = size_report(model, qmodel)

    print("Model comparison on a held-out synthetic test set")
    print("-" * 50)
    print(f"float32 accuracy : {float_acc:.3f}")
    print(f"int8    accuracy : {int8_acc:.3f}")
    print(f"accuracy drop    : {float_acc - int8_acc:+.3f}")
    print("-" * 50)
    print(f"float32 weights  : {sizes['float32_weight_bytes']:>6d} bytes")
    print(f"int8 weights     : {sizes['int8_weight_bytes']:>6d} bytes")
    print(f"compression      : {sizes['compression_x']}x smaller")


if __name__ == "__main__":
    main()
