#!/usr/bin/env python3
"""End-to-end demo: synthesise a clip, extract features, predict the keyword.

    python scripts/demo.py [--model models/kws.npz] [--label down]

This is the quickest way to see the whole pipeline work on a single example.
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from kws.dataset import LABELS, synth_keyword
from kws.features import extract
from kws.model import MLP
from kws.quantize import QuantMLP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/kws.npz")
    ap.add_argument("--label", default=None, choices=LABELS,
                    help="keyword to speak (default: a random one)")
    args = ap.parse_args()

    rng = np.random.default_rng()
    label = args.label or LABELS[rng.integers(len(LABELS))]

    wave = synth_keyword(label, rng)
    feat = extract(wave)

    model = MLP.load(args.model)
    qmodel = QuantMLP.from_float(model)

    float_pred = LABELS[int(model.predict(feat[None, :])[0])]
    int8_pred = LABELS[int(np.argmax(qmodel.forward_one(feat)))]

    print(f"spoken keyword : {label}")
    print(f"float32 model  : {float_pred}")
    print(f"int8 model     : {int8_pred}")


if __name__ == "__main__":
    main()
