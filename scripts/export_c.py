#!/usr/bin/env python3
"""Quantise a trained model and export it (plus a sample) to C headers.

    python scripts/export_c.py [--model models/kws.npz] [--label yes]

Writes c/model_data.h and c/sample.h, then prints the Python prediction for the
exported sample so you can confirm the C program agrees.
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from kws.dataset import LABELS, synth_keyword
from kws.export import write_model_header, write_sample_header
from kws.features import extract
from kws.model import MLP
from kws.quantize import QuantMLP


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/kws.npz")
    ap.add_argument("--label", default="yes", choices=LABELS,
                    help="which keyword to synthesise as the sample input")
    ap.add_argument("--outdir", default="c")
    args = ap.parse_args()

    model = MLP.load(args.model)
    qmodel = QuantMLP.from_float(model)

    # Make one example clip of the requested keyword.
    rng = np.random.default_rng(123)
    feature = extract(synth_keyword(args.label, rng))
    true_index = LABELS.index(args.label)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_model_header(qmodel, LABELS, str(outdir / "model_data.h"))
    write_sample_header(feature, true_index, LABELS, str(outdir / "sample.h"))

    pred = int(np.argmax(qmodel.forward_one(feature)))
    print(f"Wrote {outdir/'model_data.h'} and {outdir/'sample.h'}")
    print(f"Sample keyword     : {args.label}")
    print(f"Python int8 predict: {LABELS[pred]}")
    print("\nNow build the C reference:")
    print("  cd c && make run")


if __name__ == "__main__":
    main()
