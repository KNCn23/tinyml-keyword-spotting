# tinyml-keyword-spotting

A complete, dependency-light **TinyML keyword-spotting** pipeline: it takes
audio, extracts log-mel features, trains a small neural network, quantises it to
**8-bit integers**, and runs the quantised model in **pure C** — the same code
path you would flash onto a microcontroller. Everything is implemented with
nothing but **NumPy** (training side) and **standard C** (deployment side), so
it runs offline, anywhere, in seconds.

```
$ python scripts/evaluate.py
float32 accuracy : 1.000
int8    accuracy : 1.000
int8 weights     :   8352 bytes      (4.0x smaller than float32)

$ cd c && make run
predicted: yes
expected : yes
MATCH
```

---

## Table of contents

1. [Why this project](#why-this-project)
2. [The pipeline at a glance](#the-pipeline-at-a-glance)
3. [Install](#install)
4. [Quick start (one command each)](#quick-start-one-command-each)
5. [Tutorial 1 — Audio to features](#tutorial-1--audio-to-features)
6. [Tutorial 2 — Train the model](#tutorial-2--train-the-model)
7. [Tutorial 3 — Quantise to INT8](#tutorial-3--quantise-to-int8)
8. [Tutorial 4 — Run it in C (the "device")](#tutorial-4--run-it-in-c-the-device)
9. [Using real audio instead of the synthetic set](#using-real-audio-instead-of-the-synthetic-set)
10. [Project layout](#project-layout)
11. [Tests](#tests)
12. [Design notes & simplifications](#design-notes--simplifications)

---

## Why this project

"TinyML" is about running machine-learning models on tiny, cheap, low-power
hardware. Two ideas make that possible, and this repo demonstrates both
end-to-end on a keyword spotter (think *"Hey Siri"* wake-word detection):

1. **A small feature front-end + a small model.** We use a 256-number log-mel
   feature grid and a two-layer perceptron — a few thousand parameters.
2. **Integer quantisation.** Floats are expensive on an MCU; converting the
   model to int8 makes it 4× smaller and lets every multiply be a cheap
   `int8 × int8` operation, with essentially no accuracy loss.

Crucially, the C inference reproduces the Python integer math *exactly*, so you
can trust that the model you trained is the model that runs.

## The pipeline at a glance

```
   synth/real audio
         │  kws/features.py
         ▼
   log-mel features  (16 mels × 16 frames = 256 floats)
         │  kws/model.py        (train, NumPy)
         ▼
   float32 MLP  ──────────────┐
         │  kws/quantize.py    │ kws/export.py
         ▼                     ▼
   int8 MLP  ──────────▶  c/model_data.h + c/sample.h
                                 │  c/kws_infer.c
                                 ▼
                          prediction on "device"
```

## Install

Requires **Python 3.9+** and a C compiler (`gcc`/`clang`). The only Python
dependency is NumPy.

```bash
git clone https://github.com/KNCn23/tinyml-keyword-spotting.git
cd tinyml-keyword-spotting
python -m pip install -r requirements.txt
```

> On an externally-managed Python (Homebrew/Debian) use a virtualenv
> (`python -m venv .venv && source .venv/bin/activate`) or add
> `--break-system-packages` to the pip command.

## Quick start (one command each)

```bash
python scripts/train.py        # train -> models/kws.npz
python scripts/evaluate.py     # float vs int8 accuracy + size
python scripts/demo.py         # classify one random clip end-to-end
python scripts/export_c.py     # write the C headers
cd c && make run               # run the int8 model in pure C
```

---

## Tutorial 1 — Audio to features

A model never sees raw samples; it sees a **log-mel spectrogram**, a compact
time/frequency picture of the sound. Read [`kws/features.py`](kws/features.py):

1. The signal is cut into overlapping 25 ms **frames** (10 ms apart).
2. Each frame's **power spectrum** is computed with an FFT.
3. A **mel filterbank** pools the FFT bins into 16 perceptually-spaced bands.
4. We take the **log** (ears perceive loudness logarithmically) and
   average-pool the time axis to a fixed 16 frames.

The result is a `16 × 16` grid — 256 numbers — normalised for stable training.
Try it:

```python
from kws.dataset import synth_keyword
from kws.features import log_mel
import numpy as np

wave = synth_keyword("yes", np.random.default_rng(0))
print(log_mel(wave).shape)      # (16, 16)
```

## Tutorial 2 — Train the model

The classifier is a two-layer perceptron written by hand in NumPy
([`kws/model.py`](kws/model.py)): `256 → ReLU(32) → 5 classes`, trained with
mini-batch Adam and a cross-entropy loss. You can read the entire forward and
backward pass — there is no framework hiding it.

```bash
python scripts/train.py --epochs 40
```

```
  600 train / 150 test, 5 classes, feature dim 256
epoch   1  loss 0.0309  acc 0.983
...
Test accuracy: 1.000
Saved weights to models/kws.npz
```

The five classes are `yes`, `no`, `up`, `down` and `_unknown`. They are
**synthetic** (see [the next section](#using-real-audio-instead-of-the-synthetic-set))
so the project needs no dataset download; because the classes are acoustically
distinct, the tiny model learns them easily, which keeps the focus on the
TinyML mechanics.

## Tutorial 3 — Quantise to INT8

This is the heart of TinyML. [`kws/quantize.py`](kws/quantize.py) converts the
float weights to int8 using symmetric per-tensor scaling, exactly like
PyTorch/TFLite dynamic quantisation:

```
scale  = max(|W|) / 127
q      = round(W / scale)          # int8 in [-127, 127]
W ≈ q * scale                      # reconstruct when needed
```

At inference, activations are quantised on the fly, the matrix multiply runs in
**int32**, and the result is scaled back to float. Compare the two models:

```bash
python scripts/evaluate.py
```

```
float32 accuracy : 1.000
int8    accuracy : 1.000
accuracy drop    : +0.000
float32 weights  :  33408 bytes
int8 weights     :   8352 bytes
compression      : 4.0x smaller
```

Same accuracy, a quarter of the size. (This mirrors the kind of comparison in my
[onnx-quant-benchmark](https://github.com/KNCn23/onnx-quant-benchmark) project,
here taken all the way down to the C level.)

## Tutorial 4 — Run it in C (the "device")

A microcontroller has no Python, so the model is baked into C source as `const`
arrays. [`scripts/export_c.py`](scripts/export_c.py) writes two headers into
`c/`:

* `model_data.h` — int8 weights, scales, float biases, dimensions.
* `sample.h` — one example feature vector and its true label.

```bash
python scripts/export_c.py --label down
cd c && make run
```

[`c/kws_infer.c`](c/kws_infer.c) then performs the **identical** integer
arithmetic to the Python `QuantMLP` and prints the prediction:

```
class logits:
  yes        -8.9359
  no         -6.0052
  up         -10.3780
  down        5.2898
  _unknown   -9.4488
predicted: down
expected : down
MATCH
```

Because the C and Python code share the exact quantisation scheme, the
predictions always agree — that equivalence is the whole point of a TinyML
deployment flow. The committed headers let `make run` work immediately; re-run
`export_c.py` after retraining to refresh them.

## Using real audio instead of the synthetic set

The synthetic generator in [`kws/dataset.py`](kws/dataset.py) stands in for a
real corpus so the repo is self-contained. To train on real speech (e.g.
[Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)):

1. Load each `.wav` as a float array resampled to 16 kHz.
2. Feed it through `kws.features.extract(waveform)` — **unchanged**.
3. Replace `make_dataset()` with your `(features, label)` loader.

Nothing else changes: the model, quantiser, exporter and C inference all operate
on the 256-dim feature vector regardless of where it came from.

## Project layout

```
tinyml-keyword-spotting/
├── kws/
│   ├── features.py     # log-mel feature front-end
│   ├── dataset.py      # synthetic keyword generator
│   ├── model.py        # NumPy 2-layer MLP (train + inference)
│   ├── quantize.py     # INT8 post-training quantisation
│   └── export.py       # write C headers
├── scripts/            # train / evaluate / demo / export_c
├── c/
│   ├── kws_infer.c     # pure-C int8 inference (matches Python exactly)
│   ├── model_data.h    # generated: quantised weights
│   └── Makefile
└── tests/              # pytest: features, quantiser, end-to-end
```

## Tests

```bash
python -m pytest -q
```

The suite checks feature shapes and numerical stability, that quantisation
reconstructs weights within one step, and that a freshly-trained model both
hits high accuracy and survives INT8 conversion.

## Design notes & simplifications

This project optimises for *clarity* and *self-containment*, so a few choices
differ from a production system — each is a natural next step:

- **Synthetic data.** Real keyword spotting trains on recorded speech; here the
  classes are synthesised. The pipeline is identical; only the data source
  differs.
- **MLP, not a CNN.** Real KWS models are small convolutional networks over the
  spectrogram. An MLP keeps the from-scratch NumPy training readable. The
  feature grid is already CNN-ready.
- **Dynamic activation quantisation.** Activation scales are computed per
  inference. A fully static (calibrated) scheme avoids that runtime step on the
  device.
- **No audio I/O.** Features are computed from in-memory arrays; wiring up a
  microphone or `.wav` reader is left to the integrator.

## License

[MIT](LICENSE)
