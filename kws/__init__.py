"""TinyML keyword spotting: features, a NumPy MLP, INT8 quantisation, C export."""

from .features import extract, log_mel, FEATURE_DIM
from .dataset import make_dataset, synth_keyword, LABELS
from .model import MLP
from .quantize import QuantMLP, quantize_tensor, size_report

__all__ = [
    "extract",
    "log_mel",
    "FEATURE_DIM",
    "make_dataset",
    "synth_keyword",
    "LABELS",
    "MLP",
    "QuantMLP",
    "quantize_tensor",
    "size_report",
]
