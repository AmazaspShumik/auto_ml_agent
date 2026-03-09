"""
Public evaluation.

Usage:
    from evaluate import evaluate_public

    score = evaluate_public(predictions)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def evaluate_public(predictions: np.ndarray | list[float]) -> float:
    y_true = _load_targets("val_public_y.csv")
    return _score(y_true, predictions)


def _load_targets(filename: str) -> np.ndarray:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    import pandas as pd
    return pd.read_csv(path)["target"].values


def _score(y_true: np.ndarray, predictions: np.ndarray | list[float]) -> float:
    preds = np.asarray(predictions, dtype=float)
    if preds.shape[0] != y_true.shape[0]:
        raise ValueError(f"Length mismatch: {preds.shape[0]} predictions vs {y_true.shape[0]} targets")
    return float(roc_auc_score(y_true, preds))
