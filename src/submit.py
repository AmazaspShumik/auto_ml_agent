"""
Private evaluation: score predictions against private targets and log to MLflow.

Usage:
    python src/submit.py predictions.csv <mlflow_run_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python src/submit.py <predictions.csv> <mlflow_run_id>", file=sys.stderr)
        sys.exit(1)

    preds_path = Path(sys.argv[1])
    run_id = sys.argv[2]

    if not preds_path.exists():
        print(f"FAILED: predictions file not found: {preds_path}")
        sys.exit(1)

    preds = pd.read_csv(preds_path).squeeze("columns").values
    private_score = _evaluate_private(preds)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("private_val_score", private_score)

    print(f"SUBMITTED: private_val_score logged to MLflow run {run_id}.")


def _evaluate_private(preds: np.ndarray) -> float:
    y_true = pd.read_csv(DATA_DIR / "val_private_y.csv")["target"].values
    preds = np.asarray(preds, dtype=float)
    if preds.shape[0] != y_true.shape[0]:
        raise ValueError(f"Length mismatch: {preds.shape[0]} vs {y_true.shape[0]}")
    return float(roc_auc_score(y_true, preds))


if __name__ == "__main__":
    main()
