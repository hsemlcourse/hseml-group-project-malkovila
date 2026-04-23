from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | float) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k_int = int(k * n) if isinstance(k, float) and 0 < k <= 1 else int(k)
    k_int = max(1, min(k_int, n))
    order = np.argsort(-y_score)[:k_int]
    return float(np.mean(y_true[order] == 1))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    top_k_fraction: float = 0.10,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba).astype(float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        metrics[f"precision_at_top{int(top_k_fraction * 100)}"] = precision_at_k(
            y_true, y_proba, k=top_k_fraction
        )
    return metrics
