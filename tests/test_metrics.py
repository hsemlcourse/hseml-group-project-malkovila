from __future__ import annotations

import numpy as np

from src.modeling.metrics import compute_metrics, precision_at_k


def test_precision_at_k_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    assert precision_at_k(y_true, y_score, k=3) == 1.0
    assert precision_at_k(y_true, y_score, k=0.6) == 1.0


def test_precision_at_k_worst_ranking() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    assert precision_at_k(y_true, y_score, k=2) == 0.0


def test_compute_metrics_contains_expected_keys() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    y_pred = (y_proba > 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for key in ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_metrics_handles_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = y_true.copy()
    y_proba = y_true.astype(float)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
