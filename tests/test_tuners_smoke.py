from __future__ import annotations

import os


def test_tune_knn_runs_and_predicts() -> None:
    os.environ["CP2_FAST"] = "1"
    from sklearn.datasets import make_classification

    from src.modeling.tuners import tune_knn

    x, y = make_classification(n_samples=400, n_features=12, random_state=0)
    model, _params, cv_auc = tune_knn(x, y, n_iter=12)
    proba = model.predict_proba(x[:5])[:, 1]
    assert len(proba) == 5
    assert cv_auc > 0.4
