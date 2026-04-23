from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_TARGET_COL,
    CSV_TITLE_FEATURES,
    MODELS_DIR,
    REPORT_TABLES_DIR,
    SEED,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    ensure_directories,
)
from src.modeling.metrics import compute_metrics
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def _load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_baseline_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")),
        ]
    )


def train_baseline(
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    features: tuple[str, ...] = CSV_TITLE_FEATURES,
    target: str = BINARY_TARGET_COL,
    model_out: Path = MODELS_DIR / "baseline_logreg.joblib",
    metrics_out: Path = REPORT_TABLES_DIR / "baseline_metrics.csv",
) -> dict[str, dict[str, float]]:
    set_global_seed(SEED)
    ensure_directories()

    train = _load_split(train_path)
    val = _load_split(val_path)
    test = _load_split(test_path)

    missing = [c for c in features if c not in train.columns]
    if missing:
        raise ValueError(f"В train-сплите отсутствуют базовые признаки: {missing}")

    x_train = train[list(features)].to_numpy(dtype=np.float64)
    x_val = val[list(features)].to_numpy(dtype=np.float64)
    x_test = test[list(features)].to_numpy(dtype=np.float64)
    y_train = train[target].to_numpy(dtype=int)
    y_val = val[target].to_numpy(dtype=int)
    y_test = test[target].to_numpy(dtype=int)

    pipe = make_baseline_pipeline()
    log.info("Обучаем baseline LogReg на %d строк x %d признаков", *x_train.shape)
    pipe.fit(x_train, y_train)

    results: dict[str, dict[str, float]] = {}
    for split_name, x, y in (("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)):
        y_pred = pipe.predict(x)
        y_proba = pipe.predict_proba(x)[:, 1]
        results[split_name] = compute_metrics(y, y_pred, y_proba)
        log.info("%s: %s", split_name, {k: f"{v:.4f}" for k, v in results[split_name].items()})

    joblib.dump(pipe, model_out)
    log.info("Модель сохранена в %s", model_out)

    rows = []
    for split_name, metrics in results.items():
        for metric_name, value in metrics.items():
            rows.append({"model": "baseline_logreg", "split": split_name, "metric": metric_name, "value": value})
    pd.DataFrame(rows).to_csv(metrics_out, index=False)
    log.info("Метрики сохранены в %s", metrics_out)

    (metrics_out.with_suffix(".json")).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение baseline-логрегрессии на 5 готовых признаках заголовка.")
    parser.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    parser.add_argument("--val", type=Path, default=VAL_PARQUET_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PARQUET_PATH)
    parser.add_argument("--model-out", type=Path, default=MODELS_DIR / "baseline_logreg.joblib")
    parser.add_argument("--metrics-out", type=Path, default=REPORT_TABLES_DIR / "baseline_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_baseline(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    main()
