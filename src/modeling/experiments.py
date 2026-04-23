from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BINARY_TARGET_COL,
    CSV_TITLE_FEATURES,
    DATA_CHANNEL_COLS,
    MODELS_DIR,
    REPORT_TABLES_DIR,
    SEED,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    WEEKDAY_COLS,
    ensure_directories,
)
from src.features.title_features import TITLE_FEATURE_COLUMNS
from src.modeling.metrics import compute_metrics
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def _feature_set(train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    baseline_cols = [c for c in CSV_TITLE_FEATURES if c in train.columns]
    engineered_cols = [c for c in TITLE_FEATURE_COLUMNS if c in train.columns]
    context_cols = [c for c in (*DATA_CHANNEL_COLS, *WEEKDAY_COLS) if c in train.columns]
    return baseline_cols, engineered_cols, context_cols


def _evaluate(pipe, x, y) -> dict[str, float]:
    y_pred = pipe.predict(x)
    y_proba = (
        pipe.predict_proba(x)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(x)
    )
    return compute_metrics(y, y_pred, y_proba)


def run_experiments(
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    models_dir: Path = MODELS_DIR,
    metrics_out: Path = REPORT_TABLES_DIR / "experiments_cp1.csv",
) -> pd.DataFrame:
    set_global_seed(SEED)
    ensure_directories()

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    baseline_cols, engineered_cols, context_cols = _feature_set(train)
    full_cols = baseline_cols + engineered_cols + context_cols

    def to_xy(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return df[cols].to_numpy(dtype=np.float64), df[BINARY_TARGET_COL].to_numpy(dtype=int)

    experiments: list[dict[str, object]] = []

    log.info("Эксперимент 1: LogReg на baseline + инженерные + контекст (%d признаков)", len(full_cols))
    pipe1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, solver="lbfgs")),
        ]
    )
    x_train, y_train = to_xy(train, full_cols)
    pipe1.fit(x_train, y_train)
    joblib.dump(pipe1, models_dir / "exp1_logreg_full.joblib")
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        x, y = to_xy(df, full_cols)
        metrics = _evaluate(pipe1, x, y)
        experiments.append(
            {
                "experiment": "exp1_logreg_full",
                "hypothesis": "Инженерные признаки заголовка поднимают ROC-AUC на val как минимум на 0.02",
                "model": "LogReg(C=1.0) на CSV-признаках заголовка + инженерных + контекст",
                "split": split_name,
                **metrics,
            }
        )

    log.info("Эксперимент 2: решающее дерево глубины 6 на том же наборе признаков")
    pipe2 = Pipeline([("clf", DecisionTreeClassifier(max_depth=6, random_state=SEED))])
    pipe2.fit(x_train, y_train)
    joblib.dump(pipe2, models_dir / "exp2_tree_depth6.joblib")
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        x, y = to_xy(df, full_cols)
        metrics = _evaluate(pipe2, x, y)
        experiments.append(
            {
                "experiment": "exp2_tree_depth6",
                "hypothesis": "Неглубокая нелинейная модель улавливает взаимодействия признаков",
                "model": "DecisionTree(max_depth=6)",
                "split": split_name,
                **metrics,
            }
        )

    df_out = pd.DataFrame(experiments)
    df_out.to_csv(metrics_out, index=False)
    log.info("Записано %d строк в %s", len(df_out), metrics_out)
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Эксперименты CP1: LogReg на полном наборе признаков и дерево глубины 6.")
    parser.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    parser.add_argument("--val", type=Path, default=VAL_PARQUET_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PARQUET_PATH)
    parser.add_argument("--metrics-out", type=Path, default=REPORT_TABLES_DIR / "experiments_cp1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    main()
