from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import (
    BINARY_TARGET_COL,
    MODELING_EXCLUDE_COLS,
    MODELS_DIR,
    REPORT_IMAGES_DIR,
    REPORT_TABLES_DIR,
    SEED,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    ensure_directories,
)
from src.modeling.metrics import compute_metrics
from src.modeling.tuners import tune_lightgbm_optuna
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)

FINAL_MODEL_NAME = "final_lgbm_cp2.joblib"
PERM_IMPORTANCE_CSV = "permutation_importance.csv"
FINAL_METRICS_CSV = "final_metrics.csv"


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c not in MODELING_EXCLUDE_COLS]
    return df[cols].to_numpy(dtype=np.float64), df[BINARY_TARGET_COL].to_numpy(dtype=int), cols


def run_final_pipeline(
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    model_out: Path = MODELS_DIR / FINAL_MODEL_NAME,
    fast: bool = False,
) -> dict[str, Any]:
    set_global_seed(SEED)
    ensure_directories()
    if fast:
        import os

        os.environ["CP2_FAST"] = "1"

    x_tr, y_tr, feature_names = load_xy(train_path)
    x_va, y_va, _ = load_xy(val_path)
    x_te, y_te, _ = load_xy(test_path)

    model, params, _cv_auc = tune_lightgbm_optuna(x_tr, y_tr)
    log.info("Финальная модель LightGBM, CV AUC ≈ %.4f | params=%s", _cv_auc, params)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "params": params}, model_out)
    log.info("Модель сохранена в %s", model_out)

    def metrics_for(m: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        proba = m.predict_proba(x)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return compute_metrics(y, pred, proba)

    m_train = metrics_for(model, x_tr, y_tr)
    m_val = metrics_for(model, x_va, y_va)
    m_test = metrics_for(model, x_te, y_te)

    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, mm in ("train", m_train), ("val", m_val), ("test", m_test):
        row = {"split": split, **mm}
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT_TABLES_DIR / FINAL_METRICS_CSV, index=False)

    r = permutation_importance(
        model,
        x_va,
        y_va,
        n_repeats=10,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importances_mean": r.importances_mean,
            "importances_std": r.importances_std,
        }
    ).sort_values("importances_mean", ascending=False)
    imp_df.to_csv(REPORT_TABLES_DIR / PERM_IMPORTANCE_CSV, index=False)

    top_k = min(20, len(imp_df))
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.22)))
    sub = imp_df.head(top_k).iloc[::-1]
    ax.barh(sub["feature"], sub["importances_mean"], xerr=sub["importances_std"], color="teal")
    ax.set_xlabel("Перестановочная важность (↓ ROC-AUC)")
    fig.suptitle("Permutation importance на val (LightGBM)")
    fig.tight_layout()
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_IMAGES_DIR / "07_permutation_importance.png", dpi=120)
    plt.close(fig)

    result = {
        "params": params,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "model_path": str(model_out),
    }
    (REPORT_TABLES_DIR / "final_model_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Финальная LightGBM + permutation importance + метрики.")
    p.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--val", type=Path, default=VAL_PARQUET_PATH)
    p.add_argument("--test", type=Path, default=TEST_PARQUET_PATH)
    p.add_argument("--model-out", type=Path, default=MODELS_DIR / FINAL_MODEL_NAME)
    p.add_argument("--fast", action="store_true", help="CP2_FAST для меньшего числа Optuna trials")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_final_pipeline(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_out=args.model_out,
        fast=args.fast,
    )


if __name__ == "__main__":
    main()
