from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src.config import REPORT_TABLES_DIR, SEED, ensure_directories
from src.modeling.experiments_cp2 import evaluate_split, load_xy
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def evaluate_time_splits(
    train_time: Path,
    val_time: Path,
    test_time: Path,
    out_csv: Path,
) -> pd.DataFrame:
    set_global_seed(SEED)
    x_tr, y_tr, _ = load_xy(train_time)
    x_va, y_va, _ = load_xy(val_time)
    x_te, y_te, _ = load_xy(test_time)

    clf = LGBMClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(x_tr, y_tr)
    m_tr = evaluate_split(clf, x_tr, y_tr)
    m_va = evaluate_split(clf, x_va, y_va)
    m_te = evaluate_split(clf, x_te, y_te)

    row = {
        "split_strategy": "time_ordered",
        "train_roc_auc": m_tr["roc_auc"],
        "val_roc_auc": m_va["roc_auc"],
        "test_roc_auc": m_te["roc_auc"],
    }
    ensure_directories()
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    log.info("Time-split метрики: %s", row)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-time", type=Path, default=None)
    p.add_argument("--val-time", type=Path, default=None)
    p.add_argument("--test-time", type=Path, default=None)
    p.add_argument("--out", type=Path, default=REPORT_TABLES_DIR / "time_split_metrics.csv")
    return p.parse_args()


def main() -> None:
    from src.config import PROCESSED_DIR

    args = parse_args()
    train_t = args.train_time or (PROCESSED_DIR / "train_time.parquet")
    val_t = args.val_time or (PROCESSED_DIR / "val_time.parquet")
    test_t = args.test_time or (PROCESSED_DIR / "test_time.parquet")
    if not train_t.is_file():
        log.error("Нет time-parquet. Сначала: python -m src.features.time_split")
        raise SystemExit(1)
    evaluate_time_splits(train_t, val_t, test_t, args.out)


if __name__ == "__main__":
    main()
