from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    BINARY_TARGET_COL,
    FEATURES_PARQUET_PATH,
    PROCESSED_DIR,
    SEED,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    VIRAL_TARGET_COL,
    VIRAL_TOP_QUANTILE,
    ensure_directories,
)
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def time_ordered_split(
    df: pd.DataFrame,
    timedelta_col: str = "timedelta",
    train_frac: float = TRAIN_SIZE,
    val_frac: float = VAL_SIZE,
    test_frac: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение по времени: сортируем по timedelta (статьи «старше» в train)."""
    set_global_seed(seed)
    if timedelta_col not in df.columns:
        raise ValueError(f"Колонка {timedelta_col} отсутствует в датафрейме")

    d = df.sort_values(timedelta_col, ascending=False).reset_index(drop=True)
    n = len(d)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Слишком малый датасет для time-split 70/15/15")

    train = d.iloc[:n_train].copy()
    val = d.iloc[n_train : n_train + n_val].copy()
    test = d.iloc[n_train + n_val :].copy()

    pop_thr = float(train[TARGET_COL].median())
    viral_thr = float(train[TARGET_COL].quantile(VIRAL_TOP_QUANTILE))
    for part in (train, val, test):
        part[BINARY_TARGET_COL] = (part[TARGET_COL] >= pop_thr).astype(int)
        part[VIRAL_TARGET_COL] = (part[TARGET_COL] >= viral_thr).astype(int)

    log.info(
        "Time-split: train=%d val=%d test=%d | пороги pop=%.0f viral=%.0f",
        len(train),
        len(val),
        len(test),
        pop_thr,
        viral_thr,
    )
    return train, val, test


def build_time_split_parquets(
    features_path: Path = FEATURES_PARQUET_PATH,
    out_dir: Path = PROCESSED_DIR,
) -> tuple[Path, Path, Path]:
    ensure_directories()
    if not features_path.is_file():
        raise FileNotFoundError(f"Нет {features_path}; сначала make features")

    df = pd.read_parquet(features_path)
    train, val, test = time_ordered_split(df)

    train_path = out_dir / "train_time.parquet"
    val_path = out_dir / "val_time.parquet"
    test_path = out_dir / "test_time.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)
    log.info("Записаны time-parquet: %s %s %s", train_path, val_path, test_path)
    return train_path, val_path, test_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-based train/val/test по колонке timedelta.")
    p.add_argument("--input", type=Path, default=FEATURES_PARQUET_PATH)
    p.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_time_split_parquets(features_path=args.input, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
