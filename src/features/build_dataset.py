from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BINARY_TARGET_COL,
    CSV_TITLE_FEATURES,
    DATA_CHANNEL_COLS,
    FEATURES_PARQUET_PATH,
    PROCESSED_DIR,
    RAW_CSV_PATH,
    SEED,
    TARGET_COL,
    TEST_PARQUET_PATH,
    TEST_SIZE,
    TITLES_JSONL_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
    VAL_SIZE,
    VIRAL_TARGET_COL,
    VIRAL_TOP_QUANTILE,
    WEEKDAY_COLS,
    ensure_directories,
)
from src.data.parse_titles import extract_title_from_slug, parse_titles
from src.features.title_features import TITLE_FEATURE_COLUMNS, add_title_features
from src.utils.logging_setup import get_logger
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    if "n_unique_tokens" in df.columns:
        df = df[df["n_unique_tokens"] <= 1.0]
    if "n_non_stop_words" in df.columns:
        df = df[df["n_non_stop_words"] <= 1.0]
    if "n_tokens_content" in df.columns:
        df = df[df["n_tokens_content"] >= 0.0]

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    log.info("После очистки осталось %d / %d строк (%.2f%%)", len(df), before, 100 * len(df) / before)
    return df


def _attach_titles(df: pd.DataFrame, titles_jsonl: Path | None) -> pd.DataFrame:
    if titles_jsonl is not None and titles_jsonl.exists():
        log.info("Читаем кэш заголовков из %s", titles_jsonl)
        titles_df = parse_titles(input_csv=RAW_CSV_PATH, output_jsonl=titles_jsonl, mode="slug")
        titles_df = titles_df[["url", "title", "title_source"]]
        df = df.merge(titles_df, on="url", how="left")
    else:
        log.info("Кэш заголовков не найден, извлекаем их из slug'ов на лету")
        df = df.copy()
        df["title"] = df["url"].astype(str).map(extract_title_from_slug)
        df["title_source"] = "slug"
    df["title"] = df["title"].fillna("").astype(str)
    return df


def _add_targets(df: pd.DataFrame, popularity_threshold: float | None) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()
    if popularity_threshold is None:
        popularity_threshold = float(df[TARGET_COL].median())
    viral_threshold = float(df[TARGET_COL].quantile(VIRAL_TOP_QUANTILE))

    df[BINARY_TARGET_COL] = (df[TARGET_COL] >= popularity_threshold).astype(int)
    df[VIRAL_TARGET_COL] = (df[TARGET_COL] >= viral_threshold).astype(int)
    log.info(
        "Порог популярности=%.0f shares -> баланс %.3f | Порог виральности=%.0f -> баланс %.3f",
        popularity_threshold,
        df[BINARY_TARGET_COL].mean(),
        viral_threshold,
        df[VIRAL_TARGET_COL].mean(),
    )
    return df, popularity_threshold, viral_threshold


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = list(TITLE_FEATURE_COLUMNS)
    cols += [c for c in CSV_TITLE_FEATURES if c in df.columns]
    cols += [c for c in DATA_CHANNEL_COLS if c in df.columns]
    cols += [c for c in WEEKDAY_COLS if c in df.columns]
    return cols


def build_dataset(
    input_csv: Path = RAW_CSV_PATH,
    titles_jsonl: Path = TITLES_JSONL_PATH,
    output_features: Path = FEATURES_PARQUET_PATH,
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    popularity_threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    set_global_seed(SEED)
    ensure_directories()

    log.info("Читаем %s", input_csv)
    df = pd.read_csv(input_csv)
    df = _clean_raw(df)
    df = _attach_titles(df, titles_jsonl=titles_jsonl if titles_jsonl.exists() else None)
    df, pop_thr, viral_thr = _add_targets(df, popularity_threshold)

    log.info("Считаем %d инженерных признаков заголовка", len(TITLE_FEATURE_COLUMNS))
    tf = add_title_features(df["title"])
    df = pd.concat([df.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)

    feature_cols = _select_feature_columns(df)
    meta_cols = ["url", "title", "title_source", TARGET_COL, BINARY_TARGET_COL, VIRAL_TARGET_COL]
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float64)

    df_full = df[meta_cols + feature_cols].copy()
    df_full.to_parquet(output_features, index=False)
    log.info("Сохранён полный parquet с признаками: %s (%d строк, %d колонок)", output_features, *df_full.shape)

    y = df_full[BINARY_TARGET_COL].to_numpy()
    idx = np.arange(len(df_full))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, y, test_size=(VAL_SIZE + TEST_SIZE), random_state=SEED, stratify=y
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    idx_val, idx_test, _y_val, _y_test = train_test_split(
        idx_tmp, y_tmp, test_size=relative_test, random_state=SEED, stratify=y_tmp
    )

    df_train = df_full.iloc[idx_train].reset_index(drop=True)
    df_val = df_full.iloc[idx_val].reset_index(drop=True)
    df_test = df_full.iloc[idx_test].reset_index(drop=True)

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)
    log.info("Сохранены сплиты: train=%d, val=%d, test=%d", len(df_train), len(df_val), len(df_test))
    log.info(
        "Баланс классов train=%.3f val=%.3f test=%.3f",
        df_train[BINARY_TARGET_COL].mean(),
        df_val[BINARY_TARGET_COL].mean(),
        df_test[BINARY_TARGET_COL].mean(),
    )

    meta = {
        "popularity_threshold": pop_thr,
        "viral_threshold": viral_thr,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
    }
    log.info("Пайплайн завершён: %s", meta)
    return {
        "full": df_full,
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "meta": pd.Series(meta),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Очистка данных, добавление признаков заголовка и стратифицированный сплит train/val/test.",
    )
    parser.add_argument("--input", type=Path, default=RAW_CSV_PATH)
    parser.add_argument("--titles", type=Path, default=TITLES_JSONL_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Явный порог популярности (по умолчанию — медиана shares).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(
        input_csv=args.input,
        titles_jsonl=args.titles,
        output_features=args.output_dir / "features.parquet",
        train_path=args.output_dir / "train.parquet",
        val_path=args.output_dir / "val.parquet",
        test_path=args.output_dir / "test.parquet",
        popularity_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
