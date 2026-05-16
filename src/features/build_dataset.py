from __future__ import annotations

import argparse
import json
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
    SPLIT_META_PATH,
    TARGET_COL,
    TEST_FULL_PARQUET_PATH,
    TEST_PARQUET_PATH,
    TEST_SIZE,
    TITLES_JSONL_PATH,
    TRAIN_FULL_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_FULL_PARQUET_PATH,
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
    df = df.copy()
    df.columns = df.columns.str.strip()
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


def _attach_titles(df: pd.DataFrame, input_csv: Path, titles_jsonl: Path | None) -> pd.DataFrame:
    if titles_jsonl is not None and titles_jsonl.exists():
        log.info("Читаем кэш заголовков из %s", titles_jsonl)
        titles_df = parse_titles(input_csv=input_csv, output_jsonl=titles_jsonl, mode="slug")
        titles_df = titles_df[["url", "title", "title_source"]]
        df = df.merge(titles_df, on="url", how="left")
    else:
        log.info("Кэш заголовков не найден, извлекаем их из slug'ов на лету")
        df = df.copy()
        df["title"] = df["url"].astype(str).map(extract_title_from_slug)
        df["title_source"] = "slug"
    df["title"] = df["title"].fillna("").astype(str)
    return df


def _apply_targets(
    df: pd.DataFrame,
    popularity_threshold: float,
    viral_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out[BINARY_TARGET_COL] = (out[TARGET_COL] >= popularity_threshold).astype(int)
    out[VIRAL_TARGET_COL] = (out[TARGET_COL] >= viral_threshold).astype(int)
    return out


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = list(TITLE_FEATURE_COLUMNS)
    cols += [c for c in CSV_TITLE_FEATURES if c in df.columns]
    cols += [c for c in DATA_CHANNEL_COLS if c in df.columns]
    cols += [c for c in WEEKDAY_COLS if c in df.columns]
    return cols


def _meta_columns(df: pd.DataFrame) -> list[str]:
    cols = ["url", "title", "title_source"]
    if "timedelta" in df.columns:
        cols.append("timedelta")
    cols.extend([TARGET_COL, BINARY_TARGET_COL, VIRAL_TARGET_COL])
    return cols


def _write_split_meta(
    path: Path,
    *,
    pop_thr: float,
    viral_thr: float,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    stratify_median: float,
    feature_cols: list[str],
    full: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "popularity_threshold": pop_thr,
        "viral_threshold": viral_thr,
        "viral_quantile": VIRAL_TOP_QUANTILE,
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "stratify_proxy_median_shares": stratify_median,
        "threshold_source": "train_median_and_train_quantile",
        "full_features": full,
        "n_feature_columns": len(feature_cols),
        "feature_columns": feature_cols,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Мета сплита записана в %s", path)


def build_dataset(
    input_csv: Path = RAW_CSV_PATH,
    titles_jsonl: Path = TITLES_JSONL_PATH,
    output_features: Path = FEATURES_PARQUET_PATH,
    train_path: Path = TRAIN_PARQUET_PATH,
    val_path: Path = VAL_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    popularity_threshold: float | None = None,
    split_meta_path: Path = SPLIT_META_PATH,
    full: bool = False,
    train_full_path: Path = TRAIN_FULL_PARQUET_PATH,
    val_full_path: Path = VAL_FULL_PARQUET_PATH,
    test_full_path: Path = TEST_FULL_PARQUET_PATH,
) -> dict[str, pd.DataFrame]:
    """Собирает processed-датасет: FE → сплит → пороги is_popular/is_viral только по train.

    Стратификация сплита — по прокси-классу ``shares >= median(shares)`` по всей выборке,
    чтобы распределение оставалось сопоставимо с CP1; финальные метки считаются без утечки
    порога из val/test.
    """
    set_global_seed(SEED)
    ensure_directories()

    log.info("Читаем %s", input_csv)
    df = pd.read_csv(input_csv)
    df = _clean_raw(df)
    df = _attach_titles(df, input_csv=input_csv, titles_jsonl=titles_jsonl if titles_jsonl.exists() else None)

    log.info("Считаем %d инженерных признаков заголовка", len(TITLE_FEATURE_COLUMNS))
    tf = add_title_features(df["title"])
    df = pd.concat([df.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)

    n_rows = len(df)
    median_for_stratify = float(df[TARGET_COL].median())
    y_stratify = (df[TARGET_COL] >= median_for_stratify).to_numpy(dtype=int)

    idx = np.arange(n_rows)
    idx_train, idx_tmp, y_train_strat, y_tmp_strat = train_test_split(
        idx,
        y_stratify,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=SEED,
        stratify=y_stratify,
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    idx_val, idx_test, _yv, _yt = train_test_split(
        idx_tmp,
        y_tmp_strat,
        test_size=relative_test,
        random_state=SEED,
        stratify=y_tmp_strat,
    )

    shares_train = df.iloc[idx_train][TARGET_COL]
    pop_thr = float(popularity_threshold) if popularity_threshold is not None else float(shares_train.median())
    viral_thr = float(shares_train.quantile(VIRAL_TOP_QUANTILE))

    df = _apply_targets(df, pop_thr, viral_thr)

    feature_cols = _select_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float64)

    meta_cols = _meta_columns(df)
    df_full = df[meta_cols + feature_cols].copy()

    if full:
        from src.features.build_dataset_full import augment_full_features

        df_full = augment_full_features(
            df_full,
            train_idx=idx_train,
            val_idx=idx_val,
            test_idx=idx_test,
        )
        feature_cols = [c for c in df_full.columns if c not in meta_cols]

    df_full.to_parquet(output_features, index=False)
    log.info("Сохранён полный parquet с признаками: %s (%d строк, %d колонок)", output_features, *df_full.shape)

    df_train = df_full.iloc[idx_train].reset_index(drop=True)
    df_val = df_full.iloc[idx_val].reset_index(drop=True)
    df_test = df_full.iloc[idx_test].reset_index(drop=True)

    if full:
        df_train.to_parquet(train_full_path, index=False)
        df_val.to_parquet(val_full_path, index=False)
        df_test.to_parquet(test_full_path, index=False)
        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        df_test.to_parquet(test_path, index=False)
    else:
        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        df_test.to_parquet(test_path, index=False)

    log.info("Сохранены сплиты: train=%d, val=%d, test=%d", len(df_train), len(df_val), len(df_test))
    log.info(
        "Порог популярности (train)=%.0f | виральности (train q=%.2f)=%.0f",
        pop_thr,
        VIRAL_TOP_QUANTILE,
        viral_thr,
    )
    log.info(
        "Баланс is_popular train=%.3f val=%.3f test=%.3f",
        df_train[BINARY_TARGET_COL].mean(),
        df_val[BINARY_TARGET_COL].mean(),
        df_test[BINARY_TARGET_COL].mean(),
    )

    _write_split_meta(
        split_meta_path,
        pop_thr=pop_thr,
        viral_thr=viral_thr,
        seed=SEED,
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
        stratify_median=median_for_stratify,
        feature_cols=feature_cols,
        full=full,
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
        help="Явный порог популярности (иначе — медиана shares в train-подвыборке).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Расширенные признаки: TF-IDF+SVD, readability (см. build_dataset_full).",
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
        split_meta_path=args.output_dir / "split_meta.json",
        full=args.full,
        train_full_path=args.output_dir / "train_full.parquet",
        val_full_path=args.output_dir / "val_full.parquet",
        test_full_path=args.output_dir / "test_full.parquet",
    )


if __name__ == "__main__":
    main()
