from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    BINARY_TARGET_COL,
    REPORT_IMAGES_DIR,
    REPORT_TABLES_DIR,
    TARGET_COL,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    ensure_directories,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def ks_drift_report(
    train_path: Path = TRAIN_PARQUET_PATH,
    test_path: Path = TEST_PARQUET_PATH,
    top_k: int = 15,
) -> pd.DataFrame:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    exclude = {
        "url",
        "title",
        "title_source",
        BINARY_TARGET_COL,
        TARGET_COL,
        "is_viral",
    }
    cols = numeric_columns(train, exclude)
    rows: list[dict[str, object]] = []
    for c in cols:
        a = train[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        b = test[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        stat, pval = ks_2samp(a, b, alternative="two-sided")
        rows.append({"feature": c, "ks_statistic": float(stat), "p_value": float(pval)})
    out = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)
    ensure_directories()
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_TABLES_DIR / "feature_drift.csv"
    out.to_csv(csv_path, index=False)
    log.info("Drift report: %d признаков, топ сохранён в %s", len(out), csv_path)

    top = out.head(top_k)
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
    ax.barh(top["feature"][::-1], top["ks_statistic"][::-1], color="steelblue")
    ax.set_xlabel("KS statistic (train vs test)")
    ax.set_title("Топ признаков по дрифту распределения")
    plt.tight_layout()
    img_path = REPORT_IMAGES_DIR / "05_feature_drift_top.png"
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path, dpi=120)
    plt.close(fig)
    log.info("График дрифта: %s", img_path)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KS-дрифт train vs test по числовым признакам.")
    p.add_argument("--train", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--test", type=Path, default=TEST_PARQUET_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ks_drift_report(train_path=args.train, test_path=args.test)


if __name__ == "__main__":
    main()
