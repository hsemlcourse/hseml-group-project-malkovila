from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BINARY_TARGET_COL
from src.features.build_dataset import build_dataset


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 400
    urls = [f"http://mashable.com/2014/01/{(i % 28) + 1:02d}/sample-article-{i}/" for i in range(n)]
    df = pd.DataFrame(
        {
            "url": urls,
            "timedelta": rng.integers(100, 700, size=n),
            "n_tokens_title": rng.integers(4, 16, size=n).astype(float),
            "title_subjectivity": rng.random(n),
            "title_sentiment_polarity": rng.uniform(-1, 1, n),
            "abs_title_subjectivity": rng.random(n),
            "abs_title_sentiment_polarity": rng.random(n),
            "n_unique_tokens": rng.uniform(0.2, 0.9, n),
            "n_non_stop_words": rng.uniform(0.5, 1.0, n),
            "n_tokens_content": rng.integers(100, 1200, size=n).astype(float),
            "data_channel_is_tech": rng.integers(0, 2, size=n),
            "weekday_is_monday": rng.integers(0, 2, size=n),
            "is_weekend": rng.integers(0, 2, size=n),
            "shares": rng.integers(100, 50000, size=n),
        }
    )
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_stratified_split_keeps_class_balance(tmp_path: Path, synthetic_csv: Path) -> None:
    out = tmp_path
    splits = build_dataset(
        input_csv=synthetic_csv,
        titles_jsonl=out / "titles.jsonl",
        output_features=out / "features.parquet",
        train_path=out / "train.parquet",
        val_path=out / "val.parquet",
        test_path=out / "test.parquet",
    )
    train, val, test = splits["train"], splits["val"], splits["test"]

    assert len(train) + len(val) + len(test) == len(splits["full"])
    assert abs(len(train) / len(splits["full"]) - 0.70) < 0.03
    assert abs(len(val) / len(splits["full"]) - 0.15) < 0.03

    balances = [df[BINARY_TARGET_COL].mean() for df in (train, val, test)]
    for b in balances:
        assert 0.4 < b < 0.6


def test_no_url_leakage_between_splits(tmp_path: Path, synthetic_csv: Path) -> None:
    out = tmp_path
    splits = build_dataset(
        input_csv=synthetic_csv,
        titles_jsonl=out / "titles.jsonl",
        output_features=out / "features.parquet",
        train_path=out / "train.parquet",
        val_path=out / "val.parquet",
        test_path=out / "test.parquet",
    )
    urls_train = set(splits["train"]["url"])
    urls_val = set(splits["val"]["url"])
    urls_test = set(splits["test"]["url"])
    assert not (urls_train & urls_val)
    assert not (urls_train & urls_test)
    assert not (urls_val & urls_test)


def test_split_is_deterministic(tmp_path: Path, synthetic_csv: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    def _run(out: Path) -> pd.DataFrame:
        return build_dataset(
            input_csv=synthetic_csv,
            titles_jsonl=out / "titles.jsonl",
            output_features=out / "features.parquet",
            train_path=out / "train.parquet",
            val_path=out / "val.parquet",
            test_path=out / "test.parquet",
        )["train"]

    t1 = _run(out1).reset_index(drop=True)
    t2 = _run(out2).reset_index(drop=True)
    pd.testing.assert_frame_equal(t1[["url", BINARY_TARGET_COL]], t2[["url", BINARY_TARGET_COL]])
