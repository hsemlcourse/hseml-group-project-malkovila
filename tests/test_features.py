from __future__ import annotations

import pandas as pd

from src.features.title_features import (
    TITLE_FEATURE_COLUMNS,
    add_title_features,
    compute_title_features,
)


def test_feature_columns_match_schema() -> None:
    features = compute_title_features("5 Shocking Reasons You Won't Believe")
    assert set(features.keys()) == set(TITLE_FEATURE_COLUMNS)


def test_features_are_numeric_and_finite() -> None:
    features = compute_title_features("Amazon Instant Video Browser")
    for name, value in features.items():
        assert isinstance(value, float), f"{name}: ожидался float"
        assert value == value, f"{name}: получен NaN"


def test_empty_title_produces_zero_length() -> None:
    features = compute_title_features("")
    assert features["tf_title_char_len"] == 0.0
    assert features["tf_title_word_len"] == 0.0


def test_clickbait_and_structural_indicators_fire() -> None:
    text = "You Won't Believe These 7 Shocking Hacks!"
    features = compute_title_features(text)
    assert features["tf_has_number"] == 1.0
    assert features["tf_has_exclamation"] == 1.0
    assert features["tf_clickbait_word_count"] >= 2.0
    assert features["tf_clickbait_phrase_count"] >= 1.0


def test_add_title_features_vectorized_no_nan() -> None:
    titles = pd.Series(
        [
            "Amazon Instant Video Browser",
            "Why You Need This Incredible Hack",
            "",
            "Sad Crisis Shakes Global Markets",
        ]
    )
    df = add_title_features(titles)
    assert df.shape == (4, len(TITLE_FEATURE_COLUMNS))
    assert not df.isna().any().any()
    assert set(df.columns) == set(TITLE_FEATURE_COLUMNS)
