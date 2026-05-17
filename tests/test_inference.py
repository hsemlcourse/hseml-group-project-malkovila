from __future__ import annotations

import joblib
import pandas as pd
import pytest

from src.config import MODELING_EXCLUDE_COLS, MODELS_DIR, SPLIT_META_PATH, VAL_PARQUET_PATH
from src.inference.context_features import VALID_CHANNELS, VALID_WEEKDAYS
from src.inference.feature_builder import FeatureBuilder, load_feature_columns
from src.inference.predictor import NewsViralityPredictor
from src.modeling.final_model import FINAL_MODEL_NAME

MODEL_PATH = MODELS_DIR / FINAL_MODEL_NAME
SVD_PATH = MODELS_DIR / "text_tfidf_svd_artifacts.joblib"


def _artifacts_ready() -> bool:
    return (
        MODEL_PATH.exists()
        and SVD_PATH.exists()
        and SPLIT_META_PATH.exists()
        and VAL_PARQUET_PATH.exists()
    )


pytestmark = pytest.mark.skipif(not _artifacts_ready(), reason="Нужны models/ и data/processed/ (make features-cp2 && make final-model)")


def test_feature_vector_length() -> None:
    builder = FeatureBuilder()
    vec = builder.build_vector(
        "Why AI Will Change Everything in 2026",
        channel="tech",
        weekday="monday",
    )
    assert vec.shape == (len(load_feature_columns()),)
    assert vec.shape == (144,)


def test_predictor_smoke() -> None:
    pred = NewsViralityPredictor()
    out = pred.predict(
        "10 Shocking Facts About Machine Learning You Need to Know",
        channel="tech",
        weekday="friday",
        threshold=0.5,
    )
    assert 0.0 < out["probability"] < 1.0
    assert isinstance(out["is_popular"], bool)
    assert out["popularity_threshold_shares"] == 1400.0


def test_predictor_matches_parquet_row() -> None:
    df = pd.read_parquet(VAL_PARQUET_PATH)
    row = df.iloc[0]
    title = str(row["title"])
    channel_col = [c for c in df.columns if c.startswith("data_channel_is_") and row[c] == 1]
    assert len(channel_col) == 1
    channel_key = channel_col[0].replace("data_channel_is_", "")
    weekday_col = [c for c in df.columns if c.startswith("weekday_is_") and row[c] == 1]
    assert len(weekday_col) == 1
    weekday_key = weekday_col[0].replace("weekday_is_", "")

    bundle = joblib.load(MODEL_PATH)
    feature_cols = [c for c in df.columns if c not in MODELING_EXCLUDE_COLS]
    x = row[feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
    expected = float(bundle["model"].predict_proba(x)[0, 1])

    pred = NewsViralityPredictor()
    got = pred.predict(title, channel=channel_key, weekday=weekday_key)["probability"]
    assert abs(got - expected) < 0.02


def test_invalid_channel_raises() -> None:
    builder = FeatureBuilder()
    with pytest.raises(ValueError, match="канал"):
        builder.build_vector("Hello world", channel="unknown", weekday="monday")


def test_valid_channel_weekday_sets() -> None:
    assert "tech" in VALID_CHANNELS
    assert "monday" in VALID_WEEKDAYS
