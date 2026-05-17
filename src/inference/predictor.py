from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORT_TABLES_DIR, SPLIT_META_PATH
from src.inference.feature_builder import FeatureBuilder
from src.modeling.final_model import FINAL_MODEL_NAME

DEFAULT_MODEL_PATH = MODELS_DIR / FINAL_MODEL_NAME
PERM_IMPORTANCE_PATH = REPORT_TABLES_DIR / "permutation_importance.csv"


class NewsViralityPredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = SPLIT_META_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}. Запустите: make final-model"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.params: dict[str, Any] = bundle.get("params", {})

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.popularity_threshold_shares = float(meta.get("popularity_threshold", 1400.0))

        self._builder = FeatureBuilder(feature_columns=self.feature_names)
        self._top_features = self._load_top_features()

    @staticmethod
    def _load_top_features(k: int = 5) -> list[dict[str, float]]:
        if not PERM_IMPORTANCE_PATH.exists():
            return []
        imp = pd.read_csv(PERM_IMPORTANCE_PATH).head(k)
        return [
            {"feature": str(r["feature"]), "importance": float(r["importances_mean"])}
            for _, r in imp.iterrows()
        ]

    def predict(
        self,
        title: str,
        channel: str,
        weekday: str,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        x = self._builder.build_vector(title, channel, weekday)
        proba = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        return {
            "probability": proba,
            "is_popular": bool(proba >= threshold),
            "classification_threshold": threshold,
            "popularity_threshold_shares": self.popularity_threshold_shares,
            "model": "LightGBM",
            "top_features_global": self._top_features,
        }


@lru_cache(maxsize=1)
def get_predictor() -> NewsViralityPredictor:
    return NewsViralityPredictor()
