from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SPLIT_META_PATH
from src.features.build_dataset_full import TEXT_SVD_ARTIFACT_NAME
from src.features.readability import READABILITY_COLUMNS, add_readability_features
from src.features.text_vectorizers import transform_titles_with_artifacts
from src.features.title_features import add_title_features
from src.inference.context_features import encode_channel_weekday
from src.inference.csv_title_features import compute_csv_title_features


def load_feature_columns(meta_path: Path = SPLIT_META_PATH) -> list[str]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cols = meta.get("feature_columns")
    if not cols:
        raise ValueError(f"В {meta_path} нет feature_columns")
    return list(cols)


class FeatureBuilder:
    """Собирает вектор признаков (144) для одного заголовка + контекста."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        svd_artifact_path: Path | None = None,
        meta_path: Path = SPLIT_META_PATH,
    ) -> None:
        self.feature_columns = feature_columns or load_feature_columns(meta_path)
        art_path = svd_artifact_path or (MODELS_DIR / TEXT_SVD_ARTIFACT_NAME)
        if not art_path.exists():
            raise FileNotFoundError(
                f"Нет артефактов TF-IDF+SVD: {art_path}. Запустите: make features-cp2"
            )
        import joblib

        self._svd_artifacts = joblib.load(art_path)

    def build_row(self, title: str, channel: str, weekday: str) -> pd.DataFrame:
        title = (title or "").strip()
        if not title:
            raise ValueError("Заголовок не может быть пустым")

        titles = pd.Series([title])
        row: dict[str, float] = {}

        tf_df = add_title_features(titles)
        for c in tf_df.columns:
            row[c] = float(tf_df[c].iloc[0])

        read_df = add_readability_features(titles)
        for c in READABILITY_COLUMNS:
            row[c] = float(read_df[c].iloc[0])

        row.update(compute_csv_title_features(title))
        row.update(encode_channel_weekday(channel, weekday))

        svd_mat = transform_titles_with_artifacts(titles, self._svd_artifacts)
        from src.features.text_vectorizers import char_word_svd_column_names

        svd_names = char_word_svd_column_names(
            self._svd_artifacts.char_dim,
            self._svd_artifacts.word_dim,
        )
        for i, name in enumerate(svd_names):
            row[name] = float(svd_mat[0, i])

        df = pd.DataFrame([row])
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Не удалось построить признаки: {missing[:5]}...")
        return df[self.feature_columns]

    def build_vector(self, title: str, channel: str, weekday: str) -> np.ndarray:
        return self.build_row(title, channel, weekday).to_numpy(dtype=np.float64)
