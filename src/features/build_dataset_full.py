from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SEED
from src.features.readability import READABILITY_COLUMNS, add_readability_features
from src.features.text_vectorizers import (
    char_word_svd_column_names,
    fit_text_svd_on_train,
    transform_titles_with_artifacts,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

TEXT_SVD_ARTIFACT_NAME = "text_tfidf_svd_artifacts.joblib"


def augment_full_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    artifacts_dir: Path | None = None,
) -> pd.DataFrame:
    """Добавляет readability и TF-IDF+SVD признаки (SVD обучается только на train)."""
    titles = df["title"]
    read_df = add_readability_features(titles)
    for c in READABILITY_COLUMNS:
        df[c] = read_df[c].astype(np.float64).fillna(0.0)

    art = fit_text_svd_on_train(
        titles.iloc[train_idx],
        char_n_components=64,
        word_n_components=32,
        random_state=SEED,
    )
    mat = transform_titles_with_artifacts(titles, art)
    names = char_word_svd_column_names(art.char_dim, art.word_dim)
    for i, col in enumerate(names):
        df[col] = mat[:, i].astype(np.float64)

    out_dir = artifacts_dir or (MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(art, out_dir / TEXT_SVD_ARTIFACT_NAME)
        meta = {"artifact": str(out_dir / TEXT_SVD_ARTIFACT_NAME), "seed": SEED}
        (out_dir / "text_svd_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Сохранены артефакты TF-IDF+SVD в %s", out_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось сохранить артефакты SVD: %s", exc)

    return df
