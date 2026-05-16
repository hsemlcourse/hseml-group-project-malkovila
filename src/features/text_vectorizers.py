from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import SEED


@dataclass
class TextSvdArtifacts:
    """Сохранённые векторизаторы для инференса (fit на train)."""

    char_pipeline: Pipeline
    word_pipeline: Pipeline
    char_dim: int
    word_dim: int


def fit_text_svd_on_train(
    titles_train: pd.Series,
    char_n_components: int = 64,
    word_n_components: int = 32,
    random_state: int = SEED,
) -> TextSvdArtifacts:
    """Обучает TF-IDF + TruncatedSVD на train-корпусе заголовков."""
    corpus = titles_train.fillna("").astype(str).tolist()

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_svd = TruncatedSVD(n_components=char_n_components, random_state=random_state)
    char_pipeline = Pipeline([("tfidf", char_tfidf), ("svd", char_svd)])
    char_pipeline.fit(corpus)

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        sublinear_tf=True,
    )
    word_svd = TruncatedSVD(n_components=word_n_components, random_state=random_state)
    word_pipeline = Pipeline([("tfidf", word_tfidf), ("svd", word_svd)])
    word_pipeline.fit(corpus)

    return TextSvdArtifacts(
        char_pipeline=char_pipeline,
        word_pipeline=word_pipeline,
        char_dim=char_n_components,
        word_dim=word_n_components,
    )


def transform_titles_with_artifacts(titles: pd.Series, art: TextSvdArtifacts) -> np.ndarray:
    corpus = titles.fillna("").astype(str).tolist()
    x_char = art.char_pipeline.transform(corpus)
    x_word = art.word_pipeline.transform(corpus)
    if sparse.issparse(x_char):
        x_char = x_char.toarray()
    if sparse.issparse(x_word):
        x_word = x_word.toarray()
    return np.hstack([x_char, x_word])


def char_word_svd_column_names(char_dim: int, word_dim: int) -> list[str]:
    names: list[str] = [f"tfidf_char_svd_{i}" for i in range(char_dim)]
    names += [f"tfidf_word_svd_{i}" for i in range(word_dim)]
    return names
