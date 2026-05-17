from __future__ import annotations

import re

import pandas as pd

_WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z']+\b")


def compute_csv_title_features(title: str) -> dict[str, float]:
    """Прокси UCI title-полей для онлайн-инференса (только текст заголовка)."""
    text = title if title else ""
    words = _WORD_RE.findall(text)
    n_tokens = float(len(words))

    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        subjectivity = float(blob.sentiment.subjectivity)
        polarity = float(blob.sentiment.polarity)
    except Exception:
        subjectivity = 0.0
        polarity = 0.0

    return {
        "n_tokens_title": n_tokens,
        "title_subjectivity": subjectivity,
        "title_sentiment_polarity": polarity,
        "abs_title_subjectivity": abs(subjectivity),
        "abs_title_sentiment_polarity": abs(polarity),
    }


def csv_title_features_series(title: str) -> pd.Series:
    return pd.Series(compute_csv_title_features(title))
