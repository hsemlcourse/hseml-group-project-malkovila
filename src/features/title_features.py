from __future__ import annotations

import re
import string
from functools import lru_cache

import pandas as pd

from src.config import (
    CLICKBAIT_PHRASES,
    CLICKBAIT_WORDS,
    NEGATIVE_LEXICON,
    POSITIVE_LEXICON,
    SURPRISE_LEXICON,
)

_CLICKBAIT_WORDS_SET = {w.lower() for w in CLICKBAIT_WORDS}
_POSITIVE_SET = {w.lower() for w in POSITIVE_LEXICON}
_NEGATIVE_SET = {w.lower() for w in NEGATIVE_LEXICON}
_SURPRISE_SET = {w.lower() for w in SURPRISE_LEXICON}
_PUNCT_SET = set(string.punctuation)

_WORD_RE = re.compile(r"[A-Za-z']+")
_DIGIT_RE = re.compile(r"\d+")


@lru_cache(maxsize=1)
def _get_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords

            return frozenset(stopwords.words("english"))
        except Exception:  # noqa: BLE001
            return frozenset(
                {
                    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
                    "with", "is", "are", "was", "were", "be", "been", "being", "has",
                    "have", "had", "do", "does", "did", "it", "this", "that", "these",
                    "those", "at", "by", "as", "from",
                }
            )


def _safe_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    try:
        from nltk import pos_tag

        return pos_tag(tokens)
    except LookupError:
        try:
            import nltk

            nltk.download("averaged_perceptron_tagger", quiet=True)
            from nltk import pos_tag

            return pos_tag(tokens)
        except Exception:  # noqa: BLE001
            return [(t, "NN") for t in tokens]


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def compute_title_features(title: str) -> dict[str, float]:
    title = (title or "").strip()
    lower = title.lower()
    tokens = _tokenize(title)
    tokens_lower = [t.lower() for t in tokens]
    n_tokens = max(len(tokens), 1)
    n_chars = max(len(title), 1)

    capitalized = sum(1 for t in tokens if t[:1].isupper())
    all_caps = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

    digit_count = sum(1 for c in title if c.isdigit())
    punct_count = sum(1 for c in title if c in _PUNCT_SET)

    stopwords_set = _get_stopwords()
    stopword_hits = sum(1 for t in tokens_lower if t in stopwords_set)

    clickbait_word_hits = sum(1 for t in tokens_lower if t in _CLICKBAIT_WORDS_SET)
    clickbait_phrase_hits = sum(1 for phrase in CLICKBAIT_PHRASES if phrase in lower)

    positive_hits = sum(1 for t in tokens_lower if t in _POSITIVE_SET)
    negative_hits = sum(1 for t in tokens_lower if t in _NEGATIVE_SET)
    surprise_hits = sum(1 for t in tokens_lower if t in _SURPRISE_SET)

    pos_tags = _safe_pos_tag(tokens) if tokens else []
    noun_count = sum(1 for _t, p in pos_tags if p.startswith("NN"))
    verb_count = sum(1 for _t, p in pos_tags if p.startswith("VB"))
    adj_count = sum(1 for _t, p in pos_tags if p.startswith("JJ"))

    vader = _get_vader().polarity_scores(title) if title else {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

    try:
        from textblob import TextBlob

        blob = TextBlob(title) if title else None
        tb_polarity = float(blob.sentiment.polarity) if blob else 0.0
        tb_subjectivity = float(blob.sentiment.subjectivity) if blob else 0.0
    except Exception:  # noqa: BLE001
        tb_polarity, tb_subjectivity = 0.0, 0.0

    return {
        "tf_title_char_len": float(len(title)),
        "tf_title_word_len": float(len(tokens)),
        "tf_avg_word_len": float(sum(len(t) for t in tokens) / n_tokens),
        "tf_has_question": float("?" in title),
        "tf_has_exclamation": float("!" in title),
        "tf_has_number": float(bool(_DIGIT_RE.search(title))),
        "tf_digit_ratio": float(digit_count / n_chars),
        "tf_punct_ratio": float(punct_count / n_chars),
        "tf_capitalized_ratio": float(capitalized / n_tokens),
        "tf_all_caps_ratio": float(all_caps / n_tokens),
        "tf_stopwords_ratio": float(stopword_hits / n_tokens),
        "tf_clickbait_word_count": float(clickbait_word_hits),
        "tf_clickbait_phrase_count": float(clickbait_phrase_hits),
        "tf_noun_ratio": float(noun_count / n_tokens),
        "tf_verb_ratio": float(verb_count / n_tokens),
        "tf_adj_ratio": float(adj_count / n_tokens),
        "tf_positive_count": float(positive_hits),
        "tf_negative_count": float(negative_hits),
        "tf_surprise_count": float(surprise_hits),
        "tf_vader_compound": float(vader["compound"]),
        "tf_vader_pos": float(vader["pos"]),
        "tf_vader_neg": float(vader["neg"]),
        "tf_vader_neu": float(vader["neu"]),
        "tf_textblob_polarity": tb_polarity,
        "tf_textblob_subjectivity": tb_subjectivity,
    }


def add_title_features(titles: pd.Series) -> pd.DataFrame:
    rows = [compute_title_features(str(t) if pd.notna(t) else "") for t in titles]
    return pd.DataFrame(rows, index=titles.index)


TITLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "tf_title_char_len",
    "tf_title_word_len",
    "tf_avg_word_len",
    "tf_has_question",
    "tf_has_exclamation",
    "tf_has_number",
    "tf_digit_ratio",
    "tf_punct_ratio",
    "tf_capitalized_ratio",
    "tf_all_caps_ratio",
    "tf_stopwords_ratio",
    "tf_clickbait_word_count",
    "tf_clickbait_phrase_count",
    "tf_noun_ratio",
    "tf_verb_ratio",
    "tf_adj_ratio",
    "tf_positive_count",
    "tf_negative_count",
    "tf_surprise_count",
    "tf_vader_compound",
    "tf_vader_pos",
    "tf_vader_neg",
    "tf_vader_neu",
    "tf_textblob_polarity",
    "tf_textblob_subjectivity",
)
