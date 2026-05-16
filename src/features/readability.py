from __future__ import annotations

import pandas as pd


def add_readability_features(titles: pd.Series) -> pd.DataFrame:
    """Числовые признаки читабельности заголовка (англ.)."""
    try:
        import textstat  # type: ignore[import-untyped]
    except ImportError:
        return pd.DataFrame(
            {
                "tf_flesch_reading_ease": 0.0,
                "tf_gunning_fog": 0.0,
                "tf_smog_index": 0.0,
                "tf_automated_readability_index": 0.0,
            },
            index=titles.index,
        )

    rows: list[dict[str, float]] = []
    for t in titles:
        text = str(t) if pd.notna(t) else ""
        if not text.strip():
            rows.append(
                {
                    "tf_flesch_reading_ease": 0.0,
                    "tf_gunning_fog": 0.0,
                    "tf_smog_index": 0.0,
                    "tf_automated_readability_index": 0.0,
                }
            )
            continue
        rows.append(
            {
                "tf_flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
                "tf_gunning_fog": float(textstat.gunning_fog(text)),
                "tf_smog_index": float(textstat.smog_index(text)),
                "tf_automated_readability_index": float(textstat.automated_readability_index(text)),
            }
        )
    return pd.DataFrame(rows, index=titles.index)


READABILITY_COLUMNS: tuple[str, ...] = (
    "tf_flesch_reading_ease",
    "tf_gunning_fog",
    "tf_smog_index",
    "tf_automated_readability_index",
)
