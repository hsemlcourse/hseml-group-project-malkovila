"""Streamlit app for Hugging Face Spaces deployment.

Run: streamlit run app.py
Expects artifacts in ./artifacts/ directory (copied by deploy/bundle.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Path setup for standalone deployment ---
APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

sys.path.insert(0, str(APP_DIR))

# --- Minimal predictor (no full src dependency) ---

CHANNEL_ALIASES = {
    "lifestyle": "data_channel_is_lifestyle",
    "entertainment": "data_channel_is_entertainment",
    "bus": "data_channel_is_bus",
    "business": "data_channel_is_bus",
    "socmed": "data_channel_is_socmed",
    "social": "data_channel_is_socmed",
    "tech": "data_channel_is_tech",
    "technology": "data_channel_is_tech",
    "world": "data_channel_is_world",
}
WEEKDAY_ALIASES = {
    "monday": "weekday_is_monday",
    "tuesday": "weekday_is_tuesday",
    "wednesday": "weekday_is_wednesday",
    "thursday": "weekday_is_thursday",
    "friday": "weekday_is_friday",
    "saturday": "weekday_is_saturday",
    "sunday": "weekday_is_sunday",
}
VALID_CHANNELS = sorted(CHANNEL_ALIASES.keys())
VALID_WEEKDAYS = sorted(WEEKDAY_ALIASES.keys())


@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "final_lgbm_cp2.joblib"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()
    bundle = joblib.load(model_path)
    meta = json.loads((ARTIFACTS_DIR / "split_meta.json").read_text())
    perm_path = ARTIFACTS_DIR / "permutation_importance.csv"
    top_feats = []
    if perm_path.exists():
        imp = pd.read_csv(perm_path).head(5)
        top_feats = [{"feature": r["feature"], "importance": float(r["importances_mean"])} for _, r in imp.iterrows()]
    return bundle, meta, top_feats


def predict_title(title: str, channel: str, weekday: str, threshold: float, bundle, meta) -> dict:
    """Simplified prediction using only the model bundle (no full feature pipeline for HF)."""
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])

    row = {f: 0.0 for f in feature_names}

    ch_key = channel.lower().strip()
    if ch_key in CHANNEL_ALIASES:
        col = CHANNEL_ALIASES[ch_key]
        if col in row:
            row[col] = 1.0

    wd_key = weekday.lower().strip()
    if wd_key in WEEKDAY_ALIASES:
        col = WEEKDAY_ALIASES[wd_key]
        if col in row:
            row[col] = 1.0
    if wd_key in ("saturday", "sunday") and "is_weekend" in row:
        row["is_weekend"] = 1.0

    if "tf_title_char_len" in row:
        row["tf_title_char_len"] = float(len(title))
    if "tf_title_word_len" in row:
        row["tf_title_word_len"] = float(len(title.split()))

    x = np.array([[row[f] for f in feature_names]])
    proba = float(model.predict_proba(x)[0, 1])
    return {
        "probability": proba,
        "is_popular": proba >= threshold,
        "popularity_threshold_shares": float(meta.get("popularity_threshold", 1400)),
    }


# --- UI ---
st.set_page_config(page_title="News Title Virality", page_icon="📰", layout="centered")
st.title("📰 Предсказание вирусности заголовка")
st.caption("UCI Online News Popularity — LightGBM (ROC-AUC ≈ 0.68 val)")

bundle, meta, top_feats = load_model()

st.info(
    "⚠️ Это облегчённая версия (HF Spaces). Полная версия с TF-IDF+SVD и всеми фичами "
    "доступна локально через `docker compose up streamlit`."
)

mode = st.radio("Режим", ["Один заголовок", "Сравнение"], horizontal=True)
channel = st.selectbox("Канал", VALID_CHANNELS, index=VALID_CHANNELS.index("tech"))
weekday = st.selectbox("День недели", VALID_WEEKDAYS, index=0)
threshold = st.slider("Порог", 0.3, 0.7, 0.5, 0.05)

if mode == "Один заголовок":
    title = st.text_area("Заголовок", "10 Things You Need to Know About AI Today", height=80)
    if st.button("Оценить", type="primary"):
        if title.strip():
            res = predict_title(title.strip(), channel, weekday, threshold, bundle, meta)
            st.metric("P(popular)", f"{res['probability']:.1%}")
            if res["is_popular"]:
                st.success("Популярен!")
            else:
                st.info("Не в топе.")
            st.progress(min(max(res["probability"], 0.0), 1.0))
else:
    titles_text = st.text_area(
        "Заголовки (по строке)",
        "10 Things You Need to Know About AI Today\nScientists Discover New Energy Source\nBreaking: Tech Layoffs",
        height=120,
    )
    if st.button("Сравнить", type="primary"):
        titles = [t.strip() for t in titles_text.split("\n") if t.strip()]
        if len(titles) >= 2:
            rows = []
            for t in titles:
                res = predict_title(t, channel, weekday, threshold, bundle, meta)
                rows.append({"Заголовок": t[:50], "P(popular)": res["probability"], "Популярен?": "Да" if res["is_popular"] else "Нет"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Топ-фичи модели (глобальные)"):
    for f in top_feats:
        st.write(f"- `{f['feature']}`: {f['importance']:.4f}")

st.divider()
st.markdown("Исходный код: [GitHub](https://github.com/hsemlcourse/hseml-group-project-malkovila)")
