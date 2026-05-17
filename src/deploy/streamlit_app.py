from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference.context_features import VALID_CHANNELS, VALID_WEEKDAYS
from src.inference.predictor import NewsViralityPredictor

st.set_page_config(page_title="Вирусность заголовка", page_icon="📰", layout="centered")

st.title("Предсказание вирусности новостного заголовка")
st.caption(
    "Бинарная классификация: вероятность попасть в верхнюю половину по shares "
    "(порог ≈ 1400, медиана train). Датасет: UCI Online News Popularity (Mashable)."
)

try:
    predictor = NewsViralityPredictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

mode = st.radio("Режим", ["Один заголовок", "Сравнение заголовков"], horizontal=True)

channel = st.selectbox("Канал публикации", sorted(VALID_CHANNELS), index=sorted(VALID_CHANNELS).index("tech"))
weekday = st.selectbox("День недели", sorted(VALID_WEEKDAYS), index=0)
threshold = st.slider("Порог вероятности для метки «популярный»", 0.3, 0.7, 0.5, 0.05)

if mode == "Один заголовок":
    title = st.text_area(
        "Заголовок",
        value="10 Things You Need to Know About Artificial Intelligence Today",
        height=100,
    )

    if st.button("Оценить вирусность", type="primary"):
        if not title.strip():
            st.warning("Введите заголовок.")
        else:
            try:
                result = predictor.predict(title.strip(), channel=channel, weekday=weekday, threshold=threshold)
            except ValueError as exc:
                st.error(str(exc))
            else:
                proba = result["probability"]
                st.metric("Вероятность is_popular", f"{proba:.1%}")
                if result["is_popular"]:
                    st.success("Прогноз: заголовок **популярен** (выше порога по shares).")
                else:
                    st.info("Прогноз: заголовок **не в топе** по ожидаемым shares.")

                st.progress(min(max(proba, 0.0), 1.0))

                with st.expander("Детали модели"):
                    st.write(f"Модель: {result['model']}")
                    st.write(f"Порог shares (train median): {result['popularity_threshold_shares']:.0f}")
                    st.write("Топ глобальных признаков (permutation importance на val):")
                    for item in result.get("top_features_global", []):
                        st.write(f"- `{item['feature']}`: {item['importance']:.4f}")

                with st.expander("Рассчитанные признаки для этого заголовка"):
                    try:
                        row_df = predictor._builder.build_row(title.strip(), channel, weekday)
                        title_feats = [c for c in row_df.columns if c.startswith("tf_")]
                        if title_feats:
                            feat_data = row_df[title_feats].T
                            feat_data.columns = ["Значение"]
                            st.dataframe(feat_data.style.format("{:.4f}"), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось показать признаки: {e}")

else:
    st.markdown("Введите от 2 до 5 заголовков (по одному на строку):")
    titles_text = st.text_area(
        "Заголовки для сравнения",
        value="10 Things You Need to Know About AI Today\n"
              "Scientists Discover New Way to Generate Clean Energy\n"
              "Breaking: Major Tech Company Announces Layoffs",
        height=150,
    )

    if st.button("Сравнить", type="primary"):
        titles = [t.strip() for t in titles_text.strip().split("\n") if t.strip()]
        if len(titles) < 2:
            st.warning("Введите хотя бы 2 заголовка.")
        elif len(titles) > 5:
            st.warning("Максимум 5 заголовков для сравнения.")
        else:
            rows = []
            for t in titles:
                try:
                    res = predictor.predict(t, channel=channel, weekday=weekday, threshold=threshold)
                    rows.append({
                        "Заголовок": t[:60] + ("…" if len(t) > 60 else ""),
                        "Вероятность": res["probability"],
                        "Популярен?": "Да" if res["is_popular"] else "Нет",
                    })
                except ValueError as exc:
                    rows.append({"Заголовок": t[:60], "Вероятность": 0.0, "Популярен?": f"Ошибка: {exc}"})

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame({
                "Заголовок": [r["Заголовок"] for r in rows],
                "Вероятность": [r["Вероятность"] for r in rows],
            }).set_index("Заголовок")
            st.bar_chart(chart_df, horizontal=True)

            best_idx = max(range(len(rows)), key=lambda i: rows[i]["Вероятность"])
            st.success(
                f"Лучший вариант: **{titles[best_idx][:80]}** "
                f"(P = {rows[best_idx]['Вероятность']:.1%})"
            )

st.divider()
st.markdown(
    "**API:** запустите `make api` и откройте [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)"
)
