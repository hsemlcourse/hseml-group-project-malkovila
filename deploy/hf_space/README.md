---
title: News Title Virality Predictor
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# News Title Virality Predictor

Предсказание вирусности новостного заголовка — бинарная классификация (LightGBM).

- **Датасет:** UCI Online News Popularity (39 644 × 61)
- **Метрика:** ROC-AUC ≈ 0.68 (val)
- **Финальная модель:** LightGBM

## Использование

1. Введите заголовок на английском
2. Выберите канал публикации и день недели
3. Получите вероятность попадания в топ-50% по shares

## Ограничения

Облегчённая версия без полного TF-IDF+SVD пайплайна. Полная версия — через Docker.

