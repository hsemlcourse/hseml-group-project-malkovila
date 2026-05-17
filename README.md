[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Предсказание вирусности новостного заголовка

**Студент:** Мальков Илья Денисович (`idmalkov@edu.hse.ru`)

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

По признакам, вычисляемым **до публикации** статьи (текст заголовка, канал публикации, день недели), предсказать, попадёт ли она в верхнюю половину распределения `shares` на Mashable. Тема и формулировка выбраны так, чтобы модель реально была применима как ассистент редактора при A/B-тестах заголовков.

- **Задача:** бинарная классификация (`is_popular`: порог — медиана `shares` **только в train**, без утечки из val/test; фиксируется в `data/processed/split_meta.json`).
- **Датасет:** [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (39 644 × 61).
- **Главная метрика:** ROC-AUC. Сопровождающие: F1 (positive class), PR-AUC, Precision@top-10%.
- **Чекпоинт:** CP3 — деплой (FastAPI + Streamlit + Docker + публичный на HF Spaces / Render) и финальный отчёт; см. [`report/report.md`](report/report.md).

## Структура репозитория

```
.
├── data/
│   ├── raw/                   # Исходный CSV UCI + кэш заголовков (titles.jsonl)
│   └── processed/             # features.parquet, split_meta.json, train/val/test*.parquet
├── models/                    # Сериализованные модели (joblib)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA, визуализации, выбор порога
│   ├── 02_parse_titles.ipynb  # Восстановление текстов заголовков из URL
│   ├── 03_features.ipynb      # Feature engineering и MI-анализ
│   └── 04_baseline.ipynb      # Baseline LogReg + 2 эксперимента-задела
├── deploy/
│   ├── bundle.py              # Упаковка артефактов для деплоя
│   └── hf_space/              # Streamlit app для Hugging Face Spaces
├── presentation/
│   └── slides.md              # Marp-слайды для защиты (7 шт)
├── report/
│   ├── images/                # Графики, переиспользуемые в отчёте
│   ├── tables/                # Таблицы метрик и экспериментов (CSV)
│   └── report.md              # Финальный отчёт (§1–§8)
├── src/
│   ├── config.py              # Пути, SEED, лексиконы (кликбейт / сентимент)
│   ├── api/
│   │   ├── app.py             # FastAPI: /health, /version, /predict, /predict_batch
│   │   └── schemas.py         # Pydantic-схемы с Literal-валидацией
│   ├── data/
│   │   ├── download.py        # UCI id=332 → data/raw/online_news_popularity.csv
│   │   └── parse_titles.py    # Slug → HTTP → Wayback, CLI с кэшем
│   ├── deploy/
│   │   └── streamlit_app.py   # Streamlit UI (одиночный + сравнение заголовков)
│   ├── features/
│   │   ├── title_features.py  # 25 handcrafted-фич по тексту заголовка
│   │   ├── build_dataset.py   # Очистка → FE → split → порог по train
│   │   ├── build_dataset_full.py  # + TF-IDF char/word + SVD, readability
│   │   ├── time_split.py      # сплит по timedelta
│   │   └── drift_report.py    # KS train vs test
│   ├── inference/
│   │   ├── predictor.py       # NewsViralityPredictor (загрузка модели)
│   │   └── feature_builder.py # FeatureBuilder (сборка вектора 144 фич)
│   ├── modeling/
│   │   ├── metrics.py         # ROC-AUC, F1, PR-AUC, Precision@k
│   │   ├── baseline.py        # LogReg на 5 CSV title-фичах
│   │   ├── experiments.py     # Exp1 (LogReg + FE) и Exp2 (Tree depth=6)
│   │   ├── tuners.py          # Optuna / RandomizedSearch
│   │   ├── experiments_cp2.py # CP2: L1/L2, KNN, RF, XGB/LGB/Cat, калибровка, стекинг
│   │   ├── final_model.py     # финальный LGBM + permutation importance
│   │   ├── dim_reduction.py   # PCA-кривая, UMAP
│   │   └── time_split_eval.py # метрики на time-parquet
│   └── utils/
│       ├── seed.py            # SEED=42 для random/numpy/PYTHONHASHSEED
│       ├── logging_setup.py   # Единообразный логгер
│       └── mlflow_setup.py    # MLflow `file:./mlruns`
├── mlruns/                    # локальный MLflow (не коммитится)
├── tests/                     # pytest: parser, features, metrics, split, tuners
├── .github/workflows/ci.yml   # CI: ruff + pytest + NLTK
├── Dockerfile / docker-compose.yml
├── Makefile                   # make setup / data / parse / features / baseline / test / lint
├── pyproject.toml             # ruff + pytest config
├── requirements.txt           # Версии запинены
└── README.md
```

## Быстрый старт

### Локальная установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords vader_lexicon
pre-commit install   # опционально: хуки из .pre-commit-config.yaml
```

**Windows и длинные пути.** Если `pip install` падает с `OSError: No such file or directory` на файле внутри `venv\share\jupyter\labextensions\...`, это лимит длины пути (часто на путях с кириллицей и вложенными `node_modules`). Варианты: включить [длинные пути в Windows](https://pip.pypa.io/warnings/enable-long-paths); создать venv по **короткому** пути, например `python -m venv C:\venvs\newsml` и активировать его, затем `pip install -r requirements.txt` из каталога проекта. В `requirements.txt` вместо метапакета `jupyter` указан классический **`notebook==6.5.7`** — меньше глубоких зависимостей JupyterLab.

### Пайплайн CP1 одной командой

```bash
make setup      # зависимости + NLTK данные
make data       # скачать UCI id=332 в data/raw/
make parse      # восстановить заголовки из URL (slug-режим, offline)
make features   # очистка + FE + split, запись в data/processed/
make baseline   # LogReg на 5 CSV-фичах + сохранение модели и метрик
make notebooks  # прогнать все 4 .ipynb через papermill (нужны данные)
make test       # pytest (20 тестов)
make lint       # ruff check src/ — CI использует лint частично + pytest в CI
```

### Пайплайн CP2 (после `make data` и `make parse`)

Длинная цепочка: расширенные признаки, time-split, дрифт, эксперименты с бустингами и стекингом, размерность, финальная модель.

```bash
make features-cp2     # --full: TF-IDF+SVD, textstat readability (долго по CPU)
make time-split       # train_time / val_time / test_time.parquet
make drift            # report/tables/feature_drift.csv + images/05_*.png
make experiments-cp2  # mlruns/ + report/tables/experiments_cp2.csv (долго)
make dim-reduction    # PCA/UMAP графики в report/images/
make final-model      # models/final_lgbm_cp2.joblib + permutation importance
make time-metrics     # time_split_metrics.csv (нужны *_time.parquet)
```

Или обёртка (ожидает готовый `data/raw/*.csv`): `make run-all-cp2` — см. Makefile; при нехватке данных шаги упадут с понятной ошибкой.

Переменная **`CP2_FAST=1`** уменьшает число итераций Optuna/RS (используется в CI для `pytest`).

### Docker

```bash
docker-compose build
docker-compose run --rm app make run-all
# Jupyter на порту 8888:
docker-compose up jupyter
```

### Пайплайн CP3 (деплой)

```bash
# FastAPI на :8000
make api

# Streamlit UI на :8501
make streamlit

# Docker Compose (оба сервиса)
docker compose up api streamlit

# Smoke-тесты деплоя
make run-cp3-smoke

# Собрать PDF отчёт
make report-pdf

# Собрать PDF слайдов (нужен marp-cli)
make slides-pdf
```

**Публичный деплой:**
- Streamlit UI: [Hugging Face Spaces](https://huggingface.co/spaces/) — `deploy/hf_space/`
- FastAPI: [Render.com](https://render.com) — `render.yaml`
- Видео-демо: *[TODO: вставить ссылку после записи]*

### Ручной парсинг с HTTP + Wayback (опционально)

```bash
python -m src.data.parse_titles --input data/raw/online_news_popularity.csv \
    --output data/raw/titles.jsonl --mode full --workers 8
```

В `--mode full` задействуются живые запросы к Mashable и Wayback Machine — кэш в `titles.jsonl` позволяет прерывать и возобновлять процесс.

## Данные

- `data/raw/online_news_popularity.csv` — исходный UCI dataset (39 644 × 61, текущая версия UCI). Не коммитится, скачивается через `make data`.
- `data/raw/titles.jsonl` — JSONL-кэш заголовков, построчно (по одному URL). Не коммитится.
- `data/processed/features.parquet` — полный датасет с 25 инженерными фичами и бинарными таргетами.
- `data/processed/{train,val,test}.parquet` — стратифицированный сплит 70/15/15 с `random_state=42`, порог `is_popular`/`is_viral` от **train**-медианы/квантиля (`split_meta.json`).
- После `make features-cp2`: `train_full.parquet` и зеркальные `train.parquet` с сотнями признаков (базовые + TF-IDF SVD + readability).

## Результаты

CP1 — ключевая таблица (validation, после `make run-all`):

| Модель | Признаки | ROC-AUC | F1 | Accuracy | P@top10% | Примечание |
|--------|----------|---------|----|----------|----------|------------|
| baseline_logreg | 5 CSV title | 0.5469 | 0.6657 | 0.5439 | 0.6414 | минимальный эталон |
| exp1_logreg_full | CSV + 25 FE + channel + weekday (44 фичи) | **0.6771** | **0.6810** | **0.6396** | 0.7811 | FE даёт **+0.1302** AUC |
| exp2_tree_depth6 | те же 44 фичи | 0.6711 | 0.6800 | 0.6374 | **0.7845** | задел для ансамблей CP2 |

Сводные данные: [`report/tables/cp1_validation_summary.csv`](report/tables/cp1_validation_summary.csv), полные сплиты train/val/test — [`report/tables/baseline_metrics.csv`](report/tables/baseline_metrics.csv) и [`report/tables/experiments_cp1.csv`](report/tables/experiments_cp1.csv). Графики — [`report/images/`](report/images/).

**CP2** (после `make experiments-cp2` и `make final-model`): сводная таблица — [`report/tables/experiments_cp2.csv`](report/tables/experiments_cp2.csv); финальные метрики и permutation importance — [`report/tables/final_metrics.csv`](report/tables/final_metrics.csv), [`report/tables/permutation_importance.csv`](report/tables/permutation_importance.csv). Трекинг экспериментов: локальный MLflow в каталоге `mlruns/` (или задайте `MLFLOW_TRACKING_URI`).

## Отчёт

Финальный отчёт (§1–§8): [`report/report.md`](report/report.md).