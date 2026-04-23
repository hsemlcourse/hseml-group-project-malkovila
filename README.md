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

- **Задача:** бинарная классификация (`is_popular = shares >= median(shares)`).
- **Датасет:** [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (39 644 × 61).
- **Главная метрика:** ROC-AUC. Сопровождающие: F1 (positive class), PR-AUC, Precision@top-10%.
- **Чекпоинт:** CP1 — подготовка данных, самостоятельный парсинг заголовков, baseline LogReg, 2 эксперимента-задела. Подробности в [`report/report.md`](report/report.md).

## Структура репозитория

```
.
├── data/
│   ├── raw/                   # Исходный CSV UCI + кэш заголовков (titles.jsonl)
│   └── processed/             # features.parquet, train/val/test.parquet
├── models/                    # Сериализованные модели (joblib)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA, визуализации, выбор порога
│   ├── 02_parse_titles.ipynb  # Восстановление текстов заголовков из URL
│   ├── 03_features.ipynb      # Feature engineering и MI-анализ
│   └── 04_baseline.ipynb      # Baseline LogReg + 2 эксперимента-задела
├── presentation/              # Слайды (появятся на защите)
├── report/
│   ├── images/                # Графики, переиспользуемые в отчёте
│   ├── tables/                # Таблицы метрик и экспериментов (CSV)
│   └── report.md              # Финальный отчёт (заполнен до §5)
├── src/
│   ├── config.py              # Пути, SEED, лексиконы (кликбейт / сентимент)
│   ├── data/
│   │   ├── download.py        # UCI id=332 → data/raw/online_news_popularity.csv
│   │   └── parse_titles.py    # Slug → HTTP → Wayback, CLI с кэшем
│   ├── features/
│   │   ├── title_features.py  # 25 handcrafted-фич по тексту заголовка
│   │   └── build_dataset.py   # Очистка → FE → stratified split 70/15/15
│   ├── modeling/
│   │   ├── metrics.py         # ROC-AUC, F1, PR-AUC, Precision@k
│   │   ├── baseline.py        # LogReg на 5 CSV title-фичах
│   │   └── experiments.py     # Exp1 (LogReg + FE) и Exp2 (Tree depth=6)
│   └── utils/
│       ├── seed.py            # SEED=42 для random/numpy/PYTHONHASHSEED
│       └── logging_setup.py   # Единообразный логгер
├── tests/                     # pytest: parser, features, metrics, split
├── .github/workflows/ci.yml   # CI: ruff check src/ --line-length 120
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
```

### Пайплайн CP1 одной командой

```bash
make setup      # зависимости + NLTK данные
make data       # скачать UCI id=332 в data/raw/
make parse      # восстановить заголовки из URL (slug-режим, offline)
make features   # очистка + FE + split, запись в data/processed/
make baseline   # LogReg на 5 CSV-фичах + сохранение модели и метрик
make notebooks  # прогнать все 4 .ipynb через papermill (нужны данные)
make test       # pytest (20 тестов)
make lint       # ruff check src/ — CI использует эту же команду
```

### Docker

```bash
docker-compose build
docker-compose run --rm app make run-all
# Jupyter на порту 8888:
docker-compose up jupyter
```

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
- `data/processed/{train,val,test}.parquet` — стратифицированный сплит 70/15/15 с `random_state=42`.

## Результаты

CP1 — ключевая таблица (validation, после `make run-all`):

| Модель | Признаки | ROC-AUC | F1 | Accuracy | P@top10% | Примечание |
|--------|----------|---------|----|----------|----------|------------|
| baseline_logreg | 5 CSV title | 0.5469 | 0.6657 | 0.5439 | 0.6414 | минимальный эталон |
| exp1_logreg_full | CSV + 25 FE + channel + weekday (44 фичи) | **0.6771** | **0.6810** | **0.6396** | 0.7811 | FE даёт **+0.1302** AUC |
| exp2_tree_depth6 | те же 44 фичи | 0.6711 | 0.6800 | 0.6374 | **0.7845** | задел для ансамблей CP2 |

Сводные данные: [`report/tables/cp1_validation_summary.csv`](report/tables/cp1_validation_summary.csv), полные сплиты train/val/test — [`report/tables/baseline_metrics.csv`](report/tables/baseline_metrics.csv) и [`report/tables/experiments_cp1.csv`](report/tables/experiments_cp1.csv). Графики — [`report/images/`](report/images/).

## Отчёт

Финальный отчёт по CP1 (§1–§5 заполнены полностью, §6–§8 зарезервированы под CP2/CP3): [`report/report.md`](report/report.md).
