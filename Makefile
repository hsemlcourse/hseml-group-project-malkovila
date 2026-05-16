.PHONY: help setup nltk data parse features features-cp2 time-split drift eda baseline notebooks test lint format \
	run-all run-all-cp2 experiments-cp2 dim-reduction final-model time-metrics clean

PY ?= python
PIP ?= pip

help:
	@echo "Доступные цели:"
	@echo "  setup       - установить зависимости и данные NLTK"
	@echo "  nltk        - скачать необходимые корпуса NLTK"
	@echo "  data        - скачать датасет UCI Online News Popularity"
	@echo "  parse       - извлечь заголовки из URL (slug + HTTP + Wayback)"
	@echo "  features    - собрать processed-датасет с инженерными признаками"
	@echo "  features-cp2 - build_dataset --full (TF-IDF+SVD, readability, те же сплиты)"
	@echo "  time-split  - time-ordered train/val/test по timedelta → *_time.parquet"
	@echo "  drift       - KS train vs test, report/tables/feature_drift.csv + график"
	@echo "  experiments-cp2 - Optuna/RS + MLflow + experiments_cp2.csv"
	@echo "  dim-reduction - PCA AUC-кривая и UMAP (нужны processed parquet)"
	@echo "  final-model - финальный LightGBM + permutation importance"
	@echo "  time-metrics - LGBM на time-parquet → time_split_metrics.csv"
	@echo "  eda         - запустить 01_eda.ipynb через papermill"
	@echo "  baseline    - обучить базовую логистическую регрессию"
	@echo "  notebooks   - прогнать все ноутбуки через papermill"
	@echo "  test        - запустить pytest"
	@echo "  lint        - запустить ruff для src/ с настройками CI"
	@echo "  format      - отформатировать код с помощью ruff"
	@echo "  run-all     - setup + data + parse + features + baseline + notebooks + test + lint"
	@echo "  run-all-cp2 - цепочка CP2 (нужны скачанные data): features-cp2 time-split drift experiments-cp2 …"
	@echo "  clean       - удалить кэш и промежуточные файлы"

setup:
	$(PIP) install -r requirements.txt
	$(MAKE) nltk

nltk:
	$(PY) -m nltk.downloader punkt averaged_perceptron_tagger stopwords vader_lexicon

data:
	$(PY) -m src.data.download

parse:
	$(PY) -m src.data.parse_titles --input data/raw/online_news_popularity.csv \
		--output data/raw/titles.jsonl --mode slug

features:
	$(PY) -m src.features.build_dataset

features-cp2:
	$(PY) -m src.features.build_dataset --full

time-split:
	$(PY) -m src.features.time_split

drift:
	$(PY) -m src.features.drift_report

experiments-cp2:
	$(PY) -m src.modeling.experiments_cp2

dim-reduction:
	$(PY) -m src.modeling.dim_reduction

final-model:
	$(PY) -m src.modeling.final_model

time-metrics:
	$(PY) -m src.modeling.time_split_eval

eda:
	papermill notebooks/01_eda.ipynb notebooks/01_eda.ipynb

baseline:
	$(PY) -m src.modeling.baseline

notebooks:
	papermill notebooks/01_eda.ipynb notebooks/01_eda.ipynb
	papermill notebooks/02_parse_titles.ipynb notebooks/02_parse_titles.ipynb
	papermill notebooks/03_features.ipynb notebooks/03_features.ipynb
	papermill notebooks/04_baseline.ipynb notebooks/04_baseline.ipynb
	papermill notebooks/05_cp2_features.ipynb notebooks/05_cp2_features.ipynb
	papermill notebooks/06_cp2_experiments.ipynb notebooks/06_cp2_experiments.ipynb
	papermill notebooks/07_dim_reduction.ipynb notebooks/07_dim_reduction.ipynb

test:
	$(PY) -m pytest

lint:
	ruff check src/ --line-length 120

format:
	ruff format src/ tests/

run-all: setup data parse features baseline notebooks test lint

run-all-cp2: features-cp2 time-split drift experiments-cp2 dim-reduction final-model time-metrics test lint

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
