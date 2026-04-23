.PHONY: help setup nltk data parse features eda baseline notebooks test lint format run-all clean

PY ?= python
PIP ?= pip

help:
	@echo "Доступные цели:"
	@echo "  setup       - установить зависимости и данные NLTK"
	@echo "  nltk        - скачать необходимые корпуса NLTK"
	@echo "  data        - скачать датасет UCI Online News Popularity"
	@echo "  parse       - извлечь заголовки из URL (slug + HTTP + Wayback)"
	@echo "  features    - собрать processed-датасет с инженерными признаками"
	@echo "  eda         - запустить 01_eda.ipynb через papermill"
	@echo "  baseline    - обучить базовую логистическую регрессию"
	@echo "  notebooks   - прогнать все ноутбуки через papermill"
	@echo "  test        - запустить pytest"
	@echo "  lint        - запустить ruff для src/ с настройками CI"
	@echo "  format      - отформатировать код с помощью ruff"
	@echo "  run-all     - setup + data + parse + features + baseline + notebooks + test + lint"
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

eda:
	papermill notebooks/01_eda.ipynb notebooks/01_eda.ipynb

baseline:
	$(PY) -m src.modeling.baseline

notebooks:
	papermill notebooks/01_eda.ipynb notebooks/01_eda.ipynb
	papermill notebooks/02_parse_titles.ipynb notebooks/02_parse_titles.ipynb
	papermill notebooks/03_features.ipynb notebooks/03_features.ipynb
	papermill notebooks/04_baseline.ipynb notebooks/04_baseline.ipynb

test:
	$(PY) -m pytest

lint:
	ruff check src/ --line-length 120

format:
	ruff format src/ tests/

run-all: setup data parse features baseline notebooks test lint

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
