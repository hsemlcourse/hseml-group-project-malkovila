# Отчёт по проекту — CP1 / CP2

**Тема.** Предсказание вирусности новостного заголовка по его лингвистическим и структурным признакам.

**Студент.** Мальков Илья Денисович (`idmalkov@edu.hse.ru`), БИВ232.

---

## 1. Введение и постановка задачи

**Практическая ценность.** Редакторы и SMM-специалисты тратят заметную часть времени на придумывание заголовков. Если по заголовку (до публикации) можно оценить вероятность того, что статья пойдёт в топ по репостам, то A/B-тест вариантов заголовка превращается в дешёвую офлайн-задачу ранжирования.

**Формулировка задачи.** Бинарная классификация: по признакам, вычисляемым *до публикации* (текст заголовка + канал публикации + день недели), предсказать, попадёт ли статья в верхнюю половину распределения `shares`. Порог — **медиана** числа шаров в train-сплите (≈ 1 400). В экспериментах дополнительно рассматривается ablation «топ-20% → вирусные».

**Обоснование выбора бинарной постановки.** Сырое `shares` распределено с тяжёлым правым хвостом (max ≈ 8·10⁵ при медиане ≈ 1 400), из-за чего регрессия малоинтерпретируема и шумна. Бинаризация по медиане даёт ровные 50/50 классы, соответствует основному бенчмарку статьи Fernandes, Vinagre, Cortez (2015) и позволяет корректно сравниваться с литературой.

**Главная метрика — ROC-AUC.**

- Классы сбалансированы, важна общая ранжирующая способность (редактор видит скор и сортирует варианты) — это именно то, что измеряет ROC-AUC.
- Метрика не зависит от выбранного порога `0.5`, что снимает ловушку «порог подтюнили под val».

**Вторичные метрики.**

- **F1 (positive class)** — показывает баланс precision / recall при стандартном пороге; для продуктового решения это «какая доля заголовков, которые мы назвали популярными, правда такими окажется».
- **PR-AUC** — страхует от перевеса ROC-AUC при возможном дисбалансе в ablation-постановке top-20%.
- **Precision@top-10%** — для формулировки «вирусные vs нет» даёт прямой ответ: сколько из топ-10% по скору реально получат виральный трафик.

---

## 2. Поиск и описание данных

**Источник.** UCI Machine Learning Repository, dataset id=332 «Online News Popularity» (Fernandes et al., 2015, DOI [10.24432/C5NS3V](https://doi.org/10.24432/C5NS3V)). Данные собраны по публикациям сайта [Mashable](https://mashable.com) за два года (2013–2015).

**Почему именно он.**

- 39 644 строк × 61 колонка (UCI Online News Popularity v2024) — с запасом проходит порог «≥ 10 000 строк, ≥ 10 колонок».
- Содержит готовые лингво-структурные фичи по заголовку (`n_tokens_title`, `title_subjectivity`, `title_sentiment_polarity`, `abs_title_subjectivity`, `abs_title_sentiment_polarity`) плюс URL, из которого восстанавливается сам текст заголовка — то есть идеально подходит именно под тему «по тексту заголовка».
- Распределение каналов и дней недели позволяет строить честный pre-publication контекст без утечек.

**Объём и схема.**

| Параметр | Значение |
|---|---|
| Число строк (сырые) | 39 644 |
| После очистки | 39 643 |
| Всего колонок | 61 (url + timedelta + 58 фич + shares) |
| Пропуски | 0 |
| Дубликаты по URL | 0 |
| Таргет `shares` | min = 1, median = 1 400, mean ≈ 3 395, max = 843 300 |
| Порог `is_popular` (медиана) | 1 400 → баланс 0.534 |
| Порог `is_viral` (top-20%) | 3 400 → баланс 0.204 |

**Колонки, которые используем в рамках темы «по заголовку»:**

- 5 готовых CSV title-фич (см. выше).
- Восстановленный текст заголовка (см. §3).
- 25 handcrafted-фич, вычисленных из текста заголовка (см. §3).
- Pre-publication контекст: 6 флагов `data_channel_is_*`, 7 `weekday_is_*`, `is_weekend`. Это характеристики темы и момента публикации, они известны до публикации и содержательно дополняют текст заголовка.

---

## 3. Обработка и подготовка данных

### 3.1. Очистка

- **Документированная аномалия UCI.** 2 % строк имеют `n_unique_tokens = 701.0` (артефакт парсинга) и соответствующие нереальные значения `n_non_stop_words > 1`. Удаляем фильтром `n_unique_tokens <= 1.0 & n_non_stop_words <= 1.0`.
- **Пропусков нет**, импутация не требуется.
- **Дубликаты по URL** отсутствуют, `drop_duplicates(["url"])` оставлен как защитная мера в пайплайне (`src/features/build_dataset.py`).
- **Типы.** Все фичи приводятся к `float64`, бинарные таргеты — к `int`. Это гарантирует совместимость с `StandardScaler` и древовидными моделями.
- **Выбросы по `shares`.** Не удаляем: они несут сигнал о виральности (это и есть positive class). Для визуализаций в EDA применяется `log1p(shares)` для читаемых гистограмм.

### 3.2. Самостоятельный парсинг заголовков

В CSV заголовок как текст отсутствует. Чтобы не ограничиваться пятью готовыми фичами и полноценно работать «по тексту заголовка», реализован модуль [`src/data/parse_titles.py`](../src/data/parse_titles.py):

1. **Slug-парсинг URL.** Mashable использует URL вида `http://mashable.com/YYYY/MM/DD/<kebab-case-title>/`. Из slug-а одним регулярным выражением восстанавливается читабельный заголовок. Работает offline, детерминированно, 100 % покрытие.
2. **HTTP-фетч.** Опциональный режим: живая страница → `<meta property="og:title">` → `<title>` с backoff-ретраями (`tenacity`), единым `User-Agent`, rate-limit 4 запр/сек.
3. **Wayback Machine.** Fallback через `waybackpy` для статей, которые уже не отдаются Mashable.

Результаты идут в построчный кэш `data/raw/titles.jsonl`, поэтому парсинг можно прерывать и возобновлять. По умолчанию пайплайн использует slug-режим (100 % покрытие, полностью воспроизводим), HTTP/Wayback включаются вручную при необходимости кросс-проверки.

*Этот модуль рассчитывается на бонусные +4 балла критерия «самостоятельный парсинг данных».*

### 3.3. Feature engineering по заголовку

Файл [`src/features/title_features.py`](../src/features/title_features.py) вычисляет 25 фич, разбитых на 4 группы:

| Группа | Фичи |
|---|---|
| Структурные | `tf_title_char_len`, `tf_title_word_len`, `tf_avg_word_len`, `tf_has_question`, `tf_has_exclamation`, `tf_has_number`, `tf_digit_ratio`, `tf_punct_ratio`, `tf_capitalized_ratio`, `tf_all_caps_ratio` |
| Лексические | `tf_stopwords_ratio`, `tf_clickbait_word_count`, `tf_clickbait_phrase_count`, `tf_noun_ratio`, `tf_verb_ratio`, `tf_adj_ratio` |
| Сентимент | `tf_vader_compound`, `tf_vader_pos`, `tf_vader_neg`, `tf_vader_neu`, `tf_textblob_polarity`, `tf_textblob_subjectivity` |
| Эмоциональные | `tf_positive_count`, `tf_negative_count`, `tf_surprise_count` |

Кликбейт-лексикон (`CLICKBAIT_WORDS`, `CLICKBAIT_PHRASES`) и NRC-подобные лексиконы (`POSITIVE_LEXICON`, `NEGATIVE_LEXICON`, `SURPRISE_LEXICON`) зафиксированы в [`src/config.py`](../src/config.py). Все фичи — числовые, финитные, без NaN (проверяется тестом [`tests/test_features.py`](../tests/test_features.py)).

### 3.4. Визуализации

Все графики генерируются ноутбуками `notebooks/01_eda.ipynb`, `notebooks/02_parse_titles.ipynb`, `notebooks/03_features.ipynb`, `notebooks/04_baseline.ipynb` и сохраняются в [`report/images/`](images):

| Файл | Назначение |
|---|---|
| [`01_shares_distribution.png`](images/01_shares_distribution.png) | сырое и `log1p` распределение `shares`, линии медианы и top-20% |
| [`01_shares_by_channel.png`](images/01_shares_by_channel.png) | медиана `shares` по каналам (`data_channel_*`) |
| [`01_csv_title_correlations.png`](images/01_csv_title_correlations.png) | корреляции 5 готовых CSV title-фич с `log shares` (\|r\| < 0.08 — обоснование FE) |
| [`01_correlation_heatmap.png`](images/01_correlation_heatmap.png) | тепловая карта корреляций ключевых фич и таргета |
| [`02_title_length_distribution.png`](images/02_title_length_distribution.png) | распределение длин распарсенных заголовков (символы и слова) |
| [`03_feature_mi.png`](images/03_feature_mi.png) | mutual information 25 инженерных фич с бинарным таргетом |
| [`03_feature_multicollinearity.png`](images/03_feature_multicollinearity.png) | мультиколлинеарность инженерных фич |
| [`04_baseline_roc_cm.png`](images/04_baseline_roc_cm.png) | ROC и confusion matrix baseline'а на test |

![Распределение shares](images/01_shares_distribution.png)

![Shares по каналам](images/01_shares_by_channel.png)

![Mutual Information признаков](images/03_feature_mi.png)

### 3.5. Сплит данных и защита от утечек (CP2)

Реализовано в [`src/features/build_dataset.py`](../src/features/build_dataset.py):

- **Стратифицированный hold-out 70/15/15.** Перед расчётом таргета сплит выполняется по прокси-метке `shares >= median(shares)` по всей выборке (для сохранения баланса по уровню популярности), `random_state=42`. Итоговые метки `is_popular` и `is_viral` задаются **одним порогом популярности** (медиана `shares` **только в train**) и квантилем виральности (аналогично, только по train). Численные пороги и список признаков сохраняются в [`data/processed/split_meta.json`](../data/processed/split_meta.json) при сборке пайплайна.
- **Отсутствие утечек по URL.** Один и тот же URL не попадает в разные сплиты — [`tests/test_split.py`](../tests/test_split.py).
- **Колонка `timedelta`.** Сохраняется в parquet для анализа и **не входит** в модель (см. `MODELING_EXCLUDE_COLS` в [`src/config.py`](../src/config.py)), чтобы не использовать информацию, не относящуюся к post-hoc характеристикам снимка данных.
- **Детерминизм.** `SEED=42` — тест `test_split_is_deterministic`.

### 3.7. Расширенные признаки (опция `--full`)

Команда `python -m src.features.build_dataset --full` (см. [`build_dataset_full.py`](../src/features/build_dataset_full.py)): **textstat**-метрики читабельности по `title`; **TF-IDF** (char wb 3–5-граммы + словесные 1–2-граммы) с **`TruncatedSVD`**, обученным **только на train**-корпусе заголовков. Артефакты векторизатора — `models/text_tfidf_svd_artifacts.joblib`.

### 3.8. Временной сплит и дрифт

- **Time-split.** [`src/features/time_split.py`](../src/features/time_split.py): сортировка по `timedelta`, те же пропорции 70/15/15; таргеты с порогами, посчитанными на train-срезе — `data/processed/train_time.parquet` и т.д. Сравнение с random-split — в [`report/tables/time_split_metrics.csv`](tables/time_split_metrics.csv) (после `make time-metrics`).
- **Дрифт.** [`src/features/drift_report.py`](../src/features/drift_report.py): KS train vs test, [`report/tables/feature_drift.csv`](tables/feature_drift.csv), график `images/05_feature_drift_top.png`.

### 3.6. Выбор и обоснование метрики

См. §1. Главная — **ROC-AUC**, сопровождающие — **F1 (positive)**, **PR-AUC**, **Precision@top-10%**. Метрика выбрана до просмотра результатов моделей.

---

## 4. Baseline-модель

**Что.** Логистическая регрессия на 5 готовых CSV title-фичах (`CSV_TITLE_FEATURES` из [`src/config.py`](../src/config.py)) со стандартизацией (`StandardScaler`). Никакого feature engineering, никакого контекста — честный «из коробки» эталон. Код: [`src/modeling/baseline.py`](../src/modeling/baseline.py).

**Цель.** Зафиксировать нижнюю границу качества. Всё, что не превосходит baseline, не считается улучшением.

**Результаты** (автоматически выгружены в [`report/tables/baseline_metrics.csv`](tables/baseline_metrics.csv)):

| Split | ROC-AUC | F1 | Accuracy | PR-AUC | P@top10% |
|---|---|---|---|---|---|
| train | 0.5389 | 0.6638 | 0.5387 | 0.5728 | 0.6180 |
| val   | **0.5469** | 0.6657 | 0.5439 | 0.5795 | 0.6414 |
| test  | 0.5326 | 0.6615 | 0.5384 | 0.5635 | 0.6061 |

**Интерпретация.** Пять готовых CSV-фич имеют \|corr\| < 0.08 с `shares` (см. `report/images/01_csv_title_correlations.png`) — поэтому ROC-AUC чуть выше случайного (0.5469 на val). Высокий F1 ≈ 0.66 — иллюзия: recall = 0.85, но precision = 0.55, т. е. модель почти всегда предсказывает «популярно» на сбалансированных классах. Это ожидаемо и именно для этого и нужен feature engineering из §3.3.

![ROC-кривая и матрица ошибок baseline](images/04_baseline_roc_cm.png)

---

## 5. Эксперименты (CP1-задел)

Основная часть экспериментов согласно рубрике — на CP2. Здесь фиксируются 2 эксперимента, демонстрирующие формат «**Гипотеза → Как проверялось → Результат**» и дающие первые сравнения. Код: [`src/modeling/experiments.py`](../src/modeling/experiments.py); результаты — [`report/tables/experiments_cp1.csv`](tables).

### Эксп. 1 — LogReg на полном наборе фич

- **Гипотеза.** Инженерные title-фичи + pre-publication контекст (канал, день) дают прирост ROC-AUC на валидации ≥ 0.02 относительно baseline.
- **Как проверялось.** Тот же `LogisticRegression(C=1.0, random_state=42, max_iter=1000)` + `StandardScaler`, но вход — `CSV_TITLE_FEATURES ∪ TITLE_FEATURE_COLUMNS ∪ DATA_CHANNEL_COLS ∪ WEEKDAY_COLS` (44 фичи). Метрики считались на фиксированных train/val/test.
- **Результат.** ROC-AUC(val) = **0.6771** → прирост **+0.1302** над baseline (гипотеза +0.02 подтверждена с большим запасом). На test — 0.6529, переобучения почти нет. P@top-10%(val) = 0.7811 против 0.6414 у baseline — редактор, выбирая 10 % лучших по скору заголовков, получает ~78 % реально популярных вместо ~64 %.

### Эксп. 2 — DecisionTree(depth=6)

- **Гипотеза.** Нелинейная модель с ограниченной глубиной лучше линейной на смешанных структурных + лексических фичах за счёт автоматического моделирования взаимодействий (например, «длина × канал»).
- **Как проверялось.** `DecisionTreeClassifier(max_depth=6, random_state=42)` на том же полном наборе фич.
- **Результат.** ROC-AUC(val) = **0.6711** (на 0.006 ниже LogReg), но P@top-10%(val) = **0.7845** — уже чуть выше, чем у LogReg. На test: 0.6389 AUC, заметнее проседает (разрыв train–test 0.031), что типично для единичного неглубокого дерева. Дерево уступает LogReg по линейному ранжированию, но сигнализирует о потенциале ансамблей (Random Forest / XGBoost / LightGBM), которые мы раскрутим на CP2.

### Сводная таблица (val)

| Модель | Гипотеза | Признаки | ROC-AUC | F1 | Accuracy | PR-AUC | P@top10% | Комментарий |
|---|---|---|---|---|---|---|---|---|
| baseline_logreg | минимальный эталон | 5 CSV title | 0.5469 | 0.6657 | 0.5439 | 0.5795 | 0.6414 | точка отсчёта |
| exp1_logreg_full | FE + контекст дают +0.02 AUC | CSV + 25 FE + 6 channel + 7 weekday + is_weekend (44 фичи) | **0.6771** | **0.6810** | **0.6396** | **0.6912** | 0.7811 | гипотеза подтверждена, +0.1302 AUC |
| exp2_tree_depth6 | нелинейность ловит взаимодействия | те же 44 фичи | 0.6711 | 0.6800 | 0.6374 | 0.6739 | **0.7845** | лучшая P@10, задел для ансамблей на CP2 |

Полный dump по всем трём сплитам (train/val/test) — в [`report/tables/experiments_cp1.csv`](tables/experiments_cp1.csv) и сводка — в [`report/tables/cp1_validation_summary.csv`](tables/cp1_validation_summary.csv).

### 5.1. Эксперименты CP2

Полный цикл описан в коде [`src/modeling/experiments_cp2.py`](../src/modeling/experiments_cp2.py), подбор гиперпараметров — [`src/modeling/tuners.py`](../src/modeling/tuners.py): **LogReg L2/L1** + `StandardScaler` (`RandomizedSearchCV`), **KNN**, **RandomForest**, **XGBoost / LightGBM / CatBoost** (Optuna, 5-fold CV на train, при сжатии через `CP2_FAST` — быстрый режим для CI). Дополнительно: **калибровка isotonic** для LightGBM, **StackingClassifier** (RF + XGB + LGBM, мета — LogReg). Трекинг: **MLflow** (`file:./mlruns`), агрегированная таблица — **[`report/tables/experiments_cp2.csv`](tables/experiments_cp2.csv)** (заполните после `make experiments-cp2` на полном датасете).

Формат каждой строки соответствует требованию курса: **гипотеза** (`hypothesis` в CSV), **как проверялось** (модель, подбор, сплиты), **результат** (метрики `train_*`, `val_*`, `test_*`).

### 5.2. Снижение размерности

Реализовано в [`src/modeling/dim_reduction.py`](../src/modeling/dim_reduction.py): кривая **ROC-AUC (val)** для `LogReg` на **PCA** с разным `n_components` (`images/06_pca_auc_curve.png`); визуализация **UMAP-2D** на подвыборке train (`images/06_umap_scatter.png`). Дополнительно TF-IDF+SVD в §3.7 служит сжатием текстового блока признаков.

---

## 6. Финальная модель и интерпретируемость (CP2)

Скрипт [`src/modeling/final_model.py`](../src/modeling/final_model.py): повторный подбор **LightGBM** (Optuna) на train, оценка на val/test, сохранение бандла в **`models/final_lgbm_cp2.joblib`**, метрики — **[`report/tables/final_metrics.csv`](tables/final_metrics.csv)**, **permutation importance** на val — [`report/tables/permutation_importance.csv`](tables/permutation_importance.csv) и `images/07_permutation_importance.png`. Итоговый конфиг для отчёта — [`report/tables/final_model_summary.json`](tables/final_model_summary.json).

Сравнение с CP1 baseline/exp1 фиксируйте по столбцу `val_roc_auc` в `experiments_cp2.csv` и по `final_metrics.csv`.

---

## 7. Деплой — CP3

Запланировано: FastAPI `/predict`, Streamlit или Telegram-бот, скринкаст.

---

## 8. Заключение и выводы — CP3

Заполняется после деплоя и финального согласования метрик.
