from __future__ import annotations

from pathlib import Path

SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_IMAGES_DIR: Path = REPORT_DIR / "images"
REPORT_TABLES_DIR: Path = REPORT_DIR / "tables"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

RAW_CSV_PATH: Path = RAW_DIR / "online_news_popularity.csv"
TITLES_JSONL_PATH: Path = RAW_DIR / "titles.jsonl"
FEATURES_PARQUET_PATH: Path = PROCESSED_DIR / "features.parquet"
TRAIN_PARQUET_PATH: Path = PROCESSED_DIR / "train.parquet"
VAL_PARQUET_PATH: Path = PROCESSED_DIR / "val.parquet"
TEST_PARQUET_PATH: Path = PROCESSED_DIR / "test.parquet"

TRAIN_SIZE: float = 0.70
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15

TARGET_COL: str = "shares"
BINARY_TARGET_COL: str = "is_popular"
VIRAL_TARGET_COL: str = "is_viral"

VIRAL_TOP_QUANTILE: float = 0.80

CSV_TITLE_FEATURES: tuple[str, ...] = (
    "n_tokens_title",
    "title_subjectivity",
    "title_sentiment_polarity",
    "abs_title_subjectivity",
    "abs_title_sentiment_polarity",
)

DATA_CHANNEL_COLS: tuple[str, ...] = (
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
)

WEEKDAY_COLS: tuple[str, ...] = (
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
    "is_weekend",
)

CLICKBAIT_PHRASES: tuple[str, ...] = (
    "you won't believe",
    "will blow your mind",
    "this is what",
    "what happens next",
    "you need to",
    "here's why",
    "here is why",
    "here's how",
    "here is how",
    "the reason why",
    "one weird trick",
    "shocked to see",
)

CLICKBAIT_WORDS: tuple[str, ...] = (
    "you",
    "your",
    "these",
    "this",
    "why",
    "how",
    "what",
    "shocking",
    "shocked",
    "amazing",
    "incredible",
    "unbelievable",
    "secret",
    "hack",
    "hacks",
    "ultimate",
    "epic",
    "awesome",
    "insane",
    "crazy",
    "weird",
    "strange",
    "mind",
    "blown",
    "wow",
    "omg",
    "reasons",
    "things",
    "ways",
    "tips",
    "best",
    "worst",
    "top",
    "must",
    "need",
    "love",
    "hate",
    "never",
    "always",
    "finally",
    "every",
    "genius",
)

POSITIVE_LEXICON: tuple[str, ...] = (
    "love",
    "loved",
    "amazing",
    "awesome",
    "great",
    "best",
    "beautiful",
    "happy",
    "excellent",
    "fantastic",
    "wonderful",
    "brilliant",
    "incredible",
    "perfect",
    "stunning",
    "genius",
    "joy",
    "delight",
    "win",
    "winner",
    "celebrate",
    "success",
)

NEGATIVE_LEXICON: tuple[str, ...] = (
    "hate",
    "bad",
    "worst",
    "terrible",
    "awful",
    "horrible",
    "ugly",
    "sad",
    "angry",
    "fail",
    "failure",
    "disaster",
    "crisis",
    "danger",
    "dangerous",
    "death",
    "dead",
    "kill",
    "killed",
    "attack",
    "war",
    "scandal",
    "shocking",
    "shocked",
)

SURPRISE_LEXICON: tuple[str, ...] = (
    "surprise",
    "surprising",
    "unexpected",
    "sudden",
    "suddenly",
    "bombshell",
    "reveal",
    "revealed",
    "mystery",
    "mysterious",
    "twist",
    "weird",
    "bizarre",
    "strange",
    "secret",
    "hidden",
    "rare",
    "historic",
    "unprecedented",
    "shocking",
)


def ensure_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORT_IMAGES_DIR,
        REPORT_TABLES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
