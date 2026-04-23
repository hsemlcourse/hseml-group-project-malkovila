from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RAW_CSV_PATH, ensure_directories
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

UCI_DATASET_ID: int = 332


def download_dataset(output_path: Path = RAW_CSV_PATH, force: bool = False) -> Path:
    ensure_directories()
    if output_path.exists() and not force:
        log.info("Датасет уже сохранён по пути %s (используйте --force для повторной загрузки)", output_path)
        return output_path

    from ucimlrepo import fetch_ucirepo

    log.info("Загружаем датасет UCI id=%s ...", UCI_DATASET_ID)
    bundle = fetch_ucirepo(id=UCI_DATASET_ID)

    if getattr(bundle.data, "original", None) is not None:
        df: pd.DataFrame = bundle.data.original.copy()
    else:
        features: pd.DataFrame = bundle.data.features
        targets: pd.DataFrame = bundle.data.targets
        ids = getattr(bundle.data, "ids", None)
        parts = []
        if ids is not None:
            parts.append(ids.reset_index(drop=True))
        parts.extend([features.reset_index(drop=True), targets.reset_index(drop=True)])
        df = pd.concat(parts, axis=1)
    df.columns = [c.strip() for c in df.columns]

    log.info("Получено %d строк x %d колонок", df.shape[0], df.shape[1])
    df.to_csv(output_path, index=False)
    log.info("Сохранено в %s (%.1f МБ)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузка датасета UCI Online News Popularity (id=332).")
    parser.add_argument("--output", type=Path, default=RAW_CSV_PATH, help="Путь к выходному CSV.")
    parser.add_argument("--force", action="store_true", help="Перекачать файл, даже если он уже существует.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
