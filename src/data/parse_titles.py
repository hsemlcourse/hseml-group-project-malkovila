from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from src.config import RAW_CSV_PATH, TITLES_JSONL_PATH, ensure_directories
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

HTTP_TIMEOUT_S: float = 10.0
HTTP_RATE_LIMIT_S: float = 0.25
USER_AGENT: str = "Mozilla/5.0 (HSE-ML-CP1-Research-Bot; contact: idmalkov@edu.hse.ru)"

_SLUG_DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}/$")


@dataclass(frozen=True)
class TitleRecord:
    url: str
    title_slug: str
    title_http: str | None
    title_wayback: str | None
    source: str


def extract_title_from_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return ""
    parts = path.split("/")
    if len(parts) >= 4 and all(p.isdigit() for p in parts[:3]):
        slug = "/".join(parts[3:])
    elif len(parts) <= 3 and all(p.isdigit() for p in parts):
        return ""
    else:
        slug = parts[-1]
    slug = slug.strip("/")
    if not slug or slug.isdigit() or _SLUG_DATE_RE.match(slug + "/"):
        return ""
    words = [w for w in re.split(r"[-_]+", slug) if w]
    return " ".join(w.capitalize() for w in words)


def _clean_html_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for sep in (" | ", " — ", " - "):
        if sep in text:
            head, _, _tail = text.partition(sep)
            head = head.strip()
            if len(head) >= 3:
                text = head
                break
    return text


def fetch_title_http(url: str, session) -> str | None:
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = session.get(url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.text:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return _clean_html_title(og["content"])
    if soup.title and soup.title.string:
        return _clean_html_title(soup.title.string)
    return None


def fetch_title_wayback(url: str) -> str | None:
    try:
        from waybackpy import WaybackMachineCDXServerAPI
    except ImportError:
        return None

    try:
        cdx = WaybackMachineCDXServerAPI(url, user_agent=USER_AGENT)
        snapshot = cdx.newest()
        if snapshot is None:
            return None
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(snapshot.archive_url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return _clean_html_title(og["content"])
        if soup.title and soup.title.string:
            return _clean_html_title(soup.title.string)
    except Exception:  # noqa: BLE001
        return None
    return None


def _load_cache(cache_path: Path) -> dict[str, TitleRecord]:
    cache: dict[str, TitleRecord] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cache[obj["url"]] = TitleRecord(
                    url=obj["url"],
                    title_slug=obj.get("title_slug", ""),
                    title_http=obj.get("title_http"),
                    title_wayback=obj.get("title_wayback"),
                    source=obj.get("source", "slug"),
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _append_cache(cache_path: Path, record: TitleRecord) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "url": record.url,
                    "title_slug": record.title_slug,
                    "title_http": record.title_http,
                    "title_wayback": record.title_wayback,
                    "source": record.source,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _process_url(url: str, mode: str, session) -> TitleRecord:
    slug_title = extract_title_from_slug(url)
    http_title: str | None = None
    wayback_title: str | None = None
    source = "slug"

    if mode in {"http", "full"}:
        http_title = fetch_title_http(url, session)
        time.sleep(HTTP_RATE_LIMIT_S)
        if http_title:
            source = "http"

    if mode == "full" and not http_title:
        wayback_title = fetch_title_wayback(url)
        time.sleep(HTTP_RATE_LIMIT_S)
        if wayback_title:
            source = "wayback"

    if not any([slug_title, http_title, wayback_title]):
        source = "failed"

    return TitleRecord(
        url=url,
        title_slug=slug_title,
        title_http=http_title,
        title_wayback=wayback_title,
        source=source,
    )


def parse_titles(
    input_csv: Path = RAW_CSV_PATH,
    output_jsonl: Path = TITLES_JSONL_PATH,
    mode: str = "slug",
    limit: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    ensure_directories()
    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError(f"В {input_csv} отсутствует колонка 'url', найдены: {df.columns.tolist()}")
    urls = df["url"].astype(str).tolist()
    if limit is not None:
        urls = urls[:limit]

    cache = _load_cache(output_jsonl)
    to_process = [u for u in urls if u not in cache]
    log.info(
        "Всего URL: %d | Из кэша: %d | К обработке: %d | Режим: %s",
        len(urls),
        len(cache),
        len(to_process),
        mode,
    )

    if mode == "slug" or workers <= 1:
        session = None
        if mode in {"http", "full"}:
            import requests

            session = requests.Session()
        for url in tqdm(to_process, desc="парсинг заголовков"):
            rec = _process_url(url, mode=mode, session=session)
            cache[url] = rec
            _append_cache(output_jsonl, rec)
    else:
        import requests

        session = requests.Session()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_url, u, mode, session): u for u in to_process}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="парсинг заголовков"):
                rec = fut.result()
                cache[rec.url] = rec
                _append_cache(output_jsonl, rec)

    records = [cache[u] for u in urls if u in cache]
    out = pd.DataFrame(
        {
            "url": [r.url for r in records],
            "title_slug": [r.title_slug for r in records],
            "title_http": [r.title_http for r in records],
            "title_wayback": [r.title_wayback for r in records],
            "title_source": [r.source for r in records],
        }
    )
    out["title"] = out["title_http"].fillna(out["title_wayback"]).fillna(out["title_slug"]).fillna("")

    coverage = {
        "slug": (out["title_slug"].str.len() > 0).mean(),
        "http": out["title_http"].notna().mean(),
        "wayback": out["title_wayback"].notna().mean(),
        "final_non_empty": (out["title"].str.len() > 0).mean(),
    }
    log.info("Покрытие: %s", {k: f"{v:.1%}" for k, v in coverage.items()})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Восстановление заголовков статей по URL: slug-парсинг, HTTP, Wayback Machine.",
    )
    parser.add_argument("--input", type=Path, default=RAW_CSV_PATH, help="CSV со колонкой 'url'.")
    parser.add_argument("--output", type=Path, default=TITLES_JSONL_PATH, help="Путь к JSONL-кэшу.")
    parser.add_argument(
        "--mode",
        choices=["slug", "http", "full"],
        default="slug",
        help="Стратегия извлечения: slug / http / full.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Обработать только первые N URL (smoke-тест).")
    parser.add_argument("--workers", type=int, default=1, help="Количество параллельных потоков (для http/full).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parse_titles(
        input_csv=args.input,
        output_jsonl=args.output,
        mode=args.mode,
        limit=args.limit,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
