from __future__ import annotations

from src.config import DATA_CHANNEL_COLS, WEEKDAY_COLS

CHANNEL_ALIASES: dict[str, str] = {
    "lifestyle": "data_channel_is_lifestyle",
    "entertainment": "data_channel_is_entertainment",
    "bus": "data_channel_is_bus",
    "business": "data_channel_is_bus",
    "socmed": "data_channel_is_socmed",
    "social": "data_channel_is_socmed",
    "tech": "data_channel_is_tech",
    "technology": "data_channel_is_tech",
    "world": "data_channel_is_world",
}

WEEKDAY_ALIASES: dict[str, str] = {
    "monday": "weekday_is_monday",
    "tuesday": "weekday_is_tuesday",
    "wednesday": "weekday_is_wednesday",
    "thursday": "weekday_is_thursday",
    "friday": "weekday_is_friday",
    "saturday": "weekday_is_saturday",
    "sunday": "weekday_is_sunday",
}

VALID_CHANNELS: frozenset[str] = frozenset(CHANNEL_ALIASES.keys())
VALID_WEEKDAYS: frozenset[str] = frozenset(WEEKDAY_ALIASES.keys())


def normalize_channel(channel: str) -> str:
    key = channel.strip().lower().replace(" ", "_")
    if key not in CHANNEL_ALIASES:
        raise ValueError(
            f"Неизвестный канал: {channel!r}. Допустимо: {', '.join(sorted(VALID_CHANNELS))}"
        )
    return key


def normalize_weekday(weekday: str) -> str:
    key = weekday.strip().lower()
    if key not in WEEKDAY_ALIASES:
        raise ValueError(
            f"Неизвестный день недели: {weekday!r}. Допустимо: {', '.join(sorted(VALID_WEEKDAYS))}"
        )
    return key


def encode_channel_weekday(channel: str, weekday: str) -> dict[str, float]:
    ch_key = normalize_channel(channel)
    wd_key = normalize_weekday(weekday)
    out: dict[str, float] = {c: 0.0 for c in DATA_CHANNEL_COLS}
    out[CHANNEL_ALIASES[ch_key]] = 1.0
    for c in WEEKDAY_COLS:
        out[c] = 0.0
    out[WEEKDAY_ALIASES[wd_key]] = 1.0
    out["is_weekend"] = 1.0 if wd_key in ("saturday", "sunday") else 0.0
    return out
