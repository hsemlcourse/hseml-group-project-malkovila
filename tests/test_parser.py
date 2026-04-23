from __future__ import annotations

import pytest

from src.data.parse_titles import extract_title_from_slug


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "http://mashable.com/2013/01/07/amazon-instant-video-browser/",
            "Amazon Instant Video Browser",
        ),
        (
            "http://mashable.com/2014/10/31/apple-pay-launch/",
            "Apple Pay Launch",
        ),
        (
            "https://mashable.com/2013/05/20/5-ways-to-get-fired/",
            "5 Ways To Get Fired",
        ),
        (
            "http://mashable.com/2014/07/04/american-flag-emoji/",
            "American Flag Emoji",
        ),
        (
            "http://mashable.com/2013/12/25/merry-christmas",
            "Merry Christmas",
        ),
        (
            "http://mashable.com/2014/10/31/",
            "",
        ),
        (
            "",
            "",
        ),
    ],
)
def test_extract_title_from_slug(url: str, expected: str) -> None:
    assert extract_title_from_slug(url) == expected


def test_extract_title_returns_str_always() -> None:
    assert isinstance(extract_title_from_slug("http://example.com/no/date/structure"), str)
