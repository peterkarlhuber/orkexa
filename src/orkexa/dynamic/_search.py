"""Tiny keyword search shared by the registries."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import TypeVar

T = TypeVar("T")

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "der",
        "die",
        "das",
        "und",
        "oder",
        "ein",
        "eine",
        "fuer",
        "mit",
    ]
)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("_", " "))) - _STOPWORDS


def keyword_search(items: Iterable[tuple[T, str]], query: str, limit: int) -> list[T]:
    """Rank ``(item, text)`` pairs by the number of query words in ``text``.

    An empty query returns the first ``limit`` items unchanged.
    """
    pairs = list(items)
    terms = _words(query)
    if not terms:
        return [item for item, _ in pairs[:limit]]
    scored = [(len(terms & _words(text)), index, item) for index, (item, text) in enumerate(pairs)]
    scored = [entry for entry in scored if entry[0] > 0]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _, _, item in scored[:limit]]
