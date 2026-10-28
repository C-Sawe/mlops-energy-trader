"""Market headlines for the energy universe, from Alpaca's news API.

**Context for the person reading the dashboard, never an input to the
agent.** Nothing in training, inference, the CT loop or paper trading reads
this module, so news cannot change a decision; the dashboard and the
assistant say so wherever headlines appear. Feeding news to the agent is
left as future work: scoring historical headlines with a language model
would leak knowledge of what happened next into a backtest (I1).

A read-only HTTP source like `ingestion.py` (DR-10: public market data, no
personal data). Headlines are cached in memory for `NEWS.cache_seconds`, and
article text is reduced to plain text here, so nothing downstream ever
handles provider HTML.
"""
from __future__ import annotations

import html
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx

from src.config import BROKER, DATA, NEWS

logger = logging.getLogger(__name__)

MAX_ITEMS = 50  # Alpaca's page size; more than a card or an answer needs
SUMMARY_CHARS = 280
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


class NewsUnavailable(RuntimeError):
    """The provider couldn't be reached or refused the request."""


def _plain(text: str | None, limit: int | None = None) -> str:
    out = _SPACE.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()
    if limit and len(out) > limit:
        out = out[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out


def _safe_url(url: str | None) -> str | None:
    """Only web links reach the browser (no javascript: or data: URLs)."""
    return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None


def clean_article(raw: dict) -> dict | None:
    """One provider article as the dashboard uses it, or None if unusable.
    `tickers` keeps only this project's universe; `other_symbols` counts the
    rest, because a roundup that tags thirty companies is weaker evidence
    about one of them than a story about that company alone."""
    headline = _plain(raw.get("headline"))
    try:
        published = datetime.fromisoformat(str(raw.get("created_at")).replace("Z", "+00:00"))
    except ValueError:
        return None
    if not headline:
        return None
    symbols = [s for s in raw.get("symbols") or [] if isinstance(s, str)]
    ours = [s for s in symbols if s in DATA.tickers]
    return {
        "id": str(raw.get("id")),
        "published_at": published,
        "headline": headline,
        "summary": _plain(raw.get("summary"), SUMMARY_CHARS) or None,
        "source": _plain(raw.get("source")) or None,
        "url": _safe_url(raw.get("url")),
        "tickers": ours,
        "other_symbols": len(symbols) - len(ours),
    }


class NewsFeed:
    """Recent headlines for one ticker or the whole universe, cached."""

    def __init__(self, client: httpx.Client | None = None, clock=time.monotonic) -> None:
        self._client = client
        self._clock = clock
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=NEWS.data_url,
                headers={"APCA-API-KEY-ID": BROKER.api_key or "", "APCA-API-SECRET-KEY": BROKER.secret_key or ""},
                timeout=10.0,
            )
        return self._client

    def headlines(self, ticker: str | None = None, days: int = 7, limit: int = 20) -> list[dict]:
        """Newest first. `ticker` None means any of the universe."""
        if ticker is not None and ticker not in DATA.tickers:
            raise ValueError(f"ticker must be one of {list(DATA.tickers)}")
        days, limit = max(1, min(days, 30)), max(1, min(limit, MAX_ITEMS))
        key = (ticker, days, limit)
        with self._lock:
            hit = self._cache.get(key)
            if hit and self._clock() - hit[0] < NEWS.cache_seconds:
                return hit[1]
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = self._http().get("/v1beta1/news", params={
                "symbols": ticker or ",".join(DATA.tickers), "start": start,
                "limit": limit, "sort": "desc", "include_content": "false",
            })
            resp.raise_for_status()
            articles = resp.json().get("news", [])
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("news: provider request failed: %s", type(exc).__name__)
            raise NewsUnavailable("news provider unavailable") from exc
        items = [a for a in (clean_article(r) for r in articles if isinstance(r, dict)) if a]
        if ticker:
            items = [a for a in items if ticker in a["tickers"]]
        with self._lock:
            self._cache[key] = (self._clock(), items)
        return items
