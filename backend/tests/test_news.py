"""Tests for the market headlines (src/dataops/news.py, GET /news and the
assistant's get_news tool).

No network (CLAUDE.md §11): the provider is an httpx.MockTransport. Headlines
are context only, so the tests also check what must never happen: provider
HTML or unsafe links reaching the browser, a failed provider breaking the
page, and decision-making code importing news (via .importlinter, run by
test_architecture.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from src.dataops import news as news_module
from src.dataops.news import NewsFeed, NewsUnavailable, clean_article
from src.serving import api
from src.serving.api import app
from src.serving.assistant import PipelineTools


def _article(i, symbols, headline="Exxon reports results", **extra):
    return {"id": i, "created_at": f"2026-10-22T1{i % 10}:00:00Z", "headline": headline, "summary": "",
            "source": "benzinga", "url": f"https://example.com/{i}", "symbols": symbols} | extra


class Provider:
    """Counts requests and serves canned articles, or fails."""

    def __init__(self, articles=(), status=200):
        self.articles, self.status, self.requests = list(articles), status, []

    def client(self):
        def handle(request):
            self.requests.append(request)
            return httpx.Response(self.status, json={"news": self.articles, "next_page_token": None})
        return httpx.Client(base_url="https://data.example", transport=httpx.MockTransport(handle))


def test_articles_are_reduced_to_plain_text_and_safe_links():
    a = clean_article(_article(1, ["XOM", "AAPL", "MSFT"], headline="<b>Exxon</b> &amp; Chevron",
                               summary="<p>Line one</p><script>x()</script> two", url="javascript:alert(1)"))
    assert a["headline"] == "Exxon & Chevron"
    assert "<" not in a["summary"] and "Line one" in a["summary"]
    assert a["url"] is None
    assert a["tickers"] == ["XOM"] and a["other_symbols"] == 2
    assert a["published_at"] == datetime(2026, 10, 22, 11, tzinfo=timezone.utc)
    assert clean_article(_article(2, ["XOM"], headline="  ")) is None
    assert clean_article(_article(3, ["XOM"], created_at="not a date")) is None


def test_long_summaries_are_cut_at_a_word():
    a = clean_article(_article(1, ["XOM"], summary="word " * 200))
    assert len(a["summary"]) <= news_module.SUMMARY_CHARS and a["summary"].endswith("…")


def test_headlines_are_cached_then_refetched_after_the_ttl():
    provider, now = Provider([_article(1, ["XOM"])]), [1000.0]
    feed = NewsFeed(client=provider.client(), clock=lambda: now[0])
    assert len(feed.headlines("XOM")) == 1
    feed.headlines("XOM")
    assert len(provider.requests) == 1  # served from the cache
    now[0] += news_module.NEWS.cache_seconds + 1
    feed.headlines("XOM")
    assert len(provider.requests) == 2
    params = provider.requests[0].url.params
    assert params["symbols"] == "XOM" and params["sort"] == "desc"


def test_one_ticker_keeps_only_articles_tagging_it_and_all_asks_for_the_universe():
    provider = Provider([_article(1, ["XOM"]), _article(2, ["CVX"])])
    feed = NewsFeed(client=provider.client())
    assert [a["id"] for a in feed.headlines("XOM")] == ["1"]
    feed.headlines(None)
    assert provider.requests[-1].url.params["symbols"] == "XOM,CVX,SHEL,BP,NEE"
    with pytest.raises(ValueError):
        feed.headlines("AAPL")


def test_a_failing_provider_raises_a_clear_error():
    with pytest.raises(NewsUnavailable):
        NewsFeed(client=Provider(status=403).client()).headlines("XOM")


# --------------------------------------------------------------- endpoint
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    return TestClient(app)


def _keys(monkeypatch, on=True):
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        if on:
            monkeypatch.setenv(name, "test")
        else:
            monkeypatch.delenv(name, raising=False)


def test_news_is_off_without_alpaca_keys(client, monkeypatch):
    _keys(monkeypatch, on=False)
    with client:
        body = client.get("/news").json()
    assert body == {"enabled": False, "ticker": None, "items": []}


def test_news_endpoint_serves_cleaned_headlines(client, monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(api, "news_feed", NewsFeed(client=Provider([_article(1, ["XOM", "SPY"])]).client()))
    with client:
        body = client.get("/news", params={"ticker": "XOM"}).json()
    [item] = body["items"]
    assert body["enabled"] and item["headline"] == "Exxon reports results" and item["other_symbols"] == 1


def test_news_endpoint_rejects_other_tickers_and_reports_provider_failure(client, monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setattr(api, "news_feed", NewsFeed(client=Provider(status=500).client()))
    with client:
        assert client.get("/news", params={"ticker": "AAPL"}).status_code == 422
        failed = client.get("/news")
    assert failed.status_code == 503 and failed.json()["detail"] == "News is unavailable right now."


def test_news_can_be_switched_off_with_keys_set(client, monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setenv("NEWS_ENABLED", "false")
    with client:
        assert client.get("/news").json()["enabled"] is False


# --------------------------------------------------------------- assistant tool
def test_the_assistant_reads_headlines_and_says_they_are_context_only():
    feed = NewsFeed(client=Provider([_article(i, ["XOM"] + ["X"] * (i % 2)) for i in range(10)]).client())
    out = PipelineTools(repo=None, status_fn=dict, news=feed).run("get_news", {"ticker": "XOM"})
    assert len(out.result["headlines"]) == 8 and len(out.visual["items"]) == 10  # model reads fewer
    assert "never reads news" in out.result["note"]
    assert "summary" not in out.result["headlines"][0] and "url" not in out.result["headlines"][0]
    assert out.visual["type"] == "news" and out.visual["items"][0]["url"].startswith("https://")


def test_the_news_tool_degrades_when_news_is_missing_or_down():
    off = PipelineTools(repo=None, status_fn=dict).run("get_news", {})
    assert off.result["available"] is False and off.visual is None
    down = PipelineTools(repo=None, status_fn=dict, news=NewsFeed(client=Provider(status=503).client()))
    assert down.run("get_news", {}).result["available"] is False
