"""Tests for the Alpaca paper-trading client (CLAUDE.md §7's recorded
deviation). No network: `httpx.MockTransport` stands in for Alpaca's API,
consistent with this project's "no network, no database" test convention
— the same reason `test_ingestion_scheduler.py` monkeypatches
`fetch_market_data` rather than calling yfinance for real.
"""
from __future__ import annotations

import httpx
import pytest

from src.execution.alpaca_broker import AlpacaBroker, BrokerError, route_decision


def _broker(handler) -> AlpacaBroker:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://paper-api.alpaca.markets")
    return AlpacaBroker(api_key="test-key", secret_key="test-secret", client=client)


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(BrokerError):
        AlpacaBroker()


def test_submit_notional_order_sends_dollar_amount_not_share_count():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json={"id": "order-1", "status": "accepted"})

    broker = _broker(handler)
    result = broker.submit_notional_order("XOM", "buy", 123.456)

    assert seen["method"] == "POST"
    assert seen["url"].endswith("/v2/orders")
    assert b'"notional":"123.46"' in seen["body"]
    assert b'"symbol":"XOM"' in seen["body"]
    assert result["status"] == "accepted"


def test_close_position_treats_404_as_a_no_op_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "position does not exist"})

    broker = _broker(handler)
    assert broker.close_position("XOM") is None


def test_close_position_raises_on_a_genuine_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "internal error"})

    broker = _broker(handler)
    with pytest.raises(httpx.HTTPStatusError):
        broker.close_position("XOM")


def test_route_decision_maps_each_discrete_action():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "closed"})
        return httpx.Response(200, json={"id": "order", "status": "accepted"})

    broker = _broker(handler)

    route_decision(broker, "XOM", "BUY", 100.0)
    route_decision(broker, "CVX", "SELL", 100.0)
    route_decision(broker, "NEE", "LIQUIDATE", 100.0)
    result = route_decision(broker, "SHEL", "HOLD", 100.0)

    assert result is None
    assert len(calls) == 3  # HOLD made no request at all
    assert calls[0] == ("POST", "https://paper-api.alpaca.markets/v2/orders")
    assert calls[1] == ("POST", "https://paper-api.alpaca.markets/v2/orders")
    assert calls[2] == ("DELETE", "https://paper-api.alpaca.markets/v2/positions/NEE")


def test_route_decision_rejects_an_unknown_action():
    broker = _broker(lambda request: httpx.Response(200, json={}))
    with pytest.raises(BrokerError):
        route_decision(broker, "XOM", "SOMETHING_ELSE", 100.0)
