"""Alpaca paper-trading client — a recorded deviation from the approved
Blueprint (CLAUDE.md §2, documented in §7), added to test whether
`InferenceService`'s decisions (FR-10, FR-11) can actually reach a real
execution venue's order API, not to trade for real. `BROKER.base_url` is a
fixed constant pointed at Alpaca's paper endpoint — there is no live-trading
path anywhere in this module. Any resulting fill is simulated money on a
simulated account; nothing here is or should be read as a financial result
(CLAUDE.md §1).
"""
from __future__ import annotations

import logging

import httpx

from src.config import BROKER

logger = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """A credential problem or a rejected request — not the routine
    "nothing to close" case `close_position` already handles."""


class AlpacaBroker:
    """Thin wrapper over the subset of Alpaca's REST API this project
    needs. Deliberately narrow: this is an integration check of FR-10
    reaching a venue, not a general-purpose trading client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = api_key or BROKER.api_key
        secret_key = secret_key or BROKER.secret_key
        if not api_key or not secret_key:
            raise BrokerError(
                "Alpaca paper-trading credentials not configured — "
                "set ALPACA_API_KEY and ALPACA_SECRET_KEY (NFR-10: never hard-code these)"
            )
        self._client = client or httpx.Client(
            base_url=BROKER.base_url,
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
            timeout=10.0,
        )

    def get_account(self) -> dict:
        resp = self._client.get("/v2/account")
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> list[dict]:
        resp = self._client.get("/v2/positions")
        resp.raise_for_status()
        return resp.json()

    def submit_notional_order(self, ticker: str, side: str, notional: float) -> dict:
        """`notional` is a dollar amount, not a share count — matches this
        project's continuous target-weight model (FR-11) far better than
        Alpaca's default share-count orders would."""
        payload = {
            "symbol": ticker,
            "notional": f"{notional:.2f}",
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        resp = self._client.post("/v2/orders", json=payload)
        resp.raise_for_status()
        return resp.json()

    def close_position(self, ticker: str) -> dict | None:
        """Alpaca's own "close entirely" endpoint — the right primitive for
        FR-12/I5's LIQUIDATE action, which means "get to cash", not "sell
        some amount". Closing a position that doesn't exist is a
        recoverable, expected condition (a fresh paper account holds
        nothing yet), not a programming error — logged and skipped rather
        than raised, the same precedent CLAUDE.md §7 sets for
        `load_partition`'s empty-frame case."""
        resp = self._client.delete(f"/v2/positions/{ticker}")
        if resp.status_code == 404:
            logger.info("close_position(%s): no open position, nothing to close", ticker)
            return None
        resp.raise_for_status()
        return resp.json()

    def get_order(self, order_id: str) -> dict:
        resp = self._client.get(f"/v2/orders/{order_id}")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


def route_decision(
    broker: AlpacaBroker, ticker: str, discrete_action: str, notional: float
) -> dict | None:
    """Translate one FR-11 discrete action into the corresponding Alpaca
    call. HOLD is intentionally a no-op: there is nothing to route, and
    routing it as a zero-notional order would just be rejected by Alpaca
    rather than expressing "do nothing"."""
    if discrete_action == "BUY":
        return broker.submit_notional_order(ticker, "buy", notional)
    if discrete_action == "SELL":
        return broker.submit_notional_order(ticker, "sell", notional)
    if discrete_action == "LIQUIDATE":
        return broker.close_position(ticker)
    if discrete_action == "HOLD":
        return None
    raise BrokerError(f"unknown discrete_action: {discrete_action!r}")
