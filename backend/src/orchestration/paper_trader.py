"""Daily forward paper trading on Alpaca (CLAUDE.md §7's recorded deviation,
extended from a one-off validation to a scheduled cycle).

Each weekday after the US close: ingest today's bar, ask
`InferenceService.predict()` for decisions given the paper account's real
current weights, turn the raw target weights into orders
(`src.execution.rebalance`), and submit them to the *paper* account. Orders
queue for the next open, which is I2's timing on a real venue: decide on
day t's close, realise from t+1. Between cycles, `sync()` records the
account's equity, positions and fills.

What this is evidence of (CLAUDE.md §1): the architecture running forward
in time, unattended, against a real order API, out-of-sample by
construction. What it is not: a measure of profitability. Weeks or months
of paper P&L on five tickers carry no statistical weight; §10's deflated
Sharpe analysis already says what this strategy's edge is worth.

Lives in `src/orchestration`, not `src/execution`: it composes
`InferenceService` with the broker, and `.importlinter` forbids
`src.execution` from importing serving or orchestration.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from datetime import date, datetime, timezone
from enum import Enum

import pandas as pd

from src.config import DATA, PAPER
from src.dataops.repository import MarketRepository
from src.execution.alpaca_broker import AlpacaBroker
from src.execution.rebalance import OrderIntent, liquidate_all, plan_rebalance
from src.serving.inference import InferenceService

logger = logging.getLogger(__name__)

MARKET_TZ = "America/New_York"


def market_today(now: datetime | None = None) -> date:
    """The US market's calendar date — the date a bar ingested tonight
    carries, regardless of the server's own timezone."""
    ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MARKET_TZ).date()


def next_run_at(now: datetime, run_time_utc: str) -> datetime:
    """The next weekday at `run_time_utc` (HH:MM) strictly after `now`.
    A fixed wall-clock time, unlike the interval-based ingestion scheduler,
    which drifts with whenever the process happened to start. Weekend days
    are skipped here; market holidays are caught later by the "no bar for
    today" check in `run_cycle`."""
    hour, minute = (int(x) for x in run_time_utc.split(":"))
    candidate = now.astimezone(timezone.utc).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += pd.Timedelta(days=1).to_pytimedelta()
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1).to_pytimedelta()
    return candidate


def _parse_ts(value: str | None) -> datetime | None:
    return pd.Timestamp(value).to_pydatetime() if value else None


def _f(value) -> float | None:
    return float(value) if value not in (None, "") else None


class PaperStatus(str, Enum):
    IDLE = "IDLE"
    TRADING = "TRADING"


class PaperTrader:
    def __init__(
        self,
        service: InferenceService,
        repo: MarketRepository,
        tickers: tuple[str, ...] = DATA.tickers,
        min_order_notional: float | None = None,
    ) -> None:
        self.service = service
        self.repo = repo
        self.tickers = tickers
        self.min_order_notional = (
            PAPER.min_order_notional if min_order_notional is None else min_order_notional
        )
        self._lock = threading.RLock()
        self._status = PaperStatus.IDLE
        self._last_run_at: datetime | None = None
        self._last_outcome: str | None = None

    @property
    def status(self) -> PaperStatus:
        with self._lock:
            return self._status

    @property
    def last_run_at(self) -> datetime | None:
        with self._lock:
            return self._last_run_at

    @property
    def last_outcome(self) -> str | None:
        """A short human-readable result of the last cycle attempt
        ("traded 3 orders", "skipped: no bar for 2026-10-02", ...)."""
        with self._lock:
            return self._last_outcome

    # ------------------------------------------------------------------ sync
    def sync(self, broker: AlpacaBroker, now: datetime | None = None) -> dict:
        """Record account equity, cash and positions for today, and pick up
        fills for any order still open. Safe to call as often as wanted."""
        for order_id in self.repo.unfilled_paper_order_ids():
            try:
                o = broker.get_order(order_id)
            except Exception:  # noqa: BLE001 - one bad lookup must not stop the sync
                logger.exception("paper sync: could not fetch order %s", order_id)
                continue
            self.repo.update_paper_order(
                order_id,
                o.get("status", "unknown"),
                _f(o.get("filled_qty")),
                _f(o.get("filled_avg_price")),
                _parse_ts(o.get("filled_at")),
            )

        account = broker.get_account()
        positions = [
            {
                "ticker": p["symbol"],
                "qty": _f(p.get("qty")),
                "avg_entry_price": _f(p.get("avg_entry_price")),
                "current_price": _f(p.get("current_price")),
                "market_value": _f(p.get("market_value")),
                "unrealized_pl": _f(p.get("unrealized_pl")),
                "unrealized_plpc": _f(p.get("unrealized_plpc")),
            }
            for p in broker.get_positions()
        ]
        snapshot = {
            "equity": float(account["equity"]),
            "cash": float(account["cash"]),
            "positions": positions,
        }
        now = now or datetime.now(timezone.utc)
        self.repo.record_paper_account(
            market_today(now), snapshot["equity"], snapshot["cash"], positions, now
        )
        return snapshot

    # ----------------------------------------------------------------- cycle
    def run_cycle(self, broker: AlpacaBroker, now: datetime | None = None) -> str:
        """One daily cycle. Returns (and records) a short outcome string.
        Skips, rather than trades, when today's bar hasn't arrived (weekend,
        holiday, or yfinance lag) or when this signal date already traded."""
        with self._lock:
            if self._status == PaperStatus.TRADING:
                return "skipped: a cycle is already running"
            self._status = PaperStatus.TRADING
        outcome = "failed"
        try:
            outcome = self._cycle(broker, now)
            return outcome
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the scheduler to log
            outcome = f"failed: {type(exc).__name__}"
            raise
        finally:
            with self._lock:
                self._status = PaperStatus.IDLE
                self._last_run_at = datetime.now(timezone.utc)
                self._last_outcome = outcome

    def _cycle(self, broker: AlpacaBroker, now: datetime | None) -> str:
        today = market_today(now)
        account = self.sync(broker, now)

        latest = self.repo.latest_ingest_info()
        if latest is None or latest["date"] != today:
            got = latest["date"] if latest else None
            return f"skipped: no bar for {today} (latest is {got})"
        if self.repo.has_paper_cycle(today):
            return f"skipped: {today} already traded"

        equity = account["equity"]
        holdings = {
            p["ticker"]: p["market_value"] or 0.0
            for p in account["positions"]
            if p["ticker"] in self.tickers
        }
        weights = {t: max(0.0, min(1.0, holdings.get(t, 0.0) / equity)) for t in self.tickers}
        cash_weight = max(0.0, min(1.0, account["cash"] / equity))

        result = self.service.predict(weights, cash_weight)
        decision_ids = {d["ticker"]: d.get("decision_id") for d in result["decisions"]}

        if result["failsafe_triggered"]:
            intents = liquidate_all(holdings)
        else:
            raw = {d["ticker"]: d["raw_weight"] for d in result["decisions"]}
            intents = plan_rebalance(
                raw, holdings, equity, account["cash"], self.min_order_notional
            )

        submitted, failed = [], 0
        for intent in intents:
            try:
                submitted.append(self._submit(broker, intent, decision_ids.get(intent.ticker)))
            except Exception:  # noqa: BLE001 - one rejected order must not abort the rest
                logger.exception("paper cycle: order failed %s", asdict(intent))
                failed += 1
        submitted = [o for o in submitted if o is not None]

        self.repo.record_paper_cycle(
            today, result["version_id"], result["failsafe_triggered"], submitted, failed
        )
        tag = " (fail-safe)" if result["failsafe_triggered"] else ""
        return f"traded {len(submitted)} orders{tag}" + (f", {failed} failed" if failed else "")

    def _submit(self, broker: AlpacaBroker, intent: OrderIntent, decision_id: str | None) -> dict | None:
        if intent.close:
            resp = broker.close_position(intent.ticker)
            if resp is None:  # nothing was actually held
                return None
        else:
            resp = broker.submit_notional_order(intent.ticker, intent.side, intent.notional)
        return {
            "order_id": resp["id"],
            "decision_id": decision_id,
            "ticker": intent.ticker,
            "side": intent.side,
            "requested_notional": intent.notional,
            "status": resp.get("status", "new"),
        }
