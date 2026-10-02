"""Tests for the daily forward paper-trading cycle
(src/orchestration/paper_trader.py). No network: a duck-typed fake broker
stands in for Alpaca, and the repository is a temp-file SQLite database.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.config import DATA
from src.dataops import processing as P
from src.dataops.repository import MarketRepository
from src.orchestration.paper_trader import PaperTrader, market_today, next_run_at
from tests.test_processing import make_series


class FakeBroker:
    """Holds positions as {ticker: market_value}; records every call."""

    def __init__(self, cash: float = 100_000.0, positions: dict[str, float] | None = None):
        self.cash = cash
        self.positions = dict(positions or {})
        self.submitted: list[tuple[str, str, float]] = []
        self.closed: list[str] = []
        self.orders: dict[str, dict] = {}
        self.fail_tickers: set[str] = set()

    def get_account(self):
        equity = self.cash + sum(self.positions.values())
        return {"equity": str(equity), "cash": str(self.cash)}

    def get_positions(self):
        return [
            {
                "symbol": t, "qty": "10", "avg_entry_price": "100.0", "current_price": str(v / 10),
                "market_value": str(v), "unrealized_pl": "0", "unrealized_plpc": "0",
            }
            for t, v in self.positions.items()
        ]

    def submit_notional_order(self, ticker, side, notional):
        if ticker in self.fail_tickers:
            raise RuntimeError("rejected")
        self.submitted.append((ticker, side, notional))
        order_id = f"o{len(self.orders) + 1}"
        self.orders[order_id] = {"id": order_id, "status": "accepted"}
        return self.orders[order_id]

    def close_position(self, ticker):
        if ticker not in self.positions:
            return None
        self.closed.append(ticker)
        order_id = f"o{len(self.orders) + 1}"
        self.orders[order_id] = {"id": order_id, "status": "accepted"}
        return self.orders[order_id]

    def get_order(self, order_id):
        return self.orders[order_id]

    def close(self):
        pass


class FakeService:
    """Returns fixed raw weights; mimics InferenceService.predict's shape."""

    def __init__(self, weights: dict[str, float], failsafe: bool = False):
        self.weights = weights
        self.failsafe = failsafe
        self.calls: list[tuple[dict, float]] = []

    def predict(self, positions, cash_weight):
        self.calls.append((positions, cash_weight))
        return {
            "decisions": [
                {
                    "ticker": t,
                    "raw_weight": None if self.failsafe else self.weights.get(t, 0.0),
                    "discrete_action": "LIQUIDATE" if self.failsafe else "HOLD",
                    "decision_id": None,
                }
                for t in DATA.tickers
            ],
            "failsafe_triggered": self.failsafe,
            "vix_at_decision": 20.0,
            "version_id": "v-test",
            "decided_at": datetime.now(timezone.utc),
        }


def _repo(tmp_path, n_days: int = 120) -> tuple[MarketRepository, datetime]:
    """Seeded repo plus a `now` (22:00 UTC on the last bar's date) at which
    that last bar is "today's" bar in market time."""
    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 20, len(dates))})
    repo = MarketRepository(url=f"sqlite:///{tmp_path / 'test.db'}")
    repo.create_schema()
    repo.persist(P.build_feature_frame(equities, vix))
    last = dates.max()
    now = datetime(last.year, last.month, last.day, 22, 0, tzinfo=timezone.utc)
    return repo, now


def test_cycle_rebalances_toward_long_only_targets_and_records_orders(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker(cash=100_000)
    trader = PaperTrader(FakeService({"XOM": 0.3, "CVX": -0.9}), repo)

    outcome = trader.run_cycle(broker, now=now)

    assert outcome == "traded 1 orders"
    assert broker.submitted == [("XOM", "buy", 30_000.0)]  # CVX's short clamped to flat
    cycle = repo.latest_paper_cycle()
    assert cycle["signal_date"] == market_today(now)
    assert cycle["orders_submitted"] == 1
    assert repo.unfilled_paper_order_ids() == ["o1"]


def test_cycle_feeds_the_accounts_real_weights_into_predict(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker(cash=75_000, positions={"XOM": 25_000})
    service = FakeService({})
    PaperTrader(service, repo).run_cycle(broker, now=now)

    positions, cash_weight = service.calls[0]
    assert positions["XOM"] == pytest.approx(0.25)
    assert positions["CVX"] == 0.0
    assert cash_weight == pytest.approx(0.75)


def test_same_signal_date_never_trades_twice(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker()
    trader = PaperTrader(FakeService({"XOM": 0.3}), repo)

    trader.run_cycle(broker, now=now)
    outcome = trader.run_cycle(broker, now=now)

    assert outcome.startswith("skipped:") and "already traded" in outcome
    assert len(broker.submitted) == 1


def test_no_bar_for_today_skips_without_trading(tmp_path):
    """Weekend, holiday, or yfinance lag: the latest bar is not today's,
    so deciding now would re-trade yesterday's close."""
    repo, now = _repo(tmp_path)
    broker = FakeBroker()
    service = FakeService({"XOM": 0.3})
    outcome = PaperTrader(service, repo).run_cycle(broker, now=now + pd.Timedelta(days=3))

    assert outcome.startswith("skipped: no bar for")
    assert broker.submitted == [] and service.calls == []
    assert repo.latest_paper_cycle() is None


def test_failsafe_closes_every_held_position(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker(cash=50_000, positions={"XOM": 30_000, "BP": 20_000})
    outcome = PaperTrader(FakeService({}, failsafe=True), repo).run_cycle(broker, now=now)

    assert sorted(broker.closed) == ["BP", "XOM"]
    assert broker.submitted == []
    assert "fail-safe" in outcome
    assert repo.latest_paper_cycle()["failsafe_triggered"] is True


def test_one_rejected_order_does_not_abort_the_others(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker()
    broker.fail_tickers = {"XOM"}
    outcome = PaperTrader(FakeService({"XOM": 0.3, "CVX": 0.3}), repo).run_cycle(broker, now=now)

    assert broker.submitted == [("CVX", "buy", 30_000.0)]
    assert outcome == "traded 1 orders, 1 failed"
    assert repo.latest_paper_cycle()["orders_failed"] == 1


def test_sync_records_equity_positions_and_picks_up_fills(tmp_path):
    repo, now = _repo(tmp_path)
    broker = FakeBroker()
    trader = PaperTrader(FakeService({"XOM": 0.3}), repo)
    trader.run_cycle(broker, now=now)

    # The queued order fills at the next open; a later sync sees it.
    broker.orders["o1"] = {
        "id": "o1", "status": "filled", "filled_qty": "300.5",
        "filled_avg_price": "99.83", "filled_at": "2026-10-02T13:30:01Z",
    }
    broker.cash, broker.positions = 70_000, {"XOM": 30_100}
    trader.sync(broker, now=now)

    assert repo.unfilled_paper_order_ids() == []
    fills = repo.list_paper_fills("2000-01-01", ticker="XOM")
    assert fills[0]["filled_avg_price"] == pytest.approx(99.83)
    assert fills[0]["side"] == "buy"
    account = repo.list_paper_account()
    assert account[-1]["equity"] == pytest.approx(100_100)
    assert account[-1]["positions"][0]["ticker"] == "XOM"


def test_next_run_is_a_fixed_weekday_time_after_now():
    friday_late = datetime(2026, 10, 2, 23, 0, tzinfo=timezone.utc)  # Fri, after 22:00
    assert next_run_at(friday_late, "22:00") == datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)  # Mon

    tuesday_early = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
    assert next_run_at(tuesday_early, "22:00") == datetime(2026, 10, 6, 22, 0, tzinfo=timezone.utc)


def test_market_today_uses_new_york_date_not_utc():
    # 02:00 UTC on Oct 3 is still Oct 2 in New York.
    assert str(market_today(datetime(2026, 10, 3, 2, 0, tzinfo=timezone.utc))) == "2026-10-02"
