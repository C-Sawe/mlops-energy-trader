"""Tests for target weights -> paper orders (src/execution/rebalance.py).
Pure arithmetic: no broker, no network, no database."""
from __future__ import annotations

import pytest

from src.execution.rebalance import (
    OrderIntent,
    clamp_long_only,
    liquidate_all,
    plan_rebalance,
)


def test_negative_targets_clamp_to_flat_not_short():
    assert clamp_long_only({"XOM": -0.8, "CVX": 0.3}) == {"XOM": 0.0, "CVX": 0.3}


def test_positive_targets_above_one_scale_down_to_no_leverage():
    clamped = clamp_long_only({"XOM": 0.9, "CVX": 0.9, "BP": -0.4})
    assert sum(clamped.values()) == pytest.approx(1.0)
    assert clamped["XOM"] == pytest.approx(0.5)
    assert clamped["BP"] == 0.0


def test_buy_from_cash_sizes_to_target_weight():
    orders = plan_rebalance({"XOM": 0.25, "CVX": 0.0}, {}, equity=100_000, cash=100_000)
    assert orders == [OrderIntent("XOM", "buy", 25_000.0)]


def test_zero_target_on_a_held_position_closes_it_outright():
    """A sized sell could leave a fractional remainder behind; "be flat"
    is the close-position primitive, the same reasoning as LIQUIDATE."""
    orders = plan_rebalance({"XOM": -0.6}, {"XOM": 10_000}, equity=100_000, cash=90_000)
    assert orders == [OrderIntent("XOM", "sell", None, close=True)]


def test_partial_reduction_is_a_sized_sell_capped_at_what_is_held():
    orders = plan_rebalance({"XOM": 0.1}, {"XOM": 30_000}, equity=100_000, cash=70_000)
    assert orders == [OrderIntent("XOM", "sell", 20_000.0)]


def test_swap_funds_the_buy_from_the_sell_and_scales_only_the_buy():
    """The multi-ticker case CLAUDE.md §7 says every rebalance change must
    be checked against: a fully invested account (no cash) swapping one
    ticker for another. The sell executes in full; the buy is funded by
    it. Symmetric scaling of both legs would cancel and trade nothing."""
    orders = plan_rebalance(
        {"XOM": 0.0, "CVX": 1.0}, {"XOM": 100_000}, equity=100_000, cash=0.0
    )
    assert OrderIntent("XOM", "sell", None, close=True) in orders
    assert OrderIntent("CVX", "buy", 100_000.0) in orders


def test_unaffordable_buys_scale_back_proportionally_not_rejected():
    # Targets sum to 1.0 against equity 100k, but only 50k is cash: the
    # other 50k sits in a ticker the agent wants to keep.
    orders = plan_rebalance(
        {"XOM": 0.5, "CVX": 0.25, "BP": 0.25},
        {"XOM": 60_000},  # equity counts it at 60k; cash only 40k
        equity=100_000,
        cash=40_000,
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["XOM"].side == "sell" and by_ticker["XOM"].notional == 10_000.0
    # budget = 40k cash + 10k proceeds = 50k = exactly what CVX + BP want
    assert by_ticker["CVX"].notional == 25_000.0
    assert by_ticker["BP"].notional == 25_000.0

    scaled = plan_rebalance({"CVX": 0.5, "BP": 0.5}, {}, equity=100_000, cash=50_000)
    assert sum(o.notional for o in scaled) == pytest.approx(50_000.0)
    assert {o.side for o in scaled} == {"buy"}


def test_deltas_below_minimum_notional_are_skipped():
    """A "hold" target carries float rounding error (CLAUDE.md §7's
    _DUST_FRACTION note); it must not become an order every single day."""
    orders = plan_rebalance(
        {"XOM": 0.2500001}, {"XOM": 25_000}, equity=100_000, cash=75_000, min_notional=1.0
    )
    assert orders == []


def test_liquidate_all_closes_only_what_is_held():
    assert liquidate_all({"XOM": 5_000, "CVX": 0.0}) == [OrderIntent("XOM", "sell", None, close=True)]


def test_nonpositive_equity_plans_nothing():
    assert plan_rebalance({"XOM": 1.0}, {}, equity=0.0, cash=0.0) == []
