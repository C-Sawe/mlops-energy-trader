"""Tests for the baseline policies (CLAUDE.md §10.1).

A Sharpe Ratio is meaningless without these: the same conditions that
reward a learned policy also reward simply holding the assets.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rlops import baselines as B
from src.rlops.environment import TradingEnvironment
from tests.test_environment import _feature_frame, _single_ticker_frame_with_tail


def test_all_cash_never_trades_and_preserves_capital():
    frame = _feature_frame(tickers=("XOM", "CVX"))
    env = TradingEnvironment(frame, tickers=("XOM", "CVX"), transaction_cost_pct=0.01)

    result = B.run_policy(env, B.all_cash)

    assert np.all(result["equity_curve"] == pytest.approx(env.initial_cash))
    assert np.all(result["returns"] == pytest.approx(0.0))


def test_buy_and_hold_matches_frictionless_manual_return():
    tail = [100.0, 120.0, 90.0, 150.0]
    frame = _single_ticker_frame_with_tail(tail)
    env = TradingEnvironment(
        frame, tickers=("XOM",), transaction_cost_pct=0.0, drawdown_penalty_coef=0.0
    )

    starting_price = env._prices[0, 0]  # the episode may start mid-warmup, not at tail[0]
    result = B.run_policy(env, B.buy_and_hold)

    expected_final = env.initial_cash * (tail[-1] / starting_price)
    assert result["equity_curve"][-1] == pytest.approx(expected_final, rel=1e-6)


def test_buy_and_hold_does_not_rebalance_after_the_first_step():
    """After the initial allocation, buy-and-hold's target is always the
    environment's own current weights, so no further trade is issued."""
    frame = _feature_frame(tickers=("XOM", "CVX"))
    env = TradingEnvironment(frame, tickers=("XOM", "CVX"), transaction_cost_pct=0.01)
    env.reset()

    env.step(B.buy_and_hold(env, 0))
    shares_after_first_trade = env.shares.copy()

    env.step(B.buy_and_hold(env, 1))

    np.testing.assert_array_equal(env.shares, shares_after_first_trade)


def test_equal_weight_rebalanced_costs_more_with_fees_than_without():
    """CLAUDE.md §10.4: frictionless simulation flatters any policy that
    trades often. Isolate the drag by fixing the policy and price path and
    varying only the fee, so the comparison cannot be confounded by a
    rebalancing premium or penalty from the price path itself."""
    frame = _feature_frame(n=250, tickers=("XOM", "CVX", "SHEL"))
    tickers = ("XOM", "CVX", "SHEL")

    frictionless = B.run_policy(
        TradingEnvironment(frame, tickers=tickers, transaction_cost_pct=0.0),
        B.equal_weight_rebalanced,
    )
    costly = B.run_policy(
        TradingEnvironment(frame, tickers=tickers, transaction_cost_pct=0.01),
        B.equal_weight_rebalanced,
    )

    assert costly["equity_curve"][-1] < frictionless["equity_curve"][-1]


def test_random_policy_is_deterministic_given_seed():
    frame = _feature_frame(tickers=("XOM", "CVX"))
    env_a = TradingEnvironment(frame, tickers=("XOM", "CVX"))
    env_b = TradingEnvironment(frame, tickers=("XOM", "CVX"))

    result_a = B.run_policy(env_a, B.make_random_policy(seed=42), seed=0)
    result_b = B.run_policy(env_b, B.make_random_policy(seed=42), seed=0)

    np.testing.assert_array_equal(result_a["equity_curve"], result_b["equity_curve"])


def test_random_policy_loses_to_cost_drag_relative_to_all_cash_baseline():
    """CLAUDE.md §10.4: on synthetic data the random policy loses to
    transaction costs; it must never systematically beat holding cash."""
    frame = _feature_frame(n=250, tickers=("XOM", "CVX", "SHEL"))
    tickers = ("XOM", "CVX", "SHEL")

    finals = []
    for seed in range(5):
        env = TradingEnvironment(frame, tickers=tickers, transaction_cost_pct=0.01)
        result = B.run_policy(env, B.make_random_policy(seed=seed), seed=seed)
        finals.append(result["equity_curve"][-1])

    all_cash_final = TradingEnvironment(frame, tickers=tickers).initial_cash
    assert np.mean(finals) < all_cash_final
