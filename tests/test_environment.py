"""Tests for the trading MDP (FR-06, FR-07, I1, I2, I4, CLAUDE.md §7).

Traceability: I1 (no look-ahead) and I2 (t -> t+1 return timing) are the two
properties whose violation produces a backtest that cannot be reproduced
live. I4 (drawdown penalty on the increment, not the level) is what keeps
the reward from depending on episode length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataops import processing as P
from src.rlops.environment import FEATURE_COLUMNS, TradingEnvironment
from tests.test_processing import make_series


def _feature_frame(n: int = 150, tickers=("XOM", "CVX")) -> pd.DataFrame:
    equities = make_series(n, tickers=tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 28, len(dates))})
    return P.build_feature_frame(equities, vix)


def _single_ticker_frame_with_tail(tail: list[float], warmup: int = 90) -> pd.DataFrame:
    """A one-ticker frame whose *last* ``len(tail)`` closes are exactly
    ``tail``, with enough leading history for every indicator and the
    z-score window to be fully warmed up by the time the tail begins."""
    equities = make_series(warmup, tickers=("XOM",))
    last_date = equities["date"].max()
    tail_dates = pd.bdate_range(start=last_date, periods=len(tail) + 1)[1:]
    tail_df = pd.DataFrame(
        {
            "date": tail_dates,
            "ticker": "XOM",
            "open": tail,
            "high": tail,
            "low": tail,
            "close": tail,
            "volume": 1_000_000,
        }
    )
    equities = pd.concat([equities, tail_df], ignore_index=True)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.full(len(dates), 20.0)})
    return P.build_feature_frame(equities, vix)


def _land_before_tail(env: TradingEnvironment, tail_len: int) -> None:
    """Advance ``env`` with an all-cash action until the tail's first price
    is the *next* one to be traded on, without disturbing cash or shares."""
    hold = np.zeros(len(env.tickers), dtype=np.float32)
    for _ in range(env._n_steps - tail_len):
        env.step(hold)
    assert env._step_idx == env._n_steps - tail_len


# --------------------------------------------------------------- FR-06
def test_observation_and_action_space_shapes():
    tickers = ("XOM", "CVX", "SHEL")
    frame = _feature_frame(tickers=tickers)
    env = TradingEnvironment(frame, tickers=tickers)

    n = len(tickers)
    assert env.action_space.shape == (n,)
    assert env.observation_space.shape == (n * len(FEATURE_COLUMNS) + n + 1,)

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info == {}


def test_conforms_to_gymnasium_api():
    """CLAUDE.md §8: passes Stable Baselines3's own environment checker,
    the property that distinguishes this from FinRL's hmax-scaled actions."""
    from stable_baselines3.common.env_checker import check_env

    frame = _feature_frame(tickers=("XOM", "CVX"))
    env = TradingEnvironment(frame, tickers=("XOM", "CVX"))
    check_env(env, warn=True)


def test_reset_starts_fully_in_cash():
    frame = _feature_frame(tickers=("XOM",))
    env = TradingEnvironment(frame, tickers=("XOM",))
    env.reset()

    assert env.shares[0] == 0.0
    assert env.cash == pytest.approx(env.initial_cash)
    np.testing.assert_allclose(env.current_weights, [0.0], atol=1e-9)


def test_missing_normalised_columns_raise():
    frame = _feature_frame(tickers=("XOM",)).drop(columns=["close_z"])
    with pytest.raises(ValueError, match="normalised columns"):
        TradingEnvironment(frame, tickers=("XOM",))


# --------------------------------------------------------------- I1
def test_observation_contains_no_future_information():
    """A shock to the final day's price must not change day zero's
    observation — the environment-level counterpart to the DR-07 leakage
    test in test_processing.py, confirming the plumbing preserves it."""
    tickers = ("XOM",)
    equities = make_series(150, tickers=tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 28, len(dates))})
    base_frame = P.build_feature_frame(equities, vix)

    tampered_equities = equities.copy()
    last_date = tampered_equities["date"].max()
    tampered_equities.loc[tampered_equities["date"] == last_date, "close"] *= 50.0
    tampered_frame = P.build_feature_frame(tampered_equities, vix)

    base_obs, _ = TradingEnvironment(base_frame, tickers=tickers).reset()
    tampered_obs, _ = TradingEnvironment(tampered_frame, tickers=tickers).reset()

    np.testing.assert_allclose(base_obs, tampered_obs)


# --------------------------------------------------------------- I2
def test_return_is_realised_from_t_to_t_plus_one():
    """A step's reward reflects the price move from t to t+1, never the
    move that produced the observation the action was conditioned on."""
    tail = [100.0, 110.0, 99.0]
    frame = _single_ticker_frame_with_tail(tail)
    env = TradingEnvironment(
        frame, tickers=("XOM",), transaction_cost_pct=0.0, drawdown_penalty_coef=0.0
    )
    env.reset()

    np.testing.assert_allclose(env._prices[-len(tail):, 0], tail)
    _land_before_tail(env, len(tail))

    _, reward1, terminated1, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward1 == pytest.approx((110.0 - 100.0) / 100.0)
    assert terminated1 is False

    _, reward2, terminated2, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward2 == pytest.approx((99.0 - 110.0) / 110.0)
    assert terminated2 is True


# --------------------------------------------------------------- I4
def test_drawdown_penalty_applies_to_increment_not_level():
    """A drawdown that deepens is penalised; holding at the same drawdown
    level, or recovering from it, must not be penalised again."""
    tail = [100.0, 90.0, 90.0, 95.0]  # -10%, flat, partial recovery
    frame = _single_ticker_frame_with_tail(tail)
    env = TradingEnvironment(
        frame, tickers=("XOM",), transaction_cost_pct=0.0, drawdown_penalty_coef=1.0
    )
    env.reset()
    _land_before_tail(env, len(tail))

    _, reward1, _, _, _ = env.step(np.array([1.0], dtype=np.float32))  # 100 -> 90
    _, reward2, _, _, _ = env.step(np.array([1.0], dtype=np.float32))  # 90 -> 90
    _, reward3, _, _, _ = env.step(np.array([1.0], dtype=np.float32))  # 90 -> 95

    assert reward1 == pytest.approx(-0.10 - 0.10)  # -10% return, +0.10 drawdown
    assert reward2 == pytest.approx(0.0)  # flat return, drawdown unchanged
    assert reward3 == pytest.approx((95.0 - 90.0) / 90.0)  # recovering, never negative penalty


# --------------------------------------------------------------- §7 costs
def test_multi_ticker_swap_sells_in_full_and_scales_the_buy():
    """From a fully invested, zero-cash position, swapping entirely from
    one ticker to another must sell the outgoing ticker in full and only
    scale back the incoming one, ending with non-negative cash. A single
    scale factor applied uniformly to both legs would cancel out and
    execute nothing at all, since the sell and buy notionals are equal
    and opposite."""
    tickers = ("XOM", "CVX")
    frame = _feature_frame(tickers=tickers)
    env = TradingEnvironment(frame, tickers=tickers, transaction_cost_pct=0.01)
    env.reset()

    env.transaction_cost_pct = 0.0
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    env.transaction_cost_pct = 0.01

    assert env.current_weights[0] == pytest.approx(1.0, abs=1e-6)

    price_cvx = env._prices[env._step_idx, 1]
    equity = env._equity(env._prices[env._step_idx])
    naive_shares_cvx = equity / price_cvx  # what a frictionless full buy would get

    env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert env.cash >= 0.0
    assert env.shares[0] == pytest.approx(0.0, abs=1e-6)  # XOM sold in full
    # CVX was scaled back by the fee, not bought in full and not rejected.
    # Weight alone cannot show this: once cash is a negligible residual,
    # the weight reads ~1.0 whether or not the buy was actually scaled.
    assert 0.0 < env.shares[1] < naive_shares_cvx


def test_buy_from_cash_is_scaled_back_by_fee_not_rejected():
    tc = 0.01
    frame = _feature_frame(tickers=("XOM",))
    env = TradingEnvironment(frame, tickers=("XOM",), transaction_cost_pct=tc)
    env.reset()
    naive_shares = env.initial_cash / env._prices[0, 0]

    env.step(np.array([1.0], dtype=np.float32))

    assert env.cash >= 0.0
    assert 0.0 < env.shares[0] < naive_shares  # scaled back by the fee, not rejected


def test_dust_deadband_skips_negligible_trade():
    frame = _feature_frame(tickers=("XOM",))
    env = TradingEnvironment(frame, tickers=("XOM",), transaction_cost_pct=0.0)
    env.reset()
    env.step(np.array([0.3], dtype=np.float32))
    shares_before = env.shares.copy()

    target = env.current_weights.astype(np.float64) + 1e-7  # far below the 1e-6 threshold
    env.step(target.astype(np.float32))

    np.testing.assert_array_equal(env.shares, shares_before)


def test_cash_never_goes_negative_under_random_actions():
    tickers = ("XOM", "CVX", "SHEL")
    frame = _feature_frame(n=200, tickers=tickers)
    env = TradingEnvironment(frame, tickers=tickers, transaction_cost_pct=0.001)
    env.reset(seed=0)
    rng = np.random.default_rng(1)

    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.uniform(-1.0, 1.0, size=len(tickers)).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        assert env.cash >= -1e-6
