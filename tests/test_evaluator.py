"""Tests for the performance metrics (FR-13, CLAUDE.md §7, §10.3).

These are what the CT orchestrator (Sprint 4, FR-14) thresholds on, so a
metric that misbehaves on a degenerate input is a broken retraining trigger,
not a cosmetic bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.orchestration import evaluator as E


# --------------------------------------------------------------- sharpe_ratio
def test_sharpe_ratio_matches_manual_calculation():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.001, 0.02, 500)

    expected = returns.mean() / returns.std(ddof=0) * np.sqrt(252)
    assert E.sharpe_ratio(returns) == pytest.approx(expected, rel=1e-9)


def test_sharpe_ratio_of_constant_series_is_bounded_not_exploding():
    """CLAUDE.md §7: without the variance floor, this returns ~8.9e16 —
    finite, but large enough to always clear target_sharpe_threshold and
    silently suppress the FR-14 retrain trigger. The floor must keep it
    from blowing up regardless of the exact float noise in std()."""
    value = E.sharpe_ratio(np.full(20, 0.01))
    assert np.isfinite(value)
    assert abs(value) < 1e6


def test_sharpe_ratio_of_too_few_observations_is_zero():
    assert E.sharpe_ratio(np.array([])) == 0.0
    assert E.sharpe_ratio(np.array([0.01])) == 0.0


# --------------------------------------------------------------- rolling_sharpe
def test_rolling_sharpe_warmup_rows_are_nan_not_zero():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.001, 0.02, 40))
    window = 10

    rolling = E.rolling_sharpe(returns, window=window)

    assert rolling.iloc[: window - 1].isna().all()
    assert rolling.iloc[window - 1 :].notna().all()

    spot = returns.iloc[:window]
    expected = spot.mean() / np.sqrt(spot.var(ddof=0)) * np.sqrt(252)
    assert rolling.iloc[window - 1] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------- max_drawdown
def test_max_drawdown_matches_manual_calculation():
    equity_curve = np.array([100.0, 110.0, 90.0, 95.0, 80.0, 120.0])
    # Worst trough is 80 against the running peak of 110: (110-80)/110.
    expected = (110.0 - 80.0) / 110.0
    assert E.max_drawdown(equity_curve) == pytest.approx(expected)


def test_max_drawdown_of_monotonically_rising_curve_is_zero():
    assert E.max_drawdown(np.array([100.0, 105.0, 110.0])) == pytest.approx(0.0)


def test_max_drawdown_of_empty_curve_is_zero():
    assert E.max_drawdown(np.array([])) == 0.0


# --------------------------------------------------------------- cumulative_return
def test_cumulative_return_matches_manual_calculation():
    assert E.cumulative_return(np.array([100.0, 150.0])) == pytest.approx(0.5)


def test_cumulative_return_of_single_point_is_zero():
    assert E.cumulative_return(np.array([100.0])) == 0.0


# --------------------------------------------------------------- deflated_sharpe_ratio
def test_deflated_sharpe_ratio_is_a_probability():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.002, 0.01, 300)

    dsr = E.deflated_sharpe_ratio(returns, n_trials=10)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_decreases_as_trials_increase():
    """The best of many backtests shows a positive Sharpe by construction;
    correcting for more trials must make that look less significant, never
    more, holding the observed returns fixed."""
    rng = np.random.default_rng(13)
    returns = rng.normal(0.002, 0.01, 300)

    dsr_few = E.deflated_sharpe_ratio(returns, n_trials=1)
    dsr_many = E.deflated_sharpe_ratio(returns, n_trials=200)

    assert dsr_many < dsr_few


def test_deflated_sharpe_ratio_requires_minimum_observations():
    with pytest.raises(ValueError, match="observations"):
        E.deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=5)


def test_deflated_sharpe_ratio_requires_at_least_one_trial():
    with pytest.raises(ValueError, match="n_trials"):
        E.deflated_sharpe_ratio(np.full(50, 0.01), n_trials=0)
