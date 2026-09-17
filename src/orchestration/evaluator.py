"""Performance metrics for FR-13 and the evaluation protocol (CLAUDE.md §10).

These are read by the CT orchestrator (Sprint 4, FR-14) to decide whether
the incumbent model has decayed, so a metric that silently misbehaves on a
degenerate input is not a cosmetic bug: it is a broken retraining trigger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# CLAUDE.md §7: a constant (or near-constant) return series has a standard
# deviation of order 1e-18 due to float rounding, not exactly zero. Without
# a floor, that noise denominator makes the Sharpe ratio blow up to an
# enormous but finite number (order 1e16) that clears any sensible
# target_sharpe_threshold and so silently suppresses FR-14's retrain
# trigger precisely when the strategy has stopped doing anything.
_VARIANCE_FLOOR = 1e-12

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """The (annualised) Sharpe ratio of a per-period return series."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return 0.0
    mean = returns.mean()
    variance = max(returns.var(ddof=0), _VARIANCE_FLOOR)
    return float(mean / np.sqrt(variance) * np.sqrt(periods_per_year))


def rolling_sharpe(
    returns: pd.Series, window: int, periods_per_year: int = 252
) -> pd.Series:
    """FR-13: the rolling Sharpe ratio the CT orchestrator thresholds on.

    Warm-up rows (fewer than ``window`` observations) stay NaN rather than
    reporting a Sharpe of zero — a zero would be indistinguishable from a
    genuinely bad but fully-observed window, and would spuriously trigger
    FR-14 before there is enough history to justify it.
    """
    mean = returns.rolling(window, min_periods=window).mean()
    variance = returns.rolling(window, min_periods=window).var(ddof=0)
    floored = variance.clip(lower=_VARIANCE_FLOOR)
    sharpe = mean / np.sqrt(floored) * np.sqrt(periods_per_year)
    return sharpe.where(variance.notna())


def max_drawdown(equity_curve: np.ndarray) -> float:
    """The largest peak-to-trough decline, as a fraction of the peak."""
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if equity_curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return float(drawdown.max())


def cumulative_return(equity_curve: np.ndarray) -> float:
    """Total return over the curve, from its first to its last value."""
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if equity_curve.size < 2:
        return 0.0
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 1,
) -> float:
    """Bailey & Lopez de Prado (2014), already cited in Chapter 2.

    The probability that the observed (per-period) Sharpe ratio is still
    positive once corrected for having tried ``n_trials`` configurations —
    including abandoned ones (CLAUDE.md §10.3). The best of many backtests
    shows a positive Sharpe by construction; this quantifies how much of
    the observed figure that alone explains.

    ``returns`` should be the per-period series the reported Sharpe ratio
    was computed from; ``periods_per_year`` is left at 1 (non-annualised)
    to match the formula's derivation, unless the caller has a specific
    reason to annualise consistently on both sides.
    """
    returns = np.asarray(returns, dtype=np.float64)
    t = returns.size
    if t < 4:
        raise ValueError("need at least 4 return observations")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    sr_hat = sharpe_ratio(returns, periods_per_year=periods_per_year)
    skew = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))

    denom = np.sqrt(
        max(1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2, _VARIANCE_FLOOR)
    )

    if n_trials > 1:
        sharpe_variance = max(denom**2 / (t - 1), _VARIANCE_FLOOR)
        sr_0 = np.sqrt(sharpe_variance) * (
            (1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    else:
        sr_0 = 0.0

    z = (sr_hat - sr_0) * np.sqrt(t - 1) / denom
    return float(stats.norm.cdf(z))
