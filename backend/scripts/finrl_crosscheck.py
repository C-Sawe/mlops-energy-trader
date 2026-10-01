#!/usr/bin/env python3
"""CLAUDE.md §8's pending validation experiment.

Cross-checks this project's `TradingEnvironment` accounting against FinRL's
`StockTradingEnv` — a published, independently-implemented Gym environment —
by holding an identical fixed share count of each ticker in both, with
transaction costs and the drawdown penalty disabled in both, so each
reduces to plain PnL on an identical holding. Closely matching equity
curves corroborate this project's own accounting; they do not validate
FinRL's, which is the trusted reference here.

Both environments buy the same integer share count of each ticker on the
*same calendar date*, not each one's own unrestricted day zero: this
project's z-score warm-up (DR-07, zscore_window=60) means the environment's
actual episode starts ~79 trading days after the raw data begins, and FinRL
has no such warm-up. Comparing from two different start dates would confound
"does the accounting match" with "did the two environments start holding on
different days at different prices" — so FinRL is run only over the same
date range `TradingEnvironment` actually uses.

Requires `finrl`, `stable-baselines3`, `gymnasium`, `matplotlib`, and
`stockstats` — NOT part of this project's main requirements.txt (see the
commented-out `# finrl` line there), and not safe to pin there without
risking conflicts with Sprint 1/2's numpy/pandas/gymnasium versions. Install
into a *separate* virtualenv and run this script with that interpreter:

    python3 -m venv /tmp/venv-finrl
    source /tmp/venv-finrl/bin/activate
    pip install finrl gymnasium stable-baselines3 matplotlib stockstats
    /tmp/venv-finrl/bin/python scripts/finrl_crosscheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataops.processing import build_feature_frame  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402

SHARES_PER_TICKER = 100
INITIAL_CASH = 100_000.0


def _synthetic_equities(n_days: int, tickers: tuple[str, ...], seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV, shared by both environments so any
    divergence in results comes from the environments, not the data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for i, ticker in enumerate(tickers):
        close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n_days)) + i * 10
        close = np.maximum(close, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1_000_000, 20_000_000, n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _run_ours(equities: pd.DataFrame, tickers: tuple[str, ...]) -> tuple[np.ndarray, pd.DatetimeIndex]:
    dates_all = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates_all, "close": np.full(len(dates_all), 20.0)})
    frame = build_feature_frame(equities, vix)

    env = TradingEnvironment(
        frame, tickers=tickers, transaction_cost_pct=0.0, drawdown_penalty_coef=0.0
    )
    env.reset()
    aligned_dates = pd.DatetimeIndex(env._dates)

    price0 = env._prices[0]
    target_weights = (SHARES_PER_TICKER * price0 / env.initial_cash).astype(np.float32)

    equity_curve = [env.initial_cash]
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = target_weights if step == 0 else env.current_weights
        _obs, _reward, terminated, truncated, info = env.step(action)
        equity_curve.append(info["equity"])
        step += 1

    return np.asarray(equity_curve, dtype=np.float64), aligned_dates


def _run_finrl(equities: pd.DataFrame, tickers: tuple[str, ...], aligned_dates: pd.DatetimeIndex) -> np.ndarray:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    df = equities[equities["date"].isin(aligned_dates)].rename(columns={"ticker": "tic"}).copy()
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df.index = df["date"].factorize()[0]

    # StockTradingEnv._buy_stock/_sell_stock hard-code a "disabled" flag at
    # state[index + 2*stock_dim + 1] — i.e. inside the first technical
    # indicator's slot — regardless of whether tech_indicator_list is used
    # for anything else. An empty list leaves that slot out of the state
    # vector entirely and every buy/sell raises IndexError. A single
    # always-False column supplies the slot without affecting trading.
    df["disable"] = False

    n = len(tickers)
    hmax = SHARES_PER_TICKER  # exactly the buy size we want; no fractional scaling

    env = StockTradingEnv(
        df=df,
        stock_dim=n,
        hmax=hmax,
        initial_amount=INITIAL_CASH,
        num_stock_shares=[0] * n,
        buy_cost_pct=[0.0] * n,
        sell_cost_pct=[0.0] * n,
        reward_scaling=1.0,
        state_space=1 + 3 * n,  # cash + n prices + n share counts + n "disable" flags
        action_space=n,
        tech_indicator_list=["disable"],
    )
    env.reset()

    action_buy_max = np.ones(n, dtype=np.float64)  # action * hmax == hmax shares, on day 0
    hold = np.zeros(n, dtype=np.float64)

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = action_buy_max if step == 0 else hold
        _state, _reward, terminated, truncated, _info = env.step(action)
        step += 1

    return np.asarray(env.asset_memory, dtype=np.float64)


def main() -> int:
    tickers = ("XOM", "CVX", "SHEL", "BP", "NEE")
    equities = _synthetic_equities(300, tickers, seed=0)

    ours, aligned_dates = _run_ours(equities, tickers)
    theirs = _run_finrl(equities, tickers, aligned_dates)

    if len(ours) != len(theirs):
        print(f"WARNING: length mismatch — ours={len(ours)} finrl={len(theirs)}")
    n = min(len(ours), len(theirs))
    ours, theirs = ours[:n], theirs[:n]

    abs_diff = np.abs(ours - theirs)

    print(f"aligned episode length: {n} days, from {aligned_dates[0].date()} to {aligned_dates[n - 1].date()}")
    print(f"ours  initial/final equity: {ours[0]:.4f} / {ours[-1]:.4f}")
    print(f"finrl initial/final equity: {theirs[0]:.4f} / {theirs[-1]:.4f}")
    print(f"max abs diff:   {abs_diff.max():.8f}")
    print(f"max rel diff:   {(abs_diff / theirs).max():.10f}")
    print(f"correlation:    {np.corrcoef(ours, theirs)[0, 1]:.12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
