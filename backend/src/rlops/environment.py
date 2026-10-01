"""The trading MDP (FR-06, FR-07, FR-11).

A Gymnasium-compatible environment over the enriched feature frame produced
by ``src.dataops.processing.build_feature_frame``. This is the deviation
recorded in CLAUDE.md §8: FinRL's ``StockTradingEnv`` was evaluated and does
import successfully, but its reward is assigned inline with no override hook
and its actions are hmax-scaled share counts rather than continuous [-1, 1]
target weights. This module implements the specification directly instead.

Action space: one continuous target weight per ticker in [-1, 1], where the
weight is the fraction of current portfolio equity the agent wants held in
that ticker. Rebalancing toward the target happens at the close of the
current step (I2); the resulting position then realises its return over the
following step, from t to t+1.

Observation space: for each ticker, the eight backward-looking normalised
features from ``processing.normalize_rolling`` (open, high, low, close,
volume, sma_20, rsi_14, vix, all "_z"), plus the agent's own current
position weight per ticker, plus its current cash weight. For the five-
ticker universe this is 5*8 + 5 + 1 = 46 dimensions (CLAUDE.md §9). Including
the agent's own holdings is what makes the state Markovian: two identical
price histories with different current allocations require different
actions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from src.config import DATA, ENVIRONMENT

# CLAUDE.md §7: float32 actions round-trip a "hold" target weight with
# ~2e-7 relative error. Below this fraction of portfolio value, a nonzero
# delta is treated as noise rather than a trade, so a buy-and-hold baseline
# does not appear to rebalance every day.
_DUST_FRACTION = 1e-6

# Order matters: this is the same column order normalize_rolling uses, and
# the observation layout depends on it staying fixed.
FEATURE_COLUMNS = ("open", "high", "low", "close", "volume", "sma_20", "rsi_14", "vix")


class TradingEnvironment(Env):
    """FR-06: a Gym-compatible MDP over the configured equity universe."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        tickers: tuple[str, ...] = DATA.tickers,
        initial_cash: float = ENVIRONMENT.initial_cash,
        transaction_cost_pct: float = ENVIRONMENT.transaction_cost_pct,
        drawdown_penalty_coef: float = ENVIRONMENT.drawdown_penalty_coef,
    ) -> None:
        super().__init__()
        self.tickers = tuple(tickers)
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)

        self._features, self._prices, self._dates = self._prepare(feature_frame)
        self._n_steps = len(self._dates)

        n = len(self.tickers)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        obs_dim = n * len(FEATURE_COLUMNS) + n + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_idx = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(n, dtype=np.float64)
        self._peak_equity = self.initial_cash
        self._max_drawdown = 0.0

    def _prepare(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Pivot the feature frame to arrays aligned across all tickers.

        Only dates on which every ticker has a complete normalised feature
        vector are usable — the zscore warm-up window (DR-07) leaves the
        first ``zscore_window - 1`` rows NaN, and a ticker that failed to
        ingest on a given day cannot silently contribute a partial state.
        """
        z_cols = [f"{c}_z" for c in FEATURE_COLUMNS]
        missing = set(z_cols) - set(frame.columns)
        if missing:
            raise ValueError(f"feature_frame missing normalised columns: {sorted(missing)}")

        per_ticker: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            sub = frame[frame["ticker"] == ticker].sort_values("date").set_index("date")
            sub = sub.dropna(subset=z_cols + ["close"])
            if sub.empty:
                raise ValueError(f"no usable rows for ticker {ticker!r}")
            per_ticker[ticker] = sub

        common_dates = per_ticker[self.tickers[0]].index
        for ticker in self.tickers[1:]:
            common_dates = common_dates.intersection(per_ticker[ticker].index)
        common_dates = common_dates.sort_values()

        if len(common_dates) < 2:
            raise ValueError("not enough aligned observations to build an episode")

        features = [per_ticker[t].loc[common_dates, z_cols].to_numpy(dtype=np.float64) for t in self.tickers]
        prices = [per_ticker[t].loc[common_dates, "close"].to_numpy(dtype=np.float64) for t in self.tickers]

        return (
            np.concatenate(features, axis=1),
            np.stack(prices, axis=1),
            common_dates,
        )

    @property
    def current_weights(self) -> np.ndarray:
        """Each ticker's current position value as a fraction of equity."""
        prices = self._prices[self._step_idx]
        equity = self._equity(prices)
        return (self.shares * prices / equity).astype(np.float32)

    def _equity(self, prices: np.ndarray) -> float:
        equity = self.cash + float(np.dot(self.shares, prices))
        # A ruined portfolio (equity <= 0) cannot be divided into weights;
        # floor it rather than propagate inf/nan into the observation.
        return equity if equity > 0 else 1e-8

    def _observation(self) -> np.ndarray:
        prices = self._prices[self._step_idx]
        equity = self._equity(prices)
        position_weights = self.shares * prices / equity
        cash_weight = self.cash / equity
        return np.concatenate(
            [self._features[self._step_idx], position_weights, [cash_weight]]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_idx = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(len(self.tickers), dtype=np.float64)
        self._peak_equity = self.initial_cash
        self._max_drawdown = 0.0
        return self._observation(), {}

    def _rebalance(self, target_weights: np.ndarray) -> None:
        """Trade toward ``target_weights``, scaling down if unaffordable.

        CLAUDE.md §7: a fully invested portfolio holds no cash, so any
        rebalance leaves its transaction fee unfunded unless the agent
        sells marginally more than the naive target. Concretely: selling
        always funds itself (a sell's proceeds exceed its own fee for any
        transaction_cost_pct < 100%), so sell legs execute in full. Buy
        legs draw from cash-on-hand plus those sell proceeds, and are
        scaled back together if that is not enough to cover both their
        principal and their fee. A single scale factor applied to *every*
        leg (buys and sells alike) does not work here: in a pure swap
        between two tickers the sell and buy notionals are equal and
        opposite, so they cancel regardless of the scale factor and cash
        after the trade is negative for any nonzero scale — bisection
        would converge on executing nothing at all. Scaling only the buy
        legs is what makes "sell marginally more to cover the fee" true.

        Cash after the trade is monotone decreasing in how much of the
        buy leg is executed, so bisection finds the largest affordable
        fraction. Unaffordable buys are scaled down, never rejected
        outright — a hard rejection makes the action space discontinuous
        and gives the policy an uninformative gradient right at the
        affordability edge.
        """
        prices = self._prices[self._step_idx]
        equity = self._equity(prices)

        target_shares = target_weights * equity / prices
        delta_shares = target_shares - self.shares

        dust_threshold = _DUST_FRACTION * equity
        delta_notional = delta_shares * prices
        delta_shares = np.where(np.abs(delta_notional) < dust_threshold, 0.0, delta_shares)

        if not np.any(delta_shares):
            return

        sell_mask = delta_shares < 0.0
        buy_mask = delta_shares > 0.0

        sell_notional = -float(np.sum(delta_shares[sell_mask] * prices[sell_mask]))
        sell_fee = self.transaction_cost_pct * sell_notional
        available_cash = self.cash + sell_notional - sell_fee

        buy_notional_full = float(np.sum(delta_shares[buy_mask] * prices[buy_mask]))

        def cash_after(beta: float) -> float:
            buy_notional = buy_notional_full * beta
            buy_fee = self.transaction_cost_pct * buy_notional
            return available_cash - buy_notional - buy_fee

        if buy_notional_full == 0.0 or cash_after(1.0) >= 0.0:
            beta = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if cash_after(mid) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            beta = lo

        executed = np.where(buy_mask, delta_shares * beta, delta_shares)
        fee = self.transaction_cost_pct * float(np.sum(np.abs(executed) * prices))
        self.cash = max(self.cash - float(np.sum(executed * prices)) - fee, 0.0)
        self.shares = self.shares + executed

    def _reward(self, step_return: float, equity_after: float) -> float:
        """FR-07 / I4: penalise the increment in max drawdown, not its level.

        Penalising the level would bill the agent repeatedly for one
        historical loss on every subsequent step, making the reward depend
        on episode length rather than on behaviour.
        """
        self._peak_equity = max(self._peak_equity, equity_after)
        drawdown = (self._peak_equity - equity_after) / self._peak_equity
        increment = max(0.0, drawdown - self._max_drawdown)
        self._max_drawdown = max(self._max_drawdown, drawdown)
        return step_return - self.drawdown_penalty_coef * increment

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        prices_t = self._prices[self._step_idx]
        equity_before = self._equity(prices_t)

        self._rebalance(action)

        # I2: the action executed at t's close realises its return from
        # t to t+1 — advance the clock only after the trade is booked.
        self._step_idx += 1
        prices_t1 = self._prices[self._step_idx]
        equity_after = self._equity(prices_t1)

        step_return = (equity_after - equity_before) / equity_before
        reward = self._reward(step_return, equity_after)

        terminated = self._step_idx >= self._n_steps - 1
        truncated = False
        info = {"equity": equity_after, "cash": self.cash, "date": self._dates[self._step_idx]}
        return self._observation(), reward, terminated, truncated, info
