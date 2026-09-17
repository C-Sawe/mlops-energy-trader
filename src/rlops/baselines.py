"""Baseline trading policies (CLAUDE.md §10.1).

A Sharpe Ratio is meaningless in isolation: the same conditions that reward
a learned policy also reward simply holding the assets. These baselines run
through the same ``TradingEnvironment`` accounting — same transaction costs,
same bisection scaling, same dust deadband — as any learned agent, so the
comparison is apples-to-apples rather than a hand-rolled approximation.

Each policy is a callable ``(env, step) -> action`` rather than a fixed
array, because "hold the current allocation" is not a fixed target: it is
whatever the environment's own ``current_weights`` currently are, since
prices move between rebalances.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from src.rlops.environment import TradingEnvironment


class Policy(Protocol):
    def __call__(self, env: TradingEnvironment, step: int) -> np.ndarray: ...


def buy_and_hold(env: TradingEnvironment, step: int) -> np.ndarray:
    """Allocate equal weight once, then never rebalance again.

    After the first step, the target action is the position weights the
    environment already reports, so the requested delta is zero and the
    only drift is from price movement itself.
    """
    n = len(env.tickers)
    if step == 0:
        return np.full(n, 1.0 / n, dtype=np.float32)
    return env.current_weights


def equal_weight_rebalanced(env: TradingEnvironment, step: int) -> np.ndarray:
    """Target equal weight every step, forcing daily rebalancing."""
    n = len(env.tickers)
    return np.full(n, 1.0 / n, dtype=np.float32)


def all_cash(env: TradingEnvironment, step: int) -> np.ndarray:
    """Never hold a position; the capital-preservation baseline."""
    return np.zeros(len(env.tickers), dtype=np.float32)


def make_random_policy(seed: int) -> Policy:
    """A uniformly random target weight per ticker at every step.

    Its own RNG, independent of the environment's, so the same seed always
    produces the same sequence of actions regardless of what else in the
    episode consumes randomness.
    """
    rng = np.random.default_rng(seed)

    def _policy(env: TradingEnvironment, step: int) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=len(env.tickers)).astype(np.float32)

    return _policy


def run_policy(
    env: TradingEnvironment,
    policy: Policy,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Execute ``policy`` for one full episode and report its trajectory."""
    obs, _info = env.reset(seed=seed)
    equity = [env.initial_cash]
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        action = policy(env, step)
        obs, _reward, terminated, truncated, info = env.step(action)
        equity.append(info["equity"])
        step += 1

    equity_curve = np.asarray(equity, dtype=np.float64)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    return {"equity_curve": equity_curve, "returns": returns}
