#!/usr/bin/env python3
"""CLAUDE.md §12 item 2: benchmark PPO training on CPU vs MPS.

CLAUDE.md §9 already reasons about why this might go either way: PPO
alternates rollout collection (many forward passes on a single observation)
with small minibatch gradient updates, and at ~14.7K parameters the
CPU<->GPU transfer overhead can dominate the arithmetic MPS would otherwise
accelerate. This script measures it rather than assuming it, on identical
hyperparameters, an identical seed, and the real TradingEnvironment shape
(FR-06) rather than a toy environment.

Uses synthetic deterministic data (no network) since only the environment's
shape and step cost matter for a throughput benchmark, not the data content.

Usage:
    python scripts/benchmark_device.py
    python scripts/benchmark_device.py --timesteps 50000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.processing import build_feature_frame  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402


def _synthetic_feature_frame(n_days: int, tickers: tuple[str, ...], seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV, enriched exactly like real data would
    be — same shape as `tests/test_processing.make_series` but kept local so
    this script has no dependency on the test suite."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
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
    equities = pd.concat(frames, ignore_index=True)
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 28, n_days)})
    return build_feature_frame(equities, vix)


def _run(
    device: str, frame: pd.DataFrame, tickers: tuple[str, ...], timesteps: int, seed: int, warmup: int
) -> float:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return TradingEnvironment(frame, tickers=tickers)

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, device=device, seed=seed, verbose=0)

    # Untimed warm-up: MPS in particular pays a one-time kernel-compilation
    # and device-transfer setup cost on first use that would otherwise be
    # charged to the measurement rather than to steady-state throughput.
    if warmup:
        model.learn(total_timesteps=warmup)

    start = time.perf_counter()
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    elapsed = time.perf_counter() - start

    env.close()
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PPO learn() on CPU vs MPS")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=2048, help="untimed steps before the measured run")
    parser.add_argument("--days", type=int, default=2000, help="synthetic trading days to generate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tickers = DATA.tickers
    frame = _synthetic_feature_frame(args.days, tickers, seed=args.seed)

    print(f"universe: {', '.join(tickers)}  timesteps: {args.timesteps}  seed: {args.seed}")
    print(f"{'device':<10}{'seconds':>12}{'steps/sec':>14}")

    for device in ("cpu", "mps"):
        try:
            elapsed = _run(device, frame, tickers, args.timesteps, args.seed, args.warmup)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"{device:<10}  unavailable: {exc}")
            continue
        rate = args.timesteps / elapsed
        print(f"{device:<10}{elapsed:>12.2f}{rate:>14.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
