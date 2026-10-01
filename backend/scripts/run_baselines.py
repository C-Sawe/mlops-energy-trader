#!/usr/bin/env python3
"""Sprint 2 entry point: run the baseline trading policies (CLAUDE.md §10.1).

A Sharpe Ratio is meaningless in isolation — the same conditions that
reward a learned policy also reward simply holding the assets. Run this
before reporting any agent result, and again once Sprint 3 lands, so the
agent's numbers appear alongside these rather than in a vacuum.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --partition train --tickers XOM CVX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.processing import normalize_rolling, partition_chronological  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402
from src.orchestration.evaluator import (  # noqa: E402
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.rlops import baselines as B  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402

DETERMINISTIC_POLICIES = {
    "buy-and-hold": B.buy_and_hold,
    "equal-weight-rebalanced": B.equal_weight_rebalanced,
    "all-cash": B.all_cash,
}


def _print_row(name: str, sharpe: float, drawdown: float, cum_return: float, note: str = "") -> None:
    print(f"{name:<28}{sharpe:>10.3f}{drawdown:>10.3f}{cum_return:>12.3f}  {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline policies over a data partition")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--partition", choices=["train", "eval"], default="eval")
    parser.add_argument("--seeds", type=int, default=5, help="seeds for the random baseline")
    args = parser.parse_args()

    tickers = tuple(args.tickers) if args.tickers else DATA.tickers

    repo = MarketRepository()
    raw = repo.load_partition(DATA.train_start, DATA.eval_end, tickers=tickers)
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return 1

    # Normalise over the full continuous range before partitioning, so the
    # eval slice's rolling z-score windows can legitimately reach back into
    # the training period (still backward-looking, DR-07) instead of
    # wasting the first zscore_window-1 rows of every partition on warm-up.
    frame = normalize_rolling(raw)
    train, evaluation = partition_chronological(
        frame, DATA.train_start, DATA.train_end, DATA.eval_start, DATA.eval_end
    )
    target = train if args.partition == "train" else evaluation

    print(f"partition: {args.partition}  universe: {', '.join(tickers)}")
    print(f"{'policy':<28}{'sharpe':>10}{'max_dd':>10}{'cum_return':>12}")

    for name, policy in DETERMINISTIC_POLICIES.items():
        env = TradingEnvironment(target, tickers=tickers)
        result = B.run_policy(env, policy)
        _print_row(
            name,
            sharpe_ratio(result["returns"]),
            max_drawdown(result["equity_curve"]),
            cumulative_return(result["equity_curve"]),
        )

    finals, sharpes, drawdowns, returns = [], [], [], []
    for seed in range(args.seeds):
        env = TradingEnvironment(target, tickers=tickers)
        result = B.run_policy(env, B.make_random_policy(seed=seed), seed=seed)
        finals.append(result["equity_curve"][-1])
        sharpes.append(sharpe_ratio(result["returns"]))
        drawdowns.append(max_drawdown(result["equity_curve"]))
        returns.append(cumulative_return(result["equity_curve"]))
    _print_row(
        "random",
        float(np.mean(sharpes)),
        float(np.mean(drawdowns)),
        float(np.mean(returns)),
        note=f"(mean of {args.seeds} seeds)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
