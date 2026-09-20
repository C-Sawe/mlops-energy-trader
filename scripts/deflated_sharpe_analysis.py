#!/usr/bin/env python3
"""Statistical-rigor pass for §10's evaluation protocol: deflated Sharpe on
every walk-forward split (not just the single best fold), with more seeds
per split than the original sweep.

This is a validation/analysis experiment, the same kind as
`scripts/finrl_crosscheck.py` and `scripts/benchmark_device.py` — it does
not touch `ModelRegistry` or the production `model_run`/`model_version`
tables. `scripts/train_agent.py`'s sweep never persisted per-period returns
anywhere (only the summary Sharpe scalar), and `deflated_sharpe_ratio()`
needs the actual return series to estimate skew/kurtosis (Bailey & Lopez de
Prado 2014) — so reproducing every split's winning run here, in memory, is
the only way to get DSR per split without re-architecting the production
logging path for a one-off report.

Same hyperparameters as `scripts/train_agent.py`'s defaults (n_steps=2048,
timesteps=20000) — only the seed count changes, per the request to tighten
the confidence interval, not to also change what's being measured.

Usage:
    python scripts/deflated_sharpe_analysis.py
    python scripts/deflated_sharpe_analysis.py --seeds 10 --out /tmp/dsr.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.processing import (  # noqa: E402
    normalize_rolling,
    partition_chronological,
    walk_forward_splits,
)
from src.dataops.repository import MarketRepository  # noqa: E402
from src.orchestration.evaluator import deflated_sharpe_ratio, sharpe_ratio  # noqa: E402
from src.rlops.agent import PPOAgent  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Deflated Sharpe per walk-forward split, more seeds")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--eval-days", type=int, default=180)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--out", type=str, default=None, help="write the full per-split JSON report here")
    args = parser.parse_args()

    tickers = tuple(args.tickers) if args.tickers else DATA.tickers

    repo = MarketRepository()
    raw = repo.load_partition(DATA.train_start, DATA.eval_end, tickers=tickers)
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return 1

    frame = normalize_rolling(raw)
    splits = walk_forward_splits(
        DATA.train_start, DATA.eval_end, train_days=args.train_days, eval_days=args.eval_days
    )
    if not splits:
        print("configured date range is too short for these window sizes", file=sys.stderr)
        return 1

    n_trials = len(splits) * args.seeds
    print(
        f"universe: {', '.join(tickers)}  splits: {len(splits)}  seeds: {args.seeds}  "
        f"n_trials (for DSR): {n_trials}"
    )

    started = time.time()
    per_split = []
    all_seed_sharpes: list[float] = []
    best_overall = {"sharpe": -np.inf}

    for i, split in enumerate(splits):
        train_df, eval_df = partition_chronological(
            frame, split["train_start"], split["train_end"], split["eval_start"], split["eval_end"]
        )
        train_env = TradingEnvironment(train_df, tickers=tickers)
        eval_env = TradingEnvironment(eval_df, tickers=tickers)

        seed_sharpes, seed_returns = [], []
        for seed in range(args.seeds):
            agent = PPOAgent(train_env, n_steps=args.n_steps, seed=seed)
            agent.train(total_timesteps=args.timesteps)
            [result] = agent.evaluate(eval_env, n_episodes=1, seed=seed)
            sr = sharpe_ratio(result["returns"])
            seed_sharpes.append(sr)
            seed_returns.append(result["returns"])
            all_seed_sharpes.append(sr)

        best_idx = int(np.argmax(seed_sharpes))
        best_sharpe = seed_sharpes[best_idx]
        best_returns = seed_returns[best_idx]
        dsr = deflated_sharpe_ratio(best_returns, n_trials=n_trials)

        record = {
            "split": i + 1,
            "train_start": str(split["train_start"].date()),
            "train_end": str(split["train_end"].date()),
            "eval_start": str(split["eval_start"].date()),
            "eval_end": str(split["eval_end"].date()),
            "seed_sharpes": seed_sharpes,
            "best_seed": best_idx,
            "best_sharpe": best_sharpe,
            "deflated_sharpe_ratio": dsr,
        }
        per_split.append(record)
        if best_sharpe > best_overall["sharpe"]:
            best_overall = {**record, "sharpe": best_sharpe, "returns": list(map(float, best_returns))}

        elapsed = time.time() - started
        print(
            f"split {i + 1}/{len(splits)}: best_sharpe={best_sharpe:.3f} "
            f"(seed {best_idx})  DSR={dsr:.3f}  [{elapsed:.0f}s elapsed]"
        )

    best_per_split = np.array([r["best_sharpe"] for r in per_split])
    all_seeds = np.array(all_seed_sharpes)
    dsrs = np.array([r["deflated_sharpe_ratio"] for r in per_split])
    positive_splits = int((best_per_split > 0).sum())

    print("\n" + "=" * 60)
    print(f"Best-seed-per-split Sharpe, n={len(per_split)}: mean={best_per_split.mean():.3f} std={best_per_split.std(ddof=1):.3f}")
    print(f"All individual seeds pooled, n={all_seeds.size}: mean={all_seeds.mean():.3f} std={all_seeds.std(ddof=1):.3f}")
    print(f"Positive splits: {positive_splits}/{len(per_split)}")
    print(f"Deflated Sharpe per split: mean={dsrs.mean():.3f} std={dsrs.std(ddof=1):.3f} min={dsrs.min():.3f} max={dsrs.max():.3f}")
    print(f"Splits with DSR > 0.5: {int((dsrs > 0.5).sum())}/{len(per_split)}")
    print(f"Best single result overall: split {best_overall['split']} sharpe={best_overall['sharpe']:.3f} DSR={best_overall['deflated_sharpe_ratio']:.3f}")
    print(f"Total wall time: {time.time() - started:.0f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n_trials": n_trials,
                    "seeds": args.seeds,
                    "per_split": per_split,
                    "summary": {
                        "best_per_split_mean": float(best_per_split.mean()),
                        "best_per_split_std": float(best_per_split.std(ddof=1)),
                        "all_seeds_mean": float(all_seeds.mean()),
                        "all_seeds_std": float(all_seeds.std(ddof=1)),
                        "positive_splits": positive_splits,
                        "total_splits": len(per_split),
                        "dsr_mean": float(dsrs.mean()),
                        "dsr_std": float(dsrs.std(ddof=1)),
                        "dsr_min": float(dsrs.min()),
                        "dsr_max": float(dsrs.max()),
                        "splits_dsr_gt_half": int((dsrs > 0.5).sum()),
                    },
                    "best_overall": best_overall,
                },
                f,
                indent=2,
            )
        print(f"\nfull report written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
