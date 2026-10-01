#!/usr/bin/env python3
"""Sprint 3 entry point: walk-forward PPO training with a seed sweep.

For every walk-forward split (not a single train/eval split — CLAUDE.md
§10, §13), trains a fresh PPOAgent per seed, evaluates it on that split's
held-out partition, logs every run to MLflow and the `model_run` table with
its exact partition boundaries (FR-08, DR-08), and reports the mean ± std
across seeds (§10.2: a single-seed number is not a result). The
best-performing seed in each split is registered as a model_version
(FR-09) — "the selected artifact", not automatically every run; selecting
which version gets *promoted* to serving is Sprint 4's job (FR-17), not
this script's.

Usage:
    python scripts/train_agent.py
    python scripts/train_agent.py --timesteps 50000 --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.models import ModelRun  # noqa: E402
from src.dataops.processing import (  # noqa: E402
    normalize_rolling,
    partition_chronological,
    walk_forward_splits,
)
from src.dataops.repository import MarketRepository  # noqa: E402
from src.orchestration.evaluator import cumulative_return, max_drawdown, sharpe_ratio  # noqa: E402
from src.rlops.agent import PPOAgent  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402
from src.rlops.registry import ModelRegistry  # noqa: E402


def _metrics_for(result: dict) -> dict[str, float]:
    return {
        "sharpe_ratio": sharpe_ratio(result["returns"]),
        "max_drawdown": max_drawdown(result["equity_curve"]),
        "cumulative_return": cumulative_return(result["equity_curve"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward PPO training with a seed sweep")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--eval-days", type=int, default=180)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=2048)
    args = parser.parse_args()

    tickers = tuple(args.tickers) if args.tickers else DATA.tickers

    repo = MarketRepository()
    raw = repo.load_partition(DATA.train_start, DATA.eval_end, tickers=tickers)
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return 1

    # Normalise over the full range before splitting (see the note in
    # run_baselines.py / CLAUDE.md §7): still purely backward-looking
    # (DR-07), and avoids wasting the first zscore_window-1 rows of every
    # split on warm-up NaNs.
    frame = normalize_rolling(raw)
    splits = walk_forward_splits(
        DATA.train_start, DATA.eval_end, train_days=args.train_days, eval_days=args.eval_days
    )
    if not splits:
        print("configured date range is too short for these window sizes", file=sys.stderr)
        return 1

    print(f"universe: {', '.join(tickers)}  splits: {len(splits)}  seeds: {args.seeds}")
    registry = ModelRegistry()

    for i, split in enumerate(splits):
        train_df, eval_df = partition_chronological(
            frame,
            split["train_start"],
            split["train_end"],
            split["eval_start"],
            split["eval_end"],
        )
        print(
            f"\nsplit {i + 1}/{len(splits)}: "
            f"train {split['train_start'].date()}..{split['train_end'].date()}  "
            f"eval {split['eval_start'].date()}..{split['eval_end'].date()}"
        )

        run_ids, sharpes, drawdowns, returns = [], [], [], []
        for seed in range(args.seeds):
            train_env = TradingEnvironment(train_df, tickers=tickers)
            eval_env = TradingEnvironment(eval_df, tickers=tickers)

            agent = PPOAgent(train_env, n_steps=args.n_steps, seed=seed)
            agent.train(total_timesteps=args.timesteps)

            [result] = agent.evaluate(eval_env, n_episodes=1, seed=seed)
            metrics = _metrics_for(result)

            run_id = registry.log_run(
                agent,
                split["train_start"],
                split["train_end"],
                split["eval_start"],
                split["eval_end"],
                metrics,
            )
            run_ids.append(run_id)
            sharpes.append(metrics["sharpe_ratio"])
            drawdowns.append(metrics["max_drawdown"])
            returns.append(metrics["cumulative_return"])
            print(f"  seed {seed}: sharpe={metrics['sharpe_ratio']:.3f}  run_id={run_id}")

        print(
            f"  aggregate: sharpe={np.mean(sharpes):.3f}±{np.std(sharpes):.3f}  "
            f"max_dd={np.mean(drawdowns):.3f}±{np.std(drawdowns):.3f}  "
            f"cum_return={np.mean(returns):.3f}±{np.std(returns):.3f}"
        )

        best_idx = int(np.argmax(sharpes))
        best_run_id = run_ids[best_idx]
        with registry.repo.session() as session:
            mlflow_ref = session.get(ModelRun, best_run_id).mlflow_run_ref
        version_id = registry.register_version(best_run_id, artifact_uri=f"runs:/{mlflow_ref}/model")
        print(
            f"  registered best seed (seed={best_idx}, sharpe={sharpes[best_idx]:.3f}) "
            f"as version {version_id}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
