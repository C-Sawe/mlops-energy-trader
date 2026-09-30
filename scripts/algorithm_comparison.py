#!/usr/bin/env python3
"""Algorithm comparison: PPO vs A2C vs SAC vs TD3, plus a selection ensemble.

A validation/analysis experiment, the same kind as
`scripts/deflated_sharpe_analysis.py` — it never touches `ModelRegistry` or
the production `model_run`/`model_version` tables, and the production CT
loop keeps training PPO regardless of what this finds.

Protocol (CLAUDE.md §10), identical for every algorithm:
- the same 18 walk-forward splits and the same seeds
- the same training budget (`--timesteps`), each algorithm at SB3 defaults
- transaction costs and the drawdown penalty left on
- per-split baselines on the same evaluation window
- deflated Sharpe with the *total* number of configurations tried

The ensemble follows Yang et al. (2020): for each split and seed, pick the
algorithm with the best Sharpe on a validation slice, then trade the
evaluation window with it. The validation slice is carved off the *end of
the training window* (`--val-days`), so the choice is made before the
evaluation window starts and never sees it (I1/DR-06). Every algorithm
therefore trains on the train window minus that slice, so PPO's numbers here
are not directly comparable with `deflated_sharpe_analysis.py`'s, which
trained on the full window.

Each finished run is appended to a JSONL checkpoint, so an interrupted
sweep resumes where it stopped instead of starting over.

Usage:
    python scripts/algorithm_comparison.py --out results/algorithm_comparison
    python scripts/algorithm_comparison.py --seeds 2 --algorithms ppo a2c --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.processing import (  # noqa: E402
    normalize_rolling,
    partition_chronological,
    walk_forward_splits,
)
from src.dataops.repository import MarketRepository  # noqa: E402
from src.orchestration.evaluator import deflated_sharpe_ratio, sharpe_ratio  # noqa: E402
from src.rlops.agent import ALGORITHMS, RLAgent  # noqa: E402
from src.rlops.baselines import (  # noqa: E402
    all_cash,
    buy_and_hold,
    equal_weight_rebalanced,
    make_random_policy,
    run_policy,
)
from src.rlops.environment import TradingEnvironment  # noqa: E402

ENSEMBLE = "ensemble"

_FRAME: pd.DataFrame | None = None
_TICKERS: tuple[str, ...] = ()


def _init_worker(frame: pd.DataFrame, tickers: tuple[str, ...]) -> None:
    # One thread per worker: parallelism comes from running jobs side by side,
    # and letting every worker also spawn a full torch thread pool would
    # oversubscribe the cores.
    import torch

    torch.set_num_threads(1)
    global _FRAME, _TICKERS
    _FRAME, _TICKERS = frame, tickers


def _windows(split: dict, val_days: int) -> dict[str, pd.Timestamp]:
    fit_end = split["train_end"] - pd.Timedelta(days=val_days)
    return {
        "fit_start": split["train_start"],
        "fit_end": fit_end,
        "val_start": fit_end + pd.Timedelta(days=1),
        "val_end": split["train_end"],
        "eval_start": split["eval_start"],
        "eval_end": split["eval_end"],
    }


def _run_job(job: dict) -> dict:
    w = job["windows"]
    fit_df, val_df = partition_chronological(_FRAME, w["fit_start"], w["fit_end"], w["val_start"], w["val_end"])
    _, eval_df = partition_chronological(_FRAME, w["fit_start"], w["val_end"], w["eval_start"], w["eval_end"])

    started = time.time()
    agent = RLAgent(TradingEnvironment(fit_df, tickers=_TICKERS), algorithm=job["algorithm"], seed=job["seed"])
    agent.train(total_timesteps=job["timesteps"])
    [val] = agent.evaluate(TradingEnvironment(val_df, tickers=_TICKERS), seed=job["seed"])
    [ev] = agent.evaluate(TradingEnvironment(eval_df, tickers=_TICKERS), seed=job["seed"])

    return {
        "algorithm": job["algorithm"],
        "split": job["split"],
        "seed": job["seed"],
        "val_returns": val["returns"].tolist(),
        "eval_returns": ev["returns"].tolist(),
        "seconds": time.time() - started,
    }


def _baselines(frame: pd.DataFrame, tickers: tuple[str, ...], w: dict, seeds: int) -> dict[str, float]:
    _, eval_df = partition_chronological(frame, w["fit_start"], w["val_end"], w["eval_start"], w["eval_end"])
    env = TradingEnvironment(eval_df, tickers=tickers)
    out = {
        name: sharpe_ratio(run_policy(env, policy)["returns"])
        for name, policy in [
            ("buy_and_hold", buy_and_hold),
            ("equal_weight", equal_weight_rebalanced),
            ("all_cash", all_cash),
        ]
    }
    out["random_mean"] = float(
        np.mean([sharpe_ratio(run_policy(env, make_random_policy(s), seed=s)["returns"]) for s in range(seeds)])
    )
    return out


def _load_checkpoint(path: Path) -> dict[tuple, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["algorithm"], r["split"], r["seed"])] = r
    return done


def _summarise(runs: dict[tuple, dict], algorithms: list[str], n_splits: int, seeds: int) -> dict:
    """Per-strategy statistics, the ensemble built post hoc from saved runs."""
    eval_sharpe: dict[str, np.ndarray] = {}
    eval_returns: dict[str, list[list[np.ndarray]]] = {}
    for algo in algorithms:
        eval_sharpe[algo] = np.array(
            [[sharpe_ratio(np.asarray(runs[(algo, s, k)]["eval_returns"])) for k in range(seeds)] for s in range(n_splits)]
        )
        eval_returns[algo] = [
            [np.asarray(runs[(algo, s, k)]["eval_returns"]) for k in range(seeds)] for s in range(n_splits)
        ]

    # Yang et al. (2020) selection: validation Sharpe only, never eval.
    picks = [[None] * seeds for _ in range(n_splits)]
    ens_sharpe = np.zeros((n_splits, seeds))
    ens_returns: list[list[np.ndarray]] = [[None] * seeds for _ in range(n_splits)]
    for s in range(n_splits):
        for k in range(seeds):
            val = {a: sharpe_ratio(np.asarray(runs[(a, s, k)]["val_returns"])) for a in algorithms}
            best = max(val, key=val.get)
            picks[s][k] = best
            ens_sharpe[s, k] = eval_sharpe[best][s, k]
            ens_returns[s][k] = eval_returns[best][s][k]
    eval_sharpe[ENSEMBLE] = ens_sharpe
    eval_returns[ENSEMBLE] = ens_returns

    strategies = algorithms + [ENSEMBLE]
    # Every strategy's every seed on every split is a configuration tried.
    n_trials = len(strategies) * n_splits * seeds

    summary = {"n_trials": n_trials, "strategies": {}}
    for name in strategies:
        sh = eval_sharpe[name]
        best_idx = sh.argmax(axis=1)
        best = sh[np.arange(n_splits), best_idx]
        dsr = np.array(
            [deflated_sharpe_ratio(eval_returns[name][s][best_idx[s]], n_trials=n_trials) for s in range(n_splits)]
        )
        summary["strategies"][name] = {
            "pooled_mean": float(sh.mean()),
            "pooled_std": float(sh.std(ddof=1)),
            "split_mean_of_seed_means": float(sh.mean(axis=1).mean()),
            "best_per_split_mean": float(best.mean()),
            "best_per_split_std": float(best.std(ddof=1)),
            "positive_splits_seed_mean": int((sh.mean(axis=1) > 0).sum()),
            "dsr_mean": float(dsr.mean()),
            "dsr_max": float(dsr.max()),
            "splits_dsr_gt_half": int((dsr > 0.5).sum()),
            "per_split_seed_mean": sh.mean(axis=1).tolist(),
        }

    # Paired across splits: does each strategy's seed-averaged Sharpe beat
    # PPO's on the same windows more often than chance would give?
    if "ppo" in algorithms:
        ppo = eval_sharpe["ppo"].mean(axis=1)
        for name in strategies:
            if name == "ppo":
                continue
            other = eval_sharpe[name].mean(axis=1)
            diff = other - ppo
            p = float(wilcoxon(other, ppo).pvalue) if np.any(diff != 0) else 1.0
            summary["strategies"][name]["vs_ppo"] = {
                "splits_better": int((diff > 0).sum()),
                "mean_diff": float(diff.mean()),
                "wilcoxon_p": p,
            }

    flat = [p for row in picks for p in row]
    summary["ensemble_picks"] = {a: flat.count(a) for a in algorithms}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RL algorithms and a selection ensemble")
    parser.add_argument("--algorithms", nargs="*", default=sorted(ALGORITHMS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--eval-days", type=int, default=180)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=str, required=True, help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "runs.jsonl"

    tickers = DATA.tickers
    raw = MarketRepository().load_partition(DATA.train_start, DATA.eval_end, tickers=tickers)
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return 1
    frame = normalize_rolling(raw)
    splits = walk_forward_splits(DATA.train_start, DATA.eval_end, train_days=args.train_days, eval_days=args.eval_days)
    windows = [_windows(s, args.val_days) for s in splits]

    done = _load_checkpoint(checkpoint)
    jobs = [
        {"algorithm": a, "split": i, "seed": k, "windows": w, "timesteps": args.timesteps}
        for a in args.algorithms
        for i, w in enumerate(windows)
        for k in range(args.seeds)
        if (a, i, k) not in done
    ]
    # Slowest algorithms first, so the long tail doesn't sit on one worker at the end.
    jobs.sort(key=lambda j: j["algorithm"] not in ("sac", "td3"))
    total = len(args.algorithms) * len(windows) * args.seeds
    print(f"splits={len(windows)} seeds={args.seeds} algorithms={args.algorithms} "
          f"runs={total} already done={len(done)} to run={len(jobs)} workers={args.workers}", flush=True)

    started = time.time()
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(frame, tickers)) as pool, \
                checkpoint.open("a") as fh:
            for n, r in enumerate(pool.imap_unordered(_run_job, jobs), start=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[(r["algorithm"], r["split"], r["seed"])] = r
                if n % 20 == 0 or n == len(jobs):
                    print(f"[{time.time() - started:.0f}s] {len(done)}/{total} runs done", flush=True)

    baselines = [_baselines(frame, tickers, w, args.seeds) for w in windows]
    summary = _summarise(done, args.algorithms, len(windows), args.seeds)
    summary["baselines_split_mean"] = {k: float(np.mean([b[k] for b in baselines])) for k in baselines[0]}
    summary["config"] = {**vars(args), "splits": len(windows)}
    summary["windows"] = [{k: str(v.date()) for k, v in w.items()} for w in windows]
    summary["per_split_baselines"] = baselines
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"n_trials for DSR: {summary['n_trials']}")
    print(f"{'strategy':<10}{'pooled':>16}{'best/split':>12}{'+splits':>9}{'DSR max':>9}{'DSR>.5':>8}{'vs PPO (p)':>16}")
    for name, s in summary["strategies"].items():
        vs = s.get("vs_ppo")
        vs_txt = f"{vs['splits_better']}/{len(windows)} ({vs['wilcoxon_p']:.3f})" if vs else "—"
        print(f"{name:<10}{s['pooled_mean']:>8.3f}±{s['pooled_std']:<7.3f}{s['best_per_split_mean']:>12.3f}"
              f"{s['positive_splits_seed_mean']:>6}/{len(windows)}{s['dsr_max']:>9.3f}"
              f"{s['splits_dsr_gt_half']:>5}/{len(windows)}{vs_txt:>16}")
    print("baselines (mean Sharpe across splits): "
          + ", ".join(f"{k}={v:.3f}" for k, v in summary["baselines_split_mean"].items()))
    print(f"ensemble picks: {summary['ensemble_picks']}")
    print(f"wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
