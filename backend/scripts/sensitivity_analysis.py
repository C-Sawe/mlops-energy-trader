#!/usr/bin/env python3
"""Sensitivity analysis over this project's configurable thresholds.

CLAUDE.md §6: "Thresholds are configurable rather than constant so Chapter
5 can report a sensitivity analysis over them instead of defending magic
numbers. Doing that analysis is a genuine strengthening of the evaluation
— plan for it." This is that analysis, done against real data throughout,
not synthetic illustrations of what the shape "should" look like.

Three independent passes:

1. VIX critical threshold (FR-12, and the VIX-gated retrain deferral,
   CLAUDE.md §7) — what fraction of every real trading day ingested into
   this checkout would have triggered the fail-safe / retrain-deferral at
   each candidate threshold. Pure data query, no training.

2. target_sharpe_threshold (FR-14) — using the real 18-split x 10-seed
   walk-forward sweep (`scripts/deflated_sharpe_analysis.py --out ...`),
   what fraction of those 180 real training runs would read as "below
   target" (i.e. would trigger a retrain if that seed were the serving
   incumbent) at each candidate threshold. Requires that report to already
   exist — regenerating it is a real ~11-minute run, not something this
   script silently redoes.

3. transaction_cost_pct (FR-07) — re-running the equal-weight-rebalanced
   baseline (a policy that actually trades every step, unlike buy-and-hold,
   so it is the one baseline actually sensitive to cost assumptions) on
   real market data at several real cost levels.

Usage:
    python scripts/deflated_sharpe_analysis.py --seeds 10 --out /tmp/dsr.json   # once, ~11 min
    python scripts/sensitivity_analysis.py --dsr-report /tmp/dsr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA, ENVIRONMENT, RISK  # noqa: E402
from src.dataops.processing import normalize_rolling  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402
from src.orchestration.evaluator import sharpe_ratio  # noqa: E402
from src.rlops.baselines import equal_weight_rebalanced, run_policy  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402


def vix_threshold_sensitivity(repo: MarketRepository) -> None:
    print("\n" + "=" * 70)
    print("1. VIX critical threshold (FR-12 / the retrain-deferral gate)")
    print("=" * 70)

    raw = repo.load_partition(DATA.train_start, DATA.eval_end, tickers=(DATA.tickers[0],))
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return
    vix = raw["vix"].dropna()
    n = len(vix)
    print(f"real trading days with a VIX reading: {n} ({raw['date'].min().date()} to {raw['date'].max().date()})")
    print(f"{'threshold':>10} | {'days >= threshold':>18} | {'% of history':>13}")
    for threshold in (20.0, 25.0, 28.0, 30.0, 32.0, 35.0, 38.0, 40.0, 45.0, 50.0):
        hit = int((vix >= threshold).sum())
        marker = "  <- current default" if threshold == RISK.vix_critical_threshold else ""
        print(f"{threshold:>10.1f} | {hit:>18} | {100 * hit / n:>12.2f}%{marker}")
    print(
        f"\nReading: the current default ({RISK.vix_critical_threshold}) fires on "
        f"{100 * (vix >= RISK.vix_critical_threshold).sum() / n:.2f}% of real trading "
        "days in this checkout's history — rare enough to be a genuine circuit "
        "breaker, not a threshold the system spends its life near."
    )


def target_sharpe_sensitivity(dsr_report_path: str) -> None:
    print("\n" + "=" * 70)
    print("2. target_sharpe_threshold (FR-14)")
    print("=" * 70)

    path = Path(dsr_report_path)
    if not path.exists():
        print(
            f"no report at {path} — run "
            "`python scripts/deflated_sharpe_analysis.py --seeds 10 --out {path}` first "
            "(a real ~11-minute training pass; not something this script silently repeats)",
            file=sys.stderr,
        )
        return

    report = json.loads(path.read_text())
    all_seeds = np.array([s for split in report["per_split"] for s in split["seed_sharpes"]])
    best_per_split = np.array([split["best_sharpe"] for split in report["per_split"]])
    n_all, n_splits = len(all_seeds), len(best_per_split)

    print(f"using {path} — {report['seeds']} seeds x {n_splits} splits = {n_all} real training runs")
    print(f"{'threshold':>10} | {'% of all seeds below':>21} | {'% of best-per-split below':>26}")
    for threshold in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        below_all = 100 * (all_seeds < threshold).sum() / n_all
        below_best = 100 * (best_per_split < threshold).sum() / n_splits
        marker = "  <- current default" if threshold == RISK.target_sharpe_threshold else ""
        print(f"{threshold:>10.1f} | {below_all:>20.1f}% | {below_best:>25.1f}%{marker}")
    print(
        "\nReading: 'all seeds' is the honest population (what an untuned single run "
        "actually gets you); 'best-per-split' is already inflated by the same "
        "selection effect §10.3 disciplines with the deflated Sharpe ratio, included "
        "here only to show how much *more* forgiving the trigger looks if judged "
        "against a cherry-picked seed instead of a real one. A higher "
        "target_sharpe_threshold retrains more often against the honest population; "
        "the current default sits in a range where most individual seeds would "
        "already be flagged, which matches this project's own measured seed "
        "sensitivity (CLAUDE.md §10.2), not a threshold tuned to look lenient."
    )


def transaction_cost_sensitivity(repo: MarketRepository) -> None:
    print("\n" + "=" * 70)
    print("3. transaction_cost_pct (FR-07)")
    print("=" * 70)

    raw = repo.load_partition(DATA.eval_start, DATA.eval_end, tickers=DATA.tickers)
    if raw.empty:
        print("no data in the store for the configured eval window", file=sys.stderr)
        return
    frame = normalize_rolling(raw)

    print(f"eval window: {DATA.eval_start} to {DATA.eval_end}, real data, equal-weight-rebalanced baseline")
    print(f"{'cost (bps)':>10} | {'Sharpe':>8} | {'cumulative return':>18}")
    for cost_pct in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01):
        env = TradingEnvironment(frame, tickers=DATA.tickers, transaction_cost_pct=cost_pct)
        result = run_policy(env, equal_weight_rebalanced, seed=0)
        sr = sharpe_ratio(result["returns"])
        cum_return = result["equity_curve"][-1] / result["equity_curve"][0] - 1.0
        marker = "  <- current default" if cost_pct == ENVIRONMENT.transaction_cost_pct else ""
        print(f"{cost_pct * 10_000:>10.1f} | {sr:>8.3f} | {cum_return:>17.2%}{marker}")
    print(
        "\nReading: equal-weight-rebalanced trades every step by construction, so "
        "it is the baseline most exposed to cost drag — the gap between 0bps and "
        "the current default quantifies how much of any policy's apparent edge a "
        "frictionless simulation would have flattered away (CLAUDE.md §10.4)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity analysis over configurable thresholds")
    parser.add_argument("--dsr-report", default="/tmp/dsr_report_real.json")
    args = parser.parse_args()

    repo = MarketRepository()
    vix_threshold_sensitivity(repo)
    target_sharpe_sensitivity(args.dsr_report)
    transaction_cost_sensitivity(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
