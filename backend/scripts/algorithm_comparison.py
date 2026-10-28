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

Every run also writes <out>/report/: seven numbered charts and REPORT.md,
which explains how to read each chart and fills in this run's numbers.
`--report-only` rebuilds that report from the saved runs.jsonl and
summary.json, without the database or any training.

Usage:
    python scripts/algorithm_comparison.py --out results/algorithm_comparison
    python scripts/algorithm_comparison.py --seeds 2 --algorithms ppo a2c --out /tmp/smoke
    python scripts/algorithm_comparison.py --out results/algorithm_comparison --report-only
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
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


def _ensemble_picks(runs: dict[tuple, dict], algorithms: list[str], n_splits: int, seeds: int) -> list[list[str]]:
    """Yang et al. (2020) selection: for each split and seed, the algorithm
    with the best Sharpe on the validation slice. Validation only, never the
    evaluation window it is then scored on (I1). Shared by the summary and
    the report so both describe the same choices."""
    picks = []
    for s in range(n_splits):
        row = []
        for k in range(seeds):
            val = {a: sharpe_ratio(np.asarray(runs[(a, s, k)]["val_returns"])) for a in algorithms}
            row.append(max(val, key=val.get))
        picks.append(row)
    return picks


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

    picks = _ensemble_picks(runs, algorithms, n_splits, seeds)
    ens_sharpe = np.array([[eval_sharpe[picks[s][k]][s, k] for k in range(seeds)] for s in range(n_splits)])
    ens_returns = [[eval_returns[picks[s][k]][s][k] for k in range(seeds)] for s in range(n_splits)]
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


# ----------------------------------------------------------------- report
# Charts and a plain-language REPORT.md, written to <out>/report/ after every
# run (or rebuilt from saved files with --report-only). Each chart answers
# one question; REPORT.md explains how to read it and fills in this run's
# numbers, so the output can be shown as it is.
LABELS = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "td3": "TD3", ENSEMBLE: "Ensemble"}
COLORS = {"ppo": "#2a78d6", "a2c": "#eb6834", "sac": "#1baf7a", "td3": "#7a5cc4", ENSEMBLE: "#52514e"}
MARKERS = {"ppo": "o", "a2c": "s", "sac": "^", "td3": "D", ENSEMBLE: "X"}


def _run_table(runs: dict[tuple, dict], algorithms: list[str], n_splits: int, seeds: int) -> pd.DataFrame:
    """One row per run: validation and evaluation Sharpe, plus the ensemble's
    row for each split and seed (a copy of the algorithm it picked)."""
    rows = []
    for (a, s, k), r in runs.items():
        if a in algorithms and s < n_splits and k < seeds:
            rows.append({"strategy": a, "split": s, "seed": k,
                         "val_sharpe": sharpe_ratio(np.asarray(r["val_returns"])),
                         "eval_sharpe": sharpe_ratio(np.asarray(r["eval_returns"]))})
    table = pd.DataFrame(rows)
    picks = _ensemble_picks(runs, algorithms, n_splits, seeds)
    by_key = table.set_index(["strategy", "split", "seed"])
    ensemble = [{"strategy": ENSEMBLE, "split": s, "seed": k, "picked": picks[s][k],
                 **by_key.loc[(picks[s][k], s, k)].to_dict()}
                for s in range(n_splits) for k in range(seeds)]
    return pd.concat([table, pd.DataFrame(ensemble)], ignore_index=True)


def _write_report(runs: dict[tuple, dict], summary: dict, out: Path) -> list[Path]:
    """Writes the charts and REPORT.md into out/report/ and returns their paths.
    Needs matplotlib (installed with mlflow); without it, says so and skips."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib is not installed; skipping the report", file=sys.stderr)
        return []
    from scipy.stats import spearmanr

    cfg = summary["config"]
    algorithms, seeds, n_splits = list(cfg["algorithms"]), int(cfg["seeds"]), int(cfg["splits"])
    strategies = algorithms + [ENSEMBLE]
    table = _run_table(runs, algorithms, n_splits, seeds)
    stats = summary["strategies"]
    bh = summary.get("baselines_split_mean", {}).get("buy_and_hold")
    report = out / "report"
    report.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = report / name
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    # 1. How the data is split: train, choose, then score.
    if summary.get("windows"):
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(summary["windows"]) + 1.2))
        for i, w in enumerate(summary["windows"]):
            d = {key: mdates.date2num(pd.Timestamp(v)) for key, v in w.items()}
            y = len(summary["windows"]) - 1 - i
            ax.barh(y, d["fit_end"] - d["fit_start"], left=d["fit_start"], height=0.6, color="#9ec5f4")
            ax.barh(y, d["val_end"] - d["val_start"], left=d["val_start"], height=0.6, color="#eb6834")
            ax.barh(y, d["eval_end"] - d["eval_start"], left=d["eval_start"], height=0.6, color="#2a78d6")
        ax.set_yticks(range(len(summary["windows"])))
        ax.set_yticklabels([str(len(summary["windows"]) - i) for i in range(len(summary["windows"]))])
        ax.set_ylabel("Split")
        ax.xaxis_date()
        ax.grid(axis="y", visible=False)
        ax.set_title("How each split is used: train, then choose, then score", loc="left")
        ax.legend(handles=[Patch(color="#9ec5f4", label="Train each algorithm"),
                           Patch(color="#eb6834", label=f"Validation: choose ({cfg['val_days']} days)"),
                           Patch(color="#2a78d6", label=f"Evaluation: score ({cfg['eval_days']} days)")],
                  loc="upper left", ncol=3, frameon=False, fontsize=8)
        save(fig, "1_windows.png")

    # 2. Evaluation Sharpe of every run, per strategy, against buy-and-hold.
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [table.loc[table.strategy == s, "eval_sharpe"].values for s in strategies]
    box = ax.boxplot(data, widths=0.45, patch_artist=True, medianprops={"color": "black"})
    for patch, s in zip(box["boxes"], strategies):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.45)
    ax.scatter(range(1, len(strategies) + 1), [d.mean() for d in data], marker="D", color="black", zorder=3,
               label="Mean")
    if bh is not None:
        ax.axhline(bh, color="black", linestyle="--", linewidth=1, label=f"Buy-and-hold ({bh:.2f})")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels([LABELS.get(s, s) for s in strategies])
    ax.set_ylabel("Evaluation-window Sharpe ratio")
    ax.set_title(f"Every run's result ({n_splits} splits × {seeds} seeds per strategy)", loc="left")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "2_sharpe_by_strategy.png")

    # 3. A typical seed against the best seed per split: selection bias.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(strategies))
    typical = [stats[s]["pooled_mean"] for s in strategies]
    best = [stats[s]["best_per_split_mean"] for s in strategies]
    ax.bar(x - 0.2, typical, width=0.38, color="#9b9a94", label="Typical seed (mean of all runs)")
    ax.bar(x + 0.2, best, width=0.38, color="#2a78d6", label="Best seed of each split (cherry-picked)")
    if bh is not None:
        ax.axhline(bh, color="black", linestyle="--", linewidth=1, label=f"Buy-and-hold ({bh:.2f})")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(s, s) for s in strategies])
    ax.set_ylabel("Mean evaluation Sharpe")
    ax.set_title("What one untuned run gets you, against the best of several", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(min(min(typical), 0) - 0.4, max(max(best), bh or 0) + 0.8)
    save(fig, "3_best_vs_typical.png")

    # 4. Each strategy minus PPO on every split, with the Wilcoxon result.
    if "ppo" in algorithms:
        means = table.groupby(["strategy", "split"])["eval_sharpe"].mean().unstack("split")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        splits = np.arange(1, n_splits + 1)
        for s in strategies:
            if s == "ppo":
                continue
            vs = stats[s]["vs_ppo"]
            ax.plot(splits, (means.loc[s] - means.loc["ppo"]).values, marker=MARKERS[s], color=COLORS[s],
                    linewidth=1, markersize=5,
                    label=f"{LABELS.get(s, s)}: better on {vs['splits_better']}/{n_splits}, p = {vs['wilcoxon_p']:.2f}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(splits)
        ax.set_xlabel("Split")
        ax.set_ylabel("Sharpe minus PPO's\n(mean over seeds)")
        ax.set_title("Does any strategy beat PPO consistently? (above 0 = better than PPO)", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="best")
        save(fig, "4_vs_ppo_by_split.png")

    # 5. Does validation predict evaluation? The ensemble depends on it.
    algos = table[table.strategy != ENSEMBLE]
    rho, rho_p = spearmanr(algos["val_sharpe"], algos["eval_sharpe"])
    within = [spearmanr(g["val_sharpe"], g["eval_sharpe"]).statistic for _, g in algos.groupby("split")]
    within_mean = float(np.nanmean(within))
    within_positive = int(sum(w > 0 for w in within))
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in algorithms:
        part = algos[algos.strategy == s]
        ax.scatter(part["val_sharpe"], part["eval_sharpe"], s=12, alpha=0.5, color=COLORS[s],
                   marker=MARKERS[s], linewidths=0, label=LABELS.get(s, s))
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Validation Sharpe (what the ensemble chooses on)")
    ax.set_ylabel("Evaluation Sharpe (what it is scored on)")
    ax.set_title("Does a good validation score predict a good evaluation score?", loc="left")
    ax.text(0.01, 0.98, f"All runs: Spearman ρ = {rho:.2f}\nWithin splits: mean ρ = {within_mean:.2f}, "
                        f"positive in {within_positive}/{n_splits}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=len(algorithms))
    save(fig, "5_validation_vs_evaluation.png")

    # 6. Which algorithm the ensemble picked, against an even spread.
    picks = summary["ensemble_picks"]
    total_picks = sum(picks.values())
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([LABELS.get(a, a) for a in algorithms], [picks[a] for a in algorithms],
           color=[COLORS[a] for a in algorithms])
    ax.axhline(total_picks / len(algorithms), color="black", linestyle="--", linewidth=1,
               label=f"Even spread ({total_picks / len(algorithms):.0f} each)")
    for i, a in enumerate(algorithms):
        ax.text(i, picks[a], str(picks[a]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"Times picked (of {total_picks})")
    ax.set_title("Which algorithm the ensemble chose", loc="left")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "6_ensemble_picks.png")

    # 7. Deflated Sharpe: does any best result survive the number of trials?
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - 0.2, [stats[s]["dsr_mean"] for s in strategies], width=0.38, color="#9ec5f4",
           label="Mean over splits")
    ax.bar(x + 0.2, [stats[s]["dsr_max"] for s in strategies], width=0.38, color="#2a78d6",
           label="Best split")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(s, s) for s in strategies])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Deflated Sharpe ratio\n(probability the edge is real)")
    ax.set_title(f"After correcting for {summary['n_trials']} configurations tried", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    save(fig, "7_deflated_sharpe.png")

    report_md = report / "REPORT.md"
    report_md.write_text(_report_text(summary, strategies, rho, rho_p, within_mean, within_positive, written))
    written.append(report_md)
    return written


def _report_text(summary: dict, strategies: list[str], rho: float, rho_p: float,
                 within_mean: float, within_positive: int, charts: list[Path]) -> str:
    cfg, stats = summary["config"], summary["strategies"]
    n_splits, seeds = int(cfg["splits"]), int(cfg["seeds"])
    bh = summary.get("baselines_split_mean", {}).get("buy_and_hold")
    names = [LABELS.get(s, s) for s in strategies]
    beats_bh = [s for s in strategies if bh is not None and stats[s]["pooled_mean"] > bh]
    vs = {s: stats[s]["vs_ppo"] for s in strategies if "vs_ppo" in stats[s]}
    significant = [LABELS.get(s, s) for s, v in vs.items() if v["wilcoxon_p"] < 0.05]
    survivors = [LABELS.get(s, s) for s in strategies if stats[s]["splits_dsr_gt_half"] > 0]
    picks = summary["ensemble_picks"]
    chart = {p.name: p.name for p in charts}

    def vs_ppo_cell(s: str) -> str:
        if s not in vs:
            return "—"
        return f"{vs[s]['splits_better']}/{n_splits} (p = {vs[s]['wilcoxon_p']:.2f})"

    rows = "\n".join(
        f"| {LABELS.get(s, s)} | {stats[s]['pooled_mean']:.3f} ± {stats[s]['pooled_std']:.3f} "
        f"| {stats[s]['best_per_split_mean']:.3f} | {stats[s]['dsr_max']:.3f} "
        f"| {stats[s]['splits_dsr_gt_half']}/{n_splits} | {vs_ppo_cell(s)} |"
        for s in strategies
    )
    bh_line = f"Buy-and-hold over the same windows averaged **{bh:.3f}**." if bh is not None else ""
    lines = [
        "# Algorithm comparison report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/algorithm_comparison.py`. "
        f"Strategies: {', '.join(names)}. {n_splits} walk-forward splits × {seeds} seeds, "
        f"{cfg['timesteps']} training timesteps per run, every algorithm at its Stable-Baselines3 defaults, "
        "transaction costs and the drawdown penalty on.",
        "",
        "## Results at a glance",
        "",
        "| Strategy | Typical run (mean ± sd Sharpe) | Best seed per split | Deflated Sharpe, best split | Splits with deflated Sharpe > 0.5 | Better than PPO (Wilcoxon) |",
        "|---|---|---|---|---|---|",
        rows,
        "",
        bh_line,
        "",
        "**In one paragraph.** "
        + (f"{', '.join(LABELS.get(s, s) for s in beats_bh)} beat buy-and-hold on a typical run. "
           if beats_bh else "No strategy beats simply holding the stocks on a typical run. ")
        + (f"{', '.join(significant)} differ significantly from PPO. " if significant
           else "No strategy is significantly better or worse than PPO (all Wilcoxon p ≥ 0.05). ")
        + (f"{', '.join(survivors)} have at least one split whose best result survives the deflated Sharpe test."
           if survivors else "No strategy's best result survives correction for the number of models tried "
                             "(no deflated Sharpe above 0.5)."),
        "",
        "## How to read each chart",
        "",
    ]
    sections = [
        ("1_windows.png", "How the data is split",
         "Each row is one walk-forward split. Every algorithm trains on the light blue part. The ensemble "
         "then chooses an algorithm using the orange validation slice, the last days of the training "
         "period. The choice is scored on the dark blue evaluation window, which starts after validation "
         "ends. Choosing on the data you are scored on would be look-ahead; the layout rules it out."),
        ("2_sharpe_by_strategy.png", "Every run's result",
         "One box per strategy, built from every split and seed. The line inside the box is the median, "
         "the diamond is the mean, and the box covers the middle half of the runs. The dashed line is "
         "buy-and-hold. A strategy only matters if its box sits clearly above that line."),
        ("3_best_vs_typical.png", "Typical run against the best of several",
         "Grey is what a single untuned run achieves on average. Blue is what you get by training "
         f"{seeds} seeds and keeping the best one in each split. The gap between them is selection bias: "
         "keeping the best of many always looks better, even when no single run is good. Report grey, "
         "not blue."),
        ("4_vs_ppo_by_split.png", "Each strategy against PPO, split by split",
         "Each line is a strategy's mean Sharpe minus PPO's on the same split. Above zero means it did "
         "better than PPO there. The legend gives how many splits it won and the Wilcoxon signed-rank "
         "p-value: below 0.05 would mean the difference is unlikely to be chance."),
        ("5_validation_vs_evaluation.png", "Does validation predict evaluation?",
         "Each dot is one trained model. The ensemble picks whichever algorithm scored best on validation, "
         "so it can only help if a good validation score is followed by a good evaluation score: dots "
         f"rising from left to right. Here Spearman ρ = {rho:.2f} across all runs (p = {rho_p:.2g}), and "
         f"the mean within individual splits is {within_mean:.2f}, positive in {within_positive} of "
         f"{n_splits}. "
         + ("Validation scores did not carry over, so choosing on them could not add much."
            if within_mean <= 0.1 else "Validation scores carried over to some extent.")),
        ("6_ensemble_picks.png", "Which algorithm the ensemble picked",
         f"How often each algorithm was chosen ({', '.join(f'{LABELS.get(a, a)} {n}' for a, n in picks.items())}). "
         "The dashed line is an even spread. Picks close to even suggest the choice was close to random, "
         "which is what chart 5 predicts when validation carries little information."),
        ("7_deflated_sharpe.png", "Does any best result survive?",
         f"The best of {summary['n_trials']} backtests will look good by chance. The deflated Sharpe ratio "
         "(Bailey & López de Prado, 2014) estimates the probability that a best result reflects a real edge "
         "once that is accounted for. Above 0.5 would be more likely real than not."),
    ]
    for file, title, text in sections:
        if file in chart:
            lines += [f"### {title}", "", f"![{title}]({file})", "", text, ""]
    lines += [
        "## Reading this honestly",
        "",
        "- Every algorithm ran at default settings with the same budget. This compares algorithms as used, "
        "not each algorithm at its best.",
        "- The evaluation windows are simulated with transaction costs, on daily data for five stocks. "
        "Simulated results do not include real slippage or liquidity, so none of this is evidence of a "
        "profitable strategy.",
        "- The question this answers is whether swapping the learner inside the training loop would change "
        "anything. A strategy is only worth switching to if it is consistently better than PPO and survives "
        "the deflated Sharpe test.",
        "",
    ]
    return "\n".join(lines)


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
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild <out>/report/ from saved runs.jsonl and summary.json; no training")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "runs.jsonl"

    if args.report_only:
        summary_path = out / "summary.json"
        if not (checkpoint.exists() and summary_path.exists()):
            print(f"--report-only needs {checkpoint} and {summary_path}", file=sys.stderr)
            return 1
        written = _write_report(_load_checkpoint(checkpoint), json.loads(summary_path.read_text()), out)
        print("\n".join(f"wrote {p}" for p in written))
        return 0

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
    for path in _write_report(done, summary, out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
