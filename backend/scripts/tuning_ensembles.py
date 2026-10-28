#!/usr/bin/env python3
"""Hyperparameter tuning and decision ensembles: can either make the agent better?

A validation/analysis experiment, the same kind as
`scripts/algorithm_comparison.py`. It never touches `ModelRegistry` or the
production tables, and the CT loop keeps training default PPO whatever this
finds. Adopting a tuned configuration or an ensemble in production would be
a separate decision (CLAUDE.md §2).

Protocol (CLAUDE.md §10), with tuning kept apart from the final test:

- The same 18 walk-forward splits as the algorithm comparison (2 years
  train, 6 months evaluation). Splits 1-9 are the **development** periods;
  their evaluation windows end on 2021-06-07. Splits 10-18 are the
  **holdout**: every holdout evaluation day is after every development day,
  so nothing chosen in development has seen holdout data (I1).
- Each split's training window loses its last `--val-days` to a validation
  slice, which the weighted blend and the selection ensemble use. No choice
  is ever made on an evaluation window.

Phase 1, tuning (development splits only). Random search over PPO and A2C
hyperparameters (Bergstra & Bengio 2012). Each configuration trains on every
development split with several seeds and is scored by its **mean** Sharpe
over all of them: the typical seed, not the best one, because the best of
many seeds is the selection bias §10.3 exists for. SB3's defaults are always
configuration 0, so tuning can only replace them by beating them.

Phase 2, holdout. On each holdout split, five members are trained with
several seeds and saved: default PPO, tuned PPO, tuned A2C, SAC and TD3
(SAC and TD3 at defaults: tuning them would cost ~40x more compute).

Phase 3, ensembles. Ensembles combine **decisions**, not network weights:
the members are separately trained networks whose internals don't line up,
so their parts can't be combined, but each day's target weights can be
averaged. The strategies are fixed here, before any holdout result exists:

  seed_avg_ppo_default / seed_avg_ppo_tuned  one algorithm, all seeds averaged
  blend_equal      per seed: tuned PPO + tuned A2C + SAC + TD3, equal weights
  blend_all        all of those models, every seed, equal weights
  blend_weighted   the same models weighted by validation Sharpe (positive part;
                   equal weights if none is positive)
  selection        per seed: the algorithm with the best validation Sharpe
                   (Yang et al. 2020), as in algorithm_comparison.py

Everything is scored on the holdout against buy-and-hold and equal weight,
with paired Wilcoxon tests against default PPO and deflated Sharpe ratios
counting every series evaluated on the holdout (and, more conservatively,
every tuning configuration as well).

Checkpoints (tune.jsonl, holdout.jsonl, ensembles.jsonl) make an interrupted
run resume where it stopped. <out>/report/ holds the charts and REPORT.md;
`--report-only` rebuilds them from the saved files.

Usage:
    python scripts/tuning_ensembles.py --out results/tuning_ensembles
    python scripts/tuning_ensembles.py --out results/tuning_ensembles --report-only
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
from src.orchestration.evaluator import deflated_sharpe_ratio, sharpe_ratio  # noqa: E402
from src.rlops.agent import RLAgent  # noqa: E402
from src.rlops.baselines import buy_and_hold, equal_weight_rebalanced, run_policy  # noqa: E402
from src.rlops.environment import TradingEnvironment  # noqa: E402

# Search spaces. Values bracket SB3's defaults (marked in DEFAULTS below).
SPACES: dict[str, dict[str, list]] = {
    "ppo": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [512, 1024, 2048],
        "batch_size": [32, 64, 128],
        "n_epochs": [5, 10, 20],
        "gamma": [0.9, 0.95, 0.99],
        "gae_lambda": [0.9, 0.95, 1.0],
        "clip_range": [0.1, 0.2, 0.3],
        "ent_coef": [0.0, 0.001, 0.01],
        "net_arch": [32, 64, 128],
        "timesteps": [20_000, 50_000],
    },
    "a2c": {
        "learning_rate": [1e-4, 3e-4, 7e-4, 1e-3],
        "n_steps": [5, 16, 32],
        "gamma": [0.9, 0.95, 0.99],
        "gae_lambda": [0.9, 1.0],
        "ent_coef": [0.0, 0.001, 0.01],
        "net_arch": [32, 64, 128],
        "timesteps": [20_000, 50_000],
    },
}
DEFAULTS: dict[str, dict] = {
    "ppo": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "net_arch": 64, "timesteps": 20_000},
    "a2c": {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0,
            "net_arch": 64, "timesteps": 20_000},
    "sac": {"timesteps": 20_000},
    "td3": {"timesteps": 20_000},
}
# Holdout members: name -> algorithm. "tuned" members get Phase 1's winner.
MEMBERS = {"ppo_default": "ppo", "ppo_tuned": "ppo", "a2c_tuned": "a2c", "sac": "sac", "td3": "td3"}
BLEND_MEMBERS = ("ppo_tuned", "a2c_tuned", "sac", "td3")
STRATEGIES = ("ppo_default", "ppo_tuned", "a2c_tuned", "sac", "td3", "seed_avg_ppo_default",
              "seed_avg_ppo_tuned", "blend_equal", "blend_all", "blend_weighted", "selection")
PER_SEED = {"ppo_default", "ppo_tuned", "a2c_tuned", "sac", "td3", "blend_equal", "selection"}

_FRAME: pd.DataFrame | None = None
_TICKERS: tuple[str, ...] = ()


# --------------------------------------------------------------------- configs
def sample_configs(algorithm: str, n: int, seed: int = 0) -> list[dict]:
    """`n` distinct configurations, SB3's defaults first. Seeded, so a
    resumed run samples the same ones."""
    space, rng = SPACES[algorithm], np.random.default_rng(seed)
    configs, seen = [dict(DEFAULTS[algorithm])], set()
    seen.add(json.dumps(configs[0], sort_keys=True))
    while len(configs) < n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        if "batch_size" in cfg and cfg["batch_size"] > cfg["n_steps"]:
            continue
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def to_kwargs(config: dict) -> tuple[dict, int]:
    """A search configuration as SB3 keyword arguments plus a step budget."""
    kwargs = {k: v for k, v in config.items() if k not in ("timesteps", "net_arch")}
    if "net_arch" in config:
        kwargs["policy_kwargs"] = {"net_arch": [config["net_arch"], config["net_arch"]]}
    return kwargs, int(config["timesteps"])


# --------------------------------------------------------------------- splits
def windows(split: dict, val_days: int) -> dict[str, pd.Timestamp]:
    fit_end = split["train_end"] - pd.Timedelta(days=val_days)
    return {"fit_start": split["train_start"], "fit_end": fit_end,
            "val_start": fit_end + pd.Timedelta(days=1), "val_end": split["train_end"],
            "eval_start": split["eval_start"], "eval_end": split["eval_end"]}


def dev_and_holdout(all_windows: list[dict], dev_splits: int) -> tuple[list[int], list[int]]:
    """Development splits first; the holdout must start after every
    development evaluation day (I1), or the protocol is refused."""
    dev, holdout = list(range(dev_splits)), list(range(dev_splits, len(all_windows)))
    if not dev or not holdout:
        raise ValueError("need at least one development and one holdout split")
    last_dev = max(all_windows[i]["eval_end"] for i in dev)
    if min(all_windows[i]["eval_start"] for i in holdout) <= last_dev:
        raise ValueError("holdout evaluation overlaps development evaluation")
    return dev, holdout


def tune_jobs(all_windows: list[dict], dev: list[int], configs: dict[str, list[dict]], seeds: int) -> list[dict]:
    return [{"phase": "tune", "algorithm": algo, "config_id": c, "config": cfg, "split": i, "seed": k,
             "windows": all_windows[i]}
            for algo, cfgs in configs.items() for c, cfg in enumerate(cfgs) for i in dev for k in range(seeds)]


# --------------------------------------------------------------------- workers
def _init_worker(frame: pd.DataFrame, tickers: tuple[str, ...]) -> None:
    import torch

    torch.set_num_threads(1)
    global _FRAME, _TICKERS
    _FRAME, _TICKERS = frame, tickers


def _frames(w: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_df, val_df = partition_chronological(_FRAME, w["fit_start"], w["fit_end"], w["val_start"], w["val_end"])
    _, eval_df = partition_chronological(_FRAME, w["fit_start"], w["val_end"], w["eval_start"], w["eval_end"])
    return fit_df, val_df, eval_df


def _train(algorithm: str, config: dict, fit_df: pd.DataFrame, seed: int) -> RLAgent:
    kwargs, steps = to_kwargs(config)
    agent = RLAgent(TradingEnvironment(fit_df, tickers=_TICKERS), algorithm=algorithm, seed=seed, **kwargs)
    return agent.train(total_timesteps=steps)


def _run_job(job: dict) -> dict:
    """Train one model; score it on validation and evaluation. Holdout
    models are saved, because the ensembles need them afterwards."""
    started = time.time()
    fit_df, val_df, eval_df = _frames(job["windows"])
    agent = _train(job["algorithm"], job["config"], fit_df, job["seed"])
    [val] = agent.evaluate(TradingEnvironment(val_df, tickers=_TICKERS), seed=job["seed"])
    [ev] = agent.evaluate(TradingEnvironment(eval_df, tickers=_TICKERS), seed=job["seed"])
    if job.get("model_path"):
        agent.save(job["model_path"])
    out = {k: job[k] for k in ("phase", "algorithm", "split", "seed")}
    out.update({k: job[k] for k in ("config_id", "member", "model_path") if k in job})
    out.update(val_returns=val["returns"].tolist(), eval_returns=ev["returns"].tolist(),
               seconds=time.time() - started)
    return out


def rollout(env: TradingEnvironment, act) -> np.ndarray:
    """One episode where `act(obs)` gives the target weights; returns daily returns."""
    obs, _ = env.reset(seed=0)
    equity, done = [env.initial_cash], False
    while not done:
        obs, _r, terminated, truncated, info = env.step(act(obs))
        equity.append(info["equity"])
        done = terminated or truncated
    curve = np.asarray(equity, dtype=np.float64)
    return np.diff(curve) / curve[:-1]


def blend(models: list, weights: list[float] | None = None):
    """Each day: every model's target weights, averaged (weighted), clipped
    to the action space. Combines decisions only (see the module docstring)."""
    w = np.full(len(models), 1.0 / len(models)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    def act(obs: np.ndarray) -> np.ndarray:
        actions = np.stack([np.asarray(m.predict(obs), dtype=np.float64) for m in models])
        return np.clip((w[:, None] * actions).sum(axis=0), -1.0, 1.0).astype(np.float32)
    return act


def validation_weights(val_sharpes: list[float]) -> list[float]:
    """Positive validation Sharpe, normalised; equal weights if none is positive."""
    pos = np.clip(np.asarray(val_sharpes, dtype=float), 0.0, None)
    return (pos / pos.sum()).tolist() if pos.sum() > 0 else [1.0 / len(pos)] * len(pos)


def _ensemble_job(job: dict) -> dict:
    """All ensembles for one holdout split, from that split's saved members."""
    _, _, eval_df = _frames(job["windows"])
    env = lambda: TradingEnvironment(eval_df, tickers=_TICKERS)  # noqa: E731
    runs = job["runs"]  # (member, seed) -> {"model_path", "val_sharpe", "algorithm"}
    seeds = sorted({k[1] for k in runs})
    models = {key: RLAgent.load(r["model_path"], algorithm=r["algorithm"]) for key, r in runs.items()}
    out: dict[str, list] = {}
    for member in ("ppo_default", "ppo_tuned"):
        out[f"seed_avg_{member}"] = [rollout(env(), blend([models[(member, k)] for k in seeds])).tolist()]
    out["blend_equal"] = [rollout(env(), blend([models[(m, k)] for m in BLEND_MEMBERS])).tolist() for k in seeds]
    pool = [(m, k) for m in BLEND_MEMBERS for k in seeds]
    out["blend_all"] = [rollout(env(), blend([models[p] for p in pool])).tolist()]
    weights = validation_weights([runs[p]["val_sharpe"] for p in pool])
    out["blend_weighted"] = [rollout(env(), blend([models[p] for p in pool], weights)).tolist()]
    return {"split": job["split"], "strategies": out, "weighted_weights": dict(zip(map(str, pool), weights))}


def _map(fn, jobs: list[dict], workers: int, frame: pd.DataFrame, tickers: tuple[str, ...]):
    """Run jobs in a spawn pool, or in this process when workers <= 1."""
    if workers <= 1:
        _init_worker(frame, tickers)
        yield from map(fn, jobs)
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(frame, tickers)) as pool:
        yield from pool.imap_unordered(fn, jobs)


# --------------------------------------------------------------------- analysis
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def pick_best(tune_runs: list[dict], algorithm: str) -> dict:
    """Mean evaluation Sharpe per configuration over every development
    split and seed; the highest wins, the lowest id on a tie (so the default,
    id 0, is kept unless something beats it)."""
    scores: dict[int, list[float]] = {}
    for r in tune_runs:
        if r["algorithm"] == algorithm:
            scores.setdefault(r["config_id"], []).append(sharpe_ratio(np.asarray(r["eval_returns"])))
    table = sorted(((float(np.mean(v)), -cid, cid, float(np.std(v)), len(v)) for cid, v in scores.items()),
                   reverse=True)
    ranking = [{"config_id": cid, "mean_sharpe": m, "std_sharpe": sd, "runs": n} for m, _, cid, sd, n in table]
    best = ranking[0]["config_id"]
    default_rank = next(i for i, r in enumerate(ranking) if r["config_id"] == 0) + 1
    return {"best_config_id": best, "ranking": ranking, "default_rank": default_rank}


def strategy_returns(holdout_runs: list[dict], ensembles: list[dict], holdout: list[int],
                     seeds: int) -> dict[str, dict[int, list[np.ndarray]]]:
    """strategy -> split -> list of return series (one per seed, or one)."""
    by = {(r["member"], r["split"], r["seed"]): r for r in holdout_runs}
    out: dict[str, dict[int, list[np.ndarray]]] = {s: {} for s in STRATEGIES}
    for i in holdout:
        for member in MEMBERS:
            out[member][i] = [np.asarray(by[(member, i, k)]["eval_returns"]) for k in range(seeds)]
        picks = []
        for k in range(seeds):
            val = {m: sharpe_ratio(np.asarray(by[(m, i, k)]["val_returns"])) for m in BLEND_MEMBERS}
            best = max(BLEND_MEMBERS, key=lambda m: (val[m], -BLEND_MEMBERS.index(m)))
            picks.append(np.asarray(by[(best, i, k)]["eval_returns"]))
        out["selection"][i] = picks
    for e in ensembles:
        for name, series in e["strategies"].items():
            out[name][e["split"]] = [np.asarray(s) for s in series]
    return out


def summarise(returns: dict, baselines: dict[int, dict], holdout: list[int], n_tuning_configs: int) -> dict:
    n_series = sum(len(returns[s][i]) for s in STRATEGIES for i in holdout)
    strategies = {}
    ref = {i: float(np.mean([sharpe_ratio(r) for r in returns["ppo_default"][i]])) for i in holdout}
    for s in STRATEGIES:
        per_split = {i: [sharpe_ratio(r) for r in returns[s][i]] for i in holdout}
        split_mean = [float(np.mean(per_split[i])) for i in holdout]
        allv = [v for i in holdout for v in per_split[i]]
        n_rep = len(returns[s][holdout[0]])
        dsr = [deflated_sharpe_ratio(np.concatenate([returns[s][i][k] for i in holdout]), n_trials=n_series)
               for k in range(n_rep)]
        dsr_cons = [deflated_sharpe_ratio(np.concatenate([returns[s][i][k] for i in holdout]),
                                          n_trials=n_series + n_tuning_configs) for k in range(n_rep)]
        diffs = [split_mean[j] - ref[i] for j, i in enumerate(holdout)]
        p = float(wilcoxon(diffs).pvalue) if s != "ppo_default" and any(abs(d) > 1e-12 for d in diffs) else None
        strategies[s] = {
            "per_seed": s in PER_SEED,
            "mean_sharpe": float(np.mean(split_mean)),
            "std_over_runs": float(np.std(allv)),
            "per_split_mean": split_mean,
            "beats_buy_and_hold": int(sum(m > baselines[i]["buy_and_hold"] for m, i in zip(split_mean, holdout))),
            "vs_ppo_default": None if s == "ppo_default" else {
                "splits_better": int(sum(d > 0 for d in diffs)), "wilcoxon_p": p},
            "dsr_median": float(np.median(dsr)), "dsr_max": float(np.max(dsr)),
            "dsr_conservative_max": float(np.max(dsr_cons)),
        }
    return {"strategies": strategies, "n_trials_holdout": n_series,
            "n_trials_conservative": n_series + n_tuning_configs,
            "baselines_mean": {k: float(np.mean([baselines[i][k] for i in holdout]))
                               for k in ("buy_and_hold", "equal_weight")},
            "baselines_per_split": {str(i): baselines[i] for i in holdout}}


# --------------------------------------------------------------------- report
LABELS = {"ppo_default": "PPO (default)", "ppo_tuned": "PPO (tuned)", "a2c_tuned": "A2C (tuned)",
          "sac": "SAC", "td3": "TD3", "seed_avg_ppo_default": "Seed avg: PPO default",
          "seed_avg_ppo_tuned": "Seed avg: PPO tuned", "blend_equal": "Blend: 4 algorithms",
          "blend_all": "Blend: all 20 models", "blend_weighted": "Blend: validation-weighted",
          "selection": "Selection (Yang)"}


def write_report(summary: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = rep / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 1-2: tuning, every configuration's development score, default highlighted.
    for n, algo in enumerate(("ppo", "a2c"), start=1):
        t = summary["tuning"].get(algo)
        if not t:
            continue
        rank = t["ranking"]
        fig, ax = plt.subplots(figsize=(9, 4))
        colours = ["#FF9500" if r["config_id"] == 0 else ("#34C759" if r["config_id"] == t["best_config_id"]
                                                           else "#8E8E93") for r in rank]
        ax.bar(range(len(rank)), [r["mean_sharpe"] for r in rank], color=colours)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(rank)), [str(r["config_id"]) for r in rank], fontsize=7)
        ax.set_xlabel("configuration id (sorted by score)")
        ax.set_ylabel("mean Sharpe, development splits")
        ax.set_title(f"{algo.upper()} tuning: green = chosen, orange = SB3 defaults")
        save(fig, f"fig{n}_tuning_{algo}.png")

    s = summary["strategies"]
    names = list(STRATEGIES)
    bh, ew = summary["baselines_mean"]["buy_and_hold"], summary["baselines_mean"]["equal_weight"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    vals = [s[k]["mean_sharpe"] for k in names]
    ax.barh(range(len(names)), vals, color=["#007AFF" if v > bh else "#8E8E93" for v in vals])
    ax.axvline(bh, color="#FF9500", ls="--", label=f"buy-and-hold {bh:.2f}")
    ax.axvline(ew, color="#AF52DE", ls=":", label=f"equal weight {ew:.2f}")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(names)), [LABELS[k] for k in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean Sharpe over the {len(summary['holdout_splits'])} holdout splits")
    ax.set_title("Holdout: every strategy against the passive baselines (blue = above buy-and-hold)")
    ax.legend(loc="lower right")
    save(fig, "fig3_holdout_strategies.png")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    splits = list(range(len(summary["holdout_splits"])))
    xs = [summary["holdout_splits"][j] + 1 for j in splits]
    for k, style in (("ppo_default", "o-"), ("ppo_tuned", "s-"), ("seed_avg_ppo_tuned", "^-"), ("blend_all", "d-")):
        ax.plot(xs, s[k]["per_split_mean"], style, label=LABELS[k], ms=4)
    ax.plot(xs, [summary["baselines_per_split"][str(i)]["buy_and_hold"] for i in summary["holdout_splits"]],
            "k--", label="buy-and-hold")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xlabel("walk-forward split (holdout)")
    ax.set_ylabel("Sharpe (seed mean where per seed)")
    ax.set_title("Holdout, split by split")
    ax.legend(fontsize=8, ncol=3)
    save(fig, "fig4_holdout_per_split.png")

    (rep / "REPORT.md").write_text(report_text(summary))
    written.append(rep / "REPORT.md")
    return written


def report_text(summary: dict) -> str:
    s, t = summary["strategies"], summary["tuning"]
    bh = summary["baselines_mean"]["buy_and_hold"]
    n = len(summary["holdout_splits"])
    rows = "\n".join(
        f"| {LABELS[k]} | {v['mean_sharpe']:.3f} | {v['std_over_runs']:.3f} | {v['beats_buy_and_hold']}/{n} | "
        + (f"{v['vs_ppo_default']['splits_better']}/{n} (p={v['vs_ppo_default']['wilcoxon_p']:.3f})"
           if v["vs_ppo_default"] and v["vs_ppo_default"]["wilcoxon_p"] is not None else "—")
        + f" | {v['dsr_max']:.3f} | {v['dsr_conservative_max']:.3f} |"
        for k, v in s.items())
    tuning = "\n".join(
        f"- **{a.upper()}**: chose configuration {r['best_config_id']} "
        f"(development mean Sharpe {r['ranking'][0]['mean_sharpe']:.3f}); SB3's defaults ranked "
        f"{r['default_rank']} of {len(r['ranking'])}. Chosen settings: `{json.dumps(r['best_config'])}`"
        for a, r in t.items())
    best = max(s, key=lambda k: s[k]["mean_sharpe"])
    return f"""# Tuning and ensembles: report

Generated by `scripts/tuning_ensembles.py`. Development splits {summary['dev_splits'][0] + 1}-{summary['dev_splits'][-1] + 1}
(evaluation windows up to {summary['last_dev_day']}); holdout splits {summary['holdout_splits'][0] + 1}-{summary['holdout_splits'][-1] + 1}
(from {summary['first_holdout_day']}). Nothing chosen in development saw a holdout day.

## Phase 1: tuning (development only)

{tuning}

Configurations are scored by their mean Sharpe over every development split and seed, the
typical outcome rather than the best seed. Figures 1-2 show every configuration.

## Phases 2-3: holdout

| Strategy | Mean Sharpe | Spread (std) | Beats buy-and-hold | Better than default PPO (Wilcoxon p) | DSR max | DSR max, conservative |
|---|---|---|---|---|---|---|
{rows}

Buy-and-hold averages **{bh:.3f}** over the same {n} holdout windows; equal weight
{summary['baselines_mean']['equal_weight']:.3f}. The highest mean Sharpe is **{LABELS[best]}**
({s[best]['mean_sharpe']:.3f}).

How to read it:
- **Mean Sharpe** is per split (seed mean for per-seed strategies), averaged over the {n} holdout splits.
- **Spread** is the standard deviation over every run. Seed averaging and blending exist to shrink it.
- **Wilcoxon p** compares each strategy with default PPO split by split; with {n} pairs, only a
  large and consistent difference can reach p < 0.05.
- **DSR** (Bailey & López de Prado 2014) counts {summary['n_trials_holdout']} holdout series as trials; the
  conservative column adds the {summary['n_trials_conservative'] - summary['n_trials_holdout']} tuning configurations. Above 0.5 would start to
  suggest real skill.
- The strategies were fixed in the script before any holdout result existed, so choosing the
  best of them afterwards is itself a selection: quote the DSR, not the best mean.

Framing (CLAUDE.md §1): whatever these numbers show is evidence about the learner inside the CT
loop, not a claim that any strategy is profitable.
"""


# --------------------------------------------------------------------- main
def run_experiment(frame: pd.DataFrame, tickers: tuple[str, ...], all_windows: list[dict], args, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    dev, holdout = dev_and_holdout(all_windows, args.dev_splits)

    # Phase 1: tuning.
    configs = {a: sample_configs(a, args.configs, seed=args.search_seed) for a in args.tune}
    tune_path = out / "tune.jsonl"
    tune_runs = _load_jsonl(tune_path)
    done = {(r["algorithm"], r["config_id"], r["split"], r["seed"]) for r in tune_runs}
    jobs = [j for j in tune_jobs(all_windows, dev, configs, args.tune_seeds)
            if (j["algorithm"], j["config_id"], j["split"], j["seed"]) not in done]
    print(f"phase 1: {len(done)} tuning runs done, {len(jobs)} to run", flush=True)
    _drain(_run_job, jobs, args, frame, tickers, tune_path, tune_runs)
    tuning = {}
    for a in args.tune:
        tuning[a] = pick_best(tune_runs, a)
        tuning[a]["best_config"] = configs[a][tuning[a]["best_config_id"]]
        tuning[a]["configs"] = configs[a]
    chosen = {"ppo_default": DEFAULTS["ppo"], "ppo_tuned": tuning.get("ppo", {}).get("best_config", DEFAULTS["ppo"]),
              "a2c_tuned": tuning.get("a2c", {}).get("best_config", DEFAULTS["a2c"]),
              "sac": DEFAULTS["sac"], "td3": DEFAULTS["td3"]}

    # Phase 2: holdout members.
    hold_path = out / "holdout.jsonl"
    hold_runs = _load_jsonl(hold_path)
    done = {(r["member"], r["split"], r["seed"]) for r in hold_runs}
    jobs = [{"phase": "holdout", "member": m, "algorithm": MEMBERS[m], "config": chosen[m], "split": i, "seed": k,
             "windows": all_windows[i], "model_path": str(out / "models" / f"{m}_split{i + 1}_seed{k}.zip")}
            for m in MEMBERS for i in holdout for k in range(args.holdout_seeds) if (m, i, k) not in done]
    jobs.sort(key=lambda j: j["algorithm"] not in ("sac", "td3"))  # slowest first
    print(f"phase 2: {len(done)} holdout runs done, {len(jobs)} to run", flush=True)
    _drain(_run_job, jobs, args, frame, tickers, hold_path, hold_runs)

    # Phase 3: ensembles from the saved members.
    ens_path = out / "ensembles.jsonl"
    ens = _load_jsonl(ens_path)
    by = {(r["member"], r["split"], r["seed"]): r for r in hold_runs}
    jobs = [{"split": i, "windows": all_windows[i],
             "runs": {(m, k): {"model_path": by[(m, i, k)]["model_path"], "algorithm": MEMBERS[m],
                               "val_sharpe": sharpe_ratio(np.asarray(by[(m, i, k)]["val_returns"]))}
                      for m in ("ppo_default",) + BLEND_MEMBERS for k in range(args.holdout_seeds)}}
            for i in holdout if i not in {e["split"] for e in ens}]
    print(f"phase 3: {len(ens)} splits done, {len(jobs)} to run", flush=True)
    _drain(_ensemble_job, jobs, args, frame, tickers, ens_path, ens)

    baselines = {}
    for i in holdout:
        _init_worker(frame, tickers)
        _, _, eval_df = _frames(all_windows[i])
        env = TradingEnvironment(eval_df, tickers=tickers)
        baselines[i] = {"buy_and_hold": sharpe_ratio(run_policy(env, buy_and_hold)["returns"]),
                        "equal_weight": sharpe_ratio(run_policy(env, equal_weight_rebalanced)["returns"])}
    returns = strategy_returns(hold_runs, ens, holdout, args.holdout_seeds)
    summary = summarise(returns, baselines, holdout, sum(len(c) for c in configs.values()))
    summary.update(
        tuning={a: {k: v for k, v in t.items() if k != "configs"} for a, t in tuning.items()},
        dev_splits=dev, holdout_splits=holdout,
        last_dev_day=str(max(all_windows[i]["eval_end"] for i in dev).date()),
        first_holdout_day=str(min(all_windows[i]["eval_start"] for i in holdout).date()),
        config={k: v for k, v in vars(args).items() if k != "report_only"},
        weighted_blend_weights={str(e["split"]): e["weighted_weights"] for e in ens},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _drain(fn, jobs, args, frame, tickers, path: Path, sink: list) -> None:
    if not jobs:
        return
    started = time.time()
    with path.open("a") as fh:
        for n, r in enumerate(_map(fn, jobs, args.workers, frame, tickers), start=1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            sink.append(r)
            if n % 20 == 0 or n == len(jobs):
                print(f"  [{time.time() - started:.0f}s] {n}/{len(jobs)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning and decision ensembles")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--tune", nargs="*", default=["ppo", "a2c"], choices=sorted(SPACES))
    parser.add_argument("--configs", type=int, default=24, help="configurations per tuned algorithm, defaults included")
    parser.add_argument("--tune-seeds", type=int, default=3)
    parser.add_argument("--holdout-seeds", type=int, default=5)
    parser.add_argument("--dev-splits", type=int, default=9)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--eval-days", type=int, default=180)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-only", action="store_true", help="rebuild <out>/report/ from summary.json")
    args = parser.parse_args()
    out = Path(args.out)

    if args.report_only:
        path = out / "summary.json"
        if not path.exists():
            print(f"--report-only needs {path}", file=sys.stderr)
            return 1
        print("\n".join(f"wrote {p}" for p in write_report(json.loads(path.read_text()), out)))
        return 0

    from src.dataops.repository import MarketRepository

    raw = MarketRepository().load_partition(DATA.train_start, DATA.eval_end, tickers=DATA.tickers)
    if raw.empty:
        print("no data in the store; run scripts/run_ingestion.py first", file=sys.stderr)
        return 1
    frame = normalize_rolling(raw)
    splits = walk_forward_splits(DATA.train_start, DATA.eval_end, train_days=args.train_days, eval_days=args.eval_days)
    summary = run_experiment(frame, DATA.tickers, [windows(s, args.val_days) for s in splits], args, out)
    for k, v in summary["strategies"].items():
        print(f"{LABELS[k]:<28} mean {v['mean_sharpe']:>7.3f}  "
              f"beats B&H {v['beats_buy_and_hold']}/{len(summary['holdout_splits'])}  "
              f"DSR max {v['dsr_max']:.3f}")
    print(f"buy-and-hold {summary['baselines_mean']['buy_and_hold']:.3f}")
    print("\n".join(f"wrote {p}" for p in write_report(summary, out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
