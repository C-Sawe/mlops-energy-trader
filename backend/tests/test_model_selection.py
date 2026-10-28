"""How the experiment scripts pick "the best" model.

- `scripts/train_agent.py` registers the best seed per walk-forward split.
- `scripts/algorithm_comparison.py` builds the selection ensemble and tests
  each algorithm against PPO (Wilcoxon signed-rank, deflated Sharpe).

The property that matters most is I1: a choice must never be made on the
window it is then scored on. So the ensemble is checked with results where
the validation and evaluation windows disagree: a selector that peeked at
evaluation would pick the other algorithm. No real training: evaluation is
stubbed or the saved-run format is built by hand.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.dataops.models import ModelRun, ModelVersion
from src.orchestration.evaluator import sharpe_ratio
from src.rlops.agent import PPOAgent
from tests.test_ct_orchestrator import _seed_repo

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _returns(mean: float, seed: int, n: int = 40) -> list[float]:
    return list(np.random.default_rng(seed).normal(mean, 0.01, n))


GOOD, BAD = 0.004, -0.004


# --------------------------------------------------------------- algorithm_comparison.py
def _runs(spec: dict[str, tuple[float, float]], n_splits: int, seeds: int) -> dict:
    """Saved-run records in the script's checkpoint format. `spec` maps an
    algorithm to (validation mean return, evaluation mean return)."""
    runs = {}
    for a, (algo, (val_mean, eval_mean)) in enumerate(spec.items()):
        for s in range(n_splits):
            for k in range(seeds):
                base = 1000 * a + 10 * s + k  # distinct noise per run
                runs[(algo, s, k)] = {
                    "val_returns": _returns(val_mean, base),
                    "eval_returns": _returns(eval_mean, base + 500),
                }
    return runs


def test_ensemble_chooses_on_validation_never_on_evaluation():
    """PPO looks best on validation but loses on evaluation; A2C is the
    reverse. Choosing on validation must pick PPO every time. Picking A2C
    would mean the evaluation window leaked into the choice (I1)."""
    cmp = _load("algorithm_comparison")
    n_splits, seeds = 6, 2
    runs = _runs({"ppo": (GOOD, BAD), "a2c": (BAD, GOOD)}, n_splits, seeds)

    summary = cmp._summarise(runs, ["ppo", "a2c"], n_splits, seeds)

    assert summary["ensemble_picks"] == {"ppo": n_splits * seeds, "a2c": 0}
    ensemble = summary["strategies"]["ensemble"]
    assert ensemble["per_split_seed_mean"] == summary["strategies"]["ppo"]["per_split_seed_mean"]
    assert ensemble["vs_ppo"]["wilcoxon_p"] == 1.0  # identical to PPO on every split


def test_wilcoxon_flags_a_consistent_winner_and_not_a_copy():
    cmp = _load("algorithm_comparison")
    n_splits, seeds = 6, 2
    runs = _runs({"ppo": (GOOD, BAD), "a2c": (GOOD, GOOD)}, n_splits, seeds)
    runs.update({("sac", s, k): runs[("ppo", s, k)] for s in range(n_splits) for k in range(seeds)})

    summary = cmp._summarise(runs, ["ppo", "a2c", "sac"], n_splits, seeds)

    a2c = summary["strategies"]["a2c"]["vs_ppo"]
    assert a2c["splits_better"] == n_splits and a2c["mean_diff"] > 0
    assert a2c["wilcoxon_p"] < 0.05  # 6 of 6 splits: the smallest two-sided p is 0.031
    sac = summary["strategies"]["sac"]["vs_ppo"]
    assert sac == {"splits_better": 0, "mean_diff": 0.0, "wilcoxon_p": 1.0}


def test_best_per_split_is_the_top_seed_and_deflation_counts_every_trial():
    cmp = _load("algorithm_comparison")
    n_splits, seeds = 3, 4
    runs = _runs({"ppo": (GOOD, GOOD)}, n_splits, seeds)

    summary = cmp._summarise(runs, ["ppo"], n_splits, seeds)

    expected_best = np.mean([
        max(sharpe_ratio(np.asarray(runs[("ppo", s, k)]["eval_returns"])) for k in range(seeds))
        for s in range(n_splits)
    ])
    assert summary["strategies"]["ppo"]["best_per_split_mean"] == pytest.approx(expected_best)
    assert summary["n_trials"] == 2 * n_splits * seeds  # PPO and the ensemble, every seed, every split


def test_validation_slice_is_inside_training_and_before_evaluation():
    cmp = _load("algorithm_comparison")
    split = {
        "train_start": pd.Timestamp("2020-01-01"), "train_end": pd.Timestamp("2021-12-31"),
        "eval_start": pd.Timestamp("2022-01-01"), "eval_end": pd.Timestamp("2022-06-30"),
    }
    w = cmp._windows(split, val_days=90)

    assert w["fit_start"] == split["train_start"]
    assert w["fit_end"] < w["val_start"] <= w["val_end"] == split["train_end"]
    assert w["val_end"] < w["eval_start"]  # validation never touches the evaluation window


# --------------------------------------------------------------- train_agent.py
def test_sweep_registers_the_best_seed_of_each_split(tmp_path, monkeypatch):
    """Seed 1 is made the best on every split's evaluation window; the
    version registered for each split must be seed 1's run."""
    train_agent = _load("train_agent")
    repo, last_day = _seed_repo(tmp_path, n_days=400)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", str(repo.engine.url))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns.db'}")
    monkeypatch.setattr(
        train_agent, "DATA",
        SimpleNamespace(tickers=("XOM", "CVX", "SHEL", "BP", "NEE"), train_start="2020-01-01", eval_end=str(last_day)),
    )
    monkeypatch.setattr(PPOAgent, "train", lambda self, total_timesteps: self)

    mean_by_seed = {0: 0.0005, 1: 0.003, 2: -0.001}

    def fake_evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        r = np.asarray(_returns(mean_by_seed[seed], seed))
        return [{"returns": r, "equity_curve": 100_000 * np.cumprod(np.r_[1.0, 1 + r])}]

    monkeypatch.setattr(PPOAgent, "evaluate", fake_evaluate)
    monkeypatch.setattr("sys.argv", ["train_agent.py", "--seeds", "3", "--train-days", "250",
                                     "--eval-days", "90", "--n-steps", "64"])

    assert train_agent.main() == 0

    with repo.session() as session:
        versions = session.query(ModelVersion).all()
        seeds = [session.get(ModelRun, v.run_id).hyperparameters["seed"] for v in versions]
        runs = session.query(ModelRun).count()
    assert len(versions) >= 2  # several splits, one registered version each
    assert seeds == [1] * len(versions)
    assert runs == 3 * len(versions)  # every seed is logged, not only the winner


# --------------------------------------------------------------- the comparison's report
def _report_summary(cmp, runs, algorithms, n_splits, seeds):
    summary = cmp._summarise(runs, algorithms, n_splits, seeds)
    summary["config"] = {"algorithms": algorithms, "seeds": seeds, "splits": n_splits,
                         "timesteps": 100, "val_days": 90, "eval_days": 180}
    summary["baselines_split_mean"] = {"buy_and_hold": 0.5}
    start = pd.Timestamp("2020-01-01")
    summary["windows"] = [
        {k: str(v.date()) for k, v in cmp._windows({
            "train_start": start + pd.Timedelta(days=180 * i), "train_end": start + pd.Timedelta(days=180 * i + 729),
            "eval_start": start + pd.Timedelta(days=180 * i + 730), "eval_end": start + pd.Timedelta(days=180 * i + 909),
        }, 90).items()}
        for i in range(n_splits)
    ]
    return summary


REPORT_FILES = {"1_windows.png", "2_sharpe_by_strategy.png", "3_best_vs_typical.png", "4_vs_ppo_by_split.png",
                "5_validation_vs_evaluation.png", "6_ensemble_picks.png", "7_deflated_sharpe.png", "REPORT.md"}


def test_report_writes_every_chart_and_explains_this_runs_numbers(tmp_path):
    cmp = _load("algorithm_comparison")
    n_splits, seeds = 6, 2
    runs = _runs({"ppo": (GOOD, BAD), "a2c": (BAD, GOOD)}, n_splits, seeds)

    written = cmp._write_report(runs, _report_summary(cmp, runs, ["ppo", "a2c"], n_splits, seeds), tmp_path)

    assert {p.name for p in written} == REPORT_FILES
    assert all(p.stat().st_size > 0 for p in written)
    text = (tmp_path / "report" / "REPORT.md").read_text()
    assert "| A2C |" in text and "6/6 (p = 0.03)" in text  # A2C beat PPO on every split
    assert "PPO 12, A2C 0" in text  # the ensemble's picks, as chosen on validation
    assert "Validation scores did not carry over" in text  # validation and evaluation disagree here


def test_report_only_rebuilds_from_saved_files_without_the_database(tmp_path, monkeypatch):
    cmp = _load("algorithm_comparison")
    n_splits, seeds = 4, 2
    runs = _runs({"ppo": (GOOD, GOOD), "a2c": (GOOD, BAD)}, n_splits, seeds)
    with (tmp_path / "runs.jsonl").open("w") as fh:
        for (algo, s, k), r in runs.items():
            fh.write(json.dumps({"algorithm": algo, "split": s, "seed": k, **r}) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps(_report_summary(cmp, runs, ["ppo", "a2c"], n_splits, seeds)))

    def no_database(*args, **kwargs):
        raise AssertionError("--report-only must not touch the database")

    monkeypatch.setattr(cmp, "MarketRepository", no_database)
    monkeypatch.setattr("sys.argv", ["algorithm_comparison.py", "--out", str(tmp_path), "--report-only"])

    assert cmp.main() == 0
    assert {p.name for p in (tmp_path / "report").iterdir()} == REPORT_FILES
