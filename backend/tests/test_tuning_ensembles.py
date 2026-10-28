"""Tuning and ensembles (`scripts/tuning_ensembles.py`).

What must hold for the results to mean anything:
- tuning only ever sees development splits, and the holdout starts after
  every development day (I1);
- a configuration is chosen by its typical (mean) score, and SB3's defaults
  are kept unless something beats them;
- ensembles combine decisions (averaged target weights), and the weighted
  blend and the selection ensemble decide on validation, never evaluation.

No network or database beyond a temporary SQLite file; the end-to-end run
uses tiny training budgets.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import DATA
from src.dataops.processing import normalize_rolling, walk_forward_splits
from tests.test_ct_orchestrator import _seed_repo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tuning_ensembles.py"


@pytest.fixture(scope="module")
def te():
    spec = importlib.util.spec_from_file_location("tuning_ensembles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- configurations
def test_search_starts_from_the_defaults_and_samples_valid_distinct_configs(te):
    configs = te.sample_configs("ppo", 24, seed=0)
    assert configs[0] == te.DEFAULTS["ppo"]
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == 24
    assert all(c["batch_size"] <= c["n_steps"] for c in configs)
    assert te.sample_configs("ppo", 24, seed=0) == configs  # a resumed run samples the same ones
    kwargs, steps = te.to_kwargs(configs[0])
    assert steps == 20_000 and "timesteps" not in kwargs
    assert kwargs["policy_kwargs"] == {"net_arch": [64, 64]}


# --------------------------------------------------------------- development vs holdout
def test_the_real_calendar_gives_a_holdout_strictly_after_development(te):
    splits = walk_forward_splits(DATA.train_start, DATA.eval_end, train_days=730, eval_days=180)
    w = [te.windows(s, 90) for s in splits]
    dev, holdout = te.dev_and_holdout(w, 9)
    assert (dev, holdout) == (list(range(9)), list(range(9, 18)))
    assert min(w[i]["eval_start"] for i in holdout) > max(w[i]["eval_end"] for i in dev)
    jobs = te.tune_jobs(w, dev, {"ppo": te.sample_configs("ppo", 3)}, seeds=2)
    assert {j["split"] for j in jobs} == set(dev) and len(jobs) == 3 * 9 * 2


def test_an_overlapping_holdout_is_refused(te):
    day = pd.Timestamp
    w = [{"eval_start": day("2020-01-01"), "eval_end": day("2020-06-30")},
         {"eval_start": day("2020-06-01"), "eval_end": day("2020-12-31")}]
    with pytest.raises(ValueError, match="overlaps"):
        te.dev_and_holdout(w, 1)


# --------------------------------------------------------------- choosing a configuration
def _runs(algorithm, scores: dict[int, list[float]]):
    rng = np.random.default_rng(0)
    out = []
    for cid, means in scores.items():
        for k, m in enumerate(means):
            out.append({"algorithm": algorithm, "config_id": cid, "split": k, "seed": 0,
                        "eval_returns": list(rng.normal(m, 0.01, 60))})
    return out


def test_the_best_typical_score_wins_not_the_best_single_run(te):
    # Config 1 has the single best run but a poor average; config 2 is
    # steadily good. Choosing on the best run would pick 1.
    runs = _runs("ppo", {0: [0.0, 0.0, 0.0], 1: [0.006, -0.004, -0.004], 2: [0.002, 0.002, 0.002]})
    best = te.pick_best(runs, "ppo")
    assert best["best_config_id"] == 2
    assert best["default_rank"] in (2, 3)


def test_the_defaults_stay_when_nothing_beats_them(te):
    same = list(np.random.default_rng(1).normal(0.001, 0.01, 60))
    runs = [{"algorithm": "a2c", "config_id": c, "split": 0, "seed": 0, "eval_returns": same} for c in (2, 0, 1)]
    assert te.pick_best(runs, "a2c")["best_config_id"] == 0


# --------------------------------------------------------------- ensembles
class Const:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, obs):
        return self.action


def test_a_blend_averages_decisions_and_stays_in_the_action_space(te):
    act = te.blend([Const([1.0, -1.0, 0.2]), Const([0.0, -1.0, 0.4])])
    assert act(None) == pytest.approx([0.5, -1.0, 0.3])
    weighted = te.blend([Const([1.0, 1.0, 1.0]), Const([-1.0, -1.0, -1.0])], [3, 1])
    assert weighted(None) == pytest.approx([0.5, 0.5, 0.5])
    assert te.blend([Const([0.9] * 3), Const([0.9] * 3)], [1, 1])(None).max() <= 1.0


def test_validation_weights_ignore_losers_and_fall_back_to_equal(te):
    assert te.validation_weights([2.0, -1.0, 2.0]) == pytest.approx([0.5, 0.0, 0.5])
    assert te.validation_weights([-1.0, -2.0]) == pytest.approx([0.5, 0.5])


def test_selection_chooses_on_validation_never_on_evaluation(te):
    # A2C looks best on validation but worst on evaluation; a selector that
    # peeked at evaluation would choose another member.
    rng = np.random.default_rng(3)
    val = {"ppo_tuned": -0.002, "a2c_tuned": 0.004, "sac": -0.001, "td3": 0.0, "ppo_default": 0.0}
    ev = {"ppo_tuned": 0.004, "a2c_tuned": -0.004, "sac": 0.003, "td3": 0.002, "ppo_default": 0.0}
    runs = [{"member": m, "split": 0, "seed": 0, "val_returns": list(rng.normal(val[m], 0.005, 60)),
             "eval_returns": list(rng.normal(ev[m], 0.005, 60))} for m in te.MEMBERS]
    out = te.strategy_returns(runs, [], [0], seeds=1)
    a2c = next(r for r in runs if r["member"] == "a2c_tuned")
    assert np.array_equal(out["selection"][0][0], np.asarray(a2c["eval_returns"]))


# --------------------------------------------------------------- end to end
def test_a_tiny_run_trains_tunes_blends_reports_and_resumes(te, tmp_path, monkeypatch):
    tiny = {"timesteps": 64}
    monkeypatch.setattr(te, "DEFAULTS", {
        "ppo": {**te.DEFAULTS["ppo"], "n_steps": 32, "batch_size": 16, **tiny},
        "a2c": {**te.DEFAULTS["a2c"], **tiny}, "sac": tiny, "td3": tiny})
    monkeypatch.setattr(te, "SPACES", {**te.SPACES, "ppo": {
        **te.SPACES["ppo"], "n_steps": [32, 64], "batch_size": [16], "timesteps": [64], "n_epochs": [1]}})
    repo, _ = _seed_repo(tmp_path, n_days=420)
    frame = normalize_rolling(repo.load_partition("2000-01-01", "2100-01-01", tickers=DATA.tickers))
    dates = frame["date"]
    splits = walk_forward_splits(str(dates.min().date()), str(dates.max().date()), train_days=200, eval_days=60)
    w = [te.windows(s, 40) for s in splits][:3]
    args = SimpleNamespace(dev_splits=1, configs=2, search_seed=0, tune=["ppo"], tune_seeds=1,
                           holdout_seeds=2, workers=1)
    out = tmp_path / "exp"

    summary = te.run_experiment(frame, DATA.tickers, w, args, out)
    assert set(summary["strategies"]) == set(te.STRATEGIES)
    assert summary["tuning"]["ppo"]["best_config_id"] in (0, 1)
    assert len(list((out / "models").glob("*.zip"))) == len(te.MEMBERS) * 2 * 2  # members x holdout x seeds
    assert all(np.isfinite(s["mean_sharpe"]) for s in summary["strategies"].values())
    written = te.write_report(summary, out)
    assert (out / "report" / "REPORT.md") in written
    assert "Nothing chosen in development saw a holdout day" in (out / "report" / "REPORT.md").read_text()

    lines = {p: len(p.read_text().splitlines()) for p in out.glob("*.jsonl")}
    te.run_experiment(frame, DATA.tickers, w, args, out)  # resumes: nothing retrained
    assert {p: len(p.read_text().splitlines()) for p in out.glob("*.jsonl")} == lines
