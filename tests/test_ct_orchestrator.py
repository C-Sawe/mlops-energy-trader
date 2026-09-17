"""Tests for the CT orchestrator (FR-13, FR-14, FR-15, FR-17, FR-20).

Two different testing strategies are used deliberately:
  - The "no incumbent" scenarios exercise the real train -> evaluate ->
    promote pipeline end to end (tiny hyperparameters, but genuinely
    trained), because the outcome (anything beats nothing) is deterministic
    regardless of what the tiny model actually learns.
  - The acceptance-gate scenarios (FR-17) monkeypatch `PPOAgent.evaluate` to
    return fixed equity curves, because "does the candidate's model happen
    to outperform the incumbent's" is not something a few hundred timesteps
    can be made to guarantee either way — the orchestrator's promote/retain
    *decision* is what FR-17 is about, not whether training got lucky.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import DATA
from src.dataops import processing as P
from src.dataops.repository import MarketRepository
from src.orchestration.ct_orchestrator import CTOrchestrator, CTStatus
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment
from src.rlops.registry import ModelRegistry
from src.serving.inference import InferenceService
from tests.test_processing import make_series


def _seed_repo(tmp_path, n_days: int = 400, final_vix: float = 20.0):
    """A repo with `n_days` of synthetic history for the full universe,
    anchored so the caller can use the data's own last date as `as_of` —
    the orchestrator's lookback queries only find data that's actually
    there, and this project's fixtures don't reach up to whatever the real
    calendar date happens to be when the suite runs.

    Deliberately file-based, not `sqlite:///:memory:`: an in-memory SQLite
    database is not shared across connections, and the CT orchestrator's
    whole point is a background thread using a *different* connection than
    the caller's — with `:memory:` that thread sees a separate, empty
    database ("no such table"), not the one just seeded. This is purely a
    test-fixture concern; real deployments use Postgres or a file (both
    genuinely shared across connections), never `:memory:`, for exactly
    this reason if nothing else.
    """
    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix_values = np.linspace(14, 20, len(dates))
    vix_values[-1] = final_vix
    vix = pd.DataFrame({"date": dates, "close": vix_values})
    frame = P.build_feature_frame(equities, vix)

    repo = MarketRepository(url=f"sqlite:///{tmp_path / 'test.db'}")
    repo.create_schema()
    repo.persist(frame)
    return repo, dates.max().date()


def _promote_initial_model(repo, tmp_path, monkeypatch, as_of) -> str:
    monkeypatch.chdir(tmp_path)
    frame = P.normalize_rolling(repo.load_partition(pd.Timestamp(as_of) - pd.Timedelta(days=399), as_of))
    train_env = TradingEnvironment(frame, tickers=DATA.tickers)
    agent = PPOAgent(train_env, n_steps=64, seed=0).train(total_timesteps=128)

    registry = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)
    run_id = registry.log_run(
        agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", {"sharpe_ratio": 0.0}
    )
    from src.dataops.models import ModelRun

    with repo.session() as session:
        mlflow_ref = session.get(ModelRun, run_id).mlflow_run_ref
    version_id = registry.register_version(run_id, artifact_uri=f"runs:/{mlflow_ref}/model")
    repo.promote_version(version_id)
    return version_id


def _fast_orchestrator(inference_service, repo, tmp_path, monkeypatch) -> CTOrchestrator:
    monkeypatch.chdir(tmp_path)
    registry = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)
    return CTOrchestrator(
        inference_service,
        repo=repo,
        registry=registry,
        retrain_train_days=200,
        retrain_timesteps=128,
        retrain_n_steps=64,
    )


# --------------------------------------------------------------- FR-14, FR-15
def test_evaluate_triggers_retrain_when_no_incumbent(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    assert service.active_version_id is None

    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    status = orchestrator.evaluate(as_of=as_of)

    assert status == CTStatus.RETRAINING
    orchestrator._retrain_thread.join(timeout=60)

    assert orchestrator.status == CTStatus.SERVING
    assert service.active_version_id is not None  # something got promoted


def test_evaluate_is_non_blocking(tmp_path, monkeypatch):
    """FR-15: the incumbent keeps serving throughout retraining — the call
    that *triggers* retraining must return immediately, not wait for it."""
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    start = time.perf_counter()
    orchestrator.evaluate(as_of=as_of)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0  # the full retrain alone takes much longer than this
    orchestrator._retrain_thread.join(timeout=60)  # let it finish before the test ends


def test_double_evaluate_does_not_spawn_a_second_retrain(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=as_of)
    first_thread = orchestrator._retrain_thread
    status_again = orchestrator.evaluate(as_of=as_of)

    assert status_again == CTStatus.RETRAINING
    assert orchestrator._retrain_thread is first_thread  # no second thread spawned

    first_thread.join(timeout=60)


# --------------------------------------------------------------- FR-13, FR-18
def test_evaluate_persists_snapshots_and_computes_rolling_sharpe(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=as_of)
    if orchestrator._retrain_thread is not None:
        orchestrator._retrain_thread.join(timeout=60)

    assert orchestrator.last_rolling_sharpe is not None
    assert np.isfinite(orchestrator.last_rolling_sharpe)
    assert orchestrator.last_evaluated_at is not None

    snapshots = repo.list_snapshots(pd.Timestamp(as_of) - pd.Timedelta(days=60), as_of)
    assert not snapshots.empty


def test_evaluate_handles_no_data_in_window_gracefully(tmp_path, monkeypatch):
    """Found via the dashboard: evaluate() defaults `as_of` to real
    `date.today()`, but a repo whose data doesn't reach that far (e.g.
    ingestion is behind, or — as here — synthetic fixture data anchored to
    a fixed range) must not crash the request that triggered it. This is
    the same "no such data yet" condition `load_partition`'s correctly-
    shaped empty frame exists for (see test_repository.py); this test
    exercises it through the orchestrator specifically."""
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    far_future = (pd.Timestamp(as_of) + pd.Timedelta(days=3650)).date()
    status = orchestrator.evaluate(as_of=far_future)

    assert status == CTStatus.SERVING
    assert orchestrator.last_rolling_sharpe is None  # never computed, not crashed


# --------------------------------------------------------------- FR-17
def test_retrain_promotes_when_candidate_beats_incumbent(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    incumbent_version = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    losing = {"equity_curve": np.array([100_000.0, 99_000.0, 98_000.0]), "returns": np.array([-0.01, -0.0101])}
    winning = {"equity_curve": np.array([100_000.0, 101_000.0, 102_000.0]), "returns": np.array([0.01, 0.0099])}

    calls = {"n": 0}

    def fake_evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        calls["n"] += 1
        return [winning] if calls["n"] == 1 else [losing]  # candidate first, then incumbent

    monkeypatch.setattr(PPOAgent, "evaluate", fake_evaluate)

    orchestrator._retrain_and_maybe_promote(as_of)

    assert service.active_version_id != incumbent_version


def test_retrain_retains_incumbent_when_candidate_does_not_beat_it(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    incumbent_version = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    losing = {"equity_curve": np.array([100_000.0, 99_000.0, 98_000.0]), "returns": np.array([-0.01, -0.0101])}
    winning = {"equity_curve": np.array([100_000.0, 101_000.0, 102_000.0]), "returns": np.array([0.01, 0.0099])}

    calls = {"n": 0}

    def fake_evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        calls["n"] += 1
        return [losing] if calls["n"] == 1 else [winning]  # candidate first, then incumbent

    monkeypatch.setattr(PPOAgent, "evaluate", fake_evaluate)

    orchestrator._retrain_and_maybe_promote(as_of)

    assert service.active_version_id == incumbent_version

    # The rejected candidate must still be logged — "how many candidates
    # were held back" has to be an answerable question, not silently
    # discarded just because it lost the acceptance gate.
    stats = repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())
    assert stats["rejected"] == 1
