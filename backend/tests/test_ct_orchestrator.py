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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import DATA, RISK
from src.dataops import processing as P
from src.dataops.models import ModelRun
from src.dataops.repository import MarketRepository
from src.orchestration.ct_orchestrator import CTOrchestrator, CTStatus, _evaluation_start
from src.orchestration.evaluator import max_drawdown
from src.rlops.agent import PPOAgent
from src.rlops.baselines import all_cash, buy_and_hold, equal_weight_rebalanced, run_policy
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
        # In-process so monkeypatching PPOAgent.train reaches the training
        # call; the subprocess path has its own dedicated test below.
        train_in_subprocess=False,
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


def test_evaluate_defers_retrain_when_vix_is_critical(tmp_path, monkeypatch):
    """A volatile week is exactly when the rolling Sharpe is most likely to
    cross the retrain trigger — training a fresh candidate on a distorted,
    high-volatility window is the opposite of what FR-14 is for. When VIX
    is at or above the same critical threshold I5/FR-12 already uses to
    force capital preservation on inference, a below-threshold Sharpe must
    defer the retrain rather than fire it."""
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold + 5.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    losing = {"equity_curve": np.array([100_000.0, 95_000.0, 90_000.0]), "returns": np.array([-0.05, -0.0526])}
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [losing])

    status = orchestrator.evaluate(as_of=as_of)

    assert status == CTStatus.SERVING
    assert orchestrator._retrain_thread is None  # never spawned
    assert orchestrator.last_retrain_deferred_at is not None
    assert orchestrator.last_rolling_sharpe is not None
    assert orchestrator.last_rolling_sharpe < RISK.target_sharpe_threshold  # the trigger condition was real


def test_evaluate_still_retrains_below_threshold_when_vix_is_normal(tmp_path, monkeypatch):
    """The deferral must be specific to volatility, not a general brake on
    FR-14 — a poor Sharpe under ordinary VIX still triggers a retrain,
    exactly as before this fail-safe existed."""
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    losing = {"equity_curve": np.array([100_000.0, 95_000.0, 90_000.0]), "returns": np.array([-0.05, -0.0526])}
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [losing])

    orchestrator.evaluate(as_of=as_of)
    if orchestrator._retrain_thread is not None:
        orchestrator._retrain_thread.join(timeout=60)

    assert orchestrator.last_retrain_deferred_at is None
    assert orchestrator._retrain_thread is not None


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


def test_retrain_survives_a_genuine_crash_and_leaves_incumbent_active(tmp_path, monkeypatch):
    """NFR-05: a *failed* retraining cycle (training itself throws, not
    "the candidate lost the comparison") must leave the incumbent serving
    and state consistent — and, distinctly from FR-17's already-tested
    rejection path, the failure itself must be visible, not just a bare
    stderr traceback from the background thread's default exception hook."""
    repo, as_of = _seed_repo(tmp_path)
    incumbent_version = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    runs_before = repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())["total_runs"]

    def exploding_train(self, total_timesteps):
        raise RuntimeError("simulated training crash")

    monkeypatch.setattr(PPOAgent, "train", exploding_train)

    orchestrator._retrain_and_maybe_promote(as_of)

    assert service.active_version_id == incumbent_version  # untouched
    assert orchestrator.status == CTStatus.SERVING  # not stuck RETRAINING
    assert orchestrator.last_retrain_failed_at is not None
    assert "simulated training crash" in orchestrator.last_retrain_error

    # A crash before a candidate exists has nothing log_run() could record
    # against — correctly no new row, not a silent swallow of a run that
    # never happened.
    runs_after = repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())["total_runs"]
    assert runs_after == runs_before


# --------------------------------------------------------------- NFR-02 (out-of-process training)
def test_candidate_and_incumbent_are_judged_on_identical_rows(tmp_path, monkeypatch):
    """The serving process loads only the rows it needs (NFR-02) while the
    worker loads two years; the gate is only fair if both give the same
    normalised evaluation rows."""
    repo, as_of = _seed_repo(tmp_path, n_days=900)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    registry = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)
    orchestrator = CTOrchestrator(service, repo=repo, registry=registry, retrain_train_days=730,
                                  retrain_timesteps=64, retrain_n_steps=64, train_in_subprocess=False)
    seen = []
    real = PPOAgent.evaluate

    def spy(self, env, *args, **kwargs):
        seen.append((env._features.copy(), env._prices.copy(), list(env._dates)))
        return real(self, env, *args, **kwargs)

    monkeypatch.setattr(PPOAgent, "evaluate", spy)
    orchestrator._trigger_retrain(as_of, "analysis")
    _join_retrain(orchestrator)
    assert orchestrator.last_retrain_failed_at is None, orchestrator.last_retrain_error

    (cand, cand_prices, cand_dates), (inc, inc_prices, inc_dates) = seen
    assert cand_dates == inc_dates and len(cand_dates) == RISK.sharpe_evaluation_window + 1
    np.testing.assert_allclose(inc, cand, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(inc_prices, cand_prices)


def test_retrain_worker_trains_in_a_genuinely_separate_process(tmp_path, monkeypatch):
    """NFR-02's fix is only real if training leaves this process: run the
    worker under the same spawn executor production uses and prove the work
    happened under a different pid, with the artifact and Sharpe coming
    back across the process boundary."""
    import concurrent.futures
    import multiprocessing
    import os

    from src.orchestration.retrain_worker import train_and_evaluate_candidate

    repo, as_of = _seed_repo(tmp_path)
    end = pd.Timestamp(as_of)
    model_path = str(tmp_path / "candidate.zip")

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        result = pool.submit(
            train_and_evaluate_candidate,
            db_url=repo.url,
            tickers=tuple(DATA.tickers),
            train_start=str((end - pd.Timedelta(days=200)).date()),
            train_end=str((end - pd.Timedelta(days=46)).date()),
            eval_start=str((end - pd.Timedelta(days=45)).date()),
            eval_end=str(end.date()),
            n_steps=64,
            timesteps=128,
            seed=0,
            model_path=model_path,
        ).result(timeout=120)

    assert result["pid"] != os.getpid()
    assert np.isfinite(result["sharpe"])
    assert result["hyperparameters"]["algorithm"] == "ppo"
    # The artifact crossed the boundary as a file the parent can serve.
    assert PPOAgent.load(model_path).predict(
        np.zeros(5 * 8 + 5 + 1, dtype=np.float32)
    ).shape == (5,)


def test_subprocess_retrain_promotes_end_to_end(tmp_path, monkeypatch):
    """The production path (train_in_subprocess=True) through the real
    bootstrap: evaluate() spawns the retrain thread, the thread spawns the
    training process, and the promoted model serves — FR-14/16/17 wiring
    unchanged by NFR-02's process boundary."""
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    registry = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)
    orchestrator = CTOrchestrator(
        service,
        repo=repo,
        registry=registry,
        retrain_train_days=200,
        retrain_timesteps=128,
        retrain_n_steps=64,
        train_in_subprocess=True,
    )

    status = orchestrator.evaluate(as_of=as_of)
    assert status == CTStatus.RETRAINING
    orchestrator._retrain_thread.join(timeout=120)

    assert orchestrator.status == CTStatus.SERVING
    assert orchestrator.last_retrain_failed_at is None, orchestrator.last_retrain_error
    assert service.active_version_id is not None  # bootstrap promoted
    # log_run recorded the trainer's hyperparameters, not a loaded agent's
    # empty dict (FR-08/NFR-07 across the process boundary).
    with repo.session() as session:
        run = session.query(ModelRun).one()
    assert run.hyperparameters["algorithm"] == "ppo"
    assert run.hyperparameters["n_steps"] == 64


# --------------------------------------------------------------- NFR-07 (trigger provenance)
def _runs_with_trigger(repo, trigger):
    with repo.session() as session:
        return session.query(ModelRun).filter(ModelRun.trigger_reason == trigger).all()


def test_bootstrap_retrain_is_recorded_as_ct_bootstrap(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=as_of)
    orchestrator._retrain_thread.join(timeout=60)

    [run] = _runs_with_trigger(repo, "ct_bootstrap")
    assert run.trigger_rolling_sharpe is None  # nothing was evaluated; there was no incumbent
    assert float(run.trigger_vix) == pytest.approx(20.0)


def test_decay_retrain_records_the_sharpe_and_vix_that_caused_it(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    losing = {"equity_curve": np.array([100_000.0, 95_000.0, 90_000.0]), "returns": np.array([-0.05, -0.0526])}
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [losing])

    orchestrator.evaluate(as_of=as_of)
    orchestrator._retrain_thread.join(timeout=60)

    [run] = _runs_with_trigger(repo, "ct_decay")
    assert float(run.trigger_rolling_sharpe) == pytest.approx(orchestrator.last_rolling_sharpe, abs=1e-6)
    assert float(run.trigger_rolling_sharpe) < RISK.target_sharpe_threshold
    assert float(run.trigger_vix) == pytest.approx(20.0)
    assert _runs_with_trigger(repo, "ct_bootstrap") == []


# --------------------------------------------------------------- FR-18 (benchmark)
def test_evaluate_persists_a_buy_and_hold_benchmark_on_the_same_window(tmp_path, monkeypatch):
    """§10.1: the live equity chart needs a benchmark, and it is only a fair
    one if it ran over the identical window with identical environment
    settings (costs, accounting) as the incumbent it is drawn next to."""
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=as_of)
    if orchestrator._retrain_thread is not None:
        orchestrator._retrain_thread.join(timeout=60)

    end = pd.Timestamp(as_of)
    frame = orchestrator._load_normalised_frame(end - pd.Timedelta(days=200), end)
    eval_slice = frame[frame["date"] >= _evaluation_start(frame)]
    expected = run_policy(TradingEnvironment(eval_slice.reset_index(drop=True), tickers=DATA.tickers), buy_and_hold)

    snaps = repo.list_snapshots(eval_slice["date"].min(), as_of)
    assert len(snaps) == len(expected["equity_curve"])
    assert snaps["benchmark_equity"].notna().all()
    np.testing.assert_allclose(snaps["benchmark_equity"].to_numpy(), expected["equity_curve"], rtol=1e-8)
    # Both series start from the same capital on the same first day.
    assert snaps["benchmark_equity"].iloc[0] == pytest.approx(snaps["equity_value"].iloc[0])
    # And it is a real second series, not the incumbent's curve copied over.
    assert not np.allclose(snaps["benchmark_equity"].to_numpy(), snaps["equity_value"].to_numpy())


# ------------------------------------------------- D-20: the stored history
class _HoldEqualWeight:
    """A serving model that stays the same, so every change in the history
    comes from how it is stored."""

    def evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        return [run_policy(env, equal_weight_rebalanced)]


class _AllCash:
    def evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        return [run_policy(env, all_cash)]


def _history_orchestrator(tmp_path, n_days: int = 260):
    """An orchestrator that checks but never trains, over real storage."""
    repo, _ = _seed_repo(tmp_path, n_days=n_days)
    service = SimpleNamespace(
        agent=_HoldEqualWeight(), active_version_id="v",
        controls=SimpleNamespace(is_on=lambda name: False),
    )
    service.get_agent = lambda: service.agent
    orch = CTOrchestrator(service, repo=repo, registry=object())
    orch._trigger_retrain = lambda *a, **k: setattr(orch, "_status", CTStatus.SERVING)
    days = sorted(repo.load_partition("1990-01-01", "2100-01-01")["date"].unique())
    return orch, repo, service, [pd.Timestamp(d).date() for d in days]


def _history(repo):
    return repo.list_snapshots("1990-01-01", "2100-01-01").set_index("date")


def test_a_day_once_stored_is_never_rewritten_as_the_window_moves_on(tmp_path):
    """D-20: daily checks left every day that aged out of the replay window
    at exactly the initial cash, with no Sharpe and no drawdown."""
    orch, repo, _, days = _history_orchestrator(tmp_path)
    checked = days[-50:]
    seen = {}
    for day in checked:
        orch.evaluate(as_of=day)
        row = _history(repo).loc[pd.Timestamp(day)]
        seen[day] = (row["equity_value"], row["rolling_sharpe_30d"], row["max_drawdown"], row["benchmark_equity"])

    final = _history(repo)
    for day, then in seen.items():
        row = final.loc[pd.Timestamp(day)]
        now = (row["equity_value"], row["rolling_sharpe_30d"], row["max_drawdown"], row["benchmark_equity"])
        assert now == pytest.approx(then), day
    assert (final["equity_value"] == 100_000.0).sum() == 1  # the first day only
    assert final.loc[pd.Timestamp(checked[0]):, "rolling_sharpe_30d"].notna().all()


def test_the_history_continues_from_the_last_stored_value(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    for day in days[-12:]:
        orch.evaluate(as_of=day)
    history = _history(repo)

    end = pd.Timestamp(days[-1])
    frame = orch._load_normalised_frame(end - pd.Timedelta(days=200), end)
    window = frame[frame["date"] >= _evaluation_start(frame)].reset_index(drop=True)
    for column, policy in (("equity_value", equal_weight_rebalanced), ("benchmark_equity", buy_and_hold)):
        replay = run_policy(TradingEnvironment(window, tickers=DATA.tickers), policy)["equity_curve"]
        got = history[column].iloc[-1] / history[column].iloc[-2]
        assert got == pytest.approx(replay[-1] / replay[-2], rel=1e-6)
        if policy is equal_weight_rebalanced:
            assert history["max_drawdown"].iloc[-1] == pytest.approx(max_drawdown(replay), rel=1e-4)
            assert max_drawdown(replay) > 0
    grown = history["equity_value"].iloc[-1] / history["equity_value"].iloc[0] - 1
    assert history["cumulative_return"].iloc[-1] == pytest.approx(grown, rel=1e-4)


def test_the_stored_sharpe_is_the_one_the_check_used_and_never_a_partial_window(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    orch.evaluate(as_of=days[-1])
    history = _history(repo)
    assert len(history) == RISK.sharpe_evaluation_window + 1
    assert history["rolling_sharpe_30d"].iloc[:-1].isna().all()
    assert history["rolling_sharpe_30d"].iloc[-1] == pytest.approx(orch.last_rolling_sharpe, rel=1e-5)


def test_a_model_swapped_in_later_cannot_rewrite_the_record(tmp_path):
    """The replacement was chosen on the same days, so letting it rewrite
    them would hide the decline it was chosen for."""
    orch, repo, service, days = _history_orchestrator(tmp_path)
    orch.evaluate(as_of=days[-2])
    before = _history(repo)
    service.agent = _AllCash()
    orch.evaluate(as_of=days[-2])
    pd.testing.assert_frame_equal(_history(repo), before)
    orch.evaluate(as_of=days[-1])
    after = _history(repo)
    pd.testing.assert_frame_equal(after.iloc[:-1], before)
    assert after["equity_value"].iloc[-1] == pytest.approx(before["equity_value"].iloc[-1])  # cash: no change


def test_a_check_forced_onto_a_past_date_adds_nothing(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    orch.evaluate(as_of=days[-1])
    before = _history(repo)
    orch.evaluate(as_of=days[-60])
    pd.testing.assert_frame_equal(_history(repo), before)


def test_days_missed_while_nothing_ran_are_filled_without_a_sharpe(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    orch.evaluate(as_of=days[-6])
    orch.evaluate(as_of=days[-1])
    history = _history(repo)
    assert list(history.index[-6:]) == [pd.Timestamp(d) for d in days[-6:]]
    assert history["rolling_sharpe_30d"].iloc[-5:-1].isna().all()
    assert pd.notna(history["rolling_sharpe_30d"].iloc[-1])


def test_a_gap_longer_than_the_window_continues_without_inventing_a_return(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    orch.evaluate(as_of=days[-60])
    last = _history(repo)["equity_value"].iloc[-1]
    orch.evaluate(as_of=days[-1])
    history = _history(repo)
    assert pd.Timestamp(days[-31]) not in history.index  # no return is measured for the window's first day
    first_new = history.loc[pd.Timestamp(days[-30])]
    assert first_new["equity_value"] == pytest.approx(last, rel=0.1)
    assert first_new["equity_value"] != pytest.approx(100_000.0)


def test_a_history_without_a_benchmark_starts_one_level_with_the_agent(tmp_path):
    orch, repo, _, days = _history_orchestrator(tmp_path)
    repo.record_snapshot(days[-2], 91_000.0, cumulative_return=-0.09)
    orch.evaluate(as_of=days[-1])
    row = _history(repo).iloc[-1]
    assert row["benchmark_equity"] == pytest.approx(91_000.0, rel=0.1)
    assert row["benchmark_equity"] != pytest.approx(100_000.0, rel=0.01)


# ------------------------------------------------- D-15 and the shared evaluation window
_LOSING = {"equity_curve": np.array([100_000.0, 95_000.0, 90_000.0]), "returns": np.array([-0.05, -0.0526])}
_WINNING = {"equity_curve": np.array([100_000.0, 101_000.0, 102_000.0]), "returns": np.array([0.01, 0.0099])}


def _join_retrain(orchestrator) -> None:
    if orchestrator._retrain_thread is not None:
        orchestrator._retrain_thread.join(timeout=60)


def _run_count(repo) -> int:
    return repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())["total_runs"]


def _events_of_kind(repo, kind) -> list[dict]:
    events, _ = repo.list_events(limit=200)
    return [e for e in events if e["kind"] == kind]


def test_unchanged_inputs_do_not_retrain_the_same_candidate_again(tmp_path, monkeypatch):
    """D-15: the scheduler re-fires every 300 s while the incumbent is below
    target. With the same model and the same last bar, a retrain can only
    reproduce the candidate that just lost, so it must not run. It is
    deferred, not cancelled: the next bar makes it run again."""
    repo, as_of = _seed_repo(tmp_path)
    earlier = (pd.Timestamp(as_of) - pd.Timedelta(days=7)).date()
    incumbent = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [_LOSING])
    runs_before = _run_count(repo)

    orchestrator.evaluate(as_of=earlier)
    _join_retrain(orchestrator)
    assert _run_count(repo) == runs_before + 1

    for _ in range(3):
        orchestrator.evaluate(as_of=earlier)
        assert orchestrator.status == CTStatus.SERVING
    _join_retrain(orchestrator)
    assert _run_count(repo) == runs_before + 1
    assert service.active_version_id == incumbent

    deferred = _events_of_kind(repo, "retrain_deferred")
    assert len(deferred) == 3
    assert "already compared" in deferred[0]["summary"]
    # This path is not the VIX deferral, which /ct-status reports separately.
    assert orchestrator.last_retrain_deferred_at is None

    orchestrator.evaluate(as_of=as_of)  # later last bar: new inputs
    _join_retrain(orchestrator)
    assert _run_count(repo) == runs_before + 2


def test_a_promoted_candidate_is_not_retrained_on_the_same_bar(tmp_path, monkeypatch):
    """After a promotion the incumbent is the model this data just produced.
    Retraining on the same bar would rebuild it and compare it to itself."""
    repo, as_of = _seed_repo(tmp_path)
    incumbent = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    calls = {"n": 0}

    def fake_evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        calls["n"] += 1
        # Replay for the trigger (loses), candidate (wins), incumbent (loses).
        return [_WINNING] if calls["n"] == 2 else [_LOSING]

    monkeypatch.setattr(PPOAgent, "evaluate", fake_evaluate)

    orchestrator.evaluate(as_of=as_of)
    _join_retrain(orchestrator)
    promoted = service.active_version_id
    assert promoted != incumbent
    runs = _run_count(repo)

    orchestrator.evaluate(as_of=as_of)
    _join_retrain(orchestrator)
    assert _run_count(repo) == runs
    assert service.active_version_id == promoted


def test_a_crashed_retrain_is_tried_again_on_the_same_bar(tmp_path, monkeypatch):
    """A crash reached no FR-17 decision, so the same inputs have no known
    outcome yet and the next evaluation must retry them (NFR-05)."""
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [_LOSING])
    attempts = {"n": 0}

    def exploding_train(self, total_timesteps):
        attempts["n"] += 1
        raise RuntimeError("simulated training crash")

    monkeypatch.setattr(PPOAgent, "train", exploding_train)

    orchestrator.evaluate(as_of=as_of)
    _join_retrain(orchestrator)
    orchestrator.evaluate(as_of=as_of)
    _join_retrain(orchestrator)
    assert attempts["n"] == 2
    assert _events_of_kind(repo, "retrain_deferred") == []


def test_trigger_and_gate_judge_on_the_same_number_of_trading_day_returns(tmp_path, monkeypatch):
    """FR-14 and FR-17 must judge a model on the same evidence. Counting
    calendar days gave the gate about 21 returns against the trigger's 30."""
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    windows = []

    def spy(self, env, n_episodes=1, seed=None, deterministic=True):
        windows.append((env._n_steps - 1, pd.Timestamp(env._dates[0]), pd.Timestamp(env._dates[-1])))
        return [_LOSING]

    monkeypatch.setattr(PPOAgent, "evaluate", spy)

    orchestrator.evaluate(as_of=as_of)
    _join_retrain(orchestrator)

    assert len(windows) == 3  # trigger replay, candidate, incumbent
    assert {w[0] for w in windows} == {RISK.sharpe_evaluation_window}
    assert len(set(windows)) == 1


def test_gate_window_counts_trading_days_across_a_gap_and_stays_after_training(tmp_path, monkeypatch):
    """A holiday-like gap must lengthen the window in calendar days rather
    than shorten it in returns, an `as_of` past the last bar must not change
    it, and DR-06 must still hold on the boundaries that get recorded."""
    repo, last_bar = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, last_bar)
    gap_end = pd.Timestamp(last_bar) - pd.Timedelta(days=10)
    gap_start = gap_end - pd.Timedelta(days=14)
    from src.dataops.models import MarketObservation
    with repo.session() as session:
        session.query(MarketObservation).filter(
            MarketObservation.observation_date >= gap_start.date(),
            MarketObservation.observation_date <= gap_end.date(),
        ).delete()
        session.commit()

    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    returns = []

    def spy(self, env, n_episodes=1, seed=None, deterministic=True):
        returns.append(env._n_steps - 1)
        return [_LOSING]

    monkeypatch.setattr(PPOAgent, "evaluate", spy)

    weekend_after = (pd.Timestamp(last_bar) + pd.Timedelta(days=2)).date()
    orchestrator._retrain_and_maybe_promote(weekend_after)
    assert returns == [RISK.sharpe_evaluation_window] * 2

    with repo.session() as session:
        run = session.query(ModelRun).filter(ModelRun.status == "REJECTED").one()
        assert run.eval_start > run.train_end
        assert run.eval_end == last_bar

