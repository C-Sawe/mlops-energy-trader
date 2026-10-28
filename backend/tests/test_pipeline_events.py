"""Tests for the pipeline activity log (FR-20 as a history, NFR-07).

Each test drives the real code path (CT evaluation, retrain, ingestion,
the API) and checks the event it recorded, rather than calling
`record_event` directly. The timeline is only evidence of autonomy if the
events come from the steps themselves. No network: ingestion is
monkeypatched, as in test_ingestion_scheduler.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from src.config import RISK
from src.orchestration import ingestion_scheduler
from src.orchestration.events import emit
from src.orchestration.ingestion_scheduler import IngestionState
from src.rlops.agent import PPOAgent
from src.serving.api import app
from src.serving.inference import InferenceService
from tests.test_ct_orchestrator import _fast_orchestrator, _promote_initial_model, _seed_repo

LOSING = {"equity_curve": np.array([100_000.0, 99_000.0, 98_000.0]), "returns": np.array([-0.01, -0.0101])}
WINNING = {"equity_curve": np.array([100_000.0, 101_000.0, 102_000.0]), "returns": np.array([0.01, 0.0099])}


def _events(repo):
    items, _ = repo.list_events(limit=200)
    return items


def _kinds(repo):
    return [e["kind"] for e in _events(repo)]


def _always(result):
    return lambda self, env, n_episodes=1, seed=None, deterministic=True: [result]


def _candidate_then_incumbent(candidate, incumbent):
    calls = {"n": 0}

    def fake(self, env, n_episodes=1, seed=None, deterministic=True):
        calls["n"] += 1
        return [candidate] if calls["n"] == 1 else [incumbent]

    return fake


# --------------------------------------------------------------- storage
def test_events_are_listed_newest_first_and_page_backwards(tmp_path):
    repo, _ = _seed_repo(tmp_path, n_days=120)
    ids = [repo.record_event("evaluation", f"event {i}") for i in range(5)]

    page, has_more = repo.list_events(limit=2)
    assert [e["event_id"] for e in page] == ids[:-3:-1]
    assert has_more

    older, has_more = repo.list_events(limit=10, before_id=page[-1]["event_id"])
    assert [e["event_id"] for e in older] == ids[2::-1]
    assert not has_more


def test_an_unknown_event_kind_is_rejected_by_the_database(tmp_path):
    repo, _ = _seed_repo(tmp_path, n_days=120)
    with pytest.raises(IntegrityError):
        repo.record_event("made_up_kind", "should not be stored")


def test_a_failed_event_write_never_breaks_the_step_being_recorded(tmp_path, caplog):
    repo, _ = _seed_repo(tmp_path, n_days=120)

    def broken(*args, **kwargs):
        raise RuntimeError("database unavailable")

    repo.record_event = broken
    with caplog.at_level(logging.ERROR, logger="src.orchestration.events"):
        emit(repo, "evaluation", "model kept serving")  # must not raise
    assert "could not record pipeline event" in caplog.text


# --------------------------------------------------------------- CT loop
def test_an_evaluation_above_target_records_that_the_model_kept_serving(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    version = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", _always(WINNING))

    orchestrator.evaluate(as_of=as_of)

    [event] = _events(repo)
    assert event["kind"] == "evaluation"
    assert event["source"] == "scheduled"
    assert event["version_id"] == version
    assert event["detail"]["rolling_sharpe"] == pytest.approx(orchestrator.last_rolling_sharpe)
    assert "model kept serving" in event["summary"]


def test_decay_records_the_retrain_start_then_the_gate_outcome(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", _always(LOSING))  # incumbent decays; candidate no better

    orchestrator.evaluate(as_of=as_of)
    orchestrator._retrain_thread.join(timeout=60)

    rejected, started = _events(repo)
    assert started["kind"] == "retrain_started"
    assert started["detail"]["reason"] == "ct_decay"
    assert started["detail"]["rolling_sharpe"] < RISK.target_sharpe_threshold
    assert rejected["kind"] == "candidate_rejected"
    assert rejected["run_id"] is not None  # links to the logged candidate (NFR-07)
    assert rejected["version_id"] is None  # nothing was promoted


def test_a_promotion_records_the_new_version(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    old = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", _candidate_then_incumbent(WINNING, LOSING))

    orchestrator._retrain_and_maybe_promote(as_of)

    [event] = _events(repo)
    assert event["kind"] == "candidate_promoted"
    assert event["version_id"] == service.active_version_id != old
    assert event["detail"]["candidate_sharpe"] > event["detail"]["incumbent_sharpe"]


def test_the_bootstrap_is_recorded_as_the_first_model(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=as_of)
    orchestrator._retrain_thread.join(timeout=60)

    promoted, started = _events(repo)
    assert started["detail"]["reason"] == "ct_bootstrap"
    assert promoted["kind"] == "candidate_promoted"
    assert promoted["summary"].startswith("First model trained")
    assert promoted["detail"]["incumbent_sharpe"] is None  # -inf is not JSON


def test_a_volatility_deferral_is_recorded(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold + 5.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", _always(LOSING))

    orchestrator.evaluate(as_of=as_of)

    assert _kinds(repo) == ["retrain_deferred"]


def test_a_crashed_retrain_is_recorded_without_the_raw_error(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    def exploding_train(self, total_timesteps):
        raise RuntimeError("secret path /etc/something")

    monkeypatch.setattr(PPOAgent, "train", exploding_train)
    orchestrator._retrain_and_maybe_promote(as_of)

    [event] = _events(repo)
    assert event["kind"] == "retrain_failed"
    assert "secret path" not in event["summary"] + str(event["detail"])


def test_no_data_records_a_skipped_evaluation(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    orchestrator.evaluate(as_of=(pd.Timestamp(as_of) + pd.Timedelta(days=3650)).date())

    assert _kinds(repo) == ["evaluation_skipped"]


# --------------------------------------------------------------- ingestion
def test_ingestion_records_success_and_failure(tmp_path, monkeypatch):
    repo, _ = _seed_repo(tmp_path, n_days=120)
    state = IngestionState()

    monkeypatch.setattr(ingestion_scheduler, "run_ingestion_tick", lambda *a, **k: 42)
    state.run(repo)

    def failing(*a, **k):
        raise ConnectionError("yahoo unreachable")

    monkeypatch.setattr(ingestion_scheduler, "run_ingestion_tick", failing)
    with pytest.raises(ConnectionError):
        state.run(repo, source="manual")

    failed, ok = _events(repo)
    assert ok["kind"] == "ingestion_ok" and ok["detail"]["rows"] == 42
    assert failed["kind"] == "ingestion_failed" and failed["source"] == "manual"
    assert "ConnectionError" in failed["summary"]


# --------------------------------------------------------------- API
def test_force_check_is_recorded_as_manual_and_served_by_events(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    monkeypatch.setenv("DATABASE_URL", str(repo.engine.url))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns.db'}")
    monkeypatch.setattr(PPOAgent, "evaluate", _always(WINNING))

    with TestClient(app) as client:
        client.post(f"/ct/evaluate?as_of={as_of}")
        body = client.get("/events?limit=10").json()

    [event] = body["items"]
    assert event["kind"] == "evaluation"
    assert event["source"] == "manual"
    assert body["has_more"] is False


def test_events_require_auth_like_the_rest_of_the_api(tmp_path, monkeypatch):
    repo, _ = _seed_repo(tmp_path, n_days=120)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", str(repo.engine.url))
    monkeypatch.setenv("API_BEARER_TOKEN", "secret123")

    with TestClient(app) as client:
        assert client.get("/events").status_code == 401
        assert client.get("/events", headers={"Authorization": "Bearer secret123"}).status_code == 200
