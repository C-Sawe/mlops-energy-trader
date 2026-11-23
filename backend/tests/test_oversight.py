"""Tests for the operator override (FR-23, src/serving/oversight.py).

What these pin down:
  - each switch does what it says, and only makes the system more cautious;
  - the trading halt lives in the serving layer, so it survives a model
    promotion and a restart (I5, NFR-04);
  - every override is recorded with who, when and why (NFR-07, NFR-11);
  - a rollback restores a model that has served before, and refuses when it
    can't be done safely.

Fixtures are reused from the orchestrator and API tests, so the halt and
pause are exercised through the same code paths production uses.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import DATA, RISK
from src.dataops.models import ModelRun, ModelVersion
from src.orchestration.ct_orchestrator import CTStatus, RollbackRefused
from src.rlops.agent import PPOAgent
from src.serving import api
from src.serving.api import app
from src.serving.assistant import PipelineTools
from src.serving.inference import InferenceService
from tests.test_api import _promote_model, _seed_db
from tests.test_ct_orchestrator import (
    _events_of_kind,
    _fast_orchestrator,
    _promote_initial_model,
    _run_count,
    _seed_repo,
)

FLAT = {t: 0.0 for t in DATA.tickers}
LOSING = {"equity_curve": np.array([100_000.0, 95_000.0, 90_000.0]), "returns": np.array([-0.05, -0.0526])}
WINNING = {"equity_curve": np.array([100_000.0, 101_000.0, 102_000.0]), "returns": np.array([0.01, 0.0099])}


def _halt(service, by="ops@example.com"):
    service.controls.set("halt_trading", True, "VIX feed looks wrong", by)


# ------------------------------------------------------------ trading halt
def test_halt_returns_capital_preservation_without_consulting_the_model(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    assert service.predict(FLAT, 1.0)["failsafe_triggered"] is False  # VIX alone wouldn't fire it

    _halt(service)
    monkeypatch.setattr(PPOAgent, "predict", lambda *a, **k: pytest.fail("the model was consulted"))
    result = service.predict(FLAT, 1.0)

    assert result["failsafe_triggered"] is True
    assert result["failsafe_reason"] == "operator"
    assert all(d["discrete_action"] == "LIQUIDATE" and d["raw_weight"] is None for d in result["decisions"])


def test_halted_decisions_are_logged_as_operator_failsafes(tmp_path, monkeypatch):
    """NFR-11: a halt's decisions are separable from model decisions and
    from VIX trips, so later analysis can exclude or attribute them."""
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    service.predict(FLAT, 1.0)
    _halt(service)
    service.predict(FLAT, 1.0)

    items, _ = repo.list_decisions(page_size=100)
    reasons = sorted((i["failsafe_triggered"], i["failsafe_reason"]) for i in items)
    assert reasons == [(False, None)] * 5 + [(True, "operator")] * 5


def test_vix_trip_is_logged_with_reason_vix(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold + 5.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    result = InferenceService(repo=repo).predict(FLAT, 1.0)
    assert result["failsafe_reason"] == "vix"


def test_lifting_the_halt_restores_model_decisions(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    _halt(service)
    service.controls.set("halt_trading", False, None, "ops@example.com")

    result = service.predict(FLAT, 1.0)
    assert result["failsafe_triggered"] is False
    assert all(d["raw_weight"] is not None for d in result["decisions"])


def test_halt_survives_a_promotion_and_a_restart(tmp_path, monkeypatch):
    """I5: the halt must not live in anything promotion replaces, and a
    restart must not quietly lift it (NFR-04)."""
    repo, as_of = _seed_repo(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    _halt(service)

    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)  # a new model goes live
    service.reload()
    assert service.predict(FLAT, 1.0)["failsafe_reason"] == "operator"

    restarted = InferenceService(repo=repo)  # a fresh process reads the database
    assert restarted.controls.get("halt_trading")["set_by"] == "ops@example.com"
    assert restarted.predict(FLAT, 1.0)["failsafe_reason"] == "operator"


def test_a_failed_write_leaves_the_switch_unchanged(tmp_path, monkeypatch):
    """Database first, memory second: if the write fails, what the system
    obeys must still match what is stored."""
    repo, _ = _seed_repo(tmp_path)
    service = InferenceService(repo=repo)

    def broken(*a, **k):
        raise RuntimeError("database down")

    monkeypatch.setattr(repo, "set_control", broken)
    with pytest.raises(RuntimeError):
        _halt(service)
    assert service.controls.is_on("halt_trading") is False


# -------------------------------------------------------- retraining pause
def test_paused_retraining_defers_a_decay_retrain_and_says_why(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [LOSING])
    service.controls.set("pause_retraining", True, "defence demo", "ops@example.com")

    status = orchestrator.evaluate(as_of=as_of)

    assert status == CTStatus.SERVING
    assert orchestrator._retrain_thread is None
    assert orchestrator.last_rolling_sharpe < RISK.target_sharpe_threshold  # a retrain really was due
    [event] = _events_of_kind(repo, "retrain_deferred")
    assert event["detail"]["reason"] == "operator_paused"
    assert event["detail"]["paused_by"] == "ops@example.com"
    assert _events_of_kind(repo, "retrain_started") == []


def test_paused_retraining_blocks_the_bootstrap_too(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    service.controls.set("pause_retraining", True, "not yet", "ops@example.com")

    assert orchestrator.evaluate(as_of=as_of) == CTStatus.SERVING
    assert orchestrator._retrain_thread is None and _run_count(repo) == 0


def test_lifting_the_pause_lets_the_next_evaluation_retrain(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path, final_vix=20.0)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    monkeypatch.setattr(PPOAgent, "evaluate", lambda self, env, n_episodes=1, seed=None, deterministic=True: [LOSING])
    service.controls.set("pause_retraining", True, "demo", "ops@example.com")
    orchestrator.evaluate(as_of=as_of)

    service.controls.set("pause_retraining", False, None, "ops@example.com")
    orchestrator.evaluate(as_of=as_of)
    orchestrator._retrain_thread.join(timeout=60)

    assert len(_events_of_kind(repo, "retrain_started")) == 1


def test_a_pause_during_training_withholds_promotion_without_counting_a_rejection(tmp_path, monkeypatch):
    """A candidate that would have won, finishing after the operator paused
    retraining: not promoted, logged WITHHELD (not REJECTED, since the FR-17
    gate didn't turn it down), and not remembered as a finished decision, so
    the gate decides this window properly after the pause is lifted."""
    repo, as_of = _seed_repo(tmp_path)
    incumbent = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)

    calls = {"n": 0}

    def fake_evaluate(self, env, n_episodes=1, seed=None, deterministic=True):
        calls["n"] += 1
        if calls["n"] == 1:  # the candidate has just finished training; the operator pauses now
            service.controls.set("pause_retraining", True, "hold everything", "ops@example.com")
            return [WINNING]
        return [LOSING]

    monkeypatch.setattr(PPOAgent, "evaluate", fake_evaluate)
    orchestrator._retrain_and_maybe_promote(as_of, reason="ct_decay")

    assert service.active_version_id == incumbent
    with repo.session() as session:
        statuses = [r.status for r in session.query(ModelRun).all()]
    assert "WITHHELD" in statuses
    assert repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())["rejected"] == 0
    assert len(_events_of_kind(repo, "candidate_withheld")) == 1
    assert orchestrator._last_decision is None
    assert orchestrator.status == CTStatus.SERVING


# ---------------------------------------------------------------- rollback
def _two_served_models(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    first = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    second = _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)
    orchestrator = _fast_orchestrator(service, repo, tmp_path, monkeypatch)
    return repo, service, orchestrator, first, second


def test_rollback_restores_an_earlier_model_and_keeps_its_promotion_time(tmp_path, monkeypatch):
    repo, service, orchestrator, first, second = _two_served_models(tmp_path, monkeypatch)
    with repo.session() as session:
        original_promoted_at = session.get(ModelVersion, first).promoted_at

    assert orchestrator.rollback(first) == first

    assert service.active_version_id == first
    with repo.session() as session:
        restored = session.get(ModelVersion, first)
        replaced = session.get(ModelVersion, second)
        assert restored.is_active and restored.retired_at is None
        assert restored.promoted_at == original_promoted_at
        assert not replaced.is_active and replaced.retired_at is not None


def test_rollback_is_refused_while_retraining(tmp_path, monkeypatch):
    _, _, orchestrator, first, _ = _two_served_models(tmp_path, monkeypatch)
    orchestrator._status = CTStatus.RETRAINING
    with pytest.raises(RollbackRefused, match="retraining"):
        orchestrator.rollback(first)


def test_rollback_is_refused_for_a_model_that_never_served(tmp_path, monkeypatch):
    repo, service, orchestrator, _, second = _two_served_models(tmp_path, monkeypatch)
    with repo.session() as session:
        run_id = session.get(ModelVersion, second).run_id
        session.add(ModelVersion(version_id="never-served", run_id=run_id, artifact_uri="x"))
        session.commit()

    with pytest.raises(RollbackRefused, match="served before"):
        orchestrator.rollback("never-served")
    with pytest.raises(RollbackRefused, match="already serving"):
        orchestrator.rollback(second)
    assert service.active_version_id == second


# --------------------------------------------------------------------- API
def test_api_switch_needs_a_reason_and_records_who_did_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        assert client.put("/oversight/controls/halt_trading", json={"enabled": True}).status_code == 422
        assert client.put("/oversight/controls/launch_rockets", json={"enabled": True, "reason": "x"}).status_code == 422

        resp = client.put("/oversight/controls/halt_trading", json={"enabled": True, "reason": "bad data"})
        assert resp.status_code == 200
        halt = next(c for c in resp.json()["controls"] if c["name"] == "halt_trading")
        assert halt["enabled"] and halt["reason"] == "bad data" and halt["set_by"] == "local operator"

        status = client.get("/ct-status").json()
        assert next(c for c in status["controls"] if c["name"] == "halt_trading")["enabled"] is True

        predicted = client.post("/predict", json={"positions": FLAT, "cash_weight": 1.0}).json()
        assert predicted["failsafe_reason"] == "operator"

        events = client.get("/events").json()["items"]
        [override] = [e for e in events if e["kind"] == "operator_override"]
        assert override["source"] == "manual"
        assert override["detail"] == {"control": "halt_trading", "enabled": True,
                                      "reason": "bad data", "by": "local operator"}

        # Setting a switch to the state it's already in changes nothing, so
        # the original reason isn't overwritten with no event to keep it.
        body = client.put("/oversight/controls/halt_trading", json={"enabled": True, "reason": "still bad"}).json()
        assert body["controls"][0]["reason"] == "bad data"
        events = client.get("/events").json()["items"]
        assert len([e for e in events if e["kind"] == "operator_override"]) == 1


def test_api_rollback_lists_served_models_and_refuses_with_409(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    first = _promote_model(db_url, tmp_path, as_of)
    second = _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        body = client.get("/oversight").json()
        assert [v["version_id"] for v in body["versions"]] == [second, first]
        assert body["versions"][0]["is_active"] and body["pipeline_status"] == "SERVING"

        assert client.post("/oversight/rollback", json={"version_id": first, "reason": ""}).status_code == 422
        assert client.post("/oversight/rollback", json={"version_id": second, "reason": "x"}).status_code == 409

        resp = client.post("/oversight/rollback", json={"version_id": first, "reason": "new model misbehaving"})
        assert resp.status_code == 200
        assert client.get("/health").json()["active_version_id"] == first

        [event] = [e for e in client.get("/events").json()["items"] if e["kind"] == "operator_override"]
        assert event["detail"]["from_version"] == second and event["detail"]["to_version"] == first
        assert event["version_id"] == first


def test_operator_identity_never_reaches_the_assistants_language_model(tmp_path, monkeypatch):
    """DR-10: what the model reads (event summaries, the status tool) must
    not carry who set a control. The dashboard still gets it, in `detail`."""
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    first = _promote_model(db_url, tmp_path, as_of)
    _promote_model(db_url, tmp_path, as_of)
    monkeypatch.setattr(api, "_actor", lambda request: "ops@example.com")

    with TestClient(app) as client:
        client.put("/oversight/controls/halt_trading", json={"enabled": True, "reason": "bad data"})
        client.put("/oversight/controls/halt_trading", json={"enabled": False})
        client.post("/oversight/rollback", json={"version_id": first, "reason": "why"})
        tools = PipelineTools(api.repo, lambda: api.ct_status().model_dump())
        activity = tools.run("get_recent_activity", {})
        status = tools.run("get_pipeline_status", {})

    assert len(activity.result["events"]) == 3
    assert "ops@example.com" not in json.dumps([activity.result, status.result], default=str)
    assert "ops@example.com" in json.dumps(activity.visual, default=str)


def test_halt_holds_when_market_data_cannot_be_read(tmp_path, monkeypatch):
    repo, as_of = _seed_repo(tmp_path)
    _promote_initial_model(repo, tmp_path, monkeypatch, as_of)
    service = InferenceService(repo=repo)

    def broken():
        raise RuntimeError("feed down")

    monkeypatch.setattr(service, "_load_recent_frame", broken)
    with pytest.raises(RuntimeError):
        service.predict(FLAT, 1.0)  # without a halt the error is not hidden
    _halt(service)
    result = service.predict(FLAT, 1.0)
    assert result["failsafe_reason"] == "operator" and result["vix_at_decision"] is None
    assert all(d["discrete_action"] == "LIQUIDATE" for d in result["decisions"])


def test_a_rollback_whose_model_cannot_load_leaves_the_current_one_recorded_active(tmp_path, monkeypatch):
    repo, service, orchestrator, first, second = _two_served_models(tmp_path, monkeypatch)

    def broken():
        raise OSError("artifact missing")

    monkeypatch.setattr(service, "reload", broken)
    with pytest.raises(RollbackRefused, match="could not be loaded"):
        orchestrator.rollback(first)
    assert repo.get_active_version()["version_id"] == second == service.active_version_id
    with repo.session() as session:
        assert session.get(ModelVersion, second).retired_at is None
