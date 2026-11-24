"""Tests for the pipeline simulation (src/simulation).

The property that matters is isolation: a simulation must not change the real
store, MLflow or the broker, must not see a later day's data (I1), and must
not receive a credential. The end-to-end test runs the real child process on
synthetic history, so it takes some seconds.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from src.config import DATA
from src.dataops import processing as P
from src.dataops.models import Base
from src.dataops.repository import MarketRepository
from src.serving.api import app
from src.simulation import manager as M
from src.simulation.manager import SimulationManager, SimulationRefused, child_environment
from src.simulation.scenarios import HISTORY_DAYS, Scenario
from src.simulation.worker import SandboxViolation, check_isolated
from tests.test_processing import make_series


@pytest.fixture
def market(tmp_path, monkeypatch):
    """A 'real' store of synthetic history, and a scenario that fits it."""
    equities = make_series(900, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 20, len(dates))})
    repo = MarketRepository(url=f"sqlite:///{tmp_path / 'real.db'}")
    repo.create_schema()
    repo.persist(P.build_feature_frame(equities, vix))
    start = next(d for d in dates if (d - dates[0]).days >= HISTORY_DAYS).date()
    monkeypatch.setitem(M.SCENARIOS, "test", Scenario("test", "t", "d", start, 5, crash_first_retrain=False))
    managers = []

    def make(**kw):
        managers.append(SimulationManager(repo, **kw))
        return managers[-1]

    yield repo, make
    for m in managers:
        m.stop()


def _counts(repo) -> dict[str, int]:
    with repo.session() as s:
        return {t.name: s.execute(select(func.count()).select_from(t)).scalar_one() for t in Base.metadata.sorted_tables}


def _wait(condition, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


def _step(m, n):
    m.command({"cmd": "step"})
    _wait(lambda: len(m.state()["days"]) >= n or m.state()["status"] != "running")
    assert m.state()["status"] == "running"
    return m.state()["days"][-1]


def test_a_simulation_runs_the_real_loop_and_leaves_the_real_store_untouched(market, monkeypatch):
    repo, make = market
    monkeypatch.setenv("ALPACA_API_KEY", "must-not-leak")
    before, uri = _counts(repo), mlflow.get_tracking_uri()
    m = make()
    m.start("test")
    _wait(lambda: m.state()["status"] != "starting")
    s = m.state()
    assert s["status"] == "running", s
    assert [e["kind"] for e in s["events"]][:2] == ["retrain_started", "candidate_promoted"]  # the bootstrap
    workdir = m._workdir

    def newest_bar():
        with sqlite3.connect(f"file:{workdir / 'sandbox.db'}?mode=ro", uri=True) as db:
            return db.execute("select max(observation_date) from market_observation").fetchone()[0]

    first_day = m._state["days"]
    assert first_day == [] and newest_bar() < str(M.SCENARIOS["test"].start)  # I1: nothing of day 1 yet
    day = _step(m, 1)
    assert newest_bar() == day["day"]  # I1: and nothing after it
    assert day["failsafe"] is False and day["rolling_sharpe"] is not None

    m.command({"cmd": "vix", "value": 50.0})
    day = _step(m, 2)
    assert day["failsafe_reason"] == "vix" and day["vix_injected"] and day["exposure"] == 0.0
    m.command({"cmd": "vix", "value": None})
    m.command({"cmd": "control", "name": "halt_trading", "enabled": True})
    day = _step(m, 3)
    assert day["failsafe_reason"] == "operator" and not day["vix_injected"]
    assert any(e["kind"] == "operator_override" for e in m.state()["events"])
    m.command({"cmd": "control", "name": "halt_trading", "enabled": False})
    assert _step(m, 4)["failsafe"] is False

    assert _counts(repo) == before
    assert mlflow.get_tracking_uri() == uri
    pid = m._proc.pid
    m.stop()
    assert not workdir.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert m.state()["status"] == "idle"


def test_the_child_environment_holds_no_credential_and_only_sandbox_addresses(tmp_path, monkeypatch):
    secrets = {
        "DATABASE_URL": "postgresql://real", "POSTGRES_PASSWORD": "x", "ALPACA_API_KEY": "x", "ALPACA_SECRET_KEY": "x",
        "GROQ_API_KEY": "x", "SESSION_SECRET": "x", "GOOGLE_CLIENT_SECRET": "x", "API_BEARER_TOKEN": "x",
        "A_SECRET_ADDED_LATER": "x", "MLFLOW_TRACKING_URI": "http://real",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    env = child_environment(tmp_path)
    assert set(secrets) & set(env) == {"DATABASE_URL", "MLFLOW_TRACKING_URI"}
    assert str(tmp_path) in env["DATABASE_URL"] and str(tmp_path) in env["MLFLOW_TRACKING_URI"]


@pytest.mark.parametrize("name", ["DATABASE_URL", "MLFLOW_TRACKING_URI"])
@pytest.mark.parametrize("value", ["postgresql://real", "", "sqlite:////somewhere/else/x.db"])
def test_the_worker_refuses_a_store_outside_its_own_directory(tmp_path, monkeypatch, name, value):
    for k, v in child_environment(tmp_path).items():
        monkeypatch.setenv(k, v)
    check_isolated(tmp_path)
    monkeypatch.setenv(name, value)
    with pytest.raises(SandboxViolation):
        check_isolated(tmp_path)


def test_an_injected_crash_leaves_the_model_serving(market, monkeypatch):
    _, make = market
    monkeypatch.setenv("TARGET_SHARPE_THRESHOLD", "1000")  # every check retrains
    m = make()
    m.start("test")
    _wait(lambda: m.state()["status"] != "starting")
    m.command({"cmd": "crash", "enabled": True})
    crashed = _step(m, 1)
    kinds = [e["kind"] for e in m.state()["events"] if e["day"] == crashed["day"]]
    assert kinds == ["retrain_started", "retrain_failed"] and not crashed["promoted"]
    assert crashed["failed_during_retrain"] == 0
    again = _step(m, 2)
    kinds = [e["kind"] for e in m.state()["events"] if e["day"] == again["day"]]
    assert kinds[0] == "retrain_started" and kinds[1] in ("candidate_promoted", "candidate_rejected")
    assert again["served_during_retrain"] > 0 and again["failed_during_retrain"] == 0


def test_one_session_at_a_time_and_unusable_scenarios_are_refused(market, monkeypatch):
    _, make = market
    m = make()
    with pytest.raises(SimulationRefused, match="unknown"):
        m.start("nope")
    with pytest.raises(SimulationRefused, match="ready"):
        m.command({"cmd": "step"})
    monkeypatch.setitem(M.SCENARIOS, "early", Scenario("early", "", "", date(2001, 1, 2), 5))
    with pytest.raises(SimulationRefused, match="history"):
        m.start("early")
    m.start("test")
    with pytest.raises(SimulationRefused, match="already running"):
        m.start("test")


def test_an_unwatched_session_is_ended_but_a_watched_one_is_not(market):
    _, make = market
    now = [0.0]
    m = make(clock=lambda: now[0])
    m.start("test")
    workdir = m._workdir
    now[0] += M.IDLE_SECONDS - 1
    m.state(watching=True)
    now[0] += M.IDLE_SECONDS - 1
    m.reap()
    assert m._proc is not None
    now[0] += 2
    m.reap()
    assert m._proc is None and not workdir.exists() and m.state()["status"] == "idle"


def test_a_watched_session_still_ends_at_the_time_limit(market):
    _, make = market
    now = [0.0]
    m = make(clock=lambda: now[0])
    m.start("test")
    now[0] += M.MAX_SECONDS + 1
    m.state(watching=True)
    m.reap()
    assert m._proc is None


def test_a_dead_child_is_reported_without_detail_and_cleaned_up(market):
    _, make = market
    m = make()
    m.start("test")
    workdir = m._workdir
    m._proc.kill()
    _wait(lambda: m.state()["status"] == "failed", 15)
    assert m.state()["error"] == "The simulation stopped unexpectedly."
    assert not workdir.exists()


def test_api_is_off_by_default_and_behind_sign_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    with TestClient(app) as client:
        assert client.get("/simulation").json() == {"enabled": False, "scenarios": [], "state": None}
        for path, body in (("start", {"scenario": "decay"}), ("command", {"cmd": "step"}), ("stop", None)):
            refused = client.post(f"/simulation/{path}", json=body)
            # The reason too: a start with no stored history is also a 409.
            assert refused.status_code == 409 and "disabled" in refused.json()["detail"]

        monkeypatch.setenv("SIMULATION_ENABLED", "true")
        assert client.get("/simulation").json()["state"]["status"] == "idle"
        assert client.post("/simulation/start", json={"scenario": "decay"}).status_code == 409  # no history
        for bad in ({"cmd": "rm"}, {"cmd": "vix", "value": 1e9}, {"cmd": "control"},
                    {"cmd": "control", "name": "pause_paper_trading"}):
            assert client.post("/simulation/command", json=bad).status_code == 422
        monkeypatch.setenv("API_BEARER_TOKEN", "t")
        assert client.get("/simulation").status_code == 401
        assert client.post("/simulation/stop").status_code == 401
