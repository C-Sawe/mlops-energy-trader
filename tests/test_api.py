"""Tests for the FastAPI inference gateway (FR-10, FR-18, FR-19, FR-20, NFR-08).

`app` builds `repo`/`service`/`registry`/`orchestrator` in its `lifespan`,
not at import time, specifically so each test's `TestClient(app)` context
gets a fresh set built against *that test's* `DATABASE_URL` — see the
comment in `src/serving/api.py`. Every test here sets `DATABASE_URL` (and,
where a model is involved, `MLFLOW_TRACKING_URI` + chdir) before entering
the `with TestClient(app) as client:` block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.config import DATA, RISK
from src.dataops import processing as P
from src.dataops.repository import MarketRepository
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment
from src.rlops.registry import ModelRegistry
from src.serving.api import app
from tests.test_processing import make_series


def _seed_db(tmp_path, n_days: int = 250, final_vix: float = 20.0) -> tuple[str, object]:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix_values = np.linspace(14, 20, len(dates))
    vix_values[-1] = final_vix
    vix = pd.DataFrame({"date": dates, "close": vix_values})
    frame = P.build_feature_frame(equities, vix)

    repo = MarketRepository(url=db_url)
    repo.create_schema()
    repo.persist(frame)
    return db_url, dates.max().date()


def _promote_model(db_url: str, tmp_path, as_of) -> str:
    repo = MarketRepository(url=db_url)
    frame = P.normalize_rolling(
        repo.load_partition(pd.Timestamp(as_of) - pd.Timedelta(days=249), as_of)
    )
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


def _flat_positions() -> dict[str, float]:
    return {t: 0.0 for t in DATA.tickers}


# --------------------------------------------------------------- FR-10
def test_health_reports_no_active_model_before_anything_is_trained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["active_version_id"] is None


def test_predict_returns_503_without_an_active_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    with TestClient(app) as client:
        resp = client.post("/predict", json={"positions": _flat_positions(), "cash_weight": 1.0})
        assert resp.status_code == 503


def test_predict_rejects_positions_missing_a_ticker(tmp_path, monkeypatch):
    """NFR-08: malformed input is rejected (422), never crashes the service."""
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    incomplete = _flat_positions()
    del incomplete[DATA.tickers[0]]

    with TestClient(app) as client:
        resp = client.post("/predict", json={"positions": incomplete, "cash_weight": 1.0})
        assert resp.status_code == 422


def test_predict_rejects_out_of_range_weight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    positions = _flat_positions()
    positions[DATA.tickers[0]] = 2.5  # outside [-1, 1]

    with TestClient(app) as client:
        resp = client.post("/predict", json={"positions": positions, "cash_weight": 1.0})
        assert resp.status_code == 422


def test_predict_succeeds_with_an_active_model_and_normal_vix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        resp = client.post("/predict", json={"positions": _flat_positions(), "cash_weight": 1.0})
        assert resp.status_code == 200
        body = resp.json()
        assert {d["ticker"] for d in body["decisions"]} == set(DATA.tickers)
        assert body["failsafe_triggered"] is False


# --------------------------------------------------------------- I5, FR-12
def test_predict_triggers_failsafe_with_critical_vix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold + 5.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        resp = client.post("/predict", json={"positions": _flat_positions(), "cash_weight": 1.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["failsafe_triggered"] is True
        assert all(d["discrete_action"] == "LIQUIDATE" for d in body["decisions"])


# --------------------------------------------------------------- FR-19
def test_decisions_endpoint_paginates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        client.post("/predict", json={"positions": _flat_positions(), "cash_weight": 1.0})

        resp = client.get("/decisions", params={"page": 1, "page_size": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == len(DATA.tickers)
        assert len(body["items"]) == 2


# --------------------------------------------------------------- FR-18
def test_telemetry_returns_recorded_snapshots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    repo = MarketRepository(url=db_url)
    repo.record_snapshot(as_of, 101_000.0, rolling_sharpe_30d=1.1, max_drawdown=0.02, cumulative_return=0.01)

    with TestClient(app) as client:
        resp = client.get(
            "/telemetry",
            params={"start": str(pd.Timestamp(as_of) - pd.Timedelta(days=5)), "end": str(as_of)},
        )
        assert resp.status_code == 200
        points = resp.json()["points"]
        assert len(points) == 1
        assert points[0]["equity_value"] == 101_000.0


# --------------------------------------------------------------- FR-20
def test_ct_status_reflects_active_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    version_id = _promote_model(db_url, tmp_path, as_of)

    with TestClient(app) as client:
        resp = client.get("/ct-status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_version_id"] == version_id
        assert body["status"] == "SERVING"
        assert body["target_sharpe_threshold"] == RISK.target_sharpe_threshold


# --------------------------------------------------------------- auth
def test_protected_endpoint_requires_token_when_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("API_BEARER_TOKEN", "secret123")

    with TestClient(app) as client:
        unauthenticated = client.get("/ct-status")
        assert unauthenticated.status_code == 401

        authenticated = client.get("/ct-status", headers={"Authorization": "Bearer secret123"})
        assert authenticated.status_code == 200


def test_health_never_requires_a_token(tmp_path, monkeypatch):
    """The health probe must stay reachable regardless of auth config."""
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("API_BEARER_TOKEN", "secret123")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
