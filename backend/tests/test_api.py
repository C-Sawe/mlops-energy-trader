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
        # Presentation label alongside the real ID — "<Month> <Year> #<n>",
        # not a replacement for active_version_id (NFR-07 still needs it).
        import re

        assert re.match(r"^[A-Z][a-z]+ \d{4} #1$", body["active_model_label"])


def test_ct_evaluate_accepts_an_as_of_override(tmp_path, monkeypatch):
    """The default (no as_of) anchors to date.today(), which a checkout's
    ingested data may not reach — an operator needs to be able to force
    evaluation against a date the data actually covers (backfill, catching
    up post-outage, or bootstrapping a fresh checkout)."""
    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    _promote_model(db_url, tmp_path, as_of)

    import src.serving.api as api

    with TestClient(app) as client:
        # The tiny incumbent's replayed Sharpe is usually below target, so
        # this fires a real background retrain (FR-14). Keep it tiny and join
        # it before the test ends: a daemon thread still inside torch at
        # interpreter exit aborts the process ("terminate called without an
        # active exception", exit 134 on Linux CI) after every test passed.
        api.orchestrator.retrain_timesteps = 128
        api.orchestrator.retrain_n_steps = 64
        resp = client.post("/ct/evaluate", params={"as_of": str(as_of)})
        assert resp.status_code == 200
        # as_of matches the seeded data's own range, so this must actually
        # evaluate (not silently no-op) — status settles back to SERVING
        # either way, but last_evaluated_at only moves if real work happened.
        assert client.get("/ct-status").json()["last_evaluated_at"] is not None
        if api.orchestrator._retrain_thread is not None:
            api.orchestrator._retrain_thread.join(timeout=60)
            assert not api.orchestrator._retrain_thread.is_alive()


# ---------------------------------------------------- candles + paper
def test_candles_returns_ohlc_ending_at_the_latest_ingested_bar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_url, last_date = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    with TestClient(app) as client:
        body = client.get("/market/candles", params={"ticker": "XOM", "days": 30}).json()
        assert body["ticker"] == "XOM"
        assert body["candles"][-1]["date"] == str(last_date)
        assert all(c["high"] >= c["low"] for c in body["candles"])

        assert client.get("/market/candles", params={"ticker": "AAPL"}).status_code == 422


def test_paper_status_reports_disabled_and_run_is_refused(tmp_path, monkeypatch):
    """Off by default: nothing reaches Alpaca unless explicitly enabled."""
    monkeypatch.chdir(tmp_path)
    db_url, _ = _seed_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.delenv("PAPER_TRADING_ENABLED", raising=False)

    with TestClient(app) as client:
        body = client.get("/paper/status").json()
        assert body["enabled"] is False
        assert body["positions"] == [] and body["equity"] is None
        assert client.post("/paper/run").status_code == 409


def test_paper_run_trades_through_the_api_with_a_mocked_broker(tmp_path, monkeypatch):
    """End to end through the real InferenceService and a real promoted
    model, with Alpaca replaced by httpx.MockTransport (no network)."""
    import httpx

    import src.serving.api as api
    from src.execution.alpaca_broker import AlpacaBroker
    from src.orchestration import paper_trader

    monkeypatch.chdir(tmp_path)
    db_url, as_of = _seed_db(tmp_path, final_vix=RISK.vix_critical_threshold - 10.0)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAPER_TRADING_ENABLED", "true")
    _promote_model(db_url, tmp_path, as_of)

    # Today, in market time, is the seeded data's last date; and the
    # pre-trade ingestion must not reach yfinance.
    monkeypatch.setattr(paper_trader, "market_today", lambda now=None: as_of)
    monkeypatch.setattr(api.ingestion, "run", lambda *a, **k: 0, raising=False)

    orders = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/account":
            return httpx.Response(200, json={"equity": "100000", "cash": "100000"})
        if path == "/v2/positions":
            return httpx.Response(200, json=[])
        if path == "/v2/orders" and request.method == "POST":
            orders.append(request.content)
            return httpx.Response(200, json={"id": f"o{len(orders)}", "status": "accepted"})
        return httpx.Response(404, json={})

    def factory():
        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://paper-api.alpaca.markets")
        return AlpacaBroker(api_key="k", secret_key="s", client=client)

    monkeypatch.setattr(api, "_broker_factory", factory)

    with TestClient(app) as client:
        monkeypatch.setattr(api.ingestion, "run", lambda *a, **k: 0)
        body = client.post("/paper/run").json()
        assert body["last_signal_date"] == str(as_of)
        assert body["last_outcome"].startswith("traded")
        assert body["equity"] == 100000.0
        # A second press on the same signal date cannot double-trade.
        again = client.post("/paper/run").json()
        assert "already traded" in again["last_outcome"]
    for raw in orders:
        assert b'"side":"buy"' in raw  # an empty account can only buy


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
