"""Tests for the inference service (FR-10, FR-11, FR-12, FR-16, I5, NFR-11).

Building an `InferenceService` requires a repository with real market data
*and* a promoted model_version whose artifact can actually be downloaded —
this is the first place in the project those two things (dataops's store,
rlops's registry) have to work together, so the fixture here does the full
train -> register -> promote sequence rather than mocking any of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DATA, RISK
from src.dataops import processing as P
from src.dataops.repository import MarketRepository
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment
from src.rlops.registry import ModelRegistry
from src.serving.inference import InferenceService, NoActiveModelError
from tests.test_processing import make_series


def _seed_repo_and_promote_model(
    tmp_path, monkeypatch, final_vix: float, n_days: int = 250
) -> MarketRepository:
    """Builds a repo with `n_days` of synthetic data for the whole universe
    (VIX on the final day pinned to `final_vix`, so tests can force or avoid
    the fail-safe), trains a tiny PPOAgent on it, and promotes it."""
    monkeypatch.chdir(tmp_path)

    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix_values = np.linspace(14, 20, len(dates))
    vix_values[-1] = final_vix
    vix = pd.DataFrame({"date": dates, "close": vix_values})
    frame = P.build_feature_frame(equities, vix)

    repo = MarketRepository(url="sqlite:///:memory:")
    repo.create_schema()
    repo.persist(frame)

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

    return repo


def _flat_positions(weight: float = 0.0) -> dict[str, float]:
    return {t: weight for t in DATA.tickers}


# --------------------------------------------------------------- I5, FR-12
def test_failsafe_triggers_above_vix_threshold(tmp_path, monkeypatch):
    repo = _seed_repo_and_promote_model(
        tmp_path, monkeypatch, final_vix=RISK.vix_critical_threshold + 5.0
    )
    service = InferenceService(repo=repo)

    result = service.predict(_flat_positions(), cash_weight=1.0)

    assert result["failsafe_triggered"] is True
    assert all(d["discrete_action"] == "LIQUIDATE" for d in result["decisions"])
    assert all(d["raw_weight"] is None for d in result["decisions"])


def test_failsafe_does_not_trigger_below_threshold(tmp_path, monkeypatch):
    repo = _seed_repo_and_promote_model(
        tmp_path, monkeypatch, final_vix=RISK.vix_critical_threshold - 5.0
    )
    service = InferenceService(repo=repo)

    result = service.predict(_flat_positions(), cash_weight=1.0)

    assert result["failsafe_triggered"] is False
    assert all(d["discrete_action"] != "LIQUIDATE" for d in result["decisions"])


def test_failsafe_decision_is_logged_with_its_trigger_value(tmp_path, monkeypatch):
    """NFR-11: the fail-safe must be logged with the value that triggered it."""
    trigger_vix = RISK.vix_critical_threshold + 3.0
    repo = _seed_repo_and_promote_model(tmp_path, monkeypatch, final_vix=trigger_vix)
    service = InferenceService(repo=repo)

    service.predict(_flat_positions(), cash_weight=1.0)

    items, total = repo.list_decisions(page_size=100)
    assert total == len(DATA.tickers)
    assert all(item["failsafe_triggered"] for item in items)
    assert all(item["vix_at_decision"] == pytest.approx(trigger_vix) for item in items)


# --------------------------------------------------------------- FR-11
def test_discretize_maps_weight_to_action_at_configured_thresholds(tmp_path, monkeypatch):
    repo = _seed_repo_and_promote_model(
        tmp_path, monkeypatch, final_vix=RISK.vix_critical_threshold - 10.0
    )
    service = InferenceService(repo=repo)

    assert service._discretize(RISK.buy_threshold + 0.01) == "BUY"
    assert service._discretize(RISK.sell_threshold - 0.01) == "SELL"
    assert service._discretize(0.0) == "HOLD"


# --------------------------------------------------------------- FR-10
def test_predict_returns_one_decision_per_ticker(tmp_path, monkeypatch):
    repo = _seed_repo_and_promote_model(
        tmp_path, monkeypatch, final_vix=RISK.vix_critical_threshold - 10.0
    )
    service = InferenceService(repo=repo)

    result = service.predict(_flat_positions(), cash_weight=1.0)

    tickers_seen = {d["ticker"] for d in result["decisions"]}
    assert tickers_seen == set(DATA.tickers)
    assert result["version_id"] == service.active_version_id
    assert result["decided_at"] is not None


# --------------------------------------------------------------- FR-16
def test_reload_picks_up_a_newly_promoted_version(tmp_path, monkeypatch):
    repo = _seed_repo_and_promote_model(
        tmp_path, monkeypatch, final_vix=RISK.vix_critical_threshold - 10.0
    )
    service = InferenceService(repo=repo)
    first_version = service.active_version_id
    assert first_version is not None

    # Train and promote a second model, entirely independent of the running service.
    train_env = TradingEnvironment(
        P.build_feature_frame(
            make_series(250, tickers=DATA.tickers),
            pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(
                        sorted(make_series(250, tickers=DATA.tickers)["date"].unique())
                    ),
                    "close": 20.0,
                }
            ),
        ),
        tickers=DATA.tickers,
    )
    second_agent = PPOAgent(train_env, n_steps=64, seed=99).train(total_timesteps=128)
    registry = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)
    run_id = registry.log_run(
        second_agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", {"sharpe_ratio": 1.0}
    )
    from src.dataops.models import ModelRun

    with repo.session() as session:
        mlflow_ref = session.get(ModelRun, run_id).mlflow_run_ref
    second_version = registry.register_version(run_id, artifact_uri=f"runs:/{mlflow_ref}/model")
    repo.promote_version(second_version)

    reloaded = service.reload()

    assert reloaded == second_version
    assert reloaded != first_version
    assert service.active_version_id == second_version


def test_reload_sets_its_own_tracking_uri_not_assuming_one_is_already_set(tmp_path, monkeypatch):
    """`InferenceService` can legitimately be the first thing in a fresh
    process to touch MLflow — a real server restart with an already-active
    model does exactly this — so `reload()` must not assume some other
    object (like `ModelRegistry`) already called
    `mlflow.set_tracking_uri`. Found via `scripts/broker_paper_trade_test.py`,
    the first place in this project anything constructed `InferenceService`
    without a `ModelRegistry` having run first: it failed with "Run not
    found" against MLflow's own ambient default tracking URI, not this
    project's `mlruns.db`. Every other test masks this by training/promoting
    through `ModelRegistry` first, which primes the global URI as a side
    effect before `InferenceService` ever needs it."""
    repo = _seed_repo_and_promote_model(tmp_path, monkeypatch, final_vix=20.0)

    import mlflow

    # Put MLflow back in the state a genuinely fresh process starts in:
    # `set_tracking_uri` also writes `MLFLOW_TRACKING_URI` into the
    # environment as a side effect (so subprocesses inherit it), so merely
    # calling it with a "wrong" value doesn't simulate an untouched process
    # — it just makes a different value look like a deliberate override.
    # `set_tracking_uri(None)` genuinely unsets both, which is what a real
    # server restart looks like before `InferenceService.reload()` runs.
    mlflow.set_tracking_uri(None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    service = InferenceService(repo=repo)

    assert service.active_version_id is not None
    assert service.get_agent() is not None


# --------------------------------------------------------------- edge cases
def test_no_active_model_raises_on_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = MarketRepository(url="sqlite:///:memory:")
    repo.create_schema()  # no data, no model at all

    service = InferenceService(repo=repo)
    assert service.active_version_id is None

    with pytest.raises(NoActiveModelError):
        service.predict(_flat_positions(), cash_weight=1.0)
