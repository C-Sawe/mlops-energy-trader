"""Tests for MLflow run logging and artifact registration (FR-08, FR-09).

Uses a local file-based MLflow tracking URI (a directory, not a server) and
an in-memory SQLite repository — no network and no running MLflow instance
required, matching this project's test conventions (CLAUDE.md §11).
"""
from __future__ import annotations

import mlflow
import pandas as pd
import pytest

from src.dataops.repository import MarketRepository
from src.rlops.agent import PPOAgent
from src.rlops.registry import ModelRegistry
from tests.test_environment import _feature_frame
from src.rlops.environment import TradingEnvironment


@pytest.fixture
def repo():
    r = MarketRepository(url="sqlite:///:memory:")
    r.create_schema()
    yield r
    r.engine.dispose()


@pytest.fixture
def registry(repo, tmp_path, monkeypatch):
    # MLflow's artifact store defaults to a relative ./mlruns regardless of
    # where the tracking database lives; chdir into tmp_path so a test run
    # never leaves artifacts behind in the actual working directory.
    monkeypatch.chdir(tmp_path)
    return ModelRegistry(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}", repo=repo)


def _trained_agent() -> PPOAgent:
    frame = _feature_frame(n=250, tickers=("XOM", "CVX"))
    env = TradingEnvironment(frame, tickers=("XOM", "CVX"))
    return PPOAgent(env, n_steps=64, seed=0).train(total_timesteps=128)


# --------------------------------------------------------------- FR-08, DR-08
def test_log_run_persists_exact_partition_boundaries(registry, repo):
    agent = _trained_agent()
    metrics = {"sharpe_ratio": 1.2, "max_drawdown": 0.1, "cumulative_return": 0.05}

    run_id = registry.log_run(
        agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", metrics
    )

    from src.dataops.models import ModelRun

    with repo.session() as session:
        record = session.get(ModelRun, run_id)
        assert record.train_start == pd.Timestamp("2020-01-01").date()
        assert record.train_end == pd.Timestamp("2021-12-31").date()
        assert record.eval_start == pd.Timestamp("2022-01-01").date()
        assert record.eval_end == pd.Timestamp("2022-06-30").date()
        assert float(record.sharpe_ratio) == pytest.approx(1.2)
        assert record.status == "COMPLETED"


def test_log_run_accepts_timestamp_inputs_not_just_strings(registry):
    agent = _trained_agent()
    run_id = registry.log_run(
        agent,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2021-12-31"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-06-30"),
        {"sharpe_ratio": 0.0},
    )
    assert run_id is not None


def test_log_run_records_hyperparameters_in_mlflow(registry):
    agent = _trained_agent()
    run_id = registry.log_run(
        agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", {"sharpe_ratio": 0.5}
    )

    from src.dataops.models import ModelRun

    with registry.repo.session() as session:
        record = session.get(ModelRun, run_id)
        mlflow_run_id = record.mlflow_run_ref

    logged = mlflow.get_run(mlflow_run_id)
    assert float(logged.data.params["learning_rate"]) == pytest.approx(3e-4)
    assert logged.data.metrics["sharpe_ratio"] == pytest.approx(0.5)
    artifacts = mlflow.artifacts.list_artifacts(run_id=mlflow_run_id, artifact_path="model")
    assert any(a.path.endswith("model.zip") for a in artifacts)


def test_log_run_rejects_evaluation_overlapping_training(registry):
    """DR-06/I1: the CHECK constraint fires through the registry too, not
    just when constructing ModelRun directly."""
    from sqlalchemy.exc import IntegrityError

    agent = _trained_agent()
    with pytest.raises(IntegrityError):
        registry.log_run(
            agent, "2020-01-01", "2021-12-31", "2021-06-01", "2022-06-30", {}
        )


# --------------------------------------------------------------- FR-09, DR-09
def test_register_version_links_to_its_run(registry):
    agent = _trained_agent()
    run_id = registry.log_run(
        agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", {"sharpe_ratio": 0.5}
    )

    version_id = registry.register_version(run_id, artifact_uri="mlruns/0/abc/artifacts/model")

    from src.dataops.models import ModelVersion

    with registry.repo.session() as session:
        version = session.get(ModelVersion, version_id)
        assert version.run_id == run_id
        assert version.artifact_uri == "mlruns/0/abc/artifacts/model"


def test_registered_version_defaults_to_inactive(registry):
    """Promotion (FR-16/FR-17) is a deliberate later decision, not a side
    effect of registering an artifact."""
    agent = _trained_agent()
    run_id = registry.log_run(
        agent, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30", {"sharpe_ratio": 0.5}
    )
    version_id = registry.register_version(run_id, artifact_uri="mlruns/0/abc/artifacts/model")

    from src.dataops.models import ModelVersion

    with registry.repo.session() as session:
        version = session.get(ModelVersion, version_id)
        assert version.is_active is False
        assert version.promoted_at is None
