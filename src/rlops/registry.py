"""Model run logging (FR-08) and artifact registration (FR-09).

MLflow tracks hyperparameters, metrics, and the artifact itself. The
`model_run` / `model_version` tables (DR-08, DR-09) are this project's own
record of exactly which data partition and run produced each servable
artifact — the chain that makes NFR-07 and I6 satisfiable, since a served
model must trace back to its run, and a run back to its partition
boundaries. MLflow's run id is stored alongside as `mlflow_run_ref`, but the
local tables are the source of truth serving (Sprint 4) will query.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.dataops.models import ModelRun, ModelVersion
from src.dataops.repository import MarketRepository
from src.rlops.agent import PPOAgent

_DateLike = str | pd.Timestamp


def _as_date(value: _DateLike):
    return pd.Timestamp(value).date()


class ModelRegistry:
    """Owns MLflow logging and the `model_run` / `model_version` records."""

    def __init__(self, tracking_uri: str | None = None, repo: MarketRepository | None = None):
        # Defaults to a local SQLite-backed store, not a server — no
        # network or running MLflow instance is required, matching this
        # project's test conventions (CLAUDE.md §11). Not `file:./mlruns`:
        # MLflow 3.x put the plain filesystem backend into maintenance mode
        # and refuses to open one without an explicit opt-out, so the
        # forward-compatible local default is a database URI, not a flag
        # that silences the deprecation. A real deployment (Sprint 4) points
        # MLFLOW_TRACKING_URI at the docker-compose `mlflow` server instead,
        # which already uses Postgres as its backend store.
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.repo = repo or MarketRepository()

    def log_run(
        self,
        agent: PPOAgent,
        train_start: _DateLike,
        train_end: _DateLike,
        eval_start: _DateLike,
        eval_end: _DateLike,
        metrics: dict[str, float],
        experiment_name: str = "ppo-trading-agent",
    ) -> str:
        """FR-08: log hyperparameters and metrics to MLflow, then persist
        the exact partition boundaries to `model_run` (DR-08) so the run is
        reproducible from the record alone. Returns the new `model_run.run_id`.
        """
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(agent.hyperparameters)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_path = Path(tmp_dir) / "model.zip"
                agent.save(artifact_path)
                mlflow.log_artifact(str(artifact_path), artifact_path="model")
            mlflow_run_id = run.info.run_id

        record = ModelRun(
            mlflow_run_ref=mlflow_run_id,
            train_start=_as_date(train_start),
            train_end=_as_date(train_end),
            eval_start=_as_date(eval_start),
            eval_end=_as_date(eval_end),
            hyperparameters=_json_safe(agent.hyperparameters),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            max_drawdown=metrics.get("max_drawdown"),
            cumulative_return=metrics.get("cumulative_return"),
            status="COMPLETED",
        )
        with self.repo.session() as session:
            session.add(record)
            session.commit()
            return record.run_id

    def register_version(self, run_id: str, artifact_uri: str) -> str:
        """FR-09/DR-09: register a servable artifact under a run.

        Deliberately does not set `is_active` — promotion (Sprint 4, FR-16/
        FR-17) is a separate decision gated on out-of-sample acceptance, not
        an automatic consequence of registering an artifact.
        """
        version = ModelVersion(run_id=run_id, artifact_uri=artifact_uri, is_active=False)
        with self.repo.session() as session:
            session.add(version)
            session.commit()
            return version.version_id


def _json_safe(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """`model_run.hyperparameters` is a JSON column; `seed=None` and plain
    floats/ints/strs round-trip fine, but this guards against anything else
    (e.g. a numpy scalar) slipping in and failing to serialise later."""
    return {k: (v if v is None or isinstance(v, (bool, int, float, str)) else str(v)) for k, v in hyperparameters.items()}
