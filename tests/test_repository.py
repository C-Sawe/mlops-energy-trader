"""Persistence tests (FR-05, DR-05, DR-06 constraints).

Runs against in-memory SQLite so the suite needs no running PostgreSQL.
The schema is identical; the CHECK constraints under test are portable.
Note that SQLite enforces CHECK constraints but not all Postgres types,
so the integration test in Sprint 4 must re-run these against Postgres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from src.dataops import processing as P
from src.dataops.models import ModelRun, ModelVersion, TradingDecision
from src.dataops.repository import MarketRepository
from tests.test_processing import make_series


@pytest.fixture
def repo():
    r = MarketRepository(url="sqlite:///:memory:")
    r.create_schema()
    yield r
    r.engine.dispose()


def feature_frame(n=120):
    equities = make_series(n)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(14, 28, len(dates))})
    return P.build_feature_frame(equities, vix)


# --------------------------------------------------------------- FR-05
def test_persist_and_reload_roundtrip(repo):
    df = feature_frame()
    written = repo.persist(df)

    assert written == len(df)
    assert repo.count() == len(df)

    loaded = repo.load_partition(df["date"].min(), df["date"].max())
    assert len(loaded) == len(df)
    assert set(loaded["ticker"].unique()) == set(df["ticker"].unique())


def test_load_partition_respects_date_bounds(repo):
    df = feature_frame(150)
    repo.persist(df)

    dates = sorted(df["date"].unique())
    lo, hi = dates[20], dates[60]
    loaded = repo.load_partition(lo, hi)

    assert loaded["date"].min() >= pd.Timestamp(lo)
    assert loaded["date"].max() <= pd.Timestamp(hi)


def test_load_partition_filters_by_ticker(repo):
    repo.persist(feature_frame())
    loaded = repo.load_partition("2020-01-01", "2021-12-31", tickers=("XOM",))
    assert set(loaded["ticker"].unique()) == {"XOM"}


# --------------------------------------------------------------- DR-05
def test_reingestion_is_idempotent_not_duplicating(repo):
    """TC-03: re-persisting the same range leaves one row per key."""
    df = feature_frame()
    repo.persist(df)
    first = repo.count()

    repo.persist(df)  # same batch again, as the scheduled job would
    assert repo.count() == first, "re-ingestion created duplicate rows"


def test_reingestion_updates_corrected_values(repo):
    """A late correction to a published price must overwrite, not append."""
    df = feature_frame(60)
    repo.persist(df)

    corrected = df.copy()
    idx = corrected.index[10]
    key_date, key_ticker = corrected.loc[idx, "date"], corrected.loc[idx, "ticker"]
    corrected.loc[idx, "close"] = 999.99

    repo.persist(corrected)

    reloaded = repo.load_partition(df["date"].min(), df["date"].max())
    row = reloaded[
        (reloaded["date"] == key_date) & (reloaded["ticker"] == key_ticker)
    ]
    assert row["close"].iloc[0] == pytest.approx(999.99)


def test_imputation_flag_survives_persistence(repo):
    df = feature_frame(80)
    df.loc[df.index[5], "is_imputed"] = True
    repo.persist(df)

    loaded = repo.load_partition(df["date"].min(), df["date"].max())
    assert loaded["is_imputed"].sum() == 1


def test_latest_state_returns_most_recent_row(repo):
    df = feature_frame()
    repo.persist(df)

    state = repo.latest_state("XOM")
    expected = df[df["ticker"] == "XOM"]["date"].max()

    assert state is not None
    assert pd.Timestamp(state["date"]) == expected
    assert "vix" in state and "sma_20" in state


def test_latest_state_none_for_unknown_ticker(repo):
    repo.persist(feature_frame())
    assert repo.latest_state("NOSUCH") is None


def test_persist_empty_frame_is_noop(repo):
    assert repo.persist(pd.DataFrame()) == 0
    assert repo.count() == 0


# --------------------------------------------------------------- DR-06
def test_model_run_rejects_evaluation_overlapping_training(repo):
    """The look-ahead guard is enforced by the database, not just code."""
    with repo.session() as session:
        session.add(
            ModelRun(
                train_start=pd.Timestamp("2020-01-01").date(),
                train_end=pd.Timestamp("2022-12-31").date(),
                eval_start=pd.Timestamp("2022-06-01").date(),  # overlaps
                eval_end=pd.Timestamp("2023-12-31").date(),
                hyperparameters={},
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_model_run_accepts_valid_chronological_windows(repo):
    with repo.session() as session:
        run = ModelRun(
            train_start=pd.Timestamp("2020-01-01").date(),
            train_end=pd.Timestamp("2022-12-31").date(),
            eval_start=pd.Timestamp("2023-01-01").date(),
            eval_end=pd.Timestamp("2023-12-31").date(),
            hyperparameters={"learning_rate": 3e-4},
        )
        session.add(run)
        session.commit()
        assert run.run_id is not None


# --------------------------------------------------------------- NFR-07
def test_decision_traces_back_to_run_and_partition(repo):
    """Every action must resolve to the weights and data that produced it."""
    with repo.session() as session:
        run = ModelRun(
            train_start=pd.Timestamp("2020-01-01").date(),
            train_end=pd.Timestamp("2022-12-31").date(),
            eval_start=pd.Timestamp("2023-01-01").date(),
            eval_end=pd.Timestamp("2023-12-31").date(),
            hyperparameters={"learning_rate": 3e-4},
        )
        session.add(run)
        session.flush()

        version = ModelVersion(
            run_id=run.run_id, artifact_uri="mlflow://models/ppo/1", is_active=True
        )
        session.add(version)
        session.flush()

        session.add(
            TradingDecision(
                version_id=version.version_id,
                ticker="XOM",
                raw_weight=0.72,
                discrete_action="BUY",
                vix_at_decision=18.4,
                failsafe_triggered=False,
            )
        )
        session.commit()
        decision_id = session.query(TradingDecision).one().decision_id

    with repo.session() as session:
        decision = session.get(TradingDecision, decision_id)
        assert decision.version.run.train_start == pd.Timestamp("2020-01-01").date()
        assert decision.version.run.hyperparameters["learning_rate"] == 3e-4


def test_invalid_action_is_rejected(repo):
    with repo.session() as session:
        run = ModelRun(
            train_start=pd.Timestamp("2020-01-01").date(),
            train_end=pd.Timestamp("2022-12-31").date(),
            eval_start=pd.Timestamp("2023-01-01").date(),
            eval_end=pd.Timestamp("2023-12-31").date(),
        )
        session.add(run)
        session.flush()
        version = ModelVersion(run_id=run.run_id, artifact_uri="uri")
        session.add(version)
        session.flush()

        session.add(
            TradingDecision(
                version_id=version.version_id,
                ticker="XOM",
                discrete_action="PANIC",  # not in the permitted enum
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
