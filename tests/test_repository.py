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


def test_load_partition_returns_correctly_shaped_empty_frame(repo):
    """A query that matches nothing must still return the expected columns
    — `pd.DataFrame([])` has zero *columns*, not just zero rows, and a
    caller like `normalize_rolling` needs the real shape to fail gracefully
    on "no data" instead of crashing on "missing columns"."""
    repo.persist(feature_frame(60))

    empty = repo.load_partition("2099-01-01", "2099-12-31")

    assert empty.empty
    assert list(empty.columns) == [
        "date", "ticker", "open", "high", "low", "close",
        "volume", "sma_20", "rsi_14", "vix", "is_imputed",
    ]


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


def _make_version(repo, artifact_uri: str = "mlflow://models/ppo/1") -> str:
    with repo.session() as session:
        run = ModelRun(
            train_start=pd.Timestamp("2020-01-01").date(),
            train_end=pd.Timestamp("2022-12-31").date(),
            eval_start=pd.Timestamp("2023-01-01").date(),
            eval_end=pd.Timestamp("2023-12-31").date(),
        )
        session.add(run)
        session.flush()
        version = ModelVersion(run_id=run.run_id, artifact_uri=artifact_uri)
        session.add(version)
        session.commit()
        return version.version_id


# --------------------------------------------------------------- FR-19
def test_record_decision_returns_a_traceable_id(repo):
    version_id = _make_version(repo)
    decision_id = repo.record_decision(version_id, "XOM", 0.72, "BUY", 18.4)
    assert decision_id is not None


def test_list_decisions_orders_most_recent_first(repo):
    version_id = _make_version(repo)
    for i in range(3):
        repo.record_decision(version_id, "XOM", 0.1 * i, "HOLD", 15.0)

    items, total = repo.list_decisions(page=1, page_size=10)
    assert total == 3
    assert len(items) == 3
    assert items[0]["decided_at"] >= items[1]["decided_at"] >= items[2]["decided_at"]


def test_list_decisions_paginates(repo):
    version_id = _make_version(repo)
    for i in range(5):
        repo.record_decision(version_id, "XOM", 0.0, "HOLD", 15.0)

    page1, total = repo.list_decisions(page=1, page_size=2)
    page2, _ = repo.list_decisions(page=2, page_size=2)
    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert {d["decision_id"] for d in page1}.isdisjoint({d["decision_id"] for d in page2})


def test_decision_records_failsafe_flag(repo):
    version_id = _make_version(repo)
    repo.record_decision(version_id, "XOM", None, "LIQUIDATE", 40.0, failsafe_triggered=True)
    [item], _ = repo.list_decisions()
    assert item["failsafe_triggered"] is True
    assert item["raw_weight"] is None


def test_list_decisions_includes_run_and_partition(repo):
    """NFR-07's traceability chain (decision -> version -> run -> data
    partition) is only actually visible if a caller gets all of it back in
    one call, not just the version_id foreign key."""
    version_id = _make_version(repo)
    repo.record_decision(version_id, "XOM", 0.5, "BUY", 15.0)

    [item], _ = repo.list_decisions()
    assert item["run_id"] is not None
    assert item["train_start"] == pd.Timestamp("2020-01-01").date()
    assert item["train_end"] == pd.Timestamp("2022-12-31").date()
    assert item["eval_start"] == pd.Timestamp("2023-01-01").date()
    assert item["eval_end"] == pd.Timestamp("2023-12-31").date()


# --------------------------------------------------------------- FR-13, FR-18
def test_record_snapshot_and_list_snapshots_round_trip(repo):
    repo.record_snapshot("2024-01-02", 101_500.0, rolling_sharpe_30d=1.2, max_drawdown=0.05, cumulative_return=0.015)
    repo.record_snapshot("2024-01-03", 102_000.0, rolling_sharpe_30d=1.3, max_drawdown=0.04, cumulative_return=0.02)

    df = repo.list_snapshots("2024-01-01", "2024-01-31")
    assert len(df) == 2
    assert df.iloc[0]["equity_value"] == pytest.approx(101_500.0)
    assert df.iloc[1]["rolling_sharpe_30d"] == pytest.approx(1.3)


def test_record_snapshot_is_idempotent_per_date(repo):
    """Re-computing today's snapshot must update it, not duplicate it."""
    repo.record_snapshot("2024-01-02", 100_000.0)
    repo.record_snapshot("2024-01-02", 105_000.0)

    df = repo.list_snapshots("2024-01-01", "2024-01-31")
    assert len(df) == 1
    assert df.iloc[0]["equity_value"] == pytest.approx(105_000.0)


# --------------------------------------------------------------- I5, FR-16, FR-17
def test_get_active_version_returns_none_when_nothing_promoted(repo):
    _make_version(repo)  # exists, but never promoted
    assert repo.get_active_version() is None


def test_promote_version_activates_and_retires(repo):
    v1 = _make_version(repo, artifact_uri="uri-1")
    v2 = _make_version(repo, artifact_uri="uri-2")

    repo.promote_version(v1)
    active = repo.get_active_version()
    assert active["version_id"] == v1

    repo.promote_version(v2)
    active = repo.get_active_version()
    assert active["version_id"] == v2

    with repo.session() as session:
        from src.dataops.models import ModelVersion as MV

        retired = session.get(MV, v1)
        assert retired.is_active is False
        assert retired.retired_at is not None


def test_promote_version_rejects_unknown_id(repo):
    with pytest.raises(ValueError, match="no such model_version"):
        repo.promote_version("not-a-real-id")


# --------------------------------------------------------------- FR-17 (cycle stats)
def test_get_cycle_stats_counts_promoted_and_rejected(repo):
    promoted_version = _make_version(repo, artifact_uri="uri-promoted")
    repo.promote_version(promoted_version)

    with repo.session() as session:
        rejected_run = ModelRun(
            train_start=pd.Timestamp("2020-01-01").date(),
            train_end=pd.Timestamp("2022-12-31").date(),
            eval_start=pd.Timestamp("2023-01-01").date(),
            eval_end=pd.Timestamp("2023-12-31").date(),
            status="REJECTED",
        )
        session.add(rejected_run)
        session.commit()

    stats = repo.get_cycle_stats(pd.Timestamp("2000-01-01").date())
    assert stats["total_runs"] == 2
    assert stats["promoted"] == 1
    assert stats["rejected"] == 1


def test_get_cycle_stats_excludes_runs_before_the_window(repo):
    _make_version(repo)  # created "now", inside any reasonable window
    stats = repo.get_cycle_stats(pd.Timestamp("2099-01-01").date())  # window starts in the future
    assert stats["total_runs"] == 0


# --------------------------------------------------------------- ingest info
def test_latest_ingest_info_returns_most_recent_observation(repo):
    df = feature_frame(60)
    repo.persist(df)

    info = repo.latest_ingest_info()
    assert info is not None
    assert info["date"] == pd.Timestamp(df["date"].max()).date()
    assert info["vix"] is not None


def test_latest_ingest_info_none_when_store_is_empty(repo):
    assert repo.latest_ingest_info() is None
