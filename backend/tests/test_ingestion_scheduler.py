"""Tests for the scheduled ingestion tick (FR-01).

`fetch_market_data` is monkeypatched rather than hitting the network — this
project's convention (CLAUDE.md §11) is synthetic deterministic data, no
network, no real database. The fake slices a full synthetic history down to
whatever [start, end] the tick actually requested, which is what makes it
possible to test the property that matters here: recomputing indicators
from only a *trailing slice* of history must reproduce the same values a
full backfill already computed and persisted, not silently NaN them out.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA
from src.dataops import processing as P
from src.dataops.ingestion import IngestionResult
from src.dataops.repository import MarketRepository
from src.orchestration import ingestion_scheduler as S
from tests.test_processing import make_series


def _seed_full_history(tmp_path, n_days: int = 300):
    """A repo backfilled once, the way `scripts/run_ingestion.py` does it,
    over more history than any single tick's trailing window will cover."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": 18.0})

    repo = MarketRepository(url=db_url)
    repo.create_schema()
    repo.persist(P.build_feature_frame(equities, vix))
    return repo, equities, vix, dates.max().date()


def test_ingestion_tick_reproduces_warm_indicators_not_nulls(tmp_path, monkeypatch):
    """The whole point of DEFAULT_TRAILING_WINDOW_DAYS: a scheduled tick
    that only re-fetches the last ~120 days must still land sma_20/rsi_14
    for the days it's actually updating — not overwrite an already-correct,
    warmed-up indicator with NULL because its own fetch window was too
    narrow to satisfy the rolling/EMA warm-up (CLAUDE.md §7)."""
    repo, equities, vix, as_of = _seed_full_history(tmp_path)

    def fake_fetch(start, end, tickers=None, strict=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        eq = equities[
            (equities["date"] >= start_ts)
            & (equities["date"] <= end_ts)
            & (equities["ticker"].isin(tickers or DATA.tickers))
        ].reset_index(drop=True)
        vx = vix[(vix["date"] >= start_ts) & (vix["date"] <= end_ts)].reset_index(drop=True)
        return IngestionResult(equities=eq, vix=vx, requested_tickers=tuple(tickers or DATA.tickers), failed_tickers=())

    monkeypatch.setattr(S, "fetch_market_data", fake_fetch)

    rows_fetched = S.run_ingestion_tick(repo, end=as_of)
    assert rows_fetched > 0

    lookback_start = pd.Timestamp(as_of) - pd.Timedelta(days=10)
    tail = repo.load_partition(lookback_start, as_of)
    assert not tail.empty
    assert tail["sma_20"].notna().all()
    assert tail["rsi_14"].notna().all()


def test_ingestion_tick_does_not_clobber_already_warm_history_behind_it(tmp_path, monkeypatch):
    """A tick's own fetch always starts *somewhere*, and the first
    `_INDICATOR_WARMUP` trading days after that start are `NaN` within this
    computation — regardless of window width — because `compute_indicators`
    needs real prior rows *in this call*, not just a wide calendar range.
    If that NaN zone lands on dates that already had a correct value from
    an earlier, better-positioned computation, persisting it clobbers
    perfectly good history. Found 2026-09-20 via a real evaluation that
    failed with "no usable rows for ticker 'XOM'" after a scheduled tick's
    120-day window started partway through already-ingested data."""
    repo, equities, vix, as_of = _seed_full_history(tmp_path)

    def fake_fetch(start, end, tickers=None, strict=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        eq = equities[
            (equities["date"] >= start_ts)
            & (equities["date"] <= end_ts)
            & (equities["ticker"].isin(tickers or DATA.tickers))
        ].reset_index(drop=True)
        vx = vix[(vix["date"] >= start_ts) & (vix["date"] <= end_ts)].reset_index(drop=True)
        return IngestionResult(equities=eq, vix=vx, requested_tickers=tuple(tickers or DATA.tickers), failed_tickers=())

    monkeypatch.setattr(S, "fetch_market_data", fake_fetch)

    # A tick whose end is well before the seeded history's own last day, so
    # its 120-day fetch starts partway through already-populated dates —
    # exactly the scenario a daily scheduler produces once it's been
    # running a while, not just on its very first-ever tick.
    tick_end = pd.Timestamp(as_of) - pd.Timedelta(days=50)
    fetch_start = tick_end - pd.Timedelta(days=S.DEFAULT_TRAILING_WINDOW_DAYS)
    clobber_zone_end = fetch_start + pd.Timedelta(days=30)  # comfortably covers the warm-up rows

    before = repo.load_partition(fetch_start, clobber_zone_end)
    assert not before.empty
    assert before["sma_20"].notna().all()  # already correct, from the original full backfill

    S.run_ingestion_tick(repo, end=tick_end.date())

    after = repo.load_partition(fetch_start, clobber_zone_end)
    assert len(after) == len(before)
    assert after["sma_20"].notna().all()
    assert after["rsi_14"].notna().all()
    pd.testing.assert_series_equal(
        before.sort_values(["ticker", "date"])["sma_20"].reset_index(drop=True),
        after.sort_values(["ticker", "date"])["sma_20"].reset_index(drop=True),
    )


def test_ingestion_tick_is_idempotent_on_repeated_runs(tmp_path, monkeypatch):
    """DR-05: re-running the same tick must not duplicate rows — persist()
    upserts by (date, ticker), so the store's row count is unaffected."""
    repo, equities, vix, as_of = _seed_full_history(tmp_path)

    def fake_fetch(start, end, tickers=None, strict=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        eq = equities[
            (equities["date"] >= start_ts)
            & (equities["date"] <= end_ts)
            & (equities["ticker"].isin(tickers or DATA.tickers))
        ].reset_index(drop=True)
        vx = vix[(vix["date"] >= start_ts) & (vix["date"] <= end_ts)].reset_index(drop=True)
        return IngestionResult(equities=eq, vix=vx, requested_tickers=tuple(tickers or DATA.tickers), failed_tickers=())

    monkeypatch.setattr(S, "fetch_market_data", fake_fetch)

    before = repo.load_partition("2000-01-01", as_of)
    S.run_ingestion_tick(repo, end=as_of)
    S.run_ingestion_tick(repo, end=as_of)
    after = repo.load_partition("2000-01-01", as_of)

    assert len(after) == len(before)


# --------------------------------------------------------------- IngestionState
def test_ingestion_state_records_a_successful_tick(tmp_path, monkeypatch):
    """The dashboard shows this to prove FR-01's autonomous ingestion is
    actually running, not just trust it from the code — so the state it
    reads has to reflect a real completed tick, not just "nothing failed
    yet"."""
    repo, equities, vix, as_of = _seed_full_history(tmp_path)

    def fake_fetch(start, end, tickers=None, strict=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        eq = equities[
            (equities["date"] >= start_ts)
            & (equities["date"] <= end_ts)
            & (equities["ticker"].isin(tickers or DATA.tickers))
        ].reset_index(drop=True)
        vx = vix[(vix["date"] >= start_ts) & (vix["date"] <= end_ts)].reset_index(drop=True)
        return IngestionResult(equities=eq, vix=vx, requested_tickers=tuple(tickers or DATA.tickers), failed_tickers=())

    monkeypatch.setattr(S, "fetch_market_data", fake_fetch)

    state = S.IngestionState()
    assert state.status == S.IngestionStatus.IDLE
    assert state.last_attempted_at is None
    assert state.last_ok is None

    rows = state.run(repo, end=as_of)

    assert rows > 0
    assert state.status == S.IngestionStatus.IDLE  # returns to idle once the tick completes
    assert state.last_attempted_at is not None
    assert state.last_ok is True


def test_ingestion_state_records_a_failed_tick_and_still_returns_to_idle(tmp_path, monkeypatch):
    """IR-02: a failed tick must be visible (last_ok False), not silently
    swallowed — and the state must not get stuck INGESTING forever just
    because one tick failed, or the dashboard would show a permanently
    stuck pipeline after a single bad fetch."""
    repo, _, _, as_of = _seed_full_history(tmp_path)

    def failing_fetch(start, end, tickers=None, strict=True):
        from src.dataops.ingestion import IngestionError

        raise IngestionError("simulated fetch failure")

    monkeypatch.setattr(S, "fetch_market_data", failing_fetch)

    state = S.IngestionState()
    with pytest.raises(Exception):
        state.run(repo, end=as_of)

    assert state.status == S.IngestionStatus.IDLE
    assert state.last_ok is False
    assert state.last_attempted_at is not None
