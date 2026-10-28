"""Tests for the manual backfill entry point, scripts/run_ingestion.py (FR-01, DR-05).

The script is loaded by path (scripts/ is not a package). `fetch_market_data`
is replaced by a fake that slices a synthetic history, so no network is used.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.config import DATA
from src.dataops import processing as P
from src.dataops.ingestion import IngestionResult
from src.dataops.repository import MarketRepository
from tests.test_processing import make_series

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ingestion.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_ingestion", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history(n_days: int = 300):
    equities = make_series(n_days, tickers=DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": 18.0})
    return equities, vix, dates


def _fake_fetch(equities, vix):
    def fetch(start, end, tickers=None, strict=True):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        eq = equities[(equities["date"] >= s) & (equities["date"] <= e)].reset_index(drop=True)
        vx = vix[(vix["date"] >= s) & (vix["date"] <= e)].reset_index(drop=True)
        return IngestionResult(equities=eq, vix=vx, requested_tickers=tuple(DATA.tickers), failed_tickers=())
    return fetch


def test_backfill_inside_stored_history_does_not_clobber_indicators(tmp_path):
    """The real defect: `run_ingestion.py --start 2025-09-01` (and later
    --start 2026-04-01) recomputed indicators from its own first row, so the
    first 19 trading days came out NaN and the upsert overwrote correct stored
    values. The z-score window then spread each hole to ~78 unusable days."""
    script = _load_script()
    equities, vix, dates = _history()
    repo = MarketRepository(url=f"sqlite:///{tmp_path / 'test.db'}")
    repo.create_schema()
    repo.persist(P.build_feature_frame(equities, vix))

    start, end = dates[150], dates[-1]
    before = repo.load_partition(start, start + pd.Timedelta(days=40))
    assert before["sma_20"].notna().all()

    frame, _ = script.build_window(str(start.date()), str(end.date()), DATA.tickers, fetch=_fake_fetch(equities, vix))
    repo.persist(frame)

    after = repo.load_partition(start, start + pd.Timedelta(days=40))
    assert after["sma_20"].notna().all()
    assert after["rsi_14"].notna().all()
    pd.testing.assert_series_equal(
        before.sort_values(["ticker", "date"])["sma_20"].reset_index(drop=True),
        after.sort_values(["ticker", "date"])["sma_20"].reset_index(drop=True),
    )


def test_backfill_persists_only_the_requested_range(tmp_path):
    """Warm-up rows before --start are fetched for the indicators only and
    must never be written, or a backfill would touch history it wasn't asked for."""
    script = _load_script()
    equities, vix, dates = _history()
    start = dates[150]

    frame, _ = script.build_window(str(start.date()), str(dates[-1].date()), DATA.tickers, fetch=_fake_fetch(equities, vix))

    assert frame["date"].min() == start
    assert frame["sma_20"].notna().all()  # warm from the first requested day
