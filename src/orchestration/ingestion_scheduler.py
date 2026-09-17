"""Scheduled market-data ingestion (FR-01: "without manual intervention").

Sits beside `ct_orchestrator.py` in this package because both are periodic
background jobs the serving layer wires up in its `lifespan` (see
`src/serving/api.py`), and together they close the loop CLAUDE.md §6
describes as the project's contribution: this module feeds new data in,
`CTOrchestrator` evaluates and retrains against it. Without this module,
FR-14's autonomous retraining had fresh telemetry to react to, but nothing
autonomously kept that telemetry current — every ingestion until now was a
human running `scripts/run_ingestion.py` by hand.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from src.config import DATA
from src.dataops.ingestion import fetch_market_data
from src.dataops.processing import build_feature_frame
from src.dataops.repository import MarketRepository

logger = logging.getLogger(__name__)

# Wilder's RSI_14 is an EMA with unbounded memory (CLAUDE.md §7, §9): the
# smoothing never fully "forgets" data before the fetch window starts, so a
# too-narrow trailing re-fetch computes a measurably *colder* RSI than the
# one already sitting in the store from the original backfill. Because
# `persist()` is idempotent (§7), a scheduled tick would then silently
# overwrite a correct, converged indicator with a wrong, cold one for every
# date the two windows overlap. At 120 days the EMA's weight on anything
# before the fetch starts has decayed to (1 - 1/14)**120 ≈ 1.9e-4 — close
# enough to full convergence that recomputing from this window reproduces
# the full-history value, not a different one.
DEFAULT_TRAILING_WINDOW_DAYS = 120


def run_ingestion_tick(
    repo: MarketRepository | None = None,
    tickers: tuple[str, ...] | None = None,
    window_days: int = DEFAULT_TRAILING_WINDOW_DAYS,
    end: date | None = None,
) -> int:
    """FR-01: re-fetch and persist a trailing window, idempotently.

    Refetches a window rather than only "yesterday" so late corrections
    (a split/dividend adjustment finalised after its ex-date, a provider
    correction) are picked up too — `persist()` upserts by (date, ticker),
    so re-writing overlapping rows updates them rather than duplicating or
    rejecting them (DR-05 still holds: exactly one row survives per key).

    Returns the number of rows in this tick's fetched frame (not the number
    that actually changed — an unchanged trailing window still round-trips
    through fetch + persist).

    Uses `fetch_market_data`'s default `strict=True`: a failed ticker
    raises `IngestionError` rather than silently persisting a partial
    universe (IR-02). The caller (the scheduler loop in `src/serving/api.py`)
    logs and skips the tick, same as a scheduled CT evaluation failure —
    the next tick tries again rather than the process crashing.
    """
    repo = repo or MarketRepository()
    end = end or date.today()
    start = end - timedelta(days=window_days)
    result = fetch_market_data(str(start), str(end), tickers=tickers or DATA.tickers)
    frame = build_feature_frame(result.equities, result.vix)
    repo.persist(frame)
    logger.info("ingestion tick: persisted %d rows (%s to %s)", len(frame), start, end)
    return len(frame)


class IngestionStatus(str, Enum):
    """No FR/DR ID covers this directly — it exists so the dashboard can
    show FR-01's "without manual intervention" actually happening, the
    same way FR-20's CT-pipeline status shows FR-13/14 happening, rather
    than asking anyone to trust it from the code alone."""

    IDLE = "IDLE"
    INGESTING = "INGESTING"


class IngestionState:
    """Thread-safe tracker for the last ingestion tick, shared by the
    background scheduler and the on-demand `/ingest/run` endpoint in
    `src/serving/api.py` — both call `.run()` instead of the bare
    `run_ingestion_tick`, so whichever one is presently ticking is
    reflected on `/ct-status` regardless of which one triggered it.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status = IngestionStatus.IDLE
        self._last_attempted_at: datetime | None = None
        self._last_ok: bool | None = None
        self._last_error: str | None = None
        self._last_rows: int | None = None

    @property
    def status(self) -> IngestionStatus:
        with self._lock:
            return self._status

    @property
    def last_attempted_at(self) -> datetime | None:
        with self._lock:
            return self._last_attempted_at

    @property
    def last_ok(self) -> bool | None:
        with self._lock:
            return self._last_ok

    def run(
        self,
        repo: MarketRepository | None = None,
        tickers: tuple[str, ...] | None = None,
        window_days: int = DEFAULT_TRAILING_WINDOW_DAYS,
        end: date | None = None,
    ) -> int:
        with self._lock:
            self._status = IngestionStatus.INGESTING
        try:
            rows = run_ingestion_tick(repo, tickers, window_days, end)
            with self._lock:
                self._last_ok = True
                self._last_error = None
                self._last_rows = rows
            return rows
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the caller to log
            with self._lock:
                self._last_ok = False
                self._last_error = str(exc)
            raise
        finally:
            with self._lock:
                self._status = IngestionStatus.IDLE
                self._last_attempted_at = datetime.now(timezone.utc)
