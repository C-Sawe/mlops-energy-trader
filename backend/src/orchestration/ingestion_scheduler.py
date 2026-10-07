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
# one already sitting in the store from the original backfill. 120 calendar
# days is ~83 trading rows, so the EMA's weight on anything before the fetch
# starts has decayed to (1 - 1/14)**83 ≈ 2e-3 at the window's end (an
# earlier version of this comment used 120 rows, ≈1.9e-4). Measured on
# synthetic data, that leaves the newest rows within ~0.2 RSI points of the
# full-history value (test_ingestion_tick_backfills_a_gap_longer_than_its_window).
#
# That handles RSI's "colder, not wrong" problem, but not SMA_20's
# different one: `compute_indicators()` uses `min_periods=window`, so the
# first `sma_window - 1` (here, 19) rows of *any* single computation call
# are exactly `NaN`, by construction, no matter how wide the window is —
# widening the fetch doesn't give those specific rows more lookback, since
# they're only ever a fixed distance from the fetch's own leading edge, not
# from the true history. A 120-day fetch still starts *somewhere*, and
# whatever ~20 trading days land right after that start are NaN this tick.
# Since `persist()` is idempotent (§7), the next tick's fetch starts one day
# later, its own leading ~20-day NaN zone shifts one day later too, and it
# clobbers a *different* previously-good stretch — every tick permanently
# corrupts a moving 20-day window of indicator history. Found 2026-09-20,
# replaying an evaluation against `as_of=2026-08-19`: `TradingEnvironment`
# raised "no usable rows for ticker 'XOM'" because `sma_20`/`rsi_14` had
# gone NaN for 2026-06-01..18 — a stretch a scheduled tick's own 120-day
# fetch had started right before, days after that range was already
# correctly populated by an earlier, wider backfill.
_INDICATOR_WARMUP = max(DATA.sma_window, DATA.rsi_window)

DEFAULT_TRAILING_WINDOW_DAYS = 120

def _gap_warning_days(window_days: int) -> int:
    """A gap this many calendar days or longer would have defeated a fixed
    `end - window_days` fetch: its first missing dates land at or before
    the fetch's own leading warm-up rows (`_INDICATOR_WARMUP` trading rows,
    ~1.4× that in calendar days; doubled for margin), so they would be
    trimmed and never filled. Only decides when a gap backfill is logged
    as a warning; the fetch start is anchored to the last stored bar
    regardless (`_fetch_start`)."""
    return window_days - 2 * _INDICATOR_WARMUP


def _fetch_start(repo: MarketRepository, end: date, window_days: int) -> date:
    """FR-01: where this tick's fetch must start so no gap is left behind.

    Anchors the trailing window to whichever is earlier, `end` or the last
    stored bar. In steady state the two are a day or a weekend apart, so
    this is the plain trailing window. After an outage longer than the
    window (CLAUDE.md §13's real 4.5-month gap), anchoring to `end` alone
    would start the fetch *after* the last stored bar and leave a permanent
    hole. Anchoring to the last stored bar instead gives the first missing
    day the same `window_days` of lookback a normal tick gets, so RSI is as
    converged as usual and the trimmed warm-up rows fall on dates that are
    already stored, never on the gap.
    """
    latest = repo.latest_ingest_info()
    last_stored = latest["date"] if latest else None
    anchor = min(end, last_stored) if last_stored else end
    start = anchor - timedelta(days=window_days)

    if last_stored and (end - last_stored).days >= _gap_warning_days(window_days):
        logger.warning(
            "ingestion tick: last stored bar is %s, %d days before %s, longer than "
            "the %d-day trailing window covers on its own; backfilling from %s",
            last_stored, (end - last_stored).days, end, window_days, start,
        )
    return start


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

    Drops each ticker's first `_INDICATOR_WARMUP` rows before persisting —
    those rows are `NaN` by construction within *this* fetch (see the
    module-level comment above) regardless of window width, and persisting
    them would silently overwrite an already-correct value an earlier,
    better-positioned computation left there. `window_days` must stay
    comfortably larger than `_INDICATOR_WARMUP` or every ticker's trimmed
    frame comes back empty; the default (120) has wide margin.

    Gaps longer than the window are backfilled in the same tick: the
    window is anchored to the last stored bar when that is older than
    `end` (`_fetch_start`), so an outage never leaves a permanent hole that
    needs a manual `run_ingestion.py` run.

    Returns the number of rows actually persisted (post-trim), not the
    number fetched.

    Uses `fetch_market_data`'s default `strict=True`: a failed ticker
    raises `IngestionError` rather than silently persisting a partial
    universe (IR-02). The caller (the scheduler loop in `src/serving/api.py`)
    logs and skips the tick, same as a scheduled CT evaluation failure —
    the next tick tries again rather than the process crashing.
    """
    repo = repo or MarketRepository()
    end = end or date.today()
    start = _fetch_start(repo, end, window_days)
    result = fetch_market_data(str(start), str(end), tickers=tickers or DATA.tickers)
    frame = build_feature_frame(result.equities, result.vix)
    # Not groupby(...).apply(lambda g: g.iloc[n:]) — pandas 3.x's groupby
    # excludes the grouping column from what apply's callback receives,
    # which silently dropped "ticker" here and broke every insert.
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame = frame[frame.groupby("ticker").cumcount() >= _INDICATOR_WARMUP].reset_index(drop=True)
    if frame.empty:
        logger.warning(
            "ingestion tick: window_days=%d left nothing after trimming the %d-row "
            "indicator warm-up — window_days must be well above _INDICATOR_WARMUP",
            window_days, _INDICATOR_WARMUP,
        )
        return 0
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
