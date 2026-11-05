"""Fixed wall-clock weekday schedules, shared by ingestion (FR-01) and the
paper-trading cycle. A fixed time follows the market, not the process: an
interval timer restarts with every deploy, crash recovery (NFR-04) or
reload, and the first restart before it fires postpones the next run by a
full interval. Weekends are skipped; market holidays are not modelled, so a
run on a holiday just finds no new bar.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16


def _market_now(now: datetime | None = None) -> pd.Timestamp:
    ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MARKET_TZ)


def market_today(now: datetime | None = None) -> date:
    """The US market's calendar date — the date a bar ingested tonight
    carries, regardless of the server's own timezone."""
    return _market_now(now).date()


def last_closed_market_date(now: datetime | None = None) -> date:
    """The newest date whose daily bar is final: New York's date from 16:00
    New York time, the day before until then. A fetch during the session
    would store a half-finished bar as that day's close. Weekends and
    holidays are not modelled; those dates simply have no bar to fetch."""
    ts = _market_now(now)
    if ts.hour >= MARKET_CLOSE_HOUR:
        return ts.date()
    return ts.date() - timedelta(days=1)


def _at(day: datetime, run_time_utc: str) -> datetime:
    hour, minute = (int(x) for x in run_time_utc.split(":"))
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def next_run_at(now: datetime, run_time_utc: str) -> datetime:
    """The next weekday at `run_time_utc` (HH:MM) strictly after `now`."""
    candidate = _at(now.astimezone(timezone.utc), run_time_utc)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def previous_run_at(now: datetime, run_time_utc: str) -> datetime:
    """The most recent weekday at `run_time_utc` at or before `now`: the
    last run that should already have happened."""
    candidate = _at(now.astimezone(timezone.utc), run_time_utc)
    if candidate > now:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate
