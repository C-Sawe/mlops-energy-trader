"""FR-01's schedule: ingestion at a fixed weekday time after the US close,
plus one catch-up fetch at startup when the stored data is behind.

The interval scheduler this replaces waited a full day after every process
start, so repeated restarts on 2026-10-21 postponed it until two trading
days were missed. No network: ingestion is stubbed.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.orchestration.ingestion_scheduler import IngestionState, ingestion_is_behind
from src.orchestration.schedule import next_run_at, previous_run_at
from src.serving import api
from src.serving.api import app
from tests.test_ct_orchestrator import _seed_repo

UTC = timezone.utc


@pytest.mark.parametrize(
    "now, expected",
    [
        (datetime(2026, 10, 22, 22, 0, tzinfo=UTC), datetime(2026, 10, 22, 21, 30, tzinfo=UTC)),  # Thu, after
        (datetime(2026, 10, 22, 21, 30, tzinfo=UTC), datetime(2026, 10, 22, 21, 30, tzinfo=UTC)),  # exactly at
        (datetime(2026, 10, 22, 8, 0, tzinfo=UTC), datetime(2026, 10, 21, 21, 30, tzinfo=UTC)),   # Thu, before
        (datetime(2026, 10, 26, 8, 0, tzinfo=UTC), datetime(2026, 10, 23, 21, 30, tzinfo=UTC)),   # Mon -> Fri
        (datetime(2026, 10, 25, 12, 0, tzinfo=UTC), datetime(2026, 10, 23, 21, 30, tzinfo=UTC)),  # Sun -> Fri
    ],
)
def test_previous_run_is_the_last_weekday_run_at_or_before_now(now, expected):
    assert previous_run_at(now, "21:30") == expected


def test_next_run_skips_the_weekend():
    friday_late = datetime(2026, 10, 23, 22, 0, tzinfo=UTC)
    assert next_run_at(friday_late, "21:30") == datetime(2026, 10, 26, 21, 30, tzinfo=UTC)


def test_stored_data_is_behind_only_when_a_scheduled_run_was_missed(tmp_path):
    repo, last_bar = _seed_repo(tmp_path, n_days=120)
    assert last_bar.weekday() < 5  # synthetic data uses business days
    evening = datetime(last_bar.year, last_bar.month, last_bar.day, 22, 0, tzinfo=UTC)

    assert not ingestion_is_behind(repo, evening, "21:30")  # that evening's run stored this bar
    next_weekday_evening = evening + timedelta(days=1)
    while next_weekday_evening.weekday() >= 5:
        next_weekday_evening += timedelta(days=1)
    assert ingestion_is_behind(repo, next_weekday_evening, "21:30")


def test_an_empty_database_is_not_behind_it_needs_the_bootstrap(tmp_path):
    from src.dataops.repository import MarketRepository

    repo = MarketRepository(url=f"sqlite:///{tmp_path / 'empty.db'}")
    repo.create_schema()
    assert not ingestion_is_behind(repo, datetime(2026, 10, 22, 22, 0, tzinfo=UTC), "21:30")


def _startup_with(tmp_path, monkeypatch, *, catch_up: str, behind: bool) -> list:
    repo, _ = _seed_repo(tmp_path, n_days=120)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", str(repo.engine.url))
    monkeypatch.setenv("INGESTION_CATCH_UP", catch_up)
    monkeypatch.setattr(api, "ingestion_is_behind", lambda *a, **k: behind)
    calls = []
    monkeypatch.setattr(IngestionState, "run", lambda self, *a, **k: calls.append(k) or 0)

    with TestClient(app):
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)  # long enough that a second, unwanted call would show up
    return calls


def test_startup_catches_up_once_when_behind(tmp_path, monkeypatch):
    assert len(_startup_with(tmp_path, monkeypatch, catch_up="true", behind=True)) == 1


def test_startup_does_not_fetch_when_data_is_current(tmp_path, monkeypatch):
    assert _startup_with(tmp_path, monkeypatch, catch_up="true", behind=False) == []


def test_startup_catch_up_can_be_switched_off(tmp_path, monkeypatch):
    assert _startup_with(tmp_path, monkeypatch, catch_up="false", behind=True) == []


def test_ct_status_reports_the_next_scheduled_fetch(tmp_path, monkeypatch):
    repo, _ = _seed_repo(tmp_path, n_days=120)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", str(repo.engine.url))
    monkeypatch.setenv("INGESTION_TIME_UTC", "21:30")

    with TestClient(app) as client:
        at = datetime.fromisoformat(client.get("/ct-status").json()["next_ingestion_at"])

    assert (at.hour, at.minute) == (21, 30)
    assert at.weekday() < 5
    assert at > datetime.now(UTC)
