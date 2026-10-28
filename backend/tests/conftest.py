"""Suite-wide test settings.

Tests never reach the network (CLAUDE.md §11). Every `TestClient(app)`
runs the API's startup, and the ingestion scheduler's startup catch-up
would fetch from Yahoo whenever a test's synthetic data looks stale, which
it always does. Tests that exercise the catch-up turn it back on with a
stubbed ingestion.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_startup_ingestion(monkeypatch):
    monkeypatch.setenv("INGESTION_CATCH_UP", "false")
