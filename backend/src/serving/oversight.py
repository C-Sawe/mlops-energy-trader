"""Operator overrides (FR-23): the human's switches over the autonomous loop.

Three switches, each of which can only restrain the system:

  halt_trading         every `/predict` answer is capital preservation,
                       exactly as when VIX trips FR-12. The model is not
                       consulted.
  pause_retraining     the CT loop keeps evaluating and drawing its charts,
                       but never trains or promotes a candidate.
  pause_paper_trading  the daily paper cycle records a skip instead of
                       placing orders.

None of the switches can make the system trade, train or promote where it
otherwise wouldn't, and the VIX fail-safe can't be switched off.

Rollback (`CTOrchestrator.rollback`) is different: it does choose which model
decides, without a new FR-17 comparison. It is limited to versions that have
served before, and is not claimed to be more cautious.

What the switches don't do: cancel orders already queued at the broker, or
close paper positions at once. A halt reaches the paper account at its next
daily cycle, and not at all while that cycle is paused.

Owned by `InferenceService` for the same reason the VIX fail-safe is (I5):
it must survive model promotion, so it can't live in anything promotion
replaces. The orchestrator and paper trader reach it through the service
they already hold, so no new import edge is added (NFR-06).

State is the `operator_control` table, so a restart can't silently lift a
halt (NFR-04). Reads come from an in-memory copy, because `/predict` checks
`halt_trading` on every request and NFR-01's 20 ms budget has no room for an
extra query. Writes go to the database first and only then to memory: if
the write fails, nothing changes and the caller gets the error.
"""
from __future__ import annotations

import threading

from src.dataops.models import OPERATOR_CONTROLS
from src.dataops.repository import MarketRepository

# What the dashboard and the timeline call each switch.
CONTROL_LABELS = {
    "halt_trading": "Trading halt",
    "pause_retraining": "Retraining pause",
    "pause_paper_trading": "Paper-trading pause",
}


class OperatorControls:
    def __init__(self, repo: MarketRepository) -> None:
        self.repo = repo
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._state: dict[str, dict] = repo.get_controls()

    def is_on(self, name: str) -> bool:
        with self._lock:
            return bool(self._state[name]["enabled"])

    def get(self, name: str) -> dict:
        with self._lock:
            return dict(self._state[name])

    def all(self) -> list[dict]:
        """In OPERATOR_CONTROLS order, so the dashboard's list is stable."""
        with self._lock:
            return [dict(self._state[n]) for n in OPERATOR_CONTROLS]

    def set(self, name: str, enabled: bool, reason: str | None, set_by: str) -> dict:
        # One writer at a time, so two requests can't leave memory and the
        # database disagreeing about which came last. Readers don't wait.
        with self._write_lock:
            state = self.repo.set_control(name, enabled, reason, set_by)  # raises before memory changes
            with self._lock:
                self._state[name] = state
        return dict(state)
