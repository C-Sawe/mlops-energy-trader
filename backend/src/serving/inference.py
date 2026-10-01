"""Inference service: fail-safe, action mapping, hot reload (FR-11, FR-12, FR-16).

I5: the VIX fail-safe lives here, in the serving layer — not in the agent,
not in the model — because it must survive model promotion. A component
that promotion replaces (the loaded PPOAgent) cannot be where a safety
mechanism lives, or promoting a new model would silently promote away the
fail-safe with it.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd

from src.config import DATA, RISK, resolve_mlflow_tracking_uri
from src.dataops.processing import normalize_rolling
from src.dataops.repository import MarketRepository
from src.rlops.agent import PPOAgent
from src.rlops.environment import FEATURE_COLUMNS

# Calendar days of trailing history to fetch for one prediction. Needs to
# comfortably clear the zscore_window (60 trading days, DR-07) plus the
# sma_window warm-up (20), accounting for weekends/holidays.
_LOOKBACK_DAYS = 200


class NoActiveModelError(RuntimeError):
    """Raised when a prediction is requested but no model_version is active."""


class InferenceService:
    """FR-10's logic: fail-safe first, then the active model, always logged."""

    def __init__(self, repo: MarketRepository | None = None, tickers: tuple[str, ...] = DATA.tickers):
        self.repo = repo or MarketRepository()
        self.tickers = tickers
        self._lock = threading.RLock()
        self._agent: PPOAgent | None = None
        self._active_version_id: str | None = None
        self.reload()

    def reload(self) -> str | None:
        """FR-16: (re)load whichever model_version is currently active,
        without restarting the service. Safe to call while `predict()` is
        being served concurrently — the swap is atomic under the lock, so a
        caller never sees a half-loaded agent.
        """
        active = self.repo.get_active_version()
        with self._lock:
            if active is None:
                self._agent = None
                self._active_version_id = None
                return None
            # Must not assume `ModelRegistry` already set this: this
            # service can legitimately be the first thing in a fresh
            # process to touch MLflow (a real restart with an
            # already-active model does exactly this), and MLflow's own
            # ambient default tracking URI is not this project's
            # `mlruns.db` (CLAUDE.md §7 — found via
            # scripts/broker_paper_trade_test.py).
            mlflow.set_tracking_uri(resolve_mlflow_tracking_uri())
            local_path = mlflow.artifacts.download_artifacts(
                f"{active['artifact_uri']}/model.zip"
            )
            self._agent = PPOAgent.load(local_path)
            self._active_version_id = active["version_id"]
            return self._active_version_id

    @property
    def active_version_id(self) -> str | None:
        with self._lock:
            return self._active_version_id

    def get_agent(self) -> PPOAgent | None:
        """The currently-loaded incumbent, for the CT orchestrator's
        out-of-sample comparison (FR-17) — not used for prediction, which
        goes through `predict()` so the fail-safe always runs first."""
        with self._lock:
            return self._agent

    def _discretize(self, weight: float) -> str:
        """FR-11: continuous weight -> discrete action at the configured thresholds."""
        if weight > RISK.buy_threshold:
            return "BUY"
        if weight < RISK.sell_threshold:
            return "SELL"
        return "HOLD"

    def _load_recent_frame(self) -> tuple[pd.DataFrame, float | None]:
        latest_states = [s for t in self.tickers if (s := self.repo.latest_state(t))]
        if not latest_states:
            raise NoActiveModelError("no market data ingested yet")
        end = max(s["date"] for s in latest_states)
        start = end - pd.Timedelta(days=_LOOKBACK_DAYS)

        raw = self.repo.load_partition(start, end, tickers=self.tickers)
        frame = normalize_rolling(raw)

        latest_vix = None
        if not frame.empty:
            latest_row = frame.sort_values("date").iloc[-1]
            latest_vix = float(latest_row["vix"]) if pd.notna(latest_row["vix"]) else None
        return frame, latest_vix

    def _observation_from_frame(
        self, frame: pd.DataFrame, positions: dict[str, float], cash_weight: float
    ) -> np.ndarray:
        z_cols = [f"{c}_z" for c in FEATURE_COLUMNS]
        features: list[float] = []
        for ticker in self.tickers:
            sub = frame[frame["ticker"] == ticker].sort_values("date")
            if sub.empty or sub[z_cols].iloc[-1].isna().any():
                raise NoActiveModelError(
                    f"{ticker}: not enough trailing history for a complete feature vector"
                )
            features.extend(sub[z_cols].iloc[-1].tolist())

        position_weights = [positions[t] for t in self.tickers]
        return np.asarray(features + position_weights + [cash_weight], dtype=np.float32)

    def _failsafe_decisions(self) -> list[dict]:
        """FR-12: bypass inference entirely and return capital-preservation."""
        return [
            {"ticker": t, "raw_weight": None, "discrete_action": "LIQUIDATE"} for t in self.tickers
        ]

    def _model_decisions(self, observation: np.ndarray) -> list[dict]:
        with self._lock:
            if self._agent is None:
                raise NoActiveModelError("no model_version is currently active")
            action = self._agent.predict(observation, deterministic=True)
        return [
            {"ticker": t, "raw_weight": float(w), "discrete_action": self._discretize(float(w))}
            for t, w in zip(self.tickers, action)
        ]

    def predict(self, positions: dict[str, float], cash_weight: float) -> dict:
        """FR-10: the full request, fail-safe checked before any inference.

        Returns a plain dict (not a schema) so this class has no dependency
        on `src.serving.schemas` — the API layer adapts it, not the other
        way around.
        """
        frame, vix = self._load_recent_frame()
        failsafe = vix is not None and vix >= RISK.vix_critical_threshold

        if failsafe:
            decisions = self._failsafe_decisions()
        else:
            observation = self._observation_from_frame(frame, positions, cash_weight)
            decisions = self._model_decisions(observation)

        decided_at = datetime.now(timezone.utc)
        version_id = self.active_version_id
        # trading_decision.version_id is NOT NULL (it must trace to the
        # weights that produced it, I6), so a fail-safe firing before any
        # model has ever been promoted has nothing to attribute itself to
        # and cannot be persisted — it is still returned to the caller
        # below, just not logged. This is a bootstrap-only edge case, not a
        # steady-state one: once any version has been promoted, this branch
        # never applies again for the life of the deployment.
        for decision in decisions:
            if version_id is not None:
                self.repo.record_decision(
                    version_id,
                    decision["ticker"],
                    decision["raw_weight"],
                    decision["discrete_action"],
                    vix,
                    failsafe_triggered=failsafe,
                )

        return {
            "decisions": decisions,
            "failsafe_triggered": failsafe,
            "vix_at_decision": vix,
            "version_id": version_id,
            "decided_at": decided_at,
        }
