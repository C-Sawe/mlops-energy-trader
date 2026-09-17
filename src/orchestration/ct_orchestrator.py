"""The Continuous Training loop (FR-13, FR-14, FR-15, FR-17, FR-20).

This is the closed feedback loop CLAUDE.md §6 describes as the project's
actual contribution: telemetry from serving feeds a drift evaluator, which
feeds a retrain decision, which feeds back into serving via a hot reload —
autonomously, without the pipeline going down while it happens.

Sim2Real note (CLAUDE.md §1): there is no live broker connection in this
project's scope. "The incumbent's performance" here means replaying the
incumbent's own policy against the most recent *real, already-ingested*
market data through `TradingEnvironment` — a faithful simulation of what it
would have done, not a report of what it actually executed. That is exactly
the boundary Section 1.7 already concedes; this module does not pretend
otherwise.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from src.config import DATA, RISK
from src.dataops.processing import normalize_rolling, partition_chronological
from src.dataops.repository import MarketRepository
from src.orchestration.evaluator import max_drawdown, sharpe_ratio
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment
from src.rlops.registry import ModelRegistry
from src.serving.inference import InferenceService

logger = logging.getLogger(__name__)


class CTStatus(str, Enum):
    """FR-20: what the dashboard's CT pipeline indicator renders."""

    SERVING = "SERVING"
    EVALUATING = "EVALUATING"
    RETRAINING = "RETRAINING"


class CTOrchestrator:
    def __init__(
        self,
        inference_service: InferenceService,
        repo: MarketRepository | None = None,
        registry: ModelRegistry | None = None,
        tickers: tuple[str, ...] = DATA.tickers,
        retrain_train_days: int = 730,
        retrain_timesteps: int = 20_000,
        retrain_n_steps: int = 2048,
        retrain_seed: int = 0,
    ) -> None:
        self.inference_service = inference_service
        self.repo = repo or MarketRepository()
        self.registry = registry or ModelRegistry(repo=self.repo)
        self.tickers = tickers
        self.retrain_train_days = retrain_train_days
        self.retrain_timesteps = retrain_timesteps
        self.retrain_n_steps = retrain_n_steps
        self.retrain_seed = retrain_seed

        self._lock = threading.RLock()
        self._status = CTStatus.SERVING
        self._last_rolling_sharpe: float | None = None
        self._last_evaluated_at: datetime | None = None
        self._retrain_thread: threading.Thread | None = None

    @property
    def status(self) -> CTStatus:
        with self._lock:
            return self._status

    @property
    def last_rolling_sharpe(self) -> float | None:
        with self._lock:
            return self._last_rolling_sharpe

    @property
    def last_evaluated_at(self) -> datetime | None:
        with self._lock:
            return self._last_evaluated_at

    def _load_normalised_frame(self, start, end) -> pd.DataFrame:
        raw = self.repo.load_partition(start, end, tickers=self.tickers)
        return normalize_rolling(raw)

    def evaluate(self, as_of: date | None = None) -> CTStatus:
        """FR-13: replay the incumbent over the most recent evaluation
        window, persist a fresh snapshot per day (FR-18 reads these back),
        and trigger a retrain (FR-14) if the resulting rolling Sharpe has
        fallen below target — or if there is no incumbent to begin with.

        `as_of` defaults to today; tests pin it to a fixed date so synthetic
        fixture data (which does not span up to whatever "today" happens to
        be when the suite runs) is actually found by the lookback queries.
        """
        with self._lock:
            if self._status != CTStatus.SERVING:
                return self._status  # an evaluation or retrain is already in flight
            self._status = CTStatus.EVALUATING

        try:
            agent = self.inference_service.get_agent()
            if agent is None:
                logger.info("no incumbent model; triggering an initial training run")
                self._trigger_retrain(as_of)
                return self.status

            # Fetch far more than the evaluation window itself: normalize_rolling's
            # zscore_window (60 trading days, DR-07) plus the sma_window warm-up
            # need to be satisfied from *before* the window starts, or every row
            # in a lookback sized to just the window would be NaN. Normalise over
            # the full lookback, then slice down to the actual evaluation window
            # — the same "normalise before partitioning" pattern documented in
            # CLAUDE.md §7, applied to a single slice instead of a train/eval pair.
            end = pd.Timestamp(as_of or date.today())
            lookback_start = end - pd.Timedelta(days=200)
            window_start = end - pd.Timedelta(days=RISK.sharpe_evaluation_window + 15)

            frame = self._load_normalised_frame(lookback_start, end)
            if frame.empty:
                # Ingestion hasn't caught up to `as_of` yet — a recoverable,
                # expected condition (the next scheduled tick will likely
                # find data), not a programming error. Logged and skipped
                # rather than raised, so a stale ingestion job degrades the
                # CT loop visibly instead of crashing the API request that
                # triggered this evaluation.
                logger.warning(
                    "no market data in [%s, %s]; skipping this evaluation", lookback_start.date(), end.date()
                )
                with self._lock:
                    self._status = CTStatus.SERVING
                return self.status
            eval_slice = frame[frame["date"] >= window_start].reset_index(drop=True)

            env = TradingEnvironment(eval_slice, tickers=self.tickers)
            [result] = agent.evaluate(env, n_episodes=1, deterministic=True)

            self._persist_snapshots(env, result)

            recent_returns = result["returns"][-RISK.sharpe_evaluation_window :]
            rolling = sharpe_ratio(recent_returns)
            with self._lock:
                self._last_rolling_sharpe = rolling
                self._last_evaluated_at = datetime.now(timezone.utc)

            if rolling < RISK.target_sharpe_threshold:
                logger.info("rolling Sharpe %.3f below target %.3f; retraining", rolling, RISK.target_sharpe_threshold)
                self._trigger_retrain(as_of)
            else:
                with self._lock:
                    self._status = CTStatus.SERVING
        except Exception:
            with self._lock:
                self._status = CTStatus.SERVING
            raise

        return self.status

    def _persist_snapshots(self, env: TradingEnvironment, result: dict) -> None:
        """FR-13/FR-18: one snapshot per simulated day, each computed only
        from the days up to and including it — the dashboard's charts must
        never show a metric that used information from a future day."""
        dates = env._dates[: len(result["equity_curve"])]
        equity_curve = result["equity_curve"]
        returns = result["returns"]

        for i, equity in enumerate(equity_curve):
            window = equity_curve[: i + 1]
            window_returns = returns[:i][-RISK.sharpe_evaluation_window :]
            self.repo.record_snapshot(
                pd.Timestamp(dates[i]).date(),
                float(equity),
                rolling_sharpe_30d=sharpe_ratio(window_returns) if len(window_returns) >= 2 else None,
                max_drawdown=max_drawdown(window),
                cumulative_return=float(equity / equity_curve[0] - 1.0),
            )

    def _trigger_retrain(self, as_of: date | None = None) -> None:
        with self._lock:
            if self._status == CTStatus.RETRAINING:
                return
            self._status = CTStatus.RETRAINING
        thread = threading.Thread(
            target=self._retrain_and_maybe_promote, args=(as_of,), daemon=True
        )
        self._retrain_thread = thread
        thread.start()

    def _retrain_and_maybe_promote(self, as_of: date | None = None) -> None:
        """FR-15: runs in its own thread — `InferenceService.predict()` is
        untouched while this executes, since it only ever reads the
        currently-loaded agent under `InferenceService`'s own lock, and this
        method does not touch that lock until (and unless) promotion.
        """
        try:
            end = pd.Timestamp(as_of or date.today())
            train_start = end - pd.Timedelta(days=self.retrain_train_days + RISK.sharpe_evaluation_window)
            train_end = end - pd.Timedelta(days=RISK.sharpe_evaluation_window + 1)
            eval_start = train_end + pd.Timedelta(days=1)
            eval_end = end

            frame = self._load_normalised_frame(train_start, eval_end)
            train_df, eval_df = partition_chronological(frame, train_start, train_end, eval_start, eval_end)

            candidate_env = TradingEnvironment(train_df, tickers=self.tickers)
            candidate = PPOAgent(candidate_env, n_steps=self.retrain_n_steps, seed=self.retrain_seed)
            candidate.train(total_timesteps=self.retrain_timesteps)

            eval_env_candidate = TradingEnvironment(eval_df, tickers=self.tickers)
            [candidate_result] = candidate.evaluate(eval_env_candidate, n_episodes=1)
            candidate_sharpe = sharpe_ratio(candidate_result["returns"])

            incumbent = self.inference_service.get_agent()
            if incumbent is not None:
                eval_env_incumbent = TradingEnvironment(eval_df, tickers=self.tickers)
                [incumbent_result] = incumbent.evaluate(eval_env_incumbent, n_episodes=1)
                incumbent_sharpe = sharpe_ratio(incumbent_result["returns"])
            else:
                incumbent_sharpe = -np.inf  # nothing incumbent; any candidate clears the bar

            logger.info(
                "candidate sharpe=%.3f vs incumbent sharpe=%.3f", candidate_sharpe, incumbent_sharpe
            )

            if candidate_sharpe > incumbent_sharpe:
                run_id = self.registry.log_run(
                    candidate,
                    train_start,
                    train_end,
                    eval_start,
                    eval_end,
                    {"sharpe_ratio": candidate_sharpe},
                )
                from src.dataops.models import ModelRun

                with self.repo.session() as session:
                    mlflow_ref = session.get(ModelRun, run_id).mlflow_run_ref
                version_id = self.registry.register_version(run_id, artifact_uri=f"runs:/{mlflow_ref}/model")
                self.repo.promote_version(version_id)
                self.inference_service.reload()  # FR-16: hot-swap, no restart
                logger.info("promoted %s", version_id)
            else:
                # FR-17: the incumbent is retained; promotion is aborted.
                logger.info("candidate did not beat incumbent; incumbent retained")
        finally:
            with self._lock:
                self._status = CTStatus.SERVING
