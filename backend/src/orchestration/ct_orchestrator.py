"""The Continuous Training loop (FR-13, FR-14, FR-15, FR-17, FR-20, FR-23).

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

import concurrent.futures
import logging
import multiprocessing
import tempfile
import threading
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA, RISK
from src.dataops.processing import normalize_rolling, partition_chronological
from src.dataops.repository import MarketRepository
from src.orchestration.evaluator import max_drawdown, sharpe_ratio
from src.orchestration.events import emit, finite_or_none, fmt
from src.orchestration.retrain_worker import train_and_evaluate_candidate
from src.rlops.agent import PPOAgent
from src.rlops.baselines import buy_and_hold, run_policy
from src.rlops.environment import TradingEnvironment
from src.rlops.registry import ModelRegistry
from src.serving.inference import InferenceService

logger = logging.getLogger(__name__)


def _evaluation_start(frame: pd.DataFrame) -> pd.Timestamp:
    """First date of the evaluation window: the row that gives the newest
    `sharpe_evaluation_window` daily returns, counted in trading rows rather
    than calendar days. The FR-14 trigger and the FR-17 gate both use it, so
    they judge a model on the same number of returns. A calendar window
    gave the gate about 21 returns against the trigger's 30, and holidays
    could shorten the trigger's."""
    dates = pd.to_datetime(frame["date"]).drop_duplicates().sort_values()
    return dates.iloc[max(0, len(dates) - (RISK.sharpe_evaluation_window + 1))]


class RollbackRefused(RuntimeError):
    """FR-23: a rollback that can't be done safely right now, or at all."""


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
        train_in_subprocess: bool = True,
    ) -> None:
        self.inference_service = inference_service
        self.repo = repo or MarketRepository()
        self.registry = registry or ModelRegistry(repo=self.repo)
        self.tickers = tickers
        self.retrain_train_days = retrain_train_days
        self.retrain_timesteps = retrain_timesteps
        self.retrain_n_steps = retrain_n_steps
        self.retrain_seed = retrain_seed
        # NFR-02: train candidates in a spawned child process so the GIL-bound
        # `learn()` cannot stall concurrent `/predict` calls (measured at +71%
        # to +107% p95 degradation when it ran as a thread in this process —
        # CLAUDE.md §3). Tests pass False so monkeypatching `PPOAgent.train`
        # still reaches the training call; both modes run the identical
        # `retrain_worker.train_and_evaluate_candidate`.
        self.train_in_subprocess = train_in_subprocess

        self._lock = threading.RLock()
        self._status = CTStatus.SERVING
        self._last_rolling_sharpe: float | None = None
        self._last_evaluated_at: datetime | None = None
        self._last_retrain_deferred_at: datetime | None = None
        self._last_retrain_failed_at: datetime | None = None
        self._last_retrain_error: str | None = None
        self._retrain_thread: threading.Thread | None = None
        # (active version, last bar) after the newest finished FR-17 decision.
        # Training is deterministic for a fixed seed and window, so while
        # both are unchanged a retrain could only reproduce the same outcome
        # (D-15: the same candidate was retrained up to 220 times). The
        # training settings are fixed for this object's lifetime. Kept in
        # memory, so a restart costs at most one repeated retrain.
        self._last_decision: tuple[str | None, pd.Timestamp] | None = None

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

    @property
    def last_retrain_deferred_at(self) -> datetime | None:
        """Set whenever a below-threshold Sharpe would have triggered a
        retrain but a volatile VIX deferred it instead — visible evidence
        the deferral fired, not just an assumption it would."""
        with self._lock:
            return self._last_retrain_deferred_at

    @property
    def last_retrain_failed_at(self) -> datetime | None:
        """NFR-05: set when the retrain thread itself raised — training or
        evaluation genuinely crashed, not "candidate lost the comparison".
        Without this, a crash was only ever visible as a bare stderr
        traceback from Python's default thread-exception hook: nothing
        logged through this module's own logger, and no `model_run` row,
        since `log_run()` needs a trained candidate agent that may never
        have existed."""
        with self._lock:
            return self._last_retrain_failed_at

    @property
    def last_retrain_error(self) -> str | None:
        with self._lock:
            return self._last_retrain_error

    def _load_normalised_frame(self, start, end) -> pd.DataFrame:
        raw = self.repo.load_partition(start, end, tickers=self.tickers)
        return normalize_rolling(raw)

    def _current_vix(self) -> float | None:
        ingest = self.repo.latest_ingest_info()
        return ingest["vix"] if ingest else None

    def evaluate(self, as_of: date | None = None, source: str = "scheduled") -> CTStatus:
        """FR-13: replay the incumbent over the most recent evaluation
        window, append the new days to the stored history (FR-18 reads it back),
        and trigger a retrain (FR-14) if the resulting rolling Sharpe has
        fallen below target — or if there is no incumbent to begin with.

        `as_of` defaults to today; tests pin it to a fixed date so synthetic
        fixture data (which does not span up to whatever "today" happens to
        be when the suite runs) is actually found by the lookback queries.
        `source` is "scheduled" or "manual" (the dashboard's Force Check),
        recorded on the pipeline event each outcome writes.
        """
        with self._lock:
            if self._status != CTStatus.SERVING:
                return self._status  # an evaluation or retrain is already in flight
            self._status = CTStatus.EVALUATING

        try:
            agent = self.inference_service.get_agent()
            if agent is None:
                logger.info("no incumbent model; triggering an initial training run")
                self._trigger_retrain(as_of, "ct_bootstrap", vix=self._current_vix(), source=source)
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
                emit(
                    self.repo, "evaluation_skipped",
                    f"No market data up to {end.date()}: evaluation skipped until ingestion catches up",
                    source=source, detail={"window_end": str(end.date())},
                )
                with self._lock:
                    self._status = CTStatus.SERVING
                return self.status
            eval_slice = frame[frame["date"] >= _evaluation_start(frame)].reset_index(drop=True)

            env = TradingEnvironment(eval_slice, tickers=self.tickers)
            [result] = agent.evaluate(env, n_episodes=1, deterministic=True)

            # §10.1: a Sharpe Ratio is meaningless alone. Buy-and-hold over
            # the identical slice and environment settings (same costs,
            # same accounting) gives the live equity chart its benchmark.
            # No model involved, so this costs one cheap episode.
            benchmark = run_policy(TradingEnvironment(eval_slice, tickers=self.tickers), buy_and_hold)

            self._persist_snapshots(env, result, benchmark["equity_curve"])

            recent_returns = result["returns"][-RISK.sharpe_evaluation_window :]
            rolling = sharpe_ratio(recent_returns)
            with self._lock:
                self._last_rolling_sharpe = rolling
                self._last_evaluated_at = datetime.now(timezone.utc)

            if rolling < RISK.target_sharpe_threshold:
                current_vix = self._current_vix()
                if current_vix is not None and current_vix >= RISK.vix_critical_threshold:
                    # A volatile week is exactly when the rolling Sharpe is
                    # most likely to cross the retrain trigger — training a
                    # fresh candidate on a distorted, high-volatility window
                    # is the opposite of what FR-14 is for. Reuses
                    # vix_critical_threshold rather than a second magic
                    # number: "volatile enough to force capital
                    # preservation" (I5/FR-12) is the same bar as "volatile
                    # enough not to trust a retrain." Deferred, not
                    # cancelled — the next scheduled tick re-checks both
                    # Sharpe and VIX from scratch.
                    logger.info(
                        "rolling Sharpe %.3f below target %.3f, but VIX %.1f >= critical "
                        "%.1f; deferring retrain until volatility subsides",
                        rolling, RISK.target_sharpe_threshold, current_vix, RISK.vix_critical_threshold,
                    )
                    with self._lock:
                        self._last_retrain_deferred_at = datetime.now(timezone.utc)
                        self._status = CTStatus.SERVING
                    emit(
                        self.repo, "retrain_deferred",
                        f"Rolling Sharpe {fmt(rolling)} is below the {fmt(RISK.target_sharpe_threshold)} target, "
                        f"but VIX {fmt(current_vix, 1)} is at or above {fmt(RISK.vix_critical_threshold, 1)}: "
                        "retrain deferred until volatility subsides",
                        source=source,
                        detail={"rolling_sharpe": finite_or_none(rolling), "vix": current_vix,
                                "target": RISK.target_sharpe_threshold},
                    )
                    return self.status
                last_bar = pd.Timestamp(eval_slice["date"].max())
                with self._lock:
                    already_decided = self._last_decision == (self.inference_service.active_version_id, last_bar)
                if already_decided:
                    logger.info(
                        "rolling Sharpe %.3f below target %.3f, but this model was already compared "
                        "on data through %s; waiting for a new bar",
                        rolling, RISK.target_sharpe_threshold, last_bar.date(),
                    )
                    with self._lock:
                        self._status = CTStatus.SERVING
                    emit(
                        self.repo, "retrain_deferred",
                        f"Rolling Sharpe {fmt(rolling)} is below the {fmt(RISK.target_sharpe_threshold)} target, "
                        f"but this model was already compared on data through {last_bar.date()}: "
                        "retrain deferred until a new bar arrives",
                        source=source,
                        detail={"rolling_sharpe": finite_or_none(rolling), "target": RISK.target_sharpe_threshold,
                                "last_bar": str(last_bar.date()), "reason": "no_new_data"},
                        version_id=self.inference_service.active_version_id,
                    )
                    return self.status
                logger.info("rolling Sharpe %.3f below target %.3f; retraining", rolling, RISK.target_sharpe_threshold)
                self._trigger_retrain(as_of, "ct_decay", rolling_sharpe=rolling, vix=current_vix, source=source)
            else:
                with self._lock:
                    self._status = CTStatus.SERVING
                emit(
                    self.repo, "evaluation",
                    f"Rolling Sharpe {fmt(rolling)} meets the {fmt(RISK.target_sharpe_threshold)} target: "
                    "model kept serving",
                    source=source,
                    detail={"rolling_sharpe": finite_or_none(rolling), "target": RISK.target_sharpe_threshold,
                            "window_end": str(end.date())},
                    version_id=self.inference_service.active_version_id,
                )
        except Exception:
            with self._lock:
                self._status = CTStatus.SERVING
            raise

        return self.status

    def rollback(self, version_id: str) -> str | None:
        """FR-23: an operator puts back a model that has served before.

        Refused while an evaluation or retrain is in flight: a retrain
        compares its candidate against the incumbent it read when it
        started, so swapping the incumbent underneath it could let the
        candidate be promoted over a model it was never compared with. The
        orchestrator's lock is held for the swap, so no evaluation can
        start halfway through. The version keeps its original promotion
        time (`restamp=False`). Returns the version now active.

        Rolling back doesn't stop the loop: if the restored model's rolling
        Sharpe is below target, the next evaluation retrains as usual. To
        keep it, pause retraining as well.
        """
        with self._lock:
            if self._status != CTStatus.SERVING:
                raise RollbackRefused(f"the CT loop is {self._status.value.lower()}; try again when it is serving")
            versions = {v["version_id"]: v for v in self.repo.list_model_versions(limit=500)}
            target = versions.get(version_id)
            if target is None or target["promoted_at"] is None:
                raise RollbackRefused("only a model version that has served before can be restored")
            if target["is_active"]:
                raise RollbackRefused("that model version is already serving")
            previous = self.repo.get_active_version()
            self.repo.promote_version(version_id, restamp=False)
            try:
                return self.inference_service.reload()
            except Exception as exc:
                # A missing or unreadable artifact: the old model is still
                # loaded, so the record must go back to naming it.
                logger.exception("rollback to %s failed to load", version_id)
                if previous is not None:
                    self.repo.promote_version(previous["version_id"], restamp=False)
                raise RollbackRefused("that model could not be loaded; the current one is still serving") from exc

    def _persist_snapshots(
        self, env: TradingEnvironment, result: dict, benchmark_curve: np.ndarray | None = None
    ) -> None:
        """FR-13/FR-18: appends the days not yet in the stored history.

        The replay covers the newest `sharpe_evaluation_window` returns and
        starts from the initial cash every time, so its days can't be stored
        as they are: until 2026-11-24 each check rewrote all of them, and a
        day was left, as it aged to the first row of the window, at exactly
        the initial cash with no Sharpe and no drawdown (D-20). Now a day is
        written once, by the first check that sees it, and never again: the
        equity continues from the last stored value by that day's replayed
        return, so a model promoted later (chosen on those same days) can't
        rewrite the record of the one it replaced.

        The drawdown and the Sharpe are the check's own: over the replay up to
        that day, with no Sharpe unless it covers the full window. Days missed
        while nothing was running are filled from the replay, so they have no
        Sharpe; the first day of a window after a longer gap is skipped,
        because the replay measures no return for it.

        `benchmark_curve` is buy-and-hold over the same slice, chained the
        same way. Where the history has none yet it starts level with the
        agent's equity."""
        dates = env._dates[: len(result["equity_curve"])]
        equity_curve, returns = result["equity_curve"], result["returns"]
        window = RISK.sharpe_evaluation_window
        previous = self.repo.latest_snapshot()

        for i, ts in enumerate(dates):
            day = pd.Timestamp(ts).date()
            if previous is not None and (day <= previous["date"] or i == 0):
                continue
            has_benchmark = benchmark_curve is not None and i < len(benchmark_curve)
            if previous is None:
                equity, cumulative = float(equity_curve[0]), 0.0
                benchmark = float(benchmark_curve[0]) if has_benchmark else None
            else:
                growth = float(equity_curve[i] / equity_curve[i - 1])
                equity = previous["equity_value"] * growth
                cumulative = (1.0 + (previous["cumulative_return"] or 0.0)) * growth - 1.0
                base = previous["benchmark_equity"]
                base = previous["equity_value"] if base is None else base
                benchmark = base * float(benchmark_curve[i] / benchmark_curve[i - 1]) if has_benchmark else None
            recent = returns[:i][-window:]
            self.repo.record_snapshot(
                day,
                equity,
                rolling_sharpe_30d=sharpe_ratio(recent) if len(recent) == window else None,
                max_drawdown=max_drawdown(equity_curve[: i + 1]),
                cumulative_return=cumulative,
                benchmark_equity=benchmark,
            )
            previous = {"date": day, "equity_value": equity, "cumulative_return": cumulative,
                        "benchmark_equity": benchmark}

    def _trigger_retrain(
        self,
        as_of: date | None,
        reason: str,
        rolling_sharpe: float | None = None,
        vix: float | None = None,
        source: str = "scheduled",
    ) -> None:
        """`reason` is "ct_bootstrap" or "ct_decay" from `evaluate()`
        ("analysis" from measurement scripts that force a retrain) —
        recorded on the candidate's `model_run` row with the Sharpe and VIX
        the decision was made on, so autonomous retrains are queryable
        (NFR-07).

        FR-23: while an operator has paused retraining, nothing is trained,
        whichever path asked. The pause is checked here rather than in each
        caller so that no path (bootstrap, decay, a script's forced retrain)
        can get around it. It is a deferral, like the VIX one: the next
        evaluation after the pause is lifted decides afresh."""
        controls = self.inference_service.controls
        if controls.is_on("pause_retraining"):
            pause = controls.get("pause_retraining")
            logger.info("retrain (%s) not started: retraining is paused by %s", reason, pause["set_by"])
            with self._lock:
                self._last_retrain_deferred_at = datetime.now(timezone.utc)
                if self._status != CTStatus.RETRAINING:
                    self._status = CTStatus.SERVING
            because = (
                f"rolling Sharpe {fmt(rolling_sharpe)} is below the {fmt(RISK.target_sharpe_threshold)} target"
                if reason == "ct_decay" else "no model is serving yet" if reason == "ct_bootstrap"
                else f"a retrain was requested ({reason})"
            )
            emit(
                self.repo, "retrain_deferred",
                f"A retrain is due ({because}), but an operator has paused retraining: "
                "the current model keeps serving",
                source=source,
                detail={"reason": "operator_paused", "trigger": reason,
                        "rolling_sharpe": finite_or_none(rolling_sharpe), "vix": vix,
                        "paused_by": pause["set_by"]},
                version_id=self.inference_service.active_version_id,
            )
            return
        with self._lock:
            if self._status == CTStatus.RETRAINING:
                return
            self._status = CTStatus.RETRAINING
        if reason == "ct_bootstrap":
            summary = "No model is serving yet: training the first one"
        elif reason == "ct_decay":
            summary = (
                f"Rolling Sharpe {fmt(rolling_sharpe)} fell below the {fmt(RISK.target_sharpe_threshold)} "
                "target: retraining a candidate in the background while the current model keeps serving"
            )
        else:
            summary = f"Retrain started ({reason})"
        emit(
            self.repo, "retrain_started", summary, source=source,
            detail={"reason": reason, "rolling_sharpe": finite_or_none(rolling_sharpe), "vix": vix},
        )
        thread = threading.Thread(
            target=self._retrain_and_maybe_promote,
            args=(as_of, reason, rolling_sharpe, vix, source),
            daemon=True,
        )
        self._retrain_thread = thread
        thread.start()

    def _retrain_and_maybe_promote(
        self,
        as_of: date | None = None,
        reason: str | None = None,
        rolling_sharpe: float | None = None,
        vix: float | None = None,
        source: str = "scheduled",
    ) -> None:
        """FR-15: runs in its own thread, and the CPU-bound training itself
        runs in a separate *process* (`retrain_worker`, NFR-02) — a thread
        alone kept serving available (NFR-03) but not fast, because PPO's
        `learn()` holds the GIL: measured +71% to +107% p95 degradation with
        in-thread training. `InferenceService.predict()` is untouched while
        this executes; this method does not touch its lock until (and
        unless) promotion. The FR-17 decision, `log_run` and promotion all
        stay in this process — the child only trains and evaluates.
        """
        try:
            end = pd.Timestamp(as_of or date.today())
            # Three calendar days per trading row comfortably covers weekends
            # and holidays. DR-06 holds by construction: training ends the day
            # before the evaluation window starts.
            recent = self.repo.load_partition(
                end - pd.Timedelta(days=3 * RISK.sharpe_evaluation_window), end, tickers=self.tickers
            )
            if recent.empty:
                raise ValueError(f"no market data to evaluate a candidate on, up to {end.date()}")
            eval_start = _evaluation_start(recent)
            eval_end = pd.to_datetime(recent["date"]).max()
            train_end = eval_start - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=self.retrain_train_days)

            # Only the incumbent's replay needs data in this process; the
            # worker loads and partitions its own copy, so the heavy lifting
            # (normalise + train + candidate replay) all happens in the
            # child when `train_in_subprocess` is on.
            frame = self._load_normalised_frame(train_start, eval_end)
            _train_df, eval_df = partition_chronological(frame, train_start, train_end, eval_start, eval_end)

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = str(Path(tmp_dir) / "candidate.zip")
                worker_kwargs = dict(
                    db_url=self.repo.url,
                    tickers=tuple(self.tickers),
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    eval_start=str(pd.Timestamp(eval_start).date()),
                    eval_end=str(pd.Timestamp(eval_end).date()),
                    n_steps=self.retrain_n_steps,
                    timesteps=self.retrain_timesteps,
                    seed=self.retrain_seed,
                    model_path=model_path,
                )
                if self.train_in_subprocess:
                    # `spawn`, never `fork`: this process holds live threads,
                    # SQLAlchemy pools and torch state. A child exception is
                    # re-raised here, so the except below records a training
                    # crash exactly as it did for in-thread training.
                    ctx = multiprocessing.get_context("spawn")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                        worker_result = pool.submit(train_and_evaluate_candidate, **worker_kwargs).result()
                    logger.info("candidate trained in child process %s", worker_result["pid"])
                else:
                    worker_result = train_and_evaluate_candidate(**worker_kwargs)
                candidate = PPOAgent.load(model_path)
                # A loaded agent has empty hyperparameters; restore the
                # trainer's own so `log_run` records them (FR-08/NFR-07).
                candidate.hyperparameters = worker_result["hyperparameters"]
            candidate_sharpe = worker_result["sharpe"]

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

            accepted = candidate_sharpe > incumbent_sharpe
            # FR-23: a pause switched on while this candidate was training
            # still stops it being promoted. It is logged as WITHHELD, not
            # REJECTED, because the FR-17 gate didn't turn it down: an
            # operator did, and counting it as a gate rejection would
            # overstate the gate (the D-15 lesson).
            withheld = accepted and self.inference_service.controls.is_on("pause_retraining")
            # Every trained candidate is logged, win or lose — FR-17's
            # "abort promotion" must still leave a record that a candidate
            # was tried and rejected, or "how many candidates were held
            # back" is an unanswerable question rather than a queryable one.
            run_id = self.registry.log_run(
                candidate,
                train_start,
                train_end,
                eval_start,
                eval_end,
                {"sharpe_ratio": candidate_sharpe},
                status="WITHHELD" if withheld else "COMPLETED" if accepted else "REJECTED",
                trigger_reason=reason,
                trigger_rolling_sharpe=rolling_sharpe,
                trigger_vix=vix,
            )

            if withheld:
                logger.info("candidate beat the incumbent, but retraining is paused; promotion withheld")
                emit(
                    self.repo, "candidate_withheld",
                    f"Candidate beat the current model out of sample (Sharpe {fmt(candidate_sharpe)} vs "
                    f"{fmt(incumbent_sharpe)}), but an operator paused retraining while it trained: "
                    "not promoted, current model kept",
                    source=source,
                    detail={"candidate_sharpe": finite_or_none(candidate_sharpe),
                            "incumbent_sharpe": finite_or_none(incumbent_sharpe), "reason": reason},
                    run_id=run_id,
                )
                # Not recorded as a finished FR-17 decision: once the pause is
                # lifted, the gate gets to decide this window properly.
                return

            if accepted:
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

            # Only a finished decision counts: a crash above never gets here,
            # so the next evaluation retries it.
            with self._lock:
                self._last_decision = (self.inference_service.active_version_id, eval_end)

            outcome = {
                "candidate_sharpe": finite_or_none(candidate_sharpe),
                "incumbent_sharpe": finite_or_none(incumbent_sharpe),
                "eval_start": str(eval_start.date()),
                "eval_end": str(eval_end.date()),
                "reason": reason,
            }
            if accepted and not np.isfinite(incumbent_sharpe):
                emit(
                    self.repo, "candidate_promoted",
                    f"First model trained (out-of-sample Sharpe {fmt(candidate_sharpe)}): promoted and serving",
                    source=source, detail=outcome, run_id=run_id, version_id=version_id,
                )
            elif accepted:
                emit(
                    self.repo, "candidate_promoted",
                    f"Candidate beat the current model out of sample (Sharpe {fmt(candidate_sharpe)} vs "
                    f"{fmt(incumbent_sharpe)}): promoted and hot-swapped with no restart",
                    source=source, detail=outcome, run_id=run_id, version_id=version_id,
                )
            else:
                emit(
                    self.repo, "candidate_rejected",
                    f"Candidate did not beat the current model out of sample (Sharpe {fmt(candidate_sharpe)} vs "
                    f"{fmt(incumbent_sharpe)}): current model kept",
                    source=source, detail=outcome, run_id=run_id,
                )
        except Exception as exc:  # noqa: BLE001 - NFR-05: a crashed retrain must not crash the process
            # A genuine crash (bad data, an OOM, a training exception) is
            # different from FR-17's "candidate lost the comparison" — that
            # is a normal, logged outcome; this is not. Without this catch,
            # the only trace was a bare stderr traceback from Python's
            # default thread-exception hook: invisible to this module's own
            # logger, and no model_run row, since log_run() needs a trained
            # candidate that may never have existed (e.g. if training
            # itself threw). The incumbent is untouched either way — this
            # method never promotes anything until every step above
            # succeeds — so NFR-05's actual guarantee ("state consistent")
            # already held before this fix; what was missing was knowing it
            # happened at all.
            logger.exception("retrain failed; incumbent remains active")
            with self._lock:
                self._last_retrain_failed_at = datetime.now(timezone.utc)
                self._last_retrain_error = str(exc)
            # The exception text stays in the server log, as on /ct-status.
            emit(
                self.repo, "retrain_failed",
                "Retraining failed; the current model kept serving (details in the server log)",
                source=source, detail={"reason": reason},
            )
        finally:
            with self._lock:
                self._status = CTStatus.SERVING
