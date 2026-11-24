"""The simulation's child process: `python -m src.simulation.worker <dir>`.

Why a process and not an object inside the API: MLflow's tracking location is
global to a process, so a sandbox sharing the API's process could redirect a
real retrain's records into the sandbox, or the reverse. Separate processes
also keep its training off the serving interpreter (NFR-02). The manager
gives this one a fresh working directory, its own SQLite database and MLflow
store, and an environment holding no database URL or credential but those.

What is real: InferenceService, CTOrchestrator (evaluation, retrain, FR-17
gate, hot reload), the operator switches, the registry, the event log and the
environment's accounting, at production settings. What isn't: bars are
replayed from stored history and not fetched; one trading day passes per step;
and a VIX override, a training crash or a switch can be injected by the viewer.

The pipeline can only see what is in the sandbox database, and a day's bar is
written there when that day starts, so a decision can't see a later one (I1).

Protocol: one JSON command per line on stdin, one JSON message per line on
stdout. stdin closing means the parent has gone, and the process exits.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA
from src.dataops.processing import normalize_rolling
from src.dataops.repository import MarketRepository
from src.orchestration.ct_orchestrator import CTOrchestrator
from src.orchestration.events import emit, finite_or_none
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment
from src.serving.inference import InferenceService
from src.serving.oversight import CONTROL_LABELS

SERVING_PROBE_SECONDS = 0.2


class SandboxViolation(RuntimeError):
    """The process was not given an isolated environment; it refuses to run."""


def check_isolated(workdir: Path) -> None:
    """Refuses to start unless the database and MLflow store are both inside
    `workdir`. The manager sets them; this is the check that it did."""
    for name in ("DATABASE_URL", "MLFLOW_TRACKING_URI"):
        value = os.environ.get(name, "")
        if not value.startswith("sqlite:///"):
            raise SandboxViolation(f"{name} is not a sandbox SQLite file")
        if workdir.resolve() not in Path(value.removeprefix("sqlite:///")).resolve().parents:
            raise SandboxViolation(f"{name} is outside the sandbox directory")


class Simulation:
    def __init__(self, workdir: Path, scenario: dict, send) -> None:
        self.send = send
        self.days_limit = int(scenario["days"])
        market = pd.read_csv(workdir / "market.csv", float_precision="round_trip")
        market["date"] = pd.to_datetime(market["date"])
        start = pd.Timestamp(scenario["start"])
        self._market = market
        self._days = [d for d in sorted(market["date"].unique()) if d >= start][: self.days_limit]
        if not self._days:
            raise ValueError("no trading days in this window")

        self.repo = MarketRepository()
        self.repo.create_schema()
        history = market[market["date"] < self._days[0]]
        self.repo.persist(history)

        self.service = InferenceService(repo=self.repo)
        # In this process, not a further one: the injected crash patches
        # PPOAgent.train here, and the sandbox already has its own interpreter.
        self.orchestrator = CTOrchestrator(self.service, repo=self.repo, train_in_subprocess=False)
        self.vix_override: float | None = None
        self.crash_next = False
        self._crash_after_bootstrap = bool(scenario.get("crash_first_retrain"))
        self._cash, self._shares = 100_000.0, np.zeros(len(DATA.tickers))
        self._last_event = 0
        self._i = 0

        real_train = PPOAgent.train

        def train(agent, *args, **kwargs):
            if self.crash_next:
                self.crash_next = False
                raise RuntimeError("injected training failure (simulation)")
            return real_train(agent, *args, **kwargs)

        PPOAgent.train = train  # this process only

        # A new deployment has no model: the loop trains its first one.
        self._stage("retrain", None)
        last = pd.Timestamp(history["date"].max()).date()
        self.orchestrator.evaluate(as_of=last)
        self._wait_for_retrain(None)
        if self.service.active_version_id is None:
            raise RuntimeError("the first model could not be trained")
        self._events(None)
        self.crash_next = self._crash_after_bootstrap
        self.send({"type": "ready", "total": len(self._days), **self._flags()})

    # ------------------------------------------------------------- messages
    def _flags(self) -> dict:
        return {
            "flags": {
                "vix_override": self.vix_override,
                "crash_next": self.crash_next,
                **{c["name"]: c["enabled"] for c in self.service.controls.all()},
            }
        }

    def _stage(self, stage: str, day) -> None:
        self.send({"type": "stage", "stage": stage, "day": str(day) if day else None})

    def _events(self, day) -> None:
        events, _ = self.repo.list_events(limit=50)
        for e in reversed([e for e in events if e["event_id"] > self._last_event]):
            self._last_event = e["event_id"]
            self.send({
                "type": "event", "day": str(day) if day else None, "id": e["event_id"],
                "kind": e["kind"], "source": e["source"], "summary": e["summary"], "detail": e["detail"],
            })

    # ------------------------------------------------------------- commands
    def handle(self, cmd: dict) -> None:
        name = cmd.get("cmd")
        if name == "vix":
            self.vix_override = None if cmd.get("value") is None else float(cmd["value"])
        elif name == "crash":
            self.crash_next = bool(cmd.get("enabled", True))
        elif name == "control":
            control, enabled = cmd["name"], bool(cmd["enabled"])
            if self.service.controls.get(control)["enabled"] != enabled:
                self.service.controls.set(control, enabled, "Simulation", "viewer")
                emit(self.repo, "operator_override",
                     f"{CONTROL_LABELS[control]} {'switched on' if enabled else 'lifted'}", source="manual",
                     detail={"control": control, "enabled": enabled})
                self._events(self._current_day())
        self.send({"type": "flags", **self._flags()})

    def _current_day(self):
        return pd.Timestamp(self._days[self._i - 1]).date() if self._i else None

    # ------------------------------------------------------------------ day
    @property
    def finished(self) -> bool:
        return self._i >= len(self._days)

    def step(self) -> None:
        day_ts = pd.Timestamp(self._days[self._i])
        self._i += 1
        day = day_ts.date()

        self._stage("data", day)
        bars = self._market[self._market["date"] == day_ts].copy()
        real_vix = float(bars["vix"].iloc[0])
        injected = self.vix_override is not None
        if injected:
            bars["vix"] = self.vix_override
        self.repo.persist(bars)

        self._stage("decide", day)
        env = self._portfolio(day)
        weights = env.current_weights
        equity = env._equity(env._prices[env._step_idx])
        positions = {t: float(w) for t, w in zip(DATA.tickers, weights)}
        result = self.service.predict(positions, float(env.cash / equity))
        failsafe = result["failsafe_triggered"]
        targets = np.array([0.0 if failsafe else d["raw_weight"] for d in result["decisions"]])
        env._rebalance(np.clip(targets, -1.0, 1.0))
        self._cash, self._shares = env.cash, env.shares.copy()

        self._stage("evaluate", day)
        before = self.service.active_version_id
        self.orchestrator.evaluate(as_of=day)
        served = failed = 0
        if self.orchestrator._retrain_thread is not None and self.orchestrator._retrain_thread.is_alive():
            self._stage("retrain", day)
            served, failed = self._wait_for_retrain(day, positions)
        self._events(day)

        active = self.repo.get_active_version()
        self.send({
            "type": "day", "day": str(day), "index": self._i,
            "vix": real_vix if not injected else self.vix_override, "vix_injected": injected,
            "failsafe": failsafe, "failsafe_reason": result["failsafe_reason"],
            "decisions": [
                {"ticker": d["ticker"], "weight": finite_or_none(d["raw_weight"]), "action": d["discrete_action"]}
                for d in result["decisions"]
            ],
            "exposure": float(np.sum(np.abs(self._shares * env._prices[env._step_idx])) / equity),
            "rolling_sharpe": finite_or_none(self.orchestrator.last_rolling_sharpe),
            "promoted": self.service.active_version_id != before,
            "model": active["version_id"][:8] if active else None,
            "served_during_retrain": served, "failed_during_retrain": failed,
            **self._flags(),
        })
        self._stage("finished" if self.finished else "idle", day)

    def _portfolio(self, day) -> TradingEnvironment:
        """The environment at the close of `day`, holding what was held: its
        own accounting and costs, seeing only bars up to `day`."""
        frame = normalize_rolling(
            self.repo.load_partition(pd.Timestamp(day) - pd.Timedelta(days=200), day, tickers=DATA.tickers)
        )
        env = TradingEnvironment(frame, tickers=DATA.tickers)
        env._step_idx = env._n_steps - 1
        env.cash, env.shares = self._cash, self._shares.copy()
        return env

    def _wait_for_retrain(self, day, positions: dict | None = None) -> tuple[int, int]:
        """Keeps asking the serving layer for decisions while a candidate
        trains (FR-15, NFR-03), counting the answers and the failures."""
        thread, served, failed = self.orchestrator._retrain_thread, 0, 0
        while thread is not None and thread.is_alive():
            if positions is not None:
                try:
                    self.service.predict(positions, 0.0)
                    served += 1
                except Exception:  # noqa: BLE001 - a failure is the thing being counted
                    failed += 1
                self.send({"type": "serving", "day": str(day), "served": served, "failed": failed})
            thread.join(timeout=SERVING_PROBE_SECONDS)
        return served, failed


def run(workdir: Path, commands: queue.Queue, send) -> None:
    check_isolated(workdir)
    scenario = json.loads((workdir / "scenario.json").read_text())
    sim = Simulation(workdir, scenario, send)
    delay: float | None = None  # None: paused
    while True:
        try:
            cmd = commands.get(timeout=None if delay is None or sim.finished else delay)
        except queue.Empty:
            sim.step()
            continue
        if cmd is None:
            return
        if cmd.get("cmd") == "step":
            delay = None
            if not sim.finished:
                sim.step()
        elif cmd.get("cmd") == "play":
            delay = float(np.clip(cmd.get("delay", 1.0), 0.1, 10.0))
        elif cmd.get("cmd") == "pause":
            delay = None
        else:
            sim.handle(cmd)
        send({"type": "playing", "playing": delay is not None and not sim.finished})


def main() -> None:
    os.nice(10)  # the real service comes first (NFR-02)
    # Libraries print to stdout; only protocol messages may reach the parent.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def send(message: dict) -> None:
        with lock:
            protocol.write(json.dumps(message, default=str) + "\n")

    commands: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except ValueError:
                continue
        # The parent closed the pipe or died. Training can't be interrupted
        # from another thread, so leave at once; the directory is disposable.
        os._exit(0)

    threading.Thread(target=read, daemon=True).start()
    try:
        run(Path(sys.argv[1]), commands, send)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, then the process ends
        import traceback

        traceback.print_exc()
        send({"type": "error", "error": type(exc).__name__, "at": datetime.now().isoformat()})
        time.sleep(0.1)
        os._exit(1)


if __name__ == "__main__":
    main()
