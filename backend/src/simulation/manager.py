"""Starts, commands and kills the simulation process (worker.py).

One session at a time: each holds a torch process and may be training. The
market history is read here, through the ordinary repository, and handed over
as a file, so the child is never given the real database's address. The
child's environment is an allowlist, not a copy with secrets removed: a
credential added later can't reach it by being forgotten.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.config import DATA, RISK
from src.dataops.repository import MarketRepository
from src.simulation.scenarios import SCENARIOS, Scenario

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
# What a Python process needs to run, plus the settings that make the
# simulation behave as this deployment does. Nothing else is passed on.
ENV_ALLOWLIST = (
    "PATH", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV",
    "VIX_CRITICAL_THRESHOLD", "TARGET_SHARPE_THRESHOLD", "SHARPE_EVALUATION_WINDOW",
    "INITIAL_CASH", "TRANSACTION_COST_PCT", "DRAWDOWN_PENALTY_COEF",
)
IDLE_SECONDS = 600
MAX_SECONDS = 3600


class SimulationRefused(RuntimeError):
    """Can't be done in the current state; the message is for the viewer."""


def child_environment(workdir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
    env |= {
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONPATH": str(BACKEND_ROOT),
        "PYTHONUNBUFFERED": "1",
        "DATABASE_URL": f"sqlite:///{workdir / 'sandbox.db'}",
        "MLFLOW_TRACKING_URI": f"sqlite:///{workdir / 'mlruns.db'}",
        "OMP_NUM_THREADS": "1",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "DO_NOT_TRACK": "true",
    }
    return env


class SimulationManager:
    def __init__(self, repo: MarketRepository, clock=time.monotonic) -> None:
        self.repo = repo
        self._clock = clock
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._workdir: Path | None = None
        self._state: dict = self._empty()
        self._touched = self._started = clock()

    @staticmethod
    def _empty() -> dict:
        return {
            "status": "idle", "scenario": None, "stage": None, "day": None, "playing": False,
            "total": 0, "days": [], "events": [], "serving": None, "flags": {}, "error": None,
            "vix_threshold": RISK.vix_critical_threshold, "target_sharpe": RISK.target_sharpe_threshold,
        }

    # ------------------------------------------------------------- lifecycle
    def start(self, scenario_id: str) -> dict:
        scenario = SCENARIOS.get(scenario_id)
        if scenario is None:
            raise SimulationRefused("unknown scenario")
        with self._lock:
            self.reap()
            if self._proc is not None:
                raise SimulationRefused("a simulation is already running; stop it first")
            market = self._market(scenario)
            workdir = Path(tempfile.mkdtemp(prefix="pipeline-sim-")).resolve()
            try:
                market.to_csv(workdir / "market.csv", index=False)
                (workdir / "scenario.json").write_text(json.dumps(asdict(scenario), default=str))
                with open(workdir / "worker.log", "wb") as log:
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "src.simulation.worker", str(workdir)],
                        cwd=workdir, env=child_environment(workdir),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                        start_new_session=True,  # so anything it starts is killed with it
                    )
            except Exception:
                shutil.rmtree(workdir, ignore_errors=True)
                raise
            self._proc, self._workdir = proc, workdir
            self._state = self._empty() | {"status": "starting", "scenario": scenario_id}
            self._touched = self._started = self._clock()
            threading.Thread(target=self._read, args=(proc, workdir), daemon=True).start()
            return self.state()

    def _market(self, scenario: Scenario) -> pd.DataFrame:
        frame = self.repo.load_partition(scenario.history_start, scenario.end, tickers=DATA.tickers)
        first = pd.Timestamp(scenario.start)
        if frame.empty or frame["date"].min() > pd.Timestamp(scenario.history_start) + pd.Timedelta(days=10):
            raise SimulationRefused("the stored market history doesn't reach back far enough for this scenario")
        if frame.loc[frame["date"] >= first, "date"].nunique() < scenario.days:
            raise SimulationRefused("the stored market history doesn't cover this scenario")
        return frame

    def stop(self) -> dict:
        with self._lock:
            self._kill()
            self._state = self._empty()
            return self.state()

    def _kill(self) -> None:
        proc, workdir = self._proc, self._workdir
        self._proc = self._workdir = None
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)

    def reap(self) -> None:
        """Ends a session nobody is watching, or one that has run too long."""
        with self._lock:
            now = self._clock()
            if self._proc is not None and (now - self._touched > IDLE_SECONDS or now - self._started > MAX_SECONDS):
                logger.info("simulation ended: idle or over its time limit")
                self._kill()
                self._state = self._empty()

    # -------------------------------------------------------------- commands
    def command(self, message: dict) -> dict:
        with self._lock:
            if self._proc is None or self._state["status"] != "running":
                raise SimulationRefused("no simulation is ready")
            self._touched = self._clock()
            try:
                self._proc.stdin.write(json.dumps(message) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise SimulationRefused("the simulation has ended") from exc
            return self.state()

    def state(self, watching: bool = False) -> dict:
        with self._lock:
            if watching:
                self._touched = self._clock()
            return self._state | {"days": list(self._state["days"]), "events": list(self._state["events"])}

    # --------------------------------------------------------------- reading
    def _read(self, proc: subprocess.Popen, workdir: Path) -> None:
        for line in proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                if self._proc is not proc:
                    return  # stopped or replaced
                self._apply(message)
        code = proc.wait()
        with self._lock:
            if self._proc is not proc:
                return
            log = (workdir / "worker.log").read_text(errors="replace")[-2000:]
            logger.error("simulation process exited (%s): %s", code, log)
            self._kill()
            # The detail stays in the server log, as for a failed retrain.
            self._state |= {"status": "failed", "playing": False, "error": "The simulation stopped unexpectedly."}

    def _apply(self, m: dict) -> None:
        s, kind = self._state, m.get("type")
        if kind == "ready":
            s |= {"status": "running", "total": m["total"], "flags": m["flags"], "stage": "idle"}
        elif kind == "stage":
            s["stage"], s["day"] = m["stage"], m["day"]
            if m["stage"] == "data":
                s["serving"] = None
            if m["stage"] == "finished":
                s["playing"] = False
        elif kind == "serving":
            s["serving"] = {"day": m["day"], "served": m["served"], "failed": m["failed"]}
        elif kind == "event":
            s["events"].append({k: v for k, v in m.items() if k != "type"})
        elif kind == "day":
            s["flags"] = m.pop("flags")
            s["days"].append({k: v for k, v in m.items() if k != "type"})
        elif kind == "flags":
            s["flags"] = m["flags"]
        elif kind == "playing":
            s["playing"] = m["playing"]
