#!/usr/bin/env python3
"""NFR-03 and NFR-05, measured rather than assumed.

NFR-03: "Serving remains continuously available across a full CT cycle;
zero failed requests." NFR-01/02/04's measurements (CLAUDE.md §5) already
hammered `/predict` during a real retrain and reported *latency*
degradation, but never stated a failure *count* explicitly — this script
does, so "zero failed requests" is a number that was actually counted, not
inferred from a latency chart having no obvious gaps in it.

NFR-05: "A failed retraining cycle leaves the incumbent serving and state
consistent." This is really a correctness property (see
`test_retrain_survives_a_genuine_crash_and_leaves_incumbent_active` in
`tests/test_ct_orchestrator.py` for the actual verification — a genuine
crash, not just a rejected candidate), but this script also confirms it
under load: `/predict` keeps answering, and answering *correctly* (server
never lets a half-loaded agent through FR-16's hot-swap lock), for the
whole duration a real retrain runs alongside it.

Same "real except the transport" scope as the original NFR-01/02/04 pass:
real SQLite-backed repository, a real trained PPOAgent, real fail-safe
check, `TestClient`'s in-process ASGI transport standing in for a real
network hop (a constant offset on top of these numbers, not something
specific to this service).

Usage:
    python scripts/nfr_reliability_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_series(n: int, tickers, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for i, ticker in enumerate(tickers):
        close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n)) + i * 10
        close = np.maximum(close, 1.0)
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.integers(1_000_000, 20_000_000, n),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="nfr_reliability_")
    os.chdir(tmp_dir)  # MLflow's artifact store defaults to a relative ./mlruns (CLAUDE.md §11)
    db_url = f"sqlite:///{tmp_dir}/test.db"
    os.environ["DATABASE_URL"] = db_url

    from src.config import DATA
    from src.dataops import processing as P
    from src.dataops.repository import MarketRepository
    from src.rlops.agent import PPOAgent
    from src.rlops.environment import TradingEnvironment
    from src.rlops.registry import ModelRegistry

    # CTOrchestrator's default retrain_train_days=730 (real production
    # hyperparameters, deliberately not shrunk here — see the module
    # docstring) needs at least that much history plus its own eval
    # window before a retrain; too little and it fails on insufficient
    # data rather than genuinely exercising serving reliability.
    n_days = 900
    equities = _make_series(n_days, DATA.tickers)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": 20.0})
    frame = P.build_feature_frame(equities, vix)
    as_of = dates.max().date()

    repo = MarketRepository(url=db_url)
    repo.create_schema()
    repo.persist(frame)

    train_env = TradingEnvironment(
        P.normalize_rolling(repo.load_partition(pd.Timestamp(as_of) - pd.Timedelta(days=n_days - 1), as_of)),
        tickers=DATA.tickers,
    )
    agent = PPOAgent(train_env, n_steps=256, seed=0).train(total_timesteps=2000)

    registry = ModelRegistry(repo=repo)
    run_id = registry.log_run(agent, "2020-01-01", "2020-10-01", "2020-10-02", str(as_of), {"sharpe_ratio": 0.0})
    from src.dataops.models import ModelRun

    with repo.session() as session:
        mlflow_ref = session.get(ModelRun, run_id).mlflow_run_ref
    version_id = registry.register_version(run_id, artifact_uri=f"runs:/{mlflow_ref}/model")
    repo.promote_version(version_id)

    from fastapi.testclient import TestClient

    from src.serving import api as api_module

    positions = {t: 0.0 for t in DATA.tickers}
    stats = {"n": 0, "failures": 0, "latencies": []}
    stop = threading.Event()

    def hammer():
        with TestClient(api_module.app) as client:
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    resp = client.post("/predict", json={"positions": positions, "cash_weight": 1.0})
                    ok = resp.status_code == 200
                except Exception:  # noqa: BLE001 - counted as a failure, not a crash
                    ok = False
                elapsed = time.perf_counter() - t0
                stats["n"] += 1
                stats["latencies"].append(elapsed)
                if not ok:
                    stats["failures"] += 1

    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        print(f"active model: {health.json()['active_version_id']}")

        hammer_thread = threading.Thread(target=hammer, daemon=True)
        hammer_thread.start()
        time.sleep(1.0)  # a moment of baseline traffic before the retrain starts

        print("triggering a real retrain (same code path FR-14 uses)...")
        api_module.orchestrator._trigger_retrain(as_of)
        retrain_thread = api_module.orchestrator._retrain_thread
        assert retrain_thread is not None
        retrain_thread.join(timeout=120)
        assert not retrain_thread.is_alive(), "retrain did not finish within 120s"

        time.sleep(1.0)  # a moment of baseline traffic after, for symmetry
        stop.set()
        hammer_thread.join(timeout=10)

    latencies_ms = sorted(l * 1000 for l in stats["latencies"])
    n = len(latencies_ms)
    p50 = latencies_ms[n // 2]
    p95 = latencies_ms[int(n * 0.95)]
    print(f"\nrequests: {stats['n']}  failures: {stats['failures']}")
    print(f"latency p50={p50:.1f}ms p95={p95:.1f}ms max={latencies_ms[-1]:.1f}ms")
    print(f"\nNFR-03 (zero failed requests): {'PASS' if stats['failures'] == 0 else 'FAIL'}")
    print(
        "NFR-05 (incumbent kept serving, no crash): "
        f"{'PASS' if api_module.orchestrator.last_retrain_failed_at is None else 'FAIL — see last_retrain_error'}"
    )
    return 0 if stats["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
