#!/usr/bin/env python3
"""NFR-04, re-measured from a genuinely cold process.

CLAUDE.md §7 flags a real concern with the original ~8ms figure: every
existing measurement of `InferenceService` reconstruction happened in a
process where a `ModelRegistry` had already been constructed first (to
train/promote a model), which calls `mlflow.set_tracking_uri()` as a side
effect and silently primes MLflow's global state before
`InferenceService.reload()` ever needs it. `scripts/broker_paper_trade_test.py`
already proved this priming matters functionally (CLAUDE.md §7's
`resolve_mlflow_tracking_uri` entry) — this script settles whether it also
matters for the *timing* claim.

This script is meant to be invoked once per process (see the shell loop in
its own usage note below) — each invocation is a fresh `python3` process
that constructs `MarketRepository` and `InferenceService` as the *only*
things that ever touch MLflow, with no `ModelRegistry` constructed first,
matching what a real unplanned-termination-and-restart looks like against
a real, already-active model.

Usage (N=5 genuinely separate processes, matching the original methodology):
    for i in 1 2 3 4 5; do python scripts/measure_nfr04_cold.py; done
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imports happen untimed, before the measurement starts — the ~8ms original
# figure measured object *construction*, not interpreter/module-import
# overhead, and this re-measurement should isolate the same thing.
from src.dataops.repository import MarketRepository  # noqa: E402
from src.serving.inference import InferenceService  # noqa: E402


def main() -> int:
    repo = MarketRepository()
    active = repo.get_active_version()
    if active is None:
        print("no active model_version — nothing real to recover to", file=sys.stderr)
        return 1

    start = time.perf_counter()
    service = InferenceService(repo=repo)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert service.active_version_id == active["version_id"]
    print(f"{elapsed_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
