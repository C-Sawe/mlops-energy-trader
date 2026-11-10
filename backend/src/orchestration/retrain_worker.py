"""Trains a CT candidate in a separate process (NFR-02).

The retrain used to run as a thread inside the serving process, and the GIL
made FR-15 a half-truth: serving *continued* through a retrain (NFR-03 held,
zero failed requests) but latency degraded far past NFR-02's 10% bound —
three saved runs measured +71% to +107% p95 (CLAUDE.md §3 corrections,
2026-10-07). PPO's `learn()` is CPU-bound Python/torch that holds the GIL for
most of the retrain, so no amount of locking discipline in the serving thread
could fix it from inside one process.

This module is the fix: everything CPU-heavy about a candidate — loading and
normalising its training data, training, and replaying it over the evaluation
partition — happens in a `spawn`ed child process that shares no interpreter
with serving. The parent (`CTOrchestrator._retrain_and_maybe_promote`) keeps
everything that must observe or mutate shared state: the incumbent's replay
(seconds, it is one pass over ~30 rows), `log_run`, the FR-17 decision, and
promotion. The child communicates back a plain picklable dict plus the saved
model artifact on disk; it never touches MLflow, the inference service, or
the promotion tables.

`spawn`, not `fork`: the serving process holds live threads, SQLAlchemy pools
and torch state, none of which survive a fork safely on macOS or in general.
A spawned child re-imports its modules (a few seconds of torch import),
which is noise against a retrain measured in minutes on the deployment VM.
"""
from __future__ import annotations

import os

from src.dataops.processing import normalize_rolling, partition_chronological
from src.dataops.repository import MarketRepository
from src.orchestration.evaluator import sharpe_ratio
from src.rlops.agent import PPOAgent
from src.rlops.environment import TradingEnvironment


def train_and_evaluate_candidate(
    db_url: str,
    tickers: tuple[str, ...],
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    n_steps: int,
    timesteps: int,
    seed: int,
    model_path: str,
) -> dict:
    """Train one candidate and replay it over the evaluation partition.

    Runs in a child process in production (NFR-02), or in-process when
    `CTOrchestrator` is constructed with `train_in_subprocess=False` — the
    test suite's mode, so monkeypatching `PPOAgent.train` still reaches the
    training call. Either way this is the single code path that trains a CT
    candidate; the two modes differ only in which process executes it.

    Dates are ISO strings because the arguments cross a process boundary.
    The trained model is saved to `model_path`; the returned dict carries
    the out-of-sample Sharpe, the hyperparameters `log_run` needs (a loaded
    agent has an empty `hyperparameters`), and the worker's pid so the
    orchestrator can log where training actually ran.
    """
    repo = MarketRepository(url=db_url)
    frame = normalize_rolling(repo.load_partition(train_start, eval_end, tickers=tuple(tickers)))
    train_df, eval_df = partition_chronological(frame, train_start, train_end, eval_start, eval_end)

    candidate = PPOAgent(
        TradingEnvironment(train_df, tickers=tuple(tickers)), n_steps=n_steps, seed=seed
    )
    candidate.train(total_timesteps=timesteps)

    [result] = candidate.evaluate(TradingEnvironment(eval_df, tickers=tuple(tickers)), n_episodes=1)
    candidate.save(model_path)

    return {
        "sharpe": float(sharpe_ratio(result["returns"])),
        "hyperparameters": candidate.hyperparameters,
        "pid": os.getpid(),
    }
