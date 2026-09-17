"""FastAPI inference gateway (FR-10, FR-18, FR-19, FR-20, IR-04, IR-05, NFR-08).

Single-user local operation (CLAUDE.md §12's Sprint 4 note): no user table,
one optional bearer token guarding the exposed surface. `/ct/evaluate` and
the background scheduler both call the same `CTOrchestrator.evaluate()` —
the scheduler is FR-13's "on a schedule" satisfied automatically; the
endpoint exists so the dashboard can also trigger one on demand rather than
only ever waiting for the timer.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import RISK, SERVING
from src.dataops.repository import MarketRepository
from src.orchestration.ct_orchestrator import CTOrchestrator
from src.orchestration.ingestion_scheduler import DEFAULT_TRAILING_WINDOW_DAYS, run_ingestion_tick
from src.rlops.registry import ModelRegistry
from src.serving.inference import InferenceService, NoActiveModelError
from src.serving.schemas import (
    CTStatusResponse,
    DecisionLogEntry,
    DecisionLogResponse,
    EquityPoint,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TelemetryResponse,
    TickerDecision,
)

CT_EVALUATION_INTERVAL_SECONDS = int(os.environ.get("CT_EVALUATION_INTERVAL_SECONDS", "300"))
# Daily by default: yfinance's OHLCV bars only change once per trading day
# (§9), so anything more frequent just re-fetches the same values. Set
# lower for a live demo of FR-01 actually landing new rows without a human
# re-running scripts/run_ingestion.py.
INGESTION_INTERVAL_SECONDS = int(os.environ.get("INGESTION_INTERVAL_SECONDS", "86400"))

# Constructed in `lifespan`, not here: `InferenceService.__init__` queries
# the database immediately (it loads whatever model_version is active), so
# doing this at *import* time would make importing this module reach for a
# real database connection using whatever DATABASE_URL happens to be set —
# including from an unrelated earlier test. Deferring to lifespan startup
# means each `TestClient(app)` context gets a fresh, correctly-isolated set
# of instances built against whatever repo that test configured.
repo: MarketRepository
service: InferenceService
registry: ModelRegistry
orchestrator: CTOrchestrator
_scheduler_task: asyncio.Task | None = None
_ingestion_task: asyncio.Task | None = None


async def _run_scheduler() -> None:
    while True:
        await asyncio.sleep(CT_EVALUATION_INTERVAL_SECONDS)
        try:
            orchestrator.evaluate()
        except Exception:  # noqa: BLE001 - the scheduler must survive a bad tick
            logging.getLogger(__name__).exception("scheduled CT evaluation failed")


async def _run_ingestion_scheduler() -> None:
    """FR-01: ingest without manual intervention. Runs `fetch_market_data`'s
    blocking network I/O off the event loop via `asyncio.to_thread` so a
    slow or retrying fetch never stalls `/predict` (FR-15's "keep serving"
    applies here too, not just during a retrain)."""
    while True:
        await asyncio.sleep(INGESTION_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(run_ingestion_tick, repo)
        except Exception:  # noqa: BLE001 - the scheduler must survive a bad tick
            logging.getLogger(__name__).exception("scheduled ingestion tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global repo, service, registry, orchestrator, _scheduler_task, _ingestion_task
    repo = MarketRepository()
    service = InferenceService(repo=repo)
    registry = ModelRegistry(repo=repo)
    orchestrator = CTOrchestrator(service, repo=repo, registry=registry)
    _scheduler_task = asyncio.create_task(_run_scheduler())
    _ingestion_task = asyncio.create_task(_run_ingestion_scheduler())
    yield
    _scheduler_task.cancel()
    _ingestion_task.cancel()


app = FastAPI(title="MLOps Trading Inference API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-user local operation; no cross-origin secrets are at risk
    allow_methods=["*"],
    allow_headers=["*"],
)


def _verify_token(authorization: str | None = Header(default=None)) -> None:
    """IR-05/NFR-08 boundary check, not a users table (see module docstring)."""
    if SERVING.bearer_token is None:
        return
    if authorization != f"Bearer {SERVING.bearer_token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", active_version_id=service.active_version_id)


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(_verify_token)])
def predict(request: PredictRequest) -> PredictResponse:
    """FR-10. NFR-08: `PredictRequest`'s own validation already rejected a
    malformed body before this function runs; a missing model is a 503
    (the service is real but not ready), not an unhandled crash."""
    try:
        result = service.predict(request.positions, request.cash_weight)
    except NoActiveModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(
        decisions=[TickerDecision(**d) for d in result["decisions"]],
        failsafe_triggered=result["failsafe_triggered"],
        vix_at_decision=result["vix_at_decision"],
        version_id=result["version_id"],
        decided_at=result["decided_at"],
    )


@app.get("/telemetry", response_model=TelemetryResponse, dependencies=[Depends(_verify_token)])
def telemetry(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> TelemetryResponse:
    """FR-18: the equity curve, rolling Sharpe and max drawdown series."""
    end = end or date.today()
    start = start or (end - timedelta(days=180))
    df = repo.list_snapshots(start, end)
    if df.empty:
        return TelemetryResponse(points=[])
    points = [
        EquityPoint(
            date=row["date"].date(),
            equity_value=row["equity_value"],
            rolling_sharpe_30d=row["rolling_sharpe_30d"],
            max_drawdown=row["max_drawdown"],
            cumulative_return=row["cumulative_return"],
        )
        for row in df.to_dict(orient="records")
    ]
    return TelemetryResponse(points=points)


@app.get("/decisions", response_model=DecisionLogResponse, dependencies=[Depends(_verify_token)])
def decisions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
) -> DecisionLogResponse:
    """FR-19: a paginated decision log."""
    items, total = repo.list_decisions(page=page, page_size=page_size)
    return DecisionLogResponse(
        items=[DecisionLogEntry(**item) for item in items],
        page=page,
        page_size=page_size,
        total=total,
    )


CYCLE_WINDOW_DAYS = 90  # "this quarter", for the dashboard's cycle-history card


@app.get("/ct-status", response_model=CTStatusResponse, dependencies=[Depends(_verify_token)])
def ct_status() -> CTStatusResponse:
    """FR-20: serving / evaluating / retraining, for the dashboard's status indicator."""
    ingest = repo.latest_ingest_info()
    since = date.today() - timedelta(days=CYCLE_WINDOW_DAYS)
    cycle = repo.get_cycle_stats(since)
    return CTStatusResponse(
        status=orchestrator.status.value,
        active_version_id=service.active_version_id,
        rolling_sharpe=orchestrator.last_rolling_sharpe,
        target_sharpe_threshold=RISK.target_sharpe_threshold,
        last_evaluated_at=orchestrator.last_evaluated_at,
        last_ingest_date=ingest["date"] if ingest else None,
        current_vix=ingest["vix"] if ingest else None,
        vix_critical_threshold=RISK.vix_critical_threshold,
        cycle_window_days=CYCLE_WINDOW_DAYS,
        training_runs=cycle["total_runs"],
        promoted_count=cycle["promoted"],
        rejected_count=cycle["rejected"],
    )


@app.post("/ct/evaluate", response_model=CTStatusResponse, dependencies=[Depends(_verify_token)])
def trigger_evaluation(as_of: date | None = Query(default=None)) -> CTStatusResponse:
    """On-demand counterpart to the background scheduler — returns
    immediately (FR-15): if this evaluation decides to retrain, that
    happens in its own thread, same as the scheduled path.

    `as_of` is an operational escape hatch, not a routine parameter: it
    lets an operator force an evaluation anchored to a specific date
    (backfill after an ingestion gap, catching up post-outage, or — as
    here — bootstrapping against a checkout whose ingested data doesn't
    reach all the way to the literal current date) instead of always
    defaulting to `date.today()`.
    """
    orchestrator.evaluate(as_of=as_of)
    return ct_status()


@app.post("/ingest/run", response_model=CTStatusResponse, dependencies=[Depends(_verify_token)])
def trigger_ingestion(
    end: date | None = Query(default=None),
    window_days: int = Query(default=DEFAULT_TRAILING_WINDOW_DAYS, ge=1),
) -> CTStatusResponse:
    """FR-01, on-demand counterpart to the background ingestion scheduler:
    pulls fresh data immediately instead of waiting for
    INGESTION_INTERVAL_SECONDS. Runs synchronously — a sync route function
    already executes in Starlette's threadpool, so the blocking network
    call inside `run_ingestion_tick` does not stall the event loop or any
    concurrent `/predict` call.

    `end`/`window_days` exist for the same operational reason `/ct/evaluate`
    takes `as_of`: forcing a specific range for backfill or a demo, rather
    than always defaulting to `date.today()`.
    """
    run_ingestion_tick(repo, window_days=window_days, end=end)
    return ct_status()
