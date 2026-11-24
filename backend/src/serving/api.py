"""FastAPI inference gateway (FR-10, FR-18, FR-19, FR-20, FR-21, FR-23, IR-04, IR-05, NFR-08).

Access: dashboard sign-in for allowlisted people (FR-21/FR-22,
src/serving/auth.py), and one optional bearer token for scripts and
server-to-server calls. There are no roles: everyone admitted can do
everything, as in the original single-user design. `/ct/evaluate` and
the background scheduler both call the same `CTOrchestrator.evaluate()` —
the scheduler is FR-13's "on a schedule" satisfied automatically; the
endpoint exists so the dashboard can also trigger one on demand rather than
only ever waiting for the timer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from src.config import ASSISTANT, AUTH, DATA, INGESTION, NEWS, PAPER, RISK, SERVING, SIMULATION
from src.dataops.news import NewsFeed, NewsUnavailable
from src.dataops.repository import MarketRepository
from src.orchestration.ct_orchestrator import CTOrchestrator, RollbackRefused
from src.orchestration.events import emit
from src.execution.alpaca_broker import AlpacaBroker, BrokerError
from src.orchestration.ingestion_scheduler import (
    DEFAULT_TRAILING_WINDOW_DAYS,
    IngestionState,
    ingestion_is_behind,
)
from src.orchestration.paper_trader import PaperTrader, next_run_at
from src.rlops.registry import ModelRegistry
from src.serving import assistant, auth
from src.simulation.manager import SimulationManager, SimulationRefused
from src.simulation.scenarios import SCENARIOS
from src.serving.inference import InferenceService, NoActiveModelError
from src.serving.oversight import CONTROL_LABELS
from src.serving.schemas import (
    AssistantChatRequest,
    AssistantStatusResponse,
    CandlesResponse,
    CTStatusResponse,
    DecisionLogEntry,
    DecisionLogResponse,
    EquityPoint,
    EventsResponse,
    HealthResponse,
    NewsResponse,
    OperatorControlName,
    OperatorControlOut,
    OversightResponse,
    PaperStatusResponse,
    PipelineEventOut,
    PredictRequest,
    PredictResponse,
    RollbackCandidate,
    RollbackRequest,
    SetControlRequest,
    SimulationCommand,
    SimulationResponse,
    SimulationScenario,
    SimulationStartRequest,
    TelemetryResponse,
    TickerDecision,
)

CT_EVALUATION_INTERVAL_SECONDS = int(os.environ.get("CT_EVALUATION_INTERVAL_SECONDS", "300"))

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
ingestion: IngestionState
paper: PaperTrader
simulation: SimulationManager
_scheduler_task: asyncio.Task | None = None
_ingestion_task: asyncio.Task | None = None
_paper_tasks: list[asyncio.Task] = []
# Overridable so tests can substitute a broker over httpx.MockTransport —
# nothing in the test suite reaches Alpaca (CLAUDE.md §11).
_broker_factory = AlpacaBroker


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
    applies here too, not just during a retrain). Goes through `ingestion`
    (not the bare `run_ingestion_tick`) so `/ct-status` can show this tick
    actually happening — the same reasoning FR-20 already applies to the CT
    loop's own status.

    Runs every weekday at INGESTION_TIME_UTC, after the US close, not every
    N seconds from process start: an interval timer restarts with every
    deploy, crash recovery or reload, and on 2026-10-21 repeated restarts
    kept postponing it until two trading days were missed. A restart can
    still miss a scheduled time, so startup fetches once if the newest
    stored bar is older than the last scheduled run should have produced."""
    log = logging.getLogger(__name__)
    if INGESTION.catch_up_on_startup:
        try:
            behind = await asyncio.to_thread(
                ingestion_is_behind, repo, datetime.now(timezone.utc), INGESTION.run_time_utc
            )
            if behind:
                log.info("stored market data is behind the last scheduled run; catching up now")
                await asyncio.to_thread(ingestion.run, repo)
        except Exception:  # noqa: BLE001 - the scheduler must survive a bad catch-up
            log.exception("startup ingestion catch-up failed")
    while True:
        now = datetime.now(timezone.utc)
        await asyncio.sleep((next_run_at(now, INGESTION.run_time_utc) - now).total_seconds())
        try:
            await asyncio.to_thread(ingestion.run, repo)
        except Exception:  # noqa: BLE001 - the scheduler must survive a bad tick
            log.exception("scheduled ingestion tick failed")


def _paper_cycle_job(source: str = "scheduled") -> str:
    """Ingest today's bar first, so the cycle decides on the close that
    just happened rather than on whatever the interval-based ingestion
    scheduler last fetched. An ingestion failure is logged, not fatal: the
    cycle's own "no bar for today" check then skips trading."""
    try:
        ingestion.run(repo, source=source)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("paper cycle: pre-trade ingestion failed")
    broker = _broker_factory()
    try:
        return paper.run_cycle(broker, source=source)
    finally:
        broker.close()


def _paper_sync_job() -> None:
    broker = _broker_factory()
    try:
        paper.sync(broker)
    finally:
        broker.close()


async def _run_paper_scheduler() -> None:
    while True:
        now = datetime.now(timezone.utc)
        await asyncio.sleep((next_run_at(now, PAPER.run_time_utc) - now).total_seconds())
        try:
            outcome = await asyncio.to_thread(_paper_cycle_job)
            logging.getLogger(__name__).info("paper cycle: %s", outcome)
        except Exception:  # noqa: BLE001 - the scheduler must survive a bad tick
            logging.getLogger(__name__).exception("scheduled paper cycle failed")


async def _run_paper_sync() -> None:
    while True:
        await asyncio.sleep(PAPER.sync_interval_seconds)
        try:
            await asyncio.to_thread(_paper_sync_job)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("paper account sync failed")


async def _reap_simulation() -> None:
    """Ends a simulation nobody is watching, so it can't hold memory."""
    while True:
        await asyncio.sleep(30)
        try:
            await asyncio.to_thread(simulation.reap)
        except Exception:  # noqa: BLE001 - must survive a bad tick
            logging.getLogger(__name__).exception("could not reap the simulation")


def _check_auth_config() -> None:
    """CLAUDE.md §11, explicit failure: sign-in configured without a session
    secret would sign cookies with a key that changes on every restart, and
    without an allowlist would lock everyone out with no hint why."""
    if not AUTH.enabled:
        return
    missing = [
        name
        for name, value in (("SESSION_SECRET", AUTH.session_secret), ("ALLOWED_EMAILS", AUTH.allowed_emails))
        if not value
    ]
    if missing:
        methods = [*AUTH.enabled_providers, *(["password"] if AUTH.password_login else [])]
        raise RuntimeError(f"sign-in is enabled ({', '.join(methods)}) but {', '.join(missing)} is not set")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global repo, service, registry, orchestrator, ingestion, paper, simulation
    global _scheduler_task, _ingestion_task, _paper_tasks
    _check_auth_config()
    repo = MarketRepository()
    repo.create_schema()
    _app.state.repo = repo  # password sign-in reads accounts through it (src/serving/auth.py)
    service = InferenceService(repo=repo)
    registry = ModelRegistry(repo=repo)
    orchestrator = CTOrchestrator(service, repo=repo, registry=registry)
    ingestion = IngestionState()
    # NFR-01: a successful ingestion drops the cached feature frame, so a
    # same-date revision (the tick's trailing-window upsert) is picked up
    # by the next /predict rather than served stale until a new bar.
    ingestion.on_success = service.invalidate_cache
    paper = PaperTrader(service, repo)
    _scheduler_task = asyncio.create_task(_run_scheduler())
    _ingestion_task = asyncio.create_task(_run_ingestion_scheduler())
    _paper_tasks = []
    if PAPER.enabled:
        _paper_tasks = [
            asyncio.create_task(_run_paper_scheduler()),
            asyncio.create_task(_run_paper_sync()),
        ]
    simulation = SimulationManager(repo)
    reaper = asyncio.create_task(_reap_simulation())
    yield
    reaper.cancel()
    simulation.stop()
    _scheduler_task.cancel()
    _ingestion_task.cancel()
    for task in _paper_tasks:
        task.cancel()


app = FastAPI(title="MLOps Trading Inference API", lifespan=lifespan)
# Always installed: auth.current_user() reads request.session on every
# guarded request. The secret is read at import, i.e. at process start in
# any real run; the random fallback only signs cookies while sign-in is off
# (_check_auth_config refuses to start without SESSION_SECRET otherwise).
# SameSite=Lax keeps the cookie off cross-site POSTs, which is the CSRF
# guard for the state-changing endpoints.
app.add_middleware(
    SessionMiddleware,
    secret_key=AUTH.session_secret or secrets.token_urlsafe(32),
    session_cookie="mlops_session",
    max_age=7 * 24 * 3600,
    same_site="lax",
    https_only=AUTH.public_api_url.startswith("https://"),
)
if AUTH.enabled:
    # Cookies need credentialed CORS, and that needs an explicit origin:
    # "*" with credentials would let any site call the API as the signed-in
    # user. Only local dev (Vite on :5173, API on :8000) is cross-origin;
    # production serves both through nginx under one origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[AUTH.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # single-user local operation; no cross-origin secrets are at risk
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(auth.router)


def _verify_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    """IR-05/NFR-08 boundary check and FR-21's gate. Accepts either
    credential that is configured: the bearer token (scripts,
    server-to-server) or a signed-in, allowlisted session (the dashboard).
    With neither configured the API is open, as in local dev and tests."""
    token = SERVING.bearer_token
    if token is None and not AUTH.enabled:
        return
    if token is not None and authorization == f"Bearer {token}":
        return
    if AUTH.enabled and auth.current_user(request) is not None:
        return
    raise HTTPException(
        status_code=401,
        detail="sign in or supply a valid bearer token" if AUTH.enabled else "invalid or missing bearer token",
    )


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
        failsafe_reason=result["failsafe_reason"],
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
            benchmark_equity=row["benchmark_equity"],
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


def _active_model_label(active: dict | None) -> str | None:
    """A human-readable label like "September 2026 #2" for the dashboard's
    "Active model" tile — a presentation convenience only. It is never a
    substitute for `active_version_id`, which stays the real identifier
    NFR-07's traceability chain (decision -> version -> run -> partition)
    actually depends on; the frontend keeps the raw ID available too (as a
    hover title), it just isn't the headline text anymore."""
    if active is None or active["promoted_at"] is None:
        return None
    sequence = repo.get_promotion_sequence(active["promoted_at"])
    return f"{active['promoted_at'].strftime('%B %Y')} #{sequence}"


@app.get("/ct-status", response_model=CTStatusResponse, dependencies=[Depends(_verify_token)])
def ct_status() -> CTStatusResponse:
    """FR-20: serving / evaluating / retraining, for the dashboard's status indicator."""
    ingest = repo.latest_ingest_info()
    since = date.today() - timedelta(days=CYCLE_WINDOW_DAYS)
    cycle = repo.get_cycle_stats(since)
    active = repo.get_active_version()
    return CTStatusResponse(
        status=orchestrator.status.value,
        active_version_id=service.active_version_id,
        active_model_label=_active_model_label(active),
        rolling_sharpe=orchestrator.last_rolling_sharpe,
        target_sharpe_threshold=RISK.target_sharpe_threshold,
        last_evaluated_at=orchestrator.last_evaluated_at,
        last_retrain_deferred_at=orchestrator.last_retrain_deferred_at,
        last_retrain_failed_at=orchestrator.last_retrain_failed_at,
        last_ingest_date=ingest["date"] if ingest else None,
        current_vix=ingest["vix"] if ingest else None,
        vix_critical_threshold=RISK.vix_critical_threshold,
        cycle_window_days=CYCLE_WINDOW_DAYS,
        training_runs=cycle["total_runs"],
        promoted_count=cycle["promoted"],
        rejected_count=cycle["rejected"],
        autonomous_retrains=cycle["autonomous_runs"],
        autonomous_promoted=cycle["autonomous_promoted"],
        ingestion_status=ingestion.status.value,
        last_ingestion_attempted_at=ingestion.last_attempted_at,
        last_ingestion_ok=ingestion.last_ok,
        next_ingestion_at=next_run_at(datetime.now(timezone.utc), INGESTION.run_time_utc),
        controls=_controls_out(),
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
    orchestrator.evaluate(as_of=as_of, source="manual")
    return ct_status()


@app.post("/ingest/run", response_model=CTStatusResponse, dependencies=[Depends(_verify_token)])
def trigger_ingestion(
    end: date | None = Query(default=None),
    window_days: int = Query(default=DEFAULT_TRAILING_WINDOW_DAYS, ge=1),
) -> CTStatusResponse:
    """FR-01, on-demand counterpart to the background ingestion scheduler:
    pulls fresh data immediately instead of waiting for the next
    INGESTION_TIME_UTC. Runs synchronously — a sync route function
    already executes in Starlette's threadpool, so the blocking network
    call inside `run_ingestion_tick` does not stall the event loop or any
    concurrent `/predict` call.

    `end`/`window_days` exist for the same operational reason `/ct/evaluate`
    takes `as_of`: forcing a specific range for backfill or a demo, rather
    than always defaulting to the last closed session. `end` is inclusive
    and capped at that session (`run_ingestion_tick`).
    """
    ingestion.run(repo, window_days=window_days, end=end, source="manual")
    return ct_status()


@app.get("/events", response_model=EventsResponse, dependencies=[Depends(_verify_token)])
def events(
    limit: int = Query(default=50, ge=1, le=200),
    before: int | None = Query(default=None, ge=1),
) -> EventsResponse:
    """FR-20 as a history: what the pipeline did, newest first. `before` is
    the oldest `event_id` already shown, to page further back."""
    items, has_more = repo.list_events(limit=limit, before_id=before)
    return EventsResponse(items=[PipelineEventOut(**i) for i in items], has_more=has_more)


@app.get("/market/candles", response_model=CandlesResponse, dependencies=[Depends(_verify_token)])
def candles(
    ticker: str = Query(...),
    days: int = Query(default=120, ge=5, le=1500),
) -> CandlesResponse:
    """Daily OHLC for the dashboard's candlestick chart, ending at the
    latest ingested date (not `date.today()` — the same anchoring
    reasoning as `/ct/evaluate`'s `as_of`)."""
    if ticker not in DATA.tickers:
        raise HTTPException(status_code=422, detail=f"ticker must be one of {list(DATA.tickers)}")
    ingest = repo.latest_ingest_info()
    end = ingest["date"] if ingest else date.today()
    return CandlesResponse(
        ticker=ticker, candles=repo.load_candles(ticker, end - timedelta(days=days), end)
    )


# ------------------------------------------------------- operator override
def _controls_out() -> list[OperatorControlOut]:
    return [OperatorControlOut(**c, label=CONTROL_LABELS[c["name"]]) for c in service.controls.all()]


def _actor(request: Request) -> str:
    """Who an override is recorded against (FR-23). A signed-in person's
    email when sign-in is on; otherwise the credential used, since without
    sign-in there is no person to name."""
    if AUTH.enabled and (user := auth.current_user(request)) is not None:
        return user["email"]
    token = SERVING.bearer_token
    if token is not None and request.headers.get("authorization") == f"Bearer {token}":
        return "API token"
    return "local operator"


def _oversight() -> OversightResponse:
    versions = [v for v in repo.list_model_versions(limit=20) if v["promoted_at"] is not None]
    return OversightResponse(
        controls=_controls_out(),
        versions=[
            RollbackCandidate(
                version_id=v["version_id"],
                label=_active_model_label(v),
                promoted_at=v["promoted_at"],
                is_active=v["is_active"],
                sharpe_ratio=v["sharpe_ratio"],
                trigger_reason=v["trigger_reason"],
            )
            for v in versions
        ],
        pipeline_status=orchestrator.status.value,
    )


@app.get("/oversight", response_model=OversightResponse, dependencies=[Depends(_verify_token)])
def oversight() -> OversightResponse:
    """FR-23: the operator's switches, and the models a rollback can restore."""
    return _oversight()


@app.put(
    "/oversight/controls/{name}", response_model=OversightResponse, dependencies=[Depends(_verify_token)]
)
def set_control(name: OperatorControlName, body: SetControlRequest, request: Request) -> OversightResponse:
    """FR-23: switch one control on or off. Takes effect on the next request
    or tick: `/predict` reads the halt on every call, the CT loop and paper
    trader before they act. Recorded with who, when and why, both as the
    control's state and as an `operator_override` event on the timeline."""
    before = service.controls.get(name)
    if before["enabled"] == body.enabled:
        # Nothing to switch. Writing anyway would replace who set it and why
        # with no event to keep the original.
        return _oversight()
    actor = _actor(request)
    service.controls.set(name, body.enabled, body.reason, actor)
    # The operator's email goes in `detail`, never in the summary: the
    # assistant sends summaries to its language model (DR-10).
    summary = f"{CONTROL_LABELS[name]} {'switched on' if body.enabled else 'lifted'}"
    if body.reason:
        summary += f": {body.reason}"
    emit(repo, "operator_override", summary, source="manual",
         detail={"control": name, "enabled": body.enabled, "reason": body.reason, "by": actor})
    return _oversight()


@app.post("/oversight/rollback", response_model=OversightResponse, dependencies=[Depends(_verify_token)])
def rollback(body: RollbackRequest, request: Request) -> OversightResponse:
    """FR-23: restore a model that has served before, hot-swapped like a
    promotion (FR-16). 409 while the CT loop is evaluating or retraining, or
    for a version that has never served or is already serving."""
    previous = service.active_version_id
    try:
        active = orchestrator.rollback(body.version_id)
    except RollbackRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    actor = _actor(request)
    label = _active_model_label(repo.get_active_version()) or (active or "")[:8]
    emit(
        repo, "operator_override",
        f"Rolled back to {label}: {body.reason}",
        source="manual",
        detail={"control": "rollback", "from_version": previous, "to_version": active,
                "reason": body.reason, "by": actor},
        version_id=active,
    )
    return _oversight()


# ----------------------------------------------------------------- news
news_feed = NewsFeed()


@app.get("/news", response_model=NewsResponse, dependencies=[Depends(_verify_token)])
def news(
    ticker: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=20, ge=1, le=50),
) -> NewsResponse:
    """Recent headlines for one ticker, or the whole universe. Context for
    the reader only (src/dataops/news.py): no decision reads news."""
    if ticker is not None and ticker not in DATA.tickers:
        raise HTTPException(status_code=422, detail=f"ticker must be one of {list(DATA.tickers)}")
    if not NEWS.enabled:
        return NewsResponse(enabled=False, ticker=ticker, items=[])
    try:
        items = news_feed.headlines(ticker, days=days, limit=limit)
    except NewsUnavailable as exc:
        raise HTTPException(status_code=503, detail="News is unavailable right now.") from exc
    return NewsResponse(enabled=True, ticker=ticker, items=items)


PAPER_HISTORY_DAYS = 365


@app.get("/paper/status", response_model=PaperStatusResponse, dependencies=[Depends(_verify_token)])
def paper_status() -> PaperStatusResponse:
    """The forward paper account as last synced — read from the database,
    never from Alpaca directly, so the dashboard's polling never waits on
    a third-party API."""
    # One row per day, so the whole forward equity curve is small enough
    # to return in full; only fills are windowed.
    history = repo.list_paper_account()
    since = date.today() - timedelta(days=PAPER_HISTORY_DAYS)
    latest = history[-1] if history else None
    cycle = repo.latest_paper_cycle()
    return PaperStatusResponse(
        enabled=PAPER.enabled,
        status=paper.status.value,
        next_run_at=next_run_at(datetime.now(timezone.utc), PAPER.run_time_utc) if PAPER.enabled else None,
        last_run_at=paper.last_run_at,
        last_outcome=paper.last_outcome,
        last_signal_date=cycle["signal_date"] if cycle else None,
        last_cycle_failsafe=cycle["failsafe_triggered"] if cycle else None,
        equity=latest["equity"] if latest else None,
        cash=latest["cash"] if latest else None,
        synced_at=latest["synced_at"] if latest else None,
        positions=latest["positions"] if latest else [],
        equity_curve=[{"date": h["date"], "equity": h["equity"]} for h in history],
        fills=repo.list_paper_fills(since),
    )


def _require_paper_enabled() -> None:
    if not PAPER.enabled:
        raise HTTPException(status_code=409, detail="paper trading is disabled (PAPER_TRADING_ENABLED)")


@app.post("/paper/run", response_model=PaperStatusResponse, dependencies=[Depends(_verify_token)])
def trigger_paper_cycle() -> PaperStatusResponse:
    """On-demand counterpart to the daily paper scheduler. Still bound by
    the same guards — it only trades if today's bar exists and today has
    not already traded — so pressing it twice cannot double-trade."""
    _require_paper_enabled()
    try:
        _paper_cycle_job(source="manual")
    except BrokerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NoActiveModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return paper_status()


@app.post("/paper/sync", response_model=PaperStatusResponse, dependencies=[Depends(_verify_token)])
def trigger_paper_sync() -> PaperStatusResponse:
    _require_paper_enabled()
    try:
        _paper_sync_job()
    except BrokerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return paper_status()


# ----------------------------------------------------------------- simulation
def _simulation_response(state: dict) -> SimulationResponse:
    return SimulationResponse(
        enabled=True,
        scenarios=[SimulationScenario(id=s.id, title=s.title, description=s.description) for s in SCENARIOS.values()],
        state=state,
    )


def _simulate(action) -> SimulationResponse:
    if not SIMULATION.enabled:
        raise HTTPException(status_code=409, detail="the simulation is disabled (SIMULATION_ENABLED)")
    try:
        return _simulation_response(action())
    except SimulationRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/simulation", response_model=SimulationResponse, dependencies=[Depends(_verify_token)])
def simulation_state(watch: bool = Query(default=False)) -> SimulationResponse:
    """The sandbox replay of the pipeline (src/simulation). A demonstration:
    it runs in its own process and store, and nothing it does is recorded
    here. `watch` keeps a session alive while its page is open."""
    if not SIMULATION.enabled:
        return SimulationResponse(enabled=False)
    return _simulation_response(simulation.state(watching=watch))


@app.post("/simulation/start", response_model=SimulationResponse, dependencies=[Depends(_verify_token)])
def simulation_start(body: SimulationStartRequest) -> SimulationResponse:
    return _simulate(lambda: simulation.start(body.scenario))


@app.post("/simulation/command", response_model=SimulationResponse, dependencies=[Depends(_verify_token)])
def simulation_command(body: SimulationCommand) -> SimulationResponse:
    return _simulate(lambda: simulation.command(body.model_dump()))


@app.post("/simulation/stop", response_model=SimulationResponse, dependencies=[Depends(_verify_token)])
def simulation_stop() -> SimulationResponse:
    return _simulate(simulation.stop)


# ----------------------------------------------------------------- assistant
_assistant_limiter = assistant.RateLimiter()


@app.get("/assistant/status", response_model=AssistantStatusResponse, dependencies=[Depends(_verify_token)])
def assistant_status() -> AssistantStatusResponse:
    """Whether the dashboard should offer the assistant (src/serving/assistant.py)."""
    return AssistantStatusResponse(enabled=ASSISTANT.enabled, model=ASSISTANT.model if ASSISTANT.enabled else None)


@app.post("/assistant/chat", dependencies=[Depends(_verify_token)])
async def assistant_chat(body: AssistantChatRequest, request: Request) -> StreamingResponse:
    """Streams one answer as server-sent events (see assistant.run_chat).
    Behind the same sign-in as the rest of the API, and rate-limited per
    signed-in user (or client address) because every question costs money."""
    if not ASSISTANT.enabled:
        raise HTTPException(status_code=503, detail="The assistant is not configured on this server.")
    user = auth.current_user(request) if AUTH.enabled else None
    caller = user["email"] if user else (request.client.host if request.client else "unknown")
    if not _assistant_limiter.allow(caller):
        raise HTTPException(status_code=429, detail="Too many questions. Try again in a few minutes.")

    tools = assistant.PipelineTools(repo, lambda: ct_status().model_dump(), news=news_feed if NEWS.enabled else None)
    messages = [m.model_dump() for m in body.messages]

    async def events():
        try:
            async for event in assistant.run_chat(messages, tools):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception:  # noqa: BLE001 - end the stream with a message, never a half-open connection
            logging.getLogger(__name__).exception("assistant chat failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong. Please ask again.'})}\n\n"

    # X-Accel-Buffering: nginx (the deployment's proxy) would otherwise hold
    # the stream back until the answer finished.
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
