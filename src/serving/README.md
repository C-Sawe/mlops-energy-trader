# Serving Module

**Sprint 4 — Status: ⏳ Awaiting Sprint 3**

This module contains the FastAPI inference gateway and CT Orchestrator:

- Stateless ASGI FastAPI backend — non-blocking async inference
- Pydantic data validation — strict tensor format enforcement
- CT Orchestrator — Sharpe Ratio watchdog, drift detection, async retraining trigger
- Hot-swap model redeployment without server downtime

## Planned Files

```
serving/
├── main.py              # FastAPI app — routes: /predict, /metrics, /health, /telemetry
├── orchestrator.py      # MLOpsOrchestrator class — drift detection + CT loop
├── schemas.py           # Pydantic request/response models
├── model_manager.py     # Model artifact loading, versioning, hot-swap logic
└── config.py            # Sharpe threshold, retraining cooldown, server config
```

## Implementation begins: Sprint 4 (November 2026)
