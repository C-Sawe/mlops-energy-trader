# Sequence Diagram

Illustrates the chronological flow of messages between system objects during a full **Continuous Training (CT) cycle** — from alpha decay detection through to live model redeployment.

## Full CT Loop Sequence

```mermaid
sequenceDiagram
    participant Dashboard as React Dashboard
    participant FastAPI as FastAPI Server
    participant Orchestrator as CT Orchestrator
    participant Agent as PPO Agent
    participant Env as FinRL Environment
    participant DB as PostgreSQL Store
    participant Ingestor as Data Ingestor
    participant API as yfinance API

    Note over Dashboard,API: PHASE 1 — LIVE INFERENCE (steady state)

    Dashboard->>FastAPI: GET /telemetry (polling, every N ms)
    FastAPI->>Agent: predict_action(current_market_state)
    Agent-->>FastAPI: action [buy_weight, hold_weight, sell_weight]
    FastAPI-->>Dashboard: {action, sharpe_ratio, portfolio_value, mdd}
    Dashboard->>Dashboard: render_equity_curve(), render_sharpe_gauge()

    Note over Dashboard,API: PHASE 2 — DRIFT DETECTION

    FastAPI->>Orchestrator: stream_telemetry(sharpe_series)
    Orchestrator->>Orchestrator: evaluate_model_drift(sharpe_series)
    Orchestrator->>Orchestrator: sharpe < target_sharpe_threshold?

    alt Alpha Decay Detected
        Orchestrator->>Orchestrator: log_drift_event(timestamp, sharpe)
        Orchestrator->>FastAPI: notify: retraining_in_progress = true
        FastAPI-->>Dashboard: CT_status: "Retraining triggered"

        Note over Dashboard,API: PHASE 3 — BACKGROUND RETRAINING (async, non-blocking)

        Orchestrator->>Ingestor: fetch fresh OHLCV data
        Ingestor->>API: GET /download (tickers, start_date, end_date)
        API-->>Ingestor: raw OHLCV DataFrame
        Ingestor->>Ingestor: forward_fill_missing()
        Ingestor->>Ingestor: rolling_zscore_normalise()
        Ingestor->>DB: persist_to_db(clean_df)
        DB-->>Ingestor: write confirmed

        Orchestrator->>DB: fetch_batch(tickers, training_window)
        DB-->>Orchestrator: enriched training DataFrame

        Orchestrator->>Env: build_mdp(training_data)
        Env-->>Orchestrator: gym_env ready

        Orchestrator->>Agent: train(gym_env, total_timesteps)
        Agent->>Agent: run PPO policy gradient updates
        Agent-->>Orchestrator: training complete, new weights path

        Note over Dashboard,API: PHASE 4 — HOT REDEPLOY

        Orchestrator->>FastAPI: deploy_new_weights(model_path)
        FastAPI->>Agent: load_weights(model_path)
        Agent-->>FastAPI: new policy loaded ✓
        FastAPI-->>Dashboard: CT_status: "Model updated — live"

    else No Drift — Continue Normal Inference
        Orchestrator->>Orchestrator: continue monitoring
    end

    Note over Dashboard,API: SYSTEM RETURNS TO STEADY STATE (Phase 1)
```

## Key Design Decisions Illustrated

**Non-blocking async retraining:** The FastAPI ASGI server continues serving inference requests (`predict_action`) while the CT Orchestrator runs the retraining cycle asynchronously in the background — a critical architectural requirement for system availability.

**Hot-swap deployment:** `deploy_new_weights()` atomically replaces the live model artifact without server restart or downtime, maintaining uninterrupted trading execution.

**Sharpe Ratio as the drift signal:** The orchestrator uses a rolling Sharpe Ratio time-series — not raw returns — as the primary drift detector, providing a risk-adjusted, noise-resistant signal.
