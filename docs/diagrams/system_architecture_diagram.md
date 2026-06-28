# Activity Diagram

Illustrates the system's internal control flow and decision-making logic for a single localised trading cycle, including the VIX-based emergency capital-preservation fail-safe.

## Trading Cycle Activity Flow

```mermaid
flowchart TD
    A([Start: New Market Day]) --> B[Fetch latest OHLCV\nfrom PostgreSQL]
    B --> C[Enrich state observation\nSMA_20 + RSI_14 + VIX]
    C --> D{VIX > emergency\nvolatility threshold?}

    D -->|YES — extreme macro shock| E[🚨 Emergency Capital Preservation\nForce HOLD — bypass trade execution]
    E --> F[Log: VIX override triggered]
    F --> Z

    D -->|NO — normal conditions| G[Send state to PPO Agent\npredict_action]
    G --> H[Agent outputs portfolio\nweight vector]
    H --> I{Sharpe Ratio\ncheck}

    I -->|Sharpe < threshold| J[Flag: Alpha Decay Suspected]
    J --> K[Notify CT Orchestrator\ntrigger async retraining]
    K --> L[Execute current action\nwith existing weights]
    L --> M[Rebalance portfolio]

    I -->|Sharpe ≥ threshold| M

    M --> N[Record trade action\nBuy / Hold / Sell + weights]
    N --> O[Compute post-trade metrics\nSharpe · MDD · Cumulative Return]
    O --> P[Stream telemetry\nto FastAPI + Dashboard]
    P --> Q{End of trading\nperiod?}

    Q -->|NO| B
    Q -->|YES| Z([End Cycle])

    style E fill:#fce8e8,stroke:#cc0000,color:#7a0000
    style J fill:#fff4e0,stroke:#cc8800,color:#7a4400
    style K fill:#fff4e0,stroke:#cc8800,color:#7a4400
```

## Decision Logic Notes

| Decision Gate | Condition | Consequence |
|---------------|-----------|-------------|
| VIX threshold breach | VIX index exceeds predefined limit (e.g. > 35) | Trade bypassed; agent forced to HOLD; capital preserved |
| Sharpe Ratio check | Rolling Sharpe drops below `target_sharpe_threshold` | Drift flagged; CT Orchestrator notified; async retraining initiated |
| End of period | Daily cycle complete | Metrics logged; system loops to next trading day |

---

# System Architecture Diagram

Provides a layered view of the complete platform — illustrating how data flows across the four infrastructure layers and how the closed-loop CT feedback architecture operates.

## Architecture Layers

```mermaid
graph TB
    subgraph External["External Data Sources"]
        YF[yfinance API\nGlobal Energy OHLCV]
    end

    subgraph DataOps["Layer 1 — DataOps"]
        ING[Data Ingestor\nETL Pipeline]
        PG[(PostgreSQL\nTime-Series Store\nACID Compliant)]
        ING --> PG
    end

    subgraph RLOps["Layer 2 — RLOps Training"]
        FINRL[FinRL Environment\nOpenAI Gym MDP]
        PPO[PPO Agent\nStable Baselines3\nActor-Critic Network]
        FINRL --> PPO
    end

    subgraph CT["CT Orchestrator"]
        ORCH[Continuous Training Monitor\nSharpe Ratio Watchdog\nAsync Retraining Trigger]
    end

    subgraph Serving["Layer 3 — Serving"]
        FAST[FastAPI Backend\nASGI · Async\nPydantic Validation]
    end

    subgraph Presentation["Layer 4 — Presentation"]
        REACT[React.js Dashboard\nSPA · Virtual DOM\nReal-time Telemetry]
    end

    YF -->|Raw OHLCV| ING
    PG -->|Training batches| FINRL
    PPO -->|Model weights| FAST
    FAST -->|Inference + metrics| REACT
    REACT -->|Performance telemetry| ORCH
    ORCH -->|Drift detected — fetch fresh data| ING
    ORCH -->|Retrain trigger| FINRL
    ORCH -->|Hot-swap model| FAST

    style External fill:#f0f4ff,stroke:#3366cc
    style DataOps fill:#f0fff4,stroke:#33cc66
    style RLOps fill:#fff8f0,stroke:#cc8833
    style CT fill:#fff0f0,stroke:#cc3333
    style Serving fill:#f8f0ff,stroke:#9933cc
    style Presentation fill:#f0f8ff,stroke:#3399cc
```

## Data Flow Summary

```
yfinance API
    → DataIngestor (ETL: clean, normalise, Z-score)
        → PostgreSQL (ACID-compliant time-series storage)
            → FinRL MDP Environment (state/action/reward definition)
                → PPO Agent (policy gradient training)
                    → FastAPI Serving Layer (stateless async inference)
                        → React.js Dashboard (real-time telemetry)
                            → CT Orchestrator (Sharpe monitoring)
                                ↺ loops back to DataIngestor on drift detection
```

## Infrastructure Separation of Concerns

| Module | Namespace | Responsibility |
|--------|-----------|---------------|
| `dataops/` | DataOps | ETL pipelines, PostgreSQL bridging, data normalisation |
| `rlops/` | RLOps | FinRL environment, Stable Baselines3 PPO, reward engineering |
| `serving/` | Serving | FastAPI gateway, Pydantic validation, CT orchestration |
| `dashboard/` | Presentation | React.js components, telemetry visualisation |
