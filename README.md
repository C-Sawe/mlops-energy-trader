# ⚡ MLOps Continuous Training Pipeline — Energy Markets RL Agent

> **Final Year Undergraduate Thesis Project**  
> Caleb Kipchirchir · 169391 · ICS 4C  
> Strathmore University, School of Computing and Engineering Sciences · Nairobi, Kenya  
> Supervisor: Mr. Allan Vikiru  

---

## 📌 Project Status

| Chapter | Title | Status |
|---------|-------|--------|
| Chapter 1 | Introduction & Problem Statement | ✅ Complete |
| Chapter 2 | Literature Review | ✅ Complete |
| Chapter 3 | Methodology & System Design | ✅ Complete |
| Chapter 4 | System Implementation | 🔄 In Progress (Sprint 1 Complete) |
| Chapter 5 | Results, Testing & Evaluation | ⏳ Awaiting Chapter 4 |

> **Proposal Defence:** Completed — June 2026  
> **Final Submission Target:** January 2027  

---

## 🧠 What This Project Solves

Current academic research in algorithmic trading trains models in sterile, static environments. When deployed into live markets, these models suffer from **alpha decay** — their logic degrades as the market evolves. No automated infrastructure exists to detect this and fix it.

This project engineers an end-to-end **MLOps pipeline** that wraps a Reinforcement Learning trading agent in a **self-healing Continuous Training (CT) loop** — autonomously monitoring performance, detecting model drift, and retraining without human intervention.

The core academic contribution is the **Deployment Chasm** framing: the gap between academic ML prototypes and production-grade, continuously adapting systems.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-HEALING FEEDBACK LOOP                    │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ DataOps  │──▶│  RLOps   │──▶│ FastAPI  │──▶│  React.js   │  │
│  │ Layer    │   │ Training │   │ Serving  │   │  Dashboard  │  │
│  │          │   │  Layer   │   │  Layer   │   │             │  │
│  │ yfinance │   │  FinRL   │   │ Inference│   │  Portfolio  │  │
│  │PostgreSQL│   │   PPO    │   │ Gateway  │   │  Telemetry  │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘  │
│       ▲                                               │          │
│       └───────── CT Orchestrator (Sharpe Monitor) ◀──┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Layers

| Layer | Responsibility | Technology | Package | Status |
|---|---|---|---|---|
| **DataOps** | Ingest, clean, enrich and persist market data | `yfinance` + PostgreSQL + SQLAlchemy | `src/dataops` | ✅ Sprint 1 Complete |
| **RLOps** | MDP environment, PPO agent, model registry | FinRL + Stable Baselines3 + MLflow | `src/rlops` | 🔄 Sprint 2 & 3 |
| **Orchestration** | Drift detection, continuous training cycle | Python (custom) | `src/orchestration` | ⏳ Sprint 4 |
| **Serving** | Inference API, volatility fail-safe | FastAPI (ASGI) | `src/serving` | ⏳ Sprint 4 |
| **Presentation** | Real-time telemetry dashboard | React.js SPA | `frontend/` | ⏳ Sprint 4 |

Dependencies flow one way only — orchestration toward data and model concerns — with no cycles (NFR-06).

---

## 🤖 The RL Agent & Universe

- **Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines3
- **Environment:** FinRL (OpenAI Gym–compatible MDP over energy equities)
- **Action Space:** Continuous — Buy / Hold / Sell portfolio weights
- **State Space:** OHLCV + SMA_20 + RSI_14 + VIX (enriched observation)
- **Reward Function:** Risk-averse; penalizes Maximum Drawdown (MDD) heavily
- **Validation:** Walk-forward time-series cross-validation (no look-ahead bias)

### Target Equities

| Ticker | Company | Exchange |
|---|---|---|
| XOM | ExxonMobil | NYSE |
| CVX | Chevron | NYSE |
| SHEL | Shell | NYSE |
| BP | BP | NYSE |
| NEE | NextEra Energy | NYSE |

---

## 📁 Repository Structure

```
mlops-energy-trader/
│
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── docs/
│   ├── proposal/
│   │   └── Kipchirchir_169391_Proposal_Vikiru.pdf   # Approved proposal (June 2026)
│   └── diagrams/
│       ├── use_case_diagram.md
│       ├── class_diagram.md
│       ├── sequence_diagram.md
│       ├── activity_diagram.md
│       └── system_architecture_diagram.md
│
├── scripts/
│   └── run_ingestion.py      # CLI runner for data ingestion pipeline
│
├── src/
│   ├── config.py             # Database and project settings
│   ├── dataops/              # Sprint 1 — ETL pipeline (yfinance → PostgreSQL)
│   │   ├── ingestion.py
│   │   ├── processing.py
│   │   ├── models.py
│   │   └── repository.py
│   ├── rlops/                # Sprint 2 & 3 — FinRL env + PPO agent training
│   ├── orchestration/        # Continuous Training loop & drift detection
│   └── serving/              # Sprint 4 — FastAPI inference + CT orchestrator
│
└── tests/                    # Unit + integration tests
    ├── test_processing.py
    └── test_repository.py
```

---

## 🚀 Setup & Installation

Requires Python 3.11+ and Docker Desktop.

```bash
git clone https://github.com/C-Sawe/mlops-energy-trader.git
cd mlops-energy-trader

python3 -m venv mlops-env
source mlops-env/bin/activate          # Windows: mlops-env\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                   # edit credentials if needed
docker compose up -d postgres          # spin up PostgreSQL container
```

On Apple Silicon, install the MPS-enabled PyTorch build when running Sprint 3; set `device="mps"` on the PPO model.

---

## 💻 Usage

```bash
# Fetch, enrich and persist the configured universe
python scripts/run_ingestion.py --start 2015-01-01 --end 2025-12-31

# Transform only, without writing to the database
python scripts/run_ingestion.py --start 2024-01-01 --end 2024-03-01 --dry-run

# A single ticker
python scripts/run_ingestion.py --tickers XOM --start 2024-01-01 --end 2024-06-01
```

### Running Tests

```bash
python -m pytest tests/ -v
```

The test suite uses deterministic synthetic data and in-memory SQLite, so it requires neither a network connection nor a running database.

---

## ⚙️ Data Pipeline (Sprint 1 Implementation)

`build_feature_frame()` applies transformations in a strict, fixed order to prevent look-ahead bias:

1. **Corporate action adjustment** (DR-03) — before indicators, because an SMA computed across an unadjusted split is meaningless.
2. **Imputation** (DR-04) — forward-fill only, every filled row flagged `is_imputed`. Leading gaps are dropped rather than back-filled.
3. **Indicators** (FR-03) — SMA_20, and RSI_14 using Wilder's exponential smoothing rather than a simple rolling mean.
4. **VIX join** (FR-03) — market-wide, merged on date and forward-filled.
5. **Rolling Z-score** (FR-04, DR-07) — backward-looking windows only.

### Look-Ahead Leakage Prevention

The most consequential failure mode in financial ML is silent look-ahead leakage:
- `test_zscore_uses_only_backward_looking_window` mutates a future observation and asserts that no past standardized value changes.
- `partition_chronological()` raises rather than warns when the evaluation window does not begin strictly after the training window ends. The `model_run` table enforces this rule as a database `CHECK` constraint (`ck_eval_after_train`).

---

## 📋 Requirement Traceability

| Requirement | Implementation | Test |
|---|---|---|
| FR-01 | `ingestion.fetch_market_data` | *(live, manual)* |
| FR-02 | `processing.impute_missing`, `adjust_corporate_actions` | `test_missing_rows_are_forward_filled_and_flagged` |
| FR-03 | `processing.compute_indicators`, `attach_vix` | `test_sma_matches_manual_calculation`, `test_rsi_bounded_and_correct_on_monotonic_series` |
| FR-04 | `processing.normalize_rolling` | `test_zscore_uses_only_backward_looking_window` |
| FR-05 | `repository.MarketRepository.persist` | `test_persist_and_reload_roundtrip` |
| DR-03 | `processing.adjust_corporate_actions` | `test_corporate_action_adjustment_preserves_ratios` |
| DR-04 | `is_imputed` column | `test_imputation_flag_survives_persistence` |
| DR-05 | composite PK on `market_observation` | `test_reingestion_is_idempotent_not_duplicating` |
| DR-06 | `partition_chronological`, `ck_eval_after_train` | `test_partition_rejects_overlapping_windows`, `test_model_run_rejects_evaluation_overlapping_training` |
| DR-07 | backward-looking `rolling()` | `test_zscore_uses_only_backward_looking_window` |
| IR-01 | `_fetch_one` exponential backoff | *(live, manual)* |
| IR-02 | `strict=True` aborts whole batch | *(live, manual)* |
| IR-03 | `config.DatabaseConfig` | — |
| NFR-07 | `trading_decision → model_version → model_run` | `test_decision_traces_back_to_run_and_partition` |
| NFR-10 | `DatabaseConfig.__repr__` masks password | — |

---

## 🗓️ Development Sprints & Roadmap

- [x] **Sprint 1 — DataOps Foundation (Complete).** Ingestion, enrichment, persistence, schema, tests.
- [ ] **Sprint 2 — FinRL Environment.** State space, continuous action space, drawdown-penalized reward.
- [ ] **Sprint 3 — Training & Model Registry.** PPO on MPS, walk-forward validation, MLflow.
- [ ] **Sprint 4 — Serving & CT Loop.** FastAPI, VIX fail-safe, React dashboard, orchestrator.

---

## 📐 UML Diagrams

All system analysis and design diagrams are documented in [`docs/diagrams/`](./docs/diagrams/):

- [Use Case Diagram](./docs/diagrams/use_case_diagram.md)
- [Class Diagram](./docs/diagrams/class_diagram.md)
- [Sequence Diagram](./docs/diagrams/sequence_diagram.md)
- [Activity Diagram](./docs/diagrams/activity_diagram.md)
- [System Architecture Diagram](./docs/diagrams/system_architecture_diagram.md)

---

## 📄 Academic Context

- **Institution:** Strathmore University, Nairobi, Kenya
- **Programme:** BSc Informatics and Computer Science
- **School:** School of Computing and Engineering Sciences
- **Supervisor:** Mr. Allan Vikiru
- **Student:** Caleb Kipchirchir (Admission No. 169391)
- **Contact:** caleb.Kipchirchir@strathmore.edu
- **Proposal Document:** [`docs/proposal/Kipchirchir_169391_Proposal_Vikiru.pdf`](./docs/proposal/Kipchirchir_169391_Proposal_Vikiru.pdf)

---

## 📚 Key References

- Espiga-Fernández et al. (2024). *A Systematic Approach to Portfolio Optimization.* Algorithms, 17(12).
- Li et al. (2021). *FinRL-Podracer: High Performance and Scalable DRL for Quantitative Finance.* ACM.
- Kreuzberger et al. (2023). *MLOps: Overview, Definition, and Architecture.* IEEE Access.
- Meng & Chen (2026). *AI-Driven Alpha Decay.* arXiv:2605.23905.
- Xia et al. (2026). *Agentic Trading: When LLM Agents Meet Financial Markets.* arXiv:2605.19337.

---

*This project is an academic research deliverable submitted in partial fulfilment of the requirements for the award of a Bachelor of Science in Informatics and Computer Science at Strathmore University.*
