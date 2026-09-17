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
| Chapter 4 | System Implementation | 🔄 In Progress (Sprints 1–3 Complete) |
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
│  │ yfinance │   │TradingEnv│   │ Inference│   │  Portfolio  │  │
│  │PostgreSQL│   │PPO(SB3)  │   │ Gateway  │   │  Telemetry  │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘  │
│       ▲                                               │          │
│       └───────── CT Orchestrator (Sharpe Monitor) ◀──┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Layers

| Layer | Responsibility | Technology | Package | Status |
|---|---|---|---|---|
| **DataOps** | Ingest, clean, enrich and persist market data | `yfinance` + PostgreSQL + SQLAlchemy | `src/dataops` | ✅ Sprint 1 Complete |
| **RLOps** | MDP environment, baseline policies, PPO agent, registry | Gymnasium + Stable Baselines3 + MLflow | `src/rlops` | ✅ Sprints 2 & 3 Complete |
| **Orchestration** | Performance metrics, drift detection, CT cycle | Python (custom) | `src/orchestration` | ✅ Sprint 2 (metrics); ⏳ Sprint 4 (CT loop) |
| **Serving** | Inference API, volatility fail-safe | FastAPI (ASGI) | `src/serving` | ⏳ Sprint 4 |
| **Presentation** | Real-time telemetry dashboard | React.js SPA | `frontend/` | ⏳ Sprint 4 |

Dependencies flow one way only — orchestration toward data and model concerns — with no cycles (NFR-06).

---

## 🤖 The RL Agent & Universe

- **Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines3, wrapped by
  `PPOAgent` (`src/rlops/agent.py`), on `device="cpu"` — benchmarked ~12–13× faster than
  `"mps"` on the actual training hardware (`CLAUDE.md` §9), not a default left open for tuning
- **Environment:** `TradingEnvironment` (`src/rlops/environment.py`), a Gymnasium-compatible
  MDP implemented directly to spec. FinRL's `StockTradingEnv` was evaluated and does import
  successfully, but its reward has no override hook and its actions are hmax-scaled share
  counts rather than continuous weights — see the recorded deviation in `CLAUDE.md` §8, and the
  cross-check against it in `scripts/finrl_crosscheck.py` (max abs diff $0.00007 on a ~$100K
  portfolio, correlation 1.0)
- **Action Space:** Continuous target weight per ticker, in [-1, 1]
- **State Space:** 8 backward-looking z-scored features per ticker (OHLCV + SMA_20 + RSI_14
  + VIX) plus the agent's own current position weights and cash weight — 46 dimensions for
  the 5-ticker universe
- **Reward Function:** Step return minus the *increment* in maximum drawdown, never its level
- **Baselines:** buy-and-hold, equal-weight-rebalanced, all-cash, random (`src/rlops/baselines.py`)
- **Metrics:** Sharpe ratio, rolling Sharpe, max drawdown, cumulative return, deflated Sharpe
  ratio (`src/orchestration/evaluator.py`)
- **Registry:** `ModelRegistry` (`src/rlops/registry.py`) — MLflow run logging (FR-08) and
  `model_version` registration (FR-09)
- **Validation:** Walk-forward cross-validation across rolling regimes
  (`processing.walk_forward_splits`), each with a 5-seed sweep, not a single train/eval split
  or a single seed

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
│   ├── run_ingestion.py      # CLI runner for data ingestion pipeline
│   ├── run_baselines.py      # CLI runner for the baseline policies (Sprint 2)
│   ├── benchmark_device.py   # CPU vs MPS PPO throughput benchmark (Sprint 3)
│   ├── finrl_crosscheck.py   # §8 validation experiment against FinRL (Sprint 3)
│   └── train_agent.py        # Walk-forward PPO training + seed sweep (Sprint 3)
│
├── src/
│   ├── config.py             # Database, risk and environment settings
│   ├── dataops/              # Sprint 1 — ETL pipeline (yfinance → PostgreSQL)
│   │   ├── ingestion.py
│   │   ├── processing.py     # + walk_forward_splits (Sprint 3)
│   │   ├── models.py
│   │   └── repository.py
│   ├── rlops/                # Sprint 2 — MDP environment + baselines
│   │   ├── environment.py
│   │   ├── baselines.py
│   │   ├── agent.py          # PPOAgent wrapper over SB3 (Sprint 3)
│   │   └── registry.py       # MLflow logging + model_version registration (Sprint 3)
│   ├── orchestration/        # Sprint 2 — metrics; Sprint 4 — CT loop & drift detection
│   │   └── evaluator.py
│   └── serving/              # Sprint 4 — FastAPI inference + CT orchestrator
│
└── tests/                    # Unit + integration tests
    ├── test_processing.py
    ├── test_repository.py
    ├── test_environment.py
    ├── test_baselines.py
    ├── test_evaluator.py
    ├── test_agent.py
    └── test_registry.py
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

Benchmarked on an M5 (`scripts/benchmark_device.py`, CLAUDE.md §9): CPU beats MPS by ~12–13× for this policy network (14.7K parameters — small enough that CPU↔GPU transfer overhead dominates any arithmetic MPS would accelerate). Use `device="cpu"` on the PPO model in Sprint 3, not `device="mps"`.

---

## 💻 Usage

```bash
# Fetch, enrich and persist the configured universe
python scripts/run_ingestion.py --start 2015-01-01 --end 2025-12-31

# Transform only, without writing to the database
python scripts/run_ingestion.py --start 2024-01-01 --end 2024-03-01 --dry-run

# A single ticker
python scripts/run_ingestion.py --tickers XOM --start 2024-01-01 --end 2024-06-01

# Run the baseline policies over the evaluation partition (Sprint 2)
python scripts/run_baselines.py
python scripts/run_baselines.py --partition train --tickers XOM CVX

# Walk-forward PPO training with a seed sweep, logged to MLflow (Sprint 3)
python scripts/train_agent.py
python scripts/train_agent.py --timesteps 50000 --seeds 5
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
| FR-06 | `rlops.environment.TradingEnvironment` | `test_conforms_to_gymnasium_api`, `test_observation_and_action_space_shapes` |
| FR-07 | `TradingEnvironment._reward` (drawdown increment, I4) | `test_drawdown_penalty_applies_to_increment_not_level` |
| FR-13 | `orchestration.evaluator.rolling_sharpe` | `test_rolling_sharpe_warmup_rows_are_nan_not_zero` |
| I1 | `TradingEnvironment._prepare` (backward-looking alignment) | `test_observation_contains_no_future_information` |
| I2 | `TradingEnvironment.step` (t → t+1 realisation) | `test_return_is_realised_from_t_to_t_plus_one` |
| FR-07 | `PPOAgent.train` (reward from `TradingEnvironment`) | `test_agent_trains_without_error` |
| FR-08 | `ModelRegistry.log_run` | `test_log_run_records_hyperparameters_in_mlflow` |
| FR-09 | `ModelRegistry.register_version` | `test_register_version_links_to_its_run` |
| DR-08 | `ModelRegistry.log_run` → `model_run` partition columns | `test_log_run_persists_exact_partition_boundaries` |

---

## 🗓️ Development Sprints & Roadmap

- [x] **Sprint 1 — DataOps Foundation (Complete).** Ingestion, enrichment, persistence, schema, tests.
- [x] **Sprint 2 — Trading Environment (Complete).** Gymnasium MDP, continuous action space, drawdown-incremented reward, baseline policies, evaluation metrics.
- [x] **Sprint 3 — Training & Model Registry (Complete).** PPO on CPU (benchmarked ~12–13× faster than MPS), walk-forward cross-validation with a seed sweep, MLflow logging, `model_version` registration.
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
