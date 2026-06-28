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
| Chapter 4 | System Implementation | 🔄 In Progress |
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

| Layer | Technology | Function |
|-------|-----------|---------|
| **DataOps** | `yfinance` + PostgreSQL | Programmatic ingestion, rolling Z-score normalization, ACID-compliant time-series storage |
| **RLOps / Intelligence** | FinRL + Stable Baselines3 | MDP environment, PPO agent training, policy updates |
| **Serving** | FastAPI (ASGI) | Stateless async inference gateway; non-blocking background retraining |
| **Presentation** | React.js SPA | Real-time portfolio telemetry, Sharpe Ratio monitoring, trading activity logs |
| **CT Orchestrator** | Python (custom) | Monitors live Sharpe Ratio; triggers background retrain + redeploy on drift detection |

---

## 🤖 The RL Agent

- **Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines3
- **Environment:** FinRL (OpenAI Gym–compatible MDP over energy equities)
- **Action Space:** Continuous — Buy / Hold / Sell portfolio weights
- **State Space:** OHLCV + SMA_20 + RSI_14 + VIX (enriched observation)
- **Reward Function:** Risk-averse; penalizes Maximum Drawdown (MDD) heavily
- **Validation:** Walk-forward time-series cross-validation (no look-ahead bias)

### Target Equities

| Ticker | Company | Exchange |
|--------|---------|---------|
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
├── src/
│   ├── dataops/          # Sprint 1 — ETL pipeline (yfinance → PostgreSQL)
│   ├── rlops/            # Sprint 2 & 3 — FinRL env + PPO agent training
│   ├── serving/          # Sprint 4 — FastAPI inference + CT orchestrator
│   └── dashboard/        # Sprint 4 — React.js visualization frontend
│
└── tests/                # Unit + integration tests (walk-forward backtesting)
```

---

## 🗓️ Development Sprints (Agile)

| Sprint | Focus | Target |
|--------|-------|--------|
| Sprint 1 | DataOps Foundation — yfinance scrapers + PostgreSQL schema | Aug 2026 |
| Sprint 2 | FinRL Environment Configuration — MDP state/action space | Sep 2026 |
| Sprint 3 | PPO Agent Training + Risk-Averse Reward Engineering | Oct 2026 |
| Sprint 4 | FastAPI Deployment + React Dashboard + CT Loop | Nov 2026 |

---

## 📐 UML Diagrams

All system analysis and design diagrams are documented in [`docs/diagrams/`](./docs/diagrams/).

- [Use Case Diagram](./docs/diagrams/use_case_diagram.md)
- [Class Diagram](./docs/diagrams/class_diagram.md)
- [Sequence Diagram](./docs/diagrams/sequence_diagram.md)
- [Activity Diagram](./docs/diagrams/activity_diagram.md)
- [System Architecture Diagram](./docs/diagrams/system_architecture_diagram.md)

---

## 🔑 Key Concepts

**Alpha Decay** — The erosion of a trading algorithm's excess returns as markets adapt to its signals (Meng & Chen, 2026).

**Deployment Chasm** — The gap between a well-trained academic ML prototype and a production-grade, continuously adapting live system (Xia et al., 2026).

**Continuous Training (CT)** — An automated MLOps pattern where model drift triggers background retraining and redeployment without human intervention.

**Walk-Forward Validation** — Time-series cross-validation using chronological rolling windows to prevent look-ahead bias; replaces standard k-fold CV.

---

## 📄 Academic Context

- **Institution:** Strathmore University, Nairobi, Kenya
- **Programme:** BSc Informatics and Computer Science
- **School:** School of Computing and Engineering Sciences
- **Supervisor:** Mr. Allan Vikiru
- **Student:** Caleb Kipchirchir (Admission No. 169391)
- **Contact:** caleb.Kipchirchir@strathmore.edu
- **Turnitin Similarity:** 19% (bibliography and quoted text excluded)

---

## 📚 Key References

- Espiga-Fernández et al. (2024). *A Systematic Approach to Portfolio Optimization.* Algorithms, 17(12).
- Li et al. (2021). *FinRL-Podracer: High Performance and Scalable DRL for Quantitative Finance.* ACM.
- Kreuzberger et al. (2023). *MLOps: Overview, Definition, and Architecture.* IEEE Access.
- Meng & Chen (2026). *AI-Driven Alpha Decay.* arXiv:2605.23905.
- Xia et al. (2026). *Agentic Trading: When LLM Agents Meet Financial Markets.* arXiv:2605.19337.

---

*This project is an academic research deliverable submitted in partial fulfilment of the requirements for the award of a Bachelor of Science in Informatics and Computer Science at Strathmore University.*
