# Class Diagram

Models the static object-oriented backend structure. Implements OOAD principles with strict encapsulation, separation of concerns, and loose coupling across the four system modules.

## Diagram

```mermaid
classDiagram
    class DataIngestor {
        +String api_source
        +List~String~ tickers
        +String db_connection_string
        +Date last_ingestion_date
        +fetch_ohlcv(ticker: String, start: Date, end: Date) DataFrame
        +forward_fill_missing(df: DataFrame) DataFrame
        +adjust_for_splits(df: DataFrame) DataFrame
        +rolling_zscore_normalise(df: DataFrame, window: int) DataFrame
        +persist_to_db(df: DataFrame, table: String) bool
    }

    class PostgreSQLStore {
        +String host
        +String database
        +String schema
        +int port
        +connect() Connection
        +fetch_batch(ticker: String, start: Date, end: Date) DataFrame
        +write_batch(df: DataFrame, table: String) bool
        +get_latest_date(ticker: String) Date
    }

    class FinRLEnvironment {
        +List~String~ tickers
        +int initial_capital
        +float transaction_cost_pct
        +DataFrame market_data
        +reset() ObservationSpace
        +step(action: ActionSpace) Tuple
        +get_state() ObservationSpace
        +calculate_portfolio_value() float
        +build_mdp() GymEnv
    }

    class PPOAgent {
        +String policy_type
        +float learning_rate
        +float gamma
        +int n_steps
        +ndarray policy_weights
        +train(env: GymEnv, total_timesteps: int) Model
        +predict_action(observation: ndarray) ndarray
        +save_weights(path: String) bool
        +load_weights(path: String) bool
        +evaluate(env: GymEnv, n_episodes: int) Dict
    }

    class MLOpsOrchestrator {
        +float target_sharpe_threshold
        +int monitoring_window_days
        +float retraining_cooldown_hours
        +bool retraining_in_progress
        +evaluate_model_drift(sharpe_series: Series) bool
        +compute_sharpe_ratio(returns: Series) float
        +compute_max_drawdown(equity_curve: Series) float
        +trigger_retraining() AsyncTask
        +deploy_new_weights(model_path: String) bool
        +log_drift_event(timestamp: DateTime, sharpe: float) void
    }

    class FastAPIServer {
        +String host
        +int port
        +PPOAgent active_agent
        +predict(market_state: dict) ActionResponse
        +get_portfolio_metrics() MetricsResponse
        +get_system_health() HealthResponse
        +reload_model(model_path: String) bool
        +stream_telemetry() EventStream
    }

    class PortfolioDashboard {
        +String api_base_url
        +int refresh_interval_ms
        +render_equity_curve(data: TimeSeries) Component
        +render_sharpe_gauge(value: float) Component
        +render_trading_log(actions: List) Component
        +render_ct_status(status: dict) Component
        +render_drawdown_chart(data: TimeSeries) Component
    }

    DataIngestor --> PostgreSQLStore : persists to
    PostgreSQLStore --> FinRLEnvironment : supplies training data
    FinRLEnvironment --> PPOAgent : provides MDP env
    PPOAgent --> FastAPIServer : serves inference
    FastAPIServer --> MLOpsOrchestrator : exposes telemetry
    MLOpsOrchestrator --> PPOAgent : triggers retrain
    MLOpsOrchestrator --> DataIngestor : requests fresh data
    FastAPIServer --> PortfolioDashboard : REST + SSE telemetry
```

## Class Descriptions

### `DataIngestor`
Responsible for the entire ETL pipeline. Connects to the yfinance API, handles missing value imputation, adjusts for corporate actions (splits, dividends), applies rolling Z-score normalisation, and persists clean data to PostgreSQL.

### `PostgreSQLStore`
The ACID-compliant persistence layer. Acts as the single source of truth for all historical and incoming market data. Exposes batch fetch methods optimised for the PPO training loop's large sequential reads.

### `FinRLEnvironment`
Wraps the FinRL library to convert PostgreSQL-stored market data into a standard OpenAI Gym-compatible Markov Decision Process. Defines the observation space (state) and action space (portfolio weights) for the PPO agent.

### `PPOAgent`
The core intelligence. Wraps Stable Baselines3's PPO implementation. Exposes `train()` for supervised training cycles and `predict_action()` for real-time inference. Handles model artifact serialisation and loading.

### `MLOpsOrchestrator`
The primary engineering contribution of this thesis. Continuously monitors the live Sharpe Ratio. On detecting drift below `target_sharpe_threshold`, it asynchronously triggers `DataIngestor` for fresh data, initiates `PPOAgent.train()`, and calls `FastAPIServer.reload_model()` — all without blocking live inference.

### `FastAPIServer`
Stateless ASGI inference gateway. Decouples the computationally heavy retraining loop from live trading inference. Exposes REST endpoints for action prediction and telemetry, plus a Server-Sent Events (SSE) stream for the React dashboard.

### `PortfolioDashboard`
The React.js SPA client. Polls the FastAPI telemetry endpoints and renders real-time equity curves, Sharpe gauges, drawdown charts, trading logs, and CT pipeline status indicators.
