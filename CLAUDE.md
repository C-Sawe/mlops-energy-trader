# CLAUDE.md — project context

> Read this fully before touching the code. It carries decisions that are not
> obvious from the source, several of which were reached by finding and fixing
> a bug. Undoing one reintroduces a defect that has already cost time.

---

## 1. What this is

**An MLOps-Enabled Continuous Training Architecture for Reinforcement Learning
in Global Energy Markets.**

Final-year BSc Informatics and Computer Science thesis, Strathmore University,
School of Computing and Engineering Sciences. Student: Caleb Kipchirchir
(169391, ICS 4C). Supervisor: Mr. Allan Vikiru.

The system deploys a PPO reinforcement-learning trading agent for global energy
equities behind a Continuous Training (CT) loop that monitors risk-adjusted
performance, detects alpha decay, and retrains and redeploys the agent without
human intervention.

### The claim under examination

**The contribution is the architecture, not the alpha.** What the thesis
asserts is that a closed-loop CT pipeline can detect its own performance decay
and recover from it autonomously, without pipeline failure or serving
downtime. A favourable backtest number is *not* the result and must never be
presented as one.

Section 1.7 of the proposal already concedes the Sim2Real gap: simulations
cannot reproduce real transaction costs, slippage, or liquidity constraints,
and the agent trains on daily OHLCV rather than tick data. Every claim must
stay inside that boundary. If a strong result appears, the correct framing is
"the architecture responded as designed," never "the strategy is profitable."

This matters practically: when writing Chapter 5, resist any sentence that
turns an architectural result into a financial one.

---

## 2. Academic constraints

The proposal (Chapters 1–3) was approved with **Minor Corrections, 31/40**,
marked by Ms. Salome Chemiat. Chapter 4 is written. Chapter 5 is scaffolded but
must not be filled with anything not actually measured.

**Do not change without consulting the supervisor:**

- The research objectives, questions, or scope
- The equity universe (XOM, CVX, SHEL, BP, NEE)
- The declared paradigm — Object-Oriented Analysis and Design (OOAD), *not*
  SSAD/SSADM. This is why Chapter 4 has UML diagrams and a Logical Database
  Schema rather than an ERD.
- The named technology stack in Section 3.6

**Deviations from the approved design are permitted but must be recorded and
justified in Chapter 5.** The departmental guide requires this explicitly. One
such deviation already exists (see §8, FinRL).

### Outstanding items in the thesis document

These are not code tasks, but they are open and they cost marks:

1. **Section 2.3.1 citation mismatch.** The section describes an ARIMA +
   Attention-LSTM ensemble study but cites Kreuzberger et al. (2023), which is
   an MLOps survey. The correct source has not been identified. Flagged
   independently by the marker (comment #2).
2. **Section 2.3.2 citation.** Cites "Senneset, M. & Gultvedt, E. (2020)",
   which appears nowhere in the reference list. Espiga-Fernández et al. (2024)
   in the bibliography matches the section heading word-for-word. Likely a
   mid-draft source swap that was never propagated.
3. **Front matter.** Table of Contents, List of Figures and List of Tables
   contain no Chapter 4 entries. In Word: select each, press F9, "Update
   entire table". Scored directly by the marking rubric.
4. **Figure 4.8 wireframe.** Shows tech tickers (AAPL, MSFT, NVDA…) instead of
   the energy universe, action values that contradict FR-11, and a retraining
   threshold line drawn at ~0.2 instead of 1.0.

---

## 3. Current state

| Sprint | Scope | Status |
|---|---|---|
| 1 | DataOps — ingestion, features, persistence, schema | **Complete** |
| 2 | Environment — Gymnasium MDP, metrics, baselines | **Complete** |
| 3 | Training — PPO, walk-forward validation, MLflow | **Complete** |
| 4 | Serving — FastAPI, fail-safe, dashboard, CT loop | **Complete** |

```bash
source .venv/bin/activate
python -m pytest tests/ -q     # 112 passing — keep it that way (backend only; see §14 for the frontend)
```

Branches (all local, none pushed): `main` is the trunk. `sprint-1` and
`sprint-2` are retroactive markers at each sprint's completion commit, kept
for reference. `sprint-3` **was merged into `main` (fast-forward)** in the
same local session that built it — Sprint 4's serving layer depends on
Sprint 3's `agent.py`/`registry.py`, so it had to land on `main` before
`sprint-4` branched off it. `sprint-4` is where this sprint was built and,
**as of this checkout, has not been merged to `main`** — the user
explicitly asked for no further merges without asking first, after the
sprint-3 merge already happened. Don't merge `sprint-4` (or anything else)
into `main` without checking first, regardless of what any earlier version
of this file implied about a normal merge cadence.

Tests use deterministic synthetic data and in-memory SQLite. **No network and
no database required.** If a test starts needing either, that is a regression
in the test, not an improvement.

Sprint 2 landed in a local session on 2026-09-17 after the cloud session (no
code access) had already written this file describing it as complete and
measured. It was not: the checkout at that point had only Sprint 1 done, zero
lines of `src/rlops/environment.py`, and 26 tests, not 63. Treat any claim in
this file about a sprint's completeness, a test count, or a measured number
as suspect until verified against the actual checkout — cross-session
handoffs like this one are exactly how a false "Complete" propagates. Sprint 2
is now genuinely complete and the 56 is a real count from this checkout; the
gap above is recorded so the same mistake is easier to catch next time.

---

## 4. Critical invariants

These are properties whose violation produces plausible-looking results that
cannot be reproduced live. That is worse than a crash, because nothing signals
it. Treat a failure in any of these as evidence the *code* is wrong, never the
test.

**I1 — No look-ahead anywhere.**
Rolling windows are backward-looking only (DR-07). Evaluation partitions begin
strictly after training partitions end (DR-06), enforced both in
`partition_chronological()` and as a database CHECK constraint
(`ck_eval_after_train`) so an invalid partition cannot be recorded even by code
that bypasses the helper. Guarded by
`test_zscore_uses_only_backward_looking_window` and
`test_observation_contains_no_future_information`, both of which mutate a
future value and assert no past value changes. The second of these now lives
in `tests/test_environment.py` and checks the property through
`TradingEnvironment`'s own observation, not just `processing.normalize_rolling`
directly — it is the plumbing check that the MDP wrapper didn't reintroduce
leakage the DataOps layer had already closed.

**I2 — Observation/action timing.**
The observation at step *t* contains only information available at the close of
day *t*. The action is executed at *t*'s close and its return realised from *t*
to *t+1*. Verified by `test_return_is_realised_from_t_to_t_plus_one`
(`tests/test_environment.py`): `TradingEnvironment.step()` calls `_rebalance()`
at the current step's price before advancing `_step_idx`, and only then reads
the next price to compute the reward.

**I3 — Forward-fill only, never back-fill.**
Gaps are forward-filled and flagged `is_imputed` (DR-04). Back-filling is a
forward-looking operation. Rows preceding a ticker's first observation are
dropped rather than filled.

**I4 — The drawdown penalty applies to the increment, not the level.**
Penalising the *level* of max drawdown on every step bills the agent
repeatedly for one historical loss and makes reward depend on episode length
rather than on behaviour. Implemented in `TradingEnvironment._reward()`:
`self._max_drawdown` only ever increases, and the penalty is
`max(0, drawdown - self._max_drawdown)`, so a step that holds at the same
drawdown or recovers from it is never re-penalised. Guarded by
`test_drawdown_penalty_applies_to_increment_not_level`.

**I5 — The VIX fail-safe lives in the serving layer.**
Not in the agent, not in the model. It must survive model promotion, so it
cannot sit in a component that promotion replaces. This is why
`InferenceService` owns `vix_critical_threshold` and `_apply_failsafe()` in the
Chapter 4 class diagram.

**I6 — Every trading decision traces to the weights that produced it.**
`trading_decision.version_id → model_version.run_id → model_run`, which records
the exact data partition and hyperparameters. This chain is what makes NFR-07
satisfiable and is a precondition for any post-hoc analysis.

---

## 5. Requirements catalogue

Code should cite the requirement it serves. If no requirement covers a
behaviour you are adding, say so rather than inventing an ID.

### Functional

| ID | Requirement | Obj |
|---|---|---|
| FR-01 | Ingest daily OHLCV for the universe from yfinance without manual intervention | ii |
| FR-02 | Forward-fill missing observations; adjust for splits and dividends | ii |
| FR-03 | Compute SMA_20, RSI_14; append VIX | ii |
| FR-04 | Apply rolling Z-score standardisation | ii |
| FR-05 | Persist to PostgreSQL, partitioned chronologically | ii |
| FR-06 | Expose stored data as a Gym-compatible MDP | iii |
| FR-07 | Train PPO with a reward penalising maximum drawdown | iii |
| FR-08 | Log hyperparameters and metrics of every run to MLflow | iii |
| FR-09 | Register the selected artifact with a version identifier | iii |
| FR-10 | Expose an inference endpoint returning a discrete action | iv |
| FR-11 | Map continuous weight to discrete action at the defined thresholds | iii |
| FR-12 | Bypass inference and return capital-preservation when VIX exceeds critical | iv |
| FR-13 | Compute the 30-day rolling Sharpe Ratio on a schedule | iv |
| FR-14 | Trigger background retraining when rolling Sharpe falls below target | iv |
| FR-15 | Continue serving from the incumbent model throughout retraining | iv |
| FR-16 | Reload a new model without restarting the service | iv |
| FR-17 | Retain the incumbent and abort promotion if the candidate fails out-of-sample | iv |
| FR-18 | Render equity curve, rolling Sharpe, max drawdown | iv |
| FR-19 | Display a paginated decision log | iv |
| FR-20 | Surface CT pipeline status (serving / evaluating / retraining) | iv |

**FR-17 is an addition to the original Blueprint.** As originally specified,
the CT loop trained and reloaded unconditionally, which would let the system
autonomously promote a *worse* model simply because a cycle completed. Making
promotion conditional on out-of-sample acceptance prevents the closed loop from
becoming a mechanism for compounding error.

### Non-functional

| ID | Requirement | Characteristic |
|---|---|---|
| NFR-01 | ≥95% of inference requests return within **20 ms** over **100** requests | Performance |
| NFR-02 | Latency degrades by no more than **10%** during retraining | Performance |
| NFR-03 | Serving remains continuously available across a full CT cycle; zero failed requests | Reliability |
| NFR-04 | Recover to serving within **1 min** of unplanned termination, no data loss | Reliability |
| NFR-05 | A failed retraining cycle leaves the incumbent serving and state consistent | Reliability |
| NFR-06 | No circular dependencies between packages, verified by static analysis | Maintainability |
| NFR-07 | Every served model traceable to run, partition boundaries, hyperparameters | Accountability |
| NFR-08 | Payloads schema-validated; malformed input rejected, not crashed | Security |
| NFR-09 | Dashboard renders without unbounded memory growth over **[X] h** — still unmeasured | Performance |
| NFR-10 | No credential, key or connection string in source control or logs | Security |
| NFR-11 | The fail-safe takes precedence over any model output and is logged with its trigger value | Human oversight |

**The bracketed thresholds were deliberately unset until measured on the
target hardware** — the departmental guide is explicit that thresholds must
not be invented to make a requirement look measurable. NFR-01/02/04 are now
filled in from real measurements on the M5, taken 2026-09-17 via
`fastapi.testclient.TestClient` against a real `/predict` call chain (real
SQLite-backed repository, real loaded PPOAgent, real fail-safe check — the
only thing not real is the HTTP transport itself, which `TestClient` calls
in-process; a real deployment adds network round-trip on top of these
numbers, but that's a constant offset, not something specific to this
service):
- **NFR-01: p95 = 16.9ms, p99 = 77.1ms, max = 78.3ms over 100 requests** — set
  at 20ms with headroom over the measured p95, not at the median. The p99
  tail (~4.5x the p50) is worth watching if this ever needs re-measuring
  against real Postgres instead of SQLite.
- **NFR-02: 2.1% p95 degradation while a real background retrain was
  running** (triggered via `orchestrator._trigger_retrain()`, same code
  path FR-14 uses) — set at 10% for margin. This is a genuinely good result,
  not a lenient threshold chosen to pass: `predict()` and the retrain thread
  touch different model instances and only briefly share
  `InferenceService`'s lock at the moment of promotion, so FR-15's
  "continue serving throughout retraining" holds up under actual
  measurement, not just by design intent.
- **NFR-04: ~8ms mean, ~11ms max (N=5) to reconstruct `InferenceService`
  and reload the active model from scratch** — what a fresh process does on
  startup. Set at 1 minute, which is almost entirely headroom: this number
  is the *application's* readiness time once the process is actually
  running again, not the OS/process-manager's crash-to-restart time (e.g.
  systemd/supervisor restart delay), which is an infrastructure concern
  outside this service's own code and wasn't part of this measurement.
- **NFR-09 remains unmeasured** — it requires the dashboard to actually run
  for hours in a browser, which a backend measurement pass can't produce.
  Leaving it bracketed is the correct call here, not an oversight: inventing
  a number to fill the row would be exactly what the departmental guide
  warns against.

### Data

DR-01 source OHLCV + VIX via yfinance · DR-02 required fields, composite key
(date, ticker) · DR-03 split/dividend adjustment · DR-04 forward-fill, flagged
· DR-05 duplicates rejected by uniqueness constraint · DR-06 chronological
partition, no overlap · DR-07 backward-looking normalisation only · DR-08 every
run records its partition boundaries · DR-09 artifacts versioned and
restorable · DR-10 public market data only, no personal data

### Interface

IR-01 HTTPS with bounded retry + backoff · IR-02 exhausted retries fail
explicitly, leaving the prior dataset intact · IR-03 SQLAlchemy against
PostgreSQL, env-supplied connection · IR-04 documented JSON/REST contract ·
IR-05 Pydantic validation at the boundary · IR-06 registry via client API ·
IR-07 dashboard degrades gracefully when the backend is unreachable

---

## 6. Architecture

Five layers. **Dependencies flow one way only** — no package imports a package
that imports it (NFR-06).

```
presentation ──▶ serving ──▶ rlops ──▶ dataops
                     ▲          ▲         ▲
                     └──── orchestration ─┘
```

```
src/dataops/          ingestion · processing (+ walk_forward_splits)   [Sprint 1, 3]
                      · repository (+ decisions/snapshots/versions)    [Sprint 1, 4]
                      · models
src/rlops/            environment · baselines                         [Sprint 2]
                      · agent · registry                              [Sprint 3]
src/orchestration/    evaluator                                       [Sprint 2]
                      · ct_orchestrator                                [Sprint 4]
src/serving/          schemas · inference · api                       [Sprint 4]
frontend/              React + hand-rolled SVG charts, no CSS framework [Sprint 4]
scripts/              run_ingestion.py · run_baselines.py
                      · benchmark_device.py · finrl_crosscheck.py
                      · train_agent.py                                [Sprint 3]
tests/                107 tests (backend; frontend has no test suite yet)
```

The closed feedback loop that constitutes the contribution: telemetry from the
serving boundary → drift evaluator → CT orchestrator → back into ingestion,
training, and serving (hot reload). The system's output is also its input.

### Configuration

All values environment-driven (IR-03); nothing hard-coded. `src/config.py`.

| Setting | Default | Notes |
|---|---|---|
| `tickers` | XOM, CVX, SHEL, BP, NEE | ^VIX ingested separately as a market-wide feature |
| `sma_window` / `rsi_window` | 20 / 14 | |
| `zscore_window` | 60 | backward-looking |
| `train` / `eval` | 2015-01-01→2023-12-31 / 2024-01-01→2025-12-31 | eval strictly after train |
| `vix_critical_threshold` | 35.0 | FR-12 |
| `target_sharpe_threshold` | 1.0 | FR-14 |
| `sharpe_evaluation_window` | 30 | |
| `buy` / `sell` threshold | 0.5 / −0.5 | FR-11 |
| `initial_cash` | 100,000 | `EnvironmentConfig`, FR-06 |
| `transaction_cost_pct` | 0.001 | `EnvironmentConfig`, FR-07 |
| `drawdown_penalty_coef` | 1.0 | `EnvironmentConfig`, FR-07/I4 |

Thresholds are configurable rather than constant so Chapter 5 can report a
**sensitivity analysis** over them instead of defending magic numbers. Doing
that analysis is a genuine strengthening of the evaluation — plan for it.

---

## 7. Non-obvious decisions

Each of these was reached by finding a bug. Changing one without understanding
why it exists will reintroduce it.

**Transaction-cost bisection scales the buy legs only — never the sell legs.**
A fully invested portfolio holds no cash, so any rebalance leaves its fee
unfunded. The first implementation of this scaled the *entire* delta vector
(every ticker's buy and sell) by one factor found via bisection. That is
wrong: in a pure swap between two tickers (sell 100% of A, buy 100% of B), the
sell and buy notionals are equal and opposite, so they cancel regardless of
the scale factor — cash-after-trade came out negative for *any* nonzero scale,
and bisection converged on executing essentially nothing. The fix: a sell's
proceeds always exceed its own fee (for any `transaction_cost_pct < 100%`), so
sell legs execute in full and always fund themselves; only the buy legs draw
down cash-on-hand plus those proceeds, and only the buy legs are what
bisection scales back if that isn't enough to cover their principal and fee.
Cash-after-trade is monotone decreasing in the buy-side scale factor, which is
what makes bisection valid. Caught by
`test_multi_ticker_swap_sells_in_full_and_scales_the_buy`
(`tests/test_environment.py`) — a single-ticker buy-from-cash or sell-to-cash
test cannot distinguish the correct and the broken version, because with only
one leg there is nothing for the wrong symmetric scaling to get wrong. Any
future change to `TradingEnvironment._rebalance` should be checked against a
multi-ticker swap, not just a single-ticker trade.

**Unaffordable trades scale down; they are never rejected.**
Rejection makes the action space discontinuous: a target one cent too large
produces no trade whatsoever, which is an uninformative gradient for the policy
to learn from. Note that this cannot be verified by asserting the resulting
*weight* is below 1.0 — once cash is a negligible residual either way, the
weight rounds to ~1.0 regardless of whether the buy was actually scaled back.
The tests check absolute share counts against the frictionless target instead
(`test_buy_from_cash_is_scaled_back_by_fee_not_rejected`).

**`_DUST_FRACTION = 1e-6` deadband on trades.**
Actions are float32, so expressing "hold my current position" as a target
weight carries round-trip error of ~2e-7 of portfolio value (measured, not
assumed). Without a deadband above that, every hold registers as a trade and a
buy-and-hold baseline appears to rebalance daily. 1e-6 of a 100,000 portfolio
is ten cents — well below one share. Implemented in
`TradingEnvironment._rebalance`, applied to the trade's dollar notional, not
to the weight difference itself.

**`_VARIANCE_FLOOR = 1e-12` on Sharpe.**
A constant float series has standard deviation on the order of 1e-18, not
exactly zero, from float rounding noise. Without the floor,
`sharpe_ratio([0.01]*20)` returns an enormous but finite number (order 1e16),
which in production would silently satisfy `target_sharpe_threshold` and
suppress the CT retraining trigger. `rolling_sharpe` applies the same floor to
the *variance* before taking its square root, and separately masks warm-up
rows (fewer than `window` observations) to NaN rather than letting them read
as a Sharpe of zero. Both live in `src/orchestration/evaluator.py`.

**RSI uses Wilder's exponential smoothing**, `alpha = 1/window`, not a simple
rolling mean of gains and losses. Most tutorial implementations get this wrong;
the two diverge materially over the first ~3×window periods.

**Transformation order in `build_feature_frame()` is load-bearing.**
Corporate-action adjustment → imputation → indicators → VIX join →
normalisation. Indicators computed on unadjusted prices are meaningless across
a split, and normalisation must standardise final feature values, not
intermediate ones.

**`persist()` is idempotent, not append-only.**
The scheduled job refetches a trailing window to pick up late corrections, so
re-ingesting an overlapping range updates rather than raises. DR-05 still holds
— exactly one row survives per (date, ticker).

**Normalise before partitioning, not after.**
`scripts/run_baselines.py` calls `normalize_rolling()` on the full
`train_start`→`eval_end` range before calling `partition_chronological()`, not
the other way around. Normalising only within the eval partition would waste
the first `zscore_window - 1` rows of every partition on warm-up NaNs, even
though using pre-partition history to standardise an eval-partition row is
still purely backward-looking (DR-07) and leaks nothing forward. This is
distinct from DR-06/I1, which is about training never seeing evaluation data —
here the flow of information is the reverse direction (eval reading further
into the past), which is always safe.

**`load_partition()` returns a correctly-shaped empty frame, not `pd.DataFrame([])`.**
Found via the dashboard, not a test: `CTOrchestrator.evaluate()`'s "Force
Check" button hit a real 500. `pd.DataFrame([])` for a query that matched no
rows has zero *columns*, not just zero rows, so `normalize_rolling()`'s
column check failed with "missing required columns" — a confusing error
that has nothing to do with the actual problem (no market data in the
requested range, e.g. ingestion hasn't caught up to `as_of` yet). Fixed in
`MarketRepository.load_partition` to return a frame with the right columns
and zero rows; `CTOrchestrator.evaluate()` also now checks for this
explicitly and logs + returns to `SERVING` rather than letting a stale
ingestion job crash the request that triggered the evaluation. This is a
recoverable, expected condition (the next scheduled tick likely finds
data), not a programming error, so it is handled, not raised.

**`sqlite:///:memory:` is not safe across threads — use a file, even in tests.**
Found writing `tests/test_ct_orchestrator.py`: an in-memory SQLite database
is not shared across connections, and each new thread that calls
`repo.session()` can get a genuinely different connection from the pool. The
CT orchestrator's whole point is a background thread retraining while the
main thread keeps serving — with `:memory:`, that background thread's
connection sees a separate, empty database ("no such table:
market_observation"), not the one the test just seeded. Every CT
orchestrator test uses a temp-file-backed SQLite DB instead
(`sqlite:///{tmp_path}/test.db`). This is purely a test-fixture concern —
real deployments use Postgres or a file, both genuinely shared across
connections — but it would bite anyone who reached for `:memory:` for "quick
local testing" of anything that touches threading, which the CT loop
inherently does.

**A rejected candidate is logged too, not just a promoted one.**
Found while building the dashboard's "Cycle history" card, which wants a
real "held back" count. `CTOrchestrator._retrain_and_maybe_promote`
originally called `registry.log_run()` only inside the branch where the
candidate beat the incumbent — a candidate that lost the FR-17 acceptance
gate left no record anywhere that it had ever been trained. FR-17 says
"retain the incumbent and abort promotion"; it does not say "and forget the
attempt happened." Fixed: `log_run()` now takes a `status` argument, and
every candidate is logged (`"COMPLETED"` if promoted, `"REJECTED"` if not)
regardless of outcome — only `register_version()` + `promote_version()` are
conditional on winning. `MarketRepository.get_cycle_stats()` counts both.

**`/ct/evaluate` takes an optional `as_of` — `date.today()` isn't always the right anchor.**
Found bootstrapping the CT loop against this checkout's real ingested data
(2015-01-02→2025-12-30): the default `evaluate()`/telemetry both anchor to
`date.today()`, but a checkout's ingested data can lag behind the literal
current date for entirely ordinary reasons — a fresh checkout, an ingestion
gap, or (as here) a sandboxed dev clock running ahead of the last real
trading day the data covers. Anchoring to `date.today()` unconditionally
made both the backend's evaluation and the dashboard's charts go silently
blank rather than showing what data actually exists. Fixed on both ends:
`POST /ct/evaluate?as_of=YYYY-MM-DD` lets an operator force evaluation
against a specific date (`src/serving/api.py`), and the dashboard's
`fetchTelemetry()` now anchors its window to `/ct-status`'s
`last_ingest_date` instead of `new Date()` (`frontend/src/api.js`) — "last N
days" means the same thing whether ingestion is perfectly current or a few
days behind, not "blank unless today happens to have data."

---

## 8. FinRL — the recorded deviation

The proposal names FinRL. **It does work.** An earlier claim in this project
that it was broken was wrong and has been corrected.

`StockTradingEnv` imports successfully once further packages are installed.
The list of six in the previous version of this file (`alpaca_trade_api`,
`exchange_calendars`, `pytz`, `stockstats`, `wrds`, `yfinance`) turned out to
be incomplete for the current PyPI release (FinRL 0.3.7, 2026-09-17): the
actual chain that fires just from `import finrl` — it eagerly imports
`finrl.test`, which imports the env module directly — needed `gymnasium`,
`stable-baselines3`, and `matplotlib` as well, none of which are in that
original list. `stockstats` was needed; `alpaca_trade_api`,
`exchange_calendars`, `pytz`, `wrds`, `yfinance` were not, at least for
reaching `StockTradingEnv` — they may matter for FinRL's live-data
connectors, which nothing here touches. It is Gymnasium-based, 557 lines in
this release, and already models transaction costs (`buy_cost_pct`) and a
`turbulence_threshold` close in spirit to the VIX fail-safe. The drawdown
penalty *can* be added by post-processing `super().step()` in a subclass —
this was verified working.

Installed into an isolated venv, not this project's own `.venv` — FinRL's
dependency footprint is dated enough that pinning it alongside Sprint 1/2's
numpy 2.x/pandas 2.x/gymnasium 1.x stack risked real conflict for a
one-off validation experiment. See `scripts/finrl_crosscheck.py`'s docstring
for the exact install steps if this needs re-running.

It is not used as the primary environment for two reasons:

1. Its reward is assigned inline —
   `self.reward = end_total_asset - begin_total_asset` at line 330 — with **no
   override hook**, so FR-07's penalty can only be bolted on afterward rather
   than expressed in the reward itself.
2. Its actions are `hmax`-scaled share counts, not the continuous [-1, 1]
   target weights this project specifies.

`src/rlops/environment.py` implements that specification directly: `TradingEnvironment`,
under test (30 tests across `test_environment.py` and `test_baselines.py`),
passing Stable Baselines3's `check_env` (`test_conforms_to_gymnasium_api`).

**Validation experiment — done, 2026-09-17 (`scripts/finrl_crosscheck.py`).**
Both environments hold the same fixed 100-share position in each of the
5 tickers, bought on the same calendar date at the same (synthetic) price,
with transaction costs and the drawdown penalty both disabled in both, so
each reduces to plain PnL against an identical holding across a 222-day
aligned window. Result:

| | ours | FinRL |
|---|---|---|
| initial equity | 100,000.0000 | 100,000.0000 |
| final equity | 102,108.6627 | 102,108.6627 |

Max absolute difference **$0.00007237** on a ~$100–102K portfolio (max
relative difference ~7×10⁻¹⁰), correlation **1.000000000000**. That residual
is consistent with float32 action rounding on this project's side (the
initial target weight is cast to float32 before `step()` upcasts it back to
float64), not a real accounting divergence. This corroborates
`TradingEnvironment`'s valuation and return accounting against FinRL's
independently published implementation — the §8 deviation is now backed by
evidence, not just a design justification. It does *not* validate the
transaction-cost or reward mechanics specifically (both were disabled to
isolate accounting), and it used single-step buy-and-hold, not a trading
policy — see §7's transaction-cost note for what *is* tested there.

One FinRL implementation detail this surfaced, worth knowing before anyone
tries to reuse `StockTradingEnv` with `tech_indicator_list=[]`:
`_buy_stock`/`_sell_stock` hard-code a "disabled" flag at
`state[index + 2*stock_dim + 1]`, i.e. inside the first technical
indicator's slot, regardless of whether indicators are otherwise used. An
empty `tech_indicator_list` leaves that slot out of the state vector
entirely, and every buy or sell raises `IndexError`. The workaround: supply
one always-`False` dummy column, e.g. `tech_indicator_list=["disable"]`.

---

## 9. Verified empirical findings

Measured, not assumed. Cite these rather than re-deriving them. The items
below predate Sprint 2 and concern Sprint 3 (training, not yet started) —
most have now been re-verified against this checkout (dated where so); the
ones that haven't are marked explicitly.

**The policy network is tiny.** For 5 tickers × 8 features + positions + cash =
46-dimensional observation, SB3's default `MlpPolicy` (two 64-unit hidden
layers for each of π and V) is **14,731 parameters, 57.5 KB**. ResNet-50 has
25 million by comparison. The 46-dimensional figure is now confirmed by
`TradingEnvironment.observation_space` itself
(`test_observation_and_action_space_shapes`), not just asserted.

**MPS vs CPU, measured on the actual M5 (2026-09-17).**
`scripts/benchmark_device.py` runs identical `PPO("MlpPolicy", ...).learn()`
calls against `TradingEnvironment` on the real 5-ticker/46-dim observation
space, on both devices, with an untimed 2048-step warm-up before each timed
measurement (MPS pays a one-time kernel-compilation and device-transfer setup
cost on first use that would otherwise be charged to the measurement, not to
steady-state throughput). Two independent seeds, at 20,000 measured
timesteps each:

| device | seed 0 | seed 1 |
|---|---|---|
| cpu | 5,677 steps/sec | 5,718 steps/sec |
| mps | 452.6 steps/sec | 438.5 steps/sec |

**CPU wins by roughly 12–13×.** This confirms the hypothesis this section
used to only speculate about: PPO alternates rollout collection (thousands of
forward passes on a single observation) with small minibatch updates, and at
14.7K parameters the CPU↔GPU transfer overhead dominates whatever the
arithmetic would have gained from the GPU. **Use `device="cpu"` for Sprint
3** — do not pass `device="mps"` to `PPO(...)`, it is not close. A GPU would
only pay off with `CnnPolicy` on image inputs or with 50+ parallel
environments, neither of which applies here.

**Training is CPU-viable — now confirmed at full scale, not extrapolated.**
500,000 timesteps ran in **87.0 seconds** (5,750 steps/sec) on the actual M5
CPU, `scripts/benchmark_device.py --timesteps 500000`. The earlier "875
steps/sec on a weak cloud CPU → under 10 minutes" figure was itself a
measurement, just not one taken on this hardware — the M5 turns out to be
about 6.5× faster than that reference point, not merely "faster" as
previously hedged.

**Colab is not needed** and carries a real cost: a dropped 12-hour session at
the wrong moment. (Unverified against this checkout — no Colab run has been
attempted from here; the M5 CPU numbers above make it unnecessary to try.)

**Yahoo Finance is unreachable from cloud containers** (403 on CONNECT,
gateway policy). It works fine from a local machine.

**Live ingestion verified against a real response (2026-09-17, this
checkout).** `fetch_market_data()` had never been exercised against a real
yfinance response before this. It now has, for the full 5-ticker universe
plus VIX over 2024-01-01→2024-06-01, with `--dry-run` and by persisting the
real fetched frame through the ORM into SQLite (Docker Desktop was not
running locally, so this could not be run against actual PostgreSQL — see
the gap that leaves, below). Findings:
- **yfinance 1.7.0 returns MultiIndex columns even for a single-ticker
  download** — `[('Adj Close', 'XOM'), ('Close', 'XOM'), ...]`, not the flat
  frame the module's docstring implies is the single-ticker case. This is
  more aggressive than the "changes periodically" warning anticipated, but
  `_normalise_yf_frame()`'s existing `isinstance(df.columns, pd.MultiIndex)`
  branch already handles it correctly — `get_level_values(0)` drops the
  ticker level, which is safe here specifically *because* `_fetch_one` only
  ever requests one symbol at a time (`fetch_market_data` loops per ticker
  rather than batching). Flattening that same way on a genuine multi-ticker
  batch download would silently collide two tickers' `Close` columns; that
  path is not exercised anywhere in the current code, so it stayed latent
  rather than becoming a bug — worth remembering if a future change ever
  batches the download.
- **`Adj Close` is returned** with `auto_adjust=False`, for both equities and
  `^VIX` — confirmed non-trivially different from `Close` for XOM (dividend
  drift over ~2.5 years from the 2024 dates to today), and correctly equal to
  `Close` for `^VIX` (no dividends/splits on an index).
  `adjust_corporate_actions()` consumed it without changes.
- **`^VIX` returns `Volume=0`**, not a missing column — `_normalise_yf_frame`'s
  required-column check passes without special-casing this.
- Over 2024-01-01→2024-06-01, the real universe had **0 imputed rows** — all
  five tickers traded on the same calendar for this window — and **230 of 525
  rows** had a complete normalised feature vector, exactly matching
  `5 tickers × (105 − 59) = 230` from the 60-day zscore warm-up. The math
  checked out against real data, not just synthetic.
- The persisted-and-reloaded frame round-tripped through
  `MarketRepository.persist` / `load_partition` / `latest_state` correctly —
  dtypes, NaN handling, and numeric precision all held up against real
  values, not just the synthetic fixtures the test suite uses.

**That gap is now closed too (2026-09-17, same session).** Started Docker
Desktop, brought up `postgres` via `docker compose up -d postgres`, and ran
`scripts/run_ingestion.py --start 2024-01-01 --end 2024-06-01` for real (no
`--dry-run`) against it — the same 525-row universe as above, this time
landing in actual PostgreSQL rather than SQLite. Also confirmed, against the
real container, the two things `tests/test_repository.py` explicitly flags as
unverified on SQLite:
- **Idempotency (DR-05).** Running the exact same ingestion command a second
  time left the store at 525 rows, not 1050 — `persist()`'s upsert behaviour
  holds on Postgres.
- **Both CHECK constraints actually fire.** Manually inserting a `ModelRun`
  with `eval_start <= train_end` raised `IntegrityError` (`ck_eval_after_train`,
  DR-06/I1); inserting a `MarketObservation` with `high_price < low_price`
  raised `IntegrityError` (`ck_high_ge_low`) — both roundly rejected rather
  than silently accepted, and the observation count stayed at 525 after each
  rollback.

`load_partition`, `latest_state`, and `scripts/run_baselines.py` were all
re-run against this real Postgres-backed store and produced sane output
(buy-and-hold Sharpe 1.99, random policy losing money to cost drag over the
2024-04-01→2024-06-01 eval slice). The persistence path is genuinely closed
now, not just on SQLite. Docker Desktop + the `mlops_postgres` container were
left running locally after this — `docker compose down` to stop them if
they're not wanted between sessions.

---

## 10. Evaluation protocol

Do all four before reporting any agent result. Skipping them is the single
most common way an RL trading thesis fails at defence.

1. **Run the baselines.** `python scripts/run_baselines.py`
   A Sharpe Ratio is meaningless alone — the same conditions that reward a
   learned policy also reward simply holding the assets. Available:
   buy-and-hold, equal-weight-rebalanced, random, all-cash
   (`src/rlops/baselines.py`). On synthetic data buy-and-hold reaches Sharpe
   1.14; an agent must clearly separate from that.

2. **Run ≥5 seeds; report mean ± standard deviation.**
   RL is notoriously seed-sensitive. A single-seed number is not a result.
   Five runs at ~10 minutes each is one afternoon.

3. **Apply `deflated_sharpe_ratio()`** (`src/orchestration/evaluator.py`) with
   the *true* number of configurations tried, including abandoned ones. It
   implements Bailey & López de Prado (2014), already cited in Chapter 2. The
   best of fifty backtests shows a positive Sharpe by construction; this
   quantifies how much of the observed figure that explains. Using a source
   you already cite to discipline your own results reads very well at
   defence.

4. **Keep transaction costs on.** Frictionless simulation systematically
   flatters any policy that trades often. On synthetic data the random policy
   loses to cost drag against simply holding cash
   (`test_random_policy_loses_to_cost_drag_relative_to_all_cash_baseline`) —
   without costs it looks roughly flat. That contrast is itself a good
   Chapter 5 discussion point.

Also report: walk-forward validation across distinct market regimes (Section
3.3.4), not a single split.

### First real training run — 2026-09-17, this checkout

Real market data (13,825 rows, 2015-01-02→2025-12-30, all 5 tickers + VIX,
ingested for real via `scripts/run_ingestion.py` with no `--start`/`--end`
override, into the real `mlops_postgres` container) through
`scripts/train_agent.py` at real settings (no synthetic data, no toy
hyperparameters): **18 walk-forward splits × 5 seeds = 90 models**, ~20
minutes on the M5 CPU. All 90 runs logged to `model_run`; the best seed per
split registered as a `model_version` (18 total) — **none promoted**, which
is correct, not a gap: this script produces validation folds for reporting,
not a production deployment decision (`ModelRegistry.register_version()`
deliberately never sets `is_active`; see §7).

**The headline numbers, reported the way §10 asks them to be, not the
flattering way:**

| Statistic | Value | What it's measuring |
|---|---|---|
| Best-seed-per-split Sharpe, n=18 | mean 0.570, std 1.279 | what gets *registered* each split — already a maximum over 5 seeds |
| All individual seeds pooled, n=90 | mean −0.606, std 1.507 | what an *untuned* seed actually gets you |
| Positive splits | 10/18 | regime-dependent, not uniformly positive |

That gap between 0.570 and −0.606 is the selection-bias effect §10.3 exists
to catch, demonstrated with this project's own real numbers rather than a
hypothetical. **The best single result across all 90 runs** was split 11
(eval 2021-12-05→2022-06-02), seed 0, Sharpe **3.273**, `run_id`
`7f60784a-805f-4047-ad74-fad67261920c` — re-run standalone to confirm exact
reproducibility given a fixed seed (it reproduced to three decimal places)
and to recover the per-period returns MLflow didn't log a copy of:

- **Undeflated (treated as the only thing tried): PSR = 0.986** — 98.6%
  probability this Sharpe is genuinely positive, taken alone.
- **`deflated_sharpe_ratio(returns, n_trials=90)` = 0.388** — under 40%,
  correcting for having actually tried 90 configurations and reported the
  best. This is the number that belongs in Chapter 5, not the 0.986, and
  not the 3.273 either.
- **Baselines on that exact window** (§10.1, same eval partition, so the
  comparison is fair): buy-and-hold Sharpe **2.419**, equal-weight
  **2.388**, random (mean of 5 seeds) **−3.192**, all-cash **0.000**. The
  agent's 3.273 beats buy-and-hold, but buy-and-hold *itself* scored 2.4 on
  this window — Dec 2021–Jun 2022 was simply a strong period for energy
  equities. The margin over a passive baseline is real but modest once
  that's accounted for, not the standalone number suggests.

**Read this the way §1 requires, not the way it's tempting to read it.** The
correct sentence is: the walk-forward + seed-sweep + deflation + baseline
pipeline ran end to end on real data and produced a disciplined, honest
number (DSR 0.388 on the best fold) instead of an inflated one (0.986 or
3.273) — that is the architecture responding as designed. The incorrect
sentence is any version of "the strategy made 41% in six months" — that
figure is one seed, one fold, in one favourable regime, and the very
analysis above exists to stop that number from being reported as a result.

**Not yet done:** computing `deflated_sharpe_ratio` for the other 17 splits
(only the single best result was disciplined this way, as a demonstration —
a full Chapter 5 treatment would want this, or the pooled-90 framing, for
every fold); and re-running with more than 5 seeds if the defence wants
tighter confidence intervals than std≈1.3 gives.

**The CT loop's core claim — autonomous decay detection and recovery — has
now been observed live, not just unit-tested (2026-09-17, same checkout).**
Promoting one of the 18 candidates was the missing step above; doing it
through the real `/ct/evaluate` endpoint (rather than a direct database
write) exercised the whole loop end to end for the first time:

1. **Bootstrap.** `POST /ct/evaluate?as_of=2025-12-30` with no incumbent yet
   active. `CTOrchestrator` has no baseline to compare against, so it trains
   and promotes unconditionally — this is what put candidate
   `e928c835-bbcb-4531-8824-0e464037f407` into `is_active`.
2. **Decay detection, live.** A second `evaluate()` call replayed that
   incumbent over the real market data through `TradingEnvironment` and
   computed `rolling_sharpe = -3.79` — well below `target_sharpe_threshold =
   1.0` — and autonomously fired a background retrain (FR-14), the same code
   path the 300-second scheduler uses.
3. **The FR-17 gate held.** That retrain's candidate did not beat the
   incumbent out-of-sample, so it was rejected rather than promoted —
   `rejected_count` moved 0→1 — and the original incumbent stayed active.

This is the literal architectural claim under examination (§1): a closed
loop that notices its own performance decay and responds without human
intervention or serving downtime, observed running against real ingested
data rather than asserted from unit tests or design intent. It does not
demonstrate a profitable strategy — the replayed Sharpe was strongly
negative — and per §1 that is not the point: the system's response to a bad
Sharpe is exactly what was supposed to happen.

---

## 11. Code conventions

- **Cite the requirement.** Docstrings reference FR/DR/NFR/IR IDs. Keep the
  traceability table in `README.md` current — it is the basis of Chapter 5
  §5.3.
- **Add a test with any behaviour change.** Synthetic deterministic data,
  in-memory SQLite. No network, no database.
- **Never commit `.env`, credentials, or API keys.** If a secret is ever
  committed, *rotating it is required* — deleting the file is not sufficient.
  `DatabaseConfig.__repr__` masks the password deliberately (NFR-10).
- **Prefer explicit failure to silent degradation.** IR-02 exists because a
  half-written batch is indistinguishable downstream from a genuine market gap.
- Type hints throughout; `from __future__ import annotations` at module top.
- **A policy is a callable `(env, step) -> action`, not a fixed array.**
  "Hold the current allocation" is not a constant target weight — it is
  whatever `TradingEnvironment.current_weights` currently reports, since
  prices move between rebalances. See `src/rlops/baselines.py`.

---

## 12. Next tasks, in order

**1. ~~Verify live ingestion~~ — done 2026-09-17, see §9.** No code change was
needed: `_normalise_yf_frame()` already handled the real (MultiIndex, even
for a single ticker) response shape correctly, and persistence — including
both CHECK constraints and idempotency — is now confirmed against real
PostgreSQL, not just SQLite. `mlops_postgres` is currently running locally
with 525 real rows in it from this check (`docker compose down` to stop it).

**2. ~~Benchmark MPS vs CPU~~ — done 2026-09-17, see §9.** CPU wins by
~12–13× on this hardware. Use `device="cpu"` when `src/rlops/agent.py` is
built in Sprint 3 — this is now a settled decision, not an open question.

**3. ~~Run the FinRL cross-check~~ — done 2026-09-17, see §8.** Max absolute
difference $0.00007 on a ~$100K portfolio, correlation 1.000000000000. The §8
deviation is now backed by a cross-check, not just a design justification.

**4. ~~Sprint 3 — training and registry~~ — done 2026-09-17, on the
`sprint-3` branch (not yet merged to `main`).**
- `src/rlops/agent.py` — `PPOAgent`, a thin wrapper over SB3's PPO
  (`device="cpu"`, per §9's measured decision). `train`/`predict`/`evaluate`/
  `save`/`load`; `evaluate()` returns the same `{equity_curve, returns}`
  shape `baselines.run_policy` does, so an agent's results and a baseline's
  are directly comparable with no glue code.
- `src/rlops/registry.py` — `ModelRegistry.log_run()` (FR-08: MLflow
  params/metrics/artifact, then DR-08: the exact partition boundaries into
  `model_run`) and `.register_version()` (FR-09). Defaults to a local
  SQLite-backed MLflow store (`sqlite:///mlruns.db`), not `file:./mlruns` —
  MLflow 3.x put the plain filesystem backend into maintenance mode and
  refuses to open one without an explicit opt-out flag; the local default
  here is the forward-compatible database URI, not a flag that silences the
  deprecation. A real deployment points `MLFLOW_TRACKING_URI` at the
  docker-compose `mlflow` service instead.
- `src/dataops/processing.walk_forward_splits()` — rolling (train, eval)
  windows across distinct regimes, each satisfying DR-06 by construction
  (eval_start is always train_end + 1 day, not a separately-checked
  invariant).
- `scripts/train_agent.py` — runs the walk-forward × seed-sweep loop end to
  end: trains, evaluates, logs every run, and registers the best-Sharpe
  seed per split as a `model_version`. Smoke-tested against a temporary
  SQLite store spanning ~3.5 years of synthetic data: 17 splits × 2 seeds
  ran cleanly with no errors. Per-seed Sharpe varied wildly within some
  splits (e.g. one split: seed 0 → 1.67, seed 1 → −7.23) — exactly the
  seed-sensitivity §10.2 already warned about, not a bug; it is the
  concrete demonstration of why the seed sweep is there.
- 16 new tests (`test_agent.py`, `test_registry.py`, plus 4 for
  `walk_forward_splits` in `test_processing.py`), all synthetic/offline —
  the registry tests chdir into `tmp_path` (MLflow's artifact store defaults
  to a relative `./mlruns` regardless of the tracking DB location, which
  will otherwise leak a stray directory into the repo root on every test
  run — it did, once, before this fix; deleted, not committed).

**5. ~~Sprint 4 — serving and the CT loop~~ — done 2026-09-17, on the
`sprint-4` branch (branched off `main` after sprint-3 merged; not yet
merged back — see §3).**
- `src/serving/schemas.py` — Pydantic request/response models (IR-04, IR-05,
  NFR-08). `PredictRequest.positions` validates its keys are exactly the
  configured universe and every weight is in [-1, 1] — malformed input is a
  422, never an unhandled crash.
- `src/serving/inference.py` — `InferenceService`. Fail-safe checked
  *before* any model call (I5/FR-12: fetches the latest VIX first, and only
  builds the full observation / calls the agent if it's below
  `vix_critical_threshold`); FR-11's weight→action thresholds; `reload()`
  for hot-swapping the active model under a lock, so a promotion never
  serves a half-loaded agent (FR-16).
- `src/serving/api.py` — FastAPI app. `/predict` (FR-10), `/telemetry`
  (FR-18), `/decisions` (FR-19, paginated), `/ct-status` (FR-20),
  `/ct/evaluate` (on-demand trigger, same code path the scheduler uses).
  Constructs its `repo`/`service`/`orchestrator` singletons in `lifespan`,
  not at import time — `InferenceService.__init__` hits the database
  immediately, and doing that at import time would make importing the
  module reach for whatever `DATABASE_URL` happens to be set, including
  from an unrelated earlier test.
- `src/orchestration/ct_orchestrator.py` — `CTOrchestrator`. `evaluate()`
  replays the incumbent over the most recent real market data through
  `TradingEnvironment` (there's no live broker in this project's scope —
  see the module's Sim2Real note), persists a snapshot per day (FR-13),
  and triggers a background retrain (FR-14) if the resulting rolling Sharpe
  is below target. The retrain thread trains a candidate, evaluates it
  out-of-sample against the same window the incumbent was just evaluated
  on, and only promotes if the candidate's Sharpe beats the incumbent's
  (FR-17) — otherwise the incumbent is retained and nothing changes.
  Measured, not assumed: triggering a real retrain and hammering `/predict`
  throughout showed 2.1% p95 degradation (§9) — FR-15 holds up under load,
  not just by design intent.
- `frontend/` — React, plain CSS custom properties (no Tailwind), hand-drawn
  SVG charts (no Recharts), a hand-rolled critically-damped spring
  integrator (no Framer Motion). **Rebuilt from scratch on 2026-09-17** to a
  design reference the user supplied directly (superseding an earlier
  Tailwind/Framer/Recharts version built the same day) — see `frontend/README.md`
  for the design rationale in full; the highlights:
  - IR-07's graceful degradation is real, not aspirational: `src/api.js`
    falls back to `src/mock.js` (visibly marked "Example data — not live
    results") until the backend answers, and again if it later stops
    answering. Verified both states render correctly by stopping the
    backend mid-session and reloading.
  - The equity chart's buy-and-hold **benchmark series is honestly absent
    from the live path** — the backend doesn't persist one (the CT
    orchestrator only replays the incumbent, never a baseline policy,
    during evaluation) — rather than faked client-side. Only the clearly-
    labelled mock data shows the full two-series comparison.
  - The "Cycle history" card is labelled "Training runs, last N days", not
    "Retrains this quarter" — `model_run` has no field distinguishing an
    autonomous `CTOrchestrator` retrain from a manual
    `scripts/train_agent.py` sweep, so the honest label is the one that
    matches what `MarketRepository.get_cycle_stats` actually counts.
  - This surfaced a real gap the backend needed anyway: **a rejected
    candidate was never being logged at all.** `CTOrchestrator._retrain_and_maybe_promote`
    only called `registry.log_run()` inside the "candidate won" branch — a
    candidate that lost the FR-17 acceptance gate left zero record of
    having been trained. Fixed: every candidate is now logged
    (`status="REJECTED"` when not promoted), which is what makes "how many
    candidates were held back" an answerable, queryable question instead
    of always reading zero. `MarketRepository.list_decisions()` was also
    extended to join through to `model_run` for `run_id` and both
    partition boundaries, so NFR-07's traceability chain (decision →
    version → run → training partition → eval partition) is something the
    decision-detail sheet can actually *show*, not just something the
    schema makes possible.
  - Verified by actually running it: backend seeded with a trained+promoted
    model, frontend driven with Playwright (`chromium-cli` wasn't available
    in this environment), screenshotted in both light and dark mode,
    decision-detail sheet opened and closed (including the fail-safe
    example row), "Force Check" clicked, and the mock fallback exercised by
    killing the backend mid-session — `console --errors` clean throughout
    every pass. The earlier version's Playwright run is what caught the
    `load_partition` empty-frame bug documented in §7; that fix carried
    over unchanged into this rebuild.
- NFR-01/02/04 established by measurement (§9); NFR-09 explicitly left
  unmeasured (would require the dashboard actually running for hours).

A note for Sprint 4 (from before this was built, still accurate): NFR-01
scopes the system to single-user local operation, and there is **no user
table** — a single optional bearer token (`SERVING.bearer_token`, off by
default) guards the endpoints that matter, which is what got built.

---

## 13. Known risks

- **Single-source data.** yfinance is an unofficial interface to a public
  endpoint with no delivery guarantee or SLA. DR-04 and DR-05 exist so a
  supply interruption degrades the dataset *visibly* rather than corrupting it
  silently — an agent trained on quietly corrupted data fails in a way the
  drift monitoring cannot distinguish from genuine market change.
- **Sample size.** Five tickers of daily data over ten years is ~2,500
  observations per ticker. PPO is sample-inefficient. Section 1.7 concedes
  this; do not let a good backtest obscure it.
- **Overfitting to a single regime.** Walk-forward validation across distinct
  regimes is the mitigation, not a single train/test split.
- **Seed sensitivity.** See §10.2.
- **Cross-session claims that outrun the code.** §3 records a concrete case:
  this file described Sprint 2 as complete, with a specific test count and
  specific bug-fix stories, before any of `src/rlops/environment.py` existed
  in this checkout. The cloud session that wrote it had no way to check. The
  mitigation is procedural, not technical: verify a claim about *this*
  checkout's state (file existence, test counts, "measured" numbers) by
  reading the checkout, before building on it or repeating it in Chapter 5.

---

## 14. Working across two sessions

This project is worked on from two places: a **cloud session** (thesis writing,
design decisions, diagram generation — no access to this machine) and a **local
Claude Code session** (runs the code, reaches the network, trains the agent).

They share no state. **This file is the handoff mechanism in both directions.**
When something worth keeping is discovered locally — the real yfinance column
shape, measured MPS timings, a threshold established by experiment — append it
here rather than leaving it in a transcript.

Because the cloud session cannot see the checkout, anything it writes here
about the *state of the code* (what's implemented, test counts, benchmark
numbers) is necessarily a claim it cannot verify — it can describe intent and
design decisions reliably, but not measured local state. §3 and §13 record the
one time that gap produced a wrong "Complete." The asymmetry only runs one
way: a local session can and should verify before propagating; treat anything
here that reads as a measurement (a count, a timing, a "verified" claim) as a
todo to re-check against the actual checkout, not as ground truth, until it
has been.
