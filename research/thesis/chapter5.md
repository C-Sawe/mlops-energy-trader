---
title: "Chapter 5 — Discussion, Conclusions and Recommendations"
subtitle: "An MLOps-Enabled Continuous Training Architecture for Reinforcement Learning in Global Energy Markets"
author: "Caleb Kipchirchir (169391, ICS 4C) — Supervisor: Mr. Allan Vikiru"
date: "Strathmore University, School of Computing and Engineering Sciences"
---

# 5.1 Introduction

This chapter discusses the results of the implementation described in Chapter 4,
evaluates the system against the objectives and requirements established in
Chapters 1 and 3, records and justifies every deviation from the approved
design, states the limitations of the work, and sets out recommendations for
further development.

The chapter follows the boundary established in Section 1.7. The contribution
under examination is **architectural**: the claim is that a closed-loop
Continuous Training (CT) pipeline can detect degradation in its own
risk-adjusted performance and recover from it autonomously, without pipeline
failure and without serving downtime. The contribution is *not* a profitable
trading strategy, and no result in this chapter should be read as one. Where
the evaluation produced a favourable figure, the correct interpretation is that
the architecture behaved as designed; Section 5.5 shows that the statistical
discipline applied to those figures in fact declines to certify any of them as
evidence of a genuine trading edge, which is itself the intended outcome of the
evaluation protocol rather than a shortfall of it.

# 5.2 Mathematical Formulation

This section consolidates the mathematics the implementation actually uses.
Each formulation below corresponds directly to code in the delivered system,
and the governing requirement identifier is cited alongside it.

## 5.2.1 Feature engineering

**Simple moving average (FR-03).** For closing price $P_t$ of a given ticker
and window $n = 20$:

$$\mathrm{SMA}_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

The value is undefined (retained as `NaN`) for $t < n$ rather than computed
from a partial window, so that a warm-up value is never mistaken for a
converged one.

**Relative Strength Index, Wilder's formulation (FR-03).** With
$\Delta_t = P_t - P_{t-1}$, gains $G_t = \max(\Delta_t, 0)$ and losses
$L_t = \max(-\Delta_t, 0)$, the averages are exponentially smoothed with
$\alpha = 1/n$, $n = 14$:

$$\bar{G}_t = \alpha G_t + (1-\alpha)\bar{G}_{t-1}, \qquad
  \bar{L}_t = \alpha L_t + (1-\alpha)\bar{L}_{t-1}$$

$$\mathrm{RS}_t = \frac{\bar{G}_t}{\bar{L}_t}, \qquad
  \mathrm{RSI}_t = 100 - \frac{100}{1 + \mathrm{RS}_t}$$

Wilder's exponential smoothing is used deliberately in place of the simple
rolling mean of gains and losses common in tutorial implementations; the two
diverge materially over approximately the first $3n$ periods. The degenerate
cases are defined explicitly: $\bar{L}_t = 0$ with $\bar{G}_t > 0$ yields
$\mathrm{RSI}_t = 100$ by definition, and $\bar{G}_t = \bar{L}_t = 0$ yields
$50$.

**Rolling Z-score standardisation (FR-04, DR-07).** For feature $x$ and
lookback $w = 60$:

$$z_t = \frac{x_t - \mu_{t-w+1:t}}{\sigma_{t-w+1:t}}, \qquad
  \sigma_{t-w+1:t} = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1}\left(x_{t-i} - \mu\right)^2}$$

The window is inclusive of $t$ and strictly backward-looking. This is the
central no-look-ahead guarantee (invariant I1): the standardised value at time
$t$ is a function of observations at or before $t$ only. A zero-variance window
yields $z_t = 0$ rather than a division by zero, which is the correct reading —
a constant series exhibits no deviation from its own mean.

## 5.2.2 The Markov Decision Process

**State.** For $k = 5$ tickers with 8 features each, plus $k$ position weights
and one cash weight, the observation is a vector in
$\mathbb{R}^{46}$ where $46 = 5 \times 8 + 5 + 1$.

**Action.** $a_t \in [-1, 1]^k$, a vector of continuous target portfolio
weights, clipped to the interval on entry to the transition function.

**Portfolio accounting.** With share holdings $s_t \in \mathbb{R}^k$, prices
$p_t \in \mathbb{R}^k$ and cash $c_t$:

$$E_t = c_t + s_t^{\top} p_t$$

**Transition timing (invariant I2).** The action selected from the observation
at the close of day $t$ is executed at day $t$'s closing prices, and only then
is the clock advanced; the resulting return is realised from $t$ to $t+1$:

$$r_t = \frac{E_{t+1} - E_t}{E_t}$$

This ordering is enforced in code and guarded by a regression test, because an
implementation that advances the index before rebalancing would realise each
action's return over a period the agent had already observed.

**Rebalancing under transaction costs (FR-07).** Let $\delta = s^{*} - s_t$ be
the share delta toward target shares $s^{*} = a_t E_t / p_t$. Legs are
partitioned into sells ($\delta_i < 0$) and buys ($\delta_i > 0$). Writing
$N^{-}$ for total sell notional and $N^{+}$ for total buy notional, with
proportional cost $\kappa = 0.001$, cash available after sells is

$$c_{\text{avail}} = c_t + N^{-}(1 - \kappa)$$

and cash remaining after executing a fraction $\beta \in [0,1]$ of the buy legs
is

$$f(\beta) = c_{\text{avail}} - \beta N^{+}(1 + \kappa)$$

The executed fraction is $\beta^{*} = 1$ if $f(1) \geq 0$, and otherwise the
root of $f(\beta) = 0$, located by bisection. Since $f$ is monotone decreasing
in $\beta$, the root is unique and bisection is valid.

Two properties of this formulation are load-bearing and were each established
by correcting a defect. First, **only the buy legs are scaled**. A sell's
proceeds exceed its own fee for any $\kappa < 1$, so sell legs are always
self-funding and execute in full. Applying a single scale factor to buys and
sells alike fails on a pure swap between two tickers, where the sell and buy
notionals are equal and opposite and therefore cancel for *any* scale factor,
leaving cash-after-trade negative at every nonzero $\beta$ and causing
bisection to converge on executing nothing at all. Second, **unaffordable
trades are scaled down, never rejected**; rejection would make the action space
discontinuous at the affordability boundary and supply the policy with an
uninformative gradient there.

A deadband suppresses trades whose notional falls below
$10^{-6} E_t$. Actions are `float32`, so expressing "hold the current
allocation" as a target weight carries round-trip error of order
$2\times10^{-7}$ of portfolio value; without the deadband every hold would
register as a trade, and a buy-and-hold baseline would appear to rebalance
daily.

**Reward (FR-07, invariant I4).** With running peak equity
$\hat{E}_t = \max_{\tau \le t} E_\tau$, instantaneous drawdown
$d_t = (\hat{E}_t - E_t)/\hat{E}_t$, and running maximum drawdown
$D_t = \max_{\tau \le t} d_\tau$, the reward is

$$R_t = r_t - \lambda \cdot \max\left(0,\; d_t - D_{t-1}\right)$$

with penalty coefficient $\lambda = 1.0$.

The penalty applies to the **increment** in maximum drawdown, not its level.
Penalising the level would charge the agent repeatedly, on every subsequent
step, for a single historical loss — making the cumulative reward a function of
episode length rather than of behaviour. Under the formulation above, a step
that holds at the same drawdown or recovers from it incurs no penalty at all.

## 5.2.3 Policy optimisation

The agent is Proximal Policy Optimisation (Schulman et al., 2017) as
implemented in Stable-Baselines3. The clipped surrogate objective is

$$L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[
  \min\left( \rho_t(\theta)\hat{A}_t,\;
  \mathrm{clip}\left(\rho_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t \right)
\right]$$

where $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$
is the probability ratio and $\hat{A}_t$ is the advantage estimate, computed by
Generalised Advantage Estimation:

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda_{\mathrm{GAE}})^{l}\,\delta_{t+l},
\qquad \delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$$

Configured hyperparameters: learning rate $3\times10^{-4}$, discount
$\gamma = 0.99$, rollout length $n_{\text{steps}} = 2048$, on `device="cpu"`
for the reason established empirically in Section 5.3.2.

## 5.2.4 Evaluation metrics

**Sharpe ratio (FR-13).** For a per-period return series $\{r_t\}_{t=1}^{T}$
annualised over $A = 252$ trading days:

$$\mathrm{SR} = \frac{\bar{r}}{\sqrt{\max\left(\hat{\sigma}^2,\, \varepsilon\right)}}\sqrt{A},
\qquad \varepsilon = 10^{-12}$$

The variance floor $\varepsilon$ is not cosmetic. A constant return series has
a computed standard deviation of order $10^{-18}$ arising from floating-point
rounding, not exactly zero. Without the floor, that noise denominator produces
a Sharpe ratio of order $10^{16}$ — enormous but finite — which would clear any
sensible target threshold and thereby **silently suppress the FR-14 retraining
trigger at precisely the moment the strategy had stopped doing anything.** The
floor converts a broken trigger into a merely large number; Section 5.6
discusses a case where such a number was in fact observed in production
telemetry.

**Rolling Sharpe (FR-13).** The same quantity over a trailing window of
$w = 30$ observations. Warm-up rows with fewer than $w$ observations are held
as `NaN` rather than reported as zero, since a zero would be indistinguishable
from a genuinely poor but fully observed window and would spuriously fire
FR-14 before sufficient history existed to justify it.

**Maximum drawdown (FR-18).**

$$\mathrm{MDD} = \max_t \frac{\max_{\tau \le t} E_\tau - E_t}{\max_{\tau \le t} E_\tau}$$

**Deflated Sharpe Ratio (Bailey & López de Prado, 2014).** Given the observed
Sharpe $\widehat{\mathrm{SR}}$ over $T$ observations, with sample skewness
$\hat{\gamma}_3$ and kurtosis $\hat{\gamma}_4$, and $N$ independent
configurations tried, the expected maximum Sharpe under the null of no skill is

$$\mathrm{SR}_0 = \sqrt{\hat{V}}\left[(1-\varphi)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
  + \varphi\,\Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right)\right]$$

where $\varphi \approx 0.5772$ is the Euler–Mascheroni constant and $\hat{V}$
is the estimated variance of the Sharpe estimator. The deflated ratio is then

$$\mathrm{DSR} = \Phi\!\left(
  \frac{\left(\widehat{\mathrm{SR}} - \mathrm{SR}_0\right)\sqrt{T-1}}
       {\sqrt{1 - \hat{\gamma}_3\widehat{\mathrm{SR}} + \tfrac{\hat{\gamma}_4 - 1}{4}\widehat{\mathrm{SR}}^2}}
\right)$$

This is the single most important formula in the evaluation. The best of $N$
backtests exhibits a positive Sharpe ratio *by construction*; the DSR quantifies
how much of an observed figure that selection effect alone explains. Its
application in Section 5.5 is what prevents this project from reporting a
favourable backtest as a result.

**Discrete action mapping (FR-11).** The continuous weight $w$ produced by the
policy is mapped to a discrete action at the configured thresholds:

$$
\mathcal{A}(w) =
\begin{cases}
\text{BUY} & w > 0.5\\
\text{SELL} & w < -0.5\\
\text{HOLD} & \text{otherwise}
\end{cases}
$$

with `LIQUIDATE` returned for every ticker, bypassing the model entirely, when
the fail-safe condition of Section 5.4.3 holds.

# 5.3 Summary of Findings

## 5.3.1 Objective ii — Automated data pipeline

The DataOps layer ingests daily OHLCV for the five-equity energy universe
(XOM, CVX, SHEL, BP, NEE) plus the VIX volatility index, adjusts for corporate
actions, forward-fills gaps with an explicit `is_imputed` flag, computes the
specified indicators, and persists to PostgreSQL under a composite `(date,
ticker)` key.

Live ingestion was validated against real yfinance responses rather than
fixtures. Over the 2024-01-01 to 2024-06-01 window the universe produced zero
imputed rows — all five tickers traded on an identical calendar — and exactly
230 of 525 rows carried a complete normalised feature vector, matching the
analytically predicted $5 \times (105 - 59) = 230$ from the 60-day
standardisation warm-up. The arithmetic was confirmed against real data, not
merely against synthetic fixtures.

Two properties that unit tests on SQLite explicitly cannot verify were
confirmed against a real PostgreSQL container: **idempotency** (re-running an
identical ingestion left the store at 525 rows rather than 1,050, confirming
DR-05) and **both CHECK constraints firing** — an attempt to insert a model run
with `eval_start <= train_end` raised `IntegrityError` on `ck_eval_after_train`,
and an observation with `high_price < low_price` raised `IntegrityError` on
`ck_high_ge_low`. The database therefore enforces invariant I1 structurally,
not only through application code that might be bypassed.

The full historical backfill comprises **13,825 rows spanning 2015-01-02 to
2025-12-30** across all five tickers plus VIX.

## 5.3.2 Objective iii — Training and model registry

Hardware selection was settled empirically rather than assumed. Identical PPO
training runs against the real 46-dimensional observation space, with an
untimed warm-up preceding each measurement, produced:

| Device | Seed 0 | Seed 1 |
|---|---|---|
| CPU | 5,677 steps/sec | 5,718 steps/sec |
| MPS (Apple GPU) | 452.6 steps/sec | 438.5 steps/sec |

**The CPU outperforms the GPU by a factor of roughly 12–13.** The policy
network contains only **14,731 parameters (57.5 KB)**; PPO alternates rollout
collection — thousands of forward passes on a single observation — with small
minibatch updates, and at this parameter count the host-to-device transfer
overhead dominates any arithmetic advantage the GPU could offer. Training is
comfortably CPU-viable: 500,000 timesteps completed in **87.0 seconds**. This
finding is worth recording because the assumption that reinforcement learning
requires GPU acceleration is widespread and, at this problem scale, incorrect.

## 5.3.3 Objective iv — Serving and the Continuous Training loop

The serving layer exposes inference, telemetry, a paginated decision log and CT
pipeline status, with the volatility fail-safe positioned in the serving layer
rather than inside the agent (invariant I5) so that it survives model promotion
— a fail-safe residing in a component that promotion replaces would be
defeated by the very mechanism it exists to guard against.

The system's central claim was observed operating end to end, twice, against
real ingested data:

**First occurrence (bootstrap and decay detection).** With no incumbent active,
the orchestrator trained and promoted unconditionally. A subsequent evaluation
replayed that incumbent over real market data and computed a rolling Sharpe
ratio of $-3.79$, far below the target of $1.0$, and autonomously triggered a
background retrain through the identical code path the scheduler uses. The
resulting candidate failed to beat the incumbent out-of-sample and was
therefore rejected rather than promoted, leaving the incumbent serving.

**Second occurrence (steady state).** After three days running unattended, the
incumbent was evaluated against a historical window identified from recorded
telemetry as weak for it, producing a rolling Sharpe of $-2.43$. This again
triggered a genuine retrain; the candidate again failed the acceptance gate and
was rejected, and the incumbent remained active. This case is the more
representative of the two, since it begins from an already-serving incumbent
rather than from an empty system.

**Unattended operation.** Over three days with no human attention, the
scheduler performed **540 evaluations, promoting 2 candidates and rejecting
448**. That ratio is the substantive result: the FR-17 acceptance gate is
demonstrably difficult to satisfy, which is what prevents the closed loop from
becoming a mechanism for compounding error by promoting whichever candidate
happened to finish training.

# 5.4 Requirements Achievement

## 5.4.1 Functional requirements

All twenty functional requirements are implemented. Of particular note:

**FR-17 is an addition to the original Blueprint,** and is recorded here as a
design change rather than presented as part of the approved specification. As
originally specified, the CT loop trained and reloaded unconditionally. That
design permits the system to autonomously promote a *worse* model purely
because a training cycle completed. Making promotion conditional on
out-of-sample acceptance converts the closed loop from a mechanism that can
compound error into one that can only improve or hold steady. The 448
rejections recorded in Section 5.3.3 are the evidence that this gate is
load-bearing rather than decorative.

**FR-12 and the retraining trigger.** The volatility fail-safe originally
gated inference only. A gap was identified during review: the retraining
trigger of FR-14 had no volatility awareness whatsoever, and since a volatile
period is precisely when a 30-day rolling Sharpe is most likely to fall below
target, the system as designed would have tended to retrain *more* during
market stress — training a fresh candidate on a distorted window. The
orchestrator now defers, rather than cancels, a retraining cycle when VIX is at
or above the critical threshold; the next scheduled evaluation re-checks both
conditions from scratch. The same threshold is reused deliberately rather than
introducing a second tunable constant, so that the thesis defends one number
rather than two. The deferral is scoped to the decay-detection path only and
not to the bootstrap path, since deferring an initial deployment indefinitely
because ingestion began during a turbulent week would leave nothing serving at
all — a worse outcome than training once on a noisy window that the acceptance
gate would likely reject in any case.

## 5.4.2 Non-functional requirements

Thresholds were deliberately left unset in the approved design until they could
be measured on the target hardware, in accordance with the departmental
requirement that thresholds not be invented to make a requirement appear
measurable.

| ID | Threshold | Measured | Status |
|---|---|---|---|
| NFR-01 | ≥95% within 20 ms over 100 requests | p95 = 16.9 ms, p99 = 77.1 ms, max = 78.3 ms | Met |
| NFR-02 | ≤10% latency degradation during retraining | 2.1% p95 degradation | Met |
| NFR-03 | Zero failed requests across a full CT cycle | 1,331 requests, 0 failures | Met |
| NFR-04 | Recover to serving within 1 minute | 497.9 ms mean, 523.2 ms max (N = 5) | Met |
| NFR-05 | Failed retraining leaves incumbent serving | Verified by crash-injection test | Met |
| NFR-06 | No circular dependencies, static analysis | 2 `import-linter` contracts pass | Met |
| NFR-07 | Every served model fully traceable | Decision → version → run → partition | Met |
| NFR-08 | Payloads schema-validated | Pydantic; malformed input returns 422 | Met |
| NFR-09 | Dashboard stable over [X] hours | 31 min sampled; growth reclaimed by GC, no unbounded trend | Met |
| NFR-10 | No credentials in source or logs | Enforced; `__repr__` masks password | Met |
| NFR-11 | Fail-safe precedence, logged with trigger | Checked before any model call | Met |

Three observations on this table deserve emphasis.

**The NFR-01 threshold was set above the measured p95, not at the median.** The
p99 tail of 77.1 ms is roughly 4.5 times the median and is the figure worth
watching if the service is ever re-measured against PostgreSQL rather than
SQLite.

**The NFR-04 figure required a correction, and the correction is itself worth
reporting.** An initial measurement of 8 ms mean and 11 ms max was found, on
review, to have been taken from a process in which a model registry had
already been constructed before `InferenceService` was timed — which, as a
side effect, initialises the MLflow tracking store's connection and runs its
schema check. Re-measurement from five genuinely separate processes, each
constructing `InferenceService` as the first and only thing to touch MLflow —
matching what an actual unplanned termination and restart looks like — produced
497.9 ms mean and 523.2 ms max, approximately sixty times slower. The
discrepancy was confirmed directly: a second reconstruction within an
already-primed process completes in roughly 10 ms, isolating the one-time
initialisation cost as the cause. The corrected figure still clears the
one-minute threshold with substantial margin (0.83% of the budget, rather than
the 0.013% the original figure implied), so the requirement was never at
risk; the number originally reported for it was, however, wrong, and is
corrected here rather than left standing. This is recorded as a finding in
its own right: a benchmark taken in a warm process can misstate a cold-start
requirement by nearly two orders of magnitude without any single measurement
being dishonest.

**NFR-03 was counted, not inferred.** A reliability script issued continuous
requests throughout a genuine retraining cycle and explicitly counted failures
rather than concluding "probably zero" from a latency chart exhibiting no
visible gaps. Latency under that specific load was higher than NFR-01's
steady-state figure (p50 22.4 ms, p95 25.9 ms, max 92.8 ms), which is
consistent with contention between the request-handling thread and the
CPU-bound training thread in a single-process deployment — a different
measurement condition, not a regression.

**NFR-09 was measured, and the first measurement attempt was itself
misleading — which is the more instructive part of the finding.** A real
Chromium tab loaded the live dashboard and ran normally for 31 minutes while
Chrome DevTools Protocol sampled its event-listener count and JavaScript heap
size once per minute. Taken at face value the trend was alarming: a listener
count that climbed from 185 to over 2,000 and a heap that grew from 3.9 MB to
6.2 MB, both without pause. Reported as such, this would have recorded a
failure of a requirement that, on closer inspection, the system satisfies.

Four checks, each designed to falsify rather than confirm the apparent leak,
were applied before any conclusion was drawn. Directly intercepting every
`addEventListener`/`removeEventListener` call showed no net growth over an
equivalent window. Chrome DevTools' own `getEventListeners()` utility, which
queries the browser's internal listener table directly rather than through a
JavaScript-level proxy for it, reported an identical count before and after
the same interval. With every request to the backend intercepted and
discarded — so polling continued but no genuinely new data ever arrived — the
previously climbing count held perfectly flat. Repeatedly forcing garbage
collection and sampling immediately afterward showed the post-collection heap
baseline plateau after an initial rise, rather than continuing to climb across
successive cycles.

The decisive evidence arrived unprompted. At the thirty-three-minute mark of
the original run, with no garbage collection ever forced on that process,
Chromium's own collector fired on its own: the listener count fell from 2,025
to 185 and heap usage from 6.2 MB to 4.3 MB in a single sampling interval,
both values returning to the baseline every other check had already
established. This is the complete grow-then-reclaim cycle the requirement is
actually concerned with, observed naturally rather than staged, and it is
bounded rather than open-ended.

**NFR-09 is therefore met.** The dashboard does not exhibit unbounded memory
growth. The behaviour actually observed is ordinary garbage accumulation
between collection passes, which any sufficiently active web page exhibits,
and the raw Chrome DevTools listener metric tracks something closer to churn
since the last collection than to currently active listeners — making it
misleading as a standalone signal for a dashboard that redraws its charts on
every poll. The run was not extended to the originally scoped literal span of
hours: once a complete natural grow-and-reclaim cycle had been observed and
corroborated by all four independent checks, a longer run would have shown
the same cycle repeat without adding information. The methodological finding
is arguably the more citable result of the two: a single browser memory
metric, trusted without cross-checking, would have produced a false and
dramatic-looking failure of a requirement the system in fact satisfies.

## 5.4.3 Verification summary

The delivered system carries **131 backend tests** and **47 frontend tests**.
All tests use deterministic synthetic data and in-memory or temporary-file
SQLite; none requires network access or a running database. Several tests were
confirmed to catch the defect they guard against by reverting the corresponding
fix and observing the test fail — a regression test that has never been seen to
fail is an assumption, not a verification.

# 5.5 Evaluation Results

The evaluation protocol required four steps before reporting any agent result:
run the baselines, sweep multiple seeds, apply the deflated Sharpe ratio with
the true number of configurations tried, and retain transaction costs. All four
were performed.

## 5.5.1 Walk-forward validation and seed sensitivity

The full sweep comprised **18 walk-forward splits × 10 seeds = 180 training
runs** against the real 2015–2025 data, completing in 671 seconds (~11.2
minutes) of CPU time.

| Statistic | 5 seeds | 10 seeds |
|---|---|---|
| Best-seed-per-split Sharpe (n = 18) | mean 0.570, std 1.279 | mean 1.173, std 0.914 |
| All seeds pooled | mean −0.606, std 1.507 (n = 90) | mean −0.723, std 1.583 (n = 180) |
| Positive splits | 10 / 18 | 17 / 18 |
| Deflated Sharpe per split | 0.388 (one fold) | mean 0.055, std 0.073, **max 0.301** |
| Splits with DSR > 0.5 | — | **0 / 18** |

**These rows must be read in the correct direction.** Doubling the seed count
raised the best-per-split mean from 0.570 to 1.173 and tightened its standard
deviation. That movement is not the strategy improving. "Best of ten" is a
maximum order statistic, which rises and tightens mechanically as more samples
are drawn, regardless of whether any individual seed is any good. The pooled
row — what an untuned single run actually delivers — barely moved between 90
and 180 observations, and was marginally *worse* at the larger sample. That is
the honest population estimate.

## 5.5.2 Deflated Sharpe ratio

The single best result across all 180 runs was split 11 (evaluation window
2021-12-05 to 2022-06-02), seed 0, with a Sharpe ratio of **3.273**. Treated in
isolation, its Probabilistic Sharpe Ratio is 0.986 — a 98.6% probability that
the figure is genuinely positive. Corrected for the 180 configurations
actually tried, its **deflated Sharpe ratio is 0.301**. It had previously been
computed as 0.388 against a trial count of 90; honestly increasing the trial
count *lowered* it.

Across all eighteen splits, the deflated Sharpe ratio of every split's best
result is below 0.31, and **none clears 0.5**.

Baselines over split 11's identical evaluation window place this result in its
proper context: buy-and-hold returned a Sharpe of **2.419** and equal-weight
rebalancing **2.388** over the same period. December 2021 to June 2022 was
simply a strong window for energy equities. The agent's margin over a passive
baseline is real but modest, and far smaller than the standalone figure of
3.273 suggests.

**The defensible conclusion is therefore the following.** The disciplined
evaluation pipeline — walk-forward validation, seed sweep, deflation, and
baseline comparison — executed end to end on real market data and correctly
declined to certify any fold as evidence of a genuine edge. That is the
architecture responding as designed. No result in this work supports the
statement that the strategy is profitable, and the analysis above exists
specifically to prevent such a statement from being made.

Determinism was confirmed as a side effect: split 11, seed 0 reproduced a
Sharpe ratio of 3.2727 to five decimal places across two independent runs
three days apart.

## 5.5.3 Sensitivity analysis

Configurable thresholds were swept against real data so that the thesis reports
a sensitivity analysis rather than defending magic numbers.

**Volatility threshold (FR-12),** swept against all 2,766 real VIX
observations from 2015-01-02 to 2025-12-31:

| Threshold | Days at or above | % of history |
|---|---|---|
| 20.0 | 826 | 29.86% |
| 25.0 | 377 | 13.63% |
| 30.0 | 156 | 5.64% |
| **35.0 (default)** | **63** | **2.28%** |
| 40.0 | 40 | 1.45% |
| 50.0 | 19 | 0.69% |

The default engages on 2.28% of real trading days — infrequent enough to
constitute a genuine circuit breaker rather than a condition the system
inhabits. The episodes falling within that band correspond to recognised
market-stress events: the 2015–16 correction, the fourth quarter of 2018, and
the 2020 COVID crash. A materially lower threshold would place the fail-safe in
effect between 6% and 14% of the time, which is a real trade-off between missed
opportunity and caution rather than a free gain in safety.

**Retraining trigger (FR-14),** evaluated against the 180-run sweep: at the
default threshold of 1.0, **85.6% of individual seeds fail to clear the bar.**
This explains an observation already made in production rather than merely
predicting one — it is precisely why the three-day unattended run produced 540
evaluations with only 2 promotions. The threshold is not miscalibrated; it
correctly reflects the seed sensitivity measured in Section 5.5.1.

**Transaction costs (FR-07),** applied to the equal-weight-rebalanced baseline,
the only baseline that trades at every step and is therefore genuinely exposed
to cost drag:

| Cost | Sharpe | Cumulative return |
|---|---|---|
| 0 bps | 0.629 | 15.56% |
| **10 bps (default)** | **0.616** | **15.16%** |
| 20 bps | 0.603 | 14.75% |
| 100 bps | 0.501 | 11.62% |

The degradation is modest, monotonic and real. The default assumption costs
approximately 2% of the frictionless Sharpe, and even a tenfold harsher
assumption does not collapse the baseline's edge — which is itself informative,
indicating that the instruction to retain transaction costs guards against a
genuine effect rather than defending a fragile result that survives only at
exactly 10 basis points.

# 5.6 Deviations from the Approved Design

The departmental guide requires deviations from the approved design to be
recorded and justified. Five are recorded.

**1. FinRL was evaluated and not adopted as the primary environment.** The
proposal names FinRL, and it does function correctly. It was rejected as the
primary environment for two substantive reasons: its reward is assigned inline
with no override hook, so the FR-07 drawdown penalty could only be attached by
post-processing rather than expressed within the reward itself; and its actions
are `hmax`-scaled share counts rather than the continuous target weights this
specification requires.

FinRL was instead used as an **independent cross-check** on the custom
environment's valuation and return accounting. Both environments held an
identical fixed position across an aligned 222-day window with costs and
penalties disabled, reducing each to plain profit-and-loss over the same
holding. Final equity agreed to **$0.00007237 on a portfolio of approximately
$102,000** — a maximum relative difference of about $7 \times 10^{-10}$, with a
correlation of 1.000000000000. The residual is consistent with `float32` action
rounding, not an accounting divergence. The deviation is therefore supported by
evidence rather than by design argument alone.

**2. Alpaca paper-trading integration.** This is not in the approved technology
stack and is recorded as a deliberate addition, scoped narrowly. Paper trading
— simulated capital, real market structure, real order API — is a safe
architectural demonstration that remains within the Section 1.7 boundary. Live
trading with real capital was explicitly ruled out rather than left ambiguous.

That boundary is enforced in code, not merely in documentation: the broker base
URL is a **fixed constant** pointing at the paper endpoint, not an environment
variable, so no configuration error can silently redirect the system to live
trading. The integration was exercised against a real paper account: a $10
notional order on XOM filled at $162.876 per share for 0.061335003 fractional
shares, and the position was subsequently closed to flat via the
close-position endpoint.

**3. FR-17, the out-of-sample acceptance gate,** as discussed in Section 5.4.1.

**4. The autonomous ingestion scheduler.** FR-01 requires ingestion "without
manual intervention". Review established that this was a claim the code
comments made but the code did not keep: the only autonomous background task
was the CT evaluation scheduler, which re-evaluates data already present and
never fetches anything new. Every ingestion to that point had been a human
running a script. A scheduler was added to close the gap between the stated
requirement and the implemented behaviour.

**5. The volatility deferral on retraining,** as discussed in Section 5.4.1.

# 5.7 Critique and Limitations

**The Sim2Real gap remains, as conceded in Section 1.7.** The simulation cannot
reproduce real transaction costs, slippage or liquidity constraints, and the
agent trains on daily OHLCV rather than tick data. No claim in this work
extends beyond that boundary.

**Sample size.** Five tickers of daily data over ten years yields approximately
2,500 observations per ticker. PPO is sample-inefficient, and no favourable
backtest should be permitted to obscure this.

**Seed sensitivity is severe.** Section 5.5.1 quantifies it: the gap between
best-of-ten (mean 1.173) and the pooled population (mean −0.723) is the
selection effect made visible in the project's own numbers.

**Single-source data.** yfinance is an unofficial interface to a public
endpoint carrying no delivery guarantee. The forward-fill flag and the
uniqueness constraint exist so that a supply interruption degrades the dataset
*visibly* rather than corrupting it silently — an agent trained on quietly
corrupted data fails in a manner the drift monitoring cannot distinguish from
genuine market change.

**Hyperparameters were not tuned.** The evaluation varied the random seed only,
holding network size, learning rate and rollout length fixed throughout. The
work therefore establishes how variable a single configuration is, not whether
a better configuration exists.

**A latent data-integrity defect was found and corrected during this work, and
is reported rather than omitted.** The ingestion scheduler was silently
corrupting a moving window of indicator history on every scheduled tick. The
mechanism is instructive: indicator computation produces `NaN` for the first
$n-1$ rows of *any individual computation call* by construction, and because
the scheduled fetch window slides forward with the calendar, each tick's
warm-up zone landed on a different stretch of dates — and because persistence
is idempotent by upsert, each tick permanently overwrote correct values that an
earlier, better-positioned computation had produced. The defect was discovered
only because a forced evaluation against an affected window raised an error.
The correction drops each ticker's warm-up rows before persistence, and is
guarded by a test confirmed to fail against the unfixed code.

This defect is worth recording in a thesis rather than quietly repaired,
because it is exactly the failure mode the data-integrity requirements were
written to prevent and it evaded them for a period. It is also a concrete
instance of the general risk that a pipeline can corrupt its own history in a
way that produces plausible-looking but unreproducible results.

# 5.8 Conclusions

The objectives set out in Chapter 1 were met. An automated data pipeline
ingests, cleans, engineers features from and persists real market data; a PPO
agent trains against a Gym-compatible MDP with a drawdown-penalised reward and
is versioned with full traceability to its training partition and
hyperparameters; and a serving layer exposes inference behind a volatility
fail-safe while a Continuous Training loop monitors risk-adjusted performance,
detects decay, retrains autonomously and promotes only on out-of-sample
acceptance.

The central architectural claim was demonstrated rather than asserted. The
closed loop was observed detecting its own performance decay and responding
autonomously on two separate occasions against real data, and ran unattended
for three days performing 540 evaluations without pipeline failure or serving
downtime. Measured non-functional results support the claim quantitatively:
zero failed requests across 1,331 issued during a genuine retraining cycle, and
2.1% p95 latency degradation while retraining proceeded.

**The trading results do not support any claim of profitability, and this is
reported as the correct outcome rather than as a shortfall.** Not one of the
eighteen walk-forward folds produced a deflated Sharpe ratio clearing 0.5, and
the single strongest fold's figure fell from 0.388 to 0.301 once the trial
count was honestly counted. An evaluation protocol that declines to certify its
own best result is functioning precisely as intended.

# 5.9 Recommendations for Further Work

**Tune hyperparameters, not only seeds.** The present work establishes the
variance of one configuration. Annealing network size, learning rate and
rollout length would establish whether a materially better configuration
exists, and would require the deflated Sharpe ratio's trial count to be
increased accordingly — which, on the evidence of Section 5.5.2, would lower
the resulting figures further.

**Broaden the data.** Intraday data would narrow the Sim2Real gap; a second
independent data source would mitigate the single-source risk.

**Portfolio management and brokerage** are discussed separately in Section 5.10
because they carry regulatory obligations that place them outside the scope of
this thesis.

# 5.10 A Note on Production Deployment

The question of whether this architecture could operate as a live service
managing client portfolios arises naturally from the work and is addressed here
explicitly, since the answer bears on the thesis boundary.

**Technically**, the architecture already supports the majority of what such a
service requires. The serving layer, model registry, traceability chain,
fail-safe and continuous-training loop are substantially the same components a
production system would need. The principal engineering additions would be
multi-account state, per-account position reconciliation against broker
records, an audit log at the account level, and authentication and
authorisation considerably stronger than the single-token scheme appropriate to
a single-user research deployment.

**Legally, this is a regulated activity and the engineering is not the binding
constraint.** Managing investments on behalf of third parties, and operating as
a broker, require licensing by the Capital Markets Authority in Kenya, with
comparable regimes in other jurisdictions. Requirements typically extend to
capital adequacy, fit-and-proper assessment of principals, client-money
segregation, mandatory reporting and compliance officers. This is not a matter
that software architecture addresses. It is stated here in general terms and
does not constitute legal advice; the Capital Markets Authority is the
authoritative source, and qualified counsel would be required before any such
service accepted a single external client.

**Academically**, such an extension would exceed the approved scope of this
thesis and would require the supervisor's approval, since it alters the
research boundary rather than the implementation. It would also undermine the
framing that Chapter 1 establishes and this chapter has maintained throughout:
the Section 5.5 evidence explicitly declines to certify a trading edge, and
deploying capital — whether the researcher's own or a client's — on a strategy
whose own evaluation protocol refuses to certify it would contradict the
discipline the work exists to demonstrate.

The defensible path forward, in order, is therefore: extend the paper-trading
integration to a multi-account simulation demonstrating portfolio management
mechanics without regulatory exposure; establish a live track record on
simulated capital over a meaningful period; and treat licensing as a
prerequisite to, not a consequence of, any decision to accept external funds.
