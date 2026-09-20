# Thesis handoff — citations, front matter, Figure 4.8, and Chapter 5

**Purpose of this file.** Everything below is written for a *thesis-writing*
session (prose, citations, figures, Word mechanics), not a coding session.
It exists because CLAUDE.md is a dev log — accurate, but organized by
engineering session, not by the five thesis tasks this file covers. This
file pulls every number, finding, and framing rule those tasks need into one
place, cross-referenced back to CLAUDE.md so nothing here has to be taken on
faith.

**Read `CLAUDE.md` §1 before anything else in this file.** Every piece of
content below is written to respect its rule: **the contribution is the
architecture, not the alpha.** A favourable number is never the result.
Section 5's own subsection ("Framing rules") restates this with a concrete
checklist of sentences to avoid — read it before drafting Chapter 5 prose,
not after.

---

## 1. Section 2.3.1 — citation mismatch (unresolved, needs literature search)

**The problem, precisely:** Section 2.3.1 describes a study using an ARIMA +
Attention-LSTM ensemble, but cites Kreuzberger et al. (2023) — which is an
MLOps *survey* paper, not a forecasting-methods paper. This mismatch was
flagged independently by the marker (Ms. Salome Chemiat, comment #2) on the
approved proposal.

**What this needs:** the *actual* ARIMA + Attention-LSTM ensemble paper the
section was originally describing, which was never correctly cited. This is
not something derivable from the codebase — it requires a literature search.
Starting points:
- Search terms to try: `"ARIMA" "attention" "LSTM" ensemble stock forecasting`,
  `"ARIMA-LSTM" hybrid energy price forecasting`, `attention-based LSTM
  financial time series ensemble`.
- Check whether the surrounding paragraph (the sentences immediately before
  and after the citation) contains any other clue — a dataset name, an
  author's institution, a stated result (e.g. a specific RMSE or accuracy
  figure) that could be used to search for the exact paper via Google
  Scholar's "cited by" or a direct phrase search.
- If genuinely irrecoverable, the fallback is finding *a* legitimate,
  citable ARIMA + Attention-LSTM ensemble paper that supports whatever claim
  Section 2.3.1 is making, and citing that instead — but only as a last
  resort, and flag to the supervisor that the original source could not be
  recovered.

**Do not** substitute Kreuzberger et al. (2023) with another MLOps paper —
the mismatch is that an MLOps survey is cited where a forecasting-methods
paper belongs, not that the wrong MLOps paper is cited.

---

## 2. Section 2.3.2 — citation (resolvable now)

**The problem:** Section 2.3.2 cites "Senneset, M. & Gultvedt, E. (2020)",
which does not appear anywhere in the reference list. Espiga-Fernández et
al. (2024), which *is* in the bibliography, matches the section's own
heading word-for-word.

**The fix:** this reads as a mid-draft source swap that was never
propagated — replace the in-text citation "Senneset, M. & Gultvedt, E.
(2020)" with "Espiga-Fernández et al. (2024)" throughout Section 2.3.2
(check for more than one occurrence — a citation is often repeated within a
section). Confirm the replacement reads grammatically (e.g. "(Espiga-
Fernández et al., 2024)" vs. "Espiga-Fernández et al. (2024) found that...")
depending on how it's used in each sentence.

**Before finalizing:** re-read the paragraph the citation sits in and check
Espiga-Fernández et al. (2024)'s actual claims (title/abstract) genuinely
support what that paragraph asserts — the heading match is strong evidence
this is the right source, but the sentence-level claim should still agree
with it.

---

## 3. Front matter — Table of Contents, List of Figures, List of Tables

**The problem:** none of the three contain Chapter 4 entries. This is
scored directly by the marking rubric.

**The fix (mechanical, in Word):**
1. Click into the Table of Contents → References tab (or right-click →
   "Update Field") → **Update entire table** (not "update page numbers
   only" — the entries themselves are missing, not just mis-numbered).
2. Repeat for the List of Figures.
3. Repeat for the List of Tables.
4. Scroll through all three afterward and confirm every Chapter 4 figure,
   table, and heading now appears with a correct page number. If any
   Chapter 4 figure/table is missing a caption (Word only auto-includes
   captioned figures/tables), add the caption via Word's caption tool
   (right-click the figure → Insert Caption) rather than typing plain text
   above it, or it will not appear even after F9.

---

## 4. Figure 4.8 wireframe — three corrections

**The problem, exactly as flagged:**
1. Shows tech tickers (AAPL, MSFT, NVDA, …) instead of the approved energy
   equity universe.
2. Action values shown contradict FR-11's actual mapping.
3. The retraining threshold line is drawn at ~0.2 instead of the real value.

**The three corrections, with the exact real values to use:**

1. **Tickers.** Replace AAPL/MSFT/NVDA/etc. with the actual five: **XOM,
   CVX, SHEL, BP, NEE** (CLAUDE.md §2 — this universe is locked and not
   something to change while fixing the figure). `^VIX` is a separate
   market-wide feature, not a sixth "ticker" row.

2. **Action values (FR-11).** The real mapping, from `src/config.py`
   (`RiskConfig`) and `InferenceService._discretize()`:
   - continuous weight **> 0.5** → **BUY**
   - continuous weight **< −0.5** → **SELL**
   - otherwise → **HOLD**
   - **LIQUIDATE** is a separate, fourth action — it only appears when the
     VIX fail-safe (I5/FR-12) has fired, overriding the model's weight
     entirely. It is not one of the three weight-threshold outcomes above,
     and should be shown as a distinct path in the wireframe if it's shown
     at all (e.g. a branch off "VIX critical?" rather than a fourth
     weight-range bucket).
   - Whatever numeric example weights the wireframe currently shows
     (presumably illustrating a sample decision), make sure the shown
     weight-to-action pairing is consistent with the ±0.5 boundary above —
     e.g. a weight of 0.72 → BUY is correct; a weight of 0.3 → BUY would
     not be (0.3 is inside the HOLD band).

3. **Retraining threshold line.** Should be drawn at **1.0**
   (`target_sharpe_threshold`, FR-14), not ~0.2. This is the rolling
   30-day Sharpe Ratio threshold below which the CT loop autonomously
   triggers a retrain. If the figure shows a sample Sharpe trajectory
   crossing the line, an illustrative crossing anywhere below 1.0 (e.g. a
   dip from 1.4 down to 0.6) is representative of a real observed pattern —
   see §5.5 below for a real trajectory you could base the illustration on
   if it needs a realistic-looking curve rather than a stylised one.

---

## 5. Chapter 5 — content package

Chapter 5 is scaffolded but not filled in. Below is (a) the non-negotiable
framing rule, (b) a suggested section structure to map onto whatever the
existing scaffold headings actually are, and (c) the real content —
numbers, tables, and ready-to-adapt prose — for each section. Every number
below is either measured on real market data pulled from a real, cited
source (yfinance) into a real PostgreSQL store, or clearly labelled as
synthetic-data-only where that's what it is. Nothing here is invented to
make a section look complete.

### 5.0 Framing rules — read before drafting anything

CLAUDE.md §1: **the contribution is the architecture, not the alpha.**
Section 1.7 of the approved proposal already concedes the Sim2Real gap
(no real transaction costs/slippage/liquidity beyond the modelled 10bps,
daily OHLCV not tick data) — every claim in Chapter 5 must stay inside
that boundary.

**Sentences to never write**, regardless of how good a number looks:
- ❌ "The agent is profitable" / "the strategy made X% return"
- ❌ "The agent outperforms the market" (a single favourable fold beating
  buy-and-hold is not evidence the *strategy* is good — see §5.4 below,
  where the best fold's edge over buy-and-hold shrinks once the regime is
  accounted for)
- ❌ "The deflated Sharpe ratio proves the strategy works" (it does the
  opposite — it disciplines an inflated number down to an honest one)

**Sentences that are correct and defensible:**
- ✅ "The architecture responded as designed" — decay was detected,
  retraining triggered autonomously, and the acceptance gate correctly
  rejected an inferior candidate.
- ✅ "This demonstrates the closed-loop CT pipeline can detect and
  respond to performance decay without human intervention or serving
  downtime" — this is the literal thesis claim (§1), and it is now
  backed by two independent live observations (§5.5), not just design
  intent or unit tests.

### 5.1 Suggested structure

Map this onto the actual Chapter 5 scaffold headings — this is a
suggestion for *what real content exists to fill each kind of section*,
not a mandate to rename anything:

1. Experimental setup recap (data range, universe, hyperparameters)
2. Baseline results
3. Agent training results (walk-forward, seed sweep)
4. Statistical discipline (deflated Sharpe ratio)
5. Continuous Training loop — live demonstrations
6. Non-functional requirements verification
7. Sensitivity analysis
8. Recorded deviations from the approved Blueprint
9. Discussion (tie back to research questions/objectives)
10. Limitations
11. Chapter summary

### 5.2 Experimental setup

- **Universe:** XOM, CVX, SHEL, BP, NEE (energy equities) + `^VIX` (market
  volatility, joined as a feature, not traded).
- **Data:** real daily OHLCV via yfinance, 2015-01-02 → present (this
  checkout currently holds data through **2026-09-18**, backfilled and
  kept current by an autonomous scheduler — see §5.5.1). Ingested into
  real PostgreSQL, not just tested against SQLite fixtures (CLAUDE.md §9).
  **13,825+ rows** in the original real ingestion pass alone (2015–2025).
- **Features:** SMA_20, RSI_14 (Wilder's exponential smoothing, not a
  simple rolling mean — CLAUDE.md §7), VIX, all rolling Z-score
  normalised with a 60-day backward-looking window (DR-07 — no
  look-ahead, verified by `test_zscore_uses_only_backward_looking_window`
  and `test_observation_contains_no_future_information`).
- **Agent:** PPO (Stable-Baselines3), `MlpPolicy`, **14,731 parameters
  (57.5 KB)** — for scale, ResNet-50 has 25 million.
- **Hardware:** trained on CPU, not GPU — measured (not assumed):
  CPU 5,677–5,718 steps/sec vs. MPS 452.6–438.5 steps/sec on the same
  machine, a **12–13× CPU advantage**, because at 14.7K parameters the
  CPU↔GPU transfer overhead dominates any arithmetic benefit. 500,000
  timesteps trains in **87.0 seconds**.
- **Reward:** portfolio return net of transaction costs, penalised for
  the *increment* in maximum drawdown (not its level — CLAUDE.md §4, I4),
  so the agent isn't billed repeatedly for one historical loss.
- **Evaluation protocol (§10 of CLAUDE.md, followed in full):** baselines
  for context, ≥5 seeds (this pass used 10) reported as mean±std not a
  single number, deflated Sharpe ratio correcting for the true number of
  configurations tried, transaction costs kept on throughout.

### 5.3 Baseline results

On the real 2024-01-01→2025-12-31 eval window (`scripts/run_baselines.py`):
buy-and-hold, equal-weight-rebalanced, random, and all-cash are the four
baselines (`src/rlops/baselines.py`). These exist because a Sharpe ratio
is meaningless in isolation — the same market conditions that reward a
learned policy also reward simply holding the assets.

*(On synthetic fixture data only, for reference — not the real-data
number to cite in Chapter 5 — buy-and-hold reaches Sharpe 1.14; use
the real-data figures in §5.4's table instead, which are the ones that
matter.)*

### 5.4 Agent training results — walk-forward, seed sweep, deflated Sharpe

**First real training run:** 18 walk-forward splits (non-overlapping,
each satisfying DR-06 — eval strictly after train — by construction, not
by a separate check) across the full 2015–2025 real data, **10 seeds per
split** (180 real PPO training runs total, **671 seconds / ~11.2 minutes**
wall time on the CPU above).

**The headline table — report all three rows, not just the top one:**

| Statistic | Value |
|---|---|
| Best-seed-per-split Sharpe (n=18) | mean **1.173**, std **0.914** |
| All individual seeds pooled (n=180) | mean **−0.723**, std **1.583** |
| Positive splits | **17 / 18** |
| Deflated Sharpe per split (n=18) | mean **0.055**, std **0.073**, max **0.301** |
| Splits with DSR > 0.5 | **0 / 18** |

**How to write this up — the critical interpretive point.** The
best-per-split row (mean 1.173) and the pooled row (mean −0.723) tell two
different stories, and the gap between them *is* the finding, not a
contradiction to smooth over:
- **Pooled (−0.723)** is what an untuned, arbitrary single seed actually
  gets you — the honest population statistic.
- **Best-per-split (1.173)** is a maximum-order-statistic: it mechanically
  rises as you draw more seeds, regardless of whether any individual seed
  is actually good. Doubling the seed count from 5→10 alone pushed this
  number from 0.570 to 1.173 — not because training improved, but because
  drawing more samples makes it more likely one of them looks good by
  chance.
- **The deflated Sharpe ratio is what actually disciplines this.**
  Every one of the 18 splits' best result has a DSR under 0.31; **none
  clears 0.5**. Not even the single best result across all 180 runs
  (below) survives correction for how many configurations were actually
  tried.

**The single best result, reported honestly, not flatteringly:**
- Split 11 (eval window **2021-12-05 → 2022-06-02**), seed 0, Sharpe
  **3.273** — reproduced to **five decimal places (3.2727)** across two
  independent runs three days apart, confirming genuine determinism, not
  a fluke measurement.
- Undeflated (treated as the only thing tried): probabilistic Sharpe
  ratio (PSR) = **0.986** — 98.6% probability this Sharpe is genuinely
  positive, *taken alone*.
- **Deflated Sharpe ratio, correcting for 180 real trials: 0.301** (this
  fell from an earlier 0.388 computed at n=90, when only 5 seeds had been
  run — the more trials are honestly counted, the more it deflates, as
  it should).
- **Baseline comparison on that exact window** (same eval partition, so
  the comparison is fair): buy-and-hold Sharpe **2.419**, equal-weight
  **2.388**, random (mean of 5 seeds) **−3.192**, all-cash **0.000**. The
  agent's 3.273 beats buy-and-hold's 2.419 — but buy-and-hold *itself*
  scored 2.4 on this window, because Dec 2021–Jun 2022 was simply a
  strong period for energy equities. **The margin over a passive baseline
  is real but modest once that's accounted for — not what the standalone
  3.273 number suggests on its own.**

**The correct concluding sentence for this section:** "The walk-forward,
seed-sweep, and deflation pipeline ran to completion on real 2015–2025
market data and produced a disciplined, honest figure (DSR 0.301 on the
single best fold) rather than an inflated one (0.986, or the raw 3.273) —
this is the architecture responding as designed to the multiple-testing
problem inherent in any seed sweep, which is itself part of the
contribution under examination." **Never** write a version of "the
strategy achieved a 41% Sharpe-implied return" — that figure is one seed,
one fold, in one favourable regime, and the entire analysis above exists
specifically to stop that number from being reported as a result.

### 5.5 The Continuous Training loop — live demonstrations

This is the actual thesis claim (§1), observed running, twice,
independently, against real data — not asserted from unit tests or design
intent.

**5.5.1 First observation — bootstrap (2026-09-17).**
Triggered via `POST /ct/evaluate?as_of=2025-12-30` against real ingested
data (2015-01-02→2025-12-30 at that point):
1. **Bootstrap.** No incumbent existed yet, so the orchestrator trained
   and promoted a candidate (`e928c835-...`) unconditionally — this is
   the designed behaviour when there is nothing to compare against.
2. **Decay detection, live.** A second evaluation replayed that
   incumbent over real market data and computed **rolling Sharpe =
   −3.79** — well below the 1.0 target — autonomously firing a
   background retrain via the same code path the 300-second scheduler
   uses (FR-14).
3. **The acceptance gate held.** The retrained candidate did not beat
   the incumbent out-of-sample, so it was **rejected, not promoted**
   (FR-17) — the incumbent was retained.

**5.5.2 Second observation — steady state (2026-09-20), specifically
captured for this chapter.** After the bootstrap, the incumbent
(`bf65de93-...`, promoted after beating `e928c835` in a later autonomous
cycle) ran **unattended for three real days**: **540 evaluations, only
2 promotions, 448 rejections** — independent, strong evidence the
acceptance gate is genuinely hard to beat, not merely present in the
code. To capture a second complete cycle on demand (rather than wait for
one), `POST /ct/evaluate?as_of=2026-08-19` replayed the incumbent
against a real historical window already known to be weak for it
(found by scanning recorded telemetry for `rolling_sharpe_30d < 1.0`,
not by guessing a date): **rolling Sharpe = −2.43**, below target,
autonomously triggering a real retrain; the candidate again failed to
beat the incumbent out-of-sample and was rejected (`rejected_count`
448→449); the incumbent remained active throughout, with zero serving
interruption (see §5.6's NFR-03 measurement for the load-bearing proof
of that specific claim).

**How to write this up:** present both observations as independent
confirmations of the same mechanism (decay detected → retrain triggered
autonomously → acceptance gate evaluated the candidate honestly →
correct decision made either way), not as two data points averaged
together. The fact that *both* real retrains were rejected is itself
informative — it shows the gate is not a rubber stamp, and it is
consistent with §5.4's finding that most individual training runs
underperform a well-chosen incumbent.

**Sim2Real note, state explicitly:** "the incumbent's performance" in
every observation above means replaying its own policy against real,
already-ingested market data through `TradingEnvironment` — a faithful
simulation of what it would have done, not a report of what it actually
executed against a live broker. This is exactly the boundary Section 1.7
already concedes, not a new limitation being introduced here.

### 5.6 Non-functional requirements — verified, not asserted

| Requirement | Measured value | Threshold | Result |
|---|---|---|---|
| NFR-01 (latency) | p95 = 16.9ms, p99 = 77.1ms, max = 78.3ms (n=100) | ≤20ms p95 | Pass |
| NFR-02 (degradation during retrain) | 2.1% p95 increase | ≤10% | Pass |
| NFR-03 (zero failed requests) | **1,331 requests, 0 failures**, hammered continuously through a real retrain | 0 failures | Pass |
| NFR-04 (recovery time) | ~8ms mean, ~11ms max (n=5) to reconstruct the service and reload | ≤1 min | Pass (see caveat below) |
| NFR-05 (failed retrain leaves consistent state) | Verified two ways: (a) a genuine training crash — not just a rejected candidate — leaves the incumbent untouched and status returns to `SERVING`; (b) the incumbent kept answering `/predict` correctly throughout the same load test that produced the NFR-03 figure | — | Pass |
| NFR-06 (no circular dependencies) | Verified with `import-linter` against the actual dependency graph, not just the architecture diagram | — | Pass |
| NFR-09 (dashboard memory over time) | **Unmeasured** — requires the dashboard running for real hours in a browser, which a backend measurement pass cannot produce | [X] h | Not measured — state this honestly, do not invent a number |

**Caveat to state for NFR-04, in the interest of full honesty:** the ~8ms
figure may have benefited from test-fixture ordering effects (an
underlying MLflow tracking-store initialisation quirk meant every prior
measurement happened to run after another component had already "warmed"
a shared resource). This is flagged as a possible re-measurement item,
not hidden — state it as "measured under conditions that may not fully
isolate a from-cold-start process" if precision matters for the
defence.

### 5.7 Sensitivity analysis

Real data throughout, not illustrations of an expected shape (this
directly answers the approved proposal's own instruction, CLAUDE.md §6:
"so Chapter 5 can report a sensitivity analysis... instead of defending
magic numbers").

**VIX critical threshold (FR-12):**

| Threshold | Days ≥ threshold (of 2,766 real trading days, 2015–2025) | % of history |
|---|---|---|
| 20.0 | 826 | 29.86% |
| 30.0 | 156 | 5.64% |
| **35.0 (used)** | **63** | **2.28%** |
| 40.0 | 40 | 1.45% |
| 50.0 | 19 | 0.69% |

The chosen threshold fires on 2.28% of real trading days — a genuine
circuit breaker for real market-stress episodes (2015–16's correction,
2018 Q4, the 2020 COVID crash all fall in this range), not something the
system is near constantly.

**target_sharpe_threshold (FR-14):**

| Threshold | % of all 180 real seeds below | % of best-per-split (n=18) below |
|---|---|---|
| 0.0 | 67.2% | 5.6% |
| 0.5 | 75.0% | 22.2% |
| **1.0 (used)** | **85.6%** | **44.4%** |
| 1.5 | 92.8% | 61.1% |
| 2.0 | 96.1% | 83.3% |

This is a genuinely interesting result worth its own paragraph: **85.6%
of individual untuned training seeds fail to clear the retrain
threshold** at the chosen value. This is not the threshold being set
wrong — it directly explains the empirical 540-evaluations/2-promotions
result in §5.5.2, and it is consistent with this project's own measured
finding that PPO is highly seed-sensitive on this task (§5.4). The
threshold correctly reflects that sensitivity rather than papering over
it with a lenient bar.

**transaction_cost_pct (FR-07):**

| Cost | Sharpe (equal-weight-rebalanced, real 2024–25 data) | Cumulative return |
|---|---|---|
| 0 bps | 0.629 | 15.56% |
| **10 bps (used)** | **0.616** | **15.16%** |
| 100 bps | 0.501 | 11.62% |

A modest, monotonic degradation — the chosen cost assumption costs about
2% of the frictionless Sharpe, confirming the "keep transaction costs on"
instruction in the evaluation protocol is guarding against something
real rather than a fragile result that only survives at exactly the
chosen value.

### 5.8 Recorded deviations from the approved Blueprint

Two exist, both required by the departmental guide to be justified here,
not just noted in a dev log:

**FinRL** (named in the proposal). It works — an earlier internal claim
that it was broken was itself wrong and has been corrected. It was not
adopted as the primary environment because (1) its reward is assigned
inline with no override hook, so the drawdown penalty (FR-07) could only
be bolted on afterward, and (2) its actions are share-count based, not
the continuous [-1,1] target weights this project specifies. A
cross-check validation experiment held both environments to an identical
fixed position and found a **max absolute difference of $0.00007237** on
a ~$100–102K portfolio (relative difference ~7×10⁻¹⁰), **correlation
1.000000000000** — corroborating this project's own environment's
valuation and return accounting against FinRL's independently published
implementation.

**Alpaca paper-trading integration** (not in the original Section 3.6
technology stack). Added to test whether the architecture's decisions can
reach a real execution venue's order API — a paper-trading-only
integration (no path to real capital in the code), which stays inside the
"architecture, not alpha" boundary. A real $10 notional BUY order on XOM
filled at $162.876/share for 0.061335 fractional shares on Alpaca's real
paper API; the LIQUIDATE path (I5/FR-12's fail-safe primitive) correctly
closed the position via Alpaca's close-position endpoint. This is real
evidence the serving layer's output is not just internally consistent
but genuinely actionable against an external system.

### 5.9 Discussion — tie back to research objectives

Tie each finding above back to the specific research objective/question
it addresses (Chapter 1/3's numbering — insert the actual objective
numbers from the approved proposal here, since this file doesn't have
them verbatim). The throughline across all of §5.4–§5.8: every
measurement was designed to either demonstrate the CT loop's autonomous
decay-detection-and-recovery mechanism (the actual contribution) or to
honestly bound what the backtest numbers do and do not show (protecting
against overclaiming). Both halves are necessary — a chapter with only
the mechanism demonstrations and no statistical discipline would invite
the "but is the backtest just cherry-picked" question; a chapter with
only the disciplined-but-modest Sharpe numbers and no live mechanism
demonstration would undersell the actual contribution.

### 5.10 Limitations

State plainly, per CLAUDE.md §13, and do not let §5.4–5.5's genuinely
positive architectural results obscure any of these:
- **Single-source data.** yfinance is an unofficial interface with no
  delivery-guarantee SLA.
- **Sample size.** ~2,500 daily observations per ticker over ten years is
  small for a sample-inefficient algorithm like PPO.
- **Seed sensitivity.** Directly demonstrated in §5.4 and §5.7 — this is
  not a hedge, it is a measured property of this specific task.
- **Sim2Real gap**, already conceded in Section 1.7: no real transaction
  costs beyond the modelled assumption, no slippage, no liquidity
  constraints, daily bars not tick data. The Alpaca integration (§5.8)
  demonstrates the architecture *can* reach a real venue; it does not
  close this gap, since it was only exercised in paper trading with a
  synthetic starting position, not a live trading history.
- **NFR-09 remains unmeasured** — correctly left bracketed rather than
  invented (§5.6).

### 5.11 Chapter summary

One paragraph, restating §1's framing rule as the chapter's actual
conclusion: the evaluation demonstrates a closed-loop architecture that
autonomously detects its own performance decay and responds to it —
observed live, twice, independently — while an honest statistical
accounting of the underlying trading policy's performance shows no
individual fold surviving correction for the number of configurations
tried. Both results are the contribution; neither is a claim that the
trading strategy itself is profitable.

---

## Appendix — where every number above comes from, if it needs re-verifying

All of the above is drawn from `CLAUDE.md` in this same repository,
principally:
- §5 (requirements catalogue + NFR measurements)
- §6 (architecture, configuration, sensitivity analysis)
- §7 (non-obvious decisions — the ingestion scheduler, VIX-gated retrain
  deferral, crash-handling, and Alpaca deviation entries)
- §8 (FinRL cross-check)
- §9 (empirical findings — hardware, live ingestion)
- §10 (evaluation protocol, the full training run, both live CT-loop
  observations)
- §13 (known risks/limitations)

If a number here and a number in `CLAUDE.md` ever disagree, `CLAUDE.md`
is the more current source — this file was generated from it on
2026-09-20 and will not automatically pick up anything added afterward.
