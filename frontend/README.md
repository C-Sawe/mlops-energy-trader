# Telemetry dashboard — presentation layer (Sprint 4)

React implementation of Figure 4.8, satisfying FR-18 through FR-20.

```bash
cd frontend
npm install
npm run dev          # expects the FastAPI service on :8000
```

Set `VITE_API_BASE` if the backend is elsewhere, and `VITE_API_BEARER_TOKEN`
if the backend has `API_BEARER_TOKEN` set (`src/config.py`'s `ServingConfig`
— unset by default, single-user local operation).

## What is here

| File | Role |
|---|---|
| `src/theme.css` | Apple HIG token system, light and dark |
| `src/components/Charts.jsx` | Equity curve, rolling Sharpe, gauge |
| `src/components/Primitives.jsx` | Glass card, pill, status strip, spring sheet |
| `src/api.js` | FastAPI client (IR-04), graceful degradation (IR-07) |
| `src/App.jsx` | Composition — the three bands of Figure 4.8, plus the adapters that translate this project's real API shapes (`src/serving/schemas.py`) into the view-model `mock.js` already uses |
| `src/mock.js` | Placeholder telemetry, shown until `/ct-status` etc. answer, or again if the backend later goes unreachable |
| `design-reference.html` | Standalone version — opens in a browser, no build |

Verified: `npm run build` succeeds and the page renders with the sheet,
crosshair tooltips and theme toggle all working, against a real backend
seeded with a trained and promoted model (screenshotted in both themes).

## Design decisions worth preserving

**Status colours are reserved.** Green, orange and red mean good / warning /
critical and are never reused as a data series. This is why the benchmark line
is a grey dashed reference rather than a second hue — orange would have
collided with the "retraining" pipeline state.

**The two series colours were validated**, not chosen by eye, for
colour-vision-deficiency separation against the chart surface in both light and
dark modes. If you substitute them, re-validate rather than assuming.

**Identity is never carried by colour alone.** Both equity series are
direct-labelled at their end point, and the benchmark additionally differs by
dash pattern — so the chart still reads in greyscale, in print, and for a
colour-blind viewer.

**Axis labels sit on the left**, direct labels on the right. Both once claimed
the same gutter and collided.

**The gauge arc's large-arc flag is always 0.** The sweep never exceeds 180
degrees, so deriving the flag from the value splits the arc into disjoint
pieces past the halfway point.

**Motion is spring-based, never time-eased.** `useSpring` integrates a
critically damped spring (stiffness 300, damping 30, mass 1): fast to settle,
no wobble. Buttons compress under press rather than only recolouring. All of it
is suppressed under `prefers-reduced-motion`.

**Type never goes below weight 500**, per HIG guidance against thin weights for
interface text, and every figure uses tabular numerals so columns align.

**The equity chart's benchmark series is optional, not faked.** This
project's backend does not currently persist a buy-and-hold comparison
alongside the agent's equity curve — the CT orchestrator only ever replays
the incumbent, not a baseline policy, during evaluation. Rather than invent
one client-side, `EquityChart` renders a single agent line (with its own
adjusted legend/subtitle) when live data has no `bench` series, and only the
mock data — clearly marked as such — shows the full two-series comparison.
Persisting a real benchmark server-side would be a reasonable Sprint 5
addition; faking it here would not.

**The retraining threshold is read from the backend, not hardcoded.** Both
the Sharpe chart's rule and the gauge's headroom calculation use
`target_sharpe_threshold` from `/ct-status`, which is itself
`RISK.target_sharpe_threshold` (`src/config.py`) — this project's existing
"thresholds are measured/configured, never invented" convention, applied to
the dashboard.

## Information design

The analyst using this is judging whether an autonomous system remains
trustworthy, not operating it. The bands are ordered accordingly:

1. **Pipeline state** — the first question is whether the system is working.
2. **Performance** — meaningful only once state is known. The retraining
   threshold is drawn on the Sharpe chart so the analyst watches the system
   approach its own trigger, rather than learning of a retrain afterwards.
3. **Decision log** — for investigating after an anomaly, not routine
   monitoring. Each row opens a sheet showing the full attribution chain
   (decision → model version → run → both the training and evaluation
   partition), which is what makes NFR-07 visible rather than merely
   satisfied in the schema. `MarketRepository.list_decisions()` joins
   through to `model_run` specifically so this sheet can show all of it in
   one call.

A fail-safe row is deliberately present in the sample data: it is the only
place in the whole project where FR-12 and NFR-11 can be *seen* — VIX above
threshold, action `LIQUIDATE`, no raw weight, because the model was never
consulted. Trigger a real one live by pushing `VIX_CRITICAL_THRESHOLD` below
the current ingested VIX and calling `/predict`.

**"Cycle history" counts training runs, not specifically autonomous
retrains.** `ModelRun` has no field distinguishing a `CTOrchestrator`
background retrain from a manual `scripts/train_agent.py` sweep, so the
card is labelled "Training runs, last N days" rather than "Retrains this
quarter" — an honest label for what is actually measured
(`MarketRepository.get_cycle_stats`), not a more impressive-sounding one.
Every trained candidate is logged whether it wins or loses the FR-17
acceptance gate (`status="REJECTED"` for a loser), which is what makes
"held back" a real, queryable number instead of always reading zero.

## Honesty constraint

Any sample or placeholder data must be visibly marked as such. The thesis
claims an architectural result, not a financial one, and a dashboard showing a
flattering equity curve with no such marking invites exactly the
misinterpretation Chapter 1.7 rules out.
