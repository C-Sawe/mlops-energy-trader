# Dashboard (Sprint 4)

React + TypeScript + Vite, styled to an Apple HIG glass aesthetic (Tailwind
for materials/typography/shape, Framer Motion for spring-physics
interactions) — the exact design language specified for this project.

- `src/components/GlassCard.tsx` — the translucent `backdrop-blur` material
  every card is built from.
- `src/components/AppleButton.tsx` — the `whileTap={{ scale: 0.95 }}`
  "physical press" button.
- `src/components/DecisionDetailSheet.tsx` — the fluid bottom sheet (higher
  spring damping than a button, since it's a much larger element).
- `src/components/PerformanceCharts.tsx` — equity curve, rolling Sharpe,
  max drawdown (FR-18), via Recharts.
- `src/components/DecisionLog.tsx` — the paginated decision log (FR-19);
  clicking a row opens the detail sheet.
- `src/components/CTStatusPanel.tsx` / `StatusPill.tsx` — the CT pipeline
  status (FR-20), with purposeful colour (green/blue/amber for
  serving/evaluating/retraining, never decorative).
- `src/lib/api.ts` — the API client, typed to mirror `src/serving/schemas.py`
  by hand (no OpenAPI codegen step yet).
- `src/lib/usePolling.ts` — the polling hook every panel uses to stay live.

## Run

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api -> http://localhost:8000
```

The backend (`src/serving/api.py`) must be running separately and needs a
populated database with at least one promoted `model_version` for the
dashboard to show real data — see the root README's "Running the Dashboard"
section.

## No test suite yet

`npm run lint` (`tsc --noEmit`) is the only automated check. This was
verified by actually running it — backend seeded with a trained model,
frontend driven with Playwright, screenshotted in light and dark mode, one
real bug found and fixed in the process (`CLAUDE.md` §7) — not by a
committed test suite. Adding component tests (Vitest + Testing Library)
would be a reasonable next step before this dashboard is trusted
unsupervised.
