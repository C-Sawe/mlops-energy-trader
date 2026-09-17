/**
 * Placeholder telemetry, used only until the backend answers (or when it
 * stops answering mid-session).
 *
 * Everything here is generated. The dashboard marks it visibly as such, and
 * that marking must survive: the thesis claims an architectural result, not a
 * financial one, and an unlabelled equity curve invites exactly the reading
 * Section 1.7 rules out.
 */

// Deterministic PRNG so the page looks the same on every reload.
const rand = ((s) => () => (s = (s * 16807) % 2147483647) / 2147483647)(42);

const N = 124;

export const dates = (() => {
  const start = new Date(2026, 0, 5);
  return Array.from({ length: N }, (_, i) => {
    const d = new Date(start);
    d.setDate(d.getDate() + Math.round(i * 1.42));
    return d;
  });
})();

const _agent = [];
const _bench = [];
(() => {
  let a = 100000;
  let b = 100000;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    // A drawdown episode in April, so the drawdown gauge and the Sharpe dip
    // have a common cause rather than looking unrelated.
    const shock = t > 0.3 && t < 0.42 ? -0.0034 : 0;
    a *= 1 + 0.0025 + shock * 1.35 + (rand() - 0.5) * 0.0092;
    b *= 1 + 0.00208 + shock * 1.55 + (rand() - 0.5) * 0.0104;
    _agent.push(a);
    _bench.push(b);
  }
})();

export const agent = _agent;
export const bench = _bench;

export const sharpe = Array.from({ length: N }, (_, i) => {
  const t = i / (N - 1);
  const dip = Math.exp(-Math.pow((t - 0.355) / 0.085, 2)) * 0.92;
  return 1.86 - dip + Math.sin(t * 13.5) * 0.15 + (rand() - 0.5) * 0.1;
});

export const status = {
  pipeline: "SERVING",
  model: "v1.4.2",
  lastIngest: "11 Jun · 16:30",
  vix: 16.4,
  failSafeThreshold: 35,
  targetSharpe: 1.0,
};

export const decisions = [
  { ts: "2026-06-11 16:28:12", ticker: "XOM",  weight: "0.720",  action: "BUY",  vix: "15.2", failsafe: false,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
  { ts: "2026-06-11 15:42:07", ticker: "CVX",  weight: "0.140",  action: "HOLD", vix: "14.8", failsafe: false,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
  { ts: "2026-06-11 14:15:33", ticker: "SHEL", weight: "−0.680", action: "SELL", vix: "16.1", failsafe: false,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
  { ts: "2026-06-11 12:03:21", ticker: "BP",   weight: "0.310",  action: "HOLD", vix: "18.6", failsafe: false,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
  { ts: "2026-06-11 10:27:44", ticker: "NEE",  weight: "0.910",  action: "BUY",  vix: "17.3", failsafe: false,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
  // The only place in the project where FR-12 and NFR-11 can actually be seen:
  // VIX above threshold, no raw weight, because the model was never consulted.
  { ts: "2026-06-10 09:11:05", ticker: "XOM",  weight: "—", action: "LIQUIDATE", vix: "38.4", failsafe: true,
    version: "v1.4.2", run: "run_8f21c6", trainPartition: "2015-01-01 → 2023-12-31", evalPartition: "2024-01-01 → 2025-06-01" },
];

export const decisionsTotal = 42;

export const metrics = {
  rollingSharpe: 1.42,
  maxDrawdown: -12.8,
  windowDays: 90,
  trainingRuns: 3,
  promoted: 2,
  heldBack: 1,
};
