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
  modelLabel: "June 2026 #3",
  modelId: "v1.4.2",
  lastIngest: "11 Jun · 16:30",
  ingestionStatus: "IDLE",
  ingestionCheckedAt: "11/06/2026 16:30:00",
  ingestionOk: true,
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

// ------------------------------------------------------------ paper trading
// Example candles and a paper account, same "generated, labelled" rule as
// everything above — never shown once the backend answers.
const BASE_PRICE = { XOM: 112, CVX: 154, SHEL: 68, BP: 34, NEE: 72 };

export function candles(ticker) {
  const r = ((s) => () => (s = (s * 16807) % 2147483647) / 2147483647)(
    7 + ticker.charCodeAt(0) * 31 + ticker.length
  );
  const out = [];
  let close = BASE_PRICE[ticker] ?? 100;
  const d = new Date(Date.UTC(2026, 5, 1));
  while (out.length < 84) {
    d.setUTCDate(d.getUTCDate() + 1);
    if (d.getUTCDay() === 0 || d.getUTCDay() === 6) continue;
    const open = close * (1 + (r() - 0.5) * 0.012);
    close = open * (1 + (r() - 0.48) * 0.024);
    const high = Math.max(open, close) * (1 + r() * 0.008);
    const low = Math.min(open, close) * (1 - r() * 0.008);
    out.push({ date: d.toISOString().slice(0, 10), open, high, low, close, imputed: false });
  }
  return out;
}

export function paperFor(ticker) {
  const c = candles(ticker);
  const buyAt = c[30];
  const addAt = c[52];
  const fills = [
    { side: "buy", date: buyAt.date, price: buyAt.open, qty: 120 },
    { side: "buy", date: addAt.date, price: addAt.open, qty: 60 },
  ];
  const entry = (buyAt.open * 120 + addAt.open * 60) / 180;
  return { fills, entryPrice: entry, lastPrice: c[c.length - 1].close };
}

export const paper = {
  enabled: true,
  equity: 101_284.6,
  startEquity: 100_000,
  cash: 38_412.1,
  lastOutcome: "traded 3 orders",
  lastSignalDate: "2026-09-25",
  nextRunAt: "2026-09-28T22:00:00Z",
  // Same calendar as the example candles, starting from the account's
  // first sync at $100k, so the curve agrees with the P&L figure above it.
  dates: candles("XOM").slice(30).map((c) => new Date(c.date)),
  equityCurve: (() => {
    const n = candles("XOM").length - 30;
    return Array.from({ length: n }, (_, i) =>
      100_000 + (1_284.6 * i) / (n - 1) + (i && i < n - 1 ? Math.sin(i * 0.9) * 260 : 0));
  })(),
  positions: ["XOM", "CVX", "NEE"].map((t) => {
    const p = paperFor(t);
    return {
      ticker: t, qty: 180, avgEntry: p.entryPrice, current: p.lastPrice,
      marketValue: 180 * p.lastPrice, plPct: (p.lastPrice / p.entryPrice - 1) * 100,
    };
  }),
};
