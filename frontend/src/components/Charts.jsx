/**
 * Chart primitives, drawn as SVG against a single scale.
 *
 * Deliberately hand-drawn rather than pulled from a chart library: the design
 * decisions here (grey dashed benchmark as a reference rather than a peer
 * series, direct end labels, the threshold rule, axis labels on the left so
 * they never collide with those end labels) are each easier to hold in plain
 * SVG than to fight a library's defaults for.
 *
 * Conventions applied throughout:
 *   - 2px strokes, >=8px interactive markers
 *   - every axis label names a value the chart actually reaches
 *   - chart text takes theme tokens, so it reads in both modes
 *   - identity is never carried by colour alone (direct labels + dash pattern)
 *
 * `bench` (the buy-and-hold comparison) is optional throughout: this
 * project's backend does not currently persist a benchmark equity series
 * alongside the agent's (CLAUDE.md records this as a known gap, not an
 * oversight), so the live dashboard renders a single agent line with no
 * legend collision or dangling "Buy & hold" label. The mock data still
 * supplies both, so the illustrative version keeps showing the comparison.
 */
import { useEffect, useMemo, useRef, useState } from "react";

const token = (n) =>
  getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const fmtMoney = (v) => "$" + Math.round(v).toLocaleString("en-US");
const fmtDate = (d) =>
  d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });

function useScales(box, pad, n, lo, hi) {
  return useMemo(() => {
    const x = (i) => pad.l + (i / (n - 1)) * (box.w - pad.l - pad.r);
    const y = (v) =>
      pad.t + (1 - (v - lo) / (hi - lo)) * (box.h - pad.t - pad.b);
    return { x, y };
  }, [box, pad, n, lo, hi]);
}

const path = (pts, s) =>
  pts.map((v, i) => `${i ? "L" : "M"}${s.x(i).toFixed(2)} ${s.y(v).toFixed(2)}`).join(" ");

/* ------------------------------------------------------------------ hover */
function useCrosshair(ref, box, pad, n) {
  const [idx, setIdx] = useState(null);
  useEffect(() => {
    const svg = ref.current;
    if (!svg || n < 2) return;
    const plotW = box.w - pad.l - pad.r;
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      const vx = (px / r.width) * box.w;
      const i = Math.round(((vx - pad.l) / plotW) * (n - 1));
      setIdx(Math.max(0, Math.min(n - 1, i)));
    };
    const out = () => setIdx(null);
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerleave", out);
    svg.addEventListener("touchmove", move, { passive: true });
    svg.addEventListener("touchend", out);
    return () => {
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointerleave", out);
      svg.removeEventListener("touchmove", move);
      svg.removeEventListener("touchend", out);
    };
  }, [ref, box, pad, n]);
  return idx;
}

function Tooltip({ visible, x, date, rows }) {
  if (!visible) return null;
  return (
    <div
      className="tip on"
      style={{
        position: "absolute",
        left: Math.max(0, Math.min(x, 1)) * 100 + "%",
        top: 12,
        transform: x > 0.72 ? "translateX(-108%)" : "translateX(14px)",
        pointerEvents: "none",
      }}
    >
      <div className="tip-d">{fmtDate(date)}</div>
      {rows.map(([label, colour, value]) => (
        <div className="tip-row" key={label}>
          <span className="tip-lab">
            <span className="swatch" style={{ background: colour }} />
            {label}
          </span>
          <span className="tip-val num">{value}</span>
        </div>
      ))}
    </div>
  );
}

function EmptyChart({ label }) {
  return <div className="chart-empty">No {label} yet — nothing has been evaluated.</div>;
}

/* ------------------------------------------------------------ equity chart */
export function EquityChart({ dates, agent, bench, label = "Agent" }) {
  const ref = useRef(null);
  const n = agent.length;
  const hasBench = Array.isArray(bench) && bench.length === n;
  const box = { w: 760, h: 260 };
  // Left gutter holds the axis; the right gutter belongs to the direct labels.
  const pad = { l: 54, r: 104, t: 14, b: 26 };

  const [lo, hi] = useMemo(() => {
    const all = hasBench ? agent.concat(bench) : agent;
    return [Math.min(...all) * 0.985, Math.max(...all) * 1.015];
  }, [agent, bench, hasBench]);

  const s = useScales(box, pad, n, lo, hi);
  const idx = useCrosshair(ref, box, pad, n);

  if (n < 2) return <EmptyChart label="equity data" />;

  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xTicks = [0, Math.floor(n * 0.33), Math.floor(n * 0.66), n - 1];

  const series = hasBench
    ? [
        [agent, "var(--series-agent)", "Agent"],
        [bench, "var(--series-bench)", "Buy & hold"],
      ]
    : [[agent, "var(--series-agent)", label]];

  return (
    <div className="chartbox" style={{ position: "relative" }}>
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${box.w} ${box.h}`}
        role="img"
        aria-label={
          hasBench
            ? "Portfolio equity: the agent against an equal-weight buy and hold benchmark."
            : `Portfolio equity: ${label}.`
        }
      >
        <defs>
          <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--series-agent)" stopOpacity="0.20" />
            <stop offset="100%" stopColor="var(--series-agent)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {ticks.map((v) => (
          <g key={v}>
            <line
              x1={pad.l} x2={box.w - pad.r} y1={s.y(v)} y2={s.y(v)}
              stroke="var(--grid)" strokeWidth="1"
            />
            <text
              x={pad.l - 10} y={s.y(v) + 4} textAnchor="end"
              fontSize="11" fontWeight="500" fill="var(--ink-3)"
            >
              ${Math.round(v / 1000)}k
            </text>
          </g>
        ))}

        {xTicks.map((i, k) => (
          <text
            key={i} x={s.x(i)} y={box.h - 6} fontSize="11" fontWeight="500"
            fill="var(--ink-3)"
            textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}
          >
            {fmtDate(dates[i])}
          </text>
        ))}

        <path
          d={`${path(agent, s)} L ${s.x(n - 1)} ${s.y(lo)} L ${s.x(0)} ${s.y(lo)} Z`}
          fill="url(#eqfill)" stroke="none"
        />
        {hasBench && (
          <path
            d={path(bench, s)} fill="none" stroke="var(--series-bench)"
            strokeWidth="2" strokeDasharray="5 4" strokeLinecap="round"
          />
        )}
        <path
          d={path(agent, s)} fill="none" stroke="var(--series-agent)"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        />

        {series.map(([d, c, label]) => (
          <g key={label}>
            <circle
              cx={s.x(n - 1)} cy={s.y(d[n - 1])} r="4.5"
              fill={c} stroke="var(--glass-strong)" strokeWidth="2"
            />
            <text x={s.x(n - 1) + 9} y={s.y(d[n - 1]) - 7}
                  fontSize="11.5" fontWeight="600" fill="var(--ink-2)">
              {label}
            </text>
            <text x={s.x(n - 1) + 9} y={s.y(d[n - 1]) + 6}
                  fontSize="12" fontWeight="600" fill="var(--ink)">
              {fmtMoney(d[n - 1])}
            </text>
          </g>
        ))}

        {idx !== null && (
          <g>
            <line
              x1={s.x(idx)} x2={s.x(idx)} y1={pad.t} y2={box.h - pad.b}
              stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="3 3" opacity="0.55"
            />
            <circle
              cx={s.x(idx)} cy={s.y(agent[idx])} r="5.5"
              fill="none" stroke="var(--ink-2)" strokeWidth="1.5"
            />
          </g>
        )}
      </svg>

      <Tooltip
        visible={idx !== null}
        x={idx === null ? 0 : (s.x(idx) - pad.l) / (box.w - pad.l - pad.r)}
        date={idx === null ? dates[0] : dates[idx]}
        rows={
          idx === null ? [] : hasBench
            ? [
                ["Agent", token("--series-agent"), fmtMoney(agent[idx])],
                ["Buy & hold", token("--series-bench"), fmtMoney(bench[idx])],
              ]
            : [[label, token("--series-agent"), fmtMoney(agent[idx])]]
        }
      />
    </div>
  );
}

/* ------------------------------------------------------------ sharpe chart */
// A rolling Sharpe computed over a near-flat stretch of returns (the model
// holding steady for weeks) can mathematically explode to an enormous but
// finite number — CLAUDE.md §7's `_VARIANCE_FLOOR` note: the floor prevents
// an infinite value, not merely a huge one. Real, but useless on a linear
// chart: one such point would compress every meaningful value into a flat
// line near zero. Exported as a standalone function (not inlined in the
// component) specifically so this logic — the exact thing that caused a
// real bug once — is unit-testable without rendering SVG.
export const SHARPE_OUTLIER_BOUND = 20;

/** The chart's y-axis domain, computed from values actually within
 * `SHARPE_OUTLIER_BOUND` of zero; anything beyond that is plotted clipped
 * to the axis edge (see `clampForDisplay`) and marked with a chevron, not
 * silently rescaled or dropped — the tooltip and aria-label always carry
 * the true value. */
export function computeSharpeDomain(values, threshold) {
  if (values.length === 0) return [0, 2];
  const inRange = values.filter((v) => Math.abs(v) <= SHARPE_OUTLIER_BOUND);
  const basis = inRange.length ? inRange : values;
  const min = Math.min(...basis, threshold);
  const max = Math.max(...basis, threshold);
  const span = Math.max(max - min, 0.5);
  return [min - span * 0.15, max + span * 0.15];
}

/** Clips each value into `[lo, hi]` for plotting, and flags which ones were
 * actually out of range — the flag is what draws the chevron marker. */
export function clampForDisplay(values, lo, hi) {
  return {
    clampedValues: values.map((v) => Math.max(lo, Math.min(hi, v))),
    offScale: values.map((v) => v < lo || v > hi),
  };
}

export function SharpeChart({ dates, values, threshold = 1.0 }) {
  const ref = useRef(null);
  // Warm-up rows (fewer than the rolling window's worth of history) are
  // null on the backend (CLAUDE.md §7's variance-floor note), not zero —
  // drop them here rather than plotting a misleading dip to 0.
  const clean = useMemo(
    () => dates.map((d, i) => [d, values[i]]).filter(([, v]) => v != null),
    [dates, values]
  );
  const n = clean.length;
  const cleanDates = clean.map(([d]) => d);
  const cleanValues = clean.map(([, v]) => v);

  const box = { w: 760, h: 200 };
  const pad = { l: 44, r: 18, t: 24, b: 26 };

  const [lo, hi] = useMemo(
    () => computeSharpeDomain(cleanValues, threshold),
    [cleanValues, threshold]
  );

  const s = useScales(box, pad, Math.max(n, 2), lo, hi);
  const idx = useCrosshair(ref, box, pad, n);

  if (n < 2) return <EmptyChart label="Sharpe history" />;

  const { clampedValues, offScale } = clampForDisplay(cleanValues, lo, hi);

  const xTicks = [0, Math.floor(n * 0.33), Math.floor(n * 0.66), n - 1];

  return (
    <div className="chartbox" style={{ position: "relative" }}>
      <svg
        ref={ref} className="chart" viewBox={`0 0 ${box.w} ${box.h}`} role="img"
        aria-label={`Rolling Sharpe ratio against a retraining threshold of ${threshold.toFixed(2)}.`}
      >
        <defs>
          <linearGradient id="shfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--series-agent)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--series-agent)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4).map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={box.w - pad.r} y1={s.y(v)} y2={s.y(v)}
                  stroke="var(--grid)" strokeWidth="1" />
            <text x={pad.l - 10} y={s.y(v) + 4} textAnchor="end"
                  fontSize="11" fontWeight="500" fill="var(--ink-3)">
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {xTicks.map((i, k) => (
          <text key={i} x={s.x(i)} y={box.h - 6} fontSize="11" fontWeight="500"
                fill="var(--ink-3)"
                textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}>
            {fmtDate(cleanDates[i])}
          </text>
        ))}

        <path
          d={`${path(clampedValues, s)} L ${s.x(n - 1)} ${s.y(lo)} L ${s.x(0)} ${s.y(lo)} Z`}
          fill="url(#shfill)" stroke="none"
        />

        {/* The rule the orchestrator acts on (FR-14), drawn so the analyst can
            watch the system approach its own trigger rather than learn of a
            retrain only after it has started. */}
        <line x1={pad.l} x2={box.w - pad.r} y1={s.y(threshold)} y2={s.y(threshold)}
              stroke="var(--warn)" strokeWidth="2" strokeDasharray="6 5" strokeLinecap="round" />
        <text x={pad.l + 8} y={s.y(threshold) - 8}
              fontSize="11" fontWeight="600" fill="var(--warn)">
          Retraining threshold · {threshold.toFixed(2)}
        </text>

        <path d={path(clampedValues, s)} fill="none" stroke="var(--series-agent)"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={s.x(n - 1)} cy={s.y(clampedValues[n - 1])} r="4.5"
                fill="var(--series-agent)" stroke="var(--glass-strong)" strokeWidth="2" />

        {/* Off-scale points: a near-flat return stretch can send the ratio
            far beyond any sane axis range (see OUTLIER_BOUND above). Marked
            with a chevron and its true value rather than silently clipped —
            the point is genuinely there, just not at that height. */}
        {cleanValues.map((v, i) => {
          if (!offScale[i]) return null;
          const atTop = v > hi;
          const y = atTop ? pad.t : box.h - pad.b;
          return (
            <g key={i} transform={`translate(${s.x(i)}, ${y})`}>
              <path
                d={atTop ? "M -5 6 L 0 -2 L 5 6" : "M -5 -6 L 0 2 L 5 -6"}
                fill="none" stroke="var(--warn)" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round"
              />
              <text x={0} y={atTop ? 18 : -10} textAnchor="middle" fontSize="10"
                    fontWeight="600" fill="var(--warn)">
                {v.toFixed(0)}
              </text>
            </g>
          );
        })}

        {idx !== null && (
          <line x1={s.x(idx)} x2={s.x(idx)} y1={pad.t} y2={box.h - pad.b}
                stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="3 3" opacity="0.55" />
        )}
      </svg>

      <Tooltip
        visible={idx !== null}
        x={idx === null ? 0 : (s.x(idx) - pad.l) / (box.w - pad.l - pad.r)}
        date={idx === null ? cleanDates[0] : cleanDates[idx]}
        rows={idx === null ? [] : [["Sharpe", token("--series-agent"), cleanValues[idx].toFixed(2)]]}
      />
    </div>
  );
}

/* ------------------------------------------------------- candlestick chart */
// Fills are placed by *date string* ("YYYY-MM-DD"), never by Date object
// arithmetic — the same DST/timezone trap api.js's fetchTelemetry fell into
// once. Exported for the same reason computeSharpeDomain is: it's the part
// most likely to silently misplace a marker, so it's tested without SVG.

/** Map each fill onto the candle for its trading date, or the next candle
 * if that date has no bar. A fill after the last candle (it filled today,
 * and today's bar is only ingested after the close) can't be placed yet —
 * it's counted as `pending` rather than drawn at a made-up position. */
export function placeFills(candles, fills) {
  const placed = [];
  let pending = 0;
  for (const f of fills) {
    if (f.price == null || !f.date) continue;
    const index = candles.findIndex((c) => c.date >= f.date);
    if (index === -1) pending += 1;
    else placed.push({ index, side: f.side, price: f.price, qty: f.qty });
  }
  return { placed, pending };
}

/** Price axis covering every candle's range plus any reference price
 * (average entry, last price) that must stay visible even if it sits
 * outside the window's own high/low. */
export function candleDomain(candles, refPrices = []) {
  const refs = refPrices.filter((v) => v != null && Number.isFinite(v));
  if (candles.length === 0 && refs.length === 0) return [0, 1];
  const lows = candles.map((c) => c.low).concat(refs);
  const highs = candles.map((c) => c.high).concat(refs);
  const lo = Math.min(...lows);
  const hi = Math.max(...highs);
  const span = Math.max(hi - lo, Math.abs(hi) * 0.01, 0.01);
  return [lo - span * 0.08, hi + span * 0.08];
}

const fmtPrice = (v) => "$" + v.toFixed(2);
// Candle dates are calendar dates, not instants: format them in UTC so a
// viewer west of Greenwich doesn't see every label one day early.
const fmtDay = (iso) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB", {
    day: "numeric", month: "short", timeZone: "UTC",
  });

export function CandleChart({ ticker, candles, fills = [], entryPrice = null, lastPrice = null }) {
  const ref = useRef(null);
  const n = candles.length;
  const box = { w: 760, h: 300 };
  const pad = { l: 54, r: 96, t: 14, b: 26 };
  const plotW = box.w - pad.l - pad.r;
  const slot = plotW / Math.max(n, 1);

  const [lo, hi] = useMemo(
    () => candleDomain(candles, [entryPrice, lastPrice]),
    [candles, entryPrice, lastPrice]
  );
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (box.h - pad.t - pad.b);
  const cx = (i) => pad.l + slot * (i + 0.5);

  const [idx, setIdx] = useState(null);
  useEffect(() => {
    const svg = ref.current;
    if (!svg || n < 1) return;
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      const i = Math.floor(((px / r.width) * box.w - pad.l) / slot);
      setIdx(i >= 0 && i < n ? i : null);
    };
    const out = () => setIdx(null);
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerleave", out);
    return () => {
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointerleave", out);
    };
  }, [n, slot, box.w, pad.l]);

  const { placed, pending } = useMemo(() => placeFills(candles, fills), [candles, fills]);

  if (n < 2) return <EmptyChart label={`price data for ${ticker}`} />;

  const bodyW = Math.max(1.5, Math.min(slot * 0.64, 12));
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xTicks = [0, Math.floor(n * 0.33), Math.floor(n * 0.66), n - 1];

  // Right-gutter labels for the two reference lines; nudged apart if
  // they'd overlap, so neither price is ever hidden under the other.
  let entryY = entryPrice != null ? y(entryPrice) : null;
  let lastY = lastPrice != null ? y(lastPrice) : null;
  if (entryY != null && lastY != null && Math.abs(entryY - lastY) < 26) {
    const mid = (entryY + lastY) / 2;
    const up = entryY < lastY ? -13 : 13;
    entryY = mid + up;
    lastY = mid - up;
  }
  const pnlPct =
    entryPrice != null && lastPrice != null ? (lastPrice / entryPrice - 1) * 100 : null;

  const dayFills = idx === null ? [] : placed.filter((f) => f.index === idx);

  return (
    <div className="chartbox" style={{ position: "relative" }}>
      <svg
        ref={ref} className="chart" viewBox={`0 0 ${box.w} ${box.h}`} role="img"
        aria-label={
          `${ticker} daily candles` +
          (entryPrice != null ? `, average entry ${fmtPrice(entryPrice)}` : "") +
          (lastPrice != null ? `, last ${fmtPrice(lastPrice)}` : "") +
          `, ${placed.length} filled orders marked.`
        }
      >
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={box.w - pad.r} y1={y(v)} y2={y(v)}
                  stroke="var(--grid)" strokeWidth="1" />
            <text x={pad.l - 10} y={y(v) + 4} textAnchor="end"
                  fontSize="11" fontWeight="500" fill="var(--ink-3)">
              {v.toFixed(v >= 100 ? 0 : 1)}
            </text>
          </g>
        ))}

        {xTicks.map((i, k) => (
          <text key={i} x={cx(i)} y={box.h - 6} fontSize="11" fontWeight="500"
                fill="var(--ink-3)"
                textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}>
            {fmtDay(candles[i].date)}
          </text>
        ))}

        {/* Up candles hollow, down candles filled — direction is carried
            by shape as well as colour. Imputed (forward-filled, DR-04)
            bars are faded: they are not real trading. */}
        {candles.map((c, i) => {
          const up = c.close >= c.open;
          const colour = up ? "var(--good)" : "var(--critical)";
          const top = y(Math.max(c.open, c.close));
          const h = Math.max(1, Math.abs(y(c.open) - y(c.close)));
          return (
            <g key={c.date} opacity={c.imputed ? 0.35 : 1}>
              <line x1={cx(i)} x2={cx(i)} y1={y(c.high)} y2={y(c.low)}
                    stroke={colour} strokeWidth="1.2" />
              <rect x={cx(i) - bodyW / 2} y={top} width={bodyW} height={h}
                    fill={up ? "var(--glass-strong)" : colour}
                    stroke={colour} strokeWidth="1.2" rx="0.8" />
            </g>
          );
        })}

        {entryPrice != null && (
          <g>
            <line x1={pad.l} x2={box.w - pad.r} y1={y(entryPrice)} y2={y(entryPrice)}
                  stroke="var(--accent)" strokeWidth="1.6" strokeDasharray="6 5" />
            <text x={box.w - pad.r + 8} y={entryY - 2} fontSize="11" fontWeight="600"
                  fill="var(--accent)">Avg entry</text>
            <text x={box.w - pad.r + 8} y={entryY + 11} fontSize="12" fontWeight="600"
                  fill="var(--ink)">{fmtPrice(entryPrice)}</text>
          </g>
        )}

        {lastPrice != null && (
          <g>
            <line x1={pad.l} x2={box.w - pad.r} y1={y(lastPrice)} y2={y(lastPrice)}
                  stroke="var(--ink-2)" strokeWidth="1" opacity="0.7" />
            <text x={box.w - pad.r + 8} y={lastY - 2} fontSize="11" fontWeight="600"
                  fill="var(--ink-2)">
              Now{pnlPct != null ? ` ${pnlPct >= 0 ? "+" : "−"}${Math.abs(pnlPct).toFixed(1)}%` : ""}
            </text>
            <text x={box.w - pad.r + 8} y={lastY + 11} fontSize="12" fontWeight="600"
                  fill="var(--ink)">{fmtPrice(lastPrice)}</text>
          </g>
        )}

        {/* Fill markers at the actual fill price: ▲ buy, ▼ sell. Shape,
            not just colour, carries the side. */}
        {placed.map((f, k) => {
          const buy = f.side === "buy";
          const py = y(f.price);
          const d = buy
            ? `M ${cx(f.index)} ${py + 3} l -6 10 h 12 Z`
            : `M ${cx(f.index)} ${py - 3} l -6 -10 h 12 Z`;
          return (
            <g key={k}>
              <line x1={cx(f.index) - bodyW} x2={cx(f.index) + bodyW} y1={py} y2={py}
                    stroke="var(--ink)" strokeWidth="1.5" />
              <path d={d} fill={buy ? "var(--good)" : "var(--critical)"}
                    stroke="var(--glass-strong)" strokeWidth="1.5" />
            </g>
          );
        })}

        {idx !== null && (
          <line x1={cx(idx)} x2={cx(idx)} y1={pad.t} y2={box.h - pad.b}
                stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="3 3" opacity="0.55" />
        )}
      </svg>

      {pending > 0 && (
        <p className="chart-note">
          {pending} fill{pending > 1 ? "s" : ""} from today will appear once today's bar is ingested after the close.
        </p>
      )}

      {idx !== null && (
        <div
          className="tip on"
          style={{
            position: "absolute", top: 12, pointerEvents: "none",
            left: ((cx(idx) - pad.l) / plotW) * 100 + "%",
            transform: (cx(idx) - pad.l) / plotW > 0.66 ? "translateX(-108%)" : "translateX(14px)",
          }}
        >
          <div className="tip-d">{fmtDay(candles[idx].date)}{candles[idx].imputed ? " · filled gap" : ""}</div>
          {[["Open", candles[idx].open], ["High", candles[idx].high],
            ["Low", candles[idx].low], ["Close", candles[idx].close]].map(([k, v]) => (
            <div className="tip-row" key={k}>
              <span className="tip-lab">{k}</span>
              <span className="tip-val num">{fmtPrice(v)}</span>
            </div>
          ))}
          {dayFills.map((f, k) => (
            <div className="tip-row" key={"f" + k}>
              <span className="tip-lab">{f.side === "buy" ? "▲ Bought" : "▼ Sold"}</span>
              <span className="tip-val num">{fmtPrice(f.price)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ gauge */
export function Gauge({ fraction, colour = "var(--good)" }) {
  const cx = 62, cy = 66, r = 46;
  const arc = (f) => {
    const a0 = Math.PI, a1 = Math.PI + Math.PI * Math.max(0.02, Math.min(1, f));
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    // Sweep never exceeds 180 degrees, so large-arc is always 0. Deriving it
    // from the fraction splits the arc into disjoint pieces past the halfway
    // point — a bug worth not reintroducing.
    return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  };
  return (
    <svg width="124" height="80" viewBox="0 0 124 80" aria-hidden="true">
      <path d={arc(1)} fill="none" stroke="var(--grid)" strokeWidth="9" strokeLinecap="round" />
      <path d={arc(fraction)} fill="none" stroke={colour} strokeWidth="9" strokeLinecap="round" />
    </svg>
  );
}
