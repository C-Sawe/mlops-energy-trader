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
export function EquityChart({ dates, agent, bench }) {
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
    : [[agent, "var(--series-agent)", "Agent"]];

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
            : "Portfolio equity for the agent."
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
            : [["Agent", token("--series-agent"), fmtMoney(agent[idx])]]
        }
      />
    </div>
  );
}

/* ------------------------------------------------------------ sharpe chart */
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
  const [lo, hi] = useMemo(() => {
    if (n === 0) return [0, 2];
    const min = Math.min(...cleanValues, threshold);
    const max = Math.max(...cleanValues, threshold);
    const span = Math.max(max - min, 0.5);
    return [min - span * 0.15, max + span * 0.15];
  }, [cleanValues, threshold, n]);

  const s = useScales(box, pad, Math.max(n, 2), lo, hi);
  const idx = useCrosshair(ref, box, pad, n);

  if (n < 2) return <EmptyChart label="Sharpe history" />;

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
          d={`${path(cleanValues, s)} L ${s.x(n - 1)} ${s.y(lo)} L ${s.x(0)} ${s.y(lo)} Z`}
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

        <path d={path(cleanValues, s)} fill="none" stroke="var(--series-agent)"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={s.x(n - 1)} cy={s.y(cleanValues[n - 1])} r="4.5"
                fill="var(--series-agent)" stroke="var(--glass-strong)" strokeWidth="2" />

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
