import {
  Area,
  AreaChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GlassCard } from "./GlassCard";
import type { EquityPoint } from "../lib/api";

// A brand-neutral, purposeful palette: blue for the primary equity series
// (the thing being tracked), green/red reserved for gain/loss framing on
// drawdown specifically, never decorative rainbow colouring.
const BLUE = "#0A84FF";
const RED = "#FF3B30";

function GlassTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: { value: number }[];
  label?: string;
  formatter: (v: number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/20 bg-white/90 px-3 py-2 text-xs shadow-lg backdrop-blur-xl dark:border-white/10 dark:bg-black/80">
      <div className="font-medium text-black/60 dark:text-white/60">{label}</div>
      <div className="font-semibold tabular-nums text-black dark:text-white">
        {formatter(payload[0].value)}
      </div>
    </div>
  );
}

const fmtCurrency = (v: number) =>
  `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const fmtRatio = (v: number) => v.toFixed(2);
const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function EquityCurveChart({ points }: { points: EquityPoint[] }) {
  return (
    <GlassCard className="col-span-2 p-6">
      <h2 className="text-lg font-semibold tracking-tight">Equity Curve</h2>
      <p className="text-sm text-black/50 dark:text-white/50">FR-18 · portfolio value over time</p>
      <div className="mt-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
            <defs>
              <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={BLUE} stopOpacity={0.35} />
                <stop offset="100%" stopColor={BLUE} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="date" hide />
            <YAxis
              domain={["auto", "auto"]}
              width={64}
              tick={{ fontSize: 12, fill: "currentColor", opacity: 0.5 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip content={<GlassTooltip formatter={fmtCurrency} />} />
            <Area
              type="monotone"
              dataKey="equity_value"
              stroke={BLUE}
              strokeWidth={2}
              fill="url(#equityFill)"
              isAnimationActive
              animationDuration={600}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
}

export function RollingSharpeChart({ points }: { points: EquityPoint[] }) {
  return (
    <GlassCard className="p-6">
      <h2 className="text-base font-semibold tracking-tight">Rolling Sharpe (30d)</h2>
      <p className="text-sm text-black/50 dark:text-white/50">FR-13</p>
      <div className="mt-3 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
            <XAxis dataKey="date" hide />
            <YAxis hide domain={["auto", "auto"]} />
            <Tooltip content={<GlassTooltip formatter={fmtRatio} />} />
            <Line
              type="monotone"
              dataKey="rolling_sharpe_30d"
              stroke={BLUE}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive
              animationDuration={600}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
}

export function MaxDrawdownChart({ points }: { points: EquityPoint[] }) {
  return (
    <GlassCard className="p-6">
      <h2 className="text-base font-semibold tracking-tight">Max Drawdown</h2>
      <p className="text-sm text-black/50 dark:text-white/50">FR-18</p>
      <div className="mt-3 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
            <defs>
              <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={RED} stopOpacity={0.3} />
                <stop offset="100%" stopColor={RED} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="date" hide />
            <YAxis hide domain={[0, "auto"]} />
            <Tooltip content={<GlassTooltip formatter={fmtPct} />} />
            <Area
              type="monotone"
              dataKey="max_drawdown"
              stroke={RED}
              strokeWidth={2}
              fill="url(#ddFill)"
              isAnimationActive
              animationDuration={600}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
}
