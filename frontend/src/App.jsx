/**
 * Telemetry dashboard — FR-18, FR-19, FR-20.
 *
 * Band order follows the supervisory task rather than the data model: the
 * analyst's first question is whether the pipeline is healthy, performance is
 * only meaningful once that is known, and the decision log serves
 * investigation after an anomaly rather than routine monitoring.
 *
 * Adapter functions below translate this project's real API shapes
 * (src/serving/schemas.py) into the same view-model shape mock.js already
 * provides, so every component renders identically whether the data is
 * live or illustrative. Live data is frequently *incomplete* in ways the
 * mock never is — no CT evaluation has run yet, no snapshots exist yet —
 * so every field read here is null-guarded rather than assumed present.
 */
import { useEffect, useState } from "react";
import { EquityChart, SharpeChart, Gauge } from "./components/Charts.jsx";
import {
  GlassCard, CardHead, Pill, StatusStrip, Sheet, Button,
} from "./components/Primitives.jsx";
import { fetchStatus, fetchTelemetry, fetchDecisions, triggerEvaluation, poll } from "./api.js";
import * as mock from "./mock.js";

const actionClass = (a) =>
  ({ BUY: "buy", SELL: "sell", HOLD: "hold", LIQUIDATE: "liq" }[a] ?? "hold");

const shortId = (id) => (id ? id.slice(0, 8) : "—");
const fmtShortDate = (d) => d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
const fmtDateTime = (d) =>
  d.toLocaleString("en-GB", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).replace(",", "");

function adaptStatus(raw) {
  return {
    pipeline: raw.status,
    model: shortId(raw.active_version_id),
    lastIngest: raw.last_ingest_date ? fmtShortDate(new Date(raw.last_ingest_date)) : "—",
    vix: raw.current_vix,
    failSafeThreshold: raw.vix_critical_threshold,
    targetSharpe: raw.target_sharpe_threshold,
  };
}

function adaptTelemetry(raw) {
  const points = raw.points ?? [];
  const last = points[points.length - 1];
  return {
    dates: points.map((p) => new Date(p.date)),
    agent: points.map((p) => p.equity_value),
    bench: null, // not persisted server-side yet — see Charts.jsx's note
    sharpe: points.map((p) => p.rolling_sharpe_30d),
    lastMaxDrawdown: last?.max_drawdown ?? null,
  };
}

function adaptDecisions(raw) {
  return {
    items: raw.items.map((d) => ({
      ts: fmtDateTime(new Date(d.decided_at)),
      ticker: d.ticker,
      weight: d.raw_weight != null ? d.raw_weight.toFixed(3) : "—",
      action: d.discrete_action,
      vix: d.vix_at_decision != null ? d.vix_at_decision.toFixed(1) : "—",
      failsafe: d.failsafe_triggered,
      version: shortId(d.version_id),
      run: shortId(d.run_id),
      trainPartition: `${d.train_start} → ${d.train_end}`,
      evalPartition: `${d.eval_start} → ${d.eval_end}`,
    })),
    total: raw.total,
  };
}

function adaptMetrics(statusRaw, telemetryAdapted) {
  return {
    rollingSharpe: statusRaw.rolling_sharpe,
    maxDrawdown: telemetryAdapted.lastMaxDrawdown != null ? -(telemetryAdapted.lastMaxDrawdown * 100) : null,
    windowDays: statusRaw.cycle_window_days,
    trainingRuns: statusRaw.training_runs,
    promoted: statusRaw.promoted_count,
    heldBack: statusRaw.rejected_count,
  };
}

export default function App() {
  const [live, setLive] = useState(null); // null until the backend answers
  const [stale, setStale] = useState(false);
  const [selected, setSelected] = useState(null);
  const [page, setPage] = useState(1);
  const [checking, setChecking] = useState(false);
  const pageSize = 6;

  // IR-07: when the backend is unreachable the page keeps rendering and says
  // so, rather than blanking. Until data has ever loaded, this is the mock.
  useEffect(() => {
    return poll(
      async () => {
        const [s, t, d] = await Promise.all([
          fetchStatus(), fetchTelemetry(), fetchDecisions(page, pageSize),
        ]);
        return { s, t, d };
      },
      15000,
      ({ s, t, d }) => {
        if (s.ok && t.ok && d.ok) {
          setLive({
            statusRaw: s.data,
            status: adaptStatus(s.data),
            telemetry: adaptTelemetry(t.data),
            decisions: adaptDecisions(d.data),
          });
          setStale(false);
        } else {
          setStale(true);
        }
      }
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  const usingMock = live === null;
  const status = live?.status ?? mock.status;
  const decisionsView = live?.decisions ?? { items: mock.decisions, total: mock.decisionsTotal };
  const decisions = decisionsView.items;
  const dates = live?.telemetry?.dates ?? mock.dates;
  const agentSeries = live?.telemetry?.agent ?? mock.agent;
  const benchSeries = usingMock ? mock.bench : live.telemetry.bench;
  const sharpeSeries = live?.telemetry?.sharpe ?? mock.sharpe;
  const m = live ? adaptMetrics(live.statusRaw, live.telemetry) : mock.metrics;

  const toggleTheme = () => {
    const root = document.documentElement;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const current = root.getAttribute("data-theme") ?? (dark ? "dark" : "light");
    root.setAttribute("data-theme", current === "dark" ? "light" : "dark");
  };

  const forceCheck = async () => {
    setChecking(true);
    try {
      const result = await triggerEvaluation();
      if (result.ok) {
        setLive((prev) => (prev ? { ...prev, statusRaw: result.data, status: adaptStatus(result.data) } : prev));
      }
    } finally {
      setChecking(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil((decisionsView.total ?? decisions.length) / pageSize));

  return (
    <div className="wrap">
      <header className="masthead">
        <div>
          <h1>Energy Agent Telemetry</h1>
          <p>PPO trading agent · continuous training pipeline · XOM · CVX · SHEL · BP · NEE</p>
        </div>
        <div className="mast-right">
          {usingMock && (
            <span className="demo-chip">Example data — not live results</span>
          )}
          {!usingMock && stale && (
            <span className="stale">● Backend unreachable — showing last known</span>
          )}
          {!usingMock && (
            <button
              className="icon-btn"
              onClick={forceCheck}
              disabled={checking || status.pipeline !== "SERVING"}
              aria-label="Force a CT evaluation now"
              title="Force a CT evaluation now"
            >
              ⟳
            </button>
          )}
          <button className="icon-btn" onClick={toggleTheme}
                  aria-label="Switch theme" title="Switch theme">◐</button>
        </div>
      </header>

      <StatusStrip
        pipeline={status.pipeline}
        model={status.model}
        lastIngest={status.lastIngest}
        vix={status.vix}
        failSafeThreshold={status.failSafeThreshold}
      />

      <div className="panel">
        <div className="col">
          <GlassCard>
            <CardHead
              title="Portfolio equity"
              sub={
                benchSeries
                  ? "Agent versus equal-weight buy & hold, net of 10 bps costs"
                  : "Simulated by replaying the incumbent against real market data (CLAUDE.md §6)"
              }
              right={
                benchSeries && (
                  <div className="legend">
                    <span className="legend-item">
                      <span className="swatch" style={{ background: "var(--series-agent)" }} />
                      Agent
                    </span>
                    <span className="legend-item">
                      <span className="swatch dash" />Buy &amp; hold
                    </span>
                  </div>
                )
              }
            />
            <EquityChart dates={dates} agent={agentSeries} bench={benchSeries} />
          </GlassCard>

          <GlassCard>
            <CardHead
              title="Rolling Sharpe"
              sub={`Retraining triggers below ${status.targetSharpe.toFixed(2)} — the threshold the orchestrator watches`}
            />
            <SharpeChart dates={dates} values={sharpeSeries} threshold={status.targetSharpe} />
          </GlassCard>
        </div>

        <div className="col">
          <GlassCard className="gauge-card">
            <h2 className="card-title">Rolling Sharpe</h2>
            <div className="gauge-wrap">
              <Gauge fraction={m.rollingSharpe != null ? (m.rollingSharpe + 1) / 4 : 0} colour="var(--good)" />
              <div className="gauge-read">
                <div className="gauge-val num">{m.rollingSharpe != null ? m.rollingSharpe.toFixed(2) : "—"}</div>
                <div className="gauge-meta">
                  {m.rollingSharpe != null
                    ? `Threshold ${status.targetSharpe.toFixed(2)} · headroom ${(m.rollingSharpe - status.targetSharpe).toFixed(2)}`
                    : "No evaluation has run yet"}
                </div>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="gauge-card">
            <h2 className="card-title">Maximum drawdown</h2>
            <div className="gauge-wrap">
              <Gauge fraction={m.maxDrawdown != null ? Math.abs(m.maxDrawdown) / 50 : 0} colour="var(--warn)" />
              <div className="gauge-read">
                <div className="gauge-val num">{m.maxDrawdown != null ? `${m.maxDrawdown.toFixed(1)}%` : "—"}</div>
                <div className="gauge-meta">Worst peak-to-trough this period</div>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="gauge-card">
            <h2 className="card-title">Cycle history</h2>
            <div className="gauge-wrap">
              <div className="gauge-read" style={{ flex: 1 }}>
                <div className="gauge-val num" style={{ fontSize: 26 }}>
                  {m.trainingRuns}
                </div>
                <div className="gauge-meta">
                  Training runs, last {m.windowDays} days
                </div>
              </div>
            </div>
            <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Pill tone="good">{m.promoted} promoted</Pill>
              <Pill tone="warn">{m.heldBack} held back</Pill>
            </div>
          </GlassCard>
        </div>
      </div>

      <GlassCard style={{ marginTop: 14 }}>
        <CardHead
          title="Autonomous decision log"
          sub="Every action traced to the model version, run and data partition that produced it"
        />
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Timestamp</th>
                <th scope="col">Ticker</th>
                <th scope="col">Raw weight</th>
                <th scope="col">Action</th>
                <th scope="col">VIX</th>
                <th scope="col">Fail-safe</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((r, i) => (
                <tr
                  key={r.ts + r.ticker + i}
                  tabIndex={0}
                  onClick={() => setSelected(i)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(i);
                    }
                  }}
                >
                  <td className="num">{r.ts}</td>
                  <td className="tick">{r.ticker}</td>
                  <td className="num">{r.weight}</td>
                  <td className={`act ${actionClass(r.action)}`}>{r.action}</td>
                  <td className="num">{r.vix}</td>
                  <td>
                    {r.failsafe
                      ? <Pill tone="crit">Triggered</Pill>
                      : <span style={{ color: "var(--ink-3)" }}>No</span>}
                  </td>
                </tr>
              ))}
              {decisions.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: "center", color: "var(--ink-3)", padding: "24px 18px" }}>
                    No decisions logged yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="foot">
          <span>
            {decisions.length === 0
              ? "Showing 0 of 0"
              : `Showing ${(page - 1) * pageSize + 1}–${(page - 1) * pageSize + decisions.length} of ${decisionsView.total}`}
          </span>
          {!usingMock && (
            <div className="pager">
              <button className="pg" aria-label="Previous page" disabled={page <= 1}
                      onClick={() => setPage((p) => Math.max(1, p - 1))}>‹</button>
              <button className="pg" aria-current="true">{page}</button>
              <button className="pg" aria-label="Next page" disabled={page >= totalPages}
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>›</button>
            </div>
          )}
        </div>
      </GlassCard>

      {usingMock && (
        <p className="note">
          <b>Figures on this page are generated example data</b>, shown so the interface
          can be judged before the pipeline is wired to it. The agent series is not a
          backtest result and carries no claim about profitability.
        </p>
      )}

      <Sheet
        open={selected !== null}
        onClose={() => setSelected(null)}
        title="Decision detail"
        subtitle={
          selected === null
            ? ""
            : `${decisions[selected].ticker} · ${decisions[selected].ts}`
        }
      >
        {selected !== null && <DecisionDetail row={decisions[selected]} />}
      </Sheet>
    </div>
  );
}

/** NFR-07 made visible: decision → version → run → data partition. */
function DecisionDetail({ row }) {
  const kv = [
    ["Action", <span className={`act ${actionClass(row.action)}`}>{row.action}</span>],
    ["Raw weight", <span className="num">{row.weight}</span>],
    ["VIX at decision", <span className="num">{row.vix}</span>],
    ["Model version", <span className="num">{row.version}</span>],
    ["Training run", <span className="num">{row.run}</span>],
    ["Training partition", <span className="num">{row.trainPartition}</span>],
    ["Evaluation partition", <span className="num">{row.evalPartition}</span>],
  ];
  return (
    <>
      <div className="kv">
        {kv.map(([k, v]) => (
          <div className="kv-row" key={k}>
            <span className="kv-k">{k}</span>
            <span className="kv-v">{v}</span>
          </div>
        ))}
      </div>
      <div className="trace">
        {row.failsafe ? (
          <>
            The volatility fail-safe fired: VIX <b>{row.vix}</b> exceeded the critical
            threshold, so the request bypassed the agent entirely and returned a
            capital-preservation instruction (I5, FR-12).{" "}
            <b>No model weight was consulted.</b>
          </>
        ) : (
          <>
            Weight <b>{row.weight}</b> maps to <b>{row.action}</b> under the decision
            thresholds (buy above 0.5, sell below −0.5, otherwise hold — FR-11). VIX{" "}
            <b>{row.vix}</b> sat below the fail-safe threshold, so the agent's output
            was served unchanged.
          </>
        )}
      </div>
    </>
  );
}
