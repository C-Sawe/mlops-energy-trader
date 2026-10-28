/**
 * Charts and tables the assistant shows beside its answers.
 *
 * Each one is drawn from a tool's `visual` payload: rows the backend read
 * from the database (src/serving/assistant.py), never numbers the model
 * wrote. Where it helps, items are clickable and ask a follow-up question
 * through `onAsk`, so the conversation can be explored by pointing.
 */
import { CandleChart, EquityChart, SharpeChart } from "./Charts.jsx";
import { groupEvents, relativeTime } from "./Activity.jsx";
import { NewsList, NEWS_NOTE } from "./News.jsx";
import { Pill, pipelineToneAndLabel } from "./Primitives.jsx";

const short = (id) => (id ? String(id).slice(0, 8) : "—");
const num = (v, places = 2) => (v == null ? "—" : Number(v).toFixed(places));
const pct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}%`);
const day = (iso) =>
  iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "—";
const when = (iso) =>
  iso ? new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "—";

const ACTION_TONE = { BUY: "good", SELL: "crit", HOLD: "idle", LIQUIDATE: "warn" };
export const TRIGGER_LABELS = {
  ct_decay: "Retrain after performance fell below target",
  ct_bootstrap: "First model, trained because none was serving",
  manual_sweep: "Manual training sweep",
  analysis: "Analysis run",
};
export const triggerLabel = (reason) => TRIGGER_LABELS[reason] ?? "Unknown (logged before triggers were recorded)";

/** Click-to-ask on any element: mouse, Enter or Space. */
const askable = (onAsk, question, label) =>
  onAsk
    ? {
        role: "button",
        tabIndex: 0,
        title: label ?? question,
        onClick: () => onAsk(question),
        onKeyDown: (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onAsk(question);
          }
        },
      }
    : {};

function Frame({ title, note, children, source = "From the records", sourceTitle = "Drawn from the database, not written by the model" }) {
  return (
    <figure className="av">
      <figcaption>
        <span className="av-title">{title}</span>
        <span className="av-src" title={sourceTitle}>{source}</span>
      </figcaption>
      {children}
      {note && <p className="av-note">{note}</p>}
    </figure>
  );
}

/** A horizontal track with a fill to `value` and a tick at `marker`. */
export function Meter({ value, marker, lo, hi, tone = "good", label }) {
  const span = hi - lo || 1;
  const at = (v) => `${Math.max(0, Math.min(100, ((v - lo) / span) * 100))}%`;
  return (
    <div className="meter" role="img" aria-label={label}>
      <div className={`meter-fill ${tone}`} style={{ width: value == null ? 0 : at(value) }} />
      {marker != null && <div className="meter-mark" style={{ left: at(marker) }} />}
    </div>
  );
}

function StatusVisual({ status }) {
  const { tone, label } = pipelineToneAndLabel(status.status);
  const sharpe = status.rolling_sharpe;
  const target = status.target_sharpe_threshold;
  const vix = status.current_vix;
  const vixLimit = status.vix_critical_threshold;
  return (
    <Frame title="Pipeline status">
      <div className="av-tiles">
        <div className="av-tile"><span className="k">Pipeline</span><Pill tone={tone}>{label}</Pill></div>
        <div className="av-tile">
          <span className="k">Active model</span>
          <span className="v" title={status.active_version_id ?? undefined}>
            {status.active_model_label ?? short(status.active_version_id)}
          </span>
        </div>
        <div className="av-tile wide">
          <span className="k">Rolling Sharpe {sharpe != null && target != null && (sharpe >= target ? "· above target" : "· below target")}</span>
          <span className="v num">{num(sharpe)} <small>target {num(target)}</small></span>
          {sharpe == null && <small className="av-muted">Not checked since the server last started.</small>}
          {sharpe != null && target != null && (
            <Meter value={sharpe} marker={target} lo={Math.min(0, sharpe)} hi={Math.max(target * 2, sharpe * 1.1)}
                   tone={sharpe >= target ? "good" : "warn"} label={`Rolling Sharpe ${num(sharpe)} against target ${num(target)}`} />
          )}
        </div>
        <div className="av-tile wide">
          <span className="k">VIX {vix != null && vixLimit != null && (vix >= vixLimit ? "· fail-safe active" : "· fail-safe inactive")}</span>
          <span className="v num">{num(vix, 1)} <small>limit {num(vixLimit, 1)}</small></span>
          {vix != null && vixLimit != null && (
            <Meter value={vix} marker={vixLimit} lo={0} hi={Math.max(vixLimit * 1.4, vix * 1.1)}
                   tone={vix >= vixLimit ? "crit" : "good"} label={`VIX ${num(vix, 1)} against limit ${num(vixLimit, 1)}`} />
          )}
        </div>
        <div className="av-tile"><span className="k">Latest data</span><span className="v">{day(status.last_ingest_date)}</span></div>
        <div className="av-tile"><span className="k">Next fetch</span><span className="v">{when(status.next_ingestion_at)}</span></div>
        <div className="av-tile wide">
          <span className="k">Training, last {status.cycle_window_days ?? "—"} days</span>
          <span className="v num">{status.training_runs ?? "—"} runs <small>{status.promoted_count ?? 0} promoted · {status.rejected_count ?? 0} held back</small></span>
        </div>
      </div>
    </Frame>
  );
}

function TimelineVisual({ events, onAsk }) {
  const rows = groupEvents(events.map((e) => ({
    id: e.event_id, at: new Date(e.occurred_at), kind: e.kind, source: e.source, summary: e.summary, detail: e.detail ?? {},
  })));
  if (!rows.length) return <Frame title="Activity log"><p className="av-empty">No activity recorded yet.</p></Frame>;
  const now = new Date();
  return (
    <Frame title="Activity log" note="Click an entry to ask about it.">
      <ol className="activity av-activity">
        {rows.slice(0, 10).map((row) => {
          const id = row.key.slice(1);
          return (
            <li key={row.key} className={`act ${row.tone} askable`} {...askable(onAsk, `Explain event #${id} in more detail.`)}>
              <span className="act-dot" aria-hidden="true" />
              <div className="act-body">
                <div className="act-head">
                  <span className="act-label">{row.label}</span>
                  {row.count > 1 && <span className="act-count">×{row.count}</span>}
                  {row.source === "manual" && <span className="act-tag">by operator</span>}
                  <span className="act-id">#{id}</span>
                  <time className="act-time" dateTime={row.at.toISOString()}>{relativeTime(row.at, now)}</time>
                </div>
                <p className="act-summary">{row.summary}</p>
              </div>
            </li>
          );
        })}
      </ol>
    </Frame>
  );
}

function PerformanceVisual({ points, target_sharpe: target }) {
  const dates = points.map((p) => new Date(p.date));
  const equity = points.map((p) => p.equity);
  const bench = points.some((p) => p.benchmark != null) ? points.map((p) => p.benchmark ?? null) : null;
  const firstBench = points.find((p) => p.benchmark != null);
  const lastBench = [...points].reverse().find((p) => p.benchmark != null);
  const modelReturn = (equity[equity.length - 1] / equity[0] - 1) * 100;
  const benchReturn = firstBench && lastBench && firstBench !== lastBench
    ? (lastBench.benchmark / firstBench.benchmark - 1) * 100 : null;
  // Only days that have a rolling Sharpe: a missing value is not drawn as zero.
  const withSharpe = points.filter((p) => p.sharpe != null);
  const latest = withSharpe.length ? withSharpe[withSharpe.length - 1].sharpe : null;
  return (
    <Frame title="Performance against buy-and-hold"
           note="Simulated replay of the serving model over real prices, with transaction costs.">
      <div className="av-chips">
        <span className="av-chip"><span className="k">Model</span><b className="num">{pct(modelReturn)}</b></span>
        <span className="av-chip"><span className="k">Buy-and-hold</span><b className="num">{pct(benchReturn)}</b></span>
        <span className="av-chip"><span className="k">Rolling Sharpe now</span><b className="num">{num(latest)}</b></span>
        <span className="av-chip"><span className="k">Days</span><b className="num">{points.length}</b></span>
      </div>
      <EquityChart dates={dates} agent={equity} bench={bench} label="Model" />
      {withSharpe.length > 1 && (
        <SharpeChart dates={withSharpe.map((p) => new Date(p.date))} values={withSharpe.map((p) => p.sharpe)}
                     threshold={target ?? 1} />
      )}
    </Frame>
  );
}

function DecisionsVisual({ items, onAsk }) {
  if (!items.length) return <Frame title="Recent decisions"><p className="av-empty">No decisions recorded yet.</p></Frame>;
  return (
    <Frame title="Recent decisions" note="Click a decision to trace it to its model and data.">
      <div className="av-table-wrap">
        <table className="av-table">
          <thead><tr><th>When</th><th>Ticker</th><th>Weight</th><th>Action</th><th>VIX</th><th>Model</th></tr></thead>
          <tbody>
            {items.map((d) => (
              <tr key={d.decision_id} className="askable"
                  {...askable(onAsk, `Trace decision ${d.decision_id} to the model and data behind it.`, "Trace this decision")}>
                <td>{when(d.decided_at)}</td>
                <td><b>{d.ticker}</b></td>
                <td className="num">{num(d.raw_weight, 3)}</td>
                <td><Pill tone={ACTION_TONE[d.action] ?? "idle"}>{d.action}</Pill>{d.failsafe && <span className="av-flag">fail-safe</span>}</td>
                <td className="num">{num(d.vix, 1)}</td>
                <td className="mono">{short(d.version_id)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Frame>
  );
}

function WindowBar({ d }) {
  const t0 = new Date(d.train_start).getTime();
  const t1 = new Date(d.train_end).getTime();
  const e0 = new Date(d.eval_start).getTime();
  const e1 = new Date(d.eval_end).getTime();
  const span = e1 - t0 || 1;
  const left = (t) => `${((t - t0) / span) * 100}%`;
  return (
    <div className="av-windows" role="img"
         aria-label={`Trained ${day(d.train_start)} to ${day(d.train_end)}, evaluated ${day(d.eval_start)} to ${day(d.eval_end)}`}>
      <div className="av-win train" style={{ left: 0, width: left(t1) }} />
      <div className="av-win eval" style={{ left: left(e0), width: `calc(100% - ${left(e0)})` }} />
      <div className="av-win-labels">
        <span><i className="sw train" />Train {day(d.train_start)} – {day(d.train_end)}</span>
        <span><i className="sw eval" />Evaluate {day(d.eval_start)} – {day(d.eval_end)}</span>
      </div>
    </div>
  );
}

function ProvenanceVisual({ decision: d, onAsk }) {
  const hyper = Object.entries(d.hyperparameters ?? {}).filter(([k]) => k !== "device");
  return (
    <Frame title="Decision provenance" note="Every decision traces to the model that made it and the data that trained it.">
      <div className="av-chain">
        <div className="av-node">
          <span className="k">Decision</span>
          <span className="v"><b>{d.ticker}</b> <Pill tone={ACTION_TONE[d.discrete_action] ?? "idle"}>{d.discrete_action}</Pill></span>
          <span className="s">weight {num(d.raw_weight, 3)} · VIX {num(d.vix_at_decision, 1)}{d.failsafe_triggered ? " · fail-safe" : ""}</span>
          <span className="s">{when(d.decided_at)}</span>
        </div>
        <span className="av-arrow" aria-hidden="true">→</span>
        <div className="av-node askable" {...askable(onAsk, `Tell me about model version ${short(d.version_id)}.`)}>
          <span className="k">Model version</span>
          <span className="v mono">{short(d.version_id)}{d.version_is_active && <span className="av-badge">active</span>}</span>
          <span className="s">promoted {day(d.version_promoted_at)}</span>
        </div>
        <span className="av-arrow" aria-hidden="true">→</span>
        <div className="av-node">
          <span className="k">Training run</span>
          <span className="v mono">{short(d.run_id)}</span>
          <span className="s">{triggerLabel(d.trigger_reason)}</span>
          <span className="s">out-of-sample Sharpe {num(d.sharpe_ratio)}</span>
        </div>
      </div>
      <WindowBar d={d} />
      {hyper.length > 0 && (
        <div className="av-chips">
          {hyper.map(([k, v]) => <span className="av-chip small" key={k}><span className="k">{k}</span><b>{String(v)}</b></span>)}
        </div>
      )}
    </Frame>
  );
}

function ModelsVisual({ items, onAsk }) {
  const models = [...items].filter((m) => m.promoted_at).reverse(); // oldest promotion first
  if (!models.length) return <Frame title="Model history"><p className="av-empty">No model has been promoted yet.</p></Frame>;
  const values = models.map((m) => m.sharpe_ratio ?? 0);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0.5, ...values);
  const W = 640, H = 210, pad = { l: 36, r: 12, t: 22, b: 34 };
  const bw = (W - pad.l - pad.r) / models.length;
  const y = (v) => pad.t + ((hi - v) / (hi - lo)) * (H - pad.t - pad.b);
  return (
    <Frame title="Model history"
           note="Each bar is a promoted model's Sharpe on its own evaluation window when it was selected. Windows differ, so compare with care. Click a bar to ask about it.">
      <svg className="chart av-bars" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Out-of-sample Sharpe of each promoted model">
        <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} stroke="var(--grid)" />
        <text x={pad.l - 6} y={y(hi) + 4} textAnchor="end" className="tick">{hi.toFixed(1)}</text>
        <text x={pad.l - 6} y={y(0) + 4} textAnchor="end" className="tick">0</text>
        {models.map((m, i) => {
          const v = m.sharpe_ratio ?? 0;
          const x = pad.l + i * bw + bw * 0.18;
          const top = Math.min(y(v), y(0));
          return (
            <g key={m.version_id} className="askable" {...askable(onAsk, `Tell me about model version ${short(m.version_id)}.`, `Version ${short(m.version_id)}: Sharpe ${num(m.sharpe_ratio)}`)}>
              <rect x={x} y={top} width={bw * 0.64} height={Math.max(1.5, Math.abs(y(v) - y(0)))} rx="4"
                    fill={m.is_active ? "var(--series-agent)" : "color-mix(in srgb, var(--series-agent) 32%, transparent)"} />
              <text x={x + bw * 0.32} y={top - 6} textAnchor="middle" className="tick strong">{num(m.sharpe_ratio)}</text>
              <text x={x + bw * 0.32} y={H - pad.b + 16} textAnchor="middle" className="tick">
                {new Date(m.promoted_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}
              </text>
              {m.is_active && <text x={x + bw * 0.32} y={H - 4} textAnchor="middle" className="tick accent">active</text>}
            </g>
          );
        })}
      </svg>
    </Frame>
  );
}

function TrainingVisual({ days, stats }) {
  const total = stats.total_runs || 0;
  const promoted = stats.promoted || 0;
  const rejected = stats.rejected || 0;
  const other = Math.max(0, total - promoted - rejected);
  const share = (n) => (total ? `${(n / total) * 100}%` : "0%");
  return (
    <Frame title={`Training activity, last ${days} days`}
           note="Counts are training runs, not distinct models: an unchanged candidate can be retrained many times.">
      <div className="av-chips">
        <span className="av-chip"><span className="k">Runs</span><b className="num">{total}</b></span>
        <span className="av-chip"><span className="k">Promoted</span><b className="num">{promoted}</b></span>
        <span className="av-chip"><span className="k">Held back by the gate</span><b className="num">{rejected}</b></span>
        <span className="av-chip"><span className="k">Started by the pipeline</span><b className="num">{stats.autonomous_runs ?? 0}</b></span>
      </div>
      <div className="av-stack" role="img" aria-label={`${promoted} promoted, ${rejected} held back, ${other} other runs`}>
        <span className="good" style={{ width: share(promoted) }} />
        <span className="warn" style={{ width: share(rejected) }} />
        <span className="idle" style={{ width: share(other) }} />
      </div>
      <div className="av-legend">
        <span><i className="sw good" />Promoted</span><span><i className="sw warn" />Held back</span><span><i className="sw idle" />Other (sweeps, analysis)</span>
      </div>
    </Frame>
  );
}

function CandlesVisual({ ticker, candles }) {
  const rows = candles.map((c) => ({ date: c.date, open: c.open, high: c.high, low: c.low, close: c.close, imputed: Boolean(c.is_imputed) }));
  return (
    <Frame title={`${ticker} daily prices`} note="Split- and dividend-adjusted closes.">
      <CandleChart ticker={ticker} candles={rows} lastPrice={rows.length ? rows[rows.length - 1].close : null} />
    </Frame>
  );
}

function NewsVisual({ ticker, days, items }) {
  return (
    <Frame title={`Headlines · ${ticker ?? "all five"} · last ${days} days`} note={NEWS_NOTE}
           source="From the news feed" sourceTitle="Alpaca's news feed, not written by the model">
      <NewsList items={items} />
    </Frame>
  );
}

const VISUALS = {
  status: StatusVisual,
  timeline: TimelineVisual,
  news: NewsVisual,
  performance: PerformanceVisual,
  decisions: DecisionsVisual,
  provenance: ProvenanceVisual,
  models: ModelsVisual,
  training: TrainingVisual,
  candles: CandlesVisual,
};

export function Visual({ visual, onAsk }) {
  const Component = VISUALS[visual?.type];
  return Component ? <Component {...visual} onAsk={onAsk} /> : null;
}
