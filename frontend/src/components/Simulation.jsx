/**
 * Pipeline simulation: replays stored market history through the real
 * pipeline code in a sandbox process, so an audience can watch the loop work.
 *
 * Every number comes from the backend's sandbox. The page says throughout
 * that it is a replay, marks whatever the viewer injected, and shows no
 * profit figure: it demonstrates the architecture (CLAUDE.md §1).
 */
import { useEffect, useRef, useState } from "react";
import { fetchSimulation, poll, sendSimulation, startSimulation, stopSimulation } from "../api.js";
import { SharpeChart } from "./Charts.jsx";
import { CardHead, GlassCard, Pill } from "./Primitives.jsx";
import { groupEvents } from "./Activity.jsx";

const POLL_MS = 700;

export const STAGES = [
  { id: "data", label: "Data", note: "Store the day's bar" },
  { id: "decide", label: "Decide", note: "Fail-safe, then model" },
  { id: "evaluate", label: "Check", note: "30-day Sharpe vs target" },
  { id: "retrain", label: "Retrain", note: "Train, then gate" },
];

/** Whether a stage is running, was used on the latest day, or neither. A
 * day that needed no retrain must not show that stage as done. */
export function stageState(id, stage, day, events) {
  if (stage === id) return "active";
  const order = STAGES.findIndex((s) => s.id === id);
  const at = STAGES.findIndex((s) => s.id === stage);
  if (at >= 0) return order < at ? "done" : "idle";
  if (!day) return "idle";
  if (id !== "retrain") return "done";
  return events.some((e) => e.day === day.day && e.kind === "retrain_started") ? "done" : "idle";
}

/** One sentence on what the latest day showed, from its records only. */
export function explainDay(day, events) {
  if (!day) return "The sandbox is ready. Step one trading day, or press Play.";
  const kinds = events.filter((e) => e.day === day.day).map((e) => e.kind);
  const parts = [];
  if (day.failsafe_reason === "operator") parts.push("Trading is halted by the viewer, so every ticker was LIQUIDATE and the model was not consulted.");
  else if (day.failsafe_reason === "vix") parts.push(`VIX ${day.vix.toFixed(1)}${day.vix_injected ? " (injected)" : ""} tripped the fail-safe: every ticker was LIQUIDATE and the model was not consulted.`);
  else parts.push("The model chose the weights below.");
  if (kinds.includes("candidate_promoted")) parts.push("The check was below target, so a candidate was trained. It beat the current model on the gate's window and replaced it without a restart.");
  else if (kinds.includes("candidate_rejected")) parts.push("The check was below target, so a candidate was trained. It did not beat the current model, so the gate kept the current one.");
  else if (kinds.includes("candidate_withheld")) parts.push("A candidate won, but retraining was paused, so it was withheld.");
  else if (kinds.includes("retrain_failed")) parts.push("The retrain crashed. The current model kept serving.");
  else if (kinds.includes("retrain_deferred")) parts.push("A retrain was due but was deferred; the reason is in the log.");
  else if (kinds.includes("evaluation")) parts.push("The check met the target, so nothing was retrained.");
  return parts.join(" ");
}

function VixChart({ days, threshold }) {
  if (days.length < 2) return <div className="chart-empty">Two days are needed to draw VIX.</div>;
  const box = { w: 760, h: 170, l: 44, r: 18, t: 18, b: 22 };
  const top = Math.max(threshold * 1.2, ...days.map((d) => d.vix)) * 1.05;
  const x = (i) => box.l + (i * (box.w - box.l - box.r)) / (days.length - 1);
  const y = (v) => box.h - box.b - (v / top) * (box.h - box.t - box.b);
  const last = days[days.length - 1];
  return (
    <svg className="chart" viewBox={`0 0 ${box.w} ${box.h}`} role="img"
         aria-label={`VIX by simulated day, last ${last.vix.toFixed(1)}, fail-safe threshold ${threshold.toFixed(0)}`}>
      {[0, threshold].map((v) => (
        <text key={v} x={box.l - 10} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)">{v.toFixed(0)}</text>
      ))}
      <line x1={box.l} x2={box.w - box.r} y1={y(0)} y2={y(0)} stroke="var(--grid)" />
      <line x1={box.l} x2={box.w - box.r} y1={y(threshold)} y2={y(threshold)} stroke="var(--critical)" strokeWidth="2" strokeDasharray="6 5" />
      <text x={box.l + 8} y={y(threshold) - 7} fontSize="11" fontWeight="600" fill="var(--critical)">Fail-safe · {threshold.toFixed(0)}</text>
      <path d={days.map((d, i) => `${i ? "L" : "M"} ${x(i)} ${y(d.vix)}`).join(" ")} fill="none" stroke="var(--series-agent)" strokeWidth="2" />
      {days.map((d, i) => d.vix_injected && (
        <path key={d.day} d={`M ${x(i)} ${y(d.vix) - 6} l 6 6 l -6 6 l -6 -6 z`} fill="var(--warn)"><title>{`${d.day}: injected`}</title></path>
      ))}
      <text x={box.l} y={box.h - 5} fontSize="11" fill="var(--ink-3)">{days[0].day}</text>
      <text x={box.w - box.r} y={box.h - 5} fontSize="11" textAnchor="end" fill="var(--ink-3)">{last.day}</text>
    </svg>
  );
}

const failure = (r) =>
  r.reason === "conflict" ? r.detail?.detail ?? "The server refused." : "No answer from the backend. Check the state before retrying.";

export function SimulationPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [vix, setVix] = useState(50);
  const acted = useRef(0); // a poll that began before an action must not undo it

  const load = async () => {
    const at = acted.current;
    return { at, r: await fetchSimulation(true) };
  };
  useEffect(() => poll(load, POLL_MS, ({ at, r }) => { if (r.ok && at === acted.current) setData(r.data); }), []);

  const act = async (fn) => {
    acted.current += 1;
    setBusy(true);
    setError(null);
    try {
      const r = await fn();
      acted.current += 1;
      if (r.ok) setData(r.data);
      else setError(failure(r));
    } finally {
      setBusy(false);
    }
  };
  const send = (cmd, extra) => act(() => sendSimulation({ cmd, ...extra }));

  if (!data) return <GlassCard><CardHead title="Pipeline simulation" sub="Waiting for the backend…" /></GlassCard>;
  if (!data.enabled) {
    return (
      <GlassCard>
        <CardHead title="Pipeline simulation" sub="Off on this server. Set SIMULATION_ENABLED=true and restart the backend to use it." />
      </GlassCard>
    );
  }

  const s = data.state;
  const days = s.days ?? [];
  const day = days[days.length - 1] ?? null;
  const running = s.status === "running";
  const finished = running && s.total > 0 && days.length >= s.total;
  const working = running && !["idle", "finished"].includes(s.stage);
  const scenario = data.scenarios.find((x) => x.id === s.scenario);
  const flags = s.flags ?? {};
  const rows = groupEvents([...s.events].reverse().map((e) => ({ ...e, at: new Date(e.day ?? 0) })));
  const dayOf = (row) => { const e = s.events.find((x) => row.key === `e${x.id}` || row.key === `g${x.id}`); return e?.day ?? null; };

  return (
    <>
      <section className="sim-banner" role="note">
        <b>Sandbox replay.</b> Stored market history runs through the real pipeline code in a separate process with its
        own database. The live model, records and paper account are not touched. These windows were chosen to show the
        mechanism; nothing here is a result.
      </section>
      {error && <p className="ovr-error" role="alert">{error}</p>}

      {(s.status === "idle" || s.status === "failed") && (
        <GlassCard>
          <CardHead title="Choose a scenario" sub="The first model is trained when the sandbox starts, which takes several seconds." />
          {s.error && <p className="ovr-error sim-pad" role="alert">{s.error}</p>}
          <div className="ovr-body"><div className="ovr-list">
            {data.scenarios.map((x) => (
              <div className="ovr-row" key={x.id}>
                <div><div className="ovr-name">{x.title}</div><p className="ovr-does">{x.description}</p></div>
                <button className="ovr-btn primary" disabled={busy} onClick={() => act(() => startSimulation(x.id))}>Start</button>
              </div>
            ))}
          </div></div>
        </GlassCard>
      )}

      {(s.status === "starting" || running) && (
        <>
          <GlassCard>
            <CardHead
              title={scenario?.title ?? "Simulation"}
              sub={s.status === "starting" ? "Starting the sandbox and training its first model…"
                : `Trading day ${days.length} of ${s.total}${s.day ? ` · ${s.day}` : ""}`}
              right={<Pill tone={working || s.status === "starting" ? "warn" : "idle"}>
                {s.status === "starting" ? "Starting" : finished ? "Finished" : working ? "Working" : s.playing ? "Playing" : "Paused"}
              </Pill>}
            />
            <div className="ovr-body">
              <ol className="sim-stages" aria-label="Pipeline stages">
                {STAGES.map((st) => {
                  const state = stageState(st.id, s.stage, day, s.events);
                  return (
                    <li key={st.id} className={`sim-stage ${state}`} aria-current={state === "active" ? "step" : undefined}>
                      <b>{st.label}</b><span>{st.note}</span>
                      <em>{{ active: "Running", done: "Done", idle: "—" }[state]}</em>
                    </li>
                  );
                })}
              </ol>
              <p className="sim-explain" aria-live="polite">
                {s.serving && s.stage === "retrain"
                  ? `Training a candidate. Meanwhile the current model has answered ${s.serving.served} requests, ${s.serving.failed} failed.`
                  : s.status === "running" ? explainDay(day, s.events) : ""}
                {day?.served_during_retrain > 0 && s.stage !== "retrain" &&
                  ` While it trained, the current model answered ${day.served_during_retrain} requests, ${day.failed_during_retrain} failed.`}
              </p>
              <div className="ovr-actions sim-left">
                <button className="ovr-btn primary" disabled={!running || busy || finished || working || s.playing} onClick={() => send("step")}>Step one day</button>
                {s.playing
                  ? <button className="ovr-btn" disabled={busy} onClick={() => send("pause")}>Pause</button>
                  : <button className="ovr-btn" disabled={!running || busy || finished} onClick={() => send("play", { delay: 1 })}>Play</button>}
                <button className="ovr-btn danger" disabled={busy} onClick={() => act(stopSimulation)}>Stop and discard</button>
              </div>
            </div>
          </GlassCard>

          {running && (
            <div className="section-gap"><GlassCard>
              <CardHead title="Inject a fault or an override" sub="Takes effect on the next step. Injected VIX is marked; all other market data is stored history." />
              <div className="ovr-body"><div className="ovr-actions sim-left">
                <label className="sim-vix">VIX
                  <input type="number" min="5" max="100" value={vix} onChange={(e) => setVix(Number(e.target.value))} disabled={flags.vix_override != null} />
                </label>
                <button className="ovr-btn" aria-pressed={flags.vix_override != null} disabled={busy || !(vix >= 5 && vix <= 100)}
                        onClick={() => send("vix", { value: flags.vix_override != null ? null : vix })}>
                  {flags.vix_override != null ? `Stop injecting VIX ${flags.vix_override}` : "Inject VIX"}
                </button>
                <button className="ovr-btn" aria-pressed={Boolean(flags.crash_next)} disabled={busy} onClick={() => send("crash", { enabled: !flags.crash_next })}>
                  {flags.crash_next ? "Next retrain will crash (cancel)" : "Crash the next retrain"}
                </button>
                {[["halt_trading", "Halt trading", "Lift trading halt"], ["pause_retraining", "Pause retraining", "Lift retraining pause"]].map(([name, on, off]) => (
                  <button key={name} className="ovr-btn danger" aria-pressed={Boolean(flags[name])} disabled={busy}
                          onClick={() => send("control", { name, enabled: !flags[name] })}>{flags[name] ? off : on}</button>
                ))}
              </div></div>
            </GlassCard></div>
          )}

          {day && (
            <div className="panel section-gap">
              <div className="col">
                <GlassCard>
                  <CardHead title="Rolling Sharpe" sub="Replay of the current model over the latest 30 daily returns. A jump after a promotion is partly selection: the gate chose the candidate on almost the same days." />
                  <SharpeChart dates={days.map((d) => new Date(`${d.day}T00:00:00`))} values={days.map((d) => d.rolling_sharpe)} threshold={s.target_sharpe} />
                </GlassCard>
                <GlassCard>
                  <CardHead title="VIX" sub="Diamonds mark injected values" />
                  <VixChart days={days} threshold={s.vix_threshold} />
                </GlassCard>
              </div>
              <div className="col">
                <GlassCard>
                  <CardHead title={`Decisions · ${day.day}`} sub={`Model ${day.model ?? "—"} · gross exposure ${(day.exposure * 100).toFixed(0)}% of equity`} />
                  <div className="tablewrap"><table className="pos-table">
                    <thead><tr><th scope="col">Ticker</th><th scope="col">Weight</th><th scope="col">Action</th></tr></thead>
                    <tbody>{day.decisions.map((d) => (
                      <tr key={d.ticker}><td className="tick">{d.ticker}</td>
                        <td className="num">{d.weight == null ? "—" : d.weight.toFixed(3)}</td>
                        <td className={`act ${{ BUY: "buy", SELL: "sell", LIQUIDATE: "liq" }[d.action] ?? "hold"}`}>{d.action}</td></tr>
                    ))}</tbody>
                  </table></div>
                </GlassCard>
                <GlassCard className="gauge-card">
                  <h2 className="card-title">Sandbox log</h2>
                  <p className="card-sub">Written by the pipeline code, newest first</p>
                  <ol className="activity" aria-label="Sandbox log, newest first">
                    {rows.slice(0, 12).map((row) => (
                      <li key={row.key} className={`act ${row.tone}`}>
                        <span className="act-dot" aria-hidden="true" />
                        <div className="act-body">
                          <div className="act-head"><span className="act-label">{row.label}</span>
                            {row.count > 1 && <span className="act-count">×{row.count}</span>}
                            <span className="act-time">{dayOf(row) ?? "start-up"}</span></div>
                          <p className="act-summary">{row.summary}</p>
                        </div>
                      </li>
                    ))}
                  </ol>
                </GlassCard>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
