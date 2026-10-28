/**
 * Pipeline activity timeline (FR-20 as a history, NFR-07).
 *
 * Shows what the backend did on its own, from `/events`: data fetched,
 * model checked, retrain started, candidate promoted or rejected. It is the
 * part of the dashboard that makes the autonomous loop visible rather than
 * implied by the status strip.
 *
 * The CT check runs every 5 minutes, so raw events would bury everything
 * else. `groupEvents` collapses runs of routine checks into one line, and
 * repeats of the same retrain cycle (CLAUDE.md D-15) into one line with a
 * count, without hiding that they happened.
 */
import { useState } from "react";
import { GlassCard } from "./Primitives.jsx";

const KINDS = {
  ingestion_ok: { tone: "good", label: "Data fetched" },
  ingestion_failed: { tone: "crit", label: "Fetch failed" },
  evaluation: { tone: "good", label: "Model checked" },
  evaluation_skipped: { tone: "warn", label: "Check skipped" },
  retrain_started: { tone: "warn", label: "Retraining" },
  retrain_deferred: { tone: "warn", label: "Retrain deferred" },
  candidate_promoted: { tone: "good", label: "Model promoted" },
  candidate_rejected: { tone: "idle", label: "Candidate rejected" },
  retrain_failed: { tone: "crit", label: "Retrain failed" },
  paper_cycle: { tone: "idle", label: "Paper cycle" },
};

export const kindInfo = (kind) => KINDS[kind] ?? { tone: "idle", label: kind };

const sameEvent = (a, b) =>
  a.kind === b.kind && a.summary === b.summary && a.source === b.source;

function blockRepeats(events, start, size, reps) {
  const next = start + reps * size;
  if (next + size > events.length) return false;
  for (let k = 0; k < size; k += 1) {
    if (!sameEvent(events[start + k], events[next + k])) return false;
  }
  return true;
}

function single(e) {
  return { ...kindInfo(e.kind), key: `e${e.id}`, kind: e.kind, summary: e.summary,
           source: e.source, at: e.at, from: e.at, count: 1 };
}

function evaluationRun(run) {
  const values = run.map((e) => e.detail?.rolling_sharpe).filter((v) => typeof v === "number");
  const target = run[0].detail?.target;
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  const range = !values.length
    ? "rolling Sharpe"
    : lo.toFixed(2) === hi.toFixed(2)
      ? `rolling Sharpe steady at ${lo.toFixed(2)}`
      : `rolling Sharpe between ${lo.toFixed(2)} and ${hi.toFixed(2)}`;
  const vsTarget = typeof target === "number" ? `, at or above the ${target.toFixed(2)} target` : "";
  return {
    ...kindInfo("evaluation"),
    key: `g${run[0].id}`,
    kind: "evaluation",
    summary: `Checked ${run.length} times: ${range}${vsTarget}. Model kept serving.`,
    source: run[0].source,
    at: run[0].at,
    from: run[run.length - 1].at,
    count: run.length,
  };
}

/**
 * `events` newest first, as /events returns them. Returns display rows,
 * newest first: consecutive routine checks become one row, and a block of
 * one or two events repeated back to back becomes one row with a count.
 */
export function groupEvents(events) {
  const rows = [];
  let i = 0;
  while (i < events.length) {
    const e = events[i];
    if (e.kind === "evaluation") {
      let j = i;
      while (j < events.length && events[j].kind === "evaluation" && events[j].source === e.source) j += 1;
      const run = events.slice(i, j);
      rows.push(run.length === 1 ? single(e) : evaluationRun(run));
      i = j;
      continue;
    }
    let grouped = false;
    for (const size of [1, 2]) {
      let reps = 1;
      while (blockRepeats(events, i, size, reps)) reps += 1;
      if (reps > 1) {
        const block = events.slice(i, i + size);
        const all = events.slice(i, i + reps * size);
        rows.push({
          ...kindInfo(e.kind),
          key: `r${e.id}`,
          kind: e.kind,
          // Oldest first reads as cause then effect: "Retraining → Candidate rejected".
          label: block.map((b) => kindInfo(b.kind).label).reverse().join(" → "),
          summary: e.summary,
          source: e.source,
          at: e.at,
          from: all[all.length - 1].at,
          count: reps,
        });
        i += reps * size;
        grouped = true;
        break;
      }
    }
    if (!grouped) {
      rows.push(single(e));
      i += 1;
    }
  }
  return rows;
}

const pad = (n) => String(n).padStart(2, "0");
const clock = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
const dayMonth = (d) => d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });

export function relativeTime(date, now = new Date()) {
  const s = Math.round((now - date) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${dayMonth(date)} ${clock(date)}`;
}

function when(row, now) {
  if (row.count === 1 || row.at - row.from < 60000) return relativeTime(row.at, now);
  const sameDay = row.at.toDateString() === row.from.toDateString();
  return sameDay
    ? `${clock(row.from)}–${clock(row.at)}${now - row.at < 86400000 ? "" : ` ${dayMonth(row.at)}`}`
    : `${dayMonth(row.from)} – ${relativeTime(row.at, now)}`;
}

const VISIBLE = 6;

/** `limit` rows at first. With `onSeeAll` (the Overview's short version),
 * the footer links to the full timeline instead of expanding in place. */
export function ActivityCard({ events, now = new Date(), limit = VISIBLE, onSeeAll }) {
  const [expanded, setExpanded] = useState(false);
  const rows = events ? groupEvents(events) : [];
  const shown = expanded ? rows : rows.slice(0, limit);
  const oldest = events?.length ? events[events.length - 1].at : null;

  return (
    <GlassCard className="gauge-card activity-card">
      <h2 className="card-title">Pipeline activity</h2>
      <p className="card-sub">What the pipeline did on its own, newest first</p>
      {events === null ? (
        <div className="gauge-meta" style={{ marginTop: 10 }}>Waiting for the backend…</div>
      ) : rows.length === 0 ? (
        <p className="gauge-meta" style={{ marginTop: 10 }}>
          No activity recorded yet. Each data fetch, model check and retrain appears here as it
          happens; the model is checked every few minutes.
        </p>
      ) : (
        <>
          <ol className="activity" aria-label="Pipeline activity, newest first">
            {shown.map((row) => (
              <li key={row.key} className={`act ${row.tone}`}>
                <span className="act-dot" aria-hidden="true" />
                <div className="act-body">
                  <div className="act-head">
                    <span className="act-label">{row.label}</span>
                    {row.count > 1 && <span className="act-count">×{row.count}</span>}
                    {row.source === "manual" && <span className="act-tag">by operator</span>}
                    <time className="act-time" dateTime={row.at.toISOString()} title={row.at.toLocaleString("en-GB")}>
                      {when(row, now)}
                    </time>
                  </div>
                  <p className="act-summary">{row.summary}</p>
                </div>
              </li>
            ))}
          </ol>
          <div className="act-foot">
            {oldest && <span>Recorded since {dayMonth(oldest)} {clock(oldest)}</span>}
            {onSeeAll ? (
              <button className="text-btn" onClick={onSeeAll}>See all in Pipeline</button>
            ) : rows.length > limit && (
              <button className="text-btn" aria-expanded={expanded} onClick={() => setExpanded((v) => !v)}>
                {expanded ? "Show less" : `Show all ${rows.length}`}
              </button>
            )}
          </div>
        </>
      )}
    </GlassCard>
  );
}
