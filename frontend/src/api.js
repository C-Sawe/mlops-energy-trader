/**
 * Client for the FastAPI serving layer (IR-04: documented JSON over REST).
 *
 * IR-07 requires the dashboard to degrade gracefully when the backend is
 * unreachable — presenting a stale-data indication rather than failing
 * silently. Every call here therefore resolves to a shape the UI can render,
 * carrying `ok: false` rather than throwing into a blank screen.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const TOKEN = import.meta.env.VITE_API_BEARER_TOKEN;
const TIMEOUT_MS = 8000;

function authHeaders() {
  return TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};
}

async function get(path) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, {
      signal: ctrl.signal,
      headers: { Accept: "application/json", ...authHeaders() },
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return { ok: true, data: await res.json() };
  } catch (err) {
    // Distinguish "took too long" from "refused" — the analyst reads these
    // differently when judging whether the pipeline is healthy.
    const reason = err.name === "AbortError" ? "timeout" : "unreachable";
    return { ok: false, reason, error: String(err.message ?? err) };
  } finally {
    clearTimeout(timer);
  }
}

async function post(path, body) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, {
      method: "POST",
      signal: ctrl.signal,
      headers: { "Content-Type": "application/json", Accept: "application/json", ...authHeaders() },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 422) {
      // NFR-08: the service rejects malformed payloads with a description
      // rather than failing, so surface that description rather than a
      // generic error the analyst cannot act on.
      const detail = await res.json().catch(() => null);
      return { ok: false, reason: "invalid", detail };
    }
    if (res.status === 503) {
      const detail = await res.json().catch(() => null);
      return { ok: false, reason: "not_ready", detail };
    }
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return { ok: true, data: await res.json() };
  } catch (err) {
    const reason = err.name === "AbortError" ? "timeout" : "unreachable";
    return { ok: false, reason, error: String(err.message ?? err) };
  } finally {
    clearTimeout(timer);
  }
}

/** FR-20: serving / evaluating / retraining, the active model, cycle stats. */
export const fetchStatus = () => get("/ct-status");

/** FR-18: equity curve, rolling Sharpe and max drawdown, over `days` back from today. */
export function fetchTelemetry(days = 180) {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - days);
  const iso = (d) => d.toISOString().slice(0, 10);
  return get(`/telemetry?start=${iso(start)}&end=${iso(end)}`);
}

/** FR-19: paginated decision log. */
export const fetchDecisions = (page = 1, pageSize = 6) =>
  get(`/decisions?page=${page}&page_size=${pageSize}`);

/**
 * FR-10: request an action for a portfolio state. Included for completeness
 * — the dashboard is supervisory and does not normally drive inference.
 */
export const predict = (positions, cashWeight) =>
  post("/predict", { positions, cash_weight: cashWeight });

/** On-demand counterpart to the backend's scheduled CT evaluation (FR-13/14). */
export const triggerEvaluation = () => post("/ct/evaluate");

/** Poll helper — returns an unsubscribe function. */
export function poll(fn, ms, onData) {
  let alive = true;
  let timer;
  const tick = async () => {
    const result = await fn();
    if (!alive) return;
    onData(result);
    timer = setTimeout(tick, ms);
  };
  tick();
  return () => { alive = false; clearTimeout(timer); };
}
