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

/**
 * A 401 means the sign-in session ended (signed out elsewhere, expired, or
 * removed from the allowlist). AuthGate listens for this and shows the
 * login page instead of leaving the dashboard silently stale.
 */
export const AUTH_REQUIRED_EVENT = "auth:required";

function unauthorized() {
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
  return { ok: false, reason: "unauthorized" };
}

async function get(path) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, {
      signal: ctrl.signal,
      // The sign-in session is a cookie; "include" sends it in local dev,
      // where the dashboard (:5173) and API (:8000) are different origins.
      credentials: "include",
      headers: { Accept: "application/json", ...authHeaders() },
    });
    if (res.status === 401) return unauthorized();
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
      credentials: "include",
      headers: { "Content-Type": "application/json", Accept: "application/json", ...authHeaders() },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 401) return unauthorized();
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
    if (res.status === 204) return { ok: true, data: null };
    return { ok: true, data: await res.json() };
  } catch (err) {
    const reason = err.name === "AbortError" ? "timeout" : "unreachable";
    return { ok: false, reason, error: String(err.message ?? err) };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Whether sign-in is on, which providers are configured, and who (if
 * anyone) is signed in. Public, so it never triggers AUTH_REQUIRED_EVENT.
 */
export const fetchSession = () => get("/auth/session");

/** Full-page navigation, not fetch: the provider's consent screen takes over the tab. */
export const loginUrl = (provider) => `${BASE}/auth/login/${encodeURIComponent(provider)}`;

export const signOut = () => post("/auth/logout");

/**
 * Email-and-password sign-in. Not `post()`: a 401 here means "wrong
 * password", not "session ended", so it must not fire AUTH_REQUIRED_EVENT,
 * and the server's message is what the form shows.
 */
export async function passwordLogin(email, password) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}/auth/password/login`, {
      method: "POST",
      signal: ctrl.signal,
      credentials: "include",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const body = await res.json().catch(() => null);
    if (res.ok) return { ok: true, data: body };
    const detail = typeof body?.detail === "string" ? body.detail : null;
    return { ok: false, reason: res.status === 429 ? "throttled" : "rejected", detail };
  } catch (err) {
    const reason = err.name === "AbortError" ? "timeout" : "unreachable";
    return { ok: false, reason, error: String(err.message ?? err) };
  } finally {
    clearTimeout(timer);
  }
}

/** FR-20: serving / evaluating / retraining, the active model, cycle stats. */
export const fetchStatus = () => get("/ct-status");

/**
 * FR-18: equity curve, rolling Sharpe and max drawdown, over `days` back
 * from `endDate` (defaults to today).
 *
 * `endDate` should normally be `/ct-status`'s `last_ingest_date`, not a bare
 * `new Date()` — anchoring to calendar-today means the window silently goes
 * blank after any ingestion gap (a weekend the market was still open in some
 * other sense, a brief outage, DR-04's forward-fill catching up), even
 * though the data the chart actually cares about hasn't changed. Anchoring
 * to the latest data that actually exists makes "last N days" mean the same
 * thing whether ingestion is perfectly current or a few days behind.
 */
export function fetchTelemetry(days = 180, endDate) {
  const end = endDate ? new Date(endDate) : new Date();
  const start = new Date(end);
  // setUTCDate, not setDate: local-time calendar arithmetic on a date
  // parsed as UTC midnight is off by one day whenever the `days`-wide
  // window straddles a DST transition in the viewer's timezone (found
  // writing this file's tests — e.g. a 30-day window ending 2026-11-15
  // computed a start of 2026-10-15 in America/New_York, not the correct
  // 2026-10-16, because the UTC-to-local offset used going in differs
  // from the one implicitly undone by toISOString() coming out). Pure UTC
  // arithmetic throughout sidesteps the local offset entirely.
  start.setUTCDate(start.getUTCDate() - days);
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

/** Daily OHLC for one ticker, ending at the latest ingested bar. */
export const fetchCandles = (ticker, days = 120) =>
  get(`/market/candles?ticker=${encodeURIComponent(ticker)}&days=${days}`);

/**
 * Recent headlines, for one ticker or (no ticker) all five. Context only:
 * the agent never reads news. `enabled: false` when the server has no
 * Alpaca keys.
 */
export const fetchNews = (ticker, days = 7, limit = 12) =>
  get(`/news?${new URLSearchParams({ ...(ticker ? { ticker } : {}), days, limit })}`);

/**
 * The forward paper account (Alpaca paper venue): equity curve, positions
 * with entry prices, and filled orders. Read from the backend's database,
 * never from Alpaca directly.
 */
export const fetchPaper = () => get("/paper/status");

/**
 * FR-20 as a history: what the pipeline did, newest first (the activity
 * timeline). 100 events covers several hours of 5-minute checks.
 */
export const fetchEvents = (limit = 100) => get(`/events?limit=${limit}`);

/** Whether the server offers the pipeline assistant (an API key is set). */
export const fetchAssistantStatus = () => get("/assistant/status");

/**
 * Splits buffered server-sent-event text into complete events, calls
 * `onEvent` with each parsed `data:` payload, and returns the incomplete
 * remainder to prepend to the next chunk.
 */
export function drainEvents(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    for (const line of part.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch {
        // A malformed event is skipped rather than ending the answer.
      }
    }
  }
  return rest;
}

/**
 * Asks the pipeline assistant. `messages` is the conversation so far (text
 * only, ending with the question). Calls `onEvent` for each streamed event
 * (step, step_done, visual, text, follow_ups, error, done) and resolves when
 * the stream ends. No timeout: answers stream for as long as they take, and
 * `signal` lets the user stop one.
 */
export async function streamAssistant(messages, onEvent, signal) {
  let res;
  try {
    res = await fetch(`${BASE}/assistant/chat`, {
      method: "POST",
      signal,
      credentials: "include",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...authHeaders() },
      body: JSON.stringify({ messages }),
    });
  } catch (err) {
    return { ok: false, reason: err.name === "AbortError" ? "cancelled" : "unreachable" };
  }
  if (res.status === 401) return unauthorized();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const reason = { 429: "throttled", 503: "not_configured", 422: "invalid" }[res.status] ?? "failed";
    return { ok: false, reason, detail: typeof body?.detail === "string" ? body.detail : null };
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer = drainEvents(buffer + decoder.decode(value, { stream: true }), onEvent);
    }
    drainEvents(buffer + decoder.decode() + "\n\n", onEvent);
  } catch (err) {
    return { ok: false, reason: err.name === "AbortError" ? "cancelled" : "interrupted" };
  }
  return { ok: true };
}

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
