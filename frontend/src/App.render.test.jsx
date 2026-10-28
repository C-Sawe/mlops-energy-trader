/**
 * Rendering tests for the dashboard component itself, not just its adapters
 * (those are in App.test.jsx): polling, the IR-07 mock and stale fallbacks,
 * the decision-detail sheet (NFR-07, NFR-11), Force Check, and the live
 * benchmark and cycle-history readouts.
 *
 * The API module is mocked per call; its real `poll` helper is kept, and
 * driven with fake timers, so the 15 s poll loop is exercised as written.
 * waitFor is avoided on purpose: under Vitest's fake timers it can hang
 * instead of timing out, so each step advances the clock explicitly.
 */
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    fetchStatus: vi.fn(),
    fetchTelemetry: vi.fn(),
    fetchDecisions: vi.fn(),
    fetchPaper: vi.fn(),
    fetchCandles: vi.fn(),
    fetchEvents: vi.fn(),
    fetchAssistantStatus: vi.fn(),
    fetchNews: vi.fn(),
    triggerEvaluation: vi.fn(),
  };
});

import * as api from "./api.js";
import App from "./App.jsx";

const POLL_MS = 15000;
const DOWN = { ok: false, reason: "unreachable" };
const ok = (data) => ({ ok: true, data });

const statusRaw = (over = {}) => ({
  status: "SERVING",
  active_version_id: "bf65de93-d5c1-4109-912d-23a547177a31",
  active_model_label: "October 2026 #1",
  rolling_sharpe: 0.42,
  target_sharpe_threshold: 1.0,
  last_evaluated_at: "2026-10-06T12:00:00Z",
  last_retrain_deferred_at: null,
  last_retrain_failed_at: null,
  last_ingest_date: "2026-10-06",
  current_vix: 18.4,
  vix_critical_threshold: 35.0,
  cycle_window_days: 90,
  training_runs: 7,
  promoted_count: 1,
  rejected_count: 6,
  autonomous_retrains: 5,
  autonomous_promoted: 1,
  ingestion_status: "IDLE",
  last_ingestion_attempted_at: "2026-10-06T22:00:00Z",
  last_ingestion_ok: true,
  ...over,
});

/** `benchFrom`: index of the first point carrying a benchmark (null = none). */
const telemetryRaw = (benchFrom = 0) => ({
  points: Array.from({ length: 10 }, (_, i) => ({
    date: `2026-09-${String(21 + i).padStart(2, "0")}`,
    equity_value: 100000 + i * 120,
    rolling_sharpe_30d: i ? 0.5 : null,
    max_drawdown: 0.02,
    cumulative_return: i * 0.0012,
    benchmark_equity: benchFrom != null && i >= benchFrom ? 100000 + i * 60 : null,
  })),
});

const decisionsRaw = {
  items: [
    {
      decision_id: "d1", version_id: "bf65de93-d5c1-4109-912d-23a547177a31",
      run_id: "7f60784a-805f-4047-ad74-fad67261920c",
      train_start: "2024-01-01", train_end: "2025-12-31",
      eval_start: "2026-01-01", eval_end: "2026-03-31",
      decided_at: "2026-10-06T22:01:00Z", ticker: "XOM", raw_weight: 0.62,
      discrete_action: "BUY", vix_at_decision: 18.4, failsafe_triggered: false,
    },
    {
      decision_id: "d2", version_id: "bf65de93-d5c1-4109-912d-23a547177a31",
      run_id: "7f60784a-805f-4047-ad74-fad67261920c",
      train_start: "2024-01-01", train_end: "2025-12-31",
      eval_start: "2026-01-01", eval_end: "2026-03-31",
      decided_at: "2026-10-05T22:01:00Z", ticker: "CVX", raw_weight: null,
      discrete_action: "LIQUIDATE", vix_at_decision: 41.2, failsafe_triggered: true,
    },
  ],
  page: 1,
  page_size: 6,
  total: 2,
};

function backendUp({ status = statusRaw(), telemetry = telemetryRaw() } = {}) {
  api.fetchStatus.mockResolvedValue(ok(status));
  api.fetchTelemetry.mockResolvedValue(ok(telemetry));
  api.fetchDecisions.mockResolvedValue(ok(decisionsRaw));
}

function backendDown() {
  api.fetchStatus.mockResolvedValue(DOWN);
  api.fetchTelemetry.mockResolvedValue(DOWN);
  api.fetchDecisions.mockResolvedValue(DOWN);
}

/** Advance the clock and let every resolved fetch settle inside act(). */
const advance = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

/** Render, then (optionally) open a section from the navigation as a user would. */
async function renderApp(section) {
  render(<App />);
  await advance();
  if (section) await openSection(section);
}

async function openSection(name) {
  const nav = screen.getByRole("navigation", { name: "Sections" });
  fireEvent.click(within(nav).getByRole("link", { name: new RegExp(`^${name}`) }));
  await advance();
}

beforeEach(() => {
  window.location.hash = ""; // every test starts on the Overview
  vi.useFakeTimers();
  // jsdom has no matchMedia. Reduced motion makes the sheet's spring settle
  // immediately, so open/close is observable without animation frames.
  window.matchMedia = vi.fn().mockReturnValue({
    matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn(),
  });
  // Paper and candles poll on their own; their failure must not affect the rest.
  api.fetchPaper.mockResolvedValue(DOWN);
  api.fetchCandles.mockResolvedValue(DOWN);
  api.fetchEvents.mockResolvedValue(DOWN);
  api.fetchAssistantStatus.mockResolvedValue(DOWN);
  api.fetchNews.mockResolvedValue(DOWN);
});

afterEach(() => {
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe("IR-07: graceful degradation", () => {
  it("shows labelled example data while the backend has never answered", async () => {
    backendDown();
    await renderApp();

    expect(screen.getByText("Example data — not live results")).toBeInTheDocument();
    expect(screen.getByText(/Figures on this page are generated example data/)).toBeInTheDocument();
    // Nothing to force-check against when there is no backend.
    expect(screen.queryByLabelText("Force a CT evaluation now")).toBeNull();
  });

  it("switches to live data on the next poll once the backend answers", async () => {
    backendDown();
    await renderApp();
    expect(screen.getByText("Example data — not live results")).toBeInTheDocument();

    backendUp();
    await advance(POLL_MS);

    expect(screen.queryByText("Example data — not live results")).toBeNull();
    expect(screen.getByText("October 2026 #1")).toBeInTheDocument();
    expect(screen.getByLabelText("Force a CT evaluation now")).toBeInTheDocument();
  });

  it("keeps the last known live data and flags it stale when the backend dies mid-session", async () => {
    backendUp();
    await renderApp();
    expect(screen.getByText("October 2026 #1")).toBeInTheDocument();

    backendDown();
    await advance(POLL_MS);

    expect(screen.getByText(/Backend unreachable — showing last known/)).toBeInTheDocument();
    expect(screen.getByText("October 2026 #1")).toBeInTheDocument();
    // Last known live data, not a silent swap back to illustrative figures.
    expect(screen.queryByText("Example data — not live results")).toBeNull();
  });

  it("clears the stale flag when the backend comes back", async () => {
    backendUp();
    await renderApp();
    backendDown();
    await advance(POLL_MS);
    expect(screen.getByText(/Backend unreachable/)).toBeInTheDocument();

    backendUp();
    await advance(POLL_MS);
    expect(screen.queryByText(/Backend unreachable/)).toBeNull();
  });
});

describe("decision-detail sheet", () => {
  it("shows the decision → version → run → partition chain (NFR-07) and closes on Escape", async () => {
    backendUp();
    await renderApp("Decisions");

    fireEvent.click(screen.getByText("XOM", { selector: "td.tick" }));
    const sheet = screen.getByRole("dialog", { name: "Decision detail" });

    expect(within(sheet).getByText("bf65de93")).toBeInTheDocument();
    expect(within(sheet).getByText("7f60784a")).toBeInTheDocument();
    expect(within(sheet).getByText("2024-01-01 → 2025-12-31")).toBeInTheDocument();
    expect(within(sheet).getByText("2026-01-01 → 2026-03-31")).toBeInTheDocument();
    expect(sheet).toHaveTextContent("maps to BUY under the decision thresholds");

    fireEvent.keyDown(document, { key: "Escape" });
    await advance();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("explains a fail-safe decision by the VIX value that triggered it (NFR-11)", async () => {
    backendUp();
    await renderApp("Decisions");

    fireEvent.click(screen.getByText("CVX", { selector: "td.tick" }));
    const sheet = screen.getByRole("dialog", { name: "Decision detail" });

    expect(sheet).toHaveTextContent("The volatility fail-safe fired: VIX 41.2 exceeded the critical threshold");
    expect(sheet).toHaveTextContent("No model weight was consulted.");
  });

  it("opens from the keyboard as well as the mouse", async () => {
    backendUp();
    await renderApp("Decisions");

    const row = screen.getByText("XOM", { selector: "td.tick" }).closest("tr");
    fireEvent.keyDown(row, { key: "Enter" });
    expect(screen.getByRole("dialog", { name: "Decision detail" })).toBeInTheDocument();
  });
});

describe("Force Check", () => {
  it("posts to /ct/evaluate and shows the status it returns", async () => {
    backendUp();
    api.triggerEvaluation.mockResolvedValue(ok(statusRaw({ status: "RETRAINING" })));
    await renderApp();

    const button = screen.getByLabelText("Force a CT evaluation now");
    expect(button).toBeEnabled();
    fireEvent.click(button);
    await advance();

    expect(api.triggerEvaluation).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Retraining")).toBeInTheDocument();
    // Only one evaluation at a time: the button stays off until SERVING again.
    expect(screen.getByLabelText("Force a CT evaluation now")).toBeDisabled();
  });
});

describe("live equity benchmark and cycle history", () => {
  it("draws the persisted buy-and-hold benchmark next to the agent", async () => {
    backendUp();
    await renderApp();

    expect(screen.getByText("Agent versus equal-weight buy & hold, net of 10 bps costs")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /agent against an equal-weight buy and hold benchmark/ }))
      .toBeInTheDocument();
  });

  it("says from when the benchmark exists when older snapshots predate it", async () => {
    backendUp({ telemetry: telemetryRaw(4) });
    await renderApp();

    expect(screen.getByText(/benchmark recorded from/)).toBeInTheDocument();
  });

  it("falls back to the single-line chart when no snapshot has a benchmark", async () => {
    backendUp({ telemetry: telemetryRaw(null) });
    await renderApp();

    expect(screen.getByText(/Simulated by replaying the incumbent/)).toBeInTheDocument();
    expect(screen.queryByText(/buy & hold, net of 10 bps/)).toBeNull();
  });

  it("separates runs the CT loop started itself from all training runs", async () => {
    backendUp();
    await renderApp("Pipeline");

    const meta = screen.getByText(/started by the CT loop/);
    expect(meta.textContent).toMatch(/last 90 days · 5 started by the CT loop \(1 promoted\)/);
  });
});

describe("signed-in account", () => {
  it("shows the account in the masthead and signs out from there", async () => {
    backendUp();
    const onSignOut = vi.fn();
    render(<App user={{ email: "caleb@example.com", name: "Caleb", provider: "google" }} onSignOut={onSignOut} />);
    await advance();

    expect(screen.getByText("caleb@example.com")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
    expect(onSignOut).toHaveBeenCalledOnce();
  });

  it("shows no account without sign-in", async () => {
    backendUp();
    await renderApp();
    expect(screen.queryByRole("button", { name: "Sign out" })).toBeNull();
  });
});

describe("pipeline activity", () => {
  it("shows the backend's events, with a manual Force Check marked as such", async () => {
    backendUp();
    api.fetchEvents.mockResolvedValue(ok({
      items: [
        { event_id: 2, occurred_at: "2026-10-06T12:05:00Z", kind: "candidate_rejected", source: "scheduled",
          summary: "Candidate did not beat the current model out of sample (Sharpe 0.41 vs 0.64): current model kept",
          detail: {}, run_id: "r1", version_id: null },
        { event_id: 1, occurred_at: "2026-10-06T12:00:00Z", kind: "evaluation", source: "manual",
          summary: "Rolling Sharpe 1.02 meets the 1.00 target: model kept serving",
          detail: { rolling_sharpe: 1.02, target: 1.0 }, run_id: null, version_id: "v1" },
      ],
      has_more: false,
    }));
    await renderApp();

    const list = screen.getByRole("list", { name: /pipeline activity/i });
    expect(within(list).getByText("Candidate rejected")).toBeTruthy();
    expect(within(list).getByText("by operator")).toBeTruthy();
  });

  it("says it is waiting rather than showing an empty timeline while the backend is down", async () => {
    backendUp();
    await renderApp();
    // Scoped: the paper card has the same waiting message.
    const card = screen.getByRole("heading", { name: "Pipeline activity" }).closest("section");
    expect(within(card).getByText("Waiting for the backend…")).toBeTruthy();
  });
});

describe("sections (sidebar on wide screens, tab bar on narrow)", () => {
  const nav = () => screen.getByRole("navigation", { name: "Sections" });

  it("starts on the Overview and lists four sections, marking the current one", async () => {
    backendUp();
    await renderApp();
    const links = within(nav()).getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual(["Overview", "Pipeline", "Decisions", "Trading"]);
    expect(within(nav()).getByRole("link", { name: "Overview" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Overview");
    expect(screen.getByText("Portfolio equity")).toBeTruthy();
    expect(screen.queryByText("Autonomous decision log")).toBeNull(); // one section at a time
  });

  it("opens a section, updates the address, and moves focus to its title", async () => {
    backendUp();
    await renderApp("Decisions");
    expect(window.location.hash).toBe("#decisions");
    const title = screen.getByRole("heading", { level: 1 });
    expect(title).toHaveTextContent("Decisions");
    expect(document.activeElement).toBe(title);
    expect(screen.getByText("Autonomous decision log")).toBeTruthy();
    expect(screen.queryByText("Portfolio equity")).toBeNull();
  });

  it("opens the section named in the address on load, and follows the back button", async () => {
    window.location.hash = "#trading";
    backendUp();
    await renderApp();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Trading");
    expect(screen.getByText("Price & entries")).toBeTruthy();

    window.location.hash = "#pipeline";
    await act(async () => { window.dispatchEvent(new HashChangeEvent("hashchange")); });
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Pipeline");
  });

  it("links from the Overview's short activity list to the full timeline", async () => {
    backendUp();
    api.fetchEvents.mockResolvedValue(ok({ items: [
      { event_id: 1, occurred_at: "2026-10-22T10:00:00Z", kind: "ingestion_ok", source: "scheduled",
        summary: "Fetched market data", detail: null, run_id: null, version_id: null },
    ] }));
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: "See all in Pipeline" }));
    await advance();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Pipeline");
  });

  it("badges Pipeline while the loop is retraining", async () => {
    backendUp();
    api.fetchStatus.mockResolvedValue(ok(statusRaw({ status: "RETRAINING" })));
    await renderApp();
    expect(within(nav()).getByRole("link", { name: /Pipeline/ })).toHaveTextContent("(Retraining)");
  });
});
