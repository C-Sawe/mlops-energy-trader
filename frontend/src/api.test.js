/**
 * Tests for the FastAPI client (IR-04/IR-07). Every network call is
 * mocked — no real backend, matching this project's backend test
 * convention of no network required.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchDecisions, fetchTelemetry, poll, predict, triggerEvaluation } from "./api.js";

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => body,
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("fetchTelemetry", () => {
  it("anchors the window to the given endDate, not calendar-today", async () => {
    fetch.mockResolvedValue(jsonResponse(200, { points: [] }));

    await fetchTelemetry(30, "2026-09-18");

    const url = fetch.mock.calls[0][0];
    expect(url).toContain("end=2026-09-18");
    expect(url).toContain("start=2026-08-19"); // 30 days back from the 18th
  });

  it("defaults to a window ending today when no endDate is given", async () => {
    fetch.mockResolvedValue(jsonResponse(200, { points: [] }));
    const today = new Date().toISOString().slice(0, 10);

    await fetchTelemetry(30);

    expect(fetch.mock.calls[0][0]).toContain(`end=${today}`);
  });

  it("resolves ok:true with the parsed body on a 200", async () => {
    fetch.mockResolvedValue(jsonResponse(200, { points: [{ date: "2026-09-18" }] }));
    const result = await fetchTelemetry(30, "2026-09-18");
    expect(result.ok).toBe(true);
    expect(result.data.points).toHaveLength(1);
  });

  it("computes the correct start date across a DST transition, regardless of viewer timezone", async () => {
    // A 30-day window ending 2026-11-15 crosses the US's Nov 1 2026
    // fall-back. Local-time date math (setDate/getDate on a UTC-midnight-
    // parsed date) computed 2026-10-15 in America/New_York instead of the
    // correct 2026-10-16 — a real bug this test caught, fixed with
    // setUTCDate. Forces TZ explicitly: this machine's own timezone (EAT,
    // no DST) would never have failed on the old buggy code, which would
    // make this test pass trivially regardless of whether the fix works.
    const originalTZ = process.env.TZ;
    process.env.TZ = "America/New_York";
    try {
      fetch.mockResolvedValue(jsonResponse(200, { points: [] }));
      await fetchTelemetry(30, "2026-11-15");
      expect(fetch.mock.calls[0][0]).toContain("start=2026-10-16");
    } finally {
      process.env.TZ = originalTZ;
    }
  });
});

describe("fetchDecisions error handling (IR-07: degrade gracefully, never throw)", () => {
  it("returns ok:false with reason 'unreachable' on a network failure", async () => {
    fetch.mockRejectedValue(new TypeError("Failed to fetch"));
    const result = await fetchDecisions();
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("unreachable");
  });

  it("returns ok:false on a generic non-2xx status", async () => {
    fetch.mockResolvedValue(jsonResponse(500, {}));
    const result = await fetchDecisions();
    expect(result.ok).toBe(false);
  });
});

describe("predict/triggerEvaluation (NFR-08: malformed input and not-ready are distinguished)", () => {
  it("surfaces a 422 as reason 'invalid' with the validation detail", async () => {
    fetch.mockResolvedValue(jsonResponse(422, { detail: "positions missing a ticker" }));
    const result = await predict({}, 1.0);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("invalid");
    expect(result.detail.detail).toBe("positions missing a ticker");
  });

  it("surfaces a 503 as reason 'not_ready' (no active model yet)", async () => {
    fetch.mockResolvedValue(jsonResponse(503, { detail: "no active model" }));
    const result = await triggerEvaluation();
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("not_ready");
  });

  it("resolves ok:true on success", async () => {
    fetch.mockResolvedValue(jsonResponse(200, { decisions: [] }));
    const result = await predict({ XOM: 0.0 }, 1.0);
    expect(result.ok).toBe(true);
  });
});

describe("poll", () => {
  it("calls fn immediately, then again after the interval, until unsubscribed", async () => {
    vi.useFakeTimers();
    const fn = vi.fn().mockResolvedValue("tick");
    const onData = vi.fn();

    const stop = poll(fn, 1000, onData);
    await vi.advanceTimersByTimeAsync(0);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(onData).toHaveBeenCalledWith("tick");

    await vi.advanceTimersByTimeAsync(1000);
    expect(fn).toHaveBeenCalledTimes(2);

    stop();
    await vi.advanceTimersByTimeAsync(5000);
    expect(fn).toHaveBeenCalledTimes(2); // no further ticks after unsubscribing
  });
});
