/**
 * Tests for App.jsx's adapter functions — these translate the real API
 * shapes (src/serving/schemas.py) into the view-model shape every
 * component renders. Live data is frequently *incomplete* in ways the
 * mock never is (no evaluation has run yet, no snapshots exist yet), so
 * the null-guarding here is exactly what these tests exercise.
 */
import { describe, expect, it } from "vitest";
import {
  actionClass, adaptDecisions, adaptMetrics, adaptStatus, adaptTelemetry,
  fmtShortDate, shortId,
} from "./App.jsx";

describe("shortId", () => {
  it("truncates a real UUID to 8 characters", () => {
    expect(shortId("bf65de93-d5c1-4109-912d-23a547177a31")).toBe("bf65de93");
  });

  it("returns an em dash for null or undefined", () => {
    expect(shortId(null)).toBe("—");
    expect(shortId(undefined)).toBe("—");
  });
});

describe("actionClass", () => {
  it("maps every known discrete action to its CSS class", () => {
    expect(actionClass("BUY")).toBe("buy");
    expect(actionClass("SELL")).toBe("sell");
    expect(actionClass("HOLD")).toBe("hold");
    expect(actionClass("LIQUIDATE")).toBe("liq");
  });

  it("falls back to hold for an unrecognised action rather than throwing", () => {
    expect(actionClass("SOMETHING_NEW")).toBe("hold");
  });
});

describe("adaptStatus", () => {
  it("maps a fully-populated real /ct-status response", () => {
    const raw = {
      status: "SERVING",
      active_version_id: "bf65de93-d5c1-4109-912d-23a547177a31",
      active_model_label: "September 2026 #2",
      last_ingest_date: "2026-09-18",
      current_vix: 14.81,
      vix_critical_threshold: 35.0,
      target_sharpe_threshold: 1.0,
      ingestion_status: "IDLE",
      last_ingestion_attempted_at: "2026-09-20T19:09:07.252312Z",
      last_ingestion_ok: true,
    };
    const status = adaptStatus(raw);

    expect(status.pipeline).toBe("SERVING");
    expect(status.model).toBe("bf65de93");
    expect(status.modelLabel).toBe("September 2026 #2");
    expect(status.modelId).toBe(raw.active_version_id);
    expect(status.lastIngest).toBe("18 Sept");
    expect(status.vix).toBe(14.81);
    expect(status.ingestionStatus).toBe("IDLE");
    expect(status.ingestionOk).toBe(true);
    expect(status.ingestionCheckedAt).toBeTruthy();
  });

  it("falls back to the short ID when no friendly label exists yet", () => {
    const status = adaptStatus({ status: "SERVING", active_version_id: "bf65de93-d5c1-4109" });
    expect(status.modelLabel).toBe("bf65de93");
  });

  it("null-guards a fresh install with nothing evaluated or ingested yet", () => {
    const status = adaptStatus({
      status: "SERVING",
      active_version_id: null,
      last_ingest_date: null,
      last_ingestion_attempted_at: null,
      last_ingestion_ok: null,
    });
    expect(status.model).toBe("—");
    expect(status.modelLabel).toBe("—");
    expect(status.lastIngest).toBe("—");
    expect(status.ingestionCheckedAt).toBeNull();
  });
});

describe("adaptTelemetry", () => {
  it("maps real telemetry points into parallel arrays", () => {
    const raw = {
      points: [
        { date: "2026-09-16", equity_value: 100000, rolling_sharpe_30d: null, max_drawdown: 0.0 },
        { date: "2026-09-17", equity_value: 102140, rolling_sharpe_30d: 30101.4, max_drawdown: 0.0 },
        { date: "2026-09-18", equity_value: 100941, rolling_sharpe_30d: 4.79, max_drawdown: 0.0117 },
      ],
    };
    const telemetry = adaptTelemetry(raw);

    expect(telemetry.dates).toHaveLength(3);
    expect(telemetry.agent).toEqual([100000, 102140, 100941]);
    expect(telemetry.sharpe).toEqual([null, 30101.4, 4.79]);
    expect(telemetry.bench).toBeNull(); // not persisted server-side (Charts.jsx's note)
    expect(telemetry.lastMaxDrawdown).toBe(0.0117);
  });

  it("handles an empty points array without throwing (a fresh install, or no evaluation yet)", () => {
    const telemetry = adaptTelemetry({ points: [] });
    expect(telemetry.dates).toEqual([]);
    expect(telemetry.lastMaxDrawdown).toBeNull();
  });

  it("handles a missing points key the same way as an empty array", () => {
    const telemetry = adaptTelemetry({});
    expect(telemetry.dates).toEqual([]);
  });
});

describe("adaptDecisions", () => {
  it("formats a real decision log entry, including the NFR-07 traceability chain", () => {
    const raw = {
      items: [{
        decided_at: "2026-09-17T20:56:13Z",
        ticker: "XOM",
        raw_weight: -0.134,
        discrete_action: "HOLD",
        vix_at_decision: 14.3,
        failsafe_triggered: false,
        version_id: "bf65de93-d5c1-4109-912d-23a547177a31",
        run_id: "0b3dbee5-97c0-4210-b432-82d93a388289",
        train_start: "2015-01-01", train_end: "2023-12-31",
        eval_start: "2024-01-01", eval_end: "2025-12-31",
      }],
      total: 1,
    };
    const decisions = adaptDecisions(raw);

    expect(decisions.total).toBe(1);
    expect(decisions.items[0].weight).toBe("-0.134");
    expect(decisions.items[0].vix).toBe("14.3");
    expect(decisions.items[0].version).toBe("bf65de93");
    expect(decisions.items[0].trainPartition).toBe("2015-01-01 → 2023-12-31");
  });

  it("shows an em dash for a fail-safe decision, which has no raw_weight", () => {
    const raw = {
      items: [{
        decided_at: "2026-09-17T20:56:13Z", ticker: "XOM", raw_weight: null,
        discrete_action: "LIQUIDATE", vix_at_decision: 42.0, failsafe_triggered: true,
        version_id: "v1", run_id: "r1",
        train_start: "2015-01-01", train_end: "2023-12-31",
        eval_start: "2024-01-01", eval_end: "2025-12-31",
      }],
      total: 1,
    };
    expect(adaptDecisions(raw).items[0].weight).toBe("—");
  });
});

describe("adaptMetrics", () => {
  it("converts a fractional drawdown to a signed percentage", () => {
    const metrics = adaptMetrics(
      { rolling_sharpe: 4.79, cycle_window_days: 90, training_runs: 540, promoted_count: 2, rejected_count: 448 },
      { lastMaxDrawdown: 0.087 }
    );
    expect(metrics.rollingSharpe).toBe(4.79);
    expect(metrics.maxDrawdown).toBeCloseTo(-8.7, 5);
    expect(metrics.promoted).toBe(2);
    expect(metrics.heldBack).toBe(448);
  });

  it("null-guards when no snapshot has ever been recorded", () => {
    const metrics = adaptMetrics(
      { rolling_sharpe: null, cycle_window_days: 90, training_runs: 0, promoted_count: 0, rejected_count: 0 },
      { lastMaxDrawdown: null }
    );
    expect(metrics.rollingSharpe).toBeNull();
    expect(metrics.maxDrawdown).toBeNull();
  });
});

describe("fmtShortDate", () => {
  it("formats a date as day-month, matching the dashboard's compact style", () => {
    expect(fmtShortDate(new Date("2026-09-18T00:00:00Z"))).toMatch(/18/);
  });
});
