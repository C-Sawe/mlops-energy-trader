import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Pill, StatusStrip, ingestionToneAndLabel, pipelineToneAndLabel } from "./Primitives.jsx";

describe("pipelineToneAndLabel", () => {
  it("maps each real CT pipeline status to its tone and label", () => {
    expect(pipelineToneAndLabel("SERVING")).toEqual({ tone: "good", label: "Serving" });
    expect(pipelineToneAndLabel("EVALUATING")).toEqual({ tone: "idle", label: "Evaluating" });
    expect(pipelineToneAndLabel("RETRAINING")).toEqual({ tone: "warn", label: "Retraining" });
  });

  it("falls back to the raw value rather than crashing on an unrecognised status", () => {
    const { tone, label } = pipelineToneAndLabel("SOMETHING_NEW");
    expect(tone).toBe("idle");
    expect(label).toBe("SOMETHING_NEW");
  });
});

describe("ingestionToneAndLabel", () => {
  it("reads Ingesting/warn while a tick is actually running", () => {
    expect(ingestionToneAndLabel("INGESTING", null)).toEqual({ tone: "warn", label: "Ingesting" });
  });

  it("reads Idle/good after a successful tick", () => {
    expect(ingestionToneAndLabel("IDLE", true)).toEqual({ tone: "good", label: "Idle" });
  });

  it("reads Failed/crit after a failed tick", () => {
    expect(ingestionToneAndLabel("IDLE", false)).toEqual({ tone: "crit", label: "Failed" });
  });

  it("reads a plain dash when nothing has ever ticked yet (e.g. the mock fallback)", () => {
    expect(ingestionToneAndLabel(undefined, null)).toEqual({ tone: "idle", label: "—" });
  });

  it("prioritises INGESTING over a stale last-result flag", () => {
    // A tick is running right now; whatever the *previous* tick's outcome
    // was must not be what's shown while this one is in flight.
    expect(ingestionToneAndLabel("INGESTING", false).label).toBe("Ingesting");
  });
});

describe("Pill", () => {
  it("renders its text content", () => {
    render(<Pill tone="good">Serving</Pill>);
    expect(screen.getByText("Serving")).toBeInTheDocument();
  });
});

describe("StatusStrip", () => {
  const baseProps = {
    pipeline: "SERVING",
    model: "bf65de93",
    modelLabel: "September 2026 #2",
    modelId: "bf65de93-d5c1-4109-912d-23a547177a31",
    lastIngest: "18 Sept",
    ingestionStatus: "IDLE",
    ingestionCheckedAt: "20/09/2026 19:09:07",
    ingestionOk: true,
    ingestionNextAt: "Thu 00:30",
    vix: 14.81,
    failSafeThreshold: 35.0,
  };

  it("renders the friendly model label, not the raw ID, as the visible text", () => {
    render(<StatusStrip {...baseProps} />);
    expect(screen.getByText("September 2026 #2")).toBeInTheDocument();
  });

  it("shows the fail-safe as inactive when VIX is well below the critical threshold", () => {
    render(<StatusStrip {...baseProps} />);
    expect(screen.getByText(/Inactive/)).toBeInTheDocument();
  });

  it("shows the fail-safe as triggered when VIX exceeds the critical threshold (I5/FR-12)", () => {
    render(<StatusStrip {...baseProps} vix={40.0} />);
    expect(screen.getByText(/Triggered/)).toBeInTheDocument();
  });

  it("shows when the next scheduled fetch is (FR-01)", () => {
    render(<StatusStrip {...baseProps} />);
    expect(screen.getByText("Next fetch Thu 00:30")).toBeInTheDocument();
  });

  it("shows the ingestion checked-at timestamp when one exists", () => {
    render(<StatusStrip {...baseProps} />);
    expect(screen.getByText(/Checked/)).toBeInTheDocument();
  });

  describe("US market tile", () => {
    afterEach(() => vi.useRealTimers());

    it("shows closed, New York's time and the next open", () => {
      vi.useFakeTimers({ now: new Date("2026-11-05T07:51:00Z") });
      render(<StatusStrip {...baseProps} />);
      expect(screen.getByText("Closed")).toBeInTheDocument();
      expect(screen.getByText("New York 02:51 EST")).toBeInTheDocument();
      expect(screen.getByText("Opens Thu 09:30 ET")).toBeInTheDocument();
      expect(screen.getByText(/your time$/)).toBeInTheDocument();
    });

    it("says so when it is closed for a holiday", () => {
      vi.useFakeTimers({ now: new Date("2026-11-26T16:00:00Z") });
      render(<StatusStrip {...baseProps} />);
      expect(screen.getByText("Closed · Holiday")).toBeInTheDocument();
    });

    it("turns open at the opening minute without a page reload", () => {
      vi.useFakeTimers({ now: new Date("2026-11-05T14:29:30Z") });
      render(<StatusStrip {...baseProps} />);
      expect(screen.getByText("Closed")).toBeInTheDocument();

      act(() => { vi.advanceTimersByTime(30_000); });

      expect(screen.getByText("Open")).toBeInTheDocument();
      expect(screen.getByText("New York 09:30 EST")).toBeInTheDocument();
      expect(screen.getByText("Closes 16:00 ET")).toBeInTheDocument();
    });
  });
});
