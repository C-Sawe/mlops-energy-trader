import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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

  it("shows the ingestion checked-at timestamp when one exists", () => {
    render(<StatusStrip {...baseProps} />);
    expect(screen.getByText(/Checked/)).toBeInTheDocument();
  });
});
