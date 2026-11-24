import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", async (importOriginal) => ({
  ...(await importOriginal()),
  fetchSimulation: vi.fn(), startSimulation: vi.fn(), sendSimulation: vi.fn(), stopSimulation: vi.fn(),
}));

import * as api from "../api.js";
import { SimulationPage, explainDay, stageState } from "./Simulation.jsx";

const ok = (data) => ({ ok: true, data });
const day = (over = {}) => ({
  day: "2020-02-27", index: 1, vix: 39.2, vix_injected: false, failsafe: true, failsafe_reason: "vix",
  decisions: [{ ticker: "XOM", weight: null, action: "LIQUIDATE" }], exposure: 0, rolling_sharpe: -5.1,
  model: "abc12345", served_during_retrain: 0, failed_during_retrain: 0, ...over,
});
const body = (state = {}) => ({
  enabled: true,
  scenarios: [{ id: "volatility", title: "Volatility shock (2020)", description: "d" }],
  state: {
    status: "running", scenario: "volatility", stage: "idle", day: null, playing: false, total: 25, days: [], events: [],
    serving: null, flags: {}, error: null, vix_threshold: 35, target_sharpe: 1, ...state,
  },
});
const settle = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
const show = async (data) => { api.fetchSimulation.mockResolvedValue(ok(data)); render(<SimulationPage />); await settle(); };

beforeEach(() => vi.useFakeTimers());
afterEach(() => { vi.useRealTimers(); vi.resetAllMocks(); });

describe("stageState", () => {
  it("doesn't mark Retrain done on a day that retrained nothing", () => {
    expect(stageState("retrain", "idle", day(), [])).toBe("idle");
    expect(stageState("evaluate", "idle", day(), [])).toBe("done");
    expect(stageState("retrain", "idle", day(), [{ day: "2020-02-27", kind: "retrain_started" }])).toBe("done");
  });
  it("doesn't count another day's retrain", () => {
    expect(stageState("retrain", "idle", day(), [{ day: "2020-02-26", kind: "retrain_started" }])).toBe("idle");
  });
  it("marks only earlier stages done while one runs", () => {
    expect(["data", "decide", "evaluate", "retrain"].map((s) => stageState(s, "evaluate", day(), [])))
      .toEqual(["done", "done", "active", "idle"]);
  });
});

describe("explainDay", () => {
  it("says when a VIX value was injected, and only then", () => {
    expect(explainDay(day(), [])).toContain("VIX 39.2 tripped");
    expect(explainDay(day({ vix_injected: true }), [])).toContain("VIX 39.2 (injected) tripped");
  });
  it("tells a withheld candidate from a rejected one", () => {
    const e = (kind) => [{ day: "2020-02-27", kind }];
    expect(explainDay(day(), e("candidate_withheld"))).toContain("withheld");
    expect(explainDay(day(), e("candidate_rejected"))).toContain("gate kept the current one");
  });
});

describe("SimulationPage", () => {
  it("says it is a sandbox replay and not a result", async () => {
    await show(body());
    expect(screen.getByRole("note")).toHaveTextContent("nothing here is a result");
  });
  it("starts the chosen scenario and shows a refusal", async () => {
    await show(body({ status: "idle", scenario: null }));
    api.startSimulation.mockResolvedValue({ ok: false, reason: "conflict", detail: { detail: "a simulation is already running" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(api.startSimulation).toHaveBeenCalledWith("volatility");
    expect(screen.getByRole("alert")).toHaveTextContent("already running");
  });
  it("shows the day's decisions, and stops injecting with a null value", async () => {
    await show(body({ days: [day()], flags: { vix_override: 50 } }));
    expect(screen.getByText("LIQUIDATE")).toBeTruthy();
    api.sendSimulation.mockResolvedValue(ok(body()));
    fireEvent.click(screen.getByRole("button", { name: "Stop injecting VIX 50" }));
    await settle();
    expect(api.sendSimulation).toHaveBeenCalledWith({ cmd: "vix", value: null });
  });
  it("counts requests served while a candidate trains", async () => {
    await show(body({ stage: "retrain", serving: { served: 12, failed: 0 } }));
    expect(screen.getByText(/answered 12 requests, 0 failed/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Step one day" })).toBeDisabled();
  });
  it("says how to turn it on when the server has it off", async () => {
    await show({ enabled: false, scenarios: [], state: null });
    expect(screen.getByText(/SIMULATION_ENABLED=true/)).toBeTruthy();
  });
});
