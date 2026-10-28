/**
 * The activity timeline's grouping and wording (components/Activity.jsx).
 * Grouping is the part most likely to mislead if it goes wrong: it must
 * shorten the list without hiding that something happened, or how often.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ActivityCard, groupEvents, relativeTime } from "./Activity.jsx";

const T0 = new Date("2026-10-21T12:00:00Z").getTime();
let nextId = 100;
const ev = (kind, minutesAgo, summary = kind, extra = {}) => ({
  id: nextId--, at: new Date(T0 - minutesAgo * 60000), kind, source: "scheduled", summary, detail: {}, ...extra,
});
const check = (minutesAgo, sharpe, extra = {}) =>
  ev("evaluation", minutesAgo, `Rolling Sharpe ${sharpe}`, { detail: { rolling_sharpe: sharpe, target: 1 }, ...extra });

describe("groupEvents", () => {
  it("collapses consecutive routine checks into one row with the Sharpe range", () => {
    const rows = groupEvents([check(0, 1.2), check(5, 1.1), check(10, 1.15), ev("ingestion_ok", 60)]);
    expect(rows).toHaveLength(2);
    expect(rows[0].count).toBe(3);
    expect(rows[0].summary).toBe(
      "Checked 3 times: rolling Sharpe between 1.10 and 1.20, at or above the 1.00 target. Model kept serving."
    );
    expect(rows[1].label).toBe("Data fetched");
  });

  it("keeps a manual check separate from scheduled ones", () => {
    const rows = groupEvents([check(0, 1.2), check(5, 1.2, { source: "manual" }), check(10, 1.2)]);
    expect(rows.map((r) => r.source)).toEqual(["scheduled", "manual", "scheduled"]);
  });

  it("collapses a repeated retrain cycle into one counted row, cause before effect", () => {
    const cycle = (m) => [ev("candidate_rejected", m, "rejected 0.41 vs 0.64"), ev("retrain_started", m + 5, "Sharpe 0.64")];
    const rows = groupEvents([...cycle(0), ...cycle(10), ...cycle(20), ev("ingestion_ok", 90)]);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ label: "Retraining → Candidate rejected", count: 3, summary: "rejected 0.41 vs 0.64" });
  });

  it("does not merge events that differ", () => {
    const rows = groupEvents([ev("candidate_rejected", 0, "0.41 vs 0.64"), ev("candidate_rejected", 5, "0.30 vs 0.64")]);
    expect(rows.map((r) => r.count)).toEqual([1, 1]);
  });

  it("labels an unknown kind with its own name rather than dropping it", () => {
    expect(groupEvents([ev("something_new", 0)])[0]).toMatchObject({ label: "something_new", tone: "idle" });
  });
});

describe("relativeTime", () => {
  const now = new Date(T0);
  it.each([
    [30, "just now"],
    [5 * 60, "5 min ago"],
    [3 * 3600, "3 h ago"],
  ])("%i s ago reads %s", (seconds, text) => {
    expect(relativeTime(new Date(T0 - seconds * 1000), now)).toBe(text);
  });
});

describe("ActivityCard", () => {
  const now = new Date(T0);

  it("says what will appear when nothing has been recorded yet", () => {
    render(<ActivityCard events={[]} now={now} />);
    expect(screen.getByText(/No activity recorded yet/)).toBeTruthy();
  });

  it("marks operator-started actions and shows six rows until expanded", () => {
    const events = Array.from({ length: 8 }, (_, i) =>
      ev("ingestion_ok", i * 60, `fetch ${i}`, i === 0 ? { source: "manual" } : {}));
    render(<ActivityCard events={events} now={now} />);

    expect(screen.getByText("by operator")).toBeTruthy();
    expect(screen.getAllByRole("listitem")).toHaveLength(6);
    fireEvent.click(screen.getByRole("button", { name: "Show all 8" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(8);
  });

  it("states how far back the record goes", () => {
    render(<ActivityCard events={[ev("ingestion_ok", 0), ev("ingestion_ok", 120, "older")]} now={now} />);
    expect(screen.getByText(/^Recorded since/)).toBeTruthy();
  });
});

describe("groupEvents wording", () => {
  it("says the Sharpe was steady rather than 'between x and x'", () => {
    const rows = groupEvents([check(0, 2.45), check(5, 2.45)]);
    expect(rows[0].summary).toBe("Checked 2 times: rolling Sharpe steady at 2.45, at or above the 1.00 target. Model kept serving.");
  });
});
