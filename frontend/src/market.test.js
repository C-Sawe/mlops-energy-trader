import { describe, expect, it } from "vitest";
import { marketStatus } from "./market.js";

const at = (iso) => marketStatus(new Date(iso));

describe("marketStatus", () => {
  it("is closed before the open, and names that morning's open", () => {
    const s = at("2026-11-05T07:51:00Z");
    expect(s.open).toBe(false);
    expect(s.holiday).toBe(false);
    expect(s.nyTime).toBe("02:51");
    expect(s.nyZone).toBe("EST");
    expect(s.next.weekday).toBe("Thu");
    expect(s.next.time).toBe("09:30");
    expect(s.next.at.toISOString()).toBe("2026-11-05T14:30:00.000Z");
  });

  it("opens at exactly 09:30 and closes at exactly 16:00 New York time", () => {
    expect(at("2026-11-05T14:29:59Z").open).toBe(false);
    expect(at("2026-11-05T14:30:00Z").open).toBe(true);
    expect(at("2026-11-05T20:59:59Z").open).toBe(true);
    expect(at("2026-11-05T21:00:00Z").open).toBe(false);
  });

  it("names the coming close while open", () => {
    const s = at("2026-11-05T15:00:00Z");
    expect(s.next.time).toBe("16:00");
    expect(s.next.at.toISOString()).toBe("2026-11-05T21:00:00.000Z");
  });

  it("follows daylight saving: in summer time 09:30 is an hour earlier in UTC", () => {
    expect(at("2026-10-30T13:29:00Z").open).toBe(false);
    const s = at("2026-10-30T13:30:00Z");
    expect(s.open).toBe(true);
    expect(s.nyZone).toBe("EDT");
  });

  it("uses the offset of the day it names when a clock change falls in between", () => {
    // Friday in EDT; clocks go back on Sunday 1 Nov; Monday's open is in EST.
    const s = at("2026-10-30T21:00:00Z");
    expect(s.nyZone).toBe("EDT");
    expect(s.next.weekday).toBe("Mon");
    expect(s.next.at.toISOString()).toBe("2026-11-02T14:30:00.000Z");
  });

  it("is closed across a weekend, through to Monday", () => {
    const s = at("2026-11-07T16:00:00Z");
    expect(s.open).toBe(false);
    expect(s.holiday).toBe(false);
    expect(s.next.at.toISOString()).toBe("2026-11-09T14:30:00.000Z");
  });

  it("is closed on a weekday holiday, during what would be trading hours", () => {
    const s = at("2026-11-26T16:00:00Z");
    expect(s.open).toBe(false);
    expect(s.holiday).toBe(true);
    expect(s.next.weekday).toBe("Fri");
    expect(s.next.at.toISOString()).toBe("2026-11-27T14:30:00.000Z");
  });

  it("closes at 13:00 New York time on an early-close day", () => {
    const open = at("2026-11-27T17:59:00Z");
    expect(open.open).toBe(true);
    expect(open.next.time).toBe("13:00");
    expect(open.next.at.toISOString()).toBe("2026-11-27T18:00:00.000Z");
    expect(at("2026-11-27T18:00:00Z").open).toBe(false);
  });

  it("skips a holiday and the weekend after it when finding the next open", () => {
    // Christmas Eve's early close, then Christmas (Friday), then the weekend.
    const s = at("2026-12-24T18:30:00Z");
    expect(s.open).toBe(false);
    expect(s.next.weekday).toBe("Mon");
    expect(s.next.at.toISOString()).toBe("2026-12-28T14:30:00.000Z");
  });
});
