/**
 * Tests for the Sharpe chart's outlier handling — extracted into
 * `computeSharpeDomain`/`clampForDisplay` specifically so this logic (the
 * exact thing that once let a 30,101 outlier crush the entire chart to a
 * flat line — CLAUDE.md §7) is unit-testable without rendering SVG.
 */
import { describe, expect, it } from "vitest";
import { SHARPE_OUTLIER_BOUND, clampForDisplay, computeSharpeDomain, gappedPath } from "./Charts.jsx";

describe("computeSharpeDomain", () => {
  it("computes a normal domain from ordinary values", () => {
    const [lo, hi] = computeSharpeDomain([-2, -1, 0, 1, 2], 1.0);
    expect(lo).toBeLessThan(-2);
    expect(hi).toBeGreaterThan(2);
  });

  it("excludes a real outlier (the 30101.4 case) from the domain entirely", () => {
    const values = [-14, -8, -3, 1, 5, 30101.4]; // the actual value CLAUDE.md §7 records
    const [lo, hi] = computeSharpeDomain(values, 1.0);

    // The domain must stay anchored to the *normal* values, not stretch to
    // fit the outlier — this is the entire point of the fix.
    expect(hi).toBeLessThan(100);
    expect(hi).toBeGreaterThan(5);
    expect(lo).toBeLessThan(-14);
  });

  it("falls back to using all values if every single one is an outlier", () => {
    // Degenerate case: nothing "normal" to anchor to at all.
    const [lo, hi] = computeSharpeDomain([500, 600], 1.0);
    expect(hi).toBeGreaterThan(600);
    expect(lo).toBeLessThan(500);
  });

  it("always includes the retraining threshold line, even if every value is far from it", () => {
    const [lo, hi] = computeSharpeDomain([10, 11, 12], 1.0);
    expect(lo).toBeLessThanOrEqual(1.0);
  });

  it("returns a placeholder domain for an empty series without throwing", () => {
    expect(computeSharpeDomain([], 1.0)).toEqual([0, 2]);
  });

  it("never collapses to a zero-height domain for a perfectly flat series", () => {
    const [lo, hi] = computeSharpeDomain([1.0, 1.0, 1.0], 1.0);
    expect(hi - lo).toBeGreaterThan(0);
  });
});

describe("clampForDisplay", () => {
  it("passes ordinary in-range values through unchanged", () => {
    const { clampedValues, offScale } = clampForDisplay([-1, 0, 1], -5, 5);
    expect(clampedValues).toEqual([-1, 0, 1]);
    expect(offScale).toEqual([false, false, false]);
  });

  it("clips the real 30101.4 outlier to the axis edge and flags it", () => {
    const { clampedValues, offScale } = clampForDisplay([1, 30101.4, 2], -10, 10);
    expect(clampedValues).toEqual([1, 10, 2]); // clipped to hi, not left at its real value
    expect(offScale).toEqual([false, true, false]);
  });

  it("clips a large negative outlier to the low edge too", () => {
    const { clampedValues, offScale } = clampForDisplay([-9999, 0], -10, 10);
    expect(clampedValues[0]).toBe(-10);
    expect(offScale[0]).toBe(true);
  });
});

describe("SHARPE_OUTLIER_BOUND", () => {
  it("is well above any Sharpe this project has ever actually measured", () => {
    // Documented range across every CLAUDE.md finding is roughly -14 to +5;
    // the bound exists with headroom, not as a tight fit to observed data.
    expect(SHARPE_OUTLIER_BOUND).toBeGreaterThan(14);
  });
});

import { candleDomain, placeFills } from "./Charts.jsx";

const bars = [
  { date: "2026-09-28", open: 10, high: 11, low: 9, close: 10.5 },
  { date: "2026-09-29", open: 10.5, high: 12, low: 10, close: 11.5 },
  // 2026-09-30 missing (no bar)
  { date: "2026-10-01", open: 11.5, high: 12.5, low: 11, close: 12 },
];

describe("placeFills", () => {
  it("places a fill on the candle for its own trading date", () => {
    const { placed } = placeFills(bars, [{ side: "buy", date: "2026-09-29", price: 10.6 }]);
    expect(placed).toEqual([{ index: 1, side: "buy", price: 10.6, qty: undefined }]);
  });

  it("moves a fill on a date with no bar to the next candle, not the previous one", () => {
    const { placed } = placeFills(bars, [{ side: "sell", date: "2026-09-30", price: 11.8 }]);
    expect(placed[0].index).toBe(2);
  });

  it("counts a fill after the last candle as pending instead of inventing a position", () => {
    const { placed, pending } = placeFills(bars, [{ side: "buy", date: "2026-10-02", price: 12.1 }]);
    expect(placed).toEqual([]);
    expect(pending).toBe(1);
  });

  it("skips fills with no price or date (an order that hasn't filled yet)", () => {
    const { placed, pending } = placeFills(bars, [
      { side: "buy", date: null, price: 10 },
      { side: "buy", date: "2026-09-28", price: null },
    ]);
    expect(placed).toEqual([]);
    expect(pending).toBe(0);
  });
});

describe("candleDomain", () => {
  it("covers every candle's high and low with padding", () => {
    const [lo, hi] = candleDomain(bars);
    expect(lo).toBeLessThan(9);
    expect(hi).toBeGreaterThan(12.5);
  });

  it("stretches to keep an entry price outside the window visible", () => {
    const [lo] = candleDomain(bars, [7.5, null]);
    expect(lo).toBeLessThan(7.5);
  });

  it("never returns a zero-height domain for a perfectly flat series", () => {
    const flat = [{ date: "2026-09-28", open: 5, high: 5, low: 5, close: 5 }];
    const [lo, hi] = candleDomain(flat);
    expect(hi).toBeGreaterThan(lo);
  });
});

describe("gappedPath", () => {
  const identity = { x: (i) => i, y: (v) => v };

  it("draws a continuous series exactly like a plain path", () => {
    expect(gappedPath([1, 2, 3], identity)).toBe("M0.00 1.00 L1.00 2.00 L2.00 3.00");
  });

  it("lifts the pen over nulls instead of plotting them as zero", () => {
    // A benchmark persisted only from some date onward (older snapshots
    // predate it) must not dive to $0 across the missing stretch.
    expect(gappedPath([null, null, 3, 4], identity)).toBe("M2.00 3.00 L3.00 4.00");
    expect(gappedPath([1, null, 3, 4], identity)).toBe("M0.00 1.00 M2.00 3.00 L3.00 4.00");
  });

  it("returns an empty path when there is nothing to draw", () => {
    expect(gappedPath([null, null], identity)).toBe("");
  });
});
