/**
 * Tests for the Sharpe chart's outlier handling — extracted into
 * `computeSharpeDomain`/`clampForDisplay` specifically so this logic (the
 * exact thing that once let a 30,101 outlier crush the entire chart to a
 * flat line — CLAUDE.md §7) is unit-testable without rendering SVG.
 */
import { describe, expect, it } from "vitest";
import { SHARPE_OUTLIER_BOUND, clampForDisplay, computeSharpeDomain } from "./Charts.jsx";

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
