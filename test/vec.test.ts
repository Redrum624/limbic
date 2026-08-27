import { describe, expect, it } from "vitest";

import { cosine, l2normalize } from "../src/internal/vec.js";

const TOL = 1e-6;

describe("cosine", () => {
  it("is 1 for identical vectors", () => {
    const v = Float32Array.from([0.1, 0.2, 0.3, 0.4]);
    expect(cosine(v, v)).toBeCloseTo(1, 6);
  });

  it("is 0 for orthogonal vectors", () => {
    const a = Float32Array.from([1, 0, 0, 0]);
    const b = Float32Array.from([0, 1, 0, 0]);
    expect(Math.abs((cosine(a, b) as number) - 0)).toBeLessThanOrEqual(TOL);
  });

  it("is -1 for opposed vectors", () => {
    const a = Float32Array.from([1, 2, 3]);
    const b = Float32Array.from([-1, -2, -3]);
    expect(cosine(a, b)).toBeCloseTo(-1, 6);
  });

  it("is scale-invariant", () => {
    const a = Float32Array.from([1, 2, 3]);
    const b = Float32Array.from([10, 20, 30]);
    expect(cosine(a, b)).toBeCloseTo(1, 6);
  });

  it("reproduces the golden fixture's stub similarities", () => {
    const q = Float32Array.from([1, 0, 0, 0]);
    // "User adores flat whites" — 0.6/|(0.6,0.8)| = 0.6
    expect(cosine(q, Float32Array.from([0.6, 0.8, 0, 0]))).toBeCloseTo(0.6, 6);
    // "User drinks coffee every morning" — 1/|(1,0.75)| = 0.8
    expect(cosine(q, Float32Array.from([1, 0.75, 0, 0]))).toBeCloseTo(0.8, 6);
  });

  it("accepts plain number arrays as well as Float32Array", () => {
    expect(cosine([1, 0], Float32Array.from([1, 0]))).toBeCloseTo(1, 6);
  });

  // ── the MISSING cases: null, never 0 ──────────────────────────────────────

  it("returns null — not 0 — when either side is absent", () => {
    const v = Float32Array.from([1, 0]);
    expect(cosine(null, v)).toBeNull();
    expect(cosine(v, null)).toBeNull();
    expect(cosine(undefined, v)).toBeNull();
    expect(cosine(v, undefined)).toBeNull();
  });

  it("returns null for empty vectors", () => {
    expect(cosine(new Float32Array(0), new Float32Array(0))).toBeNull();
  });

  it("returns null for mismatched dimensions (two embedding models)", () => {
    expect(cosine(Float32Array.from([1, 0]), Float32Array.from([1, 0, 0]))).toBeNull();
  });

  it("returns null for a zero norm", () => {
    expect(cosine(Float32Array.from([0, 0]), Float32Array.from([1, 1]))).toBeNull();
    expect(cosine(Float32Array.from([1, 1]), Float32Array.from([0, 0]))).toBeNull();
  });

  it("returns null for non-finite components", () => {
    expect(cosine([Number.NaN, 1], [1, 1])).toBeNull();
    expect(cosine([Number.POSITIVE_INFINITY, 1], [1, 1])).toBeNull();
  });

  it("never throws", () => {
    expect(() => cosine([1, 2], [3, 4])).not.toThrow();
    expect(() => cosine(null, null)).not.toThrow();
  });
});

describe("l2normalize", () => {
  it("produces a unit vector", () => {
    const out = l2normalize([3, 4]) as Float32Array;
    expect(out[0]).toBeCloseTo(0.6, 6);
    expect(out[1]).toBeCloseTo(0.8, 6);
    expect(cosine(out, [3, 4])).toBeCloseTo(1, 6);
  });

  it("returns a new Float32Array and does not mutate the input", () => {
    const input = Float32Array.from([3, 4]);
    const out = l2normalize(input) as Float32Array;
    expect(out).not.toBe(input);
    expect(Array.from(input)).toEqual([3, 4]);
  });

  it("returns null for a zero, empty, absent or non-finite vector", () => {
    expect(l2normalize([0, 0])).toBeNull();
    expect(l2normalize(new Float32Array(0))).toBeNull();
    expect(l2normalize(null)).toBeNull();
    expect(l2normalize(undefined)).toBeNull();
    expect(l2normalize([Number.NaN, 1])).toBeNull();
  });
});
