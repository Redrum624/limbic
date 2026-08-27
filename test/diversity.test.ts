/**
 * The parts of divsel's contract that **no fixture pins** — every case in
 * `golden-selection.json` is a valid input in `[-4, 4]`, so nothing there
 * reaches a validation error, an infinite `div`, or `k > n`.
 *
 * CONFORMANCE.md at `9262375` says so explicitly ("a port must reject these on
 * its own"), which makes these the rules most likely to rot. They are asserted
 * here instead.
 */

import { describe, expect, it } from "vitest";

import {
  DiversityError,
  F32_EPSILON,
  gistSelect,
  gistSelectFull,
} from "../src/diversity.js";

const line = [[0], [1], [5], [6]];

describe("rule 13 — every invalid input is an error, never an empty result", () => {
  it("rejects an empty point matrix", () => {
    expect(() => gistSelectFull([], null, 2)).toThrow(DiversityError);
    expect(() => gistSelectFull([], null, 2)).toThrow(/empty/i);
  });

  it("rejects k === 0", () => {
    const thrown = catchError(() => gistSelectFull(line, null, 0, 1, 0.1, { metric: "euclidean" }));
    expect(thrown.code).toBe("InvalidK");
  });

  it("rejects eps outside [f32::EPSILON, 1]", () => {
    for (const eps of [0, -0.5, 1.5, Number.NaN, F32_EPSILON / 2]) {
      const thrown = catchError(() =>
        gistSelectFull(line, null, 2, 1, eps, { metric: "euclidean" }),
      );
      expect(thrown.code, `eps ${eps}`).toBe("InvalidEps");
    }
    // Both bounds are accepted. `eps === 1` is cheap (two grid entries).
    expect(() => gistSelectFull(line, null, 2, 1, 1, { metric: "euclidean" })).not.toThrow();
    // `eps === f32::EPSILON` is accepted too, but the `2/eps` ceiling builds
    // 139,548,968 entries there — divsel documents that cost rather than
    // capping it. Rule 5 gives the only affordable way to assert the bound: a
    // zero-diameter point set skips the sweep entirely, so validation is the
    // only thing `eps` reaches.
    const coincident = [[2], [2]];
    expect(() =>
      gistSelectFull(coincident, null, 2, 1, F32_EPSILON, { metric: "euclidean" }),
    ).not.toThrow();
    expect(
      gistSelectFull(coincident, null, 2, 1, F32_EPSILON, { metric: "euclidean" }).stage,
    ).toBe("greedy");
  });

  it("rejects a negative or non-finite lambda", () => {
    for (const lam of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const thrown = catchError(() =>
        gistSelectFull(line, null, 2, lam, 0.1, { metric: "euclidean" }),
      );
      expect(thrown.code, `lam ${lam}`).toBe("InvalidLambda");
    }
  });

  it("rejects a weight vector of the wrong length", () => {
    const thrown = catchError(() =>
      gistSelectFull(line, [1, 1], 2, 1, 0.1, { metric: "euclidean" }),
    );
    expect(thrown.code).toBe("WeightsLength");
  });
});

describe("rule 14 — cosine rows are normalised, and an un-normalisable row is an error", () => {
  it("rejects a zero-norm row under cosine only", () => {
    const withZero = [[0, 0], [1, 0], [0, 1]];
    const thrown = catchError(() => gistSelectFull(withZero, null, 2, 1, 0.1, { metric: "cosine" }));
    expect(thrown.code).toBe("ZeroNormRow");
    // Euclidean never normalises, so it accepts the very same buffer.
    expect(() => gistSelectFull(withZero, null, 2, 1, 0.1, { metric: "euclidean" })).not.toThrow();
  });

  it("rejects a non-finite coordinate", () => {
    const thrown = catchError(() =>
      gistSelectFull([[0], [Number.NaN]], null, 2, 1, 0.1, { metric: "euclidean" }),
    );
    expect(thrown.code).toBe("NonFinite");
  });

  it("puts orthogonal unit rows exactly 1 apart and identical rows exactly 0 apart", () => {
    // dist = clamp(1 - a.b, 0, 2): orthogonal -> 1, identical -> 0, antipodal -> 2.
    const r = gistSelectFull([[1, 0], [0, 1]], null, 2, 1, 0.1, { metric: "cosine" });
    expect(r.div).toBe(1);
    expect(r.dMax).toBe(1);
    const antipodal = gistSelectFull([[1, 0], [-1, 0]], null, 2, 1, 0.1, { metric: "cosine" });
    expect(antipodal.dMax).toBe(2);
  });
});

describe("rule 12 — k > n clamps, and the result may be shorter than k", () => {
  it("returns every point when k exceeds n", () => {
    const r = gistSelectFull(line, null, 10, 0.25, 0.1, { metric: "euclidean" });
    expect(r.selected).toHaveLength(4);
  });
});

describe("rule 18 — lambda === 0 contributes exactly 0, whatever div is", () => {
  it("reports f === g rather than NaN when div is +Infinity", () => {
    // Points::new validates coordinates, not distances: these are finite f32s
    // whose squared difference overflows, so div and d_max are +Infinity. The
    // literal `0 * Infinity` would be NaN, and a NaN loses every comparison in
    // the sweep fold, so three fields would diverge with no error on either side.
    const far = [[-3e38], [3e38]];
    const r = gistSelectFull(far, null, 2, 0, 0.1, { metric: "euclidean" });
    expect(r.dMax).toBe(Number.POSITIVE_INFINITY);
    expect(r.div).toBe(Number.POSITIVE_INFINITY);
    expect(r.f).toBe(r.g);
    expect(r.f).toBe(2);
    expect(Number.isNaN(r.f)).toBe(false);
  });
});

describe("gistSelect — the id-keyed wrapper", () => {
  it("returns ids in selection order", () => {
    const ids = ["a", "b", "c", "d"];
    const picked = gistSelect(ids, line, undefined, 2, 1, 0.1, { metric: "euclidean" });
    // Uniform weights, so f is decided entirely by div: the widest pair.
    expect(picked).toEqual(["a", "d"]);
  });

  it("rejects an id list that does not match the vectors", () => {
    expect(() => gistSelect(["a"], line, undefined, 2)).toThrow(DiversityError);
  });

  it("defaults lam to 0.5 — the origin engine's default, not divsel's 1.0", () => {
    // Three points on a line at 0, 1, 10 with unit weights and k = 2.
    // At lam = 0.5 the pair {0, 2} scores 2 + 0.5*10 = 7 either way; what is
    // being pinned is that the default is applied at all rather than divsel's 1.
    const ids = ["a", "b", "c"];
    const pts = [[0], [1], [10]];
    const atDefault = gistSelect(ids, pts, undefined, 2, undefined, 0.1, { metric: "euclidean" });
    const atHalf = gistSelect(ids, pts, undefined, 2, 0.5, 0.1, { metric: "euclidean" });
    expect(atDefault).toEqual(atHalf);
  });
});

/** Narrow a thrown value to a `DiversityError` so `code` can be asserted. */
function catchError(run: () => unknown): DiversityError {
  try {
    run();
  } catch (error) {
    if (error instanceof DiversityError) return error;
    throw error;
  }
  throw new Error("expected a DiversityError, nothing was thrown");
}
