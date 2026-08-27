/**
 * GIST diversity selection — limbic's public surface over the index-based core
 * in `src/internal/gist.ts`.
 *
 * Two entry points, because they answer different questions:
 *
 * * {@link gistSelect} is the memory-engine one — ids in, ids out.
 * * {@link gistSelectFull} is the conformance one — 0-based row indices plus
 *   every quantity divsel's `docs/CONFORMANCE.md` compares, which is what
 *   `test/diversity.golden.test.ts` runs the 22 golden cases through.
 *
 * `lam` defaults to **0.5** here — the value the origin engine's plan gives
 * `ORIGIN_MEMORY_LAMBDA`. divsel's own default is `1.0`; the difference is
 * deliberate and is called out in the README's parity section.
 */

import {
  Coverage,
  DiversityError,
  FacilityLocation,
  gist,
  Linear,
  Points,
  type DiameterMode,
  type GistResult,
  type Metric,
  type Utility,
  type UtilityKind,
} from "./internal/gist.js";

export {
  DiversityError,
  F32_EPSILON,
  type DiameterMode,
  type DiversityErrorCode,
  type GistResult,
  type Metric,
  type Stage,
  type UtilityKind,
} from "./internal/gist.js";

/** Per-point weights (linear), per-point item-id lists (coverage), or nothing. */
export type Utilities =
  | ReadonlyArray<number>
  | ReadonlyArray<ReadonlyArray<number>>
  | null
  | undefined;

/** Everything beyond `k`, `lam` and `eps` that the GIST contract exposes. */
export interface GistSelectOptions {
  /** Default `"cosine"` — divsel's Python default too. */
  metric?: Metric;
  /** Default `"linear"`. */
  utility?: UtilityKind;
  /** Rule 6: sweep `{dist(u,v)/2}` instead of the geometric grid. */
  exhaustiveThresholds?: boolean;
  /** Rule 9: `"exact"` (default) or the farthest-point double sweep. */
  diameter?: DiameterMode;
  /** Double sweeps under `diameter: "approx"`. `0` means 1; above `n` means `n`. */
  diameterSweeps?: number;
}

function buildUtility(
  pts: Points,
  kind: UtilityKind,
  utilities: Utilities,
): Utility {
  switch (kind) {
    case "linear": {
      // Rule 16: `null` means uniform unit weights, so `g(S) === |S|`.
      if (utilities === null || utilities === undefined) return Linear.uniform(pts.n);
      return new Linear(utilities as ReadonlyArray<number>);
    }
    case "coverage": {
      if (utilities === null || utilities === undefined) {
        throw new DiversityError(
          "CoverageLength",
          "gistSelect: the coverage utility needs one item-id list per point",
        );
      }
      const sets = utilities as ReadonlyArray<ReadonlyArray<number>>;
      // Rule 17: the universe is inferred as `max id + 1`; it only bounds the
      // ids, it never changes `g`.
      return new Coverage(sets, Coverage.inferUniverse(sets));
    }
    case "facility_location": {
      // Rule 8: built from the vectors alone — `utilities` is `null` for it.
      return new FacilityLocation(pts);
    }
    default: {
      const never: never = kind;
      throw new DiversityError("WeightsLength", `gistSelect: unknown utility ${String(never)}`);
    }
  }
}

/**
 * Run GIST over `vectors` and report the full result, with 0-based row indices.
 *
 * `selected` is in **selection order** — ascending only when the diametrical
 * pair won. Every other field is defined by divsel's `docs/CONFORMANCE.md`:
 * `f = g + lam*div` (exactly `g` at `lam === 0`), `div` is the minimum pairwise
 * distance or `d_max` when at most one point is selected, `threshold` is `0`
 * for `"greedy"`, `d_max` for `"diameter_pair"` and the winning grid entry for
 * `"sweep"`.
 *
 * @throws {DiversityError} on an empty point matrix, `k < 1`, an `eps` outside
 * `[1.1920929e-7, 1]`, a negative or non-finite `lam`, a zero-norm row under
 * the cosine metric, or a utility whose tables do not match the point set.
 * Never an empty result (rule 13).
 */
export function gistSelectFull(
  vectors: ReadonlyArray<ArrayLike<number>>,
  utilities: Utilities,
  k: number,
  lam = 0.5,
  eps = 0.1,
  opts: GistSelectOptions = {},
): GistResult {
  const pts = new Points(vectors, opts.metric ?? "cosine");
  const util = buildUtility(pts, opts.utility ?? "linear", utilities);
  return gist(pts, util, {
    k,
    lam,
    eps,
    exhaustiveThresholds: opts.exhaustiveThresholds ?? false,
    diameter: opts.diameter ?? "exact",
    diameterSweeps: opts.diameterSweeps ?? 1,
  });
}

/**
 * The id-keyed wrapper: pick at most `k` of `ids` maximising
 * `g(S) + lam * div(S)`, returned in selection order.
 *
 * `ids[i]` names `vectors[i]`; the mapping is the only thing this adds over
 * {@link gistSelectFull}. `utilities` of `undefined` (or `null`) means uniform
 * unit weights, so `g(S)` is just `|S|` and the answer is decided by diversity.
 */
export function gistSelect(
  ids: ReadonlyArray<string>,
  vectors: ReadonlyArray<ArrayLike<number>>,
  utilities: ReadonlyArray<number> | undefined,
  k: number,
  lam = 0.5,
  eps = 0.1,
  opts: { metric?: Metric } = {},
): string[] {
  if (ids.length !== vectors.length) {
    throw new DiversityError(
      "LengthNotMultipleOfDim",
      `gistSelect: ${ids.length} ids for ${vectors.length} vectors`,
    );
  }
  const result = gistSelectFull(vectors, utilities ?? null, k, lam, eps, {
    metric: opts.metric ?? "cosine",
    utility: "linear",
  });
  return result.selected.map((index) => ids[index]!);
}
