/**
 * GIST (arXiv:2405.18754v3), ported index-for-index from divsel — the reference
 * implementation — at commit `9262375`, under the 18-rule contract in divsel's
 * `docs/CONFORMANCE.md` at that same commit.
 *
 * Everything here is 0-based **row indices**. `src/diversity.ts` is the thin
 * id-keyed wrapper; nothing in this file knows what a `Memory` is.
 *
 * Arithmetic width is part of the contract. divsel computes every distance in
 * `f32` with a fixed 16-accumulator reduction order, and its golden values were
 * generated from that arithmetic; `Math.fround` reproduces it here so the two
 * implementations agree bit-for-bit on the distance kernels rather than merely
 * within tolerance. Utilities and the objective are `f64`, as in divsel.
 *
 * Paper Algorithm 1, counting the `function` header as line 1:
 *
 *   1: function GIST(V, g, k, eps)
 *   2:   S <- GreedyIndependentSet(V, g, 0, k)
 *   3:   d_max = max_(u,v) dist(u,v)
 *   4:   T <- (u,v) realising d_max
 *   5:   if f(T) > f(S) and k >= 2 then S <- T
 *   7:   D <- ((1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps)
 *   8:   for d in D: T <- GIS(V, g, d, k); if f(T) >= f(S) then S <- T
 *  12:   return S
 */

/** Distance metric. divsel's Python default is `"cosine"`. */
export type Metric = "cosine" | "euclidean";

/** Which submodular/modular utility supplies `g`. */
export type UtilityKind = "linear" | "coverage" | "facility_location";

/** Which branch of Algorithm 1 produced the answer. */
export type Stage = "greedy" | "diameter_pair" | "sweep";

/** How line 3's diameter is obtained. */
export type DiameterMode = "exact" | "approx";

/** Error codes mirroring divsel's `DivselError` variants (CONFORMANCE rule 13). */
export type DiversityErrorCode =
  | "ZeroDim"
  | "EmptyInput"
  | "LengthNotMultipleOfDim"
  | "NonFinite"
  | "ZeroNormRow"
  | "InvalidK"
  | "InvalidEps"
  | "InvalidLambda"
  | "WeightsLength"
  | "CoverageLength"
  | "CoverageItemOutOfRange";

/** Thrown for every invalid input. Rule 13: never an empty result. */
export class DiversityError extends Error {
  readonly code: DiversityErrorCode;
  constructor(code: DiversityErrorCode, message: string) {
    super(message);
    this.name = "DiversityError";
    this.code = code;
  }
}

/** `f32::EPSILON` — the lower bound on `eps` (CONFORMANCE rule 13). */
export const F32_EPSILON = 1.1920928955078125e-7;

// ---------------------------------------------------------------------------
// f32 kernels — divsel `crates/divsel/src/metric.rs`
// ---------------------------------------------------------------------------

/** divsel's fixed logical accumulator count. */
const LANES = 16;

const fr = Math.fround;

/**
 * `sum a[i]*b[i]` over one row pair, in `f32`, with divsel's fixed reduction
 * order: 16 independent accumulators, the tail folded into accumulator
 * `idx % 16`, then a final left-to-right reduction in index order.
 */
function dotScalar(data: Float32Array, ao: number, bo: number, dim: number): number {
  const acc = new Float32Array(LANES);
  const full = Math.floor(dim / LANES) * LANES;
  for (let base = 0; base < full; base += LANES) {
    for (let l = 0; l < LANES; l++) {
      const p = fr(data[ao + base + l]! * data[bo + base + l]!);
      acc[l] = acc[l]! + p;
    }
  }
  for (let l = 0; l < dim - full; l++) {
    const p = fr(data[ao + full + l]! * data[bo + full + l]!);
    acc[l] = acc[l]! + p;
  }
  let total = 0;
  for (let l = 0; l < LANES; l++) total = fr(total + acc[l]!);
  return total;
}

/** `sum (a[i]-b[i])^2`, sharing `dotScalar`'s reduction order. */
function sqEuclidScalar(data: Float32Array, ao: number, bo: number, dim: number): number {
  const acc = new Float32Array(LANES);
  const full = Math.floor(dim / LANES) * LANES;
  for (let base = 0; base < full; base += LANES) {
    for (let l = 0; l < LANES; l++) {
      const d = fr(data[ao + base + l]! - data[bo + base + l]!);
      acc[l] = acc[l]! + fr(d * d);
    }
  }
  for (let l = 0; l < dim - full; l++) {
    const d = fr(data[ao + full + l]! - data[bo + full + l]!);
    acc[l] = acc[l]! + fr(d * d);
  }
  let total = 0;
  for (let l = 0; l < LANES; l++) total = fr(total + acc[l]!);
  return total;
}

/**
 * `a.total_cmp(&b) == Greater`, the ordering divsel resolves every argmax with.
 * Differs from `>` only on `NaN` and on `+0` against `-0`; both are cheap
 * enough to write out rather than assume away.
 */
function totalGreater(a: number, b: number): boolean {
  if (a > b) return true;
  if (a < b) return false;
  if (Number.isNaN(a)) return !Number.isNaN(b);
  if (Number.isNaN(b)) return false;
  // Equal and both non-NaN: only the signed zeros can still be ordered.
  return Object.is(a, 0) && Object.is(b, -0);
}

// ---------------------------------------------------------------------------
// Points — divsel `crates/divsel/src/points.rs`
// ---------------------------------------------------------------------------

/** A row-major `f32` point set with a metric, matching divsel's `Points`. */
export class Points {
  readonly n: number;
  readonly dim: number;
  readonly metric: Metric;
  private readonly data: Float32Array;
  private diameterCache: [number, number, number] | null = null;

  constructor(vectors: ReadonlyArray<ArrayLike<number>>, metric: Metric) {
    if (vectors.length === 0) {
      throw new DiversityError("EmptyInput", "gist: the point matrix is empty");
    }
    const dim = vectors[0]!.length;
    if (dim === 0) {
      throw new DiversityError("ZeroDim", "gist: vectors must have at least one dimension");
    }
    const n = vectors.length;
    const data = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      const row = vectors[i]!;
      if (row.length !== dim) {
        throw new DiversityError(
          "LengthNotMultipleOfDim",
          `gist: row ${i} has length ${row.length}, expected ${dim}`,
        );
      }
      for (let j = 0; j < dim; j++) {
        const value = row[j]!;
        if (!Number.isFinite(value)) {
          throw new DiversityError(
            "NonFinite",
            `gist: coordinate (${i}, ${j}) is ${value}; every coordinate must be finite`,
          );
        }
        data[i * dim + j] = value;
      }
    }
    if (metric === "cosine") {
      // Rule 14: rows are L2-normalised on construction; a row that cannot be
      // normalised is an error, not a silently-zero row.
      for (let i = 0; i < n; i++) {
        const off = i * dim;
        const norm = fr(Math.sqrt(dotScalar(data, off, off, dim)));
        if (norm === 0 || !Number.isFinite(norm)) {
          throw new DiversityError(
            "ZeroNormRow",
            `gist: row ${i} has L2 norm ${norm} and cannot be normalised for the cosine metric`,
          );
        }
        for (let j = 0; j < dim; j++) data[off + j] = data[off + j]! / norm;
      }
    }
    this.n = n;
    this.dim = dim;
    this.metric = metric;
    this.data = data;
  }

  /** Rule 14: cosine is `clamp(1 - a.b, 0, 2)` on normalised rows; `dist(i,i) == 0`. */
  dist(i: number, j: number): number {
    if (i === j) return 0;
    const ao = i * this.dim;
    const bo = j * this.dim;
    if (this.metric === "cosine") {
      const raw = fr(1 - dotScalar(this.data, ao, bo, this.dim));
      return raw < 0 ? 0 : raw > 2 ? 2 : raw;
    }
    return fr(Math.sqrt(sqEuclidScalar(this.data, ao, bo, this.dim)));
  }

  /**
   * Rule 9: the exact diameter over `u < v`, reduced under the total order
   * *larger distance, then smaller `u`, then smaller `v`* — so ties resolve to
   * the lexicographically smallest pair. `n < 2` gives `(0, 0, 0)`.
   */
  diameter(): [number, number, number] {
    if (this.diameterCache) return this.diameterCache;
    let out: [number, number, number];
    if (this.n < 2) {
      out = [0, 0, 0];
    } else {
      let best: [number, number, number] = [Number.NEGATIVE_INFINITY, -1, -1];
      for (let i = 0; i < this.n; i++) {
        for (let j = i + 1; j < this.n; j++) {
          best = betterPair(best, [this.dist(i, j), i, j]);
        }
      }
      out = best;
    }
    this.diameterCache = out;
    return out;
  }
}

/** divsel's `better_pair`: larger distance, then smaller `u`, then smaller `v`. */
function betterPair(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  if (totalGreater(a[0], b[0])) return a;
  if (totalGreater(b[0], a[0])) return b;
  if (a[1] !== b[1]) return a[1] < b[1] ? a : b;
  return a[2] <= b[2] ? a : b;
}

// ---------------------------------------------------------------------------
// Utilities — divsel `crates/divsel/src/utility.rs`
// ---------------------------------------------------------------------------

/** Marginal-gain oracle for `g`. Selection state lives in the utility. */
export interface Utility {
  /** `g(v | S)`, in `f64`. */
  marginal(v: number, selected: readonly number[], pts: Points): number;
  /** Fold `v` into the running selection. */
  commit(v: number, pts: Points): void;
  /** Back to `g(empty) = 0`. */
  reset(): void;
  /** Modular utilities skip divsel's lazy path; kept for parity of shape. */
  readonly isLinear: boolean;
  /** divsel checks the utility's own tables after `k`/`eps`/`lambda`. */
  validate(pts: Points): void;
}

/** Rule 16: `g(S) = sum of w_v over S`; marginals independent of `S`. */
export class Linear implements Utility {
  readonly isLinear = true;
  constructor(private readonly weights: readonly number[]) {}

  /** Rule 16: `utilities: null` under a linear utility means uniform unit weights. */
  static uniform(n: number): Linear {
    return new Linear(new Array<number>(n).fill(1));
  }

  marginal(v: number): number {
    return this.weights[v]!;
  }
  commit(): void {}
  reset(): void {}
  validate(pts: Points): void {
    if (this.weights.length !== pts.n) {
      throw new DiversityError(
        "WeightsLength",
        `gist: ${this.weights.length} weights for ${pts.n} points`,
      );
    }
    for (let i = 0; i < this.weights.length; i++) {
      const w = this.weights[i]!;
      if (!Number.isFinite(w) || w < 0) {
        throw new DiversityError(
          "WeightsLength",
          `gist: weight ${i} is ${w}; every weight must be finite and >= 0`,
        );
      }
    }
  }
}

/** Rule 17: `g(S) = |union of the item sets|`; unweighted, ids deduped per row. */
export class Coverage implements Utility {
  readonly isLinear = false;
  private readonly sets: number[][];
  private readonly covered: Uint8Array;

  constructor(sets: ReadonlyArray<ReadonlyArray<number>>, universe: number) {
    this.sets = sets.map((items, row) => {
      for (const item of items) {
        if (!Number.isInteger(item) || item < 0 || item > 0xffffffff) {
          throw new DiversityError(
            "CoverageItemOutOfRange",
            `gist: coverage row ${row} holds item ${item}, outside [0, 2**32 - 1]`,
          );
        }
        if (item >= universe) {
          throw new DiversityError(
            "CoverageItemOutOfRange",
            `gist: coverage row ${row} holds item ${item}, universe is ${universe}`,
          );
        }
      }
      return [...new Set(items)].sort((a, b) => a - b);
    });
    this.covered = new Uint8Array(universe);
  }

  /** Rule 17: the universe is inferred as `max id + 1`, `0` when every row is empty. */
  static inferUniverse(sets: ReadonlyArray<ReadonlyArray<number>>): number {
    let max = -1;
    for (const items of sets) for (const item of items) if (item > max) max = item;
    return max + 1;
  }

  marginal(v: number): number {
    let count = 0;
    for (const item of this.sets[v]!) if (this.covered[item] === 0) count++;
    return count;
  }
  commit(v: number): void {
    for (const item of this.sets[v]!) this.covered[item] = 1;
  }
  reset(): void {
    this.covered.fill(0);
  }
  validate(pts: Points): void {
    if (this.sets.length !== pts.n) {
      throw new DiversityError(
        "CoverageLength",
        `gist: ${this.sets.length} coverage rows for ${pts.n} points`,
      );
    }
  }
}

/** divsel's `usable_scale`: anything not finite and positive falls back to 1. */
function usableScale(scale: number): number {
  return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

/**
 * Rule 8: `sim(i, j) = max(0, 1 - dist(i, j)/scale)`, `sim(i, i) = 1`;
 * `scale = 1.0` for cosine and the **exact** diameter for euclidean — exact
 * whatever the diameter mode is, which is rule 10's one exception.
 */
export class FacilityLocation implements Utility {
  readonly isLinear = false;
  private readonly scale: number;
  private readonly best: Float64Array;

  constructor(pts: Points) {
    this.scale = usableScale(pts.metric === "cosine" ? 1 : pts.diameter()[0]);
    this.best = new Float64Array(pts.n);
  }

  private sim(i: number, j: number, pts: Points): number {
    return Math.max(0, 1 - pts.dist(i, j) / this.scale);
  }

  marginal(v: number, _selected: readonly number[], pts: Points): number {
    let total = 0;
    for (let i = 0; i < this.best.length; i++) {
      total += Math.max(0, this.sim(i, v, pts) - this.best[i]!);
    }
    return total;
  }
  commit(v: number, pts: Points): void {
    for (let i = 0; i < this.best.length; i++) {
      const similarity = this.sim(i, v, pts);
      if (similarity > this.best[i]!) this.best[i] = similarity;
    }
  }
  reset(): void {
    this.best.fill(0);
  }
  validate(pts: Points): void {
    if (this.best.length !== pts.n) {
      throw new DiversityError(
        "WeightsLength",
        `gist: facility-location cache built for ${this.best.length} points, got ${pts.n}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Threshold sets
// ---------------------------------------------------------------------------

/**
 * Rule 7: `D = ((1+eps)^i * eps*d_max/2 : (1+eps)^i <= bound)`, built by
 * repeated multiplication in `f64` — **never** `log` + floor, because the entry
 * count is part of the contract. Each entry is cast to `f32`; consecutive
 * duplicates are dropped, which only ever collapses a zero-diameter set.
 *
 * `bound` is `2/eps` under the exact diameter and `4/eps` under `approx`
 * (rule 10); `eps` and `dMax` arrive already widened from `f32`.
 */
export function thresholdsWithBound(dMax: number, eps: number, bound: number): number[] {
  if (!(eps >= F32_EPSILON && Number.isFinite(eps))) return [];
  const out: number[] = [];
  let p = 1;
  while (p <= bound) {
    const entry = fr((p * eps * dMax) / 2);
    if (out.length === 0 || out[out.length - 1] !== entry) out.push(entry);
    p *= 1 + eps;
  }
  return out;
}

/** {@link thresholdsWithBound} at the paper's `2/eps` ceiling. */
export function thresholds(dMax: number, eps: number): number[] {
  const widened = fr(eps);
  return thresholdsWithBound(fr(dMax), widened, 2 / widened);
}

/**
 * Rule 6: `(dist(u,v)/2 : u <= v)`, ascending and exactly deduplicated. The
 * `u == v` pairs put `0` in the set, so with `d_max > 0` the sweep repeats the
 * line-2 greedy run and rule 2 relabels it — `"greedy"` is then unreachable.
 */
export function exhaustiveThresholdSet(pts: Points): number[] {
  const out: number[] = [];
  for (let u = 0; u < pts.n; u++) {
    for (let v = u; v < pts.n; v++) out.push(fr(pts.dist(u, v) / 2));
  }
  out.sort((a, b) => (totalGreater(a, b) ? 1 : totalGreater(b, a) ? -1 : 0));
  const deduped: number[] = [];
  for (const value of out) {
    if (deduped.length === 0 || deduped[deduped.length - 1] !== value) deduped.push(value);
  }
  return deduped;
}

// ---------------------------------------------------------------------------
// Greedy independent set (paper lines 2 and 9)
// ---------------------------------------------------------------------------

/**
 * `GIS(d)`: repeat up to `k` times — `C = (v not in S : dist(v, S) >= d)`
 * (rule 15, **non-strict**, with `dist(v, empty) = +Infinity`), pick
 * `argmax g(v|S)` breaking ties to the **lowest** index (rule 1), stop when `C`
 * is empty. Rule 12: the result can be shorter than `k`.
 *
 * This is divsel's `run_scan`, its own documented oracle for the lazy CELF path.
 */
export function greedyIndependentSet(
  pts: Points,
  util: Utility,
  d: number,
  k: number,
): number[] {
  util.reset();
  const budget = Math.min(k, pts.n);
  const nearest = new Float32Array(pts.n).fill(Number.POSITIVE_INFINITY);
  const chosen = new Uint8Array(pts.n);
  const selected: number[] = [];
  while (selected.length < budget) {
    let bestGain = 0;
    let bestIndex = -1;
    for (let v = 0; v < pts.n; v++) {
      // Rule 15: `v` is outside `S` and `dist(v, S) >= d` — non-strict, so two
      // points exactly `d` apart are feasible.
      if (chosen[v] === 1 || nearest[v]! < d) continue;
      const gain = util.marginal(v, selected, pts);
      // Rule 1: replace only on a strictly greater gain, scanning ascending, so
      // the lowest index wins every tie.
      if (bestIndex === -1 || totalGreater(gain, bestGain)) {
        bestGain = gain;
        bestIndex = v;
      }
    }
    if (bestIndex === -1) break;
    selected.push(bestIndex);
    util.commit(bestIndex, pts);
    chosen[bestIndex] = 1;
    nearest[bestIndex] = Number.NEGATIVE_INFINITY;
    for (let v = 0; v < pts.n; v++) {
      if (chosen[v] === 0) nearest[v] = Math.min(nearest[v]!, pts.dist(v, bestIndex));
    }
  }
  return selected;
}

/** `g(S)`, by replaying `S` through `util` in order and summing the marginals. */
export function evalG(util: Utility, s: readonly number[], pts: Points): number {
  util.reset();
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    total += util.marginal(s[i]!, s.slice(0, i), pts);
    util.commit(s[i]!, pts);
  }
  util.reset();
  return total;
}

/** Rule 4: `div(S)` is the min pairwise distance, or `d_max` when `|S| <= 1`. */
export function divWithDmax(pts: Points, s: readonly number[], dMax: number): number {
  if (s.length <= 1) return dMax;
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j < s.length; j++) best = Math.min(best, pts.dist(s[i]!, s[j]!));
  }
  return best;
}

/**
 * Rule 9's farthest-point double sweep. Each `argmax` runs over `j != source`,
 * so the pair stays distinct even when every distance is 0; ties go to the
 * lowest index, and sweeps are folded with the same total order the exact
 * diameter uses. `sweeps == 0` is treated as 1 and `sweeps > n` as `n`.
 */
export function approxDiameter(pts: Points, sweeps: number): [number, number, number] {
  if (pts.n < 2) return [0, 0, 0];
  let best: [number, number, number] = [Number.NEGATIVE_INFINITY, -1, -1];
  let current = 0;
  const runs = Math.min(Math.max(sweeps, 1), pts.n);
  for (let s = 0; s < runs; s++) {
    const a = farthestFrom(pts, current);
    const b = farthestFrom(pts, a);
    best = betterPair(best, [pts.dist(a, b), Math.min(a, b), Math.max(a, b)]);
    current = b;
  }
  return best;
}

function farthestFrom(pts: Points, from: number): number {
  let bestDistance = Number.NEGATIVE_INFINITY;
  let bestIndex = -1;
  for (let j = 0; j < pts.n; j++) {
    if (j === from) continue;
    const distance = pts.dist(from, j);
    if (totalGreater(distance, bestDistance)) {
      bestDistance = distance;
      bestIndex = j;
    }
  }
  return bestIndex;
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/** Everything GIST reports. `selected` holds 0-based row indices. */
export interface GistResult {
  /** Selection order for `"greedy"` / `"sweep"`, ascending for `"diameter_pair"`. */
  selected: number[];
  /** `f(S) = g(S) + lam * div(S)` — exactly `g(S)` at `lam == 0` (rule 18). */
  f: number;
  g: number;
  div: number;
  /** Rule 3: `0` for `"greedy"`, `d_max` for `"diameter_pair"`, the winning `d` for `"sweep"`. */
  threshold: number;
  stage: Stage;
  /** Exact under `diameter: "exact"`, the estimate `d_hat` under `"approx"`. */
  dMax: number;
}

/** Knobs of {@link gist}, mirroring divsel's `GistConfig`. */
export interface GistConfig {
  k: number;
  lam?: number;
  eps?: number;
  exhaustiveThresholds?: boolean;
  diameter?: DiameterMode;
  diameterSweeps?: number;
}

/**
 * Algorithm 1 verbatim: classic greedy, the diametrical pair compared strictly
 * (rule 3), then the threshold sweep folded ascending and compared non-strictly
 * (rule 2), so the **largest** threshold attaining the best `f` is reported.
 */
export function gist(pts: Points, util: Utility, cfg: GistConfig): GistResult {
  const lam = cfg.lam ?? 1;
  const eps = fr(cfg.eps ?? 0.1);

  // Rule 13, in divsel's own order: k, eps, lambda, then the utility's tables.
  if (!Number.isInteger(cfg.k) || cfg.k < 1) {
    throw new DiversityError("InvalidK", `gist: k must be a positive integer, got ${cfg.k}`);
  }
  if (!(eps >= F32_EPSILON && eps <= 1)) {
    throw new DiversityError(
      "InvalidEps",
      `gist: eps must lie in [${F32_EPSILON}, 1], got ${cfg.eps ?? 0.1}`,
    );
  }
  if (!(lam >= 0 && Number.isFinite(lam))) {
    throw new DiversityError("InvalidLambda", `gist: lambda must be finite and >= 0, got ${lam}`);
  }
  util.validate(pts);

  const n = pts.n;
  // Rule 12: a budget past the ground set is not an error, it just cannot bind.
  const k = Math.min(cfg.k, n);
  const mode = cfg.diameter ?? "exact";

  // Paper lines 3-4.
  const [dMax, u, v] =
    mode === "approx" ? approxDiameter(pts, cfg.diameterSweeps ?? 1) : pts.diameter();

  const evaluate = (selection: readonly number[]): [number, number, number] => {
    const gValue = evalG(util, selection, pts);
    const divValue = divWithDmax(pts, selection, dMax);
    // Rule 18: at lam == 0 the diversity term contributes exactly 0, written out
    // because div really can be +Infinity and 0 * Infinity is NaN.
    const weighted = lam === 0 ? 0 : lam * divValue;
    return [gValue + weighted, gValue, divValue];
  };

  // Paper line 2.
  let selected = greedyIndependentSet(pts, util, 0, k);
  let [f, g, div] = evaluate(selected);
  let stage: Stage = "greedy";
  let threshold = 0;

  // Paper lines 5-6: strict `>`, guarded by k >= 2 && n >= 2, reported ascending.
  if (k >= 2 && n >= 2) {
    const pair = [Math.min(u, v), Math.max(u, v)];
    const [fPair, gPair, divPair] = evaluate(pair);
    if (fPair > f) {
      selected = pair;
      f = fPair;
      g = gPair;
      div = divPair;
      stage = "diameter_pair";
      threshold = dMax;
    }
  }

  // Paper lines 7-11. Rule 5: at d_max == 0 every threshold is 0, so the sweep
  // would only repeat line 2 — it is skipped, and only `stage` can differ.
  if (dMax > 0) {
    const set = cfg.exhaustiveThresholds
      ? exhaustiveThresholdSet(pts)
      : thresholdsWithBound(dMax, eps, (mode === "approx" ? 4 : 2) / eps);
    for (const d of set) {
      const candidate = greedyIndependentSet(pts, util, d, k);
      const [fc, gc, divc] = evaluate(candidate);
      // Rule 2: non-strict, folded ascending — the largest threshold wins ties.
      if (fc >= f) {
        selected = candidate;
        f = fc;
        g = gc;
        div = divc;
        stage = "sweep";
        threshold = d;
      }
    }
  }

  util.reset();
  return { selected, f, g, div, threshold, stage, dMax };
}
