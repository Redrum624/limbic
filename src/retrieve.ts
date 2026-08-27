/**
 * The retrieval pipeline: score the pool, then diversify it with GIST.
 *
 * Ported from the origin engine `server/memory/retrieval_service.py` — `retrieve_relevant`
 * and the `_apply_diversity` / `_fill_preserving_spread` pair (`:300-424`).
 *
 * The shape is deliberately the origin engine's, including the parts that look like
 * over-engineering until you have watched them fail:
 *
 * * **Diversity changes membership, never order.** The scored pool is sorted
 *   descending, so a position in it *is* its rank; the result is assembled by
 *   index and returned in index order, so the ordering and the tie-break are
 *   identical to the no-diversity path and only *which* rows survive changes.
 * * **A memory with no embedding is diversity-neutral, not excluded.** It
 *   contributes no distance edges — it is not a point GIST sees at all — so it
 *   can take a slot GIST left open without lowering the minimum pairwise
 *   distance, which is defined over the embedded rows.
 * * **The fill may not undo the selector.** Topping the result up by score
 *   alone puts back exactly the near-duplicates GIST just rejected, which makes
 *   diversity mode return a set whose spread is bit-identical to plain top-`k`.
 *   So a candidate is admitted only if it sits at least `div(picked)` from
 *   every embedded row already chosen; otherwise the slot stays empty and the
 *   result is **short**. That short return is the point: it is the only way the
 *   selector's collapse is visible from outside, and it is the signal that
 *   `lambda` is too high for this corpus.
 * * **Diversity is never fatal.** Anything thrown out of `gistSelectFull`
 *   degrades to the top `k` by score. It is a ranking preference, not a reason
 *   to lose a chat turn.
 */

import { gistSelectFull } from "./diversity.js";
import { DEFAULT_ALL_LIMIT, comparePool } from "./internal/store-shared.js";
import { extractKeywords, scoreMemory } from "./internal/scoring.js";
import type { MemoryStore } from "./store.js";
import { DEFAULT_WEIGHTS, type Embedder, type Memory, type ScoreWeights } from "./types.js";

/** One scored row of the result. */
export interface ScoredMemory {
  memory: Memory;
  /** The final score in `[0, 1]` — the blend when a vector was comparable. */
  score: number;
}

/** Everything `retrieve` needs that is not the query. */
export interface RetrieveOptions {
  /** How many rows to score before diversifying. Default 50. */
  pool?: number;
  /** GIST's diversity weight. Default 0.5 — the origin engine's `ORIGIN_MEMORY_LAMBDA`. */
  lambda?: number;
  /** Channel weights. Default {@link DEFAULT_WEIGHTS}. */
  weights?: ScoreWeights;
  /** Embeds the query once. Omit for keyword-only scoring. */
  embedder?: Embedder;
  /** Emotion to prefer, feeding the target/family boosts. */
  targetEmotion?: string;
  /** Explicit clock. Scoring never reads the wall clock on its own. */
  now?: Date;
  /** `false` returns the top `k` by score, unchanged. Default `true`. */
  diversify?: boolean;
}

/** The default scored-pool size (`ORIGIN_MEMORY_POOL`). */
export const DEFAULT_POOL = 50;

/** The default diversity weight — the origin engine's `ORIGIN_MEMORY_LAMBDA`, not divsel's 1.0. */
export const DEFAULT_LAMBDA = 0.5;

/**
 * Embed `query` once, or return `undefined`. **Never rejects.**
 *
 * the origin engine's rule: an embedding failure degrades the cosine channel to MISSING and
 * costs nothing else. `scoreMemory` already treats a missing vector as
 * "cannot be compared" rather than as a similarity of 0, so there is nothing
 * further to handle downstream.
 */
export async function embedQuery(
  embedder: Embedder | undefined,
  query: string,
): Promise<Float32Array | undefined> {
  if (embedder === undefined) return undefined;
  try {
    const vectors = await embedder.embed([query]);
    const first = vectors[0];
    return first instanceof Float32Array && first.length > 0 ? first : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Score every memory the store hands back and sort it descending.
 *
 * The tail of the comparison is `comparePool`, the same total order the stores
 * use, so two stores holding the same rows produce the same ranking and equal
 * scores never come back in insertion order.
 */
export function scorePool(
  memories: readonly Memory[],
  query: { keywords: string[]; embedding?: Float32Array; targetEmotion?: string },
  now: Date,
  weights: ScoreWeights,
): ScoredMemory[] {
  return memories
    .map((memory) => ({ memory, score: scoreMemory(memory, query, now, weights) }))
    .sort((a, b) => (a.score === b.score ? comparePool(a.memory, b.memory) : b.score - a.score));
}

/**
 * the origin engine's `_apply_diversity`: pick at most `k` diverse-and-high-scoring rows out
 * of an already-sorted pool, returned in pool order.
 */
export function diversify(
  pool: readonly ScoredMemory[],
  k: number,
  lambda: number,
): ScoredMemory[] {
  // Positions in the pool that carry a vector. Below two of them there is no
  // pairwise distance to maximise, so there is nothing GIST can do.
  const embedded: number[] = [];
  for (const [i, row] of pool.entries()) {
    if (row.memory.embedding !== undefined && row.memory.embedding.length > 0) embedded.push(i);
  }
  if (embedded.length < 2) return pool.slice(0, k);

  const rows = embedded.map((i) => pool[i]!.memory.embedding!);
  // Every row must share a dimension for the cosine metric; a ragged corpus is
  // the caller's problem to notice, not a reason to throw inside a chat turn.
  const dim = rows[0]!.length;
  if (rows.some((row) => row.length !== dim)) return pool.slice(0, k);

  let picked: number[];
  let floor: number;
  try {
    const result = gistSelectFull(
      rows,
      // Scores are already in [0, 1]; the max() is for the "weights must be
      // >= 0" precondition, not for the arithmetic.
      embedded.map((i) => Math.max(0, pool[i]!.score)),
      k,
      lambda,
      0.1,
      { metric: "cosine", utility: "linear" },
    );
    picked = result.selected;
    floor = result.div;
  } catch {
    // Never fatal: a ranking preference is not worth a lost turn.
    return pool.slice(0, k);
  }

  const chosenRows = new Set(picked);
  const chosen = new Set(picked.map((row) => embedded[row]!));
  if (chosen.size >= k) return [...chosen].sort((a, b) => a - b).map((i) => pool[i]!);

  // `_fill_preserving_spread`: top up towards k without lowering div(chosen).
  const rowOf = new Map<number, number>();
  for (const [row, position] of embedded.entries()) rowOf.set(position, row);
  const distance = (a: number, b: number): number => cosineDistance(rows[a]!, rows[b]!);

  for (let i = 0; i < pool.length && chosen.size < k; i++) {
    if (chosen.has(i)) continue;
    const row = rowOf.get(i);
    if (row !== undefined) {
      // An embedded candidate is admitted only if it holds the floor against
      // every embedded row already chosen. In practice it almost never is:
      // whenever greedy or the sweep won, GIST stopped early precisely because
      // no unpicked row cleared its threshold, and the floor is at least that.
      let admissible = true;
      for (const other of chosenRows) {
        if (distance(row, other) < floor) {
          admissible = false;
          break;
        }
      }
      if (!admissible) continue;
      chosenRows.add(row);
    }
    // A bare memory falls straight through: it contributes no edges, so it
    // cannot lower a minimum taken over the rows that do have vectors.
    chosen.add(i);
  }

  return [...chosen].sort((a, b) => a - b).map((i) => pool[i]!);
}

/**
 * The 0.1.0 retrieval pipeline: read the pool, score it, diversify it.
 *
 * Returns at most `k` rows, in pool (i.e. score) order. It can return **fewer**
 * than `k` — see the note on the fill at the top of this file.
 */
export async function retrieve(
  store: MemoryStore,
  query: string,
  k: number,
  options: RetrieveOptions = {},
): Promise<ScoredMemory[]> {
  if (!Number.isInteger(k) || k < 1) {
    throw new TypeError(`retrieve: k must be a positive integer, got ${k}`);
  }
  const pool = options.pool ?? DEFAULT_POOL;
  const weights = options.weights ?? DEFAULT_WEIGHTS;
  const now = options.now ?? new Date();

  // One store read, capped the way the origin engine caps it: the pool is a chat turn's
  // budget, not a table scan.
  const rows = await store.all(Math.max(pool, DEFAULT_ALL_LIMIT));
  const embedding = await embedQuery(options.embedder, query);

  const scoreQuery: { keywords: string[]; embedding?: Float32Array; targetEmotion?: string } = {
    keywords: [...extractKeywords(query)],
  };
  if (embedding !== undefined) scoreQuery.embedding = embedding;
  if (options.targetEmotion !== undefined) scoreQuery.targetEmotion = options.targetEmotion;

  const scored = scorePool(rows, scoreQuery, now, weights).slice(0, pool);
  if (options.diversify === false) return scored.slice(0, k);
  return diversify(scored, k, options.lambda ?? DEFAULT_LAMBDA);
}

/**
 * `clamp(1 - a.b, 0, 2)` on L2-normalised rows — the same cosine distance
 * `gistSelectFull` uses, recomputed here for the fill's floor test.
 */
function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const norm = Math.sqrt(na) * Math.sqrt(nb);
  if (norm === 0 || !Number.isFinite(norm)) return 0;
  const raw = 1 - dot / norm;
  return raw < 0 ? 0 : raw > 2 ? 2 : raw;
}
