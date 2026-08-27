/**
 * Naive top-k against GIST diversification, on a corpus with known cluster
 * structure.
 *
 * The metric definitions are the origin engine plan Task 8's, so the two are comparable in
 * principle:
 *
 * * **redundancy** — the mean, over the selected rows, of each row's *maximum*
 *   cosine similarity to another selected row. 1.0 means every pick has a twin
 *   in the result; lower is better.
 * * **cluster coverage** — distinct clusters hit divided by clusters present,
 *   where clusters are the connected components of the graph on
 *   `cosine > 0.92`. Higher is better, and it is capped by `k / clusters`.
 * * **distinct facts per 1,000 prompt characters** — distinct clusters hit
 *   divided by the character cost of pasting the selection into a prompt. This
 *   is the one that maps to money: the same context window buys more facts.
 *
 * ⚠️ **The corpus is synthetic, and n = 1.** It is built from a fixed seed with
 * a cluster structure chosen to make redundancy visible, which is exactly the
 * shape retrieval-collapse takes but is not evidence about any real corpus.
 * the origin engine's `server/tools/memory_bench.py` exists and could supply a real one, but
 * as of 2026-08-27 the collapse-onset numbers in the origin engine's
 * `docs/benchmarks/memory-diversity.md` are **retracted** — the detector that
 * produced them could not fire — and a re-cut is in progress. The definitions
 * above are shared with it; the numbers here are not comparable to anything
 * published there yet, and no figure from it is quoted.
 *
 * Run: `npm run bench`
 */

import { bench, describe } from "vitest";

import { diversify, type ScoredMemory } from "../src/retrieve.js";
import type { Memory } from "../src/types.js";

const SEED = 0x5eed_11b1;
const CLUSTERS = 12;
const PER_CLUSTER = 5;
const DIM = 48;
const K = 8;
const LAMBDA = 0.5;
/** How much score a caller gives up per step to another cluster. */
const CLUSTER_SCORE_STEP = 0.03;
/** the origin engine plan Task 8's clustering threshold. */
const CLUSTER_EDGE = 0.92;

/** mulberry32 — small, seeded, and identical on every platform. */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b_79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function normalise(v: Float32Array): Float32Array {
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm);
  for (let i = 0; i < v.length; i++) v[i] = v[i]! / norm;
  return v;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

/** A row plus the cluster it was generated from, which the metrics need. */
interface Row extends ScoredMemory {
  cluster: number;
}

/**
 * `CLUSTERS` topics, `PER_CLUSTER` near-paraphrases each. Scores descend by
 * cluster with only a hair of jitter inside one — which is what a real query
 * produces, and why naive top-k clumps.
 */
function buildCorpus(): Row[] {
  const random = prng(SEED);
  const rows: Row[] = [];
  for (let c = 0; c < CLUSTERS; c++) {
    const base = normalise(Float32Array.from({ length: DIM }, () => random() * 2 - 1));
    for (let m = 0; m < PER_CLUSTER; m++) {
      const embedding = normalise(
        Float32Array.from(base, (x) => x + (random() * 2 - 1) * 0.04),
      );
      const memory: Memory = {
        id: `c${c}-m${m}`,
        // A realistic paraphrase length; the per-1,000-characters metric is
        // sensitive to it, so it is written out rather than left as an id.
        content: `Topic ${c}, phrasing ${m}: the user mentioned this in passing and it was worth keeping.`,
        category: "general",
        importance: 0.5,
        keywords: [],
        createdAt: "2026-08-01T00:00:00.000Z",
        lastAccessed: "2026-08-01T00:00:00.000Z",
        accessCount: 0,
        subject: "user",
        embedding,
      };
      rows.push({
        memory,
        score: 0.95 - c * CLUSTER_SCORE_STEP + random() * 0.004,
        cluster: c,
      });
    }
  }
  return rows.sort((a, b) => b.score - a.score);
}

/** Connected components of the graph on `cosine > CLUSTER_EDGE`. */
function componentsPresent(rows: readonly Row[]): number {
  const parent = rows.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i]!)));
  for (let i = 0; i < rows.length; i++) {
    for (let j = i + 1; j < rows.length; j++) {
      if (cosine(rows[i]!.memory.embedding!, rows[j]!.memory.embedding!) > CLUSTER_EDGE) {
        parent[find(i)] = find(j);
      }
    }
  }
  return new Set(rows.map((_, i) => find(i))).size;
}

interface Report {
  label: string;
  picked: number;
  redundancy: number;
  clustersHit: number;
  coverage: number;
  promptChars: number;
  factsPerThousand: number;
}

function measure(label: string, selection: readonly Row[], clustersPresent: number): Report {
  const vectors = selection.map((r) => r.memory.embedding!);
  let redundancySum = 0;
  for (let i = 0; i < vectors.length; i++) {
    let best = 0;
    for (let j = 0; j < vectors.length; j++) {
      if (i !== j) best = Math.max(best, cosine(vectors[i]!, vectors[j]!));
    }
    redundancySum += best;
  }
  const clustersHit = new Set(selection.map((r) => r.cluster)).size;
  const promptChars = selection.reduce((total, r) => total + r.memory.content.length, 0);
  return {
    label,
    picked: selection.length,
    redundancy: selection.length === 0 ? 0 : redundancySum / selection.length,
    clustersHit,
    coverage: clustersHit / clustersPresent,
    promptChars,
    factsPerThousand: (clustersHit / promptChars) * 1000,
  };
}

const corpus = buildCorpus();
const clustersPresent = componentsPresent(corpus);
const naive = corpus.slice(0, K) as Row[];

/**
 * `lambda` is swept rather than fixed, because a single number would be a
 * half-truth. GIST maximises `g(S) + lambda*div(S)`, so whether spreading wins
 * depends on how much score a caller must give up to reach another cluster.
 * Below the crossover GIST **correctly** returns top-k, and reporting only a
 * lambda above it would be picking the flattering row.
 */
const LAMBDA_SWEEP = [0, 0.5, 1, 2, 4, 8];

const reports = [
  measure("naive top-k", naive, clustersPresent),
  ...LAMBDA_SWEEP.map((lam) =>
    measure(`gistSelect lam=${lam}`, diversify(corpus, K, lam) as Row[], clustersPresent),
  ),
];

/* eslint-disable no-console */
console.log(
  `\ncorpus: ${corpus.length} memories, ${CLUSTERS} planted topics x ${PER_CLUSTER} paraphrases, ` +
    `dim ${DIM}, seed 0x${SEED.toString(16)}\n` +
    `components at cosine > ${CLUSTER_EDGE}: ${clustersPresent}   k = ${K}   ` +
    `score step between clusters: ${CLUSTER_SCORE_STEP}\n`,
);
console.table(
  reports.map((r) => ({
    strategy: r.label,
    picked: r.picked,
    redundancy: r.redundancy.toFixed(4),
    "clusters hit": `${r.clustersHit} / ${clustersPresent}`,
    coverage: r.coverage.toFixed(3),
    "prompt chars": r.promptChars,
    "facts / 1k chars": r.factsPerThousand.toFixed(3),
  })),
);
/* eslint-enable no-console */

describe("selection over a 60-memory clustered corpus", () => {
  bench("naive top-k", () => {
    corpus.slice(0, K);
  });

  bench("gistSelect (GIST diversification)", () => {
    diversify(corpus, K, LAMBDA);
  });
});
