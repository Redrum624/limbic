/**
 * Golden scoring parity — limbic vs the origin engine.
 *
 * Fixture: `test/fixtures/golden-scoring.json`, copied verbatim from the origin engine
 * `test-assets/memory/golden-scoring.json` at commit
 * `4a307a3b3553eb0b3d112f9c24649628a9c6ed04` (upstream HEAD when copied:
 * `ff9c407cd292848155a6c2982ccc4d185c518b5b`), sha256
 * `fc1e983031ddb1cace78af238cdc1defe1f491020c83e671e49e0ad6453caabd`.
 * `test/fixtures.hash.test.ts` re-hashes it; `test/fixtures/PROVENANCE.md` has
 * the full record.
 *
 * SHAPE WARNING — this fixture is a SINGLE SCENARIO, not a `cases[]` array.
 * One query, one clock, one set of weights, five memories. The reader below is
 * written for that shape; do not "generalise" it into a case loop.
 *
 * What is asserted, and what is not:
 *   - `expected_base_score` and `expected_final_score` per memory, and both
 *     `expected_order_*` arrays. These are the contract.
 *   - The per-memory `channels` block is DISPLAY-ROUNDED to 6 dp upstream
 *     (`recency: 0.707107`), so it is reported on failure for diagnosis but is
 *     never an assertion target.
 *
 * Tolerance is the fixture's own: `score_abs = 1e-6`, ABSOLUTE.
 *
 * Clock: pinned to the fixture's `now`. `scoreMemoryDetailed` takes `now` as an
 * argument and never reads the wall clock, so this test has no `vi.setSystemTime`
 * and no hidden dependency on when it runs.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_WEIGHTS,
  EMBED_BLEND,
  type Memory,
  type MemoryCategory,
} from "../src/types.js";
import {
  BASE_BLEND,
  RECENCY_HALF_LIFE_DAYS,
  daysSince,
  scoreMemoryDetailed,
  type ScoreQuery,
} from "../src/internal/scoring.js";

// ── the fixture's shape ──────────────────────────────────────────────────────

interface GoldenMemory {
  key: string;
  content: string;
  category: string;
  importance: number;
  keywords: string[];
  days_since_access: number;
  last_accessed: string;
  created_at: string;
  access_count: number;
  source_message_id: string | null;
  subject: string;
  vector: number[] | null;
  channels: {
    recency: number;
    importance: number;
    relevance: number;
    emotion: number;
    cosine: number | null;
  };
  expected_base_score: number;
  expected_final_score: number;
}

interface GoldenScoring {
  generator: string;
  schema: number;
  description: string;
  tolerance: { score_abs: number; rule: string; note: string };
  now: string;
  clock: string;
  query: string;
  query_keywords: string[];
  target_emotion: string | null;
  query_vector: number[];
  stub_vectors: Record<string, number[]>;
  embedding_model: string;
  weights: {
    recency: number;
    importance: number;
    relevance: number;
    emotion: number;
    base: number;
    cosine: number;
    recency_half_life_days: number;
  };
  formula: Record<string, string>;
  memories: GoldenMemory[];
  expected_order_base: string[];
  expected_order_embedded: string[];
}

const fixturePath = fileURLToPath(
  new URL("./fixtures/golden-scoring.json", import.meta.url),
);
const golden = JSON.parse(readFileSync(fixturePath, "utf8")) as GoldenScoring;

const TOL = golden.tolerance.score_abs;

/**
 * The fixture's timestamps are naive (no offset), as Python's
 * `datetime.now()` / `datetime.fromisoformat` produce them. Parse `now` and
 * every memory timestamp with the SAME assumed zone so their difference is
 * exact regardless of the machine's local time. Appending `Z` to both is the
 * cheapest way to guarantee that.
 */
function parseNaive(iso: string): string {
  return /[Zz]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : `${iso}Z`;
}

const NOW = new Date(parseNaive(golden.now));

function toMemory(g: GoldenMemory): Memory {
  const m: Memory = {
    id: g.key,
    content: g.content,
    category: g.category as MemoryCategory,
    importance: g.importance,
    keywords: g.keywords,
    createdAt: parseNaive(g.created_at),
    lastAccessed: parseNaive(g.last_accessed),
    accessCount: g.access_count,
    subject: g.subject as "user" | "persona",
  };
  if (g.source_message_id !== null) m.sourceMessageId = g.source_message_id;
  // `vector: null` is the MISSING case: leave `embedding` undefined entirely.
  if (g.vector !== null) m.embedding = Float32Array.from(g.vector);
  return m;
}

const memories = golden.memories.map(toMemory);
const byKey = new Map(golden.memories.map((g, i) => [g.key, memories[i] as Memory]));

const queryVector = Float32Array.from(golden.query_vector);

const baseQuery: ScoreQuery = { keywords: golden.query_keywords };
const embeddedQuery: ScoreQuery = {
  keywords: golden.query_keywords,
  embedding: queryVector,
};
if (golden.target_emotion !== null) {
  baseQuery.targetEmotion = golden.target_emotion;
  embeddedQuery.targetEmotion = golden.target_emotion;
}

/** Report the fixture's rounded channels alongside ours when a score fails. */
function diagnose(g: GoldenMemory, actual: ReturnType<typeof scoreMemoryDetailed>): string {
  return (
    `\n  memory: ${g.key} (${g.content})` +
    `\n  fixture channels (display-rounded 6dp): ${JSON.stringify(g.channels)}` +
    `\n  limbic channels:                        ${JSON.stringify({
      recency: actual.recency,
      importance: actual.importance,
      relevance: actual.relevance,
      emotion: actual.emotion,
      cosine: actual.cosine,
    })}`
  );
}

// ── the tests ────────────────────────────────────────────────────────────────

describe("golden-scoring.json: the fixture itself", () => {
  it("is the single-scenario shape this reader expects", () => {
    expect(golden.schema).toBe(1);
    expect(Array.isArray(golden.memories)).toBe(true);
    expect(golden).not.toHaveProperty("cases");
    expect(golden.memories).toHaveLength(5);
  });

  it("pins the clock this test uses", () => {
    expect(golden.now).toBe("2026-08-26T12:00:00");
    expect(Number.isNaN(NOW.getTime())).toBe(false);
  });

  it("uses an ABSOLUTE tolerance of 1e-6", () => {
    expect(golden.tolerance.score_abs).toBe(1e-6);
    expect(golden.tolerance.rule).toBe("abs(actual - expected) <= score_abs");
  });

  it("agrees with limbic's compiled-in constants", () => {
    expect(golden.weights.recency).toBe(DEFAULT_WEIGHTS.recency);
    expect(golden.weights.importance).toBe(DEFAULT_WEIGHTS.importance);
    expect(golden.weights.relevance).toBe(DEFAULT_WEIGHTS.relevance);
    expect(golden.weights.emotion).toBe(DEFAULT_WEIGHTS.emotion);
    expect(golden.weights.base).toBe(BASE_BLEND);
    expect(golden.weights.cosine).toBe(EMBED_BLEND);
    expect(golden.weights.recency_half_life_days).toBe(RECENCY_HALF_LIFE_DAYS);
  });

  it("states the MISSING-cosine rule this port implements", () => {
    expect(golden.formula.cosine_is_none).toMatch(/never a similarity of 0/);
    expect(golden.formula.final_no_vectors).toBe("base");
  });
});

describe("golden-scoring.json: per-memory scores", () => {
  for (const g of golden.memories) {
    describe(g.key, () => {
      const memory = byKey.get(g.key) as Memory;

      it("reproduces the clock the fixture recorded", () => {
        expect(daysSince(memory.lastAccessed, NOW)).toBeCloseTo(
          g.days_since_access,
          12,
        );
      });

      it(`base score == ${g.expected_base_score} (abs 1e-6)`, () => {
        const actual = scoreMemoryDetailed(memory, baseQuery, NOW);
        expect(
          Math.abs(actual.base - g.expected_base_score),
          `base drift${diagnose(g, actual)}`,
        ).toBeLessThanOrEqual(TOL);
      });

      it("base-only scoring leaves the score unblended", () => {
        // No query embedding => cosine is MISSING => final === base, for every
        // memory, including the ones that DO carry a vector.
        const actual = scoreMemoryDetailed(memory, baseQuery, NOW);
        expect(actual.cosine).toBeNull();
        expect(actual.final).toBe(actual.base);
      });

      it(`final score == ${g.expected_final_score} (abs 1e-6)`, () => {
        const actual = scoreMemoryDetailed(memory, embeddedQuery, NOW);
        expect(
          Math.abs(actual.final - g.expected_final_score),
          `final drift${diagnose(g, actual)}`,
        ).toBeLessThanOrEqual(TOL);
      });

      it("treats a null vector as MISSING, not as cosine 0", () => {
        const actual = scoreMemoryDetailed(memory, embeddedQuery, NOW);
        if (g.vector === null) {
          expect(actual.cosine).toBeNull();
          // The whole point: base, not 0.7*base.
          expect(actual.final).toBe(actual.base);
          expect(g.expected_final_score).toBe(g.expected_base_score);
        } else {
          expect(actual.cosine).not.toBeNull();
        }
      });
    });
  }
});

describe("golden-scoring.json: ranking", () => {
  function order(query: ScoreQuery): string[] {
    return golden.memories
      .map((g) => ({
        content: g.content,
        score: scoreMemoryDetailed(byKey.get(g.key) as Memory, query, NOW).final,
      }))
      .sort((a, b) => b.score - a.score)
      .map((r) => r.content);
  }

  it("expected_order_base", () => {
    expect(order(baseQuery)).toEqual(golden.expected_order_base);
  });

  it("expected_order_embedded", () => {
    expect(order(embeddedQuery)).toEqual(golden.expected_order_embedded);
  });

  it("the cosine channel actually reorders the result", () => {
    // Not a parity assertion — a guard that the fixture is still exercising the
    // blend at all. If these two ever coincide the fixture has lost its teeth.
    expect(golden.expected_order_embedded).not.toEqual(golden.expected_order_base);
  });
});

describe("golden-scoring.json: stub vectors are self-consistent", () => {
  it("query_vector matches stub_vectors[query]", () => {
    expect(golden.stub_vectors[golden.query]).toEqual(golden.query_vector);
  });

  it("every STORED memory vector matches stub_vectors[content]", () => {
    for (const g of golden.memories) {
      if (g.vector === null) continue;
      expect(g.vector, `${g.key} vector`).toEqual(golden.stub_vectors[g.content]);
    }
  });

  it("the MISSING case is a memory whose stub exists but is not stored", () => {
    // `portuguese` HAS a stub vector in the fixture and still carries
    // `vector: null`. That is deliberate: it is the "cannot be compared" case,
    // and it is what separates a correct port from one that substitutes 0.0.
    const missing = golden.memories.filter((g) => g.vector === null);
    expect(missing.map((g) => g.key)).toEqual(["portuguese"]);
    for (const g of missing) {
      expect(golden.stub_vectors[g.content]).toBeDefined();
      expect(g.channels.cosine).toBeNull();
    }
  });
});
