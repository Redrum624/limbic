/**
 * The retrieval pipeline, end to end on `MemStore`.
 *
 * The interesting assertions are not "does it return something" but the three
 * properties the origin engine's `_apply_diversity` / `_fill_preserving_spread` exist to
 * hold, each of which is a real defect if it breaks:
 *
 * 1. Diversity changes **membership**, never order — the result is still in
 *    score order, so the ordering and tie-break match the no-diversity path.
 * 2. A memory with **no embedding** is selectable but edge-free: it can take a
 *    slot GIST left open, and it never lowers the spread.
 * 3. The fill may not hand back the near-duplicates the selector rejected. If
 *    it does, diversity mode returns a set whose minimum pairwise distance is
 *    identical to plain top-`k`, which is the bug the invariant exists to stop.
 */

import { describe, expect, it } from "vitest";

import { createLimbic } from "../src/index.js";
import { diversify, retrieve, scorePool, type ScoredMemory } from "../src/retrieve.js";
import { MemStore } from "../src/store.js";
import { DEFAULT_WEIGHTS, type Embedder, type Memory } from "../src/types.js";

const NOW = new Date("2026-08-27T12:00:00.000Z");

/** A memory with everything defaulted except what a test cares about. */
function mem(id: string, partial: Partial<Memory> = {}): Memory {
  return {
    id,
    content: partial.content ?? id,
    category: partial.category ?? "general",
    importance: partial.importance ?? 0.5,
    keywords: partial.keywords ?? [],
    createdAt: partial.createdAt ?? NOW.toISOString(),
    lastAccessed: partial.lastAccessed ?? NOW.toISOString(),
    accessCount: partial.accessCount ?? 0,
    subject: partial.subject ?? "user",
    ...(partial.embedding !== undefined ? { embedding: partial.embedding } : {}),
  };
}

const vec = (...values: number[]): Float32Array => Float32Array.from(values);

/** A scored row, for the pure `diversify` tests. */
function scored(id: string, score: number, embedding?: Float32Array): ScoredMemory {
  return { memory: mem(id, embedding ? { embedding } : {}), score };
}

/** A deterministic stub embedder: one fixed vector per exact text. */
function stubEmbedder(table: Record<string, number[]>, model = "stub-1"): Embedder {
  return {
    model,
    async embed(texts: string[]): Promise<Float32Array[]> {
      return texts.map((t) => {
        const row = table[t];
        if (row === undefined) throw new Error(`stubEmbedder: no vector for ${JSON.stringify(t)}`);
        return Float32Array.from(row);
      });
    },
  };
}

describe("scorePool", () => {
  it("sorts by score descending and breaks ties the way the stores do", () => {
    // Equal scores: importance DESC, then lastAccessed DESC, then id ASC.
    const rows = [
      mem("b", { importance: 0.5 }),
      mem("a", { importance: 0.5 }),
      mem("c", { importance: 0.9 }),
    ];
    const out = scorePool(rows, { keywords: [] }, NOW, DEFAULT_WEIGHTS);
    expect(out.map((r) => r.memory.id)).toEqual(["c", "a", "b"]);
    expect(out[0]!.score).toBeGreaterThan(out[1]!.score);
    expect(out[1]!.score).toBe(out[2]!.score);
  });
});

describe("diversify — property 1: membership changes, order does not", () => {
  it("returns rows in pool order, never in selection order", () => {
    // Four rows on a line. GIST picks the extremes; the result must still read
    // low-index-first, because a position in the pool is its rank.
    const pool = [
      scored("p0", 0.9, vec(1, 0)),
      scored("p1", 0.8, vec(0.99, 0.14)),
      scored("p2", 0.7, vec(0.7, 0.71)),
      scored("p3", 0.6, vec(0, 1)),
    ];
    const out = diversify(pool, 2, 0.5);
    expect(out).toHaveLength(2);
    const ids = out.map((r) => r.memory.id);
    expect([...ids].sort()).toEqual(ids);
  });

  it("does not return the plain top-k when a near-duplicate sits second", () => {
    const pool = [
      scored("anchor", 0.90, vec(1, 0)),
      scored("duplicate", 0.89, vec(1, 0.0001)),
      scored("distinct", 0.40, vec(0, 1)),
    ];
    const byScore = pool.slice(0, 2).map((r) => r.memory.id);
    expect(byScore).toEqual(["anchor", "duplicate"]);
    const out = diversify(pool, 2, 0.5).map((r) => r.memory.id);
    expect(out).not.toEqual(byScore);
    expect(out).toContain("distinct");
  });
});

describe("diversify — property 2: a bare memory is selectable but edge-free", () => {
  it("fills a slot GIST left open with a memory that has no vector", () => {
    // Two embedded rows that are near-duplicates plus one bare row. GIST can
    // only justify one of the duplicates, so the third slot goes to the bare
    // memory — which contributes no edges and so cannot lower the spread.
    const pool = [
      scored("embedded-a", 0.9, vec(1, 0)),
      scored("embedded-b", 0.8, vec(1, 0.00001)),
      scored("bare", 0.7),
    ];
    const out = diversify(pool, 3, 4).map((r) => r.memory.id);
    expect(out).toContain("bare");
  });

  it("returns the top k by score when fewer than two rows carry a vector", () => {
    const pool = [scored("a", 0.9), scored("b", 0.8, vec(1, 0)), scored("c", 0.7)];
    expect(diversify(pool, 2, 0.5).map((r) => r.memory.id)).toEqual(["a", "b"]);
  });

  it("degrades to the top k by score rather than throwing on a ragged corpus", () => {
    const pool = [
      scored("a", 0.9, vec(1, 0)),
      scored("b", 0.8, vec(1, 0, 0)),
      scored("c", 0.7, vec(0, 1)),
    ];
    expect(() => diversify(pool, 2, 0.5)).not.toThrow();
    expect(diversify(pool, 2, 0.5).map((r) => r.memory.id)).toEqual(["a", "b"]);
  });
});

describe("diversify — property 3: the fill never lowers the spread", () => {
  it("returns fewer than k rather than re-admitting a rejected duplicate", () => {
    // An exact duplicate pair plus one orthogonal row. At a high lambda the
    // sweep's winning threshold is the full diameter, so GIST returns the two
    // rows a distance 1 apart and stops — the duplicate cannot clear `d`. The
    // fill then measures it against the floor `div(picked) == 1`, finds it at
    // distance 0 from a chosen row, and leaves the third slot empty.
    const pool = [
      scored("anchor", 0.9, vec(1, 0)),
      scored("anchor-dup", 0.89, vec(1, 0)),
      scored("orthogonal", 0.2, vec(0, 1)),
    ];
    const out = diversify(pool, 3, 64);
    expect(out.map((r) => r.memory.id)).toEqual(["anchor", "orthogonal"]);
    expect(out.length).toBeLessThan(3);
  });

  it("keeps every row when the corpus is degenerate and there is no spread to lose", () => {
    // All three coincide, so d_max is 0 and there is nothing for diversity to
    // trade against. Returning a short list here would be a regression, not a
    // signal: the honest answer is the top k by score.
    const pool = [
      scored("d0", 0.9, vec(1, 0)),
      scored("d1", 0.89, vec(1, 0)),
      scored("d2", 0.88, vec(1, 0)),
    ];
    expect(diversify(pool, 3, 64)).toHaveLength(3);
  });

  it("holds div(returned) >= div(picked) on a mixed corpus", () => {
    const pool = [
      scored("a", 0.95, vec(1, 0, 0)),
      scored("a-dup", 0.94, vec(0.9999, 0.0141, 0)),
      scored("b", 0.60, vec(0, 1, 0)),
      scored("b-dup", 0.59, vec(0.0141, 0.9999, 0)),
      scored("c", 0.20, vec(0, 0, 1)),
    ];
    const out = diversify(pool, 3, 1);
    const vectors = out
      .map((r) => r.memory.embedding)
      .filter((v): v is Float32Array => v !== undefined);
    let minimum = Number.POSITIVE_INFINITY;
    for (let i = 0; i < vectors.length; i++) {
      for (let j = i + 1; j < vectors.length; j++) {
        minimum = Math.min(minimum, cosineDistance(vectors[i]!, vectors[j]!));
      }
    }
    // The three axis vectors are mutually orthogonal (distance 1); admitting a
    // duplicate would drop the minimum to ~1e-4.
    expect(minimum).toBeGreaterThan(0.5);
  });
});

describe("retrieve — end to end on MemStore", () => {
  it("scores keyword-only when no embedder is configured", async () => {
    const store = new MemStore();
    await store.save(mem("cats", { content: "User has a cat named Whiskers", keywords: ["cat"] }));
    await store.save(mem("cars", { content: "User drives a blue car", keywords: ["car"] }));

    const out = await retrieve(store, "tell me about the cat", 1, { now: NOW });
    expect(out).toHaveLength(1);
    expect(out[0]!.memory.id).toBe("cats");
    expect(out[0]!.score).toBeGreaterThan(0);
  });

  it("caps the scored pool at `pool` before diversifying", async () => {
    const store = new MemStore();
    for (let i = 0; i < 12; i++) {
      await store.save(mem(`m${String(i).padStart(2, "0")}`, { importance: 1 - i / 100 }));
    }
    const out = await retrieve(store, "anything", 20, { now: NOW, pool: 3, diversify: false });
    expect(out).toHaveLength(3);
    expect(out.map((r) => r.memory.id)).toEqual(["m00", "m01", "m02"]);
  });

  it("rejects a non-positive k", async () => {
    await expect(retrieve(new MemStore(), "q", 0)).rejects.toThrow(TypeError);
  });

  it("degrades to keyword-only when the embedder throws", async () => {
    const store = new MemStore();
    await store.save(mem("a", { content: "alpha", keywords: ["alpha"] }));
    const broken: Embedder = {
      model: "broken",
      async embed(): Promise<Float32Array[]> {
        throw new Error("ollama is not running");
      },
    };
    const out = await retrieve(store, "alpha", 1, { now: NOW, embedder: broken });
    expect(out).toHaveLength(1);
    expect(out[0]!.memory.id).toBe("a");
  });
});

describe("createLimbic — the 0.1.0 surface", () => {
  it("remembers with defaults, then retrieves what it stored", async () => {
    const limbic = createLimbic();
    const saved = await limbic.remember("User's name is Ada");
    expect(saved.category).toBe("general");
    expect(saved.importance).toBe(0.5);
    expect(saved.subject).toBe("user");
    expect(saved.accessCount).toBe(0);
    expect(saved.id).toMatch(/^mem_[0-9a-f]{8}_\d{6}$/);

    const hits = await limbic.retrieve("what is my name", 3);
    expect(hits.map((h) => h.memory.content)).toContain("User's name is Ada");
  });

  it("issues sequential ids that do not collide between two engines on one store", async () => {
    const store = new MemStore();
    const one = createLimbic({ store });
    const two = createLimbic({ store });
    await one.remember("a");
    await one.remember("b");
    await two.remember("c");
    await two.remember("d");
    expect(await store.count()).toBe(4);
    const ids = (await store.all()).map((m) => m.id);
    expect(new Set(ids).size).toBe(4);
  });

  it("embeds on remember when an embedder is configured, and records the model", async () => {
    const limbic = createLimbic({
      embedder: stubEmbedder({ "User likes chess": [1, 0, 0] }, "stub-1"),
    });
    const saved = await limbic.remember("User likes chess");
    expect(saved.embedding).toBeInstanceOf(Float32Array);
    expect(saved.embeddingModel).toBe("stub-1");
  });

  it("stores the memory bare when the embedder is down", async () => {
    const limbic = createLimbic({
      embedder: {
        model: "down",
        async embed(): Promise<Float32Array[]> {
          throw new Error("connection refused");
        },
      },
    });
    const saved = await limbic.remember("still stored");
    expect(saved.embedding).toBeUndefined();
    expect(await limbic.store.count()).toBe(1);
  });

  it("runs a decay pass that fades what has decayed below 0.05 and keeps the rest", async () => {
    const store = new MemStore();
    const ancient = new Date(NOW.getTime() - 400 * 86_400_000).toISOString();
    // work: half-life 30d, importance 0.1 -> factor 1.3 (the >= 0.2 row does not
    // match), so effective half-life is 30/1.3 = 23.1d and 400 days is ~17 of
    // them: far below the fade threshold, and no floor applies under 0.6.
    await store.save(
      mem("faded", { category: "work", importance: 0.1, createdAt: ancient, lastAccessed: ancient }),
    );
    // relationship: half-life 365d and importance 0.9 keeps a floor of 0.3.
    await store.save(
      mem("kept", {
        category: "relationship",
        importance: 0.9,
        createdAt: ancient,
        lastAccessed: ancient,
      }),
    );
    const limbic = createLimbic({ store });
    const result = await limbic.decayPass(NOW);
    expect(result).toEqual({ decayed: 1, faded: 1 });
    expect(await store.get("faded")).toBeUndefined();
    expect(await store.get("kept")).toBeDefined();
  });

  it("reports a no-op decay pass on an empty store", async () => {
    await expect(createLimbic().decayPass(NOW)).resolves.toEqual({ decayed: 0, faded: 0 });
  });
});

/** `clamp(1 - cos, 0, 2)` — the distance the pipeline's floor test uses. */
function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const raw = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
  return raw < 0 ? 0 : raw > 2 ? 2 : raw;
}
