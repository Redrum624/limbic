/**
 * Store and engine lifecycle: the scalar decay scan (no embedding BLOBs in a
 * scheduled pass), fail-soft keyword decoding on rows limbic did not write,
 * and the one call that releases everything the engine holds.
 *
 * Requires the optional peer `better-sqlite3`, like `test/store.test.ts`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, it, vi } from "vitest";

import { createLimbic, type Embedder, type Memory } from "../src/index.js";
import { MemStore } from "../src/store.js";
import { DECAY_SCAN_PAGE, SqliteStore } from "../src/stores/sqlite.js";

const dir = mkdtempSync(join(tmpdir(), "limbic-lifecycle-"));
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
});

const NOW = new Date();
const DAY_MS = 86_400_000;

function daysAgo(days: number): string {
  return new Date(NOW.getTime() - days * DAY_MS).toISOString();
}

function mem(id: string, overrides: Partial<Memory> = {}): Memory {
  return {
    id,
    content: `content of ${id}`,
    category: "general",
    importance: 0.5,
    keywords: [],
    createdAt: NOW.toISOString(),
    lastAccessed: NOW.toISOString(),
    accessCount: 0,
    subject: "user",
    ...overrides,
  };
}

describe("SqliteStore.decayCandidates (L-05)", () => {
  it("yields the scalar decay fields for every row, in id order, without the embedding", async () => {
    const store = await SqliteStore.open(":memory:");
    await store.save(mem("b", { embedding: new Float32Array([1, 2, 3]), importance: 0.9 }));
    await store.save(mem("a", { category: "emotion", accessCount: 4 }));
    await store.save(mem("c"));

    const seen: string[] = [];
    for await (const candidate of store.decayCandidates()) {
      seen.push(candidate.id);
      // The whole point of the scan: the BLOB never crosses into JS.
      expect("embedding" in candidate).toBe(false);
      expect(typeof candidate.importance).toBe("number");
      expect(typeof candidate.category).toBe("string");
      expect(typeof candidate.createdAt).toBe("string");
      expect(typeof candidate.lastAccessed).toBe("string");
      expect(typeof candidate.accessCount).toBe("number");
    }
    expect(seen).toEqual(["a", "b", "c"]);
    store.close();
  });

  it("pages by id keyset, so deleting yielded rows never skips a row", async () => {
    const store = await SqliteStore.open(":memory:");
    const total = DECAY_SCAN_PAGE + 50; // force a page boundary
    for (let i = 0; i < total; i++) {
      await store.save(mem(`m${i.toString().padStart(6, "0")}`));
    }

    let seen = 0;
    for await (const candidate of store.decayCandidates()) {
      seen += 1;
      // Deleting mid-scan is exactly what decayPass does to faded rows; an
      // OFFSET-based scan would slide past its successor after each delete.
      await store.delete(candidate.id);
    }
    expect(seen).toBe(total);
    expect(await store.count()).toBe(0);
    store.close();
  });
});

describe("decayPass over a SqliteStore (L-05)", () => {
  it("streams candidates instead of materialising every row through all()", async () => {
    const store = await SqliteStore.open(":memory:");
    // 200 idle days at half-life 30 (emotion, importance 0.1 => factor 1.0):
    // 0.5 ** (200/30) ≈ 0.0098 < FADE_THRESHOLD, so this row fades…
    await store.save(
      mem("stale", {
        category: "emotion",
        importance: 0.1,
        createdAt: daysAgo(200),
        lastAccessed: daysAgo(200),
        embedding: new Float32Array(768),
      }),
    );
    // …while a fresh important row survives.
    await store.save(mem("fresh", { importance: 0.9 }));

    const allSpy = vi.spyOn(store, "all");
    const limbic = createLimbic({ store });
    await expect(limbic.decayPass(NOW)).resolves.toEqual({ decayed: 1, faded: 1 });
    expect(allSpy).not.toHaveBeenCalled();
    expect(await store.get("stale")).toBeUndefined();
    expect(await store.get("fresh")).toBeDefined();
    store.close();
  });

  it("still decays a store without the scan hook (MemStore fallback)", async () => {
    const store = new MemStore();
    await store.save(
      mem("stale", {
        category: "emotion",
        importance: 0.1,
        createdAt: daysAgo(200),
        lastAccessed: daysAgo(200),
      }),
    );
    await store.save(mem("fresh", { importance: 0.9 }));
    await expect(createLimbic({ store }).decayPass(NOW)).resolves.toEqual({
      decayed: 1,
      faded: 1,
    });
    expect(await store.count()).toBe(1);
  });
});

describe("SqliteStore keyword decoding fails soft (S-11)", () => {
  it("reads a malformed JSON-ish keywords column via the comma-split path instead of throwing", async () => {
    const file = join(dir, "keywords.db");
    const first = await SqliteStore.open(file);
    await first.save(mem("k1", { keywords: ["ok"] }));
    first.close();

    // A hand-edited or imported row: starts with "[" but is not JSON.
    const sqlite3 = (await import("better-sqlite3")).default;
    const raw = new sqlite3(file);
    raw.prepare("UPDATE memories SET keywords = ? WHERE id = ?").run("[broken, json", "k1");
    raw.close();

    const store = await SqliteStore.open(file);
    const row = await store.get("k1");
    expect(row?.keywords).toEqual(["[broken", "json"]);
    // A whole-table scan must not abort on the one bad row either.
    await expect(store.all()).resolves.toHaveLength(1);
    store.close();
  });
});

describe("SqliteStore and explicit resource management (L-10)", () => {
  const disposeSymbol = (Symbol as { dispose?: symbol }).dispose;

  it.skipIf(disposeSymbol === undefined)(
    "closes through Symbol.dispose, so `using store = ...` releases the handle",
    async () => {
      const store = await SqliteStore.open(":memory:");
      const dispose = (store as unknown as Record<symbol, (() => void) | undefined>)[
        disposeSymbol!
      ];
      expect(typeof dispose).toBe("function");
      dispose!.call(store);
      await expect(store.count()).rejects.toThrow();
    },
  );
});

describe("limbic.close (L-06)", () => {
  it("closes the store and disposes the embedder, and is idempotent", async () => {
    const close = vi.fn();
    const dispose = vi.fn();
    const store = Object.assign(new MemStore(), { close });
    const embedder: Embedder & { dispose: () => void } = {
      model: "fake",
      embed: async (texts: string[]) => texts.map(() => new Float32Array([1])),
      dispose,
    };

    const limbic = createLimbic({ store, embedder });
    await limbic.close();
    await limbic.close();
    expect(close).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("still closes the store when the embedder's dispose throws", async () => {
    const close = vi.fn();
    const store = Object.assign(new MemStore(), { close });
    const embedder: Embedder & { dispose: () => void } = {
      model: "fake",
      embed: async (texts: string[]) => texts.map(() => new Float32Array([1])),
      dispose: () => {
        throw new Error("native teardown failed");
      },
    };

    const limbic = createLimbic({ store, embedder });
    // The embedder's failure surfaces, but not at the cost of the store handle.
    await expect(limbic.close()).rejects.toThrow(/native teardown failed/);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("is a no-op on the defaults — MemStore has no close, no embedder configured", async () => {
    const limbic = createLimbic();
    await expect(limbic.close()).resolves.toBeUndefined();
    await expect(limbic.close()).resolves.toBeUndefined();
  });

  it("really closes a SqliteStore", async () => {
    const store = await SqliteStore.open(":memory:");
    const limbic = createLimbic({ store });
    await limbic.close();
    await expect(store.count()).rejects.toThrow();
  });
});
