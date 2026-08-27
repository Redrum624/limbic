/**
 * The `MemoryStore` contract, run against every store limbic ships.
 *
 * `describeStoreContract` is the whole point of this file: a backend is only a
 * limbic store if it is indistinguishable from the others through the
 * interface — same ordering, same case folding, same behaviour on unknown ids,
 * same round trip for a `Float32Array` embedding. Adding a backend (the
 * `SqliteCipherStore` the plan floats for wellness_companion, say) means adding
 * one call here, not a second pile of tests.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, beforeEach, describe, expect, it } from "vitest";

import { MemStore, type MemoryStore } from "../src/store.js";
import { MISSING_SQLITE_PEER, SqliteStore } from "../src/stores/sqlite.js";
import type { Memory } from "../src/types.js";

const TMP_ROOT = mkdtempSync(join(tmpdir(), "limbic-store-"));
afterAll(() => {
  rmSync(TMP_ROOT, { recursive: true, force: true });
});

let tmpCounter = 0;
function tmpDbPath(): string {
  tmpCounter += 1;
  return join(TMP_ROOT, `store-${String(tmpCounter)}.db`);
}

function mem(id: string, over: Partial<Memory> = {}): Memory {
  return {
    id,
    content: `memory ${id}`,
    category: "general",
    importance: 0.5,
    keywords: [],
    createdAt: "2026-08-20T10:00:00.000Z",
    lastAccessed: "2026-08-20T10:00:00.000Z",
    accessCount: 0,
    subject: "user",
    ...over,
  };
}

interface OpenStore {
  store: MemoryStore;
  dispose: () => Promise<void>;
}

function describeStoreContract(name: string, makeStore: () => Promise<OpenStore>): void {
  describe(`MemoryStore contract: ${name}`, () => {
    let store!: MemoryStore;
    let dispose!: () => Promise<void>;

    beforeEach(async () => {
      ({ store, dispose } = await makeStore());
    });

    afterEach(async () => {
      await dispose();
    });

    it("round-trips every field, embedding included", async () => {
      const full: Memory = {
        id: "m-full",
        content: "Marc adopted a cat named Nyx",
        category: "relationship",
        importance: 0.875,
        keywords: ["cat", "nyx", "hiking, camping"],
        sourceMessageId: "msg-7",
        createdAt: "2026-08-19T08:30:00.000Z",
        lastAccessed: "2026-08-26T11:00:00.000Z",
        accessCount: 3,
        subject: "user",
        feeling: "joy",
        emotion: { label: "joy", intensity: 0.75 },
        embedding: new Float32Array([1.5, -2.25, 0.125]),
        embeddingModel: "nomic-embed-text",
      };

      const saved = await store.save(full);
      expect(saved).toEqual(full);

      const loaded = await store.get("m-full");
      expect(loaded).toEqual(full);
      expect(loaded?.embedding).toBeInstanceOf(Float32Array);
      expect([...(loaded?.embedding ?? [])]).toEqual([1.5, -2.25, 0.125]);
      expect(loaded?.keywords).toEqual(["cat", "nyx", "hiking, camping"]);
    });

    it("keeps optional fields absent when they were never set", async () => {
      await store.save(mem("m-bare"));
      const loaded = await store.get("m-bare");
      expect(loaded?.embedding).toBeUndefined();
      expect(loaded?.embeddingModel).toBeUndefined();
      expect(loaded?.feeling).toBeUndefined();
      expect(loaded?.emotion).toBeUndefined();
      expect(loaded?.sourceMessageId).toBeUndefined();
      expect(loaded?.keywords).toEqual([]);
    });

    it("hands back copies — the caller cannot reach into the store", async () => {
      const original = mem("m-iso", {
        keywords: ["alpha"],
        embedding: new Float32Array([1, 2]),
      });
      const saved = await store.save(original);

      original.content = "mutated after save";
      original.keywords.push("beta");
      original.embedding?.set([9], 0);
      saved.content = "mutated through the return value";

      const loaded = await store.get("m-iso");
      expect(loaded?.content).toBe("memory m-iso");
      expect(loaded?.keywords).toEqual(["alpha"]);
      expect([...(loaded?.embedding ?? [])]).toEqual([1, 2]);

      loaded!.keywords.push("gamma");
      const again = await store.get("m-iso");
      expect(again?.keywords).toEqual(["alpha"]);
    });

    it("save is an upsert keyed on id", async () => {
      await store.save(mem("m-1", { content: "first" }));
      await store.save(mem("m-1", { content: "second", importance: 0.9 }));
      expect(await store.count()).toBe(1);
      const loaded = await store.get("m-1");
      expect(loaded?.content).toBe("second");
      expect(loaded?.importance).toBe(0.9);
    });

    it("get returns undefined for an unknown id", async () => {
      expect(await store.get("nope")).toBeUndefined();
    });

    it("rejects a memory with no id", async () => {
      await expect(store.save(mem(""))).rejects.toThrow(TypeError);
      expect(await store.count()).toBe(0);
    });

    it("all() orders by importance DESC, lastAccessed DESC, id ASC", async () => {
      await store.save(mem("a", { importance: 0.9, lastAccessed: "2026-08-20T10:00:00.000Z" }));
      await store.save(mem("b", { importance: 0.9, lastAccessed: "2026-08-21T10:00:00.000Z" }));
      await store.save(mem("c", { importance: 0.9, lastAccessed: "2026-08-21T10:00:00.000Z" }));
      await store.save(mem("d", { importance: 0.2, lastAccessed: "2026-08-22T10:00:00.000Z" }));

      expect((await store.all()).map((m) => m.id)).toEqual(["b", "c", "a", "d"]);
      expect((await store.all(2)).map((m) => m.id)).toEqual(["b", "c"]);
      expect(await store.all(0)).toEqual([]);
    });

    it("all() rejects a limit that is not a non-negative integer", async () => {
      await expect(store.all(-1)).rejects.toThrow(RangeError);
      await expect(store.all(1.5)).rejects.toThrow(RangeError);
    });

    it("search matches content and keywords, ASCII-case-insensitively", async () => {
      await store.save(mem("s-1", { content: "Nyx the Cat sleeps on the desk" }));
      await store.save(mem("s-2", { content: "unrelated", keywords: ["Astronomy", "Stars"] }));
      await store.save(mem("s-3", { content: "unrelated", keywords: ["hiking, camping"] }));

      expect((await store.search("nyx", 10)).map((m) => m.id)).toEqual(["s-1"]);
      expect((await store.search("CAT", 10)).map((m) => m.id)).toEqual(["s-1"]);
      expect((await store.search("astro", 10)).map((m) => m.id)).toEqual(["s-2"]);
      expect((await store.search("hiking, camp", 10)).map((m) => m.id)).toEqual(["s-3"]);
      expect(await store.search("no such thing", 10)).toEqual([]);
    });

    it("search treats the needle as literal text, not a LIKE pattern", async () => {
      await store.save(mem("p-1", { content: "battery at 50% today" }));
      await store.save(mem("p-2", { content: "no percent sign here" }));

      expect((await store.search("50%", 10)).map((m) => m.id)).toEqual(["p-1"]);
      expect((await store.search("%", 10)).map((m) => m.id)).toEqual(["p-1"]);
      expect((await store.search("_", 10)).map((m) => m.id)).toEqual([]);
    });

    it("search honours the pool ordering and the limit, and an empty needle matches all", async () => {
      await store.save(mem("q-low", { content: "shared token", importance: 0.1 }));
      await store.save(mem("q-high", { content: "shared token", importance: 0.9 }));
      await store.save(mem("q-mid", { content: "shared token", importance: 0.5 }));

      expect((await store.search("shared", 10)).map((m) => m.id)).toEqual([
        "q-high",
        "q-mid",
        "q-low",
      ]);
      expect((await store.search("shared", 2)).map((m) => m.id)).toEqual(["q-high", "q-mid"]);
      expect(await store.search("shared", 0)).toEqual([]);
      expect((await store.search("", 10)).map((m) => m.id)).toEqual(["q-high", "q-mid", "q-low"]);
      await expect(store.search("shared", -1)).rejects.toThrow(RangeError);
    });

    it("updateAccess increments the count and bumps lastAccessed", async () => {
      await store.save(mem("u-1", { accessCount: 2, lastAccessed: "2020-01-01T00:00:00.000Z" }));
      await store.updateAccess("u-1");

      const loaded = await store.get("u-1");
      expect(loaded?.accessCount).toBe(3);
      expect(loaded?.lastAccessed).not.toBe("2020-01-01T00:00:00.000Z");
      expect(Date.parse(loaded?.lastAccessed ?? "")).toBeGreaterThan(
        Date.parse("2020-01-01T00:00:00.000Z"),
      );
    });

    it("updateAccess and delete are no-ops for an unknown id", async () => {
      await store.save(mem("keep"));
      await expect(store.updateAccess("ghost")).resolves.toBeUndefined();
      await expect(store.delete("ghost")).resolves.toBeUndefined();
      expect(await store.count()).toBe(1);
    });

    it("delete removes the row and count tracks it", async () => {
      expect(await store.count()).toBe(0);
      await store.save(mem("d-1"));
      await store.save(mem("d-2"));
      expect(await store.count()).toBe(2);

      await store.delete("d-1");
      expect(await store.count()).toBe(1);
      expect(await store.get("d-1")).toBeUndefined();
      expect((await store.all()).map((m) => m.id)).toEqual(["d-2"]);
    });
  });
}

describeStoreContract("MemStore", async () => {
  const store = new MemStore();
  return { store, dispose: async () => {} };
});

describeStoreContract("SqliteStore", async () => {
  const store = await SqliteStore.open(tmpDbPath());
  return {
    store,
    dispose: async () => {
      store.close();
    },
  };
});

describe("SqliteStore specifics", () => {
  it("persists across close and reopen", async () => {
    const path = tmpDbPath();
    const first = await SqliteStore.open(path);
    await first.save(
      mem("persisted", { keywords: ["a,b", "c"], embedding: new Float32Array([0.5, 0.25]) }),
    );
    first.close();

    const second = await SqliteStore.open(path);
    const loaded = await second.get("persisted");
    expect(loaded?.keywords).toEqual(["a,b", "c"]);
    expect([...(loaded?.embedding ?? [])]).toEqual([0.5, 0.25]);
    second.close();
  });

  it("stores the embedding as a little-endian float32 BLOB, the layout the origin engine packs", async () => {
    const path = tmpDbPath();
    const store = await SqliteStore.open(path);
    await store.save(mem("blob", { embedding: new Float32Array([1.5, -2.25]) }));
    store.close();

    const mod = (await import("better-sqlite3")) as unknown as { default: new (f: string) => any };
    const db = new mod.default(path);
    const row = db.prepare("SELECT embedding FROM memories WHERE id = ?").get("blob") as {
      embedding: Buffer;
    };
    // 1.5 -> 0x3FC00000, -2.25 -> 0xC0100000, little-endian.
    expect([...row.embedding]).toEqual([0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0]);
    db.close();
  });

  it("reads the origin engine's legacy comma-joined keywords column", async () => {
    const path = tmpDbPath();
    const store = await SqliteStore.open(path);
    await store.save(mem("legacy"));
    store.close();

    const mod = (await import("better-sqlite3")) as unknown as { default: new (f: string) => any };
    const db = new mod.default(path);
    db.prepare("UPDATE memories SET keywords = ? WHERE id = ?").run("cat, nyx ,desk", "legacy");
    db.close();

    const reopened = await SqliteStore.open(path);
    expect((await reopened.get("legacy"))?.keywords).toEqual(["cat", "nyx", "desk"]);
    expect((await reopened.search("nyx", 10)).map((m) => m.id)).toEqual(["legacy"]);
    reopened.close();
  });

  it("names the peer to install when better-sqlite3 is missing", () => {
    expect(MISSING_SQLITE_PEER).toBe(
      "SqliteStore requires the optional peer better-sqlite3: npm i better-sqlite3",
    );
  });
});
