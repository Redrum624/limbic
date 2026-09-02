/**
 * `SqliteStore.open` must not leak the native handle when the schema DDL
 * throws (SQLITE_NOTADB on a garbage file, a read-only volume, a locked
 * database): each failed `open()` retry would otherwise strand one file
 * descriptor. Mocked peer, own file: whether an open fd blocks unlink is
 * platform-shaped, but "close before rethrow" is not — and `vi.mock` is
 * file-wide, so the real-peer tests live elsewhere.
 */

import { describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({ opened: 0, closed: 0 }));

vi.mock("better-sqlite3", () => {
  class FakeDatabase {
    constructor(_filename: string) {
      state.opened += 1;
    }
    exec(): never {
      throw Object.assign(new Error("file is not a database"), { code: "SQLITE_NOTADB" });
    }
    prepare(): never {
      throw new Error("prepare must not be reached before the schema exists");
    }
    close(): void {
      state.closed += 1;
    }
  }
  return { default: FakeDatabase };
});

import { SqliteStore } from "../src/stores/sqlite.js";

describe("SqliteStore.open when the schema DDL fails (L-04)", () => {
  it("closes the just-opened handle before rethrowing", async () => {
    await expect(SqliteStore.open("garbage.db")).rejects.toThrow(/not a database/);
    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);
  });
});
