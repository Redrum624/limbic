/**
 * The storage seam: the `MemoryStore` interface every limbic backend implements,
 * plus `MemStore`, the zero-dependency in-memory default.
 *
 * `SqliteStore` (optional peer `better-sqlite3`) lives in `src/stores/sqlite.ts`
 * and is held to the same contract by `test/store.test.ts`.
 */

import type { Memory } from "./types.js";
import {
  DEFAULT_ALL_LIMIT,
  asciiLower,
  assertLimit,
  assertStorable,
  cloneMemory,
  comparePool,
  matchesQuery,
} from "./internal/store-shared.js";

export { DEFAULT_ALL_LIMIT } from "./internal/store-shared.js";

/**
 * Persistence contract for memories.
 *
 * Every method is async so that a backend may be remote, file-backed or
 * synchronous without changing callers. Ordering, case folding and the
 * behaviour of unknown ids are part of the contract, not of the backend:
 *
 *   - `all` and `search` return rows ordered by `importance DESC`, then
 *     `lastAccessed DESC`, then `id ASC`.
 *   - `search` is an ASCII-case-insensitive substring match over `content` and
 *     over each entry of `keywords`. The needle is literal text: `%` and `_`
 *     are not wildcards.
 *   - `updateAccess` and `delete` are no-ops for an id the store does not hold.
 *   - `save` is an upsert keyed on `id`.
 *   - Returned memories are copies. Mutating one never reaches into the store,
 *     and `Float32Array` embeddings survive the round trip unchanged.
 */
export interface MemoryStore {
  save(m: Memory): Promise<Memory>;
  get(id: string): Promise<Memory | undefined>;
  /** Default 200, matching the origin engine's pool read. */
  all(limit?: number): Promise<Memory[]>;
  /** Substring match — parity with the origin engine's `LIKE`. */
  search(text: string, limit: number): Promise<Memory[]>;
  /** Bump `accessCount` and set `lastAccessed` to now. */
  updateAccess(id: string): Promise<void>;
  delete(id: string): Promise<void>;
  count(): Promise<number>;
}

/**
 * In-memory `MemoryStore`. The default store: no dependencies, nothing on disk.
 *
 * Insertion order is not preserved as a tie-break — `comparePool` decides the
 * order completely, so a `MemStore` and a `SqliteStore` holding the same rows
 * return the same list.
 */
export class MemStore implements MemoryStore {
  readonly #rows = new Map<string, Memory>();

  async save(m: Memory): Promise<Memory> {
    assertStorable(m);
    const stored = cloneMemory(m);
    this.#rows.set(stored.id, stored);
    return cloneMemory(stored);
  }

  async get(id: string): Promise<Memory | undefined> {
    const found = this.#rows.get(id);
    return found === undefined ? undefined : cloneMemory(found);
  }

  async all(limit: number = DEFAULT_ALL_LIMIT): Promise<Memory[]> {
    assertLimit(limit);
    return [...this.#rows.values()].sort(comparePool).slice(0, limit).map(cloneMemory);
  }

  async search(text: string, limit: number): Promise<Memory[]> {
    assertLimit(limit);
    const needle = asciiLower(text);
    return [...this.#rows.values()]
      .filter((m) => matchesQuery(m, needle))
      .sort(comparePool)
      .slice(0, limit)
      .map(cloneMemory);
  }

  async updateAccess(id: string): Promise<void> {
    const found = this.#rows.get(id);
    if (found === undefined) return;
    found.accessCount += 1;
    found.lastAccessed = new Date().toISOString();
  }

  async delete(id: string): Promise<void> {
    this.#rows.delete(id);
  }

  async count(): Promise<number> {
    return this.#rows.size;
  }
}
