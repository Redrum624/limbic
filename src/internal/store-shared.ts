/**
 * Shared store semantics.
 *
 * `MemStore` and `SqliteStore` must be indistinguishable through the
 * `MemoryStore` interface — that is what `describeStoreContract` in
 * `test/store.test.ts` asserts — so every rule that could drift between a Map
 * and a SQL engine lives here, in one implementation both stores call:
 *
 *   - pool ordering: `importance DESC, last_accessed DESC, id ASC`, matching
 *     the origin engine's `ORDER BY importance DESC, last_accessed DESC`
 *     (`server/memory/memory_service.py:337`, `:408`, `:421`, `:435`). The
 *     trailing `id ASC` is limbic's addition: the origin engine's two-key sort leaves ties
 *     in whatever order the engine returns them, which is not a contract two
 *     different stores can both satisfy.
 *   - search: substring over `content` and over each keyword, case-insensitive
 *     the way SQLite's `lower()` is case-insensitive, i.e. ASCII-only. The origin engine
 *     searches with `content LIKE '%q%' OR keywords LIKE '%q%'`
 *     (`memory_service.py:484`-ish region); limbic matches per keyword instead
 *     of against the serialized column so the result cannot depend on how the
 *     keyword list happens to be encoded on disk.
 *
 * Internal module: nothing here is exported from `src/index.ts`.
 */

import type { Memory } from "../types.js";

/** Default page size of a pool read. The origin engine's retrieval pool reads the same shape. */
export const DEFAULT_ALL_LIMIT = 200;

/**
 * Lower-case exactly the 26 ASCII letters.
 *
 * SQLite's built-in `lower()` is ASCII-only; `String.prototype.toLowerCase()`
 * is Unicode-aware. Using the JS version in `MemStore` and the SQL version in
 * `SqliteStore` would make the two stores disagree on any non-ASCII query, so
 * both stores fold case through this function and neither uses `lower()`.
 */
export function asciiLower(text: string): string {
  let out = "";
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    out += code >= 65 && code <= 90 ? String.fromCharCode(code + 32) : text[i];
  }
  return out;
}

/**
 * Byte-order-ish text comparison, mirroring SQLite's BINARY collation.
 *
 * JS compares UTF-16 code units and SQLite compares UTF-8 bytes; the two orders
 * differ only when astral-plane characters are compared against U+E000..U+FFFF.
 * ISO-8601 timestamps and ordinary ids are unaffected.
 */
function compareText(a: string, b: string): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/** Pool ordering: importance DESC, lastAccessed DESC, id ASC. */
export function comparePool(a: Memory, b: Memory): number {
  if (a.importance !== b.importance) return b.importance - a.importance;
  const byAccess = compareText(b.lastAccessed, a.lastAccessed);
  if (byAccess !== 0) return byAccess;
  return compareText(a.id, b.id);
}

/**
 * Does `memory` match an already ASCII-lower-cased `needle`?
 *
 * An empty needle matches everything, which is what `LIKE '%%'` does in the origin engine.
 */
export function matchesQuery(memory: Memory, needle: string): boolean {
  if (needle === "") return true;
  if (asciiLower(memory.content).includes(needle)) return true;
  for (const keyword of memory.keywords) {
    if (asciiLower(keyword).includes(needle)) return true;
  }
  return false;
}

/** Deep enough copy that a caller cannot reach into the store through it. */
export function cloneMemory(memory: Memory): Memory {
  const copy: Memory = { ...memory, keywords: [...memory.keywords] };
  if (memory.embedding !== undefined) copy.embedding = new Float32Array(memory.embedding);
  if (memory.emotion !== undefined) copy.emotion = { ...memory.emotion };
  return copy;
}

/** Reject the arguments that would otherwise mean different things to each store. */
export function assertLimit(limit: number, label = "limit"): void {
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError(`${label} must be a non-negative integer, received ${String(limit)}`);
  }
}

/** A store row must be addressable, or `get`/`delete`/`updateAccess` have nothing to key on. */
export function assertStorable(memory: Memory): void {
  if (typeof memory.id !== "string" || memory.id.length === 0) {
    throw new TypeError("Memory.id must be a non-empty string");
  }
  if (!Number.isFinite(memory.importance)) {
    throw new TypeError(`Memory.importance must be a finite number, received ${String(memory.importance)}`);
  }
  if (!Array.isArray(memory.keywords)) {
    throw new TypeError("Memory.keywords must be an array of strings");
  }
}
