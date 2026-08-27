/**
 * `SqliteStore` — durable `MemoryStore` on top of the optional peer
 * `better-sqlite3`.
 *
 * The peer is loaded through a dynamic `import()` inside `SqliteStore.open()`,
 * never at module load, so importing `limbic` costs nothing to a caller who
 * does not use SQLite and the core keeps its zero-required-dependency promise.
 *
 * Schema: `memories`, mirroring the origin engine's table
 * (`server/data/persona_datastore.py:94-105` plus the `embedding`,
 * `embedding_model` and `feeling` columns added at `:146-160`), in snake_case,
 * mapped to camelCase on the way out. Two deliberate differences:
 *
 *   - `id` is `TEXT PRIMARY KEY`, not `INTEGER PRIMARY KEY AUTOINCREMENT`:
 *     limbic memories carry caller-supplied string ids. Importing an the origin engine
 *     database means coercing its integer ids to strings.
 *   - `keywords` holds a JSON array, where the origin engine holds `",".join(keywords)`.
 *     limbic owns this database and a keyword containing a comma must survive
 *     the round trip. The reader still accepts the legacy comma-joined form so
 *     an imported the origin engine table reads correctly.
 *
 * `tier`, `original_content` and `compacted_at` — the origin engine's memory-compaction
 * columns — are deliberately absent: limbic 0.1.0 does not model tiers, and a
 * column nothing writes is a lie about the schema.
 */

import type { Memory, MemoryCategory } from "../types.js";
import type { MemoryStore } from "../store.js";
import {
  DEFAULT_ALL_LIMIT,
  asciiLower,
  assertLimit,
  assertStorable,
  cloneMemory,
  matchesQuery,
} from "../internal/store-shared.js";

/** Thrown when `better-sqlite3` is not installed. */
export const MISSING_SQLITE_PEER =
  "SqliteStore requires the optional peer better-sqlite3: npm i better-sqlite3";

/**
 * The slice of better-sqlite3 this file uses, declared structurally.
 *
 * Keeping the peer's own types out of limbic's surface means the generated
 * `.d.ts` never references `better-sqlite3`, so a consumer who has not
 * installed it still type-checks against `limbic`.
 */
interface SqliteStatement {
  run(...params: unknown[]): { changes: number };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
  iterate(...params: unknown[]): IterableIterator<unknown>;
}

interface SqliteDatabase {
  prepare(sql: string): SqliteStatement;
  exec(sql: string): unknown;
  close(): void;
}

type SqliteDatabaseCtor = new (filename: string) => SqliteDatabase;

interface MemoryRow {
  id: string;
  content: string;
  category: string;
  importance: number;
  keywords: string | null;
  source_message_id: string | null;
  created_at: string;
  last_accessed: string;
  access_count: number;
  subject: string;
  feeling: string | null;
  emotion_label: string | null;
  emotion_intensity: number | null;
  embedding: Uint8Array | null;
  embedding_model: string | null;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS memories (
  id                TEXT PRIMARY KEY,
  content           TEXT NOT NULL,
  category          TEXT NOT NULL DEFAULT 'general',
  importance        REAL NOT NULL DEFAULT 0.5,
  keywords          TEXT NOT NULL DEFAULT '[]',
  source_message_id TEXT,
  created_at        TEXT NOT NULL,
  last_accessed     TEXT NOT NULL,
  access_count      INTEGER NOT NULL DEFAULT 0,
  subject           TEXT NOT NULL DEFAULT 'user',
  feeling           TEXT,
  emotion_label     TEXT,
  emotion_intensity REAL,
  embedding         BLOB,
  embedding_model   TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_pool
  ON memories (importance DESC, last_accessed DESC, id ASC);
`;

const COLUMNS =
  "id, content, category, importance, keywords, source_message_id, created_at, " +
  "last_accessed, access_count, subject, feeling, emotion_label, emotion_intensity, " +
  "embedding, embedding_model";

const ORDER_BY = "ORDER BY importance DESC, last_accessed DESC, id ASC";

/** Node runs little-endian on every platform it supports; this asserts it rather than assuming. */
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function encodeEmbedding(vector: Float32Array): Uint8Array {
  if (!LITTLE_ENDIAN) {
    throw new Error(
      "SqliteStore writes embeddings as little-endian float32, byte-compatible with " +
        "the origin engine's struct.pack('<Nf') BLOBs; this platform is big-endian.",
    );
  }
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
}

function decodeEmbedding(blob: Uint8Array): Float32Array {
  if (!LITTLE_ENDIAN) {
    throw new Error("SqliteStore reads little-endian float32 BLOBs; this platform is big-endian.");
  }
  if (blob.byteLength % 4 !== 0) {
    throw new Error(
      `embedding BLOB length ${blob.byteLength} is not a multiple of 4 — not a float32 vector`,
    );
  }
  // Copy: the row buffer is transient, and a Float32Array view needs 4-byte alignment.
  const bytes = new Uint8Array(blob.byteLength);
  bytes.set(blob);
  return new Float32Array(bytes.buffer);
}

function encodeKeywords(keywords: string[]): string {
  return JSON.stringify(keywords);
}

function decodeKeywords(raw: string | null): string[] {
  if (raw === null) return [];
  const trimmed = raw.trim();
  if (trimmed.length === 0) return [];
  if (trimmed.startsWith("[")) {
    const parsed: unknown = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.map((k) => String(k));
  }
  // Legacy: an the origin engine table stores `",".join(keywords)`.
  return trimmed
    .split(",")
    .map((k) => k.trim())
    .filter((k) => k.length > 0);
}

function rowToMemory(row: MemoryRow): Memory {
  const memory: Memory = {
    id: row.id,
    content: row.content,
    category: row.category as MemoryCategory,
    importance: row.importance,
    keywords: decodeKeywords(row.keywords),
    createdAt: row.created_at,
    lastAccessed: row.last_accessed,
    accessCount: row.access_count,
    subject: row.subject as Memory["subject"],
  };
  if (row.source_message_id !== null) memory.sourceMessageId = row.source_message_id;
  if (row.feeling !== null) memory.feeling = row.feeling;
  if (row.emotion_label !== null && row.emotion_intensity !== null) {
    memory.emotion = { label: row.emotion_label, intensity: row.emotion_intensity };
  }
  if (row.embedding !== null) memory.embedding = decodeEmbedding(row.embedding);
  if (row.embedding_model !== null) memory.embeddingModel = row.embedding_model;
  return memory;
}

/** File-backed `MemoryStore`. Open it with {@link SqliteStore.open}. */
export class SqliteStore implements MemoryStore {
  readonly #db: SqliteDatabase;
  readonly filename: string;

  private constructor(db: SqliteDatabase, filename: string) {
    this.#db = db;
    this.filename = filename;
    db.exec(SCHEMA);
  }

  /**
   * Load `better-sqlite3` and open (or create) the database at `filename`.
   *
   * Pass `":memory:"` for a private in-process database.
   * Throws {@link MISSING_SQLITE_PEER} when the peer is not installed.
   */
  static async open(filename: string): Promise<SqliteStore> {
    let ctor: SqliteDatabaseCtor;
    try {
      const mod = (await import("better-sqlite3")) as unknown as { default?: unknown };
      ctor = (mod.default ?? mod) as SqliteDatabaseCtor;
    } catch (err) {
      const code = (err as { code?: unknown } | null)?.code;
      const missing =
        (code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND") &&
        String((err as { message?: unknown } | null)?.message ?? "").includes("better-sqlite3");
      if (missing) throw new Error(MISSING_SQLITE_PEER, { cause: err });
      throw err;
    }
    return new SqliteStore(new ctor(filename), filename);
  }

  /** Close the underlying database handle. Further calls throw. */
  close(): void {
    this.#db.close();
  }

  async save(m: Memory): Promise<Memory> {
    assertStorable(m);
    this.#db
      .prepare(
        `INSERT OR REPLACE INTO memories (${COLUMNS})
         VALUES (@id, @content, @category, @importance, @keywords, @source_message_id,
                 @created_at, @last_accessed, @access_count, @subject, @feeling,
                 @emotion_label, @emotion_intensity, @embedding, @embedding_model)`,
      )
      .run({
        id: m.id,
        content: m.content,
        category: m.category,
        importance: m.importance,
        keywords: encodeKeywords(m.keywords),
        source_message_id: m.sourceMessageId ?? null,
        created_at: m.createdAt,
        last_accessed: m.lastAccessed,
        access_count: m.accessCount,
        subject: m.subject,
        feeling: m.feeling ?? null,
        emotion_label: m.emotion?.label ?? null,
        emotion_intensity: m.emotion?.intensity ?? null,
        embedding: m.embedding === undefined ? null : encodeEmbedding(m.embedding),
        embedding_model: m.embeddingModel ?? null,
      });
    return cloneMemory(m);
  }

  async get(id: string): Promise<Memory | undefined> {
    const row = this.#db.prepare(`SELECT ${COLUMNS} FROM memories WHERE id = ?`).get(id) as
      | MemoryRow
      | undefined;
    return row === undefined ? undefined : rowToMemory(row);
  }

  async all(limit: number = DEFAULT_ALL_LIMIT): Promise<Memory[]> {
    assertLimit(limit);
    const rows = this.#db
      .prepare(`SELECT ${COLUMNS} FROM memories ${ORDER_BY} LIMIT ?`)
      .all(limit) as MemoryRow[];
    return rows.map(rowToMemory);
  }

  /**
   * Substring search.
   *
   * The match itself runs in JS through the same `matchesQuery` the in-memory
   * store uses, rather than as a SQL `LIKE`, so that the result cannot depend
   * on how `keywords` is serialized and cannot diverge from `MemStore` on a
   * needle containing JSON punctuation or a `%`. SQL supplies the ordering and
   * the rows are pulled lazily, so a satisfied `limit` stops the scan.
   */
  async search(text: string, limit: number): Promise<Memory[]> {
    assertLimit(limit);
    const needle = asciiLower(text);
    const out: Memory[] = [];
    if (limit === 0) return out;
    for (const row of this.#db.prepare(`SELECT ${COLUMNS} FROM memories ${ORDER_BY}`).iterate()) {
      const memory = rowToMemory(row as MemoryRow);
      if (!matchesQuery(memory, needle)) continue;
      out.push(memory);
      if (out.length >= limit) break;
    }
    return out;
  }

  async updateAccess(id: string): Promise<void> {
    this.#db
      .prepare("UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?")
      .run(new Date().toISOString(), id);
  }

  async delete(id: string): Promise<void> {
    this.#db.prepare("DELETE FROM memories WHERE id = ?").run(id);
  }

  async count(): Promise<number> {
    const row = this.#db.prepare("SELECT COUNT(*) AS n FROM memories").get() as { n: number };
    return row.n;
  }
}
