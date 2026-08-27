/**
 * limbic — local-first, emotion-aware, LLM-agnostic agent memory.
 *
 * Everything re-exported here is public API from 0.1.0 on; internal modules
 * live under `src/internal/` and are not part of the contract.
 *
 * Tasks 3-7 of the plan add the scoring, decay, embedder, diversity and
 * pipeline exports; this file currently carries the type surface (Task 1) and
 * the storage seam (Task 2).
 */

export type {
  CompleteFn,
  Embedder,
  ExtractedMemory,
  Memory,
  MemoryCategory,
  MemoryEmotion,
  ScoreWeights,
} from "./types.js";
export { DEFAULT_WEIGHTS, EMBED_BLEND } from "./types.js";

export { DEFAULT_ALL_LIMIT, MemStore, type MemoryStore } from "./store.js";
export { MISSING_SQLITE_PEER, SqliteStore } from "./stores/sqlite.js";
