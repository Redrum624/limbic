/**
 * limbic — a local-first, emotion-aware, LLM-agnostic memory engine.
 *
 * Everything exported from this file is public API from 0.1.0 on. Internal
 * modules live under `src/internal/` and are not exported.
 *
 * ```ts
 * import { createLimbic } from "limbic";
 *
 * const limbic = createLimbic();
 * await limbic.remember("User's name is Ada", { category: "personal_fact", importance: 0.9 });
 * const hits = await limbic.retrieve("what is my name", 5);
 * ```
 */

import { randomUUID } from "node:crypto";

import { calculateDecay } from "./decay.js";
import { extractFromConversation, type ChatTurn } from "./extraction.js";
import { retrieve as retrievePipeline, type RetrieveOptions, type ScoredMemory } from "./retrieve.js";
import { MemStore, type MemoryStore } from "./store.js";
import {
  DEFAULT_WEIGHTS,
  type CompleteFn,
  type Embedder,
  type ExtractedMemory,
  type Memory,
  type ScoreWeights,
} from "./types.js";

// ── Public surface ──────────────────────────────────────────────────────────

export {
  DEFAULT_WEIGHTS,
  EMBED_BLEND,
  type CompleteFn,
  type Embedder,
  type ExtractedMemory,
  type Memory,
  type MemoryCategory,
  type MemoryEmotion,
  type ScoreWeights,
} from "./types.js";

export { DEFAULT_ALL_LIMIT, MemStore, type MemoryStore } from "./store.js";
export { MISSING_SQLITE_PEER, SqliteStore } from "./stores/sqlite.js";

export {
  ACCESS_REINFORCEMENT_DAYS,
  CATEGORY_HALF_LIFE_DAYS,
  DEFAULT_HALF_LIFE_DAYS,
  IMPORTANCE_DECAY_FACTOR,
  STRENGTH_FLOOR_HIGH,
  STRENGTH_FLOOR_MEDIUM,
  calculateDecay,
  type DecayArgs,
} from "./decay.js";

export {
  DiversityError,
  F32_EPSILON,
  gistSelect,
  gistSelectFull,
  type DiameterMode,
  type DiversityErrorCode,
  type GistResult,
  type GistSelectOptions,
  type Metric,
  type Stage,
  type Utilities,
  type UtilityKind,
} from "./diversity.js";

export {
  CONVERSATION_WINDOW,
  EXTRACTION_PROMPT,
  EXTRACTION_TO_CATEGORY,
  KNOWN_EXTRACTION_TYPES,
  MIN_CONFIDENCE,
  MIN_CONVERSATION_CHARS,
  MIN_IMPORTANCE,
  buildExtractionPrompt,
  categoryFor,
  extractFromConversation,
  formatConversation,
  parseExtractionResponse,
  passesSaveGate,
  type ChatTurn,
} from "./extraction.js";

export {
  DEFAULT_LAMBDA,
  DEFAULT_POOL,
  diversify,
  retrieve,
  scorePool,
  type RetrieveOptions,
  type ScoredMemory,
} from "./retrieve.js";

export {
  EMOTION_HIGH_THRESHOLD,
  EMOTION_MEDIUM_THRESHOLD,
  RECENCY_HALF_LIFE_DAYS,
  scoreMemory,
  scoreMemoryDetailed,
  type ScoreBreakdown,
  type ScoreQuery,
} from "./internal/scoring.js";

export {
  EmbedderUnavailableError,
  isEmbedderUnavailable,
} from "./embedders/errors.js";

export {
  DEFAULT_OLLAMA_HOST,
  OllamaEmbedder,
  type OllamaEmbedderOptions,
} from "./embedders/ollama.js";
export {
  NodeLlamaCppEmbedder,
  type NodeLlamaCppEmbedderOptions,
} from "./embedders/node-llama-cpp.js";
export {
  TransformersEmbedder,
  type TransformersEmbedderOptions,
} from "./embedders/transformers.js";

// ── createLimbic ────────────────────────────────────────────────────────────

/**
 * Below this strength a memory is deleted by {@link Limbic.decayPass}.
 * The origin engine's `memory_decay.py:126` — `if current_strength < 0.05: DELETE`.
 */
export const FADE_THRESHOLD = 0.05;

/** What {@link createLimbic} accepts. Every field has a working default. */
export interface LimbicOptions {
  /** Default `new MemStore()`. */
  store?: MemoryStore;
  /** Default none — scoring runs keyword-only and nothing is embedded. */
  embedder?: Embedder;
  /** Default none — {@link Limbic.extract} then throws rather than pretending. */
  complete?: CompleteFn;
  /** Default {@link DEFAULT_WEIGHTS}. */
  weights?: ScoreWeights;
  /** Default 0.5, the origin engine's default. divsel's own default is 1.0. */
  lambda?: number;
  /** How many scored rows to diversify over. Default 50. */
  pool?: number;
}

/** The 0.1.0 engine handle. */
export interface Limbic {
  /** Store `content`, filling in the defaults and embedding it if possible. */
  remember(content: string, partial?: Partial<Memory>): Promise<Memory>;
  /**
   * Extract memories from a conversation.
   * @throws {Error} when no `complete` was configured.
   */
  extract(conversation: readonly ChatTurn[]): Promise<ExtractedMemory[]>;
  /** Score the pool and diversify it. May return fewer than `k` — see `retrieve`. */
  retrieve(query: string, k?: number, options?: RetrieveOptions): Promise<ScoredMemory[]>;
  /** The origin engine's `apply_decay_to_memories`: recompute strength, delete what has faded. */
  decayPass(now?: Date): Promise<{ decayed: number; faded: number }>;
  /** The store in use, for direct access. */
  store: MemoryStore;
}

/** Whole days between two instants, floored — the origin engine reads `timedelta.days`. */
function wholeDaysBetween(fromIso: string, now: Date): number {
  const from = Date.parse(fromIso);
  if (Number.isNaN(from)) return 0;
  return Math.floor((now.getTime() - from) / 86_400_000);
}

/**
 * Build a limbic engine.
 *
 * Nothing here touches the network unless you hand it an `embedder` that does,
 * and nothing calls an LLM unless you hand it a `complete`. Both defaults are
 * "absent", and both absences degrade rather than fail: no embedder means the
 * cosine channel is MISSING and scoring is keyword-only; no `complete` means
 * `extract` throws, because silently returning `[]` would be indistinguishable
 * from a conversation with nothing in it.
 */
export function createLimbic(options: LimbicOptions = {}): Limbic {
  const store = options.store ?? new MemStore();
  const { embedder, complete } = options;
  const weights = options.weights ?? DEFAULT_WEIGHTS;
  const lambda = options.lambda ?? 0.5;
  const pool = options.pool ?? 50;

  // Sequential ids, salted per engine instance. A bare counter would collide
  // between two engines sharing one store; deriving the id from the content
  // would collide on a repeated memory. `instance` makes the sequence unique,
  // `seq` keeps it ordered and readable.
  const instance = randomUUID().slice(0, 8);
  let seq = 0;
  const nextId = (): string => `mem_${instance}_${(++seq).toString().padStart(6, "0")}`;

  return {
    store,

    async remember(content: string, partial: Partial<Memory> = {}): Promise<Memory> {
      if (typeof content !== "string" || content.trim() === "") {
        throw new TypeError("remember: content must be a non-empty string");
      }
      const nowIso = new Date().toISOString();
      const memory: Memory = {
        id: partial.id ?? nextId(),
        content,
        category: partial.category ?? "general",
        importance: partial.importance ?? 0.5,
        keywords: partial.keywords ?? [],
        createdAt: partial.createdAt ?? nowIso,
        lastAccessed: partial.lastAccessed ?? nowIso,
        accessCount: partial.accessCount ?? 0,
        subject: partial.subject ?? "user",
      };
      if (partial.sourceMessageId !== undefined) memory.sourceMessageId = partial.sourceMessageId;
      if (partial.feeling !== undefined) memory.feeling = partial.feeling;
      if (partial.emotion !== undefined) memory.emotion = partial.emotion;
      if (partial.embeddingModel !== undefined) memory.embeddingModel = partial.embeddingModel;

      if (partial.embedding !== undefined) {
        memory.embedding = partial.embedding;
      } else if (embedder !== undefined) {
        // Never fatal: an embedder that is down costs the cosine channel, not
        // the write. The memory is stored bare and is still retrievable.
        try {
          const vectors = await embedder.embed([content]);
          const first = vectors[0];
          if (first instanceof Float32Array && first.length > 0) {
            memory.embedding = first;
            memory.embeddingModel = partial.embeddingModel ?? embedder.model;
          }
        } catch {
          /* keyword-only for this row */
        }
      }
      return store.save(memory);
    },

    async extract(conversation: readonly ChatTurn[]): Promise<ExtractedMemory[]> {
      if (complete === undefined) {
        throw new Error(
          "extract() needs a CompleteFn: createLimbic({ complete }). limbic ships no LLM.",
        );
      }
      return extractFromConversation(complete, conversation);
    },

    async retrieve(
      query: string,
      k = 5,
      overrides: RetrieveOptions = {},
    ): Promise<ScoredMemory[]> {
      const merged: RetrieveOptions = { pool, lambda, weights, ...overrides };
      if (embedder !== undefined && merged.embedder === undefined) merged.embedder = embedder;
      return retrievePipeline(store, query, k, merged);
    },

    async decayPass(now: Date = new Date()): Promise<{ decayed: number; faded: number }> {
      const total = await store.count();
      const rows = total === 0 ? [] : await store.all(total);
      let decayed = 0;
      let faded = 0;
      for (const memory of rows) {
        const strength = calculateDecay({
          originalStrength: 1,
          daysSinceCreation: wholeDaysBetween(memory.createdAt, now),
          daysSinceAccess: wholeDaysBetween(memory.lastAccessed, now),
          importance: memory.importance,
          category: memory.category,
          accessCount: memory.accessCount,
        });
        if (strength < FADE_THRESHOLD) {
          await store.delete(memory.id);
          faded += 1;
        } else {
          decayed += 1;
        }
      }
      return { decayed, faded };
    },
  };
}
