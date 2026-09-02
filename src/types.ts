/**
 * limbic core types — the 0.1.0 public type surface.
 *
 * Ported from the origin engine, a production Python memory engine:
 *   - `models.py`            (`Memory`, `Conversation`)
 *   - `memory_extraction.py` (`ExtractedMemory`, `feeling` default "neutral")
 *   - `retrieval_service.py` (scoring weights, half-life, blend)
 *
 * Deliberate deviations from the Python reference are documented in the
 * README's parity section.
 */

export type MemoryCategory =
  | "personal_fact"
  | "preference"
  | "relationship"
  | "experience"
  | "emotion"
  | "interest"
  | "work"
  | "health"
  | "general";

/**
 * The emotional reading limbic scores on.
 *
 * The origin engine reads the *source conversation's* detected emotion through
 * `_get_emotion_for_message(source_message_id)`, backed by an in-process cache
 * plus an unimplemented conversations-table lookup. limbic does not store
 * conversations, so the caller supplies the pair directly at `remember()` time
 * and it rides on the memory. `Memory.feeling` is NOT this: `feeling` is the
 * extractor's free-text tone label, whereas this is the scored `(label,
 * intensity)` pair that drives the intensity and target-emotion boosts.
 */
export interface MemoryEmotion {
  label: string;
  /** 0..1 */
  intensity: number;
}

export interface Memory {
  id: string;
  content: string;
  category: MemoryCategory; // default "general"
  importance: number; // 0..1, default 0.5
  keywords: string[];
  sourceMessageId?: string;
  createdAt: string; // ISO-8601, same convention as the origin engine
  lastAccessed: string;
  accessCount: number; // default 0
  subject: "user" | "persona"; // default "user"
  feeling?: string; // emotional tone at extraction time
  emotion?: MemoryEmotion; // scored emotion of the source turn — see MemoryEmotion
  embedding?: Float32Array; // little-endian f32 — the layout the origin engine stores in its BLOB
  embeddingModel?: string;
}

export interface ExtractedMemory {
  content: string;
  extractionType: string;
  importance: number;
  keywords: string[];
  confidence: number;
  supersedes?: string;
  subject: "user" | "persona";
  dateExpression?: string;
  feeling: string; // default "neutral"
}

export type CompleteFn = (
  prompt: string,
  opts?: { maxTokens?: number; temperature?: number },
) => Promise<string>;

export interface Embedder {
  readonly model: string;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export interface ScoreWeights {
  recency: number;
  importance: number;
  relevance: number;
  emotion: number;
}

/** The origin engine's `retrieval_service.py` RECENCY/IMPORTANCE/RELEVANCE/EMOTION_WEIGHT. */
export const DEFAULT_WEIGHTS: ScoreWeights = {
  recency: 0.25,
  importance: 0.35,
  relevance: 0.25,
  emotion: 0.15,
};

/** `final = 0.7*base + 0.3*max(0, cosine)` — the origin engine's `COSINE_WEIGHT`. */
export const EMBED_BLEND = 0.3;
