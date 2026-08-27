/**
 * Scoring — a port of the origin engine's `server/memory/retrieval_service.py`.
 *
 * Verified against the origin engine at commit `ff9c407c` (2026-08-27):
 *   `_calculate_score`           retrieval_service.py:553
 *   `_calculate_emotion_score`   retrieval_service.py:610
 *   `_emotions_related`          retrieval_service.py:672
 *   `_calculate_recency_score`   retrieval_service.py:884
 *   `_calculate_relevance_score` retrieval_service.py:898
 *   `_extract_keywords`          retrieval_service.py:926
 *   weights / thresholds         retrieval_service.py:110-125
 *
 * The pinned formula, also stated in `test/fixtures/golden-scoring.json`:
 *
 *   recency = 0.5 ** (daysSinceLastAccess / 7)
 *   base    = clamp(0.25*recency + 0.35*importance + 0.25*relevance + 0.15*emotion, 0, 1)
 *   final   = base                                           when cosine is MISSING
 *   final   = clamp(0.70*base + 0.30*max(0, cosine), 0, 1)   when a vector exists
 *
 * `cosine == null` is MISSING, never 0.0.
 *
 * Scoring never reads the wall clock: `now` is always an explicit argument, so
 * the golden fixture can pin it to 2026-08-26T12:00:00.
 */

import {
  DEFAULT_WEIGHTS,
  EMBED_BLEND,
  type Memory,
  type MemoryEmotion,
  type ScoreWeights,
} from "../types.js";
import { cosine } from "./vec.js";

/** The origin engine's `RECENCY_HALF_LIFE_DAYS = 7` (retrieval_service.py:121). */
export const RECENCY_HALF_LIFE_DAYS = 7;

/** The origin engine's `EMOTION_HIGH_THRESHOLD` (retrieval_service.py:124). */
export const EMOTION_HIGH_THRESHOLD = 0.7;
/** The origin engine's `EMOTION_MEDIUM_THRESHOLD` (retrieval_service.py:125). */
export const EMOTION_MEDIUM_THRESHOLD = 0.4;

/** The origin engine's `BASE_WEIGHT` — the complement of `EMBED_BLEND`. */
export const BASE_BLEND = 1 - EMBED_BLEND;

export interface ScoreQuery {
  /** Already-extracted query keywords — see `extractKeywords`. */
  keywords: string[];
  embedding?: Float32Array;
  targetEmotion?: string;
}

export interface ScoreBreakdown {
  recency: number;
  importance: number;
  relevance: number;
  emotion: number;
  /** `null` means MISSING — "cannot be compared" — never a similarity of 0. */
  cosine: number | null;
  base: number;
  final: number;
}

/**
 * The origin engine's `_extract_keywords` stop-word set, verbatim (retrieval_service.py:932).
 */
const STOP_WORDS: ReadonlySet<string> = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "shall",
  "can", "to", "of", "in", "for", "on", "with", "at", "by",
  "from", "as", "into", "through", "during", "before", "after",
  "above", "below", "between", "under", "again", "further",
  "then", "once", "here", "there", "when", "where", "why",
  "how", "all", "each", "few", "more", "most", "other", "some",
  "such", "no", "nor", "not", "only", "own", "same", "so",
  "than", "too", "very", "just", "and", "but", "if", "or",
  "because", "until", "while", "this", "that", "these", "those",
  "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
  "you", "your", "yours", "yourself", "yourselves", "he", "him",
  "his", "himself", "she", "her", "hers", "herself", "it", "its",
  "itself", "they", "them", "their", "theirs", "themselves",
  "what", "which", "who", "whom", "about", "am", "also",
]);

/**
 * The origin engine's `_extract_keywords`: findall of ASCII letter runs over `text.lower()`,
 * then drop stop words and words of length <= 2.
 */
export function extractKeywords(text: string): Set<string> {
  const out = new Set<string>();
  const words = text.toLowerCase().match(/[a-z]+/g);
  if (!words) return out;
  for (const word of words) {
    if (word.length > 2 && !STOP_WORDS.has(word)) out.add(word);
  }
  return out;
}

/**
 * The origin engine's `_emotions_related` — the hard-coded seven-family feelings-wheel table
 * (retrieval_service.py:678). No dependency on any sibling library.
 */
const EMOTION_FAMILIES: ReadonlyMap<string, ReadonlySet<string>> = new Map([
  ["happy", new Set(["joyful", "content", "proud", "playful", "excited", "optimistic", "peaceful"])],
  ["sad", new Set(["lonely", "vulnerable", "guilty", "depressed", "hurt", "grief", "abandoned"])],
  ["angry", new Set(["frustrated", "bitter", "mad", "aggressive", "hostile", "annoyed", "resentful"])],
  ["fearful", new Set(["scared", "anxious", "insecure", "nervous", "worried", "overwhelmed"])],
  ["surprised", new Set(["startled", "confused", "amazed", "shocked", "astonished"])],
  ["disgusted", new Set(["disappointed", "disapproving", "awful", "repelled"])],
  ["love", new Set(["intimate", "passionate", "aroused", "affectionate", "caring", "tender"])],
]);

export function emotionsRelated(a: string, b: string): boolean {
  const e1 = a.toLowerCase();
  const e2 = b.toLowerCase();
  for (const [family, members] of EMOTION_FAMILIES) {
    if (e1 === family || members.has(e1)) {
      if (e2 === family || members.has(e2)) return true;
    }
  }
  return false;
}

/** The origin engine's `_calculate_recency_score`: 0.5 to the power (daysSinceAccess / 7). */
export function recencyScore(daysSinceAccess: number): number {
  return Math.pow(0.5, daysSinceAccess / RECENCY_HALF_LIFE_DAYS);
}

/**
 * The origin engine's `_calculate_relevance_score`: Jaccard between the query keyword set and
 * `memory.keywords | extractKeywords(memory.content)`, both lower-cased. `0`
 * when either side is empty.
 */
export function relevanceScore(memory: Memory, queryKeywords: Iterable<string>): number {
  const q = new Set<string>();
  for (const k of queryKeywords) if (k) q.add(k.toLowerCase());
  if (q.size === 0) return 0;

  const all = extractKeywords(memory.content);
  for (const k of memory.keywords) if (k) all.add(k.toLowerCase());
  if (all.size === 0) return 0;

  let intersection = 0;
  for (const k of q) if (all.has(k)) intersection++;
  const union = q.size + all.size - intersection;
  if (union === 0) return 0;
  return intersection / union;
}

/**
 * The origin engine's `_calculate_emotion_score`.
 *
 * Deviation, deliberate and documented: the origin engine reaches for the source
 * conversation's emotion via `_get_emotion_for_message(source_message_id)`.
 * limbic has no conversations table, so the caller populates `memory.emotion`
 * instead and this reads it directly. Everything downstream — the intensity
 * ladder, the target match, the family match, the `min(1, ...)` cap — is the
 * Python verbatim.
 *
 * (In the origin engine today those boosts are dead code: the cache's only writer,
 * `cache_emotion_from_conversation`, has no callers, and the DB lookup is an
 * unimplemented TODO returning None. So the origin engine's production emotion score is
 * `0.3` for `category == "emotion"` and `0` otherwise — which is exactly what
 * `golden-scoring.json` exercises, since every fixture memory has a null
 * `source_message_id`. The full ladder is still the contract to port.)
 */
export function emotionScore(memory: Memory, targetEmotion?: string): number {
  let score = 0;

  if (memory.category === "emotion") score += 0.3;

  // Read through an explicit shape so this keeps compiling if `Memory` is
  // regenerated from the plan's verbatim Task 1 block, which predates the
  // `emotion` field Task 3 adds. See types.ts / MemoryEmotion.
  const data: MemoryEmotion | undefined = memory.emotion;
  if (data) {
    const { label, intensity } = data;
    if (intensity >= EMOTION_HIGH_THRESHOLD) score += 0.5;
    else if (intensity >= EMOTION_MEDIUM_THRESHOLD) score += 0.3;
    else score += intensity * 0.3;

    if (targetEmotion && label) {
      if (label.toLowerCase() === targetEmotion.toLowerCase()) score += 0.4;
      else if (emotionsRelated(label, targetEmotion)) score += 0.2;
    }
  }

  return Math.min(1, score);
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** Fractional days between an ISO-8601 timestamp and `now`. */
export function daysSince(iso: string, now: Date): number {
  return (now.getTime() - Date.parse(iso)) / 86_400_000;
}

/**
 * The four channels, the base score and the blended final in one object.
 * `scoreMemory` is the plan's public one-number signature over this.
 */
export function scoreMemoryDetailed(
  memory: Memory,
  query: ScoreQuery,
  now: Date,
  weights: ScoreWeights = DEFAULT_WEIGHTS,
): ScoreBreakdown {
  const recency = recencyScore(daysSince(memory.lastAccessed, now));
  const importance = memory.importance;
  const relevance = relevanceScore(memory, query.keywords);
  const emotion = emotionScore(memory, query.targetEmotion);

  const base = clamp01(
    weights.recency * recency +
      weights.importance * importance +
      weights.relevance * relevance +
      weights.emotion * emotion,
  );

  // MISSING on either side is MISSING for the pair. `cosine` itself also
  // returns null for a mismatched dimension or a zero norm.
  const similarity =
    query.embedding == null || memory.embedding == null
      ? null
      : cosine(query.embedding, memory.embedding);

  const final =
    similarity === null
      ? base
      : clamp01(BASE_BLEND * base + EMBED_BLEND * Math.max(0, similarity));

  return { recency, importance, relevance, emotion, cosine: similarity, base, final };
}

/**
 * `scoreMemory(m, q, now)` — the plan's pinned signature. Returns the final
 * score in [0, 1]: the base when there is no comparable vector, the blend when
 * there is.
 */
export function scoreMemory(
  memory: Memory,
  query: ScoreQuery,
  now: Date,
  weights: ScoreWeights = DEFAULT_WEIGHTS,
): number {
  return scoreMemoryDetailed(memory, query, now, weights).final;
}
