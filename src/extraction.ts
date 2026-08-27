/**
 * LLM-driven memory extraction, ported from the origin engine's
 * `server/memory/memory_extraction.py`.
 *
 * limbic supplies no LLM. The caller injects a {@link CompleteFn}; without one,
 * `createLimbic().extract()` throws rather than pretending. Everything else —
 * the prompt, the conversation window, the JSON shape, the save gate — is
 * the origin engine's, and the divergences are listed below.
 *
 * ## Deliberate divergences from the Python reference (verified 2026-08-27)
 *
 * 1. **The placeholder is substituted literally, not through a format
 *    language.** The origin engine calls `EXTRACTION_PROMPT.format(conversation=formatted)`
 *    (`memory_extraction.py:172`) on a prompt whose JSON example contains
 *    literal braces. Measured against the origin engine's own source, that call raises
 *    `KeyError: '\n  "memories"'` — `str.format` reads the example object as a
 *    replacement field. The broad `except Exception` at `:196` swallows it and
 *    returns `[]`, so **the origin engine's LLM extraction path returns no memories today**.
 *    limbic replaces the one `{conversation}` token and leaves every other
 *    brace alone, so the same prompt text actually reaches the model.
 * 2. **`extractionType` is a plain string.** The origin engine's `ExtractionType(...)`
 *    constructor raises on an unknown value and the row is dropped
 *    (`:370`, `:386`). limbic keeps unknown types and maps them to
 *    `"general"` — validate against {@link KNOWN_EXTRACTION_TYPES}, do not
 *    reject. Nothing else in the core depends on the enum being closed.
 * 3. **Extraction never saves.** The origin engine's `extract_from_conversation` writes
 *    through to storage as a side effect when `save_immediately` is set;
 *    limbic's `extract()` returns the list and `remember()` is the only writer.
 *    The save gate travels with the data as {@link passesSaveGate}.
 */

import type { CompleteFn, ExtractedMemory, MemoryCategory } from "./types.js";

/** One conversation turn, as `extract()` receives it. */
export interface ChatTurn {
  role: string;
  content: string;
}

/**
 * The origin engine's `EXTRACTION_PROMPT` (`memory_extraction.py:59`), verbatim apart from
 * the placeholder, which is substituted literally here — see divergence 1.
 */
export const EXTRACTION_PROMPT = `Analyze this conversation and extract important information worth remembering.

CRITICAL: Distinguish WHO each piece of information is about:
- "user": Facts about the human user (their name, job, pets, preferences, family, etc.)
- "persona": Facts about the AI persona herself (her activities, her pets, her friends, her hobbies from her simulated life)

If the user says "I have a cat named Whiskers" -> subject is "user"
If the assistant says "I adopted a kitten today" -> subject is "persona"
If the user asks "how's your cat?" and the assistant replies about her cat -> subject is "persona"

Conversation:
{conversation}

Extract any of the following types of information:
- FACT: Personal facts (name, age, job, location, etc.)
- PREFERENCE: What they like/dislike, favorites
- RELATIONSHIP: People they mention (family, friends, colleagues, pets)
- EVENT: Important dates, plans, or past experiences
- GOAL: Goals, aspirations, things they want to do
- EMOTION: Their emotional state or significant feelings
- INTEREST: Hobbies, interests, things they're into

For each piece of information found, provide:
1. type: The category from above
2. subject: "user" or "persona" — who is this fact about?
3. content: A clear statement of the information (e.g., "User's name is John")
4. importance: 0.0-1.0 (how important to remember)
5. keywords: Relevant keywords for retrieval
6. supersedes: If this updates previous info (e.g., "User moved from Boston" supersedes "User lives in Boston")
7. date_expression: (EVENT type only) The date/time mentioned, as written (e.g., "May 15th", "next Tuesday", "March 3rd"). null if no date.
8. feeling: (EVENT type only) Emotional tone — "excited", "nervous", "dreading", "hopeful", "neutral", etc.

Respond ONLY with valid JSON in this format:
{
  "memories": [
    {
      "type": "FACT",
      "subject": "user",
      "content": "User's name is John",
      "importance": 0.9,
      "keywords": ["name", "john"],
      "supersedes": null,
      "date_expression": null,
      "feeling": "neutral"
    }
  ]
}

If no extractable information is found, respond with: {"memories": []}
`;

/** The origin engine keeps only the last 10 turns (`_format_conversation`, `:348`). */
export const CONVERSATION_WINDOW = 10;

/** Below this many formatted characters the origin engine skips the call entirely (`:169`). */
export const MIN_CONVERSATION_CHARS = 50;

/** `max_tokens` and `temperature` the origin engine passes (`:183-184`). */
export const EXTRACTION_MAX_TOKENS = 1024;
export const EXTRACTION_TEMPERATURE = 0.3;

/** The origin engine hands every LLM extraction the same confidence (`:379`). */
export const LLM_EXTRACTION_CONFIDENCE = 0.8;

/** The save gate: `importance >= 0.4 AND confidence >= 0.6` (`:403`). */
export const MIN_IMPORTANCE = 0.4;
export const MIN_CONFIDENCE = 0.6;

/** The origin engine's `EXTRACTION_TO_CATEGORY` (`memory_extraction.py:32`). */
export const EXTRACTION_TO_CATEGORY: Readonly<Record<string, MemoryCategory>> = {
  fact: "personal_fact",
  preference: "preference",
  relationship: "relationship",
  event: "experience",
  goal: "work",
  emotion: "emotion",
  interest: "interest",
};

/** The origin engine's closed `ExtractionType` enum, kept open here — see divergence 2. */
export const KNOWN_EXTRACTION_TYPES: ReadonlySet<string> = new Set(
  Object.keys(EXTRACTION_TO_CATEGORY),
);

/** The category an extraction type maps to; unknown types fall to `"general"`. */
export function categoryFor(extractionType: string): MemoryCategory {
  return EXTRACTION_TO_CATEGORY[extractionType.toLowerCase()] ?? "general";
}

/**
 * The origin engine's `_format_conversation` (`:345`): the last {@link CONVERSATION_WINDOW}
 * turns, empty messages dropped, `ROLE: content` per line.
 */
export function formatConversation(conversation: readonly ChatTurn[]): string {
  const lines: string[] = [];
  for (const turn of conversation.slice(-CONVERSATION_WINDOW)) {
    const content = (turn.content ?? "").trim();
    if (content === "") continue;
    const role = (turn.role ?? "user").toUpperCase();
    lines.push(`${role}: ${content}`);
  }
  return lines.join("\n");
}

/** The prompt for one conversation, or `null` when it is too short to bother. */
export function buildExtractionPrompt(conversation: readonly ChatTurn[]): string | null {
  const formatted = formatConversation(conversation);
  if (formatted.length < MIN_CONVERSATION_CHARS) return null;
  // Literal substitution, not a format language — see divergence 1.
  return EXTRACTION_PROMPT.replace("{conversation}", formatted);
}

/**
 * The origin engine's `_parse_extraction_response` (`:356`): find the outermost
 * brace-delimited span, parse it, and read `memories[]`.
 *
 * **Never throws.** A malformed response, a missing object, a non-array
 * `memories`, or a row that is not an object all yield `[]` or are skipped —
 * an extraction failure must never cost the caller their turn.
 */
export function parseExtractionResponse(response: string): ExtractedMemory[] {
  // The origin engine's `re.search(r'\{[\s\S]*\}', response)` — greedy, so it spans from the
  // first `{` to the last `}`, which is what strips a model's prose preamble
  // and any trailing ``` fence.
  const first = response.indexOf("{");
  const last = response.lastIndexOf("}");
  if (first === -1 || last <= first) return [];

  let data: unknown;
  try {
    data = JSON.parse(response.slice(first, last + 1));
  } catch {
    return [];
  }
  if (typeof data !== "object" || data === null) return [];
  const rows = (data as { memories?: unknown }).memories;
  if (!Array.isArray(rows)) return [];

  const extracted: ExtractedMemory[] = [];
  for (const row of rows) {
    if (typeof row !== "object" || row === null) continue;
    const raw = row as Record<string, unknown>;

    const importance = Number(raw["importance"] ?? 0.5);
    if (!Number.isFinite(importance)) continue;

    const subject = raw["subject"];
    const keywords = Array.isArray(raw["keywords"])
      ? (raw["keywords"] as unknown[]).filter((k): k is string => typeof k === "string")
      : [];

    const memory: ExtractedMemory = {
      content: typeof raw["content"] === "string" ? raw["content"] : "",
      extractionType: String(raw["type"] ?? "fact").toLowerCase(),
      importance,
      keywords,
      confidence: LLM_EXTRACTION_CONFIDENCE,
      subject: subject === "persona" ? "persona" : "user",
      feeling: typeof raw["feeling"] === "string" ? raw["feeling"] : "neutral",
    };
    if (typeof raw["supersedes"] === "string") memory.supersedes = raw["supersedes"];
    if (typeof raw["date_expression"] === "string") {
      memory.dateExpression = raw["date_expression"];
    }
    extracted.push(memory);
  }
  return extracted;
}

/** The origin engine's save gate (`:403`): `importance >= 0.4` **and** `confidence >= 0.6`. */
export function passesSaveGate(extracted: ExtractedMemory): boolean {
  return extracted.importance >= MIN_IMPORTANCE && extracted.confidence >= MIN_CONFIDENCE;
}

/**
 * Extract memories from a conversation through `complete`.
 *
 * Returns `[]` — never throws — when the conversation is too short, the model
 * returns something unparseable, or `complete` itself rejects. That is the origin engine's
 * rule (`:196`) and the reason it holds here too: extraction runs inside a chat
 * turn, and an unhandled rejection there costs the user their reply.
 */
export async function extractFromConversation(
  complete: CompleteFn,
  conversation: readonly ChatTurn[],
): Promise<ExtractedMemory[]> {
  const prompt = buildExtractionPrompt(conversation);
  if (prompt === null) return [];
  try {
    const response = await complete(prompt, {
      maxTokens: EXTRACTION_MAX_TOKENS,
      temperature: EXTRACTION_TEMPERATURE,
    });
    return parseExtractionResponse(response);
  } catch {
    return [];
  }
}
