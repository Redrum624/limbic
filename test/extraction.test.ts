/**
 * Extraction — the prompt, the parse, and the save gate.
 *
 * The gate is the load-bearing part: the origin engine drops anything below
 * `importance 0.4` **or** `confidence 0.6` (`memory_extraction.py:403`), and a
 * port that made either bound non-strict, or used `or` where the origin engine uses two
 * independent `<` checks, would store rows the origin engine discards.
 */

import { describe, expect, it, vi } from "vitest";

import {
  CONVERSATION_WINDOW,
  EXTRACTION_MAX_TOKENS,
  EXTRACTION_PROMPT,
  EXTRACTION_TEMPERATURE,
  MIN_CONFIDENCE,
  MIN_IMPORTANCE,
  buildExtractionPrompt,
  categoryFor,
  extractFromConversation,
  formatConversation,
  parseExtractionResponse,
  passesSaveGate,
} from "../src/extraction.js";
import { createLimbic } from "../src/index.js";
import type { CompleteFn, ExtractedMemory } from "../src/types.js";

const LONG_ENOUGH = [
  { role: "user", content: "My name is Ada and I work as a mathematician in London." },
  { role: "assistant", content: "Lovely to meet you, Ada." },
];

const canned = (response: string): CompleteFn => async () => response;

describe("formatConversation", () => {
  it("renders ROLE: content, uppercasing the role", () => {
    expect(formatConversation([{ role: "user", content: "hi" }])).toBe("USER: hi");
  });

  it("drops empty and whitespace-only messages", () => {
    const out = formatConversation([
      { role: "user", content: "  " },
      { role: "assistant", content: "" },
      { role: "user", content: " kept " },
    ]);
    expect(out).toBe("USER: kept");
  });

  it("keeps only the last ten turns", () => {
    const turns = Array.from({ length: 14 }, (_, i) => ({ role: "user", content: `m${i}` }));
    const lines = formatConversation(turns).split("\n");
    expect(lines).toHaveLength(CONVERSATION_WINDOW);
    expect(lines[0]).toBe("USER: m4");
    expect(lines.at(-1)).toBe("USER: m13");
  });
});

describe("buildExtractionPrompt", () => {
  it("returns null below the 50-character floor, so no LLM call is made", () => {
    expect(buildExtractionPrompt([{ role: "user", content: "hi" }])).toBeNull();
  });

  it("substitutes the conversation literally and leaves the JSON example intact", () => {
    const prompt = buildExtractionPrompt(LONG_ENOUGH);
    expect(prompt).not.toBeNull();
    expect(prompt).toContain("USER: My name is Ada");
    expect(prompt).not.toContain("{conversation}");
    // The example object survives. the origin engine's `str.format` call cannot get this far:
    // it reads `{\n  "memories": ...}` as a replacement field and raises
    // KeyError, which its own `except Exception` then swallows.
    expect(prompt).toContain('"memories": [');
    expect(prompt).toContain('{"memories": []}');
  });

  it("keeps the prompt template itself free of stray placeholders", () => {
    const placeholders = EXTRACTION_PROMPT.match(/\{conversation\}/g) ?? [];
    expect(placeholders).toHaveLength(1);
  });
});

describe("parseExtractionResponse", () => {
  it("reads a well-formed response", () => {
    const rows = parseExtractionResponse(
      JSON.stringify({
        memories: [
          {
            type: "FACT",
            subject: "user",
            content: "User's name is Ada",
            importance: 0.9,
            keywords: ["name", "ada"],
            supersedes: null,
            date_expression: null,
            feeling: "neutral",
          },
        ],
      }),
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      content: "User's name is Ada",
      extractionType: "fact",
      importance: 0.9,
      keywords: ["name", "ada"],
      confidence: 0.8,
      subject: "user",
      feeling: "neutral",
    });
    // null supersedes / date_expression stay absent rather than becoming null.
    expect(rows[0]).not.toHaveProperty("supersedes");
    expect(rows[0]).not.toHaveProperty("dateExpression");
  });

  it("finds the JSON inside a model's prose and fences", () => {
    const rows = parseExtractionResponse(
      'Sure! Here you go:\n```json\n{"memories": [{"type": "INTEREST", "content": "likes chess", "importance": 0.6}]}\n```\nHope that helps.',
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]?.extractionType).toBe("interest");
  });

  it("returns [] on malformed JSON, without throwing", () => {
    for (const bad of ["", "no json here", "{not json}", "{", "}{", '{"memories": "nope"}']) {
      expect(() => parseExtractionResponse(bad)).not.toThrow();
      expect(parseExtractionResponse(bad), bad).toEqual([]);
    }
  });

  it("keeps an unknown extraction type instead of dropping the row", () => {
    // the origin engine's `ExtractionType(...)` raises here and the row is skipped; limbic
    // validates against the known set rather than rejecting.
    const rows = parseExtractionResponse('{"memories": [{"type": "VIBE", "content": "x"}]}');
    expect(rows).toHaveLength(1);
    expect(rows[0]?.extractionType).toBe("vibe");
    expect(categoryFor("vibe")).toBe("general");
  });

  it("defaults subject to user for anything that is not persona", () => {
    const rows = parseExtractionResponse(
      '{"memories": [{"content": "a", "subject": "cat"}, {"content": "b", "subject": "persona"}]}',
    );
    expect(rows.map((r) => r.subject)).toEqual(["user", "persona"]);
  });

  it("skips rows that are not objects", () => {
    const rows = parseExtractionResponse('{"memories": [null, 3, "x", {"content": "kept"}]}');
    expect(rows).toHaveLength(1);
    expect(rows[0]?.content).toBe("kept");
  });
});

describe("the save gate — importance >= 0.4 AND confidence >= 0.6", () => {
  const row = (importance: number, confidence: number): ExtractedMemory => ({
    content: "x",
    extractionType: "fact",
    importance,
    keywords: [],
    confidence,
    subject: "user",
    feeling: "neutral",
  });

  it("admits a row sitting exactly on both bounds", () => {
    expect(passesSaveGate(row(MIN_IMPORTANCE, MIN_CONFIDENCE))).toBe(true);
  });

  it("rejects a row a hair under either bound", () => {
    expect(passesSaveGate(row(0.399, 0.9))).toBe(false);
    expect(passesSaveGate(row(0.9, 0.599))).toBe(false);
  });

  it("needs both, not either", () => {
    expect(passesSaveGate(row(1, 0))).toBe(false);
    expect(passesSaveGate(row(0, 1))).toBe(false);
  });
});

describe("extractFromConversation", () => {
  it("passes the origin engine's max_tokens and temperature", async () => {
    const complete = vi.fn<CompleteFn>(async () => '{"memories": []}');
    await extractFromConversation(complete, LONG_ENOUGH);
    expect(complete).toHaveBeenCalledOnce();
    expect(complete.mock.calls[0]?.[1]).toEqual({
      maxTokens: EXTRACTION_MAX_TOKENS,
      temperature: EXTRACTION_TEMPERATURE,
    });
  });

  it("never calls the model for a conversation under the floor", async () => {
    const complete = vi.fn<CompleteFn>(async () => '{"memories": []}');
    await expect(extractFromConversation(complete, [{ role: "user", content: "hi" }])).resolves.toEqual([]);
    expect(complete).not.toHaveBeenCalled();
  });

  it("returns [] when the model rejects, rather than propagating", async () => {
    const complete: CompleteFn = async () => {
      throw new Error("model is down");
    };
    await expect(extractFromConversation(complete, LONG_ENOUGH)).resolves.toEqual([]);
  });
});

describe("createLimbic().extract", () => {
  it("throws when no CompleteFn was configured, rather than returning []", async () => {
    const limbic = createLimbic();
    await expect(limbic.extract(LONG_ENOUGH)).rejects.toThrow(/CompleteFn/);
  });

  it("turns a canned response into memories the caller can store", async () => {
    const limbic = createLimbic({
      complete: canned(
        '{"memories": [{"type": "FACT", "content": "User is a mathematician", "importance": 0.8, "keywords": ["job"]}]}',
      ),
    });
    const extracted = await limbic.extract(LONG_ENOUGH);
    expect(extracted).toHaveLength(1);
    expect(passesSaveGate(extracted[0]!)).toBe(true);

    const saved = await limbic.remember(extracted[0]!.content, {
      category: categoryFor(extracted[0]!.extractionType),
      importance: extracted[0]!.importance,
      keywords: extracted[0]!.keywords,
      feeling: extracted[0]!.feeling,
    });
    expect(saved.category).toBe("personal_fact");
    expect(await limbic.store.count()).toBe(1);
  });

  it("returns [] for malformed JSON without throwing", async () => {
    const limbic = createLimbic({ complete: canned("I am not JSON at all, sorry.") });
    await expect(limbic.extract(LONG_ENOUGH)).resolves.toEqual([]);
  });
});
