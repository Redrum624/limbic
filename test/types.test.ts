/**
 * Type-level sanity for the 0.1.0 surface.
 *
 * These assertions are enforced by `tsc --noEmit`, which runs over `test/` as
 * well as `src/`: `expectTypeOf` is a compile-time check with a no-op runtime.
 * The runtime `expect`s pin the constants the scoring and extraction suites
 * read as contract values, so a stray edit to a weight fails here and not
 * somewhere downstream inside a golden-parity diff.
 */

import { describe, expect, expectTypeOf, it } from "vitest";
import {
  DEFAULT_WEIGHTS,
  EMBED_BLEND,
  type CompleteFn,
  type Embedder,
  type ExtractedMemory,
  type Memory,
  type MemoryCategory,
  type MemoryEmotion,
  type ScoreWeights,
} from "../src/types.js";

describe("core types", () => {
  it("DEFAULT_WEIGHTS matches the origin engine retrieval_service.py and sums to 1", () => {
    expect(DEFAULT_WEIGHTS).toEqual({
      recency: 0.25,
      importance: 0.35,
      relevance: 0.25,
      emotion: 0.15,
    });
    const sum =
      DEFAULT_WEIGHTS.recency +
      DEFAULT_WEIGHTS.importance +
      DEFAULT_WEIGHTS.relevance +
      DEFAULT_WEIGHTS.emotion;
    expect(sum).toBeCloseTo(1, 12);
    expectTypeOf(DEFAULT_WEIGHTS).toEqualTypeOf<ScoreWeights>();
  });

  it("EMBED_BLEND is the 0.3 half of the 0.7/0.3 blend", () => {
    expect(EMBED_BLEND).toBe(0.3);
  });

  it("MemoryCategory is exactly the nine categories of the origin engine's enum", () => {
    const all: MemoryCategory[] = [
      "personal_fact",
      "preference",
      "relationship",
      "experience",
      "emotion",
      "interest",
      "work",
      "health",
      "general",
    ];
    expect(new Set(all).size).toBe(9);
    expectTypeOf<(typeof all)[number]>().toEqualTypeOf<MemoryCategory>();
  });

  it("a fully populated Memory type-checks, embedding and emotion included", () => {
    const m: Memory = {
      id: "m1",
      content: "Marc adopted a cat named Nyx",
      category: "relationship",
      importance: 0.8,
      keywords: ["cat", "nyx"],
      sourceMessageId: "msg-7",
      createdAt: "2026-08-26T12:00:00.000Z",
      lastAccessed: "2026-08-26T12:00:00.000Z",
      accessCount: 0,
      subject: "user",
      feeling: "joy",
      emotion: { label: "joy", intensity: 0.75 },
      embedding: new Float32Array([0.1, 0.2]),
      embeddingModel: "nomic-embed-text",
    };
    expect(m.embedding).toBeInstanceOf(Float32Array);
    expectTypeOf(m.category).toEqualTypeOf<MemoryCategory>();
    expectTypeOf(m.embedding).toEqualTypeOf<Float32Array | undefined>();
    expectTypeOf(m.emotion).toEqualTypeOf<MemoryEmotion | undefined>();
    expectTypeOf(m.subject).toEqualTypeOf<"user" | "persona">();
  });

  it("a minimal Memory omits every optional field", () => {
    const m: Memory = {
      id: "m2",
      content: "likes rain",
      category: "general",
      importance: 0.5,
      keywords: [],
      createdAt: "2026-08-26T12:00:00.000Z",
      lastAccessed: "2026-08-26T12:00:00.000Z",
      accessCount: 0,
      subject: "user",
    };
    expect(m.feeling).toBeUndefined();
    expect(m.sourceMessageId).toBeUndefined();
    expect(m.emotion).toBeUndefined();
    expect(m.embeddingModel).toBeUndefined();
  });

  it("ExtractedMemory carries feeling as a required field and extractionType as open text", () => {
    const e: ExtractedMemory = {
      content: "Marc's birthday is May 15th",
      extractionType: "fact",
      importance: 0.9,
      keywords: ["birthday"],
      confidence: 0.85,
      subject: "user",
      dateExpression: "May 15th",
      feeling: "neutral",
    };
    expect(e.feeling).toBe("neutral");
    expectTypeOf<ExtractedMemory["feeling"]>().toEqualTypeOf<string>();
    expectTypeOf<ExtractedMemory["extractionType"]>().toEqualTypeOf<string>();
    expectTypeOf<ExtractedMemory["supersedes"]>().toEqualTypeOf<string | undefined>();
  });

  it("CompleteFn and Embedder are satisfiable without any provider SDK", async () => {
    const complete: CompleteFn = async (prompt, opts) =>
      `${prompt}|${String(opts?.maxTokens ?? 0)}|${String(opts?.temperature ?? 0)}`;
    expect(await complete("hi", { maxTokens: 8 })).toBe("hi|8|0");

    const embedder: Embedder = {
      model: "stub",
      embed: async (texts) => texts.map((t) => new Float32Array([t.length])),
    };
    const vectors = await embedder.embed(["abc", "de"]);
    expect(vectors.map((v) => v[0])).toEqual([3, 2]);
    expectTypeOf(embedder.embed).returns.resolves.toEqualTypeOf<Float32Array[]>();
  });
});
