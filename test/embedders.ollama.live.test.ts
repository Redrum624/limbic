/**
 * OllamaEmbedder against a REAL local Ollama — **opt-in via `LIMBIC_LIVE=1`**.
 *
 * The default suite makes zero network calls: without `LIMBIC_LIVE=1` this
 * whole file is inert — no probe, no request, every test skipped loudly. The
 * gate is opt-in rather than opt-out because a default `vitest run` (and CI)
 * must never send test strings to whatever answers at `LIMBIC_OLLAMA_HOST`.
 *
 *     LIMBIC_LIVE=1 npx vitest run test/embedders.ollama.live.test.ts
 *
 * Optionally set `LIMBIC_OLLAMA_HOST` (default `http://127.0.0.1:11434`) and
 * `LIMBIC_OLLAMA_MODEL` (default `nomic-embed-text`). The file stays in the
 * suite rather than being culled: a test that never executes proves nothing,
 * and this one is the only evidence that the ported endpoint, body keys and
 * response field are right against the actual server rather than against a
 * stub that agrees with the code by construction. When opted in, it probes the
 * host once and skips the block when nothing answers; it never starts, pulls
 * or installs anything.
 *
 * Host is `127.0.0.1`, never `localhost`: on a dual-stack host where Ollama
 * binds IPv4 only, `localhost` can resolve `::1` first — measured 2026-08-27
 * at ~2.1 s per request against a few ms for the literal address.
 */

import { describe, expect, it } from "vitest";

import { OllamaEmbedder } from "../src/embedders/ollama.js";
import { cosine } from "../src/internal/vec.js";

const LIVE = process.env["LIMBIC_LIVE"] === "1";
const HOST = process.env["LIMBIC_OLLAMA_HOST"] ?? "http://127.0.0.1:11434";
const MODEL = process.env["LIMBIC_OLLAMA_MODEL"] ?? "nomic-embed-text";
const PROBE_TIMEOUT_MS = 1_500;

async function probe(): Promise<string[] | null> {
  try {
    const response = await fetch(`${HOST}/api/tags`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const body = (await response.json()) as { models?: Array<{ name?: string }> };
    return (body.models ?? []).map((m) => m.name ?? "");
  } catch {
    return null;
  }
}

// The env gate comes first: without it, not even the probe runs.
const tags = LIVE ? await probe() : null;
const modelPresent =
  tags !== null && tags.some((name) => name === MODEL || name.startsWith(`${MODEL}:`));

if (!LIVE) {
  // eslint-disable-next-line no-console
  console.info("[live] skipping: set LIMBIC_LIVE=1 to run the live Ollama tests");
} else if (tags === null) {
  // eslint-disable-next-line no-console
  console.info(`[live] skipping: no Ollama at ${HOST}`);
} else if (!modelPresent) {
  // eslint-disable-next-line no-console
  console.info(`[live] skipping: Ollama at ${HOST} has no ${MODEL} (has: ${tags.join(", ")})`);
}

describe.skipIf(!modelPresent)(`OllamaEmbedder against a real ${MODEL}`, () => {
  const embedder = new OllamaEmbedder({ host: HOST, model: MODEL });

  it("returns one finite Float32Array per input, in order, in one round trip", async () => {
    const vectors = await embedder.embed([
      "User drinks coffee every morning",
      "User adores flat whites",
      "User's manager is called Dave",
    ]);

    expect(vectors).toHaveLength(3);
    for (const vector of vectors) {
      expect(vector).toBeInstanceOf(Float32Array);
      expect(vector.length).toBeGreaterThan(0);
      expect(vector.every((c) => Number.isFinite(c))).toBe(true);
      expect(vector.some((c) => c !== 0)).toBe(true);
    }
    // One model, one dimensionality.
    expect(new Set(vectors.map((v) => v.length)).size).toBe(1);
  }, 60_000);

  it("is deterministic for the same text", async () => {
    const [a] = await embedder.embed(["coffee"]);
    const [b] = await embedder.embed(["coffee"]);
    expect(cosine(a, b)).toBeCloseTo(1, 6);
  }, 60_000);

  /**
   * The cosine channel exists for the memory that MEANS the query without
   * sharing a word with it: "User adores flat whites" has a Jaccard relevance
   * of exactly 0.0 against "coffee" and must still outrank "User's manager is
   * called Dave".
   *
   * ⚠️ Measured 2026-08-27 with `nomic-embed-text:latest`
   * (F16, 137M, 768-d), via `/api/embed`:
   *
   *   text sent verbatim          flat_whites 0.375108  dave 0.383352  portuguese 0.438127
   *   with nomic's task prefixes  flat_whites 0.494534  dave 0.450479  portuguese 0.415407
   *
   * Verbatim, the model gets it BACKWARDS — every one of those three is within
   * 0.07 of the others and the intended ranking is inverted. nomic-embed-text
   * is trained with mandatory task prefixes (`search_query:` for the query,
   * `search_document:` for the corpus) and is close to useless without them.
   *
   * That is a caller's concern, not the embedder's: `OllamaEmbedder` is a
   * transport and must send exactly the strings it is given — silently
   * prefixing them would corrupt every other model. So the assertion below
   * uses the model as documented, and the verbatim numbers are recorded here
   * rather than asserted, because they are a fact about this model version and
   * not a property of the port.
   */
  it("ranks by meaning when the model is used as documented", async () => {
    const [query, related, unrelated] = await embedder.embed([
      "search_query: coffee",
      "search_document: User adores flat whites",
      "search_document: User's manager is called Dave",
    ]);

    const near = cosine(query, related) as number;
    const far = cosine(query, unrelated) as number;
    expect(near).toBeGreaterThan(far);
  }, 60_000);

  it("sends the text verbatim — it never rewrites a prompt", async () => {
    // The guarantee the note above depends on: whatever the caller passes is
    // what reaches the model, so a caller CAN apply the prefixes and get the
    // documented behaviour.
    const [bare] = await embedder.embed(["coffee"]);
    const [prefixed] = await embedder.embed(["search_query: coffee"]);
    expect(cosine(bare, prefixed)).not.toBeCloseTo(1, 6);
  }, 60_000);

  it("reports an unknown model as EmbedderUnavailableError, not a crash", async () => {
    const broken = new OllamaEmbedder({
      host: HOST,
      model: "limbic-no-such-model-xyz",
      timeoutMs: 10_000,
    });
    await expect(broken.embed(["x"])).rejects.toThrow(/EmbedderUnavailable|not found|404/);
  }, 30_000);
});
