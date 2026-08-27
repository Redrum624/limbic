/**
 * OllamaEmbedder against a REAL local Ollama.
 *
 * This is the only test in the suite that touches the network, and it is
 * deliberately in the default run rather than under `test/integration/`
 * (which `vitest.config.ts` excludes outright): a test that never executes
 * proves nothing, and this one is the only evidence that the ported endpoint,
 * body keys and response field are right against the actual server rather than
 * against a stub that agrees with the code by construction.
 *
 * It probes the host once and SKIPS the whole block when nothing answers, so
 * CI — and any machine without Ollama — stays green. It never starts, pulls or
 * installs anything.
 *
 * Host is `127.0.0.1`, never `localhost`: on this dual-stack Windows box
 * `localhost` resolves `::1` first while Ollama binds IPv4, which measured at
 * ~2.1 s per request against a few ms for the literal address.
 */

import { describe, expect, it } from "vitest";

import { OllamaEmbedder } from "../src/embedders/ollama.js";
import { cosine } from "../src/internal/vec.js";

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

const tags = await probe();
const modelPresent =
  tags !== null && tags.some((name) => name === MODEL || name.startsWith(`${MODEL}:`));

if (tags === null) {
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
   * ⚠️ Measured on this machine, 2026-08-27, `nomic-embed-text:latest`
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
