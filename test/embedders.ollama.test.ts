/**
 * OllamaEmbedder — endpoint/body/response parity with the origin engine's
 * `ollama_client.py` `embeddings()`, exercised against a
 * stubbed `fetch`. No network.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_OLLAMA_HOST,
  OllamaEmbedder,
  parseEmbeddings,
} from "../src/embedders/ollama.js";
import { EmbedderUnavailableError, isEmbedderUnavailable } from "../src/embedders/errors.js";

interface Call {
  url: string;
  init: { method?: string; headers?: Record<string, string>; body?: string } | undefined;
}

function stubFetch(handler: (call: Call) => unknown) {
  const calls: Call[] = [];
  const fn = vi.fn(async (url: string, init?: Call["init"]) => {
    const call: Call = { url, init };
    calls.push(call);
    const result = await handler(call);
    return result as never;
  });
  return { fn, calls };
}

function ok(body: unknown) {
  return {
    ok: true,
    status: 200,
    async text() {
      return JSON.stringify(body);
    },
    async json() {
      return body;
    },
  };
}

function notOk(status: number, statusText: string, body: string) {
  return {
    ok: false,
    status,
    statusText,
    async text() {
      return body;
    },
    async json() {
      return JSON.parse(body);
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("OllamaEmbedder: construction", () => {
  it("defaults to 127.0.0.1, never localhost", () => {
    // Not cosmetic: on a dual-stack host where Ollama binds IPv4 only,
    // `localhost` can resolve ::1 first — measured 2026-08-27 at ~2.1 s per
    // request against a few ms for the literal address.
    expect(DEFAULT_OLLAMA_HOST).toBe("http://127.0.0.1:11434");
    expect(DEFAULT_OLLAMA_HOST).not.toContain("localhost");

    const e = new OllamaEmbedder({ model: "nomic-embed-text", fetch: stubFetch(() => ok({})).fn });
    expect(e.host).toBe("http://127.0.0.1:11434");
    expect(e.endpoint).toBe("http://127.0.0.1:11434/api/embed");
  });

  it("trims a trailing slash off the host", () => {
    const e = new OllamaEmbedder({
      model: "m",
      host: "http://127.0.0.1:11434/",
      fetch: stubFetch(() => ok({})).fn,
    });
    expect(e.endpoint).toBe("http://127.0.0.1:11434/api/embed");
  });

  it("rejects a missing model as a TypeError, not an availability error", () => {
    // A pipeline that catches EmbedderUnavailableError must not swallow this.
    expect(() => new OllamaEmbedder({ model: "" } as never)).toThrow(TypeError);
    expect(() => new OllamaEmbedder({} as never)).toThrow(TypeError);
  });

  it("uses the global fetch when none is injected", () => {
    vi.stubGlobal("fetch", stubFetch(() => ok({})).fn);
    expect(() => new OllamaEmbedder({ model: "m" })).not.toThrow();
  });
});

describe("OllamaEmbedder: the happy path", () => {
  it("POSTs /api/embed with {model, input} and reads json.embeddings", async () => {
    const { fn, calls } = stubFetch(() =>
      ok({ embeddings: [[1, 0, 0], [0, 0.5, 0.5]], model: "nomic-embed-text" }),
    );
    const embedder = new OllamaEmbedder({ model: "nomic-embed-text", fetch: fn });

    const vectors = await embedder.embed(["coffee", "tea"]);

    expect(calls).toHaveLength(1);
    const call = calls[0] as Call;
    expect(call.url).toBe("http://127.0.0.1:11434/api/embed");
    expect(call.init?.method).toBe("POST");
    expect(JSON.parse(call.init?.body as string)).toEqual({
      model: "nomic-embed-text",
      input: ["coffee", "tea"],
    });

    expect(vectors).toHaveLength(2);
    expect(vectors[0]).toBeInstanceOf(Float32Array);
    expect(Array.from(vectors[0] as Float32Array)).toEqual([1, 0, 0]);
    expect(Array.from(vectors[1] as Float32Array)).toEqual([0, 0.5, 0.5]);
  });

  it("sends the whole batch in ONE round trip", async () => {
    const texts = Array.from({ length: 25 }, (_, i) => `memory ${i}`);
    const { fn, calls } = stubFetch(() => ok({ embeddings: texts.map(() => [1, 2, 3]) }));
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });

    await embedder.embed(texts);

    expect(calls).toHaveLength(1);
    expect(JSON.parse(calls[0]?.init?.body as string).input).toHaveLength(25);
  });

  it("returns [] for no texts without touching the network", async () => {
    const { fn, calls } = stubFetch(() => ok({ embeddings: [] }));
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });

    expect(await embedder.embed([])).toEqual([]);
    expect(calls).toHaveLength(0);
  });

  it("honours a custom host", async () => {
    const { fn, calls } = stubFetch(() => ok({ embeddings: [[1]] }));
    const embedder = new OllamaEmbedder({
      model: "m",
      host: "http://10.0.0.5:11434",
      fetch: fn,
    });
    await embedder.embed(["x"]);
    expect(calls[0]?.url).toBe("http://10.0.0.5:11434/api/embed");
  });
});

describe("OllamaEmbedder: failure modes", () => {
  it("turns a non-200 into EmbedderUnavailableError carrying status and body", async () => {
    const { fn } = stubFetch(() =>
      notOk(404, "Not Found", JSON.stringify({ error: 'model "nope" not found' })),
    );
    const embedder = new OllamaEmbedder({ model: "nope", fetch: fn });

    await expect(embedder.embed(["x"])).rejects.toThrowError(EmbedderUnavailableError);
    await expect(embedder.embed(["x"])).rejects.toThrow(/404/);
    await expect(embedder.embed(["x"])).rejects.toThrow(/not found/);
  });

  it("turns a network error into EmbedderUnavailableError naming the endpoint", async () => {
    const { fn } = stubFetch(() => {
      throw Object.assign(new Error("connect ECONNREFUSED 127.0.0.1:11434"), {
        name: "TypeError",
      });
    });
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });

    const error = await embedder.embed(["x"]).catch((e: unknown) => e);
    expect(isEmbedderUnavailable(error)).toBe(true);
    expect((error as EmbedderUnavailableError).embedder).toBe("ollama");
    expect((error as Error).message).toContain("127.0.0.1:11434/api/embed");
    expect((error as Error).message).toContain("ECONNREFUSED");
  });

  it("reports a timeout as such", async () => {
    const { fn } = stubFetch(() => {
      throw Object.assign(new Error("aborted"), { name: "AbortError" });
    });
    const embedder = new OllamaEmbedder({ model: "m", timeoutMs: 5, fetch: fn });
    await expect(embedder.embed(["x"])).rejects.toThrow(/timed out/);
  });

  it("turns a non-JSON body into EmbedderUnavailableError", async () => {
    const { fn } = stubFetch(() => ({
      ok: true,
      status: 200,
      async text() {
        return "<html>proxy</html>";
      },
      async json(): Promise<unknown> {
        throw new SyntaxError("Unexpected token <");
      },
    }));
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });
    await expect(embedder.embed(["x"])).rejects.toThrow(/non-JSON/);
  });

  it("rejects a body with the wrong number of vectors", async () => {
    const { fn } = stubFetch(() => ok({ embeddings: [[1, 2, 3]] }));
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });
    await expect(embedder.embed(["a", "b"])).rejects.toThrow(/asked for 2 vector\(s\), got 1/);
  });

  it("rejects a non-array `embed` argument as a TypeError", async () => {
    const { fn } = stubFetch(() => ok({ embeddings: [] }));
    const embedder = new OllamaEmbedder({ model: "m", fetch: fn });
    await expect(embedder.embed("coffee" as never)).rejects.toThrow(TypeError);
  });
});

describe("parseEmbeddings", () => {
  const endpoint = "http://127.0.0.1:11434/api/embed";

  it("names the legacy endpoint when `embeddings` is missing", () => {
    // /api/embeddings (legacy) returns `embedding`, singular. limbic posts to
    // /api/embed on purpose, exactly as the origin engine's client does.
    expect(() => parseEmbeddings({ embedding: [1, 2, 3] }, 1, endpoint)).toThrow(
      /legacy \/api\/embeddings/,
    );
  });

  it("rejects a non-object payload", () => {
    expect(() => parseEmbeddings(null, 1, endpoint)).toThrow(EmbedderUnavailableError);
    expect(() => parseEmbeddings("nope", 1, endpoint)).toThrow(/expected an object/);
  });

  it("rejects an empty or non-array vector", () => {
    expect(() => parseEmbeddings({ embeddings: [[]] }, 1, endpoint)).toThrow(/non-empty array/);
    expect(() => parseEmbeddings({ embeddings: ["x"] }, 1, endpoint)).toThrow(/non-empty array/);
  });

  it("rejects a non-finite component instead of producing a NaN vector", () => {
    // A NaN would poison cosine() into returning null forever after, silently.
    expect(() => parseEmbeddings({ embeddings: [[1, null, 3]] }, 1, endpoint)).toThrow(
      /not a finite number/,
    );
    expect(() => parseEmbeddings({ embeddings: [[1, "2", 3]] }, 1, endpoint)).toThrow(
      /not a finite number/,
    );
  });

  it("narrows float64 JSON numbers to float32, the stored layout", () => {
    const [vector] = parseEmbeddings({ embeddings: [[0.1, 0.2]] }, 1, endpoint);
    expect(vector).toBeInstanceOf(Float32Array);
    expect(vector?.[0]).toBe(Math.fround(0.1));
  });
});

describe("the live suite stays opt-in (S-10)", () => {
  it("short-circuits its probe behind LIMBIC_LIVE, before any network call", async () => {
    // A source-level pin, like the fixture hashes: the property "the default
    // run makes zero network calls" cannot be observed from inside the suite,
    // so assert the gate itself — the env check and the short-circuit that
    // keeps even the probe from running without it.
    const { readFile } = await import("node:fs/promises");
    const source = await readFile(
      new URL("./embedders.ollama.live.test.ts", import.meta.url),
      "utf8",
    );
    expect(source).toContain('process.env["LIMBIC_LIVE"] === "1"');
    expect(source).toContain("LIVE ? await probe() : null");
  });
});
