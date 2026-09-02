/**
 * Embedder lifecycle: what happens when a load half-succeeds, when a server
 * stalls mid-body, and when a caller is done with the native resources.
 *
 * Companion to `test/embedders.peers.test.ts` (load-error and shape tests);
 * everything here runs against injected fakes, nothing touches a real model
 * or the network.
 */

import { describe, expect, it, vi } from "vitest";

import { NodeLlamaCppEmbedder } from "../src/embedders/node-llama-cpp.js";
import { TransformersEmbedder } from "../src/embedders/transformers.js";
import { OllamaEmbedder, type FetchLike } from "../src/embedders/ollama.js";
import { EmbedderUnavailableError, isEmbedderUnavailable } from "../src/embedders/errors.js";

describe("NodeLlamaCppEmbedder — failed context creation (L-01)", () => {
  it("disposes the model loaded by the failed attempt instead of keeping it", async () => {
    const disposeModel = vi.fn();
    const loadModel = vi.fn(async () => {
      if (loadModel.mock.calls.length === 1) {
        return {
          dispose: disposeModel,
          createEmbeddingContext: async () => {
            throw new Error("not enough memory for the context");
          },
        };
      }
      return {
        createEmbeddingContext: async () => ({
          getEmbeddingFor: async () => ({ vector: [1] }),
        }),
      };
    });
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({ getLlama: async () => ({ loadModel }) }),
    });

    await expect(e.embed(["a"])).rejects.toThrow(/could not load the embedding model/);
    // The GGUF loaded on the failed attempt must be freed, or every retry
    // stacks another full native model load on top of it.
    expect(disposeModel).toHaveBeenCalledTimes(1);

    // The retry starts clean and works.
    const vectors = await e.embed(["b"]);
    expect(Array.from(vectors[0] as Float32Array)).toEqual([1]);
    expect(loadModel).toHaveBeenCalledTimes(2);

    // dispose() has nothing left over from the failed attempt to free.
    await e.dispose();
    expect(disposeModel).toHaveBeenCalledTimes(1);
  });
});

describe("NodeLlamaCppEmbedder — dispose() during an in-flight load (L-07)", () => {
  it("does not resurrect a context whose model dispose() already released", async () => {
    const disposeModel = vi.fn();
    const disposeContext = vi.fn();
    let releaseLoad!: () => void;
    const gate = new Promise<void>((resolve) => {
      releaseLoad = resolve;
    });
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({
        getLlama: async () => ({
          loadModel: async () => ({
            dispose: disposeModel,
            createEmbeddingContext: async () => {
              await gate; // held open until the test calls releaseLoad()
              return {
                dispose: disposeContext,
                getEmbeddingFor: async () => ({ vector: [1] }),
              };
            },
          }),
        }),
      }),
    });

    const inFlight = e.embed(["a"]);
    // Let the load reach the createEmbeddingContext await before disposing.
    await new Promise((resolve) => setTimeout(resolve, 0));
    await e.dispose();
    expect(disposeModel).toHaveBeenCalledTimes(1);

    releaseLoad();
    // The load must not hand back a context whose model is already gone: the
    // context is released, the embed fails loudly, nothing is kept.
    await expect(inFlight).rejects.toThrow(/disposed/);
    expect(disposeContext).toHaveBeenCalledTimes(1);

    // A second dispose() finds nothing left over.
    await e.dispose();
    expect(disposeModel).toHaveBeenCalledTimes(1);
    expect(disposeContext).toHaveBeenCalledTimes(1);
  });
});

describe("OllamaEmbedder — timeoutMs covers the whole exchange (L-02)", () => {
  it("aborts a response whose headers arrived but whose body stalls", async () => {
    // 200 OK immediately, then a body that never comes unless the request
    // signal aborts — the shape of a server that accepted and then hung.
    const stalledFetch: FetchLike = async (_url, init) => ({
      ok: true,
      status: 200,
      text: async () => "",
      json: () =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(Object.assign(new Error("The operation was aborted"), { name: "AbortError" }));
          });
        }),
    });

    const e = new OllamaEmbedder({ model: "m", timeoutMs: 25, fetch: stalledFetch });
    const error = await e.embed(["x"]).catch((err: unknown) => err);
    expect(isEmbedderUnavailable(error)).toBe(true);
    expect((error as Error).message).toMatch(/request timed out/);
  });
});

describe("TransformersEmbedder.dispose (L-03)", () => {
  it("releases the ONNX session and tolerates a second dispose", async () => {
    const disposeSession = vi.fn();
    const extractor = Object.assign(
      vi.fn(async () => ({ data: [1, 2], dims: [1, 2] })),
      { dispose: disposeSession },
    );
    const e = new TransformersEmbedder({
      model: "m",
      load: async () => ({ pipeline: async () => extractor }),
    });

    await e.embed(["a"]);
    await e.dispose();
    await e.dispose();
    expect(disposeSession).toHaveBeenCalledTimes(1);
  });

  it("is a no-op before anything was loaded", async () => {
    const e = new TransformersEmbedder({
      model: "m",
      load: () => Promise.reject(new Error("never reached")),
    });
    await expect(e.dispose()).resolves.toBeUndefined();
  });
});

describe("OllamaEmbedder — host validation (S-09)", () => {
  it("rejects Ollama's own OLLAMA_HOST shorthand, which is not a URL", () => {
    expect(() => new OllamaEmbedder({ model: "m", host: "127.0.0.1:11434" })).toThrow(TypeError);
    expect(() => new OllamaEmbedder({ model: "m", host: "localhost:11434" })).toThrow(TypeError);
  });

  it("rejects a non-http(s) scheme at construction, not as a late fetch error", () => {
    expect(() => new OllamaEmbedder({ model: "m", host: "ftp://127.0.0.1:11434" })).toThrow(
      TypeError,
    );
    expect(() => new OllamaEmbedder({ model: "m", host: "file:///var/run/ollama" })).toThrow(
      TypeError,
    );
  });

  it("accepts http and https, still stripping a trailing slash", () => {
    expect(new OllamaEmbedder({ model: "m", host: "http://localhost:11434/" }).endpoint).toBe(
      "http://localhost:11434/api/embed",
    );
    expect(new OllamaEmbedder({ model: "m", host: "https://ollama.internal" }).host).toBe(
      "https://ollama.internal",
    );
  });

  it("does not throw EmbedderUnavailableError for a bad host — it is a programming error", () => {
    try {
      new OllamaEmbedder({ model: "m", host: "not a url at all" });
      expect.unreachable("constructor should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(TypeError);
      expect(error).not.toBeInstanceOf(EmbedderUnavailableError);
    }
  });
});
