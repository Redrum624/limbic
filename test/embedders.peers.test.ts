/**
 * The two optional-peer adapters.
 *
 * Neither peer is installed (both are `peerDependenciesMeta.optional` and the
 * core has zero required runtime dependencies), so these are load-error and
 * shape tests: the adapter must report a missing peer with an install hint
 * rather than leaking `ERR_MODULE_NOT_FOUND`, and must not touch the peer at
 * all until someone actually embeds. The real-model runs live in
 * `test/embedders.ollama.live.test.ts` and their `node-llama-cpp` equivalent is
 * out of scope until a GGUF path is configured.
 */

import { describe, expect, it, vi } from "vitest";

import { NodeLlamaCppEmbedder } from "../src/embedders/node-llama-cpp.js";
import {
  DEFAULT_EXTRACT_OPTIONS,
  TransformersEmbedder,
  splitPooledTensor,
} from "../src/embedders/transformers.js";
import { EmbedderUnavailableError, isEmbedderUnavailable } from "../src/embedders/errors.js";

const moduleNotFound = () =>
  Promise.reject(
    Object.assign(new Error("Cannot find module"), { code: "ERR_MODULE_NOT_FOUND" }),
  );

describe("NodeLlamaCppEmbedder", () => {
  it("names itself after the model file when no model name is given", () => {
    const e = new NodeLlamaCppEmbedder({ modelPath: "C:\\models\\nomic-embed-text-v1.5.Q4.gguf" });
    expect(e.model).toBe("nomic-embed-text-v1.5.Q4.gguf");
    expect(new NodeLlamaCppEmbedder({ modelPath: "/m/x.gguf", model: "custom" }).model).toBe(
      "custom",
    );
  });

  it("requires a modelPath, as a TypeError", () => {
    expect(() => new NodeLlamaCppEmbedder({ modelPath: "" })).toThrow(TypeError);
    expect(() => new NodeLlamaCppEmbedder({} as never)).toThrow(TypeError);
  });

  it("does not load the peer until embed() is called", async () => {
    const load = vi.fn(moduleNotFound);
    const e = new NodeLlamaCppEmbedder({ modelPath: "/m/x.gguf", load });
    expect(load).not.toHaveBeenCalled();
    // An empty batch is still not a reason to load a model.
    expect(await e.embed([])).toEqual([]);
    expect(load).not.toHaveBeenCalled();
  });

  it("reports a missing peer with an install hint", async () => {
    const e = new NodeLlamaCppEmbedder({ modelPath: "/m/x.gguf", load: moduleNotFound });
    const error = await e.embed(["x"]).catch((err: unknown) => err);
    expect(isEmbedderUnavailable(error)).toBe(true);
    expect((error as EmbedderUnavailableError).embedder).toBe("node-llama-cpp");
    expect((error as Error).message).toBe(
      "node-llama-cpp requires the optional peer node-llama-cpp: npm i node-llama-cpp",
    );
  });

  it("rejects a peer that loaded but has no v3 getLlama()", async () => {
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({ LlamaModel: class {} }),
    });
    await expect(e.embed(["x"])).rejects.toThrow(/exports no getLlama/);
  });

  it("wraps a model-load failure, naming the path", async () => {
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/missing.gguf",
      load: async () => ({
        getLlama: async () => ({
          loadModel: async () => {
            throw new Error("no such file");
          },
        }),
      }),
    });
    await expect(e.embed(["x"])).rejects.toThrow(/could not load the embedding model at \/m\/missing.gguf/);
  });

  it("drives getLlama -> loadModel -> createEmbeddingContext -> getEmbeddingFor", async () => {
    const loadModel = vi.fn(async () => ({
      createEmbeddingContext: async () => ({
        getEmbeddingFor: async (text: string) => ({ vector: [text.length, 1, 2] }),
      }),
    }));
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({ getLlama: async () => ({ loadModel }) }),
    });

    const vectors = await e.embed(["ab", "cdef"]);
    expect(loadModel).toHaveBeenCalledWith({ modelPath: "/m/x.gguf" });
    expect(vectors[0]).toBeInstanceOf(Float32Array);
    expect(Array.from(vectors[0] as Float32Array)).toEqual([2, 1, 2]);
    expect(Array.from(vectors[1] as Float32Array)).toEqual([4, 1, 2]);
  });

  it("loads the model once across repeated and concurrent calls", async () => {
    const loadModel = vi.fn(async () => ({
      createEmbeddingContext: async () => ({
        getEmbeddingFor: async () => ({ vector: [1] }),
      }),
    }));
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({ getLlama: async () => ({ loadModel }) }),
    });

    await Promise.all([e.embed(["a"]), e.embed(["b"]), e.embed(["c"])]);
    await e.embed(["d"]);
    expect(loadModel).toHaveBeenCalledTimes(1);
  });

  it("disposes the context and the model, and tolerates a second dispose", async () => {
    const disposeContext = vi.fn();
    const disposeModel = vi.fn();
    const e = new NodeLlamaCppEmbedder({
      modelPath: "/m/x.gguf",
      load: async () => ({
        getLlama: async () => ({
          loadModel: async () => ({
            dispose: disposeModel,
            createEmbeddingContext: async () => ({
              getEmbeddingFor: async () => ({ vector: [1] }),
              dispose: disposeContext,
            }),
          }),
        }),
      }),
    });

    await e.embed(["a"]);
    await e.dispose();
    await e.dispose();
    expect(disposeContext).toHaveBeenCalledTimes(1);
    expect(disposeModel).toHaveBeenCalledTimes(1);
  });
});

describe("TransformersEmbedder", () => {
  it("requires a model, as a TypeError", () => {
    expect(() => new TransformersEmbedder({ model: "" })).toThrow(TypeError);
    expect(() => new TransformersEmbedder({} as never)).toThrow(TypeError);
  });

  it("does not load the peer until embed() is called", async () => {
    const load = vi.fn(moduleNotFound);
    const e = new TransformersEmbedder({ model: "Xenova/all-MiniLM-L6-v2", load });
    expect(load).not.toHaveBeenCalled();
    expect(await e.embed([])).toEqual([]);
    expect(load).not.toHaveBeenCalled();
  });

  it("reports a missing peer with an install hint", async () => {
    const e = new TransformersEmbedder({ model: "m", load: moduleNotFound });
    const error = await e.embed(["x"]).catch((err: unknown) => err);
    expect(isEmbedderUnavailable(error)).toBe(true);
    expect((error as EmbedderUnavailableError).embedder).toBe("transformers");
    expect((error as Error).message).toBe(
      "transformers requires the optional peer @huggingface/transformers: npm i @huggingface/transformers",
    );
  });

  it("builds a feature-extraction pipeline and mean-pools by default", async () => {
    const extractor = vi.fn(async () => ({ data: [1, 2, 3, 4], dims: [2, 2] }));
    const pipeline = vi.fn(async () => extractor);
    const e = new TransformersEmbedder({ model: "Xenova/all-MiniLM-L6-v2", load: async () => ({ pipeline }) });

    const vectors = await e.embed(["a", "b"]);

    expect(pipeline).toHaveBeenCalledWith("feature-extraction", "Xenova/all-MiniLM-L6-v2", {});
    expect(extractor).toHaveBeenCalledWith(["a", "b"], DEFAULT_EXTRACT_OPTIONS);
    expect(DEFAULT_EXTRACT_OPTIONS).toEqual({ pooling: "mean", normalize: true });
    expect(Array.from(vectors[0] as Float32Array)).toEqual([1, 2]);
    expect(Array.from(vectors[1] as Float32Array)).toEqual([3, 4]);
  });

  it("lets the caller override the extract options", async () => {
    const extractor = vi.fn(async () => ({ data: [1, 2], dims: [1, 2] }));
    const e = new TransformersEmbedder({
      model: "m",
      extractOptions: { pooling: "cls" },
      load: async () => ({ pipeline: async () => extractor }),
    });
    await e.embed(["a"]);
    expect(extractor).toHaveBeenCalledWith(["a"], { pooling: "cls", normalize: true });
  });

  it("builds the pipeline once across repeated calls", async () => {
    const pipeline = vi.fn(async () => async () => ({ data: [1], dims: [1, 1] }));
    const e = new TransformersEmbedder({ model: "m", load: async () => ({ pipeline }) });
    await Promise.all([e.embed(["a"]), e.embed(["b"])]);
    await e.embed(["c"]);
    expect(pipeline).toHaveBeenCalledTimes(1);
  });

  it("wraps a pipeline construction failure", async () => {
    const e = new TransformersEmbedder({
      model: "nope/nope",
      load: async () => ({
        pipeline: async () => {
          throw new Error("404 model not found");
        },
      }),
    });
    await expect(e.embed(["x"])).rejects.toThrow(/could not build a feature-extraction pipeline/);
  });
});

describe("splitPooledTensor", () => {
  it("slices [batch, hidden] into one vector per input", () => {
    const out = splitPooledTensor({ data: [1, 2, 3, 4, 5, 6], dims: [3, 2] }, 3);
    expect(out.map((v) => Array.from(v))).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  it("rejects an un-pooled 3-D tensor instead of slicing it wrong", () => {
    // [batch, tokens, hidden] is what you get with pooling turned off. Slicing
    // it as if it were 2-D yields vectors of the wrong dimension, and the only
    // downstream symptom would be cosine() quietly returning null forever.
    expect(() => splitPooledTensor({ data: [1, 2, 3, 4], dims: [1, 2, 2] }, 1)).toThrow(
      /expected dims \[batch, hidden\]/,
    );
    expect(() => splitPooledTensor({ data: [1, 2, 3, 4], dims: [1, 2, 2] }, 1)).toThrow(
      EmbedderUnavailableError,
    );
  });

  it("rejects a batch that does not match the input count", () => {
    expect(() => splitPooledTensor({ data: [1, 2], dims: [1, 2] }, 2)).toThrow(
      /batch 1 but 2 input\(s\)/,
    );
  });

  it("rejects a data length inconsistent with dims", () => {
    expect(() => splitPooledTensor({ data: [1, 2, 3], dims: [2, 2] }, 2)).toThrow(
      /data length 3 != 2 \* 2/,
    );
  });

  it("rejects a tensor with no data or dims", () => {
    expect(() => splitPooledTensor({} as never, 1)).toThrow(/no `data`\/`dims`/);
  });
});
