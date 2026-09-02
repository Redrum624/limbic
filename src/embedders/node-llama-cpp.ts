/**
 * `node-llama-cpp` embedder — a fully offline, in-process GGUF option.
 *
 * Wraps the documented embedding flow
 * (https://node-llama-cpp.withcat.ai/guide/embedding):
 *
 *   getLlama() -> llama.loadModel({ modelPath }) -> model.createEmbeddingContext()
 *              -> context.getEmbeddingFor(text) -> { vector: readonly number[] }
 *
 * `node-llama-cpp` is a `peerDependenciesMeta.optional` peer, so it is reached
 * only through a dynamic `import()` and its absence is reported with an install
 * hint instead of a bare `ERR_MODULE_NOT_FOUND`. limbic's core keeps zero
 * required runtime dependencies.
 *
 * The context is created lazily on the first `embed()` and reused afterwards;
 * `dispose()` releases it and the model. A caller that never embeds never loads
 * a model.
 */

import type { Embedder } from "../types.js";
import { EmbedderUnavailableError, missingPeer } from "./errors.js";

const ADAPTER = "node-llama-cpp";
const PACKAGE = "node-llama-cpp";

interface EmbeddingContextLike {
  getEmbeddingFor(text: string): Promise<{ vector: readonly number[] }>;
  dispose?: () => Promise<void> | void;
}

interface ModelLike {
  createEmbeddingContext(options?: Record<string, unknown>): Promise<EmbeddingContextLike>;
  dispose?: () => Promise<void> | void;
}

interface LlamaLike {
  loadModel(options: { modelPath: string }): Promise<ModelLike>;
}

interface NodeLlamaCppModule {
  getLlama: (options?: Record<string, unknown>) => Promise<LlamaLike>;
}

export interface NodeLlamaCppEmbedderOptions {
  /** Absolute path to the GGUF embedding model. */
  modelPath: string;
  /** Reported as `Embedder.model`. Defaults to the model file's basename. */
  model?: string;
  /** Passed straight through to `createEmbeddingContext`. */
  contextOptions?: Record<string, unknown>;
  /** Injectable for tests — defaults to `import("node-llama-cpp")`. */
  load?: () => Promise<unknown>;
}

function basename(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

export class NodeLlamaCppEmbedder implements Embedder {
  readonly model: string;
  readonly modelPath: string;

  readonly #load: () => Promise<unknown>;
  readonly #contextOptions: Record<string, unknown>;
  #context: EmbeddingContextLike | undefined;
  #modelHandle: ModelLike | undefined;
  #pending: Promise<EmbeddingContextLike> | undefined;
  // Bumped by dispose(). A load that started under an older epoch must not
  // install its context: dispose() already released the model it belongs to.
  #epoch = 0;

  constructor(options: NodeLlamaCppEmbedderOptions) {
    if (!options || typeof options.modelPath !== "string" || options.modelPath.length === 0) {
      throw new TypeError("NodeLlamaCppEmbedder requires a non-empty `modelPath`");
    }
    this.modelPath = options.modelPath;
    this.model = options.model ?? basename(options.modelPath);
    this.#contextOptions = options.contextOptions ?? {};
    this.#load = options.load ?? (() => import(PACKAGE));
  }

  async #ready(): Promise<EmbeddingContextLike> {
    if (this.#context) return this.#context;
    // Concurrent embed() calls must not each load a model.
    this.#pending ??= this.#open();
    try {
      this.#context = await this.#pending;
      return this.#context;
    } finally {
      this.#pending = undefined;
    }
  }

  async #open(): Promise<EmbeddingContextLike> {
    let mod: NodeLlamaCppModule;
    try {
      mod = (await this.#load()) as NodeLlamaCppModule;
    } catch (cause) {
      throw missingPeer(ADAPTER, PACKAGE, cause);
    }

    if (!mod || typeof mod.getLlama !== "function") {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `${PACKAGE} loaded but exports no getLlama() — expected the v3 API (npm i ${PACKAGE}@^3)`,
      );
    }

    const epoch = this.#epoch;
    let context: EmbeddingContextLike;
    try {
      const llama = await mod.getLlama();
      const model = await llama.loadModel({ modelPath: this.modelPath });
      this.#modelHandle = model;
      context = await model.createEmbeddingContext(this.#contextOptions);
    } catch (cause) {
      // A model loaded by this failed attempt must not outlive it: the next
      // embed() re-runs #open(), and a kept handle would mean a second full
      // native model load stacked on the first, once per retry.
      const model = this.#modelHandle;
      this.#modelHandle = undefined;
      try {
        await model?.dispose?.();
      } catch {
        // The load failure is the error worth reporting, not the cleanup's.
      }
      throw new EmbedderUnavailableError(
        ADAPTER,
        `could not load the embedding model at ${this.modelPath}: ${
          cause instanceof Error ? cause.message : String(cause)
        }`,
        { cause },
      );
    }

    if (this.#epoch !== epoch) {
      // dispose() ran while the load was in flight and has already released
      // the model this context sits on; keeping the context would leak it and
      // point it at freed native state. Release it and fail the embed loudly.
      try {
        await context.dispose?.();
      } catch {
        // The disposal race is the error worth reporting, not the cleanup's.
      }
      throw new EmbedderUnavailableError(
        ADAPTER,
        "disposed while the embedding model was loading — call embed() again to reload",
      );
    }
    return context;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!Array.isArray(texts)) {
      throw new TypeError("NodeLlamaCppEmbedder.embed expects an array of strings");
    }
    if (texts.length === 0) return [];

    const context = await this.#ready();
    const out: Float32Array[] = [];
    // getEmbeddingFor takes one text; there is no batch entry point in v3.
    for (const text of texts) {
      try {
        const { vector } = await context.getEmbeddingFor(text);
        out.push(Float32Array.from(vector));
      } catch (cause) {
        throw new EmbedderUnavailableError(
          ADAPTER,
          `getEmbeddingFor failed: ${cause instanceof Error ? cause.message : String(cause)}`,
          { cause },
        );
      }
    }
    return out;
  }

  /**
   * Release the embedding context and the model. Safe to call twice, and safe
   * while a load is in flight: that load sees the epoch change, releases the
   * context it produced and rejects instead of resurrecting it.
   */
  async dispose(): Promise<void> {
    this.#epoch += 1;
    const context = this.#context;
    const model = this.#modelHandle;
    this.#context = undefined;
    this.#modelHandle = undefined;
    await context?.dispose?.();
    await model?.dispose?.();
  }
}
