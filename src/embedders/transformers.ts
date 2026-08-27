/**
 * `@huggingface/transformers` embedder — ONNX Runtime, in-process, no server.
 *
 * Wraps the feature-extraction pipeline:
 *
 *   pipeline("feature-extraction", model) -> extractor(texts, opts) -> Tensor
 *
 * The tensor carries `data` (a flat `Float32Array`) and `dims`. With
 * `{ pooling: "mean", normalize: true }` the result is `[batch, hidden]`, which
 * is the shape limbic slices back into one vector per input. Mean pooling and
 * L2 normalisation are the defaults here because a raw feature-extraction call
 * returns per-TOKEN vectors (`[batch, tokens, hidden]`) — useless as a memory
 * embedding, and a silent dimension mismatch downstream if handed through.
 *
 * `@huggingface/transformers` is a `peerDependenciesMeta.optional` peer, loaded
 * only through a dynamic `import()` so limbic's core keeps zero required
 * runtime dependencies. Its absence gets an install hint, not a bare
 * `ERR_MODULE_NOT_FOUND`.
 */

import type { Embedder } from "../types.js";
import { EmbedderUnavailableError, missingPeer } from "./errors.js";

const ADAPTER = "transformers";
const PACKAGE = "@huggingface/transformers";

interface TensorLike {
  data: ArrayLike<number>;
  dims: number[];
}

type ExtractorLike = (
  texts: string[],
  options?: Record<string, unknown>,
) => Promise<TensorLike>;

interface TransformersModule {
  pipeline: (
    task: string,
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<ExtractorLike>;
}

export interface TransformersEmbedderOptions {
  /** e.g. `"Xenova/all-MiniLM-L6-v2"`. */
  model: string;
  /** Passed to `pipeline()` — `{ dtype, device, local_files_only, ... }`. */
  pipelineOptions?: Record<string, unknown>;
  /** Merged over `{ pooling: "mean", normalize: true }`. */
  extractOptions?: Record<string, unknown>;
  /** Injectable for tests — defaults to `import("@huggingface/transformers")`. */
  load?: () => Promise<unknown>;
}

export const DEFAULT_EXTRACT_OPTIONS: Readonly<Record<string, unknown>> = {
  pooling: "mean",
  normalize: true,
};

export class TransformersEmbedder implements Embedder {
  readonly model: string;

  readonly #load: () => Promise<unknown>;
  readonly #pipelineOptions: Record<string, unknown>;
  readonly #extractOptions: Record<string, unknown>;
  #extractor: ExtractorLike | undefined;
  #pending: Promise<ExtractorLike> | undefined;

  constructor(options: TransformersEmbedderOptions) {
    if (!options || typeof options.model !== "string" || options.model.length === 0) {
      throw new TypeError("TransformersEmbedder requires a non-empty `model`");
    }
    this.model = options.model;
    this.#pipelineOptions = options.pipelineOptions ?? {};
    this.#extractOptions = { ...DEFAULT_EXTRACT_OPTIONS, ...(options.extractOptions ?? {}) };
    this.#load = options.load ?? (() => import(PACKAGE));
  }

  async #ready(): Promise<ExtractorLike> {
    if (this.#extractor) return this.#extractor;
    this.#pending ??= this.#open();
    try {
      this.#extractor = await this.#pending;
      return this.#extractor;
    } finally {
      this.#pending = undefined;
    }
  }

  async #open(): Promise<ExtractorLike> {
    let mod: TransformersModule;
    try {
      mod = (await this.#load()) as TransformersModule;
    } catch (cause) {
      throw missingPeer(ADAPTER, PACKAGE, cause);
    }

    if (!mod || typeof mod.pipeline !== "function") {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `${PACKAGE} loaded but exports no pipeline() (npm i ${PACKAGE})`,
      );
    }

    try {
      return await mod.pipeline("feature-extraction", this.model, this.#pipelineOptions);
    } catch (cause) {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `could not build a feature-extraction pipeline for ${this.model}: ${
          cause instanceof Error ? cause.message : String(cause)
        }`,
        { cause },
      );
    }
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!Array.isArray(texts)) {
      throw new TypeError("TransformersEmbedder.embed expects an array of strings");
    }
    if (texts.length === 0) return [];

    const extractor = await this.#ready();

    let tensor: TensorLike;
    try {
      tensor = await extractor(texts, this.#extractOptions);
    } catch (cause) {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `feature extraction failed: ${cause instanceof Error ? cause.message : String(cause)}`,
        { cause },
      );
    }

    return splitPooledTensor(tensor, texts.length);
  }
}

/**
 * A `[batch, hidden]` tensor -> one `Float32Array` per input.
 *
 * Exported because every failure here is a silent-wrong-answer risk: a
 * `[batch, tokens, hidden]` tensor (pooling turned off) would otherwise be
 * sliced into vectors of the wrong length, and nothing downstream would notice
 * until the cosine channel started returning `null` for mismatched dimensions.
 */
export function splitPooledTensor(tensor: TensorLike, expected: number): Float32Array[] {
  const fail = (why: string): never => {
    throw new EmbedderUnavailableError(
      ADAPTER,
      `feature extraction returned an unusable tensor: ${why}`,
    );
  };

  if (!tensor || !tensor.data || !Array.isArray(tensor.dims)) {
    return fail("no `data`/`dims`");
  }
  if (tensor.dims.length !== 2) {
    return fail(
      `expected dims [batch, hidden], got [${tensor.dims.join(", ")}] — ` +
        "this is what an un-pooled feature-extraction call looks like; keep `pooling: \"mean\"`",
    );
  }

  const batch = tensor.dims[0] as number;
  const hidden = tensor.dims[1] as number;
  if (batch !== expected) return fail(`batch ${batch} but ${expected} input(s)`);
  if (!Number.isInteger(hidden) || hidden <= 0) return fail(`hidden size ${hidden}`);
  if (tensor.data.length !== batch * hidden) {
    return fail(`data length ${tensor.data.length} != ${batch} * ${hidden}`);
  }

  const out: Float32Array[] = [];
  for (let i = 0; i < batch; i++) {
    const vector = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) {
      vector[j] = tensor.data[i * hidden + j] as number;
    }
    out.push(vector);
  }
  return out;
}
