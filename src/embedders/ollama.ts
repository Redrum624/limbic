/**
 * Ollama embedder — the default, and the only network call limbic ever makes.
 *
 * Endpoint parity with the origin engine `server/services/ollama_client.py` `embeddings()`
 * (L238-261, verified at the origin engine commit `ff9c407c`):
 *
 *   POST {host}/api/embed
 *   body {"model": ..., "input": ...}
 *   read json["embeddings"]  ->  list[list[float]]
 *
 * The legacy `POST /api/embeddings` with `{"model", "prompt"}` is deliberately
 * unsupported here, exactly as it is there. `input` accepts a string or a list;
 * the origin engine's signature takes one string, limbic always sends the array so a pool of
 * texts costs ONE round trip.
 *
 * ⚠️ The default host is `127.0.0.1`, not `localhost`. On a dual-stack Windows
 * box `localhost` resolves to `::1` first, Ollama listens on IPv4 only, and the
 * connection eats a full DNS/connect fallback before it succeeds — measured at
 * roughly 2.1 s per request on this machine versus a few ms for `127.0.0.1`.
 * That is a resolver artifact, not an Ollama one, but the default should not
 * cost a caller two seconds a turn. Pass `host` explicitly to override.
 */

import type { Embedder } from "../types.js";
import { EmbedderUnavailableError } from "./errors.js";

export type FetchLike = (
  input: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
  },
) => Promise<{
  ok: boolean;
  status: number;
  statusText?: string;
  text(): Promise<string>;
  json(): Promise<unknown>;
}>;

export interface OllamaEmbedderOptions {
  /** Default `http://127.0.0.1:11434` — see the note above about `localhost`. */
  host?: string;
  /** e.g. `"nomic-embed-text"`. Required: there is no sensible default model. */
  model: string;
  /** Per-request timeout. Default 30 s. */
  timeoutMs?: number;
  /** Injectable for tests; defaults to the global `fetch` (Node >= 20). */
  fetch?: FetchLike;
}

export const DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";
export const DEFAULT_TIMEOUT_MS = 30_000;

const ADAPTER = "ollama";

function stripTrailingSlash(host: string): string {
  return host.endsWith("/") ? host.slice(0, -1) : host;
}

export class OllamaEmbedder implements Embedder {
  readonly model: string;
  readonly host: string;
  readonly timeoutMs: number;

  readonly #fetch: FetchLike;

  constructor(options: OllamaEmbedderOptions) {
    if (!options || typeof options.model !== "string" || options.model.length === 0) {
      // A programming error, not an availability problem: do NOT dress it up as
      // an EmbedderUnavailableError that the pipeline would silently swallow.
      throw new TypeError("OllamaEmbedder requires a non-empty `model`");
    }
    this.model = options.model;
    this.host = stripTrailingSlash(options.host ?? DEFAULT_OLLAMA_HOST);
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

    const impl = options.fetch ?? (globalThis.fetch as unknown as FetchLike | undefined);
    if (typeof impl !== "function") {
      throw new TypeError(
        "OllamaEmbedder needs a global fetch (Node >= 20) or an injected `fetch`",
      );
    }
    this.#fetch = impl;
  }

  /** The endpoint this instance posts to — handy in error messages and tests. */
  get endpoint(): string {
    return `${this.host}/api/embed`;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!Array.isArray(texts)) {
      throw new TypeError("OllamaEmbedder.embed expects an array of strings");
    }
    // No texts means no round trip. An empty `input` array is also what Ollama
    // would reject, so this is both faster and kinder.
    if (texts.length === 0) return [];

    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort();
    }, this.timeoutMs);
    // Node keeps the event loop alive for a pending timer; this one must not.
    (timer as unknown as { unref?: () => void }).unref?.();

    let response: Awaited<ReturnType<FetchLike>>;
    try {
      response = await this.#fetch(this.endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: this.model, input: texts }),
        signal: controller.signal,
      });
    } catch (cause) {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `Ollama unreachable at ${this.endpoint}: ${describe(cause)}`,
        { cause },
      );
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.text()).slice(0, 200);
      } catch {
        // A body we cannot read is not more interesting than the status.
      }
      throw new EmbedderUnavailableError(
        ADAPTER,
        `Ollama returned ${response.status}${
          response.statusText ? ` ${response.statusText}` : ""
        } from ${this.endpoint}${detail ? `: ${detail}` : ""}`,
      );
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch (cause) {
      throw new EmbedderUnavailableError(
        ADAPTER,
        `Ollama returned a non-JSON body from ${this.endpoint}: ${describe(cause)}`,
        { cause },
      );
    }

    return parseEmbeddings(payload, texts.length, this.endpoint);
  }
}

function describe(e: unknown): string {
  if (e instanceof Error) {
    return e.name === "AbortError" ? "request timed out" : `${e.name}: ${e.message}`;
  }
  return String(e);
}

/**
 * `json.embeddings` -> `Float32Array[]`, with every shape complaint turned into
 * an `EmbedderUnavailableError` rather than a `TypeError` deep in a loop.
 */
export function parseEmbeddings(
  payload: unknown,
  expected: number,
  endpoint: string,
): Float32Array[] {
  const fail = (why: string): never => {
    throw new EmbedderUnavailableError(
      ADAPTER,
      `Ollama returned an unusable body from ${endpoint}: ${why}`,
    );
  };

  if (typeof payload !== "object" || payload === null) {
    return fail(`expected an object, got ${payload === null ? "null" : typeof payload}`);
  }
  const raw = (payload as { embeddings?: unknown }).embeddings;
  if (!Array.isArray(raw)) {
    return fail(
      raw === undefined
        ? "no `embeddings` field (the legacy /api/embeddings endpoint returns `embedding`, singular — limbic posts to /api/embed on purpose)"
        : "`embeddings` is not an array",
    );
  }
  if (raw.length !== expected) {
    return fail(`asked for ${expected} vector(s), got ${raw.length}`);
  }

  const out: Float32Array[] = [];
  for (let i = 0; i < raw.length; i++) {
    const vector = raw[i];
    if (!Array.isArray(vector) || vector.length === 0) {
      return fail(`embeddings[${i}] is not a non-empty array`);
    }
    const typed = new Float32Array(vector.length);
    for (let j = 0; j < vector.length; j++) {
      const component = vector[j];
      if (typeof component !== "number" || !Number.isFinite(component)) {
        return fail(`embeddings[${i}][${j}] is not a finite number`);
      }
      typed[j] = component;
    }
    out.push(typed);
  }
  return out;
}
