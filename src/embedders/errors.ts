/**
 * The one error type every `Embedder` throws.
 *
 * The origin engine's rule, which limbic keeps: an embedding failure is never fatal to a
 * turn. The retrieval pipeline catches this and degrades to keyword-only
 * scoring — the cosine channel simply goes MISSING, which is exactly the case
 * `scoreMemory` already handles. So callers need one type to catch, and it must
 * be distinguishable from a programming error (a `TypeError` from bad
 * arguments) that should NOT be swallowed.
 */
export class EmbedderUnavailableError extends Error {
  override readonly name = "EmbedderUnavailableError";

  /** Which adapter failed: "ollama", "node-llama-cpp", "transformers". */
  readonly embedder: string;

  constructor(embedder: string, message: string, options?: { cause?: unknown }) {
    super(message, options as ErrorOptions);
    this.embedder = embedder;
  }
}

/** True for an `EmbedderUnavailableError` from any realm. */
export function isEmbedderUnavailable(e: unknown): e is EmbedderUnavailableError {
  return e instanceof Error && e.name === "EmbedderUnavailableError";
}

/**
 * The install hint an optional peer raises when it is not present. The peers are
 * `peerDependenciesMeta.optional`, so a plain `ERR_MODULE_NOT_FOUND` is what a
 * user sees otherwise, and it does not say what to install.
 */
export function missingPeer(
  embedder: string,
  pkg: string,
  cause: unknown,
): EmbedderUnavailableError {
  return new EmbedderUnavailableError(
    embedder,
    `${embedder} requires the optional peer ${pkg}: npm i ${pkg}`,
    { cause },
  );
}
