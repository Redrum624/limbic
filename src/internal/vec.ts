/**
 * Vector operations — a port of the origin engine `server/memory/vectorops.py`.
 *
 * The one semantic that matters for parity: `cosine` returns `null`, not `0`,
 * for every case where a similarity does not exist (either side missing or
 * empty, mismatched dimensions, a zero or non-finite norm, non-finite
 * components). `null` means "these two cannot be compared" and falls back to
 * the unblended base score; a returned `0` would be a real similarity of zero
 * and would cost the memory 30% of its score. the origin engine's docstring calls this out
 * explicitly and `golden-scoring.json` pins it:
 *
 *     "cosine_is_none": "base - 'cannot be compared' is the MISSING case,
 *                        never a similarity of 0."
 */

export type Vector = ArrayLike<number>;

/**
 * Cosine similarity in [-1, 1], or `null` when undefined. Never throws.
 *
 * Accumulates in float64 — matching the origin engine's `cosine`, which uses numpy float64
 * (or `math.fsum`) precisely because there is no external reference
 * implementation to match bit-for-bit here, so plain accuracy is the target.
 */
export function cosine(
  a: Vector | null | undefined,
  b: Vector | null | undefined,
): number | null {
  if (a == null || b == null) return null;
  const n = a.length;
  if (n === 0 || n !== b.length) return null;

  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }

  // NaN / Infinity components propagate into these three sums, so one check
  // covers every non-finite input as well as an overflowing square.
  if (!Number.isFinite(dot) || !Number.isFinite(na) || !Number.isFinite(nb)) {
    return null;
  }

  const normA = Math.sqrt(na);
  const normB = Math.sqrt(nb);
  if (normA === 0 || normB === 0) return null;

  // Divide twice rather than by (normA * normB): for very small vectors the
  // product can land in the subnormal range and lose bits before the division.
  const value = dot / normA / normB;
  return Number.isFinite(value) ? value : null;
}

/**
 * L2-normalise into a new `Float32Array`, or `null` when the norm is zero or
 * the input is non-finite / empty — the same "no usable answer" sentinel.
 */
export function l2normalize(v: Vector | null | undefined): Float32Array | null {
  if (v == null) return null;
  const n = v.length;
  if (n === 0) return null;

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = v[i] as number;
    if (!Number.isFinite(x)) return null;
    sum += x * x;
  }
  if (!Number.isFinite(sum) || sum === 0) return null;

  const norm = Math.sqrt(sum);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (v[i] as number) / norm;
  return out;
}
