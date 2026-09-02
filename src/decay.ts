/**
 * Memory decay — a port of the origin engine's `memory_decay.py`.
 *
 * Verified against the origin engine (2026-08-27): `CATEGORY_HALF_LIFE_DAYS`,
 * `IMPORTANCE_DECAY_FACTOR` and `calculate_decay`, all in `memory_decay.py`.
 *
 * The formula, verbatim from the Python:
 *
 *   halfLife           = CATEGORY_HALF_LIFE_DAYS[category] ?? 60
 *   effectiveHalfLife  = (halfLife + accessCount * 5) / importanceFactor
 *   strength           = originalStrength * 0.5 ** (daysSinceAccess / effectiveHalfLife)
 *   floor              importance >= 0.8 -> max(strength, 0.3)
 *                      importance >= 0.6 -> max(strength, 0.1)
 *   result             = round(strength, 3)
 *
 * Note that `daysSinceCreation` is part of the signature and is deliberately
 * unused: the origin engine's `calculate_decay` takes it and then sets
 * `decay_time = days_since_access`, decaying on time since last *access*, not
 * since creation. Keeping the parameter keeps the two signatures aligned, and
 * dropping the argument silently would be a behaviour change waiting to happen.
 */

/** The origin engine's `CATEGORY_HALF_LIFE_DAYS` (memory_decay.py). Unknown category => 60. */
export const CATEGORY_HALF_LIFE_DAYS: Readonly<Record<string, number>> = {
  personal_fact: 180, // Personal facts remembered longer
  preference: 90, // Preferences fade over time
  relationship: 365, // Relationship info very persistent
  experience: 60, // Experiences fade unless reinforced
  emotion: 30, // Emotional memories consolidate or fade
  interest: 45, // Interests can shift
  work: 30, // Work details fade quickly
  health: 60, // Health info moderately persistent
};

/** Half-life for a category not in the table. The origin engine: `.get(category, 60)`. */
export const DEFAULT_HALF_LIFE_DAYS = 60;

/**
 * The origin engine's `IMPORTANCE_DECAY_FACTOR` (memory_decay.py), as ordered pairs.
 *
 * The origin engine scans `sorted(..., reverse=True)` and takes the FIRST threshold that is
 * `<= importance`, so this list is already in descending threshold order and is
 * scanned the same way. Importance below 0.2 matches nothing and the factor
 * stays at its initial `1.0`.
 */
export const IMPORTANCE_DECAY_FACTOR: ReadonlyArray<readonly [number, number]> = [
  [1.0, 0.5], // Very important: half the decay rate
  [0.8, 0.7],
  [0.6, 0.9],
  [0.4, 1.1],
  [0.2, 1.3], // Unimportant: faster decay
];

/** The origin engine: each access adds 5 days to the half-life. */
export const ACCESS_REINFORCEMENT_DAYS = 5;

/** Floors — "very important memories never fully fade" (memory_decay.py). */
export const STRENGTH_FLOOR_HIGH = 0.3; // importance >= 0.8
export const STRENGTH_FLOOR_MEDIUM = 0.1; // importance >= 0.6

export interface DecayArgs {
  originalStrength: number;
  /** Present for signature parity with the origin engine; not used by the formula. */
  daysSinceCreation: number;
  daysSinceAccess: number;
  importance: number;
  category: string;
  accessCount: number;
}

/**
 * Python's `round(x, 3)`: round-half-to-EVEN on the value's exact decimal
 * expansion, not JavaScript's round-half-away-from-zero.
 *
 * This matters. `Math.round(0.0625 * 1000) / 1000` is `0.063`; Python's
 * `round(0.0625, 3)` is `0.062`, and `0.0625` is reachable here — it is
 * `0.5 ** 4`, i.e. any memory sitting at exactly four effective half-lives with
 * `originalStrength = 1.0`. `toFixed(20)` is specified to be computed from the
 * exact value of the double, so a true tie shows up as `...5` followed by
 * zeros and anything else does not.
 */
export function roundHalfEven3(x: number): number {
  if (!Number.isFinite(x)) return x;

  const negative = x < 0;
  const digits = Math.abs(x).toFixed(20); // "d.dddddddddddddddddddd"
  const dot = digits.indexOf(".");
  const frac = digits.slice(dot + 1);

  const keep = `${digits.slice(0, dot)}${frac.slice(0, 3)}`; // scaled by 1000
  const rest = frac.slice(3);
  const first = rest.charCodeAt(0) - 48;

  let scaled = Number(keep);
  if (first > 5) {
    scaled += 1;
  } else if (first === 5) {
    const tie = /^0*$/.test(rest.slice(1));
    // Exactly halfway -> to even. Above halfway -> up.
    if (!tie || scaled % 2 === 1) scaled += 1;
  }

  const out = scaled / 1000;
  return negative ? -out : out;
}

/**
 * Current strength of a memory after decay, rounded to 3 dp.
 *
 * A modified exponential decay: the category half-life is lengthened by every
 * access (reinforcement) and by importance (an important memory decays slower),
 * then floored so that important memories never fully fade.
 */
export function calculateDecay(args: DecayArgs): number {
  const { originalStrength, daysSinceAccess, importance, category, accessCount } = args;

  const halfLife = CATEGORY_HALF_LIFE_DAYS[category] ?? DEFAULT_HALF_LIFE_DAYS;

  // Access reinforcement: each access adds to the half-life.
  let effectiveHalfLife = halfLife + accessCount * ACCESS_REINFORCEMENT_DAYS;

  // Importance factor: first threshold <= importance wins, descending scan.
  let importanceFactor = 1.0;
  for (const [threshold, factor] of IMPORTANCE_DECAY_FACTOR) {
    if (importance >= threshold) {
      importanceFactor = factor;
      break;
    }
  }

  effectiveHalfLife = effectiveHalfLife / importanceFactor;

  // Decay is measured from the last ACCESS, not from creation.
  const decayFactor = Math.pow(0.5, daysSinceAccess / effectiveHalfLife);

  let newStrength = originalStrength * decayFactor;

  if (importance >= 0.8) {
    newStrength = Math.max(newStrength, STRENGTH_FLOOR_HIGH);
  } else if (importance >= 0.6) {
    newStrength = Math.max(newStrength, STRENGTH_FLOOR_MEDIUM);
  }

  return roundHalfEven3(newStrength);
}
