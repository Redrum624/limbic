/**
 * Decay parity — limbic vs the origin engine's `memory_decay.py`.
 *
 * The `CASES` table below was NOT hand-authored. It was produced by executing
 * the origin engine's own `MemoryDecayManager.calculate_decay` at commit `ff9c407c`
 * (2026-08-27) over the argument grid each row's name describes, and
 * transcribing its return values. Regenerating it means re-running the upstream
 * Python, not editing numbers here.
 *
 * The grid covers: every category in the table plus two absent ones (the
 * `.get(category, 60)` default), the full importance ladder across all five
 * `IMPORTANCE_DECAY_FACTOR` thresholds and below the lowest one, both strength
 * floors and the values just under them, access reinforcement, original-strength
 * scaling, zero elapsed time, decay-on-access-not-creation, fractional days,
 * and a rounding tie that discriminates Python's round-half-to-even from
 * JavaScript's round-half-up.
 */

import { describe, expect, it } from "vitest";

import {
  ACCESS_REINFORCEMENT_DAYS,
  CATEGORY_HALF_LIFE_DAYS,
  DEFAULT_HALF_LIFE_DAYS,
  IMPORTANCE_DECAY_FACTOR,
  STRENGTH_FLOOR_HIGH,
  STRENGTH_FLOOR_MEDIUM,
  calculateDecay,
  roundHalfEven3,
} from "../src/decay.js";

interface DecayCase {
  name: string;
  originalStrength: number;
  daysSinceCreation: number;
  daysSinceAccess: number;
  importance: number;
  category: string;
  accessCount: number;
  expected: number;
}

const CASES: DecayCase[] = [
  { name: "personal_fact at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 180, daysSinceAccess: 180, importance: 0.5, category: "personal_fact", accessCount: 0, expected: 0.467 },
  { name: "personal_fact fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "personal_fact", accessCount: 0, expected: 1.0 },
  { name: "preference at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 90, daysSinceAccess: 90, importance: 0.5, category: "preference", accessCount: 0, expected: 0.467 },
  { name: "preference fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "preference", accessCount: 0, expected: 1.0 },
  { name: "relationship at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 365, daysSinceAccess: 365, importance: 0.5, category: "relationship", accessCount: 0, expected: 0.467 },
  { name: "relationship fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "relationship", accessCount: 0, expected: 1.0 },
  { name: "experience at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 60, daysSinceAccess: 60, importance: 0.5, category: "experience", accessCount: 0, expected: 0.467 },
  { name: "experience fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "experience", accessCount: 0, expected: 1.0 },
  { name: "emotion at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "emotion", accessCount: 0, expected: 0.467 },
  { name: "emotion fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "emotion", accessCount: 0, expected: 1.0 },
  { name: "interest at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 45, daysSinceAccess: 45, importance: 0.5, category: "interest", accessCount: 0, expected: 0.467 },
  { name: "interest fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "interest", accessCount: 0, expected: 1.0 },
  { name: "work at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.467 },
  { name: "work fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "work", accessCount: 0, expected: 1.0 },
  { name: "health at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 60, daysSinceAccess: 60, importance: 0.5, category: "health", accessCount: 0, expected: 0.467 },
  { name: "health fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "health", accessCount: 0, expected: 1.0 },
  { name: "general at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 60, daysSinceAccess: 60, importance: 0.5, category: "general", accessCount: 0, expected: 0.467 },
  { name: "general fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "general", accessCount: 0, expected: 1.0 },
  { name: "not_a_category at one nominal half-life", originalStrength: 1.0, daysSinceCreation: 60, daysSinceAccess: 60, importance: 0.5, category: "not_a_category", accessCount: 0, expected: 0.467 },
  { name: "not_a_category fresh", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 0, importance: 0.5, category: "not_a_category", accessCount: 0, expected: 1.0 },
  { name: "importance 0.0", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.0, category: "work", accessCount: 0, expected: 0.5 },
  { name: "importance 0.1", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.1, category: "work", accessCount: 0, expected: 0.5 },
  { name: "importance 0.2", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.2, category: "work", accessCount: 0, expected: 0.406 },
  { name: "importance 0.3", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.3, category: "work", accessCount: 0, expected: 0.406 },
  { name: "importance 0.4", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.4, category: "work", accessCount: 0, expected: 0.467 },
  { name: "importance 0.5", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.467 },
  { name: "importance 0.6", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.6, category: "work", accessCount: 0, expected: 0.536 },
  { name: "importance 0.7", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.7, category: "work", accessCount: 0, expected: 0.536 },
  { name: "importance 0.8", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.8, category: "work", accessCount: 0, expected: 0.616 },
  { name: "importance 0.9", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.9, category: "work", accessCount: 0, expected: 0.616 },
  { name: "importance 1.0", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 1.0, category: "work", accessCount: 0, expected: 0.707 },
  { name: "floor 0.3 for importance 0.8 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 0.8, category: "work", accessCount: 0, expected: 0.3 },
  { name: "floor 0.3 for importance 1.0 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 1.0, category: "work", accessCount: 0, expected: 0.3 },
  { name: "floor 0.1 for importance 0.6 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 0.6, category: "work", accessCount: 0, expected: 0.1 },
  { name: "floor 0.1 for importance 0.7 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 0.7, category: "work", accessCount: 0, expected: 0.1 },
  { name: "no floor for importance 0.5 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 0.5, category: "work", accessCount: 0, expected: 0.0 },
  { name: "no floor for importance 0.59 after 10 years", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 3650, importance: 0.59, category: "work", accessCount: 0, expected: 0.0 },
  { name: "work, 30 days, 0 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.467 },
  { name: "work, 30 days, 1 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 1, expected: 0.52 },
  { name: "work, 30 days, 2 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 2, expected: 0.564 },
  { name: "work, 30 days, 5 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 5, expected: 0.66 },
  { name: "work, 30 days, 10 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 10, expected: 0.751 },
  { name: "work, 30 days, 50 accesses", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 50, expected: 0.922 },
  { name: "originalStrength 1.0", originalStrength: 1.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.467 },
  { name: "originalStrength 0.8", originalStrength: 0.8, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.373 },
  { name: "originalStrength 0.5", originalStrength: 0.5, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.233 },
  { name: "originalStrength 0.25", originalStrength: 0.25, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.117 },
  { name: "originalStrength 0.0", originalStrength: 0.0, daysSinceCreation: 30, daysSinceAccess: 30, importance: 0.5, category: "work", accessCount: 0, expected: 0.0 },
  { name: "zero days since access", originalStrength: 1.0, daysSinceCreation: 100, daysSinceAccess: 0, importance: 0.5, category: "work", accessCount: 0, expected: 1.0 },
  { name: "old but freshly accessed", originalStrength: 1.0, daysSinceCreation: 3650, daysSinceAccess: 1, importance: 0.5, category: "work", accessCount: 0, expected: 0.975 },
  { name: "fractional 3.5 days", originalStrength: 1.0, daysSinceCreation: 3.5, daysSinceAccess: 3.5, importance: 0.5, category: "emotion", accessCount: 0, expected: 0.915 },
  { name: "banker's-rounding tie: exactly four half-lives, strength 0.0625", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 120, importance: 0.1, category: "work", accessCount: 0, expected: 0.062 },
  { name: "just below a tie: exactly five half-lives, strength 0.03125", originalStrength: 1.0, daysSinceCreation: 0, daysSinceAccess: 150, importance: 0.1, category: "work", accessCount: 0, expected: 0.031 },
];

describe("calculateDecay: parity with the origin engine's memory_decay.py", () => {
  it("covers every category in the table", () => {
    for (const category of Object.keys(CATEGORY_HALF_LIFE_DAYS)) {
      expect(
        CASES.some((c) => c.category === category),
        `no case exercises category ${category}`,
      ).toBe(true);
    }
  });

  for (const c of CASES) {
    it(`${c.name} => ${c.expected}`, () => {
      expect(
        calculateDecay({
          originalStrength: c.originalStrength,
          daysSinceCreation: c.daysSinceCreation,
          daysSinceAccess: c.daysSinceAccess,
          importance: c.importance,
          category: c.category,
          accessCount: c.accessCount,
        }),
      ).toBe(c.expected);
    });
  }
});

describe("the ported tables", () => {
  it("CATEGORY_HALF_LIFE_DAYS is the origin engine's table verbatim", () => {
    expect(CATEGORY_HALF_LIFE_DAYS).toEqual({
      personal_fact: 180,
      preference: 90,
      relationship: 365,
      experience: 60,
      emotion: 30,
      interest: 45,
      work: 30,
      health: 60,
    });
  });

  it("IMPORTANCE_DECAY_FACTOR is the origin engine's table, already in descending scan order", () => {
    expect(IMPORTANCE_DECAY_FACTOR).toEqual([
      [1.0, 0.5],
      [0.8, 0.7],
      [0.6, 0.9],
      [0.4, 1.1],
      [0.2, 1.3],
    ]);
    const thresholds = IMPORTANCE_DECAY_FACTOR.map(([t]) => t);
    expect(thresholds).toEqual([...thresholds].sort((a, b) => b - a));
  });

  it("an unknown category falls back to 60 days", () => {
    expect(DEFAULT_HALF_LIFE_DAYS).toBe(60);
    // `general` is a real limbic category with no entry in the origin engine's table.
    expect(CATEGORY_HALF_LIFE_DAYS["general"]).toBeUndefined();
    expect(
      calculateDecay({
        originalStrength: 1,
        daysSinceCreation: 0,
        daysSinceAccess: 60,
        importance: 0.1,
        category: "general",
        accessCount: 0,
      }),
    ).toBe(
      calculateDecay({
        originalStrength: 1,
        daysSinceCreation: 0,
        daysSinceAccess: 60,
        importance: 0.1,
        category: "experience", // half-life 60, same as the default
        accessCount: 0,
      }),
    );
  });
});

describe("behavioural properties", () => {
  const base = {
    originalStrength: 1,
    daysSinceCreation: 0,
    daysSinceAccess: 30,
    importance: 0.5,
    category: "work",
    accessCount: 0,
  };

  it("decays on time since ACCESS, not since creation", () => {
    const freshlyAccessed = calculateDecay({
      ...base,
      daysSinceCreation: 3650,
      daysSinceAccess: 1,
    });
    const stale = calculateDecay({ ...base, daysSinceCreation: 1, daysSinceAccess: 3650 });
    expect(freshlyAccessed).toBeGreaterThan(stale);
    // daysSinceCreation is inert: varying it alone changes nothing.
    expect(calculateDecay({ ...base, daysSinceCreation: 0 })).toBe(
      calculateDecay({ ...base, daysSinceCreation: 100000 }),
    );
  });

  it("each access lengthens the half-life by 5 days", () => {
    expect(ACCESS_REINFORCEMENT_DAYS).toBe(5);
    let previous = -1;
    for (const accessCount of [0, 1, 2, 5, 10, 50]) {
      const strength = calculateDecay({ ...base, accessCount });
      expect(strength).toBeGreaterThan(previous);
      previous = strength;
    }
  });

  it("is monotonically non-increasing in days since access", () => {
    let previous = Number.POSITIVE_INFINITY;
    for (let days = 0; days <= 200; days += 5) {
      const strength = calculateDecay({ ...base, daysSinceAccess: days });
      expect(strength).toBeLessThanOrEqual(previous);
      previous = strength;
    }
  });

  it("scales linearly with originalStrength before rounding", () => {
    expect(calculateDecay({ ...base, originalStrength: 0 })).toBe(0);
    const full = calculateDecay({ ...base, originalStrength: 1 });
    const half = calculateDecay({ ...base, originalStrength: 0.5 });
    expect(Math.abs(half - full / 2)).toBeLessThanOrEqual(0.001);
  });

  it("returns originalStrength unchanged at zero elapsed days", () => {
    expect(calculateDecay({ ...base, daysSinceAccess: 0 })).toBe(1);
    expect(calculateDecay({ ...base, daysSinceAccess: 0, originalStrength: 0.42 })).toBe(0.42);
  });

  it("floors importance >= 0.8 at 0.3 and importance >= 0.6 at 0.1", () => {
    expect(STRENGTH_FLOOR_HIGH).toBe(0.3);
    expect(STRENGTH_FLOOR_MEDIUM).toBe(0.1);
    const ancient = { ...base, daysSinceAccess: 36500 };
    expect(calculateDecay({ ...ancient, importance: 1.0 })).toBe(0.3);
    expect(calculateDecay({ ...ancient, importance: 0.8 })).toBe(0.3);
    expect(calculateDecay({ ...ancient, importance: 0.7 })).toBe(0.1);
    expect(calculateDecay({ ...ancient, importance: 0.6 })).toBe(0.1);
    // Just under the floor thresholds there is no floor at all.
    expect(calculateDecay({ ...ancient, importance: 0.59 })).toBe(0);
    expect(calculateDecay({ ...ancient, importance: 0.0 })).toBe(0);
  });

  it("importance slows decay across the ladder", () => {
    const strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((importance) =>
      calculateDecay({ ...base, daysSinceAccess: 90, importance }),
    );
    // 0.0 and 0.2 straddle the factor 1.0 / 1.3 boundary — importance below the
    // lowest threshold keeps the initial factor of 1.0 and so decays SLOWER
    // than importance 0.2, which is the origin engine's behaviour, not a port bug. From 0.2
    // upward the ladder is strictly increasing.
    const ladder = strengths.slice(1);
    for (let i = 1; i < ladder.length; i++) {
      expect(ladder[i] as number).toBeGreaterThan(ladder[i - 1] as number);
    }
  });
});

/**
 * `roundHalfEven3` was additionally validated against CPython's `round(x, 3)`
 * over a 4,335-value sweep (4,000 uniform randoms plus every i/16, i/32 and
 * i/64 in [0, 1], every 0.0005-offset midpoint below 0.2, and 0.5**k for
 * k in 0..19) — zero disagreements. The sweep needs CPython to produce its
 * truth values, so it is not committed; the dyadic ties below are the ones
 * that actually discriminate the two rounding modes.
 */
describe("roundHalfEven3: Python's round(x, 3), not JavaScript's", () => {
  it("rounds an exact tie to even, where Math.round would go up", () => {
    // Every value here is a dyadic rational, so it IS exactly halfway at 3 dp.
    // Expected values are CPython's round(x, 3).
    expect(roundHalfEven3(0.0625)).toBe(0.062);
    expect(Math.round(0.0625 * 1000) / 1000).toBe(0.063); // the wrong answer
    expect(roundHalfEven3(0.1875)).toBe(0.188); // tie up, to the even 188
    expect(roundHalfEven3(0.3125)).toBe(0.312);
    expect(roundHalfEven3(0.4375)).toBe(0.438);
    expect(roundHalfEven3(0.5625)).toBe(0.562);
    expect(roundHalfEven3(0.8125)).toBe(0.812);
    expect(roundHalfEven3(0.9375)).toBe(0.938);
  });

  it("does NOT treat a decimal-looking near-tie as a tie", () => {
    // 0.0125 and 0.0375 are not dyadic, so neither is exactly halfway: the
    // nearest double to 0.0125 sits just ABOVE it and the one to 0.0375 just
    // BELOW, and CPython rounds them 0.013 and 0.037 accordingly. A naive
    // "round half even on the printed decimal" would get both wrong.
    expect(roundHalfEven3(0.0125)).toBe(0.013);
    expect(roundHalfEven3(0.0375)).toBe(0.037);
  });

  it("rounds ordinary values the ordinary way", () => {
    expect(roundHalfEven3(0.4674)).toBe(0.467);
    expect(roundHalfEven3(0.4676)).toBe(0.468);
    expect(roundHalfEven3(1)).toBe(1);
    expect(roundHalfEven3(0)).toBe(0);
    expect(roundHalfEven3(0.9999)).toBe(1);
  });

  it("is sign-symmetric and passes non-finite values through", () => {
    expect(roundHalfEven3(-0.0625)).toBe(-0.062);
    expect(roundHalfEven3(-0.4676)).toBe(-0.468);
    expect(Number.isNaN(roundHalfEven3(Number.NaN))).toBe(true);
    expect(roundHalfEven3(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
  });
});
