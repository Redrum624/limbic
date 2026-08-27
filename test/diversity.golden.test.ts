/**
 * Cross-language conformance: limbic's TypeScript GIST against divsel's Rust
 * reference, through the committed golden fixture.
 *
 * ## Provenance
 *
 * | | |
 * |---|---|
 * | fixture | `C:\Dev\divsel\test-assets\golden-selection.json`, copied byte-for-byte |
 * | fixture commit | `d0f8ac8` — `test(golden): complete the CONFORMANCE.md contract, add cases 21-22` (2026-08-25) |
 * | fixture `upstream_sha256` | `73713cd2d7ec9bb23659a9ee05235600e3fa358b318f2095c3eb80eab2e458a7` |
 * | schema / generator | `1` / `divsel 0.1.0`, 22 cases |
 * | **rules** | `docs/CONFORMANCE.md` at divsel commit **`926237566b1ca6919be239b9ce1bda3f7fb71c76`** — `fix(conformance): a claim the measurement disproved, and the field the fix missed` (2026-08-26) |
 * | rules `conformance_sha256` | `829bc08705ad13377624d08616254b1cd21b780f60c82b23e3a1e627e77be24b` (the blob at that commit; recorded, not re-hashed here — limbic has no divsel checkout to read) |
 *
 * divsel rewrote its history on 2026-08-26 (`d022aef`), so both hashes above
 * name objects that are still reachable but no longer sit on `main` under those
 * abbreviations. The fixture bytes are what the contract is; they are pinned by
 * sha256 here and re-hashed on every run, so a drifting copy fails mechanically
 * rather than silently.
 *
 * ## The rules this file implements
 *
 * `docs/CONFORMANCE.md` changed **normatively** after divsel's lot-11 ledger
 * checkpoint, at `e9447ac` and `9262375`. The superseded blanket bound
 * `abs(actual - expected) <= f_rel * max(1, abs(expected))` applied to every
 * field was measured to fail correct ports **69 times, by up to 8.1x**, and is
 * replaced by:
 *
 * * `tol(x) = f_rel * max(1, abs(x))` per **primitive**;
 * * `expected_g`, `expected_div`, `expected_d_max` against their own `tol`;
 * * `expected_f` against `tol(expected_g) + lam * tol(expected_div)`, because
 *   `f = g + lam*div` is **derived**, not a primitive — an error in `div`
 *   reaches `f` multiplied by `lam`, a caller's parameter with no ceiling;
 * * `expected_threshold` is a **selected grid entry**, not a measurement. The
 *   rule stays `abs(a - e) <= tol(e)` and is deliberately not widened;
 *   consecutive entries stand a factor `1 + eps` apart, so a port either picks
 *   the same entry and agrees to the last ulp or misses by five orders.
 * * `expected_selected` and `expected_stage` are exact-equality fields.
 *
 * Read from divsel at the pinned commit, never from the origin engine's port, which still
 * implements the superseded rule.
 *
 * ## Scope
 *
 * The full contract: both metrics, all three utilities, exhaustive thresholds,
 * and the approximate-diameter double sweep. **No case is skipped** — case 20,
 * the only optional one, runs like the rest.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { gistSelectFull, type GistSelectOptions, type Metric, type UtilityKind } from "../src/diversity.js";

/** The fixture bytes as committed in divsel at `d0f8ac8`. */
const FIXTURE_SHA256 = "73713cd2d7ec9bb23659a9ee05235600e3fa358b318f2095c3eb80eab2e458a7";

/**
 * `docs/CONFORMANCE.md` at divsel `9262375` — recorded so a future revision of
 * the rules is a visible diff here rather than a silent one. Nothing re-hashes
 * a divsel checkout at test time: limbic has no dependency on that path.
 */
const CONFORMANCE_COMMIT = "926237566b1ca6919be239b9ce1bda3f7fb71c76";

/** sha256 of that `docs/CONFORMANCE.md` blob, recorded for the same reason. */
const CONFORMANCE_SHA256 = "829bc08705ad13377624d08616254b1cd21b780f60c82b23e3a1e627e77be24b";

interface GoldenCase {
  name: string;
  note: string;
  metric: Metric;
  utility: UtilityKind;
  vectors: number[][];
  utilities: number[] | number[][] | null;
  k: number;
  lam: number;
  eps: number;
  exhaustive_thresholds: boolean;
  diameter: "exact" | "approx";
  diameter_sweeps: number;
  expected_selected: number[];
  expected_f: number;
  expected_g: number;
  expected_div: number;
  expected_threshold: number;
  expected_stage: "greedy" | "diameter_pair" | "sweep";
  expected_d_max: number;
}

interface Golden {
  generator: string;
  paper: string;
  schema: number;
  tolerance: { f_rel: number; selected: string };
  cases: GoldenCase[];
}

const fixturePath = fileURLToPath(new URL("./fixtures/golden-selection.json", import.meta.url));
const fixtureBytes = readFileSync(fixturePath);
const golden = JSON.parse(fixtureBytes.toString("utf8")) as Golden;

/** `tol(x) = f_rel * max(1, abs(x))`, with `f_rel` read from the file. */
const tol = (x: number): number => golden.tolerance.f_rel * Math.max(1, Math.abs(x));

describe("divsel golden-selection.json", () => {
  it("is byte-identical to the upstream fixture", () => {
    expect(createHash("sha256").update(fixtureBytes).digest("hex")).toBe(FIXTURE_SHA256);
  });

  it("is the schema and generator this reader was written for", () => {
    expect(golden.schema).toBe(1);
    expect(golden.generator).toBe("divsel 0.1.0");
    expect(golden.tolerance.f_rel).toBe(1e-6);
    expect(golden.tolerance.selected).toBe("exact");
    expect(golden.cases).toHaveLength(22);
    // The rules this reader implements come from one pinned commit; recording it
    // in an assertion keeps the citation in the failing output, not just a comment.
    expect(CONFORMANCE_COMMIT).toBe("926237566b1ca6919be239b9ce1bda3f7fb71c76");
    expect(CONFORMANCE_SHA256).toHaveLength(64);
  });
});

describe("GIST conformance — all 22 cases, none skipped", () => {
  for (const [index, c] of golden.cases.entries()) {
    it(`case ${index + 1}: ${c.name}`, () => {
      const opts: GistSelectOptions = {
        metric: c.metric,
        utility: c.utility,
        exhaustiveThresholds: c.exhaustive_thresholds,
        diameter: c.diameter,
        diameterSweeps: c.diameter_sweeps,
      };
      const r = gistSelectFull(c.vectors, c.utilities, c.k, c.lam, c.eps, opts);

      // Exact-equality fields: the index list including order, and the stage.
      expect(r.selected).toEqual(c.expected_selected);
      expect(r.stage).toBe(c.expected_stage);

      // Primitives, each against its own tol().
      expect(Math.abs(r.g - c.expected_g)).toBeLessThanOrEqual(tol(c.expected_g));
      expect(Math.abs(r.div - c.expected_div)).toBeLessThanOrEqual(tol(c.expected_div));
      expect(Math.abs(r.dMax - c.expected_d_max)).toBeLessThanOrEqual(tol(c.expected_d_max));

      // A selected grid entry, not a measurement — same bound, stated separately
      // because it means something different (see the header).
      expect(Math.abs(r.threshold - c.expected_threshold)).toBeLessThanOrEqual(
        tol(c.expected_threshold),
      );

      // `f = g + lam*div` is derived, so its budget is derived too: div's own
      // budget multiplied by the same lam the objective multiplies it by.
      expect(Math.abs(r.f - c.expected_f)).toBeLessThanOrEqual(
        tol(c.expected_g) + c.lam * tol(c.expected_div),
      );
    });
  }
});
