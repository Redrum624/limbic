/**
 * Fixture drift gate.
 *
 * Golden fixtures are copied verbatim from upstream and are the conformance
 * contract. Re-hash them on every run so an edit, a partial re-copy or a
 * CRLF conversion fails here — loudly, with the two hashes side by side —
 * instead of silently redefining what "parity" means.
 *
 * Provenance for each entry lives in `test/fixtures/PROVENANCE.md`.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

interface FixtureProvenance {
  file: string;
  /** Upstream repo the bytes came from. */
  upstream: string;
  /** Path inside that repo. */
  upstreamPath: string;
  /** Commit at which the upstream file was last changed. */
  sourceCommit: string;
  /** Upstream HEAD at the moment the bytes were copied. */
  upstreamHeadAtCopy: string;
  bytes: number;
  upstreamSha256: string;
}

export const FIXTURES: readonly FixtureProvenance[] = [
  {
    file: "golden-scoring.json",
    upstream: "a private Python engine by the same author",
    upstreamPath: "test-assets/memory/golden-scoring.json",
    sourceCommit: "4a307a3b3553eb0b3d112f9c24649628a9c6ed04",
    upstreamHeadAtCopy: "ff9c407cd292848155a6c2982ccc4d185c518b5b",
    bytes: 5582,
    upstreamSha256:
      "fc1e983031ddb1cace78af238cdc1defe1f491020c83e671e49e0ad6453caabd",
  },
  {
    file: "golden-selection.json",
    upstream: "divsel (Redrum624/divsel, public) — https://github.com/Redrum624/divsel",
    upstreamPath: "test-assets/golden-selection.json",
    // divsel rewrote its history on 2026-08-26 (`d022aef`), so this commit is
    // the post-rewrite hash of the change that completed the fixture to 22
    // cases; its pre-rewrite name was `d0f8ac8`. Both objects are still
    // reachable and the fixture bytes are byte-identical at either.
    sourceCommit: "b9a7a9cfc05a421b264c5d8684776a3d102455da",
    upstreamHeadAtCopy: "d022aef648ae14ad381cb3082b2ca93140d9a786",
    bytes: 19365,
    upstreamSha256:
      "73713cd2d7ec9bb23659a9ee05235600e3fa358b318f2095c3eb80eab2e458a7",
  },
];

const here = fileURLToPath(new URL(".", import.meta.url));

describe("golden fixture provenance", () => {
  for (const fixture of FIXTURES) {
    describe(fixture.file, () => {
      const bytes = readFileSync(`${here}fixtures/${fixture.file}`);

      it("matches the recorded upstream_sha256", () => {
        const actual = createHash("sha256").update(bytes).digest("hex");
        expect(
          actual,
          `${fixture.file} drifted from ${fixture.upstream}:${fixture.upstreamPath} ` +
            `at ${fixture.sourceCommit}. Re-copy it verbatim; do not edit it here.`,
        ).toBe(fixture.upstreamSha256);
      });

      it("matches the recorded byte length", () => {
        expect(bytes.byteLength).toBe(fixture.bytes);
      });

      it("is stored with LF line endings, as upstream", () => {
        expect(bytes.includes(0x0d)).toBe(false);
      });
    });
  }
});
