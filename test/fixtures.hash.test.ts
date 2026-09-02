/**
 * Fixture drift gate.
 *
 * Golden fixtures are the conformance contract. Re-hash them on every run so
 * an edit, a partial re-copy or a CRLF conversion fails here — loudly, with
 * the two hashes side by side — instead of silently redefining what "parity"
 * means.
 *
 * `golden-selection.json` is byte-verbatim from upstream. `golden-scoring.json`
 * is upstream's bytes with exactly two metadata strings (`generator`,
 * `description`) neutralised for publication on 2026-08-29 — no number, key or
 * byte of formatting changed — so its hash pins the published bytes, not the
 * upstream ones. Provenance for each entry lives in
 * `test/fixtures/PROVENANCE.md`.
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
  /**
   * sha256 of the committed bytes — upstream's bytes for a verbatim copy, the
   * published bytes where PROVENANCE.md records a neutralisation.
   */
  upstreamSha256: string;
}

export const FIXTURES: readonly FixtureProvenance[] = [
  {
    file: "golden-scoring.json",
    upstream: "a private Python engine by the same author",
    upstreamPath: "test-assets/memory/golden-scoring.json",
    // The origin engine is unpublished, so its commit ids identify nothing a
    // reader can reach; only the dates are recorded.
    sourceCommit: "(private, unpublished; file last changed 2026-08-26)",
    upstreamHeadAtCopy: "(private, unpublished; copied 2026-08-27)",
    // Two metadata strings (`generator`, `description`) neutralised for
    // publication on 2026-08-29; every number, key and byte of formatting is
    // upstream's. The hash pins the published bytes.
    bytes: 5599,
    upstreamSha256:
      "89755bf6a74bd7408fbe0a8313d0bb370a18130f35bbfc567d632d90fb5cf349",
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
          `${fixture.file} drifted from ${fixture.upstream}:${fixture.upstreamPath}. ` +
            `Restore the pinned bytes (see test/fixtures/PROVENANCE.md); do not edit it here.`,
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
