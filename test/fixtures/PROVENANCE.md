# Fixture provenance

Golden fixtures are copied in from their upstream repository, never regenerated
here, and their **values** are never edited. `golden-selection.json` is
byte-verbatim; `golden-scoring.json` carries one documented exception — two
metadata strings neutralised for publication (see its section). Every entry
below is pinned by `sha256`, and `test/fixtures.hash.test.ts` re-hashes each
file on every run so a silent edit or a line-ending conversion fails the suite
instead of quietly changing what "parity" means.

## `golden-scoring.json`

| Field | Value |
|---|---|
| Upstream | A private Python engine by the same author ("the origin engine") |
| Upstream path | `test-assets/memory/golden-scoring.json` |
| Produced by | The origin engine's scoring code — its `retrieval_service.py` is the only definition of these numbers |
| Source commit (file last changed) | (private, unpublished), 2026-08-26 |
| Upstream `HEAD` when copied | (private, unpublished) (2026-08-27) |
| Copied on | 2026-08-27 |
| Size | 5599 bytes |
| Line endings | LF (243), CR (0) — held by `.gitattributes` (`test/fixtures/*.json -text`) |
| `sha256` (published bytes) | `89755bf6a74bd7408fbe0a8313d0bb370a18130f35bbfc567d632d90fb5cf349` |

**Not byte-verbatim — one documented exception.** On 2026-08-29, in preparation
for publication, exactly two metadata strings were neutralised in the committed
copy: `generator` (now `"the origin engine's scoring code (Task 5)"`) and
`description` (the same sentence with the engine named only as "the origin
engine"). Nothing else changed — not a byte of the numbers, keys, or formatting.
The `sha256` above therefore pins the **published** bytes, not the upstream
ones; the numeric content is upstream's, untouched.

**Shape.** A *single scenario*, not a `cases[]` array. Top-level keys:
`generator, schema, description, tolerance, now, clock, query, query_keywords,
target_emotion, query_vector, stub_vectors, embedding_model, weights, formula,
memories, expected_order_base, expected_order_embedded`.

**What the parity test asserts.** `expected_base_score`, `expected_final_score`
and both `expected_order_*` arrays, at the fixture's own tolerance —
`score_abs = 1e-6`, **absolute** (`abs(actual - expected) <= 1e-6`). The
per-memory `channels` block is *display-rounded to 6 dp* upstream, so it is
useful for diagnosis and is **not** an assertion target.

**Clock.** Recency is measured against `now = 2026-08-26T12:00:00`. The origin engine's
`_calculate_recency_score` reads `datetime.now()` directly; limbic's
`scoreMemory` takes `now` as an explicit argument and never reads the wall
clock, which is what makes this fixture evaluable at all.

**`cosine == null` is MISSING, never `0.0`** — the `portuguese` memory has
`vector: null` and its `expected_final_score` equals its `expected_base_score`.
A port that substituted `0.0` would return `0.7 * 0.425 = 0.2975` and fail.

## `golden-selection.json`

| Field | Value |
|---|---|
| Upstream | `divsel` — https://github.com/Redrum624/divsel (public) |
| Upstream path | `test-assets/golden-selection.json` |
| Produced by | divsel's own generator, `python/tools/gen_golden.py` (brute-force oracle over every subset of size `<= k`) |
| Source commit (file last changed) | `b9a7a9cfc05a421b264c5d8684776a3d102455da` — *"test(golden): complete the CONFORMANCE.md contract, add cases 21-22"*, 2026-08-25 |
| Upstream `HEAD` when copied | `d022aef648ae14ad381cb3082b2ca93140d9a786` (2026-08-27) |
| Copied on | 2026-08-27 |
| Size | 19365 bytes |
| Line endings | LF (1123), CR (0) — held by `.gitattributes` (`test/fixtures/*.json -text`), as upstream's own `.gitattributes` holds it there |
| `upstream_sha256` | `73713cd2d7ec9bb23659a9ee05235600e3fa358b318f2095c3eb80eab2e458a7` |

**The rules, and where they come from.** The fixture is only half the contract;
the other half is divsel's `docs/CONFORMANCE.md`, read at commit
**`926237566b1ca6919be239b9ce1bda3f7fb71c76`** (*"fix(conformance): a claim the
measurement disproved, and the field the fix missed"*, 2026-08-26),
`sha256 829bc08705ad13377624d08616254b1cd21b780f60c82b23e3a1e627e77be24b`.

> ⚠️ Read those rules **from divsel**, never from the origin engine's port, which still
> implements the superseded blanket tolerance.

**divsel rewrote its history on 2026-08-26** (`d022aef`), so the abbreviations
cited in this file and in `test/diversity.golden.test.ts` are pre-rewrite names
of objects that are still reachable but no longer on `main`: fixture `d0f8ac8`
→ `b9a7a9c`, conformance `9262375` → `02c546f`, and `main` is now `d022aef`.
The fixture **bytes** are identical at every one of them, which is what the
`sha256` above pins.

**Shape.** `generator: "divsel 0.1.0"`, `schema: 1`, `tolerance: {f_rel: 1e-6,
selected: "exact"}`, and `cases[]` with 22 entries. There is **no `ids` field**:
`expected_selected` is a list of **0-based row indices in selection order**.

**What the parity test asserts** (`test/diversity.golden.test.ts`), under the
rules as they stand at `9262375` — the contract changed *normatively* at
`e9447ac` and `9262375`, and the old blanket bound was measured to fail correct
ports 69 times by up to 8.1x:

| Field | Rule |
|---|---|
| `expected_selected` | **Exact** list equality, order included |
| `expected_stage` | Exact string equality |
| `expected_g`, `expected_div`, `expected_d_max` | `abs(a - e) <= tol(e)`, `tol(x) = f_rel * max(1, abs(x))` |
| `expected_threshold` | Same bound — but it is a **selected grid entry**, not a measurement, so the bound never decides anything: entries stand a factor `1 + eps` apart |
| `expected_f` | `abs(a - e) <= tol(expected_g) + lam * tol(expected_div)` — because `f = g + lam*div` is **derived**, and an error in `div` reaches `f` multiplied by `lam` |

**Scope: the full contract.** Both metrics, all three utilities (`linear`,
`coverage`, `facility_location`), exhaustive thresholds and the approximate
diameter. **No case is skipped**, case 20 included — so limbic has no exemption
to declare in its conformance report.

**Measured fidelity.** limbic reproduces divsel's `f`, `g`, `div`, `threshold`
and `d_max` **exactly** on all 22 cases — worst budget share `0` of the
tolerance, not merely inside it — because `src/internal/gist.ts` reproduces
divsel's `f32` distance kernel (16 logical accumulators, tail folded into
`idx % 16`, fixed-order reduction) with `Math.fround` rather than computing in
`f64`.
