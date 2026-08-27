# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-27

First feature-complete release. **Not published to npm**: `npm publish` has not
run, so the `limbic` name is still unclaimed and this version exists only as a
tag and a build.

### Added

- Scaffold: TypeScript (strict, `noUncheckedIndexedAccess`) + tsup dual
  ESM/CJS build with types for both + vitest, Node >= 20, CI on node 20 and 22.
- `src/types.ts` — the 0.1.0 core type surface (`Memory`, `ExtractedMemory`,
  `Embedder`, `CompleteFn`, `ScoreWeights`, `DEFAULT_WEIGHTS`, `EMBED_BLEND`).
- `src/store.ts` — the `MemoryStore` seam and `MemStore`, the zero-dependency
  in-memory default. `src/stores/sqlite.ts` — `SqliteStore` behind the optional
  `better-sqlite3` peer, held to the same contract suite.
- `src/internal/vec.ts` — `cosine` (returns `null` for "cannot be compared",
  never `0`) and `l2normalize`.
- `src/internal/scoring.ts` — the four-channel base score plus the cosine blend,
  ported from the origin engine `server/memory/retrieval_service.py`. `scoreMemory` takes
  `now` explicitly and never reads the wall clock.
- `src/decay.ts` — `calculateDecay`, ported from the origin engine
  `server/memory/memory_decay.py`, including Python's round-half-to-even at
  3 dp.
- `src/embedders/ollama.ts` — `OllamaEmbedder` (POST `/api/embed`).
  `src/embedders/node-llama-cpp.ts`, `src/embedders/transformers.ts` — optional
  peer adapters loaded through dynamic `import()` with an install hint.
- `src/internal/gist.ts` + `src/diversity.ts` — GIST (arXiv:2405.18754v3)
  ported index-for-index from divsel, with `gistSelect` (ids in, ids out) and
  `gistSelectFull` (indices plus `f`, `g`, `div`, `threshold`, `stage`, `dMax`).
  Full contract: both metrics, all three utilities, exhaustive thresholds and
  the approximate-diameter double sweep.
- `src/extraction.ts` — the origin engine's `EXTRACTION_PROMPT`, the 10-turn window, the
  JSON parse and the save gate (`importance >= 0.4` **and**
  `confidence >= 0.6`).
- `src/retrieve.ts` — score the pool, then diversify it, porting the origin engine's
  `_apply_diversity` / `_fill_preserving_spread` pair.
- `src/index.ts` — `createLimbic({ store, embedder, complete, weights, lambda,
  pool })` returning `remember` / `extract` / `retrieve` / `decayPass` / `store`.
- `bench/redundancy.bench.ts` — naive top-`k` against GIST over a clustered
  corpus, sweeping `lambda`. `examples/ollama-companion.ts` and
  `examples/node-llama-cpp.ts`.

### Conformance

- **divsel `golden-selection.json`: 22 of 22 cases pass, none skipped** —
  case 20 (approximate diameter), the only optional case, included. Numeric
  agreement is **exact**: the worst tolerance-budget share across all 22 cases
  × 5 numeric fields is `0`, because the distance kernel reproduces divsel's
  `f32` 16-accumulator arithmetic rather than computing in `f64`.
- Rules read from divsel's `docs/CONFORMANCE.md` at commit `9262375`
  (`sha256 829bc087…`), which supersedes the earlier blanket tolerance — that
  one was measured to fail correct ports 69 times by up to 8.1×. `expected_f`
  is bounded as `tol(expected_g) + lam*tol(expected_div)` because `f` is
  derived; `expected_threshold` is a selected grid entry, not a measurement.
- the origin engine `golden-scoring.json`: base score, final score and both ranking arrays
  at the fixture's own absolute `1e-6`, clock pinned to its `now`.
- Both fixtures are copied byte-for-byte and re-hashed on every test run
  (`test/fixtures.hash.test.ts`); provenance is in
  `test/fixtures/PROVENANCE.md`.

### Notes on the port

- `lambda` defaults to **0.5** (the origin engine's `ORIGIN_MEMORY_LAMBDA`); divsel's own
  default is `1.0`. The bench shows the crossover between them on a clustered
  corpus, so limbic's default is the conservative one.
- the origin engine's `EXTRACTION_PROMPT.format(...)` raises `KeyError` on its own JSON
  example and the exception is swallowed, so the origin engine's LLM extraction path returns
  `[]` today. limbic substitutes the placeholder literally instead.
- Unknown extraction types are kept and mapped to `general` rather than
  dropped.

## [0.0.1]

- Name claim placeholder. Never published.
