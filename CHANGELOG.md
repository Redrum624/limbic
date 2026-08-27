# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Entries are added per plan task as they land. Tasks 2 (stores), 6 (GIST
> diversity) and 7 (extraction / retrieval / `createLimbic`) are in the tree
> but not yet listed here.

### Added

- Scaffold: TypeScript (strict) + tsup dual ESM/CJS build + vitest, Node >= 20.
- `src/types.ts` — the 0.1.0 core type surface (`Memory`, `ExtractedMemory`,
  `Embedder`, `CompleteFn`, `ScoreWeights`, `DEFAULT_WEIGHTS`, `EMBED_BLEND`).
- `src/internal/vec.ts` — `cosine` (returns `null` for "cannot be compared",
  never `0`) and `l2normalize`.
- `src/internal/scoring.ts` — the four-channel base score plus the cosine blend,
  ported from the origin engine `server/memory/retrieval_service.py`.
- Golden scoring parity test against the origin engine's `test-assets/memory/golden-scoring.json`,
  with the fixture's `sha256` pinned and re-hashed on every run.
- `src/decay.ts` — `calculateDecay`, ported from the origin engine `server/memory/memory_decay.py`.
- `src/embedders/ollama.ts` — `OllamaEmbedder` (POST `/api/embed`).
- `src/embedders/node-llama-cpp.ts`, `src/embedders/transformers.ts` — optional
  peer adapters loaded through dynamic `import()`.

## [0.0.1]

- Name claim placeholder.
