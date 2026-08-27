# limbic

**Local-first, emotion-aware agent memory for TypeScript.** Importance, decay, emotional salience and diversity-aware retrieval — fully offline, LLM-agnostic, zero cloud calls.

> Status: **pre-0.1, in progress** — the build harness (TypeScript strict, tsup dual ESM/CJS, vitest, CI on node 20 and 22),
> the core types and the storage layer (`MemStore` plus an optional `better-sqlite3` `SqliteStore`) are in.
> The npm name is **still unclaimed**: `npm publish` has not run, so treat availability as unverified from here on.
> The implementation plan lives at
> [`docs/superpowers/plans/2026-08-21-limbic-v0.1.md`](docs/superpowers/plans/2026-08-21-limbic-v0.1.md).

**Upstream status (verified 2026-08-26):**

| Upstream | State | What limbic takes from it |
|---|---|---|
| **divsel** 0.1.0 — Rust GIST reference + Python bindings (`C:\Dev\divsel`, [GitHub](https://github.com/Redrum624/divsel), **public**) | **Finished.** Merged to `main` at `9262375`; CI green on HEAD (134 Rust / 155 Python). The crates.io / PyPI publish and the `v0.1.0` tag are still pending user-run steps (`docs/RELEASE.md`) — **but they gate nothing here**: the fixture is consumed from the checkout, not from a registry | `test-assets/golden-selection.json` (22 cases) and `docs/CONFORMANCE.md` — the diversity-selection contract `gistSelect` must reproduce |
| **the origin engine memory-engine plan** (`the origin engine's checkout\docs\superpowers\plans\2026-08-21-memory-retrieval-diversity-extraction-seam.md`) | **9 of 12 tasks done** (corrected 2026-08-26 — an earlier version of this row said “0 of 12”). Tasks 1–9 plus a finish pass are complete; T10 (activate LLM extraction), T11 (cut the `memory/` couplings) and T12 remain. `memory-engine-spec.md`, `golden-scoring.json`, `diversity.py`, `vectorops.py` and `memory_bench.py` **all exist and are tracked** | **`test-assets/memory/golden-scoring.json`** — the scoring contract, which means **Task 3 is unblocked**. The decay and extraction reference is still `server/memory/` source, cited by file:line in the plan |
| **aura-life** 0.1.0 (`C:\Dev\aura-life`) | **Finished** 2026-08-26 — but it is the origin engine's *life-simulation* engine and persona pipeline, extracted from `server/engine/` + `server/personas/`. The memory engine did not move | Nothing. It is orthogonal to this port |

## Why

Every TypeScript memory option today is either a **cloud service client** (Zep, Mem0 Platform), a framework-bound store with no memory model (LangChain.js post-0.3 LangGraph stores — namespace/key JSON, no scoring), or an engine whose defaults phone home to OpenAI (mem0ai OSS). None of them model what a *companion-grade* memory needs: **importance**, **forgetting curves**, **emotional salience**, and **retrieval that doesn't return five copies of the same fact**.

`limbic` is a TypeScript port of the production memory engine (`server/memory/`) from [the origin engine](the origin engine) (private), a shipped offline AI-companion server — scoring, decay and extraction semantics proven in a real product, verified here by **cross-language golden fixtures**: diversity selection must reproduce the 22-case fixture of the Rust reference implementation, [divsel](https://github.com/Redrum624/divsel), and scoring must match the Python engine to 6 decimal places on the same inputs. **Both fixtures now exist** — `divsel/test-assets/golden-selection.json` and `the origin engine/test-assets/memory/golden-scoring.json` — so Tasks 3 and 6 are both unblocked.

## Design commitments

- **Local-first, always.** The only network call `limbic` can ever make is to an Ollama host *you* configure. Embeddings run via `node-llama-cpp`, Ollama `/api/embed`, or `@huggingface/transformers` — all optional peers, all offline.
- **Zero required runtime dependencies** in the core. SQLite persistence (`better-sqlite3`) is an optional peer; an in-memory store is built in.
- **LLM-agnostic extraction.** You inject a `complete(prompt) => Promise<string>` function; `limbic` never bundles a provider SDK.
- **Diversity-aware retrieval.** GIST max-min diversification (Google Research, NeurIPS 2025 — arXiv:2405.18754) instead of naive top-k, so retrieved context stops being redundant.
- **Apache-2.0.**

## Name

The limbic system is the brain circuitry where emotion and memory meet. That is exactly the scope of this library.

*(npm name `limbic` verified available 2026-08-21 and re-checked 2026-08-26 — `GET https://registry.npmjs.org/limbic` returns HTTP 404. Claiming it with a 0.0.1 stub is Task 0 of the plan. Re-check at the moment of claiming: availability is not a promise.)*
