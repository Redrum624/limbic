# limbic

**Local-first, emotion-aware agent memory for TypeScript.** Importance, decay, emotional salience and diversity-aware retrieval — fully offline, LLM-agnostic, zero cloud calls, zero required runtime dependencies.

> **Status: 0.1.0, feature-complete and green — not yet published.**
> `npm publish` has **not** run, so `limbic` is **not** on the registry and the
> name remains unclaimed. Treat availability as unverified. Install from a
> checkout until that changes.
>
> The implementation plan lives at
> [`docs/superpowers/plans/2026-08-21-limbic-v0.1.md`](docs/superpowers/plans/2026-08-21-limbic-v0.1.md).

```ts
import { createLimbic } from "limbic";

const limbic = createLimbic();

await limbic.remember("User's name is Ada", { category: "personal_fact", importance: 0.9 });
await limbic.remember("User is allergic to shellfish", { category: "health", importance: 0.95 });

const hits = await limbic.retrieve("what should I avoid cooking?", 3);
// [{ memory: { content: "User is allergic to shellfish", ... }, score: 0.71 }, ...]
```

That runs with **no dependencies, no model, no network and nothing on disk**. Add an embedder and the cosine channel switches on; add a `complete` function and extraction switches on. Neither is required, and neither failing is fatal.

---

## Why

Every TypeScript memory option today is either a **cloud service client** (Zep), a framework-bound store with no memory model (LangChain.js post-0.3 LangGraph stores — namespace/key JSON, no scoring), or an engine whose defaults phone home to OpenAI (`mem0ai` OSS). None of them model what a *companion-grade* memory needs: **importance**, **forgetting curves**, **emotional salience**, and **retrieval that does not return five paraphrases of the same fact**.

| Incumbent | What it actually is | Gap limbic fills |
|---|---|---|
| `mem0ai` 3.1.6 | A real in-process OSS Node engine — but the defaults are OpenAI for the LLM *and* `text-embedding-3-small` for embeddings, and there is no importance / decay / emotion model | local-first defaults; affect and forgetting curves |
| `@getzep/zep-js`, `@getzep/zep-cloud` | Service client SDKs. There is no in-process engine to run | an engine you can run offline |
| LangChain.js ≥ 0.3 | Memory abstractions deprecated in favour of LangGraph `BaseStore`: namespace/key JSON with get/put/search and no scoring | the memory *model* itself |
| LlamaIndex.TS 0.12 | `createMemory()` is FIFO plus blocks; prioritisation is a bare integer | scoring, decay, emotion, diversity |
| `node-llama-cpp` 3.20 | Local embeddings, and a good one — but it is an inference library, not a memory engine | limbic *consumes* it, as an optional peer |

limbic is a TypeScript port of the production memory engine (`server/memory/`) from a private Python engine by the same author — a shipped offline AI-companion server. The semantics are not reinvented here; they are ported and then **proved equivalent by cross-language golden fixtures**.

## Design commitments

- **Local-first, always.** The only network call limbic can make is to an Ollama host *you* configure. There is no telemetry and no default cloud provider.
- **Zero required runtime dependencies** in the core. `better-sqlite3`, `node-llama-cpp` and `@huggingface/transformers` are optional peers, loaded through dynamic `import()` with an install hint if they are absent.
- **LLM-agnostic.** You inject `complete(prompt) => Promise<string>`. limbic bundles no provider SDK, and never will.
- **Nothing at the edges is fatal.** An embedder that is down costs the cosine channel, not the write and not the turn. Diversity that throws degrades to top-`k` by score.
- **Apache-2.0.**

## Install

```sh
npm i limbic                       # core — no dependencies
npm i limbic better-sqlite3        # + SQLite persistence
npm i limbic node-llama-cpp        # + in-process GGUF embeddings
npm i limbic @huggingface/transformers   # + transformers.js embeddings
```

Node ≥ 20 (limbic uses the global `fetch`). Dual ESM/CJS, types resolved for both.

## Quickstart

### 1. Pure in-memory — no dependencies at all

```ts
import { createLimbic } from "limbic";

const limbic = createLimbic();
await limbic.remember("User practises Portuguese every morning", {
  category: "interest",
  importance: 0.6,
  keywords: ["portuguese", "morning", "practice"],
});

const hits = await limbic.retrieve("what does the user study?", 5);
```

Without an embedder, scoring is keyword-only: the cosine channel is **MISSING**, which is not the same as a similarity of `0` and is handled as such throughout.

### 2. Ollama

```ts
import { OllamaEmbedder, createLimbic } from "limbic";

const limbic = createLimbic({
  embedder: new OllamaEmbedder({ host: "http://127.0.0.1:11434", model: "nomic-embed-text" }),
  complete: async (prompt, opts) => { /* your chat call */ return "..."; },
});
```

`OllamaEmbedder` POSTs `{ model, input }` to `{host}/api/embed` and reads `json.embeddings` — the same endpoint, body keys and response field the origin engine's `ollama_client.py` uses. The legacy `/api/embeddings` is deliberately unsupported, as it is there. Use the IPv4 literal rather than `localhost`: Node's resolver can hand back `::1` on a machine where Ollama listens on IPv4 only.

Full script: [`examples/ollama-companion.ts`](examples/ollama-companion.ts).

### 3. Fully in-process — `node-llama-cpp` + SQLite

```ts
import { NodeLlamaCppEmbedder, SqliteStore, createLimbic } from "limbic";

const limbic = createLimbic({
  store: await SqliteStore.open("./memories.db"),
  embedder: new NodeLlamaCppEmbedder({ modelPath: "/models/nomic-embed-text-v1.5.Q4_K_M.gguf" }),
});
```

No server, no socket, no network stack. Full script: [`examples/node-llama-cpp.ts`](examples/node-llama-cpp.ts).

## API

### `createLimbic(options?)`

| Option | Default | Meaning |
|---|---|---|
| `store` | `new MemStore()` | Any `MemoryStore`. `SqliteStore` is built in behind an optional peer. |
| `embedder` | none | Any `Embedder`. Absent means keyword-only scoring and no vectors stored. |
| `complete` | none | Any `CompleteFn`. Absent means `extract()` **throws** rather than silently returning `[]`. |
| `weights` | `DEFAULT_WEIGHTS` | `{ recency: 0.25, importance: 0.35, relevance: 0.25, emotion: 0.15 }` |
| `lambda` | `0.5` | GIST's diversity weight. See the note below — divsel's own default is `1.0`. |
| `pool` | `50` | How many scored rows to diversify over. |

Returns:

```ts
remember(content: string, partial?: Partial<Memory>): Promise<Memory>
extract(conversation: ChatTurn[]): Promise<ExtractedMemory[]>
retrieve(query: string, k?: number, options?: RetrieveOptions): Promise<ScoredMemory[]>
decayPass(now?: Date): Promise<{ decayed: number; faded: number }>
store: MemoryStore
```

### Scoring

```
recency   = 0.5 ^ (daysSinceLastAccess / 7)
base      = clamp(0.25*recency + 0.35*importance + 0.25*relevance + 0.15*emotion, 0, 1)
final     = base                                        (no comparable vector)
final     = clamp(0.70*base + 0.30*max(0, cosine), 0, 1)  (both vectors present)
```

`scoreMemory(memory, query, now)` takes `now` **explicitly** and never reads the wall clock. That is what makes the scoring fixture evaluable at all — the origin engine's `_calculate_recency_score` calls `datetime.now()` directly.

### Decay

`calculateDecay` is the origin engine's `memory_decay.py` verbatim: per-category half-lives (`relationship` 365 d down to `emotion`/`work` 30 d), an importance factor that stretches or shrinks the half-life, `+5` days of half-life per access, and floors so that a memory at importance ≥ 0.8 never falls below 0.3. `decayPass()` deletes anything below **0.05** and reports `{ decayed, faded }`.

### Retrieval and diversity

`retrieve` scores the pool, then hands the rows that carry a vector to GIST (arXiv:2405.18754v3), maximising `g(S) + lambda * div(S)`.

Three properties are load-bearing, and each has a test that fails if it breaks:

1. **Diversity changes membership, never order.** The pool is sorted by score, so a position in it *is* its rank; the result is assembled by index, so the ordering and tie-break are identical to the no-diversity path.
2. **A memory with no embedding is selectable but edge-free.** It is not a point GIST sees, so it can take a slot GIST left open and cannot lower the spread.
3. **The fill never re-admits what the selector rejected.** Topping the result up by score alone puts the near-duplicates straight back, which makes diversity mode return a set whose spread is identical to plain top-`k`. So `retrieve` can return **fewer than `k`** rows. That short return is a signal, not a bug: it means `lambda` is too high for this corpus.

## Parity

This is the part that is worth reading. limbic does not claim to behave "like" its references; it reproduces their committed fixtures.

### Diversity — divsel's `golden-selection.json`

| | |
|---|---|
| Fixture | 22 cases, schema 1, generator `divsel 0.1.0`, copied byte-for-byte (`sha256 73713cd2…`) and re-hashed on every test run |
| Rules | divsel `docs/CONFORMANCE.md` at commit `9262375` (`sha256 829bc087…`) |
| Cases passing | **22 of 22 — none skipped**, case 20 (approximate diameter, the only optional case) included |
| Scope | both metrics, all three utilities (`linear`, `coverage`, `facility_location`), exhaustive thresholds, approximate diameter |
| Numeric agreement | **exact** — worst tolerance-budget share across all 22 cases × 5 numeric fields is **0** |

That last row is not a rounding claim. divsel computes every distance in `f32` with a fixed 16-accumulator reduction order, and its golden values were generated from that arithmetic; `src/internal/gist.ts` reproduces the same kernel with `Math.fround` rather than computing in `f64`, so the two agree bit-for-bit instead of merely within `1e-6`. It is also not a prose claim: the suite asserts it, gated at 0.1% of each field's budget and logging the measured share, with the small slack there only because ECMA-262 specifies `Math.sqrt` as *implementation-approximated* rather than correctly rounded.

The tolerance rules matter and changed recently. divsel's earlier blanket bound `abs(a − e) <= f_rel*max(1, abs(e))` on every field was **measured to fail correct ports 69 times, by up to 8.1×**, and was replaced at `e9447ac`/`9262375` by a per-primitive `tol(x)`, with `expected_f` bounded as `tol(expected_g) + lam*tol(expected_div)` because `f = g + lam*div` is *derived*, and `expected_threshold` treated as a selected grid entry rather than a measurement. limbic implements the current rules, read from divsel — **not** from the origin engine's port, which still implements the superseded one.

The reader is not vacuous, and that is measured too. Four deliberate rule mutations reproduce CONFORMANCE.md's own published counts exactly:

| Mutation | limbic fails | CONFORMANCE.md says |
|---|---|---|
| strict `>` in the sweep fold (rule 2) | 15 of 22 | 15 |
| argmax ties to the highest index (rule 1) | 9 of 22 | nine |
| `lambda` doubled | 20 of 22 | 20 |
| `div(\|S\| ≤ 1) = 0` instead of `d_max` (rule 4) | case 7 alone | case 7 |

> **`lambda` defaults to `0.5` in limbic**, matching the origin engine's own default. **divsel's own default is `1.0`.** The bench below shows the crossover sitting between them on a clustered corpus, so this default is the conservative one: it diversifies less, not more.

### Scoring — the origin engine's `golden-scoring.json`

Copied byte-for-byte (`sha256 fc1e9830…`), clock pinned to the fixture's `now = 2026-08-26T12:00:00`, asserted at the fixture's own **absolute** `1e-6` on `expected_base_score`, `expected_final_score` and both ranking arrays. `cosine == null` means **MISSING**, never `0.0` — a port that substituted `0.0` would fail the `portuguese` row.

### Deliberate deviations from the Python reference

| limbic | The origin engine | Why |
|---|---|---|
| `id: string` | `Optional[int]` autoincrement | limbic must support non-SQL stores and caller-supplied ids. Coerce to string when importing an origin-engine DB. |
| `extractionType: string` | closed `ExtractionType` enum | Keeps the core open to user-defined types. Unknown types map to `general` and are **not** rejected; the origin engine drops the row. |
| Prompt placeholder substituted literally | `EXTRACTION_PROMPT.format(...)` | The origin engine's call raises `KeyError: '\n  "memories"'` on its own JSON example — `str.format` reads it as a replacement field — and the broad `except Exception` swallows it, so **the origin engine's LLM extraction path returns `[]` today**. limbic replaces the one `{conversation}` token and leaves the example intact. |
| `extract()` never writes | writes through when `save_immediately` | `remember()` is the only writer. The save gate travels with the data as `passesSaveGate`. |
| `Memory.emotion?: { label, intensity }` | read from the source conversation | The origin engine scores the *conversation's* detected emotion, which limbic does not store; the caller supplies the pair. `Memory.feeling` is the extractor's free-text tone and is **not** what is scored. |
| `scoreMemory(m, q, now)` | reads `datetime.now()` | An explicit clock is what makes the fixture evaluable. |

## Bench

`npm run bench` — 60 memories, 12 planted topics × 5 near-paraphrases each, dim 48, fixed seed, `k = 8`, clusters counted as connected components at `cosine > 0.92`. Metric definitions are the origin engine's plan Task 8 definitions.

| strategy | redundancy ↓ | clusters hit | coverage ↑ | facts / 1k prompt chars ↑ |
|---|---|---|---|---|
| naive top-`k` | 0.9794 | 2 / 12 | 0.167 | 3.086 |
| `gistSelect` λ=0 | 0.9794 | 2 / 12 | 0.167 | 3.086 |
| `gistSelect` λ=0.5 *(default)* | 0.9794 | 2 / 12 | 0.167 | 3.086 |
| `gistSelect` λ=1 | **0.1387** | **8 / 12** | **0.667** | **12.346** |
| `gistSelect` λ=2 | 0.1387 | 8 / 12 | 0.667 | 12.346 |
| `gistSelect` λ=4 | 0.1387 | 8 / 12 | 0.667 | 12.346 |
| `gistSelect` λ=8 | 0.1387 | 8 / 12 | 0.667 | 12.346 |

Above the crossover the same eight slots and the same 648 prompt characters carry **four times the distinct facts**, and redundancy falls from "every pick has a near-twin" to "almost none does". Cost: **≈3.1 ms** per selection at this size against ~70 ns for a slice — GIST runs `2 + |D|` greedy passes, 32 thresholds at `eps = 0.1`.

**Read the top three rows as the honest half.** At `λ ≤ 0.5` GIST returns exactly top-`k` here, and that is *correct*: it maximises `g(S) + λ·div(S)`, and below the crossover the score given up to reach another cluster outweighs the diversity gained. Whether the default `0.5` is right for your corpus depends on how steep your score gradient is across topics. Sweep it.

**Caveats, stated rather than buried.** The corpus is **synthetic and n = 1**, built from a fixed seed with a cluster structure chosen to make redundancy visible. The origin engine's `server/tools/memory_bench.py` exists and could supply a real corpus, but as of 2026-08-27 the collapse-onset figures in the origin engine's `docs/benchmarks/memory-diversity.md` are **retracted** — the detector that produced them could not fire — and a re-cut is in progress. The definitions are shared with it; no figure from it is quoted here.

## Upstream

| Upstream | State | What limbic takes |
|---|---|---|
| **divsel** 0.1.0 — Rust GIST reference + Python bindings ([GitHub](https://github.com/Redrum624/divsel), public) | Finished; not yet tagged or published to crates.io / PyPI — which gates nothing here, since the fixture is consumed from a checkout | `test-assets/golden-selection.json` and `docs/CONFORMANCE.md` — the contract `gistSelect` reproduces |
| **The origin engine** (private) | Memory-engine plan 9 of 12 done; the spec, both fixtures, `diversity.py`, `vectorops.py` and `memory_bench.py` all exist and are tracked | `test-assets/memory/golden-scoring.json`, plus `server/memory/` as the reference for scoring, decay and extraction |
| **A sibling library** 0.1.0 | Finished — but it is the origin engine's *life-simulation* half, extracted from `server/engine/` and `server/personas/` | Nothing. It is orthogonal to this port. |

## Name

The limbic system is the brain circuitry where emotion and memory meet. That is exactly the scope of this library.

*(The npm name was verified available on 2026-08-21 and re-checked 2026-08-26 — `GET https://registry.npmjs.org/limbic` returned HTTP 404. It has **not** been claimed: no publish has run. Availability is not a promise; re-check at the moment of claiming.)*

## License

Apache-2.0 © 2026 Redrum624.
