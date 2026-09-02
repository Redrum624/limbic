/**
 * A companion loop backed by a local Ollama: embeddings from `/api/embed`,
 * extraction from a chat model, retrieval diversified with GIST.
 *
 * Nothing here reaches past the host you configure. Ollama's default is
 * `http://127.0.0.1:11434` — the IPv4 literal, not `localhost`, because Node's
 * resolver may hand back `::1` on a machine where Ollama is listening on IPv4
 * only.
 *
 * Prerequisites:
 *   ollama pull nomic-embed-text
 *   ollama pull llama3.2
 *
 * Run (`tsx` is fetched by npx; it is not a devDependency):
 *   npx tsx examples/ollama-companion.ts
 */

import {
  OllamaEmbedder,
  categoryFor,
  createLimbic,
  passesSaveGate,
  type ChatTurn,
  type CompleteFn,
} from "../src/index.js";

const HOST = process.env["OLLAMA_HOST"] ?? "http://127.0.0.1:11434";
const CHAT_MODEL = process.env["OLLAMA_CHAT_MODEL"] ?? "llama3.2";
const EMBED_MODEL = process.env["OLLAMA_EMBED_MODEL"] ?? "nomic-embed-text";
// Bound the chat call the way OllamaEmbedder bounds its own requests: a stalled
// server must fail the turn, not hang it forever. Twice the embedder's 30 s
// default, because generation streams tokens where embedding returns at once.
const COMPLETE_TIMEOUT_MS = 60_000;

/**
 * The one thing limbic never bundles: a provider. `complete` is any function
 * that turns a prompt into text — Ollama here, but an OpenAI call, a local
 * llama.cpp handle or a canned string in a test all satisfy the same type.
 */
const complete: CompleteFn = async (prompt, opts) => {
  const response = await fetch(`${HOST}/api/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(COMPLETE_TIMEOUT_MS),
    body: JSON.stringify({
      model: CHAT_MODEL,
      prompt,
      stream: false,
      options: {
        num_predict: opts?.maxTokens ?? 1024,
        temperature: opts?.temperature ?? 0.3,
      },
    }),
  });
  if (!response.ok) throw new Error(`ollama /api/generate: HTTP ${response.status}`);
  const body = (await response.json()) as { response?: unknown };
  return typeof body.response === "string" ? body.response : "";
};

async function main(): Promise<void> {
  const limbic = createLimbic({
    embedder: new OllamaEmbedder({ host: HOST, model: EMBED_MODEL }),
    complete,
    // The origin engine's default. Raise it if retrieval keeps returning
    // paraphrases of one fact; the bench in `bench/redundancy.bench.ts` shows
    // where the crossover sits on a clustered corpus.
    lambda: 0.5,
  });

  const conversation: ChatTurn[] = [
    { role: "user", content: "I'm Ada. I moved to Lisbon last month and I'm learning Portuguese." },
    { role: "assistant", content: "Lisbon is lovely. How is the Portuguese going?" },
    { role: "user", content: "Slowly! I practise every morning before work. I'm a compiler engineer." },
  ];

  // 1. Extract. Everything below the save gate is dropped, exactly as the origin engine
  //    drops it: importance >= 0.4 AND confidence >= 0.6.
  const extracted = await limbic.extract(conversation);
  for (const candidate of extracted) {
    if (!passesSaveGate(candidate)) continue;
    await limbic.remember(candidate.content, {
      category: categoryFor(candidate.extractionType),
      importance: candidate.importance,
      keywords: candidate.keywords,
      subject: candidate.subject,
      feeling: candidate.feeling,
    });
  }
  console.log(`extracted ${extracted.length}, stored ${await limbic.store.count()}`);

  // 2. Retrieve. The query is embedded once; the pool is scored on recency,
  //    importance, keyword relevance and emotion, then diversified.
  const hits = await limbic.retrieve("what do I do for work?", 5);
  for (const hit of hits) {
    console.log(`${hit.score.toFixed(4)}  ${hit.memory.content}`);
  }

  // 3. Forget. Run this on a schedule, not per turn: it deletes anything whose
  //    strength has decayed below 0.05.
  console.log(await limbic.decayPass());
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
