/**
 * Fully in-process: embeddings from a GGUF model through `node-llama-cpp`, and
 * persistence in SQLite through `better-sqlite3`. No server, no socket, no
 * network stack at all.
 *
 * Both are **optional peers**. limbic's core has zero runtime dependencies;
 * these adapters `import()` their package lazily and throw
 * `EmbedderUnavailableError` with an install hint if it is not there, so a
 * consumer who never touches them never installs them.
 *
 * Prerequisites:
 *   npm i node-llama-cpp better-sqlite3
 *   # and a GGUF embedding model on disk, e.g. nomic-embed-text-v1.5.Q4_K_M.gguf
 *
 * Run:
 *   MODEL_PATH=/path/to/model.gguf npx tsx examples/node-llama-cpp.ts
 */

import {
  NodeLlamaCppEmbedder,
  SqliteStore,
  createLimbic,
  isEmbedderUnavailable,
} from "../src/index.js";

async function main(): Promise<void> {
  const modelPath = process.env["MODEL_PATH"];
  if (modelPath === undefined) {
    throw new Error("set MODEL_PATH to a GGUF embedding model");
  }

  // The store owns its own file. `SqliteStore` mirrors the origin engine's `memories`
  // table, embeddings included: a BLOB of little-endian f32, byte-compatible
  // with the layout the origin engine writes.
  const store = await SqliteStore.open("./limbic-example.db");

  const limbic = createLimbic({
    store,
    embedder: new NodeLlamaCppEmbedder({ modelPath }),
    // No `complete`: this example never calls an LLM, so `extract()` would
    // throw rather than quietly return nothing. Scoring and retrieval do not
    // need one.
  });

  await limbic.remember("User's cat is called Whiskers", {
    category: "relationship",
    importance: 0.8,
    keywords: ["cat", "whiskers", "pet"],
  });
  await limbic.remember("User is allergic to shellfish", {
    category: "health",
    importance: 0.95,
    keywords: ["allergy", "shellfish"],
  });

  for (const hit of await limbic.retrieve("tell me about the pet", 3)) {
    console.log(`${hit.score.toFixed(4)}  ${hit.memory.content}`);
  }
}

main().catch((error: unknown) => {
  if (isEmbedderUnavailable(error)) {
    // The adapter says what to install rather than leaving the caller with a
    // bare ERR_MODULE_NOT_FOUND.
    console.error(error.message);
  } else {
    console.error(error);
  }
  process.exitCode = 1;
});
