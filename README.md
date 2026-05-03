# @paulrobello/webllm

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

High-performance LLM inference in the browser via WebGPU, backed by llama.cpp's
`ggml-webgpu` backend. Supports hierarchical multi-model scheduling with
frame-budget-aware execution for interactive applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Overview](#api-overview)
- [Embeddings](#embeddings)
- [Conversation persistence](#conversation-persistence)
- [Development](#development)
- [Evaluation & Live Dashboard](#evaluation--live-dashboard)
- [License](#license)

## Features

- **GGUF model parsing** — read quantized GGUF binary format directly in the browser
- **Multi-model scheduling** — priority-based cooperative scheduler with configurable frame budgets
- **KV cache management** — paged KV cache with multi-sequence sharing and cross-session prompt caching
- **Character system** — personas with system prompts, streaming chat, and tool / function calling
- **Lightweight WGSL path** — pure TypeScript + WGSL shaders for sub-50M parameter models (no WASM)
- **Memory management** — GPU buffer pool with pressure detection and priority-based eviction
- **Game loop integration** — `requestAnimationFrame`-aware scheduling for real-time applications
- **Tokenization** — SentencePiece (SPM) and Byte Pair Encoding (BPE) tokenizers
- **Evaluation harness** — micro-benchmarks, offline task evaluation, browser-driven chat regression with profile-based sweeps, and a live SSE + SQLite dashboard for side-by-side comparison of multiple runs

## Installation

```bash
bun add @paulrobello/webllm
```

## Quick Start

The library ships a WebAssembly module (`webllm-wasm.js` + `webllm-wasm.wasm`,
plus a `webllm-wasm-mem64.*` variant for models > 3.5 GiB) that must be served
from your application alongside the JS bundle. `WebLLM.loadModelFromBuffer`
picks the right variant based on the model file size; pass an explicit
`wasmUrl` to override.

```typescript
import { WebLLM } from "@paulrobello/webllm";

// 1. Acquire a WebGPU device.
if (!navigator.gpu) throw new Error("WebGPU unavailable");
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("requestAdapter() returned null");
const device = await adapter.requestDevice();

// 2. Fetch the GGUF model into memory.
const buffer = await fetch("/models/llama-3.2-3b-q4_k_m.gguf")
  .then((r) => r.arrayBuffer());

// 3. Load the model. This parses the GGUF, instantiates the WASM backend,
//    uploads weights to the GPU, and returns an engine bound to the model.
const { engine, handle } = await WebLLM.loadModelFromBuffer(
  buffer,
  "shopkeeper",
  {
    device,
    memoryBudget: 2048 * 1024 * 1024, // 2 GB VRAM headroom for KV cache
    cacheDir: "indexeddb://webllm-cache",
    frameBudgetMs: 8,
  },
);

// 4a. Streaming chat completion (OpenAI-style messages):
for await (const chunk of engine.chatCompletion(handle.id, [
  { role: "system", content: "You are a friendly shopkeeper." },
  { role: "user", content: "What do you sell?" },
], { maxTokens: 256, temperature: 0.7 })) {
  if (chunk.text) process.stdout.write(chunk.text);
  if (chunk.done) console.log("\nstats:", chunk.stats);
}

// 4b. …or build a Character with a persistent system prompt and tools:
const npc = engine.createCharacter({
  modelId: handle.id,
  systemPrompt: "You are a friendly shopkeeper in a fantasy village.",
  temperature: 0.7,
  maxTokens: 256,
  tools: [{
    name: "check_inventory",
    description: "Check if an item is in stock",
    parameters: {
      item: { type: "string", required: true, description: "The item to check" },
    },
    handler: async (args) => db.query(args.item as string),
  }],
});

for await (const token of npc.chat("What do you sell?")) {
  dialogueBox.addText(token);
}

await engine.removeCharacter(npc.id);
await engine.shutdown();
```

> **Heads-up.** The `loadModelFromBuffer` factory creates the engine for you.
> If you need to wire several models against a shared engine instance, build
> additional models with the same pattern and reuse the returned `engine`
> reference, or pre-build the inference pipeline by hand and call
> `engine.adoptPreloadedModel(name, pipeline)` instead.

## Architecture

The TypeScript orchestration layer sits on top of two interchangeable
inference backends: a WASM-compiled `ggml-webgpu` core for quantized
production models, and a pure-WGSL path for tiny models that bypasses WASM
entirely. Both backends share the same tokenization, sampling, streaming,
and scheduling infrastructure.

```mermaid
graph TD
    API[Developer API<br/>WebLLM · Character · ToolSystem]
    Orchestration[TypeScript Orchestration<br/>Scheduler · KVCache · StreamRouter · MemoryPool<br/>GGUF Parser · Tokenizer · Sampler · GameLoop]
    WASM[ggml-webgpu WASM<br/>Quantized ops, production path]
    WGSL[Lightweight WGSL Path<br/>Pure TypeScript + WGSL<br/>Embeddings, Classifiers, Tiny models]
    Runtime[WebGPU Runtime<br/>Device · Queue · Pipeline Cache]

    API --> Orchestration
    Orchestration --> WASM
    Orchestration --> WGSL
    WASM --> Runtime
    WGSL --> Runtime

    style API fill:#0d47a1,stroke:#2196f3,stroke-width:3px,color:#ffffff
    style Orchestration fill:#1b5e20,stroke:#4caf50,stroke-width:2px,color:#ffffff
    style WASM fill:#e65100,stroke:#ff9800,stroke-width:2px,color:#ffffff
    style WGSL fill:#4a148c,stroke:#9c27b0,stroke-width:2px,color:#ffffff
    style Runtime fill:#37474f,stroke:#90a4ae,stroke-width:2px,color:#ffffff
```

## API Overview

| API | Description |
|-----|-------------|
| `WebLLM` | Main engine — initialization, model loading, character management |
| `Character` | Chat persona with system prompt, tools, and streaming output |
| `CharacterManager` | Lifecycle management for character instances |
| `ToolSystem` | Function / tool calling with XML and JSON pattern parsing |
| `Tokenizer` | SPM and BPE tokenization with encode / decode |
| `Sampler` | Token sampling with temperature, top-k, top-p, repetition penalty |
| `Generator` | Autoregressive generation loop with async generators |
| `StreamRouter` | Fan-out token streaming to multiple consumers with backpressure |
| `GgufParser` | GGUF binary format parser for model files |
| `ModelLoader` | Model loading with hyperparameter and tokenizer extraction |
| `KVCache` | Paged KV cache with multi-sequence sharing |
| `InferenceSession` | Per-session inference state tracking |
| `Scheduler` | Priority-based cooperative task scheduler |
| `MemoryPool` | GPU buffer allocation with pressure-based eviction |
| `ModelManager` | Multi-model lifecycle and memory coordination |
| `PipelineCache` | IndexedDB-backed WebGPU pipeline cache |
| `GameLoop` | Frame-budget-aware game loop for inference ticks |
| `GgmlWasm` | WebAssembly bridge for ggml-webgpu tensor operations |
| `LightweightModel` | Pure WGSL inference for small models |
| `engine.removeCharacter(id)` | Remove a registered character by id |
| `engine.shutdown()` | Release GPU buffers, dispose inference engines, and shut down WASM modules for all loaded models |

## Embeddings

`engine.embed(modelId, text)` returns an L2-normalized `Float32Array` for use
in semantic search, clustering, or RAG pipelines. The method dispatches across
three tiers in priority order:

| Tier | Model type | Example | Quality |
|------|-----------|---------|---------|
| 1 — Encoder | Bidirectional BERT/RoBERTa | `bge-large-en-v1.5` | Highest (purpose-built) |
| 2 — Causal-embedder | Causal-LM fine-tuned for retrieval | `qwen3-embedding-0.6b-hyb` | High (MTEB-competitive) |
| 3 — Chat-model (bucket D) | Chat model with `embeddingCapable: true` | `qwen3-8b-iq3m` | Good (5-15% MTEB delta vs tier 2) |

**Which tier should I use?**

- **General-purpose retrieval** — use a dedicated encoder (tier 1) or
  causal-embedder (tier 2). They are purpose-trained for semantic similarity
  and deliver the best MTEB scores.
- **In-domain agent retrieval with a single loaded model** — bucket D (tier 3)
  lets the chat model already in memory serve embedding queries without loading
  a second model. The 5-15% MTEB delta is acceptable for most agent dialogue
  and episodic memory use cases; the VRAM savings are significant on the
  16 GB floor.
- **Uncertainty** — benchmark both on your workload. The dispatch is
  transparent: load only the models you need and the right tier is used
  automatically.

**Registering a bucket D model** — set `embeddingCapable: true` in the model's
registration entry in `eval/models.ts`. Only models that have passed the parity
gate (≥ 0.90 cosine similarity against a dedicated embedder on the canonical
benchmark suite) are eligible. See
`eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md` for the current list.

## Conversation persistence

Conversations and their KV state evaporate on page reload. Apps that
want to preserve conversation state across reloads use the
`exportConversation` / `importConversation` engine primitives plus the
optional `IndexedDBConversationStore` helper.

```ts
import { WebLLM } from "@paulrobello/webllm";
import { IndexedDBConversationStore } from "@paulrobello/webllm/persistence";

const webllm = await WebLLM.init({ memoryBudget: 8 * 1024 ** 3, worker: true });
const { handle: model } = await webllm.loadModelFromUrl(url, "qwen3-8b");
const store = new IndexedDBConversationStore("my-app-conversations");

let conv;
const blob = await store.get("user-42-session");
try {
  conv = blob
    ? await webllm.importConversation(model.id, blob)
    : await webllm.createConversation(model.id);
} catch (e) {
  // IncompatibleConversationError or CorruptBlobError → discard, restart
  await store.delete("user-42-session");
  conv = await webllm.createConversation(model.id);
}

// Per turn, after chatCompletion settles:
const fresh = await webllm.exportConversation(conv);
await store.put("user-42-session", fresh);
```

Apps that need OPFS, server-side sync, or encrypted-at-rest implement
their own store against the same `Uint8Array` contract — the engine
primitives don't depend on any specific storage backend.

The wire format is `magic[4] + uint32 LE headerLen + JSON header + raw kvBytes`
with magic bytes `WLKV`. The header carries an integer `schemaVersion`,
a `ModelFingerprint` (architecture, vocab, layer/head shape, RoPE base,
quantization, tokenizer hash), the conversation options, the token-ID
prefix, the KV byte size, and a save timestamp. `importConversation`
refuses any blob whose fingerprint doesn't match the loaded model and
throws `IncompatibleConversationError` (with a `reason` distinguishing
schema/fingerprint/tokenizer mismatch). Corrupt bytes throw
`CorruptBlobError`. Quota errors from the IndexedDB helper surface as
`PersistenceQuotaError`. See
[`docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`](docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md)
for the full taxonomy and worker-mode marshaling details.

## Development

The Makefile is the single source of truth for tooling. `make help` lists
every target with descriptions.

```bash
make install          # Install dependencies (bun install)
make checkall         # fmt + lint + typecheck + test — the ship gate
make test             # Run the Bun test suite
make build            # Bundle src/ into dist/
make wasm-build       # Rebuild the ggml-webgpu WASM (requires emsdk)
```

A single test: `bun test tests/<file>.test.ts` or
`bun test -t "<pattern>"`.

### Browser smoke test

```bash
make smoke-serve      # Build + serve smoke-test/ on http://localhost:8031
make smoke-open       # Open the smoke-test page in the default browser
```

The smoke-test page accepts URL overrides for `thinking`, `ctx`, `max`,
`temp`, `topK`, `topP`, `rep`, `seed`, `prompt`, and `profile` — see
`smoke-test/real-model-page.js` for the full parser.

### Benchmarks

```bash
make bench-perf                       # Mitata micro-benchmarks (no browser)
make bench-inference                  # End-to-end Chrome inference perf
make bench-chat-smoke-matrix          # Default browser-driven chat matrix
make bench-chat-smoke-matrix-full     # Full matrix incl. Qwen3 thinking-on
make bench-browser-eval PROFILE=<p>   # Real-browser accuracy eval for one profile (needs dashboard)
make bench-full                       # Speed + accuracy across the full profile set (needs dashboard)
```

Browser-driven targets automatically restart a fresh smoke-test server each
run. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for methodology and
metric definitions.

## Evaluation & Live Dashboard

The repo ships a Bun-backed SSE dashboard at
[`smoke-test/dashboard.html`](smoke-test/dashboard.html) for comparing runs
across models, profiles, and sampling parameters in real time.

```bash
make dashboard-serve   # SSE backend on http://localhost:8033, SQLite-persisted
```

Point browser benches at the dashboard with `WEBLLM_LIVE_BENCH_URL`:

```bash
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \
  bun run eval/chat-smoke-matrix.ts --profiles llama-vs-qwen
```

Each run also writes a JSON record to `eval/reports/smoke-runs/` as a
durable archive independent of the dashboard's SQLite store.

```mermaid
graph LR
    Matrix[chat-smoke-matrix]
    Smoke[chat-smoke run]
    Page[Smoke page in Chrome<br/>driven by agentchrome]
    JSON[eval/reports/smoke-runs/<br/>JSON per run]
    Backend[Live SSE backend<br/>eval/live-server.ts · port 8033]
    DB[(SQLite<br/>smoke-runs.db)]
    Dashboard[Dashboard UI<br/>smoke-test/dashboard.html]

    Matrix --> Smoke
    Smoke --> Page
    Page --> Smoke
    Smoke --> JSON
    Smoke -- POST /ingest --> Backend
    Backend --> DB
    Backend -- SSE /stream --> Dashboard

    style Matrix fill:#0d47a1,stroke:#2196f3,stroke-width:2px,color:#ffffff
    style Smoke fill:#0d47a1,stroke:#2196f3,stroke-width:2px,color:#ffffff
    style Page fill:#1b5e20,stroke:#4caf50,stroke-width:2px,color:#ffffff
    style JSON fill:#37474f,stroke:#90a4ae,stroke-width:2px,color:#ffffff
    style Backend fill:#e65100,stroke:#ff9800,stroke-width:3px,color:#ffffff
    style DB fill:#4a148c,stroke:#9c27b0,stroke-width:2px,color:#ffffff
    style Dashboard fill:#4a148c,stroke:#9c27b0,stroke-width:2px,color:#ffffff
```

Profiles (`eval/smoke-profiles.ts`) pin `{ model, thinking, temperature,
topK, topP, repetitionPenalty, seed, contextLength, maxTokens, prompt }`
for reproducible comparison. Profile sets like `llama-vs-qwen`,
`temperature-sweep`, and `thinking-modes` group related profiles for
one-command sweeps.

## Related Documentation

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — benchmark methodology and metrics
- [`docs/DOCUMENTATION_STYLE_GUIDE.md`](docs/DOCUMENTATION_STYLE_GUIDE.md) — documentation conventions
- [`CLAUDE.md`](CLAUDE.md) — repo guidance for Claude Code sessions

## License

MIT
