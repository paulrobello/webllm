# @paulrobello/webllm

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

High-performance LLM inference in the browser via WebGPU, backed by llama.cpp's ggml-webgpu backend. Supports hierarchical multi-model scheduling with frame-budget-aware execution for interactive applications.

## Features

- **GGUF Model Parsing** — Read quantized GGUF binary format directly in the browser
- **Multi-Model Scheduling** — Priority-based cooperative scheduler with configurable frame budgets
- **KV Cache Management** — Paged KV cache with multi-sequence sharing and cross-session prompt caching
- **Character System** — Personas with system prompts, streaming chat, and tool/function calling
- **Lightweight WGSL Path** — Pure TypeScript + WGSL shaders for sub-50M parameter models (no WASM)
- **Memory Management** — GPU buffer pool with pressure detection and priority-based eviction
- **Game Loop Integration** — `requestAnimationFrame`-aware scheduling for real-time applications
- **Tokenization** — SentencePiece (SPM) and Byte Pair Encoding (BPE) tokenizers

## Installation

```bash
bun add @paulrobello/webllm
```

## Quick Start

```typescript
import { WebLLM } from "@paulrobello/webllm";

// Initialize the engine
const engine = await WebLLM.init({
  device: adapter.requestDevice(),
  cacheDir: "indexeddb://webllm-cache",
  memoryBudget: 2048 * 1024 * 1024, // 2GB VRAM
  frameBudgetMs: 8,
});

// Load a quantized model
const model = await engine.loadModel("llama-3.2-3b-q4_k_m.gguf", {
  priority: 0,
  contextLength: 4096,
});

// Create a character with streaming chat
const npc = engine.createCharacter({
  modelId: model.id,
  systemPrompt: "You are a friendly shopkeeper in a fantasy village.",
  temperature: 0.7,
  maxTokens: 256,
  tools: [{
    name: "check_inventory",
    description: "Check if an item is in stock",
    parameters: {
      item: { type: "string", required: true, description: "The item to check" },
    },
    handler: async (args) => db.query(args.item),
  }],
});

// Stream responses
for await (const token of npc.chat("What do you sell?")) {
  dialogueBox.addText(token);
}

// Cleanup
await engine.removeCharacter(npc.id);
await engine.shutdown();
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Developer API                      │
│          WebLLM │ Character │ ToolSystem             │
├─────────────────────────────────────────────────────┤
│              TypeScript Orchestration                │
│  Scheduler │ KVCache │ StreamRouter │ MemoryPool    │
│  GGUF Parser │ Tokenizer │ Sampler │ GameLoop       │
├──────────────────┬──────────────────────────────────┤
│  ggml-webgpu     │  Lightweight WGSL Path           │
│  (WASM core)     │  (Pure TypeScript + WGSL)        │
│  Quantized ops   │  Embeddings, Classifiers, Tiny   │
├──────────────────┴──────────────────────────────────┤
│                  WebGPU Runtime                      │
│          Device │ Queue │ Pipeline Cache             │
└─────────────────────────────────────────────────────┘
```

## API Overview

| Class | Description |
|-------|-------------|
| `WebLLM` | Main engine — initialization, model loading, character management |
| `Character` | Chat persona with system prompt, tools, and streaming output |
| `CharacterManager` | Lifecycle management for character instances |
| `ToolSystem` | Function/tool calling with XML and JSON pattern parsing |
| `Tokenizer` | SPM and BPE tokenization with encode/decode |
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

## Development

```bash
bun install                  # Install dependencies
bun test                     # Run tests
bun run bench                # Run benchmarks
bun run checkall             # Format, lint, typecheck, test
```

## License

MIT
