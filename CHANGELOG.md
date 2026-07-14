# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

## [0.1.0] - 2026-07-14

Initial public shape of `@paulrobello/webllm`. This is a coarse summary of
the state at the first tagged version; it is not a full retroactive
history. The commit log and `TODO_ARCHIVE.md` carry the per-feature detail.

### Added

- **WebGPU inference over a patched `ggml-webgpu`** — quantized GGUF models
  run in-browser via a WASM-compiled llama.cpp backend on branch
  `webllm-browser-patches`; a pure-WGSL path handles sub-50M-parameter
  models without WASM.
- **Multi-architecture forward graph** — Llama, Qwen, Mistral, Phi-3,
  Gemma 2, Gemma 4 E2B (PLE + dual-RoPE + shared-KV + SWA), BERT-family
  encoders (Arctic-Embed, BGE, Nomic, Jina v2), and the Qwen3-Embedding
  causal-embedder.
- **Three-tier `engine.embed()` dispatch** — encoder → causal-embedder →
  chat-model tap ("bucket D"), gated per-model by `embeddingCapable: true`
  with parity gates against PyTorch references.
- **Character system** — personas with system prompts, streaming chat, and
  tool / function calling (XML + JSON parsing).
- **Worker mode** — off-main-thread WebGPU + WASM via a typed RPC layer
  with an error codec that reconstructs the typed error hierarchy across
  `postMessage`.
- **Conversation persistence** — `exportConversation` /
  `importConversation` over a `WLKV` wire format with
  `schemaVersion: 1` (see the [Persistence compatibility
  surface](#persistence-compatibility-surface) note below), plus an
  optional `IndexedDBConversationStore` helper.
- **MEMORY64 / wasm64 support** — models larger than 3.5 GiB route to
  `webllm-wasm-mem64.{js,wasm}` automatically.
- **Evaluation framework** — five dimensions (tool-calling, reasoning,
  instruction-following, semantic-reasoning, embedding) with 56 tasks and
  nine scoring methods, driven through real WebGPU via Chrome.
- **Live SSE + SQLite dashboard** — real-time run comparison across models,
  profiles, and sampling parameters, with backfill via `make import-reports`.
- **30 registered benchmark models** across ultrafast / fast / balanced /
  quality tiers, including IQ3_M 8B entries and a hybrid-quant
  Qwen3-Embedding.
- **Greedy temperature=0 accuracy doctrine** — single-pass stable signal
  for accuracy benches; bench-session envelope for dashboard rollup.

### Changed

- Adopted the `webllm-browser-patches` llama.cpp branch carrying ten local
  patches (stack safety, ASYNCIFY bundle, request-based async readback,
  profiling, u32 UB fix, JSPI `wgpu::WaitAny`). Rebases are classified
  into §27 / §28 / §32 templates and documented per-cycle in
  `docs/LLAMA_CPP_PATCHES.md`.

### Persistence compatibility surface

The conversation-persistence wire format carries an integer
`schemaVersion` in its header (currently `1`). A schema bump is a breaking
change for previously-saved blobs and **must** receive a CHANGELOG entry
under a new version section. `importConversation` refuses blobs whose
schema or model fingerprint does not match the loaded model and throws
`IncompatibleConversationError` with a `reason` distinguishing
schema/fingerprint/tokenizer mismatch; corrupt bytes throw
`CorruptBlobError`.

[Unreleased]: https://github.com/paulrobello/webllm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paulrobello/webllm/releases/tag/v0.1.0
