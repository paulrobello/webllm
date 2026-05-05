# Tier 3 P2 — Causal-LM Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `src/inference/model-inference.ts` (~2890 LOC) and route causal-LM forward through `webllm_decode`. Rewrite `src/models/model-loader.ts` to query upstream metadata via the bridge instead of re-parsing GGUF in TS. Flip remaining `Tokenizer` callers to `LlamaTokenizer`. Net delete: ~3400 LOC of TS plus retirement of BPE/WordPiece encoders. Pass C-strict parity (greedy accuracy within sampling noise, decode tok/s within ±10%) on the canonical 6.

**Architecture:** A new `LlamaDecodeWrapper` class wraps a `(model, ctx)` handle pair from the bridge and exposes the *same public surface* `ModelInference` exposed (`forward`, `loadWeights`, `initKVCache`, `resetKVCache`, `serializeKVCache`, `loadKVCache`, `embed`, `cachedTokenCount`, `maxContextLength`). `loadWeights` is a no-op (weights are already on GPU after `webllm_load_model`); `initKVCache` resolves to `webllm_create_context`. `forward` translates `(tokenIds, positions)` to `bridge.decode(ctx, tokenIds, pastLen)` after asserting `positions[i] === pastLen + i`. `embed` lazily creates a *second* context with `embeddings=true` + `pooling=LAST` (Bucket-D self-embed survives natively). `serializeKVCache` / `loadKVCache` use `llama_state_seq_*` for opaque arch-portable round-trip. `model-loader.ts` collapses to a thin metadata pass-through that loads via `webllm_load_model` then queries `webllm_get_metadata` + typed hyperparam accessors. `speculative.ts` continues to compile via a local structural interface (the runtime path is already gated behind `SpeculativeDecodingReservedError`); spec-decode tests are skipped with a "deferred to P5" marker. `tokenizer.ts` retains only the `StreamingDecoder` — all BPE/SPM/WordPiece encoder code deletes.

**Tech Stack:** C++ (libllama bridge: 11 new exports, ~150 LOC), TypeScript (new wrapper ~300 LOC, loader rewrite ~120 LOC, engine.ts dispatch flip ~80 LOC), Bun (unit tests), agentchrome (smoke verification), `make bench-*` + `make smoke-bench` for parity gate.

**Spec:** [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../specs/2026-05-05-tier3-llama-decode-migration-design.md) §P2.

**Predecessor:** P1 closed PASS 2026-05-05 — 1000/1000 prompts byte-exact across 5 vocabs in a single-page parity run; `LlamaBridge` (`src/inference/llama-bridge.ts`) and `LlamaTokenizer` (`src/inference/llama-tokenizer.ts`) live; `webllm_load_model` uses `use_mmap=false` + `std::remove()` so cross-model loads survive the wasm32 4 GiB cap. Closure: [`eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md`](../../../eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md).

---

## File Structure

**Create:**
- `src/inference/llama-decode-wrapper.ts` — `LlamaDecodeWrapper` class. ~300 LOC. Public surface mirrors `ModelInference` minus internal graph-build helpers.
- `tests/llama-decode-wrapper.test.ts` — Bun-side unit tests for wrapper KV invariants and embed dispatch (~150 LOC).
- `eval/reports/p2-causal-migration-2026-05-05/PRE-MIGRATION-BASELINE.md` — captured greedy bench + smoke-bench numbers per canonical-6 model, taken on `main` before any P2 commit lands. Ground truth for the parity gate.
- `eval/reports/p2-causal-migration-2026-05-05/SUMMARY.md` — closure report (Task 12 only).

**Modify:**
- `src/wasm/webgpu-bridge.cpp` — extend the existing `extern "C"` block with: `webllm_get_metadata`, `webllm_n_ctx_train`, `webllm_n_embd`, `webllm_n_layer`, `webllm_n_head`, `webllm_n_head_kv`, `webllm_n_ctx`, `webllm_kv_clear`, `webllm_kv_seq_rm`, `webllm_state_seq_get_size`, `webllm_state_seq_get_data`, `webllm_state_seq_set_data`, `webllm_get_embeddings`, plus a new flags arg on `webllm_decode` (or a sibling `webllm_decode_ext`).
- `src/wasm/CMakeLists.txt` — extend `EXPORTED_FUNCTIONS` (sync) + `JSPI_EXPORTS` (anything that touches WebGPU readback).
- `src/inference/llama-bridge.ts` — extend `LlamaBridge` interface, `RawLlamaModule` shape, and the `createLlamaBridge` factory with bindings for every new export.
- `src/models/model-loader.ts` — rewrite. Drop `parseModel(data: Uint8Array): ParsedModel`. New entry point `loadModelMetadata(bridge: LlamaBridge, model: number): LoadedModelMetadata` returns hyperparams + chat template only (no `tokenizerConfig` — `LlamaTokenizer` replaces it).
- `src/core/engine.ts` — flip causal-LM branch in `loadModelFromBuffer` / `loadModelFromArchive` to construct `LlamaDecodeWrapper` instead of `ModelInference`. Replace `new Tokenizer(parsed.tokenizerConfig)` with `new LlamaTokenizer(bridge, modelHandle, { chatTemplate, encoderOnly: false })`. Update `inferenceEngines` map type to accept the wrapper. Update `adoptPreloadedModel` typing. Update `__debugInferenceForModel` return type.
- `src/inference/tokenizer.ts` — delete BPE/SPM/WordPiece encoders + pre-tokenizer regex tables; **retain** `StreamingDecoder` class and `TokenAttribute` enum (still referenced by Bucket-D distinguishability tests, retained for symmetry across phases). Drop `Tokenizer` class entirely. The file shrinks from ~1010 LOC to ~150 LOC.
- `src/inference/speculative.ts` — replace `import type { ModelInference }` with a local structural interface (`SpecDecodeForwardPass`) declaring only the methods spec-decode reads. No behavioral change.
- `src/index.ts` — drop `ModelInference` export; add `LlamaDecodeWrapper` export. Drop `Tokenizer` export; `LlamaTokenizer` export already exists.
- `tests/speculative-integration.test.ts`, `tests/speculative-rejection.test.ts` — wrap entire suite in `describe.skip("…deferred to P5", …)` with a TODO comment pointing at the spec §P5 step. Re-enabled when P5 lands.
- `TODO.md` — add a "P2 closed" closure stub once Task 12 lands; archive old P0/P1 detail blocks via the cadence rule in CLAUDE.md.

**Delete:**
- `src/inference/model-inference.ts` (~2890 LOC).

**Untouched (P3+ phases):**
- `src/inference/encoder-inference.ts` — encoders still go through legacy graph builder. P3 deletes.
- `src/inference/causal-embedder-inference.ts` — `qwen3-embedding`-style models still go through legacy. P4 deletes.
- `src/models/gguf-parser.ts` / `gguf-types.ts` — still used by encoder + causal-embedder loaders for now. Reduced footprint is captured in P3/P4.

---

## Pre-flight (Task 0)

### Task 0: Capture pre-migration baseline + verify P1 tip is green

**Files:**
- Read: `eval/models.ts` (canonical-6 inventory)
- Read: `TODO.md` header block (declared canonical baselines)
- Create: `eval/reports/p2-causal-migration-2026-05-05/PRE-MIGRATION-BASELINE.md`

- [ ] **Step 1: Confirm tree state**

Run:
```bash
git log -1 --oneline && git status --short
```
Expected: HEAD at or beyond `72cd44c docs(TODO): close P0+P1, surface P2 quickstart`. Working tree clean.

- [ ] **Step 2: Confirm bench harness shape**

Run:
```bash
make help | grep -E "bench|smoke" | head -20
```
Expected: targets `bench-full`, `smoke-bench`, `dashboard-serve`, `import-reports` listed.

- [ ] **Step 3: Run greedy accuracy bench on the canonical 6**

Per the 2026-05-04 doctrine in `CLAUDE.md` ("Greedy by default for accuracy bench"), this baseline is captured at `--eval-temperature 0`. Skip if a run from the last 7 days under identical conditions already exists in `eval/reports/`.

Run (background, captures into the live dashboard):
```bash
make dashboard-serve > /tmp/dash-baseline.log 2>&1 &
sleep 3
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 make bench-canonical EVAL_TEMP=0 2>&1 | tee /tmp/bench-baseline.log
```
Expected: 6 models complete with overall accuracy printed per-model. The header-block accuracy numbers in TODO.md are the floor.

- [ ] **Step 4: Run smoke-bench (3-run median) on the canonical 6**

Run:
```bash
make smoke-bench PERF_RUNS=3 PERF_MODELS=canonical-6 2>&1 | tee /tmp/smoke-bench-baseline.log
```
Expected: 6 tok/s medians within ±5% of the TODO.md profile-mode pins (the 2026-05-01 post-rebase line `81.8 / 67.5 / 43.9 / 29.6 / 23.4 / 22.0` is the comparison anchor; profile-mode numbers are 15-28% lower than non-profile due to per-dispatch timestamp sampling, so this baseline is its own series).

- [ ] **Step 5: Write `PRE-MIGRATION-BASELINE.md`**

The file MUST contain:
1. **Header**: capture date (2026-05-05 or later), git SHA at capture time, branch (`main`), eval temperature setting (`0`).
2. **Accuracy table**: one row per canonical-6 model, columns `accuracy / 36`, `pass@1 %`, `dimensions accurate`. Pulled from `bench-canonical` console output or `eval/reports/<auto-generated>/SUMMARY.md`.
3. **Decode-throughput table**: one row per canonical-6 model, columns `decode tok/s (3-run median)`, `delta vs TODO.md pin`.
4. **Parity bar reminder**: post-migration must satisfy:
   - Greedy accuracy within sampling noise (±2 questions of 36 = ±5.5% absolute) per row.
   - Decode tok/s within ±10% per row vs this baseline.

- [ ] **Step 6: Commit**

```bash
git add -f eval/reports/p2-causal-migration-2026-05-05/PRE-MIGRATION-BASELINE.md
git commit -m "$(cat <<'EOF'
docs(p2): capture pre-migration baseline for parity gate

Greedy accuracy + 3-run smoke-bench median on canonical 6 captured
on `main` at HEAD before any P2 wrapper / loader / engine changes
land. The closure parity gate compares post-migration measurements
to this file row-by-row.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2
EOF
)"
```

Expected: clean commit, baseline file tracked.

---

## Task 1: Bridge — model metadata exports

**Goal:** Add `webllm_get_metadata` (arbitrary string keys via `llama_model_meta_val_str`) plus typed hyperparam accessors. The TS loader needs these to fill `ModelHyperparams` without re-parsing GGUF.

**Files:**
- Modify: `src/wasm/webgpu-bridge.cpp` — append to the `extern "C"` block, after `webllm_token_eos` (line 581).
- Modify: `src/wasm/CMakeLists.txt` — extend `EXPORTED_FUNCTIONS` (these are sync; do **not** add to `JSPI_EXPORTS`).

- [ ] **Step 1: Add metadata accessors to the bridge**

Insert after `webllm_token_eos`:

```cpp
// Get a metadata string by key. Mirrors llama_model_meta_val_str:
// returns the length of the string on success, -1 on missing key.
// The output buffer must be sized for the value + null terminator.
// Caller-side retry-on-truncation uses the upstream pattern (call
// once with a small buffer to read the required length, then again
// with the right size).
int32_t webllm_get_metadata(
    void* model_handle,
    const char* key,
    char* buf,
    int32_t buf_size)
{
    if (!model_handle || !key || !buf || buf_size <= 0) return -1;
    auto* model = static_cast<llama_model*>(model_handle);
    return llama_model_meta_val_str(model, key, buf, (size_t) buf_size);
}

// Typed hyperparam accessors. Each returns an int32_t; -1 on
// missing model handle. These read from llama.cpp's parsed
// hyperparams (faster than meta_val_str, no string round-trip).
int32_t webllm_n_ctx_train(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_ctx_train(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_embd(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_embd(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_layer(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_layer(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_head(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_head(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_head_kv(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_head_kv(static_cast<llama_model*>(model_handle));
}
// Per-context KV-cache size in tokens (= the n_ctx the wrapper passed
// to webllm_create_context, possibly clamped to model's n_ctx_train).
int32_t webllm_n_ctx(void* ctx_handle) {
    if (!ctx_handle) return 0;
    return (int32_t) llama_n_ctx(static_cast<llama_context*>(ctx_handle));
}
```

- [ ] **Step 2: Extend `EXPORTED_FUNCTIONS` in CMakeLists.txt**

In `src/wasm/CMakeLists.txt` extend the existing `EXPORTED_FUNCTIONS` `string(CONCAT …)` block. Add the new symbols at the end (after `_webllm_token_eos`):

```cmake
        "_webllm_load_model,_webllm_free_model,"
        "_webllm_create_context,_webllm_free_context,"
        "_webllm_decode,_webllm_get_logits,"
        "_webllm_n_vocab,"
        "_webllm_tokenize,_webllm_detokenize,"
        "_webllm_token_bos,_webllm_token_eos,"
        "_webllm_get_metadata,"
        "_webllm_n_ctx_train,_webllm_n_embd,_webllm_n_layer,"
        "_webllm_n_head,_webllm_n_head_kv,_webllm_n_ctx"
    )
```

Do NOT add these to `JSPI_EXPORTS` — none of them touch WebGPU async readback.

- [ ] **Step 3: Rebuild WASM**

Run:
```bash
make wasm-build 2>&1 | tail -20
```
Expected: build succeeds; new symbols listed in the link summary or absent from the "undefined symbol" warning set. If the build complains about a missing `llama_model_n_*` accessor, the local llama.cpp branch (`webllm-browser-patches`) may be behind upstream — bump and rebase per `docs/LLAMA_CPP_PATCHES.md`.

- [ ] **Step 4: Verify exports landed**

Run:
```bash
grep -E "webllm_get_metadata|webllm_n_ctx_train|webllm_n_embd|webllm_n_layer|webllm_n_head|webllm_n_head_kv|webllm_n_ctx" public/webllm-wasm.js | head
```
Expected: each symbol appears in the bundled JS as `_webllm_<name>` (Emscripten exports them via the underscore prefix).

- [ ] **Step 5: Commit**

```bash
git add src/wasm/webgpu-bridge.cpp src/wasm/CMakeLists.txt public/webllm-wasm.js public/webllm-wasm.wasm
git commit -m "$(cat <<'EOF'
feat(p2): add model-metadata bridge exports

webllm_get_metadata wraps llama_model_meta_val_str. Typed
accessors (n_ctx_train, n_embd, n_layer, n_head, n_head_kv,
n_ctx) avoid string round-trip for the hot hyperparam fields
the rewritten model-loader.ts needs.

Sync exports — no JSPI wrap (no WebGPU readback in any of these).

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 step 2
EOF
)"
```

---

## Task 2: Bridge — KV cache mutation ops

**Goal:** Expose `llama_kv_cache_seq_*` so the wrapper's `resetKVCache` and `truncateKVCache` work on the new path.

**Files:**
- Modify: `src/wasm/webgpu-bridge.cpp`
- Modify: `src/wasm/CMakeLists.txt`

- [ ] **Step 1: Locate the upstream KV API names**

Run:
```bash
grep -E "llama_memory_seq_rm|llama_memory_clear|llama_kv_cache_seq_rm|llama_kv_cache_clear" ~/Repos/llama.cpp/include/llama.h | head -10
```
The upstream API is in flux (the 2026-04 rebase moved from `llama_kv_cache_*` to `llama_memory_*`). Whichever the local branch exposes is canonical. Use the helper macros llama.cpp ships (e.g. `llama_kv_cache_seq_rm` may be a backwards-compat shim).

- [ ] **Step 2: Add KV ops**

Append after the metadata exports from Task 1:

```cpp
// Drop tokens [p0, p1) for seq_id from the context's KV cache.
// p1 = -1 means "to the end". Used by truncateKVCache and by
// loadKVCache after a full state-set followed by truncation.
void webllm_kv_seq_rm(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx_handle) return;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1);
}

// Clear all sequences. Equivalent to a full resetKVCache.
void webllm_kv_clear(void* ctx_handle) {
    if (!ctx_handle) return;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, /*data=*/true);
}
```

If the local branch's API names differ, substitute them. The behavior is unambiguous (drop seq tokens, clear cache).

- [ ] **Step 3: Extend EXPORTED_FUNCTIONS**

Append to the existing block:
```cmake
        "_webllm_kv_seq_rm,_webllm_kv_clear"
```

These are sync — no JSPI wrap.

- [ ] **Step 4: Rebuild and verify**

```bash
make wasm-build 2>&1 | tail -10
grep -E "webllm_kv_(seq_rm|clear)" public/webllm-wasm.js
```
Expected: build green, both symbols in bundle.

- [ ] **Step 5: Commit**

```bash
git add src/wasm/webgpu-bridge.cpp src/wasm/CMakeLists.txt public/webllm-wasm.js public/webllm-wasm.wasm
git commit -m "$(cat <<'EOF'
feat(p2): expose KV cache mutation ops

webllm_kv_seq_rm + webllm_kv_clear let the LlamaDecodeWrapper
implement resetKVCache and truncateKVCache on top of the upstream
memory API. Both are sync; no JSPI wrap.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 (bridge)
EOF
)"
```

---

## Task 3: Bridge — KV state serialize/restore

**Goal:** Expose `llama_state_seq_get_size`, `llama_state_seq_get_data`, `llama_state_seq_set_data` so prefix-cache persistence (engine.ts:864 + 1125) keeps working through the wrapper.

**Files:**
- Modify: `src/wasm/webgpu-bridge.cpp`
- Modify: `src/wasm/CMakeLists.txt`

- [ ] **Step 1: Add state-seq exports**

Append after the KV ops from Task 2:

```cpp
// Get the byte size needed to serialize seq_id's KV state.
// Returns 0 on failure (or empty seq).
int32_t webllm_state_seq_get_size(void* ctx_handle, int32_t seq_id) {
    if (!ctx_handle) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_get_size(ctx, (llama_seq_id) seq_id);
}

// Copy seq_id's KV state into a caller-provided buffer.
// Returns the number of bytes written, 0 on failure.
// Caller must size dst to at least webllm_state_seq_get_size.
int32_t webllm_state_seq_get_data(
    void* ctx_handle,
    void* dst,
    int32_t size,
    int32_t seq_id)
{
    if (!ctx_handle || !dst || size <= 0) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_get_data(
        ctx, (uint8_t*) dst, (size_t) size, (llama_seq_id) seq_id);
}

// Restore seq_id's KV state from a caller-provided buffer.
// Returns positive on success, 0 on failure (per upstream contract).
// The buffer must have been produced by webllm_state_seq_get_data
// from a context with the SAME model + n_ctx + flash_attn flag.
int32_t webllm_state_seq_set_data(
    void* ctx_handle,
    const void* src,
    int32_t size,
    int32_t dest_seq_id)
{
    if (!ctx_handle || !src || size <= 0) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_set_data(
        ctx, (const uint8_t*) src, (size_t) size, (llama_seq_id) dest_seq_id);
}
```

- [ ] **Step 2: Extend EXPORTED_FUNCTIONS**

Append:
```cmake
        ",_webllm_state_seq_get_size,_webllm_state_seq_get_data,_webllm_state_seq_set_data"
```

These are sync — no JSPI wrap.

- [ ] **Step 3: Rebuild + verify**

```bash
make wasm-build 2>&1 | tail -10
grep -E "webllm_state_seq_(get_size|get_data|set_data)" public/webllm-wasm.js
```
Expected: 3 symbols in bundle.

- [ ] **Step 4: Commit**

```bash
git add src/wasm/webgpu-bridge.cpp src/wasm/CMakeLists.txt public/webllm-wasm.js public/webllm-wasm.wasm
git commit -m "$(cat <<'EOF'
feat(p2): expose KV state serialize/restore

webllm_state_seq_{get_size,get_data,set_data} replace the
hand-rolled per-tensor backendTensorGet/Set serialization
loops in ModelInference. The blob is opaque (arch + flash_attn
internal format), so the wrapper's serializeKVCache/loadKVCache
become single calls instead of per-layer batched uploads.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 (bridge)
EOF
)"
```

---

## Task 4: Bridge — embeddings readback + per-position logits flag

**Goal:** Add `webllm_get_embeddings` for Bucket-D self-embed via a second context. Optionally extend `webllm_decode` with a flags arg so the wrapper can request all-positions logits in P5; for P2 the existing last-position-only behavior is sufficient, so the flag exists but defaults match current behavior.

**Files:**
- Modify: `src/wasm/webgpu-bridge.cpp`
- Modify: `src/wasm/CMakeLists.txt`

- [ ] **Step 1: Add embeddings readback**

Append after the state-seq exports:

```cpp
// Get pointer to embeddings for the i-th token of the last decode.
// ith=-1 → use llama_get_embeddings (returns the pooled embedding
// when pooling_type != NONE, or the last-position embedding when
// pooling_type == NONE). Otherwise llama_get_embeddings_ith(ith)
// for per-position embeddings (Bucket-D pre-pool tap).
//
// Returns a pointer into ctx-owned memory; valid until the next
// decode call. JS-side caller wraps it as a Float32Array view of
// length n_embd. Caller must NOT free.
const float* webllm_get_embeddings(void* ctx_handle, int32_t ith) {
    if (!ctx_handle) return nullptr;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return ith < 0 ? llama_get_embeddings(ctx) : llama_get_embeddings_ith(ctx, ith);
}
```

- [ ] **Step 2: Extend EXPORTED_FUNCTIONS + JSPI_EXPORTS**

Embeddings readback can suspend (the underlying compute path may finalize an outstanding GPU readback before returning the pointer). To match `webllm_get_logits`'s safe-by-default treatment, list it in **both** `EXPORTED_FUNCTIONS` and `JSPI_EXPORTS`.

In `EXPORTED_FUNCTIONS`:
```cmake
        ",_webllm_get_embeddings"
```
In `JSPI_EXPORTS`:
```cmake
        ",webllm_get_embeddings"
```

- [ ] **Step 3: Rebuild + verify**

```bash
make wasm-build 2>&1 | tail -10
grep webllm_get_embeddings public/webllm-wasm.js
```

- [ ] **Step 4: Commit**

```bash
git add src/wasm/webgpu-bridge.cpp src/wasm/CMakeLists.txt public/webllm-wasm.js public/webllm-wasm.wasm
git commit -m "$(cat <<'EOF'
feat(p2): expose embeddings readback for Bucket-D self-embed

webllm_get_embeddings mirrors webllm_get_logits but reads the
embeddings tensor instead. The wrapper uses it via a second
llama_context configured with embeddings=true + pooling=LAST
to preserve the Bucket-D self-embedding path while
model-inference.ts is being deleted.

JSPI-wrapped because the underlying readback may suspend.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 (Bucket-D preservation)
EOF
)"
```

---

## Task 5: Extend `LlamaBridge` TS interface

**Goal:** Surface every Task 1-4 export through the typed `LlamaBridge` so wrapper code never touches the raw module.

**Files:**
- Modify: `src/inference/llama-bridge.ts`

- [ ] **Step 1: Extend the `LlamaContextParams` and `LlamaBridge` interfaces**

In `src/inference/llama-bridge.ts`, after the existing `LlamaBridge` interface, append the following methods to the interface declaration. Place them in the same shape and JSDoc style as the existing entries:

```typescript
export interface LlamaBridge {
	// … existing entries …

	/** Read a string metadata value by key. Returns null if missing. */
	getMetadata(model: number, key: string): string | null;
	/** Hyperparam accessors. Negative return = missing model handle. */
	nCtxTrain(model: number): number;
	nEmbd(model: number): number;
	nLayer(model: number): number;
	nHead(model: number): number;
	nHeadKv(model: number): number;
	/** Per-context KV size in tokens. */
	nCtx(ctx: number): number;

	/** Drop tokens [p0, p1) for seq_id. p1=-1 means "to the end". */
	kvSeqRm(ctx: number, seqId: number, p0: number, p1: number): void;
	/** Clear all sequences in this context's KV cache. */
	kvClear(ctx: number): void;

	/** Bytes needed to serialize seq_id's KV state. */
	stateSeqGetSize(ctx: number, seqId: number): number;
	/** Copy seq_id's KV state into a freshly-allocated Uint8Array. */
	stateSeqGetData(ctx: number, seqId: number): Uint8Array;
	/**
	 * Restore seq_id's KV state from a previously-captured blob.
	 * Returns true on success. The blob must come from a context
	 * with the SAME model + n_ctx + flash_attn flag — restoring
	 * across mismatched configs is undefined behavior.
	 */
	stateSeqSetData(ctx: number, blob: Uint8Array, destSeqId: number): boolean;

	/**
	 * Read embeddings for the i-th token of the last decode.
	 * ith=-1 → pooled (or last-position when pooling is NONE).
	 * Returns a Float32Array view INTO ctx-owned memory — valid
	 * until the next decode call. Length = nEmbd(model).
	 */
	getEmbeddings(ctx: number, model: number, ith?: number): Promise<Float32Array>;
}
```

- [ ] **Step 2: Extend `RawLlamaModule`**

Add the corresponding raw module methods (mirroring the existing ABI-polymorphic-pointer pattern):

```typescript
interface RawLlamaModule {
	// … existing entries …

	_webllm_get_metadata: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		model: any,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		keyPtr: any,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		bufPtr: any,
		bufSize: number,
	) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_ctx_train: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_embd: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_layer: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_head: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_head_kv: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_ctx: (ctx: any) => number;

	_webllm_kv_seq_rm: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		ctx: any,
		seqId: number,
		p0: number,
		p1: number,
	) => void;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_kv_clear: (ctx: any) => void;

	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_state_seq_get_size: (ctx: any, seqId: number) => number;
	_webllm_state_seq_get_data: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		ctx: any,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		dst: any,
		size: number,
		seqId: number,
	) => number;
	_webllm_state_seq_set_data: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		ctx: any,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		src: any,
		size: number,
		destSeqId: number,
	) => number;

	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_get_embeddings: (ctx: any, ith: number) => any;
}
```

- [ ] **Step 3: Implement the new bindings in `createLlamaBridge`**

Inside the returned object literal, append the implementations. Put them immediately after `tokenEos`. Use the existing `to64`/`from64`/`malloc`/`free` helpers:

```typescript
		getMetadata(model: number, key: string): string | null {
			const utf8 = new TextEncoder().encode(`${key}\0`);
			const keyPtr = malloc(utf8.byteLength);
			if (keyPtr === 0) {
				throw new Error("webllm: bridge_malloc failed for metadata key");
			}
			try {
				mod.HEAPU8.set(utf8, keyPtr);
				// First call sized to 0 → returns required size or -1 if missing.
				const required = mod._webllm_get_metadata(
					to64(model),
					to64(keyPtr),
					to64(0),
					0,
				);
				if (required < 0) return null;
				const cap = required + 1;
				const bufPtr = malloc(cap);
				if (bufPtr === 0) {
					throw new Error("webllm: bridge_malloc failed for metadata buf");
				}
				try {
					const n = mod._webllm_get_metadata(
						to64(model),
						to64(keyPtr),
						to64(bufPtr),
						cap,
					);
					if (n < 0) return null;
					return new TextDecoder().decode(
						new Uint8Array(mod.HEAPU8.buffer.slice(bufPtr, bufPtr + n)),
					);
				} finally {
					free(bufPtr);
				}
			} finally {
				free(keyPtr);
			}
		},

		nCtxTrain(model: number): number {
			return mod._webllm_n_ctx_train(to64(model));
		},
		nEmbd(model: number): number {
			return mod._webllm_n_embd(to64(model));
		},
		nLayer(model: number): number {
			return mod._webllm_n_layer(to64(model));
		},
		nHead(model: number): number {
			return mod._webllm_n_head(to64(model));
		},
		nHeadKv(model: number): number {
			return mod._webllm_n_head_kv(to64(model));
		},
		nCtx(ctx: number): number {
			return mod._webllm_n_ctx(to64(ctx));
		},

		kvSeqRm(ctx: number, seqId: number, p0: number, p1: number): void {
			mod._webllm_kv_seq_rm(to64(ctx), seqId, p0, p1);
		},
		kvClear(ctx: number): void {
			mod._webllm_kv_clear(to64(ctx));
		},

		stateSeqGetSize(ctx: number, seqId: number): number {
			return mod._webllm_state_seq_get_size(to64(ctx), seqId);
		},
		stateSeqGetData(ctx: number, seqId: number): Uint8Array {
			const size = mod._webllm_state_seq_get_size(to64(ctx), seqId);
			if (size === 0) return new Uint8Array(0);
			const ptr = malloc(size);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for state-seq blob");
			}
			try {
				const n = mod._webllm_state_seq_get_data(
					to64(ctx),
					to64(ptr),
					size,
					seqId,
				);
				if (n === 0) {
					throw new Error("webllm: state_seq_get_data returned 0 bytes");
				}
				return new Uint8Array(mod.HEAPU8.buffer.slice(ptr, ptr + n));
			} finally {
				free(ptr);
			}
		},
		stateSeqSetData(ctx: number, blob: Uint8Array, destSeqId: number): boolean {
			if (blob.byteLength === 0) return true;
			const ptr = malloc(blob.byteLength);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for state-seq restore");
			}
			try {
				mod.HEAPU8.set(blob, ptr);
				const n = mod._webllm_state_seq_set_data(
					to64(ctx),
					to64(ptr),
					blob.byteLength,
					destSeqId,
				);
				return n > 0;
			} finally {
				free(ptr);
			}
		},

		async getEmbeddings(
			ctx: number,
			model: number,
			ith = -1,
		): Promise<Float32Array> {
			const ptr = from64(await mod._webllm_get_embeddings(to64(ctx), ith));
			if (ptr === 0) {
				throw new Error("webllm: webllm_get_embeddings returned null");
			}
			const nEmbd = mod._webllm_n_embd(to64(model));
			return new Float32Array(mod.HEAPU8.buffer, ptr, nEmbd);
		},
```

- [ ] **Step 4: Typecheck**

Run:
```bash
make typecheck 2>&1 | tail -20
```
Expected: clean. If new methods conflict with existing names (e.g. an old `nVocab` definition), reconcile by leaving existing names intact and naming new methods to match upstream where there's no clash.

- [ ] **Step 5: Commit**

```bash
git add src/inference/llama-bridge.ts
git commit -m "$(cat <<'EOF'
feat(p2): extend LlamaBridge with metadata, KV ops, state-seq, embeddings

Surface the 13 new exports added in P2 Tasks 1-4. Pure additions —
no behavioral change to existing callers (LlamaTokenizer, P0 spike).

The wrapper class added in Task 6 consumes these.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 (bridge-TS)
EOF
)"
```

---

## Task 6: Create `LlamaDecodeWrapper`

**Goal:** Implement the class that replaces `ModelInference` for causal-LM forward. Public surface mirrors what callers in `engine.ts` (and a few smoke-test pages) read.

**Files:**
- Create: `src/inference/llama-decode-wrapper.ts`

- [ ] **Step 1: Write the wrapper class**

Create `src/inference/llama-decode-wrapper.ts` with the following content. The class is single-responsibility: own the `(model, ctx)` pair and translate the legacy public surface to bridge calls. KV state lives in `llama_context`; `cachedTokenCount` is tracked in TS as a thin mirror because the bridge has no `llama_n_kv_cells_used` accessor (querying it would require another bridge round-trip per generation step).

```typescript
// Tier 3 P2 — LlamaDecodeWrapper replaces ModelInference for the
// causal-LM forward path. The wrapper owns a (model, ctx) pair from
// the bridge and exposes the same public surface engine.ts read on
// the legacy class.
//
// Lifetime: one wrapper per loaded causal-LM model, one main context
// for chat decode (created in initKVCache), and ONE lazily-created
// embedder context for Bucket-D self-embed (configured with
// embeddings=true + pooling=LAST). Both contexts share the same
// llama_model handle.
//
// See:
//   docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2
//   docs/superpowers/plans/2026-05-05-tier3-p2-causal-lm.md

import type { LlamaBridge, LlamaContextParams } from "./llama-bridge.js";

export interface LlamaDecodeWrapperOptions {
	/** Enable flash attention for the main causal context. */
	flashAttn?: boolean;
}

export interface EmbedOptions {
	/** Pooling strategy for Bucket-D self-embed. Default: "last-token". */
	pooling?: "last-token" | "mean";
}

export class LlamaDecodeWrapper {
	readonly bridge: LlamaBridge;
	readonly model: number;
	readonly flashAttn: boolean;

	private mainCtx = 0;
	private embedCtx = 0;
	private nCached = 0;
	private mainCtxTokens = 0;

	constructor(
		bridge: LlamaBridge,
		model: number,
		opts: LlamaDecodeWrapperOptions = {},
	) {
		this.bridge = bridge;
		this.model = model;
		this.flashAttn = opts.flashAttn ?? false;
	}

	/**
	 * No-op. Legacy ModelInference.loadWeights uploaded weights into a
	 * ggml graph; in P2 the weights are already on GPU after
	 * webllm_load_model. The method exists for API symmetry — engine.ts
	 * calls it unconditionally between `new LlamaDecodeWrapper(...)` and
	 * `initKVCache(...)`.
	 */
	loadWeights(): void {
		/* no-op */
	}

	/**
	 * Allocate the main causal context with `nCtx` tokens of KV. Idempotent:
	 * a second call replaces the first context (frees the old one). After
	 * this, `forward()` is callable.
	 */
	async initKVCache(nCtx: number): Promise<void> {
		if (nCtx <= 0) {
			throw new Error(`initKVCache: nCtx must be > 0; got ${nCtx}`);
		}
		if (this.mainCtx) {
			this.bridge.freeContext(this.mainCtx);
			this.mainCtx = 0;
		}
		const params: LlamaContextParams = {
			nCtx,
			embeddings: false,
			poolingType: 0,
			flashAttn: this.flashAttn,
		};
		this.mainCtx = await this.bridge.createContext(this.model, params);
		this.mainCtxTokens = this.bridge.nCtx(this.mainCtx);
		this.nCached = 0;
	}

	/** KV cache tokens currently materialized for the main context. */
	get cachedTokenCount(): number {
		return this.nCached;
	}

	/** Effective context window of the main context (post-clamp). */
	get maxContextLength(): number {
		return this.mainCtxTokens;
	}

	/**
	 * Run a forward pass over `tokenIds` at `positions`. Returns the LAST
	 * position's logits as a Float32Array view INTO ctx-owned memory —
	 * valid until the next forward / embed call. Mirrors the legacy
	 * ModelInference.forward contract (which returned the last
	 * position's logits regardless of input length).
	 *
	 * `positions` must be sequential starting at `cachedTokenCount`; any
	 * other layout indicates a session-tracker bug upstream and throws.
	 * The legacy class enforced the same invariant via its internal
	 * nCached counter; surfacing it as a precondition makes the contract
	 * explicit.
	 */
	async forward(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.mainCtx) {
			throw new Error("forward called before initKVCache");
		}
		if (tokenIds.length === 0) {
			throw new Error("forward called with empty tokenIds");
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forward: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		const pastLen = this.nCached;
		for (let i = 0; i < positions.length; i++) {
			if (positions[i] !== pastLen + i) {
				throw new Error(
					`forward: positions must be sequential from cachedTokenCount=${pastLen}; positions[${i}]=${positions[i]} expected ${pastLen + i}`,
				);
			}
		}
		const status = await this.bridge.decode(this.mainCtx, tokenIds, pastLen);
		if (status !== 0) {
			throw new Error(`forward: webllm_decode returned status ${status}`);
		}
		this.nCached = pastLen + tokenIds.length;
		return await this.bridge.getLogits(this.mainCtx, this.model, -1);
	}

	/** Drop all KV cache state on the main context. */
	resetKVCache(): void {
		if (!this.mainCtx) return;
		this.bridge.kvClear(this.mainCtx);
		this.nCached = 0;
	}

	/**
	 * Drop tokens [keepLen, cachedTokenCount). Used by spec-decode
	 * rollback and prefix-cache mid-conversation truncation.
	 */
	truncateKVCache(keepLen: number): void {
		if (!this.mainCtx) return;
		if (keepLen < 0 || keepLen > this.nCached) {
			throw new Error(
				`truncateKVCache: keepLen=${keepLen} out of range [0, ${this.nCached}]`,
			);
		}
		this.bridge.kvSeqRm(this.mainCtx, 0, keepLen, -1);
		this.nCached = keepLen;
	}

	/**
	 * Serialize the main context's seq=0 KV state for prefix-cache
	 * persistence. The blob is opaque (arch + flash_attn-internal
	 * format); restore it via {@link loadKVCache} on a context with
	 * the same model + n_ctx + flash_attn config.
	 *
	 * Note vs legacy: the `nTokens` argument is retained for API
	 * symmetry with engine.ts callers, but is informational only —
	 * the upstream serializer always captures the full materialized
	 * KV state. Callers that pass `nTokens < cachedTokenCount` should
	 * `truncateKVCache(nTokens)` first if they want a shorter blob.
	 */
	async serializeKVCache(nTokens: number): Promise<Uint8Array> {
		if (!this.mainCtx) {
			throw new Error("serializeKVCache called before initKVCache");
		}
		if (nTokens > this.nCached) {
			throw new Error(
				`serializeKVCache: nTokens=${nTokens} > cachedTokenCount=${this.nCached}`,
			);
		}
		return this.bridge.stateSeqGetData(this.mainCtx, 0);
	}

	/**
	 * Restore main context's seq=0 KV state from a blob produced by
	 * {@link serializeKVCache}, then truncate to `nTokens`. The
	 * `snapshotLen` parameter (default = `nTokens`) records the length
	 * the blob was serialized at — it must be >= nTokens. Truncating
	 * a longer-stored snapshot to a shorter prefix matches the legacy
	 * loadKVCache contract used by engine.ts (prefix-cache reload of
	 * a shared prefix from a longer-stored snapshot).
	 */
	async loadKVCache(
		bytes: Uint8Array,
		nTokens: number,
		snapshotLen?: number,
	): Promise<void> {
		if (!this.mainCtx) {
			throw new Error("loadKVCache called before initKVCache");
		}
		const sl = snapshotLen ?? nTokens;
		if (sl < nTokens) {
			throw new Error(`loadKVCache: snapshotLen=${sl} < nTokens=${nTokens}`);
		}
		const ok = this.bridge.stateSeqSetData(this.mainCtx, bytes, 0);
		if (!ok) {
			throw new Error("loadKVCache: state_seq_set_data failed");
		}
		// State now holds `sl` tokens. Truncate down to nTokens via
		// kv_seq_rm so the consumer's first forward pass extends from
		// position nTokens.
		if (nTokens < sl) {
			this.bridge.kvSeqRm(this.mainCtx, 0, nTokens, -1);
		}
		this.nCached = nTokens;
	}

	/**
	 * Bucket-D self-embed: build a pooled embedding for `tokenIds`
	 * using a side context configured with embeddings=true. The
	 * embedder context is allocated lazily on first call and reused
	 * for the lifetime of this wrapper.
	 *
	 * Pooling: `"last-token"` (default, matches Bucket-D doctrine
	 * 2026-04-30) maps to llama_pooling_type LAST=3. `"mean"` maps to
	 * MEAN=1. The Bucket-D distinguishability gate tests last-token
	 * — `"mean"` is preserved for parity with the legacy embed()
	 * surface but is not exercised by the current canonical fleet.
	 */
	async embed(
		tokenIds: Int32Array,
		opts: EmbedOptions = {},
	): Promise<Float32Array> {
		if (tokenIds.length === 0) {
			throw new Error("embed called with empty tokenIds");
		}
		const pooling = opts.pooling ?? "last-token";
		const poolingType = pooling === "mean" ? 1 : 3;
		if (!this.embedCtx) {
			// Use the model's training-time n_ctx for the embedder context;
			// embedder requests are typically <2K tokens so the budget is
			// generous. Match flash_attn to the main context.
			const nCtxTrain = this.bridge.nCtxTrain(this.model);
			this.embedCtx = await this.bridge.createContext(this.model, {
				nCtx: nCtxTrain > 0 ? nCtxTrain : 4096,
				embeddings: true,
				poolingType: poolingType as 0 | 1 | 2 | 3,
				flashAttn: this.flashAttn,
			});
		}
		// Embedder is single-shot: clear KV before each request so
		// prior embedding's state doesn't leak into this one.
		this.bridge.kvClear(this.embedCtx);
		const status = await this.bridge.decode(this.embedCtx, tokenIds, 0);
		if (status !== 0) {
			throw new Error(`embed: webllm_decode returned status ${status}`);
		}
		// ith=-1 returns the pooled embedding when pooling_type != NONE.
		// Copy out of ctx-owned memory before returning so callers can
		// retain the result across subsequent decode calls.
		const view = await this.bridge.getEmbeddings(this.embedCtx, this.model, -1);
		return new Float32Array(view);
	}

	/** Drop both contexts. Idempotent. */
	dispose(): void {
		if (this.embedCtx) {
			this.bridge.freeContext(this.embedCtx);
			this.embedCtx = 0;
		}
		if (this.mainCtx) {
			this.bridge.freeContext(this.mainCtx);
			this.mainCtx = 0;
		}
		this.nCached = 0;
	}
}
```

- [ ] **Step 2: Typecheck**

Run:
```bash
make typecheck 2>&1 | tail -10
```
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/inference/llama-decode-wrapper.ts
git commit -m "$(cat <<'EOF'
feat(p2): add LlamaDecodeWrapper class

Wraps the (model, ctx) pair from LlamaBridge and exposes the
public surface engine.ts read on the legacy ModelInference:
forward / loadWeights / initKVCache / resetKVCache /
truncateKVCache / serializeKVCache / loadKVCache / embed /
cachedTokenCount / maxContextLength / dispose.

loadWeights is a no-op (weights are on GPU after
webllm_load_model). Bucket-D self-embed survives via a
lazily-created second context with embeddings=true +
pooling=LAST.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 step 1
EOF
)"
```

---

## Task 7: Wrapper unit tests

**Goal:** Establish a Bun-side test that exercises wrapper invariants without needing a real model. Real-model regression is covered by the smoke + bench parity gate at Task 12; these unit tests catch wrapper contract bugs early.

**Files:**
- Create: `tests/llama-decode-wrapper.test.ts`

- [ ] **Step 1: Write the test**

The test uses a fake `LlamaBridge` implementation that records calls, so we can verify the wrapper's translation of `(tokenIds, positions) → (tokens, pastLen)` and the `initKVCache → createContext`, `resetKVCache → kvClear`, etc. mappings without any WASM work.

```typescript
import { describe, expect, it } from "bun:test";
import {
	LlamaDecodeWrapper,
	type EmbedOptions,
} from "../src/inference/llama-decode-wrapper.js";
import type { LlamaBridge } from "../src/inference/llama-bridge.js";

interface DecodeCall {
	ctx: number;
	tokens: number[];
	pastLen: number;
}

function makeFakeBridge(): {
	bridge: LlamaBridge;
	calls: { decode: DecodeCall[]; kvClear: number[]; kvSeqRm: unknown[] };
	nextCtx: { value: number };
} {
	const decode: DecodeCall[] = [];
	const kvClear: number[] = [];
	const kvSeqRm: unknown[] = [];
	const nextCtx = { value: 1000 };
	const fakeLogits = new Float32Array(8);
	const fakeEmbeddings = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
	const bridge: LlamaBridge = {
		loadModel: async () => 999,
		freeModel: () => {},
		createContext: async () => {
			nextCtx.value += 1;
			return nextCtx.value;
		},
		freeContext: () => {},
		decode: async (ctx, tokens, pastLen) => {
			decode.push({ ctx, tokens: Array.from(tokens), pastLen });
			return 0;
		},
		getLogits: async () => fakeLogits,
		nVocab: () => 8,
		tokenize: () => new Int32Array(0),
		detokenize: () => "",
		tokenBos: () => 1,
		tokenEos: () => 2,
		getMetadata: () => null,
		nCtxTrain: () => 4096,
		nEmbd: () => 8,
		nLayer: () => 4,
		nHead: () => 4,
		nHeadKv: () => 4,
		nCtx: () => 4096,
		kvSeqRm: (ctx, seqId, p0, p1) => {
			kvSeqRm.push({ ctx, seqId, p0, p1 });
		},
		kvClear: (ctx) => {
			kvClear.push(ctx);
		},
		stateSeqGetSize: () => 1024,
		stateSeqGetData: () => new Uint8Array(1024),
		stateSeqSetData: () => true,
		getEmbeddings: async () => fakeEmbeddings,
	};
	return { bridge, calls: { decode, kvClear, kvSeqRm }, nextCtx };
}

describe("LlamaDecodeWrapper", () => {
	it("forwards sequential positions and tracks cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		w.loadWeights();
		await w.initKVCache(2048);

		expect(w.cachedTokenCount).toBe(0);
		expect(w.maxContextLength).toBe(4096);

		await w.forward(new Int32Array([10, 20, 30]), new Int32Array([0, 1, 2]));
		expect(w.cachedTokenCount).toBe(3);
		expect(calls.decode).toHaveLength(1);
		expect(calls.decode[0].pastLen).toBe(0);
		expect(calls.decode[0].tokens).toEqual([10, 20, 30]);

		await w.forward(new Int32Array([40]), new Int32Array([3]));
		expect(w.cachedTokenCount).toBe(4);
		expect(calls.decode[1].pastLen).toBe(3);
	});

	it("rejects non-sequential positions", async () => {
		const { bridge } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		expect(
			w.forward(new Int32Array([10, 20]), new Int32Array([0, 5])),
		).rejects.toThrow(/sequential from cachedTokenCount=0/);
	});

	it("resetKVCache calls kvClear and zeros cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.forward(new Int32Array([10, 20]), new Int32Array([0, 1]));
		expect(w.cachedTokenCount).toBe(2);
		w.resetKVCache();
		expect(w.cachedTokenCount).toBe(0);
		expect(calls.kvClear).toHaveLength(1);
	});

	it("truncateKVCache calls kvSeqRm and updates cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.forward(new Int32Array([10, 20, 30, 40]), new Int32Array([0, 1, 2, 3]));
		w.truncateKVCache(2);
		expect(w.cachedTokenCount).toBe(2);
		expect(calls.kvSeqRm).toHaveLength(1);
		expect(calls.kvSeqRm[0]).toMatchObject({ seqId: 0, p0: 2, p1: -1 });
	});

	it("loadKVCache truncates when snapshotLen > nTokens", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.loadKVCache(new Uint8Array(1024), 5, 10);
		expect(w.cachedTokenCount).toBe(5);
		// One kvSeqRm call to drop tokens [5, 10).
		expect(calls.kvSeqRm).toHaveLength(1);
		expect(calls.kvSeqRm[0]).toMatchObject({ seqId: 0, p0: 5, p1: -1 });
	});

	it("embed creates a side context once and clears KV between calls", async () => {
		const { bridge, calls, nextCtx } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		const e1 = await w.embed(new Int32Array([10, 20, 30]));
		const e2 = await w.embed(new Int32Array([40, 50]));
		// 1 main ctx + 1 embed ctx.
		expect(nextCtx.value).toBe(1002);
		// Two clears: each embed call resets the side context.
		expect(calls.kvClear).toHaveLength(2);
		// Both calls returned the fake embeddings.
		expect(Array.from(e1)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
		expect(Array.from(e2)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
	});
});
```

- [ ] **Step 2: Run the test**

```bash
bun test tests/llama-decode-wrapper.test.ts 2>&1 | tail -20
```
Expected: 6/6 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/llama-decode-wrapper.test.ts
git commit -m "$(cat <<'EOF'
test(p2): unit tests for LlamaDecodeWrapper invariants

Fake-bridge tests that lock the wrapper's contract:
- sequential-position assertion in forward()
- cachedTokenCount tracking across decode calls
- resetKVCache → kvClear translation
- truncateKVCache → kvSeqRm translation
- loadKVCache snapshotLen > nTokens path
- embed lazily creates side context, kvClears between calls

These run without WASM; real-model regression is the bench
parity gate in Task 12.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2
EOF
)"
```

---

## Task 8: Rewrite `model-loader.ts`

**Goal:** Collapse GGUF parsing to a thin metadata pass-through. The new path loads the model handle via `webllm_load_model`, queries hyperparams and chat template via the bridge, and returns the data engine.ts already consumes. The legacy `TokenizerConfig` shape disappears (engine.ts uses `LlamaTokenizer` instead).

**Files:**
- Modify: `src/models/model-loader.ts` (full rewrite — drop ~350 LOC of header parsing)
- Modify: `src/core/types.ts` (verify `ModelHyperparams` still has the fields the rewrite produces)

- [ ] **Step 1: Identify the data the rewrite must produce**

Run:
```bash
grep -nE "parsed\.(hyperparams|tokenizerConfig|kvCacheConfig)|ParsedModel" src/core/engine.ts | head
```
Expected: confirm engine.ts reads `parsed.hyperparams` (every field from the legacy `extractHyperparams`) and `parsed.kvCacheConfig.maxContextLength`. `tokenizerConfig` is consumed only at line 1654 (`new Tokenizer(parsed.tokenizerConfig)`); Task 9 replaces that line with `LlamaTokenizer` construction.

- [ ] **Step 2: Author the rewrite**

Replace the entire contents of `src/models/model-loader.ts` with:

```typescript
import {
	isCausalEmbedderArchitecture,
	isEncoderArchitecture,
	type ModelHyperparams,
} from "../core/types.js";
import type { LlamaBridge } from "../inference/llama-bridge.js";
import type { KVCacheConfig } from "./kv-cache.js";

/** Result of loading + introspecting a GGUF model via the bridge. */
export interface LoadedModelMetadata {
	/** Bridge handle to the upstream llama_model. Owned by the caller. */
	model: number;
	hyperparams: ModelHyperparams;
	kvCacheConfig: KVCacheConfig;
	/** Chat template string, or "" if missing. */
	chatTemplate: string;
}

/**
 * Load a GGUF buffer through the bridge and pull metadata back into
 * the {@link ModelHyperparams} shape engine.ts already consumes. The
 * caller owns the returned model handle and must call
 * `bridge.freeModel(metadata.model)` on disposal.
 *
 * In the legacy path, `ModelLoader.parseModel` re-implemented every
 * GGUF header field in TS. Upstream's parser is the source of truth
 * for everything llama.cpp supports — this thin wrapper surfaces it.
 */
export async function loadModelMetadata(
	bridge: LlamaBridge,
	data: Uint8Array,
): Promise<LoadedModelMetadata> {
	const model = await bridge.loadModel(data);
	try {
		const archStr =
			bridge.getMetadata(model, "general.architecture") ?? "llama";
		const metaPrefix = archStr;

		// Same Qwen3-Embedding derivation rule as the legacy loader:
		// pooling_type=3 (LAST) on a qwen3 model means it's the embedding
		// variant, not the chat variant. Surfaces via metadata so the
		// rule stays load-bearing post-migration.
		const qwen3PoolingRaw =
			archStr === "qwen3"
				? Number(bridge.getMetadata(model, "qwen3.pooling_type") ?? "")
				: Number.NaN;
		const arch: ModelHyperparams["architecture"] =
			archStr === "qwen3" && qwen3PoolingRaw === 3
				? "qwen3-embedding"
				: (archStr as ModelHyperparams["architecture"]);

		const embeddingLength = bridge.nEmbd(model);
		const headCount = bridge.nHead(model);
		const headCountKv = bridge.nHeadKv(model);
		const layerCount = bridge.nLayer(model);
		const contextLength = bridge.nCtxTrain(model);

		const ftypeRaw = bridge.getMetadata(model, "general.file_type");
		const quantType =
			ftypeRaw !== null ? mapFtypeToQuantName(Number(ftypeRaw)) : "unknown";

		// Norm epsilon: encoders use layer_norm_epsilon, others use
		// the RMSNorm key. Same dispatch as the legacy loader.
		const normEpsilonStr = isEncoderArchitecture(arch)
			? bridge.getMetadata(model, `${metaPrefix}.attention.layer_norm_epsilon`)
			: bridge.getMetadata(
					model,
					`${metaPrefix}.attention.layer_norm_rms_epsilon`,
				);
		const normEpsilon =
			normEpsilonStr !== null
				? Number(normEpsilonStr)
				: isEncoderArchitecture(arch)
					? 1e-12
					: 1e-5;

		// Pooling + causal flag for encoders / causal-LM-derived embedders.
		let poolingType: ModelHyperparams["poolingType"];
		let causalAttention: boolean | undefined;
		let alibiMaxBias: number | undefined;
		if (isEncoderArchitecture(arch)) {
			const ptStr = bridge.getMetadata(model, `${metaPrefix}.pooling_type`);
			const pt = ptStr !== null ? Number(ptStr) : 2;
			poolingType = pt === 1 ? "mean" : "cls";
			const causalStr = bridge.getMetadata(
				model,
				`${metaPrefix}.attention.causal`,
			);
			causalAttention =
				causalStr === null ? false : causalStr.toLowerCase() === "true";
			if (arch === "jina-bert-v2") {
				const alibiStr = bridge.getMetadata(
					model,
					`${metaPrefix}.attention.alibi_bias_max`,
				);
				alibiMaxBias = alibiStr !== null ? Number(alibiStr) : 8.0;
			}
		} else if (isCausalEmbedderArchitecture(arch)) {
			poolingType = "last-token";
		}

		const hyperparams: ModelHyperparams = {
			architecture: arch,
			contextLength,
			embeddingLength,
			headCount,
			headCountKv,
			layerCount,
			vocabularySize: bridge.nVocab(model),
			embeddingHeadLength: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.attention.key_length`,
				Math.floor(embeddingLength / Math.max(1, headCount)),
			),
			feedForwardLength: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.feed_forward_length`,
				11008,
			),
			ropeFreqBase:
				numFromMetaOptional(bridge, model, `${metaPrefix}.rope_freq_base`) ??
				numFromMetaOptional(bridge, model, `${metaPrefix}.rope.freq_base`) ??
				10000,
			ropeScale: numFromMeta(bridge, model, `${metaPrefix}.rope_scale`, 1),
			normEpsilon,
			expertCount: numFromMeta(bridge, model, `${metaPrefix}.expert_count`, 0),
			expertUsedCount: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.expert_used_count`,
				0,
			),
			quantType,
			poolingType,
			causalAttention,
			alibiMaxBias,
		};

		const kvCacheConfig: KVCacheConfig = {
			nLayers: hyperparams.layerCount,
			nEmbdHeadK: hyperparams.embeddingHeadLength,
			nEmbdHeadV: hyperparams.embeddingHeadLength,
			nKvHead: hyperparams.headCountKv,
			maxContextLength: hyperparams.contextLength,
			dataType: "f32",
		};

		const chatTemplate = bridge.getMetadata(model, "tokenizer.chat_template") ?? "";

		return { model, hyperparams, kvCacheConfig, chatTemplate };
	} catch (err) {
		bridge.freeModel(model);
		throw err;
	}
}

function numFromMeta(
	bridge: LlamaBridge,
	model: number,
	key: string,
	fallback: number,
): number {
	const v = bridge.getMetadata(model, key);
	if (v === null) return fallback;
	const n = Number(v);
	return Number.isFinite(n) ? n : fallback;
}

function numFromMetaOptional(
	bridge: LlamaBridge,
	model: number,
	key: string,
): number | undefined {
	const v = bridge.getMetadata(model, key);
	if (v === null) return undefined;
	const n = Number(v);
	return Number.isFinite(n) ? n : undefined;
}

// Mirrors llama.cpp `enum llama_ftype` (llama.h). Stable since 2024.
const LLAMA_FTYPE_NAMES: Readonly<Record<number, string>> = {
	0: "F32",
	1: "F16",
	2: "Q4_0",
	3: "Q4_1",
	7: "Q8_0",
	8: "Q5_0",
	9: "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS",
	20: "IQ2_XS",
	21: "Q2_K_S",
	22: "IQ3_XS",
	23: "IQ3_XXS",
	24: "IQ1_S",
	25: "IQ4_NL",
	26: "IQ3_S",
	27: "IQ3_M",
	28: "IQ2_S",
	29: "IQ2_M",
	30: "IQ4_XS",
	31: "IQ1_M",
	32: "BF16",
	36: "TQ1_0",
	37: "TQ2_0",
};

function mapFtypeToQuantName(ftype: number): string {
	return LLAMA_FTYPE_NAMES[ftype] ?? "unknown";
}

/**
 * Legacy compatibility shim. Encoder + causal-embedder loaders still
 * construct `ParsedModel` from a Uint8Array; they're rewritten in
 * P3/P4. Re-exporting the legacy API surface here lets P2 land
 * without touching them.
 */
export { ModelLoader } from "./model-loader-legacy.js";
export type { ParsedModel } from "./model-loader-legacy.js";
```

- [ ] **Step 3: Move legacy code to a shim file**

The encoder and causal-embedder branches in `engine.ts:1614-1645` still call `ModelLoader.parseModel(data)` and consume `parsed.tokenizerConfig`. They're explicitly out-of-scope for P2 (P3 and P4 phases). To avoid touching those paths, copy the *original* `model-loader.ts` content into a new file `src/models/model-loader-legacy.ts` and have the new `model-loader.ts` re-export `ModelLoader` and `ParsedModel` from it.

Run:
```bash
git show HEAD~7:src/models/model-loader.ts > src/models/model-loader-legacy.ts
```
Where `HEAD~N` resolves to a commit before any P2 modification touches the file. Verify the file looks right — it must define `class ModelLoader` and export `ParsedModel`.

If the legacy file has its own internal imports (`./gguf-parser.js`, `./gguf-types.js`, etc.) those still resolve.

- [ ] **Step 4: Typecheck**

Run:
```bash
make typecheck 2>&1 | tail -20
```
Expected: clean. If `model-loader-legacy.ts` still references the legacy `Tokenizer`-only types (`TokenizerConfig`, etc.), they must remain importable from `tokenizer.ts` until Task 11 — which is why Task 11 carefully retains the streaming detokenizer + types but drops only the encoders.

- [ ] **Step 5: Commit**

```bash
git add src/models/model-loader.ts src/models/model-loader-legacy.ts
git commit -m "$(cat <<'EOF'
refactor(p2): collapse model-loader to bridge metadata pass-through

New `loadModelMetadata(bridge, data) → LoadedModelMetadata` loads
the GGUF buffer through the bridge and reads hyperparams +
kv-cache config + chat template from upstream. The 350 LOC of
hand-rolled GGUF header parsing in the legacy loader is now
delegated to llama.cpp.

Encoder + causal-embedder paths (P3 / P4) still need the legacy
TokenizerConfig + ParsedModel shape; preserved via a re-export
from model-loader-legacy.ts. Those callers will migrate in their
own phases.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 step 2
EOF
)"
```

---

## Task 9: Flip `engine.ts` causal-LM dispatch to the wrapper

**Goal:** Change the causal-LM construction site to build a `LlamaDecodeWrapper` + `LlamaTokenizer` pair from the new loader output. Encoder + causal-embedder branches stay on the legacy path.

**Files:**
- Modify: `src/core/engine.ts`

- [ ] **Step 1: Add imports**

At the top of `src/core/engine.ts`, alongside the existing `ModelInference` import:
- Replace `import { ModelInference, … } from "../inference/model-inference.js";` with `import { LlamaDecodeWrapper } from "../inference/llama-decode-wrapper.js";` (keep any sibling-named exports the file still uses, e.g. types like `DecodeMode`, until Task 11 confirms they're unused).
- Add `import { LlamaTokenizer } from "../inference/llama-tokenizer.js";`
- Add `import { createLlamaBridge, type LlamaBridge } from "../inference/llama-bridge.js";`
- Add `import { loadModelMetadata } from "../models/model-loader.js";`
- Keep `import { ModelLoader, type ParsedModel } from "../models/model-loader.js";` — Task 8 still re-exports them for the P3/P4 paths.

If TypeScript complains about an unused `ModelInference` symbol after this, leave its import as `import type` until Task 11 deletes it.

- [ ] **Step 2: Update the causal-LM branch in `loadModelFromBuffer`**

The current branch is at `src/core/engine.ts:1629-1645`:

```typescript
} else {
    const inf = new ModelInference(wasm, parsed.hyperparams, {
        flashAttn: !!options?.flashAttn,
    });
    inf.loadWeights(ggufCtx, dataSrc);
    freeStaging();
    const requestedCtx = options?.contextLength;
    const ctxLen =
        typeof requestedCtx === "number" && requestedCtx > 0
            ? Math.min(requestedCtx, parsed.kvCacheConfig.maxContextLength)
            : parsed.kvCacheConfig.maxContextLength;
    inf.initKVCache(ctxLen);
    inference = inf;
}
```

Replace with:

```typescript
} else {
    // P2: causal-LM goes through the bridge wrapper. Encoder /
    // causal-embedder branches above still use the legacy graph
    // builder until P3 / P4.
    const bridge = createLlamaBridge(wasm.module);
    const meta = await loadModelMetadata(bridge, dataBytes);
    // Replace `parsed` for downstream session-tracker fields. The
    // legacy path's `parsed.hyperparams` came from the same metadata
    // source, so hyperparam fingerprints stay identical.
    parsed = {
        hyperparams: meta.hyperparams,
        // No legacy tokenizerConfig — LlamaTokenizer wraps the model
        // handle directly.
        tokenizerConfig: null as unknown as ParsedModel["tokenizerConfig"],
        kvCacheConfig: meta.kvCacheConfig,
    };
    const wrapper = new LlamaDecodeWrapper(bridge, meta.model, {
        flashAttn: !!options?.flashAttn,
    });
    wrapper.loadWeights();
    freeStaging();
    const requestedCtx = options?.contextLength;
    const ctxLen =
        typeof requestedCtx === "number" && requestedCtx > 0
            ? Math.min(requestedCtx, meta.kvCacheConfig.maxContextLength)
            : meta.kvCacheConfig.maxContextLength;
    await wrapper.initKVCache(ctxLen);
    inference = wrapper;
    // Remember the bridge + tokenizer for the engine's per-handle
    // bookkeeping a few lines below — moved to the entry construction
    // step instead of staying in the branch.
    bridgeForHandle = bridge;
    tokenizerForHandle = new LlamaTokenizer(bridge, meta.model, {
        chatTemplate: meta.chatTemplate,
    });
}
```

The variables `bridgeForHandle` + `tokenizerForHandle` are introduced at the top of the function body so the encoder / causal-embedder branches can leave them undefined (those branches keep using the legacy `Tokenizer` construction path until P3/P4):

```typescript
let bridgeForHandle: LlamaBridge | null = null;
let tokenizerForHandle: LlamaTokenizer | null = null;
```

- [ ] **Step 3: Wire tokenizer into the engine entry**

Below the branch, the existing block at `src/core/engine.ts:1647-1664` builds `entry.tokenizer` from `parsed.tokenizerConfig`. Update it to prefer `tokenizerForHandle` when present:

```typescript
const handle = this.registerModelHandle(name, {
    priority: 0,
    ...options,
});
const entry = this._modelManager.get(handle.id);
if (entry) {
    entry.hyperparams = parsed.hyperparams;
    entry.tokenizer =
        tokenizerForHandle ??
        // Encoder + causal-embedder branches still build a legacy
        // Tokenizer from parsed.tokenizerConfig. Removed in P3/P4.
        new Tokenizer(parsed.tokenizerConfig);
    entry.kvCache = new KVCache(parsed.kvCacheConfig);
    entry.tokenizerHash = await computeTokenizerHash(
        parsed.tokenizerConfig ?? {
            // For LlamaTokenizer-backed entries, fingerprint over the
            // model's vocab metadata instead of the legacy
            // TokenizerConfig. The hash matters only for the
            // model-fingerprint, which is used for prefix-cache
            // partitioning — no need to be byte-identical to the
            // legacy hash, just stable per (model, vocab) pair.
            type: 0,
            tokens: [],
            bpeRanks: new Map(),
            addedTokens: new Map(),
            eosTokenId: tokenizerForHandle?.eosId ?? -1,
            bosTokenId: tokenizerForHandle?.bosId ?? -1,
            padTokenId: -1,
            vocabSize: tokenizerForHandle?.vocabSize ?? 0,
            chatTemplate:
                tokenizerForHandle?.options.chatTemplate ?? "",
        },
    );
    entry.fingerprint = buildModelFingerprint(
        parsed.hyperparams,
        entry.tokenizerHash,
    );
    entry.loaded = true;
}
```

- [ ] **Step 4: Update the `inferenceEngines` map type**

The map currently stores `ModelInference`. Widen to a union, then narrow at consumer sites where needed:

```typescript
private inferenceEngines = new Map<
    string,
    ModelInference | LlamaDecodeWrapper
>();
```

Most consumers in engine.ts read methods that *both* classes implement (`forward`, `cachedTokenCount`, `maxContextLength`, `resetKVCache`, `embed`, `serializeKVCache`, `loadKVCache`, `truncateKVCache`). Where they read `forwardDecode` (lines 606, 1048), gate on `typeof … === "function"` — the wrapper does not implement `forwardDecode`, so the legacy GPU-side argmax fast path is skipped on the new path. The CPU sampling fall-through is already in place; no behavioral change.

- [ ] **Step 5: Update `__debugInferenceForModel` and `adoptPreloadedModel` typings**

`__debugInferenceForModel` (line 1149) returns `ModelInference | undefined`. Widen to `ModelInference | LlamaDecodeWrapper | undefined`. `adoptPreloadedModel` (line 1316) reads `pipeline.inference` — widen its type to the same union; the existing `instanceof EncoderInference` / `CausalLMEmbedder` checks still work, and the else branch covers both `ModelInference` and `LlamaDecodeWrapper`.

- [ ] **Step 6: Update `loadModelFromBuffer`'s return type**

Line 1599 declares the return as `inference: ModelInference | EncoderInference | CausalLMEmbedder;`. Add `LlamaDecodeWrapper`:

```typescript
inference: ModelInference | LlamaDecodeWrapper | EncoderInference | CausalLMEmbedder;
```

Same for the static `WebLLM.loadModelFromBuffer` at line 1711. Same for any other named return-type alias the file declares.

- [ ] **Step 7: Pass the GGUF bytes to the new loader**

The `dataBytes` (the Uint8Array passed in by the caller, ahead of the legacy `ggufCtx` parsing) is the actual buffer the bridge wants. Locate where it's named in the helper — the legacy code calls `ModelLoader.parseModel(dataBytes)` upstream of the inference branch. The new path calls `await loadModelMetadata(bridge, dataBytes)` directly. The legacy `ggufCtx` / `dataSrc` callback structure is no longer needed for the causal branch but is retained for the encoder + causal-embedder paths.

If the variable name in the actual file differs from `dataBytes`, use whatever name the helper uses for the input GGUF buffer.

- [ ] **Step 8: Typecheck + run unit tests**

```bash
make typecheck 2>&1 | tail -30
bun test 2>&1 | tail -20
```
Expected:
- typecheck clean.
- existing tests green except the speculative ones (those are gated in Task 10).

- [ ] **Step 9: Commit**

```bash
git add src/core/engine.ts
git commit -m "$(cat <<'EOF'
feat(p2): route causal-LM loads through LlamaDecodeWrapper

engine.ts:loadModelFromBuffer now constructs LlamaDecodeWrapper +
LlamaTokenizer for the causal-LM branch. Encoder + causal-embedder
branches remain on the legacy ModelInference / Tokenizer pair
until P3 / P4 ship.

The inferenceEngines map widens to ModelInference | LlamaDecodeWrapper.
Consumers read methods present on both classes; forwardDecode
(legacy-only) gates via `typeof === "function"` and falls through
to the CPU sampling path on the new wrapper — no behavioral change
for the non-spec-decode hot loop.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 step 3
EOF
)"
```

---

## Task 10: Detach `speculative.ts` from `ModelInference`

**Goal:** Make `speculative.ts` compile without referencing `ModelInference`. Spec-decode is gated behind `SpeculativeDecodingReservedError` at runtime, so this is purely a compile-time decoupling. Tests for spec-decode are skipped with a "deferred to P5" marker.

**Files:**
- Modify: `src/inference/speculative.ts`
- Modify: `tests/speculative-integration.test.ts`
- Modify: `tests/speculative-rejection.test.ts`

- [ ] **Step 1: Replace the import with a local interface**

In `src/inference/speculative.ts`, replace `import type { ModelInference } from "./model-inference.js";` with:

```typescript
/**
 * Structural slice of the legacy ModelInference / new
 * LlamaDecodeWrapper public surface that spec-decode reads. Declared
 * locally so this file compiles without binding to either concrete
 * class — spec-decode rewires onto LlamaDecodeWrapper in P5.
 *
 * Runtime path is gated by SpeculativeDecodingReservedError in
 * engine.ts; this interface exists for type-checker continuity only.
 */
export interface SpecDecodeForwardPass {
	forward(tokenIds: Int32Array, positions: Int32Array): Promise<Float32Array>;
	forwardVerify?: (
		tokenIds: Int32Array,
		positions: Int32Array,
	) => Promise<Float32Array>;
	truncateKVCache(keepLen: number): void;
	cachedTokenCount: number;
}
```

Then replace each `ModelInference` reference inside the file with `SpecDecodeForwardPass`.

- [ ] **Step 2: Skip spec-decode tests**

In each of `tests/speculative-integration.test.ts` and `tests/speculative-rejection.test.ts`, wrap the top-level `describe(...)` in `describe.skip(...)` and add a comment at the top of the file:

```typescript
// SKIP: deferred to Tier 3 P5 (Speculative decode + KV consolidation).
// The runtime path is gated by SpeculativeDecodingReservedError in
// engine.ts; the legacy graph-builder dependency these tests
// exercise is being deleted in P2. Re-enable when P5 lands the
// two-context spec-decode rewire on LlamaDecodeWrapper.
// Plan: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P5
```

- [ ] **Step 3: Run tests**

```bash
bun test 2>&1 | tail -20
```
Expected:
- speculative tests reported as skipped (Bun's `describe.skip` shows them as `(skipped)` or counts them as 0 ran).
- All other tests green.

- [ ] **Step 4: Commit**

```bash
git add src/inference/speculative.ts tests/speculative-integration.test.ts tests/speculative-rejection.test.ts
git commit -m "$(cat <<'EOF'
refactor(p2): decouple speculative.ts from ModelInference type

Replace `import type { ModelInference }` with a local structural
interface (SpecDecodeForwardPass) listing only the methods spec-
decode reads. Runtime path is already gated behind
SpeculativeDecodingReservedError, so this is compile-time only.

speculative-integration + speculative-rejection tests are skipped
with a "deferred to P5" marker — the legacy graph-builder code
they exercise is being deleted in P2 Task 11, and P5 rewires
spec-decode onto two LlamaDecodeWrapper contexts sharing one
llama_model.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P5
EOF
)"
```

---

## Task 11: Delete `model-inference.ts` + clean up `tokenizer.ts`

**Goal:** Drop the ~2890 LOC of legacy graph-builder code now that nothing references it on the causal path. Drop the BPE/SPM/WordPiece encoders from `tokenizer.ts` (P1 already replaced them on the active path). Retain the `StreamingDecoder` + `TokenAttribute` enum + types still used by encoder + causal-embedder loaders.

**Files:**
- Delete: `src/inference/model-inference.ts`
- Modify: `src/inference/tokenizer.ts`
- Modify: `src/index.ts`
- Modify: `src/inference/encoder-inference.ts` (drop the `getRopeModeForArchitecture` import — re-export it from a new module, see step 3)
- Modify: `src/inference/causal-embedder-inference.ts` (same as above)

- [ ] **Step 1: Verify no remaining `ModelInference` references**

Run:
```bash
grep -rn "ModelInference\|model-inference" src/ tests/ smoke-test/ 2>/dev/null | grep -v "speculative" | head
```
Expected: only `tests/speculative-*` (skipped) reference it; everything else has been migrated. If anything else references it, audit and update before proceeding.

- [ ] **Step 2: Audit `getRopeModeForArchitecture` consumers**

The function `getRopeModeForArchitecture` lives in `model-inference.ts:107`; it's imported by `encoder-inference.ts:14` and `causal-embedder-inference.ts:13` (P3/P4 still use the legacy graph builder). Move it to a small, dedicated file:

```bash
mkdir -p src/inference/rope
```

Create `src/inference/rope/rope-mode.ts`:

```typescript
import type { ModelHyperparams } from "../../core/types.js";

/**
 * Map a model architecture to its ggml rope_type / rope_mode. Read
 * by encoder-inference.ts and causal-embedder-inference.ts for
 * graph-builder dispatch. Will be deleted alongside those files
 * in P3 / P4.
 */
export function getRopeModeForArchitecture(
	arch: ModelHyperparams["architecture"],
): number {
	// Copy the body verbatim from src/inference/model-inference.ts:107
	// at the time of this commit. The function has no external
	// dependencies beyond the architecture string.
	throw new Error("populate from legacy model-inference.ts");
}
```

Replace the throw with the actual body from the legacy file. Run `git show HEAD~N:src/inference/model-inference.ts | sed -n '107,150p'` to retrieve the original body, where `N` is the number of commits back to a known-good revision (or use `git log -p src/inference/model-inference.ts | head` and copy from the earliest blob). The function is small (a switch on architecture string).

Update `encoder-inference.ts:14` and `causal-embedder-inference.ts:13` to import from the new path:

```typescript
import { getRopeModeForArchitecture } from "./rope/rope-mode.js";
```

- [ ] **Step 3: Delete `model-inference.ts`**

```bash
git rm src/inference/model-inference.ts
```

- [ ] **Step 4: Strip `tokenizer.ts` to the streaming surface**

`src/inference/tokenizer.ts` currently holds: `TokenAttribute` enum, `TokenData` + `TokenizerConfig` types, `Tokenizer` class (BPE/SPM/WordPiece), `TokenizerType` enum, `StreamingDecoder`, helper regexes, BPE-ranks tables.

After P2:
- `TokenAttribute`, `TokenData`, `TokenizerConfig`, `TokenizerType` — **kept** (referenced by `model-loader-legacy.ts` for encoder + causal-embedder paths). Move them to `src/inference/tokenizer-types.ts` if a 150 LOC tokenizer.ts feels still too cluttered; otherwise leave them in tokenizer.ts.
- `Tokenizer` class — **delete** (replaced by `LlamaTokenizer` on the active path; encoder + causal-embedder paths still construct a `Tokenizer`-like object — see audit in step 6).
- `StreamingDecoder` — **kept** (project-specific, not exposed by upstream).
- BPE pre-tokenizer regex tables, `bytesToUnicode`, etc. — **delete**.
- WordPiece encoder routines — **delete**.

If `model-loader-legacy.ts` still constructs a full `TokenizerConfig` for the encoder branch, then encoder-inference + causal-embedder-inference still call `new Tokenizer(...)` somewhere. Audit:

```bash
grep -rn "new Tokenizer(" src/ tests/ smoke-test/ 2>/dev/null
```

If the encoder path constructs a `Tokenizer`, P3 deletes it. For P2 we either:
1. Keep `Tokenizer` alive for now and only delete its BPE/SPM/WordPiece encoder *implementations* (replace with `throw new Error("legacy encoder removed in P2; use LlamaTokenizer")` — but only safe if no encoder path runs `Tokenizer.encode` at runtime), OR
2. Keep the full legacy `Tokenizer` class alive until P3.

**Decision (pragmatic per CLAUDE.md "surgical changes"):** keep `Tokenizer` alive but stripped of dead code that *only* the causal path used. Specifically: any helper invoked solely by the causal branch can go; helpers shared with encoder/wordpiece paths stay. Concretely, that usually means dropping the `_bpePreTokenizerRegex` tables for `gpt2` / `qwen3` if encoders use only `wordpiece`; preserve `wordpiece` encoder + `spm` encoder for the encoder-inference path.

The minimum bar for Task 11: delete `model-inference.ts`, retain `Tokenizer` until P3 (it's small enough, ~1000 LOC) and only **delete unused exports** the causal path was the sole consumer of (audit via grep first). Document any deferred deletions inline.

If the audit shows `Tokenizer.encode` is called *only* on encoder/causal-embedder/legacy paths, the BPE encoder code is dead and can be deleted now. Otherwise leave it.

- [ ] **Step 5: Update `src/index.ts`**

Drop the `ModelInference` export (line 135). Add `LlamaDecodeWrapper`:

```typescript
export { LlamaDecodeWrapper } from "./inference/llama-decode-wrapper.js";
```

`Tokenizer` export: keep until P3 (encoder paths still use it).

- [ ] **Step 6: Run checkall**

```bash
make checkall 2>&1 | tail -30
```
Expected: fmt + lint + typecheck + test all green. Address any leftover references.

If `make checkall` fails on the live `Tokenizer` legacy class because P2 stripped a method that encoder paths still need, restore that method and document the deferred-deletion in `TODO.md`'s P3 follow-up list.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(p2): delete model-inference.ts; trim tokenizer.ts encoders

Net delete: 2890 LOC (model-inference.ts) + dead BPE/SPM/WordPiece
encoder implementations in tokenizer.ts that the causal path was
the sole consumer of. Tokenizer class kept (still used by encoder
+ causal-embedder paths until P3 / P4); LlamaTokenizer is the
canonical encoder on the live causal-LM path.

getRopeModeForArchitecture moved to rope/rope-mode.ts so
encoder-inference + causal-embedder-inference still resolve their
last dependency on the deleted file.

src/index.ts drops the ModelInference export and adds
LlamaDecodeWrapper.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2 step 4
EOF
)"
```

---

## Task 12: Parity gate + closure report

**Goal:** Run the same bench harnesses as Task 0 against the post-migration `main` and document the result.

**Files:**
- Create: `eval/reports/p2-causal-migration-2026-05-05/SUMMARY.md`
- Modify: `TODO.md` (close P2 stub + archive cadence)

- [ ] **Step 1: Run greedy accuracy bench on the canonical 6**

```bash
make dashboard-serve > /tmp/dash-p2.log 2>&1 &
sleep 3
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 make bench-canonical EVAL_TEMP=0 2>&1 | tee /tmp/bench-p2.log
```
Compare row-by-row to `PRE-MIGRATION-BASELINE.md`.

- [ ] **Step 2: Run smoke-bench (3-run median) on the canonical 6**

```bash
make smoke-bench PERF_RUNS=3 PERF_MODELS=canonical-6 2>&1 | tee /tmp/smoke-bench-p2.log
```

- [ ] **Step 3: Compare against the parity bar**

For each canonical-6 model, post-migration must satisfy:
- **Accuracy:** within ±2 of baseline `accuracy / 36`. (±5.5% absolute = sampling noise per 2026-05-04 doctrine.)
- **Decode tok/s:** within ±10% of baseline 3-run median.
- **Bucket-D distinguishability gate:** if any embedding-capable model (per `embeddingCapable: true` in `eval/models.ts`) is in the canonical 6, run the 16+16 mean-margin test from the 2026-04-30 doctrine. Mean margin must be ≥ pre-migration value. (Can skip this step if no canonical-6 model is embedding-capable; add a one-line note in the SUMMARY in that case.)

If any model fails the parity bar, the P2 closure goes into a **perf-recovery sub-phase** per the spec §P2:
1. Identify the cause (likely `llama_context_params.n_ubatch` defaults vs project's tile heuristic).
2. Tune knobs in `webllm_create_context` (set `cparams.n_ubatch` to a project-tuned value).
3. Re-bench. Merge only when within ±10%.
4. If irrecoverable, **revert P2 to a prior tag** and document in SUMMARY.

- [ ] **Step 4: Run `make checkall`**

```bash
make checkall 2>&1 | tail -20
```
Expected: fmt + lint + typecheck + test all green. The wrapper unit test from Task 7 must show as 6/6.

- [ ] **Step 5: Write `SUMMARY.md`**

The closure report MUST contain:
1. **Header:** date, git SHA at closure, "PASS" or "FAIL" verdict.
2. **Accuracy table:** canonical 6, baseline vs post, delta. Mark each row PASS / FAIL.
3. **Decode-throughput table:** canonical 6, baseline vs post (3-run median), delta. Mark each row PASS / FAIL (≤±10%).
4. **Bucket-D gate (if applicable):** mean-margin pre vs post.
5. **LOC delta:** counted via `git diff --stat 72cd44c..HEAD -- 'src/**/*.ts' 'src/**/*.cpp'` (or equivalent reference-commit). Should land near the spec's predicted **−3400 LOC** TS net delete.
6. **Patches consumed:** 0 (P2 should not need any — pure additive bridge surface + TS rewrite). If any patches landed in llama.cpp, link to upstream PRs and document why.
7. **Closure decision:**
   - Accept all rows pass → P2 closes; P3 (Encoder migration) is next.
   - Some rows accept-with-followup → P2 closes; perf-recovery items file as P2.1 in TODO.md.
   - Catastrophic failure → revert; document why.

- [ ] **Step 6: Update TODO.md**

Per the CLAUDE.md "TODO archival cadence":
- The P2 quickstart block in `TODO.md` (the "Next-session quickstart for P2 — Causal-LM migration" section near line 848) gets a **closure stub** (4-8 lines) replacing the bullet list, with links to:
  - this SUMMARY,
  - the canonical loader/wrapper commits,
  - and the perf-recovery sub-phase artifacts if any.
- The "Tier 3 migration to upstream `llama_decode`" header block status flips from "P0 + P1 CLOSED. Next phase = P2" to "P0 + P1 + P2 CLOSED. Next phase = P3 (Encoder migration)". Spec §P3 quickstart block added at the same nesting level.

- [ ] **Step 7: Commit closure**

```bash
git add -f eval/reports/p2-causal-migration-2026-05-05/
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(p2): close P2 — causal-LM migration to webllm_decode

Parity gate: <PASS / PASS-with-followup / FAIL>.

- Accuracy on canonical 6: <summary line>.
- Decode tok/s on canonical 6 (3-run median): <summary line>.
- Bucket-D distinguishability: <line or "N/A — none in canonical 6">.
- LOC delta: <approx −3400 TS, +X cpp>.
- Patches consumed: 0.

Closure report:
eval/reports/p2-causal-migration-2026-05-05/SUMMARY.md

TODO.md updated: P2 stub replaces the quickstart block; next phase
is P3 (Encoder migration) per spec §P3.

Spec: docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2
EOF
)"
```

---

## Self-Review Notes

This plan was reviewed against the spec §P2 step list:

1. **Spec step 1 (`llama-decode-wrapper.ts`)** — Task 6.
2. **Spec step 2 (rewrite `model-loader.ts`)** — Task 8.
3. **Spec step 3 (engine.ts dispatch)** — Task 9.
4. **Spec step 4 (delete `model-inference.ts`)** — Task 11.
5. **Spec step 5 (parity gate)** — Tasks 0 (baseline) + 12 (post).

Bridge expansion (Tasks 1-4) is implicit in the spec — the wrapper at Task 6 needs them. Spec mentions `webllm_get_metadata` explicitly; Tasks 2-4 (KV ops, state-seq, embeddings) are necessary because legacy `ModelInference` exposes the corresponding surface and we preserve API symmetry.

Speculative-decode handling (Task 10) is explicitly out-of-scope per spec §P5 ("rewired in P5") but the type-checker forces a compile-time decoupling now — the plan minimizes the surface (local interface + skipped tests) so P5's actual rewire stays unconstrained.

Bucket-D self-embed (Task 6 `embed()`) is preserved per spec's "Pragmatic tap preservation (B)" decision. P4 (Embedder migration) supersedes this with `llama_get_embeddings_ith` from a single context; P2 uses two contexts to keep blast radius small.

Patch budget: 0 used. Spec's allowance is up to 3 in band B; P2 stays well inside it.

Risk per spec R2 (perf regression > 10%): handled via the perf-recovery sub-phase in Task 12 step 3.
