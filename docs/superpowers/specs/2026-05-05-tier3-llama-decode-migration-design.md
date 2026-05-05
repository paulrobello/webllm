# Tier 3 migration to upstream `llama_decode` — design

Status: proposed, 2026-05-05
Scope: TODO.md item "Tier 3 migration to upstream `llama_decode`
(NEW DIRECTION 2026-05-05)"
Trigger: LFM2.5-350M support request 2026-05-05; investigation
revealed LFM2's hybrid short-conv + GQA layers would require
~500-1000 LOC of new TS graph code that already exists in upstream
`src/models/lfm2.cpp`. This design closes the broader question
("should the TS graph builder layer move to WASM at all?") rather
than just shipping LFM2.

## Problem

This repo's forward-pass surface — `src/inference/model-inference.ts`
(~2890 LOC), `src/inference/encoder-inference.ts` (~565 LOC),
`src/inference/causal-embedder-inference.ts`, and the bulk of
`src/inference/tokenizer.ts` (~1000 LOC of WordPiece + BPE +
pre-tokenizer regex) — re-implements in TypeScript what upstream
`llama.cpp` already implements in C++ for every supported
architecture. Each new architecture (LFM2, Mamba, Qwen3-MoE, GLM,
DeepSeek, hybrids) currently requires a bespoke TS graph builder
mirroring `src/models/<arch>.cpp` upstream. That is sustainable for
the four existing arch families (`llama` / `qwen*` / `phi3` /
`gemma`-ish + BERT family) but has no upper bound: the upstream
arch list grows roughly monthly.

Two compounding costs:

1. **Per-rebase divergence risk.** The 2026-05-04 rebase (`§32`)
   landed cleanly because the patch surface is confined to
   `ggml-webgpu/`. As soon as graph-builder logic depends on
   per-arch tensor naming or pooling conventions that upstream
   evolves (e.g. the recent LFM2 `LLM_TENSOR_OUTPUT_NORM_LFM2`
   "fix for wrong tensor name" rename), TS divergence requires
   manual fixup at every rebase.
2. **Per-arch implementation cost.** LFM2 alone is 500-1000 LOC of
   new TS plus a separate conv-state-cache class plus per-layer
   dispatch on `attention.head_count_kv`. Mamba would need full
   SSM_SCAN plumbing. Each new arch is a multi-day project that
   pays no perf dividend (the C++ source already exists).

A 2026-04-21 design (`docs/superpowers/specs/2026-04-21-decode-graph-build-in-c-design.md`)
proposed moving decode graph build to C as a *perf* lever; that
was deferred (commit `091641a`) because per-step time was
GPU-compute-bound, not graph-build-bound. The current question is
different: it's about *arch portability + maintenance*, not perf.
Per-step perf is expected to be approximately neutral — the win
is scope reduction.

## Goals

1. **Future-proof the project.** Any architecture upstream
   `llama.cpp` supports works in webllm without TS-side
   reimplementation. New hybrid / SSM / MoE architectures land via
   `eval/models.ts` registration plus chat-template wiring only.
2. **Eliminate maintenance burden.** Net delete from TypeScript:
   ~4500-5000 LOC of forward-pass + tokenizer code (per component
   table below). Retain TS only
   for orchestration (engine, sampler, generation, stream-router,
   chat-template, KV-cache lifecycle hooks, public API surface).
3. **Ship LFM2.5-350M as the trailing victory lap.** P6 is
   `eval/models.ts` registration + ChatML stop tokens + smoke
   test. If P0-P5 succeed, LFM2 is trivial.

## Non-goals

- **Not a perf project.** Decode tok/s is expected to be neutral
  (within ±10% of pre-migration on the canonical 6, per stage D
  parity). The 2026-04-21 measurement showed graph build is not
  the per-step bottleneck.
- **Not a public-API redesign.** `LoadedModelMetadata`,
  `ModelArchitecture`, `ChatMessage`, etc., remain stable in shape.
  Internal sources change; consumer-visible types do not.
- **Not a tokenizer-feature reduction.** Streaming detokenizer
  state machine (`tokenizer.ts` `prevText` differential decode)
  is preserved. Only the BPE/WordPiece encoders themselves move
  to upstream.
- **Not a sampler rewrite.** Sampling stays in TS. `llama_decode`
  returns logits; TS picks the next token. Greedy / top-k /
  temperature / steering all unchanged.

## Architecture

### Before

```
TS engine → ModelInference.forward()
   ↓ (~440 wasm.opXxx FFI calls per token, hand-rolling the graph)
ggml graph
   ↓ wasm.graphCompute(graph)
ggml-webgpu backend
   ↓
GPU
```

Per-arch dispatch (split/fused-QKV, NEOX/normal RoPE, phi3 quirks,
flash-attention gating, prefill tiling) all lives in TS.

### After

```
TS engine
   ↓ (1 ASYNCIFY round trip per token)
A3 hybrid bridge surface
    ├─ webllm_decode (async wrapper)         ← hot path
    ├─ webllm_encode (async wrapper)         ← BERT-family
    ├─ webllm_load_model (cwrap)             ← buffer → llama_model
    ├─ webllm_create_context (cwrap)         ← model → llama_context
    ├─ webllm_get_logits / embeddings        ← stateless reads
    ├─ webllm_kv_seq_*                       ← KV ops (reset, rm, cp)
    ├─ webllm_tokenize / detokenize          ← thin pass-through
    └─ webllm_get_metadata                   ← arch, vocab, eos, etc.
   ↓
llama.cpp (patched: 9 existing + budgeted ≤3 new core)
   ↓
ggml-webgpu (unchanged)
   ↓
GPU
```

Per-arch dispatch lives entirely in upstream `src/models/<arch>.cpp`.

## Components

| Component | Status | LOC delta |
|---|---|---|
| `src/core/engine.ts` | retained | ~0 |
| `src/inference/sampler.ts` | retained | ~0 |
| `src/inference/generation.ts` | retained | ~0 |
| `src/inference/stream-router.ts` | retained | ~0 |
| `src/inference/chat-template.ts` | retained | ~0 |
| `src/inference/speculative.ts` | rewired in P5 | ~−100 (delete drafter graph plumbing) |
| `src/inference/model-inference.ts` | **deleted in P2** | **−2890** |
| `src/inference/encoder-inference.ts` | **deleted in P3** | **−565** |
| `src/inference/causal-embedder-inference.ts` | **deleted in P4** | **−~300** |
| `src/inference/tokenizer.ts` | mostly deleted P1 | **−~1000** (retain ~150 LOC streaming detokenizer) |
| `src/models/model-loader.ts` | rewritten in P2 | **−~400** (GGUF parsing collapses to metadata pass-through) |
| `src/inference/llama-decode-wrapper.ts` (new) | added in P2 | **+~150** |
| `src/inference/llama-encode-wrapper.ts` (new) | added in P3 | **+~80** |
| `src/wasm/webgpu-bridge.cpp` | extended | **+~400 C++** for the A3 surface |
| **Net TypeScript delta** | | **~−5000 LOC** |

## Bridge surface (A3 hybrid)

```c
// webgpu-bridge.cpp — new exports

// === Async wrappers (single ASYNCIFY round trip) ===

// Hot decode path: tokens → logits in ctx buffer.
// Returns 0 on success, non-zero llama.cpp status on failure.
int32_t webllm_decode(
    void*    ctx_handle,
    int32_t* token_ids,        // length n_tokens, host pointer (heap32 view)
    int32_t  n_tokens,
    int32_t  past_len,
    int32_t  flags             // bit 0 = WANT_LOGITS, bit 1 = WANT_EMBEDDINGS
);

// Encoder path: tokens → pooled embedding in ctx buffer.
int32_t webllm_encode(
    void*    ctx_handle,
    int32_t* token_ids,
    int32_t  n_tokens
);

// === Direct cwrap bindings (stateless, no async) ===

void*   webllm_load_model(void* gguf_buf, int32_t n_bytes, /* params */);
void    webllm_free_model(void* model_handle);

void*   webllm_create_context(
    void*  model_handle,
    int32_t n_ctx,
    int32_t embeddings,        // 0 = causal LM, 1 = embedder mode
    int32_t pooling_type,      // 0 = NONE, 1 = MEAN, 2 = CLS, 3 = LAST
    int32_t flash_attn         // 0 / 1
);
void    webllm_free_context(void* ctx_handle);

float*  webllm_get_logits(void* ctx_handle, int32_t ith);
float*  webllm_get_embeddings(void* ctx_handle, int32_t ith);

void    webllm_kv_seq_rm(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1);
void    webllm_kv_seq_cp(void* ctx_handle, int32_t src, int32_t dst, int32_t p0, int32_t p1);
void    webllm_kv_clear(void* ctx_handle);

int32_t webllm_tokenize(
    void*    model_handle,
    const char* text,
    int32_t  n_text,
    int32_t* tokens_out,       // pre-alloc'd by JS
    int32_t  n_tokens_max,
    int32_t  add_bos,
    int32_t  parse_special
);
int32_t webllm_detokenize(
    void*    model_handle,
    int32_t* tokens,
    int32_t  n_tokens,
    char*    text_out,
    int32_t  n_text_max
);

// Metadata pass-through — fills a pre-alloc'd struct in JS heap.
void    webllm_get_metadata(void* model_handle, void* meta_out);
```

The metadata struct mirrors the current `LoadedModelMetadata` shape
so JS-side consumers see no change.

## Phase plan

Order: A (tokenizer-first, LFM2 last). Each phase is its own commit
chain on a feature branch; merged only after parity gate passes.

### P0 — Spike

**Goal:** prove the ABI + ASYNCIFY patch viability end-to-end on
one existing model. No user-visible change.

**Steps:**
1. Add `webllm_load_model`, `webllm_create_context`,
   `webllm_decode`, `webllm_get_logits` to bridge (minimum
   subset).
2. Apply ASYNCIFY-aware patch to `llama_decode`'s compute path
   (or write a `llama_decode_async` shim — discover during spike).
3. Apply heap-aware `llama_kv_cache_init` patch if needed (likely
   needed for 16 GiB ceiling models).
4. Build new patched WASM. Existing `make wasm-build` + bridge
   updates.
5. Stand up a one-off TS test harness: load TinyLlama
   Q4_0, decode "The capital of France is" greedy until first
   non-whitespace token, assert top-1 == "Paris".

**Parity gate:** "it works" — top-1 = "Paris", no console errors.

**Exit decision:** if ASYNCIFY patch turns out to need invasive
restructuring, escalate patch budget (B → C) or abort Tier 3 with
a P0 closure report.

### P1 — Tokenizer migration

**Goal:** replace BPE / WordPiece encoders in `tokenizer.ts` with
`llama_tokenize` calls. Retain ~150 LOC streaming detokenizer
state machine (project-specific behavior not exposed by upstream).

**Steps:**
1. Add `webllm_tokenize` / `webllm_detokenize` bridge exports.
2. Rewrite `Tokenizer.encode()` and `Tokenizer.decode()` to
   delegate to bridge. Retain `prevText` streaming logic, special-
   token registration (`getId`), and chat-template-stop-token
   wiring.
3. Add `tests/tokenizer-llama-cpp-parity.test.ts`: 200-prompt
   fixture across all current vocabs (gpt2, llama-bpe, qwen2,
   qwen35, wordpiece). Diff TS encoder output vs `llama_tokenize`.
   Must be byte-identical.
4. Existing `tests/wordpiece-golden.test.ts`,
   `tests/chat-template.test.ts` continue to pass unchanged.
5. Delete BPE / WordPiece / pre-tokenizer regex code in
   `tokenizer.ts`.

**Parity gate (D-byte-exact):** all 200 fixture prompts encode
byte-identically; existing test suite green.

### P2 — Causal-LM migration

**Goal:** delete `model-inference.ts`. Causal LM forward goes
through `webllm_decode`.

**Steps:**
1. Add `llama-decode-wrapper.ts` (~150 LOC) implementing the same
   public surface `ModelInference` exposed (`forward`,
   `forwardSingle`, `loadWeights`, `initKVCache`).
2. Rewrite `model-loader.ts`: GGUF parsing → `webllm_load_model`
   + `webllm_get_metadata`. Retain only the chat-template-family
   detection routing (which keys off `general.architecture` +
   chat template string — both surfaced via metadata).
3. Update `engine.ts:loadModelFromUrl` to use the new wrapper.
   Three-way isEncoder / isCausalEmbedder / isCausalLM dispatch
   stays; only the causal branch flips to wrapper.
4. Delete `model-inference.ts`.
5. Run `make checkall` + `make bench-*` (canonical 6, greedy
   bench per 2026-05-04 doctrine) + `make smoke-bench PERF_RUNS=3`.

**Parity gate (C-strict):**
- Accuracy on canonical 6 within sampling noise (greedy bench).
- Decode tok/s within ±10% on canonical 6 vs pre-migration baseline
  (header block in TODO.md is the source of truth).
- All existing `tests/*.test.ts` green.

**Sub-phase: perf recovery.** If perf regresses >10% on any
canonical-6 model and the cause is identified (e.g.
`llama_context_params.n_ubatch` defaults vs project's tile
heuristic), file a perf-recovery sub-phase. Tune knobs; re-bench;
merge only when within ±10%. If irrecoverable, document and revert
P2.

### P3 — Encoder migration

**Goal:** delete `encoder-inference.ts`. BERT / nomic / jina forward
goes through `webllm_encode`.

**Steps:**
1. Add `webllm_encode` async wrapper using `llama_encode` (or
   `llama_decode` with `embeddings=true` + `pooling_type=MEAN/CLS`
   per arch).
2. Add `llama-encode-wrapper.ts` (~80 LOC).
3. Update `engine.ts` encoder branch + `engine.embed`
   dispatch.
4. Delete `encoder-inference.ts`.
5. Run encoder cosine parity vs HF reference vectors (existing
   `eval/reports/<probe>/`-style ref capture pattern).

**Parity gate (C-strict):** cosine vs HF reference ≥ pre-migration
value (currently 0.76 on arctic-embed-s synonym pair).

### P4 — Embedder migration

**Goal:** delete `causal-embedder-inference.ts`. Causal-LM-as-
embedder reads via `llama_get_embeddings_ith` with `pooling_type=NONE`.

**Steps:**
1. Configure context with `embeddings=true`, `pooling_type=NONE`
   for embedder use of causal LMs.
2. Read post-`output_norm` hidden state via
   `webllm_get_embeddings`.
3. Update `engine.embed` dispatch.
4. Delete `causal-embedder-inference.ts`.
5. Run Bucket-D distinguishability gate (16 synonym + 16 antonym
   pairs, mean-margin per 2026-04-30 doctrine).

**Parity gate (C-strict + Bucket-D):** mean-margin ≥ pre-migration
value on the canonical embedder fleet.

### P5 — Speculative decode + KV consolidation

**Goal:** rewire speculative decode to use two `llama_context`s
sharing one `llama_model`. Consolidate KV cache lifecycle under
`llama_kv_cache`.

**Steps:**
1. Two `llama_context` handles per spec-decode session: drafter
   (smaller `n_ctx`) + verifier (full).
2. KV operations (`webllm_kv_seq_*`) wrap `llama_kv_cache_seq_*`
   directly. Project's `KVCache` class becomes a thin handle
   wrapper.
3. Update `speculative.ts` to call new bridge primitives.
4. Run §C-v2-A spec-decode test fixtures (closed cycle but
   fixtures still exist in `tests/`).

**Parity gate (perf-strict):** spec-decode acceptance rate within
±5% of pre-migration on §C-v2-A fixtures. KV reuse functional
(verified via prefix-cache test suite —
`docs/superpowers/specs/2026-05-01-prefix-cache-design.md` invariants).

### P6 — LFM2 registration

**Goal:** ship LFM2.5-350M-Q8_0.

**Steps:**
1. Extend `ModelArchitecture` union in `src/core/types.ts` with
   `"lfm2"`.
2. Add registration entry in `eval/models.ts` pointing to
   `LiquidAI/LFM2.5-350M-GGUF`. ChatML chat template (auto-
   detected). Stop tokens `<|im_end|>` (id 7) + `<|endoftext|>`
   (id 2).
3. Smoke test in browser via `agentchrome` (reuse session per
   CLAUDE.md).
4. 36-prompt eval + smoke-bench (matches 13B / Phi-3 closure
   pattern).

**Parity gate (D-it-works + 36-prompt eval):** ≥60% accuracy on
36-prompt eval (existing arch-bring-up gate). Decode tok/s
recorded for header block; no perf gate.

## Patch budget

Pragmatic (B): up to ~3 new core llama.cpp patches. Each must be:

- Confined to a single concern (ASYNCIFY suspend, heap-pressure
  KV init, one unknown reserve).
- ≤30 LOC ideally, ≤80 LOC absolute cap.
- Documented in `docs/LLAMA_CPP_PATCHES.md` with an upstream-PR /
  issue link so the patch can disappear over time.
- Reviewed during the next rebase cycle for upstream merge.

**Budgeted:**
1. **`llama_decode` ASYNCIFY-aware compute path.** Internal call
   to `ggml_backend_sched_graph_compute_async` needs to suspend
   inside the WebGPU async readback path. Either patch
   `llama_decode` directly or add `llama_decode_async`. Discover
   exact shape during P0.
2. **`llama_kv_cache_init` heap-pressure-aware sizing.** The 16
   GiB unified-memory floor (`CLAUDE.md` hardware doctrine)
   constrains KV allocation differently than upstream's default.
   May need a custom allocator hook or a `n_ctx_train` override
   path.
3. **Reserve.** Held against unknowns surfaced during P0-P2.

**Escape valve:** if P0 reveals patches >3 are needed, escalate to
Liberal (C). Document the escalation in TODO.md before continuing.

## Research-tap policy

Pragmatic (B). Surviving / dropped:

| Tap | Status | Mechanism |
|---|---|---|
| Bucket-D self-embed (post-`output_norm` hidden) | **survives** | `llama_get_embeddings_ith` w/ `pooling_type=NONE` |
| Speculative decode | **rewired in P5** | Two `llama_context`s + `llama_model` shared weights |
| `debugLayerOutput` (per-layer activation tap) | **deleted** | Parity tests rewritten against logits / public embeddings |
| Per-op timing | **simplified** | `llama_perf_*` aggregate counters replace fine-grained breakdown |
| Custom FA gating per arch | **deleted** | `llama_context_params.flash_attn` |
| Prefill tiling heuristic (§22) | **deleted** | `llama_context_params.n_ubatch` / `n_batch` |

## Risk register

- **R1 ASYNCIFY blocker.** P0 may reveal `llama_decode`'s compute
  path needs deeper restructuring than a single suspend-point
  patch (e.g. internal scheduler creates new async boundaries).
  *Mitigation:* P0 is explicitly the "kill switch" phase.
  Failing P0 means escalating patch budget or aborting Tier 3
  with a closure report; no later phases are committed without
  P0 green.
- **R2 P2 perf regression > 10%.** llama.cpp's prefill scheduler
  (n_ubatch / n_batch defaults, internal tile sizing) may behave
  differently than the project's hand-tuned §22 prefill tiling.
  *Mitigation:* perf-recovery sub-phase before merge; canonical-6
  baselines from TODO.md header block are the source of truth;
  irrecoverable regression triggers P2 revert.
- **R3 Tokenizer fixture mismatch in P1.** Edge cases (chatml
  `<|im_end|>` lookup, Mistral trailing-space-before-`[/INST]`,
  WordPiece phantom-space convention) may diverge from
  `llama_tokenize`. *Mitigation:* byte-exact 200-prompt fixture
  gate exposes divergence at the test layer, never mid-decode.
  Failing fixtures get a small bridge-side adapter (e.g. for
  webllm-specific stop-token lookup that upstream doesn't surface).
- **R4 KV cache allocator interaction.** Project's `KVCache` class
  has WASM-heap-aware sizing; `llama_kv_cache_init` allocates
  eagerly via the model's backend. Memory pressure on 16 GiB
  floor models may trip OOM where current sizing doesn't.
  *Mitigation:* P5 adds the heap-pressure patch (budget item 2);
  fallback is to override `n_ctx` per-model in `eval/models.ts`.
- **R5 Tokenizer streaming detokenizer regression.** The retained
  ~150 LOC `prevText` differential decode is project-specific
  state held across calls; if its inputs change semantically (e.g.
  `llama_detokenize` returns different byte sequences than the
  current TS path on partial UTF-8), streaming output may corrupt.
  *Mitigation:* tests/streaming-detokenize tests retain coverage;
  fixtures cover partial-emoji and partial-CJK cases.
- **R6 Public-API drift.** `LoadedModelMetadata` consumers may
  observe shape changes if metadata extraction has fewer fields
  via `llama_get_metadata` than the current GGUF parser exposes.
  *Mitigation:* preserve the existing struct shape verbatim in
  `webllm_get_metadata`; spike-test against existing consumers
  before P2 merge.

## Success criteria

- **P0:** smoke test ("Paris") green on TinyLlama through
  `webllm_decode`. ASYNCIFY patch + heap patch (if needed) merged
  into `webllm-browser-patches` branch. Patch count ≤9+budget.
- **P1:** 200-prompt fixture parity + existing test suite green.
  ~1000 LOC deleted from `tokenizer.ts`.
- **P2:** Canonical-6 within stage D parity (accuracy + ±10% perf).
  `model-inference.ts` deleted (~2890 LOC).
- **P3:** Encoder cosine parity ≥ pre-migration. `encoder-inference.ts`
  deleted (~565 LOC).
- **P4:** Bucket-D distinguishability gate ≥ pre-migration. Causal-
  embedder TS class deleted.
- **P5:** Spec-decode acceptance rate ±5%. KV consolidated under
  `llama_kv_cache`.
- **P6:** LFM2.5-350M registered, smoke green, 36-prompt eval ≥60%.
- **Aggregate:** ~4500-5000 net LOC deleted from TS (per
  component-table arithmetic; +~400 LOC C++ in bridge offset).
  Future arch
  additions become `eval/models.ts` registrations + chat-template
  family entries only.

## Open questions

- **OQ1 `llama_model_load_from_buffer`.** Stock upstream uses
  `llama_model_load_from_file`; loading from a JS-side ArrayBuffer
  may need a small bridge helper that mounts the buffer in
  Emscripten MEMFS or a direct `llama_model_loader` invocation.
  Resolve in P0.
- **OQ2 Embedder mode + flash_attn interaction.** Some embedder
  configs disable FA upstream; verify the project's current FA
  gating matches what `llama_context_params.flash_attn=1`
  produces. Resolve in P3-P4.
- **OQ3 Spec-decode KV-share semantics.** Whether two
  `llama_context`s on one `llama_model` share weight buffers
  cleanly under WebGPU (vs. duplicating tensors) determines spec-
  decode memory cost. Resolve in P5.
- **OQ4 Greedy bench reproducibility.** Per 2026-05-04 doctrine
  bench is greedy; verify `llama_decode` produces deterministic
  logits given identical seed + identical inputs (it should — no
  sampler involvement). Resolve in P2.
