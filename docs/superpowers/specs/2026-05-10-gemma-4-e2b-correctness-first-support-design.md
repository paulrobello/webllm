# Gemma 4 E2B inference support — correctness-first staging

**Date:** 2026-05-10
**Status:** Draft, awaiting user review
**Target SKU:** `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M (~3.11 GB)
**Symlink:** `smoke-test/models/gemma-4-e2b-it-q4km.gguf`
**Probe artifact:** Phase 1 load attempt 2026-05-10 (no commit — page snapshot in conversation transcript)

## 1. Context

Google released Gemma 4 on 2026-04-02 under Apache 2.0 in four SKUs
(E2B, E4B, 26B A4B MoE, 31B Dense). Only **E2B (2.3B effective)** and
**E4B (4.5B effective)** fit the project's 8B parameter ceiling and 16 GB
hardware floor. Upstream llama.cpp shipped full text-path support via
PRs #21343 / #21390 / #21500 / #21534 / #21612 / #21739 / #22027,
all present in the local `webllm-browser-patches` branch baseline.
ggml-webgpu's GGUF reader handles the 262K vocab via existing chunked
binding dispatch (confirmed by probe — weights loaded in 0.4 s).

**Why ship this:** E2B Q4_K_M (3.11 GB) is the smallest Apache-2.0
model in the fleet with frontier-class capability at this scale and
the first Gemma-family model that ships through the GGUF/WASM path
(the existing `gemma-2-2b-q4f16` registration targets the deprecated
MLC/pure-WGSL path). Adding Gemma 4 also retires the longest-deferred
architecture in `TODO.md` (the "Wave-1 architecture Gemma 2" item was
deferred 2026-04-29 with 5+ gaps — those gaps now have a budget).

## 2. Probe findings (Phase 1, 2026-05-10)

Loaded the Q4_K_M GGUF in the unmodified smoke harness. Result:

- ✅ GGUF parsed: `arch=gemma4 emb=1536 heads=8/1 layers=35 vocab=262144 ctx=131072`
- ✅ Weights loaded in 0.4 s (262K vocab × Q4_K within 128 MiB binding cap)
- ✅ KV cache allocated: 4096 slots × 35 layers
- ✅ Tokenizer ready: `encode("hello") = [23391]`
- ❌ Shader-cache warmup failed (`unreachable`)
- ❌ Generation: `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)` at
  `opReshape3d` in `ModelInference.buildQKV`

Full GGUF metadata dump confirms seven architectural deltas vs the
project's current causal-LM dispatch:

| GGUF key | Value | Project impact |
|---|---|---|
| `gemma4.attention.key_length` | **512** | Global head_dim ≠ 1536/8=192 — `hp.embeddingHeadLength` is scalar, code path computes Q reshape to `(headDim, 8, n_tokens)` against 192 instead of 512 |
| `gemma4.attention.key_length_swa` | **256** | **Different head_dim** on SWA layers — scalar storage cannot represent both |
| `gemma4.attention.sliding_window_pattern` | `(T,T,T,T,F)×7` | 4 local + 1 global per cycle — project has no SWA dispatch |
| `gemma4.attention.sliding_window` | 512 | Window size for local layers |
| `gemma4.attention.shared_kv_layers` | **20** | Last 20 of 35 layers reuse KV from earlier layer — per-layer KV-cache allocator assumes independence |
| `gemma4.embedding_length_per_layer_input` | **256** | Per-Layer Embeddings table — net-new path |
| `gemma4.feed_forward_length` | `[6144×15, 12288×20]` | FFN dim varies per layer — scalar `hp.feedForwardLength` cannot represent |
| `gemma4.final_logit_softcapping` | **30.0** | Still present in Gemma 4 (`logit_softcap` plumbed in `ggml-wasm.ts`; needs GGUF-driven value through registration) |
| `gemma4.rope.dimension_count` / `_swa` | 512 / 256 | Dual RoPE dimensions per layer-type |
| `gemma4.rope.freq_base` / `_swa` | 1M / 10k | Dual RoPE freq_base per layer-type |
| `gemma4.embedding_length` | 1536 | hidden |
| `gemma4.attention.head_count` / `_kv` | 8 / 1 | 8:1 GQA (more aggressive than Gemma 2/3) |
| `gemma4.block_count` | 35 | layers |

## 3. Scope decisions

1. **SKU coverage.** E2B first (this spec). E4B as a follow-on probe
   after Stage 5 closes; same architectural code path so no extra
   work expected beyond a new registration + smoke validation.
2. **Stage 1 fidelity.** **Correctness-first** (user decision, 2026-05-10).
   Stages 1–3 ship a "sloppy but right" shape: full causal attention
   on every layer (no SWA window), PLE table GPU-resident, shared-KV
   materialized as separate per-layer K/V allocations. Stage 4 lifts
   real SWA; Stage 5 lifts ref-shared KV. PLE size to be confirmed by
   GGUF `tensor_info` probe before Stage 3 (back-of-envelope on
   `35 × 262144 × 256` = 2.35 B elements suggests ~1.2 GB at Q4_K,
   ~2.4 GB at Q8, ~4.7 GB at f16 — community reports ~4.7 GB at Q8
   for an E4B-equivalent CPU offload, so this needs ground truth
   before sizing the residency decision). CPU offload deferred unless
   the GPU residency cost is unacceptable.
3. **Staging shape.** Linear 5-stage (user decision, 2026-05-10),
   matching the §17/§19/§20 cadence and the Phi-3 closure 6-phase
   pattern (compressed because Stage 5 absorbs phases 5–6 of Phi-3).
4. **Patch budget.** Expect 0 patches in Stages 1–3 (all changes
   land in webllm TS). Stages 4 (SWA) and 5 (shared-KV) likely
   require 1 patch each to ggml-webgpu inside the
   `webllm-browser-patches` branch. Patch stack grows 9 → ~11.
5. **Gate philosophy.** Each stage has both a **build gate**
   (`make checkall` green) and a **runtime gate** (a measurable
   behavioral assertion in browser). No stage merges until its
   runtime gate fires positive on the canonical smoke harness.

## 4. Stage design

### Stage 1 — Per-layer hyperparams refactor (foundation)

**Purpose.** Convert scalar `embeddingHeadLength`, `feedForwardLength`,
`ropeDimensionCount`, and `ropeFreqBase` into per-layer arrays
(or per-layer-type pairs `{global, swa}`). For existing models
(Llama / Mistral / Qwen / Phi / Gemma 2), the array is filled with
the scalar replicated `layerCount` times — zero behavioral delta.
For Gemma 4, populated from GGUF per-key values.

**Files (estimated touch):**
- `src/core/types.ts` — `ModelHyperparams` field changes (per-layer
  arrays, plus retained scalars as convenience accessors for the
  "all layers same" case)
- `src/models/model-loader.ts` — GGUF read paths populate arrays
- `src/inference/model-inference.ts` — every `hp.embeddingHeadLength` /
  `hp.feedForwardLength` site accepts a layer-index parameter and
  indexes the array
- `src/inference/causal-embedder-inference.ts` — same
- `src/inference/encoder-inference.ts` — same
- Test surface: 6+ test files reference these scalars; verify all
  still pass via diff-mode

**Stage 1 gates:**
- **Build:** `make checkall` green (747 → 747 tests pass)
- **Runtime A (no regression):** smoke harness loads + greedy-decodes
  one token correctly on TinyLlama / qwen3-0.6b / qwen3-1.7b
  (compare against `generatedIds[0]` from current dashboard records)
- **Runtime B (Gemma 4 hp populated):** `?model=gemma-4-e2b-it-q4km`
  page log shows `hp.embeddingHeadLength` array `[512×35]` (or per-
  layer-type pair `{global:512, swa:256}` depending on impl choice)

**Closure artifact:** `eval/reports/gemma-4-stage1-per-layer-hp-<date>/SUMMARY.md`

### Stage 2 — Gemma 4 surface wiring (template / sampler / registration / softcap)

**Purpose.** Land the non-kernel surface required to dispatch
Gemma 4 through `ModelInference` end-to-end: chat template, stop
tokens, sampler defaults, `final_logit_softcapping` value, and
the registration entry.

**Files (estimated touch):**
- `src/core/types.ts` — add `"gemma4"` to `ModelArchitecture` union
- `src/inference/chat-template.ts` — new `formatGemma4` formatter
  (port the GGUF jinja template — supports tool calls via custom
  format that PR #21326 added a PEG parser for)
- `src/inference/chat-template.ts:25` — extend `detectChatTemplate`
  to map Gemma 4-specific markers (likely also `<start_of_turn>` —
  needs probe-time verification)
- `src/core/engine.ts:605/1060` — stop-token registration includes
  Gemma 4's `<end_of_turn>` and any new turn-end tokens
- `src/core/sampling-profiles.ts` — `GEMMA4_DEFAULTS` profile
  (temperature 0 for bench; runtime defaults TBD against the model's
  HF generation config — typically `temp=1.0, top_p=0.95, top_k=64`
  per Gemma model card)
- `src/inference/model-inference.ts` — read `final_logit_softcapping`
  from GGUF metadata, plumb through to `opSoftMaxExt` call site
  (the existing `logit_softcap` parameter; currently always 0.0)
- `eval/models.ts` — new `gemma-4-e2b-it-q4km` registration entry
- Bundle regeneration: `bun build eval/models.ts → smoke-test/webllm-models.js`

**Stage 2 gates:**
- **Build:** `make checkall` green
- **Runtime (smoke chat):** browser smoke loads, generates greedy
  5 tokens on prompt `"The capital of France is"` → output is
  coherent ASCII (not control chars, not repetition of input); does
  not yet require semantic correctness (still using fallback attention)
- **Runtime (stop-token):** multi-turn chat ends on `<end_of_turn>`
  cleanly, no runaway

**Closure artifact:** `eval/reports/gemma-4-stage2-surface-wiring-<date>/SUMMARY.md`

### Stage 3 — PLE injection + dual RoPE dispatch

**Purpose.** Add the two correctness-critical new paths so generation
matches the model's training distribution: (a) Per-Layer Embeddings
(256-dim per layer added into the residual stream) and (b) dual RoPE
dispatch (the global RoPE has 512 dim @ 1M base for global-attention
layers, 256 dim @ 10k base for SWA layers — even though all layers
will still be running full attention until Stage 4, the RoPE dim/base
must match per-layer-type or the K and Q angles will be wrong).

**Files (estimated touch):**
- `src/models/model-loader.ts` — load `per_layer_token_embd.weight`
  table (35 × 262144 × 256 → ~9.2 GB at f16, ~1.3 GB at Q4_K — confirm
  GGUF stores it quantized; verify it fits the 128 MiB binding cap via
  same chunked path as token_embd, else patch needed)
- `src/inference/model-inference.ts` — at each layer's start, look up
  `per_layer_embd[layer_idx, token_id, :]` for each token and add into
  the residual; dual-RoPE dispatch in `buildQKV`'s RoPE step (read the
  per-layer rope_dim + freq_base from the Stage 1 arrays)
- May need a new `opAddPerLayerEmbedding` helper or just reuse `opAdd`
  with a strided view into the PLE table

**Stage 3 gates:**
- **Build:** `make checkall` green
- **Runtime (semantic):** greedy 5 tokens on `"The capital of France is"`
  produces `Paris` (or " Paris" — tokenizer-dependent) as the first
  generated token. This is the canonical "model semantically works"
  gate the project uses for new arch validation (cf. Phi-3 closure,
  bucket D parity check)
- **Eval gate:** 36-prompt eval ≥40% (greedy @ temp=0). Phi-3
  scored 72%; the looser 40% gate here accounts for the residual
  quality cost of Stage 4's still-pending real SWA — a model trained
  with 5:1 local:global will degrade noticeably under all-global
  attention but should still demonstrate baseline competence.

**Closure artifact:** `eval/reports/gemma-4-stage3-ple-dualrope-<date>/SUMMARY.md`

### Stage 4 — Real sliding-window attention

**Purpose.** Replace the Stage-3 "all-global" fallback with real
sliding-window attention on the 4-of-5 layers marked local in the
GGUF pattern. Window size 512 tokens; mask + KV-cache-window logic
match the upstream llama.cpp implementation.

**Files (estimated touch):**
- `src/inference/model-inference.ts` — per-layer attention dispatch:
  use windowed mask for SWA layers, full causal mask for global layers
  (current behavior). KV cache reads are bounded to the window for
  SWA layers — for the correctness-first path we still *write* the
  full cache (memory-wasteful but mechanically simple); a follow-on
  micro-optimization can size the SWA layers' KV pools to
  `min(window, maxCtx)`.
- `src/wasm/ggml-bindings.cpp` (or equivalent) — verify the WGSL
  attention path masks correctly with a window offset; may need
  a new mask builder if the existing one assumes full causal only.
  **Risk:** if ggml-webgpu's softmax-with-mask path can't express
  the window mask shape, this stage takes 1 llama.cpp patch.

**Stage 4 gates:**
- **Build:** `make checkall` green
- **Eval gate:** 36-prompt eval ≥60% (matches Phi-3 closure standard).
  The lift from Stage 3's 40% to 60%+ here is the **load-bearing
  validation** that real SWA is correct — if eval doesn't lift,
  the SWA mask is wrong.
- **Long-context probe:** generate 1000 tokens; no quality cliff at
  the 512-token window boundary (smoke-test that the SWA layers are
  attending to within-window tokens correctly)

**Closure artifact:** `eval/reports/gemma-4-stage4-swa-<date>/SUMMARY.md`

### Stage 5 — Shared-KV reference sharing + smoke-bench + closure report

**Purpose.** Wire the last 20-of-35 layers' shared K/V references
into the KV cache allocator (currently materializing duplicate K/V
buffers wasted ~3 GB), then run the smoke-bench profile-mode pass
for the speed headline, then write the closure report.

**Files (estimated touch):**
- `src/models/kv-cache.ts` — allow a layer's K/V tensors to point
  at another layer's allocated buffer (ref-shared); per-conversation
  snapshot/load paths must respect the sharing graph
- `src/inference/model-inference.ts` — the `attn_k.weight` /
  `attn_v.weight` lookup at each shared layer reads from the
  source layer's projection result (or loads the same weight tensor
  if GGUF stores duplicates — verify by inspecting `tensor_info`
  for `blk.20.attn_k.weight` and confirming it's a reference to
  `blk.0.attn_k.weight` or similar)
- `src/persistence/indexeddb-store.ts` — KV serialization must
  handle ref-shared layout (no naive `concat(per_layer_kv)`)

**Stage 5 gates:**
- **Build:** `make checkall` green
- **Memory gate:** VRAM usage on E2B Q4_K_M drops by ≥2 GB vs Stage 4
  baseline (target: ~5 GB → ~3 GB after ref-sharing; verifies the
  ref-share landed)
- **Bench gate:** smoke-bench profile-mode 3-run median ≥ **10 tok/s**
  (loose floor — E2B is small enough that ~30+ tok/s is plausible;
  10 is the "didn't accidentally regress 10×" guard)
- **Eval gate:** 36-prompt eval at least as good as Stage 4 closure
  (target: ≥60%, ideally higher since the ref-shared KV is
  load-bearing for the model's trained behavior)
- **Closure report:** `eval/reports/gemma-4-e2b-validation-<date>/SUMMARY.md`
  following the Phi-3 closure template. Includes: arch deltas table,
  per-stage commit list, eval results, bench results, follow-up items.

## 5. Out of scope (this spec)

- **E4B SKU** — same architecture, deferred to a Stage-5-follow-on
  probe; expected to be a registration + smoke validation only.
- **PLE CPU offload** — community pattern for large-vocab Gemma 4
  variants. GPU-resident E2B PLE costs ~150 MB (35 × 256 × 262144 × Q4_K).
  Defer unless memory pressure makes it unviable on the 16 GB floor.
- **MoE 26B A4B and dense 31B SKUs** — exceed the 8B ceiling; defer
  to External-trigger candidates.
- **Multimodal (vision / audio)** — vision adapters ship in
  `mmproj-*.gguf` files; out of scope for chat-only.
- **MTP drafter** — Gemma 4's multi-token-prediction drafter is
  documented in upstream llama.cpp Discussion #22735 (still open).
  Defer until upstream support lands.

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stage 1 hp refactor breaks existing models (silent regression) | Medium | Stage 1 runtime gate requires `generatedIds[0]` match against current dashboard records on 3 representative models |
| Stage 3 PLE table exceeds 128 MiB binding cap at f16 | Low (Q4_K likely fits) | Chunked dispatch already proven on `token_embd`; same path applies |
| Stage 4 SWA mask shape unsupported by ggml-webgpu | Medium | Pre-implementation probe: synthetic windowed-mask softmax through existing op; if fails, scope 1 llama.cpp patch |
| Stage 5 ref-shared KV breaks persistence/snapshot | Medium | Persistence test surface explicitly covered (`engine-conversation-persistence.test.ts`); skip-gate stays at 33 |
| Eval gate misfires (model semantically wrong but eval grades okay) | Low | Phi-3 closure showed eval signal is reliable at temp=0 greedy; spot-check with same `"capital of France"` smoke that gated Stage 3 |
| Wall-clock blows out beyond 5 sessions | Medium | Stage gates are independent — partial credit lands (e.g., Stages 1–3 ship even if Stage 4 SWA stalls); intermediate state has documented utility |

## 7. References

- **GGUF metadata dump:** captured 2026-05-10 in conversation transcript
  (run via `gguf-py/GGUFReader` from local llama.cpp checkout)
- **llama.cpp Gemma 4 PRs in `webllm-browser-patches` baseline:**
  #21343 (vocab fix), #21390 (final_logit_softcapping read),
  #21500 (add_bos), #21534 (vocab tests + edge case),
  #21612 (per-layer projections first-layer), #21739
  (shared-KV tail attn_k optional), #22027 (model type detection)
- **Tool-call PEG parser:** llama.cpp PR #21326
  (`normalize_gemma4_to_json()`)
- **Research note:**
  `~/ParsidionVault/Research/gemma-4-support-feasibility.md`
  (2026-05-10; load-bearing for SKU sizing and licensing context;
  partially superseded by Stage 1 probe — research undercounted
  architectural deltas)
- **Phi-3 closure template:**
  `eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`
  (canonical 6-phase pattern; Stages 5 of this spec compress phases 5–6)
- **HuggingFace Q4_K_M source:** `unsloth/gemma-4-E2B-it-GGUF`
- **CLAUDE.md doctrines in play:**
  per-binding 128 MiB cap, single-model-active deployment,
  always-commit-before-work, phased execution, hfdownloader-first

## 8. Acceptance criteria

The work is "done" when:

1. All 5 stages have green build + runtime gates
2. Stage 5 closure report lands in `eval/reports/gemma-4-e2b-validation-<date>/`
3. `make checkall` is green (skip-count ≤ 33 baseline)
4. Dashboard shows a Gemma 4 E2B entry under the normal benchmark
   model fleet (visible via `populateDropdown`)
5. `TODO.md` "Active next steps" section moves the Gemma 4 block
   from queued to a closure stub linking to the Stage 5 report
6. Local llama.cpp patch stack growth bounded at +2 max (Stages 4/5)
