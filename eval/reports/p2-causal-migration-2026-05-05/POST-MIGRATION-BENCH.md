# P2.1 — Post-migration bench retake (HALTED — major perf regression)

> **Date:** 2026-05-05
> **Tip:** `bd7ae4b fix(p2): port smoke-test page to LlamaDecodeWrapper` (post-Task-11 cleanup)
> **Verdict:** **HALTED** — first canonical-6 model showed an **18× decode-throughput regression** vs the legacy baseline. The full sweep was stopped before further measurements; per spec §P2 R2 (perf regression > 10%) this triggers a perf-recovery sub-phase or a P2 revert.

## What happened

After landing the smoke-page port (commit `bd7ae4b` — fixes a `ModelInference is not a constructor` blocker that Task 11 missed and a heap-grow-detachment bug in `bridge.loadModel` when the caller's buffer lives in the WASM heap), the smoke test loads tinyllama end-to-end successfully through the new `LlamaDecodeWrapper` path. Functional correctness is preserved: model loads, tokenizes, generates 64 coherent tokens, returns clean output. **But the throughput is catastrophic:**

```
[7/8] Generated 64 tokens in 18.3s (prefill: 8026ms, decode: 10274ms,
      6.2 tok/s, finish=max-tokens, tokensIn=57)
User: Tell one short joke.
Assistant: Here's a one-liner that sums up the short joke: "There is
a person who lives inside a giant pineapple."
```

| Metric | Legacy (TODO.md pin) | Post-P2 | Regression |
|---|---|---|---|
| Decode tok/s | 110.8 | 6.2 | **−94% (18×)** |
| Prefill (57 tokens) | ~50 ms | 8026 ms | **160× slower** |
| Per-decode-token | ~9 ms | ~160 ms | 18× |

The spec budgets ±10%; this is two orders of magnitude over. **No further canonical-6 measurements were taken** — the regression magnitude is unambiguous from a single model.

## Diagnosis

**Most likely root cause:** `llama_decode` is not optimized for token-by-token decode in the single-batch=1 case. Each call rebuilds the compute graph from scratch (full forward pass through every layer's tensor allocator), where the legacy `ModelInference.forwardSingle` cached the graph across decodes and only re-uploaded position + token-ids tensors per step. For a 22-layer model at batch=1, the graph build cost dominates everything else.

**Secondary contributors (smaller):**
- 2 JSPI suspend points per token (`_webllm_decode` + `_webllm_get_logits`) vs 1 in legacy (`graphCompute`). ~2-20 ms of suspend overhead per token.
- JS-side sampling reads back the full 32 K-vocab logits per token (~128 KB per readback). Legacy used GPU-side `forwardDecode` (argmax/topk) when greedy. Maybe ~5-10 ms per token.
- Per-decode `bridge_malloc` of 4 bytes for the single-token batch (no impact in steady state).

The first item explains 14-16× of the regression; the rest is noise.

## Spec gate

Per [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md) §P2 R2:

> **R2 P2 perf regression > 10%.** llama.cpp's prefill scheduler may
> behave differently than the project's hand-tuned tiling. Stage D
> allows a perf-recovery sub-phase before merge — tune
> `llama_context_params.n_ubatch` first.

> **Sub-phase: perf recovery.** If perf regresses >10% on any
> canonical-6 model and the cause is identified (e.g.
> `llama_context_params.n_ubatch` defaults vs project's tile
> heuristic), file a perf-recovery sub-phase. Tune knobs; re-bench;
> merge only when within ±10%. If irrecoverable, document and revert
> P2.

The regression is real, identified, and far above the budget. Stage D fires.

## Recovery candidates

In order of estimated impact and complexity:

1. **Graph-cache patch on `llama_context`** *(highest impact, highest cost)*. Add a llama.cpp patch (band C — Liberal — escalation needed) that reuses the compute graph across decode calls when the batch shape is unchanged. This is what the legacy `ModelInference.forwardSingle` did manually. Upstream may already be working on this; check recent issues / PRs on `ggerganov/llama.cpp`. Likely the ONLY recovery path that closes the prefill regression.

2. **`n_ubatch` tuning** *(low impact at batch=1)*. Spec recommends this first. Default in llama.cpp is 512; for single-token decode it's a no-op. Worth a quick measurement but unlikely to recover materially.

3. **GPU-side sampling for greedy/topk** *(modest impact)*. Add a `webllm_decode_argmax` bridge primitive that runs argmax on the logits before readback, returning just the token id. Saves ~10 ms × 64 tokens = ~600 ms per generation. Doesn't address prefill or the bulk of the per-token cost.

4. **JSPI batching across decode steps** *(speculative, modest)*. Group multiple decode calls into one suspend transition. Needs a `forwardN(tokensList, sampleFn)` bridge primitive that loops in WASM. Risky — sampling state lives in JS, not WASM.

5. **Revert P2** *(no recovery; preserves prior architecture)*. Roll back `4bb644c..bd7ae4b`. Tier 3's arch-portability win goes away. P3-P5 stay deferred until upstream graph-cache is available or a different migration path is chosen.

## Recommendation

**Pause P2 at the current tip and surface a decision point to the user:**

- **Path A:** Investigate option #1 (llama.cpp graph caching). Likely a multi-day deep dive into `llama-context.cpp` and `llama-graph.cpp`. Patch budget escalates from band B (3 reserved) to band C (Liberal). High-uncertainty — the upstream maintainers may have intentional reasons single-token graph caching isn't already there.

- **Path B:** Accept perf regression and ship anyway. Counter to spec stage D and to the "agent + Three.js coexistence" project doctrine in CLAUDE.md (6 tok/s on tinyllama makes agents noticeably sluggish; 4-6 sec per turn just for decode). NOT recommended.

- **Path C:** Revert P2. Roll back the 11 implementation commits (`4bb644c..374cc46`) plus the smoke-page port (`bd7ae4b`). Keep the closure docs as a record. The arch-portability motivation for Tier 3 stays valid; revisit when upstream `llama_context` ships graph caching for single-token decode (track via upstream issues).

## Artifacts

- Smoke log capture: `[7/8] Generated 64 tokens in 18.3s (prefill: 8026ms, decode: 10274ms, 6.2 tok/s)` — single tinyllama run on `bd7ae4b`.
- Bench harness wedge from earlier session: separate root cause (stale tab + JS error from smoke-page `ModelInference` reference), now fixed by `bd7ae4b`. Re-running the full bench-profile sweep would now exercise the new path on all 6 canonical models — but each model would compound the same per-token regression.

## Next steps (pending user decision)

This document deliberately stops short of taking corrective action. The choice between Path A, B, and C is high-stakes and outside the autonomy granted to the implementer agent.

---

## P2.1.A — Path A investigation findings (2026-05-05)

The user selected **Path A** (investigate llama.cpp graph caching). The investigation produced a **negative result**: none of the candidate band-B/C patches recover the regression. Path A is concluded; a fresh decision is required.

### Empirical findings (four diagnostics)

**H1 — Per-call graph rebuild** *(headline hypothesis from §"Diagnosis")*: **DISPROVEN.**
Wired a `webllm_perf_counter(ctx, field)` bridge export reading `llama_perf_context()` directly. After running a 64-token decode on tinyllama, the wrapper logged `n_reused / n_decode ≈ 97%`. Upstream's graph-reuse fast path (`can_reuse(gparams)`, `gf_res_prev`, `n_reused` counter) **is engaging** on ~every decode step. The graph build cost is **not** the dominant per-token cost.

**H2 — JSPI `emscripten_sleep(1)` polling loop in `ggml_backend_webgpu_wait_queue`**: **DISPROVEN.**
Patched `~/Repos/llama.cpp/ggml/src/ggml-webgpu/`:
- `CMakeLists.txt`: added `target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_JSPI=1)` under the `GGML_WEBGPU_JSPI` branch.
- `ggml-webgpu.cpp`: changed `#ifdef __EMSCRIPTEN__` → `#if defined(__EMSCRIPTEN__) && !defined(GGML_WEBGPU_JSPI)` in `wait_queue` and `map_buffer`. JSPI now falls through to the native `wgpu::Instance::WaitAny` path (one suspend per call, no polling).
- Rebuilt webllm WASM, re-ran tinyllama smoke. **Per-token decode: ~158 ms → ~155 ms.** Within noise (~2%); no material recovery.

**H3 — end-of-graph `wait_queue` is the bottleneck**: **DISPROVEN.**
As a diagnostic, commented out the `wait_queue` call at end of `graph_compute`. Per-token decode unchanged at ~155 ms. Reverted the diagnostic; restored the `wait_queue` call. (The H2 patch — JSPI uses `WaitAny` — is correctness-preserving regardless of perf, and remains uncommitted in `~/Repos/llama.cpp/`.)

**H4 — batch-decode amortization**: did not test. Single-batch decode is the load-bearing case for agent + chat workloads; even if larger batches amortized the cost, the per-token interactive latency is what matters.

### Conclusion

**Time IS spent inside `_webllm_decode`, but not in any single isolated stage** (graph build ✗, polling waits ✗, end-of-graph sync ✗). The remaining suspect is **JS↔WASM boundary cost per WebGPU dispatch under JSPI**: a single transformer forward issues ~150 dispatch calls (per-layer attention + MLP shaders), each crossing the boundary, and observed per-token cost (~150 ms) divides into ~1 ms per dispatch — consistent with documented JSPI suspend/resume overhead on Chrome/Apple V8.

This is **intrinsic to the emscripten + Dawn + JSPI stack**, **NOT fixable with band-B or band-C llama.cpp patches**.

### Refined decision

The original A/B/C tree narrows. New options:

- **Path A.4 — Revert JSPI; restore ASYNCIFY** *(multi-day, speculative)*. The 14 new P2 bridge exports (`webllm_load_model`, `webllm_decode`, etc.) currently rely on JSPI for `await`-able transitive calls into the WebGPU backend. Switching back to ASYNCIFY requires re-listing every transitive caller of `emscripten_sleep` in `ASYNCIFY_IMPORTS` / `ASYNCIFY_ADD`, plus regression-testing ASYNCIFY_STACK_SIZE against the new call-graph. Risk: ASYNCIFY may have its own per-suspend cost on the WebGPU dispatch path; we'd be trading one async mechanism for another with no measured upside.

- **Path B — Accept regression; ship slow**. 6.2 tok/s on tinyllama violates the "agent + Three.js coexistence" doctrine in [CLAUDE.md](../../../CLAUDE.md). Not recommended.

- **Path C — Revert P2 entirely**. Roll back commits `4bb644c..374cc46` + smoke-page port `bd7ae4b`. Keep this report and `SUMMARY.md` as the record. Tier 3's arch-portability win is deferred until upstream eliminates the per-dispatch JSPI boundary cost (track via emscripten / Dawn issues — likely needs WebGPU's future WASM-native bindings, multi-quarter horizon at minimum).

### Recommendation

**Path C** is the lowest-risk option. Path A.4 is speculative and would consume another multi-day sub-phase with no measurement supporting that ASYNCIFY's overhead is materially lower than JSPI's on this workload. The arch motivation for Tier 3 is real, but the regression is two orders of magnitude over budget — shipping P2 in the current form would inflict the regression on every downstream consumer.

### Artifacts (uncommitted, awaiting decision)

- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/CMakeLists.txt` + `ggml-webgpu.cpp`: JSPI `WaitAny` patch (13 lines). Correctness-preserving regardless of which path is chosen — should be committed (or upstreamed) under any path.
- `src/wasm/webgpu-bridge.cpp` + `CMakeLists.txt`: `webllm_perf_counter` diagnostic export (~25 LOC). Useful for future perf probes; can be kept as-is or trimmed.
- `src/inference/llama-decode-wrapper.ts`: `__mainCtxHandle` getter (`@internal`, 7 LOC). Same — keep or trim.

---

## P2.1.B — Architectural reframe: the JSEP pattern (2026-05-05)

The framing of Path A/B/C/A.4 above misses a third real architecture. The original A/B/C tree implicitly assumed any GPU-accelerated path **must** route every WebGPU call through WASM; under that constraint, the per-dispatch shim cost is intrinsic. That assumption is wrong — there's a production existence proof to the contrary.

### The existence proof: transformers.js + ONNX Runtime Web

[transformers.js](https://huggingface.co/docs/transformers.js/index) ships browser-side LLM inference using ONNX Runtime Web (`onnxruntime-web`). Its WebGPU execution provider is implemented as **JSEP** — the JS Execution Provider — visible in the ORT-Web source tree at `onnxruntime/js/web/lib/wasm/jsep/`. The architecture:

1. ORT-Web's WASM core handles model loading, graph optimization, and kernel-dispatch decisions.
2. **WebGPU EP kernels are implemented in TypeScript, not C++** (`jsep/webgpu/ops/*.ts`).
3. When the WASM scheduler picks a kernel, it calls back into JS via a registered callback table (Emscripten's `Module["jsepRunKernel"]`).
4. The JS kernel records WebGPU dispatches **directly against `navigator.gpu`** — no Dawn, no `emdawnwebgpu`, no C++→JS shim.

The WASM↔JS boundary in ORT-Web's WebGPU EP is at the **kernel** level (one crossing per op, ~20-50 per token), not the **WebGPU command** level (hundreds per token). The per-dispatch shim cost we hit in Tier 3 P2 v1 simply doesn't exist in their architecture.

This is not theoretical. transformers.js + ORT-Web has been shipping JSEP for two years and is the de-facto template for browser GPU LLM inference.

### Path 3 — JSEP-style architecture for webllm

A faithful adaptation to the llama.cpp + ggml stack:

- **WASM-side:** llama.cpp's tensor allocator, graph builder, scheduler, KV cache, sampler, tokenizer all stay. A new ggml backend (`ggml-jsep` or similar) replaces `ggml-webgpu` in the WASM build. Its kernel implementations don't issue Dawn/wgpu calls — they emit `EM_ASM` / Emscripten callbacks that hand the JS side a {op type, operand handles, dims, strides} descriptor and await a "kernel done" callback.
- **JS-side:** a TS module implements the WebGPU recording for each ggml op (matmul, RMS norm, rotary, attention, MLP variants, quant-dequant, etc.). It speaks WebGPU directly via `navigator.gpu` — same model the legacy `ModelInference` uses, but driven by the WASM scheduler instead of a TS-side graph.
- **Shared:** WGSL shaders are reused from `ggml-webgpu`'s shader directory (which is already pre-generated and string-embedded today).

### Comparison table

| Path | Architecture | Maintenance burden | 8B perf ceiling | Existence proof |
|---|---|---|---|---|
| Legacy `ModelInference` | TS-side WebGPU, hand-rolled per upstream shader change | High — manual port of every upstream `ggml-webgpu` change | Good (110 tok/s tinyllama observed; ~25-35 tok/s 8B Q4 expected) | webllm `model-inference.ts` itself |
| Tier 3 P2 v1 (`ggml-webgpu` in WASM via Dawn) | C++ kernels in WASM, Dawn shim → JS WebGPU per call | Low — upstream rebases just work | **Capped by shim cost** (~6 tok/s tinyllama observed) | This report |
| **Path 3 — JSEP-style** | WASM scheduler, TS kernels recording WebGPU directly | Medium — TS kernel ports per ggml op (smaller + more stable than full shaders) | Should match legacy (no shim cost; same JS WebGPU API surface) | **transformers.js + ORT-Web JSEP** |

### Why this changes the recommendation

Path 3 captures most of Tier 3's maintenance win (the load-bearing parts — graph builder, KV cache, sampler, tokenizer, ggml ops layout — stay upstream), without paying the per-dispatch shim cost. The cost is medium-effort: TS implementations of each ggml op the project actually uses. That's a finite, well-bounded surface (matmul + a handful of normalizations + rotary + attention + MLP variants) — not the open-ended "track every upstream shader change" burden that motivated Tier 3 in the first place.

The legacy `ModelInference` is ~2890 LOC of TS that already does this for its hand-rolled graph. JSEP-style would reuse most of those shader recordings but drive them from a WASM scheduler instead of a TS one. **The legacy code becomes the kernel reference, not a dead artifact.**

### Recommendation

**Partial revert + plan P2-v2 around Path 3.**

- Revert Tasks 5-11 (wrapper + dispatch flip + delete + smoke-page port) to restore working causal-LM throughput.
- **Keep** Tasks 1-4 + 5 (bridge expansion: model metadata, KV cache mutation, state-seq, embeddings readback, TS bridge interface). These are additive C++ exports + TS wiring that any JSEP-style path will need to expose WASM-side state to JS.
- **Keep** the `webllm_perf_counter` diagnostic — useful for verifying graph-reuse + dispatch counts under a JSEP-style backend.
- **Keep** the JSPI `WaitAny` patch in llama.cpp (correctness, useful regardless).
- File a fresh P2-v2 plan in TODO.md for the next session targeting JSEP-style architecture. Phase 1 of that work is a research probe: read ORT-Web's `jsep/` source to understand the kernel-dispatch ABI, then prototype a single op (matmul) end-to-end through the new path and measure tok/s.

### Decision tree (refined)

- **Path 3 (recommended):** Plan P2-v2 around JSEP-style architecture. Multi-week design + prototype phase before the implementation phases. Real existence proof reduces architectural risk.
- **Path C (revert + accept indefinite Tier 3 deferral):** Lower upfront cost, higher long-term maintenance burden as upstream `ggml-webgpu` evolves. Reasonable fallback if Path 3's prototype phase reveals unexpected blockers.
- **Path B (ship slow):** Still not recommended.
- **Path A.4 (revert JSPI; restore ASYNCIFY):** Now clearly dominated. Same shim, same per-call cost. Drop.
