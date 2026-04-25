# WebLLM Project Status & Roadmap

> **Date:** 2026-04-25
> **Status:** End-to-end browser inference is working for both causal LMs and
> BERT-style encoders (Arctic-Embed). `make bench-full` now drives the
> generative profiles plus the embedding profiles end-to-end into the live
> SQLite-backed dashboard. The dashboard has a dedicated Embeddings section
> (cosine, latency, throughput) and the per-dimension / temperature-sweep /
> Accuracy×Speed charts split Qwen thinking-on vs thinking-off into
> distinct series. The eval suite separates chat-style semantic reasoning
> from true embedding-vector tasks. **The library is now the single source
> of truth for decode**: `engine.chatCompletion` drives [7/8], the
> interactive chat box, and bench mode through the same `Generator.generate`
> that public consumers hit. The parallel decode loop in
> `createSmokeCompletionRunner` is gone. Decode-mode selection is now
> per-step (greedy / topk / full) so steps without active steering state
> stay on the topk fast path even when the config configures steering.
> Current smoke-bench baselines (browser, TinyLlama Q4_0 + Qwen3 0.6B Q4f16,
> realistic sampling, 3-trial median on 2026-04-25 after the qwen3 steering
> topk path landed):
> - tinyllama-1.1b-chat-q4_0: **~105 tok/s decode**
> - qwen3-0.6b-q4f16, thinking-off: **~85 tok/s decode**
> - qwen3-0.6b-q4f16, thinking-on: **~93 tok/s decode** (was ~17 — see
>   `3e5be59`: CPU post-filter top-K replaced the full-vocab readback +
>   JS sampling pipeline that was costing ~76 tok/s on Qwen3's 152K vocab)
>
> **Plan files:** `docs/superpowers/plans/2026-04-20-webllm-implementation.md` (Phase 1)

---

## Project Milestones

### Completed

- [x] GGUF parser for model metadata + tokenizer
- [x] SPM + BPE tokenizer (encode/decode, ▁ normalization, byte fallback)
- [x] Emscripten WASM build for ggml-webgpu backend
- [x] Full transformer forward pass (embedding, attention, FFN, RMSNorm)
- [x] Multi-template chat formatting (llama2, chatml, gemma, phi3, llama3, mistral-v7, zephyr)
- [x] Auto-prepend default system message for models without one
- [x] Multi-turn chat in browser (KV cache reset + full prompt reformat)
- [x] Sampling (temperature, top-k, top-p, repetition penalty)
- [x] KV cache for incremental decoding
- [x] Decode profiling pipeline (`eval/perf.ts`, `make bench-inference`)
- [x] Performance optimizations (items 2, 3, 5, 7, 8, 9, 11 below)
- [x] GPU-side ARGMAX/TOP_K logits reduction (item 11)
- [x] `make smoke-bench` end-to-end benchmark target with agentchrome
- [x] Semantic-reasoning eval dimension split from true embedding-vector tasks
- [x] Live benchmark dashboard migrated to Chart.js with richer comparison charts
- [x] Model support roadmap documented in `docs/MODEL_SUPPORT.md`
- [x] Public streaming APIs (`generateStream`, `chatCompletion`) wired through `Generator` + `InferenceSession`
- [x] Encoder forward pass for BERT-style embedding models (`WebLLM.embed()`, Arctic-Embed-s/m)
- [x] `make bench-full` exercises arctic-embed profiles end-to-end alongside generative profiles
- [x] Dashboard "Embeddings" section (cosine per task, median latency, throughput)
- [x] Charts that key on `modelId` now split Qwen thinking-on vs thinking-off into distinct series

---

## Cumulative Bug Fix History

1. Embedding lookup used `opCpy` Q4_0→F32 (unsupported; replaced with `ggml_get_rows`).
2. Leaf input data (`posTensor`, `tokenIdsTensor`, mask) must be written with `backendTensorSet` *after* `backendAllocCtxTensors`.
3. SPM tokenizer: ▁ normalization (encode + decode), code-point iteration, byte-fallback via `<0xHH>` text.
4. KV writes were orphaned by `graph_build_forward_expand` (unreachable from logits) — now explicitly expanded per layer.
5. KV writes ordered BEFORE attention reads in the graph node list.
6. RMSNorm gamma was never multiplied in — now applied at all three norm sites (attn, ffn, final).
7. Custom `GGML_OP_DIAG_MASK_INF` shader broken past head 0; later replaced by `ggml_soft_max_ext` with explicit causal mask tensor.
8. **V cache permute used wrong `ggml_permute` arguments** — silent shape mismatch in subsequent cpy scrambled V values. Fixed `(2, 0, 1, 3)` → `(1, 2, 0, 3)`.
9. WASM build -O1 → -O3 (3.4MB → 1.77MB).
10. Sampling wired in via `Sampler` class (temp / top-k / top-p / repetition penalty).
11. `ggml_soft_max_ext` + `op_get_rows` WASM bindings added.
12. Multi-turn chat garbled output — TinyLlama without a system message interprets Zephyr markers as comparison operators. Fixed by auto-prepending DEFAULT_SYSTEM in `formatChatPrompt`.
13. GPU TOP_K decode path reshaped logits as `[vocab, 1]` before `ggml_get_rows`; ggml gathers along row dimension, so the graph produced `[vocab, topK]` and later failed reshape assertions. Fixed by reshaping logits to `[1, vocab]` before `opGetRows`.
14. Dashboard Temperature sweep hot series produced data but could render invisibly because `CHART_COLORS.red` was missing. Fixed by extracting shared temperature-sweep data construction and defining the hot color as `#f85149`.
15. **Encoder V permute tripped `ggml_mul_mat`'s `is_transposed` assertion** — `permute(v3, 1,2,0,3)` produced the right logical shape `[N, headDim, nHeads]` but left `nb[0] > nb[1]`. Wrapped in `opCont` to match llama.cpp's no-KV-cache BERT path.
16. **BERT WordPiece vocab follows llama.cpp's phantom-space convention** — `convert_hf_to_gguf.py::BertModel.set_vocab` rewrites the HF vocab so word-initial tokens gain a `▁` (U+2581) prefix and `##xyz` continuations have the `##` stripped. Our tokenizer was looking up `happy` / `##ful` (HF style) and missing every entry, producing `[CLS][UNK][SEP]` for every input. Rewrote `wpSubword` and `decodeWordPiece` to match `llm_tokenizer_wpm_session`. HF golden fixture in `tests/wordpiece-golden.test.ts` now guards it.
17. **`ggml-webgpu` silently no-op'd `GGML_OP_NORM`** — only `RMS_NORM` and `L2_NORM` were in `supports_op`, so LayerNorm fell through and the result buffer kept its zero-init contents. Combined with `GGML_CPU=OFF` (no scheduler fallback) every encoder forward produced bit-identical output regardless of input. Added a `LAYER_NORM` variant to `row_norm.wgsl` (Σx + Σx² in one pass), registered the pipeline, and dispatched `GGML_OP_NORM` through `ggml_webgpu_row_norm`. See `docs/LLAMA_CPP_PATCHES.md` patch #9.
18. **Bench-full smoke page hard-coded the causal-LM path at step [4/8]** — `ModelInference.loadWeights` failed on BERT GGUFs with `Weight "output_norm.weight" not found`. Page now branches on `arch === "bert"` and uses `EncoderInference`, skipping KV cache / generation / reference-encoder steps with explanatory pass logs.
19. **HF downloader picked Q4_K_M for arctic-embed** because the MLC-style `q0f32` defaultQuant didn't match any file in the GGUF repo and `q4_k_m` was first in the fallback list. Added a `ggufFilePattern?: string` field to `BenchmarkModel`; arctic-embed pins `"f16"` and the picker checks it ahead of `defaultQuant`.
20. **Tokenizer.encode("") returned `[]` for WORDPIECE** — bypassed the `[CLS] ... [SEP]` framing via an unconditional empty-string short-circuit in `encode()`. WORDPIECE now always frames; other tokenizer types keep returning `[]`.
21. **Score-over-time chart was blank despite a populated DB** — `renderSeriesChart` was defined but never invoked from the `render()` loop, so the panel always showed the bar-empty placeholder. Adding the call to the render loop (between `renderFinishChart` and `renderTable`) fixed it. Also fixed: `seriesLoaded` was sticky after the first fetch, so SSE-delivered evals were invisible to the chart; now reset on every `eval_complete` event. The category x-axis was missing its `labels` array, so even when called the points had nowhere to plot — now built from the sorted union of timestamps.
22. **Dashboard charts keyed on `modelId` collapsed Qwen thinking-on/off** — Temperature sweep, per-dimension grouped, and Accuracy×Speed scatter all shared a key for both Qwen modes; latest-wins silently overwrote one with the other. Group keys now include `thinking`; series labels gain a `" (think)"` suffix when thinking is on so non-thinking-capable models keep their existing labels.
23. **`engine.generateStream` qwen3-chatml wiring diverged from the smoke loop in 4 places** discovered while consolidating onto the library. Effects: (a) `maskedTokensWhileThinking` and `maskedTokensAfterThinkingUntilAnswer` were missing `<|endoftext|>`, so the model could emit it mid-think and either get a stray stop or pollute the chain-of-thought; (b) `maskedTokensAfterAnswerStarts` mistakenly included `<|im_end|>`, which is the chat EOS — the model could not terminate normally during the visible answer and qwen3 thinking-on always ran to `max-tokens`; (c) `<|endoftext|>` wasn't auto-added as a stop token; (d) the smoke loop's first-post-`</think>` leading-whitespace guard (forces `</think>` to be followed by a token starting with whitespace) had no library counterpart. Fixed all four; added `requireLeadingWhitespaceAfterThinking` to `GenerationConfig` for parity, and threaded the seed through `engine.generateStream`'s internal `Sampler` (added `CompletionConfig.seed`) so smoke runs are reproducible through the public API.
24. **`Generator.generate` computed `gpuMode` once, statically, before the decode loop** — `requiresFullLogitsSteering = (any qwen3 mask set configured)` forced `gpuMode = "full"` for the entire run. Once the smoke loop migrated onto the library, qwen3 thinking-off ran at ~17 tok/s on the full path instead of ~83 tok/s on the topk path, even on steps where no steering state was active. Replaced with per-step dynamic selection: `greedy` if sampler is greedy + no penalty; `topk` if `sampler.topK > 0` AND no current steering state (`thinkDepth === 0 && !waitingForVisibleAnswer && !hasVisibleAnswerText`); else `full`. The smoke loop's old code had this dynamic check inline; the library now matches.

---

## Debug Tools

`src/inference/model-inference.ts` has instrumented debug helpers. From the browser console on `smoke-test/real-model.html`:

```js
await window.inference.debugReadEmbeddingRow(1);  // BOS

window.inference.resetKVCache();
await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
await window.inference.debugReadKCache(0, 64*4, 0);
await window.inference.debugReadVCache(0, 64*4, 0);

await window.inference.debugReadNormWeight("attn0", 8);

await window.inference.debugLayerOutput(
  22172, 0, "layer_output"
  // or: "pre_attn" | "attn_normed" | "attn_q" | "attn_k" | "attn_v"
  //     | "attn_out" | "post_attn" | "ffn_normed" | "ffn_gate" | "ffn_up"
  //     | "ffn_hidden" | "ffn_out"
);
```

`smoke-test/real-model.html` stashes `window.{inference, tokenizer, parsedModel}` for console use.

---

# Inference Performance Optimizations

Baseline (pre-optimization): ~44 tok/s decode, ~130 ms prefill on TinyLlama 1.1B
Q4_0 via Emscripten WebGPU in-browser.

**Steady-state decode baselines (2026-04-25, non-profile, realistic
sampler):** TinyLlama Q4_0 ~107 tok/s · Qwen3 0.6B thinking-off ~83 tok/s
· Qwen3 0.6B thinking-on ~17 tok/s. These are the canonical numbers
to compare new perf work against.

**Profile-mode hotspot ranking (2026-04-25, `--profile`).** Captured
through `make smoke-bench PERF_RUNS=3` after fixing three latent
harness bugs (`953c560`) that had been masking trace collection since
the consolidation landed. `--profile` enables ggml-webgpu's detailed
backend timing and perturbs throughput measurably:

| Profile          | Steady-state | Profile-mode | Perturbation |
|------------------|--------------|--------------|--------------|
| TinyLlama Q4_0   | ~107 tok/s   | 76 tok/s     | -29%         |
| Qwen3-0.6 off    | ~83 tok/s    | 55 tok/s     | -34%         |
| Qwen3-0.6 on     | ~17 tok/s    | 14.6 tok/s   | -14%         |

Cite steady-state numbers for throughput claims; cite profile-mode
ratios for hotspot ranking. Mixing the two is what derailed the
"Apr-23 regression" investigation.

Per-step decode breakdown by mode (medians from the runs above):

| Bucket (ms/step)        | TinyLlama topk | Qwen3-off topk | Qwen3-on full |
|-------------------------|----------------|----------------|---------------|
| graphComputeMs          | 11.67 (88.8%)  | 13.91 (88.8%)  | 22.62 (94.5%) |
| downloadResultMs        |  1.20 ( 9.1%)  |  1.37 ( 8.8%)  |  0.93 ( 3.9%) |
| build + alloc + upload  |  0.26 ( 2.0%)  |  0.36 ( 2.3%)  |  0.35 ( 1.5%) |
| **totalMs**             | **13.14**      | **15.66**      | **23.93**     |

Backend attribution (% of `graphComputeMs`, profile mode only):

| Field                    | TinyLlama topk | Qwen3-off topk | Qwen3-on full |
|--------------------------|----------------|----------------|---------------|
| backendMatmulMs          | 3.85 (33.0%)   | 4.05 (29.1%)   | 6.31 (27.9%)  |
| backendEncodeOverheadMs  | 2.71 (23.2%)   | 4.07 (29.2%)   | 3.53 (15.6%)  |
| backendAttentionMs       | 0.40 ( 3.5%)   | 0.49 ( 3.5%)   | 0.75 ( 3.3%)  |
| backendDispatchCount     | 450/token      | 629/token      | 619/token     |

Headlines:
- **Decode is graph-compute-bound across all three profiles** (~89–95%
  of step time). Readback is a 4–9% slice; further async-readback work
  has tiny headroom.
- **Matmul leads on TinyLlama (33%) but encode/dispatch overhead leads
  on Qwen3-off (29.2%).** Qwen3 dispatches ~40% more ops per token (629
  vs 450) — graph-shape reduction has more leverage there than on TinyLlama.
- **Qwen3 thinking-on previously routed through `mode=full` while
  steering state was active**, costing ~8.7 ms/step extra graph
  compute and a 152K-float JS sampling pipeline. The full-path
  numbers above are now historical — Active Step §2 landed
  (commit `3e5be59`) and thinking-on now runs through topk + CPU
  mask filter at ~93 tok/s steady-state (5.4× lift over the 17 tok/s
  full path). The matmul jump (4.05 → 6.31 ms) between the old qwen3
  modes is unexplained but no longer load-bearing now that thinking-on
  doesn't hit the full path; flagged for any future graph-shape work.
- **Consolidation tightened TinyLlama dispatches and matmul share**
  vs the stale 2026-04-22 numbers (489 → 450 dispatches/token, matmul
  share 40.4% → 33.0%). Treat that as a quiet consolidation win, not
  a headline.

Items in rough order of expected impact. Each entry explains the idea, where
the code lives today, the expected win, and the risk/tradeoff.

---

## High impact

### 1. Decode graph reuse (deferred)
- **Where**: `src/inference/model-inference.ts::forward()`
- **Today**: every call to `forward()` calls `ctxCreate`, builds the full
  ~440-op graph, calls `backendAllocCtxTensors`, uploads leaf inputs, runs
  compute, then `ctxFree`s. For a decode step with `nTokens=1` the graph
  shape is identical across steps.
- **Change**: cache the graph for `nTokens=1` decode. Only update:
  - leaf inputs (`posTensor`, `tokenIdsTensor`, mask row for current position)
  - the K/V cache offsets (encoded into the graph nodes via `pastLen * kNb1`
    constants, so this needs either graph reuse with runtime offset or
    rebuilding just the KV views each step)
- **Expected**: 2–5× decode throughput. The JS-side graph construction +
  WASM asyncify round trip is currently a meaningful chunk of per-step cost.
- **Risk**: the ggml graph stores absolute offsets inside view tensors.
  Reusing the graph requires either:
  - Adding a C-side helper that mutates view tensor offsets in place, or
  - Refactoring KV cache layout so writes always go to a fixed slot and the
    "real" position is a permutation applied separately, or
  - Pre-building graphs for every possible past-length (memory hungry)
- **Profile measurement (2026-04-21):**
  - mean total per decode step: 16.75 ms
  - ctxCreate: 0.00 ms (0.0%)
  - buildGraph: 0.21 ms (1.3%)
  - backendAlloc: 0.05 ms (0.3%)
  - uploadLeaves: 0.02 ms (0.1%)
  - graphCompute: 6.92 ms (41.3%)
  - downloadLogits: 9.53 ms (56.9%)
  - teardown: 0.02 ms (0.1%)
  - Non-GPU overhead (ctxCreate + buildGraph + backendAlloc + teardown) = 1.7%.
- **Status**: deferred. Big structural change relative to its current
  expected win at 17.5 ms/step. Revisit if we can measure that graph
  building (not GPU compute) is actually the bottleneck.
- **Phase A skipped:** GPU compute + logits download dominate; moving graph
  build to C can at best claw back ~1.7% and isn't worth the C-side
  maintenance burden.

### 2. Re-enable batched compute passes in the WebGPU backend ✅ DONE
- **Where**: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
- **Fix**: flipped `batch_compute_passes` to `true`. Safe now that
  overlap-only conflict detection (item 3) doesn't schedule inter-dispatch
  CopyBufferToBuffer calls for most ops.
- **Actual gain**: marginal on top of item 3 (most of the 33% came from
  item 3). Still worth keeping for larger models where driver overhead
  of many compute passes would dominate.

### 3. Refine buffer-conflict detection (overlap-only) ✅ DONE
- **Where**: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
  `__EMSCRIPTEN__` block in `ggml_backend_webgpu_build_multi`.
- **Was**: created a temp GPU buffer for EVERY same-buffer-different-offset
  binding pair, even when the ranges didn't actually overlap. WebGPU's
  validation only rejects overlapping ranges, not disjoint ones.
- **Fix**: added an `overlap` check (`!(a_end <= b_start || b_end <= a_start)`)
  before creating a temp buffer. Most attention/FFN op bindings turned out
  to be disjoint slices of the shared graph buffer.
- **Actual gain**: **+28% decode throughput** (44 → 56 tok/s). The bulk of
  the original overhead was unnecessary `CopyBufferToBuffer` calls.

---

## Medium impact

### 4. Enable flash attention in the browser ❌ BLOCKED UPSTREAM
- **Where**: `ggml-webgpu.cpp::ggml_backend_webgpu_device_supports_op`
  under `GGML_OP_FLASH_ATTN_EXT` — currently `#ifndef __EMSCRIPTEN__`.
- **Blocker**: emdawnwebgpu does NOT expose
  `wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix`. Only plain
  `Subgroups` is available. llama.cpp's flash-attention shaders use
  subgroup-matrix operations specifically.
- **For now**: defer. Reconsider when Chrome ships stable subgroup-matrix
  support and emdawnwebgpu rolls.

### 5. Fused SwiGLU op ✅ DONE
- **Where**: `src/inference/model-inference.ts` FFN section.
- **Actual gain**: +1–2% (58 → ~58.5 tok/s). Modest — FFN compute is
  dominated by the three mul_mats, not by silu/mul. Kept for cleanliness.

### 6. F16 KV cache ❌ NET LOSS AT SHORT CONTEXT
- **Tried**: switched K and V to `F16`. WebGPU backend handled F16×F32
  mul_mat correctly — no correctness regression.
- **Measured**: **-7.7%** decode throughput (55.3 tok/s vs 59.9 baseline).
  F16×F32 mul_mat not as fast as F32×F32 for small matrices, and F32→F16
  conversion on every KV write adds dispatch overhead at short contexts.
- **Reverted**: code stays F32. Reconsider for long-context workloads
  (1000+ tokens) where bandwidth savings on attention reads will overtake
  write-path overhead.

---

## Low impact / polish

### 7. Skip redundant `opCont` calls ✅ DONE
- **Where**: K/V cache writes in `forward()`.
- **Actual gain**: +0–2% on top of item 5. Still worth keeping: fewer
  dispatches, cleaner code.

### 8. Skip mask tensor for nTokens=1 decode ✅ DONE (partial)
- **Where**: `forward()` mask allocation + upload + softmax_ext call.
- **Actual gain**: flat. Still a cleanup. Pre-allocating mask at
  `initKVCache` time deferred.

### 9. Reduce JS↔WASM boundary crossings ✅ DONE
- **Where**: `src/wasm/webgpu-bridge.cpp` + `src/inference/ggml-wasm.ts`
  + `forward()` in `model-inference.ts`.
- **Change**: added `backend_tensor_set3` in C bridge — single bundled WASM
  call for pos + ids + mask instead of 2–3 separate hops.
- **Actual gain**: **+5–7% decode throughput** (55.6 → 58.7 tok/s median).

### 10. Benchmark the current pipeline ✅ DONE
- **Where**: `eval/perf.ts` + `make smoke-bench` + `make bench-inference-save`.
- **Current Task 5 profiled investigation baseline**: 93.5 tok/s on the
  profiled TinyLlama-1.1B Q4_0 browser run (`PERF_RUNS=3`), with median-run
  wall time 2027 ms, `graphComputeMs` mean 9.96 ms, and `downloadResultMs`
  mean 0.62 ms. Read this as a profiling baseline for hotspot ranking, not as
  the new steady-state browser throughput baseline.
- `make smoke-bench` — end-to-end: builds WASM+JS, starts server, launches
  agentchrome (headed), runs 3 perf iterations with `--profile`, cleans up.
  All smoke targets (`smoke-serve`, `smoke-open`, `smoke-run`, `smoke-bench`)
  depend on `smoke-test` for fresh builds.

### 11. GPU-side ARGMAX/TOP_K logits reduction ✅ DONE (negligible gain)
- **Where**: `src/wasm/webgpu-bridge.cpp` (C bridge), `src/inference/ggml-wasm.ts`
  (TS bindings), `src/inference/model-inference.ts::forwardDecode()`,
  `src/inference/generation.ts` (decode loop routing),
  `src/inference/sampler.ts::sampleFromTopK()`,
  `src/core/engine.ts` (wiring), `smoke-test/real-model.html` (both code paths).
- **What**: Added `ggml_argmax` and `ggml_top_k` to the WASM bridge. New
  `forwardDecode()` method builds the same transformer graph but appends
  ARGMAX/TOP_K tail ops, downloading 4 bytes (greedy) or k×8 bytes (topk)
  instead of 128KB (32K×float32) full logits. Generation loop auto-selects
  mode: greedy (temp=0, no penalty), topk (topK>0), or full (fallback).
  Smoke test step 7 and chat handler both use the greedy path.
- **Actual gain**: **+0.5%** (58.7 → 59.0 tok/s). Negligible.
- **Why**: The readback bottleneck is synchronization latency, not data size.
  At the time of measurement, `downloadFromTensor()` still paid queue/map wait
  latency that dominated the ~9.5ms readback regardless of whether the payload
  was 4 bytes or 128KB. Reducing data size only saved the final memcpy
  (~0.01ms for 128KB).
- **Follow-up completed**: the browser stack now has a real request-based async
  readback path:
  - `~/Repos/llama.cpp/ggml/include/ggml-webgpu.h`
  - `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
  - `src/wasm/webgpu-bridge.cpp`
  - `src/inference/ggml-wasm.ts::downloadFromTensor()`
  This adds backend `begin / poll / finish / cancel` support, wires it through
  the WASM bridge, and uses heap allocation safely across async boundaries.
- **Current status**: correctness/integration is fixed, browser smoke passes,
  and the latest profiled run shows the completion-driven path has reduced
  `downloadResultMs` to a 0.64 ms mean. Readback is now a smaller slice of
  decode latency than the rest of the step.
- **Infrastructure value**: The ARGMAX/TOP_K bridge functions, `forwardDecode()`,
  generation-loop routing, and the new async readback request API are all useful
  foundations for the next round of latency-hiding work. The
  `sampleFromTopK()` sampler method enables CPU sampling on GPU-reduced
  candidate sets for temperature > 0.

---

## Dashboard & Visualization

Live bench dashboard at `smoke-test/dashboard.*` (served by
`eval/live-server.ts` on port 8033). Each section below is independently
shippable on top of the existing eval/run data — no new bench metadata
needs to be collected.

### 12. Convert existing charts to a proper charting library ✅ DONE
- **Where**: `smoke-test/dashboard.js`, `smoke-test/dashboard.html`,
  `smoke-test/dashboard.css`, `smoke-test/vendor/chart.umd.min.js`.
- **Done**: introduced self-hosted Chart.js 4 UMD and converted the main
  dashboard charts to managed Chart.js instances with dark-theme colors,
  legends, tooltips, and dynamic chart-host sizing.
- **Follow-up**: use `make vendor-refresh` after bumping `chart.js` to refresh
  the vendored browser bundle.

### 13. Accuracy × Speed scatter chart ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderScatterChart`,
  `smoke-test/dashboard.html`.
- **What**: one dot per profile. X = mean tok/s, Y = eval `overall`.
- **Answers**: "which profile should we actually ship?"

### 14. Per-dimension grouped bars per model ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderDimGroupedChart`,
  `smoke-test/dashboard.html`.
- **What**: one row per model (cold profile only); grouped bars for
  `tool-calling / reasoning / instruction-following / semantic-reasoning`.
- **Answers**: "which model do I pick for workload X?"

### 15. Temperature sweep per dimension ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderTempSweepChart`,
  `smoke-test/dashboard-charts.js`, `tests/dashboard-charts.test.ts`.
- **What**: per (model, dimension), cold / warm / hot grouped bars.
- **Answers**: "is temperature hurting me on dimension X?"
- **Regression covered**: the hot bucket now has explicit data + color coverage
  so it cannot disappear silently.

### 16. Thinking on vs off delta (Qwen) ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderThinkingDeltaChart`,
  `smoke-test/dashboard.html`.
- **What**: two-bar pairs per dimension comparing Qwen thinking off/on at
  matched temperature.
- **Answers**: "is thinking worth the extra decode time, and on which
  dimensions?"

### 17. Time-to-first-token (prefill latency) chart ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderTtftChart`,
  `smoke-test/dashboard.html`.
- **What**: horizontal bar chart of `oneShot.prefillMs` per profile.
- **Answers**: "how long until the first token for each profile?"

### 18. Finish reason breakdown ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderFinishChart`,
  `smoke-test/dashboard.html`.
- **What**: stacked horizontal bars showing `eos / max-tokens / stop-token /
  error / unknown` counts by profile.
- **Answers**: "is this profile producing clean completions, or is it
  running off the end?"

### 19. Score over time (regression detection) ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderSeriesChart`,
  `eval/live-db.ts::loadEvalSeries`, `eval/live-server.ts` `/evals/series`.
- **What**: line chart of `overall` across successive eval runs by profile.
- **Answers**: "did a code change regress anything?"

### 20. Quantization comparison (future — requires multi-quant models)
- **Where**: new panel.
- **What**: same model at q4 vs q8 vs f16 — accuracy delta vs speed delta.
- **Blocker**: today every entry in `eval/models.ts` has exactly one
  quant. Needs multi-quant registrations to be meaningful. Deferred.

### 21. Dedicated Embeddings dashboard section ✅ DONE
- **Where**: `smoke-test/dashboard.html` (new section divider + three
  panels), `smoke-test/dashboard.js`
  (`renderEmbeddingCosineChart` / `renderEmbeddingLatencyChart` /
  `renderEmbeddingThroughputChart`),
  `smoke-test/dashboard-charts.js`
  (`buildEmbeddingCosineChartData` / `buildEmbeddingLatencyChartData` /
  `buildEmbeddingThroughputChartData`),
  `tests/dashboard-charts.test.ts`.
- **What**: separate "Embeddings" section at the bottom of the dashboard
  with three panels — per-task cosine similarity, median ms-per-text
  latency, and texts/sec throughput. Per-dimension grouped chart now
  excludes embedding-only evals and drops the `embedding` column so
  generative-model rows aren't cluttered.
- **Answers**: "how fast and how good is each embedding model?"

### 22. Accuracy × Speed model colour key ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderScatterChart`.
- **What**: scatter dots are grouped by `(modelId, thinking)` into one
  dataset per model+mode, each with its own colour from an 8-stop
  palette. The chart's top legend acts as the colour key. Stable colour
  assignment via sorted keys.
- **Answers**: "which dot is which model?" without needing to hover.

---

## Won't-do (for now)

- **Smaller quants (Q2_K / Q3_K)**: quality/speed tradeoff, not a pipeline improvement.
- **Speculative decoding**: requires a drafter model; large project.
- **Custom kernel fusion beyond GLU**: huge effort, marginal win.

---

## Next Steps

### Decode hotspot decision / rebaseline (2026-04-22 Task 5)
- **Current hotspot:** matmul path tuning remains the lead target.
- **Profile-mode hotspot evidence:** `make smoke-bench PERF_RUNS=3
  PERF_MODEL=tinyllama-1.1b-chat-q4_0` reported median **93.5 tok/s**, median
  run **184 ms prefill**, **331 ms decode**, and **2027 ms wall time** for the
  full page completion. Across 90 single-token greedy decode traces,
  `graphComputeMs` averaged **9.96 ms / 91.8%** of step time while
  `downloadResultMs` averaged **0.62 ms / 5.7%**. Because this came from
  `--profile` mode, use it for hotspot ranking and direction, not as a new
  steady-state throughput claim.
- **Backend attribution:** `backendMatmulMs` averaged **4.02 ms / 40.4% of
  graph time**, ahead of `backendEncodeOverheadMs` (**2.81 ms / 28.2%**) and
  `backendAttentionMs` (**0.40 ms / 4.0%**). `backendDispatchCount` stayed at
  **489** per token; that supports the encode/dispatch-overhead suspicion but
  is not itself a timed bottleneck metric.
- **Decision:** keep structural follow-up deferred. If perf work resumes, the
  current profiled traces suggest keeping the next optimization pass narrow and
  targeting matmul first, with encode overhead as the secondary decode-compute
  suspect.
- **Update (2026-04-22):** Matmul follow-up attempt (increase legacy Q outputs per wg) showed no meaningful retained gain and was reverted.
- **Update (2026-04-25):** baseline moved to ~110 tok/s after the GPU TOP_K
  smoke-loop wiring (commit `9156deb`).
- **Update (2026-04-25, library-as-source-of-truth consolidation):** the
  smoke loop has been deleted and replaced with a thin
  `engine.chatCompletion` adapter; `Generator.generate` now picks the
  decode mode dynamically per step. Numbers in this section pre-date
  both 2026-04-25 changes — **all bucket percentages here are stale**.
  Re-profile first; see "Active next steps §1" below.

### Completed on 2026-04-24

- Fixed GPU TOP_K row gathering by reshaping logits to `[1, vocab]` before
  `opGetRows`.
- Added `WEBLLM_ASSERTIONS` / `wasm-build-debug` for preserving WASM abort
  diagnostics when needed.
- Added the `semantic-reasoning` eval dimension and moved chat-style semantic
  tasks out of the true embedding-vector track.
- Added cosine-similarity scoring helpers and regression tests for embedding
  vector scoring.
- Captured eval sampler/context params so the dashboard can bucket temperature
  and show run details.
- Migrated live dashboard charts to self-hosted Chart.js and implemented
  accuracy × speed, per-dimension grouped bars, temperature sweep, Qwen
  thinking deltas, TTFT, finish reasons, and score-over-time.
- Fixed the Temperature sweep hot bucket rendering regression with shared
  chart-data tests.
- Documented model support and follow-up roadmap in `docs/MODEL_SUPPORT.md`.
- Wired the public streaming APIs (`WebLLM.generateStream()` and
  `WebLLM.chatCompletion()`) through `Generator` + `InferenceSession`, with
  stop-token handling, abort metadata, Qwen thinking controls, and tests in
  `tests/engine-streaming-api.test.ts`.
- 2026-04-24: Encoder forward pass shipped. `WebLLM.embed(modelId, text)`
  drives a BERT-style bidirectional graph (input embed + post-norm
  attention + GeLU FFN, mean/CLS pooling, L2-normalized output) on the
  Arctic-Embed-s GGUF; smoke step `[8/8]` passes with `cosine('happy',
  'joyful') ≈ 0.77`. Bring-up uncovered three load-bearing fixes:
  V-permute → `opCont` to satisfy `ggml_mul_mat`'s `is_transposed`
  assertion; tokenizer rewritten for llama.cpp's phantom-space BERT
  vocab convention (HF golden fixture in `tests/wordpiece-golden.test.ts`
  guards it); and `GGML_OP_NORM` added to the patched ggml-webgpu
  backend (commit `68f1738d5`, see `docs/LLAMA_CPP_PATCHES.md` patch #9).

### Completed on 2026-04-25

- Wired arctic-embed-s/m profiles into `make bench-full`. New
  `embedding?: boolean` flag on `SmokeProfile`; `bench.ts` skips the
  `chat-smoke` (speed) phase for embedding profiles; `browser-eval.ts`
  auto-restricts embedding-only models to the embedding dimension.
  `eval/models.ts` gains a `ggufFilePattern?: string` field so the HF
  downloader pins the verified F16 GGUF instead of falling through to
  Q4_K_M. `smoke-test/real-model-page.js` branches on `arch === "bert"`
  and routes through `EncoderInference`; `engine.adoptPreloadedModel`
  now accepts either inference type. End-to-end:
  `8/8 tasks passing · overall 93%` for arctic-embed-s.
- HF golden WordPiece fixture: `scripts/extract-bert-vocab.ts` dumps
  the real Arctic-Embed-s vocab (30522 tokens) to JSON;
  `scripts/generate-bert-golden.py` generates HF reference encodings
  for 16 probe strings (single words, multi-word, subword splits,
  casing, accents, punctuation carve-outs, empty string);
  `tests/wordpiece-golden.test.ts` asserts byte-for-byte parity.
- Dashboard restructure:
  - New "Embeddings" section with cosine, median latency, and
    throughput panels.
  - Per-dimension grouped chart now excludes embedding-only evals and
    drops the `embedding` column; renders `null` (not `0`) for
    dimensions a model wasn't scored on.
  - Temperature sweep, per-dimension grouped, and Accuracy×Speed scatter
    now key on `(modelId, thinking)` so Qwen thinking-on and
    thinking-off render as distinct rows / colours.
  - Accuracy × Speed scatter coloured by model with the chart's own top
    legend acting as the colour key.
  - Score over time: wired into the render loop (was defined but never
    called), category x-axis given a labels array built from the sorted
    union of timestamps, `seriesLoaded` cache invalidated on every
    `eval_complete` event.
- TODO.md records an Apr-23 ~50% smoke-bench tok/s drop; bisect attributed
  it to the encoder-innocent commit `5542bef`. See Next Steps §1 for the
  2026-04-25 root-cause finding (sampler-config change in the smoke page,
  not an engine regression).
- Re-baselined item 11 with the `?slowpath=1` URL gate (temporary, not
  committed): full 32 K-vocab readback costs only ~0.1 ms/token over the
  4-byte ARGMAX readback even post-async-readback. The "negligible gain"
  framing for `forwardDecode("greedy")` is correct. The entire ~10 ms/token
  gap between greedy and realistic sampling lives in the JS sampling
  pipeline.
- `perf(smoke): route realistic-sampler decode through GPU TOP_K path`
  (commit `9156deb`). Added a topk middle branch in
  `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` that
  calls `inference.forwardDecode(..., "topk", sampler.topK)` and feeds
  the reduced indices/values into `sampler.sampleFromTopK(...)`. Gated
  to skip when qwen masking/thinking state is active. **Measured impact
  (TinyLlama Q4_0, 3 trials median): 52.9 → 110.7 tok/s (2.1×)**;
  recovers 96% of the way to the greedy upper bound (114.8 tok/s).
  Qwen3 thinking-off also benefits (~76 tok/s); thinking-on routes
  through the unchanged full path (~16.6 tok/s).
- **TODO §2 done — library is now the single source of truth for
  decode** (committed as `6865a2c`). The smoke decode loop
  in `createSmokeCompletionRunner` was a 200-line duplicate of
  `Generator.generate` that silently dropped throughput when the topk
  fast path landed on one side but not the other. Consolidation steps:
  - **Library fixes (Phase 1):** `engine.generateStream` qwen3 wiring
    parity (bug-fix #23 above); `requireLeadingWhitespaceAfterThinking`
    added to `GenerationConfig` and wired through
    `Generator.generate`'s post-`</think>` sampling guard.
  - **Library extension (Phase 2A):** `CompletionConfig.seed` added to
    `src/core/chat-types.ts`; `engine.generateStream` threads it into
    the internal `Sampler({ ... seed })` construction.
  - **Library refactor (Phase 3):** dynamic per-step decode-mode
    selection in `Generator.generate` (bug-fix #24 above).
  - **Smoke-side rewrite (Phase 2B):**
    `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` now
    a ~50-line adapter over `engine.chatCompletion`. Deleted 11 dead
    helpers (`getForbiddenReentryTokens`, `getThinkingTokenIds`, the 3
    qwen `getMaskedTokens*`, `getExtraStopTokenIds`, `maskTokenLogits`,
    `isVisibleTextToken`, `isWhitespaceOnlyTextToken`,
    `tokenStartsWithWhitespace`, `decodeForDebug`).
    `smoke-test/real-model-page.js` constructs the WebLLM engine +
    `adoptPreloadedModel` once after [6/8] (covers both causal-LM and
    encoder paths); reused by [7/8], the interactive chat box, and
    bench mode. `smoke-test/real-model-bench.js` accepts engine +
    handleId from caller (no longer creates its own GPU adapter +
    engine). `smoke-test/real-model-runtime.js` drops manual prompt
    tokenization + KV reset; passes the full message array through.
    `tests/real-model-runtime.test.ts` rewritten for the new signature.
  - **Browser smoke verification (this session, single-run on tab
    `52C698CC3FF17A7A9B85EC5CB5EC67E2`, port 50840):** tinyllama
    106.9 tok/s · 64 tokens · finish=max-tokens; qwen3 thinking-off
    83.4 tok/s · 25 tokens · finish=eos; qwen3 thinking-on 17.3 tok/s
    · 236 tokens · finish=eos; embed cosine=0.76 on all three. Output
    text byte-identical across two tinyllama re-runs (seed=12345).
    Console: no errors. `make checkall`: 390 pass / 5 skip / 0 fail
    across 43 files. Committed as `6865a2c`.
- **Profile harness re-baselined** (commit `953c560`). Three latent
  bugs in `eval/perf.ts` had been silently masking every `--profile`
  run since the consolidation: greedy-only trace filter dropped the
  realistic-sampler topk traces; `buildSmokeTestUrl` API drift swallowed
  `?perf=` and `?profile=1`; and `fetchDecodeTraces` couldn't parse
  agentchrome's `output_file` overflow envelope (>16 KB results).
  Fixed all three, added `--thinking` to perf.ts, then captured the
  fresh medians that now drive the "Inference Performance
  Optimizations" preamble.

### Resumption checklist (start a fresh session here)

1. `make checkall` — confirm 390 pass / 5 skip / 0 fail.
2. Read the "Inference Performance Optimizations" preamble for the
   2026-04-25 profile-mode hotspot ranking. The qwen3 thinking-on
   numbers there are historical — that path now runs at ~93 tok/s
   via `3e5be59`.
3. Pick the next entry point from "Active next steps" §3 onward.
   The largest single remaining lever is gone (§2 done). Likely
   next candidates: re-profile to see what bucket is now dominant
   on Qwen3 thinking-on (suspect: matmul + encode overhead, the
   same as thinking-off), or move to §4 (test on a larger model).

### Historical context (for archive — do not action again)

1. **RESOLVED (2026-04-25): Apr-23 smoke-bench "regression" is a benchmark
   methodology change, not an engine regression.** Bisect (TS bundle only;
   WASM and llama.cpp HEAD constant via `make smoke-test` rebuilds):

   | Commit | Sampler used by smoke page | Decode path | tok/s |
   |---|---|---|---|
   | `0548cd4` (last fast point) | `Sampler({ temperature: 0 })` | `forwardDecode` (4-byte readback) | **118.9** |
   | `d111560` (profiling commit) | greedy | `forwardDecode` | **118.5** |
   | `d131cf0` (KV cap commit) | greedy | `forwardDecode` | **119.6** |
   | `5542bef` (qwen stability) | `makeSmokeSampler` (temp 0.7, topK 40, topP 0.95, repPenalty 1.05) | `forward` (full 32K logits) + JS topK/topP | **56.6** (page-reported) |
   | `main` (HEAD a0d5b9a) | same as 5542bef | `forward` + JS topK/topP | **59.1** |
   | `main` + `?temp=0&rep=1` URL override | greedy override | `forwardDecode` | **116.8** |

   **Root cause**: the new smoke decode loop in `5542bef` (and inherited by
   the page-shell extractions through HEAD) only takes the GPU-reduced
   ARGMAX fast path when `sampler.isGreedy && sampler.noPenalty`
   (`smoke-test/real-model-smoke.js` → `getSmokeSamplingConfig`,
   `smoke-test/real-model-page.js` decode loop). The new realistic sampler
   has `temperature 0.7` and `repetitionPenalty 1.05`, so every step falls
   through to `inference.forward()` — full 32K-vocab logits download plus
   JS-side temperature/topK/topP/penalty work — instead of the 4-byte
   greedy readback the old greedy page used.

   **Engine evidence**: the same `main` build serving the same WASM hits
   116.8 tok/s when the URL forces `temp=0&rep=1`. That matches the 118.9
   tok/s baseline at `0548cd4` to within noise. There is no decode-path
   regression in the inference engine, ggml-webgpu, or async readback.

   **Decision (2026-04-25)**: realistic sampling is the new canonical
   smoke-bench baseline. Do not re-baseline against the historical
   `~115` / `~93.5` greedy-path numbers — they measured a different
   workload and are retired as comparison targets. `forwardDecode` (greedy)
   stays available as the upper-bound diagnostic via `?temp=0&rep=1`, but
   smoke-bench tracks the realistic-sampling number going forward.

   **Re-baseline of item 11 (2026-04-25)**, controlled comparison on HEAD
   with TinyLlama-1.1B Q4_0 (3 trials each, medians shown):

   | Scenario | Decode path | tok/s | ms/token |
   |---|---|---|---|
   | A — greedy + `forwardDecode` (4 B readback) | fast | **114.8** | 8.7 |
   | B — greedy + `forward` (128 KB readback, argmax JS) | mid | **115.9** | 8.7 |
   | C — sampled + `forward` + JS topK/topP/penalty | slow | **52.9** | 19.0 |

   A vs B: ~0.1 ms/token. **Item 11's "negligible" framing still holds**
   even post-async-readback — full 32 K-vocab readback costs essentially
   nothing extra over the 4-byte ARGMAX readback. My earlier hedge was
   wrong; do not block on re-baselining item 11 again.

   B vs C: ~10 ms/token. **The entire 2× slowdown is the JS sampling
   pipeline** (`Sampler.sample()` over 32 K floats: temperature scale +
   topK selection + topP normalization, plus `applyRepetitionPenalty`
   over the recent 64-token window). If decode optimization resumes,
   that is the lever — vectorize / partial-sort / GPU-side topK before
   chasing readback or graph-reuse. The temporary `?slowpath=1` URL
   gate used to capture scenario B is not committed; re-add it from
   git history if you want to re-measure.

   **Fix landed (2026-04-25)**: the GPU TOP_K path already existed in
   `Generator.generate` (`src/inference/generation.ts` 3-way branch
   greedy/topk/full) but the smoke decode loop only branched greedy/full,
   so realistic-sampler steps fell through to full-vocab readback + JS
   topK on 32 K floats. Added a topk middle branch in
   `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` that
   calls `inference.forwardDecode(..., "topk", sampler.topK)` and feeds
   the reduced indices/values into `sampler.sampleFromTopK(...)` (which
   already applies repetition penalty + temperature + topP on the
   k-element set). Gated to skip when qwen masking/thinking state is
   active (`thinkDepth > 0`, `waitingForVisibleAnswer`,
   `hasVisibleAnswerText`) so the GPU's pre-mask top-K choice can't
   leak masked tokens. Measured impact (TinyLlama Q4_0, 3 trials):
   ~53 → ~111 tok/s (2.1×). Qwen3 thinking-off also benefits
   (~76 tok/s with coherent output); thinking-on routes through the
   full path unchanged.

2. **DONE (2026-04-25): library is now the single source of truth for
   decode.** See "Completed on 2026-04-25 (cont.)" above and bug-fix
   entries #23 and #24 for the full landed change set.

---

### Active next steps

1. **DONE (2026-04-25): re-profiled decode against the canonical
   pipeline.** Numbers landed in the "Inference Performance
   Optimizations" preamble above. Three latent harness bugs had been
   hiding the entire profile path since the consolidation; fixed in
   `953c560`. The fresh numbers ratify the same broad direction the
   stale 2026-04-22 profile pointed at — graph compute dominates,
   matmul + encode/dispatch overhead are the leads — but with the
   refinement that **encode overhead leads on Qwen3 (29.2% of graph)
   while matmul leads on TinyLlama (33%)**. The biggest single lever
   is still §2 below: getting qwen3 thinking-on off the full path.

2. **DONE (2026-04-25): qwen3 steering routes through topk + CPU mask
   filter** (commit `3e5be59`). Replaced the full-vocab readback /
   JS sampling pipeline with a `K + STEERING_TOPK_BUFFER` GPU TOP_K
   request followed by CPU-side mask filtering inside
   `Generator.generate`. Decision driven by the diagnostic capture
   on 2026-04-25: masked tokens land in top-K of full-vocab logits
   0.31% of the time, top-(K+10) 0.41% — the masks live deep in
   the tail, so CPU post-filter virtually never exhausts the pool
   and GPU-side WGSL masking would be over-engineering. Measured
   impact: **17.3 → 93.0 tok/s, 5.4×** — much larger than the
   ~24 tok/s prediction because the hot bucket was the JS sampler
   over Qwen3's 152K vocab, not the readback. `waitingForVisibleAnswer`
   stays on the full path because its whitespace-guard resampling
   needs full-vocab access (~2 of 236 captured steps). Output
   coherent on smoke step [8/8]; no regression on TinyLlama or
   Qwen3 thinking-off.

3. **Decode graph reuse** (item 1 in "Inference Performance
   Optimizations" preamble) remains deferred. The 2026-04-21 profile
   measured non-GPU overhead (ctxCreate + buildGraph + backendAlloc +
   teardown) at 1.7% of decode-step time — too small to chase. After
   the throughput uplift and the consolidation, that fraction may
   have grown; re-evaluate as part of §1's profile pass before
   committing to the C-side refactor.

4. **Test on a larger model** (Phi-2, Llama-2-7B). Bigger models
   shift the matmul/dispatch-overhead ratio and may change which
   lever is highest-impact. Worth running through `make bench-full`
   alongside §1's profile pass — same fresh decode-mode pipeline.

5. The latent 3+ binding buffer-conflict edge case in
   `ggml_backend_webgpu_build_multi` (item 3 in preamble) remains
   untested — no llama op hits it today.

6. **JSPI feasibility checkpoint** remains a follow-up investigation,
   not the next implementation step.
   - **Go/no-go:** no-go for the current milestone; the
     completion-driven readback path is the active baseline.
   - **What would have to change if revisited:** flip the WASM build
     from the current ASYNCIFY setup toward JSPI-related flags in
     `src/wasm/CMakeLists.txt`, replace
     `ggml-wasm.ts::callWithAsyncify()` with direct JSPI-compatible
     async export handling, re-audit Emscripten runtime exports to
     remove Asyncify-specific methods and keep only the JSPI-needed
     surface, assess whether the local `~/Repos/llama.cpp` branch's
     `ggml-webgpu: browser + ASYNCIFY support bundle` needs a
     parallel JSPI patch path, and verify browser support/behavior
     on the actual target matrix before any migration.

---

## Environment

```bash
source ~/emsdk/emsdk_env.sh
make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
cd smoke-test && python3 -m http.server 8031
# Cache-bust: ?v=$(date +%s)
```

## Local Dependencies

This repo depends on a local patched llama.cpp at `~/Repos/llama.cpp/` on branch
**`webllm-browser-patches`**. Patches (full inventory in `docs/LLAMA_CPP_PATCHES.md`):
1. `ggml: iterative ggml_visit_parents_graph for WASM stack safety`
2. `ggml-webgpu: browser + ASYNCIFY support bundle`
3. `ggml-webgpu: request-based browser readback API`
4. `ggml-webgpu: harden async readback request cleanup`
5. `ggml-webgpu: notify browser async readback completion`
6. `ggml-webgpu: add opt-in browser graph profiling`
7. `ggml-webgpu: specialize browser decode matmul dispatch` (paired with patch 8)
8. `Revert "ggml-webgpu: specialize browser decode matmul dispatch"` (effective no-op vs patch 6)
9. `ggml-webgpu: add GGML_OP_NORM (LayerNorm) support` — load-bearing for the BERT encoder path; without it `engine.embed()` returns bit-identical output for every input.
