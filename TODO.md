# Inference Performance Optimizations

Baseline (pre-optimization): ~44 tok/s decode, ~130 ms prefill on TinyLlama 1.1B
Q4_0 via Emscripten WebGPU in-browser.

**Current: ~58 tok/s decode (+33%), ~125 ms prefill** after items 2, 3, and 5.

Per-step decode is ~17.5 ms, which matches what `ggml_backend_webgpu_graph_compute`
can achieve in the browser for this graph size. Further decode speedup likely
requires either graph reuse (item 1) to cut JS/WASM rebuild overhead, or
flash attention (item 4) to cut the per-step GPU work at longer contexts.

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
- **Status**: deferred. Big structural change relative to its current
  expected win at 17.5 ms/step. Revisit if we can measure that graph
  building (not GPU compute) is actually the bottleneck.

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

### 4. Enable flash attention in the browser
- **Where**: `ggml-webgpu.cpp::ggml_backend_webgpu_device_supports_op`
  under `GGML_OP_FLASH_ATTN_EXT` — currently `#ifndef __EMSCRIPTEN__`.
- **Today**: flash attention is unconditionally disabled in the browser
  because it requires `supports_subgroup_matrix`, which was assumed
  absent.
- **Change**: runtime-check `supports_subgroup_matrix` and subgroup
  support via Dawn. Chrome Canary with WebGPU-subgroups supports this.
  Enable the path when available; fall back otherwise.
- **Expected**: sub-linear memory and compute for attention (`O(n)`
  vs `O(n²)` with a 2K context). For the current single-prompt smoke
  test this is modest; for real chat with long context it's the
  difference between usable and unusable.
- **Risk**: gated on browser support. Need to probe capabilities and
  not crash when absent.

### 5. Fused SwiGLU op ✅ DONE
- **Where**: `src/inference/model-inference.ts` FFN section.
- **Change**: replaced `opMul(opSilu(gate), up)` with `opSwigluSplit(gate, up)`
  which calls `ggml_glu_split(..., GGML_GLU_OP_SWIGLU)`. Added WASM binding
  and exported `_op_swiglu_split`.
- **Actual gain**: +1–2% (58 → ~58.5 tok/s). Modest — FFN compute is
  dominated by the three mul_mats, not by silu/mul. Kept for cleanliness.

### 6. F16 KV cache
- **Where**: `src/inference/model-inference.ts::initKVCache`.
- **Today**: KV cache is `F32`. At `maxCtx=2048, nKvHeads=4,
  headDim=64, nLayers=22`, that's 2048×64×4×4 bytes × 22 layers × 2
  (K+V) ≈ **46 MB**. Every attention step reads all of it.
- **Change**: allocate K/V as `F16`. On write, convert (or upload
  pre-converted bytes). Check `supports_op` for F16 read in mul_mat;
  the backend already handles F16 K.
- **Expected**: half the attention bandwidth; noticeable on longer
  contexts. Prefill gets 10–15%, decode scales with context.
- **Risk**: need to confirm WebGPU backend's mul_mat supports F16 K
  with F32 Q (llama.cpp default). Currently V is also F32 — making V
  F16 too drops another 2× on the V side.

---

## Low impact / polish

### 7. Skip redundant `opCont` calls ✅ DONE
- **Where**: K/V cache writes in `forward()`.
- **Change**: dropped the `opCont` before `opCpy` for both K and V writes.
  `ggml_cpy` handles strided source via `src->nb[]` in the shader params,
  so the extra contiguity dispatch was pure overhead (44 dispatches per
  forward: 2 × 22 layers).
- **Actual gain**: roughly flat after accounting for noise — between
  +0 and +2% on top of item 5. Still worth keeping: fewer dispatches,
  cleaner code, and it saves scratch tensor allocations in the graph
  context.

### 8. Skip mask tensor for nTokens=1 decode ✅ DONE (partial)
- **Where**: `forward()` mask allocation + upload + softmax_ext call.
- **Change**: for single-token decode the causal mask is trivially all
  zeros (one query, all prior keys visible). Set `needsMask = nTokens > 1`,
  skip the `tensorNew2d` and `backendTensorSet` when false. `opSoftMaxExt`
  accepts null mask (our WASM bridge and the webgpu backend both handle it).
- **Actual gain**: flat — the mask was small enough that the skip doesn't
  show up in benchmarks. Still a cleanup.
- **Not done**: pre-allocating the mask tensor at `initKVCache` time.
  Would require views into a persistent tensor and still fresh uploads
  per step; deferred.

### 9. Reduce Asyncify boundary crossings
- **Where**: every JS↔WASM hop for `backendTensorSet` + the graphCompute.
- **Today**: each tensor upload is a separate call with Asyncify suspend
  setup.
- **Change**: bundle the position/token-id/mask uploads into a single
  WASM-side helper that takes one pointer. Reduces FFI overhead.
- **Expected**: low single digits; only worth doing alongside (1).
- **Risk**: low.

### 10. Benchmark the current pipeline ✅ DONE
- **Where**: `eval/perf.ts` + `make bench-inference` + `make bench-inference-save`.
- **Change**:
  - Added TinyLlama 1.1B Chat to `eval/models.ts` model catalog.
  - New `eval/perf.ts` drives the smoke-test page in a running
    agentchrome-attached Chrome, reloads it N times with cache busting,
    scrapes `Generated N tokens in Ts (prefill: Mms, decode: Dms, T tok/s)`
    from the DOM, and prints a per-run + median table.
  - Compares median tok/s to `eval/reports/perf-baseline.json`; `--save`
    writes current run as the new baseline.
  - Wired into `Makefile` as `bench-inference` / `bench-inference-save`
    and `package.json` as `bench:inference`.
- **Actual gain**: 0 tok/s — infra only. Makes every future
  optimization measurable with a 3-run median instead of squinting at
  the smoke-test output.
- **Current baseline**: 59.9 tok/s (3-run median) for TinyLlama-1.1B Q4_0 on
  "The capital of France is".

---

## Won't-do (for now)

- **Smaller quants (Q2_K / Q3_K)**: quality/speed tradeoff, not a
  pipeline improvement. Can be layered later.
- **Speculative decoding**: requires a drafter model; large project.
- **Custom kernel fusion beyond GLU**: huge effort, marginal win.
