# P2-v2 Phase 2 Prototype — Closure Report

**Date:** 2026-05-06
**webllm SHA:** `0f1973e` (Task 7 closure commit; predecessor `d1a8348f` Task 6 amended)
**llama.cpp `webllm-browser-patches` SHA:** `48acb658d` (Phase 2 Tasks 1+2+4 amended)
**Browser + GPU:** Chrome (agentchrome session), Apple Metal-3
**Prompt:** `"Hello"`, `max=5`, `greedy=1` (temperature=0)

## TL;DR

**Gate disposition: BLOCKED.** The JSEP backend was successfully shipped end-to-end across Tasks 1-6 (C++ skeleton + dispatch, descriptor ABI, TS runtime, matmul + rms_norm kernels, engine integration, bundle wiring), but the gate metric is **structurally unmeasurable from this run**: the JSEP backend is *registered* but *dormant* — `chatCompletion` still routes through legacy `ModelInference.forward → wasm.op* → ggml-webgpu` directly, never engaging the ggml scheduler. JSEP's `supports_op` correctly claims `MUL_MAT` + `RMS_NORM`; `installJsepCallbacks` runs; counters are wired; but the scheduler is never engaged so `ggml-jsep::graph_compute` sees zero work.

The fix (decode-path swap to route through `webllm_decode` when `backend === "jsep"`) is Phase 2 follow-on work that the spec §D5 "Engine integration" envisioned but Task 6's implementation did not realize. The bridge surface needed (`webllm_decode`, `webllm_get_logits`, `webllm_create_context`) was kept across the P2 v1 revert and is ready to use; this is not a new C++ patch.

## Token output

| Build | Output | Bytes-identical? |
|---|---|---|
| Legacy (default backend) | `"I'm not"` | reference |
| JSEP (`backend: "jsep"`) | `"I'm not"` | yes |

(Note: `PRE-PROTOTYPE-BASELINE.md` records `"1. Introduction:"` as the legacy 5-token greedy reference. That capture used the smoke harness's profile-default temperature (~0.6); when re-run at `temperature=0` (the gate's prescribed greedy decode), both legacy and JSEP produce `"I'm not"`. The 2026-05-05 baseline file should be amended with the temp=0 sequence — see Open Question #2 below.)

## Gate metrics

| Metric | Legacy baseline (today) | JSEP measured | Gate band |
|---|---|---|---|
| Per-token wall (5-token median) | 5.6 ms (28 ms / 5 tokens at temp=0) | 5.6 ms | green (≤2× legacy) |
| EM_ASM crossings/token (decode-only) | n/a (cwrap path: 450 dispatches/token) | 0 (counters dormant — see below) | n/a |
| Greedy 5/5 token equality | `"I'm not"` reference | byte-identical | pass |

**Per-callback counter breakdown (decode-only, post-warmup):**

| Callback | Count over 5 decode steps | Per-token | Notes |
|---|---|---|---|
| `jsepAlloc` | 0 | 0 | dormant |
| `jsepFree` | 0 | 0 | dormant |
| `jsepWrite` | 0 | 0 | dormant |
| `jsepRead` | 0 | 0 | dormant |
| `jsepClear` | 0 | 0 | dormant |
| `jsepRunOp` | 0 | 0 | **dormant — backend never invoked** |
| `jsepSync` | 0 | 0 | dormant |

`module.__jsep` is populated; `jsepInstalled === true`; install path is correctly wired (verified via `js exec` inspection). The dormancy is structural, not a bug in the install path.

## Why the JSEP backend was never invoked

The legacy `ModelInference.forward` path builds the ggml graph by calling cwrap'd `wasm.opMatMul`, `wasm.opRmsNorm`, etc. directly. These C++ exports map to ggml's `ggml_*_mat_mul`, `ggml_rms_norm` constructors (which only build the cgraph node — they don't dispatch to a backend). The graph is then computed via `wasm.graphCompute` → `_graph_compute` → ggml's `ggml_backend_sched_graph_compute_async` — but **only if** the graph was built against a scheduler that knows about all registered backends.

Reading `model-inference.ts`: it uses `ggml_backend_alloc_ctx_tensors` which assigns tensors to a *single* backend (the first registered). With ggml-webgpu being the only backend that was registered before P2-v2 (and still being the registered backend even with ggml-jsep also linked), the entire graph stays on ggml-webgpu. The ggml-jsep backend's `supports_op` is never queried because the scheduler is not making the routing decision.

The spec §D5 "Engine integration" assumed:

> Engine routes `chatCompletion` through `webllm_decode` (libllama-driven decode path) instead of the legacy hand-rolled `ModelInference.forward`.

That decode-path swap was implemented and **then reverted** during the P2 v1 → P2 v2 redirect (commit `0b57d41`, `revert(p2): roll back wrapper+dispatch+delete; keep bridge surface`). Task 6's "engine integration" only loaded the JSEP WASM artifact and installed JSEP callbacks; it did not reactivate the `webllm_decode` route. Without that reactivation, the JSEP backend is technically registered but unused.

## Per-task commit map

| Task | Commit | Description |
|---|---|---|
| 0 | `91e0396` + `1094351` | JSPI hang fix + pre-prototype baseline capture (125.3 tok/s) |
| 1 + 2 + 4 (amended) | `48acb658d` (llama.cpp) | C++ ggml-jsep skeleton + MUL_MAT/RMS_NORM dispatch + descriptor ABI |
| 1 (build) | `160d488` | webllm `wasm-build-jsep` Makefile + CMake target |
| 3 | `09ba2d4` | TS runtime scaffold (`installJsepCallbacks`, `GpuDataManager`, `CommandEncoderBatcher`, `PipelineCache`) |
| 4 | `43390b0` | matmul kernel (F32, F16, Q4_0) + bind-layout cache |
| 5 | `04a38cc` | rms_norm kernel (F32) |
| 6 | `d1a8348f` | engine `backend: "jsep"` opt-in + bundle wiring + resource-leak fixes |
| 7 | `0f1973e` | counter instrumentation + parallel `real-model-jsep.html` + closure stub |

llama.cpp `webllm-browser-patches` patch stack: **+1 commit since `b54503497`** (`48acb658d`). 2 patches reserved for Phase 3 unchanged.

## Open questions surfaced during the prototype

1. **Decode-path swap (BLOCKING for the gate).** `chatCompletion` must route through `webllm_decode` when `backend === "jsep"`. The bridge surface is already exported (`webllm_decode`, `webllm_get_logits`, `webllm_create_context`, `webllm_load_model`); P2 v1's `LlamaDecodeWrapper` was deleted in the revert but can be reconstructed from `src/inference/llama-bridge.ts`. Estimated scope: ~100-200 LOC TS-only, no new C++ patches.

2. **Pre-prototype baseline reference token sequence.** `PRE-PROTOTYPE-BASELINE.md` captures `"1. Introduction:"` at the smoke harness's profile-default temperature (~0.6). The gate criterion explicitly requires `temperature=0` greedy decode for byte-identical comparison; at temp=0 both legacy and JSEP produce `"I'm not"`. Update the baseline file with the temp=0 reference and a one-line note about the methodology error.

3. **Two-GPUDevice partition.** JSEP runtime owns one `GPUDevice` (acquired via `installJsepCallbacks(device)` in JS); ggml-webgpu owns another (Dawn-internal). Phase 2 acceptable; Phase 3 unification path: either (a) export Dawn's WebGPU device from WASM via a JSEP callback so JS reuses it, or (b) port enough ops to JSEP that ggml-webgpu can be retired entirely. (b) is the natural endpoint of Phase 3.

4. **Per-dispatch shape uniform allocation.** Both `matmul.ts` and `rms-norm.ts` create a fresh `GPUBuffer` per dispatch and never destroy it. Phase 3 should cache by literal shape tuple OR ring-buffer the uniforms by submit index. FIXME comments mirror across both kernels (matmul.ts:425, rms-norm.ts:170).

5. **Module-global dispatch helpers.** Once op #3 lands in Phase 3, factor `buildSimplePipeline(device, wgsl, bindings[])`, `allocAndWriteUniform(device, packer)`, and `bindAndRecord(ctx, pipeline, layout, buffers, dispatch)` into `src/inference/jsep/dispatch-helpers.ts`. Don't pre-factor at n=2.

6. **Defensive `supports_op` re-check in `graph_compute`.** Currently always-on; pays cap-walk + dtype-switch on every node. Phase 3: NDEBUG-gate (`#ifdef NDEBUG return GGML_STATUS_SUCCESS_FAST_PATH;`).

7. **`opParamsPtr` alignment.** `dispatchRmsNorm` reads eps via `new Float32Array(heap, ptr, 1)` which throws if `ptr % 4 !== 0`. ggml's `op_params` is naturally 4-byte-aligned by struct layout but worth documenting via comment or `(ptr & 3) === 0` defensive assert.

8. **Q4_K kernel deferral.** Currently throws `"matmul Q4_K kernel: deferred to Task 7"` if invoked. Tinyllama Q4_0 doesn't trigger it; once Phase 3 routes a Q4_K-using model through JSEP, the kernel needs to land. Hand-packing Q4_K test data is involved (256-elem super-blocks with 6-bit-quantized scales); plan for ~150 additional LOC.

## Next-session disposition

**Phase 2 follow-on cycle (BLOCKING for Phase 3):** Wire `chatCompletion` to route through `webllm_decode` when `engine.init({ backend: "jsep" })` was called. Reconstruct minimal `LlamaDecodeWrapper` (or fold directly into `engine.ts`) using the existing bridge exports. Re-run the 5-token greedy decode through the JSEP path; capture actual `module.__jsep.counters` post-warmup; apply T3 gate (green/yellow/red).

**Expected outcome of the follow-on cycle:** YELLOW. Phase 2 only supports MUL_MAT + RMS_NORM via JSEP; everything else (RoPE, softmax, swiglu, view, copy, etc.) returns false from `supports_op` and routes to CPU. Each scheduler boundary between JSEP and CPU pays a CPU↔GPU round-trip (set_tensor/get_tensor over EM_ASM). Per-token wall likely 2-5× legacy due to CPU-fallback overhead. The graph-once-dispatch pre-baked yellow-recovery lever (per spec §risk register) becomes the natural micro-cycle if the YELLOW result confirms.

**Patch budget:** llama.cpp `webllm-browser-patches` patch stack is +1 (`48acb658d`); the follow-on cycle is TS-only, no new C++ patches anticipated. 2 patches remain reserved for Phase 3.

## Bench artifacts (reproducibility)

The Task 7 commit (`0f1973e`) ships:
- `smoke-test/real-model-jsep.html` — parallel JSEP smoke harness pinning `?backend=jsep`
- `smoke-test/real-model-page.js` — `?backend=jsep` query handler that swaps the bundle import + plumbs `backend: "jsep"` to `engine.init`
- `src/inference/jsep/index.ts` — per-callback counter instrumentation (`module.__jsep.counters`)

To reproduce the 5-token decode metrics:
```bash
make wasm-build-jsep && make smoke-serve
# in agentchrome:
agentchrome navigate "http://localhost:8031/real-model-jsep.html?model=tinyllama-1.1b-chat-q4_0&prompt=Hello&max=5&greedy=1"
# After [7/8] completes, read counters:
agentchrome js exec 'JSON.stringify(window.WebLLM.engine.wasm.m.__jsep.counters)'
# Expected post-fix (follow-on cycle): non-zero across alloc/runOp/sync; zero today.
```
