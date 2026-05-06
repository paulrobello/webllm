# P2-v2 Phase 2 Prototype — Closure Report

**Date:** 2026-05-06 (originally drafted), revised 2026-05-06 after Task 8 spike
**webllm SHA:** Task 8 spike commit (this PR); predecessor `4872307` (Task 7 SUMMARY); Task 7 implementation `0f1973e`
**llama.cpp `webllm-browser-patches` SHA:** `48acb658d` (Phase 2 Tasks 1+2+4 amended)
**Browser + GPU:** Chrome (agentchrome session 64702), Apple Metal-3
**Prompt:** `"The capital of France is"` 6-token fixture, `max=5`, greedy (Task 8 spike harness)

## TL;DR

**Gate disposition: BLOCKED-on-scheduler-buft-routing.** The Task 8 follow-on spike (`smoke-test/p2-v2-spike.html`, drives `webllm_decode` directly — bypasses any "did engine route correctly?" question) **revised the Task 7 diagnosis**. The JSEP backend is **NOT dormant** — it is fully engaged at allocation time:

```
load_tensors:     jsep_buf model buffer size =   455.06 MiB
llama_kv_cache: layer  0..21: dev = JSEP
llama_kv_cache:   jsep_buf KV buffer size =   11.00 MiB
llama_context: backend_ptrs.size() = 2
```

But `llama_new_context_with_model` aborts during scheduler reservation:

```
graph_reserve: reserving a graph for ubatch with n_tokens = 1, ...
ggml-backend.cpp:898: pre-allocated tensor (blk.0.attn_q.weight) in
  a buffer (jsep_buf) that cannot run the operation (NONE)
Aborted()
```

The scheduler's reserve-time pass walks every JSEP-resident leaf and
asserts the leaf's buffer-backend supports the consumer op. Phase 2's
narrow `supports_op` (MUL_MAT + RMS_NORM only) misses ops libllama's
tinyllama graph relies on (likely GET_ROWS / VIEW / TRANSPOSE / PERMUTE
or a non-F32-dst matmul path), so the scheduler rejects the partition
*before* any compute runs.

This **invalidates the Task 7 closure's "dormant" framing**. Task 7
inferred dormancy from `chatCompletion`'s legacy routing path, but the
real shape was masked by `chatCompletion` failing earlier (load
succeeded, context creation aborted, no decode reached). The spike
unmasks this by calling `webllm_decode` directly and hitting the same
abort at `webllm_create_context`.

Detailed root cause + fix options in
[`SPIKE-RESULTS.md`](./SPIKE-RESULTS.md). Three remediation paths
identified:
- **Option A** — broaden `supports_op` + add CPU-fallback stub dispatch (smallest delta)
- **Option B** — narrow weight residency (CPU/ggml-webgpu owns weights, JSEP gets only intermediates via `offload_op`)
- **Option C** — full JSEP port (Phase 3 endpoint)

## Token output

**Decode never ran.** The 5-token greedy continuation was never reached
because `webllm_create_context` aborts. No JSEP-path token output exists
to compare against the legacy reference. The legacy reference at
temperature=0 is `"I'm not"` (preserved from the Task 7 capture).

## Gate metrics — N/A (decode never ran)

| Metric | Legacy ref (temp=0) | JSEP measured | Gate band |
|---|---|---|---|
| Per-token wall (5-token median) | 5.6 ms | **N/A** — abort in createContext | BLOCKED |
| EM_ASM crossings/token (decode-only) | n/a | **N/A** | BLOCKED |
| Greedy 5/5 token equality | `"I'm not"` | **N/A** | BLOCKED |

**Per-callback counter breakdown (model-load only — decode never ran):**

| Callback | Count at abort | Per-token | Notes |
|---|---|---|---|
| `jsepAlloc` | ~25-50 (inferred) | n/a | model + KV buffers allocated successfully |
| `jsepFree` | 0 | n/a | nothing freed before abort |
| `jsepWrite` | ~200+ (inferred) | n/a | one per weight tensor uploaded |
| `jsepRead` | 0 | n/a | no readback before abort |
| `jsepClear` | 0 | n/a | no clear before abort |
| `jsepRunOp` | **0** | n/a | abort precedes any op dispatch |
| `jsepSync` | 0 | n/a | no flush before abort |

The "model+KV buffers actually live in jsep_buf" finding is from the
console: `jsep_buf model buffer size = 455.06 MiB` and per-layer
`llama_kv_cache: layer N: dev = JSEP`. Counter snapshot in TS would
need a `try/catch` around `bridge.createContext` to capture the exact
allocation count; the spike harness was structured to snapshot
post-load (which succeeded) and post-decode (which never ran).

The Task 7 "dormant — counters all zero" reading came from the
`chatCompletion` path that fails earlier (engine init / model load
through legacy route doesn't even hit JSEP allocation). The spike
proves JSEP's allocation path runs; only its op-coverage path is
incomplete.

## Why the spike's `webllm_create_context` aborts

The Task 8 spike harness drives `webllm_decode` directly through the
`createLlamaBridge` surface — bypassing both `chatCompletion` and the
engine's legacy routing entirely. It still aborts, which proves the
issue is not the engine routing layer (Task 7's hypothesis) but the
**scheduler-reservation contract**.

`llama_new_context_with_model` calls `ggml_backend_sched_reserve` to
pre-allocate worst-case memory and validate that every scheduler-
generated graph node can run on its assigned backend. Per
`ggml-backend.cpp:898`, this check walks every leaf tensor whose
buffer is owned by a backend `B` and asserts `B->supports_op(consumer)`
for the op that consumes it. When `B = jsep_buf`, the assert fails for
ops outside Phase 2's narrow `supports_op` matrix.

ggml-jsep's `device_supports_op` (ggml-jsep.cpp:500-540) returns true
only for:

- `GGML_OP_MUL_MAT` with `src1=F32`, `src0 ∈ {F32,F16,Q4_0,Q4_K}`, `dst=F32`
- `GGML_OP_RMS_NORM` with `src0=F32`, `dst=F32`

Tinyllama's first-layer ggml graph uses additional ops on JSEP-resident
weights — likely `GET_ROWS` (token embedding lookup), `VIEW`/`PERMUTE`
(attention head reshape), `TRANSPOSE` (matmul prep), `MUL_MAT` with
non-F32 dtype combinations, or the F16-output variant. The narrow
matrix rejects these, and the scheduler aborts at reserve time.

This means **no Phase-2-scoped configuration of JSEP can run a real
model end-to-end** without either:

- broadening `supports_op` (Option A — even with stub dispatch that
  returns `STATUS_NOT_IMPLEMENTED` to force CPU fallback), or
- changing weight residency so JSEP doesn't own the leaf tensors at
  all (Option B — `offload_op` semantics).

Option A is the smallest-delta unblock; the existing `runOp` already
returns `STATUS_NOT_IMPLEMENTED` for unhandled ops, but the C++
`supports_op` returns false at reserve time, which is stricter (no
fallback path is offered to the scheduler). Reversing that — return
true and let `runOp` decline — is the pattern the upstream backends
(BLAS, CPU) use.

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
| 8 | this commit | `p2-v2-spike.{html,src.ts}` + `make wasm-build-jsep` wires bundle build; runs `webllm_decode` directly; SUMMARY.md revised with real failure mode |

llama.cpp `webllm-browser-patches` patch stack: **+1 commit since `b54503497`** (`48acb658d`) — Task 8 added no C++ patches. 2 patches reserved for Phase 3 unchanged.

## Open questions surfaced during the prototype

1. ~~**Decode-path swap (BLOCKING for the gate).**~~ **RESOLVED via Task 8 spike — was a misdiagnosis.** The Task 8 spike drives `webllm_decode` directly via `createLlamaBridge` and hits the same abort. The blocker is not engine routing but `supports_op` op-coverage at scheduler reserve time. See "Why the spike's `webllm_create_context` aborts" above.

2. **Pre-prototype baseline reference token sequence.** `PRE-PROTOTYPE-BASELINE.md` captures `"1. Introduction:"` at the smoke harness's profile-default temperature (~0.6). The gate criterion explicitly requires `temperature=0` greedy decode for byte-identical comparison; at temp=0 the legacy path produces `"I'm not"`. Update the baseline file with the temp=0 reference and a one-line note about the methodology error. (JSEP-path reference still pending until the BLOCKED-on-buft-routing finding clears.)

3. **Two-GPUDevice partition.** JSEP runtime owns one `GPUDevice` (acquired via `installJsepCallbacks(device)` in JS); ggml-webgpu owns another (Dawn-internal). Phase 2 acceptable; Phase 3 unification path: either (a) export Dawn's WebGPU device from WASM via a JSEP callback so JS reuses it, or (b) port enough ops to JSEP that ggml-webgpu can be retired entirely. (b) is the natural endpoint of Phase 3.

4. **Per-dispatch shape uniform allocation.** Both `matmul.ts` and `rms-norm.ts` create a fresh `GPUBuffer` per dispatch and never destroy it. Phase 3 should cache by literal shape tuple OR ring-buffer the uniforms by submit index. FIXME comments mirror across both kernels (matmul.ts:425, rms-norm.ts:170).

5. **Module-global dispatch helpers.** Once op #3 lands in Phase 3, factor `buildSimplePipeline(device, wgsl, bindings[])`, `allocAndWriteUniform(device, packer)`, and `bindAndRecord(ctx, pipeline, layout, buffers, dispatch)` into `src/inference/jsep/dispatch-helpers.ts`. Don't pre-factor at n=2.

6. **Defensive `supports_op` re-check in `graph_compute`.** Currently always-on; pays cap-walk + dtype-switch on every node. Phase 3: NDEBUG-gate (`#ifdef NDEBUG return GGML_STATUS_SUCCESS_FAST_PATH;`).

7. **`opParamsPtr` alignment.** `dispatchRmsNorm` reads eps via `new Float32Array(heap, ptr, 1)` which throws if `ptr % 4 !== 0`. ggml's `op_params` is naturally 4-byte-aligned by struct layout but worth documenting via comment or `(ptr & 3) === 0` defensive assert.

8. **Q4_K kernel deferral.** Currently throws `"matmul Q4_K kernel: deferred to Task 7"` if invoked. Tinyllama Q4_0 doesn't trigger it; once Phase 3 routes a Q4_K-using model through JSEP, the kernel needs to land. Hand-packing Q4_K test data is involved (256-elem super-blocks with 6-bit-quantized scales); plan for ~150 additional LOC.

## Next-session disposition

**Phase 2 follow-on micro-cycle (BLOCKING for Phase 3):** Apply
**Option A** — broaden `supports_op` to return true for the ops
libllama touches on JSEP-resident leaves, with `runOp` returning
`STATUS_NOT_IMPLEMENTED` for the ones not yet kernelized. This is
the upstream-backend pattern (BLAS/CPU) and the smallest delta to
unblock scheduler reservation.

Concrete plan:
1. Inventory the ops tinyllama's first-layer graph touches on
   `attn_q.weight` (and any other JSEP-resident leaf). The
   `graph_reserve` log-flood after enabling `GGML_LOG_DEBUG` will
   list them. Likely set: `GET_ROWS`, `VIEW`, `PERMUTE`, `RESHAPE`,
   `TRANSPOSE`, plus matmul dtype permutations.
2. Add those ops to `device_supports_op` returning `true` (gate
   on cap + dtype where needed; otherwise unconditional).
3. In `graph_compute`, for each unhandled op call ggml's CPU-
   fallback path (`ggml_compute_forward_*`) with backed-up host
   tensors, OR mark the op as not-our-problem and let the scheduler
   route via a CPU buffer copy.
4. Re-run the spike harness; capture `__jsep.counters` post-decode;
   apply T3 gate (likely YELLOW per spec §risk register — high
   crossing rate due to fallback round-trips).

If Option A turns out to require non-trivial fallback wiring (more
than ~50 LOC C++), pivot to Option B (narrow weight residency,
JSEP only owns intermediates).

**Expected outcome:** YELLOW band on per-token wall (2-5× legacy due
to CPU↔GPU round-trips per fallback boundary), but green on byte-
identical token output. The yellow-recovery lever (graph-once
dispatch / batched JSEP allocation) becomes the natural follow-on
micro-cycle.

**Patch budget:** llama.cpp `webllm-browser-patches` patch stack is
+1 (`48acb658d`); Option A is C++ only (`ggml-jsep.cpp` change),
will add ~30-100 LOC and no new patch files. Estimate +1 patch
commit. 2 patches remain reserved for the rest of Phase 3.

## Bench artifacts (reproducibility)

Task 7 (`0f1973e`):
- `smoke-test/real-model-jsep.html` — parallel JSEP smoke harness pinning `?backend=jsep`
- `smoke-test/real-model-page.js` — `?backend=jsep` query handler that swaps the bundle import + plumbs `backend: "jsep"` to `engine.init`
- `src/inference/jsep/index.ts` — per-callback counter instrumentation (`module.__jsep.counters`)

Task 8 (this commit):
- `smoke-test/p2-v2-spike.{html,src.ts}` — direct-decode spike (bypasses engine; drives `webllm_decode` via `createLlamaBridge`)
- `Makefile::wasm-build-jsep` — adds `bun build smoke-test/p2-v2-spike.src.ts` line
- `eval/reports/p2-v2-prototype-2026-05-05/SPIKE-RESULTS.md` — full diagnostic + remediation options
- `eval/reports/p2-v2-prototype-2026-05-05/spike-console-task8.log` — raw browser console capture (384 lines)

To reproduce the spike abort:
```bash
make wasm-build-jsep        # builds WASM + spike bundle
agentchrome --port 64702 navigate --tab <ID> "http://localhost:8031/p2-v2-spike.html?v=task8-N"
# Wait for #log to contain "DONE" or "FAIL"; FAIL is expected pre-Option-A
agentchrome --port 64702 console read --tab <ID> --limit 5000 | grep "ggml-backend.cpp:898"
```
