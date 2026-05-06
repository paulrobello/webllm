# P2-v2 Phase 2 Prototype — Closure Report

**Date:** 2026-05-06 (originally drafted), revised 2026-05-06 after Task 8 spike + Task 9 + Task 10 + Task 11/12 + Task 13/14 micro-cycles
**webllm SHA:** `039f448` (Task 11 device-hint); predecessor `4ea3d39` (Task 10 closure); predecessor Task 9 closure; predecessor Task 8 spike; predecessor `4872307` (Task 7 SUMMARY); Task 7 implementation `0f1973e`
**llama.cpp `webllm-browser-patches` SHA:** `49413d8e9` (Task 10 offload_op + supports_buft); predecessor `7919d1839` (Task 9 metadata-op allowlist); predecessor `48acb658d` (Phase 2 Tasks 1+2+4) — **patch stack frozen at +3; Task 11 was zero-patch**
**Browser + GPU:** Chrome (agentchrome session 64702), Apple Metal-3
**Prompt:** `"The capital of France is"` 6-token fixture, `max=5`, greedy (Task 8 spike harness)

## TL;DR (Task 13 + Task 14 update — synthetic offload probe)

**Gate disposition: OUTCOME E — routing-validated, execution-blocked-by-tensor-import-gap.** Task 13 (commit `4353594`) added a `webllm_synthetic_offload_probe` bridge entry-point that skips libllama and constructs a tiny ggml graph (F32 MUL_MAT, A=[128×64] B=[128]) with both src tensors allocated on `ggml_backend_cpu_buffer_type()` (host memory), then runs it through `ggml_backend_sched_new(..., op_offload=true)` with backends `[jsep, webgpu, cpu]`. Task 14 ran the probe in-browser at `?v=task14-3`. After two harness fixes (a) cache-bust the dynamic `webllm-wasm-jsep.js` import to defeat the browser's module cache after `make wasm-build-jsep`, (b) add `webllm_synthetic_offload_probe` to `JSPI_EXPORTS` in `CMakeLists.txt` so the entry frame can suspend through webgpu's async readback — the probe ran and produced this signal:

| Metric | Value | Meaning |
|---|---|---|
| `jsep.counters.runOp` delta | **1** | scheduler called `offload_op(MUL_MAT) → true` and dispatched to `jsepRunOp` ✅ |
| `jsep.counters.sync` delta | 1 | scheduler synchronized JSEP after dispatch (so the routing path expects results) |
| `jsep.counters.alloc/free/write/read/clear` delta | 0 each | no cross-backend buffer copy fired — JSEP did not pre-allocate or pre-fill from host_buf |
| Throw site | `dispatchMatmul → GpuDataManager.get(handle 0xBE3000)` | JSEP's kernel got a host-memory pointer it doesn't manage |

**Routing-layer validation: ✅ CONFIRMED.** Task 10's `offload_op` + `supports_buft(host_buft)` patch IS exercised correctly when src tensors really live on host_buf. Phase 2's architectural premise — that JSEP can advertise itself as a willing offload target and the scheduler will honor it — is proven viable at the routing layer. This is the missing piece that Outcome D's chat-decode test could not show (because libllama puts everything in webgpu_buf, never host_buf).

**Execution-layer gap: ❌ TENSOR-IMPORT SHIM MISSING.** When the scheduler dispatches a MUL_MAT to JSEP whose src tensors live on the CPU backend's host_buf, JSEP's `dispatchMatmul` looks the src handle up in its own `GpuDataManager` and finds nothing — the handle is the raw host pointer, not a JSEP buffer ID. Two viable fixes for Phase 3:

1. **JSEP-side import shim**: when `dispatchMatmul`/`dispatchRmsNorm` see a src whose buffer isn't JSEP-owned, allocate a transient JSEP buffer, blit the bytes via `device.queue.writeBuffer(...)` from the source backend's storage (CPU host_buf is `mod.HEAPU8`-readable; webgpu_buf would require a GPU↔GPU copy via `copyBufferToBuffer`), then dispatch with the temporary handle. ~50-100 LOC; lives entirely in JSEP TS, no llama.cpp patch.
2. **Pre-allocation contract**: have JSEP's `supports_buft` return true ONLY for bufts JSEP can directly read (its own + ggml-webgpu's via cross-backend GPU copy), and rely on the scheduler's pre-existing `cpy_tensor_async` mechanism to mirror tensors before dispatch. Cleaner architecturally but requires the scheduler to actually invoke the copy — needs verification that scheduler does this for offload routing (it may only do it for non-offload buft mismatches).

Path (1) is simpler and self-contained. Path (2) is more elegant if the scheduler cooperates.

**Phase 3 entry decision (firmed up by Task 14 data).** Two architecture endpoints remain:

- **Option A-prime (full JSEP residency)**: kernel SET_ROWS / GET_ROWS / ROPE / SOFT_MAX / MUL / ADD + matmul dtype permutations (~1k LOC of WGSL + dispatch). Pair with **device-hint inversion** (pass `params.devices = {jsep_dev, NULL}` instead of `{webgpu_dev, NULL}`) so libllama places weights+KV in jsep_buf. JSEP runs the entire graph; tensor-import shim never needed because all tensors are JSEP-resident. **Architecturally cleanest.**
- **Option B-prime (offload-routed mixed residency)**: keep weights+KV in webgpu_buf, register JSEP, add the tensor-import shim from path (1) above, plus the same kernel coverage as Option A-prime (because the scheduler will dispatch every supported op to JSEP via offload_op, not just MUL_MAT/RMS_NORM). **More plumbing, less architectural payoff.**

Recommendation: **Option A-prime** as the canonical Phase 3 entry. The synthetic probe has banked the value of Phase 2's offload-routing work — it's documented and reproducible — without committing to Option B-prime's plumbing tax.

**Patch budget unchanged.** llama.cpp `webllm-browser-patches` still at +3. Task 13 was zero-patch. Both Phase 3 paths above are also zero-patch.

## TL;DR (Task 11 + Task 12 update)

**Gate disposition: PARTIAL-UNBLOCK / OUTCOME D — chat works correctly with JSEP registered, but JSEP runOp counters stay at zero (JSEP structurally dormant). Device-hint succeeded at its stated mechanical goal — libllama now enumerates only WebGPU as the GPU device, weights+KV land in `webgpu_buf`, scheduler routes ALL ops to ggml-webgpu, no abort. The kept invariant Phase 2 wanted to demonstrate (JSEP kernel running an op end-to-end inside a real chat decode) was not achieved this cycle.**

**Concrete measurements (Task 12 spike, `?v=task11-1`):**

| Metric | Value |
|---|---|
| Generated text (greedy 5 tokens after "The capital of France is") | `"Paris.\n\n2"` (semantically correct) |
| Token IDs | `[3681, 29889, 13, 13, 29906]` |
| Decode wall (5 tokens, per-token-decode loop with logits readback) | 61.5 ms total → 12.30 ms/token → ~81 tok/s |
| Prefill wall (6 tokens) | 238 ms |
| Model load (304 MiB Q4_0 → GPU) | 304 ms |
| JSEP `runOp` deltas (decode window) | **0** (also `alloc/free/write/read/clear/sync` = 0) |
| Stderr proof of device-hint firing | `[webllm] JSEP build detected: pinning libllama devices to WebGPU only` + `llama_prepare_model_devices: using device WebGPU (WebGPU) (unknown id) - 4095 MiB free` (cf. Task 10 stderr: `using device JSEP (JSEP) ... 128 MiB free`) |

**Why JSEP is dormant in this configuration.** With JSEP excluded from `model->devices`, libllama's GPU buft_list contains only `[webgpu_buft, ...cpu_bufts]`. All weights+KV land in `webgpu_buf`. The scheduler's `offload_op` path (`ggml-backend.cpp:921`) is gated on `ggml_backend_buffer_is_host(src->buffer)` — webgpu_buf is not a host buffer, so the offload-to-JSEP path never fires. JSEP's broadened `supports_buft` (Task 10) accepts ggml-webgpu's buft as a runnable input, but the scheduler only checks `supports_buft` when it's actively considering a non-default backend for an op — and with WebGPU the only enumerated GPU backend, every op's natural assignment is webgpu.

**This was the predicted side-effect of approach (3) ("Option B + explicit device hint") in Task 10's Phase 3 paths.** Task 10 listed it as the "cleanest" unblock path, but cleanest meant *unblocks the abort* — it explicitly did NOT mean *exercises JSEP*. Task 11's device-hint successfully accomplishes (1) and (2) below; (3) is the unmet Phase-2 ambition:

1. ✅ **JSEP backend infrastructure verified.** Build, link, register, install JS callbacks, allocate buffers, run alongside ggml-webgpu without breaking inference. The descriptor ABI, encoder batcher, callback table, and per-binding cap discipline are all proven correct end-to-end.
2. ✅ **Coexistence verified.** A JSEP-built WASM (registers BOTH JSEP and ggml-webgpu) produces identical chat output to a non-JSEP build (registers only ggml-webgpu). No regressions.
3. ❌ **JSEP-resident kernel run inside chat decode** — not demonstrated. Would require either:
   - **Forced JSEP residency**: weights+KV in jsep_buf, all consumer ops kernel'd in JSEP (Outcome A path before Task 9; needs ~1k LOC of additional kernels — SET_ROWS, GET_ROWS, ROPE, SOFT_MAX, MUL, ADD, etc.).
   - **Mixed residency with active offload**: weights in CPU host_buf, JSEP picks up MUL_MAT/RMS_NORM via offload_op. Currently libllama doesn't produce this routing — `n_gpu_layers=999` always picks a GPU buft for tensors with GPU consumers, regardless of which GPU device is enumerated. Would need either `n_gpu_layers=0` (kills perf) or a probe with a small synthetic graph that puts weights on host_buf and verifies offload_op fires (separate measurement, not a chat-decode demonstration).

**Phase 2 gate verdict.** Phase 2 set out to demonstrate JSEP's architectural viability by routing real chat ops through JSEP-side kernels. With Tasks 1-12 completed, the **architecture is proven viable** (JSEP can be registered alongside ggml-webgpu without breaking inference, the kernel dispatch path works in isolation per Task 11's spike harness building cleanly), but **end-to-end JSEP-routed chat was not achieved**. Phase 3 must own the kernel-coverage push if JSEP-routed inference is the actual goal.

**Phase 3 entry recommendation.** Two paths remain:
- **Option A-prime (full kernel port)**: SET_ROWS, GET_ROWS, ROPE, SOFT_MAX, MUL, ADD, plus matmul dtype permutations. ~1k LOC of WGSL + dispatch code. With device-hint reverted (or kept and inverted to pin libllama to JSEP), weights live in jsep_buf and JSEP runs the entire graph. This is the "full JSEP port" — the architectural endpoint Phase 2 was a stepping stone toward.
- **Synthetic offload probe**: skip libllama entirely; build a tiny graph in raw ggml with weights on host_buf and a MUL_MAT consumer; verify the scheduler routes the MUL_MAT to JSEP via `offload_op`. Validates the Task 10 patch in its native habitat. ~100 LOC. Doesn't unblock real chat but proves the kernel dispatch path is exercised correctly — a credible smoke test for the Phase 2 work that exists.

The user signaled (in archived plan) that the strategic value of JSEP is freedom from llama.cpp's WGSL surface as upstream evolves. Both Phase 3 paths preserve that value; Option A-prime cashes it in, the synthetic probe banks it.

**Patch budget for Phase 3:** llama.cpp stack still at +3. Option A-prime would not require new patches (kernels live in JSEP backend). Synthetic probe also zero-patch.

## TL;DR (Task 10 update)

**Gate disposition: STILL-BLOCKED — Option B (narrow JSEP residency via `supports_buft` + `offload_op`) was insufficient. Same SET_ROWS abort persists, because the offload-routing path the patch enables is gated on weights living in CPU host memory — but in our build libllama places weights+KV directly into `jsep_buf`. Phase 3 must pivot to Option A-prime (broad op coverage starting with SET_ROWS) or solve the deeper "JSEP is the only enumerated GPU device" issue.**

Task 10 broadened `ggml_backend_jsep_device_supports_buft` to accept CPU host bufts and ggml-webgpu's buft (named `"WebGPU"`), and added `ggml_backend_jsep_device_offload_op` returning true for MUL_MAT (F32/F16/Q4_0/Q4_K × F32 → F32) and RMS_NORM (F32 → F32). Together these advertise JSEP's interest in running its compute even when leaves live in another backend's buft.

Spike harness behavior with the new patch: identical to Task 9 — same abort on `cache_k_l0 (view)` in `jsep_buf` cannot run `SET_ROWS`. The patch is correct per spec but does not change the failure path.

**Root cause of Outcome C — different from spec's premise:**

The spec's Option B assumed weights would land in CPU host_buf or webgpu_buf once JSEP advertised it could run ops on those bufts. In practice, `llama_prepare_model_devices` shows only ONE GPU device active: `using device JSEP (JSEP) (unknown id) - 128 MiB free`. WebGPU is registered but absent from `model->devices`. With only JSEP enumerated as GPU, libllama's GPU buft_list for JSEP-typed layers is `[jsep_buft, ...cpu_bufts]`. `select_buft` walks this list and picks the FIRST buft for which `buft_supported(buft, dev, ADD-fn)` returns true.

The `offload_op` path at `ggml-backend.cpp:921` only fires when `src_backend_id == sched->n_backends - 1` AND `ggml_backend_buffer_is_host(src->buffer)` — i.e., when weights are on the CPU host backend. Our weights live in `jsep_buf` (a non-host GPU buft), so the offload path never triggers. JSEP runs MUL_MAT/RMS_NORM via direct backend assignment (not offload) in this configuration anyway, but the SET_ROWS leaf check fires before any compute runs.

**Why WebGPU is absent from `model->devices`:** the dedup loop at `llama.cpp:195-202` uses uninitialized stack memory through `ggml_backend_dev_props props;` — `ggml_backend_dev_get_props` does memset the struct, but neither webgpu nor JSEP's `get_props` writes `device_id` (both leave it as nullptr after memset). Both null → dedup returns false → both should land in `gpus`. Empirically only one does. Possible explanations not yet confirmed: a third filter (e.g., props.memory_free comparison), an enumeration-order issue masking a logic bug, or webgpu's device_id being non-null via a path not visible in the source. Out of Phase-2 scope to chase; surfaces as a Phase 3 prerequisite.

**Phase 3 disposition:** Option B alone cannot land the gate. Three remediation paths now identified:

1. **Option A-prime** — kernel SET_ROWS, GET_ROWS, ROPE, SOFT_MAX, MUL, ADD, plus matmul dtype permutations. Effectively the back half of "full JSEP port". ~1k LOC.
2. **Option B + reg-order swap** — register JSEP BEFORE webgpu in `ggml-backend-reg.cpp` so libllama's GPU enumeration prefers webgpu (lower index → main_gpu candidate). Combined with Task 10's broadened supports_buft + offload_op, weights would land in webgpu_buf and JSEP would offload-route MUL_MAT/RMS_NORM. **But** depends on first solving the "WebGPU absent from model->devices" puzzle.
3. **Option B + explicit device hint** — pass `params.devices = {webgpu_dev}` to `llama_model_load_from_file` from `webllm_load_model` so JSEP is excluded from libllama's view entirely. JSEP still gets MUL_MAT/RMS_NORM via offload_op (the offload check is at scheduler level, not at libllama device-list level). Cleanest path; doesn't require investigating libllama's dedup bug.

**Patch budget exhausted.** llama.cpp `webllm-browser-patches` is at +3 (Phase 2 base, Task 9 metadata, Task 10 supports_buft+offload_op). Phase 3 either negotiates +1 more or uses approach (3) above which requires no llama.cpp patches.

## TL;DR (Task 9 update)

**Gate disposition: STILL-BLOCKED — Option A (metadata-op allowlist) was insufficient. Task 9 unblocked the leaf check; abort moved to a real consumer op (SET_ROWS) on the KV cache. Phase 3 must broaden the op surface beyond MUL_MAT/RMS_NORM or pivot to Option B.**

Task 9 broadened `ggml_backend_jsep_device_supports_op` to allowlist `NONE`, `VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE` (with a matching fast-path early-`continue` in `graph_compute` so `jsepRunOp` never sees them). This let `ggml_backend_sched_reserve` past the Task 8 abort point — the leaf weight check (`tensor->op == NONE` for `blk.0.attn_q.weight`) now passes.

The spike progressed from "abort at first weight" all the way to **KV cache allocated + scheduler entered `graph_reserve`** before aborting on a different op:

```
llama_kv_cache:   jsep_buf KV buffer size =    11.00 MiB
llama_context: backend_ptrs.size() = 2
sched_reserve: max_nodes = 1608
graph_reserve: reserving a graph for ubatch with n_tokens = 1, ...
ggml-backend.cpp:898: pre-allocated tensor (cache_k_l0 (view)) in
  a buffer (jsep_buf) that cannot run the operation (SET_ROWS)
Aborted()
```

`SET_ROWS` is the KV-cache write op (not metadata) — it actually mutates K/V. Other unsupported real consumer ops will follow once SET_ROWS is unblocked: `GET_ROWS` (token embedding), `ROPE`, `SOFT_MAX`, `MUL`, `ADD`, etc. Each Phase-2-shipped JSEP-resident leaf (model weights + KV cache) is consumed by ops Phase 2 doesn't kernel.

**Conclusion:** Option A alone is insufficient. The Phase-2-scoped MUL_MAT+RMS_NORM dispatch matrix cannot run a real model end-to-end. Phase 3 must either (a) widen the op kernel set materially (closer to "full JSEP port"), or (b) take **Option B** — narrow weight/KV residency so JSEP doesn't own those tensors and the scheduler routes consumer ops to ggml-webgpu/CPU.

## Original TL;DR (pre-Task-9, retained for context)

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
| 8 | Task 8 commit | `p2-v2-spike.{html,src.ts}` + `make wasm-build-jsep` wires bundle build; runs `webllm_decode` directly; SUMMARY.md revised with real failure mode |
| 9 | `7919d1839` (llama.cpp) | `ggml-jsep`: metadata-op allowlist (NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE) in `supports_op` + matching `graph_compute` fast-path. Unblocks reserve-time leaf check; abort moves to SET_ROWS on KV cache (real op, not metadata). |

llama.cpp `webllm-browser-patches` patch stack: **+2 commits since `b54503497`** (`48acb658d` Phase 2; Task 9 metadata-op allowlist). **1 patch reserved for Phase 3.**

## Open questions surfaced during the prototype

1. ~~**Decode-path swap (BLOCKING for the gate).**~~ **RESOLVED via Task 8 spike — was a misdiagnosis.** The Task 8 spike drives `webllm_decode` directly via `createLlamaBridge` and hits the same abort. The blocker is not engine routing but `supports_op` op-coverage at scheduler reserve time. See "Why the spike's `webllm_create_context` aborts" above.

2. **Pre-prototype baseline reference token sequence.** `PRE-PROTOTYPE-BASELINE.md` captures `"1. Introduction:"` at the smoke harness's profile-default temperature (~0.6). The gate criterion explicitly requires `temperature=0` greedy decode for byte-identical comparison; at temp=0 the legacy path produces `"I'm not"`. Update the baseline file with the temp=0 reference and a one-line note about the methodology error. (JSEP-path reference still pending until the BLOCKED-on-buft-routing finding clears.)

3. **Two-GPUDevice partition.** JSEP runtime owns one `GPUDevice` (acquired via `installJsepCallbacks(device)` in JS); ggml-webgpu owns another (Dawn-internal). Phase 2 acceptable; Phase 3 unification path: either (a) export Dawn's WebGPU device from WASM via a JSEP callback so JS reuses it, or (b) port enough ops to JSEP that ggml-webgpu can be retired entirely. (b) is the natural endpoint of Phase 3.

4. **Per-dispatch shape uniform allocation.** Both `matmul.ts` and `rms-norm.ts` create a fresh `GPUBuffer` per dispatch and never destroy it. Phase 3 should cache by literal shape tuple OR ring-buffer the uniforms by submit index. FIXME comments mirror across both kernels (matmul.ts:425, rms-norm.ts:170).

5. **Module-global dispatch helpers.** Once op #3 lands in Phase 3, factor `buildSimplePipeline(device, wgsl, bindings[])`, `allocAndWriteUniform(device, packer)`, and `bindAndRecord(ctx, pipeline, layout, buffers, dispatch)` into `src/inference/jsep/dispatch-helpers.ts`. Don't pre-factor at n=2.

6. **Defensive `supports_op` re-check in `graph_compute`.** Currently always-on; pays cap-walk + dtype-switch on every node. Phase 3: NDEBUG-gate (`#ifdef NDEBUG return GGML_STATUS_SUCCESS_FAST_PATH;`).

7. **`opParamsPtr` alignment.** `dispatchRmsNorm` reads eps via `new Float32Array(heap, ptr, 1)` which throws if `ptr % 4 !== 0`. ggml's `op_params` is naturally 4-byte-aligned by struct layout but worth documenting via comment or `(ptr & 3) === 0` defensive assert.

8. **Q4_K kernel deferral.** Currently throws `"matmul Q4_K kernel: deferred to Task 7"` if invoked. Tinyllama Q4_0 doesn't trigger it; once Phase 3 routes a Q4_K-using model through JSEP, the kernel needs to land. Hand-packing Q4_K test data is involved (256-elem super-blocks with 6-bit-quantized scales); plan for ~150 additional LOC.

## Next-session disposition (revised post-Task-9)

**Phase 3 entry — Option A insufficient, choose A-prime or B.**

Task 9 took the **first-step** of Option A (metadata-op allowlist) and
proved it unblocks the leaf check but **doesn't unblock inference** —
the next abort is on `SET_ROWS` (a real KV-cache mutation op),
followed by an open-ended tail of consumer ops (GET_ROWS, ROPE,
SOFT_MAX, MUL, ADD, ...) that all touch JSEP-resident tensors.

**Two real-Phase-3 paths now:**

- **Option A-prime — broad op coverage.** Kernel SET_ROWS,
  GET_ROWS, ROPE, SOFT_MAX, MUL, ADD, plus matmul dtype permutations
  (F16-out, Q4_K). Each is a WGSL kernel + dispatch wiring + bind-
  layout cache entry. Effectively the back half of "full JSEP port"
  without the framework. Estimate: 6-10 ops × ~150 LOC each = ~1k
  LOC C++/WGSL/TS. Patch budget: well within the remaining 1
  reserved patch (all C++ changes are inside `ggml-jsep.cpp`; new
  TS files add no patches).

- **Option B — narrow weight residency.** Switch `device_supports_buft`
  + `offload_op` so JSEP only owns intermediates produced by JSEP-
  dispatched ops. Model weights stay in `ggml-webgpu` (or CPU); KV
  cache stays in `ggml-webgpu`. JSEP becomes opt-in per-op rather
  than backend-of-record for the whole partition. Smaller surface
  but requires understanding ggml's `offload_op` semantics + handling
  the cross-backend tensor handoff. Estimate: ~200-300 LOC C++.

**Recommendation:** Option B is the lower-risk Phase 3 entry. It
keeps JSEP a kernel-server rather than a backend-of-record,
preserves the patch budget, and keeps the per-binding cap surface
small. Option A-prime is the natural endpoint **if** the gate band
on Option B comes back red (e.g., cross-backend handoff dominates).

**Patch budget:** llama.cpp `webllm-browser-patches` patch stack is
+2 (Phase 2 `48acb658d`; Task 9 metadata-op allowlist). **1 patch
remains reserved for Phase 3.** Option B fits in 1 patch; Option
A-prime would consume the remaining patch and likely require a
second one if dtype permutations land separately.

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
