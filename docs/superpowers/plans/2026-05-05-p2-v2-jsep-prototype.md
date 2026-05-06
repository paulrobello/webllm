# P2-v2 — JSEP-style backend single-op prototype implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user's CLAUDE.md says "Plan approval IS execution approval" — when invoked, begin Task 0 immediately without asking for confirmation. Each task ships as a separate commit per the `Always commit before work` doctrine.

**Goal:** Validate the JSEP-style architecture for webllm by shipping a `ggml-jsep` backend that handles **MUL_MAT + RMS_NORM only**; everything else falls through to the CPU backend via ggml's scheduler. Run a 5-token tinyllama Q4_0 greedy decode through the prototype and compare per-token wall-clock + JS↔WASM crossing count against the legacy `ModelInference` baseline. Gate decision (green/yellow/red) decides whether Phase 3 (full op coverage, ~40 TS kernels) plan-write begins.

**Architecture:** A new `ggml-jsep` ggml backend in llama.cpp's tree (`ggml/src/ggml-jsep/ggml-jsep.cpp`) registers an 8-entry callback table on `Module` at WASM init. `graph_compute(cgraph)` walks `cgraph->nodes` linearly; for each `MUL_MAT` or `RMS_NORM` node it emits one `EM_ASM_INT(Module.jsepRunOp, ...)` into JS. Other op kinds return `false` from `supports_op` so the scheduler peels them to a CPU backend. JS-side TS modules under `src/inference/jsep/` provide: callback installation, GPU-data manager (handle → `GPUBuffer` map, bucketed reuse), command-encoder batcher (single open encoder, flushed on N-dispatches or sync), pipeline cache keyed on (op, dtype, shape-deps), and per-op WGSL kernels for matmul + rms_norm ported from existing `model-inference.ts`. Engine integration is opt-in via `engine.init({ backend: "jsep" })`; legacy default unchanged.

**Tech Stack:** C++ (~600 LoC ggml-jsep.cpp + CMake hook + 1-line registration on `webllm-browser-patches`), TypeScript (~800-1000 LoC under `src/inference/jsep/`), WGSL (matmul + rms_norm kernels adapted from `model-inference.ts`), Bun (unit goldens), agentchrome (browser smoke), `webllm_perf_counter` + a new EM_ASM-counter hook for gate metrics.

**Spec:** [`docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md`](../specs/2026-05-05-p2-v2-jsep-prototype-design.md).

**Predecessor:** Phase 1 (research probe) closed 2026-05-05 — three artifacts at `eval/reports/p2-v2-jsep-research-2026-05-05/`:
- `JSEP-ABI-MENTAL-MODEL.md` (commit `ec18120`) — JSEP ABI characterization
- `GGML-BACKEND-CONTRACT.md` (commit `cec7172`) — minimum vtable surface
- `GGML-OP-CATALOG.md` (commit `b6d807a`) — bounded op set for Phase 3

Predecessor migration history: P2 v1 reverted 2026-05-05 (commit `0b57d41`) after measuring 18× decode regression from `emdawnwebgpu`-driven per-`wgpu`-API-call shim crossings. Bridge surface kept across the revert; legacy `ModelInference` restored at canonical-6 baseline.

---

## File Structure

**Create (TypeScript):**
- `src/inference/jsep/index.ts` — public `installJsepCallbacks(module)`; registers the 8-entry table on `Module.jsep*`. ~80 LOC.
- `src/inference/jsep/gpu-data-manager.ts` — `GpuDataManager` class: `alloc(size) → handle`, `free(handle)`, `get(handle) → {buffer, size}`, bucketed free-list keyed on size class. ~200 LOC.
- `src/inference/jsep/command-encoder.ts` — `CommandEncoderBatcher`: single open `GPUCommandEncoder`; `record(programInfo, bindGroup, dispatch)`; `flush()` on N-dispatches (default 16) or explicit sync. ~120 LOC.
- `src/inference/jsep/pipeline-cache.ts` — `PipelineCache`: `Map<string, GPUComputePipeline>`; key built from `(op, dtype-tuple, shape-deps)`. ~100 LOC.
- `src/inference/jsep/ops/matmul.ts` — WGSL kernel + dispatch logic for `MUL_MAT`. F32×F32, F16×F16, Q4_0×F32, Q4_K×F32 dtype paths. ~400 LOC including WGSL strings.
- `src/inference/jsep/ops/rms-norm.ts` — WGSL kernel + dispatch logic for `RMS_NORM`. F32 only. ~150 LOC.
- `tests/jsep-buffer-roundtrip.test.ts` — Task 3 verification.
- `tests/jsep-matmul-golden.test.ts` — Task 4 verification.
- `tests/jsep-rms-norm-golden.test.ts` — Task 5 verification.
- `eval/reports/p2-v2-prototype-<CLOSE_DATE>/SUMMARY.md` — Task 7 closure report.
- `eval/reports/p2-v2-prototype-<CLOSE_DATE>/PRE-PROTOTYPE-BASELINE.md` — Task 0 baseline.

**Create (C++ / build):**
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` — backend impl. ~600 LOC. Lives on `webllm-browser-patches`; one new patch.
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/CMakeLists.txt` — gated on `-DGGML_BACKEND_JSEP=ON`.
- 1-line registration in `~/Repos/llama.cpp/ggml/src/ggml-backend-reg.cpp` under the existing `#ifdef`/registration-list pattern.

**Modify:**
- `src/wasm/CMakeLists.txt` — add a second build target / variant for the jsep WASM artifact (or extend existing target with a `WEBLLM_BACKEND_JSEP` flag). Output: `dist/webllm-wasm-jsep.{js,wasm}`.
- `package.json` — add `bun build` script `build:jsep` producing `dist/webllm-bundle-jsep.js` (includes `src/inference/jsep/**`).
- `Makefile` — add `wasm-build-jsep` target alongside existing `wasm-build`. `make checkall` continues to validate the default build only; jsep build runs under `make wasm-build-jsep` opt-in.
- `src/core/engine.ts` — accept optional `backend?: "default" | "jsep"` in `init({...})`. When `"jsep"`: load `webllm-wasm-jsep.{js,wasm}` instead of standard pair, call `installJsepCallbacks(module)` before `webllm_load_model`. Default (`"default"` or unset) preserves current behavior.
- `src/index.ts` — export `JsepBackend` config type; do not change existing exports.
- `TODO.md` — Task 7 closure stub + green/yellow/red disposition.

**Untouched:**
- `src/inference/model-inference.ts` — legacy graph builder stays the default; not deleted in Phase 2.
- `src/inference/encoder-inference.ts`, `causal-embedder-inference.ts` — encoder + embedder paths unchanged.
- All other engine code (sampler, generation, stream-router, chat-template, tokenizer) — shared between legacy and jsep builds.

---

## Pre-flight (Task 0)

### Task 0: Capture pre-prototype baseline + verify session is green

**Files:**
- Read: `eval/models.ts` (tinyllama Q4_0 registration)
- Read: `TODO.md` header block (canonical baselines)
- Create: `eval/reports/p2-v2-prototype-<CLOSE_DATE>/PRE-PROTOTYPE-BASELINE.md`

- [ ] **Step 1: Confirm tree state**

Run:
```bash
cd /Users/probello/Repos/webllm && git log -1 --oneline && git status --short
cd /Users/probello/Repos/llama.cpp && git log -1 webllm-browser-patches --oneline && git status --short
```
Expected: webllm HEAD at `f3e87f7 docs(p2-v2.spec): expand Phase 2 to matmul + RMS_NORM` or later, working tree clean. llama.cpp `webllm-browser-patches` at `b54503497 ggml-webgpu: use wgpu::WaitAny under JSPI instead of polling loop` or later, working tree clean.

- [ ] **Step 2: Daily upstream cadence check**

Run:
```bash
cd /Users/probello/Repos/llama.cpp && git fetch origin
git log webllm-browser-patches..origin/master --oneline -- ggml/src/ggml-webgpu/ ggml/include/
```
If non-empty: pause Phase 2 and apply the §32 rebase procedure first (the rebase may surface free wins or regressions that change the gate baseline). If empty: proceed.

- [ ] **Step 3: Verify legacy decode baseline still green**

Run:
```bash
cd /Users/probello/Repos/webllm && make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
```
Expected: ≥100 tok/s decode (TODO.md pin: 110.8 tok/s; allow ±10% drift). If lower than 100: investigate before starting Phase 2 — the gate baseline must be the current-day legacy number, not the pinned historical number.

- [ ] **Step 4: Capture EM_ASM crossing count for the legacy path**

For T3's "EM_ASM crossings/token" metric, we need a comparable number for the legacy `ModelInference`. Add a one-shot counter:

```bash
cd /Users/probello/Repos/webllm
grep -n "wasm\.op[A-Z]" src/inference/model-inference.ts | wc -l
```
Records the total `wasm.op*` invocation count from the source as a static upper bound. Capture from a 5-token decode trace via the existing `webllm_perf_counter` instrumentation (`smoke-test/real-model.html?profile=1` exposes per-step dispatch counts in the dashboard payload).

- [ ] **Step 5: Write `PRE-PROTOTYPE-BASELINE.md`**

The file MUST contain:
1. **Header**: capture date, webllm git SHA, llama.cpp `webllm-browser-patches` SHA, browser + GPU.
2. **tinyllama Q4_0 5-token decode**: greedy generated tokens (sequence), per-token wall (5-run median), total decode time, total dispatches reported by `webllm_perf_counter`, EM_ASM crossings/token (= dispatches/token; one EM_ASM per dispatch in legacy).
3. **Gate target table**: copy from spec §T3 with concrete numbers filled in for "Legacy baseline" column.

This file is the comparison ground-truth for Task 7's gate decision.

- [ ] **Step 6: Commit baseline**

```bash
cd /Users/probello/Repos/webllm
git add -f eval/reports/p2-v2-prototype-*/PRE-PROTOTYPE-BASELINE.md
git commit -m "docs(p2-v2): pre-prototype baseline — tinyllama Q4_0 5-token decode"
```

---

## Task 1: C++ skeleton — empty `ggml-jsep` backend

**Files:**
- Create: `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp`
- Create: `~/Repos/llama.cpp/ggml/src/ggml-jsep/CMakeLists.txt`
- Modify: `~/Repos/llama.cpp/ggml/CMakeLists.txt` — add jsep subdirectory under `#ifdef GGML_BACKEND_JSEP`
- Modify: `~/Repos/llama.cpp/ggml/src/ggml-backend-reg.cpp` — add jsep registration entry
- Modify: webllm `src/wasm/CMakeLists.txt` — add `-DGGML_BACKEND_JSEP=ON` to a new `wasm-build-jsep` target
- Modify: webllm `Makefile` — add `wasm-build-jsep` target

- [ ] **Step 1: Author `ggml-jsep.cpp` skeleton**

Implement the full vtable surface from spec §D2 with all op-handling stubs:
- `ggml_backend_jsep_buffer_type_i`: `get_name="jsep_buf"`, `alloc_buffer`, `get_alignment=256`, `get_max_size=128 MiB`, `is_host=false`.
- `ggml_backend_jsep_buffer_i`: `get_base` (return `(void*)0x2000` per spec §D1 sentinel), `set_tensor`, `get_tensor`, `memset_tensor`, `clear`, `free_buffer`. All implemented as `EM_ASM_*` to the matching `Module.jsep*` callback.
- `ggml_backend_jsep_device_i`: `supports_op` returns `false` for all ops in this task; `supports_buft` returns true iff `buft == jsep_buft`.
- `ggml_backend_jsep_i`: `get_name="jsep"`, `free`, `synchronize`, `graph_compute` empty (returns `GGML_STATUS_SUCCESS`).
- `ggml_backend_reg_i`: `get_name="jsep"`, `get_device_count=1`, `get_device`, `api_version=2`.

Forward-declare the 8 `Module.jsep*` callbacks (their actual JS bodies arrive in Task 3). For Task 1, the callbacks need only return success-like values (alloc returns a stub handle, sync returns immediately, etc.) so the WASM build links.

- [ ] **Step 2: CMake hook**

`ggml-jsep/CMakeLists.txt`:
```cmake
add_library(ggml-jsep STATIC ggml-jsep.cpp)
target_link_libraries(ggml-jsep PRIVATE ggml-base)
target_compile_definitions(ggml-jsep PRIVATE GGML_BACKEND_JSEP_BUILD)
```

`ggml/CMakeLists.txt` — add the conditional:
```cmake
if (GGML_BACKEND_JSEP)
    add_subdirectory(src/ggml-jsep)
endif()
```

- [ ] **Step 3: Backend registration**

`ggml-backend-reg.cpp` — add jsep alongside existing entries:
```cpp
#ifdef GGML_BACKEND_JSEP
    register_backend(ggml_backend_jsep_reg());
#endif
```

- [ ] **Step 4: webllm CMake + Makefile hooks**

`src/wasm/CMakeLists.txt` — accept a top-level `-DWEBLLM_BACKEND=jsep` flag; when set, add `-DGGML_BACKEND_JSEP=ON` to the llama.cpp configure step and append `-jsep` to the output artifact basename.

`Makefile` — `wasm-build-jsep` target invokes `cmake -DWEBLLM_BACKEND=jsep` then build, outputs `dist/webllm-wasm-jsep.{js,wasm}`. Default `wasm-build` unchanged.

- [ ] **Step 5: Commit Task 1**

```bash
cd /Users/probello/Repos/llama.cpp
git add ggml/src/ggml-jsep/ ggml/CMakeLists.txt ggml/src/ggml-backend-reg.cpp
git commit -m "ggml-jsep: skeleton backend (no ops yet) — Phase 2 Task 1"

cd /Users/probello/Repos/webllm
git add src/wasm/CMakeLists.txt Makefile
git commit -m "build(jsep): wasm-build-jsep target gated on -DWEBLLM_BACKEND=jsep"
```

- [ ] **Step 6: Verification**

```bash
cd /Users/probello/Repos/webllm
make wasm-build-jsep 2>&1 | tail -20
ls -la dist/webllm-wasm-jsep.* | head -3
```
Expected: build succeeds; `webllm-wasm-jsep.js` + `webllm-wasm-jsep.wasm` exist; both are within ±15% size of the default artifact (no ops added yet, so size delta is small).

Smoke check: load the jsep WASM into a stub harness (`bun run -e 'const m = await import("./dist/webllm-wasm-jsep.js"); console.log(typeof m.default)'`) — must not throw. (Will throw on `Module.jsepAlloc undefined` if any backend init is triggered, which is fine; we just need to confirm linking succeeded.)

---

## Task 2: C++ op dispatch — MUL_MAT + RMS_NORM `supports_op` + `graph_compute`

**Files:**
- Modify: `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp`

- [ ] **Step 1: Flip `supports_op` for the two op kinds**

```cpp
static bool ggml_backend_jsep_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Cap check: per-binding 128 MiB doctrine (CLAUDE.md).
    // Reject any op where any tensor exceeds the cap.
    const size_t cap = 128 * 1024 * 1024;
    if (ggml_nbytes(op) > cap) return false;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (op->src[i] && ggml_nbytes(op->src[i]) > cap) return false;
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            // Phase 2 dtype matrix: src1=F32, src0 ∈ {F32, F16, Q4_0, Q4_K}.
            if (op->src[1]->type != GGML_TYPE_F32) return false;
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_K:
                    return op->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        }
        case GGML_OP_RMS_NORM: {
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        }
        default:
            return false;
    }
}
```

- [ ] **Step 2: Implement `graph_compute`**

```cpp
static ggml_status ggml_backend_jsep_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_backend_jsep_device_supports_op(/* dev */, node)) {
            // Should not happen: scheduler peels unsupported ops to CPU before reaching us.
            return GGML_STATUS_FAILED;
        }

        // Pack source handles as length-prefixed int32_t array on the C stack.
        int32_t src_handles[1 + GGML_MAX_SRC];
        int32_t n_src = 0;
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            src_handles[1 + s] = (int32_t)(intptr_t)node->src[s]->data;
            n_src++;
        }
        src_handles[0] = n_src;

        int32_t dst_handle = (int32_t)(intptr_t)node->data;

        int status = EM_ASM_INT({
            return Module.jsepRunOp($0, $1, $2, $3, $4);
        }, (int)node->op, (int)src_handles, dst_handle, (int)node->op_params, GGML_MAX_OP_PARAMS);

        if (status != 0) return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}
```

- [ ] **Step 3: Verification**

Build + run a probe binary (or the existing webllm WASM harness) that constructs a 32×32 F32×F32 `ggml_mul_mat` graph and a separate 64-elem F32 `ggml_rms_norm` graph. Each must trigger exactly one `Module.jsepRunOp` call with the matching opcode. Use a stub `Module.jsepRunOp` that logs `(op, n_src, dst)` and returns 0.

Expected log:
```
jsepRunOp op=GGML_OP_MUL_MAT n_src=2 dst=<handle>
jsepRunOp op=GGML_OP_RMS_NORM n_src=2 dst=<handle>   (src[1] = weight tensor)
```

For non-matmul/non-rms_norm ops, `supports_op` returns false → scheduler routes them elsewhere → `jsepRunOp` is **not** called. Verify by building a 5-op graph (matmul + add + rms_norm + softmax + matmul) and confirming exactly 2 `jsepRunOp` calls (the two matmuls); the softmax-routed CPU backend handles the others.

- [ ] **Step 4: Commit Task 2**

```bash
cd /Users/probello/Repos/llama.cpp
git add ggml/src/ggml-jsep/ggml-jsep.cpp
git commit --amend  # extend the Task 1 patch
```
**Important:** Phase 2's patch budget is 1 patch on `webllm-browser-patches`. Amend Task 1's commit rather than creating a new one. This keeps the patch stack at +1.

---

## Task 3: TS jsep runtime scaffold — buffer paths only

**Files:**
- Create: `src/inference/jsep/index.ts`
- Create: `src/inference/jsep/gpu-data-manager.ts`
- Create: `src/inference/jsep/command-encoder.ts`
- Create: `src/inference/jsep/pipeline-cache.ts`
- Create: `tests/jsep-buffer-roundtrip.test.ts`

- [ ] **Step 1: GpuDataManager**

`src/inference/jsep/gpu-data-manager.ts`:
- Class `GpuDataManager(device: GPUDevice)`.
- `alloc(size: number): number` — pulls from bucketed free-list (size classes 1KB / 4KB / 16KB / 64KB / 256KB / 1MB / 4MB / 16MB / 64MB / 128MB) or creates a new `GPUBuffer` with `usage: STORAGE | COPY_SRC | COPY_DST`. Returns an integer handle (incrementing counter).
- `free(handle: number): void` — returns to bucket.
- `get(handle: number): { buffer: GPUBuffer, size: number }` — throws on invalid handle.
- `write(handle, offset, hostPtr: number, size: number, wasmHeap: ArrayBuffer)` — `device.queue.writeBuffer` from a `Uint8Array` view of the WASM heap.
- `readAsync(handle, offset, hostPtr, size, wasmHeap): Promise<void>` — `commandEncoder.copyBufferToBuffer` to a staging buffer + `mapAsync`. Resolves under JSPI/ASYNCIFY.

- [ ] **Step 2: CommandEncoderBatcher**

`src/inference/jsep/command-encoder.ts`:
- Class `CommandEncoderBatcher(device, options?: { maxDispatch?: number })`.
- `record({ pipeline, bindGroup, dispatchX, dispatchY, dispatchZ })` — opens `passEncoder` lazily; dispatches.
- `flush()` — `passEncoder.end()` + `device.queue.submit([encoder.finish()])`; resets state.
- Auto-flush when `pendingDispatchCount >= maxDispatch` (default 16). Mirrors ORT-Web's flush threshold (`backend-webgpu.ts:200`).

- [ ] **Step 3: PipelineCache**

`src/inference/jsep/pipeline-cache.ts`:
- Class `PipelineCache(device)`.
- `getOrCreate(key: string, builder: () => GPUComputePipeline): GPUComputePipeline` — memoize.

- [ ] **Step 4: index.ts — installJsepCallbacks**

`src/inference/jsep/index.ts`:
- `installJsepCallbacks(module, device)`:
  - Construct `GpuDataManager`, `CommandEncoderBatcher`, `PipelineCache`.
  - Register on `module`:
    - `module.jsepAlloc = (size) => mgr.alloc(size)`
    - `module.jsepFree = (h) => mgr.free(h)`
    - `module.jsepWrite = (h, off, ptr, size) => mgr.write(h, off, ptr, size, module.HEAPU8.buffer)`
    - `module.jsepRead = (h, off, ptr, size) => mgr.readAsync(h, off, ptr, size, module.HEAPU8.buffer)` (note: returns a Promise; ASYNCIFY/JSPI handles the suspend)
    - `module.jsepClear = (h, value, off, size) => mgr.clear(h, value, off, size)` (small helper that records a `fillBuffer` or zero-write)
    - `module.jsepRunOp = (op, srcHandlesPtr, dstHandle, opParamsPtr, paramsLen) => { /* dispatch table — populated in Tasks 4 + 5 */ return STATUS_NOT_IMPLEMENTED; }`
    - `module.jsepSync = () => batcher.flush()`
- Constants: `STATUS_OK = 0`, `STATUS_NOT_IMPLEMENTED = 1`, `STATUS_FAILED = -1`.

- [ ] **Step 5: Unit test — buffer roundtrip**

`tests/jsep-buffer-roundtrip.test.ts`: against a real `GPUDevice` (skip with `!HAS_WEBGPU` guard), exercise alloc → write → clear → write again → read; compare round-tripped Float32Array against original within bit-exact equality. ~80 LOC.

- [ ] **Step 6: Verification**

```bash
cd /Users/probello/Repos/webllm
bun test tests/jsep-buffer-roundtrip.test.ts
```
Expected: passes (or skips gracefully if `!HAS_WEBGPU` in Bun environment — browser smoke covers WebGPU paths).

Make sure `bun build src/inference/jsep/index.ts` produces a working bundle. `make checkall` should still pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/inference/jsep/ tests/jsep-buffer-roundtrip.test.ts
git commit -m "feat(jsep): TS runtime scaffold — buffer paths, no ops yet"
```

---

## Task 4: TS matmul kernel

**Files:**
- Create: `src/inference/jsep/ops/matmul.ts`
- Create: `tests/jsep-matmul-golden.test.ts`
- Modify: `src/inference/jsep/index.ts` — wire matmul into `jsepRunOp` dispatch

- [ ] **Step 1: Port matmul WGSL from `model-inference.ts`**

`src/inference/jsep/ops/matmul.ts`:
- Export `dispatchMatmul(ctx, op, srcs, dst, opParams)`.
- Identify dtype combo from src/dst types (passed via op metadata or extracted from descriptor).
- Build cache key: `mat-{src0_dtype}-{src1_dtype}-{dst_dtype}-{shape-rank}`.
- WGSL string per dtype combo:
  - F32×F32 → F32 (canonical baseline)
  - F16×F16 → F32 (Q8_0/Q4_0-equivalent path; legacy default for tinyllama Q4_0)
  - Q4_0×F32 → F32 (with on-the-fly dequant; weights stored as Q4_0 blocks of 32 elems)
  - Q4_K×F32 → F32 (super-block dequant; weights stored in Q4_K format)
- Each WGSL string adapted line-for-line from `model-inference.ts` matmul WGSL near `opMulMat`. Workgroup tile size 16×16.

- [ ] **Step 2: Wire into `jsepRunOp` dispatch**

`src/inference/jsep/index.ts`:
```typescript
const GGML_OP_MUL_MAT = /* numeric value from ggml.h */;
module.jsepRunOp = (op, srcHandlesPtr, dstHandle, opParamsPtr, paramsLen) => {
    const srcs = readSrcHandles(module, srcHandlesPtr); // returns int32_t[]
    if (op === GGML_OP_MUL_MAT) {
        return dispatchMatmul(ctx, op, srcs, dstHandle, opParamsPtr) ? STATUS_OK : STATUS_FAILED;
    }
    return STATUS_NOT_IMPLEMENTED;
};
```
(Note: also need shape + dtype info per src/dst — these come from a sidecar metadata block populated by C++ when calling `jsepRunOp`. May require a small C++-side amendment to pack metadata into `opParamsPtr` or a separate parameter.)

- [ ] **Step 3: T1 golden test — matmul**

`tests/jsep-matmul-golden.test.ts`:
- F32×F32, square (32×32 × 32×32). Expected: matches Float32Array reference.
- F16×F16, tall+thin (128×16 × 16×64). Expected: matches reference.
- Q4_0×F32, square (32×32 × 32×32). Q4_0 inputs hand-packed; reference matmul is performed on dequantized values.
- Tolerance: `||delta||_∞ ≤ 1e-4` for F32 inputs, `≤ 1e-2` for quantized (dequant introduces small per-block error).

- [ ] **Step 4: Verification**

```bash
bun test tests/jsep-matmul-golden.test.ts
```
Expected: 3 cases pass. If Q4_0 fails, debug the dequant kernel before moving on (most likely cause of subtle wrong-results in Phase 3).

- [ ] **Step 5: Commit Task 4**

```bash
git add src/inference/jsep/ops/matmul.ts src/inference/jsep/index.ts tests/jsep-matmul-golden.test.ts
git commit -m "feat(jsep): matmul kernel — F32, F16, Q4_0, Q4_K dtype paths"
```

---

## Task 5: TS rms_norm kernel

**Files:**
- Create: `src/inference/jsep/ops/rms-norm.ts`
- Create: `tests/jsep-rms-norm-golden.test.ts`
- Modify: `src/inference/jsep/index.ts` — wire rms_norm into `jsepRunOp` dispatch

- [ ] **Step 1: Port rms_norm WGSL from `model-inference.ts`**

`src/inference/jsep/ops/rms-norm.ts`:
- Export `dispatchRmsNorm(ctx, op, srcs, dst, opParams)`.
- Cache key: `rms-norm-f32-{lastDim}` (last-dim governs unrolling).
- WGSL: per-row reduction `sum(x[i]²) / N`, multiply by `1 / sqrt(mean + eps)`, scale by weight (`srcs[1]`). Single dispatch per row of input.
- `op_params` field 0 = eps (f32). Read via `module.HEAPF32[opParamsPtr / 4]`.

- [ ] **Step 2: Wire into dispatch**

`src/inference/jsep/index.ts`:
```typescript
const GGML_OP_RMS_NORM = /* numeric value from ggml.h */;
module.jsepRunOp = (op, ...) => {
    if (op === GGML_OP_MUL_MAT) return dispatchMatmul(...) ? STATUS_OK : STATUS_FAILED;
    if (op === GGML_OP_RMS_NORM) return dispatchRmsNorm(...) ? STATUS_OK : STATUS_FAILED;
    return STATUS_NOT_IMPLEMENTED;
};
```

- [ ] **Step 3: T1 golden test — rms_norm**

`tests/jsep-rms-norm-golden.test.ts`:
- Typical attention width: input (1×2048), weight (2048,), eps 1e-5.
- Small width: input (1×64), weight (64,), eps 1e-6.
- Reference: `out = (x / sqrt(mean(x²) + eps)) * weight`.
- Tolerance: `||delta||_∞ ≤ 1e-4`.

- [ ] **Step 4: Verification**

```bash
bun test tests/jsep-rms-norm-golden.test.ts
```
Expected: 2 cases pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/inference/jsep/ops/rms-norm.ts src/inference/jsep/index.ts tests/jsep-rms-norm-golden.test.ts
git commit -m "feat(jsep): rms_norm kernel — F32 path"
```

---

## Task 6: Engine integration + bundle wiring

**Files:**
- Modify: `src/core/engine.ts` — accept `backend?: "default" | "jsep"` in init
- Modify: `package.json` — add `build:jsep` script
- Modify: `src/index.ts` — export `JsepBackend` config type
- Modify: `Makefile` — extend `wasm-build-jsep` to also produce the bundle

- [ ] **Step 1: Engine init plumbing**

`src/core/engine.ts`:
- Add `backend?: "default" | "jsep"` to `WebLLMConfig`.
- In init: branch on `config.backend === "jsep"`:
  - Load `webllm-wasm-jsep.{js,wasm}` instead of standard pair.
  - Construct GPUDevice (existing path), then `installJsepCallbacks(module, device)` before `webllm_load_model`.
- Default branch unchanged.

- [ ] **Step 2: Bundle build**

`package.json`:
```json
{
  "scripts": {
    "build:jsep": "bun build src/index-jsep.ts --outdir dist --target browser --format esm"
  }
}
```
`src/index-jsep.ts` is a thin shim that re-exports from `src/index.ts` plus eagerly imports `src/inference/jsep/index.ts` so the JSEP runtime lands in the bundle.

`Makefile` — `wasm-build-jsep` target ends with `bun run build:jsep`.

- [ ] **Step 3: Verification**

```bash
make wasm-build-jsep
make checkall  # default build still green
ls -la dist/webllm-{wasm,bundle}-jsep.* | head -5
```
Expected: jsep artifacts present, default `make checkall` green (no regression to legacy path).

Browser smoke:
```bash
make smoke-serve &
# In agentchrome: navigate to http://localhost:8031/?backend=jsep&model=tinyllama-1.1b-chat-q4_0
```
Page loads without console errors. (Decode may not work yet because RoPE/SwiGLU/etc. need CPU fallback support — verify the load + tokenizer init path runs clean; actual decode is Task 7.)

- [ ] **Step 4: Commit Task 6**

```bash
git add src/core/engine.ts src/index.ts src/index-jsep.ts package.json Makefile
git commit -m "feat(jsep): engine.init({ backend: 'jsep' }) opt-in + bundle wiring"
```

---

## Task 7: End-to-end smoke + gate report

**Files:**
- Create: `eval/reports/p2-v2-prototype-<CLOSE_DATE>/SUMMARY.md`
- Modify: `TODO.md` — closure stub
- Modify (possibly): `smoke-test/real-model.html` — add `?backend=jsep` query parameter handling

- [ ] **Step 1: 5-token tinyllama greedy decode through jsep**

Browser: navigate to `http://localhost:8031/?backend=jsep&model=tinyllama-1.1b-chat-q4_0&prompt=Hello&max=5&greedy=1&profile=1`.

Capture from `#log` and `webllm_perf_counter` payload:
1. Generated tokens (sequence).
2. Per-token wall-clock (median over 5 tokens).
3. Total decode time.
4. Total `Module.jsepRunOp` invocation count (from a counter wired in Task 3).
5. CPU-fallback `set_tensor` + `get_tensor` invocation counts.
6. Total EM_ASM crossings/token.

- [ ] **Step 2: Compare to legacy baseline (from Task 0)**

Cross-check tokens: must be byte-identical (greedy → deterministic). If divergent: prototype has a correctness bug (WGSL kernel, dequant, or descriptor encoding); fix before declaring gate.

- [ ] **Step 3: Gate decision**

Apply the gate from spec §T3:

| Metric                | Result | Band |
|-----------------------|--------|------|
| Per-token wall        | <fill> | green / yellow / red |
| EM_ASM crossings/token | <fill> | green / yellow / red |
| Greedy 5/5 token equality | <pass/fail> | required |

- **Green:** Phase 3 plan-write begins next session.
- **Yellow:** investigate the dominant cost (likely candidate: per-node EM_ASM cost itself). The graph-once dispatch lever is the pre-baked yellow-recovery path — propose a separate micro-cycle to test it before Phase 3.
- **Red:** stop. The JSEP architecture is incompatible with our scheduler shape under the current toolchain. Consider Tier 2 partial migration or re-evaluate.

- [ ] **Step 4: Write `SUMMARY.md`**

Required structure:
1. **Header**: capture date, commits used, browser + GPU.
2. **Token output**: sequence of 5 tokens, comparison to legacy.
3. **Metrics table**: matches the gate table above with concrete numbers.
4. **Gate disposition**: green / yellow / red with one-paragraph rationale.
5. **Per-task commit map**: list the 7 commits (Task 0 baseline → Task 7 closure).
6. **Open questions surfaced during prototype**: anything that came up that should feed the Phase 3 plan-write.

- [ ] **Step 5: Update TODO.md**

Replace the "Tier 3 migration to upstream `llama_decode` (REDIRECTED 2026-05-05)" section's "Next-session quickstart for P2-v2" subsection with a closure stub:
```
Phase 2 prototype CLOSED <CLOSE_DATE> — gate <green/yellow/red>.
- Spec: docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md
- Plan: docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md
- Closure: eval/reports/p2-v2-prototype-<CLOSE_DATE>/SUMMARY.md
- Disposition: <one sentence per gate disposition — Phase 3 plan-write begins / yellow-recovery cycle queued / Tier 3 redirect>
```

Add the seven commits' SHAs in the commit map.

- [ ] **Step 6: Final commit**

```bash
git add -f eval/reports/p2-v2-prototype-*/SUMMARY.md TODO.md
git commit -m "docs(p2-v2): close Phase 2 prototype — gate <COLOR>"
```

---

## Verification gate (Phase 2 closes only when all hold)

1. `bun test tests/jsep-buffer-roundtrip.test.ts tests/jsep-matmul-golden.test.ts tests/jsep-rms-norm-golden.test.ts` — all green.
2. `make checkall` — green (default-backend path unchanged).
3. `make wasm-build-jsep` — succeeds; produces `dist/webllm-{wasm,bundle}-jsep.*`.
4. Browser 5-token tinyllama greedy decode through jsep produces byte-identical output to legacy.
5. T3 gate metrics in `eval/reports/p2-v2-prototype-<CLOSE_DATE>/SUMMARY.md` are green or yellow band (red invalidates the thesis).
6. `TODO.md` carries a closure stub with the gate disposition.

## Risk register (mirrored from spec §Risks)

R1 ASYNCIFY/JSPI for `get_tensor` async readback — reuse existing `webllm-browser-patches` async-readback bundle (`846e0685e`, `702d40ee9`, `55fba3670`, `ff362d4ae`).
R2 Per-EM_ASM cost too high — pre-baked yellow-recovery via graph-once dispatch in Phase 3 spec/plan.
R3 Bundle-size regression — separate jsep entry point so legacy users aren't taxed.
R4 Pipeline-cache correctness — T1 goldens cover ≥3 dtype combos for matmul + 2 shape regimes for rms_norm; cache keys include all dtype + shape-rank components.
R5 Scheduler thrash from CPU-fallback — expected; gate is on per-EM_ASM rate, not absolute tok/s.

## Patch budget tracking

Band B has 3 reserved. Phase 2 uses **1** (the ggml-jsep skeleton patch on `webllm-browser-patches`, amended through Task 1 + Task 2 to keep stack +1). Remaining: 2 for Phase 3.
