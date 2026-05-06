# P2-v2 JSEP-style backend — Phase 2 prototype design

Status: proposed, 2026-05-05
Scope: TODO.md item "Tier 3 migration to upstream `llama_decode`
(REDIRECTED 2026-05-05)" — Phase 2 single-op prototype.
Trigger: P2 v1 measured an 18× decode regression from per-WebGPU-API-call
shim crossings under `emdawnwebgpu` (`eval/reports/p2-causal-migration-2026-05-05/POST-MIGRATION-BENCH.md`).
Phase 1 research probe closed 2026-05-05 (commits `ec18120` /
`cec7172` / `b6d807a`).

## Problem

P2 v1 routed causal-LM through `ggml-webgpu` compiled inside WASM. The
`ggml-webgpu` backend issues `wgpu` API calls from C++; under
Emscripten + the `emdawnwebgpu` Dawn-on-WebGPU port each call becomes a
synchronous JS shim crossing. For a tinyllama-Q4_0 single-token decode,
that's hundreds of crossings per layer × 22 layers per token. Path A
investigation (`commits c8e1dc6 + fe167aa`) ruled out graph-cache misses,
JSPI polling, and end-of-graph waits — the cost is intrinsic to running
`ggml-webgpu` inside WASM under the current toolchain. The fix has to be
architectural.

The redirect: a new ggml backend (`ggml-jsep`) where all WebGPU recording
happens in JS. ggml's tensor allocator + graph builder + scheduler stay
in WASM; the new backend's `graph_compute` walks the cgraph in C++ and
emits exactly **one EM_ASM per node** (or per graph, eventually) into a
JS-side dispatcher that records WebGPU compute passes and submits them.
Pattern modeled on ORT-Web's JSEP (`microsoft/onnxruntime`,
`js/web/lib/wasm/jsep/`); see Phase 1 research at
`eval/reports/p2-v2-jsep-research-2026-05-05/`.

This document specifies **Phase 2 only**: a two-op (matmul + RMS_NORM)
stub backend that proves the JS↔WASM crossing rate is acceptable, gating
green-light for Phase 3 (full op coverage). RMS_NORM joins matmul because
the two are structurally adjacent in every causal-LM layer (norm-before-
matmul, twice per layer); covering both keeps a contiguous run of jsep
ops together and gives a more realistic gate read than matmul-only would.

## Goals

1. **Validate the architectural thesis.** Per-token JS↔WASM crossing
   count for the prototype must drop by ≥10× vs `ggml-webgpu` under
   `emdawnwebgpu`, and per-token wall-clock must be within 2× of the
   legacy `ModelInference` (`src/inference/model-inference.ts`) baseline.
   Both are measured on tinyllama Q4_0 × 5-token decode.
2. **Minimum viable scope.** Matmul + RMS_NORM only; everything else
   CPU-fallback via ggml's scheduler. No new TS kernels for non-{matmul,
   rms_norm} ops in Phase 2.
3. **Coexistence.** Legacy `ModelInference` stays the default. Phase 2
   ships behind a feature flag (`engine.init({ backend: "jsep" })`); no
   regression risk to current canonical-6 baseline.
4. **Patch frugality.** ≤1 new llama.cpp patch (band B has 3 reserved;
   target 1 used here, leaving 2 for Phase 3+).

## Non-goals (deferred)

- **Graph-once EM_ASM dispatch.** Phase 1.1 §5 noted ggml's
  `graph_compute(cgraph)` lets us serialize the entire graph and cross
  to JS once, which JSEP cannot. Phase 2 stays per-node to mirror JSEP
  exactly; graph-once is a Phase 3 lever measured *after* the per-node
  baseline.
- **Other op kernels.** Phase 1.3 catalogued ~40 TS kernels needed for
  Phase 3; only matmul + RMS_NORM ship in Phase 2. ROPE, SwiGLU, ADD,
  SOFT_MAX, FLASH_ATTN_EXT, etc. all stay on CPU-fallback.
- **Encoder / embedder paths.** Phase 2 is causal-LM-only; encoder
  + embedder migrate as P3-encoder / P3-embedder cycles.
- **L2_NORM for Qwen3.** Phase 1.3 §6 flagged this as a Phase 3
  newcomer; not load-bearing for tinyllama prototype.
- **Worker-mode + dual-mode wiring.** Phase 2 runs main-thread only;
  worker integration falls out of Phase 3.

## Architectural decisions

### D1 — Per-node EM_ASM (locked for Phase 2)

The `ggml-jsep` `graph_compute(cgraph)` impl walks `cgraph->nodes`
linearly. For each `MUL_MAT` node it emits one `EM_ASM_INT` call into
JS:

```cpp
// in ggml-jsep.cpp
EM_ASM_INT({
    return Module.jsepRunOp($0, $1, $2, $3, $4);
}, op_code, src_handles_ptr, dst_handle, op_params_ptr, params_len);
```

For all other op kinds, `supports_op` returns `false` and the scheduler
peels the node to a CPU backend (Phase 1.2 §3). Per-token expected
crossing count for tinyllama Q4_0:

| Op kind                                | Count/layer | Lives on | Crossings/layer |
|----------------------------------------|-------------|----------|-----------------|
| `MUL_MAT` (Q/K/V/O + gate/up/down + attn QK + attn·V) | 9 | jsep   | 9 EM_ASM        |
| `RMS_NORM` (attn_norm + ffn_norm)      | 2           | jsep     | 2 EM_ASM        |
| RoPE, SwiGLU split, residuals, attn softmax/mask | ~12-16 | CPU      | ~16-24 set_tensor/get_tensor (transitions, not per-op) |
| **Per-layer total**                    |             |          | **~27-35**      |

× 22 layers + 2 for embedding lookup + final norm + sampling tail
≈ **600-800 crossings/token** for the prototype. Adding RMS_NORM to jsep
fuses each (norm → matmul) pair into one contiguous jsep run, which is
why crossings drop ~25% vs matmul-only despite only adding 2 ops/layer.
Comparison:

- Legacy `ModelInference`: ~25 dispatches × 22 layers + 8 tails =
  **~558 dispatches/token** (Phase 1.3 §1; one EM_ASM per dispatch).
- `ggml-webgpu` under `emdawnwebgpu`: per-`wgpu`-API-call crossings, not
  per-op — empirically dominates decode time at the 18× regression
  measured in P2 v1.

Phase 2 prototype is expected to land **roughly comparable to legacy
crossing count** (~1.1-1.5× more) and **dramatically below `ggml-webgpu`
crossing rate**. That's the right shape — the prototype is still
handicapped by CPU-fallback `set_tensor`/`get_tensor` traffic at
RoPE/SwiGLU/attention boundaries that disappears when Phase 3 ships the
rest of the kernels jsep-side, but RMS_NORM-on-jsep already kills the
norm-boundary traffic that would otherwise dominate at small models.

### D2 — Callback table shape

JSEP's 11-entry callback table (Phase 1.1 §1) trims to **8 entries** for
ggml's surface (no per-kernel registration phase since ggml's
`op_params` travels in the cgraph itself; no `JsepOutput` analogue
because outputs are already pre-allocated by the scheduler):

| Callback                                         | Purpose                                       | EM_ASM call site            |
|--------------------------------------------------|-----------------------------------------------|-----------------------------|
| `jsepInit(callbacks)`                            | one-time registration                         | first `ggml_backend_jsep_init` |
| `jsepAlloc(size: number) → handle: number`      | `ggml_backend_buffer_type_i.alloc_buffer`     | per buffer alloc            |
| `jsepFree(handle: number)`                       | `ggml_backend_buffer_i.free_buffer`           | per buffer free             |
| `jsepWrite(handle, offset, hostPtr, size)`      | `ggml_backend_buffer_i.set_tensor`            | host → GPU upload           |
| `jsepRead(handle, offset, hostPtr, size)` async | `ggml_backend_buffer_i.get_tensor` (ASYNCIFY/JSPI) | GPU → host readback     |
| `jsepClear(handle, value, offset, size)`        | `ggml_backend_buffer_i.{memset_tensor,clear}` | KV-cache zero               |
| `jsepRunOp(op, srcHandles, dstHandle, opParams, paramsLen) → status` | `graph_compute` per node               | per `MUL_MAT` node          |
| `jsepSync()`                                     | `ggml_backend_i.synchronize`                  | end-of-graph or readback    |

`srcHandles` is a pointer to a length-prefixed `int32_t[]` (typically
`[count, h_src0, h_src1, ...]`) in WASM linear memory. `opParams` is a
pointer to `tensor->op_params` raw bytes (currently unused for matmul,
but reserved for future ops).

### D3 — JS-side module layout

```
src/inference/jsep/
  index.ts             — public init() that registers callbacks via Module
  gpu-data-manager.ts  — handle → {GPUBuffer, size} map; bucketed reuse
  command-encoder.ts   — batcher (one open encoder; flush on N or sync)
  pipeline-cache.ts    — Map<key, GPUComputePipeline>; key = (op, dtype, shape-deps)
  ops/
    matmul.ts          — WGSL kernel + dispatch logic (Phase 2)
    rms-norm.ts        — WGSL kernel + dispatch logic (Phase 2)
```

Patterned on ORT-Web's `js/web/lib/wasm/jsep/{init.ts, backend-webgpu.ts,
webgpu/{program-manager.ts, gpu-data-manager.ts, ops/}}`. Both kernels
are **adapted from existing `model-inference.ts` WGSL** (matmul near
`opMulMat`, RMS_NORM near `opRmsNorm`); same dequant + workgroup
layout, just rebuilt around the JSEP descriptor shape.

The `jsep` runtime initializes lazily on the first
`engine.init({ backend: "jsep" })` and registers itself into the
WASM `Module` global.

### D4 — Bundle layout

A second WASM artifact `dist/webllm-wasm-jsep.{js,wasm}` ships alongside
the existing `dist/webllm-wasm.{js,wasm}`. The legacy bundle is
unchanged; the jsep bundle includes:

- llama.cpp built with `-DGGML_BACKEND_JSEP=ON` (new CMake flag)
- The jsep TS module (`src/inference/jsep/**`) bundled into
  `dist/webllm-bundle-jsep.js`
- All other engine code (`src/engine/**`, `src/inference/tokenizer.ts`,
  etc.) shared with the legacy build via a `bun build` shared-imports
  strategy

Phase 3 collapses the two artifacts back into one once jsep is the
default.

### D5 — Engine integration

`engine.init({ backend?: "default" | "jsep" })`. New optional field;
default `"default"` preserves current behavior. When `"jsep"`, the
engine:

1. Loads `webllm-wasm-jsep.{js,wasm}` instead of the standard pair.
2. Calls `installJsepCallbacks(Module)` from
   `src/inference/jsep/index.ts` before any `webllm_load_model` call.
3. Skips the legacy `ModelInference` graph-build path entirely;
   `engine.generate` flows through `webllm_decode` (the upstream
   `llama_decode` wrapper that already shipped via P1 + P2 v1 path-A
   surface, retained across the revert).

The C++ side discovers the registered backend via
`ggml_backend_load_all`-style registration; the scheduler picks it up
automatically when an `MUL_MAT`-bearing graph is built.

## Test plan

### T1 — Unit goldens (matmul + rms_norm)

`tests/jsep-matmul-golden.test.ts` + `tests/jsep-rms-norm-golden.test.ts`:

**matmul:**
1. Allocate two F32 buffers via `jsepAlloc`; upload reference tensors
   via `jsepWrite` (e.g. 32×32 × 32×32 = 32×32 result).
2. Invoke `jsepRunOp(GGML_OP_MUL_MAT, [a, b], c, ...)` directly (no
   ggml; pure JS-side test).
3. Read `c` back via `jsepRead`.
4. Compare against numpy / Float32Array reference within
   `||delta||_∞ ≤ 1e-4`.
5. At least 3 dtype combinations (F32×F32, F16×F16, Q4_0×F32) and
   2 shape regimes (square + tall+thin).

**rms_norm:**
1. Allocate F32 buffer for input + F32 buffer for weight; upload.
2. Invoke `jsepRunOp(GGML_OP_RMS_NORM, [x, weight], y, op_params=eps)`.
3. Read `y` back; compare against `(x / sqrt(mean(x²) + eps)) * weight`
   within `||delta||_∞ ≤ 1e-4`.
4. Two shape regimes: typical attention/FFN width (2048, 4096) and
   small (64, 128).

Both run in Bun against a stub Module that exposes the jsep callbacks
installed against a test WebGPU context. **Fast (<1 s each); runs in CI
gate.**

### T2 — End-to-end smoke (5-token tinyllama decode)

Browser smoke at `smoke-test/jsep-matmul-smoke.html` (or extending
`real-model.html` with a `?backend=jsep` query param):

1. Boot `engine.init({ backend: "jsep" })`, load tinyllama Q4_0.
2. Generate 5 tokens greedy from `"Hello"`.
3. Assert tokens are byte-identical to legacy `ModelInference` output
   (greedy → deterministic).
4. Capture metrics: per-token wall, total EM_ASM count (instrumented
   via `Module.jsepRunOp` wrapper that increments a counter),
   `webllm_perf_counter` deltas.

### T3 — Phase 2 → Phase 3 gate

Single canonical run reported in
`eval/reports/p2-v2-prototype-<CLOSE_DATE>/SUMMARY.md`:

| Metric                             | Legacy baseline | Gate (jsep)       | Decision |
|------------------------------------|-----------------|-------------------|----------|
| Per-token wall (5-token median)    | ~9.0 ms         | ≤18 ms (2× legacy) | green    |
|                                    |                 | 18-45 ms (2-5×)   | yellow   |
|                                    |                 | >45 ms            | red      |
| EM_ASM crossings/token             | n/a (one bridge call/dispatch) | <1500           | green    |
|                                    |                 | 1500-4000         | yellow   |
|                                    |                 | >4000             | red      |
| Greedy token equality (5/5)        | reference       | byte-identical    | required |

- **Green:** Phase 3 plan written; full op port begins.
- **Yellow:** investigate before plan-write. Likely fix: switch from
  per-node to graph-once EM_ASM (pre-baked Phase 3 lever).
- **Red:** re-evaluate the JSEP thesis. The architecture itself may be
  incompatible with our scheduler shape; consider Tier 2 (partial
  migration) or fall back to legacy + spot-fixes for new architectures.

## Phase 2 task breakdown

Seven sequential tasks (see companion plan
`docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md`):

1. **C++ skeleton** — `ggml/src/ggml-jsep/ggml-jsep.cpp` with vtables
   wired, all `supports_op` returning `false`, `graph_compute` empty.
   CMake + emcmake hook gated on `-DGGML_BACKEND_JSEP=ON`. Patch lands
   on `webllm-browser-patches`. Verify: `make wasm-build` (the jsep
   variant) succeeds; loading the resulting WASM into a stub host that
   never invokes any op runs without crashing.
2. **C++ op dispatch + JS callback registration** — wire `jsepInit`
   callback table; flip `supports_op` to true for `(MUL_MAT, src1=F32,
   src0 ∈ {F32, F16, Q4_0, Q4_K})` and `(RMS_NORM, src=F32, dst=F32)`;
   emit one EM_ASM per qualifying node. Verify: `supports_op` reports
   yes for matmul + rms_norm, no for everything else; a probe binary
   that builds a 32×32 matmul cgraph triggers exactly one
   `Module.jsepRunOp` call; same probe for an RMS_NORM cgraph triggers
   exactly one call with `op=GGML_OP_RMS_NORM`.
3. **TS jsep runtime scaffold** — `src/inference/jsep/{index.ts,
   gpu-data-manager.ts, command-encoder.ts, pipeline-cache.ts}` with no
   ops yet. `installJsepCallbacks(Module)` registers all 8 callbacks;
   alloc/free/write/read/clear paths exercised by a unit test that
   round-trips a Float32Array. Verify: T1-style unit test for
   buffer-only paths passes (no matmul/rms_norm yet — `jsepRunOp`
   returns `STATUS_NOT_IMPLEMENTED` for both op codes).
4. **TS matmul kernel** — `src/inference/jsep/ops/matmul.ts` ported
   from `model-inference.ts`'s WGSL. Pipeline cache keyed on
   `(MUL_MAT, src0_dtype, src1_dtype, dst_dtype, shape-rank)`. Verify:
   matmul golden passes for F32/F32, F16/F16, and Q4_0/F32, square +
   tall+thin shapes.
5. **TS rms_norm kernel** — `src/inference/jsep/ops/rms-norm.ts` ported
   from `model-inference.ts`'s WGSL. Pipeline cache keyed on
   `(RMS_NORM, dtype, last-dim-size)`. Verify: rms_norm golden passes
   for typical attention width (2048) and small (64), eps from
   `op_params`.
6. **Engine integration + bundle wiring** — `engine.init({ backend:
   "jsep" })` plumbing; `bun build` produces
   `dist/webllm-wasm-jsep.{js,wasm}` + `dist/webllm-bundle-jsep.js`.
   Verify: `make checkall` green; the jsep bundle loads in a browser
   tab without errors.
7. **End-to-end smoke + gate report** — T2 + T3. Verify: gate decision
   pinned in `eval/reports/p2-v2-prototype-<DATE>/SUMMARY.md`. Update
   `TODO.md` with closure stub + green/yellow/red disposition.

## Patch budget

Band B (3 reserved) → Phase 2 uses **1 patch**:

- `ggml-jsep: skeleton backend with matmul + rms_norm dispatch` — adds
  `ggml/src/ggml-jsep/ggml-jsep.cpp` (~600 LoC) + CMake hook +
  registration line in the backend loader. Single patch on
  `webllm-browser-patches`. The C++ side is op-agnostic (one EM_ASM
  hook handles all `supports_op`-accepted node kinds); adding RMS_NORM
  alongside matmul costs only the `supports_op` case statement and 1
  unit test in step 2. Likely upstreamable to `llama.cpp` as
  `GGML_BACKEND_JSEP` once Phase 3 stabilizes — *not* a goal for Phase 2.

## Risk register

- **R1 — ASYNCIFY for `get_tensor`.** `ggml_backend_buffer_i.get_tensor`
  is sync-signature; `jsepRead` is async (GPU `mapAsync`). The existing
  `webllm-browser-patches` async-readback bundle (`846e0685e`,
  `702d40ee9`, `55fba3670`, `ff362d4ae`) handles exactly this for
  `ggml-webgpu`; the same hooks apply to `ggml-jsep`. *Mitigation:*
  reuse the bundle wholesale; if it doesn't apply cleanly, flag in
  Task 1 verification and escalate.
- **R2 — Per-EM_ASM cost still too high.** If even one EM_ASM per node
  is hot under JSPI, the prototype lands in T3-yellow or T3-red.
  *Mitigation:* the spec already pre-bakes graph-once dispatch as a
  yellow-recovery lever (Phase 3 task 0). If it lands red, escalate to
  re-evaluate JSEP itself.
- **R3 — Bundle-size regression.** Two WASM artifacts roughly double
  the on-disk size for users who download both. *Mitigation:* Phase 2
  ships them as separate entry points; users on default never
  download the jsep bundle. Phase 3 collapses them.
- **R4 — Pipeline-cache correctness.** WGSL kernels for matmul and
  rms_norm have per-shape and per-dtype variants (workgroup-size
  tuning, dequant code paths, last-dim reduction loop unrolling).
  Wrong cache key → silent wrong results. *Mitigation:* T1 goldens
  exercise ≥3 dtype combinations + 2 shape regimes for matmul, ≥2
  shape regimes for rms_norm; cache keys include all dtype + shape-rank
  components.
- **R5 — Scheduler thrash from CPU-fallback.** Every layer's non-matmul
  ops bounce through CPU with `set_tensor` + `get_tensor` host-staging.
  *This is expected* (it's why the gate is on per-EM_ASM rate, not
  absolute tok/s) but if it surfaces a *correctness* issue (e.g. the
  CPU backend rejects an op `ggml-webgpu` would have accepted), the
  prototype can't run at all. *Mitigation:* Task 1 verification
  includes "scheduler successfully builds a tinyllama graph with a
  matmul-only-jsep + cpu-everything-else backend pair"; if CPU fallback
  fails for any op, fall back to the alternate strategy of declaring
  `supports_op = true` for all ops and crashing in the `graph_compute`
  body for non-matmul (gives clearer error).

## Open questions

1. **JSPI vs ASYNCIFY for the WASM build.** The P1 closure used JSPI
   (`b4d4b48`); the existing async-readback patches were written for
   ASYNCIFY. Confirm in Task 1 that the JSPI build still picks up the
   readback bundle correctly. (Likely yes — JSPI is a strict superset
   of ASYNCIFY's suspend semantics.)
2. **Should `jsepInit` register everything or per-op?** Spec assumes
   one-shot registration of the 8-entry table. JSEP also has separate
   per-kernel registration (`jsepCreateKernel`) that's elided here
   because ggml has no equivalent. If a Phase 3 op needs per-instance
   pipeline state (e.g. matmul with pre-baked `int K` constants), we
   may want to revisit; for Phase 2 it's not needed.
3. **What does `webllm_perf_counter` measure for jsep?** It currently
   reads `llama_perf_context()`. With `ggml-jsep` registered, those
   per-graph-compute counters still tick. Confirm the readout
   semantics are still meaningful (Phase 1.2 §6 noted yes; verify
   in Task 6).

## Success criteria (composite)

Phase 2 closes green when **all four** hold:

1. T1 goldens pass for both matmul (F32/F32 + ≥1 quantized path) and
   rms_norm (typical + small width).
2. T2 5-token tinyllama greedy decode produces byte-identical output to
   legacy.
3. T3 gate metrics land in green or yellow band (red invalidates the
   thesis; trigger redirect).
4. `make checkall` green; bundle artifacts ship without breaking the
   default-backend path.
