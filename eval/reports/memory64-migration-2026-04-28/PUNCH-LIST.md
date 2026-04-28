# MEMORY64 migration — call-site punch list

**Date:** 2026-04-28
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Audit script:** [`audit-grep.sh`](./audit-grep.sh) — re-run after each phase to confirm no regression.

## Phase 1 targets (JS-side `_malloc` / `_free` migration)

Replace each `m._malloc` / `m._free` call (or wrapper around it) with
`m._bridge_malloc` / `m._bridge_free`. The two custom exports already
live in both binaries (see `src/wasm/CMakeLists.txt:51`). Wrappers must
normalize BigInt return values from wasm64 to `number` because no single
call site allocates >2 GiB (largest tensor at 30B IQ3_M ≈ 850 MB).

| File:line | Caller | Notes |
|---|---|---|
| `src/inference/ggml-wasm.ts:257` | `GgmlWasm.malloc()` | Single source of truth — every other call routes through here. |
| `src/inference/ggml-wasm.ts:261` | `GgmlWasm.free()` | Mirrors `.malloc()`. |
| `src/inference/ggml-wasm.ts:373,378` | `uploadToTensor` | Internal use of `this.malloc/this.free` — covered by Phase 1 if the wrappers are migrated. |
| `src/inference/ggml-wasm.ts:389,398` | `uploadToTensorChunked` | Same. |
| `src/inference/ggml-wasm.ts:415,424` | `uploadRangeChunked` | **Load-bearing for GGUF streaming.** Heap-grow detachment is already handled. |
| `src/inference/ggml-wasm.ts:434,457` | `beginDownloadFromTensor` | Async readback heap allocation. |
| `src/inference/encoder-inference.ts:476,503,512,522` | encoder forward + parity probe | Each routes through `wasm.malloc/free`. |
| `src/inference/model-inference.ts:675,723,1039,1084,1614,1659,2024,2029` | decode/forward heap scratch | All use `wasm.malloc/free`. |
| `smoke-test/real-model-page.js:306,333,364,386,438` | GGUF streaming loader | **Phase 3 dependency.** The `wasm.malloc(total)` at line 306 allocates the entire model file; for 13B Q4_K_S that's ~7.4 GiB — exceeds 2^31. Phase 1 routes the call through `_bridge_malloc`; Phase 3 confirms the BigInt size argument flows correctly. |
| `smoke-test/mem64-probe.html` (probe only) | already on `bridge_malloc` per §31a | confirm no regression. |
| `tests/ggml-wasm.test.ts:10,11,48,52` | mock `Module` interface declarations | Test fixtures expose `_malloc`/`_free` typed as `(size: number) => number`. Phase 1 must add `_bridge_malloc`/`_bridge_free` to the mock surface (or pivot the wrappers to call `_bridge_*` and update the mock) so the unit tests still drive the wrappers. |

## Phase 2 targets (bridge ABI hardening — `int32_t` → `size_t`)

| File:line | Function | Change | Why |
|---|---|---|---|
| `src/wasm/webgpu-bridge.cpp:59` | `ctx_create(int32_t mem_size)` | `→ size_t` | mem_size already cast to `size_t` internally; signature should match. ggml metadata budget at 30B is ~1 MB — no functional cap, but consistent ABI. |
| `src/wasm/webgpu-bridge.cpp:129` | `tensor_set_data(void*, const void*, int32_t size)` | `→ size_t` | Single-tensor uploads <2 GiB at 30B (largest = embedding ≈ 850 MB); promotion is conservative safety only. |
| `src/wasm/webgpu-bridge.cpp:133` | `tensor_get_data(void*, void*, int32_t size)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:187,192` | `op_view_2d/3d(... int32_t offset)` | `→ size_t` | View offsets within graph allocator buffer; theoretical >2 GiB at very-long prefill but not exercised at typical seq=2048. |
| `src/wasm/webgpu-bridge.cpp:296` | `backend_tensor_set(... int32_t offset, int32_t size)` | `→ size_t` | Tensor-buffer offset+size pairs. |
| `src/wasm/webgpu-bridge.cpp:303-310` | `backend_tensor_set3(... int32_t sz1/sz2/sz3)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:313` | `backend_tensor_get(... int32_t offset, int32_t size)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:317-321` | `backend_tensor_get_async_begin(... int32_t offset, int32_t size)` | `→ size_t` | Body already casts to `size_t`. |
| `src/wasm/webgpu-bridge.cpp:328-329` | `backend_tensor_get_async_finish(... int32_t size)` | `→ size_t` | Body already casts to `size_t`. |
| `src/wasm/webgpu-bridge.cpp:272-273` | `graph_new(int32_t size)` | `→ size_t` | Body already casts to `size_t`; `size` is graph-node count, never near 2 GiB. |

**Net effect:** zero functional change at the ≤30B ceiling (no individual
transfer exceeds 2 GiB), but signatures stop pretending to cap at 2 GiB.
Required for the linker to emit BigInt-marshaling JS shims under wasm64
when those bindings cross >2 GiB single buffers in a future scope
expansion (e.g. concat-batched encoder graph buffers).

## Phase 3 targets (GGUF loader BigInt boundary)

| File:line | Concern | Required change |
|---|---|---|
| `smoke-test/real-model-page.js:306` | `wasm.malloc(total)` where `total` can be >4 GiB | After Phase 1, the JS wrapper calls `_bridge_malloc(BigInt(total))`. Phase 3 verifies that no intermediate `Number` narrowing occurs anywhere in the call chain when `total > 2^31`. |
| `smoke-test/real-model-page.js:318` | `wasm.heapU8.set(value, modelPtr + received)` | `modelPtr` is normalized to `number` (safe — Number can represent up to 2^53). `received` is `number`. The `set` call is safe so long as `modelPtr + received < 2^53`. For 13B at 7.4 GiB, `modelPtr` is ~10^10 well under 2^53 ≈ 9×10^15. |
| `smoke-test/real-model-page.js:341` | `new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len)` | Same — `modelPtr + off < 2^53`. |
| `src/inference/ggml-wasm.ts:415-424` | `uploadRangeChunked` heap-grow detachment | Pre-existing fix re-derives `dataAt(off, end - off)` per chunk after `malloc`. Verify it survives BigInt return values from `_bridge_malloc`. |

**No source-code change required at the GGUF parser layer
(`src/models/gguf-parser.ts`).** It operates on `Uint8Array` views — the
heap pointer never crosses its API boundary. Confirmed by audit
(`audit-gguf.txt` returns no hits in `src/models/`).

## Phase 7 target (>4 GiB validation candidate)

| Candidate | Approx. size | Why |
|---|---:|---|
| `mistral-7b-instruct-v0.3-q5km` | ~5.0 GiB | First step above the wasm32 4 GiB cap; same arch as the canonical 7B Q4_K_S baseline → tightest pre/post comparison. |
| `mistral-13b-instruct-q4ks` | ~7.4 GiB | Plan-target 13B; depends on a coherent Q4_K_S-quant 13B GGUF being available on HF. |
| `llama-3.1-13b-instruct-iq3m` | ~5.4 GiB | Alternative 13B if Mistral-13B Q4_K_S is unavailable. |

Phase 7 picks one based on availability and registers it via
`eval/models.ts` + `eval/smoke-profiles.ts`.

## Audit deltas vs canonical baseline (2026-04-28)

The audit-script outputs were cross-checked against the canonical
baseline tables above. Findings:

- **Line numbers — all confirmed.** Every Phase 1 / Phase 2 / Phase 3
  entry's line number in the canonical baseline matches the live source
  tree on `main` at the start of the migration.
- **Stack-allocator audit filter tightened post-review.** The
  initial `audit-grep.sh` excluded only `node_modules` from the
  stack-section grep, which let the Emscripten-emitted minified glue
  in `src/wasm/build/webllm-wasm.js` and `src/wasm/build-mem64/a.out.js`
  dominate `audit-stack.txt` (~370 KB / 5500 lines, mostly
  `stackAlloc` / `stackSave` / `stackRestore` runtime definitions).
  Code review flagged this as unfit for the script's "re-run after
  each phase as a regression check" purpose. The filter was tightened
  to mirror the JS-section filter (also excludes
  `build/|build-mem64/|webllm-bundle|webllm-wasm`). `audit-stack.txt`
  is now 17 lines of first-party callers only:
  - `src/inference/ggml-wasm.ts:264-282` — wrapper definitions
    (`stackSave`, `stackRestore`, `stackAlloc`, `withStack`).
  - `src/inference/model-inference.ts:2013-2022, 2060-2066` — two
    callers using `withStack`/manual save-restore for 4-byte scalar
    pos/id pointers. **Outside MEMORY64 migration scope** (stack
    alloc is BigInt-clean under wasm64; the runtime handles the
    marshaling automatically).
- **Test-fixture mock (new entry, added):** `tests/ggml-wasm.test.ts`
  declares `_malloc(size: number): number` and `_free(ptr: number):
  void` at lines 10-11 and provides matching mock implementations at
  lines 48-52. The canonical baseline did not list these. Added to
  Phase 1 table as a fixture-update obligation: when the wrappers move
  to `_bridge_malloc`/`_bridge_free`, the test mock must surface those
  exports too (either in addition to or in place of `_malloc`/`_free`)
  or the unit tests will silently exercise stale code.
- **`audit-js.txt` lines 12-13** show
  `src/inference/ggml-wasm.ts:302 (_ctx_free)` and
  `src/inference/ggml-wasm.ts:817 (_backend_buffer_free)`. These are
  matched by the `_free\b` regex but are domain-specific bridge
  exports, not generic allocator calls. **No migration action
  required.** Documented here so a future re-run isn't mistaken for a
  regression.
- **GGUF loader `wasm.malloc`/`wasm.free` count confirmed.** The
  canonical baseline's `306, 333, 364, 386, 438` set is exhaustive —
  `audit-gguf.txt` finds the same five sites and no others.
