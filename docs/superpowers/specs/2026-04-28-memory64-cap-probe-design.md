# MEMORY64 Cap Probe — Design Spec

**Date:** 2026-04-28
**Status:** Spec approved (sections 1–3)
**Plan:** _to be written by `superpowers:writing-plans` after spec review_
**Closure report (post-execution):** `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`

## 1. Context

The webllm project caps targeted models at **30B parameters** (project
constraint set 2026-04-28). The current Emscripten WASM build pins
`-sMAXIMUM_MEMORY=4GB`, which limits in-browser models to ~3.95 GB single
allocations: TinyLlama Q4_0, Qwen3-0.6B/1.7B Q8_0, smollm2-360M/1.7B Q4_0,
qwen2.5-1.5B/3B Q4_0, llama-3.2-3B Q4_0, qwen3-4B Q4_0, mistral-7B Q4_K_S
(3953 MB), llama-3.1-8B IQ3_M (3609 MB), qwen3-8B IQ3_M (~3252 MB).

The bands above 4 GiB that fall **inside** the 30B ceiling — and would
therefore become reachable under MEMORY64 + a higher `MAXIMUM_MEMORY` — are:

- 8B at Q4_K_S/Q4_K_M (4.5–5.5 GB) — possible quality lift over IQ3_M, not
  measured.
- 13B at Q4_K_S (~7–9 GB) — currently entirely unreachable; the biggest
  concrete fleet expansion the ceiling unlocks.
- 30B at IQ3/Q3 (~13–14 GB) — the new ceiling; pushes Chrome's per-tab
  memory hard.

Per project policy (set 2026-04-28):

- **Probe-first is the default.** When a lever's gain is unmeasured, run a
  probe phase before committing.
- **Complexity ≠ implementation time.** Score levers on maintenance
  burden, surface area, risk to load-bearing invariants, reversibility,
  and external-dependency exposure — never on duration.

The load-bearing risk axis for MEMORY64 on this stack is **ASYNCIFY ×
MEMORY64 interaction**. The `webgpu-bridge.cpp` build pins
`-sASYNCIFY_STACK_SIZE=1048576` and the WebGPU device-acquisition path
relies on it. Whether that combo works under wasm64 on current Emscripten +
Chrome is unknown and not derivable from documentation alone.

## 2. Goal

Produce a **measured, decision-grade answer** to one question:

> Is MEMORY64 a viable infrastructure path for the webllm browser-side
> stack at the 30B project ceiling — and if so, what's Chrome's actual
> heap ceiling on the dev box?

The probe is **not** a partial migration. It is a separate, bail-cheap
build target that informs whether to commission a follow-up "P2-class"
full bridge migration spec. If the probe fails, MEMORY64 is closed with
measurement evidence; if it passes, the follow-up spec becomes the next
brainstorm.

## 3. Architecture + scope

### 3.1 Build target

> **Amended 2026-04-28** after a Phase 1 sub-probe found that wasm32 and
> wasm64 object files cannot be linked together
> (`wasm-ld: error: ... wasm32 object file can't be linked in wasm64 mode`).
> The original "thin bridge-only" architecture is structurally impossible.
> The corrected architecture uses **two parallel CMake build directories**
> with the same `src/wasm/CMakeLists.txt` source, gated by a cache option.
> Sub-probe also confirmed `ggml-base` and `ggml-webgpu` compile cleanly
> as wasm64 — the rebuild is feasible, just needs orchestration.

**Two parallel build directories.**

| Build dir | Cache option | Output | Used by |
|---|---|---|---|
| `src/wasm/build/` | `WEBLLM_BUILD_MEM64=OFF` (default) | wasm32 `webllm-wasm.{js,wasm}` + wasm32 ggml archives | `make wasm-build` (live pipeline) |
| `src/wasm/build-mem64/` | `WEBLLM_BUILD_MEM64=ON` + `CMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1` | wasm64 `webllm-wasm-mem64.{js,wasm}` + wasm64 ggml archives | `make mem64-probe` (this probe) |

**`src/wasm/CMakeLists.txt` gating.** A single CMakeLists.txt drives both
build dirs. The new `WEBLLM_BUILD_MEM64` cache option determines which
`add_executable` is created:

- `OFF` (default) → `add_executable(webllm-wasm webgpu-bridge.cpp)` — current behavior, bit-identical to pre-amendment.
- `ON` → `add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)` with `target_link_options` adding the three flag deltas below.

`add_subdirectory(${LLAMA_CPP_DIR}/ggml ...)` runs unconditionally in both
modes; under `ON` mode the global `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS`
include `-sMEMORY64=1`, which propagates to the ggml subdirectory and
produces wasm64 ggml archives.

**Three flag deltas on the mem64 target's `target_link_options`** (in
addition to flags inherited from the wasm32 target template):

- `-sMEMORY64=1` — flips pointer width to 64-bit on the WASM side at link.
- `-sWASM_BIGINT=1` — pointer values cross the JS↔WASM boundary as BigInt
  rather than truncated Number.
- `-sMAXIMUM_MEMORY=16GB` — probes the upper bound of what Chrome will
  grant. If Chrome caps lower, Phase 3 (§4.3) reports the actual ceiling.

The existing `-sMAXIMUM_MEMORY=4GB` from the wasm32 target is replaced
(not augmented) on the mem64 target.

All other link flags inherit from the existing target template unchanged
(`-sALLOW_MEMORY_GROWTH=1`, `-sASYNCIFY_STACK_SIZE=1048576`, `-O3`,
exported-functions list, `-sMODULARIZE=1 -sEXPORT_ES6=1`, etc.).

**Source files.** Zero touch to `webgpu-bridge.cpp` — sub-probe confirmed
it compiles cleanly as wasm64 (uses `size_t` casts at every pointer
arithmetic site). If a wasm64-only mismatch surfaces in a future
toolchain bump, that's a regression to investigate, not a closed
question.

**Output names.** `webllm-wasm-mem64.js` / `webllm-wasm-mem64.wasm`
materialized at `src/wasm/build-mem64/webllm-wasm-mem64.{js,wasm}`.
Copied to `smoke-test/` by the new Make target.

**Bit-identical guarantee.** The existing `webllm-wasm` target, the live
TS bridge, `make checkall`, and the live smoke pipeline are unchanged
because:

1. The wasm32 build dir is independent — never sees `WEBLLM_BUILD_MEM64=ON`.
2. The CMakeLists.txt change is gated by the cache option; under the
   default `OFF` path, the configure produces an identical build graph
   to pre-amendment.
3. The mem64 build dir lives at a separate path; `make wasm-build`
   doesn't traverse it.

The mem64 binary is dead code on `main` until a follow-up promotes it.

### 3.2 Harness

A standalone page at `smoke-test/mem64-probe.html` (~50–100 LOC of HTML +
JS). Self-contained — no import of `webllm-bundle.js`, no SDK dependency,
no shared module state with the live smoke page.

Loads `webllm-wasm-mem64.js` as an ES module, runs four phases (§4),
writes results to a `#log` div and the console. Emits a structured JSON
summary at the end so agentchrome's `runtime.evaluate` can extract it
without screen-scraping:

```js
window.__memory64ProbeResult = {
  phase1: "ok" | "fail: <reason>",
  phase2: "ok" | "fail: <reason>",
  phase3_cap_bytes: <Number>,
  phase3_iterations: <Number>,
  phase4: "ok" | "fail: <reason>",
  emscripten_version: "<string>",
  user_agent: "<string>",
  module_bytes: <Number>,
  init_wall_ms: <Number>,
};
```

### 3.3 Make target

`make mem64-probe` runs:

1. CMake configure of `src/wasm/build-mem64/` with
   `-DWEBLLM_BUILD_MEM64=ON -DCMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1`
   plus the same `-DGGML_WEBGPU=ON` / disabled-backends list as the
   wasm32 build. Idempotent — first run configures, subsequent runs
   skip configure if the cache is fresh.
2. Build via `cmake --build src/wasm/build-mem64 --target webllm-wasm-mem64 --config Release -j`. This pulls ggml-base + ggml-webgpu as wasm64 transitively.
3. Copy `webllm-wasm-mem64.{js,wasm}` from `src/wasm/build-mem64/` to `smoke-test/`.
4. `make smoke-restart` (existing port-8031 target).
5. Echo `Open http://localhost:8031/mem64-probe.html?v=$(date +%s)`.

The user opens that URL in the existing agentchrome session; the harness
self-runs on page load. The user (or a follow-up agentchrome script)
copies `window.__memory64ProbeResult` and the `#log` text into the
closure report.

## 4. Probe phases

Each phase logs its name + outcome to the `#log` div and console. A
failure halts subsequent phases — we are collecting evidence, not
auto-recovering.

### 4.1 Phase 1 — Module load + ASYNCIFY round-trip

1. Load `webllm-wasm-mem64.js` as an ES module.
2. Call `_webgpu_init()`. This routes through
   `ggml_backend_webgpu_init()` and exercises async device-acquisition —
   the asyncify path that depends on `ASYNCIFY_STACK_SIZE=1048576`.
3. Call `_webgpu_shutdown()`.

**This is the load-bearing risk axis.** If ASYNCIFY × MEMORY64 is
incompatible on current Emscripten + Chrome, this is where it surfaces.

**Logged:** module byte size, init return code, init wall time, any
console errors observed during the phase.

### 4.2 Phase 2 — BigInt ABI smoke

Tests that pointer values cross JS↔WASM correctly under wasm64 (i.e.
return as `bigint`, not truncated `number`) and that the heap round-trip
through `HEAPU8` works for BigInt-typed offsets. **Does not** call
`_tensor_set_data` / `_backend_alloc_ctx_tensors` — those entangle with
the WebGPU backend allocation path which is out of scope for this probe
(`ctx_create` runs with `no_alloc=true`, so a tensor returned by
`tensor_new_1d` has `tensor->data == nullptr` until backend allocation).

Sequence:

1. `ctxIdx = _ctx_create(65536)` — establishes a context with 64 KiB of
   metadata budget (plenty for one tensor's metadata under `no_alloc`).
   Returns `int32_t` (stack index), so the value is a regular Number.
2. `tensorPtr = _tensor_new_1d(0 /* GGML_TYPE_F32 */, 4)`. The return
   type is `void*` — under wasm64 this **must come back as `bigint`**.
   Pass criterion: `typeof tensorPtr === "bigint" && tensorPtr > 0n`.
3. `dataPtr = _malloc(16n)`. `_malloc` takes `size_t`, which is 64-bit
   under wasm64 — pass `16n` (BigInt). Pass criterion:
   `typeof dataPtr === "bigint" && dataPtr > 0n`.
4. Write 4 F32 values via `new Float32Array(HEAPU8.buffer,
   Number(dataPtr), 4).set([1, 2, 3, 4])`. Read back via a fresh
   `Float32Array` view at the same offset and compare element-by-element.
5. `_free(dataPtr)`, `_ctx_free()`.

The harness uses BigInt arithmetic only where the value could exceed
2^32 (the cap probe in §4.3); for the 16-byte heap round-trip, narrowing
to `Number` at the `HEAPU8` boundary is safe because each individual
offset is well under 2^53.

**Logged:** each call's return value and `typeof`, `dataPtr` printed in
hex (proves it crosses the ABI as BigInt), match/mismatch result.

### 4.3 Phase 3 — Cap probe

Sequential `_malloc(1 GiB)` loop, up to 16 iterations. After each
successful malloc, `HEAPU8.fill(0xab, ...)` the first 64 KiB of the new
region — forces an actual page commit, not a lazy reservation. Stop on
the first malloc that returns `0n` or throws.

**Logged:** per-iteration `(i, ptr_value, total_committed_bytes)`; final
`total_committed_bytes` and iteration count. **The total is the headline
number for the closure report** — Chrome's practical heap ceiling on the
dev box.

**Why commit, not just reserve:** lazy-allocated pages don't actually
pressure-test the system; we want the value Chrome will sustain in
practice during a real model load.

### 4.4 Phase 4 — Clean shutdown

`_webgpu_shutdown()`. Confirms the runtime is still healthy after the cap
probe. Logs return state and any console output.

## 5. Pass/fail thresholds + decision rule

| Phase | Pass condition | Fail meaning |
|---|---|---|
| 1 | `_webgpu_init()` returns 0, `_webgpu_shutdown()` completes, no console errors **other than** the benign `adapter_info:` line emitted by the WebGPU backend (per `CLAUDE.md`). | **MEMORY64 lever is closed.** Document Emscripten + Chrome versions, link upstream issue if one exists, defer until external state changes. |
| 2 | `_tensor_new_1d` and `_malloc` both return `bigint` non-zero values; the 4-element F32 round-trip through `HEAPU8` matches element-by-element. | BigInt migration has gotchas. Lever still potentially viable but follow-up bridge-migration scope grows materially; document the failure mode. |
| 3 | Informational only — outcome is a number. | n/a. |
| 4 | `_webgpu_shutdown()` returns cleanly, no non-benign console errors (same `adapter_info:` exclusion as Phase 1). | Runtime instability under heap pressure. Document as a caveat for any cap close to the measured ceiling. |

### 5.1 Decision rule

The probe exists to inform exactly one decision:

- **All earlier phases pass + Phase 3 ≥ 8 GiB committed.** MEMORY64 lever
  is **viable**. Promote to a follow-up "P2-class" spec covering: full TS
  bridge migration to BigInt, GGUF streaming validation under wasm64,
  one real-model load (target: 8B Q4_K_S or 13B Q4_K_S depending on
  Phase 3 ceiling), perf comparison vs wasm32 baseline.
- **Phase 1 fails.** Close MEMORY64 with measurement evidence. No
  follow-up until Emscripten / Chrome state changes.
- **Phase 1 passes, Phase 2 fails.** Narrower follow-up: investigate the
  specific ABI failure before committing more surface.
- **All earlier phases pass, Phase 3 < 6 GiB.** MEMORY64 infra works but
  **doesn't unlock 13B Q4_K_S** (the fleet expansion that motivated the
  probe). Close as "infrastructure functional but doesn't pay for itself
  at the ≤30B ceiling without Chrome moving its cap."
- **All earlier phases pass, 6 ≤ Phase 3 < 8 GiB.** Ambiguous — viable
  for some 8B Q4_K_S targets, marginal for 13B. Closure recommends a
  narrower follow-up that picks one 8B Q4_K_S model and measures load +
  first-token under wasm64.

### 5.2 Closure report contract

Single file: `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`,
mirroring the §22 / §29 closure-report shape. Required fields:

- Phase outcomes (P1, P2, P4 boolean; P3 cap value).
- The `window.__memory64ProbeResult` JSON blob verbatim.
- Emscripten version (from `emcc --version`), Chrome version + dev-box
  hardware specs.
- Decision-rule branch taken (one of §5.1).
- Link to follow-up spec (if any) or explicit "lever closed" with cited
  reason.
- Reproduction commands (`make mem64-probe` + the navigated URL).

A TODO `§31` closure entry references this report.

## 6. Out-of-scope (deliberately deferred)

The following are **not** produced, attempted, or documented by this
probe:

- TS bridge migration (`src/inference/ggml-wasm.ts`,
  `src/inference/model-inference.ts`) to BigInt pointers.
- GGUF streaming via `uploadRangeChunked` under wasm64 (heap-grow
  detachment under BigInt views is its own probe).
- Real model load, decode, or any tok/s measurement.
- `make checkall` / unit test additions.
- Any change to the live `webllm-wasm` build behavior — bit-identical
  pre/post.
- Performance comparison vs the wasm32 build.
- Any `src/inference/*.ts` change.

If any becomes relevant during probe execution, it's a new spec — not a
probe scope expansion.

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Emscripten doesn't support `-sMEMORY64=1 + ASYNCIFY` simultaneously | Medium — historically this combo had issues; current state unverified. | This is exactly what Phase 1 measures. A failure here is the probe doing its job. |
| `webgpu-bridge.cpp` has a wasm64-only signed/unsigned mismatch | ~~Low — already uses `size_t` casts.~~ **Closed 2026-04-28 by sub-probe** — bridge compiles cleanly as wasm64. | n/a |
| ggml-base / ggml-webgpu have a wasm64-only build issue | ~~Unknown.~~ **Closed 2026-04-28 by sub-probe** — both compile cleanly as wasm64 with `CMAKE_C_FLAGS=-sMEMORY64=1` propagated. | n/a |
| wasm32 ggml + wasm64 bridge cannot link | ~~High~~ **Discovered 2026-04-28 sub-probe Phase 1.** Architecture amended to parallel build dirs (§3.1) so ggml is rebuilt as wasm64 in `src/wasm/build-mem64/`; no cross-architecture link attempt. | Architecture amendment (§3.1). |
| Chrome caps `MAXIMUM_MEMORY` well below 8 GiB | Medium — Chrome's per-tab ceiling has historically been conservative on macOS. | Phase 3 measures it; decision rule has explicit branches for `< 6 GiB` and `6-8 GiB`. |
| Probe binary is large enough to slow page load | Low — same source as live binary; ~32 KB delta from extra runtime support at most. | Logged as `module_bytes`; informational only. |
| Agentchrome `runtime.evaluate` returns `BigInt` values that the harness can't serialize | Low. | The structured JSON blob narrows BigInts to Numbers (`Number(cap_bytes)`) at the serialization boundary, after the BigInt arithmetic has done its work. |
| `make wasm-build` regression from CMakeLists.txt cache-option gating | Low — gated `if(WEBLLM_BUILD_MEM64)` branch is `OFF` by default; existing build dir never sees the option. | Phase 1 verifies `make checkall` + `make wasm-build` both pass post-amendment. |

## 8. Files touched

**Modified:**
- `src/wasm/CMakeLists.txt` — adds a `WEBLLM_BUILD_MEM64` cache option
  gating which `add_executable` is created. Default `OFF` reproduces
  the pre-amendment build graph; `ON` produces the mem64 target. No
  other behavior change.
- `Makefile` — adds the `mem64-probe` target with its own cmake
  configure step in `src/wasm/build-mem64/`.
- `.gitignore` — adds `src/wasm/build-mem64/` to the existing
  `src/wasm/build/` exclusion line.
- `tsconfig.json` — adds `src/wasm/build-mem64` to the existing
  `exclude` array (which already lists `src/wasm/build`). Required
  because `make mem64-probe` emits `.js` files into the new build
  dir that `tsc --noEmit` would otherwise scan and reject as
  type-check input. Discovered post-Phase-2 amendment 2026-04-28
  when `make checkall` failed on the cmake-generated runtime
  scaffolding inside the mem64 build dir; the parallel exclusion
  mirrors the existing wasm32 build-dir pattern.

**New:**
- `smoke-test/mem64-probe.html` — standalone harness.
- `eval/reports/memory64-probe-2026-04-28/SUMMARY.md` — produced after
  execution.
- `src/wasm/build-mem64/` — parallel CMake build dir containing wasm64
  ggml archives + the mem64 binary. Gitignored.

**Unchanged:**
- `src/wasm/webgpu-bridge.cpp` — sub-probe confirmed wasm64-clean.
- All `src/inference/*.ts` files — bridge migration is out of scope.
- `src/wasm/build/` — existing wasm32 build dir is untouched.
- `make wasm-build` behavior — bit-identical pre/post (the cache-option
  default preserves the pre-amendment build graph).
- Test files; `make checkall` 428/11/0 holds pre/post.

## 9. Review checklist (for the post-execution closure)

- [ ] Phase 1 outcome documented with init wall time and any console errors.
- [ ] Phase 2 outcome documented with the actual BigInt pointer values
      observed (proves the > 4 GiB-able ABI).
- [ ] Phase 3 cap bytes documented as the headline number.
- [ ] Phase 4 outcome documented.
- [ ] `window.__memory64ProbeResult` JSON included verbatim.
- [ ] Emscripten + Chrome + dev-box hardware versions cited.
- [ ] Decision-rule branch (§5.1) explicitly identified.
- [ ] Follow-up spec linked (if viable) or "lever closed" reason cited.
- [ ] TODO §31 entry references the report.
