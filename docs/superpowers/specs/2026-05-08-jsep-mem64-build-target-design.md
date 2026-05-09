# JSEP + MEMORY64 build target — design

> **SUPERSEDED 2026-05-08 — negative-result closure.** Implementation attempt
> hit two architectural blockers (static_assert at `ggml-jsep.cpp:830`
> guarding JSEP against MEMORY64; `host_mirror` weight duplication
> inside the wasm heap regardless of cap). See
> [`../../../eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`](../../../eval/reports/jsep-mem64-2026-05-08/SUMMARY.md)
> for the full blocker catalogue and re-evaluation triggers. Stage 4.36's
> deferred-subset closure under "path 3 — mathematical interpolation
> acceptance" remains the operative stance. This document is retained as
> the historical record of the attempt.

Status: proposed, 2026-05-08
Scope: TODO.md "Stage 4.36 closed — Phase 3 closed for testable subset
(2026-05-08)" → re-enablement path 1 from
[`STAGE-4.36-RESULT.md`](../../../eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)
"Deferred subset" section.
Trigger: Phase 3 JSEP causal-LM decode reached parity for the testable
subset of the canonical-6 (TinyLlama, Qwen3-0.6B, Qwen3-1.7B) but
deferred the 7B+ subset (mistral-7b-q4ks, llama-3.1-8b-iq3m,
qwen3-8b-iq3m) because all three exceed the wasm32 4 GiB JSEP heap cap.

## Problem

The JSEP build (`webllm-wasm-jsep.{js,wasm}`, produced by
`make wasm-build-jsep`) links `-sMAXIMUM_MEMORY=4GB`
(`src/wasm/CMakeLists.txt:249`). The spike harness loads a GGUF into a
JS `Uint8Array`, then `bridge.loadModel(buf)` `malloc`s a copy inside
the WASM heap; libllama allocates KV cache + scratch on top. GGUFs ≥
~3.5 GiB do not leave headroom for the 7B-8B canonical-6 entries:

| Model | GGUF size | Margin under 4 GiB |
|-------|-----------|--------------------|
| mistral-7b-instruct-v0.3-q4ks | 4.14 GiB | -0.14 GiB (overflow) |
| llama-3.1-8b-instruct-iq3m | 3.78 GiB | 0.22 GiB (insufficient) |
| qwen3-8b-iq3m | 3.90 GiB | 0.10 GiB (insufficient) |

The non-JSEP wasm64 build (`webllm-wasm-mem64.{js,wasm}`, produced by
`make wasm-build-mem64`) lifts the cap to 16 GiB by setting
`-sMEMORY64=1`, `-sMAXIMUM_MEMORY=16GB`, and `-sWASM_BIGINT=1`. It
already ships and runs the canonical-6 7B+ models successfully through
the legacy graph-builder path. The JSEP build does not yet have a wasm64
variant; the two flags (`WEBLLM_BUILD_MEM64=ON`, `WEBLLM_BACKEND=jsep`)
are mutually exclusive in `CMakeLists.txt` by omission, not design — the
`WEBLLM_BUILD_MEM64` block at lines 186-220 always links non-JSEP
(`llama ggml-base ggml-webgpu`), with no `ggml-jsep`, no `JSPI_EXPORTS`
link option, no `WEBLLM_PIN_TO_JSEP=1` define, no `OUTPUT_NAME` suffix.

This document specifies a `wasm-build-jsep-mem64` target that combines
both flags, plus the harness wiring needed to run the deferred subset
through Stage 4.36's parity gate against `webllm-wasm-mem64.js` as the
non-JSEP reference.

## Goals

1. **Build matrix.** A `make wasm-build-jsep-mem64` target produces
   `webllm-wasm-jsep-mem64.{js,wasm}` from a separate
   `src/wasm/build-jsep-mem64/` build dir. Combines `-sMEMORY64=1`,
   `-sMAXIMUM_MEMORY=16GB`, `-sWASM_BIGINT=1`, `-sJSPI_EXPORTS=...`,
   `WEBLLM_PIN_TO_JSEP=1`, and links `ggml-jsep`. Bundle wiring
   (`webllm-bundle-jsep.js`) unchanged — the bundle stays static; the
   harness picks the WASM module dynamically at import time.

2. **Parity gate.** Re-run Stage 4.36's parity gate on the deferred
   subset. For each of mistral-7b-q4ks, llama-3.1-8b-iq3m,
   qwen3-8b-iq3m, the JSEP+mem64 spike produces `generatedIds[0..4]`
   that exactly match the non-JSEP `webllm-wasm-mem64.js` reference for
   the same prompt. Match the format and rigor of
   [`canonical6-refs.json`](../../../eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json).

3. **Per-model dispatch.** Each entry in the harness MODELS registry
   gets `requiresMem64: boolean` (default false). On startup, the
   harness dynamic-imports the wasm32 or wasm64 JSEP WASM based on the
   selected model's flag. A size-threshold assertion (HEAD on
   `ggufUrl`, fail if `Content-Length > 3.5 GiB` and
   `requiresMem64=false`) catches "forgot the flag" mistakes.

4. **Phase A → B → C ordering with go/no-go after Phase A.** Phase A
   (build target only) lands first as the cap-probe-doctrine
   "bump first, characterize second" step. If Phase A fails to link,
   stop before sinking time on harness work. Each phase touches ≤5
   files per CLAUDE.md "phased execution" workflow policy.

## Non-goals (out of scope)

- **Streaming-loader migration.** Re-enablement path 2 from
  STAGE-4.36-RESULT.md (chunked HEAPU8 loader). Not needed once the
  cap is lifted; mistral-7b's 2× heap residency during load
  (~8.3 GiB peak) fits in 16 GiB. Stays deferred for an external
  trigger.
- **Computed dispatch via parsed GGUF metadata + nCtx.** Considered
  during brainstorming and rejected as over-engineered for the
  current model fleet. The explicit `requiresMem64` flag plus the
  size-threshold assertion catches all current and foreseeable cases
  with ~2 lines of harness code instead of a two-stage GgufParser
  refactor.
- **Unified `make wasm-build-all` umbrella target.** Each build target
  stays standalone; callers invoke the variant they need.
- **Changes to `wasm-build-mem64` (non-JSEP) or `wasm-build-jsep`
  (wasm32) targets.** Both ship as-is; only additive new target.
- **Re-running Stage 4.36's testable subset (TinyLlama / Qwen3-0.6B /
  Qwen3-1.7B) on the new wasm64-JSEP build.** The testable subset's
  parity gate stays on the wasm32-JSEP build. Re-confirming under
  wasm64 is not required for closing the deferred-subset gate.
- **Quantization variants beyond what's already registered.** Only
  the existing canonical-6 deferred entries are validated.
- **Models above the 8B parameter ceiling** per the CLAUDE.md
  "Model-size ceiling: 8B parameters" policy.

## Architecture

### Build matrix change (Phase A)

`src/wasm/CMakeLists.txt` currently has two top-level branches:

```
if(WEBLLM_BUILD_MEM64)
    # wasm64 non-JSEP only — no ggml-jsep, no JSPI_EXPORTS
    add_executable(webllm-wasm-mem64 ...)
else()
    add_executable(webllm-wasm ...)
    if(WEBLLM_BACKEND_JSEP)
        # wasm32 JSEP — links ggml-jsep, sets WEBLLM_PIN_TO_JSEP=1
    else()
        # wasm32 non-JSEP
    endif()
endif()
```

Refactor: thread the `WEBLLM_BACKEND_JSEP` branching into the
`WEBLLM_BUILD_MEM64` block, mirroring what the wasm32 branch does:

```
if(WEBLLM_BUILD_MEM64)
    add_executable(webllm-wasm-mem64 ...)
    if(WEBLLM_BACKEND_JSEP)
        target_link_libraries(... ggml-jsep)
        set_target_properties(... OUTPUT_NAME "webllm-wasm-jsep-mem64")
        target_compile_definitions(... WEBLLM_PIN_TO_JSEP=1)
        # add JSPI_EXPORTS to target_link_options
    else()
        # existing non-JSEP wasm64 path
    endif()
endif()
```

Output basename: `webllm-wasm-jsep-mem64.{js,wasm}`. Build dir:
`src/wasm/build-jsep-mem64/`.

`Makefile`: add `wasm-build-jsep-mem64` target mirroring
`wasm-build-jsep` (lines 130-173) but combined with the
`-DWEBLLM_BUILD_MEM64=ON -DCMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1`
flags from `wasm-build-mem64` (lines 104-128). Copy artifacts to
`smoke-test/`.

Phase A acceptance: `make wasm-build-jsep-mem64` produces a `.js` +
`.wasm` artifact pair without link errors. A new
`smoke-test/mem64-jsep-probe.html` (~30 LOC) dynamic-imports the new
WASM module, calls `webgpu_init`, logs the result. Confirms JSPI
promise wrapping survives the wasm64 ABI shift before any model load.
Closure stub: artifact size + `Module.HEAP8.length` post-init vs the
16 GiB cap.

### Reference capture (Phase B)

`smoke-test/p2-v2-ref-probe.{html,src.ts}` runs the non-JSEP
reference. Today it only imports `webllm-wasm.js` (wasm32). Changes:

1. Add 3 entries to MODELS for the deferred subset:
   ```ts
   "mistral-7b-q4ks": {
       ggufUrl: "/models/mistral-7b-instruct-v0.3.Q4_K_S.gguf",
       promptText: "The capital of France is",
       requiresMem64: true,
   },
   "llama-3.1-8b-iq3m": {
       ggufUrl: "/models/Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf",
       promptText: "The capital of France is",
       requiresMem64: true,
   },
   "qwen3-8b-iq3m": {
       ggufUrl: "/models/Qwen3-8B-IQ3_M.gguf",
       promptText: "The capital of France is",
       requiresMem64: true,
   },
   ```
   (Exact `ggufUrl` paths: confirmed against `eval/models.ts` /
   `models/` directory during Phase B implementation.)
2. Replace the static `import("./webllm-wasm.js")` call with a
   dispatch:
   ```ts
   const wasmModule = MODEL_ENTRY.requiresMem64 ? "webllm-wasm-mem64.js" : "webllm-wasm.js";
   const createModule = (await import(`./${wasmModule}${cacheBust}`)).default;
   ```
3. Add the size-threshold assertion: HEAD `ggufUrl`, throw if
   `Content-Length > 3.5 GiB && !requiresMem64` with a clear message
   instructing the caller to set `requiresMem64: true`.
4. Capture refs for the 3 new entries via agentchrome navigation:
   `?model=mistral-7b-q4ks` → record `generatedIds[0..4]`,
   `generatedText`, `logitStep0`. Repeat for the other two.
5. Merge captured refs into
   `eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json`.
   Update the file's `comment` to reflect that all 6 canonical models
   are now covered. Existing 3 entries unchanged.

### Parity sweep (Phase C)

`smoke-test/p2-v2-spike.{html,src.ts}` runs the JSEP probe under test.
Mirror Phase B changes in MODELS and the WASM-module dispatcher. Then
run the spike for each deferred model:

- `?model=mistral-7b-q4ks` → record `generatedIds[0..4]`.
- `?model=llama-3.1-8b-iq3m` → record `generatedIds[0..4]`.
- `?model=qwen3-8b-iq3m` → record `generatedIds[0..4]`.

Compare each against the corresponding entry in `canonical6-refs.json`.
Pass = exact integer-array match for `generatedIds[0..4]`.

Closure report: `eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`. Sections:
TL;DR (3 models × pass/fail), build artifacts and sizes, peak
`Module.HEAP8.length` per model, parity tables, patch-stack delta (0 —
no llama.cpp patches expected), and a one-line link to update
[`STAGE-4.36-RESULT.md`](../../../eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)
"Deferred subset" closure.

`TODO.md` update: replace the Stage 4.36 closure stub's "Deferred
subset" paragraph with a "Phase 3 closure extended to full canonical-6
2026-05-08" stub. Move the 7B+ deferral section to `TODO_ARCHIVE.md` per
the CLAUDE.md TODO archival cadence.

## Risk register

1. **JSPI re-entrancy under wasm64.** wasm32-JSEP exercises JSPI;
   wasm64 non-JSEP exercises JSPI; the *intersection* (i64 args at the
   `WebAssembly.promising` boundary) is novel. Mitigation: Phase A's
   smoke probe loads the WASM and calls `webgpu_init` before any model
   load, so a JSPI/wasm64 ABI mismatch surfaces in the cheapest
   possible context.

2. **Per-tensor i32 high-half assumption** (`src/inference/jsep/ops/matmul.ts:30,32,164`).
   The JSEP descriptor packer puts `ne[0..3]` and `nb[0..3]` low-half
   in i32 slots, with high half "zero on wasm32 with sub-2GB tensors."
   8B IQ3_M individual tensors (token_embd, lm_head) stay <2 GiB
   (vocab × hidden × dtype-bytes ≪ 2 GB), so high half is zero under
   wasm64 too. Mitigation: assert in Phase C harness that the high-half
   slots remain zero on first matmul descriptor write; flag if
   non-zero. (Defensive — likely a no-op observation.)

3. **`new Uint8Array(buf, offset, len)` indexing under 16 GiB heap.**
   JS Number addresses ArrayBuffer up to 2^53, so a 16 GiB heap
   (~2^34 bytes) is well within reach. Mitigation: none required;
   document the math in the closure report.

4. **2× heap residency on GGUF load.** `bridge.loadModel(buf)` does
   `bridge_malloc(buf.byteLength) + HEAPU8.set(buf, ptr)` after the
   harness already holds the buf in JS. For mistral-7b 4.14 GiB GGUF,
   peak resident ≈ 8.3 GiB during load + libllama scratch — fits in
   16 GiB but with limited headroom. Mitigation: log
   `Module.HEAP8.length` immediately before and after
   `webllm_load_model` in the spike to confirm peak; if it approaches
   the 16 GiB cap, the streaming-loader path becomes load-bearing.

5. **Bundle-side dynamic import path resolution.**
   `webllm-bundle-jsep.js` is built by `bun run build:jsep`; it
   doesn't statically import the JSEP WASM (the harness does the
   `import()`). The harness is the only consumer that needs to know
   about `webllm-wasm-jsep-mem64.js`. Mitigation: bundle layout
   unchanged; only harness `.html` + `.src.ts` files touched in
   Phase B/C.

6. **Pre-existing assertion failure paths under wasm64.** The original
   MEMORY64 migration's load-bearing lesson was "Phase 1 BigInt-coverage
   gap surfaced only at Phase 4 integration" — i.e., a missing pointer
   ABI translation can hide all the way to a real model run. Mitigation:
   `llama-bridge.ts:241-264` already auto-detects the ABI and threads
   `to64`/`from64` at every cwrap boundary. JSEP-side direct
   `Module.HEAPU8.buffer` accesses (`src/inference/jsep/index.ts:214`,
   `:215`, `:227`, `:252`, `:279`) take pre-translated `hostPtr` JS
   Numbers — already 64-safe.

## Phasing

### Phase A — build target

Files touched (≤4):
- `src/wasm/CMakeLists.txt` (refactor `WEBLLM_BUILD_MEM64` block to
  thread JSEP branching).
- `Makefile` (new `wasm-build-jsep-mem64` target).
- `smoke-test/mem64-jsep-probe.html` (new ~30-LOC linkage smoke).
- (Possibly) `Makefile` again for `smoke-test` rule update if mem64-jsep
  binary needs to be co-located in `smoke-test/`.

Acceptance:
- `make wasm-build-jsep-mem64` exits 0; produces
  `webllm-wasm-jsep-mem64.js` + `webllm-wasm-jsep-mem64.wasm` in
  `src/wasm/build-jsep-mem64/`.
- `make checkall` green (no TS regressions).
- `mem64-jsep-probe.html` loads, runs `webgpu_init`, logs
  `Module.HEAP8.length` post-init. No abort, no JSPI mismatch.

Commits:
- `feat(wasm): add wasm-build-jsep-mem64 target combining JSEP + MEMORY64`
- `docs(reports): jsep-mem64 Phase A closure — build target verified`

### Phase B — reference capture

Files touched (≤3):
- `smoke-test/p2-v2-ref-probe.src.ts` (3 new MODELS entries +
  per-model `requiresMem64` dispatch + size-threshold assertion).
- `smoke-test/p2-v2-ref-probe.html` (no changes expected; if `<title>`
  or comment-banner mentions wasm32-only, update).
- `eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json`
  (additive merge — 3 new entries; update `comment` field).

Acceptance:
- Three successful captures via agentchrome (or `make smoke-serve` +
  manual nav). Each yields `generatedIds[0..4]` and
  `generatedText` matching the one-shot output.
- `canonical6-refs.json` validates as JSON, contains all 6 entries.
- `make checkall` green.

Commits:
- `feat(harness): mem64 dispatch + 3 deferred-subset entries in ref-probe`
- `docs(reports): canonical6-refs.json — capture mistral-7b-q4ks, llama-3.1-8b-iq3m, qwen3-8b-iq3m refs`

### Phase C — parity sweep

Files touched (≤4):
- `smoke-test/p2-v2-spike.src.ts` (mirror Phase B changes).
- `smoke-test/p2-v2-spike.html` (parallel to ref-probe; banner update
  if needed).
- `eval/reports/jsep-mem64-2026-05-08/SUMMARY.md` (new closure report).
- `TODO.md` + `TODO_ARCHIVE.md` (closure-stub swap; archive 7B+
  deferral block).

Acceptance:
- Three successful spike runs, one per deferred model, each producing
  exact-match `generatedIds[0..4]` against `canonical6-refs.json`.
- Closure report banks: build artifact sizes, peak
  `Module.HEAP8.length` per model, parity table.
- `make checkall` green.
- `TODO.md` reflects the closure; `TODO_ARCHIVE.md` carries the
  archived 7B+ deferral section.

Commits:
- `feat(harness): mem64 dispatch + 3 deferred-subset entries in spike`
- `docs(reports): jsep-mem64 Phase 3 closure — full canonical-6 parity`
- `docs(TODO): Phase 3 closure extended to canonical-6; archive 7B+ deferral`

## Testing

`make checkall` green per phase (fmt + lint + typecheck + test). No
new unit tests — the parity gate is the test, per Stage 4.36
doctrine: closure-report-driven, harness output is the assertion.

## Open questions

None. All design choices locked in via brainstorming Q1-Q3:
- Q1: ship-grade scope (parity sweep on deferred subset).
- Q2: per-model `requiresMem64` flag (recommended path).
- Q3 (revised): explicit flag + size-threshold assertion.

Phasing locked via Q3-original: three-phase plan with go/no-go after
Phase A.
