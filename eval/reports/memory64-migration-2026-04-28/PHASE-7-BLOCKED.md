# MEMORY64 migration — Phase 7 BLOCKED

**Date:** 2026-04-28
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Model:** mistral-7b-instruct-v0.3-q5km (4.8 GiB on disk; HF mirror: bartowski/Mistral-7B-Instruct-v0.3-GGUF)
**Binary:** webllm-wasm-mem64.{js,wasm} (wasm64) — current local build off `c919efa`

## Headline

Phase 7 cannot close. The wasm64 binary loads the >4 GiB Q5_K_M model
fine end-to-end through KV-cache init and tokenizer setup, but blows
up at the **first compute graph** with a JS-side `_wgpuDeviceCreateBindGroup`
TypeError ("Required member is undefined" on `GPUBufferBinding.buffer`).
Reproduces in BOTH paths: smoke-test direct chat AND `make smoke-bench
PERF_MODEL=mistral-7b-instruct-v0.3-q5km PERF_RUNS=3 WASM_VARIANT=mem64`.

This is a real WebGPU-bridge bug that Phase 5's canonical-6 parity
sweep did not catch because none of the canonical 6 use a Q5_K family
quant (canonical pins are Q4_0 / Q4_K_S / Q3_K_M / IQ3_M / IQ4_XS).

## What works under wasm64 (probed today)

- `qwen3-0.6b-q4f16` (arch=qwen3, vocab=151936): full chat passes,
  76.7 tok/s decode.
- `llama-3.2-1b-q4f16` (arch=llama, vocab=128256): full chat passes,
  91.6 tok/s decode.
- Phase 5 historical: mistral-7b-instruct-v0.3-q4ks under wasm64 ran
  29.1 / 29.4 / 28.5 tok/s — i.e. arch=llama with Mistral architecture
  is fine under wasm64 at the same param count, just at Q4_K_S not
  Q5_K_M.

## What fails

`mistral-7b-instruct-v0.3-q5km` under wasm64:

```
[1/8] WebGPU backend initialized
[2/8] Model fetched: 5136.2 MB in 1.6s         ← >4 GiB load works
[3/8] GGUF parsed: arch=llama emb=4096 heads=32/8 layers=32 vocab=32768 ctx=32768
[4/8] Weights loaded in 1.5s                   ← all weight uploads succeed
[5/8] KV cache: 4096 slots x 32 layers
[6/8] Tokenizer ready: vocab=32768
[6/8] Shader-cache warmup failed (continuing): Failed to execute 'createBindGroup'
      on 'GPUDevice': Failed to read the 'entries' property from 'GPUBindGroupDescriptor':
      Failed to read the 'resource' property from 'GPUBindGroupEntry': Failed to read
      the 'buffer' property from 'GPUBufferBinding': Required member is undefined.
[7/8] Generation failed: <same TypeError>
```

JS stack (truncated):

```
TypeError: Required member is undefined.
    at webllm-wasm-mem64.js:1:97390
    at _wgpuDeviceCreateBindGroup (webllm-wasm-mem64.js:1:97425)
    at webllm-wasm-mem64.wasm:wasm-function[1186]
    at wrapper (...:100410)
    at invoke_jjj (...:121609)
    at webllm-wasm-mem64.wasm:wasm-function[974]
```

Diagnostic narrowing: the >4 GiB GGUF parses correctly into
`arch=llama emb=4096 heads=32/8 layers=32 vocab=32768`, weight upload
to GPU succeeds (so per-tensor `_bridge_malloc` + heapU8.set offsets
are fine across the wasm64 BigInt boundary), KV cache allocates
(more big mallocs OK), tokenizer constructs. The very first compute
graph (warmup) is where it dies — which is the first place the
**Q5_K matmul kernel** is exercised.

## Hypothesis (not yet root-caused)

The `_wgpuDeviceCreateBindGroup` import bridge in the wasm64 build
reads a `WGPUBindGroupDescriptor` struct from wasm memory and
returns a JS `GPUBindGroupDescriptor`. With `-sMEMORY64=1`, struct
member offsets and pointer fields inside that descriptor change
size (32→64 bit). Either:

1. The Emscripten-generated `_wgpuDeviceCreateBindGroup` shim has a
   wasm64 layout bug specific to a binding shape only Q5_K-family
   kernels emit (Q5_K's matmul probably binds an extra scratch /
   scales buffer the Q4_K kernels don't), OR
2. The patched `ggml-webgpu` Q5_K shader wiring on the
   `webllm-browser-patches` branch passes a `WGPUBufferBinding`
   with a buffer handle that's read as 0 / undefined under
   MEMORY64 pointer width.

This is a real wasm64 correctness gap, not a model-size gap. It
would also reproduce for any future Q5_K_S / Q5_K_L quant under
wasm64. It does NOT block ≤4 GiB Q5_K_M targets running under
wasm32 (any of those would still pass through the wasm32 binary
via `pickWasmUrl`'s ≤3.5 GiB → wasm32 branch).

## Hypothesis update — 2026-04-28 (after retry on Mistral-Nemo)

The "Q5_K-family-specific" framing above is **wrong**. A retry
attempt with Mistral-Nemo-Instruct-2407 Q4_K_S (~6.63 GiB,
registered at commit `ca01d4f → 388142b`) — chosen because Q4_K_S
is a **Phase-5-validated kernel family** — produced the **identical**
`_wgpuDeviceCreateBindGroup` `Required member is undefined` error
at the warmup graph. This rules out the kernel-family hypothesis.

The pattern that actually fits all three datapoints is **model
working-set size**, not kernel type:

| Model | Quant | File size | wasm64 outcome |
|---|---|---:|---|
| Mistral-7B Q4_K_S | Q4_K | 3.95 GiB | ✅ runs (Phase 5: 28.5-29.4 tok/s) |
| Mistral-7B Q5_K_M | Q5_K | 5.10 GiB | ❌ bind-group fail at warmup |
| Mistral-Nemo Q4_K_S | Q4_K | 6.63 GiB | ❌ bind-group fail at warmup |

Working hypothesis (replaces the prior Q5_K kernel hypothesis):
the bug surfaces when the buffer slot in a `WGPUBindGroupEntry`
references memory whose **address exceeds 2³² in wasm64 address
space**. Phase 5's Mistral-7B Q4_K_S keeps the entire model below
the 4 GiB mark; Q5_K_M and Mistral-Nemo's data spans both halves.
Likely site:

1. **Emscripten's auto-generated `_wgpuDeviceCreateBindGroup` JS
   shim** reads each `WGPUBindGroupEntry` from wasm memory; the
   `WGPUBuffer` handle field is `void*` (i64 in wasm64). If the
   shim has a hardcoded 32-bit offset somewhere in the struct
   layout, the buffer field reads as `undefined` whenever the
   handle is non-zero in its upper 32 bits. (Emscripten 5.0.6 ref
   `6ea9c28c38cdd40c1032fa04400c9d16230ee180` — pinned, may
   already be patched upstream).
2. **Patched `ggml-webgpu`** Q5_K and Q4_K matmul kernels both
   touch model weights through bindings; if the C-side struct
   construction passes a buffer pointer/handle that's been
   computed via 32-bit arithmetic anywhere in the path, it'd
   trim the high bits and emit zero/garbage to the JS shim.
3. **Buffer slicing on the WebGPU side** — Apple's M4 Max has
   `maxStorageBufferBindingSize = 2 GiB`. A single 6.6 GiB GGUF
   blob mapped as a sub-binding might exceed that limit and the
   browser silently fails the binding; but the error message
   would more likely call that out specifically rather than
   "Required member is undefined".

This is now task #543's actual scope: **wasm64 bind-group failure
above the 2³² model-data threshold, kernel-family-independent.**

## Implications for Phase 6 / migration closure

Phase 6's `pickWasmUrl` change shipped a "production wasm64
binary that auto-routes >3.5 GiB models." That binary builds and
loads >4 GiB GGUFs into memory correctly (verified through
`Weights loaded in 2.4s` in this retry), but **cannot run
inference on them** — the very class of models it was added to
support. Migration closure by candidate-pivot is no longer
viable: every >4 GiB target hits the same bind-group bug.

Path forward options:
- **A1:** Triage the bind-group bug as the actual blocker.
  Diagnostic plan: instrument the patched ggml-webgpu Q4_K
  matmul to log the buffer pointer/offset/size before
  `wgpuCreateBindGroup`, run Mistral-Nemo Q4_K_S, see whether
  the offending field is bridge_malloc'd above 2³² or below.
- **A2:** Document the bug as a known limitation and ship the
  dual binary with the wasm64 path **only proven for ≤4 GiB
  models that the wasm32 path also handles** — i.e. dual binary
  has no current value-add. Migration stays nominally closed
  for the ≤4 GiB ceiling but loses the 13B/30B aspiration.
- **A3:** Revert Phase 6's auto-routing default. Keep wasm64
  as an opt-in that throws a known-issue warning at load
  time when the caller selects it for a >4 GiB GGUF until A1
  lands.

## Two secondary gaps surfaced

1. **The smoke-test page does NOT auto-pick wasm64.** `pickWasmUrl`
   in `src/core/engine.ts` (Phase 6) only routes through the
   `WebLLM.loadModel` engine path. `smoke-test/real-model-page.js`
   has its own ad-hoc binary picker that defaults to wasm32 unless
   the URL carries `?wasm=mem64`. `bench-browser-eval` does NOT
   plumb that param, so a >4 GiB profile run via
   `make bench-browser-eval` silently tries to malloc 5 GiB inside
   the 4-GiB-cap wasm32 heap and dies with a JS-side TypedArray
   "offset is out of bounds" on the first chunk write. (`make
   smoke-bench` does plumb it via `WASM_VARIANT=mem64` env var.)
2. **Phase 5's parity sweep did not exercise Q5_K**. The
   "wasm64-vs-wasm32 ±3% on the canonical 6" gate is real but
   underspecified — it covers Q4_0/Q4_K_S/Q3_K_M/IQ3_M/IQ4_XS only.

## What I did do

- Step 1 (HF availability) — already done by the parent.
- Step 2 — added `mistral-7b-instruct-v0.3-q5km` model entry to
  `eval/models.ts` directly after the Q4_K_S entry (5400 MB
  vramMB hint, Q5_K_M ggufFilePattern, all other fields cloned).
  Comment block above it cites Phase 7 of the migration and the
  `pickWasmUrl` ≥3.5 GiB threshold.
- Step 3 — added `mistral-7b-v0.3-q5km-warm` smoke profile in
  `eval/smoke-profiles.ts` mirroring the Q4_K_S entry.
- Step 4 — `make checkall` clean (fmt + lint + typecheck + 451
  tests pass).
- Step 5 — `make bench-profile PROFILES=...` runs the bun-only
  speed harness which writes a no-engine zero-score JSON; switched
  to `make bench-browser-eval PROFILE=mistral-7b-v0.3-q5km-warm
  WEBLLM_LIVE_BENCH_URL=http://localhost:8033`. Stalled at model-
  load with "offset is out of bounds" — that's the gap (1) above
  (smoke-test page picked wasm32 by default).
- Manual probe with `&wasm=mem64` appended to the eval URL: model
  loaded fine through step 5/8, then died at step 6/8 (warmup) with
  the bind-group bug above. **This is the actual blocker.**
- Step 6 — `make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q5km
  PERF_RUNS=3 WASM_VARIANT=mem64` reproduces the same bind-group
  failure. Speed gate unmeasurable.
- Working set (informational, model loaded successfully so this is
  measured not estimated):
  - Model file: 4.8 GiB on disk (5136.2 MB reported by browser).
  - KV cache @ ctx=4096 (32 L × 32 H × 128 hd × 4096 ctx × 2 fp16
    × 2 K+V): 2.0 GiB (note: ctx=4096 here, not the 2048 the task
    spec assumed → 2× the task-spec estimate).
  - Activations + scratch + WebGPU buffers: ~1-2 GiB observed
    (reached step 6/8 with no OOM).
  - Total at warmup-failure point: ~8 GiB allocated, well under
    the 16 GiB toolchain ceiling.

## What I did NOT do

- Step 7-9 (author closure report, run cleanup, update TODO.md,
  commit). The migration is not closed. Closure-claiming docs
  would be inaccurate.
- I did NOT revert the `eval/models.ts` / `eval/smoke-profiles.ts`
  additions. They're useful as the standing reproducer for the
  bind-group bug. They're sitting unstaged for the parent agent
  to decide whether to commit as a `chore(eval): register q5km
  reproducer` (with a comment pointing at this file) or revert.

## Suggested follow-up by the parent

1. **Triage the bind-group bug.** The fastest path is probably to
   run the same Q5_K_M model under the wasm32 binary at smaller
   ctx if there's any quant-shrunk variant available, or to add
   debug logging in the patched `ggml-webgpu` Q5_K matmul kernel
   (or in the Emscripten-generated `_wgpuDeviceCreateBindGroup`
   shim) under MEMORY64 to see which buffer slot is undefined.
   The patched llama.cpp branch (`webllm-browser-patches` per
   `docs/LLAMA_CPP_PATCHES.md`) is the most likely fix site.
2. **Plumb `pickWasmUrl` (or its equivalent) into the smoke-test
   page.** Right now the smoke-test page is the canonical surface
   for `bench-browser-eval` but only uses wasm64 if `?wasm=mem64`
   is explicit. Either auto-pick from the model registry's
   `vramMB` (or a new explicit `requiresWasm64` flag), or have
   `eval/browser-eval.ts` `profileToUrlParams` inject `wasm=mem64`
   when the model entry warrants it.
3. **Tighten the Phase 5 gate retroactively** — add at least one
   Q5_K-family model to the parity sweep so the kernel surface is
   covered. Mistral-7B Q5_K_S (~4.6 GiB, also >4 GiB cap) or
   llama-3.2-3B Q5_K_M (would fit under wasm32 → direct compare)
   would both exercise the Q5_K kernel.
4. Once (1) is fixed, re-run the Phase 7 task as written. The
   model entry, profile, dashboard, and reproducer are all in
   place; only the kernel correctness work blocks closure.

## Reproduction (failing)

```bash
make wasm-build                           # both wasm32 + wasm64 binaries
make smoke-serve &
agentchrome navigate http://localhost:8031/real-model.html?\
model=mistral-7b-instruct-v0.3-q5km&ctx=4096&temp=0.6&wasm=mem64
# → fails at [6/8] shader-cache warmup with the bind-group TypeError above

make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q5km \
  PERF_RUNS=3 WASM_VARIANT=mem64
# → "Fatal: Timed out waiting for smoke-test result line on the page"
```
