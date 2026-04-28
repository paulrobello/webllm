# llama.cpp Browser Patches

Inventory of the local `webllm-browser-patches` branch this repo builds
against, plus the procedure for rebasing it onto newer upstream
`llama.cpp` masters.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Patch Inventory](#patch-inventory)
- [Rebase Procedure](#rebase-procedure)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

## Overview

This repo's `make wasm-build` target compiles the WASM module from a
**local, patched** `llama.cpp` checkout — not the upstream release. The
patches address stack safety, async / ASYNCIFY interactions, and a
request-based async readback API that the browser integration depends on.

Without the branch checked out, the WASM build either fails or produces a
binary that crashes the page during inference.

## Prerequisites

- Local clone at `~/Repos/llama.cpp/`
- Branch `webllm-browser-patches` checked out:
  ```bash
  cd ~/Repos/llama.cpp && git checkout webllm-browser-patches
  ```
- Emscripten SDK sourced (`~/emsdk/emsdk_env.sh`)

## Patch Inventory

The branch currently carries eleven commits on top of upstream `master`,
in the order shown (oldest first). Commit 7 and its revert (commit 8)
are kept as a pair pending a proper replacement; treat them as a no-op
until you hear otherwise. Last rebased onto upstream master 2026-04-27
to tip `434b2a1ff` ("ggml-webgpu: add Q1_0 support (#22374)"). The
2026-04-26 → 2026-04-27 delta was 13 commits; 3 of them touched
`ggml-webgpu/` (Q1_0 kernel addition #22374, fast i-quant mat-vec
kernels #22344, performance-portable register-tile / subgroup matmul
tuning #22241) — zero conflicts on rebase, all 11 patches replayed
cleanly. WASM rebuild + checkall (427/11/0) + browser smoke
(TinyLlama Q4_0 120 tok/s, encoder cosine 0.76, zero console errors)
verified post-rebase. Safety branch
`webllm-browser-patches-pre-rebase-2026-04-27` preserves the
pre-rebase tip (`a536df4f4`). See notes below patch 9 for the status
of FA enablement on browser decode (unchanged).

### 1. ggml: iterative `ggml_visit_parents_graph` for WASM stack safety

The recursive graph visitor overflows the JS / WASM stack on deep
transformer graphs. Rewritten as an explicit heap-allocated stack so
deep models load without a stack-overflow abort.

### 2. ggml-webgpu: browser + ASYNCIFY support bundle

- ASYNCIFY-safe wait / map paths
- Non-aborting device error handler
- Per-dispatch compute-pass fallback with overlap-only conflict detection
- `GGML_OP_DIAG_MASK_INF` shader

### 3. ggml-webgpu: request-based browser readback API

Adds a real request-based async GPU readback API for browser callers:
`begin` / `poll` / `finish` / `cancel` around queue completion + buffer
map. The TypeScript side calls this via `ggml-wasm.ts` rather than the
synchronous path that the native backend uses.

### 4. ggml-webgpu: harden async readback request cleanup

Fixes async request cleanup and cancellation lifecycle so pending
callbacks do not race buffer teardown during browser readback.

### 5. ggml-webgpu: notify browser async readback completion

Adds a completion callback so JS callers can wake the right `await`
without spinning on poll. Used by `ggml-wasm.ts`'s async readback path.

### 6. ggml-webgpu: add opt-in browser graph profiling

Per-graph profile buckets (total / encode-overhead / matmul /
attention / dispatch count) plus a C-exported accessor surface so the
TS harness (`eval/perf.ts`) can read them without scraping console
output. Adds `profile_timestamps_active` to the per-context profiling
state so timestamp queries only fire when a profile run is active.

### 7. ggml-webgpu: specialize browser decode matmul dispatch

Decode-phase specialization for the matmul dispatch path.

### 8. Revert "ggml-webgpu: specialize browser decode matmul dispatch"

Kept as a pair with commit 7 until the specialization is replaced.
Effectively a no-op relative to commit 6's state.

### 9. ggml-webgpu: add `GGML_OP_NORM` (LayerNorm) support

Upstream `ggml-webgpu` only registers `GGML_OP_RMS_NORM` and
`GGML_OP_L2_NORM`; `GGML_OP_NORM` falls through to `supports_op=false`
and silently no-ops on a CPU-less build. That broke webllm's BERT
encoder path (`EncoderInference`), producing bit-identical embeddings
for every input. This patch extends `row_norm.wgsl` with a
`LAYER_NORM` variant that computes both Σx and Σx² in one workgroup
pass, derives variance, and emits `(x − mean) / sqrt(var + eps)`;
registers the pipeline in `ggml-webgpu-shader-lib.hpp`; and dispatches
`GGML_OP_NORM` through `ggml_webgpu_row_norm`. Touches the same three
files as the other `ggml-webgpu` patches (`ggml-webgpu.cpp`,
`ggml-webgpu-shader-lib.hpp`, `wgsl-shaders/row_norm.wgsl`).

### 10. ggml-webgpu: split LAYER_NORM accumulation loop from legacy norms

The patch 9 LAYER_NORM addition hoisted the inner load into
`let v = src[...]` and replaced `pow(v, 2.0)` with `v * v`, which is
correct for LAYER_NORM but also subtly changed codegen on the
RMS_NORM / L2_NORM path that shader already used. This patch splits
the inner accumulation so the LAYER_NORM branch keeps the
single-load form (needed for `sum_x`) while the RMS_NORM / L2_NORM
branch reverts to the original `pow(src[...], 2.0)` form. Behavior
is byte-identical to the pre-LAYER_NORM shader on the legacy paths.

### 11. ggml-webgpu: fix UB shift-by-32 in `load_u32_at_src{,0}`

The unaligned u32 helpers in `wgsl-shaders/common_decls.tmpl` compute
`hi << (32u - shift)` where `shift = (byte_offset & 0x3u) * 8u`. When
`byte_offset` is u32-aligned (`shift == 0`), this becomes `hi << 32u`,
which is undefined behavior in WGSL (shift count must be < bit_width).
The trailing `select(shifted, lo, shift == 0u)` was meant to mask the
UB result, but on Tint/Dawn the UB leaks and corrupts the returned
word on aligned reads. This corrupted Q3_K mul_mat_vec and Q3_K
get_rows for Q3_K_M models; Q4_K_S happened to issue unaligned loads
in the affected lanes and was unaffected. The fix branches explicitly
on `byte_in_word == 0` and returns the aligned word directly, never
executing the UB shift. Verified: Mistral-7B Q3_K_M coherent at
24.4 tok/s (was gibberish); Q4_K_S regression-safe at 36.0 tok/s.
Closes webllm bug #28.

### Note: FA browser engagement (2026-04-25)

The 2026-04-25 rebase pulled in upstream `13d36cf89`
("ggml-webgpu: enable FLASH_ATTN_EXT on browser without subgroup
matrix"). On Qwen3-1.7B decode shapes (N=1, head_dim 128, GQA 16:8,
K=2048+, mask present) the new
`ggml_webgpu_flash_attn_get_decisions` returns the
SUBGROUP_MATRIX path; browser hits the `supports_op = false`
branch and falls back to the manual attention path (multi-step
softmax + KV reads). Dispatch count and `backendAttentionMs`
on profile-mode runs are unchanged from pre-rebase.

The new VEC and TILE paths apparently target different shape
regions (longer K, prefill seq>1, different head_dim). Engaging
FA on this workload requires investigation into
`ggml_webgpu_flash_attn_get_decisions` to understand which
heuristics gate it out — see `TODO.md` Active Step §6 path (a).
**Prefill (seq>1) was not measured** — the browser `--profile`
trace filter is `nTokens=1`, so FA *might* engage on prefill
without us seeing it in the dispatch count.

## Rebase Procedure

To pick up a newer upstream `llama.cpp` master:

```bash
cd ~/Repos/llama.cpp
git checkout webllm-browser-patches

# Safety net — so you can recover if resolution goes sideways
git branch "webllm-browser-patches-backup-$(date +%Y%m%d-%H%M%S)"

git fetch origin
git rebase origin/master
# Conflicts land almost exclusively in:
#   ggml/src/ggml-webgpu/ggml-webgpu.cpp
# Secondary files to watch:
#   ggml.c
#   ggml/include/ggml-webgpu.h
```

Then, back in the WebLLM repo:

```bash
make wasm-build
make bench-inference      # perf regression check
# Open smoke-test/real-model.html in Chrome and run the browser smoke test
```

Both the perf regression check and the browser smoke test must pass
before pushing.

### Known conflict shape: `shader_gpu_time_ms`

Our commit 6 (opt-in browser graph profiling) was authored against an
older upstream where `shader_gpu_time_ms` lived on `global_ctx`.
Upstream has since moved it back onto the per-context struct, so
`ctx->global_ctx->shader_gpu_time_ms` reads from our patch must be
rewritten to `ctx->shader_gpu_time_ms`. The `profile_timestamps_active`
field our patch adds is orthogonal and must be kept — merge both sides
rather than picking one.

## Troubleshooting

**Untracked `kompute/` subtree.** The local `~/Repos/llama.cpp` checkout
may show an untracked `ggml/src/ggml-kompute/kompute/` path. That's a
separate upstream / sub-repo artifact, not part of this project's patch
work, and can be ignored when assessing whether the dependency is clean.

**Regression after a rebase.** If a browser regression reappears after
rebasing, inspect the local branch *before* assuming the bug is in
WebLLM. Start with the three files above — the four patches all touch
them or their call paths.

**Keep `-sASYNCIFY_STACK_SIZE=1048576` in the WASM build** unless there
is a verified replacement strategy. The browser readback path relies on
it.

## Related Documentation

- [`CLAUDE.md`](../CLAUDE.md) — repo guidance + regression lessons-learned
- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — benchmark methodology and metrics
