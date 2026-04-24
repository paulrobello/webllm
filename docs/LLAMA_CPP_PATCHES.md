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

The branch currently carries eight commits on top of upstream `master`,
in the order shown (oldest first). Commit 7 and its revert (commit 8)
are kept as a pair pending a proper replacement; treat them as a no-op
until you hear otherwise.

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
