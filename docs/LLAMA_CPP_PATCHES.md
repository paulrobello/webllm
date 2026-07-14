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

The branch currently carries **ten commits** on top of upstream
`master`, in the order shown (oldest first). Commit 7 and its revert
(commit 8) are kept as a pair pending a proper replacement; treat them
as a no-op until you hear otherwise. Commit 10 (wgpu::WaitAny under
JSPI) was added 2026-05-12.

Last rebased onto upstream master 2026-07-14 to tip `bf2c86ddc`
("server : refactor prompt cache state ownership"). Local tip:
`18ee82988`. The 2026-05-12 → 2026-07-14 delta was the largest rebased
to date: 883 commits, 37 touching `ggml/src/ggml-webgpu/` or
`ggml/include/` — FA refactor #23834, MMVQ + legacy MUL_MAT cleanup
#23594 (deleted `mul_mat.wgsl`), k-quant matmul refactor #24225,
i-quant mul_mat perf #24530, profiling timestamp flush #22995,
batch_compute_passes guard #23457, concat aliasing #24000. 10 patches
replayed with **2 conflict rounds** (patch 2 browser+ASYNCIFY bundle:
`batch_compute_passes` gating + deferred-copies merge; patch 6
profiling: capability fields, dispatch `#ifdef` early-return
restructure, `profile_timestamps_active` runtime gating, buffer-init
`supports_timestamp_query` guard, overflow-flush 4-arg signature) + 2
post-rebase compile fixups (`18ee82988`); 8 patches clean.

Sweep classification: **§27 (perf-neutral maintenance rebase)**. 6/6
canonical models within ±1.0% of the same-day pre-rebase baseline,
zero regression. Encoder G3 cosine-0.76 parity VERIFIED via chat-smoke
`[8/8]` (embed('happy')·embed('joyful') cosine=0.76, ‖v‖=1.00 — matches
baseline, encoder numerics unchanged). `make checkall` clean
(782/0/36 skip). Full sweep matrix + conflict log at
[`eval/reports/llama-cpp-rebase-2026-07-14/SUMMARY.md`](../eval/reports/llama-cpp-rebase-2026-07-14/SUMMARY.md);
same-day pre-rebase baseline at
[`eval/reports/pre-rebase-baselines-2026-07-14/`](../eval/reports/pre-rebase-baselines-2026-07-14/).
Safety branch `webllm-browser-patches-backup-2026-07-14` preserves
pre-rebase tip `4192e05ba`.

#### Earlier rebase (2026-05-12, §32)

Rebased onto upstream master 2026-05-12 to tip `856c3adac`. 102 commits
ahead; 2 ggml-webgpu landings (#22906 mulmat-q refactor for gpt-oss-20b,
#22808 multimodal precision/FA rework). 10 effective patches reapplied
with zero manual conflicts; 18 JSEP probe commits dropped per the
2026-05-08 negative-closure. Sweep classified §32 (small regression,
accepted): Pass 2 sweep tok/s −2% to −11%, matmul +4-9%, thermal
contamination significant. Tip `4192e05ba`. Sweep matrix at
[`eval/reports/llama-cpp-rebase-2026-05-12/SUMMARY.md`](../eval/reports/llama-cpp-rebase-2026-05-12/SUMMARY.md).

#### Earlier rebase (2026-05-04, §27 hybrid)

Rebased onto upstream master 2026-05-04 to tip `a817a22bc`
("ggml : implement fast walsh-hadamard transform for kv rotation").
Local tip: `fc1f81242`. The 2026-05-01 → 2026-05-04 delta was small;
1 upstream commit touched `ggml-webgpu/`: `d4b0c22f9` (LayerNorm ops
#22406), which **subsumes** our local patches `72b6d001e`
(`GGML_OP_NORM` support) and `c775ac26d` (LAYER_NORM split). Both
local patches **dropped** during the rebase; the 9-patch stack
replayed cleanly with zero conflicts.

Sweep classification: **§27 hybrid (maintenance free win, perf
neutral)**. Encoder-parity gate PASS at cosine 0.76 vs 0.76
reference (synonym pair "happy" / "joyful" on arctic-embed-s, ‖v‖
1.00) — upstream's two-pass mean-then-variance algorithm is more
numerically stable than our single-pass `Σx²/n − mean²`, but the
difference is in the f32 noise floor. `bun test` clean (741/0/33
skip baseline). Cross-day perf vs 2026-05-01 baseline: noise-level
on 5/6 models; mistral-7b -14% outlier flagged for next-session
rerun (single-data-point cross-day not enough to declare a real
regression). Full sweep matrix at
[`eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md`](../eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md);
same-day pre-rebase control at
[`eval/reports/pre-rebase-baselines-2026-05-04/`](../eval/reports/pre-rebase-baselines-2026-05-04/)
(captured anomalously cold; SUMMARY documents the
"same-day-floor-anchored-on-noise" failure mode as a process
lesson).

#### Earlier rebase (2026-05-01, §27)

Rebased onto upstream master 2026-05-01 to tip `b97ebdc98`
("llama-quant : fix --tensor-type when default qtype is overridden").
The 2026-04-29 → 2026-05-01 delta was small; 2 of the upstream commits
touched `ggml-webgpu/`: `c3c150539` (mul-mat / mul-mat-id vectorize
fix #22578 — load-bearing for chat decode) and `aab68217b` (upscale
shader #22419 — image-gen op, not exercised by the chat fleet). The
rebase replayed all 11 patches with **one trivial conflict**: both
upstream `aab68217b` (UPSCALE) and our local browser bundle
(`9b009201f`, the rebased `7bbb2416c`) added new `case` arms to the
same `switch (op)` in `ggml-webgpu.cpp`. Resolved by keeping both
arms. WASM sizes: wasm32 2,482,377 bytes, wasm64 2,526,224 bytes.
Ship gate 513/12/0. Post-rebase tip: `e29753286`.

Sweep classification: **§27 template — broad free win, no
regressions**. Same-day same-tip pre-rebase control captured per
§32a doctrine; comparing post-rebase against that baseline (NOT the
4-day-old 2026-04-28 matrix — environmental floor drifted -3 to
-8% in profile mode over the gap), every model improved or held
(+0.4% to +8.0% tok/s) and matmul median time decreased uniformly
(-1.4% to -5.0%). The win attributes cleanly to `c3c150539`'s
vectorize-corner fix on the per-decode mat-vec kernel. Full sweep
matrix at
[`eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md`](../eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md);
same-day pre-rebase control at
[`eval/reports/pre-rebase-baselines-2026-05-01/SUMMARY.md`](../eval/reports/pre-rebase-baselines-2026-05-01/SUMMARY.md).

#### Earlier rebase (2026-04-29, §32)

Rebased onto upstream master 2026-04-29 to tip `b1d5f5b44`
("sync : ggml"). The 2026-04-28-eve → 2026-04-29 delta was 14 commits;
1 of them touched `ggml-webgpu/` (#22492 FlashAttention support-check
fix — adds an early-out when `decisions.path == FLASH_ATTN_PATH_NONE`
plus a "set path to none if kv_tile doesn't fit" safety net). The
rebase replayed all 11 patches cleanly with zero conflicts. WASM
sizes: wasm32 2,249,421 bytes (-229 vs §32 baseline), wasm64
2,292,124 bytes. Ship gate 452/11/0. Safety branch
`webllm-browser-patches-backup-20260429-*` preserves the pre-rebase
tip (`3b8ade2a2`).

Sweep classification: **§32 template — small regression, accepted;
don't revert**. 4 of 6 models show −3% to −5% post-rebase profile-mode
deltas; 2 of 6 (both IQ3_M) are flat within noise. The single relevant
upstream change (`d6a509400`) lives inside the cold-path
`ggml_backend_webgpu_device_supports_op` (per graph build, not per
dispatch), so the deltas are most plausibly 3-run profile-mode noise
rather than a real per-token regression. Net read: stay current; the
option value of clean future free-win wins outweighs the noise-band
delta. Full sweep matrix at
[`eval/reports/llama-cpp-rebase-2026-04-29/SUMMARY.md`](../eval/reports/llama-cpp-rebase-2026-04-29/SUMMARY.md).

#### Earlier rebase (2026-04-28-eve, §32)

Rebased onto upstream master to tip `f9f33654a` ("vulkan: Coalesce
Q4_K/Q5_K scale loads (#21751)"). The 2026-04-27 → 2026-04-28-eve
delta was 10 commits; 1 of them touched `ggml-webgpu/` (#22456 buffer
aliasing refactor for `ssm_scan` — helper `webgpu_tensor_offset`
renamed to `ggml_webgpu_tensor_offset` and `view_offs` folded into the
helper body). The rebase replayed all 11 patches cleanly but the
renamed helper produced a compile error in patch 3 (request-based
browser readback API still referenced the old name). The §32 rebase
initially landed the rename adoption as a forward fix-up (patch 12)
to avoid history rewriting on the long-lived branch. **Patch 12 was
subsequently squashed back into patch 3** (2026-04-28, post-§31b
cleanup pass) — patch 3's diff for the affected line now reads
`ggml_webgpu_tensor_offset(tensor) + offset` directly. WASM
byte-identical pre/post squash (2,249,650 bytes); ship gate 428/11/0
unchanged. Safety branches preserved:
`webllm-browser-patches-pre-rebase-2026-04-28-eve` (pre-§32 tip
`981859864`), `webllm-browser-patches-pre-squash-2026-04-28`
(pre-squash tip `c4af89356`).

WASM rebuild + checkall (428/11/0) + 6-model bench sweep verified
post-§32. Sweep result: 5 of 6 models within ±5% noise band of §27
post-rebase baselines; **llama-3.1-8b-iq3m regresses ~6%** (29.0 →
27.2 tok/s, consistent across 5 runs). Likely cause is #22456's
aliasing-logic refactor interacting with GQA + IQ3_M kernels — same
quant on qwen3-8b-iq3m (different attention shape) is essentially
flat. See `eval/reports/llama-cpp-rebase-2026-04-28-eve/SUMMARY.md`
for the full sweep matrix.

The 2026-04-26 → 2026-04-27 delta (§27) was 13 commits; 3 of them
touched `ggml-webgpu/` (Q1_0 kernel addition #22374, fast i-quant
mat-vec kernels #22344, performance-portable register-tile / subgroup
matmul tuning #22241) — zero conflicts, all 11 patches replayed
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

### 9. ggml-webgpu: fix UB shift-by-32 in `load_u32_at_src{,0}`

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

### 10. ggml-webgpu: use wgpu::WaitAny under JSPI instead of polling loop

Replaces the `emscripten_sleep(1)` busy-wait polling loop in the
request-based async readback path (patches 3–5) with `wgpu::WaitAny`
under JSPI, so the browser event loop blocks efficiently on queue
completion instead of polling. Added 2026-05-12. Depends on JSPI being
enabled in the WASM build (`-sJSPI` / `GGML_WEBGPU_JSPI`); the TS
binding for the readback completion awaits the JSPI-promised export
(see the CLAUDE.md `JSPI_EXPORTS` regression lesson — only exports
whose TS bindings await the result belong on the list).

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
