# llama.cpp rebase + sweep cycle — 2026-07-14

**Classification: §27 — perf-neutral maintenance rebase (6/6 models within ±1%, zero regression).** New local tip `18ee82988` on upstream `bf2c86ddc`.

Rebased `webllm-browser-patches` from upstream `856c3adac` (2026-05-12
cycle, local tip `4192e05ba`) to `bf2c86ddc` (2026-07-14). 883 commits
ahead overall; **37 touched `ggml/src/ggml-webgpu/` or `ggml/include/`**.
New local tip: `18ee82988` (10 patches + 1 fixup for the profiling
conflict resolution). This is the largest gap rebased across in the
project's history (prior cycles were 10–14-commit deltas).

## Patch stack

10 patches replayed onto `bf2c86ddc`; **2 conflict rounds**, both
resolved; 8 patches applied cleanly. Safety branch
`webllm-browser-patches-backup-2026-07-14` preserves pre-rebase tip
`4192e05ba`.

| # | patch | result |
|---|-------|--------|
| 1 | `cc95e5bf0` iterative `ggml_visit_parents_graph` (WASM stack safety) | clean |
| 2 | `b7d18d742` browser + ASYNCIFY bundle (batch_compute_passes, overlap conflict-detection, DIAG_MASK_INF) | **conflict — resolved** |
| 3 | `995155a21` request-based browser readback API | clean |
| 4 | `a9ae1a515` harden async readback cleanup | clean |
| 5 | `4bd244985` notify browser async readback completion | clean |
| 6 | `a0507d65a` opt-in browser graph profiling | **conflict — resolved** |
| 7 | `95a0f3379` specialize browser decode matmul dispatch | clean |
| 8 | `af4cdc584` Revert "specialize browser decode matmul dispatch" | clean |
| 9 | `28408d023` fix UB shift-by-32 in `load_u32_at_src` | clean |
| 10 | `2b8f68b99` `wgpu::WaitAny` under JSPI | clean |
| fixup | `18ee82988` correct profiling dispatch after rebase | post-rebase compile fixes |

## Notable upstream ggml-webgpu landings

- `e8c54893f` FlashAttention refactor + quantization standardization (#23834) — 986+/950− across 11 files; rewrote `flash_attn.wgsl` (271 lines), `shader-lib.hpp` (659), `ggml-webgpu.cpp` (416).
- `1506d39e7` MMVQ path for Q4/Q8/Q2_K/Q4_K + legacy MUL_MAT cleanup (#23594) — 714+/931−; **deleted `mul_mat.wgsl` (747 lines)**.
- `1e1aca09d` k-quant prefill + matmul refactor for Q4/Q5/Q8 (#24225) — 267+/543−.
- `6e9007ae6` i-quants mul_mat perf + prefill speedup (#24530).
- `527045bfb` flush gpu profile timestamp before queryset overflow (#22995) — 10+.
- `54121f732` batch_compute_passes guard (#23457) — 22+/13−.
- `1705d434f` concat buffer overlap/aliasing (#24000) — 79+/33−.
- `5ec717d12` / `32e41fa5b` / `4c1c3ac09` flash-attn-vec subgroup tuning.

## Conflict resolutions

### Patch 2 (`b7d18d742`, browser + ASYNCIFY bundle) — 2 regions in `ggml-webgpu.cpp`
- **Dispatch loop (`build_multi`):** upstream `54121f732` restructured to gate on `ctx->batch_compute_passes` (context-member bool, line 263) inside one loop; our patch gated on `if (ctx->active_compute_pass)` across two loops **and** added the load-bearing `__EMSCRIPTEN__` deferred-copies loop (overlap-only conflict detection — patch 2 item 3, the +28% decode win). Merged: adopted upstream's `ctx->batch_compute_passes` gating **and** preserved the deferred-copies loop in the else branch.
- **`graph_compute` local var:** upstream removed our local `bool batch_compute_passes = true;` (now a context member, default `true`; `graph_compute` uses `ctx->` at 3468/3495). Took HEAD (dropped the dead local + its comment).

### Patch 6 (`a0507d65a`, opt-in graph profiling) — 4 regions in `ggml-webgpu.cpp`
- **Capability struct:** kept both `supports_dot_product` (upstream) and `supports_timestamp_query` (ours, used at 4400/4402/4508).
- **Dispatch `#ifdef` (`build_multi`):** upstream's compile-time `#ifdef/#else/#endif` would dispatch nothing when profiling compiled-but-inactive; our patch 6 gates profiling at **runtime** via `profile_timestamps_active`. Resolved with early-`return result` from the profile path + `#endif` (non-profile path always compiles). *(Post-rebase compile fix `18ee82988` — the initial "take HEAD" left the `#ifdef` unterminated; the runtime gate is load-bearing for `eval/perf.ts --profile`.)*
- **`graph_compute` profile state:** kept `ctx->profile_timestamps_active = collect_gpu_profile;` + `ctx->batch_compute_passes = !collect_gpu_profile;` (bare local → `ctx->` member).
- **Profiling buffer init:** merged upstream's `batch_compute_passes = false` (init) with our `if (supports_timestamp_query)` guard (browser-safety: don't create timestamp buffers on devices without support).
- **Overflow-flush call:** upstream `527045bfb`'s call site (3 args) updated to pass `&graph_profile` (4th arg) matching our 4-arg `collect_profile_results` signature. *(Post-rebase compile fix `18ee82988`.)*

The patches-doc "known conflict" `shader_gpu_time_ms` was a **non-issue** this
cycle: all reads use the per-context `ctx->shader_gpu_time_ms` (line 349); no
`global_ctx->` references remained (upstream's move + our code align cleanly).

## Pre-rebase baseline (Phase 1, profile-mode, 2026-07-14, local tip `4192e05ba`)

| Model | tok/s (median of 3) |
|-------|--------------------:|
| tinyllama-1.1b-chat-q4_0 | 87.1 |
| qwen3-0.6b-q4f16 | 72.0 |
| qwen3-1.7b-q4f16 | 52.1 |
| mistral-7b-instruct-v0.3-q4ks | 29.7 |
| llama-3.1-8b-instruct-iq3m | 22.8 |
| qwen3-8b-iq3m | 21.3 |

All exit 0. qwen3-1.7b run-3 dip (37.4 vs 52.1) is the thermal-contamination
signature noted in the 2026-05-12 cycle.

## Post-rebase verification (Phase 4)

- **WASM build:** both wasm32 + mem64 compile clean against rebased `bf2c86ddc`.
- **`make checkall`:** 782 pass / 36 skip / 0 fail (818 tests); fmt + lint + typecheck + typecheck:tests clean.
- **TinyLlama Q4_0 profile smoke:** 86.2 tok/s (vs pre-rebase 87.1, −0.9 / −1.0%, noise). Profiling attribution populated (`backendMatmulMs` 6.25, `backendEncodeOverheadMs` 1.60, `backendAttentionMs` 0.60, 450 dispatches/token) — confirms the patch-6 `#ifdef`/signature fix works at runtime.
- **Encoder (arctic-embed-s):** embed-perf single-short p50 3.60ms, single-long 17.00ms, batch 98.8 texts/sec, exit 0. **G3 cosine-0.76 parity VERIFIED** via the chat-smoke `[8/8]` step — `embed('happy')·embed('joyful') cosine=0.76`, ‖v‖=1.00 (matches the 0.76 baseline within tolerance; not the identical-vectors >0.999 bug). Encoder numerics unchanged post-rebase.

## Post-rebase sweep (Phase 5) + classification

Same-day same-environment deltas vs the Phase 1 baseline above (profile-mode,
median of 3 runs, exit 0 on every model):

| Model | pre-rebase | post-rebase | Δ tok/s | Δ % |
|-------|-----------:|------------:|--------:|----:|
| tinyllama-1.1b-chat-q4_0 | 87.1 | 87.3 | +0.2 | +0.2% |
| qwen3-0.6b-q4f16 | 72.0 | 71.7 | −0.3 | −0.4% |
| qwen3-1.7b-q4f16 | 52.1 | 51.9 | −0.2 | −0.4% |
| mistral-7b-instruct-v0.3-q4ks | 29.7 | 30.0 | +0.3 | +1.0% |
| llama-3.1-8b-instruct-iq3m | 22.8 | 22.7 | −0.1 | −0.4% |
| qwen3-8b-iq3m | 21.3 | 21.2 | −0.1 | −0.5% |

**Classification: §27 — perf-neutral maintenance rebase.** All 6 models
within ±1.0% of the same-day pre-rebase baseline; zero regressions, zero
improvements. The upstream matmul/FA refactors (`1506d39e7`, `1e1aca09d`,
`e8c54893f`) and the i-quant perf landing (`6e9007ae6`) are perf-neutral on
the canonical fleet — pipeline reorganization, not algorithmic change.
Adopt the new baseline (`bf2c86ddc` / local tip `18ee82988`); staying
current preserves option value (next cycle's free wins land cleanly). No
follow-up work needed. Encoder cosine-0.76 G3 parity verified (cosine=0.76
via chat-smoke `[8/8]`, matching baseline).

## Re-evaluation triggers
- Next upstream cadence check fires when `git log webllm-browser-patches..origin/master -- ggml/src/ggml-webgpu/ ggml/include/` is non-empty again.
- Encoder cosine-0.76 parity: verified 2026-07-14 (cosine=0.76 via chat-smoke `[8/8]`). Re-check if a numerical-shift signal appears.
