# Stage 4.16 — H1-vs-divert handle/offset cross-correlation + EM_ASYNC_JS fix

**Status:** PARTIAL OUTCOME A — H1 fire-and-forget Promise bug found and fixed; decode flips from stuck-at-593 ("ntiuhuihnerquant") to varied output ("inonic boso-") with confident logits. Not yet "Paris" — a downstream bug remains; queue Stage 4.17 to localize.

**Date:** 2026-05-07
**Patch stack:** llama.cpp `webllm-browser-patches` 11 → 12 (P9 — `EM_ASYNC_JS` for `ggml_jsep_read`).

## TL;DR

The Stage 4.15 disambiguation matrix was a partial map. **All four documented sub-hypotheses (mirror-mismatch / offset-mismatch / undersized-read / fires-on-different-node) were RULED OUT** by Probe 6: H1 fires for the right node (Qcur-0), with the right `bctx_handle` (26), the right `dst_offset` (4194304), and the right `dst_size` (49152). Cross-correlated 1:1 against Probe 5's divert-write entry for the same `(handle=26, offset=4194304, ne=[2048,6])`. Per the Stage 4.16 brief's risk register, this collapsed onto the **fifth outcome**: "the bug moves to JS-side `Module.jsepRead`'s record fetch or in JSPI ordering."

Rather than chase JSPI ordering, I added a tiny instrumentation tweak — `mirror_post_h1`: a 4-float peek at `host_mirror[bctx->handle]+dst_offset` issued via `EM_ASM` *immediately* after the H1 `ggml_jsep_read` call returned. Smoking gun: for the load-bearing Qcur-0 fire (idx=1), the peek reads `[0, 0, 0, 0]` — H1 had returned but host_mirror at the H1's own `(handle, offset)` was still post-allocation zeros. **`ggml_jsep_read` was declared `EM_JS(void, ...)` not `EM_ASYNC_JS`, so under JSPI the JS body's returned Promise was never awaited; the readback ran asynchronously and host_mirror updates landed AFTER the C++ caller had moved on.** A comment at the EM_JS site claimed "the await happens implicitly because this EM_JS site executes from a JSPI-promised export" — that's incorrect. JSPI suspends an export only when it calls a *promising* import; EM_JS imports are not promising unless explicitly marked, which is what `EM_ASYNC_JS` does (it routes the body through `Asyncify.handleAsync`, which Emscripten 5.0.7 lowers onto `WebAssembly.Suspending` when JSPI is enabled).

Fix: change `EM_JS(void, ggml_jsep_read, ...)` → `EM_ASYNC_JS(void, ggml_jsep_read, ...)` and `await` the inner `Module.jsepRead` call. Patch stack: 11 → 12.

After fix:
- Decode no longer stuck at `topId=593/topVal=0.159`; new `topId=297/topVal=10.46` with **finiteCount=32000, no NaN/Inf**.
- `LOGIT_STATS_STEP0.first8 = [-8.39, -8.11, 1.14, -5.41, -5.62, -4.41, -6.30, -7.71]` — real distinct logits across step 0.
- `GENERATED_TOKENS = [297, 8927, 13601, 29877, 29899]` (was `[593, 5871, 15669, 15565, 12150]` — now 5 distinct values from a different distribution).
- `GENERATED_TEXT = "inonic boso-"` (was `"ntiuhuihnerquant"`). Both gibberish; the new output exhibits English-letter morphology, the old did not.
- Per-token decode 458.5 ms (vs Stage 4.15 baseline 107.7 ms / Stage 4.5 baseline 25.04 ms; ~18× regression vs 4.5). Expected — H1 now actually waits per-runOp for ~1602 GPU readbacks per prefill+5-decode. Optimization (dirty-bit, batched readback, peeled consumer-only) deferred to Stage 5+ now that correctness is on the table.
- `COUNTER_DELTAS = {alloc:0, free:0, write:4404, read:1602, clear:0, runOp:1602, sync:3671}` — `read` count unchanged from Stage 4.15 (the JS counter increments synchronously inside `module.jsepRead` regardless of whether the Promise is awaited; the bug was that nothing waited *for* the Promise, not that the Promise was never created).

## Probe 6 — implementation

C++ instrumentation added inside the H1 block in `ggml_backend_jsep_graph_compute` (`ggml-jsep.cpp:779-786`), guarded by `#ifdef __EMSCRIPTEN__`. Per H1 fire (capped at 256 entries), records:

```js
{
  idx,             // sequential probe-fire counter
  node_op,         // node->op
  bctx_handle,     // the handle H1 reads from
  dst_offset,      // jsep_tensor_handle(node) — H1's offset
  dst_size,        // ggml_nbytes(node) — H1's read size
  tensor_name,     // node->name
  view_src_handle, // node->view_src ? vctx->handle : -1
  view_src_offs,   // node->view_offs (or -1)
  mirror_post_h1,  // 4 f32 from host_mirror[handle]+dst_offset, AFTER H1 returns
}
```

`mirror_post_h1` was the load-bearing addition that flipped sub-hypothesis selection from #1-#4 (handle/offset/size mismatch) to #5 (JSPI ordering). Without that peek, the matrix would have closed inconclusive — the C++-side and JS-side handle/offset values were structurally identical.

Probe 6 has been removed from the source as part of the fix commit; the structural fix supersedes it. Probe 5 remains in `src/inference/jsep/ops/matmul.ts` (gated on `__stage415DivertProbe`) since it has zero cost when ungated and is a permanent regression check.

## Cross-correlation table — Q/K projection load-bearing window

| Probe 5 (divert write) | | | Probe 6 (H1 read) | | | Match |
|---|---|---|---|---|---|---|
| idx | (h, o) | ne / size | idx | (h, o) | size / name | |
| 1 | (26, 4194304) | [2048,6,1,1] / 49152 | 1 | (26, 4194304) | 49152 / "Qcur-0" | ✓ exact |
| 2 | (26, 4194304) | [256,6,1,1] / 6144 | 2 | (26, 4194304) | 6144 / "Kcur-0" | ✓ exact |
| 3 | (26, 6295552) | [256,6,32,1] / 196608 | 5 | (26, 6295552) | 196608 / "kq-0" | ✓ exact (across SET_ROWS in between) |

Pre-fix `mirror_post_h1` at idx=1 (Qcur-0): `[0, 0, 0, 0]` — H1 fire-and-forget, host_mirror unchanged.
Pre-fix `mirror_post_h1` at idx=0 (norm-0 in-place): `[-0.0013, 0.0019, -0.0019, 0.0038]` — masked the bug for in-place ops where host_mirror was pre-populated by `set_tensor` at model load.

## Disambiguation matrix outcome

| Sub-hypothesis | Predicted finding | Observed |
|---|---|---|
| H1-mirror-mismatch | bctx_handle ≠ dst.bufHandle | RULED OUT (both 26) |
| H1-offset-mismatch | dst_offset ≠ dst.offset | RULED OUT (both 4194304) |
| H1-undersized-read | dst_size < bytes_written | RULED OUT (both 49152) |
| H1-fires-on-different-node | MUL_MAT absent in Probe 6 | RULED OUT (Probe 6 idx=1 IS Qcur-0 MUL_MAT) |
| **Fifth outcome — JSPI ordering** | **all match but data not landing** | **CONFIRMED via `mirror_post_h1`** |

## Root-cause analysis — JSPI + EM_JS contract

The wasm-side `ggml_jsep_read` was declared:

```cpp
EM_JS(void, ggml_jsep_read, (int32_t handle, int32_t offset, int32_t host_ptr, int32_t size), {
    if (Module.jsepRead) {
        // Promise-returning under JSPI; the await happens implicitly because
        // this EM_JS site executes from a JSPI-promised export.
        return Module.jsepRead(handle, offset, host_ptr, size);
    }
})
```

The comment is wrong. Under JSPI:

1. `WebAssembly.promising`-wrapped exports (listed in `JSPI_EXPORTS`) suspend when they call a `WebAssembly.Suspending`-wrapped import.
2. `EM_JS` imports are not auto-Suspending. The body executes as a plain JS function; if it returns a Promise, that Promise is the C function's return value cast to the declared return type. For `void`, the return is discarded.
3. `EM_ASYNC_JS` (in `<emscripten/em_js.h>:73`) wraps the body with `Asyncify.handleAsync(async () => <body>)`, which Emscripten lowers to `WebAssembly.Suspending` under `-sJSPI=1` (the build's mode — see `src/wasm/CMakeLists.txt:223`'s `-sJSPI_EXPORTS=...`).

The fix:

```cpp
EM_ASYNC_JS(void, ggml_jsep_read, (int32_t handle, int32_t offset, int32_t host_ptr, int32_t size), {
    if (Module.jsepRead) {
        await Module.jsepRead(handle, offset, host_ptr, size);
    }
})
```

`Module.jsepRead` already returns `Promise<void>` from `dataManager.readAsync`. The `await` makes the wasm-side caller suspend until the readback's `mapAsync` completes and `dest.set(mapped)` writes into `module.HEAPU8.buffer` (which IS `host_mirror`'s underlying memory).

## Why H1-inverse (jsepWrite) is unaffected

`ggml_jsep_write` is also `EM_JS(void, ...)` but `Module.jsepWrite` calls `dataManager.write` → `device.queue.writeBuffer`, which is **synchronous** from JS's perspective (the queued write doesn't return a Promise; WebGPU validates and stages internally). No await needed. ✓

## Outcome verdict

**PARTIAL Outcome A.** Logits and decoded tokens are now real-but-incorrect, vs Stage 4.15's stuck-at-confident-wrong. The fix is structurally correct and load-bearing. The remaining bug is downstream — likely either:

1. **Kernel correctness at production shape.** Q4_K matmul self-test at K=2048 PASSED in Stage 4.3, but additional production shapes (e.g., output projection M=2048,K=2048; ffn_gate K=5632; lm_head [vocab,K=2048]) haven't been exhaustively self-tested. A small numerical bias would compound through 22 layers and produce coherent-letter gibberish.
2. **Cross-backend boundary not fully sealed.** With H1 + H1-inverse both firing properly, host_mirror should be coherent. But H1-inverse only syncs `node->src[s]` tensors per runOp. If the scheduler emits a JSEP runOp whose actual GPU input is updated by a CPU op AFTER H1-inverse's pre-pass (e.g., between graph_compute slices via direct host_mirror writes that don't trigger ggml_jsep_*), the JSEP read would still pick up stale GPU data.
3. **CPU-fallback ROPE / SOFT_MAX correctness.** TinyLlama's ROPE base/freq parameters or position handling may have a subtle bug in the CPU backend's interaction with the JSEP-resident KV cache. Stage 4.13's "K and V slot labels swapped" diagnosis was retracted by Stage 4.14, but the underlying scheduling pattern bears re-examination now that data actually flows.

Stage 4.17 brief (next session): per-attention-output readback at end of each layer (RMS_NORM(attn_out), pre/post-layer-norm activations) compared against a CPU-only reference run. Cheapest probe to localize the divergence.

## Files changed

- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:149-163` — `EM_JS(void, ggml_jsep_read, ...)` → `EM_ASYNC_JS(void, ggml_jsep_read, ...)` with `await` on `Module.jsepRead` and a multi-line comment explaining the bug and the discovery path. Patch stack: 11 → 12.
- (transient) `ggml-jsep.cpp:779-786 H1 block` — Probe 6 EM_ASM with `mirror_post_h1` peek added to confirm fifth-outcome disambiguation. **Removed in the fix commit** since the structural fix supersedes it; Probe 5 (JS-side gated on `__stage415DivertProbe`) remains as a permanent regression check.

## Build + checkall

`make wasm-build-jsep` green. `make checkall` green:
- `bun run fmt` — 137 files, no fixes
- `bun run lint` — 137 files, no fixes
- `bun run typecheck` + `bun run typecheck:tests` — clean
- `bun test` — 747 pass / 36 skip / 0 fail / 39170 expect() / 783 tests across 82 files in 2.43 s

## Patch stack

llama.cpp `webllm-browser-patches`:
- P1-P6 (Phase 2 + Stages 1, 1.5, 2, 4.4 F1, 4.5 H1, 4.9 H1-inverse, 4.11/4.12/4.13/4.14 diagnostics) — unchanged.
- **P9 (Stage 4.16): `EM_ASYNC_JS` for `ggml_jsep_read`** — 1-LOC structural fix + comment.

11 → 12 patches. Diff is 14 lines (added comment + macro change + await keyword).

## Decode performance regression — accepted, deferred

| Stage | Per-token | Δ vs prior | Reason |
|---|---:|---|---|
| 4.5 (H1 in place, fire-and-forget) | 25.04 ms | baseline | H1 doesn't actually block |
| 4.9 (H1-inverse pre-pass added) | 131.80 ms | +5.3× | per-runOp host→GPU upload of every src |
| 4.13/4.14 (instrumentation overhead) | 127.40 ms | within noise | |
| 4.15 (Probe 5 deferred mapAsync, lands outside decode timing) | 107.7 ms | -16% | mapAsync drains post-decode |
| **4.16 (EM_ASYNC_JS — H1 actually awaits)** | **458.52 ms** | **+18× vs 4.5** | per-runOp synchronous GPU readback (~1602 fires per token) |

The regression is correctness tax, not a kernel issue. Stage 5+ optimizations (dirty-bit tracking on host_mirror writes, batched readback at slice boundaries, peeled-consumer-only sync) are queued behind Outcome A confirmation.
