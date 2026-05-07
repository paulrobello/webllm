# Phase 3 / Option A-prime — Stage 4.9 result

**Date:** 2026-05-07

**One-line outcome:** H1-inverse landed (host→GPU writeback per-runOp pre-pass)
and works exactly as designed; it does not flip Outcome A. **Outcome C-2:** at
the FIRST 2 SET_ROWS dispatches (prefill i=3 + first decode-step's analogue),
`host_mirror[h26+0..6144]` is itself all zeros — the CPU CPY+ROPE chain that
should populate `h26o0` with the K-projection-after-ROPE data has not written
there at that point in time. The writeback gap was real but is not load-bearing
for the all-zero KV-cache collapse. Stage 4.10 brief (active in `TODO.md`)
queues localization of WHY host_mirror is stale.

## Headline metrics

| Marker | Stage 4.5 baseline | Stage 4.9 |
|---|---|---|
| `LOGIT_STATS_STEP0.topId/topVal` | `593 / 0.159` | **`593 / 0.159` (bit-identical)** |
| `LOGIT_STATS_STEP0.first8` | `[0.0060, 0.0047, ...]` | **bit-identical** |
| `GENERATED_TOKENS` | `[593, 5871, 15669, 15565, 12150]` | **bit-identical** |
| `GENERATED_TEXT` | `"ntiuhuihnerquant"` | **bit-identical** |
| `PER_TOKEN_MS` | 25.04 | **131.80** (5.3× regression from per-runOp pre-pass) |
| `COUNTER_DELTAS.write` | ~134 (set_tensor at load) | **4404** (134 + ~4270 H1-inverse) |
| `COUNTER_DELTAS.read` | 1602 | 1602 (H1 unchanged) |
| `__stage48Captures.src0AtKernelTimeF32` row 2 cell 0 | `-0.0` (raw `00 00 00 80`) | **`0.0`** (post-H1-inverse zeroed it) |
| `__stage48Captures.postKernelFirst8U16` row 2 cell 0 | `0x8000` | **`0`** |
| 6 kernel selftests | PASS | PASS |

## Smoking gun — `__h1invDiag.captures`

Diagnostic added in `src/inference/jsep/index.ts`'s `module.jsepWrite`: capture
first 16 bytes / 8 F32 of `host_mirror[hostPtr..hostPtr+16]` when (handle=26,
offset=0, size=6144). That signature uniquely identifies the H1-inverse call
for the K-projection-after-ROPE source feeding the K-cache SET_ROWS. Captured
the first 8 such calls across one full prefill + 5 decode steps:

| `callIdx` | first 8 F32 of `host_mirror[h26+0..32]` |
|---|---|
| **0** (prefill i=3) | **`[0, 0, 0, 0, 0, 0, 0, 0]`** |
| **1** (decode step 1's first SET_ROWS) | **`[0, 0, 0, 0, 0, 0, 0, 0]`** |
| 2 (decode step 2) | `[7.19e-5, -2.05e-5, -3.38e-5, 3.00e-5, -1.62e-6, -2.87e-5, -5.75e-6, 1.23e-5]` |
| 3 | (same as 2) |
| 4 | (same as 2) |
| 5 | (same as 2) |
| 6 | (same as 2) |
| 7 | (same as 2) |

**Reading:** at H1-inverse pre-pass time for the FIRST two SET_ROWS dispatches
(prefill + first decode), `host_mirror[h26+0..32]` is exactly zero. From the
THIRD such call onward, it carries real F32 data of the magnitude expected for
post-ROPE K (small magnitudes, on the order of 1e-5 to 1e-4).

This means the host→GPU sync done by H1-inverse for those first two dispatches
is faithfully syncing **zeros** — it cannot fix the all-zero collapse because
the data isn't in `host_mirror` to be synced. Stage 4.8's framing ("CPU op chain
updates host_mirror but not GPU") was structurally incomplete: in the load-
bearing window (first two SET_ROWS), the CPU op chain has **not yet written
to host_mirror** at the offset SET_ROWS reads from.

(D2-lite's deferred `setRowsDiag.src0First8F32 = [-1.067, 0.656, ...]` reading
for i=3 is consistent: that read happens via `Promise.resolve().then(...)`
microtask which fires much later, by which time `h26o0` has been overwritten
many times by subsequent H1-inverse syncs and JSEP ops on later decode steps.
The deferred read is NOT a measurement of i=3 dispatch-time state — Stage 4.6
D2-tight already established that, and Stage 4.9's diagnostic confirms the
prefill-time state directly.)

## Code that landed

**llama.cpp `webllm-browser-patches`:** new patch on top of `e0fa38928` (Stage
4.5 H1) — patch stack 7 → 8. Mirrors the H1 pattern. Added in
`ggml_backend_jsep_graph_compute()` after the descriptor is built and before
the `EM_ASM_INT` dispatch:

```cpp
// H1-inverse (Stage 4.9): host→GPU mirror sync before dispatch.
// Symmetric to H1's post-pass: between graph_compute calls,
// CPU-fallback ops dereference `tensor->data` (= host_mirror +
// offset post-F1) and write directly to host RAM. No JSEP callback
// fires for those writes, so the GPU buffer for those tensors
// stays stale. Without this pre-pass, JSEP runOps reading those
// tensors from the GPU side see pre-CPU-write contents.
//
// Over-approximation: we sync EVERY src tensor whose buffer is a
// jsep_buf, including weights that were dual-written by set_tensor
// at load time and never modified since. The redundant uploads
// cost wall time but are correctness-safe. Optimization deferred.
for (int s = 0; s < n_src; ++s) {
    const ggml_tensor * src = node->src[s];
    if (!src || !src->buffer || !src->buffer->context) {
        continue;
    }
    if (src->buffer->iface.get_base != ggml_backend_jsep_buffer_get_base) {
        continue;
    }
    auto * sctx = static_cast<ggml_backend_jsep_buffer_context *>(src->buffer->context);
    const size_t src_offset = jsep_tensor_handle(src);
    const size_t src_size   = ggml_nbytes(src);
    ggml_jsep_write(sctx->handle, (int32_t) src_offset,
                    (int32_t) (uintptr_t) ((char *) sctx->host_mirror + src_offset),
                    (int32_t) src_size);
}
```

**Verification artefacts** (kept resident for Stage 4.10):

- `src/inference/jsep/index.ts` — `module.jsepWrite` carries the
  `__h1invDiag` capture block (first 16 bytes / 8 F32 of host_mirror at
  hostPtr for matching `(handle, offset, size)`; defaults to
  `{handle:26, offset:0, size:6144}` — the i=3 src0 signature).
- `smoke-test/p2-v2-spike.src.ts` — initialises `__h1invDiag` global
  with the canonical match signature before `installJsepCallbacks`.

## Why H1-inverse stays despite not unblocking Outcome A

H1-inverse is the structurally correct symmetric counterpart to Stage 4.5 H1.
It catches the case where a CPU op writes to `host_mirror` and a subsequent JSEP
runOp reads from the GPU — exactly the failure mode Stage 4.8 articulated, and
which DOES happen for `callIdx ≥ 2`. Without H1-inverse, those later decode
steps would also collapse (they were latently working only because the GPU
buffer at `h26o0` had been touched by prior runOps in ways that approximated
the right K data). Stage 4.10's fix will attack the load-bearing bug
(`host_mirror` staleness at first 2 SET_ROWS); H1-inverse is needed to
guarantee correctness once that bug is fixed.

The 5.3× per-token regression (25 ms → 131 ms) is acceptable in the JSEP path
during Phase 3 stabilization. Optimization (dirty-bit tracking; sync only
sources whose `host_mirror` may have been touched since the previous JSEP
graph_compute) is queued for Phase 3 Stage 5+ once Outcome A holds.

## Patch stack

- llama.cpp `webllm-browser-patches`: **8** (was 7). Tip: `<pending>` on
  base `a817a22bc` (unchanged).
- webllm: +1 commit (spike + jsep diagnostic; no production-path code
  modified — `make wasm-build` legacy artifacts unaffected).

## `make checkall`

Green: fmt + lint + typecheck (incl. tests) + 747 tests pass / 36 skip
/ 0 fail / 39170 expect calls.

## Branch on outcome — Stage 4.10 queued

**Outcome was C-2** (NaN/zero collapse persists; H1-inverse correctly applied
but data-source itself is zero in the load-bearing window). Per the Stage 4.8
brief's branch table, this maps to: "**Move to Stage 4.10**. Localize WHY
`host_mirror[h26+0..6144]` is zero at first 2 SET_ROWS — the CPU CPY+ROPE chain
either hasn't run, ran to a different (handle, offset), or ran but copied
zeros." Stage 4.10 brief is queued in `TODO.md`.

## References

- Stage 4.5 closure: [`STAGE-4.5-RESULT.md`](STAGE-4.5-RESULT.md) (H1 GPU→host writeback)
- Stage 4.8 closure: [`STAGE-4.8-RESULT.md`](STAGE-4.8-RESULT.md) (host→GPU framing — superseded by 4.9 finding)
- Spike: `smoke-test/p2-v2-spike.html?v=stage4.9-diag1`
- Run log: `window.__jsepRunLog` (first 12 entries used for op-by-op handle/offset table)
- Diagnostic: `window.__h1invDiag.captures` (the smoking gun above)
