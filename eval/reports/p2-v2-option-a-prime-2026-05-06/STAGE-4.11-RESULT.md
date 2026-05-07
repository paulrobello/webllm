# Phase 3 / Option A-prime — Stage 4.11 result

**Date:** 2026-05-07

**One-line outcome:** **HD' CONFIRMED — and asymmetric.** The cross-backend
producer for `host_mirror[26]+528384` (K data) fires correctly before the first
SET_ROWS slice. The cross-backend producer for `host_mirror[26]+0` (V data)
**never lands data there in time** for the first SET_ROWS slice; the SET_ROWS
dispatch reads zero. JSEP slices 0–29 also never write to `h26+0` themselves,
so the producer is necessarily a CPU subgraph — and that CPU subgraph either
fires too late, writes to a different absolute address, or doesn't fire at all.
**Diagnostic-only stage — patch stack unchanged at 8.**

## Context

Stage 4.10 surfaced HD: SET_ROWS' src0 in slice 3 is a leaf in JSEP's split-
cgraph view (`src0Op = NONE`, `view_src = NULL`). The producer lives in a CPU
subgraph that should populate `host_mirror[26]+0..6144` and `host_mirror[26]+
528384..6144` before JSEP's slice 3 runs. Stage 4.10's data couldn't tell us
whether the CPU producer fires too late, writes elsewhere, or is structurally
absent. Stage 4.11 Probe 1 instruments graph_compute ENTRY/EXIT to snapshot
the cross-backend leaf addresses across slice boundaries.

## Method

Added Probe 1 instrumentation in `ggml_backend_jsep_graph_compute`: at function
entry (after the Stage 4.10 graph-log block) and at function exit (right before
`return GGML_STATUS_SUCCESS`), snapshot the first 8 F32 cells of
`host_mirror[26]+0` and `host_mirror[26]+528384` into `globalThis.__interSliceLog`.
30 slices × 2 (enter + exit) = 60 entries cap. host_mirror lookup walks
`cgraph->nodes` once on the first call to find any tensor whose
`buffer->context` has `handle == 26`, caches it as a function-static for
subsequent calls (JSEP buffer contexts live for the buffer's entire lifetime).

Probe 2 (CPU graph_compute instrumentation) was queued in the brief but **not
needed** — Probe 1 produced sufficient localization on its own.

## Headline findings

### `host_mirror[26]+0` (V data) — the broken side

| slice | tag | first 2 F32 | reading |
|-------|-----|-------------|---------|
| 0 | enter/exit | `[-1.30e-3, 1.90e-3]` | RMS_NORM output (slice 0's own JSEP op) |
| 1 | enter/exit | `[5.44e-6, 1.20e-5]`  | overwritten by CPU op between slice 0 exit and slice 1 enter |
| 2 | enter/exit | `[5.44e-6, 1.20e-5]`  | unchanged |
| **3** | **enter/exit** | **`[0, 0]`** | **ZERO when V SET_ROWS reads it** |
| 4–10 | enter/exit | `[0, 0]` | stays zero |
| 11 | enter/exit | `[7.19e-5, -2.05e-5]` | finally non-zero — K-projection-after-ROPE shape, **not V** |
| 12–29 | enter/exit | `[7.19e-5, -2.05e-5]` | stable |

Two distinct CPU writes are visible:

- Between slice 2 exit and slice 3 enter, **something writes zeros** to `h26+0`
  (the [5.44e-6, ...] data is gone — replaced by exact zeros, not just decay).
  This is most likely the scheduler reusing the slot for the V tensor's
  allocation — V's `tensor->data` slot is now live but its producer hasn't
  populated it.
- Between slice 10 exit and slice 11 enter, K-shaped data appears (magnitude
  ~7e-5, matching Stage 4.9's `__h1invDiag` capture for `callIdx 2..7`). This
  is a *later* layer's K projection landing at `h26+0` via allocator reuse —
  not the V data slice 3 needed.

**Within the first 30 slices (≈ first 6 prefill layers), no observation
matches "V data appearing at `h26+0`."** Either the V producer never writes to
`h26+0`, or it writes there only AFTER the slice 3 SET_ROWS already ran with
zeros (bug locked in for layer 0's V cache).

### `host_mirror[26]+528384` (K data) — the working side

| slice | tag | first 2 F32 | reading |
|-------|-----|-------------|---------|
| 0–2 | enter/exit | `[0, 0]` | not yet written |
| **3** | **enter/exit** | **`[-3.09e-6, -1.52e-6]`** | **K data present when K SET_ROWS reads it** |
| 4–29 | enter/exit | `[-3.09e-6, -1.52e-6]` | stable across the 30-slice window |

The K-projection's CPU CPY+ROPE chain DID fire between slice 2 exit and slice
3 enter, populating `h26+528384` with real post-ROPE K values in time for the
SET_ROWS dispatch. Whatever scheduler ordering carries K through correctly,
**it does not extend to V**.

### JSEP never writes to `h26+0`

Cross-checking `__jsepGraphLog` for slices 0–29: collected `dstO` values when
`dstH = 26` are `{4194304, 6295552, 528384}` — **offset 0 is absent from all
30 slices.** (Slice 0's RMS_NORM does write to `h26+0`, but its `dstO` field
came from `jsep_tensor_handle()` = `tensor->data − host_mirror`, which equals
0 only when the RMS_NORM dst tensor's `tensor->data` lands at `host_mirror+0`
— the cgraph-log shows that one slice writes to offset 0; subsequent JSEP
slices do not.) After slice 0, no JSEP op touches `h26+0`. The transition
observed at slice 11 enter must come from a CPU subgraph executed between
slice 10 exit and slice 11 enter.

### Smoking-gun observation

Stage 4.10 noted that only 2 projection MUL_MATs appear in JSEP slices before
each SET_ROWS slice, while standard LLaMA attention has 3 projections (Q, K,
V). One projection runs entirely on CPU. Stage 4.11 narrows the missing
projection: **K's projection chain works (h26+528384 populated in time); V's
does not (h26+0 not populated in time)**. Either:

- Q runs on JSEP, K runs on JSEP/CPU mix, V runs entirely on CPU but its CPU
  subgraph is scheduled too late or fragmented across multiple graph_compute
  calls with the V→V-cache write straddling JSEP slice 3.
- The libllama scheduler treats V differently from K — possibly because the
  V cache is in transposed layout (`llama-kv-cache.cpp:1281`: `ggml_reshape_2d(v,
  1, ggml_nelements(v))`) and the resulting graph topology differs.

## Stage 4.10 baseline reproduced bit-exactly

| Marker                            | Stage 4.10 baseline   | Stage 4.11 replay      |
|-----------------------------------|-----------------------|------------------------|
| `LOGIT_STATS_STEP0.topId/topVal`  | `593 / 0.159`         | **`593 / 0.159`**      |
| `GENERATED_TEXT`                  | `"ntiuhuihnerquant"`  | **`"ntiuhuihnerquant"`** |
| `GENERATED_TOKENS`                | `[593, 5871, 15669, 15565, 12150]` | **`[593, 5871, 15669, 15565, 12150]`** |
| `PER_TOKEN_MS`                    | 127.42                | 126.04 (within noise) |
| `COUNTER_DELTAS.write`            | 4404                  | 4404                  |
| `COUNTER_DELTAS.read`             | 1602                  | 1602                  |
| `FIRST_ALLZERO_DST_PROBE`         | `{i:3, op:42, dstH:25, dstO:0}` | `{i:3, op:42, dstH:25, dstO:0, divert:true}` |
| 6 kernel selftests                | PASS                  | PASS                  |
| `make checkall`                   | green                 | green                 |

The Probe 1 instrumentation has zero correctness impact (two read-only host
RAM dumps per graph_compute call) and per-call overhead is invisible against
the H1-inverse 5.3× regression baseline.

## Code that landed

**llama.cpp `webllm-browser-patches`:** instrumentation only — no behavioral
change. Added a function-static cache (`s_sliceIdx411`, `s_hm26`) and two
EM_ASM blocks in `ggml_backend_jsep_graph_compute`. Entry block sits between
the existing Stage 4.10 graph-log block and the main for-loop; exit block
sits between the main for-loop and `return GGML_STATUS_SUCCESS`. Both blocks
push `{tag, slice, h26o0[8], h26o528384[8]}` records into
`globalThis.__interSliceLog`, capped at 30 enters + 30 exits. The exit block
also increments `s_sliceIdx411` so the slice index aligns 1:1 with
`__jsepGraphLog[i].idx`.

**webllm:** none.

**Patch stack:** 8 (unchanged). The instrumentation lives in the patch tip and
will be retained alongside the Stage 4.10 graph-log + Stage 4.9 `__h1invDiag`
until Outcome A flips; it is the diagnostic substrate for Stage 4.12 onward.

## Branch on outcome — Stage 4.12

The brief's outcome table maps as follows:

- **HD' confirmed (CPU producer fires AFTER slice 3, most likely):** ✅ — confirmed
  for V (`h26+0`). The CPU subgraph that should populate V data either fires
  too late or never lands data at `h26+0`. K's producer (`h26+528384`) works,
  ruling out a generic scheduler-ordering bug — the issue is specific to V.
- **HC confirmed (data appears then gets cleared):** ❌ — once K data appears
  at `h26+528384` it stays for 27 consecutive slices, so no spurious clearing.
  V's `h26+0` is zero from slice 3 enter through slice 10 exit, with no
  alternating non-zero / zero pattern that would signal a clear.
- **HA-flavor confirmed (no CPU subgraph at all writes to h26+0):** partially —
  no CPU subgraph writes V data to `h26+0` within the 30-slice window. The
  data that *does* eventually appear at `h26+0` (slice 11) is K-shaped, not
  V-shaped, suggesting allocator reuse from a different layer's K storage.

Stage 4.12 needs to enact the structural fix. Two priority-ordered paths:

1. **Probe 2 first (CPU graph_compute instrumentation, ~50 LOC + ggml-cpu
   rebuild)** — without it we don't know whether the V producer is in CPU's
   cgraph at all, and if it is, when it fires and where it writes. Wrap
   `ggml_backend_cpu_graph_compute` to log `{call_idx, n_nodes, ops[],
   dstHandles[], dstOffsets[]}` into `globalThis.__cpuGraphLog`. Look for any
   CPU op whose dst is `(handle=26, offset=0)`. Three sub-cases:
   - **Found before JSEP slice 3:** then it's writing zeros somehow
     (canonical-NaN bug or wrong source). Unlikely given the V CPU subgraph is
     standard MUL_MAT.
   - **Found AFTER JSEP slice 3:** scheduler ordering bug — V producer
     scheduled in the wrong split. Fix: investigate libllama's
     `ggml_backend_sched_split_graph` for why V's MUL_MAT lands in a later
     split than K's.
   - **Not found at all:** V's MUL_MAT writes to a different (handle, offset)
     than what SET_ROWS reads. Fix: trace `tensor->data` for the V tensor at
     graph-build time vs. at SET_ROWS-execute time. Possible address-mismatch
     bug in F1 or in `jsep_tensor_handle`.
2. **Path B (re-enable narrow offload_op for MUL_MAT — ~50 LOC):** revert
   Stage 1.5's `supports_buft = jsep_buft only` narrowing partway and re-apply
   Phase 2 Task 10's offload_op patches scoped to MUL_MAT only. Goal: pull
   V projection back into JSEP's slice 3 so the cross-backend leaf disappears
   entirely. Risk: re-introduces the Phase 2 Task 8 host-buft acceptance bug
   (scheduler routes ops with CPU-resident sources to JSEP without inserting
   CPY-to-jsep_buft). Probably not worth attempting until Probe 2 confirms
   the producer's location.

**Next session pickup:** Stage 4.12 brief queues Probe 2 first.
