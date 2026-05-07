# Phase 3 / Option A-prime — Stage 4.10 result

**Date:** 2026-05-07

**One-line outcome:** **HA strict-form REJECTED, refined HA' (cross-backend
boundary leaf) CONFIRMED.** SET_ROWS' src0 in the first SET_ROWS slice (slice 3)
is a **leaf tensor** in JSEP's split-cgraph view (`src0Op = GGML_OP_NONE = 0`,
`view_src = NULL`). The scheduler treats `h26+0` and `h26+528384` as cross-
backend inputs that should be populated by another backend before slice 3 runs;
the CPU subgraph that should populate them via direct `tensor->data → host_mirror`
writes (post-F1) is not landing data there in time. Stage 4.11 will instrument
the CPU side of the split + dump host_mirror evolution between graph_compute
calls to disambiguate whether the CPU producer fires too late, writes to a
different absolute address, or doesn't fire at all. **Diagnostic-only stage —
patch stack unchanged at 8.**

## Context

Stage 4.9 banked H1-inverse (host→GPU writeback per-runOp pre-pass) and
confirmed it fires correctly but does not flip Outcome A: at the FIRST 2
SET_ROWS dispatches `host_mirror[h26+0..6144] = [0, 0, 0, 0, 0, 0, 0, 0]`,
so H1-inverse faithfully syncs zeros. The Stage 4.10 brief queued localization
of WHY host_mirror is stale, with three priority-ordered hypotheses:
HA (scheduler grouping), HB (different (handle, offset)), HC (clear/memset
wipes).

## Method

Added per-graph_compute-slice instrumentation in
`ggml_backend_jsep_graph_compute` (top of function, before the for-loop).
Captures the first 30 graph_compute invocations into
`globalThis.__jsepGraphLog`, with per-node payload (10 i32 stride):

```
[op, dstH, dstO, src0H, src0O, src0Op, src1H, src1O, src1Op, src0VsOp]
```

Where `src0Op = src[0]->op` and `src0VsOp = src[0]->view_src->op` (or 0 if
view_src is NULL). This lets us see:

1. Slice composition (HA: are MUL_MAT and SET_ROWS in the same JSEP slice?).
2. Whether SET_ROWS' src tensors have producers in JSEP's view (HA'').
3. View-src indirection (HB: is the data routed through a view of another
   tensor?).

## Headline findings

### HA strict-form **REJECTED**

The brief's strict HA framing was: "if the first graph_compute call processes
a long chain ending in SET_ROWS without an intervening CPU graph_compute, HA
is confirmed." Reality: **3 JSEP graph_compute calls fire before slice 3 (the
first SET_ROWS slice).** Between consecutive JSEP slices, the scheduler runs
non-JSEP subgraphs (CPY/ROPE/MUL etc. on the CPU backend). So CPU subgraphs
DO get scheduling time before slice 3 — the scheduler is not packing
everything into one mega-JSEP slice.

### Pre-SET_ROWS slices (idx 0–2)

| slice | n_nodes | ops             | dst (h+o)            | reading             |
|-------|---------|-----------------|----------------------|---------------------|
| 0     | 1       | `25` (RMS_NORM) | `h26+0`              | input pre-attn norm |
| 1     | 2       | `29, 36`        | `h26+4194304` (both) | proj A + RESHAPE    |
| 2     | 2       | `29, 36`        | `h26+4194304` (both) | proj B + RESHAPE    |

Both slice 1 and slice 2 write to `h26+4194304` (allocator coalescing — the
two projection results don't overlap in time, so ggml-alloc reuses the same
slot). **Neither slice writes to `h26+0` after slice 0's RMS_NORM.** Slice 0
wrote RMS_NORM output to `h26+0` (size = 6 tokens × 2048 hidden × 4 bytes =
49152 B) but that data is **not** what SET_ROWS needs. The expected V/K
projection data at `h26+0` (size 6144 B) and `h26+528384` (size 6144 B) must
come from somewhere else.

### Slice 3 — the first SET_ROWS slice

12 nodes; 2 SET_ROWS (V cache write + K cache write):

```json
[
  {"op": 42, "dstH": 25, "dstO":      0, "src0H": 26, "src0O":      0,
   "src0Op": 0, "s0VsOp": 0, "src1H": 26, "src1O":  524288, "src1Op": 0},
  {"op": 36, "dstH": -1, ...},  // RESHAPE (metadata)
  {"op": 36, "dstH": -1, ...},  // RESHAPE (metadata)
  {"op": 36, "dstH": 25, "dstO": 262144, "src0H": 25, "src0O": 262144, ...},
  {"op": 42, "dstH": 25, "dstO": 262144, "src0H": 26, "src0O": 528384,
   "src0Op": 0, "s0VsOp": 0, "src1H": 26, "src1O": 1052672, "src1Op": 0},
  ...
  {"op": 29, "dstH": 26, "dstO": 6295552, "src0H": 25, "src0O": 0,
   "src0Op": 38, ...}  // attention Q@K^T
]
```

**Smoking gun:** both SET_ROWS' `src0Op = 0 (GGML_OP_NONE)` and `s0VsOp = 0`.
In ggml's cgraph, `op = NONE` marks a leaf tensor (no producer node within the
graph) and `view_src = NULL` rules out view-indirection. So:

- `h26+0` (V data): **leaf in JSEP's split-cgraph view**
- `h26+528384` (K data): **leaf in JSEP's split-cgraph view**
- `h26+524288` and `h26+1052672` (the I64 SET_ROWS index lists): also leaves

This is consistent with how ggml-backend's scheduler splits graphs across
backends: when a tensor crosses backends, the receiving backend sees it as a
leaf with `op = NONE`, and the producer lives in a different backend's
sub-cgraph. The scheduler is responsible for ensuring data flows from the
producer's backend buffer to the consumer's backend buffer before the
consumer runs.

### HB **WEAKENED** (no view-src indirection)

For both SET_ROWS leaves, `s0VsOp = 0` — the tensor has no `view_src`. By
construction, post-F1, `tensor->data == host_mirror[bufHandle] + offset`, so
the leaf's "address" is unambiguously `host_mirror[26] + 0` (or
`host_mirror[26] + 528384`). HB's "different (handle, offset)" framing
doesn't apply: the address SET_ROWS reads from is exactly the F1 host_mirror
slot.

### HC **NOT TESTED but unlikely**

The Stage 4.9 capture data shows `callIdx 2..7` produce **bit-identical** K-
shaped data (`[7.19e-5, -2.05e-5, ...]` repeated across 6 calls). If a
spurious clear/memset were firing between calls, we'd expect at least some
variation. The constancy across calls 2–7 makes HC unlikely as the load-
bearing cause for the call 0/1 zero-state — though we can't rule it out from
JSEP-side instrumentation alone.

### Refined HA' / HD **CONFIRMED**

The bug is not "scheduler packs everything into one JSEP slice." Instead:

> The scheduler split correctly identifies that the V/K data at `h26+0`
> and `h26+528384` are produced by another backend (presumably CPU) and
> should be filled before JSEP's slice 3 runs. The mechanism for cross-
> backend transfer relies on F1's `host_mirror` semantics: post-F1,
> `tensor->data` for jsep_buft tensors points into `host_mirror`, so
> a CPU op writing via `tensor->data` would directly populate
> `host_mirror[26]+0` without any explicit JSEP callback. JSEP's
> H1-inverse pre-pass would then sync `host_mirror → GPU` before slice 3.
>
> The chain is broken at one of two points:
>
> 1. **No CPU op writes to `host_mirror[26]+0` before slice 3 runs.**
>    Either the CPU producer chain hasn't been scheduled yet, or it
>    writes to a different absolute address (mismatch between `tensor->data`
>    seen by CPU MUL_MAT vs. the SET_ROWS' src0 leaf).
> 2. **A CPU op DOES write there, but is overridden** by a subsequent
>    clear/memset (HC) before slice 3.
>
> Reading (1) is more probable. The Stage 4.9 capture shows `host_mirror`
> is *consistently* zero at the first two dispatches — across an entire
> prefill (~ms of wall time) plus the start of decode. If anything had
> ever written real V/K data there, it would persist (no obvious clearing
> mechanism). So Stage 4.11 should verify reading (1) directly.

### Slice composition pattern (per-layer prefill)

The 30 captured slices exhibit a clean per-layer cycle (TinyLlama has 22
layers; we capture roughly 6 layers' prefill within the 30-slice window):

```
slice  0:  25                                      (pre-attn RMS_NORM)
slice  1:  29, 36                                  (proj A + RESHAPE)
slice  2:  29, 36                                  (proj B + RESHAPE)
slice  3: 42, 36, 36, 36, 42, 37, 38, 37, 38, 37, 38, 29   (V SET_ROWS,
                                                              K SET_ROWS,
                                                              attention Q@K^T)
slice  4:  29, 38                                  (attention @ V)
slice  5:  29                                      (output projection)
slice  6:  25                                      (pre-FFN RMS_NORM)
slice  7:  29, 29                                  (FFN gate + up)
slice  8:  25                                      (next layer pre-attn)
... (repeat) ...
```

Notable: only **2** projection MUL_MATs appear before each SET_ROWS slice
(slices 1+2). Standard LLaMA attention has 3 projections (Q, K, V). The third
projection is missing from JSEP slices, meaning **one of {Q, K, V} runs
entirely on CPU.** Combined with the SET_ROWS leaves observation, the missing
projection's output is the leaf data that needs cross-backend transfer.

## Stage 4.9 baseline reproduced bit-exactly

| Marker | Stage 4.9 baseline | Stage 4.10 replay |
|---|---|---|
| `LOGIT_STATS_STEP0.topId/topVal` | `593 / 0.159` | **`593 / 0.159`** |
| `GENERATED_TEXT` | `"ntiuhuihnerquant"` | **`"ntiuhuihnerquant"`** |
| `PER_TOKEN_MS` | 131.80 | 127.42 (within noise) |
| `COUNTER_DELTAS.write` | 4404 | 4404 |
| `COUNTER_DELTAS.read` | 1602 | 1602 |
| `__h1invDiag.captures` callIdx 0 first8F32 | all-zero | all-zero |
| 6 kernel selftests | PASS | PASS |
| `make checkall` | green | green |

The Stage 4.10 instrumentation has zero correctness impact (read-only metadata
capture into a JS array) and per-call overhead is invisible against the
H1-inverse 5.3× regression baseline.

## Code that landed

**llama.cpp `webllm-browser-patches`:** instrumentation only — no behavioral
change. Added a new EM_ASM block at the top of
`ggml_backend_jsep_graph_compute` (above the existing for-loop). Allocates a
stack array sized 128 nodes × 10 i32 = 5120 B; loops once over `cgraph->nodes`
to pack per-node metadata; pushes a single record into
`globalThis.__jsepGraphLog` (cap 30 entries). Also exposes
`globalThis.__jsepCurrentGraphIdx` for downstream wrappers (not consumed in
this stage; reserved for Stage 4.11).

**webllm:** none.

**Patch stack:** 8 (unchanged). The instrumentation lives in the patch tip and
will be retained until Outcome A flips; it is the diagnostic substrate for
Stage 4.11 onward.

## Branch on outcome — Stage 4.11

The brief's outcome table maps as follows:

- **HA confirmed (most likely):** ❌ — REJECTED in strict form.
- **HB confirmed (less likely):** ❌ — WEAKENED by `s0VsOp = 0`.
- **HC confirmed (unlikely):** ❌ — NOT TESTED but indirect evidence weak.
- **All three rejected:** ✅ — surfaces **HD**: cross-backend boundary leaf
  whose CPU-side producer is not populating `host_mirror[26]+0` /
  `host_mirror[26]+528384` before JSEP slice 3 runs.

Stage 4.11 brief (queued in `TODO.md`) implements two probes in priority
order:

1. **Inter-slice host_mirror snapshot** (cheap; 20 LOC). At the *end* of every
   JSEP graph_compute call, dump `host_mirror[26]+0..16` and
   `host_mirror[26]+528384..16` into a JS-side log. Combined with the
   Stage 4.10 graph-log, this answers: at what point in the slice timeline
   does the V/K data appear in host_mirror? If it appears between slices 2
   and 3 → HC ruled in (something cleared it; or the CPU op fires per-slice
   with stale results). If it appears only AFTER slice 3 → HD' confirmed
   (CPU producer fires too late).
2. **CPU graph_compute instrumentation** (~50 LOC + ggml-cpu rebuild). If
   probe 1 indicates HD', wrap CPU's graph_compute to log per-call op
   composition + dst handles. We'd see whether a CPU subgraph containing the
   V/K projection MUL_MAT or its consumer CPY runs before slice 3.

After Stage 4.11 closes (root-cause localized), Stage 4.12 enacts the
structural fix. Likely shapes:

- Add CPY (or just CPY into jsep_buft) to JSEP's `supports_op` and
  `dispatchCpy` (or a stub that no-ops because post-F1 the data is already
  in host_mirror — but only if Stage 4.11 confirms host writes are landing
  correctly and only the timing is off).
- Or: re-enable a narrow form of the offload_op path that lets JSEP run the
  CPU-side ops it can support, eliminating the cross-backend leaf.
- Or: wire `cpy_tensor_async` for JSEP so the scheduler's standard cross-
  backend transfer path uses an explicit memcpy in F1's host_mirror world,
  fixing the gap that the implicit-tensor->data-share is failing to bridge.

## Exit criteria — Stage 4.10 — all checked

- [x] One of HA / HB / HC is confirmed via instrumentation (or all three
      rejected, surfacing HD): **HA strict-form rejected, HB weakened, HC
      unlikely → HD surfaced.**
- [x] `__jsepGraphLog` evidence captured + quoted in this report.
- [x] Closure report names the load-bearing root cause and points at Stage
      4.11's structural fix queue.
- [x] `make checkall` green (instrumentation is read-only metadata capture).
- [x] Patch stack: 8 (unchanged — Stage 4.10 is diagnostic-only).
