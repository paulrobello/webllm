# Phase 3 / Option A-prime — Stage 4.12 result

**Date:** 2026-05-07

**One-line outcome:** **PARTIAL CLOSE / Probe 2 reframed.** The brief's
three predicted sub-cases ("CPU writes V to a different (handle, offset)",
"V's CPU op scheduled into a later split", "V's MUL_MAT inputs are zero")
all assumed the V producer would appear as a node in `ggml_backend_cpu_
graph_compute`'s cgraph. **It does not.** Across 30 CPU graph_compute
calls × 42 nodes, **zero** nodes have a jsep-resident `dst` *or* a
jsep-resident `src0` / `src1`. The cross-backend writes into `jsep_buf`
flow through `ggml_backend_jsep_buffer_set_tensor` — not through the
cgraph the CPU backend executes. A second probe layer instruments
set_tensor calls and gives the smoking gun: **V's data lands at
`(h26, 0)` as a 6144-byte set_tensor write of all zeros**, while
**K's data lands at `(h26, 528384)` as a 6144-byte set_tensor write of
real K-shaped f32**. The asymmetry is at the cross-backend boundary,
upstream of any visible cgraph node — V's CPU-side producer either
doesn't run, runs but produces zeros, or runs with its result dst
allocated zero and the actual MUL_MAT skipped. Stage 4.13 will localize
*which* of the three. Patch stack: **8 → 9** (one diagnostic patch
adds the resolver helper + CPU graph instrumentation + jsep set_tensor
instrumentation; reverts to 8 once Stage 4.13 lands the real fix).

## Context

Stage 4.11 confirmed HD' for the V side only — at slice 3 entry,
`host_mirror[26]+528384` (K) holds real K-shaped f32 data, but
`host_mirror[26]+0` (V) is all zeros. Slices 0–29 in `__jsepGraphLog`
never write to `h26+0`, so V's producer must be on CPU. Stage 4.12's
brief queued Probe 2 — instrument `ggml_backend_cpu_graph_compute` to
log per-node `(op, dstH, dstO, src0H, src0O, src0Op, src1H, src1O,
src1Op)` for the first 30 CPU calls, then run the same `(handle=26,
offset=0)` vs `(handle=26, offset=528384)` filter we ran on the JSEP
log.

## Method

### Step 1 — Probe 2 (CPU graph_compute instrumentation)

Added a public helper `ggml_jsep_resolve_tensor(t, &handle, &offset)`
in `ggml-jsep.cpp` (declared in `ggml/include/ggml-jsep.h`, weak-linked
from `ggml-cpu.cpp` so non-jsep builds compile cleanly). Mirrors Stage
4.10's predicate `iface.get_base == ggml_backend_jsep_buffer_get_base`.

Instrumented `ggml_backend_cpu_graph_compute` to log
`__cpuGraphLog` (cap 30 calls, 128 nodes/call, 9-i32 stride per
node — same shape as Stage 4.10's `__jsepGraphLog` minus
`s0VsOp`). Each node's `dstH/dstO/src0H/src0O/src1H/src1O` is
`-1` if the corresponding tensor is not in `jsep_buf`.

### Step 2 — Probe 2 follow-up (set_tensor instrumentation)

After Probe 2 returned empty for both V and K targets, instrumented
`ggml_backend_jsep_buffer_set_tensor` to log calls into
`__setTensorLog` (cap 200, gated on `ctx->handle == 26` to skip
weight uploads). Each entry records `(handle, offset, size,
f32_first4)`.

## Probe 2 (CPU graph_compute log)

```js
const cpuLog = globalThis.__cpuGraphLog || [];
cpuLog.flatMap((c, ci) => c.nodes
  .map((n, ni) => ({callIdx: ci, nodeIdx: ni, ...n}))
  .filter(n => n.dstH === 26 && n.dstO === 0)
);
// → []  (V — not produced by any visible CPU op)

cpuLog.flatMap((c, ci) => c.nodes
  .map((n, ni) => ({callIdx: ci, nodeIdx: ni, ...n}))
  .filter(n => n.dstH === 26 && n.dstO === 528384)
);
// → []  (K — also not produced by any visible CPU op)
```

Distribution of `(dstH, dstO)` across all CPU nodes: **`-1:-1` for
all 42 nodes**. No CPU node writes to *any* jsep_buf address.
Distribution of `dstH` alone: `Map { -1 → 42 }`. The CPU backend
operates entirely on tensors that have already been split into
CPU buft (CPU heap) by ggml-backend's scheduler.

This means the brief's three predicted sub-cases all rest on a
false premise. The cross-backend producer of `host_mirror[26]+0`
and `host_mirror[26]+528384` is **not a cgraph node** the CPU
backend executes — it is a `set_tensor` call from the scheduler
that copies a CPU buft result *into* jsep_buf.

CPU op composition (sample):

| callIdx | n_nodes | first op | summary |
|---------|---------|----------|---------|
| 0 | 1 | 40 (GET_ROWS) | embedding lookup |
| 1 | 1 | 7 (MUL) | RMS_NORM gain × normed |
| 2 | 3 | 48, 29, 36 (ROPE, MUL_MAT, RESHAPE) | post-K-cache attention path |
| 3 | 2 | 48, 37 (ROPE, VIEW) | metadata |
| 4 | 1 | 46 (SOFT_MAX) | attention scores |
| 5 | 1 | 35 (CONT) | contiguous copy |
| 6 | 1 | 2 (ADD) | residual add |
| 7 | 1 | 7 (MUL) | mlp gate |
| 8 | 3 | 95, 29, 2 (?, MUL_MAT, ADD) | mlp + output projection |
| ... | ... | ... | layer pattern repeats |

The CPU sees: ROPE, SOFT_MAX, CONT, ADD, MUL, RESHAPE, VIEW, GET_ROWS,
plus a few MUL_MATs (attention `Q@K^T` + output projection family).
Notably, **no MUL_MAT here is a Q/K/V projection**: the only MUL_MATs
have `src1Op = 7` (MUL — a fused norm output) or `src1Op = 95`. The
Q/K/V projections live in `__jsepGraphLog` slices 0–29 (slices 1 + 2
are 2 of 3 projections; the third is the missing leaf).

## set_tensor follow-up — the smoking gun

```js
globalThis.__setTensorLog.slice(0, 5)
```

| idx | handle | offset | size | f32_first4 |
|-----|--------|--------|------|------------|
| 0 | 26 | 0 | 49152 | `[-1.30e-3, 1.90e-3, -1.94e-3, 3.83e-3]` (input embedding shape: 6 tok × 2048 dim × 4 B = 49152) |
| 1 | 26 | 0 | 49152 | `[5.44e-6, 1.20e-5, -1.36e-4, -1.13e-4]` (RMS_NORM output, allocator reuses slot) |
| **2** | **26** | **0** | **6144** | **`[0, 0, 0, 0]`** (V SET_ROWS source — **all zeros**) |
| 3 | 26 | 524288 | 48 | `[0, 0, 1.4e-45, 0]` (V SET_ROWS indices, int payload) |
| **4** | **26** | **528384** | **6144** | **`[-3.09e-6, -1.52e-6, 4.02e-6, 4.66e-6]`** (K SET_ROWS source — **valid K**) |
| 5 | 26 | 1052672 | 12288 | `[0, 0, 7.17e-43, 0]` (K SET_ROWS indices) |

Smoking gun: entries 2 and 4 are the SAME shape (6144 bytes = 1536
floats = 6 tokens × 256 K/V-head-dim × 4 bytes), targeting the same
buffer (h26), at the two offsets the slice-3 SET_ROWS reads from.
**Entry 2 (V) is all-zeros; entry 4 (K) is real K data.** This
is consistent with V's MUL_MAT either:

- **Sub-case CPU-A — "scratch passes through"**: the CPU buft tensor
  for V's projection result was allocated and zero-initialized; the
  scheduler then `set_tensor`'d the (uninitialized-but-zero) buffer
  into jsep_buf without ever running the MUL_MAT. Possible if the
  scheduler's split decision marked V's MUL_MAT as someone else's
  responsibility, then forgot to assign it.
- **Sub-case CPU-B — "wrong inputs"**: V's MUL_MAT did run on CPU
  but read its src1 (input) or src0 (weight) from a stale / zero
  CPU buft. Plausible if the input-embedding `set_tensor` to a CPU
  buft staging buffer fired AFTER V's MUL_MAT ran.
- **Sub-case CPU-C — "JSEP merger"**: V's MUL_MAT is in JSEP's
  cgraph (one of slices 0–10), but its dst is allocated at the
  *same* (h26, 0) the input-embedding occupies. Allocator-coalesced
  alias destroys V's output before SET_ROWS reads it. Stage 4.11's
  closure noted slice 0 RMS_NORM legitimately writes `[-1.30e-3,
  1.90e-3]` to (h26, 0); a CPU op then overwrites with `[5.44e-6,
  1.20e-5]`; another op zeros the slot. The set_tensor sequence
  matches: entry 0 → entry 1 → entry 2 are exactly that pattern
  in cross-backend-write form (input embedding upload → RMS_NORM
  result CPY-back → V projection result CPY-back-as-zeros).

The K side works because K's projection result lands at offset
`+528384` — a fresh allocator slot the input embedding never
touched. V's projection result lands at offset `0` — the slot
the input embedding occupies first, then RMS_NORM overwrites,
then a third write delivers zeros.

## What lands

Spike chat path **bit-identical** to Stage 4.5/4.9/4.10/4.11
baseline (instrumentation is read-only):

| metric | Stage 4.11 | Stage 4.12 | delta |
|--------|------------|------------|-------|
| `topId / topVal` | 593 / 0.159 | 593 / 0.159 | 0 |
| `GENERATED_TEXT` | `"ntiuhuihnerquant"` | `"ntiuhuihnerquant"` | 0 |
| 6 selftests | PASS | PASS | 0 |
| `make checkall` | green | green | 0 |
| Per-token decode (ms) | 126.04 | 129.36 | +2.6% (within noise; CPU log adds ~30 EM_ASMs at prefill, ~0 during decode steady state) |

`__interSliceLog` length 60, `__jsepGraphLog` length 30,
`__cpuGraphLog` length 30, `__setTensorLog` length 200 (capped).
Stage 4.11 entry markers reproduce: `s3enter.h26o0 = [0, 0]`,
`s3enter.h26o528384 ≈ [-3.09e-6, -1.52e-6]` (matches set_tensor
entry 4 — same K data, same offset).

Patch stack: **8 → 9** (one Stage-4.12 diagnostic patch +
public-helper export). The patch will revert as part of Stage
4.13's structural fix, dropping back to 8 + the real fix.

## What does NOT land

- **Outcome A (real English decode)** — not flipped. The Stage
  4.12 brief framed both Probe 2 and a structural fix as in-scope,
  but the data doesn't yet point at a single fix shape. The three
  CPU-A / CPU-B / CPU-C sub-cases each suggest a different fix
  (re-route V to JSEP, fix scheduler ordering for V's input prep,
  break the allocator coalescing). Stage 4.13 must disambiguate
  before landing the fix.

- **Allocator-coalescing hypothesis (CPU-C) confirmation** — strongly
  suggested by the matching set_tensor sequence at offset 0 (49152 →
  49152 → 6144) but not yet proven. A direct check would correlate
  the cgraph build's tensor allocation (which logical tensor maps to
  which (handle, offset)) with the runtime set_tensor sequence.

## Files touched (Stage 4.12 — diagnostic only)

| file | change |
|------|--------|
| `~/Repos/llama.cpp/ggml/include/ggml-jsep.h` | +9 LOC — declare `ggml_jsep_resolve_tensor` |
| `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` | +60 LOC — implement resolver + Probe 2 follow-up set_tensor logging |
| `~/Repos/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp` | +95 LOC — weak extern decl + EM_ASM `__cpuGraphLog` block |

`make wasm-build-jsep` green; `make checkall` green; selftests pass.

## Diagnosis (Stage 4.13 hand-off)

The next probe must distinguish CPU-A vs CPU-B vs CPU-C. Cheapest
sequence:

1. **CPU-C check** (allocator coalescing) — instrument `set_tensor`
   to record the *current* graph_compute_idx (already-emitted
   `__jsepGraphLog`/`__cpuGraphLog` IDs work as scaffolding). If
   the size=6144 zero write to (h26, 0) fires *after* the input-
   embedding 49152-byte writes but *before* slice 3, AND the dst
   tensor at the cgraph level is V's projection output, CPU-C is
   confirmed. Fix shape: pin V's projection dst to a non-coalesced
   slot via `ggml_set_input` / `ggml_format_name` or similar in
   libllama's graph builder, OR widen JSEP's `supports_op` /
   `offload_op` to capture V's MUL_MAT in JSEP rather than CPU.
2. **CPU-B check** (wrong inputs) — instrument the CPU MUL_MAT
   in `ggml-cpu/ops.cpp`'s MUL_MAT path with first-byte logging
   on src0/src1/dst. If V's MUL_MAT runs but reads zeros, fix is
   to ensure the input is set_tensor'd into a CPU staging buft
   before the MUL_MAT.
3. **CPU-A check** (op skipped) — temporarily widen jsep
   `supports_op` for *all* MUL_MAT shapes that pass the cap
   check, and re-run the spike. If V's MUL_MAT now runs on
   JSEP and produces real data, CPU-A was load-bearing — and
   the structural fix is to keep V on JSEP via offload_op or
   a libllama-side hint.

CPU-C is the most likely sub-case given the size-49152 → size-6144
pattern at offset 0, so Stage 4.13 should start there.

Stage 4.12 closes here: the brief asked for localization + fix in
one stage; the data showed the localization layer needed splitting
and the fix's shape isn't yet single-valued. **The Stage 4.13 brief
in `TODO.md` queues the disambiguation probe, then the structural
fix — same shape as 4.10 → 4.11 (probe-first, fix-second).**
