# Stage 4.2 — jsepWrite byte-dump probe

**Date:** 2026-05-06
**Status:** **CLOSED — diagnosis localized; Outcome A "Paris" decode NOT achieved.**
**Patch stack:** 6 (unchanged; no llama.cpp patches added).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.2-final-baseline`
**Per-token decode:** ~24.3 ms (post-Stage-4.1 baseline; Stage 4.2 was diagnostic-only).

## Setup

Per the Stage 4.2 brief, instrumented `smoke-test/p2-v2-spike.src.ts` with
five new probes (none touching `src/`):

1. **`JSEPWRITE_LOG`** — first 30 `mod.jsepWrite` calls after model load,
   capturing `(handle, offset, size, first8 f32, first16 raw u8)`. The
   wrap fires after `installJsepCallbacks`, so model-load weight uploads
   (134 calls) are excluded.
2. **`JSEPREAD_LOG`** — first 30 `mod.jsepRead` calls (post-await `.then()`
   handler captures dst data, with try/catch for heap-detach safety).
3. **`JSEPRUN_LOG`** — first 30 `mod.jsepRunOp` calls. Captures the C++-side
   descriptor (`op`, `n_src`, `dst.{handle,offset}`, per-src
   `{handle,offset,type}`), the dispatch return status, and a `divert`
   flag (computed JS-side as `srcs.some(sr.h === dstH)` — matches the
   in-source aliasing condition for matmul/RMS_NORM/SET_ROWS).
4. **`PREPREFILL_BUF11`** — direct WebGPU readback of `dataManager.get(11)`
   at the offsets that JSEP ops target (0, 524288, 2101248, 4194304,
   6295552), captured **before** any prefill op dispatches.
5. **`POSTPREFILL_BUF11`** — same readback after `bridge.decode(prompt)`
   returns. By then all prefill JSPI awaits have resolved, so this is
   the steady-state result of every dispatch the prefill graph emitted.
6. **`GPU_ERR_LOG` / `GPU_ERR_COUNT`** — `device.addEventListener("uncapturederror", …)`
   captures any GPUValidationError / GPUOutOfMemoryError / internal error
   that surfaces during the run.
7. **`LIVE_BUFFERS`** — `dataManager.handles.entries()` snapshot post-load
   (handle → bucketed size).

## Empirical findings

### 1. Buffer 11 is zero-initialized post-load (not NaN-initialized)

```
PREPREFILL_BUF11 = {
  "0":       [0,0,0,0,0,0,0,0],
  "524288":  [0,0,0,0,0,0,0,0],
  "2101248": [0,0,0,0,0,0,0,0],
  "4194304": [0,0,0,0,0,0,0,0],
  "6295552": [0,0,0,0,0,0,0,0]
}
```

This contradicts the Stage 3.5 framing that called the corruption
signature an "uninitialized memory pattern from a NaN-initialized GPU
buffer". On Apple/Chrome/Dawn for these `STORAGE | COPY_SRC | COPY_DST`
buffers, the post-allocation contents observed by a `copyBufferToBuffer`
+ `mapAsync` round-trip are **all zeros**, not NaN.

`counters@load = {alloc:6, free:0, write:134, read:0, clear:1, runOp:0,
sync:3}` — six allocs + one `jsepClear` during model load. The 134
writes are the model-weight uploads to bufs 6/7/8/9 (see
`LIVE_BUFFERS`). Buf 11 (64 MiB, the activation/intermediate buffer)
gets `clear`-ed once during load and is otherwise untouched by load
flow.

### 2. Live buffer inventory

```
LIVE_BUFFERS = [
  {h:6,  size:134217728, bucket:9},   // 128 MiB — model weights (Q4_K)
  {h:7,  size:134217728, bucket:9},   // 128 MiB — additional weights
  {h:8,  size:134217728, bucket:9},   // 128 MiB — additional weights
  {h:9,  size:134217728, bucket:9},   // 128 MiB — output / lm_head?
  {h:10, size:16777216,  bucket:7},   // 16 MiB  — KV cache shard
  {h:11, size:67108864,  bucket:8},   // 64 MiB  — activations + scratch
]
```

Offset 4194304 = 4 MiB, well inside buf 11 (size 64 MiB). Out-of-bounds
ruled out.

### 3. `jsepRunOp` dispatches all hit the divert path (first 30 captured)

Sample (op 25 = RMS_NORM, 29 = MUL_MAT, 42 = SET_ROWS; src `t12` =
GGML_TYPE_Q4_K, `t27` = GGML_TYPE_I64, `t1` = F16, `t0` = F32):

```
i= 0 RMS_NORM   dst=[11+0]        src0=[11+0/F32]    status=0 divert=true
i= 1 MUL_MAT    dst=[11+4194304]  src0=[6+0/Q4K]     src1=[11+0/F32]    status=0 divert=true
i= 2 MUL_MAT    dst=[11+4194304]  src0=[6+2359296/Q4K] src1=[11+0/F32]  status=0 divert=true
i= 3 SET_ROWS   dst=[10+0]        src0=[11+0/F32]    src1=[11+524288/I64] src2=[10+0/F16] status=0 divert=true
i= 4 SET_ROWS   dst=[10+262144]   src0=[11+528384/F32] src1=[11+1052672/I64] src2=[10+262144/F16] status=0 divert=true
…
```

All 30 logged ops report `divert=true` because the Llama scheduler
packs activation src1 + matmul output dst into the *same* jsep_buf
(buf 11), and SET_ROWS is structurally `dst = view(src[2])`. So the
divert path (Stage 3.5/4.1) fires for every JSEP-supported op in the
prefill graph that we captured.

The first non-divert dispatch — if any — must be **further into the
graph** than runOp i=29 (only ~1 layer's worth of ops captured at
RUN_MAX=30). The likely candidate is `lm_head` at the end of the graph,
which projects to a logits tensor that may live in a different
JSEP buffer (likely buf 9, the 128 MiB output buffer above) — making
`dst.bufHandle ≠ src*.bufHandle` and routing it through the batched
(non-divert) path.

### 4. **GPU_ERR_COUNT = 0** — every dispatch passes WebGPU validation

```
GPU_ERR_LOG   = []
GPU_ERR_COUNT = 0
```

Stage 3.5's smoking gun was a same-pass aliasing validation error that
silently dropped the dispatch. **Stage 4.2 confirms validation now
passes for every dispatch.** The current failure mode is *not* a
silent rejection at `encoder.finish()`.

### 5. **POSTPREFILL_BUF11 — every JSEP-targeted offset reads canonical NaN**

```
POSTPREFILL_BUF11 = {
  "0":        [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN] (firstBytes [0,0,192,127] x4)
  "524288":   [0,…,1.4e-45,…]          ← position-id slot (int64 read as f32 = denormals)
  "528384":   NaN
  "1052672":  [0,…,7.17e-43,…]         ← position-id slot
  "2101248":  NaN
  "4194304":  NaN
  "6295552":  NaN
  "17829888": NaN
  "35655680": NaN
}
```

The `[0,0,192,127]` byte pattern is **0x7fc00000** little-endian —
canonical IEEE 754 quiet NaN. Every f32 slot in buf 11 that the JSEP
op chain *should* have written to instead reads as NaN. The only
non-NaN slots are the int-typed position-id arrays (524288, 1052672)
that CPU ops upload directly via `jsepWrite` of int64 data.

This is **not** the corruption signature from Stage 3.5
(`[-5.16e-5, 142.08, -4.48, -7.43e+18, …]` — that was the value
*pulled to CPU before* the (since-fixed) silent matmul drop).
Stage 4.2's signature is canonical NaN, which is computed output
from a JSEP shader, not stale GPU memory.

### 6. CPU `jsepWrite` payloads — corruption present, but not canonical NaN

Out of 30 captured `jsepWrite` calls:

- 4 entries write valid data (e.g., `i=0` writes the embedding output
  to offset 0 with first8 = `[-0.0013, 0.0019, -0.0019, 0.0038, …]`).
- 5 entries write the Stage-3.5 corruption signature
  (`[-5.16e-5, 142.08, -4.48, -7.43e+18, …]` — uninitialized-CPU-heap
  pattern).
- 21 entries write canonical-NaN-filled buffers (`[0,0,192,127] x N`).

The canonical-NaN payloads are the CPU faithfully copying back what it
read from JSEP (via `jsepRead`) → the cascade is JSEP→CPU→JSEP, with
the CPU as a passive courier of NaN that originated in a JSEP shader.

### 7. `jsepRead` returns canonical NaN for most offsets

The first `jsepRead` (i=0, offset 0, size 49152) returns the valid
embedding `[-0.336, 0.492, -0.502, 0.989, …]`. **Every subsequent
read** of buf 11 at offsets the schedule treats as activation slots
returns canonical NaN. Two reads at offset 6295552 (i=7 size 135168
and i=8 at offset 17829888 size 135168) return real small floats
(~1e-4 range) — but reads at the SAME offset 6295552 with size 196608
or 49152 return NaN. So the 6295552 region has *some* valid data in
its lower bytes (133168 of them) and NaN in its upper bytes.

### 8. Final logits are all zero, not NaN

```
LOGIT_STATS_STEP0 = {first8:[0,0,0,0,0,0,0,0],
                     finiteCount:32000, hasNaN:false, hasInf:false,
                     minFinite:0, maxFinite:0}
GENERATED_TOKENS = [0,0,0,0,0]
```

If the lm_head input (last hidden state) is NaN — and per finding 5,
the activation buffer contains NaN at every offset — the lm_head
matmul output should also be NaN. **It isn't.** Logits are exactly
zero. PREPREFILL_BUF11 confirms zero is the GPU buffer's initial
state. So:

> **The lm_head matmul dispatch DID NOT WRITE TO ITS DESTINATION
> BUFFER.** The dst slot stayed at its post-allocation zero state.

This points to a *second* failure mode separate from the
NaN-cascade in buf 11: a JSEP op (likely the lm_head non-divert
dispatch routed through the batched encoder) silently produces no
output.

## Diagnosis (per brief Step 3 branch)

**Branch (c) primarily:** "CPU writes look correct AND go to the right
offset; the bug is downstream" — confirmed in part. CPU `jsepWrite`
delivers the embedding correctly and faithfully copies what `jsepRead`
returns. The bug is downstream of CPU↔JSEP marshaling.

**Two distinct downstream bugs surfaced by Stage 4.2:**

### Bug A — JSEP-supported ops compute (and write) canonical NaN

The very first `jsepRunOp` (RMS_NORM, dst=[11+0], src0=[11+0]) operates
on a known-valid input (jsepRead i=0 retrieved the raw embedding from
offset 0). After the prefill graph runs, offset 0 of buf 11 reads
canonical NaN. So either:

1. **RMS_NORM produces NaN at production shape (rows=6, cols=2048).** The
   Stage 3 self-test only exercises rows=1, cols=2048. The kernel's
   per-thread row-sum loop has no reduction across threads, so each
   thread independently recomputes the row sum — correct in isolation
   but untested at multi-row scale.
2. **A later op (likely matmul i=25, which targets dst=[11+0]) computes
   NaN** because its src1 (offset 2101248) is already NaN (read from
   the chain).
3. **MUL_MAT at production shape** (M=2048, K=2048, N=6) has a Q4_K
   kernel bug that yields NaN — Stage 3 self-test only validates
   K=256, M=1, N=1. The Q4_K dequant per super-block is identical for
   K=256 vs K=2048, but the inner loop runs 8x more super-blocks; if
   a single block has `d=NaN` (f16 unpacking issue) or any per-row
   sum overflows to ±Inf, the entire row collapses to NaN.

### Bug B — lm_head dispatch (non-divert) silently doesn't write

Final logits stay at zero (the buffer's pre-prefill state), not NaN.
For NaN inputs to produce zero output, the dispatch must not have
landed at all. Since this is *the last op* and `divert=true` for the
first 30 captured ops, the lm_head likely goes through the
**non-divert / batched path** (when dst lives in a different JSEP
buffer than the activation source). The batched encoder's submit
either fails silently somewhere not surfaced as a `uncapturederror`,
or the bind group has a subtle issue (e.g., `dst.offset` outside the
buffer, wrong copy-back, etc.).

We have not yet captured runOp entries beyond i=29 (RUN_MAX=30), so
the lm_head's exact `(op, dst, srcs)` and divert-status is unknown.

## What was eliminated

- ✗ "Buffer is NaN-initialized and ops never write" — disproven by
  PREPREFILL = zeros.
- ✗ "WebGPU validation silently rejects dispatches" — disproven by
  GPU_ERR_COUNT = 0 (Stage 3.5's bug was real but is now actually
  fixed by the divert pattern; nothing equivalent fires now).
- ✗ "tempDst.destroy() too early invalidates in-flight copy" —
  disproven by removing the destroy and observing identical
  POSTPREFILL_BUF11 NaN pattern.
- ✗ "Type mismatch — Q4_0 file misread as Q4_K" — disproven by
  GGUF metadata (`general.file_type=15` = Q4_K_M) matching the
  descriptor's `t12` (Q4_K).
- ✗ "Out-of-bounds buffer access" — disproven by LIVE_BUFFERS
  showing buf 11 is 64 MiB and dst.offset never exceeds ~36 MiB.

## Stage 4.3 — queued probes

Two parallel sub-probes, either of which (alone) might flip Outcome A:

### 4.3a — RMS_NORM multi-row + MUL_MAT production-shape selftests

Add to `smoke-test/p2-v2-spike.src.ts`, before [4/8] webgpu_init:

1. **Multi-row RMS_NORM**: rows=6, cols=2048, deterministic input
   pattern; CPU reference; assert `maxAbsDelta < 1e-5` on every row.
2. **MUL_MAT at production shape**: M=2048, K=2048, N=6, src0=Q4_K
   (use a hand-quantized synthetic block as in Stage 3, or extract a
   real layer-0 Q tensor from buf 6 via `jsepRead`); CPU reference;
   assert `maxAbsDelta < 1e-3`.

Either failing localizes Bug A. Both passing → the kernel is fine and
Bug A is in graph orchestration (cross-dispatch sync, bind-group reuse,
shape uniform staleness, etc.) — pivot to instrumented per-op readback.

### 4.3b — capture all `runOp` dispatches + correlate with NaN producer

Raise `RUN_MAX` from 30 to ~1700 (covers full prefill + 5 decode steps
worth of dispatches). Run the spike, scan the log for:

- The **first runOp with `divert=false`** — likely lm_head; verify dst
  buffer handle, offset, and dst tensor shape match expectations.
- Any runOp that returns `status ≠ 0`.
- Counts of (op, divert) — should be ~22 layers × N_ops_per_layer for
  the divert ops; lm_head should appear once.

Then add an async-scheduled GPU readback after each of the first ~10
runOps (use `setTimeout(0)` to defer; collect results into an array
indexed by runOp index). The first runOp whose dst reads back as NaN
is the first NaN producer.

## Files touched (all under `smoke-test/`, no `src/` changes)

- `smoke-test/p2-v2-spike.src.ts` — added Stage 4.2 diagnostic blocks:
  - jsepWrite/jsepRead/jsepRunOp wrappers (window-exposed `__jsepWriteLog`,
    `__jsepReadLog`, `__jsepRunLog`).
  - Pre-prefill and post-prefill GPU buffer dumps (`PREPREFILL_BUF11`,
    `POSTPREFILL_BUF11`).
  - WebGPU `uncapturederror` capture (`__gpuErrLog`).
  - Live-buffer inventory (`LIVE_BUFFERS`).
  - All blocks marked `Stage 4.2 …` so Stage 4.3 / Stage 5 can identify
    and remove them before `make checkall` ship-gate.
- Two transient probe edits to `src/inference/jsep/ops/{matmul,rms-norm,set-rows}.ts`
  (commented out `tempDst.destroy()`) were tested and **reverted** —
  no source changes ship.

## Closure summary for `TODO.md`

Stage 4.2 closes with diagnosis localized to **two parallel bugs**:

- **Bug A**: JSEP-supported ops produce canonical NaN at every
  activation/intermediate offset in buf 11 by post-prefill, despite
  zero validation errors and the divert path firing on every dispatch.
  Likely first NaN producer is RMS_NORM at multi-row shape (untested)
  or MUL_MAT at production K=2048 shape (untested).
- **Bug B**: lm_head dispatch (presumed non-divert, batched path)
  silently leaves its dst buffer at the post-allocation zero state.
  Final logits are all zero rather than the NaN-cascade we'd expect
  from NaN inputs through a working matmul.

Stage 4.3 brief queues 4.3a (kernel selftest at production shapes)
and 4.3b (full-graph runOp capture + per-op readback).
