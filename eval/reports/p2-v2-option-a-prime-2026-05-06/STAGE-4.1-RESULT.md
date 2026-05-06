# Stage 4.1 — SET_ROWS divert (structural fix landed; not the load-bearing cause of Outcome C)

**Date:** 2026-05-06
**Outcome:** **Exit criterion (b)** from the Stage 4.1 brief. SET_ROWS
divert lands cleanly with measured behavior matching the hypothesis
(100% src[2] aliasing rate, 264/264 dispatches diverted), but
`LOGIT_STATS_STEP0` remains all-zero and `GENERATED_TOKENS = [0,0,0,0,0]`.
The SET_ROWS aliasing was a real latent bug worth fixing, but it is
*not* the load-bearing cause of the Outcome C all-zero collapse. Next
suspect is **CPU-side writeback** (`jsepWrite`) for unsupported ops
(GET_ROWS / MUL / ADD / SCALE / SOFT_MAX / ROPE), to be investigated
in **Stage 4.2** per the brief's Step 4 fallback.

## TL;DR

Stage 3.5 identified a WebGPU compute-pass synchronization-scope
violation as the Outcome-C root cause for matmul + RMS_NORM, then landed
divert fixes for both. Stage 4.1's hypothesis: SET_ROWS has the same
structural bug because `dst = view_tensor(src[2])` is *definitionally*
aliased — KV cache writes always have `dst.bufHandle === src[2].bufHandle`.

**Measurements confirmed the hypothesis exactly:**

| Run | total | aliasesSrc0 | aliasesSrc1 | aliasesSrc2 | divert fires |
|---|---|---|---|---|---|
| Step 1 (counter only) | 264 | 0 | 0 | **264 (100%)** | n/a |
| Step 3 (after divert) | 264 | 0 | 0 | 264 | **264 (100%)** |

So src0/src1 never alias dst (the bound inputs are clean), and src[2]
*always* aliases dst (the structural view relationship holds for every
KV cache write). The divert correctly handles the alias case.

**But the fix doesn't unblock Outcome A.** Logits remain identical to
the post-Stage-3.5 baseline (all-zero, finiteCount 32000, min=max=0)
and per-token decode is 23.74 ms vs the 24.30 ms Stage-3.5 baseline
— within noise. The divert overhead (264 extra
createBuffer + 2 × copyBufferToBuffer + queue.submit per
prefill+5-decode-step run) is invisible at this scale.

This rules out SET_ROWS as the upstream producer corrupting
`attn_q.src1`. The activation feeding the first attn_q matmul still
shows the same `[-5e-5, 142.08, -4.48, -7.4e+18, ...]` byte pattern
(uninitialized memory) that Stage 3.5 caught — so something earlier in
the prefill graph is failing silently. The Stage 3.5 candidate ranking
puts CPU-side writeback as the next suspect.

## Why land the divert anyway

It's still correct. The aliasing rate is structural — every SET_ROWS
dispatch has `dst.bufHandle === src[2].bufHandle`. Even if WebGPU's
validator happens to *not* fire on the SET_ROWS path right now (because
the bound bind-group entries themselves don't alias — only the wider
batched-pass usage might), the latent failure mode exists. Future
scheduler changes that pack additional ops binding `src[2]`'s buffer
read-only into the same batched pass as a SET_ROWS could trigger the
same encoder.finish() rejection that Stage 3.5 caught for matmul.

The divert pattern is small (~80 LOC) and adds zero measurable cost in
the canonical workload. Keeping it removes a class of regressions the
debugger would otherwise have to re-discover later.

## Steps executed

### Step 0 — verify Stage 3.5 baseline reproduces

`?v=stage4.1-replay`:
- `Q4K_SELFTEST` delta 4.5e-6 ✓
- `RMSNORM_SELFTEST` maxAbsDelta 8.3e-7 ✓
- `LOGIT_STATS_STEP0 first8 = [0,0,0,0,0,0,0,0]`, min=max=0, finiteCount=32000 ✓
- `GENERATED_TOKENS = [0,0,0,0,0]`, GENERATED_TEXT = "" ✓
- `PER_TOKEN_MS = 24.30` ✓ (matches Stage 3.5 closure baseline)

Outcome C reproduces cleanly; ready to instrument.

### Step 1 — SET_ROWS aliasing rate measurement

Added a temporary `__setRowsStats` counter to `dispatchSetRows`
incrementing total + per-src-bufHandle alias hits. Added a
`SETROWS_STATS` readout to `smoke-test/p2-v2-spike.src.ts`. The brief
hypothesized `dst.bufHandle === src0.bufHandle` (the matmul template's
within-bind-group alias check); my counter widened the check to all
three srcs because the brief explicitly noted the descriptor layout
might map src[2] (the destination buffer in ggml semantics) to
`desc.srcs[2]` rather than to `srcs[0]` or `srcs[1]`.

`?v=stage4.1-setrows-stats` result:

```
SETROWS_STATS = {"total":264,"aliasesSrc0":0,"aliasesSrc1":0,"aliasesSrc2":264}
```

100% src[2] aliasing — exactly the structural pattern the brief
predicted, just on a different descriptor slot than the matmul
template assumed.

### Step 2 — divert implementation

Implemented in `src/inference/jsep/ops/set-rows.ts`. Mirrors the
matmul Stage 3.5 divert pattern with two SET_ROWS-specific extensions:

1. **Alias check broadened to src[2]:**

   ```ts
   const src2BufHandle = desc.srcs.length > 2 ? desc.srcs[2].bufHandle : -1;
   const dstAliasesSrc =
       dst.bufHandle === src0.bufHandle ||
       dst.bufHandle === src1.bufHandle ||
       (src2BufHandle >= 0 && dst.bufHandle === src2BufHandle);
   ```

2. **Read-modify-write semantics** (vs matmul's write-only divert):
   SET_ROWS is a *partial* update — the kernel writes only the rows
   selected by the indices and the F16 dst path uses an
   `atomicCompareExchangeWeak` loop. A naive temp + post-copy pattern
   would clobber the unwritten rows with uninitialized temp memory
   AND would feed the CAS path the wrong "current" value. Pre-copy
   real dst → temp before the dispatch fixes both.

   ```ts
   enc.copyBufferToBuffer(dstRec.buffer, dst.offset, tempDst, 0, dstSize);
   pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
   pass.end();
   enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstSize);
   ```

3. **dstSize derivation handles strided cache_v transposed view**
   (the `ne[0]=1, ne[1]=ggml_nelements(v)` shape from
   `llama-kv-cache.cpp:1281`):

   ```ts
   let dstSize = dst.ne[0] * dst.nb[0];
   for (let d = 1; d < 4; d++) {
       if (dst.ne[d] > 0) dstSize = Math.max(dstSize, dst.ne[d] * dst.nb[d]);
   }
   ```

   Validated against `dstRec.size` to catch out-of-bounds before the
   pre-copy reads past the buffer end.

The divert lives in its own command encoder; `encoderBatcher.flush()`
runs first so the diverted dispatch can't conflict with batched
neighbours. `tempDst.destroy()` after `queue.submit` is
documented-safe — pending GPU work continues using the underlying
memory.

### Step 3 — verify Outcome A (negative)

Re-ran spike at `?v=stage4.1-divert-verify` with a `__setRowsDivertFires`
counter added to confirm the divert path actually executes.

```
LOGIT_STATS_STEP0 = {"step":0,"first8":[0,0,0,0,0,0,0,0],"topVal":0,"topId":0,"hasNaN":false,"hasInf":false,"finiteCount":32000,"minFinite":0,"maxFinite":0}
SETROWS_STATS = {"total":264,"aliasesSrc0":0,"aliasesSrc1":0,"aliasesSrc2":264}
SETROWS_DIVERT_FIRES = 264
GENERATED_TOKENS = [0,0,0,0,0]
GENERATED_TEXT = ""
PER_TOKEN_MS = 23.74
COUNTER_DELTAS = {"alloc":0,"free":0,"write":1206,"read":1266,"clear":0,"runOp":1602,"sync":3671}
```

- Divert fires for every SET_ROWS call (264/264). ✓
- Logits still all-zero. ✗ (Outcome A not achieved)
- Per-token unchanged within noise (23.74 ms vs Stage-3.5 24.30 ms baseline).
- JSEP counters unchanged from Stage 3.5 baseline — divert path is
  invisible to `__jsep.counters` (it allocates raw `GPUBuffer`s
  outside the JSEP-buf flow and submits its own command encoder
  separately).

This matches the brief's exit criterion (b) verbatim. The remaining
suspect is **CPU-side writeback** for ops the JSEP backend doesn't
support (`~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:584-650`
lists the supports_op gate; CPU fallback for unsupported ops triggers
`get_tensor` → `jsepRead` and `set_tensor` → `jsepWrite` round-trips).
A wrong offset in `jsepWrite` would write CPU output back to the
wrong slot of jsep_buf, corrupting whatever tensor lives there
(possibly the activation feeding attn_q).

## Closure cleanup

- Removed the temporary `__setRowsStats` and `__setRowsDivertFires`
  counters from `set-rows.ts` and the matching `SETROWS_STATS` /
  `SETROWS_DIVERT_FIRES` log lines from `p2-v2-spike.src.ts`. The
  divert itself stays.
- `make checkall` green: 747 pass / 0 fail (Biome formatted
  `set-rows.ts` once for trailing-whitespace nits in the new comment
  block).
- Patch stack unchanged at +6 (zero llama.cpp changes).

## What ships

- `src/inference/jsep/ops/set-rows.ts`: divert with read-modify-write
  semantics. ~80 LOC added.
- `smoke-test/p2-v2-spike.src.ts`: cleanup-only (diagnostic readouts
  removed).

## Outcome / next

**Stage 4.1 closes** with the SET_ROWS divert landed as a structural
correctness fix (not as the Outcome-C unblock).

**Stage 4.2 queued** per the brief's Step 4 fallback: instrument
`jsepWrite` to log the first 8 bytes of each CPU-side write during
prefill, identify which CPU-fallback op (most likely GET_ROWS for
the initial token-embedding lookup, but MUL/ADD for attention norm
biases are also plausible) is targeting the wrong jsep_buf offset,
and either fix the offset arithmetic or move the op to JSEP.

## Files / commits

- Code: `src/inference/jsep/ops/set-rows.ts` (divert path).
- Closure: this file (`STAGE-4.1-RESULT.md`) — force-add (in gitignored
  reports tree).
- TODO: closure stub queues Stage 4.2 with the brief's Step 4 jsepWrite
  byte-dump as the entry point.
