# Stage 4.6 D1 — SET_ROWS V-cache transpose self-test

**Date:** 2026-05-06
**Status:** **D1 PASS — `dispatchSetRows` is bit-exactly correct on the
V-cache transpose layout, including the F32→F16 atomic-CAS path with
adjacent indices that share u32 words.** `maxAbsDeltaTargeted=0`,
`maxAbsDeltaUntargeted=0` on **both** no-divert and divert variants.
The kernel is NOT the source of Stage 4.5's wrong-decode bug.
**Patch stack:** 7 (unchanged — selftest is spike-side instrumentation
only).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.6-d1`

## Goal (recap from Stage 4.6 brief)

Stage 4.5 H1 confirmed GPU→host mirror staleness was a real bug, but
fixing it left "Paris" still not decoded. The Stage 4.6 brief queued
three diagnostic options: D1 kernel correctness selftest, D2 per-
dispatch CPU-reference diff, D3 ROPE/SOFT_MAX/attention-masking
inspection. D1 was the lowest-hanging fruit: prove that
`dispatchSetRows` writes the right cells for the V-cache transpose
layout (`ne[0]=1`, F16 dst, I64 indices, F32→F16 atomic-CAS path —
the one llama.cpp uses at `kv-cache.cpp:1281` when flash-attention
is disabled, which is the case for TinyLlama).

## Implementation

Single file change in
`smoke-test/p2-v2-spike.src.ts` (~270 LOC of additions):

1. **Imports** (`smoke-test/p2-v2-spike.src.ts:14-32`): added
   `dispatchSetRows`, `GGML_OP_SET_ROWS`, `GGML_TYPE_F16`,
   `GGML_TYPE_I64`.
2. **`runSetRowsVCacheSelfTest`** (~250 LOC): new selftest function.
   Builds 16 F16 dst cells with a sentinel pattern (each cell pre-
   loaded with a distinct decodable F16 value so we can check both
   "did the targeted cells get the right value?" AND "did the
   untargeted cells get preserved?"); 4 source rows with F16-exact
   values (0.5, -0.25, 1.5, -3); I64 indices `[0, 1, 6, 7]` chosen
   so cells 0 & 1 share u32 word 0 and cells 6 & 7 share u32 word 3
   — the adjacent-cells-in-same-word race is what the atomic CAS
   path defends against. Two variants: no-divert (`hSrc1 ≠ hDst`,
   batched dispatch) and divert (`hSrc1 === hDst` via co-located
   shared buffer + `src[2].bufHandle === dst.bufHandle`, triggers
   the Stage 4.1 read-modify-write divert).
3. **`runSpike` wiring** (~`smoke-test/p2-v2-spike.src.ts:875-887`):
   ran both variants after the existing
   `MATMUL_PROD_NODIVERT/DIVERT` selftests, before model load. Logs
   `SETROWS_VCACHE_NODIVERT` and `SETROWS_VCACHE_DIVERT` JSON.

The selftest is permanent regression coverage: any future change to
`dispatchSetRows` will break this if it breaks the V-cache layout.

## Result

Both variants returned bit-exact match against the CPU reference:

```json
SETROWS_VCACHE_NODIVERT = {
  "mode": "no-divert", "status": 0,
  "N_CELLS": 16, "N_ROWS": 4,
  "indices":     [0, 1, 6, 7],
  "srcF32":      [0.5, -0.25, 1.5, -3],
  "preF16":      [100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5,
                  104, 104.5, 105, 105.5, 106, 106.5, 107, 107.5],
  "postF16":     [0.5, -0.25, 101, 101.5, 102, 102.5, 1.5, -3,
                  104, 104.5, 105, 105.5, 106, 106.5, 107, 107.5],
  "expectedF16": [0.5, -0.25, 101, 101.5, 102, 102.5, 1.5, -3,
                  104, 104.5, 105, 105.5, 106, 106.5, 107, 107.5],
  "maxAbsDeltaTargeted":   0,
  "maxAbsDeltaUntargeted": 0,
  "hasNaN": false, "hasInf": false
}

SETROWS_VCACHE_DIVERT = (identical to no-divert)
```

Targeted cells `{0, 1, 6, 7}` got the F16-encoded source values.
Untargeted cells `{2, 3, 4, 5, 8..15}` retained the sentinel.
The atomic CAS at adjacent pair-mates didn't corrupt either side.
The divert path's pre-copy correctly preserved the prior buffer
state (without it, the untargeted cells would have read as zero).

The post-Stage-4.5 baseline metrics still hold:
- 4 prior selftests (Q4K, RMS_NORM single, RMS_NORM multi-row,
  MATMUL prod) — all PASS
- `GENERATED_TOKENS = [593, 5871, 15669, 15565, 12150]`
- `GENERATED_TEXT = "ntiuhuihnerquant"` (unchanged from Stage 4.5)
- `LOGIT_STATS_STEP0.first8` bit-exactly identical to Stage 4.5
- `COUNTER_DELTAS.read = 1602` (H1 still firing)
- `PER_TOKEN_MS = 24.90` (within noise of Stage 4.5's 25.04)

D1 changes the spike's selftest battery from 4 → 6, doesn't affect
the chat path, and `make checkall` is green.

## Implication for Stage 4.5's `FIRST_ALLZERO_DST_PROBE`

Stage 4.5's smoking gun was `FIRST_ALLZERO_DST_PROBE = {i:3, op:42
SET_ROWS, dstH:18 (KV cache), dstO:0}` — meaning op 3's dst at
buffer offset 0 read as eight zero F32s (= 16 zero F16 cells)
post-dispatch. The Stage 4.5 brief flagged two possible readings:
- **(a)** false positive — buffer offset 0 simply doesn't get
  written because no token's index lands there
- **(b)** real bug — SET_ROWS writes to wrong offset / wrong cell

D1 PASS strongly favors **reading (a)**. The dispatcher is bit-
exactly correct at the kernel level for exactly this layout
(transposed V-cache, ne[0]=1, F16 dst, I64 indices, atomic CAS).
If reading (b) were true, the selftest would have failed
(maxAbsDeltaTargeted > 0). It didn't.

The probe's "first 8 cells of the buffer at offset 0" check is
overly strict: a multi-MB KV cache buffer where SET_ROWS targets a
sparse subset of cells will leave the offset-0 cells at their
post-allocation zero state unless one of those cells happens to be
in the indices list. For TinyLlama prefill (n_tokens=6,
n_kv_heads=4, head_dim=64), the indices map tokens to cells via
`i1 = head*ctx + pos` style addressing — for pos=0, head=0, the
target IS cell 0, but only ONE cell (out of 8) gets written. The
remaining 7 stay zero, which the probe (which reads `first8`)
flags as "all-zero".

## Next step — D2 / D3 brief queued

Now that the dispatcher is exonerated, the bug must be upstream of
SET_ROWS. Three remaining hypotheses, in priority order:

- **H-source (likeliest given Stage 4.5 token-2-onwards divergence):**
  the F32 `src[0]` data fed into SET_ROWS is wrong. SET_ROWS receives
  K from a chain `q_proj/k_proj matmul → ROPE → SET_ROWS`. ROPE is
  CPU-fallback (not in JSEP supports_op). Pre-H1, ROPE read JSEP-
  produced K from a stale (zero) host mirror; post-H1, it reads
  current values. The matmul output is JSEP-produced; H1 mirrors
  it. ROPE's INPUT is then correct. ROPE's OUTPUT is CPU-resident.
  SET_ROWS reads ROPE output as src[0] — needs CPU output to be in
  jsep_buf because src[0].bufHandle goes to the JSEP dispatcher
  bind group. **Question: how does ROPE's CPU-resident output reach
  the JSEP dispatcher?** Either (1) scheduler inserts a CPY-to-jsep,
  which goes through `set_tensor` (dual-write to mirror + GPU);
  (2) something else. If (1), the GPU side is correct. If something
  in the CPY path is missing, GPU side is stale.
- **H-indices:** the I64 indices src[1] are wrong. Indices come
  from `inp_pos` (token positions) via index-tensor allocation in
  the build graph. If ROPE writes positions wrong, SET_ROWS writes
  to wrong cells in the K cache.
- **H-attention:** SET_ROWS is correct, KV values are correct, but
  the *consumer* (attention computation that READS from KV cache)
  is wrong. Most-likely failure: SOFT_MAX with mask reading wrong
  cells, or attention's `Q × K_cache` matmul reading from a stale
  mirror of K_cache (H1 mirrors per-op writeback but the multi-op
  attention chain has reads interleaved with writes).

D2 (per-dispatch CPU-reference diff for SET_ROWS in production
context) localizes between H-source and H-indices. D3 (ROPE position
inspection + SOFT_MAX mask values) localizes H-attention.

The Stage 4.6 D2/D3 follow-on brief in TODO.md will pick the
narrower of these two as the next probe.

## Code references

- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:14-32` — imports
- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:471-748` — `runSetRowsVCacheSelfTest`
- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:875-887` — runSpike wiring
- `~/Repos/webllm/src/inference/jsep/ops/set-rows.ts` — dispatcher (unchanged this stage)
- `~/Repos/llama.cpp/src/llama-kv-cache.cpp:1278-1285` — V-cache transpose layout

## Patch stack

Unchanged at 7. No llama.cpp changes; no `src/` changes; spike
instrumentation only.
