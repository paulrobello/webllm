# Stage 4.6 D2-lite — per-SET_ROWS source/indices/dst probe

**Date:** 2026-05-06
**Status:** **D2-lite landed; source data is sensible across all 10
captured SET_ROWS dispatches; indices are sensible; but the FIRST
SET_ROWS dispatch (i=3, K-cache layer 0) shows `dstPostFirst8U16 = [0, 0, 0, 0, 0, 0, 0, 0]` at end-of-decode readback while every
other captured SET_ROWS shows non-zero F16 cells.** Strongly
suggests **i=3 silently fails to write its dst** OR the dst buffer
region is overwritten with zeros between the dispatch and the
readback. Diagnosis sub-step closed; tighter Stage 4.7 probe
queued.
**Patch stack:** 7 (unchanged — D2-lite is spike instrumentation
only).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.6-d2lite`

## Goal

Stage 4.6 D1 proved `dispatchSetRows` is bit-exactly correct in
isolation. The remaining hypotheses (H-source / H-indices / H-
attention) need data from the production graph. D2-lite captures
src[0] (F32 K/V data), src[1] (I64 indices), and dst (F16 cells)
for the first 10 SET_ROWS dispatches via three additional staging
copies + mapAsyncs scheduled per-runOp via Promise microtasks.

## Implementation

`smoke-test/p2-v2-spike.src.ts` (~135 LOC of additions):
1. New `setRowsDiag` array + `SetRowsDiagEntry` type capturing op
   metadata (bufHandles, offsets, types, ne[0..3]) plus deferred
   readback fields.
2. Inside the wrapped `mod.jsepRunOp`: when op==42 (SET_ROWS) and
   we haven't yet captured `SET_ROWS_DIAG_COUNT=10`, build an entry
   with full ne arrays for src[0], src[1], dst.
3. Schedule a deferred Promise microtask that copyBufferToBuffer's
   32 bytes from src[0] (8 F32), 64 bytes from src[1] (8 I64; we
   read low 32 bits of each), and 16 bytes from dst (8 F16 cells).
   mapAsync each, populate the entry.
4. Await `setRowsDiagPromises` after `dstProbePromises` before
   emitting summary; log
   `SETROWS_DIAG_FIRST5 = ${JSON.stringify(setRowsDiag.slice(0, 5))}`.

`make checkall` green; spike chat path unchanged (decoded text
still "ntiuhuihnerquant", per-token decode 24.80 ms vs Stage-4.5
25.04 ms — within noise).

## Key data — all 10 captured SET_ROWS dispatches

Reading the first 10 SET_ROWS in TinyLlama's prefill+decode
graph (op=42, dstH=23 = the unified KV-cache buffer):

| i  | dstO    | dstNe          | src0Ne     | src1Ne | src0First8F32 (truncated)               | src1First8Idx                           | dstPostFirst8U16                                  |
|----|---------|----------------|------------|--------|-----------------------------------------|-----------------------------------------|---------------------------------------------------|
| 3  | 0       | [256, 512]     | [256, 6]   | [6]    | -1.067, 0.656, -0.110, -0.110, ...     | 0, 1, 2, 3, 4, 5, 0, 0                  | **0, 0, 0, 0, 0, 0, 0, 0** ← anomaly             |
| 4  | 262144  | [1, 131072]    | [1, 1536]  | [1536] | -0.0006, 0.0009, 0.0011, ...           | 0, 512, 1024, 1536, 2048, 2560, 3072, 3584 | 32820, 33200, 734, 362, 33433, 210, 0, 0       |
| 14 | 524288  | [256, 512]     | [256, 6]   | [6]    | -0.0006, 0.0009, 0.0011, ...           | 0, 1, 2, 3, 4, 5, 0, 0                  | 32820, 32794, 67, 78, 32952, 162, 32863, 57    |
| 15 | 786432  | [1, 131072]    | [1, 1536]  | [1536] | -1.067, 0.656, -0.110, -0.110, ...     | 0, 512, 1024, 1536, 2048, 2560, 3072, 3584 | 1206, 36837, 3289, 1855, 608, 2495, 0, 0       |
| 26 | 1048576 | [256, 512]     | [256, 6]   | [6]    | -1.067, 0.656, ...                     | 0, 1, 2, 3, 4, 5, 0, 0                  | 1206, 33112, 33335, 503, 32795, 33250, 32864, 206 |
| 27 | 1310720 | [1, 131072]    | [1, 1536]  | [1536] | -0.0006, 0.0009, ...                   | 0, 512, 1024, 1536, 2048, ...           | 33168, 35526, 60, 4291, 36740, 4202, 0, 0      |
| 39 | 1572864 | [256, 512]     | [256, 6]   | [6]    | -1.067, 0.656, ...                     | 0, 1, 2, 3, 4, 5, 0, 0                  | 1206, 33112, 33335, 503, 32795, 33250, 32864, 206 |
| 40 | 1835008 | [1, 131072]    | [1, 1536]  | [1536] | -0.0006, 0.0009, ...                   | 0, 512, 1024, ...                       | 33284, 5286, 39662, 37190, 6570, 1642, 0, 0    |
| 51 | 2097152 | [256, 512]     | [256, 6]   | [6]    | -0.0006, 0.0009, ...                   | 0, 1, 2, 3, 4, 5, 0, 0                  | 32820, 32794, 67, 78, 32952, 162, 32863, 57    |
| 52 | 2359296 | [1, 131072]    | [1, 1536]  | [1536] | -1.067, 0.656, ...                     | 0, 512, 1024, ...                       | 2372, 38037, 36850, 3934, 35498, 39516, 0, 0   |

Pattern in dstNe:
- `[256, 512]` = K-cache layer L (non-transposed; ne[0]=head_dim*n_kv_heads=64*4=256, ne[1]=n_ctx=512)
- `[1, 131072]` = V-cache layer L (transposed FA-disabled per `kv-cache.cpp:1281`; ne[0]=1, ne[1]=ggml_nelements(v))

Pattern in dstO progression:
- Layer 0 K @ 0; V @ 262144
- Layer 1 K @ 524288; V @ 786432
- Layer 2 K @ 1048576; V @ 1310720
- ... (each layer = 2 × 262144 bytes = 524288 bytes total)

So the 10 captured SET_ROWS span the first 5 layers' K+V cache writes. Only **i=3 (the very first SET_ROWS, layer 0 K-cache at dstO=0)** has all-zero dst.

## Source-data and indices look correct

`src0First8F32` for every dispatch shows sensible F32 values
(magnitudes < 1.5, finite, no NaN/Inf). Two distinct patterns repeat:
- K projection: `[-1.067, 0.656, -0.110, -0.110, 0.082, 0.082, ...]`
  — note the `(2,3)` and `(4,5)` pair-mate identity, consistent
  with ROPE rotation at position 0 (cos=1, sin=0 → identity rotation
  applied to K dimension pairs). Or with a coincidental K projection
  pattern.
- V projection: `[-0.0006, 0.0009, 0.0011, ...]` — small values,
  typical of V projection magnitudes.

`src1First8Idx` for K-cache writes: `[0, 1, 2, 3, 4, 5, 0, 0]` —
nr=6 valid indices (the 6 prefill token positions), padding zeros
beyond. Correct.

`src1First8Idx` for V-cache writes: `[0, 512, 1024, 1536, 2048, 2560, 3072, 3584]` — strided
indices with stride 512 for the transposed-V layout. With nr=1536,
many more indices follow; we're seeing the first 8.

**H-indices is REJECTED.** Indices look exactly correct.
**H-source is WEAKENED.** Source data is sensible. The "garbage
inputs" framing the Stage 4.5 brief proposed for ROPE-pre-H1
isn't visible at end-of-decode readback.

## The i=3 anomaly

i=3 stands alone: same dst shape as i=14, i=26, i=39, i=51
(all `[256, 512]` K-cache); same indices pattern (`[0,1,2,3,4,5]`);
sensible src0 data; but dst at end-of-decode reads all zeros.

Two readings:
- **Reading R1**: i=3's dispatch silently failed. The K-cache layer
  0 was never written. Position 0's K cache at layer 0 stays zero
  for the entire prefill+decode. Attention at every subsequent
  decode step computes against a zeroed-out position-0 K. Token 1
  is the immediate downstream consumer (it runs against prefill
  KV cache), so wrong K for position 0 explains why decoded tokens
  diverge starting at token 2 — but actually wait, token 1 also
  uses position-0 K, and it's deterministic across H1 on/off.
  Hmm. So R1 doesn't fully explain the symptom.
- **Reading R2**: i=3 wrote correctly, but a later op overwrote
  cells 0..7 of dst with zeros. Possible vectors: a ggml_jsep_clear
  call on the same range, or a divert pre-copy from another op
  that happens to target overlapping cells. None of the captured
  SET_ROWS write into K-cache offset 0, but we only captured 10 of
  ~110+ total SET_ROWS dispatches; later dispatches (decode-time
  K-cache writes for token positions 6..10 at layer 0) DO target
  K-cache layer 0 at offsets corresponding to those positions.
  Position 6's K cache lives at cells 6*256..7*256-1, NOT cells
  0..255. So those decode-time SET_ROWS shouldn't overwrite
  position 0's K. Unless dst.divert pre-copy semantics are
  different from what I expect.

Reading R1 has the right shape — the bug is in i=3 specifically —
but doesn't fully explain why token 1 is deterministic across H1
on/off. (If position-0 K is zero in both Stages 4.4 and 4.5, token 1
should be identical and SUBSEQUENT tokens should differ in some
specific pattern related to H1 mirror state, which they do.)

Actually, R1 might be self-consistent. Tokens 0+1 are deterministic
across H1 on/off because: (a) prefill's last-token logits aren't
affected by KV reads (token 0 = first prediction uses no KV); (b)
the first decode step's KV reads are deterministically zero for
position-0 K-cache layer 0 in BOTH Stage 4.4 and 4.5 (so token 1
matches). Tokens 2+ start to differ because each step's K cache
WRITE for the new token gets H1's writeback differently; token 2's
K cache at position 6 is different across H1 on/off, leading to
divergence.

But that requires position-0 K to be zero in BOTH. The K-cache
position-0 in Stage 4.4 was certainly zero (host_mirror
initialization, no GPU→host writeback). In Stage 4.5 with H1, if
i=3 silently fails, it's also zero. So R1 is consistent.

## Stage 4.7 brief — capture dst IMMEDIATELY post-dispatch

The end-of-decode readback is 5 decode steps + ~110 SET_ROWS
dispatches removed from i=3. Many things could overwrite cells 0..7
in that window. To localize whether i=3 ITSELF wrote to dst correctly,
we need to read dst RIGHT AFTER i=3's dispatch — before any later
op runs.

Stage 4.7 D2-tight: schedule the post-dispatch readback to fire
SYNCHRONOUSLY at the end of `mod.jsepRunOp` for the wrapped
SET_ROWS calls. Since `mod.jsepRunOp` is await-able under JSPI,
we can `await` an `dataManager.readAsync(...)` BEFORE returning.
That gives us dst-immediately-after-dispatch.

Alternative: insert a per-SET_ROWS GPU-fence or queue.onSubmittedWorkDone
before the next op. Heavier; do D2-tight first.

Implementation outline:
```ts
mod.jsepRunOp = async (...) => {
    ...
    const status = origRunOp(...);
    if (op === GGML_OP_SET_ROWS_VAL && setRowsDiag.length < SET_ROWS_DIAG_COUNT) {
        // SYNCHRONOUS readback (await before return)
        const dstNow = await readBufferSlice(dstH, dstO, 16);
        diag.dstImmediateFirst8U16 = dstNow;
    }
    return status;
};
```

If `dstImmediateFirst8U16` for i=3 is non-zero (matches expected
F16 values from src0), then Reading R2 is correct: a later op
overwrites with zeros. Localize the overwriter via a per-cell
sentinel pattern (write a known marker at i=3's dst pre-dispatch,
check after every later op which one zeroes it).

If `dstImmediateFirst8U16` for i=3 is zero, then Reading R1 is
correct: i=3's dispatch silently failed. Investigate the
dispatcher's behaviour for the FIRST SET_ROWS divert (corner case:
the encoder batcher's first flush, the temp dst's first allocation
+ destroy, the bind group's first creation).

## Code references

- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:1448-1497` — D2-lite SetRowsDiagEntry + capture
- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:1599-1700` (approx) — deferred readback scheduling
- `~/Repos/webllm/smoke-test/p2-v2-spike.src.ts:1939-1948` — log emission for SETROWS_DIAG_FIRST5
- `~/Repos/webllm/src/inference/jsep/ops/set-rows.ts:418-494` — divert path under investigation
- `~/Repos/webllm/src/inference/jsep/index.ts:173-216` — jsepWrite/jsepRead flush semantics

## Patch stack

Unchanged at 7. D2-lite is spike instrumentation only.
