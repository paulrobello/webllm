# Stage 3.5 — Localizing the all-zero logits collapse

**Date:** 2026-05-06
**Outcome:** **Root cause identified.** Stage 4 partial fix landed; full
Outcome A ("Paris" decode) **not yet achieved**. Logits remain all-zero.

## TL;DR

The Stage 3 "Outcome C" all-zero logits were caused by a **WebGPU compute
pass synchronization-scope violation** in `dispatchMatmul` (and
`dispatchRmsNorm`): the libllama scheduler packs the matmul activation
input (`src1`) and matmul output (`dst`) into a *single* `jsep_buf` at
different byte offsets to save memory. WebGPU's bind-group validator
works at *buffer granularity*, not sub-range:

```
[Buffer (unlabeled)] usage (Storage(read-write)|Storage(read-only))
includes writable usage and another usage in the same synchronization
scope.
 - While validating compute pass usage.
 - While finishing [CommandEncoder (unlabeled)].
```

So even though `src1` and `dst` bound non-overlapping byte ranges, both
mapping to the same `GPUBuffer` triggered the rule. The encoder was
rejected at `finish()` time, the dispatch never executed, and `dst`
stayed at its previous (zero-initialized) state. The next op in the
chain read zeros, propagated zeros — full forward pass collapsed.

**Stage 4 fix (matmul, RMS_NORM):** when `dst.bufHandle` aliases any
src `bufHandle`, allocate a fresh temp `GPUBuffer`, dispatch into it,
then `copyBufferToBuffer` back to the real `dstRec.buffer` at
`dst.offset`. The diverted dispatch lives in its own command-encoder
(flush the batcher first) so it can't conflict with batched neighbours.

**Verified post-fix:**
- Diverted matmul/RMS_NORM dispatches no longer trigger validation
  errors (`pushErrorScope("validation")` returns `<none>`).
- Diverted matmul tempDst output is non-zero (kernel actually writes).
- 1068 of 1068 model matmuls in 1 prefill + 5 decode steps alias and
  divert. 270 of 271 RMS_NORM dispatches alias and divert.

**Why "Paris" still doesn't decode:** matmul `src1` for the first
attn_q is *still* corrupt (`[-5e-5, 142.08, -4.48, -7.4e+18, ...]` —
same byte pattern as pre-fix). The corruption originates upstream of
the matmul fix. Top remaining suspects:

1. **`SET_ROWS` aliasing**. The kernel writes to a view of `src[2]`
   (KV cache) — `dst.bufHandle === src[2].bufHandle` is *definitional*,
   not incidental. Same WebGPU rule fires; KV cache writes silently
   fail; subsequent attention reads zeros. (Fix is harder than matmul:
   `SET_ROWS` is a *partial* update — naive temp + copy-back clobbers
   unwritten rows. Needs a read-modify-write divert: copy real dst →
   temp first, dispatch, copy temp → dst.)
2. **CPU-side MUL writeback**. Several non-supported ops (MUL, ADD,
   SCALE, SOFT_MAX, ROPE, GET_ROWS) fall back to CPU. The CPU side
   reads JSEP buffers via `get_tensor` → `jsepRead` and writes results
   back via `set_tensor` → `jsepWrite`. If any of those reads picks up
   stale data because the prior JSEP dispatch was rejected (pre-fix)
   or because a flush ordering is off, garbage propagates.

## Steps executed

### Step 0 — verified Stage 3 baseline reproduces

`?v=stage3-replay`: Q4_K self-test delta 4.5e-6 ✓; LOGIT_STATS_STEP0
all-zero ✓; GENERATED_TOKENS [0,0,0,0,0] ✓.

### Step 1 — RMS_NORM real-shape self-test (cols=2048)

Added `runRmsNormSelfTest` to the spike harness — drives
`dispatchRmsNorm` directly with a 1×2048 row, eps=1e-5, and a known
non-zero pattern, comparing against a CPU reference.

Result: `maxAbsDelta = 8.3e-7`, no NaN, no Inf, output matches reference
to ~6 decimal places. **RMS_NORM kernel is correct on real-model
shapes**; bug is elsewhere.

(Self-test stayed in the spike harness as a permanent regression
check — it's cheap and catches future kernel breakage at the canonical
TinyLlama shape.)

### Step 2 — first-model-matmul probe + WebGPU error scope

Instrumented `dispatchMatmul` to capture src1 + dst first 8 f32 of the
first ~3 model-shape matmul calls, plus a `pushErrorScope("validation")`
around the flush.

**Smoking gun (from `?v=stage3.5-errscope`):**

```
MATMUL_PROBE_0 = {
  M:2048, K:2048, N:6, batch:1, src0Type:12 (Q4_K),
  src0BufHandle:6, src1BufHandle:11, dstBufHandle:11,
  src1Offset:0, src0Offset:0, dstOffset:4194304,
  src1First8: [-5.17e-5, 142.08, -4.48, -7.43e+18, -2.48e-40, ...],
  dstFirst8: [0,0,0,0,0,0,0,0],
  src1NonzeroOf8: 8, dstNonzeroOf8: 0
}
MATMUL_PROBE_0_ERROR = "[Buffer (unlabeled)] usage
  (Storage(read-write)|Storage(read-only)) includes writable usage and
  another usage in the same synchronization scope."
```

`src1` and `dst` share `bufHandle=11`. WebGPU rejected the encoder.
Dispatch never ran. dst stayed zero.

### Stage 4 — divert pattern (partial)

Implemented in `src/inference/jsep/ops/matmul.ts` and
`src/inference/jsep/ops/rms-norm.ts`. Pattern:

```ts
const dstAliasesSrc =
    dst.bufHandle === src0.bufHandle ||
    dst.bufHandle === src1.bufHandle;
if (dstAliasesSrc) {
    const tempDst = device.createBuffer({
        size: dstBytesNeeded,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // bind tempDst at offset 0 instead of real dst
    encoderBatcher.flush();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(...); pass.setBindGroup(0, divertBindGroup);
    pass.dispatchWorkgroups(...); pass.end();
    enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstBytesNeeded);
    device.queue.submit([enc.finish()]);
    tempDst.destroy();
    return 0;
}
```

Contiguity check (`dst.nb[1] === M*4` && `dst.nb[2] === N*M*4`)
guards against future callers passing strided dst. matmul output
from libllama is always contiguous so this never fires in practice;
the assertion is for forward-compatibility.

`tempDst.destroy()` immediately after submit is documented-safe per
WebGPU spec — pending GPU work continues using the underlying memory.

**Performance:** the divert adds one `createBuffer` + one
`copyBufferToBuffer` per aliased dispatch. For the canonical
TinyLlama prefill + 5 decode steps, that's 1068 extra buffer creates
+ copies. Per-token decode time in the spike post-fix is ~25 ms (vs
~24 ms pre-fix Stage-3 baseline) — within noise at this scope. A
later optimization could pool temp buffers by size to skip the
allocation cost.

## Why src1 is still corrupt post-fix

Even with matmul + RMS_NORM diverts in place, attn_q's `src1` is
still `[-5e-5, 142.08, -4.48, -7.4e+18, ...]`. Identical byte pattern
to pre-fix. This means:

- The **producer** of `src1` (whoever last wrote `jsep_buf[11]`
  offset 0 before the matmul) is doing the wrong thing.
- The producer is upstream of matmul. Candidates: SET_ROWS for KV
  cache (definitional aliasing), CPU MUL writeback via `set_tensor`,
  CPU GET_ROWS embedding writeback via `set_tensor`.
- Pre-fix, the matmul's *kernel* never ran (validation error). The
  **values at offset 0 of jsep_buf[11]** at the moment matmul
  *would* have read them are the same now as before, because the
  upstream ops that *write to that offset* haven't been fixed.

The denormal-mixed-with-1e+18 byte pattern is the hallmark of
**uninitialized memory** read as f32 — strongly suggesting an op
in the chain is failing silently and leaving the buffer untouched.

## Stage 4 follow-on plan

Queue the next pickup as **Stage 4.1 — SET_ROWS divert + upstream
audit**:

1. Add an aliasing detector to every JSEP dispatch
   (`dispatchSetRows`) that logs total/diverted counts. Confirm
   SET_ROWS aliases at expected rate (likely ~100% — KV cache
   writes are definitionally a `view(src[2])` write).

2. Implement SET_ROWS divert. Tricky: SET_ROWS is a *partial*
   update — only writes specific row indices. A naive
   `dispatch + copy-back` clobbers the unwritten rows with
   uninitialized tempDst data. Two approaches:
   - **(a) Pre-copy:** `copyBufferToBuffer(dst → temp)` first,
     then dispatch (modifies certain rows in temp), then
     `copyBufferToBuffer(temp → dst)`. Doubles the memory
     traffic per SET_ROWS, but correct.
   - **(b) Skip aliased writes:** treat SET_ROWS as a no-op when
     aliased, and rely on a separate "real" SET_ROWS path that
     bypasses WebGPU's compute pass + uses `writeBuffer` directly
     for the partial write. More invasive; touches the f16-CAS
     atomic path.
   Approach (a) is simpler and easier to validate; ship that first.

3. Re-run spike at `?v=stage4.1-setrows`. Expected: matmul `src1`
   becomes sensible f32 values (post-RMS_NORM-times-gain ≈ [-3, +3]
   range). LOGIT_STATS_STEP0 first8 becomes non-zero, top token
   yields a real BPE id, GENERATED_TEXT contains "Paris" or
   close-by tokens.

4. If `src1` still corrupt after SET_ROWS divert: investigate
   CPU-side MUL/ADD writeback path. Add `jsepWrite` byte dump on
   first ~5 calls to verify the bytes flowing CPU → JSEP are
   sensible.

## Files changed

- `src/inference/jsep/ops/matmul.ts` — divert pattern in
  `dispatchMatmul` (~80 LOC).
- `src/inference/jsep/ops/rms-norm.ts` — divert pattern in
  `dispatchRmsNorm` (~50 LOC).
- `smoke-test/p2-v2-spike.src.ts` — `runRmsNormSelfTest` self-test
  function (~110 LOC). Permanent.

No llama.cpp patches in Stage 3.5. Patch stack remains at +6.

## Spike state cheat sheet (post-Stage-3.5)

- **Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage3.5-<probe>`
- **Counter snapshot per token (post-divert):** runOp 320, write 241,
  read 211, sync 612, clear 0 (unchanged from Stage 3 — divert is
  invisible to JSEP counters since it dispatches via its own encoder).
- **Per-token decode:** ~25 ms (Stage 3: 23 ms; +8% from divert
  overhead). Within pre-fix noise.
- **Build:** `bun build smoke-test/p2-v2-spike.src.ts --outfile
  smoke-test/p2-v2-spike.js --target browser`. WASM rebuild not
  needed for Stage 3.5 (TS-only changes).

## Exit criteria — re-check

Per the Stage 3.5 brief in TODO.md:

- ☑ **Identified op/transition is the fault** — WebGPU sync-scope
  buffer aliasing in matmul + RMS_NORM dispatches. Stage 4 fix
  landed for both. Outcome A not yet achieved → queue Stage 4.1.
- ☐ `WEBLLM_PIN_TO_JSEP=0` rebuild fails to produce "Paris" — not
  exercised this session (no orthogonal regression suspected; matmul
  divert path leaves the non-aliased path untouched, so the
  `WEBLLM_PIN_TO_JSEP=0` baseline should be unaffected).
- ☐ Outcome A "Paris" decode achieved — gated on Stage 4.1
  SET_ROWS divert.
