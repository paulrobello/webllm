# Stage 4.3 — production-shape kernel selftests + full-graph runOp capture

**Date:** 2026-05-06
**Status:** **CLOSED — Bug A localized to ggml-jsep CPU-fallback path; Outcome A "Paris" decode NOT achieved.**
**Patch stack:** 6 (unchanged; no llama.cpp patches added in Stage 4.3).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.3b-fullgraph`
**Per-token decode:** ~23.9 ms (within noise of Stage-4.2 baseline 24.3 ms).

## Goal

Per the Stage 4.3 brief in `TODO.md`:

- **Step 1 (Stage 4.3a):** drive the existing JSEP kernels at *production
  shapes* in isolation to localize whether Bug A (post-prefill canonical
  NaN at every JSEP-supported op output offset) is a kernel-correctness
  issue.
- **Step 2 (Stage 4.3b):** capture every dispatch in the full prefill
  graph (1602 ops, not just the first 30) plus a deferred per-runOp dst
  readback so we can identify the first NaN producer.

## Stage 4.3a — production-shape kernel selftests

Added four new selftests in `smoke-test/p2-v2-spike.src.ts`, all running
**before** `_webgpu_init()` so they exercise the kernels with a fresh
runtime (no model loaded yet, no encoderBatcher state):

| Selftest                       | Shape                  | Mode      | Result |
|--------------------------------|------------------------|-----------|--------|
| `RMSNORM_MULTIROW_NODIVERT`    | rows=6, cols=2048      | no-divert | **PASS** — maxAbsDelta = 2.0e-6, no NaN/Inf, perRowMaxDelta consistent ~1e-6 across all 6 rows |
| `RMSNORM_MULTIROW_DIVERT`      | rows=6, cols=2048      | divert    | **PASS** — identical metrics to no-divert (maxAbsDelta = 2.0e-6) |
| `MATMUL_PROD_NODIVERT`         | M=64, K=2048, N=6, Q4_K | no-divert | **PASS** — maxAbsDelta = 3.5e-4, no NaN, no zeros, results match CPU reference |
| `MATMUL_PROD_DIVERT`           | M=64, K=2048, N=6, Q4_K | divert    | **PASS** — identical metrics to no-divert |

The Q4_K production-shape selftest specifically exercises the K=2048
code path (8 super-blocks per row), which the existing K=256
`Q4K_SELFTEST` did not cover. The divert variant places `src1` and
`dst` in the same shared GPU buffer at distinct aligned offsets so
`dispatchMatmul`'s `dst.bufHandle === src1.bufHandle` check fires the
divert path (matches the production runOp i=1 layout where
`dst=buf19@4194304` and `src1=buf19@0`).

The multi-row RMS_NORM selftest specifically tests `gid.x` row indexing
across 6 rows with distinct per-row patterns
(`x[r,c] = (c%17 - 8)*0.1 + r*0.01`). A row-stride or per-thread
row-sum bug would surface as cross-row contamination at rows=6 even
though the existing rows=1 selftest passes.

**Outcome (Stage 4.3a):** Both kernels are correct at production
shapes in both non-divert and divert modes. **Bug A is upstream of the
kernel inputs**, not in MUL_MAT or RMS_NORM correctness, and not in
the divert path lifecycle (tempDst destroy after submit, encoder
batcher flush ordering, etc.).

## Stage 4.3b — full-graph runOp capture + per-op readback

Bumped MAX_LOG (writes/reads) to 3000 and `RUN_MAX` to 1700 so the
full prefill graph (1602 dispatches, 1206 writes, 1266 reads) is
captured. Added a unified `evtSeq` counter so `jsepWrite` /
`jsepRead` / `jsepRunOp` interleave in the order JSEP actually called
them — Stage 4.2's per-stream indices made it impossible to tell
whether `jsepWrite[i=1]` happened before or after `jsepRunOp[i=1]`.

Also added a deferred per-runOp dst readback for the first 30
dispatches. The probe schedules a microtask that runs a staging
copy + `mapAsync` from `dst[dstO..+32)`. Probes settle via
`Promise.allSettled(dstProbePromises)` before the summary is emitted.

**Probe artifact note:** the deferred reads run AFTER the wasm sync
chain returns, by which time many *later* runOps + writes have also
fired. So `FIRST_NAN_DST_PROBE = runOp i=0` is *consistent* with the
true first-NaN-producer being any op between i=0 and i≈30 — the
microtask deferral is too coarse to isolate per-runOp timing. The
load-bearing diagnosis comes from the unified `evtSeq` interleave,
not from the dst probes.

### Empirical event timeline (seq 0–8)

```text
seq 0  W i=0  buf19@0  size=49152  first8=[-0.0013, 0.0019, -0.0019, 0.0038, ...]   ← valid embedding
seq 1  O i=0  RMS_NORM dst=buf19@0 src=buf19@0 (in-place divert)
seq 2  R i=0  buf19@0  size=49152  first8=[-0.336, 0.492, -0.502, 0.989, ...]      ← valid normed output
seq 3  W i=1  buf19@0  size=49152  first8=[-5e-5, 142.08, -4.48, -7.4e18, ...]     ← GARBAGE
seq 4  O i=1  MUL_MAT dst=buf19@4194304 src0=buf14@0 (Q4_K) src1=buf19@0 (divert)
seq 5  R i=1  buf19@4194304  size=49152  first8=[NaN×8]                            ← canonical NaN bytes
seq 6  O i=2  MUL_MAT dst=buf19@4194304 src0=buf14@2359296 (Q4_K) src1=buf19@0
seq 7  R i=2  buf19@4194304  size=6144   first8=[NaN×8]
seq 8  W i=2  buf19@0  size=6144  first8=[-5e-5, 142.08, ...]                      ← same garbage repeat
```

`R i=0` is the input to a CPU-fallback op; `W i=1` is its output back to
the same offset. Between them, **the only thing that runs is one CPU
op**. In a llama transformer, the op directly after `RMS_NORM` is
`MUL` (per-channel gain): `out[r,c] = normed[r,c] * attn_norm.weight[c]`
where `attn_norm.weight` is F32 shape [2048].

The `W i=1` first8 implies the per-channel weights look like
`[1.5e-4, 290, 8.96, -7.5e18, -7.5e-39, -1.65e-34, -1.1e-8, -3.8e18, ...]`
— wildly out of range for an RMSNorm gain (which should be near 1.0).
These bytes do not look like F32 attn_norm.weight values; they look
like uninitialized wasm-heap memory interpreted as f32.

### Distribution of jsepWrite / jsepRead by handle

```text
writesByHandle = { 19: 1206 }
readsByHandle  = { 19: 1266 }
```

**Every single CPU↔JSEP transfer in the prefill graph targets handle
19 (the 64 MiB activations / scratch buffer). ZERO transfers target
the weight buffers (handles 14-17, 4 × 128 MiB of model weights).**

This is the smoking gun. The CPU-fallback ops (MUL, ADD, ROPE,
SOFTMAX, SCALE, SILU, GET_ROWS, CPY) need access to weight tensors
like `blk.0.attn_norm.weight`. If they were copying weights from
JSEP to CPU heap before computing, we would see jsepRead calls on
handles 14-17. We see none. Therefore the CPU side is reading
weight data from somewhere else.

## Root-cause diagnosis (Bug A)

The bug is in the **JSEP backend's host-pointer pretense**.
`ggml_backend_jsep_buffer_get_base` (`ggml-jsep.cpp:219-222`) returns
the sentinel `GGML_JSEP_PTR_BASE = 0x2000`, and tensor handles are
encoded as offsets relative to this sentinel. Each jsep-resident
tensor's `tensor->data` is therefore `0x2000 + per-tensor-handle-encoded-offset`,
which the rest of ggml-backend treats as a *valid host pointer* (since
`get_base` looks like a host buft).

The standard ggml backend scheduler, when it splits a graph and routes
a CPU op (e.g. MUL) with JSEP-resident sources, consults
`tensor->data` directly for the inputs. Because `data` looks like a
host pointer, the scheduler **does not insert a get_tensor copy** —
it expects the CPU op to dereference `data` and read the bytes.
Dereferencing `0x2000 + offset` reads uninitialized wasm heap (or
unrelated memory previously written there), and the CPU op's output is
garbage.

Activation tensors *do* go through `jsepRead` (1266 reads on handle 19)
because the **graph_compute boundary between JSEP runOps and CPU
fallback ops** must transfer the data. The mechanism that drives those
reads is unknown from this trace alone (it could be the standard
backend scheduler's split logic for tensors whose backend is jsep but
whose consumer is on CPU; it could be a webllm-specific code path).
But for **leaf weight tensors** that were uploaded to GPU during model
load (`alloc=6, write=134` at counters@load), no such copy happens —
the CPU op consumes `tensor->data = sentinel + offset` directly and
gets garbage.

This explains every Stage 3.5 / 4.1 / 4.2 observation:

- **Stage 3.5** found that activations bound for MUL_MAT input were
  corrupted before MUL_MAT ran. Stage 3.5 attributed this to an
  "uninitialized GPU memory pattern". Wrong locale: the corruption
  is in CPU-computed *output* fed back to the GPU buffer, not raw GPU
  memory. The bytes look uninitialized because the *weights* the CPU
  multiplied by are from uninitialized RAM.
- **Stage 4.1** ruled out SET_ROWS aliasing as the load-bearing cause
  by adding RMW divert. Correctly: SET_ROWS aliasing was real but not
  the Outcome C cause.
- **Stage 4.2** proved validation errors aren't the cause
  (GPU_ERR_COUNT=0), aliasing isn't the cause (every JSEP op diverts
  cleanly), and the GPU buffer starts zero-initialized.
- **Stage 4.3a** rules out kernel correctness — JSEP kernels work at
  production shapes in both modes.

The garbage-feeding cascade then taints every downstream op:

1. RMS_NORM dispatches correctly → buf19@0 holds valid normed output.
2. CPU MUL reads valid normed input, multiplies by garbage weights →
   writes garbage f32 (some denormal, some 1e+18, some near zero) to
   buf19@0.
3. JSEP MUL_MAT reads `buf19@0` (garbage) × `buf14@0` (Q4_K weights, valid),
   accumulates K=2048 multiplies → most accumulators overflow to
   ±Infinity; the kernel writes Inf or `Inf - Inf = NaN` to dst.
4. NaN propagates through every subsequent op (multiplied, added, normed,
   set_rows'd) — POSTPREFILL probes confirm NaN at every JSEP-supported
   op output offset.
5. lm_head (Q6_K, NOT supported by JSEP — `supports_op` only allows
   {F32, F16, Q4_0, Q4_K} for src0) routes to CPU. The CPU lm_head
   reads NaN-poisoned residual stream input AND garbage weights. Result
   = anything from NaN to all-zero. Stage 4.2 saw all-zero
   (`LOGIT_STATS_STEP0 = {first8: [0,...,0], finiteCount: 32000}`),
   suggesting the CPU lm_head's MUL_MAT computed `NaN * garbage_q6k`,
   which probably saturated to 0 in some integer reduction step.

So **Bug B (lm_head all-zero) is a downstream symptom of Bug A**, not
an independent bug. Once Bug A is fixed and the residual stream is
finite, lm_head should produce real logits.

## Why this hasn't fired before

Stages 1, 1.5, 2 measured intermediate-stage instrumentation (driver +
self-tests in isolation, not the full prefill graph) and never crossed
the JSEP/CPU boundary on weight tensors. The first time this manifested
was in **Stage 3** when the full TinyLlama graph ran end-to-end
through ggml-jsep — the same Stage 3 that produced "Outcome C all-zero
logits". Every subsequent stage (3.5, 4.1, 4.2) saw the same symptom
because the underlying CPU↔JSEP boundary bug was unchanged.

The reason it took until Stage 4.3 to localize: prior stages
focused on the GPU side (kernel correctness, divert path, aliasing,
buffer initialization, validation errors). The CPU side received less
scrutiny because `jsepWrite` / `jsepRead` SEEMED to be working
(they were — for activations). The interleaved `evtSeq` view was the
missing tool: it surfaces the CPU-side compute step as a read+write
pair on the same offset, with valid-input-becomes-garbage-output as
the diff signature.

## Files modified in Stage 4.3

- `smoke-test/p2-v2-spike.src.ts`:
  - Added `runMatmulProductionSelfTest(runtime, "no-divert" | "divert")`
    + helper `buildSyntheticQ4KMatrix(M, K)` (production-shape Q4_K
    matmul self-test; permanent regression check).
  - Added `runRmsNormMultiRowSelfTest(mod, runtime, "no-divert" | "divert")`
    (multi-row RMS_NORM self-test; permanent regression check).
  - Refactored handle-11 hardcoding to dynamic `actHandle` lookup
    (selftests advance the data manager handle counter, so the
    activations buffer's handle drifts; lookup uses
    `bucket ≤ 8 && size ≥ 32 MiB` filter).
  - Added unified `evtSeq` counter to `jsepWrite` / `jsepRead` /
    `jsepRunOp` wrappers + bumped `MAX_WRITE/MAX_READ/RUN_MAX` to
    3000/3000/1700.
  - Added per-runOp deferred dst readback (`dstProbes`,
    `dstProbePromises`) for the first `DST_PROBE_COUNT=30` ops.
  - Added `Promise.allSettled(dstProbePromises)` await before summary
    emission.
  - Replaced full-log emission with summary + first-30 slice (full
    logs remain on `window.__jsepWriteLog` / `__jsepReadLog` /
    `__jsepRunLog` / `__dstProbes` for targeted `js exec` fetches).

No `src/` changes. No llama.cpp patches.

## Eliminated hypotheses

| Hypothesis (entering Stage 4.3) | Verdict | Evidence |
|---|---|---|
| Q4_K MUL_MAT kernel buggy at large K | ❌ rejected | MATMUL_PROD_NODIVERT passes at K=2048 |
| RMS_NORM kernel buggy at multi-row | ❌ rejected | RMSNORM_MULTIROW_NODIVERT passes at rows=6 |
| Divert path corrupts dst (tempDst destroy / cpyBufToBuf) | ❌ rejected | _DIVERT selftests match _NODIVERT exactly |
| Buffer initialization (NaN-init) | ❌ rejected (Stage 4.2) | PREPREFILL_BUF11 = all zeros |
| WebGPU validation errors | ❌ rejected (Stage 4.2) | GPU_ERR_COUNT = 0 |
| SET_ROWS divert misses an alias | ❌ rejected (Stage 4.1) | All 264 SET_ROWS divert correctly |
| **CPU ops dereference sentinel `0x2000 + offset` for jsep-resident weights → read garbage** | ✅ **load-bearing** | 1266 jsepReads + 1206 jsepWrites all on handle 19 (activations); zero on weight handles 14-17. CPU MUL output (`W i=1` first8) is consistent with `valid_input × uninitialized_RAM`. |

## What needs to happen for Outcome A

The fix is in `ggml-jsep.cpp`. Two architectural options:

**Option F1: dual-resident weights.** During model load, also keep a
host copy of every weight tensor. `tensor->data` could keep returning
the sentinel for GPU dispatch routing, but a parallel host buffer
holds the real bytes. CPU ops dereference the host buffer instead of
the sentinel.

- **Cost:** +638 MiB host RAM for TinyLlama (the full GGUF). On the
  16 GB / 32 GB hardware baseline this is fine; for 8B models
  (~5 GB) it's still under the 10–11 GB ceiling.
- **Pro:** smallest blast radius — touches only the buffer interface
  + one allocator hook. CPU ops keep their current code path.
- **Con:** doubles memory pressure for weights.

**Option F2: lazy get_tensor on CPU-op boundaries.** Hook
`ggml_backend_jsep_buffer_get_tensor` to be called by the scheduler
whenever a CPU op needs a JSEP source. This requires either:

- the standard ggml-backend scheduler to insert these copies (which it
  appears not to do because `get_base` lies about residency), OR
- a webllm-specific scheduler shim that walks the graph pre-execute,
  collects all CPU op sources that live on jsep, and pre-stages them
  via `get_tensor` into a CPU buffer (then patches `tensor->data` for
  those source pointers).

- **Pro:** zero extra host RAM at idle (only the active op's sources
  are temporarily resident).
- **Con:** larger patch, requires understanding the scheduler's
  per-op tensor-data path. Risk of subtle breakage at split
  boundaries.

**Recommended:** F1 (dual-resident) for the first fix. It's the
mechanical option, gets us to Outcome A fastest, and the host-RAM
cost is acceptable under the 8B / 32 GB doctrine. F2 is a follow-on
optimization once Outcome A flips and we want to claw back the
duplicate memory.

## Per-token decode + smoke regression deltas

| Stage | Per-token ms | Decode ratio vs Stage 3 baseline 23.0 |
|---|---|---|
| Stage 3 (kernel correct, no Q4_K) | 23.0 | 1.00× |
| Stage 3.5 (Q4_K landed) | 24.30 | 1.06× (divert overhead) |
| Stage 4.1 (SET_ROWS RMW divert) | 23.74 | 1.03× |
| Stage 4.2 (diagnostic-only) | 24.34 | 1.06× |
| Stage 4.3 (4.3a + 4.3b instrumentation) | 23.92 | 1.04× |

All deltas are within noise. Stage 4.3 instrumentation (selftests +
deferred dst probes) adds ~50 ms each at startup but no per-token
impact during steady-state decode.

## What lands on `main`

- `smoke-test/p2-v2-spike.src.ts` (selftests + diagnostic instrumentation;
  permanent until Outcome A flips and we ship — the kernel selftests
  in particular are permanent regression checks even after the spike
  retires).

## Closure

Stage 4.3 closes with **Bug A diagnosis localized; Outcome A NOT
achieved**. The next stage is **Stage 4.4 — implement F1 dual-resident
weight buffers in ggml-jsep**, which should flip Outcome A directly
(Bug B falls out as a downstream symptom of Bug A and resolves once
the residual stream stays finite).
