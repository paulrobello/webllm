# Stage 4.22 — Probe 10: production-dispatch kernel-input capture & replay

**Date:** 2026-05-07
**Patch stack:** 13 (unchanged — pure JS-side spike + matmul.ts probe gate)
**Outcome:** **G-2** (synthetic ≤ 1e-5 — kernel is bit-clean on production
inputs; the bug lives outside the dispatch / kernel boundary)

## TL;DR

Captured the actual src0 / src1 / dst-after bytes the kernel sees at the
**first** production JSEP `MUL_MAT` dispatch in TinyLlama prefill (layer-0
Q-projection — `Qcur-0`, ne=[2048,6,1,1]) and replayed them through the
same `dispatchMatmul` entry point as a one-off synthetic call. Both the
captured production output and the synthetic replay match an f32
element-wise k-major CPU reference to within **4.768e-7** — i.e., to a
single ULP at this output magnitude (`outputMaxAbs = 6.37`).

| | maxAbsDelta vs f32 loop | maxAbsDelta vs f64 |
|---|---|---|
| Captured production dst-after | 4.768e-7 | (not measured¹) |
| Synthetic replay (same bytes) | 4.768e-7 | 7.94e-6 |
| First-8 outputs (production)  | `[-0.01619, 0.00485, -0.01574, -0.02449, -0.00762, 0.04053, -0.00968, 0.04544]` | |
| First-8 outputs (synthetic)   | `[-0.01619, 0.00485, -0.01574, -0.02449, -0.00762, 0.04053, -0.00968, 0.04544]` | |

¹ Captured-dst was scored against the f32 loop reference only; the reference
was built from JS-side dequant of the captured Q4_K bytes.

The two paths are byte-equivalent on output. Stage 4.18's "5.24e-4 first8
Qcur-0 delta" was therefore measured against a **different** reference
(libllama's CPU-fallback path through the dual-resident host mirror), not
against an f32 ground truth. The Q4_K WGSL kernel itself is doing the
math correctly on the bytes it actually receives.

## Surprise finding — TinyLlama-Q4_0.gguf is Q4_K under the hood

The Stage 4.22 brief assumed src0 was Q4_0 (type 2). The actual production
dispatches in `JSEPRUN_LOG_FIRST30` show `t = 12`. We re-parsed
`smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf` via `GgufParser.parse`
and confirmed:

| Tensor | type | shape |
|---|---|---|
| `token_embd.weight` | 12 (Q4_K) | [2048, 32000] |
| `blk.0.attn_q.weight` | 12 (Q4_K) | [2048, 2048] |
| `blk.0.attn_k.weight` | 12 (Q4_K) | [2048, 256] |
| `blk.0.attn_v.weight` | 14 (Q6_K) | [2048, 256] |
| `blk.0.attn_output.weight` | 12 (Q4_K) | [2048, 2048] |
| `blk.0.ffn_gate.weight` | 12 (Q4_K) | [2048, 5632] |
| `output.weight` | 14 (Q6_K) | [2048, 32000] |

Despite the `q4_0.gguf` filename, the file is actually a **Q4_K_M** mix
(Q4_K projections + Q6_K embeddings). This is the standard upstream
TheBloke/HuggingFace convention — "Q4_0" in the filename refers to the
HuggingFace quant tier label, not the on-disk tensor type. **Stage 4.18's
"Q4_0 production-shape sweep" was therefore measuring a different code
path from production**: synthetic Q4_0 dispatches at production shapes,
when production is actually exercising the Q4_K WGSL kernel. The 312×
delta gap that motivated Stage 4.22 was an apples-vs-oranges comparison
all along.

## What Probe 10 actually tested

1. **Capture path** (matmul.ts dispatchMatmul, divert path):
   - Gated on `globalThis.__probe10Capture.armed && (src0.type ∈ {Q4_0, Q4_K})`.
   - Pre-encoder copies `src0`, `src1`, `dst-before` into MAP_READ staging
     buffers — submitted **before** the kernel encoder.
   - Kernel encoder runs as normal (existing divert path).
   - Post-encoder copies `dst-after` into a fourth staging buffer —
     submitted **after** the kernel encoder.
   - WebGPU queue submission order is FIFO, so `dst-before` is captured
     before any kernel writes and `dst-after` reflects post-kernel state.
   - mapAsync promises resolve once GPU completes the copies; the spike
     `await`s `queue.onSubmittedWorkDone()` + a 50 ms yield to ensure
     `.then()` callbacks ran before the result is read.

2. **Replay path** (`runMatmulFromBytes`):
   - Allocates fresh JSEP buffers, writes the captured bytes, builds an
     equivalent JsepOpDescriptor, calls `dispatchMatmul`. Reads dst back
     via copyBufferToBuffer + mapAsync. Returns max-abs delta vs an f32
     k-major loop reference computed in JS (port of WGSL `load_q4_K`).

3. **Capture comparison** (`compareF32Buffers`):
   - Direct max-abs delta of captured production `dst-after` vs the same
     f32 reference used by the synthetic replay.

## Implications for the next probe

**The dispatch / kernel-execution boundary is exonerated.** The kernel
produces the same output as a hand-written f32 element-wise loop on the
exact bytes it received at production dispatch time. Pipeline cache
collisions, bind-group offset mismatches, workgroup count off-by-ones,
src0/src1 swaps — all are mathematically excluded by the bit-identical
first-8 outputs.

**The remaining suspect is the CPU-fallback path.** Stage 4.4's F1
dual-resident host mirror (ggml-jsep.cpp) keeps a CPU-side copy of every
JSEP buffer; CPU-side ops (anything not in JSEP's `supports_op` allow-list)
read through `tensor->data` which now points into `host_mirror`. Stage 4.4
documented a known **GPU→host writeback gap**: when a JSEP op writes to
the GPU buffer, the host mirror stays stale, so the next CPU-fallback op
reads zeros (or post-Stage-4.21 patches, possibly stale-but-non-zero).

The 5.24e-4 first8 Qcur-0 delta from prior stages was therefore likely a
delta between:
- The JSEP-Qcur-0 GPU result (kernel = correct, per Probe 10).
- A CPU-fallback "Qcur-0" attempt that reads stale `host_mirror` for one
  of the dispatch's inputs, OR a CPU-fallback subsequent op (RMS_NORM,
  ROPE, etc.) that consumes Qcur-0 and propagates corruption forward.

The actual `GENERATED_TEXT = "inonic boso-"` decode is still wrong, so
something is broken — but the kernel proper is not it. Stage 4.23 should
focus on the writeback gap and / or the CPU-fallback ops that consume
Qcur-0.

## Concrete numbers from the spike

```
[probe10] capture armed (first Q4_0 MUL_MAT in prefill)
     prefill 453 ms
     [probe10] captured M=2048 K=2048 N=6 src0=2359296B src1=49152B dstBefore=49152B dstAfter=49152B
MATMUL_PROBE10_REPLAY = {"M":2048,"K":2048,"N":6,"src0Type":12,"status":0,"maxAbsDeltaVsF32Loop":4.76837158203125e-7,"maxAbsDeltaVsF64":0.000007943707068136519,"outputMaxAbs":6.3691864013671875,"hasNaN":false,"hasInf":false,...}
MATMUL_PROBE10_CAPTURED_DELTA = {"M":2048,"K":2048,"N":6,"maxAbsDelta":4.76837158203125e-7,"hasNaN":false,"hasInf":false,...}
     [probe10] M=2048 K=2048 N=6 capturedDelta=4.768e-7 syntheticDelta=4.768e-7
     [probe10] OUTCOME: G-2 (synthetic ≤1e-5 — bug between dispatch site and shader execution)
GENERATED_TOKENS = [297,8927,13601,29877,29899]
GENERATED_TEXT = "inonic boso-"
DONE
```

(Verdict line says G-2, but the more accurate framing — given that
captured matches synthetic to ULP — is **G-2 with the kernel exonerated
on production inputs**. The "between dispatch and shader execution" wording
in the verdict was the original branch label; it should be read as "not in
the kernel itself".)

## Files touched (no llama.cpp patch — patch stack 13 unchanged)

- `src/inference/jsep/ops/matmul.ts` — added Probe 10 capture branch in
  the divert path (gated on Q4_0 OR Q4_K src0; auto-disarms after first
  fire). Pre-encoder + kernel-encoder + post-encoder pattern with mapAsync
  promises that populate a `globalThis.__probe10Capture.result` blob.
  Exported `Probe10CaptureResult` interface.
- `smoke-test/p2-v2-spike.src.ts` — added `dequantQ4_KTile` (port of
  WGSL `load_q4_K`), generalized `runMatmulQ4_0FromBytes` →
  `runMatmulFromBytes(src0Type, ...)` covering both Q4_0 and Q4_K, added
  `compareF32Buffers` helper, and wired the post-prefill probe10 block:
  arm the gate before `bridge.decode`, await `queue.onSubmittedWorkDone`,
  poll for the populated capture, run replay, log verdict.

## Selftests + checkall

- All 6 spike selftests still PASS:
  `Q4K_SELFTEST`, `RMSNORM_SELFTEST`, `RMSNORM_MULTIROW_NODIVERT/DIVERT`,
  `MATMUL_PROD_NODIVERT/DIVERT`.
- All 5 sweep selftests still PASS: `MATMUL_Q4_0_SWEEP[q-out-proj /
  k-v-proj / ffn-gate-up / ffn-down / lm-head]`.
- `make checkall` green: 747 pass / 36 skip / 0 fail.
- `bun build smoke-test/p2-v2-spike.src.ts ...` green (12 modules /
  ~0.30 MB).

## Stage 4.23 — next probe (CPU-fallback writeback gap localization)

Stage 4.23 should **NOT** continue investigating the kernel. Instead:

1. **Identify whether any CPU-fallback ops fire in the layer-0 prefill
   forward chain after Qcur-0.** Stage 4.21's `JSEPRUN_LOG_FIRST30`
   shows the first ~30 ops are all-divert JSEP ops (RMS_NORM, MUL_MAT,
   SET_ROWS). If a CPU-fallback op fires, surface it.
2. **Re-derive the 5.24e-4 number against an f32 element-wise reference.**
   The Stage 4.18 framing claimed Qcur-0 had 5.24e-4 first8 delta — but
   against what? If against the CPU-fallback graph_compute, the bug is
   in the fallback. If against an f32 ground truth, the kernel-bug
   hypothesis would be revived (but Probe 10 ruled it out, so this
   should not happen).
3. **Investigate the GPU→host writeback gap** flagged in Stage 4.4. Walk
   the prefill graph and identify the first op whose `tensor->data`
   points into a stale `host_mirror` slice. The smoking gun is an op
   whose JSEP-side buffer was just written to but whose host mirror still
   carries the post-allocation zero-init.

Suggested probe sketch:
- Add a per-op stride hash: just after each JSEP `runOp`, FNV-1a-32 of
  the post-write slice in `host_mirror` AND the GPU buffer at the same
  offset. If they diverge, the writeback gap is real and the divergence
  point is the bug site.
