# STAGE-4.34 RESULT — WGSL kqv MUL_MAT GQA-broadcast localization

**Date:** 2026-05-08
**Branch:** main · llama.cpp `webllm-browser-patches` tip `ebc7c3d82` (patch
stack 14, unchanged from Stage 4.29)
**Outcome:** **P-21-explicit-divide-needed** (revised from initial
"P-21-other" classifier emit) + **P-21b-bug-reproduces** (host-CPU
selftest).

## TL;DR

The WGSL kqv MUL_MAT kernel ignores GQA broadcast. Probe 21 captured
ggml passing `src0.nb=[2, 512, 128, 262144]` for the kq MUL_MAT — the
**permuted K-cache layout** (dim-fast=2, head-medium=128, pos-slow=512).
Probe 21b's host-CPU selftest at the exact kq shape (M=256, K=64, N=6,
src0.ne[2]=4, src1.ne[2]=32) demonstrated in isolation that the kernel
matches a GQA-correct reference *only* on Q-head 0 (Δ=9.5e-6, f32 noise)
and diverges on every other head (Δ=4.35–56.5 across heads 1–31).
Stage 4.35 ships the one-line WGSL fix (`src0_batch_idx = batch / r2`)
+ Probe 21b as a permanent regression guard.

## Probe 21 Shape A (read the kernel + log nb[2] for kq MUL_MAT)

### Static read

`src/inference/jsep/ops/matmul.ts:543-740` (`dispatchMatmul`):
- Line 578 — `batchCount = max(1, src1.ne[2]) * max(1, src1.ne[3])` is
  derived from **src1 only**. For GQA `src1.ne[2]=32` (n_q_head) but
  `src0.ne[2]=4` (n_kv_head). The kernel dispatches 32 z-grid invocations
  but src0 only carries 4 K-heads.
- Line 646 — `shapeData[7] = src0.nb[2]`. The shape uniform's
  `src0_batch_bytes` is set to whatever ggml provides on the descriptor.

`src/inference/jsep/ops/matmul.ts:267-487` (WGSL kernels f32, f16, q4_0,
q4_K):
- All four kernels compute the src0 byte offset as
  `byte_off = batch * shape.src0_batch_bytes + m * shape.src0_row_bytes +
  k * elsize`.
- **No GQA-aware divide.** No `batch / r2` step.

### Captured descriptor for kq MUL_MAT

```
STAGE434_PROBE21 = {
  "M":          256,        // = src0.ne[1] = n_kv_pos
  "K":          64,         // = src0.ne[0] = head_dim
  "N":          6,          // = src1.ne[1] = current Q positions
  "batchCount": 32,         // = src1.ne[2] = n_q_head
  "src0_type":  1,          // F16
  "src0_ne":    [64, 256, 4, 1],
  "src0_nb":    [2, 512, 128, 262144],
  "src1_ne":    [64, 6, 32, 1],
  "src1_nb":    [4, 8192, 256, 49152],
  "dst_ne":     [256, 6, 32, 1],
  "dst_nb":     [4, 1024, 6144, 196608]
}
```

### Layout interpretation

src0 is the **permuted K-cache** view (created by `ggml_permute(K, 0, 2,
1, 3)` upstream of the kqv matmul):

- Physical K-cache layout in memory: **dim-fast (`nb[0]=2`), head-medium
  (`128 = 64 dim × 2 byte`), pos-slow (`512 = 4 head × 128 byte`)**.
- `nb[3]=262144` = 4 head × 256 pos × 64 dim × 2 byte × 2 (the cache
  buffer holds room for n_ctx ≥ 512 even though only the first 256
  positions are active during this prefill).

After permute, the logical view ne=[64, 256, 4, 1] = [head_dim, n_kv_pos,
n_kv_head] reads through the same physical buffer with shuffled nb. The
key value: **`src0.nb[2]=128` is the head stride**, not the contiguous
batch stride (which would be `M * nb[1] = 256 × 512 = 131072`).

### Classifier emit vs. revised reading

The Stage 4.34 brief's classifier expected:
- `nb[2]=0` ⇒ stride-trick-zero
- `nb[2] = M * nb[1] = 131072` ⇒ explicit-divide-needed
- otherwise ⇒ other

The captured `nb[2]=128` failed both rules and emitted `P-21-other`.
This was the brief's heuristic missing the **permuted-layout case**:
ggml passes a head-fast stride that's *smaller* than `nb[1]`, expressing
"each batch slot is one K-head's slice within a single pos row". The
WGSL kernel's `batch * 128` walks this head-fast stride correctly for
batches 0–3 (mapping to K-heads 0–3 at pos 0) but for batch ≥ 4 walks
into the next pos row's K-head 0 — exactly the **stair-step pattern**
Stage 4.33 observed.

The correct reading is **P-21-explicit-divide-needed**: the kernel must
treat the batch dim as the Q-head index and divide by `r2 =
src1.ne[2] / src0.ne[2] = 8` before scaling by `src0.nb[2]`, so all
8 Q-heads in a GQA group read from the same K-head slice.

## Probe 21b Shape B (host-CPU selftest at kq shape)

`runKqGqaSelfTest` allocates F32 src0 / src1 / dst at the captured kq
shape and layout (permuted K-cache: nb=[4, 1024, 256, 262144] in F32),
fills with deterministic content (`sin(d*0.013) + 0.07*(h+1) +
0.003*(p%11) - 0.0001*p` for src0; `cos(k*0.011) + 0.05*b - 0.04*n` for
src1), calls `dispatchMatmul`, then compares dst against:

1. **GQA-correct reference**: `dst[m,n,b] = sum_k src0[head=b/r2, pos=m,
   dim=k] * src1[k,n,b]` — what ggml's MUL_MAT semantics requires.
2. **Byte-formula reference**: `dst[m,n,b] = sum_k
   src0_buf[(b*nb[2] + m*nb[1] + k*4)/4] * src1[k,n,b]` — what the
   current WGSL kernel computes (no GQA divide, OOB returns 0).

### Per-head Δ vs GQA reference

```
head  0: 9.54e-6   ← f32 noise (kernel matches GQA at b=0)
head  1: 4.35e+0   ← diverges
head  2: 9.15e+0
head  3: 1.44e+1
head  4: 4.07e+1   ← jump (pos-shift = 1 compounds with head shift)
head  5: 4.24e+1
head  6: 4.42e+1
head  7: 4.60e+1
head  8: 4.21e+1
...
head 23: 5.65e+1   ← largest delta
head 24: 4.84e+1
...
head 31: 5.62e+1
```

`maxAbsDeltaVsGqa = 5.65e+1` — six orders of magnitude above f32 noise.
**Only Q-head 0 matches** the GQA reference. This pins the bug at the
WGSL kernel level: at the exact kq shape, in synthetic isolation,
`dispatchMatmul` does *not* compute GQA-broadcast MUL_MAT.

### Verdict

```
[probe21b] OUTCOME: P-21b-bug-reproduces (head 0 matches GQA reference;
heads ≥1 diverge — kernel ignores GQA broadcast, needs `batch / r2`
divide)
```

### Why production stair-steps in zero-count but the selftest doesn't

The production K-cache buffer holds 256 pos slots but only positions
0–5 have been written during the 6-token prefill. The remaining 250
positions are zero-initialized by GPU buffer allocation. When the
buggy kernel reads `pos = m + b/4` with `m + b/4 >= 6`, it lands in
the zero region → output of `kq[m, *, b]` is exactly 0. With seq_len=6
and causal masking, this is the precise pattern Stage 4.33 observed
(36/36 active for heads 0–3, 30/36 for heads 4–7, ..., 0/36 for
heads 24–31).

The selftest fills *all* src0 positions with non-zero values, so OOB
reads still return non-zero data — every output element is non-zero
(1536/head). The bug surfaces instead in the **per-head delta
magnitude** which grows with pos-shift.

Both production and selftest signatures point at the same root cause.

## Surviving structural suspects after Stage 4.34

1. ❌ Output-projection (`attn_output.weight`) byte-integrity — closed
   Stage 4.28 / 4.30.
2. ❌ `ffn_norm.weight` gain-vector mis-load — closed Stage 4.30.
3. ❌ first8-window blindness on `kqv_out-0` — closed Stage 4.31.
4. ❌ Divergence at `kq-0` (Q × K^T) — confirmed Stage 4.32, localized
   Stage 4.33, **kernel-level root cause confirmed Stage 4.34**.
5. ✅ **WGSL kqv MUL_MAT GQA broadcast — kernel-level confirmed.**
   Stage 4.35 ships the fix.

## Stage 4.35 fix sketch (preview — full brief in TODO closure)

The kernel needs a per-shape `r2` parameter (or computed from src0.ne[2]
/ src1.ne[2] in dispatchMatmul) and an explicit divide:

```wgsl
// In load_*: replace
let row_byte_base: u32 = batch * shape.src0_batch_bytes + m * shape.src0_row_bytes;
// with
let src0_batch_idx: u32 = batch / shape.r2;
let row_byte_base: u32 = src0_batch_idx * shape.src0_batch_bytes + m * shape.src0_row_bytes;
```

The shape uniform layout grows by one u32 (`r2`). For non-broadcast
dispatches (`src0.ne[2] == src1.ne[2]`), `r2 = 1` and the divide is a
no-op — no precision impact on existing paths.

Probe 21b stays in tree as a permanent regression guard. After the fix
lands, re-run the selftest and verify `head0Match=true` AND
`maxAbsDeltaVsGqa < 1e-3` (uniformly correct). Then re-run the full
spike and verify `__spikeResult.generatedIds[0] === 3681` (matches
the non-JSEP reference output for "Paris" prompt).

## Files touched

- `src/inference/jsep/ops/matmul.ts` — Probe 21 one-shot capture (gated
  by `__stage434Probe21Arm`), stashes ne/nb in `__stage434Probe21`.
- `smoke-test/p2-v2-spike.src.ts` — Probe 21 arming, Probe 21b
  `runKqGqaSelfTest` helper + post-selftest verdict synthesis. Probe
  21b lives in tree as a regression guard for Stage 4.35's fix.

## Artifacts

- This file: `STAGE-4.34-RESULT.md`
- Raw spike output: `STAGE-4.34-spike-output.txt`
- Stage 4.33 closure (the input to this stage):
  `STAGE-4.33-RESULT.md`
