# Stage 4.23 — Probe 11 reframe: 5.24e-4 provenance via ref-probe vs spike

**Date:** 2026-05-07
**Patch stack:** 13 (unchanged — no llama.cpp / WASM rebuild this stage)
**Outcome:** **H-3** (writeback-gap hypothesis misframed; the 5.24e-4 first8
Qcur-0 delta originates from the WGSL Q4_K matmul kernel disagreeing with
libllama's CPU Q4_K matmul on the same Q-projection inputs — not from a
host_mirror staleness problem).

## TL;DR

Stage 4.22's queue brief gave Stage 4.23 three checkpoint items, in order:
re-derive 5.24e-4 against an f32 reference, surface CPU-fallback ops in
layer-0 prefill, then chase the host-mirror writeback gap. Item 1 turns
out to dominate the others — Item 1's outcome refutes the writeback-gap
framing entirely.

A side-by-side diff of the **spike's** `__stage417Checkpoints` (JSEP build,
patch stack 13) and a **non-JSEP reference probe's** `__refCheckpoints`
(`webllm-wasm.js` — same llama.cpp tip; WebGPU compiled but doesn't engage
on TinyLlama's per-token shapes, so Q-proj falls back to libllama's CPU
Q4_K dequant + GEMM) on the first 12 nodes of layer-0 prefill:

| idx | name           | ref backend | spike backend | first8 maxAbsDelta |
|----:|----------------|-------------|---------------|-------------------:|
| 0   | attn_norm-0    | CPU         | CPU           | 1.0e-7             |
| 1   | Qcur-0         | CPU         | **jsep_buf**  | **5.242e-4**       |
| 2   | Qcur-0 (view)  | CPU         | jsep_buf      | 5.242e-4           |
| 3   | Qcur-0 (post)  | CPU         | CPU           | 5.242e-4           |
| 4   | Vcur-0         | CPU         | CPU           | 1.0e-9             |
| 5   | Vcur-0 (view)  | CPU         | CPU           | 1.0e-9             |
| 6   | Kcur-0         | CPU         | **jsep_buf**  | **3.376e-4**       |
| 7   | Kcur-0 (view)  | CPU         | jsep_buf      | 3.376e-4           |
| 8   | Kcur-0 (post)  | CPU         | CPU           | 3.376e-4           |
| 9   | kq-0           | CPU         | jsep_buf      | 1.194e-2           |
| 10  | kq_soft_max-0  | CPU         | CPU           | 0                  |
| 11  | kqv_out-0      | CPU         | CPU           | 0                  |

**Key reading:**

1. `attn_norm-0` (the input to Q-proj, K-proj, V-proj) runs on CPU on
   **both** sides and is bit-clean — single-ULP delta. So whatever
   differs at Q-proj cannot be blamed on a divergent input.
2. `Vcur-0` runs on CPU on the spike too. Its output matches the
   reference to ~1e-9. This is conclusive: when the spike takes the
   same code path as the reference, the output agrees to numerical
   precision. The divergence is path-dependent, not input-dependent.
3. `Qcur-0` and `Kcur-0` are the spike's only JSEP-path-only nodes in
   the first 12 ops. They are the only nodes that diverge by
   ≫ ULP. Their delta magnitudes (5.24e-4, 3.38e-4) scale with the
   K-dimension of the matmul (Q's output is 2048-wide, K's is 256-wide;
   both matmuls share the same 2048-K accumulator depth, so partial-
   sum rounding accumulates similarly — the delta tracks output-row
   magnitudes, not the matmul size).

The 5.24e-4 number is therefore the **JSEP-WGSL-Q4_K matmul vs
libllama-CPU-Q4_K matmul** disagreement on the same input bytes — a
**kernel-vs-kernel rounding / dequant mismatch**, not a host-mirror
writeback gap.

## Why Stage 4.22's writeback-gap hypothesis missed this

Stage 4.22 closed by hypothesizing that the 5.24e-4 was a delta between
"JSEP-Qcur-0 GPU result" and "a CPU-fallback Qcur-0 attempt that reads
stale `host_mirror`". That framing assumed both Q computations happen
*within the spike*: one on GPU, one shadowed via the dual-resident
host mirror.

But the historical 5.24e-4 number traces back to Stage 4.17, which
introduced the 96-checkpoint diff between **the JSEP spike and a
non-JSEP reference run** (a separate WASM module loading
`webllm-wasm.js` in a separate browser tab). Stage 4.17 captured
both sides via the cb_eval node dump and diffed them post-hoc. The
"5.24e-4 at Qcur-0 idx=0" number is the elementwise max-abs delta
between those two **separate runs**. There was never a host-mirror
component to the comparison.

Stage 4.22's f32 self-consistency check (`dequantQ4_KTile` JS port of
WGSL `load_q4_K`, matched to GPU output to 4.768e-7) verified that
**the WGSL kernel's matmul is faithful to its own dequant logic** —
but it did not verify that the WGSL dequant logic agrees with
libllama's `dequantize_row_q4_K`. Those are different things, and the
5.24e-4 says they differ by ~1e-3 per element after a 2048-K accumulate.

## Hypothesis split for Stage 4.24

The kernel reads bit-clean weight bytes (Stage 4.21 confirmed via FNV
hash against the GGUF source) and bit-clean activations (Stage 4.22's
captured src1 and `attn_norm-0` row 0 in this stage's diff both show
parity). So the disagreement lives in **how Q4_K matmul is computed**
between WGSL and libllama. Two competing root causes, both testable:

- **H-3a (likely): Dequant disagreement.** WGSL `load_q4_K` (and the
  JS port `dequantQ4_KTile` Stage 4.22 used as its f32 reference) does
  not reconstruct super-block scales / mins identically to libllama's
  `dequantize_row_q4_K`. K-quants pack a 6-bit `scale_q` and a 6-bit
  `min_q` into 12-bit pairs across 16 nibbles, with two `d` and `dmin`
  f16 super-block-level scales. Any inconsistency in nibble unpacking
  or scale/min reconstruction produces a per-element bias that grows
  proportionally with weight-row magnitude.
- **H-3b (less likely): Accumulation-order disagreement.** Both kernels
  reconstruct identical f16/f32 weights, but accumulate the 2048-K
  partial products in different orders (CPU GEMM does row-major
  AVX/NEON dot-products; the WGSL kernel does workgroup-tiled
  partial sums). At f32 precision, accumulation order can change the
  result by ~K × 1e-7 = 2e-4, which is in the same ballpark as 5e-4
  but wouldn't fully explain it on its own.

H-3a is the priority because it's both larger-magnitude and easier to
disprove — Stage 4.24 only needs to dequant the captured Q4_K bytes
with each kernel and diff the resulting f32 weight tile.

## What changed in this stage

- **Untracked → committed:** `smoke-test/p2-v2-ref-probe.{html,src.ts,js}`
  was the Stage 4.17 Probe 7 reference checkpoint capture (loads
  TinyLlama Q4_K_M through the non-JSEP `webllm-wasm.js`, arms the
  cb_eval node dump, runs the same prefill + 5-decode the JSEP spike
  runs, exposes captured `[CHECKPOINT]` lines on
  `window.__refCheckpoints`). This stage commits it so the comparison
  is reproducible from a clean checkout.
- **No code or kernel change.** Patch stack 13; no llama.cpp rebase or
  WASM rebuild required to derive the H-3 framing.

## Concrete numbers from this stage

```
ref-probe URL : http://localhost:8031/p2-v2-ref-probe.html?v=stage4.23-rd1&ingest=off
spike URL     : http://localhost:8031/p2-v2-spike.html?v=stage4.22-probe10-2&ingest=off

REF  (non-JSEP, 108 checkpoints)
  GENERATED_TOKENS = [3681, 29889, 13, 13, 29906]   ← 3681 = " Paris" ✅
  LOGIT_STATS_STEP0 = {"topId":3681, "topVal":13.043}
  PER_TOKEN_MS = 153.48 (uncached cold model load)

SPIKE (JSEP, 108 checkpoints — name parity with ref)
  GENERATED_TOKENS = [297, 8927, 13601, 29877, 29899]
  GENERATED_TEXT = "inonic boso-"
  PER_TOKEN_MS = 392.12

idx=1 Qcur-0 first8:
  ref   : [-0.0162862,  0.00477356, -0.0153744,  -0.0246583,  -0.00764941, 0.040016,  -0.00979206, 0.0449144]
  spike : [-0.0161895,  0.00484894, -0.0157384,  -0.0244936,  -0.00762007, 0.0405341, -0.00967809, 0.0454386]
  delta : [ 9.67e-5,    7.54e-5,    3.64e-4,      1.65e-4,     2.93e-5,    5.18e-4,    1.14e-4,    5.24e-4]
                                                                                                  ^^^^^^^^^^
                                                                            historical Qcur-0 first8 delta
```

## Files touched

- `smoke-test/p2-v2-ref-probe.html` (committed; was untracked).
- `smoke-test/p2-v2-ref-probe.src.ts` (committed; was untracked).
- `smoke-test/p2-v2-ref-probe.js` (committed; was untracked — bundled
  via `bun build smoke-test/p2-v2-ref-probe.src.ts --outfile
  smoke-test/p2-v2-ref-probe.js --target browser`).
- This closure report.

## Selftests

- `make checkall`: not re-run this stage (no source change vs Stage 4.22
  tip).
- Spike + ref-probe both reach the `DONE` marker through their full
  prefill + 5-decode pipeline.

## Stage 4.24 — next probe (libllama vs WGSL Q4_K dequant cross-check)

Goal: localize whether H-3a (dequant) or H-3b (accumulation-order)
explains the 5.24e-4. Probe 12 sketch:

1. Capture the Q4_K bytes for `blk.0.attn_q.weight` (already captured by
   Stage 4.22's Probe 10 — `__probe10Capture.result.src0Bytes`).
2. Dequant via the JS-ported `dequantQ4_KTile` (in `p2-v2-spike.src.ts`)
   to get the WGSL-equivalent f32 weight tile `W_wgsl`.
3. Dequant the same bytes via libllama's `ggml_dequantize_row_q4_K` —
   either via an `EM_ASYNC_JS` shim into the WASM module (cheapest;
   re-uses the same WASM that the spike already loads) or via a
   from-spec hand-port to JS.
4. Diff `W_wgsl` against `W_llama` element-wise. Branch:
   - `maxAbsDelta > 1e-5`: **H-3a confirmed** — fix the WGSL dequant.
   - `maxAbsDelta ≤ 1e-5`: **H-3b confirmed** — investigate matmul
     accumulation order. Probe 13 would re-implement the WGSL matmul
     with a libllama-style row-major partial-sum order and re-measure.

Exit criteria: a probe verdict line `[probe12] dequantDeltaMax=<x>
OUTCOME: H-3a/H-3b` and either a fix lands and `GENERATED_TEXT` flips
toward `" Paris"`, or Stage 4.24 closure documents Stage 4.25 with the
chosen branch's diagnostic plan.
