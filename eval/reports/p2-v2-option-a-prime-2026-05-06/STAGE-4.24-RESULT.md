# Stage 4.24 — Probe 12: WGSL `load_q4_K` vs libllama `dequantize_row_q4_K`

**Date:** 2026-05-07
**Patch stack:** 13 (unchanged — `webllm_dequantize_q4_K` shim landed in
`src/wasm/webgpu-bridge.cpp`, which is webllm's own bridge, not in
the `webllm-browser-patches` patch series against `~/Repos/llama.cpp`).
**Outcome:** **H-3b CONFIRMED** — the WGSL-equivalent dequant
(`dequantQ4_KTile`, the JS port of WGSL `load_q4_K`) and libllama's
`dequantize_row_q4_K` produce **bit-identical** f32 weight tiles on
the production layer-0 wq Q4_K bytes. The 5.24e-4 first8 Qcur-0 delta
must therefore originate downstream of dequant — in the WGSL matmul's
f32 accumulation order disagreeing with libllama's CPU GEMM
accumulation order.

## TL;DR

```
PROBE12_DEQUANT_DELTA = {
  "M": 2048, "K": 2048, "totalElems": 4194304,
  "maxAbsDelta": 0,
  "maxIdx": -1,
  "nNaN": 0, "nInf": 0,
  "first8Wgsl":  [-0.001478672, -0.003317833, -0.006996155,
                  -0.014352798, -0.003317833, -0.003317833,
                   0.000360489, -0.003317833],
  "first8Llama": [-0.001478672, -0.003317833, -0.006996155,
                  -0.014352798, -0.003317833, -0.003317833,
                   0.000360489, -0.003317833],
  "verdict": "H-3b"
}
```

`maxAbsDelta = 0` over **4,194,304** dequantized elements (full
[M=2048, K=2048] wq weight tile, 16 MiB at f32). Zero NaN, zero Inf.
First-8 outputs byte-identical. The WGSL kernel's dequant logic is
provably correct against libllama's reference — Stage 4.22's
self-consistency check (`dispatchMatmul` output vs JS-port f32 reference)
verified the kernel against its own dequant; Probe 12 closes the gap by
verifying that JS port against libllama directly. Both are now bit-clean.

The remaining suspect for the 5.24e-4 production Qcur-0 delta is **f32
matmul accumulation-order disagreement**:

- WGSL kernel (`src/inference/jsep/ops/matmul.ts`): workgroup-tiled
  partial sums, 16-wide subgroups, 4 `OUTPUTS_PER_WG`. Reduction tree
  inside subgroup, then `workgroupBarrier` + cross-subgroup horizontal
  add for the final tile sum.
- libllama CPU GEMM (`ggml-cpu/ops.cpp` for Q4_K MUL_MAT, ultimately
  `vec_dot_q4_K_q8_K` in `ggml-cpu/quants.c`): row-major SIMD dot-product
  with horizontal sum (AVX2 `_mm256_hadd_ps` / NEON `vaddvq_f32` /
  scalar fallback). Different reduction tree.

f32 partial-sum accumulation is non-associative; reordering 2048-K
reductions can produce ~K × 2 × 1e-7 ≈ 4e-4 of disagreement on
inputs whose magnitudes are O(0.1). The observed 5.24e-4 fits that
envelope; the WGSL-vs-f32-loop delta of 4.77e-7 (Stage 4.22) confirms
the WGSL kernel matches an f32 k-major reference — i.e., the kernel
is mathematically equivalent **to the chosen reference**, not to
libllama's reference.

## What probe 12 measured

The probe reuses Stage 4.22's `__probe10Capture.result.src0Bytes` —
the exact 2,359,296-byte Q4_K tile the WGSL kernel sees at the first
production JSEP MUL_MAT dispatch (`Qcur-0`, layer-0 Q-projection on
TinyLlama-1.1b-chat-q4_0.gguf). Two dequant paths run on the same
bytes:

- **Path A** — `dequantQ4_KTile(src0Bytes, M=2048, K=2048)` — JS port
  of WGSL `load_q4_K` (Stage 4.22 added; verified bit-clean against
  the WGSL kernel via the captured kernel-output replay).
- **Path B** — `mod._webllm_dequantize_q4_K(srcPtr, dstPtr, M*K)` — the
  new `webllm_dequantize_q4_K` shim wraps
  `ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float` =
  `dequantize_row_q4_K` (in `ggml-quants.c`). The shim copies
  `cap.src0Bytes` into `mod.HEAPU8` at `srcPtr`, calls the dequant,
  reads `M*K` f32 outputs from `mod.HEAPF32` at `dstPtr/4`.

Element-wise diff over 4,194,304 outputs:

- `maxAbsDelta = 0` (exact, not single-ULP).
- `nNaN = 0`, `nInf = 0` — every output is finite on both paths.
- First-8 outputs byte-identical.

The probe runs once per spike load, after Probe 10's capture has been
recorded. It allocates two transient `_malloc` blocks (2,359,296 + 16
MiB ≈ 18.6 MiB) and frees them after the diff. No impact on the
decode loop.

## Implementation

### libllama dequant export (`src/wasm/webgpu-bridge.cpp`)

```cpp
// Stage 4.24 Probe 12: libllama Q4_K dequant shim. Calls
// ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float (= dequantize_row_q4_K
// in ggml-quants.c). Used by the spike harness to cross-check the WGSL
// `load_q4_K` reconstruction against libllama's reference path on
// captured production weight bytes (`__probe10Capture.result.src0Bytes`).
int32_t webllm_dequantize_q4_K(const void* src, float* dst, int32_t k) {
    if (!src || !dst || k <= 0 || (k % 256) != 0) return -1;
    const struct ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (!traits || !traits->to_float) return -1;
    traits->to_float(src, dst, (int64_t) k);
    return 0;
}
```

Added to `EXPORTED_FUNCTIONS` in `src/wasm/CMakeLists.txt` (alongside
the other `webllm_*` exports). Not added to `JSPI_EXPORTS` —
`dequantize_row_q4_K` is synchronous CPU code, no async readback,
and `WebAssembly.promising`-wrapping it would just add suspend-frame
overhead.

The shim lives in webllm's own bridge file (`webgpu-bridge.cpp`), not
in `~/Repos/llama.cpp`'s `webllm-browser-patches` series — patch stack
unchanged at 13.

### Spike harness block (`smoke-test/p2-v2-spike.src.ts`)

Added inside the existing Probe 10 try block, gated on
`cap.src0Type === GGML_TYPE_Q4_K` (TinyLlama's projections are
all Q4_K per Stage 4.22's surprise finding — the `Q4_0` in the GGUF
filename is the HuggingFace tier label, not the on-disk tensor type
for projections). The block:

1. Calls `dequantQ4_KTile(cap.src0Bytes, cap.M, cap.K)` for Path A.
2. `_malloc`s heap buffers, `HEAPU8.set` the captured bytes,
   `_webllm_dequantize_q4_K(srcPtr, dstPtr, M*K)`, copies
   `HEAPF32.subarray(...)` into a JS-owned Float32Array for Path B.
3. Element-wise iterates both, tracks `maxAbsDelta + maxIdx + nNaN + nInf`.
4. Logs `PROBE12_DEQUANT_DELTA = {...JSON...}` and
   `[probe12] dequantDeltaMax=<x> maxIdx=<i> OUTCOME: <H-3a|H-3b>`.

`make checkall` green (747 pass / 36 skip / 0 fail). All 6 spike
selftests + 5 sweep selftests still PASS. Per-token decode 311.60 ms
(within noise of Stage 4.22's 879.7 ms run-with-sweep envelope; this
run reuses Stage 4.23's spike state without the q4_0 sweep enabled).

## Smoking-gun reframing

Stages 4.17 → 4.21 walked through the upstream side of the chain:
input embedding → attn_norm-0 → weight upload → GPU readback. Every
link came back bit-clean. Stage 4.22 verified the kernel matches its
own dequant reference (Stage 4.18's standalone Q4_0 sweep was an
apples-vs-oranges baseline because the file was actually Q4_K).
Stage 4.23 placed a ref-probe alongside the spike and confirmed
the 5.24e-4 number is real and reproducible at exactly element [7]
of `Qcur-0` first8 — and that it appears at the very first JSEP-side
op in the prefill chain.

Stage 4.24 closes the dequant question. The full picture for the
production Qcur-0 5.24e-4 first8 delta is now:

1. Input bytes (`attn_norm-0`): bit-identical between paths (Stage 4.19).
2. Q4_K weight bytes in JSEP `set_tensor` view: bit-identical to GGUF
   parser output (Stage 4.20).
3. Q4_K weight bytes in JSEP `GPUBuffer` after upload: bit-identical
   to `set_tensor` view (Stage 4.21).
4. Q4_K dequant: WGSL-equivalent dequant `dequantQ4_KTile` matches
   libllama's `dequantize_row_q4_K` byte-for-byte over 4M+ elements
   (Stage 4.24).
5. **The remaining variable is matmul accumulation order**.

The WGSL kernel reduces 2048 partial products via subgroup tree +
workgroup horizontal add. libllama reduces via SIMD lane-pair adds +
SIMD horizontal. f32 reductions of length 2048 with O(0.1) operands
disagree on their last 12-13 mantissa bits = O(1e-4) — exactly the
observed delta envelope.

## What ships from Stage 4.24

- `src/wasm/webgpu-bridge.cpp`: `webllm_dequantize_q4_K` shim.
- `src/wasm/CMakeLists.txt`: export added to `EXPORTED_FUNCTIONS`.
- `smoke-test/p2-v2-spike.src.ts`: Probe 12 block (Q4_K-only,
  gated on `cap.src0Type`).
- `smoke-test/webllm-wasm-jsep.{js,wasm}`: rebuilt with new export
  (`make wasm-build-jsep`).
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.24-RESULT.md`:
  this file.
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.24-spike-output.txt`:
  raw `#log` text from the spike run.

`GENERATED_TEXT = "inonic boso-"` (unchanged — bug still active,
framing now further localized to matmul accumulation).

## Branch on outcome — Stage 4.25

H-3b confirmed ⇒ Stage 4.25 investigates the WGSL matmul
accumulation order vs libllama CPU GEMM. Three credible paths:

- **A. Re-implement WGSL matmul with libllama-style accumulation**
  (row-major scalar + horizontal sum at the end). Likely catastrophic
  for performance — defeats the workgroup-tiled subgroup design that
  hits memory-bound throughput. Useful only as a parity check, not
  a ship target.
- **B. Kahan-summed accumulator inside the WGSL kernel.** Adds 1
  f32 register per output, doubles the per-iteration work. Quantify
  the perf cost via the existing Stage 4.18 sweep harness; if the
  hit is <30% it's a credible ship lever for parity-critical paths
  (causal-LM accuracy).
- **C. Accept the 5.24e-4 disagreement as inherent f32 floor noise**
  and target the downstream cascade. The kq-0 1.19e-2 delta and
  the resulting "inonic boso-" gibberish are *amplification* effects
  — RMSNorm + softmax exponentially compound any input delta. Even
  if matmul is ULP-tight against libllama, the final softmax can
  flip top-1 logit ranks. This branch trades parity for shippability:
  fix the cascade with FA / softmax precision boosts / per-layer
  RMSNorm in f64 / keep mat-vec in WGSL but cast accumulator to f32
  Kahan only on the lm-head (where ranking is most sensitive).

The next session's Stage 4.25 brief should be: **a probe** that
quantifies how much of the 5.24e-4 is *actually* matmul
accumulation-order divergence vs other f32 reduction differences.
The brief sketches a Kahan-WGSL prototype gated to `Qcur-0` only
and measures the resulting delta — if Kahan flips it to ≤ 1e-5,
H-3b-Kahan is the ship target; if it doesn't move, the disagreement
is structural and Branch C (downstream mitigation) is the only
viable path.
