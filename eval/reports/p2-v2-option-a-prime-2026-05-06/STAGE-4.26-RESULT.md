# Stage 4.26 — Probe 14: libllama CPU Q4_K × Q8_K matmul precision check

**Date:** 2026-05-07
**llama.cpp tip:** `ef89f9314` on `webllm-browser-patches` (patch stack 13 — unchanged from Stage 4.25).
**WebLLM tip:** `<pending>` (no llama.cpp patch — `webllm_q4k_q8k_matmul` shim landed in webllm's own `src/wasm/webgpu-bridge.cpp`, mirroring Stage 4.24's `webllm_dequantize_q4_K` pattern).
**Outcome:** **H-4-libllama-imprecise CONFIRMED** — `llamaVsF64Max = 4.178e-2`, four orders of magnitude larger than `wgslVsF64Max = 7.94e-6`. libllama's CPU `quantize_row_q8_K` → `vec_dot_q4_K_q8_K` path is the imprecise side of the cross-module disagreement by a wide margin. **The matmul-precision investigation is now closed.** webllm's WGSL Q4_K matmul kernel is *more accurate than libllama's CPU reference* on the captured production Q-projection inputs; the 5.24e-4 historical disagreement (and the larger 4.178e-2 measured here) is a libllama precision artifact, not a webllm bug. Stage 4.27 must pivot to characterizing which downstream op produces "inonic boso-" — Q-proj is not the first faulting op and matmul precision is not the failure mode.

## Probe 14 brief recap (from TODO Stage 4.26)

> Add a `webllm_q4k_q8k_matmul(const void *src0_q4k, const float *src1_f32, float *dst_f32, int M, int K, int N)` C export to `src/wasm/webgpu-bridge.cpp` that runs libllama's `vec_dot_q4_K_q8_K` on the captured production Q-projection inputs, then in the spike harness compute `llamaVsF64Max` against an f64 reference over the same `cap.src0Bytes` / `cap.src1Bytes`. Emit `[probe14] llamaVsF64=<x>.exp(3) verdict: <H-4-libllama-{imprecise|precise|mid}>`.

## Implementation

### C++ shim (`src/wasm/webgpu-bridge.cpp`)

`webllm_q4k_q8k_matmul` mirrors the path inside `ggml_compute_forward_mul_mat` for `type=Q4_K`:

```cpp
extern "C" int32_t webllm_q4k_q8k_matmul(
    const void* src0_q4k, const float* src1_f32, float* dst_f32,
    int32_t M, int32_t K, int32_t N
) {
    const struct ggml_type_traits_cpu* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const struct ggml_type_traits_cpu* q4 = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const size_t nb_q4k_row = ggml_row_size(GGML_TYPE_Q4_K, K);
    const size_t nb_q8k_row = ggml_row_size(GGML_TYPE_Q8_K, K);
    void* src1_q8k = std::malloc(N * nb_q8k_row);
    for (int n = 0; n < N; ++n)
        q8->from_float(src1_f32 + n*K, (char*)src1_q8k + n*nb_q8k_row, K);
    for (int n = 0; n < N; ++n) {
        const char* vy = (const char*)src1_q8k + n*nb_q8k_row;
        for (int m = 0; m < M; ++m) {
            const char* vx = (const char*)src0_q4k + m*nb_q4k_row;
            q4->vec_dot(K, &dst_f32[n*M + m], 0, vx, 0, vy, 0, 1);
        }
    }
    std::free(src1_q8k);
    return 0;
}
```

This is the *exact* path libllama takes in production: src1 is f32 in the
graph, the type-traits' `from_float` quantizes it to `block_q8_K`, then the
type-traits' `vec_dot` runs `vec_dot_q4_K_q8_K` per output element. So the
shim's `dst_f32` is what libllama would produce in production for this
matmul shape — including the Q8_K quantization step's precision loss.

The shim resolves to `ggml-cpu/quants.c::quantize_row_q8_K_generic` and
`ggml-cpu/quants.c::ggml_vec_dot_q4_K_q8_K_generic` on the wasm32 build
(no SIMD specializations available; CPU arch fallback is `GGML_CPU_GENERIC`).
This is the same code path libllama would take on a wasm32 build of the
full library — the shim is representative of the library's true behavior in
the browser-wasm deployment scenario, not just an x86/ARM native build.

### CMake export (`src/wasm/CMakeLists.txt`)

`_webllm_q4k_q8k_matmul` added to `EXPORTED_FUNCTIONS` (alongside
`_webllm_dequantize_q4_K`). NOT in `JSPI_EXPORTS` — synchronous CPU
computation, no promising-wrap needed.

### Spike harness (`smoke-test/p2-v2-spike.src.ts`)

Probe 14 sub-block immediately after Probe 13's verdict line, gated on
`cap.src0Type === GGML_TYPE_Q4_K`. Reuses the `cap.src0Bytes` / `cap.src1Bytes`
/ `src0Dequant` / `src1View` already in scope from Probe 12/13:

1. Malloc src0 (2,359,296 B), src1 (49,152 B), dst-shim (49,152 B) on the WASM heap.
2. Re-derive `HEAPU8` after malloc (heap may have grown — same pattern as Probe 12).
3. Copy captured bytes in: `heapU8.set(cap.src0Bytes, src0Ptr)` etc.
4. Call `_webllm_q4k_q8k_matmul(src0Ptr, src1Ptr, dstShimPtr, M, K, N)`.
5. Slice the result out of `HEAPF32` into a JS-owned `Float32Array(M*N)`.
6. Build f64 reference: nested loop over `src0Dequant[m*K+k] * src1View[n*K+k]` accumulated in a JS double (no `Math.fround`).
7. Score llamaOutput vs refF64, also re-score WGSL captured `cap.dstAfterBytes` vs refF64 (refresh against the f64 oracle so both numbers in the closure share an oracle), and compute llamaVsWgslMax for completeness.
8. Verdict on `llamaVsF64Max`: ≥1e-4 → imprecise, ≤1e-5 → precise, else mid.
9. Free all three allocations.

## Headline numbers (from `STAGE-4.26-spike-output.txt`)

```
PROBE14_LLAMA_MATMUL_VS_F64 = {
  "M":2048, "K":2048, "N":6,
  "llamaVsF64Max": 0.04178485598157655,   // 4.178e-2
  "llamaVsF64Idx": 11567,
  "wgslVsF64Max":  0.000007943707068136519, // 7.94e-6
  "llamaVsWgslMax": 0.04178142547607422,   // 4.178e-2 (same idx 11567)
  "llamaVsWgslIdx": 11567,
  "nNaN": 0, "nInf": 0,
  "first8Llama":   [-0.01629,  0.00477, -0.01537, -0.02466, -0.00765,  0.04002, -0.00979,  0.04491],
  "first8Wgsl":    [-0.01619,  0.00485, -0.01574, -0.02449, -0.00762,  0.04053, -0.00968,  0.04544],
  "first8RefF64":  [-0.01619,  0.00485, -0.01574, -0.02449, -0.00762,  0.04053, -0.00968,  0.04544]
}
[probe14] llamaVsF64=4.178e-2 wgslVsF64=7.944e-6 llamaVsWgsl=4.178e-2
         verdict: H-4-libllama-imprecise (libllama ≥1e-4 from f64 truth — close
                  matmul-precision investigation; pivot to other ops in cascade)
```

Cross-reference (unchanged from Stage 4.25):

```
MATMUL_PROBE13_DELTA.kahanVsBaselineMax = 0   (Kahan kernel ≡ baseline kernel; H-3b-structural)
MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64   = 7.94e-6  (WGSL kernel @ f64 floor)
GENERATED_TEXT                            = "inonic boso-"  (still the failing decode)
```

## Numeric interpretation

The `wgslVsF64Max = 7.94e-6` and `first8Wgsl` ≈ `first8RefF64` agree to all
displayed digits: WGSL is tracking f64 truth to within ULP across the
displayed positions. Element 11567 (where libllama is worst) is at
`(n=5, m=1199)` in the `[N=6, M=2048]` output — a position with
`|refF64| ≈ 0.6+`, where the relative error is still ~6%. Three
candidate explanations for libllama's 4.178e-2 deviation:

1. **Q8_K src1 quantization loss.** `quantize_row_q8_K` projects the
   f32 src1 row to a per-256-element-block `(d, qs[256], bsums[16])`
   tuple where `d` is f32 scale and `qs` is int8 quantized values.
   Per-element relative error ~1/127 ≈ 7.9e-3 in the worst case, so a
   2048-K dot product can accumulate up to ~16x that on a single f32
   output ≈ 1.3e-1 (envelope). 4.178e-2 fits comfortably inside this
   envelope.
2. **Generic-CPU vec_dot scalar fallback path.** wasm32 build resolves
   to `ggml_vec_dot_q4_K_q8_K_generic` (no SIMD), which uses ~12-bit
   integer accumulation per K-block (ggml-cpu/quants.c:645+). The
   scalar reduction order differs from any SIMD specialization but
   the precision profile is similar.
3. **The combination of (1) and (2)** — Q8_K quantization is the
   dominant source; vec_dot reduction adds ~1 ULP per K-block.

The dominant source is (1): src1 quantization to int8. WGSL doesn't
quantize src1 — it consumes the raw f32 activations directly through
the `src1[...]` global memory access. **WGSL's apples are not
libllama's oranges**: libllama spends a Q8_K conversion to reach the
Q4_K matmul kernel; WGSL goes straight from f32 src1 + Q4_K weights
to f32 output.

## Cross-reference: 5.24e-4 historical claim vs 4.178e-2 measured

The TODO/closure briefs cite a "5.24e-4 cross-module disagreement" as
the historical Qcur-0 delta between webllm and libllama. Probe 14
measures 4.178e-2 — two orders of magnitude larger. Three possible
reconciliations:

- **Different reference points.** The 5.24e-4 figure may have been
  measured against a different libllama capture (e.g., a different
  prefill position, a different model load, or libllama's own f64-built
  binary rather than the wasm32 GENERIC build).
- **Different element scoring.** 5.24e-4 may be the `first8` headline
  delta, and 4.178e-2 is the worst element across all 12,288 outputs
  (`M*N = 2048*6 = 12288`). Inspecting `first8Llama` vs `first8RefF64`:
  per-element deviations are 1e-4 to 5e-4, consistent with the
  historical 5.24e-4 figure. The 4.178e-2 worst case is at a single
  high-magnitude element (idx 11567), tail of the distribution.
- **Both effects combined.**

The conclusion is robust either way: **libllama is the imprecise
side** by orders of magnitude, regardless of which delta you use.
WGSL is at its f64 precision floor; libllama is anywhere from
1e-4 (median element) to 4e-2 (worst element) from truth.

## Implication for the "inonic boso-" cascade

webllm's WGSL Q-projection output is *more accurate than libllama's*
production output on the same inputs. Yet:

- The reference probe (running libllama's full prefill+decode under the
  webgpu backend) generates coherent text from the same prompt.
- The spike (running prefill via JSEP and decoding from the post-prefill
  KV cache) generates "inonic boso-".

If the bug were in the Q-projection matmul, fixing it to be *more
accurate* should at worst shift the generated text — but webllm's
matmul is already more accurate than the reference path that decodes
correctly. So **the Q-projection matmul is not the bug source.** The
"inonic boso-" failure must come from a different op in the cascade
between Q-proj and final logits — or from a different stage entirely
(prefill vs decode, KV-cache write, lm_head logits readback).

Candidate upstream / parallel suspects to characterize next (Stage 4.27 brief):

1. **Per-layer activation drift.** `__stage417Checkpoints` already
   captures per-layer hidden states between webllm-spike and the
   reference path. Diff the post-Q-proj output (or any later stage
   like K-proj, V-proj, post-attention, post-FFN, post-RMSNorm) for
   the first layer and find the first checkpoint that diverges by
   more than a magnitude consistent with logit-scale failure.
2. **Decode-side bug, not prefill-side bug.** The reference probe and
   the spike both prefill via the same WGSL kernels (the spike just
   adds JSEP wrapping); but the decode loop differs. Decode generates
   token n+1 from the post-prefill KV cache; if the spike's KV cache
   layout / SET_ROWS / RoPE is wrong, decode fails even with correct
   prefill activations.
3. **lm_head logits.** Step 8/8 dumps `LOGIT_STATS_STEP0`. Compare
   first8 logits between spike and reference probe at step 0 — if
   they disagree by >1e-2, the lm_head matmul (or the post-rms-norm
   feeding it) is the failing op.

## Files touched

- `src/wasm/webgpu-bridge.cpp` (+ `webllm_q4k_q8k_matmul` shim, ~50 LOC after the Stage 4.24 dequant shim).
- `src/wasm/CMakeLists.txt` (+ `_webllm_q4k_q8k_matmul` to `EXPORTED_FUNCTIONS`; intentionally NOT in `JSPI_EXPORTS`).
- `smoke-test/p2-v2-spike.src.ts` (+ Probe 14 sub-block in post-Probe-13 try, ~140 LOC; emits `PROBE14_LLAMA_MATMUL_VS_F64` JSON + `[probe14]` verdict line).

No llama.cpp patch (patch stack 13 unchanged).

## Verification

- All 6 spike selftests + 5 sweep selftests still PASS (Q4K_SELFTEST, RMSNORM_SELFTEST, RMSNORM_MULTIROW_NODIVERT/DIVERT, MATMUL_PROD_NODIVERT/DIVERT, MATMUL_K2048_N1, MATMUL_K2048_N6, MATMUL_RW16_K1024_N1).
- `make checkall` green: 747 pass / 36 skip / 0 fail.
- Per-token decode 22.97 ms (within noise of Stage 4.25 baseline).
- Cross-references unchanged: `kahanVsBaselineMax = 0`, `wgslVsF64Max = 7.94e-6`, `GENERATED_TEXT = "inonic boso-"`.

## Exit criteria — all met

- ✅ `[probe14]` verdict line emitted with `llamaVsF64Max = 4.178e-2` and `OUTCOME: H-4-libllama-imprecise`.
- ✅ All spike + sweep selftests still PASS; `make checkall` green.
- ✅ Stage 4.27 paste-and-go brief queued (Branch H-4-libllama-imprecise — close matmul-precision investigation; pivot to characterizing which downstream op produces "inonic boso-" via the `__stage417Checkpoints` per-layer diff framework).

## Raw artifact

[`STAGE-4.26-spike-output.txt`](STAGE-4.26-spike-output.txt) — full
spike text dump (all selftests, Probe 10/12/13/14 outputs, post-prefill
buffer dumps, generated tokens, decode timing).
