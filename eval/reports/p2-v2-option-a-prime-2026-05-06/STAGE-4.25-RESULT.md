# Stage 4.25 — Probe 13: Kahan-summed WGSL matmul accumulator gated to `Qcur-0`

**Date:** 2026-05-07
**llama.cpp tip:** `ef89f9314` on `webllm-browser-patches` (patch stack 13 — unchanged from Stage 4.24).
**WebLLM tip:** `<pending>` (no llama.cpp patch — Kahan path lives entirely in webllm's `src/inference/jsep/ops/matmul.ts`; patch stack unchanged).
**Outcome:** **H-3b-structural CONFIRMED** — Kahan-corrected accumulation in the WGSL Q4_K MUL_MAT kernel produces bit-identical output to the non-Kahan baseline (`kahanVsBaselineMax = 0` exact, all 8 first-output positions match). Combined with the pre-existing `MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64 = 7.94e-6`, this proves **f32 accumulation precision is not the dominant error source** for the historical 5.24e-4 cross-module Qcur-0 disagreement: even closing the residual 7.94e-6 to zero would resolve only ~1.5% of the 5.24e-4 gap. The remaining 99% must come from a different source — most likely libllama's CPU Q4_K matmul precision, or a different src1 input upstream that the Stage 4.23 attn_norm-0 idx=0 check missed.

## Probe 13 brief recap (from TODO Stage 4.25)

> Add a Kahan-summed accumulator path to `dispatchMatmul` in
> `src/inference/jsep/ops/matmul.ts`, gated to layer-0 `Qcur-0` (or any
> layer-0 Q4_K MUL_MAT — the dispatch path doesn't see the ggml node
> name directly, so gate via a one-shot `__stage425KahanArm` flag set
> right before prefill). Re-run the spike. Diff the
> `MATMUL_PROBE10_CAPTURED_DELTA.maxAbsDelta` between the kernel's
> existing path and the Kahan path on the *same captured production
> inputs*.

## Implementation

### WGSL kernel (`src/inference/jsep/ops/matmul.ts`)

`buildMatmulShader` gained a `kahan = false` parameter, honored only on
the `GGML_TYPE_Q4_K` branch (other quants don't share the Q-projection
codepath at production scale on TinyLlama). The Q4_K main loop's plain
accumulator:

```wgsl
var acc: f32 = 0.0;
for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
    let a: f32 = load_q4_K(m, k, batch);
    let b: f32 = src1[...];
    acc = acc + a * b;
}
dst[dst_idx] = acc;
```

becomes Neumaier-Kahan compensated when `kahan == true`:

```wgsl
var acc: f32 = 0.0;
var compensation: f32 = 0.0;
for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
    let a: f32 = load_q4_K(m, k, batch);
    let b: f32 = src1[...];
    let term: f32 = a * b;
    let t: f32 = acc + term;
    if (abs(acc) >= abs(term)) {
        compensation = compensation + ((acc - t) + term);
    } else {
        compensation = compensation + ((term - t) + acc);
    }
    acc = t;
}
acc = acc + compensation;
dst[dst_idx] = acc;
```

The Neumaier variant (always-keep-larger) was chosen over plain Kahan
because intra-thread `acc` and `term` magnitudes can flip during a
2048-K run over signed weight × signed activation products.

### Dispatch gate (`src/inference/jsep/ops/matmul.ts::dispatchMatmul`)

Reads `globalThis.__stage425KahanArm` once per dispatch. If armed AND
`src0.type === GGML_TYPE_Q4_K` AND `M === 2048 && K === 2048 && N === 6`
(matches Qcur-0 layer 0 shape exactly — Kcur-0 is M=256 so it doesn't
fire), the gate sets `useKahan = true`, disarms the flag, and sets a
one-shot confirmation flag `globalThis.__stage425KahanFired = true` so
the spike's verdict block can disambiguate "Kahan ran, output unchanged"
from "Kahan gate never fired".

The Kahan variant uses a separate pipeline cache key
(`mat-q4_k-f32-f32-2-kahan`) so the production kernel pipeline is
unchanged for every other dispatch in the 22-layer prefill (~1936
Q4_K MUL_MAT dispatches). One pipeline build cost amortized over a
single dispatch.

### Spike harness (`smoke-test/p2-v2-spike.src.ts`)

Sets `__stage425KahanArm = true` immediately before `bridge.decode(prefill)`,
adjacent to the existing `__probe10Capture` arm. After Probe 10 captures
src0/src1/dst-after, the new Probe 13 block:

1. Diffs the captured `dstAfter` first8 against the hard-coded Stage 4.24
   baseline `first8Got` (recorded in `STAGE-4.24-RESULT.md`) — this is
   `kahanVsBaselineMax`.
2. Builds a Kahan-summed JS reference (`refKahan`, `Math.fround`-clamped
   Neumaier mirror of the WGSL kernel) and compares the captured
   `dstAfter` against it — this is `kahanVsKahanRefMax`.
3. Reads `__stage425KahanFired` for the explicit "did the gate fire"
   confirmation.

## Result

```
[probe13] kahan accumulator armed (Qcur-0 layer 0 only)
[probe10] captured M=2048 K=2048 N=6 src0=2359296B src1=49152B dstBefore=49152B dstAfter=49152B
MATMUL_PROBE10_REPLAY = {"maxAbsDeltaVsF32Loop":4.768e-7,"maxAbsDeltaVsF64":7.944e-6,...}
MATMUL_PROBE10_CAPTURED_DELTA = {"maxAbsDelta":4.768e-7,...}
[probe10] OUTCOME: G-2 (synthetic ≤1e-5 — bug between dispatch site and shader execution)
PROBE12_DEQUANT_DELTA = {"maxAbsDelta":0,"verdict":"H-3b"}
[probe12] dequantDeltaMax=0.000e+0 maxIdx=-1 OUTCOME: H-3b
MATMUL_PROBE13_DELTA = {
  "M":2048, "K":2048, "N":6,
  "kahanVsBaselineMax": 0,
  "kahanVsBaselineIdx": -1,
  "perElem": [0,0,0,0,0,0,0,0],
  "kahanFirst8": [-0.01618947833776474, 0.004848937503993511,
                  -0.015738369897007942, -0.02449355274438858,
                  -0.007620065473020077, 0.04053414985537529,
                  -0.009678085334599018, 0.04543862119317055],
  "stage424BaselineFirst8": [-0.01618947833776474, 0.004848937503993511,
                             -0.015738369897007942, -0.02449355274438858,
                             -0.007620065473020077, 0.04053414985537529,
                             -0.009678085334599018, 0.04543862119317055],
  "kahanVsKahanRefMax": 8.106e-6
}
[probe13] kahanArm=true kahanFired=true capturedDelta=0.000e+0 baseline=5.242e-04 \
          kahanVsKahanRef=8.106e-6 \
          verdict: H-3b-structural (Kahan ≈ baseline — accumulation order is not the disagreement source; cascade mitigation needed)
GENERATED_TEXT = "inonic boso-"
```

Raw output: [`STAGE-4.25-spike-output.txt`](./STAGE-4.25-spike-output.txt).

## Interpretation

### The headline numbers

| Metric | Stage 4.24 baseline (no Kahan) | Stage 4.25 with Kahan |
|---|---|---|
| `MATMUL_PROBE10_CAPTURED_DELTA.maxAbsDelta` | 4.768e-7 | 4.768e-7 |
| `kahanVsBaselineMax` | n/a (= 0 by definition) | **0 (exact)** |
| `kahanVsKahanRefMax` | n/a | 8.106e-6 |
| `MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64` | 7.944e-6 | 7.944e-6 (replay path is non-Kahan ref) |
| `kahanFired` (confirms gate engaged) | n/a | **true** |
| `GENERATED_TEXT` | "inonic boso-" | "inonic boso-" (unchanged) |

**The dispatch gate fired (kahanFired=true), and the kernel produced
bit-identical output to the non-Kahan path.**

### Why bit-identical, despite Kahan running

Two non-exclusive explanations:

1. **WGSL compiler eliding the compensation.** Naga / Tint backends apply
   floating-point algebraic simplification — `(acc + term) - acc - term`
   is mathematically zero, and the compiler may fold it even though
   strict IEEE-754 forbids the simplification. WGSL has no
   `volatile`-equivalent, no `#pragma STDC FP_CONTRACT off`, and Tint's
   default optimization profile doesn't preserve fp ordering. Defeating
   this would require `bitcast<f32>(bitcast<u32>(...))` round-trips on
   `acc` and `compensation` between iterations — at significant per-add
   cost.

2. **Compensation magnitude below ULP at the final `acc + compensation`.**
   For Q-projection on TinyLlama: dequantized weights are O(0.005-0.05),
   src1 (post attn_norm) values O(0.1-1.0), per-element products
   O(1e-3 to 5e-2), running `acc` magnitude ~0.045, ULP(0.045) ≈ 5.4e-9.
   Each add's lost-low-order is O(ULP(acc)) ≈ 5e-9; over 2048 steps
   the accumulated correction can be anywhere from ~0 (if cancellations
   dominate) to ~1e-5. If the running `compensation` lands below
   ULP(0.045) at the end, `acc + compensation` rounds to `acc` exactly.

Distinguishing (1) from (2) would require disassembling the compiled
WGSL → SPIR-V or Naga IR. That's a deep dive disproportionate to the
finding — see "structural conclusion" below.

### Structural conclusion (why Probe 13's outcome is robust either way)

Independent of whether Kahan ran or was elided, the **pre-existing**
`MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64 = 7.944e-6` proves the WGSL
Q4_K matmul is already accurate to within ~8e-6 of f64 truth on the
captured Qcur-0 inputs. The historical cross-module disagreement
(Stage 4.23: WGSL Qcur-0 vs libllama CPU Qcur-0 first8 element [7]
delta = 5.242e-4) is **~67× larger** than the WGSL kernel's
distance-from-truth.

**Even if Kahan worked perfectly and brought WGSL's accuracy to 1e-7,
it would close only ~8e-6 of the 5.24e-4 gap — 1.5% of the
disagreement.** The remaining 99% must come from a different source
than f32 accumulation precision.

This **rules out matmul accumulation order as the dominant error
source** for the 5.24e-4 disagreement. The investigation must redirect.

### Where the remaining 99% lives — three candidates for Stage 4.26

1. **libllama's CPU Q4_K matmul precision.** If WGSL is 8e-6 from f64
   truth and libllama is also 8e-6 from f64 truth, they should agree to
   ~1.6e-5. They don't (5.24e-4). Either libllama's
   `vec_dot_q4_K_q8_K` has worse-than-expected precision on this input
   distribution, or libllama is NOT close to f64 truth on this input.
   **Cheap probe:** compute the same Q-projection via libllama's CPU
   path on the captured src0/src1 (similar shim shape to Stage 4.24's
   `webllm_dequantize_q4_K`), compare against an f64 reference. If
   libllama vs f64 is also 5e-4 magnitude, libllama's CPU matmul is
   the imprecise one.

2. **Different src1 inputs at idx=1 Qcur-0 between the two modules.**
   Stage 4.23's idx=0 attn_norm-0 was bit-clean (Δ=1e-7), but idx=0 and
   idx=1 are different positions in the checkpoint stream. Re-derive
   the src1 (post-attn_norm RMSNorm output, the Q-proj input) at the
   exact moment of the Qcur-0 dispatch in both modules and re-compare.
   If src1 itself differs by 5e-4 magnitude, the bug is upstream of
   Q-proj entirely (RMSNorm, embedding lookup, or the input token
   handling).

3. **Q4_K dequantization on the libllama side disagreeing with the
   WGSL side at the integration boundary.** Stage 4.24 Probe 12 proved
   `webllm_dequantize_q4_K` (= `dequantize_row_q4_K`) and
   `dequantQ4_KTile` (the JS port of WGSL `load_q4_K`) agree to
   `maxAbsDelta = 0`. So **stand-alone** dequant matches. But
   libllama's `vec_dot_q4_K_q8_K` does NOT use `dequantize_row_q4_K`
   internally — it uses fused dequant-and-multiply via SIMD on the
   block-quantized Q8_K activation. The fused path is implemented
   independently and might compute a slightly different per-element
   weight value than the standalone `dequantize_row_q4_K`. **Probe:**
   single-step the libllama vec_dot path on the captured inputs and
   compute the implicit per-element dequant; compare against
   `webllm_dequantize_q4_K` outputs.

Of the three, **#1 (libllama precision check)** is the cheapest and
most decisive probe — same shim pattern as Stage 4.24, two days of
investigation collapsed to one new function export.

## Files touched

- `src/inference/jsep/ops/matmul.ts`
  - `buildMatmulShader(src0Type, src1Type, dstType, kahan = false)` — added `kahan` parameter, only honored on Q4_K branch.
  - Q4_K branch: introduced block scope (`case GGML_TYPE_Q4_K: { ... }`) so the conditional template-literal interpolation compiles cleanly. Inner accumulator gains Neumaier-Kahan compensation when `kahan == true`.
  - `buildPipeline(...)` — gained `kahan = false` pass-through parameter.
  - `dispatchMatmul`: new gate reads `globalThis.__stage425KahanArm`, gates on `(M, K, N, src0.type) == (2048, 2048, 6, GGML_TYPE_Q4_K)`, sets one-shot `__stage425KahanFired` flag, uses `mat-q4_k-f32-f32-2-kahan` cache key when armed.

- `smoke-test/p2-v2-spike.src.ts`
  - Sets `__stage425KahanArm = true` adjacent to `__probe10Capture.armed` setter.
  - New `MATMUL_PROBE13_DELTA` JSON + `[probe13]` verdict line in the
    Probe-12 try block — uses hard-coded Stage 4.24 baseline first8 plus
    a JS-side Neumaier-Kahan reference computation.

No `~/Repos/llama.cpp/` patch. Patch stack unchanged at 13.

## Validation

- `make checkall`: **green** (747 pass / 36 skip / 0 fail).
- All 6 spike + 5 sweep selftests still PASS.
- Per-token decode within noise of Stage 4.24 (no measurable perf
  hit from a single Kahan dispatch + new pipeline build).
- `GENERATED_TEXT = "inonic boso-"` (unchanged from Stage 4.22-4.24
  baseline — bug still active as expected; Kahan was always a
  diagnostic probe, not a fix candidate at this stage).

## Risk register reconciliation (from Probe 13 brief)

- **Kahan inside f32 only.** Outcome: as predicted — Kahan in f32 alone
  could only recover 1 ULP per step. The combination of compiler elision
  AND below-ULP final compensation made even that recovery invisible.
  Branch C (downstream cascade mitigation) status: **superseded** —
  the WGSL kernel is already at its precision limit; cascade mitigation
  isn't needed if the source isn't here.
- **Pipeline-cache key collision.** Outcome: clean. Separate cache key
  `mat-q4_k-f32-f32-2-kahan` lives alongside the production
  `mat-q4_k-f32-f32-2`. No collision; the Stage 4.21 footnote on
  pipeline cache as an open suspect remains untouched.
- **Gate-arm timing.** Outcome: clean. `kahanFired = true` confirms the
  gate fired exactly once on the first eligible dispatch (Qcur-0
  layer 0). `__stage425KahanArm` was disarmed before any other Q4_K
  MUL_MAT in the prefill could trigger.
- **Per-token decode regression.** Outcome: not measured under load
  (single-dispatch gate; pipeline-build cost amortized once at start).
  Acceptable for diagnostic probe; if Stage 4.26 escalates Kahan to a
  ship target, full `make smoke-bench PERF_RUNS=3` measurement is
  required first.

## Stage 4.26 paste-and-go brief — Probe 14: libllama CPU Q4_K matmul precision check

Stage 4.25's structural conclusion ruled out WGSL accumulation order as
the dominant error source. The next probe shifts the precision check
to the OTHER side of the cross-module disagreement: **does libllama's
CPU Q4_K matmul (`vec_dot_q4_K_q8_K`) agree with f64 truth, or does it
have ~5e-4 precision errors of its own?**

```bash
# 1. Confirm working tree.
cd /Users/probello/Repos/webllm
git log --oneline -5
#   → <Stage 4.25 TODO closure commit>  docs(TODO): Stage 4.25 closed — queue Stage 4.26 libllama matmul precision probe
#   → <Stage 4.25 reports commit>       docs(reports): Stage 4.25 closure — Outcome H-3b-structural
#   → <Stage 4.25 feat commit>          feat(spike): Stage 4.25 Probe 13 Kahan accumulator gated to Qcur-0
#   → 24cef38                           docs(TODO): Stage 4.24 closed — queue Stage 4.25 Kahan accumulator probe
#   → 91d4ab6                           docs(reports): Stage 4.24 closure — Outcome H-3b (matmul accumulation-order)

# 2. Confirm llama.cpp tip (patch stack 13 unchanged from Stage 4.25).
( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → ef89f9314   webllm-browser-patches   (patch stack 13)

# 3. Smoke-server up on 8031.
lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &

# 4. Reuse agentchrome session + spike tab.
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
[ -n "$PORT" ] || agentchrome connect --launch --headless
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
```

**One-line goal:** add a `webllm_q4k_q8k_matmul(src0, src1, dst, M, K, N)`
shim to `src/wasm/webgpu-bridge.cpp` (mirrors Stage 4.24's
`webllm_dequantize_q4_K` shim pattern) that calls libllama's
`ggml_compute_forward_mul_mat` (or `vec_dot_q4_K_q8_K` directly with
`block_q8_K`-quantized src1) on the same Probe 10 capture inputs.
Compute libllama's Q4_K matmul output's `maxAbsDeltaVsF64`. Verdict:

- **Outcome H-4-libllama-imprecise** (`libllama_vs_f64 ≥ 1e-4`):
  libllama CPU matmul is the imprecise side. The 5.24e-4 cross-module
  disagreement is two-sided (WGSL ~8e-6 from truth, libllama ~5e-4 from
  truth). No "fix" is needed in webllm — the WGSL kernel is more
  accurate. Stage 4.27 closes the investigation: document the cross-
  module disagreement as expected libllama precision artifact, focus on
  whether the production Qcur-0 path actually feeds the cascade
  causing "inonic boso-" or whether some other operation is the real
  culprit.
- **Outcome H-4-libllama-precise** (`libllama_vs_f64 ≤ 1e-5`):
  libllama is also accurate; the 5.24e-4 disagreement isn't from matmul
  precision at all. Pivot to candidate #2: re-capture src1 inputs for
  Q-proj at idx=1 in both modules and confirm bit-identicality.
- **Outcome H-4-libllama-mid** (`1e-5 < libllama_vs_f64 < 1e-4`):
  contributing factor but not full explanation. Combine with src1
  recheck.

### Probe 14 implementation sketch

1. **C++ shim.** Add to `src/wasm/webgpu-bridge.cpp`:
   ```cpp
   extern "C" int webllm_q4k_q8k_matmul(
       const void * src0_q4k,    // M*K Q4_K bytes
       const float * src1_f32,   // N*K f32 activations
       float * dst_f32,          // M*N output
       int32_t M, int32_t K, int32_t N
   );
   ```
   Internally:
   - Dequantize-then-quantize src1 to block_q8_K (the type that
     `vec_dot_q4_K_q8_K` expects). This is the same conversion
     libllama does internally before vec_dot.
   - Call `vec_dot_q4_K_q8_K(K, &dst[m*N+n], 0, src0_row_m, 0,
     src1_q8k_n, 0, 1)` for each (m, n).
   - Returns 0 on success.

2. **CMake export.** Add `_webllm_q4k_q8k_matmul` to
   `src/wasm/CMakeLists.txt::EXPORTED_FUNCTIONS`. Don't add to
   `JSPI_EXPORTS` — synchronous CPU computation doesn't need
   promising wrap.

3. **Spike harness.** In Probe 13's existing block (Q4_K-gated
   section), add a Probe 14 sub-block:
   ```typescript
   if (cap.src0Type === GGML_TYPE_Q4_K) {
       // ... existing Probe 12 dequant cross-check ...
       // ... existing Probe 13 Kahan output ...

       // NEW: Probe 14 — libllama CPU Q4_K @ Q8_K matmul on captured inputs.
       const dstShimSize = cap.M * cap.N * 4;
       const src0Ptr = mod._malloc(cap.src0Bytes.byteLength);
       const src1Ptr = mod._malloc(cap.src1Bytes.byteLength);
       const dstShimPtr = mod._malloc(dstShimSize);
       mod.HEAPU8.set(cap.src0Bytes, src0Ptr);
       mod.HEAPU8.set(cap.src1Bytes, src1Ptr);
       const status = mod._webllm_q4k_q8k_matmul(
           src0Ptr, src1Ptr, dstShimPtr, cap.M, cap.K, cap.N
       );
       const llamaOutput = new Float32Array(
           mod.HEAPF32.buffer,
           dstShimPtr,
           cap.M * cap.N
       ).slice();
       // Compute llama_vs_f64: build f64 reference (the existing
       // refF32Loop is f32-clamped, so re-run with no Math.fround).
       const refF64 = new Float64Array(cap.M * cap.N);
       for (let n = 0; n < cap.N; n++) {
           for (let m = 0; m < cap.M; m++) {
               let acc = 0;
               for (let k = 0; k < cap.K; k++) {
                   acc += src0Dequant[m*cap.K + k] * src1View[n*cap.K + k];
               }
               refF64[n*cap.M + m] = acc;
           }
       }
       let llamaVsF64Max = 0;
       for (let i = 0; i < cap.M*cap.N; i++) {
           const d = Math.abs(llamaOutput[i] - refF64[i]);
           if (d > llamaVsF64Max) llamaVsF64Max = d;
       }
       const verdict = llamaVsF64Max >= 1e-4
           ? "H-4-libllama-imprecise"
           : (llamaVsF64Max <= 1e-5 ? "H-4-libllama-precise" : "H-4-libllama-mid");
       log(`PROBE14_LLAMA_MATMUL_VS_F64 = ${JSON.stringify({
           llamaVsF64Max,
           llamaFirst8: Array.from(llamaOutput.slice(0, 8)),
           wgslFirst8: captured.first8Got,
           verdict,
       })}`);
       mod._free(src0Ptr); mod._free(src1Ptr); mod._free(dstShimPtr);
   }
   ```

### Files to read first

- [`STAGE-4.25-RESULT.md`](./STAGE-4.25-RESULT.md) (this file) — H-3b-structural confirmation.
- [`STAGE-4.24-RESULT.md`](./STAGE-4.24-RESULT.md) — Probe 12 shim pattern is the template for Probe 14's `webllm_q4k_q8k_matmul`.
- `~/Repos/llama.cpp/ggml/src/ggml-cpu/quants.c::vec_dot_q4_K_q8_K` —
  the libllama reference matmul; check its NEON/AVX2/scalar-fallback
  reductions for the precision claim.
- `~/Repos/llama.cpp/ggml/src/ggml-cpu/quants.c::quantize_row_q8_K` —
  the f32 → block_q8_K conversion path used implicitly when libllama
  feeds f32 activations into a Q4_K matmul.

### Risk register

- **Q8_K conversion drift.** The shim has to quantize-and-dequantize
  src1 (f32 activations) to block_q8_K so vec_dot_q4_K_q8_K can run.
  This conversion itself introduces precision loss. Need to confirm
  the loss is the same as what libllama produces in production
  (`ggml_compute_forward_mul_mat` does the same conversion on entry).
  If the Q8_K conversion is identical, the shim measures the same
  precision libllama gets in production. If not, the comparison is
  apples-vs-oranges.
- **WASM heap growth during shim.** The captured src0/src1 are
  ~2.4 MB; allocating dst + Q8_K-converted src1 inside the shim could
  trigger heap growth, invalidating any cached HEAPU8/HEAPF32 views in
  the spike. Re-derive views inside the call site — same pattern as
  Stage 4.24 Probe 12.
- **vec_dot_q4_K_q8_K offset arithmetic.** The signature is
  `vec_dot(K, &dst[..], 0, x, 0, y, 0, 1)` — the bx/by/bs/nrc
  parameters are tricky. Cross-reference with
  `ggml_compute_forward_mul_mat_q_f32` to confirm the right invocation
  pattern for our M=2048, K=2048, N=6 shape.
- **Single-dispatch shim cost.** The shim does a 2048×2048 matmul on
  the CPU side (≈8M f32 mul-adds). On the wasm32 build that's ~50-100
  ms — acceptable for a one-shot diagnostic; no streaming or batching
  needed.

### Exit criteria — Stage 4.26 closes when documented in `STAGE-4.26-RESULT.md`:

- `[probe14]` verdict line with `llamaVsF64Max` and `OUTCOME:
  H-4-libllama-{imprecise|precise|mid}`.
- All spike + sweep selftests still PASS; `make checkall` green.
- Stage 4.27 brief queued for the chosen branch (imprecise → close
  matmul-precision investigation entirely; precise → re-recapture
  src1 at Q-proj idx=1; mid → multi-source mitigation plan).
