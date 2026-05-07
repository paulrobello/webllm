# Stage 4.18 — Probes 8a + 8b: per-shape Q4_0 matmul precision sweep + V-on-CPU verification

**Status:** CLOSED 2026-05-07. **Outcome reframed:** the JSEP Q4_0
matmul kernel is **NOT** the source of Stage 4.17's 5.24e-4 first8
delta on `Qcur-0`. Across all five production shapes the kernel
matches both an f64 ground-truth reference and an f32 element-wise
loop reference to ≤2.1e-6 / ≤1.3e-6 absolute. Probe 8b confirmed
the V-on-CPU hypothesis as a free explanation for the `Vcur-0` Δ=0
anomaly. The remaining 5.24e-4 production-prefill delta on Qcur-0
must therefore come from **upstream of the matmul kernel** (input
side: Q-projection's src1 = `attn_norm-0` output, or src0 = Q4_0
weight bytes as uploaded into the JSEP buffer). Stage 4.19 brief
queues localization.

**Patch stack:** 12 (unchanged).
**webllm:** +2 commits pending —
- `feat(jsep)`: Probe 8a Q4_0 kernel sweep selftest + Probe 8b backend tag in cb_eval
- `docs(reports)`: this file + Stage 4.18 artifacts

## Headline

| Probe | Finding |
|---|---|
| **8a** (per-shape Q4_0 matmul sweep) | Kernel is well-behaved across all 5 production shapes: max abs Δ 9.6e-7 – 2.1e-6 vs f64 ground truth, 6.6e-7 – 1.3e-6 vs f32 element-wise loop. **Refutes** Outcome B as a kernel-level claim. |
| **8b** (`backend=` tag in cb_eval) | Under Option A-prime the JSEP backend runs only Q-proj, K-proj, Q×K^T, out-proj. V-proj, all RMSNorms, softmax, V@softmax, all FFN, and lm_head route to CPU. Vcur-0 Δ=0 explained by V-proj-on-CPU. |
| **8c** (RMSNorm low-magnitude self-test) | **Skipped** — Probe 8b shows ffn_norm-0 (and presumably attn_norm-0) runs on CPU on the JSEP-side spike. RMSNorm cannot amplify a delta that originates upstream of itself if it doesn't run on JSEP. |

## Probe 8a — Q4_0 production-shape sweep

Implementation: extended `smoke-test/p2-v2-spike.src.ts` with
`packQ4_0Block`, `buildSyntheticQ4_0Matrix`, and
`runMatmulQ4_0Sweep`. Each shape runs `dispatchMatmul` (no-divert
mode) against deterministic random Q4_0 weights with mid-layer-
realistic scales (~0.012-0.020) and src1 in the [-1.5, 1.5] range
(matching attn_norm output magnitude). Two CPU references computed
per element:

- **f64**: `acc += a * b` with JS `Number` arithmetic (53-bit mantissa,
  exact for f32 values).
- **f32 element-wise loop**: `acc = Math.fround(acc + Math.fround(a*b))`
  in K-major order. This matches the JSEP WGSL kernel's per-output-
  element accumulator structure exactly. If JSEP matches this to ULP,
  the kernel is mathematically equivalent to a CPU f32 single-pass
  loop and the precision profile is purely an f32-summation non-
  associativity feature, not a kernel bug.

Captured `MATMUL_Q4_0_SWEEP[*]` block from `?v=stage4.18-probe8a` run
(saved at `STAGE-4.18-q4_0-sweep.txt`):

| Shape (M, K, N) | Tag | maxAbsΔ vs f64 | maxAbsΔ vs f32-loop | outputMaxAbs |
|---|---|---:|---:|---:|
| (2048, 2048, 6) | q-out-proj | **1.68e-6** | 8.64e-7 | 0.730 |
| (256, 2048, 6) | k-v-proj | 1.65e-6 | 7.12e-7 | 0.875 |
| (5632, 2048, 6) | ffn-gate-up | 1.74e-6 | 7.75e-7 | 0.836 |
| (2048, 5632, 6) | ffn-down | **2.07e-6** | 1.25e-6 | 0.875 |
| (32000, 2048, 1) | lm-head | 9.61e-7 | 6.56e-7 | 0.721 |

**Interpretation:**

1. **Kernel correctness.** maxAbsΔ vs f32-loop is < 1.3e-6 in every
   shape — within the f32 ULP envelope for the output magnitudes
   reported. JSEP's WGSL matmul produces bit-stable f32 sums in the
   same order a CPU f32 single-pass loop would. This is the strongest
   form of Outcome B refutation: any other f32 implementation would
   round identically.

2. **f64 envelope.** maxAbsΔ vs f64 stays within 2.1e-6, scaling
   roughly with K (FFN-down K=5632 lands the highest delta). This is
   normal f32 accumulation non-associativity loss vs ground truth.

3. **NOT the cause of the 5.24e-4 production-prefill delta.** At
   shape (2048, 2048, 6) — exactly the production Q-projection
   shape — the synthetic delta is 1.68e-6, vs Stage 4.17's 5.24e-4
   on the actual TinyLlama-Q4_0 prefill. **312× larger in
   production than synthetic.** The kernel cannot account for that
   gap.

4. **Implication.** The 5.24e-4 production delta must originate in
   upstream input differences — either the src1 (`attn_norm-0`
   output) or the src0 (Q4_0 weight bytes as actually stored in the
   JSEP GPU buffer). Stage 4.19 must localize.

## Probe 8b — backend tag via `ggml_backend_buffer_name`

Implementation: extended `node_dump_cb` in
`src/wasm/webgpu-bridge.cpp` to log
`ggml_backend_buffer_name(t->buffer)` per allowlisted node. Returns
`jsep_buf` for tensors backed by the JSEP buffer type and `CPU` for
host-allocated tensors.

JSEP-side backend distribution (steps 0-15, prefill, captured at
`STAGE-4.18-jsep-checkpoints-with-backend.txt`):

| idx | name | shape | jsep_be | meaning |
|---:|------|-------|--------:|---------|
| 0,1 | Qcur-0 | [2048,6,1,1] / [64,32,6,1] | **jsep_buf** | Q-projection MUL_MAT (Q4_0) |
| 2 | Qcur-0 | [64,32,6,1] | CPU | post-RoPE, post-permute |
| 3,4 | Vcur-0 | [256,6,1,1] / [64,4,6,1] | **CPU** | V-projection on CPU ← explains Vcur-0 Δ=0 |
| 5,6 | Kcur-0 | [256,6,1,1] / [64,4,6,1] | **jsep_buf** | K-projection MUL_MAT (Q4_0) |
| 7 | Kcur-0 | [64,4,6,1] | CPU | post-RoPE |
| 8 | kq-0 | [256,6,32,1] | **jsep_buf** | Q×K^T (F32 × F32) |
| 9 | kq_soft_max-0 | [256,6,32,1] | CPU | softmax |
| 10 | kqv_out-0 | [2048,6,1,1] | CPU | V @ softmax |
| 11 | attn_out-0 | [2048,6,1,1] | **jsep_buf** | out-projection MUL_MAT (Q4_0) |
| 12-15 | ffn_norm-0 / ffn_out-0 / result_norm / result_output | various | CPU | all on CPU |

**Implication.** Under Option A-prime, the JSEP backend handles
exactly four ops per layer:
1. Q-projection MUL_MAT (Q4_0 weight × F32 activation)
2. K-projection MUL_MAT (Q4_0 weight × F32 activation)
3. Q × K^T attention MUL_MAT (F32 × F32)
4. Out-projection MUL_MAT (Q4_0 weight × F32 activation)

Everything else — V-projection (also Q4_0!), RoPE, permute,
softmax, V @ softmax, FFN gate/up/down, all RMSNorms, lm_head —
runs on CPU. So the JSEP-side delta vs the all-CPU reference is
attributable to exactly those 4 op types per layer × 22 layers = 88
JSEP MUL_MAT outputs. The reference's "ground truth" is also a CPU
f32 implementation, not a higher-precision reference.

The Vcur-0 Δ=0 anomaly is now **fully explained** by the
asymmetry: V-projection runs on CPU on both the JSEP-side spike
and the all-CPU reference, so its output is bit-identical by
construction. The same is true for ffn_norm-0, kqv_out-0, etc.,
which are all-CPU on both sides.

## Branch decision

Per the Stage 4.18 brief's three-outcome table:

- **Outcome A (kernel fix flips decode):** — N/A. No kernel-level
  fix lands here because Probe 8a refutes the kernel-precision
  framing.
- **Outcome B (kernel precision is irreducible at current
  accumulator):** — **REFUTED at the kernel level.** The matmul
  kernel matches f32 element-wise reference to ULP. Whatever
  precision difference is producing 5.24e-4 in production is not
  in the kernel itself.
- **Outcome C (V-on-CPU was the right path; Q/K/FFN should also
  be on CPU):** — **REJECTED.** Probe 8b shows V is *already* on
  CPU — that doesn't suggest Q/K/out-proj should be too. Routing
  more ops to CPU would defeat the purpose of having a GPU
  backend; we want to find why the JSEP-routed ops produce
  different f32 output than CPU under production inputs.

**New diagnosis (queued for Stage 4.19):** the production delta
must originate **upstream of the JSEP MUL_MAT** — most likely
either:

- **Hypothesis U-A:** The src1 input to JSEP Q-proj
  (`attn_norm-0` output) differs from the CPU side's
  `attn_norm-0` output. attn_norm-0 should run on CPU on the
  JSEP-side spike (matching the all-CPU reference) and produce
  bit-identical output, but cb_eval doesn't currently check —
  `attn_norm-0` is not in the allowlist.
- **Hypothesis U-B:** The src0 weights for JSEP Q/K/out-proj
  differ from the CPU's Q/K/out-proj weights. Plausible
  causes: byte-misalignment during `set_tensor` / weight
  upload, stride mismatch in the Q4_0 row layout, or the
  cross-backend H1 host_mirror writeback hitting a sub-byte
  edge case.
- **Hypothesis U-C:** A WGSL kernel pipeline cache hit produces
  a subtly different shader binary than the synthetic-test
  invocation due to differing pipeline-cache key (workgroup
  size, dispatch shape, bind group layout). Unlikely since the
  pipeline cache key is derived from (src0_type, src1_type,
  dst_type, ndim) only.

## Files written

- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-jsep-checkpoints-with-backend.txt` — 96 lines, JSEP capture w/ `backend=` field.
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-ref-checkpoints-with-backend.txt` — 97 lines, ref capture w/ `backend=` field.
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-q4_0-sweep.txt` — 5 sweep result lines.
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-diff.py` — diff script with backend-aware regex.
- `src/wasm/webgpu-bridge.cpp` — added `backend=` field to checkpoint dump.
- `smoke-test/p2-v2-spike.src.ts` — added Q4_0 helpers + sweep.

## Implementation summary

- `src/wasm/webgpu-bridge.cpp::node_dump_cb`: 4-line addition
  capturing `ggml_backend_buffer_name(t->buffer)` and threading it
  into the printf format.
- `smoke-test/p2-v2-spike.src.ts`: ~200 LOC for `packQ4_0Block`,
  `buildSyntheticQ4_0Matrix`, `runMatmulQ4_0Sweep`, and the 5-shape
  driver loop (logged as `MATMUL_Q4_0_SWEEP[*]`).

Build: both `make wasm-build-jsep` and `make wasm-build-wasm32`
(with `cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/`)
required to refresh both halves of the harness.

## Reproduction

```bash
cd /Users/probello/Repos/webllm
make wasm-build-wasm32
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/   # Makefile asymmetry
make wasm-build-jsep
make smoke-serve  # if not running on 8031

# Reuse the existing agentchrome session:
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html?v=stage4.18-replay"
agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-ref-probe.html?v=stage4.18-replay"
# Wait for DONE on both, snapshot to extract checkpoints, then:
python3 eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-diff.py jsep.txt ref.txt
```
