# ggml Op Catalog for P2-v2

**Date:** 2026-05-05
**Purpose:** Bounded list of ggml ops a `ggml-jsep` backend must implement
to cover the canonical 6 (causal LMs) + encoder + causal-LM-embedder fleets.
Drives Phase 3 sizing.
**Method:** Static analysis of the existing legacy WebGPU graph builders
(known-working on the canonical 6) cross-referenced against
`ggml-webgpu`'s `supports_op` (upper bound). No instrumented decode trace —
deferred to Phase 3 planning when per-model exact sets matter.
**ggml ref:** `webllm-browser-patches @ b54503497`.

## 1. Causal-LM op set (canonical 6)

Source: `src/inference/model-inference.ts:*` — `wasm.op*` invocation grep.
This is the working hand-rolled WebGPU graph that the canonical 6 currently
runs on; its op set is by definition sufficient for those 6 models.

| TS bridge call    | ggml op                  | Notes                                                       |
|-------------------|--------------------------|-------------------------------------------------------------|
| `opAdd`           | `GGML_OP_ADD`            | residuals, biases (Qwen2 attn QKV bias, Phi-3 fused FFN)    |
| `opArgmax`        | `GGML_OP_ARGMAX`         | greedy sampling tail (`engine.ts` greedy path)              |
| `opCont`          | `GGML_OP_CONT`           | post-permute materialization                                |
| `opCpy`           | `GGML_OP_CPY`            | KV-cache writes                                             |
| `opFlashAttn`     | `GGML_OP_FLASH_ATTN_EXT` | gated by FA-shape-routing; not all models engage            |
| `opGetRows`       | `GGML_OP_GET_ROWS`       | embedding lookup, output projection in some setups          |
| `opMul`           | `GGML_OP_MUL`            | RMSNorm scale, residual blending                            |
| `opMulMat`        | `GGML_OP_MUL_MAT`        | dominant op (~33-65% of decode by graph share)              |
| `opPermute`       | `GGML_OP_PERMUTE`        | shape transposes for QKV / FFN gate-up                      |
| `opReshape`       | `GGML_OP_RESHAPE`        | shape coercions (zero-cost in ggml; metadata only)          |
| `opRmsNorm`       | `GGML_OP_RMS_NORM`       | pre-attn / pre-FFN norms                                    |
| `opRope`          | `GGML_OP_ROPE`           | Llama-style RoPE + Qwen3-NEOX variant                       |
| `opSoftMaxExt`    | `GGML_OP_SOFT_MAX`       | attention softmax + sampling tails                          |
| `opSwigluSplit`   | `GGML_OP_GLU`            | SwiGLU FFN (ggml encodes variant in `op_params`)            |
| `opTopK`          | `GGML_OP_TOP_K`          | top-k sampling tail                                         |
| `opView`          | `GGML_OP_VIEW`           | KV-cache slicing, fused-tensor unpacking                    |

**Total: 16 ops.**

## 2. Encoder additions (BERT-style: jina, nomic, bge)

Source: `src/inference/encoder-inference.ts`. Three ops over §1, none of
the §1 ops dropped (encoder uses everything except the causal-LM-only
sampling tail: argmax/topk).

| TS bridge call | ggml op                         | Notes                                |
|----------------|---------------------------------|--------------------------------------|
| `opGelu`       | `GGML_OP_UNARY` (GELU variant)  | jina/bge FFN activation              |
| `opNorm`       | `GGML_OP_NORM`                  | LayerNorm (mean+var) vs RMSNorm-only |
| `opSilu`       | `GGML_OP_UNARY` (SILU variant)  | nomic GeGLU branch                   |

**Total with encoders: 19 ops.**

## 3. Causal-LM-embedder additions (bucket D — qwen3-8b-iq3m self-embed)

Source: `src/inference/causal-embedder-inference.ts`. Strict **subset** of
§1 (no argmax / topk / cpy / flash-attn — embedder runs prefill only,
extracts hidden state, no sampling). No new ops introduced.

## 4. Bounded surface for Phase 3

**19 distinct ggml ops** cover the working fleet today. With encoder dtype
permutations (F32 / F16 weights, Q4_K / IQ3_M / Q4_0 / Q5_K / Q8_0 quants
on `MUL_MAT` `src0`), the Phase 3 TS-kernel surface fits the original
"~20-30 kernels" estimate at the low end.

Per-op estimated TS WGSL kernel count (one kernel per (op, dtype)
combination that actually appears):

| Op                       | Variants                            | TS kernels |
|--------------------------|-------------------------------------|-----------:|
| `MUL_MAT`                | F16, F32, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_M (8 quants exercised) |  8 |
| `MUL`, `ADD`, `SUB`, `DIV` | F32, F16                          |  8 (4×2)   |
| `RMS_NORM`               | F32                                 |  1         |
| `NORM`                   | F32                                 |  1         |
| `ROPE`                   | F32 (Llama / NEOX dispatch in op_params) |  1    |
| `SOFT_MAX`               | F32 (mask F16/F32)                  |  2         |
| `GLU` (SwiGLU)           | F32, F16                            |  2         |
| `UNARY` (GELU, SILU)     | F32                                 |  2         |
| `FLASH_ATTN_EXT`         | F32 Q F16 KV (canonical shape)      |  1         |
| `GET_ROWS`               | F32, F16, Q4_K, Q4_0, Q8_0, IQ3_M, hyb-tier embedders |  ≥6 |
| `CPY`                    | F32→F16, F16→F32, F32→F32, F16→F16 |  4         |
| `CONT`                   | F32, F16                            |  2         |
| `PERMUTE`, `VIEW`, `RESHAPE` | metadata-only — no kernels       |  0         |
| `ARGMAX`                 | F32                                 |  1         |
| `TOP_K`                  | F32                                 |  1         |
| **Total estimate**       |                                     | **40**     |

So the actual kernel count lands at ~40 once dtype permutations are
counted. This is at the high end of the "~20-30" range from the TODO
write-up; not a blocker for Phase 3 — these are independent kernels with
small per-kernel scopes (most are 50-200 lines of WGSL, copy-adaptable
from the existing `model-inference.ts` WGSL strings).

## 5. Upper bound (`ggml-webgpu`)

`ggml-webgpu/ggml-webgpu.cpp:4409-4795` (`ggml_backend_webgpu_device_supports_op`)
declares **106 case statements** across `GGML_OP_*` and `GGML_UNARY_OP_*`.
That is the safety ceiling — anything outside this set has no WebGPU
implementation upstream either, so we can't be asked for more.

Not exercised by the working fleet but listed as supported by `ggml-webgpu`
(potential future-arch ops): `MUL_MAT_ID` (MoE), `SSM_CONV` / `SSM_SCAN`
(Mamba), `GATED_DELTA_NET` (RWKV-7), `CONV_2D`/`IM2COL` (CNN encoders),
`CUMSUM`, `SUM_ROWS`, `ARGSORT`, `L2_NORM` (Qwen3 q/k norm — **likely
needed** if Phase 3 swaps the hand-rolled Qwen3 path for `llama_decode`
since llama.cpp's Qwen3 builder uses `GGML_OP_L2_NORM` natively).

## 6. Open questions / things that may surface at instrumentation time

1. **`L2_NORM` for Qwen3.** `model-inference.ts` open-codes Qwen3's q/k
   normalization using `RMS_NORM` + a constant-`d`-axis trick. llama.cpp's
   canonical Qwen3 builder uses `GGML_OP_L2_NORM`. Switching to
   `llama_decode` (the P2/P2-v2 thesis) means `ggml-jsep` must support
   `L2_NORM` even though the legacy graph doesn't. **+1 kernel, F32.**
2. **`CPY` dtype matrix.** `model-inference.ts` only exercises a subset of
   the F16↔F32 + I32 conversions `ggml-webgpu` supports
   (`ggml-webgpu.cpp:4447-4452`). Phase 3 instrumentation should confirm
   `llama_decode` doesn't introduce I32 paths (KV-cache index ops?).
3. **Bias adds in `MUL_MAT`.** llama.cpp sometimes fuses the bias add into
   `MUL_MAT` via `op->src[2]` (verified in some upstream paths). If our
   `MUL_MAT` kernel doesn't accept a bias src, the scheduler will split
   that into separate `MUL_MAT` + `ADD` nodes — fine for correctness, but
   measure the dispatch-count delta in Phase 3.
4. **`GET_ROWS` dtype permutations for embedders.** Hybrid-quant
   embedders (`token_embd` Q4_K, rest F16) push `GET_ROWS` over Q4_K
   F32-output paths. Already covered by `ggml-webgpu`'s `GET_ROWS` case
   (`ggml-webgpu.cpp:4461-4467`); confirm at instrumentation time.

## 7. Decision

Use this catalog as the **Phase 3 sizing input** (40-kernel TS surface,
small per-kernel scope, copy-adaptable from existing WGSL). Do not block
Phase 2 prototype on it — Phase 2 only needs `MUL_MAT` (one kernel,
matmul-only stub backend; everything else CPU-fallback via the scheduler).

If the Phase 2 prototype gates green and Phase 3 plan-write begins, run
the proper instrumented decode trace (1.3-deferred) to harden this
catalog into a per-model exact set before assigning kernel-implementer
subagents.
