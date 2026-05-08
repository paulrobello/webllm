# Stage 4.32 Result — Outcome P-19-upstream-cascade

**Date:** 2026-05-07
**Status:** CLOSED
**Outcome:** **P-19-upstream-cascade** — Divergence confirmed at `kq-0` (Q×K^T).

## Analysis

### 1. Checkpoint Summary (Prefill, idx 11)

| Name | JSEP AbsMax | Ref AbsMax | Delta |
| :--- | :--- | :--- | :--- |
| `kq-0` | 31.9758 | 52.9305 | **20.9547** |
| `kq_soft_max-0` | 1.0 | 1.0 | 0.0 |
| `kqv_out-0` | 0.0495885 | 0.117065 | **0.0674762** |

### 2. Element-wise Analysis: `kqv_out-0` (n=12288)

- **Max Abs Delta:** 0.117065
- **First Divergent Index:** 64
- **Divergent Indices (>1e-5):** 11849 (96.43%)
- **JSEP Zero Count:** 10752 (87.50%)
- **Longest JSEP Zero Run:** 1792

### 3. Diagnosis

The divergence is massive at the very first attention matmul (`kq-0`, idx 9). While `Qcur-0` and `Kcur-0` showed ~5e-4 deltas (likely due to accumulation order or precision), the resulting `kq-0` divergence is orders of magnitude larger.

The 87.5% zero count in `kqv_out-0` on the JSEP path is the "smoking gun" for the gibberish output. This contiguous block of zeros likely corresponds to indices where the JSEP `kq_soft_max-0` (which is CPU-side but consumes JSEP's `kq-0`) produced zeros due to the corrupted `kq-0` inputs.

### 4. Next Steps (Stage 4.33)

Pivoting investigation upstream to the `kq-0` (Q×K^T) matmul.

- **Probe 20 (Shape A):** Element-wise capture and diff for `kq-0`.
- **Probe 20b (Shape B):** Hash-verification of `Qcur-0` and `Kcur-0` bytes as seen by the `kq-0` kernel (localize if corruption happens during the `Qcur-0` -> `kq-0` transfer or if it's the matmul kernel itself).

Stage 4.33 brief: [STAGE-4.33-BRIEF.md](STAGE-4.33-BRIEF.md)
