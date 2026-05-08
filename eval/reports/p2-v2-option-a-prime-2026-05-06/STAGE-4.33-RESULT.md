# Stage 4.33 Result — Outcome P-20-block-bounded

**Date:** 2026-05-08
**Status:** CLOSED
**Outcome:** **P-20-block-bounded** — JSEP `kq-0` non-zero coverage degrades in a clean stair-step by head index, consistent with a head-axis tile-bounds / GQA-broadcast bug in the WGSL `dispatchMatmul` kernel (`src/inference/jsep/ops/matmul.ts:543`).

## Analysis

### 1. Aggregate-stat side-by-side (prefill, idx 9 = `kq-0`)

| Tensor | Idx | JSEP AbsMax | Ref AbsMax | Delta |
| :--- | :--- | :--- | :--- | :--- |
| `kq-0` | 9 | 31.9758 | 52.9305 | **20.9547** |
| `kq_soft_max-0` | 10 | 1.0 | 1.0 | 0.0 |
| `kqv_out-0` | 11 | 0.0496 | 0.117 | 0.0675 |

Decode steps (idx 27, 45, 63, 81, 99) show the same family of deltas — JSEP `kq-0` AbsMax ranges 14–57 against ref 33–76. Decode-step divergence is *consequence*, not cause; analysis focuses on **prefill idx 9** (the only forward pass where both runs receive the same input).

### 2. Element-wise `kq-0` (n=49152, prefill)

- **Max Abs Delta:** 52.93 (full magnitude of ref signal at some index → JSEP wrote 0 where ref has the full value).
- **First Divergent Index:** 1 (JSEP=-1.178, Ref=-1.169, Δ=0.0094 — small precision-level disagreement at the *very second* element).
- **Total Divergent Indices:** 1151 / 49152 (2.34%).
- **JSEP Zero Count:** 48648 / 49152 (98.97%).
- **Ref Zero Count:** 48000 / 49152 (97.66%).
- **Longest JSEP Zero Run:** 12543.

The 250 (out of 256) zero positions per Q-row in `kq-0` are *expected* (KV cache positions 6+ are unpopulated in prefill — Q × 0 = 0). The **648-element gap between JSEP zeros (48648) and ref zeros (48000)** localizes the structural bug: JSEP zeroed out **648 / 1152 = 56.3% of the active 6×6 attention region**.

### 3. Spatial pattern — stair-step by head, 4-head bands (the smoking gun)

`kq-0` shape is `[256 KV, 6 Q, 32 heads, 1]` (innermost first). Active region per head = 6 K-positions × 6 Q-positions = 36 expected non-zero values. Per-head non-zero count on JSEP:

```
heads  0–3 : 36 / 36   (full)
heads  4–7 : 30 / 36   (missing 6 = 1 K-pos × 6 Q-rows)
heads  8–11: 24 / 36   (missing 12 = 2 K-pos × 6 Q-rows)
heads 12–15: 18 / 36   (missing 18 = 3 K-pos × 6 Q-rows)
heads 16–19: 12 / 36   (missing 24 = 4 K-pos × 6 Q-rows)
heads 20–23:  6 / 36   (missing 30 = 5 K-pos × 6 Q-rows)
heads 24–31:  0 / 36   (missing 36 = entire 6×6 region)
```

REF (CPU non-JSEP) is **36 / 36** for every head — no head is degraded.

Every (head, q) cell on JSEP is short by exactly the band's missing-K count (heads 4–7: 5/6 per cell; heads 8–11: 4/6; …; heads 24–31: 0/6). The pattern is **uniform across Q-position** — per-Q non-zero count is exactly 84 / 192 for *every* Q value. This rules out a Q-axis bug.

### 4. Diagnosis — block-bounded WGSL coverage on the head axis

The dispatch parameters in `dispatchMatmul` (`src/inference/jsep/ops/matmul.ts:657`) are:

```ts
const dispatchX = Math.ceil(M / TILE_M);    // ceil(256 / 16) = 16
const dispatchY = Math.ceil(N / TILE_N);    // ceil(6 / 16) = 1
const dispatchZ = batchCount;                // src1.ne[2] * src1.ne[3] = 32
```

`batchCount` for `kq-0` is computed from src1 (Q) which has 32 heads — so the dispatch grid *nominally* covers all 32 heads. Yet JSEP's output shows heads 24–31 are entirely zero and heads 4–23 are progressively degraded.

**Strongest hypothesis: GQA broadcast mishandling.** TinyLlama is GQA 8:1 (`n_head=32`, `n_head_kv=4`). For `kq = K^T × Q`:

- K (src0): `ne = [head_dim=64, n_kv_pos=256, n_kv_heads=4, 1]`
- Q (src1): `ne = [head_dim=64, n_q_tokens=6, n_q_heads=32, 1]`

The WGSL kernel at `matmul.ts:460-485` indexes `src0` directly with `gid.z` (the dispatched batch index, range 0–31). With src0 batch dim = 4, indexing `src0[batch=4..31] * src0_batch_bytes` reads either out-of-bounds memory or memory adjacent to the K cache — neither of which produces the *observed* clean stair-step.

The 4-head band size (matching `n_kv_heads=4`) and the 6-step degradation count (matching the Q sequence length 6) suggest the kernel is computing dst at **stride positions inside the K cache that overlap incorrectly across heads** — each later head's writes overlap further into the KV cache region than the earlier head's, with the overlap consuming one more K-position per band.

A purely OOB-zero hypothesis would show *all* heads ≥ 4 at zero, not the stair-step. The graceful degradation by band of 4 implicates a **ggml-style stride-based GQA broadcast that the WGSL kernel partially honours but truncates at one of the bounds**, rather than an outright batch-indexing failure.

This is consistent with **P-20-block-bounded** as predicted in the Stage 4.32 brief — Stage 4.34 must inspect `dispatchMatmul`'s pipeline selection / shape uniform packing for the GQA case, and the WGSL kernel's `load_*` row-byte computation when `src0.ne[2] != src1.ne[2]`.

### 5. Reproducibility

```bash
# Fresh ref capture (after rebuild with widened kq-0 IDX-DUMP filter)
cd /Users/probello/Repos/webllm
make smoke-test                              # rebuilds non-JSEP wasm32 with new filter
python3 log_receiver.py &                    # POST receiver on :8032
agentchrome --port "$PORT" navigate "http://localhost:8031/p2-v2-ref-probe.html?v=stage4.33-rebuild" --tab "$REF_TAB"
# Wait for DONE; if fetch CORS failed, dump window.__stderrLines via `js exec`:
agentchrome js exec --tab "$REF_TAB" '(() => ((window).__stderrLines || []).join("\n"))()'
# Save persisted-output JSON's `result` field as STAGE-4.33-ref.txt.

# Diff + spatial pattern:
python3 eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.33-diff.py
python3 eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.33-pattern.py
```

### 6. Next Steps (Stage 4.34)

Investigate the WGSL kqv MUL_MAT kernel's GQA broadcast handling.

- **Probe 21 (Shape A — read the kernel):** Inspect `dispatchMatmul` and the WGSL source it builds (`buildShader` in `src/inference/jsep/ops/matmul.ts`). Check whether `src0_batch_bytes` (passed via `shape.src0_batch_bytes` = `src0.nb[2]`) carries the ggml stride-based GQA broadcast trick (i.e. `nb[2]` set so `batch_idx * nb[2]` wraps modulo `ne[2]`), or whether the kernel needs an explicit divide (`src0_batch = batch / r2` where `r2 = src1.ne[2] / src0.ne[2]`).
- **Probe 21b (Shape B — selftest at kq-0 shape):** Add a JS-side selftest inside `dispatchMatmul` (gated by `__stage434SelftestArm`) that runs the kernel against two known-good f32 inputs at the kq-0 shape (M=256, K=64, N=6, src0_ne[2]=4, src1_ne[2]=32) and compares the output with a host-CPU MUL_MAT. Tells us in isolation whether the kernel-as-built has the GQA bug, independent of any libllama wiring.

Files to read first:
- `src/inference/jsep/ops/matmul.ts:543-740` — `dispatchMatmul`, shape uniform packing, dispatch grid.
- `src/inference/jsep/ops/matmul.ts:267-487` — WGSL kernels (f32, f16, q4_0, q4_K) — the `load_*` functions all use `batch * batch_bytes + m * row_bytes` directly without GQA-aware divide.
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:compute_op` — JSEP backend's MUL_MAT dispatch site. Determines whether ggml is doing the broadcast on the libllama side before handing the descriptor to JSEP, or relying on the kernel to handle it.
- `~/Repos/llama.cpp/ggml/src/ggml.c::ggml_mul_mat` — canonical ggml MUL_MAT semantics including the ne12/ne02 broadcast rule.

Branch on Probe 21 outcome:
- **P-21-stride-trick**: ggml's nb[2] for src0 already encodes the broadcast (e.g. nb[2]=0 or wrap-modulo). Kernel was supposed to honour it but truncates somewhere. Stage 4.35 = identify the truncation site in WGSL.
- **P-21-explicit-divide-needed**: ggml expects the consumer (JSEP) to do the GQA divide. Kernel needs a one-line fix: `let src0_batch: u32 = batch / r2;`. Stage 4.35 = ship the fix + selftest.
- **P-21-other**: The kernel and stride trick are both fine; the bug is upstream in how the descriptor is packed by `ggml-jsep.cpp`. Stage 4.35 = inspect the descriptor on the libllama side.
