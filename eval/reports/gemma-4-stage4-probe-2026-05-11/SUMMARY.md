# Gemma 4 Stage 4.0 — Windowed-mask feasibility probe

**Date:** 2026-05-11
**Status:** ✅ PASS by shader inspection — no llama.cpp patch required.
**Outcome:** Stage 4.1 can proceed; the WebGPU softmax + FA kernels accept
arbitrary banded (windowed) causal masks via mask-content alone, with no
shape, dtype, or stride changes.

---

## Question

> Will `ggml-webgpu`'s `opSoftMaxExt` and `opFlashAttn` accept a banded
> windowed causal mask, or does either kernel bake in a causal assumption
> (`-inf` at `k > q`) that would force a llama.cpp patch?

The campaign-Q2 risk register flagged a patch-budget hit if the shader
hard-coded causality. Stage 4.0 resolves the question before any
implementation lands.

## Method

Code inspection of the canonical WGSL sources on the local patched
branch `webllm-browser-patches` of `~/Repos/llama.cpp`:

- `ggml/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl`
- `ggml/src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl`

A runtime synthetic probe was considered but ruled out as redundant: the
kernels' mask handling is straight-line additive code with no branching
on position indices. The shader text is the authoritative answer.

## Findings

### `soft_max.wgsl`

The per-element softmax-input computation runs at two spots in the kernel
body (max-reduction loop and exp-sum loop):

```wgsl
// line 184 (max-reduction pass)
let val = src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col);

// line 211 (exp-sum pass, same construction)
let val = select(src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col),
                 cache[col], col < CACHE_SIZE);
```

Where:

- `mask_val(i)` returns `f32(mask[i])` when `HAS_MASK` is defined, else
  `0.0` (lines 130–138). No element-index branching, no position-based
  zeroing.
- `slope` is the ALiBi multiplier:
  ```wgsl
  // line 174
  let slope = select(1, select(pow(params.m1, ...), pow(params.m0, ...), head < params.n_head_log2),
                     params.max_bias > 0);
  ```
  With `max_bias = 0` (project default — every `opSoftMaxExt` call site
  in `model-inference.ts` passes `0`), `slope = 1.0`, so the mask is
  added verbatim. ALiBi is opt-in and not in scope for Gemma 4.

The mask is indexed at `i_src1_row + col` where `i_src1_row` is built
from the strided 3D coordinates (line 169) with broadcast support
(`i3 % ne13`, `i2 % ne12`). Shape-and-stride pathway is identical to
the current causal mask consumption.

**Conclusion:** the only thing distinguishing a "causal" mask from a
"banded causal" (SWA) mask is the value of `mask_val(i)` at given
positions. Same tensor, same shape, same dtype, same strides — only
the byte content changes.

### `flash_attn.wgsl`

Identical conclusion. The mask-application kernel (`calc_softmax_term`
at line 222–235):

```wgsl
fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32) -> f32 {
    var v = select(FLOAT_MIN,
                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
                   kv_idx < KV_TILE);
#ifdef LOGIT_SOFTCAP
    v = params.logit_softcap * tanh(v);
#endif
#ifdef MASK
    let mask_val = select(0.0, f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
    let mask_term = slope * mask_val;
    v += mask_term;
#endif
    return v;
}
```

`v += mask_term`. No position-driven masking inside the shader. The
mask is loaded into shared memory at the tile boundary (lines 463–472)
exactly as written by the caller — `mask_shmem[elem_idx] = mask[mask_idx]`.

`slope` here is the same ALiBi multiplier (line 299), gated on
`max_bias > 0`. Inactive for Gemma 4.

## Verdict

✅ **No llama.cpp patch needed for Stage 4.1.** Both attention paths
(`opSoftMaxExt` for the manual chain; `opFlashAttn` for the fused path
when shape/dtype gates match) accept a banded windowed causal mask via
mask content alone. Stage 4.1 can introduce per-layer mask selection
without touching shader source.

## Stage 4.1 implementation path (pre-flighted)

1. **Mask-builder helper** (pure JS). Extract the inline mask-fill at
   `model-inference.ts:3514-3532` (inside `uploadLeaves`) into a helper
   `writeCausalMaskF16(view, totalLen, nTokens, pastLen, padCols, swaWindow?)`.
   When `swaWindow` is `undefined` or `Infinity`, behavior is the current
   full-causal mask (zero-delta). When finite, position `(q, k)` is
   visible iff `k ≤ pastLen + q AND k > pastLen + q - swaWindow`.

   A pure-function unit test (Bun) asserting bit-identical output for
   the global case vs current code, plus a hand-traced SWA reference
   for `window=2, nTokens=3, pastLen=0` (3×3 banded pattern), gives
   the Stage 4.1 regression net.

2. **Per-layer mask pair.** Allocate two F16 mask tensors when any
   layer has SWA active (`hp.slidingWindowPattern?.some(b => b)`):
   `globalMask` (full causal) and `swaMask` (banded, `window = hp.swaWindow`).
   Both share the existing `[totalLen, maskPaddedCols]` shape, so the
   FA / softmax binding signatures don't change.

   Single-mask models (TinyLlama, Qwen, Mistral, Llama 3, Phi) skip
   `swaMask` entirely — `globalMask` is the existing tensor, zero
   memory delta. The cost of carrying a second mask tensor for Gemma
   models is `totalLen × maskPaddedCols × 2 bytes` (e.g., 1024 × 32 ×
   2 = 64 KiB) — negligible.

3. **Per-layer dispatch.** Where attention is built (four forward
   methods × `manualAttention` + FA paths), select the mask based on
   `hp.slidingWindowPattern?.[il] === true ? swaMask : globalMask`. For
   models without SWA, the lookup returns the single mask tensor (the
   global one) and behavior is bit-identical.

## Risks retired

- ❎ "Softmax shader has baked-in causal assumption" — falsified.
- ❎ "FA shader rejects non-causal masks" — falsified.
- ❎ "Patch budget +1 for SWA mask" — not triggered; budget unchanged.

## Risks still open (deferred to Stage 4.1+)

- Two mask tensors instead of one ~doubles per-layer mask-alloc cost
  (still tiny; smoke-bench cross-check at Stage 4.4).
- Gemma 2's `swa_period` schema differs from Gemma 4's per-layer
  boolean array — addressed in Stage 4.2.
- Long-context regression (no quality cliff at the 512 boundary) —
  addressed in Stage 4.3 long-context probe.

## Artifacts

- This report.
- Stage 4.1 PR will reference this as the probe closure (single inline
  link in the PR body / `model-inference.ts` mask-builder docstring).
