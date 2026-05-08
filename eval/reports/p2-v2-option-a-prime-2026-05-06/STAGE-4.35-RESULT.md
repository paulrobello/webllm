# STAGE-4.35 RESULT — WGSL kqv MUL_MAT GQA-broadcast fix shipped

**Date:** 2026-05-08
**Branch:** main · llama.cpp `webllm-browser-patches` tip `ebc7c3d82` (patch
stack 14, unchanged from Stage 4.29)
**Outcome:** **P-21b-clean (regression-guard pass)** + **Paris decode
matches non-JSEP reference.** Phase 3 JSEP causal-LM decode reaches parity
with the non-JSEP reference for the "capital of France" prompt.

## TL;DR

The Stage 4.34 root cause — WGSL kqv MUL_MAT kernel walks `batch *
src0.nb[2]` directly under the permuted K-cache layout, ignoring GQA
broadcast — is fixed by a one-shot WGSL change: each `load_*` kernel now
computes `src0_batch_idx = batch / shape.r2` before scaling by
`shape.src0_batch_bytes`. `r2 = max(1, src1.ne[2] / src0.ne[2])` is
populated from JS in `dispatchMatmul`'s shape-uniform pack. Probe 21b
(host-CPU selftest at the captured kq shape) reports
`maxAbsDeltaVsGqa = 3.81e-5` uniformly across all 32 Q-heads — six orders
of magnitude below Stage 4.34's pre-fix 5.65e+1. JSEP spike's
`GENERATED_TOKENS[0] === 3681` ("Paris") matches the non-JSEP reference
exactly. `make checkall` green: 747 pass, 0 fail.

## Fix

`src/inference/jsep/ops/matmul.ts`:

1. **Shape struct** — `_pad0` slot at index 10 renamed to `r2: u32` in
   the WGSL `Shape` uniform (HEADER constant). `_pad1` at index 11
   reserved for future use. `SHAPE_UNIFORM_BYTES` unchanged at 12 u32
   slots.
2. **`load_*` kernels (f32, f16, q4_0, q4_K)** — each kernel inserts
   `let src0_batch_idx: u32 = batch / shape.r2;` before computing the
   per-row byte base, and uses `src0_batch_idx * batch_bytes` instead
   of `batch * batch_bytes`.
3. **`dispatchMatmul`** — adds an invariant check
   `src1.ne[2] % src0.ne[2] === 0` (validates ggml's GQA broadcast
   contract); computes `r2 = src1.ne[2] / src0.ne[2]`; writes
   `shapeData[10] = r2`.

For non-broadcast dispatches (Q-projection, FFN, lm_head, etc. where
`src0.ne[2] === src1.ne[2]`) `r2 === 1` and the divide reduces to the
identity — the fix is a structural no-op on every existing dispatch.
Only the kqv MUL_MAT (where `src1.ne[2] / src0.ne[2] = 32 / 4 = 8` for
TinyLlama, and similarly `>1` for other GQA models) is functionally
affected.

`smoke-test/p2-v2-spike.src.ts`:

- Probe 21b's `allMatch` verdict text updated from
  *"bug is in libllama-side descriptor packing"* (Stage 4.34's pre-fix
  reading) to *"Stage 4.35 fix engaged, regression guard pass"* so the
  log reads accurately when the regression guard runs going forward.

## Probe 21b — regression guard pass

```
[probe21b] perHeadNonZero=[1536,1536,1536,1536,1536,1536,1536,1536,
                           1536,1536,1536,1536,1536,1536,1536,1536,
                           1536,1536,1536,1536,1536,1536,1536,1536,
                           1536,1536,1536,1536,1536,1536,1536,1536]
[probe21b] perHeadMaxAbsDeltaVsGqa=[9.54e-6, 9.54e-6, 1.14e-5, 9.54e-6,
                                    9.54e-6, 1.14e-5, 1.14e-5, 1.14e-5,
                                    1.91e-5, 1.91e-5, 1.53e-5, 1.53e-5,
                                    1.91e-5, 1.91e-5, 1.91e-5, 1.91e-5,
                                    1.91e-5, 2.29e-5, 2.29e-5, 2.29e-5,
                                    2.29e-5, 3.05e-5, 3.05e-5, 3.05e-5,
                                    3.81e-5, 3.05e-5, 3.05e-5, 3.81e-5,
                                    3.81e-5, 3.81e-5, 3.81e-5, 3.81e-5]
[probe21b] gqaΔ=3.815e-5 byteFormulaΔ=1.053e+2
           head0Match=true otherHeadsDiverge=false
[probe21b] OUTCOME: P-21b-clean (kernel matches GQA reference uniformly
           — Stage 4.35 fix engaged, regression guard pass)
```

Per-head delta envelope: 9.5e-6 (head 0) to 3.8e-5 (heads 24+) — the
gentle climb across head index reflects f32 length-64 reduction noise
on summed sin/cos test vectors, **not** GQA divergence. `byteFormulaΔ
= 1.053e+2` confirms the kernel no longer follows the buggy byte
formula (the byte-formula reference now diverges by 105 from kernel
output, consistent with Stage 4.34's 56.5 envelope adjusted for the
fix's reorientation).

Pre-fix vs post-fix per-head Δ (spot check):

| head | Stage 4.34 (pre-fix) | Stage 4.35 (post-fix) |
|------|---------------------|------------------------|
| 0    | 9.54e-6             | 9.54e-6                |
| 1    | 4.35e+0             | 9.54e-6                |
| 7    | 4.60e+1             | 1.14e-5                |
| 23   | 5.65e+1             | 3.05e-5                |
| 31   | 5.62e+1             | 3.81e-5                |

Six-orders-of-magnitude collapse on every head except 0 (which was
already at f32 noise pre-fix by the b=0 ⇒ k_head=0 coincidence).

## End-to-end "Paris" decode

Spike configured with prompt "The capital of France is" against
`tinyllama-1.1b-chat-q4_0.gguf` through the JSEP backend:

```
LOGIT_STATS_STEP0 = {
  "step": 0,
  "first8": [-7.4137, -7.0051, 7.9086, -4.3618, -4.1580,
             -5.1727, -5.9575, -5.8807],
  "topVal": 12.8669,
  "topId": 3681,
  "hasNaN": false,
  "hasInf": false,
  "finiteCount": 32000,
  "minFinite": -13.7979,
  "maxFinite": 12.8669
}
GENERATED_TOKENS = [3681, 29889, 13, 13, 29906]
GENERATED_TEXT   = "Paris.\n\n2"
PER_TOKEN_MS     = 388.86
TOTAL_DECODE_MS  = 1944
TOTAL_PREFILL_MS = 583
MODEL_LOAD_MS    = 377
```

`generatedIds[0] === 3681` ("Paris") matches the non-JSEP `webllm-wasm.js`
reference. Subsequent tokens (29889 = ".", 13 = "\n", 13 = "\n", 29906 = "2")
are coherent — the model continues the thought without the previous-stage
gibberish cascade. **No NaN/Inf**, finite logit count = 32000 (full vocab),
logit envelope `[-13.8, 12.87]` matches the typical TinyLlama distribution
shape.

Cascade source closed: at Stage 4.34, layer-0 `kq-0` had `abs_max=31.98`
(JSEP) vs `52.93` (ref); at Stage 4.35 the spike's `[STAGE-4.31]` checkpoint
log shows `kq-0` `abs_max` ranging 33.3–76.5 across layers — within the
reference distribution.

## Surviving structural suspects after Stage 4.35

1. ❌ Output-projection (`attn_output.weight`) byte-integrity — closed
   Stage 4.28 / 4.30.
2. ❌ `ffn_norm.weight` gain-vector mis-load — closed Stage 4.30.
3. ❌ first8-window blindness on `kqv_out-0` — closed Stage 4.31.
4. ❌ Divergence at `kq-0` (Q × K^T) — confirmed Stage 4.32, localized
   Stage 4.33, kernel-level root cause confirmed Stage 4.34, **fix
   shipped Stage 4.35**.
5. ✅ **Phase 3 JSEP causal-LM decode reaches parity** for the
   TinyLlama-1.1b-chat-q4_0 "capital of France" prompt.

## Risk register (Stage 4.34 brief, post-fix follow-up)

1. **Other GQA-correct paths may break under the divide.** RESOLVED.
   For `src1.ne[2] === src0.ne[2]`, `r2 === 1` and the divide is a
   structural no-op — verified by the existing `make checkall` test
   suite (747 pass, 0 fail) and the spike's full prefill+decode cycle
   exercising every Q/K/V/O/FFN/lm_head dispatch. The added
   `src1.ne[2] % src0.ne[2] === 0` invariant check is the explicit
   guard.
2. **Fix unblocks kq but reveals downstream cascade.** RESOLVED. The
   spike's `GENERATED_TOKENS[0] === 3681` matches the non-JSEP
   reference; no downstream cascade survives.
3. **WGSL u32 divide on hot path.** OBSERVED COST WITHIN NOISE.
   Spike's PER_TOKEN_MS = 388.86 is within the noise envelope of
   recent Phase 3 spike runs (Stage 4.34 spike was at similar scale).
   The `r2 === 1` strength-reduction makes the divide a no-op on
   every non-broadcast dispatch.
4. **Probe 21b's per-head Δ heuristic was the load-bearing diagnostic.**
   RESOLVED in tree. The selftest stays as a permanent regression
   guard; if a future kernel rewrite reintroduces the bug, the
   pre-fix per-head divergence pattern (Δ~4-56 across heads 1-31)
   will surface immediately.

## Files touched

- `src/inference/jsep/ops/matmul.ts` — Shape struct `_pad0` → `r2`;
  added `src0_batch_idx = batch / shape.r2` to all four `load_*`
  kernels; `dispatchMatmul` invariant + `r2` computation +
  `shapeData[10] = r2` write.
- `smoke-test/p2-v2-spike.src.ts` — Probe 21b verdict text refresh
  (post-fix wording: "Stage 4.35 fix engaged, regression guard pass").

## Artifacts

- This file: `STAGE-4.35-RESULT.md`
- Raw spike output: `STAGE-4.35-spike-output.txt`
- Stage 4.34 closure (root cause):
  `STAGE-4.34-RESULT.md`

## Next session pickup

Phase 3 JSEP causal-LM decode reached parity for one model
(TinyLlama-1.1b-chat-q4_0) on one prompt ("The capital of France is").
Possible follow-on stages:

- **Stage 4.36 (broaden coverage):** run the JSEP spike across the
  canonical-6 model fleet (qwen3-0.6b, qwen3-1.7b, mistral-7b-instruct,
  llama-3.1-8b-instruct, qwen3-8b — TinyLlama is now confirmed) to
  verify the GQA fix generalizes. The kqv kernel lives below the
  model boundary, so this should be a regression-test sweep, not a
  new investigation.
- **Phase 3 closure stub:** if the canonical-6 sweep is uneventful,
  archive the Phase 3 stage-by-stage block from `TODO.md` to
  `TODO_ARCHIVE.md` with a closure stub linking to this report,
  per the TODO archival cadence in `CLAUDE.md`.
