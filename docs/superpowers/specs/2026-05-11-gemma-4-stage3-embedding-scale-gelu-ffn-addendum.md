# Gemma 4 E2B Stage 3 — Embedding scale + GELU FFN addendum (Tasks 3.3f / 3.3g)

**Date:** 2026-05-11
**Status:** Follow-on to `2026-05-10-gemma-4-stage3-correction-no-altup.md` (authoritative no-AltUp scope correction). Surfaced during the 2026-05-11 post-3.3e smoke probe; not present in either the original Stage 3 spec or the no-AltUp correction.

## Why this addendum exists

After Tasks 3.3a–e landed (commits `ba0f90e` through `c4e5659`), the Gemma 4 E2B smoke probe reaches `[8/8]` and decodes 64 tokens, but output is `<unused14><unused11>…<eos>…` garbage rather than coherent text. Stage 3's gates (PLE injection, QK norm, post-attention norm, post-FFW norm, per-layer output scale) all wired correctly. Diagnosis of the residual gap traced two Gemma-family architectural pieces that are universally applied to the Gemma family in upstream llama.cpp but are NOT shared with the Llama/Qwen/Mistral/Phi families that the project's prior fleet covered. Hence they were never wired into the generic forward path.

## Architectural gaps

### Task 3.3f — Embedding scaling by `sqrt(n_embd)`

Reference: `~/Repos/llama.cpp/src/models/gemma4.cpp:149`:

```cpp
inpL = build_inp_embd(model.tok_embd);
// important: do not scale the input embeddings (matches gemma3-text)
inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
cb(inpL, "inp_scaled", -1);
```

After the token-embedding lookup, Gemma family scales the residual stream by `sqrt(embedding_length)` whenever the input is **token IDs** (not pre-computed embeddings). Llama / Qwen / Mistral / Phi do not. The Gemma 4 E2B `embedding_length = 1536`, so the current residual stream is off by `1/sqrt(1536) ≈ 0.0255` — a ~39× attenuation through the rest of the forward pass.

**Project sites to patch** (all in `src/inference/model-inference.ts`):

| Method | Line | Current op |
|---|---|---|
| `forwardSingle` | 1175 | `const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);` |
| `forwardForEmbedding` | 1582 | (same) |
| `forwardAllPositions` | 1894 | (same) |
| `forwardDecode` | 2261 | (same) |

Note: `forwardForEmbedding` also takes token IDs (not pre-computed embeddings), so the scale applies there too. The "do not scale pre-embedding inputs" branch from gemma4.cpp:149 is not currently reachable from any caller — all four methods enter through token IDs — so the unconditional scaled branch is the right one to wire.

**Op:** `wasm.opScale(x, Math.sqrt(hp.embeddingLength))`. Inserted **immediately after** the `opGetRows` call.

### Task 3.3g — FFN activation: GELU not SiLU

Reference: `~/Repos/llama.cpp/src/models/gemma4.cpp:315-320`:

```cpp
cur = build_ffn(cur,
        model.layers[il].ffn_up,   NULL, NULL,
        model.layers[il].ffn_gate, NULL, NULL,
        model.layers[il].ffn_down, NULL, NULL,
        NULL,
        LLM_FFN_GELU, LLM_FFN_PAR, il);
```

`LLM_FFN_GELU + LLM_FFN_PAR` (parallel mode) means the FFN computes `gelu(gate) * up`. The project's current op `wasm.opSwigluSplit(gate, up)` is `silu(gate) * up`. Llama / Qwen / Mistral / Phi all use SwiGLU; Gemma family swaps the gate activation to GELU.

**Project sites to patch** (all in `src/inference/model-inference.ts`):

| Method | Line | Current op |
|---|---|---|
| `forwardSingle` | 1390 | `const ffnHidden = wasm.opSwigluSplit(gate, up);` |
| `forwardForEmbedding` | 1677 | (same) |
| `forwardAllPositions` | 2070 | (same) |
| `forwardDecode` | 2433 | (same) |

**Op replacement (Gemma 4 only):** `wasm.opMul(wasm.opGelu(gate), up)`. Other architectures retain `opSwigluSplit`.

## Predicate choice

Two viable shapes:

1. Per-arch boolean hyperparams (`scaleEmbedding?: boolean`, `ffnActivation?: "silu" | "gelu"`).
2. Direct `hp.architecture === "gemma4"` ternary at the four sites each.

**Selected: option 2** (`hp.architecture === "gemma4"` ternary). Rationale:

- These are arch-family decisions, not tensor-presence decisions. The Gemma 4 codepath in upstream llama.cpp encodes them as hard rules tied to the architecture identifier, not as runtime-configurable flags read from GGUF.
- Adding two boolean hyperparams costs a `types.ts` field, a loader default, and per-call evaluation; the architecture string is already on `hp` and the comparison is free.
- Mirrors precedent at `encoder-inference.ts:387` (`arch === "nomic-bert" ? wasm.opSilu(gate) : wasm.opGelu(gate)`) which selects per-arch activation inline.
- A future Gemma variant lands in the same arch string, so the predicate keeps working without changes.

If a third Gemma-family arch (e.g. a future Gemma 5) lands with a different scale/activation choice, the predicate becomes `hp.architecture === "gemma4" || hp.architecture === "gemma5"` — same single-line change at four sites.

## Gates

- **Static gate:** `make checkall` green (no regression in 762-test suite).
- **Non-regression gate:** `?model=tinyllama-1.1b-chat-q4_0` smoke probe still ≥145 tok/s, embed cosine 0.76. The predicate is `hp.architecture === "gemma4"` so non-Gemma paths are bit-identical (just an extra comparison + branch per forward).
- **Stage 3 closure gate (Task 3.5, after 3.3f + 3.3g land):** `?model=gemma-4-e2b-it-q4km` smoke probe with greedy continuation of `"The capital of France is"` produces `Paris` (or ` Paris`) as the first generated token. Then 36-prompt eval ≥40%.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Embedding scale applied at wrong layer (e.g. after pre-layer norm) | Low | gemma4.cpp:149 is explicit — applied to `inpL` before the layer loop, immediately after `build_inp_embd`. All four project sites have `opGetRows(...)` as the very first op, so insertion point is unambiguous. |
| GELU op cost vs SwiGLU (perf regression) | Low | `opGelu` is already used in PLE injection (line 1088) and the encoder FFN (`encoder-inference.ts:387`). The `opGelu(gate)` + `opMul(.., up)` pair is one node more than fused `opSwigluSplit` but trivially within the +160 graph headroom already allocated for PLE. |
| 3.3f alone produces no qualitative change (because 3.3g is also load-bearing) | Medium | Ship them as separate commits but plan to evaluate together. If 3.3f alone produces ASCII text but not "Paris," that's expected; the semantic gate is the post-3.3g probe. If 3.3f alone produces no change at all, that's also acceptable — ship 3.3g immediately without iterating. |
| `forwardForEmbedding` includes the scale and silently changes embedder output for any future Gemma-family embedder | Low | No Gemma-family embedder is currently registered (bucket D embedders are Llama/Qwen-derived). If a Gemma embedder lands later, the scale is the correct semantic behavior (matches upstream llama.cpp's gemma3-text path); no special-casing needed. |

## Tasks

- **3.3f:** Insert `opScale(x, sqrt(hp.embeddingLength))` after `opGetRows(weights.tokEmb, ...)` at the four sites above, gated on `hp.architecture === "gemma4"`. Commit as `feat(inference): Gemma embedding scaling (Task 3.3f)`.
- **3.3g:** Replace `opSwigluSplit(gate, up)` with `hp.architecture === "gemma4" ? wasm.opMul(wasm.opGelu(gate), up) : wasm.opSwigluSplit(gate, up)` at the four sites above. Commit as `feat(inference): Gemma GELU FFN activation (Task 3.3g)`.
- **3.5:** Closure smoke probe + 36-prompt eval; closure report at `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-11/SUMMARY.md`.

## Post-3.3g diagnosis (surfaced 2026-05-11 after smoke probe)

After 3.3f + 3.3g landed (commits `63c1a6d` + `79dd05d`), the Gemma 4 E2B
smoke probe was re-run and still produced `<unused14><unused11>…<eos>…`
garbage output. Decode 74.2 tok/s, embedder probe still 0.76 cosine
(non-regression OK). Re-diagnosed against `gemma4.cpp` and found two
additional Gemma-family architectural pieces that diverge from the
project's generic causal-LM forward path. Tracking them as 3.3h / 3.3i.

### Task 3.3h — V projection bare RMSNorm

Reference: `~/Repos/llama.cpp/src/models/gemma4.cpp:220-221`:

```cpp
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
```

Gemma 4 applies `ggml_rms_norm(Vcur, eps)` with **no gain** (no
`attn_v_norm` weight tensor — bare RMSNorm) to the V projection before
the attention call. Gemma 3 does **not** do this; it's unique to Gemma 4.
Llama / Qwen / Mistral / Phi all pass V to attention without any norm.

**Project site:** `src/inference/model-inference.ts:929` — inside
`buildQKV`, just before `return { qReady, kReady, v3 }`. After the v3
reshape3d to `[headDim, headCountKv, nTokens]`, apply
`wasm.opRmsNorm(v3, hp.normEpsilon)` when
`hp.architecture === "gemma4"`. The bare-RMS-normed tensor becomes
the new `v3` returned to the caller and flows into the KV cache write.

**Why it matters:** with `v3` un-normed but `qReady` / `kReady` normed,
the attention output `softmax(QK^T) · V` is dimensionally inconsistent:
V values have ~`sqrt(headDim)` magnitude while attention probabilities
have already been computed against normalized Q/K. The V contribution
to the residual stream is then mis-scaled, distorting downstream layers.

### Task 3.3i — Final-logit softcap misuse in flash attention

Reference: `~/Repos/llama.cpp/src/models/gemma4.cpp:11`:

```cpp
hparams.f_attention_scale = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
```

And `gemma4.cpp:379-383` (post-lm-head, NOT inside attention):

```cpp
if (hparams.f_final_logit_softcapping) {
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
}
```

Gemma 4 has **no** attention softcap. The `f_final_logit_softcapping = 30.0`
field is applied to **logits after lm_head matmul**, not inside the
attention computation.

**Current project bug** (`src/inference/model-inference.ts:1350`):

```ts
const attnOut = wasm.opFlashAttn(
    qp, fullK, fullV, maskTensor,
    1.0 / Math.sqrt(headDim),
    0.0, // max_bias
    hp.finalLogitSoftcap ?? 0.0, // ← MISUSE: should always be 0 for Gemma 4
);
```

For Gemma 4 (`finalLogitSoftcap = 30.0` read from GGUF), this means
flash attention is computing `tanh(QK^T / 30) * 30` on every layer.
That tanh-squash crushes all attention scores to a narrow band, making
attention nearly uniform across the context. The residual stream
accumulates noise instead of meaningful information. Very plausible
root cause for the unused-token output.

**Project sites to patch:**

| Method | Line | Fix |
|---|---|---|
| `forwardSingle` (FA branch) | 1350 | Replace `hp.finalLogitSoftcap ?? 0.0` with `0.0`. |
| `forwardForEmbedding` (FA branch) | 2057 | Already `0.0` (no fix needed — confirm). |
| `forwardAllPositions` (FA branch) | 2429 | Already `0.0` (no fix needed — confirm). |
| `forwardDecode` (FA branch) | 3110 | Already `0` (no fix needed — confirm). |

Only the `forwardSingle` site is wrong. The other three already pass
0 — the misuse was localized.

**Post-lm-head softcap (deferred):** the canonical Gemma 4 path applies
`scale(1/s) → tanh → scale(s)` to logits **after** `opMulMat(output_weight, finalNorm)`.
For the Stage 3 closure gate (greedy / temp=0), this is a no-op on the
argmax — `s * tanh(x/s)` is monotonically increasing, so the top-1
token is unchanged whether or not the softcap is applied. Defer to
Stage 4 or a later cycle; not load-bearing for greedy smoke probe.

If/when wired, requires a new `wasm.opTanh` binding (currently absent —
see `src/inference/ggml-wasm.ts`; `_op_tanh` would need to be exported
from `webgpu-bridge.cpp` and added to `EXPORTED_FUNCTIONS` in
`src/wasm/CMakeLists.txt`. **Per CLAUDE.md, do NOT add to JSPI_EXPORTS** —
`ggml_tanh` is non-suspending CPU-side graph build).

### Updated tasks

- **3.3h:** Inside `buildQKV`, apply `wasm.opRmsNorm(v3, hp.normEpsilon)`
  before return when `hp.architecture === "gemma4"`. Commit as
  `feat(inference): Gemma V bare-RMS-norm (Task 3.3h)`.
- **3.3i:** Replace `hp.finalLogitSoftcap ?? 0.0` with literal `0.0` at
  `forwardSingle:1350`. The other three FA sites already pass 0. Commit
  as `fix(inference): drop Gemma final-logit-softcap misuse in FA (Task 3.3i)`.
- **3.5:** Closure smoke probe + 36-prompt eval still pending — gated
  on 3.3h + 3.3i landing. If output is now coherent (any ASCII-readable
  English), 3.5 may also gate on a later post-lm-head softcap fix; for
  now, greedy gate on first token = "Paris" / " Paris".
