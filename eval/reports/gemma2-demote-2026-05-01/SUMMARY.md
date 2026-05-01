# Gemma 2 2B — DEMOTED from `full` set (2026-05-01)

## Outcome

`gemma-2-2b-q4f16` removed from the `SMOKE_PROFILE_SETS.full` array
in `eval/smoke-profiles.ts`. Profile definition (`gemma-2-2b-warm`)
and model registration in `eval/models.ts` retained for future
re-probe. Model is **not** ship-eligible on the current patched
`ggml-webgpu` backend.

This mirrors the §phi-3.5-bucket-D demote pattern: registration
stays, profile stays, the named-set inclusion is what gates the
canonical sweep.

## Symptom

First run of `gemma-2-2b-warm` after adding it to `full` (commit
`22742da`, 2026-05-01) emitted coherent-token-but-incoherent-order
gibberish:

```
User: Tell one short joke.
Assistant: RSSSF suprême suprême estúdio estúdio estúdio estúdio
estúdio estúdio estúdio estúdio estúdio estúdio estúdio estúdio
estúdio estúdio estúdio estúdio estúdio estúdio estúdio estúdio
estúdio estúdio estúdio Estudi EstudiEstudi estúdio estúdio
estúdio estúdio suprême suprême suprême suprême suprême suprême
suprême suprême suprême suprême suprême suprême suprême suprême
suprême suprême suprême suprême suprême suprême suprême suprême
suprême suprême suprême suprême suprême suprême
```

Smoke metadata: 64 tokens / 1.5 s / 54.0 tok/s / `finish=max-tokens`
(model never emitted `<end_of_turn>` — kept looping until the
token cap).

## Diagnosis

The output pattern (real tokens in real languages — Portuguese
*estúdio*, French *suprême* — but in degenerate repeating order)
is the fingerprint of a forward pass that runs to completion with
**broken internal math**, not a load failure or a tokenization
failure.

Verified that the input-side stack is correct:

1. **Architecture is supported in upstream `llama.cpp`**:
   `LLM_ARCH_GEMMA2` is defined in `llama-arch.{cpp,h}`.
2. **Chat template is present in the GGUF** at offset 6028365
   (≈6 MB into the file), correctly contains `<start_of_turn>` /
   `<end_of_turn>` markers; `detectChatTemplate()` returns
   `"gemma"`; `formatGemma()` produces the expected wrapper.
3. **`general.architecture = gemma2`** (not the legacy `gemma`),
   `general.name = Gemma 2 2b It`.
4. **GGUF carries the gemma2-specific tensors**: `blk.N.post_attention_norm.weight`
   and `blk.N.post_ffw_norm.weight` are present in the metadata
   block (Gemma 1 doesn't have these).

So the model loads with the right weights and the right prompt.
The failure is downstream — in the WebGPU compute path.

## Why `ggml-webgpu` likely can't run gemma2 correctly

Gemma 2 introduced four architecture-specific features that
diverge from Gemma 1, all of which the upstream CPU and CUDA
paths handle and any of which could silently break a forward
pass on a backend that omits them:

1. **Alternating sliding-window attention.** Even-indexed layers
   use full attention; odd layers use sliding-window with a
   model-specific window size. A backend that defaults to full
   attention everywhere would over-attend and bias logits.
2. **Post-attention LayerNorm and post-FFW LayerNorm.** Two
   *additional* RMSNorm operations per block, beyond the
   pre-attention and pre-FFW norms used by Gemma 1 / llama.
   `blk.N.post_attention_norm.weight` and
   `blk.N.post_ffw_norm.weight` need to be applied at specific
   points in the residual stream. A backend that maps "all RMSNorm
   tensors" without distinguishing pre/post would either skip them
   or apply them in the wrong place.
3. **Soft-cap on attention and output logits.** Gemma 2 clips
   attention logits to ±50 and final output logits to ±30 via
   `tanh(x/cap) * cap`. Skipping the cap inflates logits far
   beyond the calibration range, biasing the softmax toward the
   highest-logit tokens (consistent with the observed
   single-token-loop behavior).
4. **Tied output ↔ embedding weights.** Gemma 2 reuses the input
   embedding matrix as the output projection (no separate
   `output.weight`). A backend that allocates a zero / random
   `output.weight` because none is provided would produce noise
   logits.

We did not bisect *which* of the four is missing — the diagnosis
is sufficient to demote. Re-enablement would require either
inspecting the `ggml-webgpu` fork's gemma2 coverage explicitly
or running each feature in isolation against a reference, which
is not load-bearing for the project's 8B-ceiling agent + Three.js
use case.

## Decision

1. **Remove `gemma-2-2b-warm` from `SMOKE_PROFILE_SETS.full`**
   (`eval/smoke-profiles.ts:405`).
2. **Retain the profile and the model registration.** A future
   ggml-webgpu update that adds gemma2 coverage would let a
   re-probe flip the inclusion back without reconstructing the
   metadata.
3. **Inline comment at the profile definition** points to this
   report so future readers see the failure mode without grep'ing
   git history.
4. **Don't fix the kernel side.** Patching gemma2-specific kernels
   into the `ggml-webgpu` fork is a multi-day effort against an
   actively-rebasing upstream. The 8B project ceiling already
   covers the project's load-bearing model classes via Llama,
   Mistral, and Qwen3 — gemma is not the gating quant family.

## Re-enablement criteria

Before flipping `gemma-2-2b-warm` back into `full`:

1. Re-run the smoke harness against a fresh build of the patched
   `webllm-browser-patches` branch.
2. Verify the output to "Tell one short joke." is a coherent
   English joke, not a token-loop.
3. If still gibberish: do not flip. Either the four gemma2 quirks
   are still uncovered, or new ones have been introduced.
4. If coherent: also run a quick eval pass (`make bench-browser-eval
   PROFILE=gemma-2-2b-warm`) to confirm the four-dimension scores
   are non-degenerate (>0.4 overall is a reasonable floor for a
   2B model).

## Artifacts

- `smoke-test/models/gemma-2-2b-q4f16.gguf` — 1.6 GB GGUF, retained
  on disk so the demote can be reverted with one profile-array edit
  (no re-download).
- This summary; no per-run logs (the failed run was visible in the
  smoke tab and not POSTed to the dashboard since the smoke
  server was already down by the time the bench harness moved to
  the next profile — see commit history around `32e7a5e`).

## Process notes

This cycle is the 2026-05-01 follow-up to the `full` set expansion
in commit `22742da`. The expansion was correctly motivated (every
registered chat model should be reachable from the canonical
sweep) but didn't validate per-model output quality before
inclusion. The right cadence going forward is:

1. **Add to `full`** only after a one-shot smoke run produces a
   coherent (not necessarily good) reply.
2. **For new architectures** (anything beyond llama / qwen2 / qwen3
   / mistral / phi3 / smollm), run the smoke check explicitly
   before inclusion — these are the cases most likely to expose
   ggml-webgpu coverage gaps.

Other today-added entries (`qwen2.5-coder-1.5b-warm`,
`mistral-7b-v0.3-iq4xs-warm`) use known-good architectures
(qwen2 and mistral respectively) and are unlikely to repro this
failure mode, but a quick smoke verification on the next
`bench-full` cycle would close the audit loop.
