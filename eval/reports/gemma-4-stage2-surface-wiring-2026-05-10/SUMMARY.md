# Gemma 4 Stage 2 — Surface wiring closure

**Date:** 2026-05-10
**Spec:** docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md
**Plan:** docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md

**Commits:**
- 848333e — Task 2.1: gemma2/gemma3 to ModelArchitecture union (gemma4 added in Task 1.3)
- f779a22 — Task 2.2: formatGemma4 + detectChatTemplate('gemma4')
- ef0bb32 — Task 2.3: stop-token widening to gemma4
- fb45c01 — Task 2.4: GEMMA4_DEFAULTS sampling profile
- ad741db — Task 2.5: final_logit_softcap plumb
- 72ede26 — Task 2.6: BENCHMARK_MODELS registration

## Build gate
`make checkall` — green. 759 pass / 36 skip / 0 fail.

## Runtime gate (adjusted)

The plan's original "5 coherent ASCII tokens" gate is structurally
unachievable at Stage 2 because the `buildQKV` reshape3d assertion
that crashed Phase 1 is gated on Stage 3 Task 3.4 (swap scalar
`hp.embeddingHeadLength` for the per-layer array). Stage 2's diff
adds surface wiring only; the per-layer dispatch in `model-inference.ts`
is untouched.

**Adjusted gate (what was measured):**

| Step | Status |
|---|---|
| [1/8] WebGPU init | OK |
| [2/8] Fetch GGUF | OK — 3106.7 MB in 1.3s (browser cache hit) |
| [3/8] Parse GGUF | OK — arch=gemma4 emb=1536 heads=8/1 layers=35 vocab=262144 ctx=131072 |
| [4/8] Upload weights | OK — loaded in 0.4s |
| [5/8] KV cache | OK — 4096 slots x 35 layers |
| [6/8] Tokenizer | OK — vocab=262144, encode("hello")=[23391] |
| [6/8] Shader warmup | FAIL (non-fatal, continuing) — "unreachable" (same GGML assert site) |
| [7/8] Generation | FAIL — buildQKV reshape3d assertion — expected, Stage 3 fixes |

**Page-title crash banner:** `/Users/probello/Repos/llama.cpp/ggml/src/ggml.c:3648: GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed` — identical assertion to Phase 1 probe crash.

**Observations:**

The page recognized `model=gemma-4-e2b-it-q4km` immediately — `arch=gemma4` printed in
the model descriptor line, confirming the registration wired correctly from eval/models.ts.
Load steps [1/8] through [6/8] Tokenizer completed cleanly with no new failures vs
Stage 1.

The shader-cache warmup at `[6/8]` emitted "unreachable" and continued (non-fatal path)
rather than aborting. This is the same WASM assertion site as the generation failure —
warmup calls into a forward pass that hits `buildQKV` early during shader compilation,
triggering the same shape mismatch that kills generation.

Generation at [7/8] failed with the identical stack as Phase 1:

```
RuntimeError: unreachable
  at wasm-function[1852]:0x16e6fc
  at abort
  at __abort_js
  at wasm-function[662]:0x44d69   <- ggml_assert
  at wasm-function[55]:0x36da
  at wasm-function[96]:0x5e82
  at wasm-function[1831]:0x16e350 <- ggml_reshape_3d
  at GgmlWasm.opReshape3d (webllm-bundle.js:2444)
  at ModelInference.buildQKV (webllm-bundle.js:3020)
  at ModelInference.forwardSingle (webllm-bundle.js:3082)
```

`GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)` — the scalar `hp.embeddingHeadLength`
(256, the SWA head dim) is passed as `ne2` for a global-attention layer where the true
head dim is 512. The element-count product does not match and aborts.

**Console summary (5 messages):**
1. `adapter_info: vendor_id:0 vendor:apple architecture:metal-3` — benign.
2. `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed` (x2, error)
3. `Aborted()` (x2, error)

No new console errors vs Stage 1. The crash site and stack trace are bit-for-bit
identical — Stage 2's surface wiring produced no regressions and no surprises.

**Chat template surface (not directly observable from smoke page log):**
The `arch=gemma4` line in the model descriptor confirms detection is working.
The smoke page does not echo the formatted prompt text, so the
`<start_of_turn>user` / `<end_of_turn>` template output cannot be visually
confirmed from the log — that will be verified via a unit test in Stage 3 or
by enabling prompt-echo in the smoke harness.

## Stage 2 deliverables landed
- `formatGemma4` available + auto-detected for Gemma 4 templates (`detectChatTemplate` returns `"gemma4"`)
- `<end_of_turn>` (token 107) registered as stop token when `tmpl === "gemma4"`
- `GEMMA4_DEFAULTS` sampling profile (`temperature: 1.0, topK: 64, minP: 0.05`) auto-selected when `architecture === "gemma4"`
- `final_logit_softcap = 30.0` plumbed from `hp.finalLogitSoftcap` through the softmax dispatch site in `model-inference.ts`
- `gemma-4-e2b-it-q4km` registered in BENCHMARK_MODELS fleet (eval/models.ts); recognized by smoke harness URL routing

## Follow-ups
- Stage 3 fixes the per-layer head_dim read in `buildQKV` (Task 3.4 swaps scalar `hp.embeddingHeadLength` for `hp.embeddingHeadLengthPerLayer[layerIdx]`), unblocking generation
- Stage 3 also adds PLE injection — combined, Stage 3 produces the canonical "Paris" semantic gate
- Optional: add prompt-echo to the smoke page so Gemma 4 template output is directly visible in the `[6/8]` log block
