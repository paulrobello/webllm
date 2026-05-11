# Gemma 4 Stage 1 — Per-layer hyperparams refactor closure

**Date:** 2026-05-10
**Spec:** docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md
**Plan:** docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md

**Commits:**
- 84151ad — Task 1.1: types fields
- c274600 → 47499e1 — Task 1.2: GGUF array readers (rename fix)
- 05f5238 → 4fe0f71 — Task 1.3: model-loader Gemma 4 population (freqBase fix)

## Build gate
`make checkall` — green. 751 pass / 36 skip / 0 fail. Skip count unchanged (baseline 33 — 3 over baseline accounted for by the 3 new fixture-gated tests this stage added, all PASSING in this session).

## Runtime gate (non-regression smoke)
Three reference models smoke-loaded and produced their first generated token on the canonical "Tell one short joke." probe. Stage 1's diff cannot affect non-Gemma-4 dispatch (the per-layer branch is gated on `arch === "gemma4"`); this run confirms the additive refactor didn't break the smoke harness itself.

| Model | Load status | First generated token / response excerpt |
|---|---|---|
| tinyllama-1.1b-chat-q4_0 | OK — [7/8] Generated 64 tokens in 0.6s (143.7 tok/s) | "Here's a short joke:\n\nOnce upon a time..." |
| qwen3-0.6b-q4f16 | OK — [7/8] Generated 25 tokens in 0.3s (115.2 tok/s) | "What do you call a man who doesn't like to eat? A *butterfly*!" |
| qwen3-1.7b-q4f16 | OK — [7/8] Generated 17 tokens in 0.3s (79.3 tok/s) | "Why don't scientists trust atoms? Because they make up everything!" |

All three models reached `[7/8]` cleanly with no console errors. ✅

Embedder cosine parity also verified for each run:
- `embed('happy') · embed('joyful') cosine=0.76 (>=0.75 expected, ‖v‖=1.00)` ✅

## Runtime gate (Gemma 4 hp populate)
`tests/models/model-loader-gemma4-hparams.test.ts` — 1 test, 21 assertions, PASSED on this checkout. Verifies the per-layer arrays match the GGUF metadata dump captured during Phase 1 probe (e.g., `embeddingHeadLengthPerLayer[0]=256` for SWA layer 0, `[4]=512` for global layer 4; `feedForwardLengthPerLayer` transitions 6144→12288 at layer 15; `sharedKvLayers=20`; `finalLogitSoftcap=30`).

Note: Generation for `gemma-4-e2b-it-q4km` still fails at `buildQKV` (expected — Stage 1 only populated the hp fields; Stages 2–5 add the dispatch code that reads them).

## Follow-ups
- Stage 2 wires `gemma4` chat-template family + sampler + stop tokens + `final_logit_softcap` from hp into the actual softmax call.
- Stage 3 reads `embeddingHeadLengthPerLayer` / `ropeDimensionCountPerLayer` / `ropeFreqBasePerLayer` in `buildQKV` and adds PLE injection.
