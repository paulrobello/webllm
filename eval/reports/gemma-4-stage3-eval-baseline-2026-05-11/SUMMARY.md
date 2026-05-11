# Gemma 4 E2B — Stage 3 closure eval baseline (2026-05-11)

**Status:** Baseline captured. Score below Stage 3 gate (≥40%) — opens a
Stage 3-quality probe per the TODO.md pickup plan.

## Headline

- Eval ID: `bench-1778534531604-16bcgh`
- Model: `gemma-4-e2b-it-q4km` (unsloth Q4_K_M; 3.11 GB)
- Profile: `gemma-4-e2b-warm` (newly registered in `eval/smoke-profiles.ts`)
- Eval temperature: 0 (greedy, per the 2026-05-04 cutover policy)
- Context length: 4096
- Total tasks: 48 (full eval suite minus embedding tasks; `capabilities.embedding=false`)
- **Overall score: 4.33 / 48 = 9%** (gate ≥40%)

## Per-dimension breakdown

| Dimension              | Pass / Total | Pct    |
|------------------------|--------------|--------|
| instruction-following  | 2.33 / 12    | 19.4 % |
| reasoning              | 0 / 12       |  0   % |
| semantic-reasoning     | 0 / 12       |  0   % |
| tool-calling           | 2 / 12       | 16.7 % |

Tool-calling is registered as `capabilities.toolCalling = false` for
Gemma 4 (PEG tool format not implemented); the 2 passing tool-calling
prompts are flukes (one-word `"I'm"` outputs that the lenient scorer
accepted).

## Speed pass

PASS. Chat smoke regression on prompt `"Tell one short joke."` produced:

> Why did the chicken cross the road?

Finish reason: `stop-token`. Engine is decoding cleanly; the bug is not
in the engine path.

## Failure-mode taxonomy

Across the 44 non-tool-calling tasks, two dominant output patterns:

1. **Refusal-style clarification request** ("Please provide...", "It
   seems like you are asking about...", "Please clarify what you
   mean?"). Model treats the user turn as incomplete and asks for
   more context instead of answering. Affects most reasoning /
   instruction prompts.
2. **Immediate `<eos>`** (semantic-reasoning emb-001 / emb-002 / emb-
   007 / emb-009 / emb-010 / emb-011). Model emits the stop token
   immediately or after a few empty tokens.
3. **Mild repetition** (rs-006, rs-008, tc-005) — e.g. "what you are
   asking about what you are asking about what you are asking about".

This is **not** the degenerate `<unused...>` / `_cownt_cownt` failure
mode from before Phase 5 — output is fluent English. The model is
producing the wrong *response strategy*, not malformed tokens.

## Hypotheses (priority order, per the TODO pickup plan)

1. **Default-system injection.** `chat-template.ts:451` calls
   `shouldInjectDefaultSystem(template)`, which returns `true` unless
   the template has both `enable_thinking` and `<think>` markers.
   The unsloth Gemma 4 GGUF template has neither, so every user
   turn is preceded by the injected default system message
   (`"You are a helpful assistant. Answer questions directly and
   concisely."`). The Gemma 4 IT model's native template only emits
   a `<|turn>system\n` segment when a real system role is present;
   injecting a default may push the IT model into a "wait for further
   user instruction" mode — consistent with the "Please provide..."
   pattern across reasoning + semantic-reasoning. **This is the
   cheapest first lever and the next step.**
2. **Greedy-degeneracy on small Gemma IT variants.** Documented
   small-Gemma trait at temp=0. Would explain mild repetition but
   not the refusal-style outputs.
3. **Stage 4 SWA gap.** All-global fallback. Each task generates
   <512 tokens so likely not the dominant factor here.

## Parity context (recap from Phase 4 + 5)

- End-of-stack cosine vs HF reference: 0.9722 on
  `"The capital of France is"` (after shared-KV wiring).
- Top-1 argmax: MATCH (id 9079 "Paris").
- Top-16 overlap: 13/16.
- Forward pass produces the right next-token for continuation
  prompts. The bug is downstream of forward correctness — at the
  chat-prompt assembly layer, the sampler, or both.

## Next step

Implement Gemma 4 default-system suppression and re-run the eval as
an A/B against this baseline. Tracked under Task 3.5 (in progress)
in `TODO.md` as a follow-on of Phase 5.

## Artifacts

- Eval row: `eval/reports/smoke-runs.db` evals.eval_id =
  `bench-1778534531604-16bcgh`
- Bench log: `/tmp/gemma4-bench.log` (transient; key lines copied
  here)
- Phase 5 closure (immediate predecessor):
  [`eval/reports/gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md`](../gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md)
