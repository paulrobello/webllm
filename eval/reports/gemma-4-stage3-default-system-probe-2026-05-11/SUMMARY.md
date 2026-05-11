# Gemma 4 E2B — default-system suppression A/B probe (NEGATIVE, 2026-05-11)

**Outcome:** **Hypothesis rejected.** Suppressing the default-system
injection for Gemma 4 leaves the accuracy score unchanged at 9% and
*degrades* the speed-pass smoke output. Suppression path is reverted;
the load-bearing question (why Gemma 4 produces refusal-style
clarifications on direct prompts) remains open.

## Setup

A/B against the baseline captured in
[`../gemma-4-stage3-eval-baseline-2026-05-11/SUMMARY.md`](../gemma-4-stage3-eval-baseline-2026-05-11/SUMMARY.md).

- **A (baseline)** — `chat-template.ts:shouldInjectDefaultSystem`
  unchanged: returns `true` unless the template carries both
  `enable_thinking` and `<think>`. Eval ID
  `bench-1778534531604-16bcgh`.
- **B (probe)** — added two early-return branches: skip injection
  when the template uses `<|turn>` *or* `<start_of_turn>` (Gemma
  family). Eval ID `bench-1778534784389-74023e`.

Both runs: greedy temp 0, 48 tasks, ctx 4096, same WASM tip
(post-Phase-5 `d8a0835`), same dashboard ingest.

## Result

| Metric                          | A (baseline) | B (probe) | Δ      |
|---------------------------------|--------------|-----------|--------|
| Overall                          | 4.33 / 48 (9 %) | 4.33 / 48 (9 %) | **0** |
| instruction-following            | 2.33 / 12 (19.4 %) | 2.33 / 12 (19.4 %) | 0 |
| reasoning                        | 0 / 12 (0 %)   | 0 / 12 (0 %)   | 0 |
| semantic-reasoning               | 0 / 12 (0 %)   | 0 / 12 (0 %)   | 0 |
| tool-calling                     | 2 / 12 (16.7 %) | 2 / 12 (16.7 %) | 0 |

Per-task outputs are **bit-identical** at temp=0 across rs-001,
rs-005, emb-001, if-001 (sampled). Speed-pass smoke (temp=0.6,
prompt `"Tell one short joke."`):

- A: `"Why did the chicken cross the road?"` — clean punchline form.
- B: `"Here are a few jokes for you! Pick your favorite:\n\n**chicken**\n\n**road**\n\n**chicken**..."` — degenerates into single-word repetition.

## Why it was a no-op on the eval path

`eval/tasks/reasoning.ts` and the other dimension files set
`task.systemPrompt = "You are a helpful assistant. Answer questions
directly and concisely."` on every task. `src/evaluation/runner.ts`
forwards `task.systemPrompt` into the `CharacterConfig`, which
constructs the conversation with `messages[0] = { role: "system",
content: task.systemPrompt }`. In `chat-template.ts:formatChatPrompt`
that triggers the `hasSystem = true` short-circuit *before*
`shouldInjectDefaultSystem` is consulted — so the suppression
branch is unreachable from the eval path.

The smoke path (chat-smoke) does *not* supply a system message and
*does* go through the suppression branch. Output changes there
confirm the wiring works; it just doesn't help — the model wanders
into a single-word loop when given no system context, suggesting
small-Gemma-IT greedy-degeneracy rather than a "default-system
poisoning" mode.

## What this rules out

- Default-system message text is not poisoning Gemma 4's responses
  on the eval suite. The eval tasks supply the same string as the
  default; the model performs poorly with or without our path's
  injection because the path is never on.
- The "Please provide..." failure mode is **not** caused by the
  system role appearing in the prompt — it persists when the
  identical content arrives as `task.systemPrompt`.

## What it leaves open

The failure mode is fluent English but the wrong response strategy
("Please provide the question you would like me to answer?" to
`"What is 2 + 2?"`; immediate `<eos>` on emb-001). Hypotheses still
on the table (in rough priority for the next probe):

1. **Greedy-degeneracy on small Gemma-IT variants.** Documented for
   Gemma 2/3 in the wider community; small ITs trained with
   instruction tuning can collapse to repetition or stock
   clarification at temperature 0. **Cheap test:** rerun the eval at
   temp 0.6 (profile-native) — bypasses the 2026-05-04 greedy
   policy for this single model. If accuracy jumps materially,
   document the small-Gemma exception and either ship Stage 3 at
   the higher temp or note temp 0 as a known weakness.
2. **`forwardSingle` vs `forwardPrefill` divergence post-shared-KV.**
   Parity Phase 4 verified `forwardPrefill` against HF at L0/L15/L34
   on `"The capital of France is"`. The eval path calls
   `forwardPrefill` once on the full chat-formatted prompt, then
   loops `forwardSingle` for each generated token. If
   `forwardSingle` has a different bug post-shared-KV (e.g. the
   `n_layer_kv_from_start=15` ref-share isn't wired for decode
   steps), the first token may be correct and the rest garbage —
   consistent with "I'm" → "Please provide..." continuations.
3. **Real SWA (Stage 4).** All-global fallback should be
   approximately right for <512-token generations but may interact
   with the long chat-formatted prompts (system + user can hit
   ~150-200 tokens before generation starts).
4. **Sampler / stop-token boundaries.** Gemma 4 has both `<|turn>`
   (105) and `<turn|>` (106) plus the BOS/EOS pair from the vocab.
   If one of those leaks into the active stop-token set during
   eval but not smoke, the model would emit short fragments
   (`"I'm"`) and stop. emb-001's bare `<eos>` is consistent with
   this.

## Revert

`src/inference/chat-template.ts:shouldInjectDefaultSystem` restored
to the pre-probe single-clause form. Working tree clean against
`HEAD = 2fb5821 docs(report): Task 3.5 baseline ...`.

## Doctrine

This is a §28 negative-result closure per the project rebase doctrine
(CLAUDE.md "Rebase + sweep cycle doctrine"). The lever's
resurrection paths are: rerun if a later architectural fix lands and
the smoke-path degradation reverses. Until then, the suppression
stays out of the codebase.
