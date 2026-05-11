# Gemma 4 E2B — temperature sweep probe (NEGATIVE, 2026-05-11)

**Outcome:** **Hypothesis rejected.** Raising the accuracy-pass
temperature from greedy 0 to profile-native 0.6 does **not** lift
Gemma 4 out of the refusal / `<eos>` failure mode. Per-task outputs
follow the same pattern; the failure is not sampling-noise.

## Setup

Re-run `bun run eval/bench.ts --profiles gemma-4-e2b-warm
--eval-temperature 0.6` against the same WASM tip (post-Phase 5,
`d8a0835`), same model `gemma-4-e2b-it-q4km`, same dashboard.

Tool-calling tasks are excluded automatically at any temperature
above the 0.2 cold-temperature ceiling (`eval/browser-eval.ts`), so
the comparison is on the 36 non-tool-calling tasks.

## Result

| Metric              | Greedy 0 (baseline) | Temp 0.6 (probe) |
|---------------------|---------------------|------------------|
| Non-tool-calling     | 2.33 / 36 ≈ **6.5 %**  | 2.33 / 36 ≈ **6.5 %** |
| instruction-following | 2.33 / 12 = 19.4 %  | 2.33 / 12 = 19.4 % |
| reasoning           | 0 / 12 = 0 %        | 0 / 12 = 0 % |
| semantic-reasoning  | 0 / 12 = 0 %        | 0 / 12 = 0 % |

Sample outputs at temp 0.6 (compare to baseline):

| Task    | Baseline (greedy) | Probe (temp 0.6) |
|---------|-------------------|-------------------|
| rs-001  | `"Please provide the question you would like me to answer?"` | `"Please provide the question you would like me to answer?"` |
| rs-005  | `"I am a large language."` | `"I am a large language."` |
| emb-001 | `<eos>` | `<empty>` |

The **same exact responses** appear at both temperatures on
sampled tasks. The 2.33 instruction-following score is identical
across both runs — strong signal the failure is deterministic
relative to the prompt, not stochastic.

## What this rules out

- Small-Gemma-IT greedy-degeneracy is not the dominant cause.
- The 2026-05-04 greedy-by-default policy is not artificially
  suppressing Gemma 4's score.

## What it leaves open

Same three deeper hypotheses from the default-system probe SUMMARY:

1. **`forwardSingle` vs `forwardPrefill` divergence post-shared-KV.**
   Parity Phase 4 verified `forwardPrefill` on
   `"The capital of France is"` (cosine 0.9722, top-1 MATCH). The
   eval path:
   - prefills the chat-formatted prompt once
   - decodes 1 token at a time via `forwardSingle` thereafter

   If `forwardSingle` doesn't honor the
   `n_layer_kv_from_start=15` ref-share at decode time, the
   first generated token can be correct (consistent with the
   sparse `"I'm"` outputs on tool-calling tasks) and every
   subsequent token reads garbage from a misaligned KV slot.
   The "Please provide..." / fluent-but-irrelevant outputs are
   compatible with this: the model is decoding *somewhere*, just
   not against the activation it just emitted.

2. **Real SWA (Stage 4).** All-global fallback. Chat-formatted
   prompts run 150-200 tokens system+user and the longer eval
   outputs push past 256-512 tokens generated. The 5-of-30 SWA
   layers may need the windowed mask to keep local-attention
   layers from leaking across turn boundaries.

3. **Stop-token leak / sampler boundary audit.** The bare-`<eos>`
   outputs on emb-001..emb-002 + several near-empty outputs in
   the probe imply something is emitting a stop token on the
   first decode step. Both `<|turn>` (105) and `<turn|>` (106)
   need to be on the active stop-set without bleeding into
   first-token candidates.

The cheapest next probe is **(3) stop-token audit** — pure
inspection + 1 smoke-test cycle. (1) forward-single parity needs
a new tap + a fresh capture-server cycle. (2) SWA is the largest
investment.

## Doctrine

This is the second §28 closure in the Task 3.5 chain. The
suppression and temperature probes both closed identically; the
project's "cheap levers first" approach has exhausted what it can
probe without an architectural intervention or a new instrument.

Recommend surfacing to the user before opening probe (1) or (3):
the next move is a planning decision (which probe / which
sequence), not a one-line fix.
