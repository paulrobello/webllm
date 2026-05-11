# Gemma 4 E2B — stop-token audit (NEGATIVE on accuracy, 2026-05-11)

**Outcome:** Audit found a real (cosmetic) engine artifact, but it is
**not** the root cause of the Stage 3 9 % accuracy gap. Gemma 4 emits
`<eos>` (id 1) as its **first response token** on many eval prompts —
the model itself is producing the wrong response, not the engine
stop-token wiring. Engine change reverted; the audit's instrumentation
is preserved here for the next probe.

## Setup

Three-step audit:

1. **Token-ID inventory.** Dumped the unsloth Gemma-4 / Gemma-3N
   GGUF vocab to enumerate the special tokens and the declared
   stop / EOS / BOS:

   | Token         | id  | role                                |
   |---------------|-----|-------------------------------------|
   | `<pad>`       | 0   | padding                             |
   | `<eos>`       | 1   | literal vocab token                 |
   | `<bos>`       | 2   | BOS (matches GGUF `bos_token_id=2`) |
   | `<|turn>`     | 105 | start-of-turn (matches Phase 5)     |
   | `<turn|>`     | 106 | end-of-turn / GGUF `eos_token_id`   |
   | `<end_of_turn>` | NOT IN VOCAB | classical Gemma 2/3 literal |

   GGUF metadata fields:
   - `tokenizer.ggml.bos_token_id = 2`
   - `tokenizer.ggml.eos_token_id = 106`

   Critically: **the GGUF-declared EOS is id 106 (`<turn|>`), not id 1
   (`<eos>`).** The pre-Phase-5 engine code claimed "`<eos>` (id 1)"
   was the declared EOS — that comment is incorrect for the unsloth
   variant.

2. **Engine wiring fix candidate (REVERTED).** Hypothesis: when the
   model emits id 1, the engine's stop set
   `{106}` doesn't match, so the literal `<eos>` string leaks into
   the assistant output and decode runs past the intended end. Patch
   tested: add `addChatStopToken(genConfig, tokenizer, config,
   "<eos>")` alongside the existing `<turn|>`/`<end_of_turn>`
   registration in both engine `generateStream` and
   `chatCompletionWithConversation` paths.

   Effect on 48-task eval at greedy temp 0:
   - Overall 4.33 / 48 = **9 %** (same as baseline, identical
     per-task outputs sampled). Eval ID
     `bench-1778535517951-uct3wy`.

   Effect on the smoke chat (browser console probe via agentchrome):
   - `console.log` confirmed `genConfig.stopTokens = Array(2)` →
     `[106, 1]` after the fix, `eosId = 106` (the declared one).
   - emb-001 probe direct via `engine.chatCompletion(...)` still
     produced `{ done: false, text: "<eos>", tokenId: 1 }` as
     the first chunk, then `{ done: true, finishReason:
     "stop-token", tokenCount: 1, ... }`.

3. **Root cause traced to `generation.ts:296`.** The Generator's
   prefill→first-decode path **unconditionally yields the very
   first sampled token before any stop-token / EOS check**:

   ```ts
   // generation.ts:289-298 — first decode after prefill
   if (waitingForVisibleAnswer && isVisibleTextToken(...)) { ... }
   yield sampledId;          // ← unconditional first yield
   session.pushToken(sampledId);
   recentTokens.push(sampledId);
   let generatedCount = 1;
   // … autoregressive loop starts here, with stop-token checks
   while (!session.shouldStop(sampledId, eosTokenId)) { … }
   ```

   The stop-token check (line 534) only applies inside the
   autoregressive `while` loop, which runs *after* the first yield.
   So the first sample is always emitted to the caller, even if it
   is a stop token.

   When Gemma 4 fails on an eval prompt by emitting `<eos>` (id 1)
   as the very first response token:
   - The engine yields the chunk `{ text: "<eos>", tokenId: 1 }`
     (decoder.push renders id 1 → `"<eos>"`).
   - The loop iterates once more, samples something in
     `stopTokens` (id 106 in our trace), and breaks with
     `finishReason: "stop-token"`, `tokenCount: 1`.
   - The caller sees the assistant text as exactly `"<eos>"`.

   This explains the bare-`<eos>` outputs on emb-001..emb-007 and
   the empty outputs on emb-006/emb-008/emb-012 (where the first
   token must be id 106 = `<turn|>`, which decodes to "").

## What this rules out

Both `(2) stop-token leak audit` candidates from the temp-sweep
closure are now closed:

- Engine `stopTokens` for Gemma 4 is correctly populated. Adding
  `<eos>` (id 1) doesn't change accuracy because the model's
  failure is *emitting `<eos>` as the answer*, not the engine
  failing to stop on it.
- The literal `"<eos>"` artifact in assistant text is a downstream
  cosmetic of the `generation.ts:296` first-token-yield design.
  Fixing it would change the rendered text from `"<eos>"` to `""`
  (empty), but the scoring would still fail — `contains "joyful"`
  doesn't match either string. **Cosmetic fix only; not load-
  bearing for Stage 3.**

## What this confirms (about the model)

Gemma 4 is genuinely producing **`<eos>` as its first response
token** on many eval prompts. Not a sampling artifact, not a
prompt-formatting artifact (the prompt is valid; the same prompt
fragments work in smoke when the user prompt is structurally
simpler). The model itself is broken at the architecture-wiring
level on the eval-task prompt shapes.

This narrows the remaining hypothesis space to two:

1. **`forwardSingle` vs `forwardPrefill` divergence post-shared-KV.**
   The chat-formatted eval prompt (~150-200 tokens system+user) is
   prefilled in one big forward call, then the first decode step
   uses `forwardSingle` with the prefilled KV. If `forwardSingle`
   doesn't honor the shared-KV layout (Phase 4 only verified
   prefill), the first sampled logits are reading from a corrupted
   residual stream — and the model is essentially predicting
   end-of-turn because nothing makes sense.
2. **Real SWA (Stage 4).** All-global fallback may leak
   long-context dependencies on the 5-of-30 SWA layers, especially
   at the chat-prompt scale (system + user + the per-layer PLE
   injections compound).

The cheapest next probe is now hypothesis (1): tap `forwardSingle`
activations at L0/L15/L34 on the first decode step after a chat-
formatted prompt is prefilled, compare to HF reference. The Phase
4 parity infrastructure already exists; the new instrument is just
a per-step capture instead of the per-layer prefill capture.

## Engine follow-up (NOT this campaign)

`generation.ts:296` unconditionally yields the first sampled
token. A defensible cleanup would be to gate that first yield on
the same stop / EOS checks that the autoregressive loop applies
(lines 530-540). This would prevent the literal `<eos>` artifact
in chat output across **all** models that ever emit a stop token
as their first response. It is a separate concern from Gemma 4
and should land in its own commit.

**Not pursuing in this session** because:
- It doesn't move Gemma 4 accuracy off 9 % — the model is the
  problem.
- It risks cross-model regressions (e.g. a Qwen3 thinking-on path
  that intentionally yields a `<think>` token first); needs a
  separate per-model audit + test surface.
- File under "engine UX cleanup" for a future cycle.

## Revert

`src/inference/engine.ts` restored to the pre-audit state (both
`generateStream` and `chatCompletionWithConversation` gemma4
branches register only the existing `<end_of_turn>`/`<turn|>` stop
token). Working tree clean against `HEAD = 4eb08e5 docs(TODO):
Task 3.5 — Stage 3 gate MISSED at 9% (both cheap probes §28-
closed)`.

## Doctrine

Third §28 closure in the Task 3.5 chain. With the stop-token
audit now ruled out, only the heavier-investment probes remain
(`forwardSingle` parity + SWA). Recommend pause for user
direction on whether to invest in Stage 4 SWA or the
`forwardSingle` tap instrument.
