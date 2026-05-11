# Stage 3 — Task 3.3l Phase 5 closure: chat-template tokenization fix

**Date:** 2026-05-11
**Status:** ✅ CORE FIX LANDED — gates met.
**Target:** `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M
**Commits:** *(this report bundles with the feat/test/docs commits)*

## Headline

- Chat smoke went from "only `<eos>` tokens" → **coherent English output
  that cleanly stops on the turn-boundary token**.
- `tokensIn` dropped 75 → 41-46 for `"The capital of France is"` wrapped
  via the chat template (38-45% reduction; the residual is the
  default-system prompt + BOS, both intentional).
- `<|turn>` / `<turn|>` now resolve to single token ids 105 / 106 (vs.
  prior 7-token BPE decomposition each).
- Generation finishes with `finish=stop-token` after 13 tokens on the
  prior repro prompt — exact wiring works through the full chat
  completion pipeline (engine → sampler → streaming decoder).

## Root cause

The unsloth Gemma-4 / Gemma-3N IT GGUFs ship a vocab in which the
turn-boundary tokens are stored with non-standard literal names:

| Token id | Vocab literal | Classical Gemma name |
|---------:|:--------------|:---------------------|
| 105      | `<\|turn>`    | `<start_of_turn>`    |
| 106      | `<turn\|>`    | `<end_of_turn>`      |

The vocab also stores the broader special-token family under
`<\|x>` / `<x\|>` syntax (e.g. `<\|tool_call>`, `<\|think\|>`,
`<\|channel>`). The GGUF chat template uses these literals directly:

```
{{- '<|turn>system\n' -}}   {# or 'user', 'model', 'tool' #}
{{- '<turn|>\n' -}}
{{- '<|turn>model\n' -}}    {# generation prompt #}
```

The project's `formatGemma4` formatter was hard-coded to emit the
classical `<start_of_turn>` / `<end_of_turn>` literals. Since the
vocab has no entries for those classical strings, the SPM tokenizer
fell through to byte-level fragmentation and split each literal into
~7 unrelated BPE pieces. The model therefore received untrained input
on every turn boundary, and decode collapsed to repeated `<eos>`
emission (or earlier-session degenerate `<unused…>` / `_cownt_cownt…`
patterns) regardless of how correct the underlying arithmetic was
(Phase 4 had already verified end-of-stack cosine 0.9722 + top-1
"Paris" match against the HF reference on the **bare** prompt).

## Fix

1. **`src/inference/chat-template.ts:formatGemma4`** — added a
   template-string sniff: when the active chat template contains the
   literal `<\|turn>` substring, emit `<\|turn>` / `<turn\|>` instead
   of `<start_of_turn>` / `<end_of_turn>`. Classical Gemma 2 / 3
   templates (no `<\|turn>` substring) keep their existing behavior —
   zero regression.

2. **`src/core/engine.ts`** (two call sites: `chatCompletion` and
   `chat` paths) — `addChatStopToken` now picks the same literal that
   the template uses. Required because `addChatStopToken` resolves via
   `tokenizer.getId(text)` and silently no-ops if the literal isn't a
   single-token vocab entry; without this, the unsloth variant had no
   explicit stop registration and would only stop via `eosTokenId`
   (which is id 106 = `<turn\|>` — the same token, but registering it
   explicitly is the cleaner contract).

3. **`tests/inference/chat-template-gemma4.test.ts`** — new test
   locking in the unsloth-variant output shape; existing tests
   continue to assert the classical shape so dispatch correctness is
   visible at the test layer.

## Verification

### Test gate

`bun test tests/inference/chat-template-gemma4.test.ts` — 6 pass / 0
fail (was 5; new test for unsloth variant).

### Ship gate

`make checkall` green: 763 pass / 36 skip / 0 fail across 85 test
files, plus fmt / lint / typecheck.

### Browser smoke (greedy, temp=0, max_tokens=24)

| Prompt | Pre-fix output | Post-fix output | `tokensIn` |
|:-------|:---------------|:----------------|-----------:|
| `The capital of France is` | `<eos><eos>…` × N | `Please provide the text you would like me to help you with.` (finish=stop-token, 13 tokens) | 75 → 46 |
| `What is the capital of France?` | (same `<eos>` × N) | `Could you please clarify what you mean by "capital"?…` then conversational repetition | — → 41 |

### Tokenizer probe (browser console)

```
tok.encode("<|turn>")              → [105]               ✓
tok.encode("<turn|>")              → [106]               ✓
tok.encode("<|turn>user\nThe capital of France is<turn|>\n<|turn>model\n")
                                   → 17 tokens           ✓
                                   first 10: [105, 2364, 107, 818, 270, 41626, 270, 1340, 270, 31756]
                                   (105 = <|turn>, 2364 = "user", 107 = "\n", 818 = "The", …)
```

## Gates (per TODO.md Phase 5 spec)

- ✅ **Greedy chat smoke produces coherent English** (not `<eos>` only).
- ✅ **Special-token resolution**: `<\|turn>` / `<turn\|>` encode to
  single ids (105 / 106), matching the GGUF vocab.
- ✅ **Stop-token wiring**: smoke stops on `finish=stop-token` at the
  turn boundary, not via `max-tokens`.
- ✅ **Ship gate**: `make checkall` green.

## Remaining quality observations (Stage 3 follow-on, not a Phase 5
issue)

The model's content quality at greedy temp=0 is conversational and
prone to mid-stream repetition (e.g. "Please tell me what you want to
ask?" emitted twice). Possible factors, none of which are tokenization
bugs:

- **Default-system injection.** `formatChatPrompt` prepends a generic
  "You are a helpful assistant…" system message because Gemma 4's
  template doesn't carry the `enable_thinking` + `<think>` markers
  that `shouldInjectDefaultSystem` keys off. The GGUF's actual
  template only emits a `<\|turn>system\n` block when a real system
  role is present — injecting a default may push the IT model into a
  "wait for further user instruction" mode.
- **Greedy degeneracy on 2B IT models.** Documented behavior for
  small instruction-tuned Gemmas; the bench harness is greedy-by-
  default per the 2026-05-04 policy, so the 36-prompt eval will
  surface absolute accuracy regardless of this anecdotal probe.
- **No SWA yet.** Stage 4 (real sliding-window attention) hasn't
  landed; for prompts < 512 tokens this shouldn't materially affect
  greedy decode, but it removes a potential source of long-context
  drift.

Filing these as Stage 3 closure follow-ups (Task 3.5) — run the
36-prompt eval, then decide whether to suppress the default-system
injection for Gemma 4 specifically.

## Sources

- Vocab probe: `agentchrome js exec` against `window.tokenizer` in the
  smoke tab (probe scripts inlined in commit message).
- GGUF chat-template excerpts: same probe, dumping
  `tokenizer.options.chatTemplate` substrings around `<\|turn>` /
  `<turn\|>` occurrences (6 hits total at offsets 8305, 9276, 9885,
  10574, 16541, 16753).
- Phase 4 end-of-stack parity baseline:
  `eval/reports/parity-gemma-4-e2b-shared-kv-2026-05-11/SUMMARY.md`.
