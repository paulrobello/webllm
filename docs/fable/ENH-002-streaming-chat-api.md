# ENH-002 — Streaming Token API Through `chat()`/`chatCompletion` and the Worker Proxy

> **Status**: proposed · **Effort/risk**: Medium · **Depends on**: ARC-004 (engine restructure) recommended first; coordinates with ENH-004

## Goal

Consumers receive visible text incrementally as tokens decode — via an `onToken?: (delta: string) => void`
option on the chat APIs — in both inline and worker modes. Invariant: the concatenation of all
deltas for a request equals the final `text` exactly.

## Current state

- `Generator.generate()` (`src/inference/generation.ts:184`) is an AsyncGenerator yielding token
  IDs; the engine drains it and decodes once at the end, so callers see nothing until completion.
- `src/inference/tokenizer.ts` already ships `StreamingDecoder` (~:954) implementing the
  full-redecode delta pattern (`push(tokenId) → delta`; redecode is ~µs vs the 10-50 ms GPU step —
  effectively free).
- The generation loop runs a steering state machine (think-token suppression, dual stop tokens for
  Qwen3 — 151645 AND 151643 masked during the post-`</think>` window; CLAUDE.md regression
  lesson). **Raw per-token deltas are therefore wrong**: suppressed/think text must not be
  emitted.
- The worker path proxies via typed messages (`src/core/worker-bridge.ts`,
  `src/core/webllm-proxy.ts`) with an existing event/callback mechanism (engine `eventHandlers`)
  — read both before designing the message.
- **Discovery required**: grep `onToken\|stream\|delta` in `src/core/engine.ts`,
  `src/core/types.ts`, `src/inference/generation.ts` — the conversation path may already have
  partial streaming (AUDIT.md ARC-004 mentions "streaming" among `chatCompletionWithConversation`'s
  concerns). If a streaming hook exists, extend it to all chat surfaces rather than adding a
  parallel one.

## Implementation steps

1. **Locate the visible-text accumulator**: in `Generator.generate`, find where post-steering
   *visible* text is assembled (the `hasVisibleAnswerText` / `waitingForVisibleAnswer` region).
   The delta source must be *that* stream, not raw yielded token IDs.
2. **Emit deltas at the generator level**: add an optional `onVisibleDelta?: (delta: string) => void`
   to `Generator.generate`'s existing options/params (Read the signature first). Implement by
   tracking `prevVisibleText` and emitting `visibleText.slice(prevVisibleText.length)` whenever it
   grows — the full-redecode delta pattern. This is naturally correct across suppression windows
   (during suppression the visible text doesn't grow → no delta; if buffered text is flushed
   after `</think>`, one larger delta emits).
3. **Thread through the engine**: add `onToken?: (delta: string) => void` to the chat options type
   in `src/core/types.ts` (optional — additive, non-breaking). In `chatCompletion` /
   `chatCompletionWithConversation`, pass it down as `onVisibleDelta`.
4. **Worker RPC**: in `worker-bridge.ts` add a one-way worker→main message
   `{ type: "token_delta", requestId, delta }`. Worker host side: when the request options carried
   a (stripped, non-cloneable) callback marker, emit the message per delta. Proxy side
   (`webllm-proxy.ts`): before posting the request, replace the function with a `hasOnToken: true`
   marker (functions don't survive postMessage — the abort-signal stripping code in the proxy is
   the pattern to copy); route incoming `token_delta` messages to the stored callback by
   `requestId`; clean up the map on completion/error/abort.
5. **Ordering guarantee**: deltas must stop before the final result resolves. Flush/emit the last
   delta before returning; on abort, stop emitting immediately (no deltas after the
   abort resolution).
6. **Tests** (`tests/generation.test.ts` + a new proxy test):
   - Mock forwardPass emitting a known token sequence including a think-block: assert
     `deltas.join("") === result.text` and that no delta contains `<think>` content.
   - Abort mid-stream: no deltas after abort.
   - Proxy surface test: `onToken` present in worker mode delivers ≥1 delta and joins to the
     final text (can run under Bun with the existing worker test scaffolding if present — check
     `tests/` for existing proxy tests to extend).
7. **Docs**: README chat section gains the `onToken` example; DOC-008's JSDoc mentions it.

## Files to touch

- `src/inference/generation.ts`, `src/core/engine.ts`, `src/core/types.ts`
- `src/core/worker-bridge.ts`, `src/core/webllm-proxy.ts`
- `tests/generation.test.ts` (+ possibly a new `tests/streaming.test.ts`), `README.md`

## Verification

1. `make checkall` — including the new join-invariant tests.
2. Browser: chat page (`docs/CHAT_PAGE.md` checklist) on a qwen3 model — text appears
   progressively, thinking content never flashes, generation stops at end-of-turn (both stop
   tokens honored), stop button works mid-stream.
3. Worker mode in the browser (the smoke page's worker toggle) shows the same progressive output.

## Rollback

The option is additive; revert the commits and the previous drain-then-decode behavior returns.
No wire-format or persisted-state changes are involved.
