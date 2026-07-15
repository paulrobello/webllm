# ENH-004 — Extract `ConversationTurnRunner` from `chatCompletionWithConversation`

> **Status**: proposed · **Effort/risk**: Medium · **Depends on**: ARC-004 (ModelRecord consolidation) MUST land first — same file, overlapping lines. Coordinates with ENH-002 (streaming) — if both are scheduled, do this one first so streaming lands in the runner, not the monolith.

## Goal

`engine.ts::chatCompletionWithConversation` (CC 44, out-degree 47 — the engine's widest bridge per
par-mem betweenness analysis) becomes a thin delegation to a `ConversationTurnRunner` whose named
phases (lock → snapshot-load → prefill → decode → snapshot-save → unlock) can each be read,
tested, and modified in isolation.

## Current state

- `src/core/engine.ts:842-1195` (~350 lines at `eccf6e6`; **re-locate after ARC-004** — anchors
  will have moved) single-handedly does: conversation lock chaining, KV snapshot load/save via
  `ConversationPool` (an articulation point, in-degree 144), delta-prefill, decode loop
  invocation, streaming/stop handling, and stats assembly.
- Mitigations already present: excellent numbered-step comments (use them as the extraction seams,
  exactly like QA-005 uses the `[N/8]` markers) and typed errors.
- KV snapshot rules (vault, load-bearing): reads use `await downloadFromTensor()` (never
  `tensor.getData()` on WebGPU tensors); writes are sync `uploadToTensor()`; no `stackAlloc`
  across `await`. The extraction must move this code verbatim — these patterns encode fixed bugs.

## Implementation steps

1. **Map the seams**: Read the full method; list the numbered steps and the locals each consumes/
   produces. Build a `TurnContext` type from that list (conversation handle, model record, options,
   accumulated tokens/text, timing marks).
2. **Create `src/core/conversation-turn-runner.ts`**: class with constructor-injected dependencies
   (the inference pipeline, `ConversationPool`, tokenizer, chat-template encoder — enumerate from
   what the method actually touches, post-ARC-004 these live on the `ModelRecord`). One public
   `run(ctx): Promise<ChatCompletionResult>` composed of private per-phase methods named after the
   existing step comments.
3. **Move code verbatim, phase by phase** — each phase move is a checkpoint: move one phase's
   block into a private method, delegate from the original, run the conversation test suite
   (`bun test -t "conversation"` — confirm the actual test-name pattern first via
   `grep -rl "chatCompletionWithConversation\|conversation" tests/`), proceed.
4. **Lock semantics unchanged**: the lock-chaining logic (whatever `Promise`-chain or mutex the
   method uses to serialize turns per conversation) stays byte-identical — concurrency bugs here
   are silent corruption. If the chain lives in engine state, the runner receives an
   already-acquired scope rather than owning acquisition; choose whichever keeps the diff smaller
   and note the choice.
5. **`chatCompletionWithConversation` becomes**: build `TurnContext` → `runner.run(ctx)` → map
   result. Public signature unchanged.
6. **No behavior edits**: no reordering of snapshot save vs. stats, no "while I'm here"
   improvements. Streaming hooks (ENH-002) land as a separate change on the runner.

## Files to touch

- `src/core/engine.ts`, `src/core/conversation-turn-runner.ts` (new)
- Possibly `src/core/types.ts` (TurnContext, if not kept file-private — prefer file-private)
- Read-only: `src/core/conversation-pool.ts`, `tests/` conversation suites

## Verification

1. `make checkall` after every phase move (step 3 cadence).
2. Browser: the chat page multi-turn flow + `docs/CHAT_PAGE.md` checklist — specifically a
   multi-turn conversation with page-reload restore (`chat-restore.js` path) to exercise
   snapshot save/load round-trips.
3. par-mem after-check (optional): reindex and confirm `chatCompletionWithConversation`'s
   out-degree dropped substantially (bridge de-risked).

## Rollback

Pure extraction — `git revert` of the extraction commits restores the monolith. No API, wire, or
persistence changes.
