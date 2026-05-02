# Prefix-cache interleaved validation — qwen3-8b-iq3m

**Model:** qwen3-8b-iq3m (~1100-token per-NPC distinct persona; 4 NPC × 2 ticks × 2 patterns; round-robin matrix; maxTokens=32; FA on)
**Date:** 2026-05-02
**Verdict:** **PASS** — Pattern B tick-2 wall is -14227 ms / -84.0% faster than Pattern A tick-2.

## Headline (wall-time)

| pattern | tick-1 wall (median) | tick-2 wall (median) | tick-2 prefillMs (median) |
|---|---|---|---|
| A (`chatCompletion(modelId, ...)`) | 15546.6 ms | 16945.5 ms | 15592.4 ms |
| B (`chatCompletion(conv, ...)`)    | 15167.7 ms | 2718.8 ms | 45.4 ms |

## Why this matrix is different

The at-scale probe (`probe-prefix-cache-at-scale-2026-05-01`) ran 4 NPCs × 2 ticks **sequentially per NPC** (`NPC_1 t1 → NPC_1 t2 → NPC_2 t1 → ...`). Pattern A's per-model session tracker preserved each NPC's own KV across its own t1 → t2 boundary, so tick-2 only re-prefilled the divergent tail (~80 tokens). Pattern B paid an extra save+load on top, making it ~12-20% slower regardless of prefix size.

This probe **interleaves** by round-robining all tick-1s first, then all tick-2s, with **per-NPC distinct personas** (~1100 tokens each, NPC id embedded in the first sentence). After NPC_4 tick-1 finishes, the session tracker holds NPC_4's KV. When NPC_1 tick-2 fires, the longest-shared-token-prefix with the cached state is just the small framework intro (~30 tokens). Pattern A is forced to re-prefill the entire NPC_1 persona (~1100 tokens, ~14 s at 12.5 ms/token) on every tick-2 call.

Pattern B reloads the per-conv snapshot for that NPC and prefills only the divergent tail.

## Pattern A per-call detail (round-robin order)

```
  goblin_1     tick=1  prefill= 14395.1ms  wall= 15761.4ms  output="Okay, let's break this down. The"
  wolf_2       tick=1  prefill= 14174.0ms  wall= 15546.6ms  output="Okay, let's see. The scenario is"
  merchant_3   tick=1  prefill= 13579.3ms  wall= 14944.2ms  output="Okay, let's break this down. The"
  guard_4      tick=1  prefill= 13687.1ms  wall= 15057.4ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill= 15592.4ms  wall= 16945.5ms  output="Okay, let's see. The goblin_1 is"
  wolf_2       tick=2  prefill= 14972.9ms  wall= 16373.4ms  output="Okay, let's break this down. Wol"
  merchant_3   tick=2  prefill= 15454.4ms  wall= 16873.0ms  output="Okay, let's see. The user is ask"
  guard_4      tick=2  prefill= 15735.6ms  wall= 17144.1ms  output="Okay, let's think through this. "
```

## Pattern B per-call detail (round-robin order)

```
  goblin_1     tick=1  prefill=    46.4ms  wall= 16036.8ms  output="Okay, let's break this down. The"
  wolf_2       tick=1  prefill=    43.8ms  wall= 15167.7ms  output="Okay, let's see. The scenario is"
  merchant_3   tick=1  prefill=    44.5ms  wall= 15095.1ms  output="Okay, let's see. The scenario is"
  guard_4      tick=1  prefill=    45.6ms  wall= 15041.8ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill=    43.8ms  wall=  2809.0ms  output="Okay, let's think about this. Th"
  wolf_2       tick=2  prefill=    45.6ms  wall=  2710.1ms  output="Okay, let's break this down. Wol"
  merchant_3   tick=2  prefill=    45.4ms  wall=  2718.8ms  output="Okay, let me try to figure this "
  guard_4      tick=2  prefill=    41.8ms  wall=  2691.8ms  output="Okay, let's see. The guard_4 is "
```

## Verdict rationale

**PASS** — Pattern B tick-2 wall 2719 ms vs A's 16946 ms — 84% savings. Per-conv prefix cache delivers in the interleaved regime.

The shape of the win matches the predicted bandwidth-bound floor: Pattern B's tick-2 wall (~2.7 s) is dominated by the post-Phase-1b save+load cost (~1.3 s) plus the small tail prefill (~80 tokens) and decode. Pattern A is forced into a full ~1100-token re-prefill on every cross-NPC tick-2 call (~15 s prefill + ~1.4 s decode = ~16.5 s wall).

## Engine session-tracker bug surfaced and fixed

The first run of this probe (commit `eaba6b0`) produced an anomalous Pattern A `goblin_1 tick-2` row: 2429 ms wall, 1057 ms prefill (vs ~14500 ms for the other three tick-2 calls), with output `"Okay, let's see. The user provid..."` that did not reference goblin. Pattern B's `goblin_1 tick-2` produced `"Okay, let's think about this. Th..."` against the correctly-loaded snapshot.

**Root cause:** `src/core/engine.ts:prepareChatPrompt` had a delta-encoding fast-path that triggered on `promptMessages.length > prevMsgCount` without verifying the cached prompt's leading messages still matched. After `guard_4` tick-1, the session held `messageCount=2`; when `goblin_1` tick-2 fired (`length=4`), the engine took the delta path and prefilled only the new tail on top of `guard_4`'s KV — fast but silently wrong.

**Fix:** added a `cachedMessages: ChatMessage[]` snapshot to `ConversationSession` and a `leadingMessagesMatch(cached, next)` guard that the fast-path condition now requires (additional AND clause). Mismatch falls through to the existing full-reset branch. Regression test in `tests/engine-streaming-api.test.ts` ("session-tracker delta path is skipped when leading messages diverge") drives the bug repro using a realistic fake that tracks `cachedTokenCount` — without the fix, call 3's first forward starts at position 7 (delta path took stale KV); with the fix, position is 0 (full reset).

**Post-fix re-run (this report):** all 4 of Pattern A's tick-2 outputs now reference their NPC by name (`goblin_1`, `Wol[f_2]`, `guard_4`) or honestly summarize (`merchant_3`'s "user is asking"). Prefill costs are uniformly ~14500-15700 ms — the honest re-prefill cost. The PASS verdict holds; the apples-to-apples comparison now shows Pattern B is the only correct option for interleaved multi-conversation workloads, not just the faster one.

**Implication:** conversation-handle mode (`createConversation` + `chatCompletion(conv, ...)`) is required for correctness in any workload that interleaves multiple distinct conversations on the same model. The session-tracker fast-path can no longer silently hand back the wrong KV — but in interleaved use, the fast-path simply doesn't fire, and Pattern A pays the full re-prefill on every cross-conv tick-2. Pattern B is ~6× faster wall-time and produces the correct output.

## Caveats

- **Single-run probe.** No averaging across page loads.
- **prefillMs on the conv path is structurally biased low** (manual mid-prefill happens before `generateTextStream`'s timed window). Wall-time is the source of truth.
- **`?fa=on` is required.** Conversation handles only work in FA mode.
- **Personas are programmatically generated** (paragraph repeated 6×). Each NPC's persona is internally repetitive but distinct from siblings via the embedded id.
