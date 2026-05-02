# Prefix-cache interleaved validation — qwen3-8b-iq3m

**Model:** qwen3-8b-iq3m (~1100-token per-NPC distinct persona; 4 NPC × 2 ticks × 2 patterns; round-robin matrix; maxTokens=32; FA on)
**Date:** 2026-05-02
**Verdict:** **PASS** — Pattern B tick-2 wall is -13151 ms / -83.0% faster than Pattern A tick-2.

## Headline (wall-time)

| pattern | tick-1 wall (median) | tick-2 wall (median) | tick-2 prefillMs (median) |
|---|---|---|---|
| A (`chatCompletion(modelId, ...)`) | 15465.2 ms | 15853.5 ms | 14487.7 ms |
| B (`chatCompletion(conv, ...)`)    | 15180.0 ms | 2702.2 ms | 46.2 ms |

## Why this matrix is different

The at-scale probe (`probe-prefix-cache-at-scale-2026-05-01`) ran 4 NPCs × 2 ticks **sequentially per NPC** (`NPC_1 t1 → NPC_1 t2 → NPC_2 t1 → ...`). Pattern A's per-model session tracker preserved each NPC's own KV across its own t1 → t2 boundary, so tick-2 only re-prefilled the divergent tail (~80 tokens). Pattern B paid an extra save+load on top, making it ~12-20% slower regardless of prefix size.

This probe **interleaves** by round-robining all tick-1s first, then all tick-2s, with **per-NPC distinct personas** (~1100 tokens each, NPC id embedded in the first sentence). After NPC_4 tick-1 finishes, the session tracker holds NPC_4's KV. When NPC_1 tick-2 fires, the longest-shared-token-prefix with the cached state is just the small framework intro (~30 tokens). Pattern A is forced to re-prefill the entire NPC_1 persona (~1100 tokens, ~14 s at 12.5 ms/token) on every tick-2 call.

Pattern B reloads the per-conv snapshot for that NPC and prefills only the divergent tail.

## Pattern A per-call detail (round-robin order)

```
  goblin_1     tick=1  prefill= 14449.3ms  wall= 15822.9ms  output="Okay, let's break this down. The"
  wolf_2       tick=1  prefill= 14090.9ms  wall= 15465.2ms  output="Okay, let's break this down. The"
  merchant_3   tick=1  prefill= 13636.3ms  wall= 14992.2ms  output="Okay, let's break this down. The"
  guard_4      tick=1  prefill= 13748.9ms  wall= 15123.0ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill=  1057.1ms  wall=  2428.7ms  output="Okay, let's see. The user provid"
  wolf_2       tick=2  prefill= 14691.2ms  wall= 16057.8ms  output="Okay, let's see. The scenario is"
  merchant_3   tick=2  prefill= 14468.9ms  wall= 15829.8ms  output="Alright, let's see. The user is "
  guard_4      tick=2  prefill= 14487.7ms  wall= 15853.5ms  output="Okay, let's think through this. "
```

## Pattern B per-call detail (round-robin order)

```
  goblin_1     tick=1  prefill=    41.1ms  wall= 15924.7ms  output="Okay, let's break this down. The"
  wolf_2       tick=1  prefill=    41.4ms  wall= 15180.0ms  output="Okay, let's break this down. The"
  merchant_3   tick=1  prefill=    44.5ms  wall= 15035.3ms  output="Okay, let's break this down. The"
  guard_4      tick=1  prefill=    45.8ms  wall= 15158.2ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill=    46.2ms  wall=  2763.9ms  output="Okay, let's think about this. Th"
  wolf_2       tick=2  prefill=    46.2ms  wall=  2702.2ms  output="Okay, let's see. The wolf_2 is a"
  merchant_3   tick=2  prefill=    43.1ms  wall=  2665.1ms  output="Okay, let's think through this. "
  guard_4      tick=2  prefill=    45.6ms  wall=  2673.6ms  output="Okay, so the guard_4 is facing a"
```

## Verdict rationale

**PASS** — Pattern B tick-2 wall 2702 ms vs A's 15854 ms — 83% savings. Per-conv prefix cache delivers in the interleaved regime.

The shape of the win matches the predicted bandwidth-bound floor: Pattern B's tick-2 wall (~2.7 s) is dominated by the post-Phase-1b save+load cost (~1.3 s) plus the small tail prefill (~80 tokens) and decode. Pattern A is forced into a full ~1100-token re-prefill on every cross-NPC tick-2 call (~14.5 s prefill + ~1.4 s decode = ~15.9 s wall).

## Engine session-tracker bug surfaced (Pattern A `goblin_1` tick-2)

Pattern A's `goblin_1` tick-2 row is anomalously fast (2429 ms wall, 1057 ms prefill — close to Pattern B's ~2700 ms) **and produces a corrupt output**. Compare:

- Pattern B `goblin_1` tick-2: `"Okay, let's think about this. Th..."` (correct — reasoning against `goblin_1`'s loaded KV)
- Pattern A `goblin_1` tick-2: `"Okay, let's see. The user provid..."` (no reference to `goblin_1` — the model is reasoning against `guard_4`'s stale KV)

Compare to Pattern B's `wolf_2` tick-2 (`"Okay, let's see. The wolf_2 is a..."`) and `guard_4` tick-2 (`"Okay, so the guard_4 is facing a..."`) — those reference the NPC by name, confirming Pattern B preserves the right context. Pattern A's `goblin_1` does not.

**Root cause:** the engine's session-tracker delta-encoding fast-path at `src/core/engine.ts:934-944`:

```ts
if (
    promptMessages.length > prevMsgCount &&
    prevMsgCount > 0 &&
    session.currentPosition === inf.cachedTokenCount
) {
    const delta = formatChatDelta(promptMessages, prevMsgCount, ...);
    promptTokens = tokenizer.encode(delta);
} else {
    /* full re-encode + KV reset */
}
```

The condition trusts `prevMsgCount` as a continuation signal without verifying that the leading messages actually match the cached state. In the round-robin matrix, after `guard_4` tick-1 the session has `messageCount=2`. When `goblin_1` tick-2 fires (`length=4`), the engine takes the delta path — it formats and prefills **only** the new tail (assistant + user-2) on top of `guard_4`'s KV. Fast but wrong.

Subsequent tick-2 calls (`wolf_2`, `merchant_3`, `guard_4`) all have `length=4` and `prevMsgCount=4`, so the condition fails and they fall through to the full-reset branch — that's why their prefills are honest ~14.5 s.

This means Pattern A's win in the **sequential** at-scale probe was at least partly the same delta-encoding bug masquerading as a prefix cache: each NPC's tick-2 was a 4-vs-2 message-count growth on top of its own tick-1 KV, which happened to be correct *because* tick-1 and tick-2 shared a leading prefix. The bug doesn't manifest in the sequential matrix because the cached KV happens to match the new prompt's leading prefix; it manifests here because the cached KV is from a different NPC.

**Implications:**

1. **Conversation-handle mode (Pattern B) is required for correctness, not just performance**, in any workload that interleaves multiple distinct conversations on the same model. The session-tracker fast-path silently produces wrong outputs.
2. The session-tracker's delta-encoding fast-path needs a guard: verify the leading `prevMsgCount` messages still match the cached prompt before taking the shortcut, or always fall through to full-reset on cross-conv use. Filed as a follow-up.
3. The probe's median calculation (4 samples, take index 2 of sorted) correctly excluded the anomaly, so the headline 83 % win still represents the honest re-prefill case.

## Caveats

- **Single-run probe.** No averaging across page loads.
- **prefillMs on the conv path is structurally biased low** (manual mid-prefill happens before `generateTextStream`'s timed window). Wall-time is the source of truth.
- **`?fa=on` is required.** Conversation handles only work in FA mode.
- **Personas are programmatically generated** (paragraph repeated 6×). Each NPC's persona is internally repetitive but distinct from siblings via the embedded id.
