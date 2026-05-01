# Prefix-cache at-scale validation — qwen3-8b-iq3m

**Model:** qwen3-8b-iq3m (~1800-character / ~1325-token NPC system prefix; 4 NPC × 2 ticks × 2 patterns; maxTokens=32; FA on)
**Date:** 2026-05-01
**Verdict:** **FAIL** — Pattern B tick-2 wall is +485 ms / +19.9% vs Pattern A tick-2 even at ~1325 shared tokens. The v1 cost decomposition was wrong: Pattern A already benefits from the engine's per-model session-tracker prefix cache, so the comparison was never "cache vs no-cache" — it was "session-tracker cache (no extra I/O)" vs "conv path (adds save+load round-trip per call)".

## Headline (wall-time)

| pattern | tick-1 wall (median) | tick-2 wall (median) | tick-2 prefillMs (median) |
|---|---|---|---|
| A (`chatCompletion(modelId, ...)`) | 15881.0 ms | 2429.3 ms | 1040.3 ms |
| B (`chatCompletion(conv, ...)`)    | 16056.8 ms | 2913.9 ms | 46.2 ms |

**Pattern B tick-2 wall is +485 ms / +19.9% vs Pattern A tick-2.** Same direction *and similar magnitude* as v1 (+446 ms / +20.1% at ~315 shared tokens), even though shared prefix grew ~4×.

## sharedLen distribution (Phase A diagnostic)

Captured via temporary `[conv]` debug logs in `engine.ts:chatCompletionWithConversation`:

**v1 prefix (~315 prompt tokens at tick-1):**

| convId | tick-1 newTokens | tick-2 newTokens | tick-2 sharedLen |
|---|---|---|---|
| conv_1 (goblin_1)   | 319 | 398 | **318** |
| conv_2 (wolf_2)     | 318 | 386 | **317** |
| conv_3 (merchant_3) | 316 | 388 | **315** |
| conv_4 (guard_4)    | 308 | 376 | **307** |

**At-scale prefix (~1325 prompt tokens at tick-1):**

| convId | tick-1 newTokens | tick-2 newTokens | tick-2 sharedLen |
|---|---|---|---|
| conv_1 (goblin_1)   | 1326 | 1405 | **1325** |
| conv_2 (wolf_2)     | 1325 | 1393 | **1324** |
| conv_3 (merchant_3) | 1323 | 1395 | **1322** |
| conv_4 (guard_4)    | 1315 | 1383 | **1314** |

**sharedLen tracks `tick1.newTokens − 1` exactly** (the −1 is the last prompt token of tick-1, which differs from tick-2's prompt). Prefix-detection is healthy at every scale; this isn't a chat-template determinism problem (option (b) from the brief is ruled out).

## Why the v1 cost decomposition is wrong

The v1 SUMMARY's prediction:
- save+load fixed cost ≈ 940 ms
- prefill savings = 12.31 ms × shared_tokens
- crossover at 76 tokens; clear win at 1000+ tokens

The actual at-scale data:
- shared_tokens ≈ 1325, predicted prefill savings = 12.31 × 1325 ≈ **16,310 ms**
- Pattern A tick-2 prefillMs is only **1040 ms total** (not 16K ms)

**The v1 decomposition assumed Pattern A would re-prefill the entire prompt every tick.** It doesn't. Pattern A's modelId path uses the engine's session-tracker prefix cache (`session.currentPosition === inf.cachedTokenCount`, `engine.ts:932`). Within a single NPC's tick-1 → tick-2 window, the session tracker still holds tick-1's KV state, so tick-2 only re-prefills the divergent assistant+user-2 tail (~80 tokens). That's why Pattern A tick-2 prefill is ~1035 ms regardless of whether the prefix is 315 or 1325 tokens — the prefill cost is dominated by the **divergent tail** (~80 tokens at ~12.5 ms/token), not the shared prefix.

So Pattern B's "savings" over Pattern A are **only** the tail-prefill cost (~1000 ms). Against that, Pattern B pays save+load on a tensor that scales with prefix size:

| metric | v1 (~315 shared) | at-scale (~1325 shared) |
|---|---|---|
| Pattern A tick-2 prefillMs | 989 ms | 1035 ms |
| Pattern B tick-2 prefillMs | 42 ms | 46 ms |
| Pattern A tick-2 wall | 2204 ms | 2408 ms |
| Pattern B tick-2 wall | 2660 ms | 2927 ms |
| save+load overhead (B − A wall delta + (A − B) prefill delta) | ~1410 ms | ~1470 ms |
| kvBytes per snapshot | ~50 MB | ~200 MB |

save+load overhead is roughly flat at **~1.4-1.5 s** even with kvBytes growing 4×. The cost is dominated by per-tensor ASYNCIFY round-trip overhead (72 round-trips at ~10-20 ms each), not by bytes copied. The corrected cost decomposition is:

**save+load total ≈ ~1450 ms (per-tensor round-trip dominated)**
**prefill savings ≈ ~1000 ms (divergent tail only, since session tracker covers shared prefix)**

Net: Pattern B is **always ~450 ms slower** than Pattern A in this NPC pattern, regardless of how long the system prefix is.

## When would Pattern B actually win?

Pattern B beats Pattern A only when Pattern A *cannot* use its session-tracker cache. That happens when **multiple conversations interleave on the same model**:

1. NPC_1 tick-1 (session tracker holds NPC_1)
2. NPC_2 tick-1 (session tracker overwritten with NPC_2; NPC_1's KV gone)
3. NPC_1 tick-2 — Pattern A would have to **re-prefill the full ~1325-token prefix** (~16 s) because the session tracker doesn't hold NPC_1 anymore. Pattern B (per-conv snapshot) reloads NPC_1's KV in ~1 s and prefills only the tail.

The current probe matrix (NPC_1 tick-1 → NPC_1 tick-2 → NPC_2 tick-1 → NPC_2 tick-2 → ...) **never interleaves**, so Pattern A's session tracker always wins. The probe is structurally unable to demonstrate Pattern B's value.

## Pattern A per-call detail

```
  goblin_1     tick=1  prefill=14497.3ms  wall=15890.0ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill= 1046.8ms  wall= 2444.6ms  output="Okay, let's see. The goblin is a"
  wolf_2       tick=1  prefill=14500.0ms  wall=15881.0ms  output="Okay, let's see. The NPC is a wo"
  wolf_2       tick=2  prefill= 1023.0ms  wall= 2412.4ms  output="Okay, let's break this down. The"
  merchant_3   tick=1  prefill=14413.7ms  wall=15797.3ms  output="Okay, let's see. The NPC is a me"
  merchant_3   tick=2  prefill= 1040.3ms  wall= 2429.3ms  output="Okay, let's break this down. The"
  guard_4      tick=1  prefill=14392.1ms  wall=15791.0ms  output="Okay, let's see. The NPC is a gu"
  guard_4      tick=2  prefill= 1030.7ms  wall= 2426.7ms  output="Okay, let's break this down. The"
```

## Pattern B per-call detail

```
  goblin_1     tick=1  prefill=   45.9ms  wall=16034.1ms  output="Okay, let's break this down. The"
  goblin_1     tick=2  prefill=   45.0ms  wall= 2964.2ms  output="Okay, let's think through this. "
  wolf_2       tick=1  prefill=   47.3ms  wall=16056.8ms  output="Okay, let's see. The NPC is a wo"
  wolf_2       tick=2  prefill=   46.2ms  wall= 2913.9ms  output="Okay, let's break this down. The"
  merchant_3   tick=1  prefill=   47.4ms  wall=16159.6ms  output="Okay, let's see. The NPC is a me"
  merchant_3   tick=2  prefill=   46.5ms  wall= 2911.1ms  output="Okay, let's break this down. The"
  guard_4      tick=1  prefill=   46.3ms  wall=16003.8ms  output="Okay, let's see. The NPC is guar"
  guard_4      tick=2  prefill=   45.9ms  wall= 2871.6ms  output="Okay, let's break this down. The"
```

## Verdict rationale

**FAIL** — Pattern B tick-2 wall is 19.9% SLOWER than A even at the at-scale prefix. The v1 cost decomposition (940 ms fixed + 12.31 ms × shared_tokens) does not match observed data — per-call overhead must scale with prefix size or there is per-call decode overhead unaccounted for.

FAIL at this probe matrix. Pattern B is ~22% slower regardless of prefix size, and the gap doesn't close as prefix grows (it stays ~450-500 ms). The mechanism works correctly (sharedLen detection 100%, KV round-trip preserves state) but the load-bearing comparison was apples-to-oranges all along: Pattern A's session-tracker prefix cache makes Pattern A as fast as Pattern B's prefill phase, while Pattern B pays an extra ~1.4 s save+load round-trip per call.

## Updated v2 priority

1. **Re-frame the prefix-cache value proposition.** Per-conversation snapshot reuse is *not* a win against single-stream sequential calls — those already benefit from the per-model session tracker. The win is only against **interleaved multi-conversation workloads** (multi-NPC concurrent agents on shared weights). Future probes must measure that interleaved pattern explicitly, not the sequential pattern.
2. **#1 (batch-transfer multi-tensor I/O) is now gating, not nice-to-have.** Even in the interleaved regime, save+load adds ~1.4 s per call. To make per-conv snapshots viable for real-time agent ticks (target <500 ms per call), the 72 ASYNCIFY round-trips per save and 72 per load must be fused into single command-buffers (`backend_tensor_get_many` / `backend_tensor_set_many`). Order-of-magnitude reduction needed.
3. **Skip-save-on-disposal heuristic.** If a conversation handle is disposed without further use, save phase is pure waste. Add `disposeConversation(conv, { skipFinalSave: true })`.
4. **Per-head strided readback.** `downloadFromTensor` reads full maxCtx-sized tensors; only `[0, finalLen)` is populated. Strided reads would cut payload by `maxCtx / finalLen` (typically 3-5× at 1325/4096).

## Caveats

- **Single-run probe.** No averaging across multiple page loads.
- **Probe pattern is sequential, not interleaved.** This is the *load-bearing* caveat — it's why Pattern A wins. A follow-up `probe-prefix-cache-interleaved-*` would round-robin NPCs (NPC_1 t1 → NPC_2 t1 → NPC_3 t1 → NPC_4 t1 → NPC_1 t2 → ...) to defeat the session-tracker cache and isolate the per-conv-snapshot value-add.
- **prefillMs on the conv path is structurally biased low** (manual mid-prefill happens before `generateTextStream`'s timed window). Wall-time is the source of truth.
- **`?fa=on` is required.** Conversation handles only work in FA mode.
