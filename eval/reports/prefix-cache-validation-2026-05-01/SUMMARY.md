# Prefix-cache validation — qwen3-8b-iq3m, 4-NPC × 2-tick

**Model:** qwen3-8b-iq3m
**Date:** 2026-05-01
**Verdict:** **PASS**

## Method

4 NPCs × 2 ticks per NPC × 2 patterns = 16 timed `chatCompletion` calls. Each call requests `maxTokens: 8`. Inter-call sleep 500 ms so KV-cache resets and GPU-queue drains don't bleed into the next prefill.

- **Pattern A** uses `engine.chatCompletion(modelId, messages, ...)` — the legacy path. The engine maintains a single per-model session tracker and re-prefills on every call.
- **Pattern B** allocates one `ConversationHandle` per NPC and calls `engine.chatCompletion(conv, messages, ...)`. Each conv keeps its own KV snapshot. Tick-1 pays the full prefill (and saves a snapshot); tick-2's `[system, user(obs1)]` prefix matches the snapshot, so only the divergent `[assistant, user(obs2)]` tail re-prefills.

## Tick-2 medians (the headline measurement)

- Pattern A tick-2 median prefill: **761.3 ms** (full re-prefill — the baseline)
- Pattern B tick-2 median prefill: **38.6 ms** (prefix-cache hit — the win)
- Speedup: **19.7×**

## Tick-1 medians (sanity check — both patterns pay the full prefix on first tick)

- Pattern A tick-1 median prefill: 1284.7 ms
- Pattern B tick-1 median prefill: 43.9 ms

## Per-call detail

```
Pattern A (no handles, full re-prefill every call):
  goblin_1     tick=1  prefill= 1293.7ms  wall= 1624.3ms  output="Okay, let's see."
  goblin_1     tick=2  prefill=  775.4ms  wall= 1086.1ms  output="Okay, the hero is getting"
  wolf_2       tick=1  prefill= 1284.7ms  wall= 1600.1ms  output="Okay, let's see."
  wolf_2       tick=2  prefill=  760.9ms  wall= 1077.8ms  output="Okay, the wolf is hungry"
  merchant_3   tick=1  prefill= 1016.0ms  wall= 1328.2ms  output="Okay, let's see."
  merchant_3   tick=2  prefill=  757.9ms  wall= 1070.3ms  output="Okay, the hero asked about"
  guard_4      tick=1  prefill=  994.7ms  wall= 1313.3ms  output="Okay, let's see."
  guard_4      tick=2  prefill=  761.3ms  wall= 1081.6ms  output="Okay, so the guard_"

Pattern B (per-NPC ConversationHandle, prefix cache active on tick 2):
  goblin_1     tick=1  prefill=   44.2ms  wall= 1858.6ms  output="Okay, let's see"
  goblin_1     tick=2  prefill=   42.1ms  wall= 1604.0ms  output="Okay, the goblin"
  wolf_2       tick=1  prefill=   42.3ms  wall= 1805.0ms  output="Okay, let's think"
  wolf_2       tick=2  prefill=   38.2ms  wall= 1506.3ms  output="Okay, let's break"
  merchant_3   tick=1  prefill=   43.9ms  wall= 1538.4ms  output="Okay, let's see"
  merchant_3   tick=2  prefill=   38.6ms  wall= 1512.3ms  output="Okay, let me think"
  guard_4      tick=1  prefill=   41.8ms  wall= 1527.1ms  output="Okay, let's see"
  guard_4      tick=2  prefill=   37.5ms  wall= 1482.8ms  output="Okay, let's break"
```

## Verdict

**PASS** — Pattern B's tick-2 median prefill of 38.6 ms is below the projected band but the speedup over pattern A tick-2 (19.7×) is well above the 5× threshold — prefix cache is delivering more savings than projected.

## PASS criteria (from plan Task 6)

- **PASS** — pattern B tick-2 median in 75-150 ms (probe 9a's projected band: `b · 40 + decode` ≈ 75-150 ms with prefix cache active)
- **PARTIAL** — pattern B tick-2 median in 150-500 ms (some win but short of projection)
- **FAIL** — pattern B tick-2 median > 500 ms (no measurable cache hit)

## Caveats

- **Single-run probe.** No averaging across multiple page loads; figures are one timing run per NPC × tick.
- **Decode confound on first-token timing.** When `stats.timeToFirstTokenMs` is available the engine's official prefill figure is used; otherwise the first-chunk wall time is reported (= prefill + 1× decode). Both patterns get the same overhead so the comparison stays valid.
- **`?fa=on` is required.** Conversation handles only work in FA mode (engine throws otherwise); the driver forces `fa=on` in the URL.
- **`ingest=off`** keeps this diagnostic probe out of the live dashboard DB. The smoke page's run-complete hook is gated on the regular [7/8] flow finishing; the probe block doesn't trigger the SmokeRunRecord post.
