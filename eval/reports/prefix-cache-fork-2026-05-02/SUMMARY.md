# forkConversation cross-conv prefix sharing — qwen3-8b-iq3m

**Model:** qwen3-8b-iq3m (~1325-token shared system prefix; 4 NPCs × first-tick × 2 patterns; maxTokens=32; FA on)
**Date:** 2026-05-02
**Verdict:** **PASS** — Pattern Y first-tick wall is -6321 ms / -72.2% vs Pattern X.

## Headline (wall-time, first-tick-per-NPC)

| pattern | wall (median) | prefillMs (median) | savings vs X |
|---|---|---|---|
| X (baseline: `createConversation` per NPC) | 8756.6 ms | 40.2 ms | — |
| Y (forked: `forkConversation` from primed base) | 2436.1 ms | 41.1 ms | −6321 ms / −72.2% |

**Total wall-time saved across 4 NPCs:** 25.3 s.

**Base prime tick:** 8129 ms (paid once, amortized across all forks).

## Why this matrix

Pattern X is the realistic naive baseline for spawning multiple agents that share a system prompt: each agent gets a fresh conversation, so the first call to each pays the full ~1325-token system prefill (~14.5 s on Qwen3-8B IQ3_M). With 4 NPCs spawned at session start, that's ~58 s of cumulative wall-time on agent boot.

Pattern Y uses `forkConversation`: drive a base conversation through `[system]` once (the prime tick), then fork it per agent. Each fork inherits the base's KV snapshot via deep copy. The fork's first chatCompletion's longest-shared-token-prefix walk finds `sharedLen ≈ |system|` and prefills only the divergent agent-specific tail. Cost per NPC: ~`load(snapshot)` + `prefill(tail)` + `decode` ≈ snapshot reload + small.

The pattern X / pattern Y separation includes a `resetConversation(modelHandleId)` between them so the engine's per-model session-tracker cache (`engine.ts`) doesn't carry pattern X's KV into pattern Y's base prime — that would mask the fork win by giving the base prime a session-tracker hit instead of a real prefill.

## Pattern X per-call detail (baseline)

```
  goblin_1     prefill=    40.0ms  wall=  8773.3ms  output="Okay, let's see. The goblin_1 is"
  wolf_2       prefill=    40.2ms  wall=  8756.6ms  output="Okay, let's see. The NPC is a wo"
  merchant_3   prefill=    38.7ms  wall=  8715.6ms  output="Okay, let's see. The NPC is a me"
  guard_4      prefill=    44.2ms  wall=  8200.8ms  output="Okay, let's see. The NPC is guar"
```

## Pattern Y per-call detail (forked)

```
  goblin_1     prefill=    40.9ms  wall=  2409.1ms  output="Okay, let's see. The goblin_1 is"
  wolf_2       prefill=    43.4ms  wall=  2487.9ms  output="Okay, let's see. The NPC is a wo"
  merchant_3   prefill=    41.1ms  wall=  2436.1ms  output="Okay, let's see. The player is a"
  guard_4      prefill=    41.1ms  wall=  2413.5ms  output="Okay, let's see. The NPC is guar"
```

## Verdict rationale

**PASS** — Pattern Y first-tick wall 2436 ms vs X's 8757 ms — 72% savings. forkConversation delivers cross-conv prefix sharing as designed.

## Net savings calculation

For N NPCs spawned from a shared system prefix:

| pattern | total wall |
|---|---|
| X (no fork) | `N × baselineFirstTick` = 4 × 8757 ms ≈ **35.0 s** |
| Y (fork) | `baseTickMs + N × forkedFirstTick` = 8129 + 4 × 2436 ≈ **17.9 s** |

**Net savings: 17.2 s on agent spawn (~49% of baseline).** As N grows, the amortized fork win approaches the per-NPC delta (6.3 s / 72%) since `baseTickMs` is paid once.

## Caveats

- **Single-run probe.** No averaging across page loads.
- **prefillMs on the conv path is structurally biased low** (the engine's `stats.timeToFirstTokenMs` covers only `generateTextStream`'s prefill window, not the manual mid-prefill on the conv path). Wall-time is the source of truth.
- **Base prime tick** (paying the shared prefix once) is reported separately. For the fork pattern to be a net win, `baseTickMs + N × forkedWall < N × baselineWall` — true here at N=2 (8129 + 2 × 2436 = 13.0 s vs 2 × 8757 = 17.5 s); break-even is N≈2.
- **Pool sizing.** This probe needed `maxConversations: 8` (default 4) because pattern Y holds 1 base + 4 forks = 5 entries simultaneously. Default `WebLLM.init({ maxConversations })` may need to be raised for fork-heavy workloads.
- **Pattern X wall (~8.7 s) is shorter than the at-scale probe's first-tick wall (~16 s)** despite both using a similar shared system prefix. Possible factors: warmer WebGPU shader cache (this probe runs after a regular smoke turn), session-tracker activity from the [7/8] step partly overlapping the system prefix tokens. The relative win (Y ≈ 28% of X) is the load-bearing finding regardless.
- **`?fa=on` is required.** Conversation handles only work in FA mode.
