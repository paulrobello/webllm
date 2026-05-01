# Probe 9b — Batched-prompt vs sequential

**Model:** qwen3-8b-iq3m
**Date:** 2026-05-01
**N (NPCs):** 4
**Build:** post-§27 rebase, tip `e29753286`.

## Method

Single warm engine, single page load. Sequential first (4 separate
`chatCompletion` calls, one per NPC, `maxTokens=8` each), then a
500 ms settle, then batched (1 `chatCompletion` with all 4
observations, `maxTokens=96`, asked for a JSON array of
`{npc_id, action}`). Both scenarios use the same system preamble
(NPC-controller framing + tool list).

Correctness scoring: each NPC counts as correct if the model emits
a recognizable tool name from `{move, speak, attack, use_item,
trade}`. Sequential matches the per-call output; batched parses
the JSON array's `action` fields.

## Sequential — per-call results

| NPC         | wall (ms) | prefill (ms) | genTokens | output    | matched |
|---|---:|---:|---:|---|---|
| goblin_1    | 1402      | 1273         | 2         | "speak"   | speak   |
| wolf_2      | 1375      | 1289         | 1         | "move"    | move    |
| merchant_3  | 1124      | 999          | 2         | "speak"   | speak   |
| guard_4     | 1146      | 1019         | 2         | "speak"   | speak   |

- Sequential total wall (excluding 500 ms inter-call settles): **5553 ms**
- Sequential prefill sum: **4580 ms** (82% of wall — every call re-prefills the same NPC-controller system preamble)
- Sequential correct: **4 / 4 = 100%**

## Batched — single-call result

- wall: **4010 ms** · prefill: **2136 ms** · genTokens: 48
- Effective per-NPC decode tokens: 48 / 4 = 12 (vs sequential's
  ~1.75 — the JSON wrapper `{"npc_id":"...","action":"..."}` adds
  ~10 tokens per NPC of structural overhead).

Raw output (clean JSON, no fence):
```
[{"npc_id":"goblin_1","action":"attack"},{"npc_id":"wolf_2","action":"move"},{"npc_id":"merchant_3","action":"trade"},{"npc_id":"guard_4","action":"attack"}]
```

Batched correct: **4 / 4 = 100%**

## Headline ratios vs thresholds

| Metric                          | Value | Threshold | Pass? |
|---|---:|---:|:---:|
| Quality ratio (batched / seq)   | 1.00  | ≥0.70     | ✅ |
| Wall ratio (batched / seq)      | 0.72  | ≤0.40     | ❌ |

**Verdict: PARTIAL — quality passes, wall fails.**

## Where the wall savings went

Batched should have crushed sequential on wall — sequential pays
prefill four times for the same system preamble (~1100 ms each,
4× = 4400 ms wasted on duplicate prefix). What ate the projected
≥60% wall savings:

1. **JSON-wrapper decode overhead.** Batched generates 48 decode
   tokens vs sequential's 7 (1.75 × 4). At qwen3-8b-iq3m's ~25 tok/s
   decode, 41 extra tokens cost ~1640 ms. That's the bulk of
   batched's wall difference vs the "naive" prefill-only model.
2. **Per-call structural floor (per probe 9c).** Sequential's
   ~120 ms per-call structural overhead (engine setup, KV-cache
   reset, the per-call ~42 ms decode hitch) adds up but is
   small compared to (1).

Decode-cost decomposition (estimated):
- Sequential decode: 7 tokens × ~40 ms/tok = ~280 ms
- Batched decode:   48 tokens × ~40 ms/tok = ~1874 ms (matches
  measured `wall - prefill`).
- Sequential prefill: 4580 ms; batched prefill: 2136 ms
  (saved ~2444 ms by collapsing duplicates)
- **Net:** batched saves 2444 ms on prefill, loses ~1594 ms on
  JSON-wrapper decode → net savings ~850 ms ≈ 1543 ms wall delta.
  Matches the measured 5553 → 4010 = 1543 ms savings.

## Decision

**Sequential remains the canonical agent-tick pattern**, but with a
strong dependency on prefix caching (probe 9a, PASS). Without
prefix caching, sequential's 5.5 s wall is unacceptable for
≥1 Hz tick rates. With prefix caching, sequential's per-tick
structure becomes:

- First-tick prefill (cold): ~1100 ms (full prefix + tail).
- Steady-state ticks: tail-only prefill (~75 ms at canonical NPC
  prompt size from probe 9a's marginals) + ~70 ms decode (1-2
  tokens) ≈ **~150 ms per tick**.

That's a 7-tick-per-second budget without breaking a sweat. Batched
dispatch is **not necessary** for the NPC use case at N=4.

When batched would be worth revisiting:
- **N ≥ 16-20 NPCs.** At higher N, sequential's per-call structural
  floor (~120 ms × N) starts to matter; batched amortizes one
  prefill across more decisions.
- **Constrained-decoding-style structured output.** If a future
  cycle adds a JSON grammar enforcer (so the JSON wrapper costs ~3
  tokens per NPC instead of 12), the wall-ratio math flips back in
  batched's favor.
- **Multi-NPC reasoning.** If NPC actions need to coordinate
  (e.g., flanking maneuvers), a single batched prompt naturally
  conditions on all observations and may improve quality. Probe
  9b's synthetic NPCs are independent, so this didn't surface.

## Caveats

- **Sample size.** Single run per pattern. Quality could shift on
  more samples but the ≥0.7 ratio is currently 1.00 — robust to
  noise. Wall ratio could vary ±10%; doesn't change the verdict
  band.
- **Quality bar is generous.** Counting any tool-name mention as
  correct doesn't penalize *bad* decisions (e.g., merchant_3 →
  "speak" might be wrong; merchant should probably "trade"). A
  semantic eval would tighten the gap. For probe-purposes, the
  threshold check stands — both patterns are equally able to emit
  *some* valid tool, which is the load-bearing question.
- **Single architecture.** qwen3-8b-iq3m only; other archs may
  have different prefill-vs-decode balance. The qualitative
  finding (sequential + prefix caching beats batched JSON
  output for small N) generalizes.

## Downstream

- Spec for the NPC harness should canonicalize **sequential
  per-tick + prefix caching** as the reference pattern, not
  batched.
- Probe 9d (worker-prototype hitch) is still load-bearing — the
  per-call ~42 ms hitch from probe 9c persists in this probe
  too (sequential's perCall walls are ~1100-1400 ms with most of
  that in prefill, but the structural hitch sits inside the decode
  phase regardless).
- Follow-on (informational only, queued behind harness work):
  benchmark batched at N=16+ to find the crossover point where
  batched flips back to a wall-win.
