# Probe 9a — Prefill prefix-cache decomposition

**Model:** qwen3-8b-iq3m
**Date:** 2026-05-01
**Runs per fixture:** 3 (cache-busted, full page reload each run)
**Build:** post-§27 rebase, tip `e29753286`.

## Method

Three NPC-shaped prompts that share an explicit (PREFIX, TAIL) split.
PREFIX is the kind of stable system+tool material that an NPC harness
re-uses across ticks; TAIL is the per-tick variable observation. The
two **deltas** between adjacent fixtures isolate the marginal cost of
prefix-extension tokens (Plong–Pshort, same tail) and tail-extension
tokens (Tlong–Tshort, same prefix), allowing direct readback of
`a = ms/prefix-token` and `b = ms/tail-token` without depending on
chat-template overhead or any constant baseline term.

Smoke harness extension: `[7/8]` result line now carries
`tokensIn=N` so the probe reads each fixture's actual encoded token
count instead of relying on label estimates.

## Per-fixture medians

| Fixture                | tokensIn | prefill samples (ms) | median (ms) |
|---|---:|---|---:|
| probe9a-Pshort-Tshort  | 85       | 977, 977, 995        | 977         |
| probe9a-Pshort-Tlong   | 190      | 2102, 2459, 2469     | 2459        |
| probe9a-Plong-Tshort   | 305      | 3570, 3686, 4078     | 3686        |

Run 1 of `Pshort-Tlong` was 357 ms below the other two (likely a
shader-cache warmup tail since the smoke page does a full reload per
run). Picking the median rather than the mean discards it; verdict
is robust either way.

## Marginal token costs

- Prefix delta (Plong − Pshort, same tail): **220 tokens → 2709 ms**
  ⇒ **a = 12.31 ms / prefix-token**.
- Tail delta (Tlong − Tshort, same prefix): **105 tokens → 1482 ms**
  ⇒ **b = 14.11 ms / tail-token**.
- Ratio **b / a = 1.15** — prefill is essentially linear in token
  count with a small (15%) per-token premium for tail-position tokens.
  Physically plausible: tail tokens sit at higher absolute positions
  in the sequence so attention computes against more KV-cache rows;
  the modest size of the premium says the linear-FFN term still
  dominates over the attention-quadratic term at these prompt
  lengths (85–305).

## Projection at canonical NPC prompt (P = 400, T = 40)

Linear extrapolation, ignoring the constant baseline term:

- Predicted prefill: a·400 + b·40 = **4924 + 564 = 5488 ms**.
- Prefix's share of prefill latency: **89.7%**.

The extrapolation has two known imperfections — a is the marginal
cost when *extending* an already-200-token prefix, so it slightly
overstates per-token cost at positions [0, 200); and the model's
real prefill at 440 tokens may include a small attention-quadratic
bump beyond the linear projection. Neither shifts the verdict: even
in the worst case where b = a (zero context-length penalty), prefix
is still 91% of total tokens at P=400/T=40, so prefix caching saves
≥90% of prefill latency.

## Caveats

- **Single model.** Probe ran on `qwen3-8b-iq3m` only. Other archs
  (Llama-3, Mistral) might land at slightly different b/a ratios,
  but all transformers in the canonical 6 share the same big-O
  shape (linear FFN + quadratic attention) so the qualitative
  result generalizes.
- **Fixed canonical prompt size assumption.** The 400-token prefix
  is what an NPC harness with system prompt + a couple of tool
  schemas typically lands at; deployments with much smaller stable
  context (≤50 tokens) would see prefix share fall toward 50%
  threshold and prefix caching's marginal value drop.
- **Constant term ignored.** Linear fit didn't model the
  pure-overhead `c` term (BOS / role tokens, KV-cache setup, first
  shader dispatch). Including it would only increase prefix's share
  proportional to prompt length, since `c` is split between
  prefix-share and tail-share by their token-fraction.

## Verdict

**PASS** — prefix accounts for **≥50% of canonical NPC-tick prefill**
(measured 89.7%; threshold robust to the worst-case b=a substitution).

## Downstream decision

KV-cache-per-conversation-on-shared-weights multiplexing — currently
**deferred** per CLAUDE.md "Single-model-active deployment" doctrine
— is now a **load-bearing work item** for any NPC harness that runs
≥1 Hz tick rate on `qwen3-8b-iq3m`-class models. At the projected
5.5 s prefill cost per tick, a freshly-prefilled-from-scratch
approach blows past one-tick-per-second budget by a factor of 5.5×;
prefix caching collapses that to ~0.6 s tail-only prefill,
comfortably under one tick at typical observation tail sizes.

This finding promotes prefix caching from "deferred speculative" to
"queued behind probe 9b/9c/9d closure" — once those probes also
land their data, a full NPC-harness spec can land. Probe 9b
(batched-vs-sequential per-tick scaling) and 9c (hitch-warmup) are
independent and should run next.

The current `engine.ts` keeps one KV cache per loaded model, fine
for single-active-conversation agents but **does not** support
concurrent independent agent conversations on shared weights. The
spec for this work needs to address: (i) per-conversation KV-cache
ownership and re-attach on `chatCompletion`, (ii) prefix-prefix
diff detection (avoid re-prefilling identical prefixes across
ticks), (iii) cache eviction policy when total KV memory exceeds
the per-binding 128 MiB cap.
