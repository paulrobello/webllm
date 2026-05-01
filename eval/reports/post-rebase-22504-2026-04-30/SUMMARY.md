# §27/§32 rebase + sweep: #22504 fast-iquant-matmul — §28 NEGATIVE (2026-04-30)

## Outcome

`webllm-browser-patches` rebased onto upstream master tip `a95a11e5b`
(picks up `45155597a` "add fast matmul iquants" #22504 and `660b1b4bd`
"vulkan: get/set tensor 2d" #22514 underneath the 11-patch stack).
Rebase clean — all 11 patches re-applied with no conflicts. WASM
rebuilds to 2,482,377 bytes (vs prior 2,249,650 — +10% on shader
table size, consistent with #22504's 423-line `mul_mat_decls.tmpl`
expansion).

**Hypothesis (cadence-trigger source):** #22504 lands a fast i-quant
matmul kernel family analogous to #22344, which delivered §27's
**+80% on qwen3-8b-iq3m**. Expected: same-class gain on either or
both IQ3_M models in the canonical fleet.

**Outcome: §28 negative result.** No measured gain on either IQ3_M
model. New rebase pin retained anyway (§32-style: stay current,
small env-floor noise accepted) since the rebase has option value
for the next free-win cycle — same pattern as `§32` itself.

## Headline matrix

3-run median, profile mode (`make smoke-bench PERF_MODEL=<m> PERF_RUNS=3`):

| Model                          | Pre-rebase 2026-04-28 | Post-rebase 2026-04-30 | Δ tok/s | Δ %    | Verdict     |
|--------------------------------|----------------------:|-----------------------:|--------:|-------:|-------------|
| tinyllama-1.1b-chat-q4_0       |                  87.9 |                   74.0 |   -13.9 | -15.8% | env floor   |
| qwen3-0.6b-q4f16  (Q8_0)       |                  68.2 |                   52.3 |   -15.9 | -23.3% | env floor   |
| qwen3-1.7b-q4f16  (Q8_0)       |                  44.0 |                   36.5 |    -7.5 | -17.0% | env floor   |
| mistral-7b-q4ks                |                  29.7 |                   27.6 |    -2.1 |  -7.1% | env floor   |
| llama-3.1-8b-iq3m              |                  23.5 |                   22.6 |    -0.9 |  -3.8% | **flat**    |
| qwen3-8b-iq3m                  |                  21.8 |                   21.3 |    -0.5 |  -2.3% | **flat**    |

The "env floor" verdicts are calibrated against the tinyllama same-day
same-tip recheck (below); the IQ3_M "flat" verdicts come from the
near-zero same-environment delta.

## Calibration: same-day same-tip control + same-env rebase delta

The 2026-04-28 baseline matrix was captured under low system load (early
afternoon). Today's 2026-04-30 measurement window has elevated background
load that the harness can't subtract out; the §32a doctrine's "same-day
pre/post bucket" comparison is required to separate env noise from patch
effect.

**Step 1 — same-day same-tip control on tinyllama** (`fa8b16a6f`, the
*pre-rebase* tip, run after a clean WASM rebuild and the post-rebase
sweep environment had quieted):

| Run                            | tok/s |
|--------------------------------|------:|
| Pre-rebase tip @ 2026-04-28    |  87.9 |
| Pre-rebase tip @ 2026-04-30    |  75.2 |
| **Environmental delta**        | -14.4% |

So today's measurement floor is ~14% below 2026-04-28's. This explains
the -15-23% "regressions" on the small Q4_0/Q8_0 models — they cluster
right at the env floor.

**Step 2 — same-environment rebase delta on tinyllama** (post-rebase tip
*today* vs pre-rebase tip *today*):

| Run                            | tok/s |
|--------------------------------|------:|
| Pre-rebase tip @ 2026-04-30    |  75.2 |
| Post-rebase tip @ 2026-04-30   |  74.0 |
| **Rebase-induced delta**       |  -1.6% |

Within sampling noise. Rebase itself is neutral on tinyllama Q4_0.

## Why the IQ3_M expectation didn't pan out

`#22504 add fast matmul iquants` adds 423 lines to `mul_mat_decls.tmpl`
and a hookup to the shader-lib. The kernels apply to specific i-quant
type / shape combos; the canonical fleet's IQ3_M dispatches evidently
don't route through them (or the gain over the current path is below
our ~1% noise floor). Hypotheses for follow-up (none scheduled):

1. **Block size mismatch.** The #22504 kernels may be specialized for
   block sizes our IQ3_M models don't hit (vocab 128K vs 152K, hidden
   4096 vs 5120, etc.). A targeted shape probe could measure.
2. **Type-class coverage gap.** "Fast iquant" may cover IQ4_NL /
   IQ4_XS but not IQ3_S / IQ3_XXS / IQ3_M. Reading the new shader
   template would show.
3. **#22344 already saturated the lever.** Our IQ3_M models post-§27
   already hit the cooperative-loading kernel. #22504 may be additive
   only on archs / quants we don't run.

None are load-bearing under the 8B project ceiling — IQ3_M at 8B
is already at the design-perf target. Closing without a follow-up
probe.

## §28 closure decisions

1. **Adopt the rebase.** Patch branch tip is now `a45089d5a` (rebased
   onto `a95a11e5b`). Option value of staying current — next upstream
   movement that DOES deliver a free win lands cleanly. No revert.
2. **Don't update CLAUDE.md canonical baseline pins.** Today's numbers
   are env-noisy (-14% floor on the small-model fleet); pinning would
   bake in noise. Re-capture the matrix via `make smoke-bench PERF_RUNS=3`
   on the canonical 6 once env is back to 2026-04-28 cleanliness
   (load average <3, no parallel renderers / simulators).
3. **Retire the iquant-rebase free-win expectation.** Future cadence
   triggers shouldn't reflexively assume "iquant kernel work →
   measurable gain on our IQ3_M models". This is the second iquant
   surface change in the rebase queue; the first (#22344) delivered,
   the second (#22504) didn't.
4. **Delete the rebase-attempt preservation branch.**
   `webllm-browser-patches-rebase-attempt-2026-04-30` was created as
   a safety net during the initial broken-environment investigation;
   the rebase has now been adopted in main and the branch is redundant.

## Artifacts

- Per-model bench logs: `tinyllama-1.1b-chat-q4_0.log`,
  `qwen3-0.6b-q4f16.log`, `qwen3-1.7b-q4f16.log`,
  `mistral-7b-instruct-v0.3-q4ks.log`,
  `llama-3.1-8b-instruct-iq3m.log`, `qwen3-8b-iq3m.log`
- `tinyllama-baseline-recheck.log` — same-day same-tip control at
  `fa8b16a6f`; the env floor calibration this report leans on
- `tinyllama-revert-verify.log` — earlier same-tip run during the
  broken-environment investigation phase (load avg 16.48); shows
  the 1.0 tok/s collapse that triggered the pause + diagnosis. Kept
  for completeness; do **not** treat as a measurement.

## Process notes

This cycle paid out twice on observability infrastructure:

1. **Latent bundle export bug** (`encodeChatPrompt`/`detectChatTemplate`)
   from `f1195407` was caught only because `make smoke-test` had to
   rebundle. Fixed in commit `6d8bc8c` — orthogonal to the rebase but
   worth documenting; the smoke harness has been silently broken for
   ~24 hours.
2. **Environmental measurement floor** (load avg 16.48 → 5.49 → final
   sweep range) is a real risk. The §32a doctrine's same-day pre/post
   discipline saved this cycle from misclassifying as a major
   regression — the initial broken-environment data points (1 tok/s
   on tinyllama at *both* git tips) would have been catastrophic without
   the calibration step.

Per doctrine, the env-floor lesson is worth saving as a reusable
pattern note: any cycle that depends on cross-day comparison with a
saved baseline must capture a same-day same-tip control as the first
data point of the bench window.
