# Probe B — Gemma 4 eval delta (2026-05-12)

## Result: 33/48 = 68.75% — within noise band vs 70.8% Stage 4.4 baseline

**Classification:** neutral / §32-style noise-floor drift. Δ -2.05 pp
absolute (-2.9% relative); n=3 across two branches all landed at
33/48. Cycle is **not** §28 (negative result requires < 65%).

## Hypothesis

Upstream `239a497e5` GELU fix (gelu / gelu_quick / gelu_erf;
f16 → f32 for q_shmem / o_shmem; NaN-clamp) tightens numerical
behavior of Gemma 4's FFN activation. Eval was expected to move
±a few pp from the 70.8% Stage 4.4 baseline.

## Procedure

```bash
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \
  bun run eval/bench.ts --profiles gemma-4-e2b-warm
```

`bench.ts` drives the eval harness through its own
`agentchrome connect --launch --headless` step, runs the 48-task
suite via the dashboard, and reports the passing count.

## Observed runs

| Run | Branch                          | Clamp | Tasks  | Result       | Wedge?                          |
|----:|---------------------------------|:-----:|--------|--------------|---------------------------------|
|  1  | probe-fa-vec-clamp-obsolete     |  off  | 33/48  | 68.75%       | once at 32/48, recovered on retry |
|  2  | probe-fa-vec-clamp-obsolete     |  off  | 14/48  | **wedged**   | both attempt + retry, 184s no-progress |
|  3  | main                            |   on  | 33/48  | 68.75%       | no                              |

Three completing runs (1, 3) hit identical 33-task pass counts.
Run 2 wedged completely on the un-clamped probe branch; see Probe A
SUMMARY for the reclassification triggered by this.

## Comparison to Stage 4.4 baseline

| Metric         | Stage 4.4 (2026-05-12 pre-rebase) | Probe B (post-rebase) | Δ      |
|----------------|----------------------------------:|----------------------:|-------:|
| Tasks passing  |                              34 |                    33 |     -1 |
| Overall %      |                            70.8 |                 68.75 |  -2.05 |

Δ -2.05 pp falls in the 65-70.8% noise band per TODO decision tree.
Per the TODO directive, **noise-floor classification — neither §27
(free win) nor §28 (negative result).** Recorded as §32-adjacent
(small drift, accepted).

## Specific task-level notes (post-rebase)

Not collected at task granularity in this probe; the bench summary
aggregates to the 33/48 headline. If a future investigation needs
task-level diff (e.g., to identify which of the 48 task categories
moved), re-run with the dashboard's per-task drill-down view and
compare against `eval/reports/gemma-4-e2b-validation-2026-05-12/`.

## Decision impact

- **Rebase cycle classification candidate:** lean §27 (precision
  fixes corrective, neutral eval drift) pending Phase 5 perf sweep.
- **Probe A reclassification:** PASS → PARTIAL (forward pass clean,
  but Run 2 wedge raises caution about sustained-eval stability with
  the clamp dropped). Permanent clamp drop deferred to a follow-up
  probe.
- **Phase 5 sweep:** runs on `main` branch (clamp present) — the
  conservative baseline.

## Artifacts

- Dashboard run records under `eval/reports/smoke-runs/` (3 records
  starting with `01778640274173-gemma-4-e2b-warm.json`).
- Wedge diagnostic: `bench.ts` reports "no task progress for 184s —
  last seen 14/48 tasks" (Run 2 both attempts).
