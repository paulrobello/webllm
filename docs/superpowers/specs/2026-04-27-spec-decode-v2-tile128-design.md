# §26 §C-v2-A re-measurement under §22 tile=128 — design

**Status:** brainstorming → spec
**Date:** 2026-04-27
**Branch:** `feat/spec-decode-v2-greedy` (side branch, retained post-§C-v2-A close)
**Predecessor closures:** §C v1 (§19), §C v2-A (side-branch §22.5, tip `646320c`), §22 (prefill-tile, on `main` `0c50e03`).

## 1. Goal

Produce one decisive empirical datapoint: does §22's `prefillTileSize=128`
change §C-v2-A's gate verdict at the canonical target/drafter pair? The
result becomes the closure datapoint either way — no mechanism claim is
required, and a confirming negative result is as informative as a positive.

This cycle follows the §17/§18/§19/§20/§24 measure-and-close pattern, not
§22's gated-ship pattern. No `src/` change is planned; if gates pass we
open a follow-on cycle to rebase + ship, not extend this one.

## 2. Why we doubt the premise

§C-v2-A's verify graph is **K+1 = 5 tokens**. §22's tile chunking only
fires when `nTokens > prefillTileSize = 128`. The verify graph is three
orders of magnitude below that threshold and will never be split.

The premise the resumption checklist documented — "tile=128 partially
alleviates the per-step K+1 verify cost" — has no obvious mechanism. The
honest framing for this cycle is empirical: cheaper to measure than to
hypothesize further. The expected outcome is **gates fail again** and the
TODO closure pivots from "candidate, not a conclusion" to "definitive
closure of §C-v2-A under all known levers."

## 3. Scope

### 3.1 Integration approach (cherry-pick, not rebase)

The side branch is 18 commits behind `main` and was last touched
2026-04-27 (`646320c`) before §22 / §23 / §24 / §25 landed. Merge-base is
`77a5e118`, pre-§22.

Pull only the §22 implementation commits onto the side branch:

| Commit    | Subject                                                   |
|-----------|-----------------------------------------------------------|
| `8e21036` | §22 Phase 0 diagnostic (cherry-pick if no conflict; drop if it costs) |
| `c38fb8f` | §22 Task 1 — `prefillTileSize` ctor option + tile dispatcher |
| `f281ac3` | §22 Task 2 — equivalence stub                             |
| `2fcc334` | §22 Task 3 — smoke `?prefillTile=N` URL param             |
| `18e1677` | §22 Task 4 — `--prefill-tile <n>` flag on `eval/perf.ts`  |

Skipping §23 (`0c50e03` registry auto-default) is deliberate — it adds a
second variable. We pass `--prefill-tile 128` explicitly to `perf.ts` so
the measured variable is unambiguous.

**Expected conflict surface:** `eval/perf.ts` (the side branch added
`--drafter` / `--draft-length`; §22 adds `--prefill-tile`). Both follow
the same flag-parsing structure; conflict should be a 5-line additive
merge. `src/inference/model-inference.ts` may also see a small conflict
around the forward path. If the §22 Phase-0 diagnostic
(`8e21036`) conflicts non-trivially, drop it — it is evidence-only and
does not affect the measurement.

### 3.2 Matrix

Four cells, 3-trial median, decode 64 tokens, temperature 0, greedy
contract. Single representative high-α prompt (not the full 36-question
`bench-full` set — that is unnecessary for a measure-and-close).

| Cell | Target          | Drafter                   | Workload           | Tile |
|------|-----------------|---------------------------|--------------------|------|
| 1    | qwen3-8b-iq3m   | —                         | high-α templated   | 128  |
| 2    | qwen3-8b-iq3m   | —                         | low-α creative     | 128  |
| 3    | qwen3-8b-iq3m   | qwen3-0.6b-q4f16, K=4     | high-α templated   | 128  |
| 4    | qwen3-8b-iq3m   | qwen3-0.6b-q4f16, K=4     | low-α creative     | 128  |

- High-α fixture: **`prefill-256`** from `eval/fixtures/long-prompts.ts`
  (already on the side branch). Templated, deterministic continuation
  ("Given that context, briefly explain what makes a software
  engineering team effective.") — drafter and target should frequently
  agree on argmax. Expected α ≥ 0.4.
- Low-α fixture: **`creative-low-alpha`** from `eval/fixtures/long-prompts.ts`
  (added `831da95`). Open-ended creative writing, expected α < 0.3.
- K = 4 only. §C-v2-A close concluded K=4 is the break-even ceiling at
  α ≈ 0.5; K sweeps add variables that don't bear on the gate question.

**Fixture pin caveat.** The §C-v2-A close commit (`646320c`) preserved
gate numbers (5.7 / 16.0 / 12.7 / 16.2 tok/s) but did **not** preserve
the high-α fixture name. Only `creative-low-alpha` is reproducible
verbatim. We adopt `prefill-256` as the high-α stand-in for §26 and
treat the §26 cell-1 baseline (qwen3-8b-iq3m alone, no drafter) as the
authoritative within-cycle baseline against which gate 1 is evaluated.
Cross-cycle comparison against `646320c`'s 16.0 tok/s figure is
informational only.

### 3.3 Decision rule

Unchanged from the §C-v2-A spec, applied to the new numbers:

- **Gate 1 — speedup.** Cell 3 / cell 1 tok/s ≥ **1.5×**.
- **Gate 2 — safety.** Cell 4 / cell 2 tok/s ≥ **0.95×**, AND adaptive
  gate fires within first 16 spec steps on cell 4.

Outcomes:

| Gate 1 | Gate 2 | Action |
|--------|--------|--------|
| FAIL   | FAIL   | **Definitive close** of §C-v2-A. TODO §26 records the closure; side branch retained as archived infra. Resurrection requires architectural change (MEMORY64 → 70B target, faster K+1 verify, …). |
| PASS   | FAIL   | Conditional. Open §C-v2-B cycle: greedy-only contract + adaptive disengage tightened. Out of scope for this cycle. |
| FAIL   | PASS   | Same as both-fail — gate 1 is the load-bearing one. |
| PASS   | PASS   | Open ship cycle: rebase side branch onto `main`, run full coherence + bench-full, follow §22 gated-ship template. Out of scope for this cycle. |

## 4. Failure modes I am watching for

| Symptom                                              | Likely cause                                                                                       | Mitigation                                                                                              |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Cell 1 or 3 drifts >10% from §C-v2-A close baseline (16.2 / 5.85 tok/s) | System load drift; intervening commits on side branch (`6995061` Qwen3-chat unblock, `40217af` perf.ts URL injection) | Re-run the single cell on a quiet system before declaring; document any baseline shift. |
| Cherry-pick conflict in `model-inference.ts` is non-trivial | §22's `forwardSingle()` rename (touched the same forward path the v2-A driver wires through) | Resolve manually preserving v2-A's `forwardVerifyArgmax`; add a smoke-test pass before measuring.       |
| WASM rebuild needed but emsdk not sourced            | `c38fb8f` may touch C++ bridge                                                                     | `source ~/emsdk/emsdk_env.sh && make wasm-build` before measuring; verify `webllm-wasm.{js,wasm}` mtime updates. |
| Cold-shader noise on first trial                     | Documented in §6 of "Active next steps" — first decode after WASM rebuild is shader-compile time   | 3-trial median already absorbs this; use medians, not means.                                            |
| α drops to 0 on cell 3                               | Drafter not engaging (contract gate firing); or vocab mismatch from rebased branches                | Read the engagement-time logs; the v2-A engagement-gate test confirms the drafter handle is wired.      |
| Adaptive gate doesn't fire on cell 4                 | Threshold tuning issue; or low-α fixture doesn't actually produce low α                            | Verify by inspecting per-step α trace; the v2-A driver logs `recentAccepts` per step.                  |

## 5. Output artifacts

- `eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh` — driver
  script, mirrors `eval/reports/prefill-tiling-2026-04-27/run-matrix.sh`
  structure.
- `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md` — cell
  table + gate evaluation + decision-rule outcome.
- `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-{1,2,3,4}.log` —
  raw 3-trial perf.ts output per cell.
- TODO §26 closure entry on `main` (the next reader looks at `main`, not
  the side branch).
- Side branch tip ends on a `docs(TODO): §26 …` commit; branch retained
  per §C-v2-A archive policy.

## 6. What this cycle is NOT

- **Not** a `src/` change cycle. Zero implementation; the side branch
  already carries the v2-A driver, contract, K+1 verify, and tests.
- **Not** a rebase of the side branch onto `main`.
- **Not** a new test cycle. `make checkall` count on `main` stays
  427/11/0; the side branch's checkall is unchanged from `646320c`.
- **Not** a mechanism investigation. If the result surprises, we open a
  follow-on diagnostic cycle.

## 7. Estimated work

| Phase                                    | Estimate    |
|------------------------------------------|-------------|
| Cherry-pick §22 commits + conflict resolution | 15-30 min   |
| WASM rebuild (if needed)                 | 5-10 min    |
| 4 cells × 3 trials × ~30 s + cold-load   | ~25 min     |
| SUMMARY + TODO §26 + commits             | 15 min      |
| **Total**                                | **~75-90 min** |

## 8. Plan reference

To be created next at
`docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md` via
`superpowers:writing-plans`. Plan structure mirrors §22's: explicit
phases, measurable gates per phase, measure-and-close decision rule.
