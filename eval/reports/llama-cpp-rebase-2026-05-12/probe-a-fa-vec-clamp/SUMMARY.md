# Probe A — FA-VEC clamp obsolescence (2026-05-12)

## Result: PARTIAL — forward pass PASSES, but eval-time stability ambiguous; clamp drop deferred

The 485 ms forward-pass probe completed cleanly on the rebased build
with `9ea3bfc` reverted (no trap, healthy argmax 50429 logit 9.495).
However, Probe B (48-task eval on `gemma-4-e2b-warm`) wedged twice on
the un-clamped probe branch when re-invoked back-to-back, then ran
clean once on main with the clamp present — see
[`../probe-b-gemma4-eval/SUMMARY.md`](../probe-b-gemma4-eval/SUMMARY.md).

**Decision:** **keep `9ea3bfc` on the rebased branch.** The
permanent drop is filed as a follow-up probe with a tighter test
matrix (multi-iteration eval stability on a longer trace, plus
explicit prefill-TTFT measurement on Gemma 4 to confirm the recovery
that would justify the risk). Today's Probe A confirms only that
upstream `239a497e5` removed the *immediate* unreachable trap, not
that the broader path is wedge-free under sustained load.

## Hypothesis

Upstream `239a497e5` "ggml-webgpu: address precision issues for
multimodal" rewrote the FA path-selection helper
(`ggml-webgpu-shader-lib.hpp`), bumped q_shmem / o_shmem from f16 to f32,
restructured PATH_VEC shared-memory math, and added q_type/dst_type to
the pipeline key. We hypothesized this fixes the LDS-budget constraint
that originally forced the `src0.ne[1] < 20` ceiling — making the
TS-side `prefillTileSize = 16` clamp redundant.

## Procedure

1. Rebased patch stack onto upstream `856c3adac` (10 effective patches).
2. Built wasm32 + wasm64 (clean — no conflicts, no missing symbols after
   removing the stranded `_ggml_cpu_set_weight_hash_probe` export).
3. Created probe branch `probe-fa-vec-clamp-obsolete` on top of `main`
   tip `8117fe2`; reverted `9ea3bfc` (one commit on the probe branch).
4. Rebundled smoke-test artifacts (TS bundle picks up the revert; WASM
   already rebuilt in Phase 2).
5. Navigated existing headless tab to:
   `http://localhost:8031/fa-prefill-probe.html?model=gemma-4-e2b-it-q4km&ctx=4096&path=forward&chat=1&v=$(date +%s)`

## Observed signals — forward-pass probe

```
[2/6] streaming GGUF from ./models/gemma-4-e2b-it-q4km.gguf...
[4/6] weights uploaded, KV ctxLen=4096
[6/6] forward() → forwardSingle...
[6/6] OK in 0.481s · rows=1 · last-row argmax=50429 (logit=9.495)
[FA-PREFILL-PROBE-DONE-PASS]
```

- **No `RuntimeError: unreachable` console trap.**
- **Healthy argmax** (token id 50429, logit 9.495) — not the
  uniform-noise pattern that NaN-driven sampling produces.
- Forward completed in **481 ms** (single forward through ctx=4096
  prefill graph; un-clamped tile size).

## Why "PARTIAL" instead of "PASS"

The Probe A page exercises one forward pass (485 ms) and exits. It
proves the original unreachable-trap path is no longer reachable. It
does **not** prove the un-clamped FA-VEC path is wedge-free under
sustained eval load. In Probe B, running the 48-task eval on the
un-clamped probe branch wedged on its second consecutive invocation
(184s no-progress timeout, both attempt + retry), while the
clamp-present main branch completed cleanly. The instability could
be environmental (headed Chrome state accumulation, GPU process
pressure) or could indicate that the un-clamped path is more
sensitive to the wedge regime — Probe A alone cannot distinguish.

## Filed follow-up — "FA-VEC clamp permanent-drop validation"

Trigger: free cycle slot or another rebase cycle introducing FA path
changes.

Plan:
1. Run Probe B (gemma-4-e2b-warm eval) n=3 on the un-clamped branch
   with explicit Chrome restart between iterations.
2. Measure TTFT on a 2,238-token prefill (the Stage 4.4 baseline
   case, 42.3s at clamped q_tile=1) with the clamp dropped — confirm
   the recovery that would justify the risk.
3. If n=3 eval all complete cleanly AND TTFT drops measurably, land
   the clamp drop with confidence. Otherwise, file a deeper
   investigation into whether the wedge correlates with the
   un-clamped path or is independent Gemma 4 instability.

## Artifacts

- `page-snapshot.txt` — accessibility-tree snapshot at forward-pass probe completion.
- Probe branch deleted post-probe (was `probe-fa-vec-clamp-obsolete`
  at `56d78fe` — a one-commit revert of `9ea3bfc` on top of `main`
  tip `8117fe2`). The follow-up probe re-creates it via
  `git revert 9ea3bfc` on a fresh probe branch.
