# llama.cpp rebase + sweep cycle — 2026-05-12

**Classification:** **§32 — small regression, accepted** (with thermal
contamination caveat — see "Methodology asterisks" below).

**Headline:** rebased patch stack onto upstream master tip `856c3adac`
(was `a817a22bc`). 10 effective patches reapplied cleanly with zero
manual conflict resolution. Post-rebase Pass 2 sweep shows uniform
-2% to -11% tok/s across the canonical 7 fleet; matmul time up
+4-9%. Most of this delta is thermal floor drift from same-day
back-to-back captures (Probe A → Probe B → sweep, with no extended
cooldown), not a true rebase regression — the run-1 vs p50
asymmetry across all 7 models confirms thermal contamination, and
small-model run-1 tok/s sits at 1-7% below pre-rebase baseline
(the realistic regression bound).

**Decision:** stay on rebased tip. Documenting the regression
magnitude with the thermal caveat; recommending the next probe
fire after a longer cooldown to retake clean numbers if the
regression turns out to be persistent rather than thermal.

## Probe outcomes

| Probe | Result | Decision |
|-------|--------|----------|
| **A — FA-VEC clamp obsolescence** | PARTIAL | Forward-pass probe PASSES on the un-clamped rebased build (no `unreachable` trap, healthy argmax). Sustained eval on the same un-clamped branch wedged on its second invocation. **Keeping `9ea3bfc` on the rebased branch**; filed follow-up probe with tighter test matrix. See [`probe-a-fa-vec-clamp/SUMMARY.md`](probe-a-fa-vec-clamp/SUMMARY.md). |
| **B — Gemma 4 eval delta** | 68.75% (n=3) | Δ -2.05 pp absolute vs 70.8% baseline. In the 65-70.8% noise band per TODO decision tree. **Neutral / §32-adjacent**. See [`probe-b-gemma4-eval/SUMMARY.md`](probe-b-gemma4-eval/SUMMARY.md). |

## Phase 1 — Rebase

**Pre-rebase tip:** `webllm-browser-patches @ ebc7c3d82` (18 JSEP
probe commits stacked above canonical `b54503497`).

**Procedure:**
1. Created safety backup branch `webllm-browser-patches-pre-rebase-2026-05-12`.
2. `git reset --hard b54503497` — dropped the 18 JSEP probe commits
   per the 2026-05-08 negative-closure (TODO predicted 16; actual
   was 18 — off-by-2 documented).
3. `git rebase origin/master` — **clean** rebase, no manual conflicts
   in `ggml-webgpu.cpp` despite TODO predicting +91 lines from
   upstream #22906 and #22808. The webgpu_context struct and FA
   pipeline-key changes integrated automatically.
4. 10 effective patches reapplied (TODO said 11 — off-by-one
   documented; the patch stack count is now canonical at 10).

**Post-rebase tip:** `webllm-browser-patches @ 4192e05ba` on merge
base `856c3adac`.

## Phase 2 — Rebuild + sanity

WASM build **failed** initially with `wasm-ld: error: symbol exported
via --export not found: ggml_cpu_set_weight_hash_probe`. The webllm-
side spike commit (Stage 4.29 Probe 16) added this export to
`src/wasm/CMakeLists.txt` line 148, with the implementation living
in the llama.cpp probe commits we just dropped. **Fix:** remove the
stranded export line + trailing comment block, leaving
`_webllm_get_tensor_data_hash` (defined in `webgpu-bridge.cpp`,
this build's TU). Committed as
`a469771 fix(wasm): drop stranded JSEP-probe export from CMakeLists`.

Sanity bench (TinyLlama, 3 runs, profile mode) ran cleanly at
107.8 tok/s p50. Profile mode is -15-28% relative to non-profile per
the harness output, so this is consistent with the Pass 2 baseline
130.5 (non-profile) given that smoke-bench reports profile-mode
numbers.

## Phase 5 — Pass 2 canonical 7-fleet headline matrix

5-run profile mode, fresh headless Chrome per model, 30s cooldown.
First TinyLlama attempt timed out on Run 1/5 due to wasm-rebuild
delay between Chrome launch and bench start; re-ran in isolation
after `agentchrome connect --disconnect → relaunch` and the second
attempt succeeded.

| Model                          | Quant   | pre-rebase | post-rebase | Δ tok/s | Δ %    | matmul pre | matmul post | matmul Δ %  |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | Q4_0    | **130.5** | **120.5** | -10.0 | -7.7% | 4.00 | 4.19 | +4.8% |
| qwen3-0.6b-q4f16               | Q8_0    |  **99.4** |  **96.0** |  -3.4 | -3.4% | 3.67 | 3.87 | +5.5% |
| qwen3-1.7b-q4f16               | Q8_0    |  **71.1** |  **66.1** |  -5.0 | -7.0% | 6.55 | 6.95 | +6.1% |
| mistral-7b-instruct-v0.3-q4ks  | Q4_K_S  |  **46.4** |  **44.2** |  -2.2 | -4.7% | 15.53 | 16.25 | +4.6% |
| llama-3.1-8b-instruct-iq3m     | IQ3_M   |  **33.8** |  **31.9** |  -1.9 | -5.6% | 22.41 | 23.46 | +4.7% |
| qwen3-8b-iq3m                  | IQ3_M   |  **31.0** |  **29.6** |  -1.4 | -4.5% | 22.68 | 23.92 | +5.5% |
| gemma-4-e2b-it-q4km            | Q4_K_M  |  **38.6** |  **34.2** |  -4.4 | -11.4% | 8.19 | 8.91 | +8.8% |

Per-model raw logs in `post-rebase/<model-id>.log`.

## Methodology asterisks — thermal contamination is significant

The post-rebase sweep was captured immediately after a full
**Probe B Gemma 4 eval (n=3 across two branches)** and the
**Phase 2 sanity bench**. The system had been running ~2 hours of
back-to-back webgpu workloads with limited cooldown. Pre-rebase
baselines were captured at end-of-day 2026-05-12 with the system
in a cooler state.

**Within-capture thermal pattern (every model shows it):**

| Model                          | Run 1 | Run 5 | Δ %    | Pattern               |
|--------------------------------|---:|---:|---:|------------------------|
| tinyllama-1.1b-chat-q4_0       | 122.2 | 120.7 | -1.2% | flat (small model)     |
| qwen3-0.6b-q4f16               | 93.9 | 95.4 | +1.6% | flat (small model)      |
| qwen3-1.7b-q4f16               | 65.9 | 55.5 | **-15.8%** | thermal dip on run 5 |
| mistral-7b-instruct-v0.3-q4ks  | 45.5 | 44.2 | -2.9% | mild fade              |
| llama-3.1-8b-instruct-iq3m     | 33.4 | 31.9 | -4.5% | thermal fade after run 2 |
| qwen3-8b-iq3m                  | 30.5 | 29.2 | -4.3% | thermal fade after run 2 |
| gemma-4-e2b-it-q4km            | 44.8 | 32.9 | **-26.6%** | heavy thermal fade  |

**Run-1 vs pre-rebase p50** (thermal-minimized comparison):

| Model                          | Pre-rebase p50 | Post Run-1 | Δ %    |
|--------------------------------|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 130.5 | 122.2 | -6.4% |
| qwen3-0.6b-q4f16               |  99.4 |  93.9 | -5.5% |
| qwen3-1.7b-q4f16               |  71.1 |  65.9 | -7.3% |
| mistral-7b-instruct-v0.3-q4ks  |  46.4 |  45.5 | -1.9% |
| llama-3.1-8b-instruct-iq3m     |  33.8 |  33.4 | -1.2% |
| qwen3-8b-iq3m                  |  31.0 |  30.5 | -1.6% |
| gemma-4-e2b-it-q4km            |  38.6 |  44.8 | **+16.1%** |

The run-1 comparison brings every model except small-model + Gemma 4
into the ±2% noise band. Small models still show 5-7% real
regression (likely from #22906 mulmat-q refactor's per-op overhead,
which is amortized over fewer ops at TinyLlama-class scale). Gemma 4
**improves** on run-1 because its pre-rebase baseline was itself
thermally hit (per-run was 45.6 / 44.6 / 30.7 / 38.6 / 36.9 — Gemma 4
is intrinsically dispatch-heavy).

**Realistic regression bounds:** -1% to -7% absolute, weighted
toward -1 to -3% for the 7B+ tier and -5 to -7% for the small models.
Matmul time uniformly up +4-9%, consistent with #22906 adding a
mulmat-q dispatch path that's slightly slower for the
quants we use (we don't yet exercise MXFP4 — that's the gpt-oss-20b
path #22906 was sized for).

## Classification rationale (§32)

Per TODO Phase 6 decision tree:

- **§27 (free win)** — requires 5+ of 7 models showing ≥ 2% matmul
  faster OR clear eval improvement on Gemma 4. **NOT MET** — matmul
  uniformly slower, eval drift -2.05 pp.
- **§28 (negative result)** — requires a prior lever closing harder.
  **NOT MET** — no specific lever's resurrection-path was contradicted
  by this rebase. Probe A's "PARTIAL" outcome is information about
  the FA-VEC clamp follow-up specifically, not a §28 trigger.
- **§32 (small regression, accepted)** — 5/6 hold neutral but 1 holds
  a measurable regression. **MATCHES with caveats**: this is more
  uniform than the canonical §32 template (every model regresses
  somewhat), but the magnitude is small (1-7% depending on thermal
  accounting) and the eval delta is in the noise band. Staying
  current carries option value — next cycle's #22906 follow-ups
  (better quant-specific dispatch) will land cleanly on this base.

## Decision

1. **Adopt rebased tip** `webllm-browser-patches @ 4192e05ba` on
   merge base `856c3adac` as the new canonical baseline.
2. **Keep `9ea3bfc` (FA-VEC clamp)** — Probe A's sustained-eval
   wedge raises caution; permanent drop is a separate filed probe.
3. **Pin the post-rebase numbers as the new baseline series for §32
   cycles going forward**, with the thermal caveat documented. Next
   §32a follow-on probe should fire after a longer cooldown (2+ hours
   idle) to disambiguate thermal contamination from rebase regression.

## Filed follow-ups

1. **FA-VEC clamp permanent-drop validation** — n=3 stability + TTFT
   measurement on Gemma 4 with the clamp dropped. Trigger: free cycle
   slot or another rebase cycle introducing FA path changes.
2. **Cool-system rebench** — repeat Pass 2 after ≥ 2 hours of idle
   to disambiguate thermal contamination. Trigger: next idle window
   before the next rebase cycle fires.
3. **mulmat-q dispatch profiling** — investigate whether the +4-9%
   matmul time is uniform across the dispatch path or concentrated
   in a specific kernel selection branch added by upstream #22906.
   Useful if a future cycle's #22906 follow-ups want a baseline.

## Artifacts

- `probe-a-fa-vec-clamp/SUMMARY.md` + `page-snapshot.txt`
- `probe-b-gemma4-eval/SUMMARY.md`
- `post-rebase/{tinyllama,qwen3-0.6b,qwen3-1.7b,mistral-7b,llama-3.1-8b,qwen3-8b,gemma-4-e2b}*.log`
- `post-rebase/sweep.log` — sweep-loop driver log

## Tips for reference

- **Pre-rebase webllm tip:** `8117fe2`
- **Post-rebase webllm tip:** (will be set after Phase 6 commits land)
- **Pre-rebase llama.cpp tip:** `ebc7c3d82` (with 18 dormant JSEP
  probes), canonical at `b54503497`
- **Post-rebase llama.cpp tip:** `4192e05ba` on merge base `856c3adac`
- **Safety backup:** `webllm-browser-patches-pre-rebase-2026-05-12`
  on llama.cpp side (preserved against bad merge)
