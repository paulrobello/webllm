# Pre-rebase profile-mode baselines — canonical 6 + Gemma 4 (2026-05-12)

**Captured:** 2026-05-12 / 2026-05-13 (UTC) in two passes. This file
documents the **clean retake** (Pass 2): the noisy Pass 1 was
superseded after the 3-run-per-model headless capture showed high
run-to-run variance from cold-shader bias on early runs and thermal
contention on later runs. Pass 2 keeps the same regime but adds
mitigations described under "Capture methodology" below.

**webllm tip:** `f8c4c65` (`feat(stage5.2): add gemma-4-e2b-warm to
bench-full profile set`).
**llama.cpp `webllm-browser-patches` tip:** `ebc7c3d82` (16 inert JSEP
experimental probe commits on top of canonical `b54503497`; JSEP is
gated `OFF` in the WASM build per `ggml/CMakeLists.txt:236` and CPU
backend is `OFF` per `src/wasm/CMakeLists.txt`, so the probes don't
compile into our WASM artifact — behaviorally equivalent to
`b54503497`).
**WASM:** `smoke-test/webllm-wasm.wasm` 3,767,720 bytes (May 12 17:20).

**Procedure (Pass 2 — canonical regime):**
- `agentchrome connect --disconnect` → `sleep 30` (thermal cooldown)
  → `agentchrome connect --launch --headless` → `sleep 5` (settle)
  **before every model**
- `make smoke-bench PERF_MODEL=<m> PERF_RUNS=5` (profile mode, 5 runs)
- 7 models total (canonical 6 + `gemma-4-e2b-it-q4km`)
- Dashboard left running; runs auto-ingested

**Trigger:** §32a pre-rebase doctrine — capture before the next rebase
fires, so a follow-on probe gets a same-tip baseline for diagnosis.
Also serves as the Stage 5.1 non-regression check: verify the NEOX
RoPE fix (Q1.6 / Gemma family branch extensions, `c8c8447` …
`31d53a5`) didn't move non-Gemma models. Pass 2 also closes the
Stage 4.4 watch-item (smoke-harness speed timeout for
`gemma-4-e2b-warm`).

---

## Headline matrix — Pass 2 (clean retake)

5-run profile mode, fresh headless Chrome per model, 30s cooldown.
"p50" is the median tok/s row's full set of columns (matches per-run
file output); spread shows max−min across the 5 runs as % of p50.

| Model                          | Quant   | tok/s p50 | spread | matmul med (ms / %) | encode med (ms / %) | dispatch/token |
|---|---|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | Q4_0    | **130.5** |  2.5%  | 4.00 / 58.4%        | 1.50 / 22.2%        | 450  |
| qwen3-0.6b-q4f16               | Q8_0    |  **99.4** |  3.6%  | 3.67 / 50.2%        | 2.10 / 28.4%        | 629  |
| qwen3-1.7b-q4f16               | Q8_0    |  **71.1** | 34.6%* | 6.55 / 56.3%        | 2.20 / 20.2%        | 629  |
| mistral-7b-instruct-v0.3-q4ks  | Q4_K_S  |  **46.4** |  3.0%  | 15.53 / 76.7%       | 2.90 / 14.4%        | 650  |
| llama-3.1-8b-instruct-iq3m     | IQ3_M   |  **33.8** |  5.0%  | 22.41 / 81.5%       | 2.70 /  9.6%        | 652  |
| qwen3-8b-iq3m                  | IQ3_M   |  **31.0** |  4.2%  | 22.68 / 79.2%       | 3.30 / 11.6%        | 805  |
| gemma-4-e2b-it-q4km            | Q4_K_M  |  **38.6** | 38.6%* | 8.19 / 47.6%        | 4.10 / 23.1%        | 1040 |

\* qwen3-1.7b had one 47.4 tok/s thermal-dip outlier among five runs
(71.1 / 71.0 / 72.0 / 71.2 / 47.4); median absorbs it. Gemma 4 is
intrinsically dispatch-heavy (1040 dispatches per token vs 450-805 for
the canonical 6) — runs vary more because per-dispatch micro-stalls
accumulate; per-run tok/s: 45.6 / 44.6 / 30.7 / 38.6 / 36.9. Matmul
time is stable in both cases (qwen3-1.7b matmul samples: mean 6.54,
median 6.55, p90 6.88; Gemma 4 matmul: mean 8.22, median 8.19, p90
8.72) — the variance is in the rest of the pipeline, not in compute.

Per-model raw logs in this directory (`<model-id>.log`).

## Cross-baseline drift vs `pre-rebase-baselines-2026-05-04/`

| Model                          | 2026-05-04 | 2026-05-12 (Pass 2) | Δ tok/s | Δ %    | matmul med Δ              |
|---|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       |  80.4 | 130.5 | +50.1 | +62.3% | 4.19 → 4.00  (-4.5%)  |
| qwen3-0.6b-q4f16               |  62.5 |  99.4 | +36.9 | +59.0% | 4.00 → 3.67  (-8.3%)  |
| qwen3-1.7b-q4f16               |  41.7 |  71.1 | +29.4 | +70.5% | 6.88 → 6.55  (-4.8%)  |
| mistral-7b-instruct-v0.3-q4ks  |  28.7 |  46.4 | +17.7 | +61.7% | 16.58 → 15.53 (-6.3%) |
| llama-3.1-8b-instruct-iq3m     |  23.3 |  33.8 | +10.5 | +45.1% | 23.66 → 22.41 (-5.3%) |
| qwen3-8b-iq3m                  |  21.3 |  31.0 |  +9.7 | +45.5% | 24.12 → 22.68 (-6.0%) |

**Matmul absolute time moved -4.5% to -8.3% across the canonical 6 —
uniformly faster, never slower.** This is not the "bit-identical
matmul" claim Pass 1 made (which was based on noisier 3-run data
captured on a Chrome instance that had inherited state from prior
model loads). The Pass 2 fresh-Chrome-per-model regime exposes
~5-8% additional compute headroom that prior captures' Chrome state
was eating. Most likely contributors to this delta, in priority order:

1. **Methodology**: Pass 2 launches a brand-new Chrome instance with
   zero accumulated GPU contexts immediately before each model bench
   and waits 30s for thermal settling. Prior captures (incl. Pass 1)
   reused Chrome across multiple model loads with no cooldown.
2. **Code drift since 2026-05-04**: webllm tip moved from somewhere on
   the post-§27 baseline to `f8c4c65`; load-bearing changes include
   per-layer headCount plumbing (`447ff82`), FA-VEC `prefillTileSize`
   clamp (`9ea3bfc`), and parity-capture instrumentation (`2c32f80`).
   None of these were intended to move canonical-fleet decode, but
   any could plausibly contribute single-digit-% via dispatch-path
   side effects.
3. **llama.cpp drift**: from `fc1f81242` to `b54503497` (the WaitAny
   under JSPI commit replaces a poll loop — could reduce per-dispatch
   tail latency).

**Stage 5.1 non-regression conclusion stands and is strengthened**:
the NEOX RoPE fix (Q1.6 + `getRopeModeForArchitecture` family
extensions) is architecture-gated to gemma2/gemma3/gemma4 and did
**not** regress non-Gemma models — matmul time moved in the favorable
direction across all 6. Non-Gemma models on `bench-full` will see a
mild tailwind, not a regression, when Gemma 4 joins the fleet at
Stage 5.2.

## Cross-baseline drift vs noisy Pass 1 (same-day)

For completeness — comparing Pass 1's 3-run-with-shared-Chrome capture
to Pass 2's 5-run-with-fresh-Chrome capture exposes how much variance
the Pass 1 methodology had:

| Model                          | Pass 1 (noisy) | Pass 2 (clean) | Δ %   | matmul Δ % |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 116.1 | 130.5 | +12.4% | -4.5% |
| qwen3-0.6b-q4f16               |  87.2 |  99.4 | +14.0% | -9.6% |
| qwen3-1.7b-q4f16               |  64.4 |  71.1 | +10.4% | -6.6% |
| mistral-7b-instruct-v0.3-q4ks  |  43.3 |  46.4 |  +7.2% | -6.3% |
| llama-3.1-8b-instruct-iq3m     |  33.0 |  33.8 |  +2.4% | -5.8% |
| qwen3-8b-iq3m                  |  29.4 |  31.0 |  +5.4% | -6.7% |
| gemma-4-e2b-it-q4km            |  43.4 |  38.6 | -11.1% | -8.1% |

Two reading points:

- **Smaller models gain more from Pass 2's methodology.** TinyLlama
  +12%, Qwen3-0.6B +14%, Qwen3-8B only +5%. Encode-overhead share
  drops further when there's less compositor / Chrome-state
  contention; smaller models spend a larger fraction of decode in
  the encode path, so they benefit more.
- **Gemma 4 went DOWN in the headline because of its variance, not
  its compute.** Matmul moved -8.1% (faster), consistent with the
  rest of the fleet. But Gemma 4's per-token dispatch count (1040)
  makes it most exposed to micro-stalls; the 5-run median caught a
  more-representative-but-slower number. Pass 1's 3-run capture
  happened to put one fast and one slow run, with the third
  splitting the difference. **Recommended for Gemma 4 specifically**:
  capture 7-9 runs and discard 2 outliers for higher-confidence
  speed numbers.

## Capture methodology — important asterisks

1. **Pass 1 (noisy, superseded).** First sweep this date used a single
   headless Chrome with restarts only before 7B+ models. Variance
   patterns visible: TinyLlama 116.1 / 115.7 / 124.4 (run 3 fastest —
   cold-shader bias); qwen3-1.7b 62.5 / 64.4 / 64.7 (climbing as
   cache warmed); llama-3.1-8b 33.0 / 33.4 / 31.4 (run 3 slowest —
   thermal); Gemma 4 43.4 / 44.4 / 30.7 (run 3 dropped 31% — heavy
   thermal). Median-of-3 swung 5-30% depending on which extreme ran
   when. Pass 1 logs are not preserved on disk (Pass 2 overwrote in
   place).

2. **Pass 2 methodology (now canonical):**
   - **Fresh Chrome per model** via
     `agentchrome connect --disconnect && sleep 30 && agentchrome
     connect --launch --headless && sleep 5` before each
     `make smoke-bench` invocation. Eliminates VRAM accumulation,
     compositor state carry-over, and prior-model thermal residue.
   - **5 runs per model** (up from 3). Median-of-5 absorbs one
     extreme without flipping (worked for the qwen3-1.7b 47.4
     outlier; almost worked for Gemma 4 — see follow-up below).
   - **30s thermal cooldown** between `agentchrome connect
     --disconnect` and the next `--launch`. Empirically sufficient
     for the canonical 6; Gemma 4 is dispatch-heavy enough that
     30s wasn't always enough.
   - **Dashboard ingest left on.** All 7 × 5 = 35 runs auto-flowed
     into `eval/reports/smoke-runs.db`; superseding older noisy
     data points by virtue of newer timestamps in the dashboard's
     "latest-N" filters.

3. **Cross-comparing 2026-05-12 (Pass 2) against 2026-05-04 is
   methodologically unequal** — 2026-05-04 was captured on the user's
   pre-existing interactive Chrome with concurrent tabs. The Δ in
   that direction includes both real environmental savings (clean
   headless) AND minor code drift since 05-04. For future §32 / §27
   / §28 adjudication, the next post-rebase sweep should use the
   Pass 2 regime (fresh-Chrome-per-model + 30s cooldown + 5 runs +
   headless) to keep the baseline series internally comparable.

4. **Per-binding 128 MiB cap doctrine still in force.** This capture
   uses wasm32 default builds (no MEMORY64); Gemma 4 fits with
   hybrid Q4_K_M, canonical 6 are well under the cap.

## Bucket profile observations

- **Matmul share grew uniformly** because encode + everything-else
  shrank more than matmul did. TinyLlama 38% → 58%, Qwen3-0.6B 32%
  → 50%, Qwen3-1.7B 36% → 56%, Mistral 50% → 77%, Llama-3.1-8B 59%
  → 82%, Qwen3-8B 56% → 79%, Gemma 4 47% → 48%. The Gemma 4 share
  barely moved because Gemma 4's 1040 dispatches/token keeps the
  non-matmul fraction large regardless of regime.
- **Per-dispatch encode cost in Pass 2 headless Chrome:** TinyLlama
  450 × 1.50 ms / 450 = ~3.3 µs/dispatch. Mistral 650 × 2.90 / 650
  = ~4.5 µs. Llama-3.1-8B 652 × 2.70 / 652 = ~4.1 µs. Gemma 4 1040
  × 4.10 / 1040 = ~3.9 µs. The dispatch-cost-per-op is now uniform
  across the fleet at 3.3-4.5 µs — prior interactive captures saw
  5+ µs with high variance.
- **Dispatch counts unchanged.** All 7 models match their
  registration-derived dispatch counts; no graph-structure
  regression since 2026-05-04.

## Follow-up / open items

- **Gemma 4 speed re-take with 7-9 runs** would tighten the headline
  tok/s number; current 38.6 p50 reflects real variance, not a code
  problem (matmul is stable). Not gating Stage 5; queue for the
  next sweep cycle.
- **Pre-existing pre-rebase tip log under** the noisy 2026-05-12 commit
  (`ca761e1`) listed the JSEP probe commits in full. The list is the
  same here — `b54503497` is the canonical webllm tip; everything
  above it on `webllm-browser-patches` is dormant in the WASM build.

---

## Pre-rebase tip log

```
ebc7c3d82 feat(jsep-probe): Stage 4.29 Probe 16 — CPU-side set_tensor weight-hash hook
1d1d64f76 feat(jsep-probe): Stage 4.28 Probe 15 — extend weight-hash allowlist 2 → 7
ef89f9314 feat(jsep): Stage 4.20 P10 — weight-upload FNV-1a probe in set_tensor
fc376580e jsep: Stage 4.16 P9 — EM_ASYNC_JS for ggml_jsep_read
ddeb2fb6e ggml-cpu,jsep: Stage 4.14 Probe 4 — post-compute CPU dst readback + JSEP get_tensor log
3b0e40d6f ggml-cpu,jsep: Stage 4.13 Probe 3 — CPU MUL_MAT capture + set_tensor name/data_addr
b50f92fd3 ggml-cpu,jsep: Stage 4.12 Probe 2 — CPU graph_compute + jsep set_tensor instrumentation
1bae15c27 ggml-jsep: Stage 4.11 graph_compute host_mirror entry/exit snapshot
7e92cc731 ggml-jsep: Stage 4.10 graph_compute slice instrumentation
e5b138abb ggml-jsep: Stage 4.9 H1-inverse — host→GPU writeback per-runOp pre-pass
e0fa38928 ggml-jsep: Stage 4.5 H1 — GPU→host writeback in graph_compute
9deefb954 ggml-jsep: F1 dual-resident host mirror — Phase 3 / Option A-prime Stage 4.4
53c66649f ggml-jsep: descriptor ABI buf_handle/offset split — Phase 3 / Option A-prime Stage 2
d0075e9a6 ggml-jsep: narrow supports_buft to JSEP-only — Phase 3 / Option A-prime Stage 1.5
d8b80dee2 ggml-jsep: SET_ROWS supports_op — Phase 3 / Option A-prime Stage 1
[--- canonical webllm-browser-patches tip below ---]
b54503497 ggml-webgpu: use wgpu::WaitAny under JSPI instead of polling loop
fc1f81242 ggml-webgpu: fix UB shift-by-32 in load_u32_at_src{,0} for aligned offsets
920c988a1 Revert "ggml-webgpu: specialize browser decode matmul dispatch"
db2a3c38d ggml-webgpu: specialize browser decode matmul dispatch
009119b07 ggml-webgpu: add opt-in browser graph profiling
702d40ee9 ggml-webgpu: notify browser async readback completion
55fba3670 ggml-webgpu: harden async readback request cleanup
846e0685e ggml-webgpu: add request-based browser readback API
ff362d4ae ggml-webgpu: browser + ASYNCIFY support bundle
17517488a ggml: iterative ggml_visit_parents_graph for WASM stack safety
a817a22bc ggml : implement fast walsh-hadamard transform for kv rotation (#21352) (#22631)
```

11 effective webllm patches on top of upstream `a817a22bc` (the 16
JSEP probe commits are dormant in this WASM build).
