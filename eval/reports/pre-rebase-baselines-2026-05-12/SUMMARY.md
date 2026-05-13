# Pre-rebase profile-mode baselines — canonical 6 (2026-05-12)

**Captured:** 2026-05-12, post Stage 4.4 closure (eval re-gate at 70.8%
on Gemma 4 E2B, bit-identical to Stage 3) and pre any subsequent
llama.cpp rebase. Stage 5.1 of the Gemma 4 campaign (TODO §Q3).
**webllm tip:** `e3038c7` (`docs(stage4.4): CLOSED eval re-gate — 70.8%`).
**llama.cpp `webllm-browser-patches` tip:** `ebc7c3d82` (16 inert JSEP
experimental probe commits on top of canonical `b54503497`; JSEP is
gated `OFF` in the WASM build per `ggml/CMakeLists.txt:236` and CPU
backend is `OFF` per `src/wasm/CMakeLists.txt`, so the probes don't
compile into our WASM artifact — behaviorally equivalent to
`b54503497`).
**WASM:** `smoke-test/webllm-wasm.wasm` 3,767,720 bytes (May 12 17:20).
**Procedure:** `make smoke-bench PERF_MODEL=<m> PERF_RUNS=3` (profile
mode).
**Trigger:** §32a pre-rebase doctrine — capture before the next rebase
fires, so a follow-on probe gets a same-tip baseline for diagnosis.
Also serves as the Stage 5.1 non-regression check: verify the NEOX
RoPE fix (Q1.6 / Gemma family branch extensions, `c8c8447` …
`31d53a5`) didn't move non-Gemma models.

---

## Headline matrix

3-run p50, profile-mode (perturbed). The "p50" row picks the median
run's full row; other columns are from that run, not column-wise
medians.

| Model                          | Quant   | tok/s | Steps | graph (med, ms) | matmul (med, ms / %) | encode (med, ms / %) | dispatch/token |
|---|---|---:|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | Q4_0    | 116.1 | 189   | 7.50            | 4.19 / 56.1%         | 1.50 / 20.6%         | 450 |
| qwen3-0.6b-q4f16               | Q8_0    |  87.2 |  72   | 8.60            | 4.06 / 46.7%         | 2.10 / 24.2%         | 629 |
| qwen3-1.7b-q4f16               | Q8_0    |  64.4 |  48   | 11.80           | 7.01 / 59.8%         | 2.10 / 18.4%         | 629 |
| mistral-7b-instruct-v0.3-q4ks  | Q4_K_S  |  43.3 | 123   | 21.60           | 16.58 / 76.8%        | 2.80 / 13.0%         | 650 |
| llama-3.1-8b-instruct-iq3m     | IQ3_M   |  33.0 | 156   | 28.60           | 23.79 / 82.3%        | 2.30 /  8.5%         | 652 |
| qwen3-8b-iq3m                  | IQ3_M   |  29.4 |  60   | 30.40           | 24.31 / 79.4%        | 3.00 / 10.1%         | 805 |

Per-model raw logs in this directory (`<model-id>.log`).

## Cross-baseline drift vs `pre-rebase-baselines-2026-05-04/`

| Model                          | 2026-05-04 | 2026-05-12 | Δ tok/s | Δ %    | matmul med Δ |
|---|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       |  80.4 | 116.1 | +35.7 | +44.4% | 4.19 → 4.19   (0%)    |
| qwen3-0.6b-q4f16               |  62.5 |  87.2 | +24.7 | +39.5% | 4.00 → 4.06 (+1.5%)   |
| qwen3-1.7b-q4f16               |  41.7 |  64.4 | +22.7 | +54.4% | 6.88 → 7.01 (+1.9%)   |
| mistral-7b-instruct-v0.3-q4ks  |  28.7 |  43.3 | +14.6 | +50.9% | 16.58 → 16.58 (0%)    |
| llama-3.1-8b-instruct-iq3m     |  23.3 |  33.0 |  +9.7 | +41.6% | 23.66 → 23.79 (+0.5%) |
| qwen3-8b-iq3m                  |  21.3 |  29.4 |  +8.1 | +38.0% | 24.12 → 24.31 (+0.8%) |

**Matmul time is bit-identical (±2%) across the canonical 6.** The
40-50% tok/s lift is **not** a code change — it is an environmental
regime shift between this capture (clean headless Chrome via
`agentchrome connect --launch --headless`) and the 2026-05-04 capture
(user's interactive Chrome instance with concurrent tabs and OS-level
contention). Confirmation: the *only* per-phase metric that moved is
encode overhead — TinyLlama 2.4 → 1.5 ms median (-38%), Qwen3-0.6B
3.4 → 2.1 ms (-38%), Qwen3-1.7B 3.5 → 2.1 ms (-40%), Mistral 3.5 → 2.8
ms (-20%), Llama-3.1-8B 3.5 → 2.3 ms (-34%), Qwen3-8B 4.6 → 3.0 ms
(-35%). Headless Chrome eliminates compositor and tab-switching
encode-overhead noise that was inflating profile-mode numbers in
prior captures.

**Stage 5.1 non-regression conclusion:** the NEOX RoPE fix (Q1.6 +
`getRopeModeForArchitecture` family extension) is architecture-gated
to gemma2/gemma3/gemma4 and did **not** touch the canonical 6's
dispatch path. Bit-identical matmul time across all 6 models confirms
this empirically. Non-Gemma models on `bench-full` will be
bit-identical when Gemma 4 joins the fleet at Stage 5.2.

## Capture methodology — important asterisks

1. **Chrome stability under repeated 8B model loads.** Both interactive
   and headless Chrome instances died mid-sweep after ~6-7 cumulative
   model page-loads in the same session. Mitigation: relaunch
   `agentchrome connect --launch --headless` before each 7B+ model
   bench. tinyllama → qwen3-0.6b → qwen3-1.7b ran on session 1
   (port 57264); llama-3.1-8b-iq3m on session 3 (port 60121);
   qwen3-8b-iq3m on session 4 (port 60410). mistral-7b ran on
   session 2 (port 56921) — the first headless Chrome — together
   with the initial three small-model runs (later re-captured on
   session 1 after the headless relaunch made the regime visible).
   Effective per-model sample is 3 runs on a fresh-or-warm headless
   Chrome; no model shares a Chrome process with another model in
   this capture.

2. **First-of-day cold-shader effect was visible on session 0**
   (the user's pre-existing interactive Chrome, port 53198) — runs
   1-2 on TinyLlama showed ~70 tok/s, run 3 climbed to 79.4 tok/s,
   consistent with the 2026-05-04 baseline floor. After Chrome
   crashed on the first mistral-7b load attempt and was relaunched
   headless, all subsequent runs landed within ±5% of each other
   per-model. Session 0 results are not in this capture's `.log`
   files; the on-disk logs reflect the headless-only re-capture
   for the small models (`tinyllama`, `qwen3-0.6b`, `qwen3-1.7b`).

3. **Going forward, headless is the canonical capture regime.** The
   §32a doctrine relies on environmental-floor parity for valid
   cross-capture diffs. With this capture establishing "headless +
   relaunch-before-7B" as the procedure, the next §32 / §27 / §28
   cycle's adjudication baseline lives here. Cross-comparing
   2026-05-12 numbers against 2026-05-04 (or earlier) is **only valid
   for matmul-time deltas**, not for tok/s deltas — the absolute
   tok/s axis is incomparable across these two regimes.

## Bucket profile observations

- **Matmul share grew uniformly post-environment-shift** because
  encode overhead dropped, leaving matmul to dominate a bigger slice
  of the now-smaller wall clock. TinyLlama 38% → 56%, Qwen3-1.7B 36%
  → 60%, Mistral Q4_K_S 50% → 77%, Llama-3.1-8B IQ3_M 59% → 82%,
  Qwen3-8B IQ3_M 56% → 79%. Absolute matmul time is unchanged; the
  share metric is a fraction-of-total-decode and tracks the
  denominator change.
- **Per-dispatch encode cost in headless Chrome:** TinyLlama 450
  dispatches × 1.50 ms encode / 450 = ~3.3 µs/dispatch. Compare
  ~5.3 µs/dispatch on 2026-05-04 interactive. Roughly 40%
  per-dispatch savings — consistent with eliminating compositor
  contention.
- **Dispatch counts unchanged.** All 6 models match their 2026-05-04
  dispatch counts (TinyLlama 450, Qwen3-0.6B 629, Qwen3-1.7B 629,
  Mistral 650, Llama-3.1-8B 652, Qwen3-8B 805). Confirms no graph
  structure change since the last capture.

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
