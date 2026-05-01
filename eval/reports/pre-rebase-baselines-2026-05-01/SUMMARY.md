# Pre-rebase profile-mode baselines — canonical 6 (2026-05-01)

**Captured:** 2026-05-01, before pulling upstream
`c3c150539` (ggml-webgpu: Fix vectorized handling in mul-mat and
mul-mat-id, #22578) and `aab68217b` (upscale shader, #22419).
**Build (pre-rebase):** `webllm-browser-patches` tip `a45089d5a`
(11-patch stack on upstream `a95a11e5b`); WASM
2,482,377 / 2,526,224 bytes (mem32 / mem64).
**Procedure:** `make smoke-bench PERF_MODEL=<m> PERF_RUNS=3`
(profile mode).
**Trigger:** §32a doctrine — capture same-day same-tip control
*before* the rebase, so the post-rebase sweep can be compared
against a baseline that matches the local environmental floor
rather than the 4-day-old `pre-rebase-baselines-2026-04-28/`
matrix.

---

## Headline matrix

3-run median, profile-mode (perturbed). All medians are column-wise
medians of the 3 runs. Model-id "Quant" reflects the actual GGUF
file shipped to the engine (registration name vs file may differ —
see TODO header).

| Model                          | Quant   | tok/s | Steps | graph (med, ms) | matmul (med, ms / %) | encode (med, ms / %) | dispatch/token |
|---|---|---:|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | Q4_0    | 80.4  | 189   | 10.80           | 4.19 / 38.4%         | 2.40 / 22.1%         | 450 |
| qwen3-0.6b-q4f16               | Q8_0    | 62.5  | 72    | 11.80           | 4.00 / 31.9%         | 3.40 / 27.3%         | 629 |
| qwen3-1.7b-q4f16               | Q8_0    | 41.7  | 48    | 18.60           | 6.88 / 35.8%         | 3.50 / 18.3%         | 629 |
| mistral-7b-instruct-v0.3-q4ks  | Q4_K_S  | 28.7  | 189   | 33.10           | 16.58 / 49.6%        | 3.50 / 10.6%         | 650 |
| llama-3.1-8b-instruct-iq3m     | IQ3_M   | 23.3  | 156   | 40.30           | 23.66 / 58.5%        | 3.50 /  8.7%         | 652 |
| qwen3-8b-iq3m                  | IQ3_M   | 21.3  | 60    | 42.50           | 24.12 / 55.7%        | 4.60 / 11.4%         | 805 |

Per-model raw logs in this directory (`<model-id>.log`).

## Cross-day drift vs `pre-rebase-baselines-2026-04-28/`

The §32a doctrine exists precisely because of cases like this:
4 days, an in-flight rebase or two, and the environmental floor
shifts in profile mode. Smaller models drifted more (less wall time
per profile timestamp sample → noise dominates):

| Model                          | 2026-04-28 | 2026-05-01 | Δ tok/s | Δ %    |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 87.9       | 80.4       | -7.5    | -8.5%  |
| qwen3-0.6b-q4f16               | 68.2       | 62.5       | -5.7    | -8.4%  |
| qwen3-1.7b-q4f16               | 44.0       | 41.7       | -2.3    | -5.2%  |
| mistral-7b-instruct-v0.3-q4ks  | 29.7       | 28.7       | -1.0    | -3.4%  |
| llama-3.1-8b-instruct-iq3m     | 23.5       | 23.3       | -0.2    | -0.9%  |
| qwen3-8b-iq3m                  | 21.8       | 21.3       | -0.5    | -2.3%  |

This is not a code regression — it's the floor-drift signal §32a
warned about. Without this same-day capture, comparing 2026-04-28
against the post-rebase sweep would have shown spurious "regressions"
on the small models even where the rebase is a clean improvement.
**Use these 2026-05-01 numbers — not the 2026-04-28 numbers — when
adjudicating the post-rebase sweep at
`eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md`.**

## Bucket profile observations

- **Encoder share preserved.** TinyLlama 22%, Qwen3-0.6B 27%, 7-8B
  9-11%. Same shape as the 2026-04-28 capture; per-dispatch encoder
  cost ~5.2-5.7 µs continues to hold.
- **Matmul share preserved.** TinyLlama 38%, Qwen3-1.7B 36%, Mistral
  Q4_K_S 50%, IQ3_M 8B 56-58%. The 7B+ K-quant / IQ-quant compute
  share is unchanged from 2026-04-28.
- **Drift is dominated by `graphCompute` p90 widening at small
  models** — TinyLlama p90 12.30 vs 11.30 in 2026-04-28; the long
  tail moved without the median shifting much. Likely background
  GPU contention on the local box; no actionable code issue.

---

## Pre-rebase tip log

```
e29753286 ggml-webgpu: fix UB shift-by-32 (post-rebase)
...
a45089d5a ggml-webgpu: fix UB shift-by-32 (pre-rebase tip)
dccef7c15 ggml-webgpu: split LAYER_NORM accumulation loop from legacy norms
96a89c4ea ggml-webgpu: add GGML_OP_NORM (LayerNorm) support
7757db139 Revert "ggml-webgpu: specialize browser decode matmul dispatch"
3e25ef547 ggml-webgpu: specialize browser decode matmul dispatch
0dbd678d3 ggml-webgpu: add opt-in browser graph profiling
c0354355c ggml-webgpu: notify browser async readback completion
b1fe593ff ggml-webgpu: harden async readback request cleanup
6d70254c7 ggml-webgpu: add request-based browser readback API
7bbb2416c ggml-webgpu: browser + ASYNCIFY support bundle
b3c444219 ggml: iterative ggml_visit_parents_graph for WASM stack safety
a95a11e5b ggml-webgpu: Improve performance of mat-vec and mat-mat for MUL_MAT_ID (#22464)
```

11 webllm patches on top of upstream `a95a11e5b`.
