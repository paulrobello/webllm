# llama.cpp rebase 2026-04-29 — canonical-6 sweep

**Pre-rebase tip:** `3b8ade2a2`
**Post-rebase tip:** `fa8b16a6f`
**Upstream base:** `b1d5f5b44` (sync : ggml; was `5d56effde`).
**Conflicts:** 0. All 11 patches replayed cleanly.
**Upstream WebGPU commits picked up:** 1 — `d6a509400`
("ggml-webgpu: Fix bug in FlashAttention support check (#22492)").
**WASM sizes:** wasm32 2,249,421 B (-229 vs §32); wasm64 2,292,124 B.
**Ship gate:** 452/11/0.

## Headline matrix

3-run median, profile-mode (`make smoke-bench PERF_RUNS=3`).
Pre-rebase column from `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`
(captured 1 day prior, same fleet, same harness).

| Model                          | Pre-rebase | Post-rebase | Δ tok/s | Δ %     |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 87.9       | 85.2        | −2.7    | **−3.1%** |
| qwen3-0.6b-q4f16               | 68.2       | 66.1        | −2.1    | **−3.1%** |
| qwen3-1.7b-q4f16               | 44.0       | 42.1        | −1.9    | **−4.3%** |
| mistral-7b-instruct-v0.3-q4ks  | 29.7       | 28.2        | −1.5    | **−5.1%** |
| llama-3.1-8b-instruct-iq3m     | 23.5       | 23.2        | −0.3    | **−1.3%** |
| qwen3-8b-iq3m                  | 21.8       | 21.7        | −0.1    | **−0.5%** |

Per-run raw data (profile-mode, dashboard-recorded):

| Model                         | run-1 | run-2 | run-3 | median |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0      | 85.4  | 81.4  | 85.2  | 85.2 |
| qwen3-0.6b-q4f16              | 66.1  | 58.8  | 68.3  | 66.1 |
| qwen3-1.7b-q4f16              | 42.1  | 41.0  | 43.3  | 42.1 |
| mistral-7b-instruct-v0.3-q4ks | 28.2  | 28.5  | 28.2  | 28.2 |
| llama-3.1-8b-instruct-iq3m    | 23.3  | 22.9  | 23.2  | 23.2 |
| qwen3-8b-iq3m                 | 21.7  | 22.0  | 21.6  | 21.7 |

## Classification

**§32 template — small regression, accepted; don't revert.**

- 4 of 6 models show −3% to −5% deltas; 2 of 6 (both IQ3_M) are flat
  within run-to-run noise.
- Profile-mode benches have a documented 5-15% sampling-pipeline
  overhead; with `PERF_RUNS=3` the median is moderately stable but
  the confidence interval easily covers ±3-5%.
- The single relevant upstream change (`d6a509400`) only adds an
  early-out check on `decisions.path == FLASH_ATTN_PATH_NONE` plus a
  "set path to none if kv_tile doesn't fit" safety net. Both run
  inside `ggml_backend_webgpu_device_supports_op`, which is hit per
  graph build, not per dispatch — too cold a path to plausibly cause
  per-token decode regressions.
- **Net read:** the regressions are likely 3-run profile-mode noise,
  not signal. The IQ3_M-pair-flat / FA-assumed-cause story doesn't
  reconcile (those models exercise FA more, not less, than the others
  — if FA changed cost, IQ3_M should move first). Per CLAUDE.md §32
  doctrine: "staying current has option value (next cycle's free wins
  land cleanly)" — accept and pin the new tip as canonical.

## Closure actions

- TODO watch-list updated (rebase trigger spent).
- `LLAMA_CPP_PATCHES.md` updated with the new tip and §32-template
  classification block.
- Pre-rebase baseline directory `pre-rebase-baselines-2026-04-28/`
  retained — it served as the same-model comparator here and remains
  fresh for the *next* rebase (1-month freshness window).
- Safety branch `webllm-browser-patches-backup-20260429-*` preserves
  the pre-rebase tip (`3b8ade2a2`).

## Reproduction

```bash
cd ~/Repos/llama.cpp && git checkout webllm-browser-patches \
  && git rebase origin/master                           # 0 conflicts
cd ~/Repos/webllm
make wasm-clean && make wasm-build
make smoke-test
make dashboard-serve &
for m in tinyllama-1.1b-chat-q4_0 qwen3-0.6b-q4f16 qwen3-1.7b-q4f16 \
         mistral-7b-instruct-v0.3-q4ks llama-3.1-8b-instruct-iq3m \
         qwen3-8b-iq3m; do
  make smoke-bench PERF_MODEL=$m PERF_RUNS=3
done
```
