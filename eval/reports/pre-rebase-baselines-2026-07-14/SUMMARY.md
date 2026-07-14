# Pre-rebase baselines — 2026-07-14

Profile-mode (`make smoke-bench PERF_MODEL=<m> PERF_RUNS=3 --profile`)
baselines captured 2026-07-14 against local tip `4192e05ba` (the
2026-05-12 cycle's post-tip), BEFORE the 2026-07-14 rebase onto
upstream `bf2c86ddc`. Same-day same-tip control for the post-rebase
sweep classification (§32a doctrine).

| Model | tok/s (median of 3) |
|-------|--------------------:|
| tinyllama-1.1b-chat-q4_0 | 87.1 |
| qwen3-0.6b-q4f16 | 72.0 |
| qwen3-1.7b-q4f16 | 52.1 |
| mistral-7b-instruct-v0.3-q4ks | 29.7 |
| llama-3.1-8b-instruct-iq3m | 22.8 |
| qwen3-8b-iq3m | 21.3 |

All exit 0. qwen3-1.7b run-3 dip (37.4 vs 52.1) is the thermal-contamination
signature noted in the 2026-05-12 cycle. Raw per-run output: `baseline-run.log`.
Post-rebase sweep + classification: `../llama-cpp-rebase-2026-07-14/SUMMARY.md`.
