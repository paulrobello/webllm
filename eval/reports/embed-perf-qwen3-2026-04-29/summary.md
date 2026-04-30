# §D Embedder Perf — Baseline

Date: 2026-04-30T04:44:11.086Z

## Single-text latency (p50 wall ms; non-profile)

| Model | Fixture | p50 ms | p90 ms | mean ms | reps |
|-------|---------|-------:|-------:|--------:|-----:|
| qwen3-embedding-0.6b-hyb | short | 77.20 | 79.40 | 77.20 | 30 |
| qwen3-embedding-0.6b-hyb | long | 113.60 | 117.40 | 114.04 | 30 |

## Batch throughput (texts/sec; non-profile)

| Model | Fixture | p50 wall ms | texts/sec | trials |
|-------|---------|------------:|----------:|-------:|
| qwen3-embedding-0.6b-hyb | batchMixed | 6134.7 | 10.4 | 30 |
