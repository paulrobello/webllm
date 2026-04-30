# §D Embedder Perf — Baseline

Date: 2026-04-30T16:15:12.906Z

## Single-text latency (p50 wall ms; non-profile)

| Model | Fixture | p50 ms | p90 ms | mean ms | reps |
|-------|---------|-------:|-------:|--------:|-----:|
| qwen3-8b-iq3m | short | 1000.10 | 1000.20 | 1000.10 | 3 |
| qwen3-8b-iq3m | long | 1999.80 | 2000.20 | 1999.83 | 3 |

## Batch throughput (texts/sec; non-profile)

| Model | Fixture | p50 wall ms | texts/sec | trials |
|-------|---------|------------:|----------:|-------:|
