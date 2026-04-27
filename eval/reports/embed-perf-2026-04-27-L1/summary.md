# §D Encoder Perf — Baseline

Date: 2026-04-27T21:52:33.157Z

## Single-text latency (p50 wall ms; non-profile)

| Model | Fixture | p50 ms | p90 ms | mean ms | reps |
|-------|---------|-------:|-------:|--------:|-----:|
| snowflake-arctic-embed-s-q0f32-b4 | short | 34.20 | 35.30 | 34.04 | 30 |
| snowflake-arctic-embed-s-q0f32-b4 | long | 26.30 | 27.30 | 25.89 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | short | 53.40 | 56.20 | 53.70 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | long | 37.90 | 39.50 | 36.55 | 30 |

## Batch throughput (texts/sec; non-profile)

| Model | Fixture | p50 wall ms | texts/sec | trials |
|-------|---------|------------:|----------:|-------:|
| snowflake-arctic-embed-s-q0f32-b4 | batchMixed | 1902.9 | 33.6 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | batchMixed | 2843.8 | 22.5 | 30 |
