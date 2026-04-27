# §D Encoder Perf — Baseline

Date: 2026-04-27T21:13:16.069Z

## Single-text latency (p50 wall ms; non-profile)

| Model | Fixture | p50 ms | p90 ms | mean ms | reps |
|-------|---------|-------:|-------:|--------:|-----:|
| snowflake-arctic-embed-s-q0f32-b4 | short | 34.00 | 34.40 | 33.45 | 30 |
| snowflake-arctic-embed-s-q0f32-b4 | long | 25.70 | 29.50 | 26.13 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | short | 52.00 | 52.40 | 51.82 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | long | 41.90 | 43.70 | 41.24 | 30 |

## Batch throughput (texts/sec; non-profile)

| Model | Fixture | p50 wall ms | texts/sec | trials |
|-------|---------|------------:|----------:|-------:|
| snowflake-arctic-embed-s-q0f32-b4 | batchMixed | 1909.5 | 33.5 | 30 |
| snowflake-arctic-embed-m-q0f32-b4 | batchMixed | 3011.2 | 21.3 | 30 |
