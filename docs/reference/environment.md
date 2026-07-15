# Environment Variables

Reference for every environment variable and Makefile override the benchmark
harnesses, dev servers, and WASM build consume. Use this when tuning
timeouts, forcing a WASM variant, pinning the accuracy-pass temperature, or
pointing a harness at the live dashboard.

## Table of Contents

- [Runtime environment variables](#runtime-environment-variables)
- [Build-time CMake variables](#build-time-cmake-variables)
- [Makefile overrides](#makefile-overrides)
- [Related Documentation](#related-documentation)

## Runtime environment variables

These are read from `process.env` by the Bun harnesses under `eval/` and by
the dev servers. Unless noted, they are optional.

| Variable | Consumed by | Default | Effect |
|----------|-------------|---------|--------|
| `WEBLLM_LIVE_BENCH_URL` | `eval/live-client.ts`, `eval/import-reports.ts`, `eval/cli.ts`, `eval/browser-eval.ts`, `eval/chat-smoke.ts`, `eval/bench.ts` | unset | Base URL of the live dashboard backend (normally `http://localhost:8033`). When set, harnesses stream eval/smoke events to it via POST. `make bench-browser-eval` requires it and fails fast if unset. `make import-reports` falls back to it when `--url` is not passed. |
| `WEBLLM_BENCH_SESSION_ID` | `eval/bench.ts`, `eval/cli.ts`, `eval/browser-eval.ts` | unset | Parent bench-session id threaded into each per-model `eval_started` event so the dashboard can roll up overall progress (model X/Y · task A/B). Minted by `eval/bench.ts` per invocation. |
| `WEBLLM_BENCH_EVAL_TEMPERATURE` | `eval/bench.ts`, `eval/cli.ts`, `eval/browser-eval.ts` | unset (profile temperature applies) | Overrides the sampling temperature for the accuracy pass without redefining every profile. Precedence: CLI `-t` > this env var > profile > default. See the comparability warning below. |
| `WEBLLM_HARD_TIMEOUT_MS` | `eval/browser-eval.ts` | `600000` (10 min) | Hard ceiling on a browser-eval run. Raise for >4 GiB MEMORY64 targets where cold-cache GGUF fetch + the prompt set can total 15-20 min. |
| `WEBLLM_STALL_TIMEOUT_MS` | `eval/browser-eval.ts` | `180000` (3 min) | Bails when the page stops making progress (`__benchStatus` never published, or `completedTasks` stops advancing — e.g. a WASM abort). Clock resets on first status or task advance. Raise for >4 GiB MEMORY64 targets where a cold GGUF fetch can exceed 3 min before the first status. |
| `WEBLLM_SMOKE_RUNS_DIR` | `eval/smoke-runs.ts` | `eval/reports/smoke-runs` | Directory smoke-run JSON records are written to (the durable archive independent of the dashboard SQLite store). |
| `WEBLLM_WASM_VARIANT` | `eval/perf.ts` | unset (wasm32) | When set to `mem64`, the smoke page picks the wasm64 binary via `?wasm=mem64`. `make smoke-bench` threads this from the `WASM_VARIANT` Makefile var. |

> **Warning — comparability.** `WEBLLM_BENCH_EVAL_TEMPERATURE` changes result
> comparability. The project's greedy-temp doctrine (see `CLAUDE.md`)
> default-pins accuracy benches to temperature 0 so a single pass per model
> gives stable signal; cross-day or cross-run comparisons are only valid at
> the same temperature. If you set this env var to a non-zero value, tag the
> run accordingly on the dashboard — pre-cutover (April 2026 and earlier)
> and greedy baselines must not be cross-compared.

## Build-time CMake variables

These are CMake cache variables passed via `-D` (or set as Make variables
that the WASM build targets forward). They configure the WASM artifact, not
the runtime.

| Variable | Consumer | Default | Effect |
|----------|----------|---------|--------|
| `WEBLLM_ASSERTIONS` | `Makefile`, `src/wasm/CMakeLists.txt` | `0` | Enables Emscripten `-sASSERTIONS=1` in the WASM build (slower, preserves abort messages). Set `WEBLLM_ASSERTIONS=1` on the make invocation, or use `make wasm-build-debug` which forces it to 1. |
| `WEBLLM_BUILD_MEM64` | `src/wasm/CMakeLists.txt`, `Makefile` (`build-mem64`) | `OFF` | Builds the wasm64 `webllm-wasm-mem64` target instead of `webllm-wasm`. Must combine with `-DCMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1` so ggml is also built as wasm64. Used for the MEMORY64 cap probe and >4 GiB model support. |
| `WEBLLM_BACKEND` | `src/wasm/CMakeLists.txt`, `Makefile` (`wasm-build-jsep`) | `default` | Selects the ggml backend mix: `default` (canonical ASYNCIFY-era ggml-webgpu) or `jsep` (experimental JSEP-style prototype; `@experimental`, ships without semver guarantees). |

## Makefile overrides

These are Make variables (`make <target> VAR=value`), not process-env. They
shape `make` targets but are not read by the TS code directly except where
noted (e.g. `WASM_VARIANT` is forwarded into `WEBLLM_WASM_VARIANT`).

| Variable | Default | Effect |
|----------|---------|--------|
| `SMOKE_PORT` | `8031` | Port the smoke-test static server (`make smoke-serve`) listens on. |
| `DASHBOARD_PORT` | `8033` | Port the live dashboard SSE backend (`make dashboard-serve`) listens on. |
| `DASHBOARD_HOST` | `0.0.0.0` (Makefile) | Host the dashboard binds. Note: the servers in `eval/live-server.ts` and `eval/smoke-serve.ts` default their own `DEFAULT_HOST` to loopback; pass `--host` explicitly to bind elsewhere. |
| `DASHBOARD_DB` | `eval/reports/smoke-runs.db` | SQLite path for the dashboard's persisted run/eval data. |
| `PERF_MODEL` | `tinyllama-1.1b-chat-q4_0` | Model id exercised by `make smoke-bench`. |
| `PERF_RUNS` | `3` | Number of runs `make smoke-bench` averages over. |
| `PERF_DRAFTER` | unset | Optional drafter model id forwarded to `eval/perf.ts` for speculative-decode probes. |
| `PERF_EXTRA` | unset | Extra args forwarded to `eval/perf.ts`. |
| `MODEL` | `hermes-3-llama-3.2-3b-q4f16` | Default model id for model-loading targets that take one. |
| `WASM_VARIANT` | unset (wasm32) | Forwarded into `WEBLLM_WASM_VARIANT` for `make smoke-bench`; set to `mem64` to exercise the wasm64 binary. |
| `PROFILE` | unset | Profile name passed to `make bench-browser-eval` (selects model + sampling defaults from `eval/smoke-profiles.ts`). |

> **Note:** `make help` lists every target with descriptions; it is the
> single source of truth for what each target does and which variables it
> honors.

## Related Documentation

- [`docs/BENCHMARKS.md`](../BENCHMARKS.md) — benchmark methodology, dimensions, and metrics
- [`CLAUDE.md`](../../CLAUDE.md) — greedy-temp doctrine, dashboard ingest paths, port reservations
- [`Makefile`](../../Makefile) — canonical target definitions (`make help`)
