# Probe 9c — Hitch warmup

**Model:** qwen3-8b-iq3m
**Date:** 2026-05-01
**Frame-probe calls per scenario:** 5

## Method

Two scenarios on the same page-load cadence. Control runs the existing `?frameProbeCalls=5` multi-call probe with no modification. Warmup runs the same probe with a 4-token throwaway `chatCompletion` inserted between baseline-rAF window close and the timed probe start (`?frameProbeWarmup=1`). The 500ms inter-call settle that already exists between subsequent calls is also applied after the warmup throwaway.

Threshold: warmup brings call-0 `decode.max` into the band of subsequent calls' `decode.max` (control's call-1..4 median).

## Control (no warmup) — per-call frame stats

```
  call 0: prefill_max=9.4ms, decode_p95=9.2ms, decode_max=41.7ms
  call 1: prefill_max=9.2ms, decode_p95=9.1ms, decode_max=41.7ms
  call 2: prefill_max=9.3ms, decode_p95=9.2ms, decode_max=41.7ms
  call 3: prefill_max=9.3ms, decode_p95=8.9ms, decode_max=41.6ms
  call 4: prefill_max=9.3ms, decode_p95=9.1ms, decode_max=50.0ms
```

## Warmup — per-call frame stats

```
  call 0: prefill_max=9.3ms, decode_p95=9.2ms, decode_max=41.6ms
  call 1: prefill_max=9.2ms, decode_p95=8.9ms, decode_max=41.6ms
  call 2: prefill_max=275.0ms, decode_p95=9.1ms, decode_max=58.3ms
  call 3: prefill_max=9.3ms, decode_p95=9.2ms, decode_max=42.1ms
  call 4: prefill_max=9.2ms, decode_p95=9.1ms, decode_max=40.8ms
```

## Headline comparison

- Control call-0 decode_max: **41.7 ms**
- Warmup  call-0 decode_max: **41.6 ms**
- Control subsequent (calls 1..4) decode_max median: **41.7 ms**

## Verdict

**FAIL** — warmup does not move the needle on call-0 decode_max.
Decision: hitch is not driven by per-shape JIT; warmup-throwaway is not the right intervention. Investigate alternative (KV-cache pre-allocation, frame-pacing changes, worker migration).
