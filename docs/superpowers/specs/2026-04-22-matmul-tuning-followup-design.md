# Matmul Tuning Follow-up Design

**Date:** 2026-04-22
**Status:** Proposed
**Scope:** One narrow matmul hypothesis cycle only

---

## Goal

Define a tightly scoped follow-up optimization cycle that tests exactly one browser WebGPU matmul tuning hypothesis against the current accepted decode-compute baseline, keeping the work attributable, reversible, and smoke-safe.

## Current verified state

The completed decode-compute profiling cycle established that the remaining decode bottleneck is still inside backend compute rather than readback:

- Task 5 profiled investigation baseline: `93.5 tok/s`
- median run: `184 ms` prefill, `331 ms` decode, `2027 ms` full page wall time
- `graphComputeMs`: `9.96 ms` mean / `91.8%` of decode-step time
- `downloadResultMs`: `0.62 ms` mean / `5.7%`
- `backendMatmulMs`: `4.02 ms` mean / `40.4%` of graph time
- `backendEncodeOverheadMs`: `2.81 ms` mean / `28.2%`
- `backendAttentionMs`: `0.40 ms` mean / `4.0%`

This means matmul remains the strongest current optimization target, but the previous retained cycle also showed that broad or weakly differentiated backend tweaks should be rejected quickly when they do not produce a clear signal.

## Constraints

- Keep the current browser baseline stable:
  - browser smoke still requires visible success output and no relevant console/runtime errors
  - `graphCompute()` remains async-capable and awaited
  - no JSPI work in this cycle
- Limit scope to one hypothesis only:
  - one chosen matmul specialization target
  - one backend change set
  - one accept-or-revert decision
- Preserve the current benchmark path:
  - `make smoke-bench PERF_RUNS=3 PERF_MODEL=tinyllama-1.1b-chat-q4_0`
- Do not treat profile-mode throughput as the new shipping baseline
- Keep structural follow-up deferred unless a retained optimization win clearly justifies reopening that question

## Options considered

### Option 1: Single decode-shape dispatch specialization
Choose one decode-heavy matmul family and tune only its dispatch/workgroup selection for the single-token browser path.

**Pros**
- Best fit for one narrow cycle
- Smallest rollback surface
- Most attributable result

**Cons**
- May show no signal if dispatch policy is not the real limiter

### Option 2: Single-path shader rewrite
Target the same operator family but rewrite shader logic or data/layout behavior rather than only dispatch policy.

**Pros**
- Higher upside if the shader is the actual bottleneck

**Cons**
- Higher backend risk
- Harder to keep attributable
- More likely to regress browser stability

### Option 3: Adjacent encode-overhead reduction instead of matmul tuning
Use the same evidence but optimize backend setup around matmul rather than the kernel path itself.

**Pros**
- Could pay off if matmul timing partially reflects surrounding overhead

**Cons**
- No longer a pure matmul follow-up
- Less aligned with the current hotspot ranking

## Recommendation

Proceed with **Option 1: single decode-shape dispatch specialization**.

This is the best fit for the current evidence and the requested one-cycle scope:

- `backendMatmulMs` is still the largest timed backend bucket
- the previous cycle already proved that unconvincing backend tweaks should be reverted rather than accumulated
- a dispatch-specialization experiment gives the cleanest accept/revert decision while minimizing browser-risk surface area

---

## Design

### Scope

The follow-up plan should define exactly one optimization loop:

1. confirm the current hotspot using the accepted Task 5 baseline
2. choose one decode-shaped matmul specialization hypothesis in local `ggml-webgpu`
3. implement exactly one narrow backend change
4. rebuild and run smoke/perf verification
5. retain the change only if it shows a meaningful positive signal and clean browser behavior

This is not a multi-hypothesis campaign. If the chosen hypothesis is flat, noisy, or regressive, the cycle ends with a revert and a documented no-signal outcome.

### Target hypothesis shape

The default target class for this cycle is:

- specialize browser WebGPU dispatch policy for a single-token decode matmul path in local `ggml-webgpu`

The implementation plan may refine the exact path, but it must stay within a single chosen family and a single concrete hypothesis such as:

- decode-specific workgroup/layout choice
- decode-specific dispatch geometry selection
- removal of one path-specific setup branch from the chosen decode matmul route

The plan must not bundle multiple matmul hypotheses together.

### Architecture boundaries

Primary code touchpoint:

- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`

Secondary files are allowed only if they are strictly required to support the one chosen backend hypothesis, for example a directly related shader-selection file.

Within the `webllm` repo, changes should be limited to:

- documentation refresh if a retained win changes the current project story
- benchmark/reporting updates only if required to explain the result clearly

This cycle should not introduce:

- new profiling features unless necessary to validate the chosen hypothesis
- graph reuse work
- fused attention work
- structural backend plan reuse
- multiple backend optimizations in one pass

### Decision flow

Inputs:

- the accepted Task 5 profiled investigation baseline
- the current hotspot ordering (`backendMatmulMs` first, `backendEncodeOverheadMs` second)
- the known prior no-signal optimization attempt

Decision rule:

- if the narrow hypothesis improves throughput or the target hotspot meaningfully, while smoke remains clean, retain it
- if the signal is weak, ambiguous, or negative, revert it
- do not start a second hypothesis in the same plan

### Error handling and safety

Treat any of the following as failure conditions:

- smoke-test visible failure
- relevant browser console/runtime errors
- benchmark instability caused by the change
- no meaningful positive signal
- scope creep into multi-hypothesis or structural work

Safety rules:

- one backend hypothesis only
- benchmark with the existing harness
- revert no-signal changes
- keep the current profiling-vs-shipping-baseline caveat explicit
- defer structural follow-up by default

## Verification strategy

The plan should require:

### Pre-change baseline

Capture the current baseline with the existing benchmark path before changing the backend.

### Rebuild and smoke

```bash
make wasm-build
make smoke-test
```

### Full verification

```bash
make checkall
make smoke-bench PERF_RUNS=3 PERF_MODEL=tinyllama-1.1b-chat-q4_0
```

### Acceptance gate

Evaluate:

- throughput change (`tok/s`)
- hotspot deltas from the current profiling output
- clean browser page output
- clean relevant console output

If the change is retained, refresh project docs honestly. If the change is reverted, record the no-signal outcome briefly and stop.

## Out of scope

- JSPI migration
- renewed readback-focused optimization work
- multiple matmul hypotheses in one cycle
- encode-overhead optimization as the primary work item
- graph reuse or fused-attention structural work
- larger-model optimization work

## Implementation handoff

The implementation plan should be split into:

1. baseline capture and hypothesis lock-in
2. one narrow backend matmul change
3. rebuild and verification
4. retain-or-revert decision
5. doc refresh only if warranted by the retained outcome
