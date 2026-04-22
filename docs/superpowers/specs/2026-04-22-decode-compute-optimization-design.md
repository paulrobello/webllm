# Decode Compute Optimization Design

**Date:** 2026-04-22
**Status:** Proposed
**Priority order:** C (low-risk incremental profiling/cleanup) → A (TinyLlama decode throughput) → B (larger-model scalability)

---

## Goal

Reduce remaining browser decode latency now that async readback is no longer the dominant bottleneck, using a low-risk sequence that first improves hotspot attribution inside `graphCompute()`, then targets the most valuable compute-side optimization for single-token decode.

## Current verified state

Fresh verified browser perf after the readback work shows:

- ~`127.0 tok/s` in the final smoke-bench verification run
- separate profiling runs around `122.8–129.3 tok/s`
- `downloadResultMs` reduced to roughly `0.45–0.64 ms`
- `graphComputeMs` is now roughly `7.1–7.3 ms` and about 90% of single-token decode time

This means the next optimization step should focus on **decode compute**, not more readback work.

## Constraints

- Keep the current browser baseline stable:
  - `GGML_WEBGPU_JSPI=OFF`
  - `graphCompute()` remains async-capable and awaited
  - browser smoke policy still requires both visible pass output and clean console output
- Avoid broad architectural churn until we have finer-grained evidence inside `graphCompute()`
- Prefer changes that preserve the current public JS API and the current local `~/Repos/llama.cpp` patch branch workflow

## Options considered

### Option 1: Deeper decode compute profiling first
Add finer-grained profiling so `graphCompute()` is no longer a single opaque bucket.

**Pros**
- Lowest-risk path
- Best fit for current priority order
- Reduces guesswork before touching backend kernels

**Cons**
- Little or no direct speedup by itself
- Requires some profiling/plumbing work before optimization work starts

### Option 2: Single-token decode matmul optimization first
Directly optimize the hot `mul_mat`/decode kernel path in local `ggml-webgpu`.

**Pros**
- Best practical near-term throughput upside
- Targets the operator family most likely dominating current decode cost

**Cons**
- Higher backend risk
- Without deeper profiling, this is still partly informed guessing

### Option 3: Structural optimization first
Pursue browser-safe fused attention or backend command/graph-plan reuse immediately.

**Pros**
- Potentially high upside
- More relevant to later larger-model scaling

**Cons**
- Highest complexity/risk
- Not justified before improved attribution

## Recommendation

Proceed in three phases:

1. **Phase 1 — Decode compute attribution**
2. **Phase 2 — Matmul-focused decode optimization**
3. **Phase 3 — Reassess structural options**

This sequence matches the requested priority order:
- first lowest-risk cleanup/profiling
- then TinyLlama throughput improvements
- then broader scalability-oriented architecture work

---

## Design

### Phase 1 — Decode compute attribution

#### Objective
Break the current `graphComputeMs` bucket into enough detail to choose the next optimization target with confidence.

#### Scope
Add profiling or instrumentation that can answer:
- how much of decode compute is spent in matmul-heavy paths
- how much is spent in attention-specific work
- whether backend encode/bind/dispatch overhead is meaningfully visible
- whether any obvious pass or pipeline churn remains in the current browser decode path

#### Likely files
- `src/inference/model-inference.ts`
- `src/inference/ggml-wasm.ts`
- `src/wasm/webgpu-bridge.cpp`
- `src/wasm/CMakeLists.txt`
- possibly local `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`

#### Design boundaries
- Keep the existing top-level perf path working:
  - `smoke-test/real-model.html`
  - `eval/perf.ts`
- Do not introduce JSPI work here
- Do not bundle kernel rewrites into the attribution pass

#### Success criteria
After Phase 1, we should be able to say which one is dominant with evidence:
- decode matmuls
- attention subgraph
- backend command encoding/setup overhead

### Phase 2 — Matmul-focused decode optimization

#### Objective
Improve TinyLlama single-token decode throughput by optimizing the operator family most likely dominating `graphCompute()`.

#### Scope
Target decode-shaped `mul_mat` behavior in local `ggml-webgpu`, especially the many skinny/small matmuls in:
- q/k/v projections
- attention score/apply steps
- output projection
- FFN gate/up/down projections

#### Likely files
- primary: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
- possible secondary touchpoints:
  - shader generation/selection files in local `ggml-webgpu`
  - `src/inference/model-inference.ts` if a light model-side graph change is needed for better backend specialization

#### Design boundaries
- Prefer backend tuning/specialization over broad model-graph rewrites
- Preserve current JS API and smoke/perf harness workflow
- Keep changes narrow enough to benchmark incrementally after each meaningful backend change

#### Success criteria
- measurable TinyLlama decode throughput improvement over the current ~127 tok/s verification baseline
- no browser smoke regression
- clean console under smoke policy

### Phase 3 — Reassess structural options

#### Objective
Decide whether higher-complexity changes are justified after profiling and matmul tuning.

#### Candidates
- browser-safe fused attention / flash-attention-like path
- backend command/graph-plan reuse for 1-token decode
- larger-model-oriented follow-up work

#### Decision rule
Only pursue these if Phase 1 + Phase 2 evidence shows they are the best remaining leverage point.

---

## Error handling and safety

- Keep all current smoke validation requirements in place
- Verify every optimization against the current patched `~/Repos/llama.cpp` branch
- Treat any browser-specific console/runtime regression as a failure, even if throughput improves
- Avoid introducing changes that require speculative runtime support (for example JSPI) during this workstream

## Verification strategy

For each phase:

```bash
make checkall
make smoke-test
make smoke-bench PERF_RUNS=1 PERF_MODEL=tinyllama-1.1b-chat-q4_0
```

For backend-heavy changes, also verify the dependency branch:

```bash
git -C ~/Repos/llama.cpp branch --show-current
```

Expected:
- `webllm-browser-patches`

## Out of scope

- JSPI migration
- renewed readback-focused optimization work
- speculative decoding
- quantization tradeoff experiments as the main optimization strategy
- broad API redesign

## Implementation handoff

The implementation plan should be split into:
1. profiling/instrumentation pass
2. hotspot analysis checkpoint
3. backend matmul optimization pass
4. perf rebaseline and TODO refresh
5. optional structural follow-up only if justified by the new evidence
