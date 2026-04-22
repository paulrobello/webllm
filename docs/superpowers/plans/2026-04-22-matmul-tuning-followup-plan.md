# Matmul Tuning Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test exactly one narrow browser WebGPU matmul tuning hypothesis against the current accepted decode-compute baseline, retaining it only if it shows a meaningful positive signal without regressing smoke stability.

**Architecture:** Run a single-hypothesis loop. First, capture a fresh baseline and lock the exact matmul hypothesis. Second, implement one narrow backend dispatch-specialization change in local `ggml-webgpu`. Third, rebuild, verify, and either retain the change with an honest doc refresh or revert it cleanly if the signal is weak, noisy, or negative.

**Tech Stack:** TypeScript, Bun, Emscripten, WebGPU, ggml-webgpu, local patched `~/Repos/llama.cpp`, agentchrome smoke benchmarking

---

## File structure

### Primary files to modify
- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
  - Implement the one chosen decode-shaped matmul specialization hypothesis.
- `TODO.md`
  - Refresh only if the backend change is retained or if a reverted no-signal result needs a brief roadmap/status note.

### Optional secondary files
- Directly related local `ggml-webgpu` shader-selection or shader source files only if strictly required by the single chosen hypothesis.
- `eval/reports/perf-baseline.json` only if a retained result is being saved locally as a refreshed artifact.

### Verification targets
- `make wasm-build`
- `make checkall`
- `make smoke-test`
- `make smoke-bench PERF_RUNS=3 PERF_MODEL=tinyllama-1.1b-chat-q4_0`

---

## Task 1: Capture a fresh baseline and lock the exact hypothesis

**Files:**
- Modify: `TODO.md` only if you choose to record the locked hypothesis before implementation
- Modify: none otherwise

- [ ] **Step 1: Confirm the local backend branch is correct**

Run:
```bash
git -C ~/Repos/llama.cpp branch --show-current
```

Expected: `webllm-browser-patches`

- [ ] **Step 2: Capture a fresh benchmark baseline before changing code**

Run:
```bash
make smoke-bench PERF_RUNS=3 PERF_MODEL=tinyllama-1.1b-chat-q4_0
```

Record these values from the median profiled run:
- `tok/s`
- `prefill ms`
- `decode ms`
- `wall time`
- `graphComputeMs`
- `backendMatmulMs`
- `backendEncodeOverheadMs`
- `backendAttentionMs`

- [ ] **Step 3: Lock exactly one backend hypothesis in writing**

Write a short hypothesis note in your task scratchpad or in `TODO.md`. It must be this specific:

```text
Chosen hypothesis:
Specialize one decode-shaped matmul dispatch path in ggml-webgpu for single-token browser decode.

Chosen path:
<name the exact path/family>

Concrete change:
<one dispatch/workgroup/layout or path-selection change only>

Why this one:
<tie directly to current profiling evidence>
```

Rules:
- choose one path only
- choose one concrete backend change only
- do not bundle a second fallback idea into the same task

- [ ] **Step 4: Reject vague hypotheses before implementation**

If the hypothesis cannot be stated as one exact path plus one exact change, stop and refine it before editing code.

- [ ] **Step 5: Commit only if a project doc was changed**

If you updated `TODO.md`:
```bash
git add TODO.md
git commit -m "docs: record narrow matmul tuning hypothesis"
```

If no tracked file changed, skip the commit.

---

## Task 2: Implement one narrow backend matmul specialization

**Files:**
- Modify: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
- Modify: one directly related local shader/shader-selection file only if required by the chosen hypothesis

- [ ] **Step 1: Read the exact current backend path before editing**

Inspect the specific function(s) implementing the chosen path in:
- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
- any directly related shader file, if needed

Record the current dispatch/workgroup/layout behavior so the before/after change is attributable.

- [ ] **Step 2: Make exactly one narrow backend change**

Allowed examples:
- one decode-specific dispatch geometry selection change
- one decode-specific workgroup/layout choice
- one removal of an unnecessary branch/setup path on the chosen decode matmul route

Not allowed in this task:
- multiple path changes
- broad cleanup/refactor mixed with optimization
- encode-overhead optimization unrelated to the chosen matmul path
- structural work like graph reuse or fused attention

- [ ] **Step 3: Re-read the edited backend file and verify scope stayed narrow**

Confirm that the resulting diff:
- touches only the chosen path
- does not introduce a second optimization idea
- does not change unrelated browser behavior

- [ ] **Step 4: Commit the backend hypothesis implementation**

Run:
```bash
git -C ~/Repos/llama.cpp add ggml/src/ggml-webgpu/ggml-webgpu.cpp
git -C ~/Repos/llama.cpp commit -m "ggml-webgpu: specialize browser decode matmul dispatch"
```

If a second local backend file was required, add it to the same commit. Do not include unrelated files.

---

## Task 3: Rebuild and run the acceptance benchmark

**Files:**
- Modify: none

- [ ] **Step 1: Rebuild the WASM/backend artifacts**

Run:
```bash
make wasm-build
make smoke-test
```

Expected:
- build completes successfully
- smoke assets refresh successfully

- [ ] **Step 2: Run repository verification**

Run:
```bash
make checkall
```

Expected: pass, or only pre-existing known warnings that are already accepted by the project state.

- [ ] **Step 3: Run the acceptance benchmark**

Run:
```bash
make smoke-bench PERF_RUNS=3 PERF_MODEL=tinyllama-1.1b-chat-q4_0
```

Capture:
- median `tok/s`
- median `prefill ms`
- median `decode ms`
- median `wall time`
- `graphComputeMs`
- `backendMatmulMs`
- `backendEncodeOverheadMs`
- `backendAttentionMs`
- browser console cleanliness under the smoke policy

- [ ] **Step 4: Compare directly against the pre-change baseline**

Write a short comparison in this exact shape:

```text
Before:
- tok/s: <value>
- decode ms: <value>
- graphComputeMs: <value>
- backendMatmulMs: <value>

After:
- tok/s: <value>
- decode ms: <value>
- graphComputeMs: <value>
- backendMatmulMs: <value>

Conclusion:
- retain | revert
- reason: <one sentence>
```

- [ ] **Step 5: Reject weak or ambiguous signals**

If the result is flat, noisy, regressive, or introduces any smoke/runtime regression, treat the hypothesis as failed and move to Task 4 revert flow.

---

## Task 4: Revert on no-signal, or retain with documentation on success

**Files:**
- Modify: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp` only if reverting
- Modify: `TODO.md` only if retained or if a brief no-signal note is warranted
- Optional: `eval/reports/perf-baseline.json`

- [ ] **Step 1: If the hypothesis failed, revert it immediately**

Run:
```bash
git -C ~/Repos/llama.cpp log --oneline -n 1
git -C ~/Repos/llama.cpp revert --no-edit HEAD
```

Expected: the narrow optimization commit is reverted cleanly.

If the hypothesis clearly succeeded, skip this step.

- [ ] **Step 2: If retained, refresh docs honestly**

Update `TODO.md` only if the win changes the project story in a meaningful way.

Required doc rules:
- do not overclaim profile-mode throughput as a shipping baseline
- state the retained change narrowly
- keep structural follow-up deferred unless the retained signal truly changes priorities

- [ ] **Step 3: If reverted, record the no-signal outcome briefly only if useful**

Optional note shape:

```text
Matmul follow-up attempt (<short hypothesis name>) showed no meaningful retained gain and was reverted.
```

Keep it brief and factual.

- [ ] **Step 4: Run the final verification state for whichever branch state remains**

If retained:
```bash
make checkall
make smoke-test
make smoke-bench PERF_RUNS=1 PERF_MODEL=tinyllama-1.1b-chat-q4_0
```

If reverted:
```bash
make checkall
make smoke-test
```

Expected: the final checked-in state is clean for the chosen outcome.

- [ ] **Step 5: Commit the final state**

If retained and docs changed in `webllm`:
```bash
git add TODO.md eval/reports/perf-baseline.json
git commit -m "docs: refresh retained matmul tuning result"
```

If reverted and you added a project note in `webllm`:
```bash
git add TODO.md
git commit -m "docs: note reverted matmul tuning attempt"
```

If the final state required no tracked `webllm` file changes, skip the repo-local commit.

The local `~/Repos/llama.cpp` repo should already contain either:
- the retained optimization commit, or
- the revert commit

Do not create extra commits beyond those needed to represent the final accepted state.

---

## Self-review

- Spec coverage: the plan covers one hypothesis only, baseline capture, one narrow backend change, full verification, and retain-or-revert handling.
- Placeholder scan: every task states concrete commands and decision rules; no TBD/TODO placeholders remain.
- Type consistency: the plan does not introduce new public APIs or new profiling fields, so it stays consistent with the accepted profiling/reporting system.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-22-matmul-tuning-followup-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
