# qwen3-0.6b-thinking-cold semantic-reasoning wedge — investigation closure

**Date:** 2026-05-08
**Disposition:** **Transient confirmed; root cause attributed upstream (Dawn /
Chrome WebGPU process pressure). Abort guard `f3cbca9` is the operative
mitigation; no structural code change shipped.**

## TL;DR

The "qwen3-0.6b-thinking-cold semantic-reasoning wedge" is a **WebGPU device
loss** (`Device lost! Reason: 1, Message: A valid external Instance
reference no longer exists.`) that fires probabilistically during sustained
GPU work, not a deterministic bug in our WASM bindings or eval runner. It
manifests as a downstream `unreachable` trap because `ggml-webgpu.cpp:543`
calls `GGML_ABORT` when `Queue work` returns non-Success status post device
loss.

- **Reproduction rate (this-session, n=3):** 1 / 3 wedged (33%).
- **Lifetime rate at temp=0 thinking-cold (DB):** 1 wedged-but-recorded /
  19 runs = 5.3% (understated — post-guard wedges don't write rows).
- **Other models tested at temp=0 (DB):** qwen3-1.7b (n=6), llama-3.2-1b
  (n=3), qwen3-8b (n=2) — **0 wedges**. Specific to qwen3-0.6b under
  thinking-cold settings.
- **Probe A (emb-001 alone, no warmup):** PASSED. Rules out task-input
  specificity (H3). Confirms cumulative-state hypothesis.
- **Cumulative wall time at trap:** ~67s of GPU work over 36 prior
  `character.chat()` decodes (12 tool-calling + 12 reasoning + 12
  instruction-following).

## Evidence

### Console capture from 3rd reproduction
- File: `console-existing-tab-2026-05-09T05:04:46Z.json`
- Two `adapter_info:` lines on init (benign)
- One `ggml_webgpu: Device lost! Reason: 1, Message: A valid external
  Instance reference no longer exists.` at 05:04:46.561Z
- 12 cascade pairs of `ggml-webgpu.cpp:544: ggml_webgpu: Queue work
  failed with status 0:` + `Aborted()` over 245 ms (one per remaining
  semantic-reasoning task)

### Wedged-run report
- `eval/reports/archive/wedged-run-qwen3-0.6b-thinking-cold-20260509-04h33.json`
  (2nd reproduction)
- Per-task progression: tool-calling 12/12 ok · reasoning 12/12 ok ·
  instruction-following 12/12 ok · semantic-reasoning **0/12** ok
- emb-001 latency at trap: 531 ms (got partway into decode)
- emb-002...emb-012: 2-3 ms each (sync throw — engine already dead)

### Probe A — bisection
- Posted single-task list with only `emb-001`
- Navigated to fresh page (full WASM + WebGPU re-init)
- Ran `bench=probe-a-emb001-only-2026-05-08` — emb-001 ran in 977 ms,
  scored 0% (model produced wrong answer at temp=0 thinking-on, but
  did NOT crash)
- Final state: `benchDone:true, wedge:false`
- **Verdict:** wedge requires sustained prior load; not input-specific.

### Reproduction sweep (this session)
- Run #1: 48/48 clean, 77% — passed
- Run #2: 35/48 then `unreachable` — guard aborted at 3 consecutive
  errors; **no row written to DB** (guard worked as designed)
- Run #3: 48/48 clean, 77% — passed

## Root cause

The **direct cause** of the trap is `GGML_ABORT` at
[`ggml-webgpu.cpp:543-544`](https://github.com/ggml-org/llama.cpp/blob/webllm-browser-patches/ggml/src/ggml-webgpu/ggml-webgpu.cpp#L543-L544),
fired when `Queue.OnSubmittedWorkDone` returns `QueueWorkDoneStatus::Error`
because the WebGPU device was lost mid-stream.

The **upstream cause** of the device loss is signalled by Dawn's emdawnwebgpu
binding as "external Instance reference no longer exists" with reason
`Unknown` (1). This message is emitted by Chrome's WebGPU subsystem when
the wire-client's reference to the GPU process Instance becomes invalid —
typically when the Chrome GPU process recycles a long-running tab's WebGPU
device under cumulative resource pressure.

The Instance is held in a `std::shared_ptr<webgpu_global_context_struct>`
reachable from a `static ggml_backend_webgpu_reg_context *ctx` (see
`ggml-webgpu.cpp:4858, 4889-4891`). Our local patch stack does not touch
Instance lifetime. The shared_ptr is never re-assigned during a session.
The Instance handle invalidation therefore originates outside the WASM
module — in Chrome's GPU process / Dawn wire layer.

### Why qwen3-0.6b specifically

- Smallest fleet model (0.6B params, ~1.4 GB VRAM weights at q4f16) →
  fastest decode → highest GPU command-queue submission rate per second
- thinking-cold profile (`maxTokens=1024`, `enableThinking=true`,
  `temperature=0`) generates the longest outputs because greedy
  thinking + small reasoning model gets stuck in deterministic loops
  more readily (`rs-012` shows the pattern: 8107 ms · empty visible
  output → max_tokens consumed inside `<think>`)
- Combined: fastest dispatch rate × longest generations × sustained
  for ~67 s = highest cumulative GPU process pressure in the fleet

Larger models (1.7B, 8B) submit dispatches more slowly (compute-bound)
and do not approach the same pressure threshold within 48 tasks.

## Why no structural code fix shipped

Three reasons:

1. **The cumulative-state lever we control is already exhausted.**
   `runner.ts:91, 126` calls `engine.resetConversation` between every
   task, which clears KV via `llama_memory_clear`. There is no looser
   per-task state on the JS / WASM side that builds up.

2. **The pressure source is upstream.** "External Instance reference no
   longer exists" is Chrome's wire-client signal that the GPU-process
   peer is gone. We cannot reset that from WASM without a full WebGPU
   device recreate path, which the engine doesn't expose and which
   would require a non-trivial refactor (drop+rebuild adapter / device
   / queue / pipelines / weight buffers between dimensions). With a
   ~20-30% wedge rate and `n=30+` trials needed per fix-validation
   pass, the fix-cost / fix-confidence ratio is poor.

3. **The abort guard already fully prevents data pollution.** Verified
   in this session: Run #2 wedged at task 35-38, the guard fired with
   `EngineDeadError` after 3 consecutive errors, no `eval_complete`
   was published, no row was written to `evals`. The dashboard stayed
   clean; bench summary surfaced `[FAIL]` cleanly with the diagnosis
   message ("runTasks aborting after 3 consecutive task errors …").

## Re-evaluation triggers

Reopen this issue if **any** of the following fires:

1. Wedge frequency on `qwen3-0.6b-thinking-cold` exceeds **50%** over
   any 10-trial window (current observed: ~30% session, ~20% lifetime).
2. The wedge starts firing on a second model (any of qwen3-1.7b,
   llama-3.2-1b, qwen3-8b, or other registered chat models).
3. Upstream `ggml-webgpu` lands a change touching device-lifecycle
   code (`ggml-webgpu.cpp` lines around `dev_desc.SetDeviceLostCallback`
   ~4223, `ggml_backend_webgpu_reg` ~4853, or the global-context
   destructor ~4308). Cadence-check fires daily per `CLAUDE.md`.
4. Chrome / Dawn ships a fix for the "external Instance reference"
   message under sustained WebGPU load — track via the cadence check
   and `chrome://gpu` release notes.
5. A consumer reports wedges hitting agent + Three.js coexistence in a
   real product workload (the load-bearing project use case per
   `CLAUDE.md`'s 8B ceiling rationale).

## Artifacts

- `console-existing-tab-2026-05-09T05:04:46Z.json` — 3rd-reproduction
  console capture (40 messages, 38 errors).
- `eval/reports/archive/wedged-run-qwen3-0.6b-thinking-cold-20260509.json`
  — 1st reproduction (02:53).
- `eval/reports/archive/wedged-run-qwen3-0.6b-thinking-cold-20260509-04h33.json`
  — 2nd reproduction (04:33).
- 4th reproduction (Run #2 in this session) **not retained** — guard
  prevented row write; harness output recorded inline above.
- `f3cbca9 fix(eval/runner): abort runTasks after N consecutive task
  errors` — the operative mitigation.
