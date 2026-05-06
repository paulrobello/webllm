# P2-v2 Pre-prototype baseline — tinyllama Q4_0 5-token greedy decode

**Capture date:** 2026-05-05
**webllm SHA:** `bcb4247` (`docs(TODO): redirect P2-v2 next-session opener to spec + plan`)
**llama.cpp `webllm-browser-patches` SHA:** `b54503497` (`ggml-webgpu: use wgpu::WaitAny under JSPI instead of polling loop`)
**Browser + GPU:** Chrome 147.0.7727.138 headless, Apple Metal-3 (per
`ggml_webgpu: adapter_info: vendor: apple | architecture: metal-3`)

## Status: **BLOCKED**

Task 0 cannot capture the legacy decode baseline because the smoke-test
chat-completion path **hangs deterministically** at the shader-cache
warmup stage on the current `main` HEAD. Three reproductions on three
fresh `agentchrome connect --launch --headless` Chrome sessions all
exhibited the identical failure mode:

- Steps [1/8] through [6/8] (WebGPU init, GGUF parse, weight upload,
  KV cache init, tokenizer construction) all complete in ~1-2 s as
  expected.
- The shader-cache warmup `chatCompletion(maxTokens=2)` call
  (`smoke-test/real-model-page.js:957-971`) never returns. Page is
  pinned at "85%" / "[6/8] Tokenizer ready" indefinitely.
- Synchronous CDP `js exec` probes succeed (`window.inference.constructor.name
  === "ModelInference"`, `window.engine` and `window.handleId` are set,
  `forward` and `resetKVCache` are functions).
- **Async** CDP `js exec` calls into the engine (`inference.forward`,
  `engine.chatCompletion`) hang the entire renderer — even a
  `Promise.race([forward(...), timeout(30s)])` wrapper times out the
  CDP `Runtime.evaluate` call itself, indicating the async path is
  wedged inside the WASM/WebGPU layer rather than just slow.
- No console errors are emitted; only the benign
  `ggml_webgpu: adapter_info: …` line appears.
- `lsof :8031`, `lsof :8033`, GGUF md5, and disk artifacts (bundles
  rebuilt at 2026-05-05 20:50 PT during the smoke-bench attempt)
  all check out clean.

## Upstream cadence check

```
$ cd ~/Repos/llama.cpp && git fetch origin
$ git log webllm-browser-patches..origin/master --oneline -- ggml/src/ggml-webgpu/ ggml/include/
(empty)
```

Empty → no upstream rebase pending. Step 2 of the plan passes.

## Root cause hypothesis

Local `webllm-browser-patches` HEAD `b54503497`
("ggml-webgpu: use wgpu::WaitAny under JSPI instead of polling loop")
**landed today, 2026-05-05 20:00 PT**, ~3 hours before this Task 0
attempt. webllm's wasm32 build forces `GGML_WEBGPU_JSPI=ON`
(`src/wasm/CMakeLists.txt:30`), so the new `wgpu::WaitAny` code path
is the active queue-wait / map-buffer mechanism for the chat decode
hot path. Since:

- Tinyllama Q4_0 last produced healthy bench numbers at 2026-05-05
  02:59 UTC (106.3 tok/s) per the live dashboard
  (`http://localhost:8033/runs?limit=200`).
- webllm tree has only doc commits since `bcb4247`; no source changes.
- The hang is in async readback, which is the exact contract
  CLAUDE.md flags as load-bearing
  ("`graphCompute()` must be treated as async-capable in the browser
  integration and awaited before tensor readback. Async tensor
  readback must not use `stackAlloc` across `await` boundaries").

…the new `WaitAny` integration is the most plausible regression
locus. The polling path (`emscripten_sleep(1)` loop) it replaced is
known-good with the existing async-readback contract; the new path
may need additional ASYNCIFY/JSPI cooperation that hasn't been wired
through the smoke flow yet.

## Three-attempt repro log

| Attempt | Browser PID | Tab strategy | URL params | Outcome |
|---|---|---|---|---|
| 1 | 91965 (pre-existing session) | reused existing tab | `perfTrace=1` | Hung at [6/8]; CDP async eval timeout |
| 2 | 99224 (fresh `--launch --headless`) | perf.ts auto-created | `perfTrace=1` | Hung at [6/8]; identical signature |
| 3 | 71903 (fresh `--launch --headless`) | manually re-navigated, no `perfTrace` | `perf=noTrace&ingest=off` | Hung at [6/8]; identical signature |

`perfTrace=1` ruled out as a cause. Existing-tab vs fresh-tab ruled
out as a cause. Browser-state contamination ruled out by 3 fresh
launches.

## Static call-site count for `wasm.op*` in `model-inference.ts`

```
$ grep -n "wasm\.op[A-Z]" src/inference/model-inference.ts | wc -l
212
```

This is the upper bound on per-graph distinct call types in the
legacy `ModelInference` path (Task 0 step 4 metric (a)). The
**per-decode-step crossing count** (metric (b)) cannot be captured
until the decode hang is resolved.

## Legacy 5-token decode metrics (UNCAPTURED)

| Metric | Value | Notes |
|---|---|---|
| Decode tok/s (median, 3 runs) | **N/A** | Bench timed out after 360 s on Run 1/3 |
| Per-token wall (ms, median) | **N/A** | derives from tok/s |
| Dispatches/token (median) | **N/A** | from `webllm_perf_counter`; never reached |
| Static `wasm.op*` callsites in model-inference.ts | 212 | upper bound on distinct call types |

## T3 gate target table (Legacy baseline column UNFILLED)

| Metric | Legacy baseline | Gate (jsep) green | yellow | red |
|---|---|---|---|---|
| Per-token wall (5-token median) | **TBD** (gate ≥100 tok/s pin would set this to 9.0 ms; not measured today) | ≤2× legacy | 2-5× | >5× |
| EM_ASM crossings/token | n/a (cwrap path: dispatches/token, value TBD) | <1500 | 1500-4000 | >4000 |
| Greedy 5/5 token equality | reference (token sequence: TBD) | byte-identical | — | — |

## Generated tokens (greedy, max=5, prompt="Hello")

**N/A** — decode never produced a token. The 02:59 UTC dashboard
trace at 106.3 tok/s used a different (non-greedy, non-5-token)
prompt fixture, so it is not a substitute for the gate's reference
sequence.

## Recommendation

Phase 2 implementation **must not** begin until either:

1. The `b54503497` `wgpu::WaitAny` path is debugged or reverted such
   that `make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0
   PERF_RUNS=3` produces ≥100 tok/s decode end-to-end, and a 5-token
   greedy reference token sequence is captured into this file; OR
2. The plan is amended so that the gate's "Legacy baseline" column
   uses the most recent green tinyllama Q4_0 datapoint
   (2026-05-05 02:59 UTC, 106.3 tok/s, dispatchCount unrecorded)
   as a stale reference, with the explicit caveat that
   `dispatchCount/token` cannot be back-filled without a working
   smoke decode.

Option 1 preserves the gate's load-bearing property (current-day
legacy number is the comparison ground-truth). Option 2 ships Phase 2
on stale data and accepts the corresponding noise.

The `b54503497` revert path is one commit deep
(`git checkout webllm-browser-patches~1` would land on `fc1f81242`),
making investigation cheap. Surgery on the `WaitAny` integration is
out of Task 0's scope and explicitly out of "surgical changes"
doctrine for this session.

## Notes

- This file is the comparison ground-truth for Task 7's gate decision.
- "EM_ASM crossings/token" in the gate refers to the jsep prototype's
  per-node EM_ASM cost. The legacy column reads "n/a" per spec §T3
  because the legacy path uses cwrap'd `op*` exports, not EM_ASM.
  The closest comparable number is `dispatchCount/token`.
- Until the hang is resolved, **all six other canonical-fleet models
  are likely affected** (qwen3-0.6b, qwen3-1.7b, mistral-7b, llama-3.1-8b,
  qwen3-8b) since they share the same chat-decode async-readback path.
  This is a fleet-wide regression, not tinyllama-specific.
