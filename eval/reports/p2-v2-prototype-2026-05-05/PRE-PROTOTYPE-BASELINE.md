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

Phase 2 implementation **must not** begin until the legacy chat-decode
hang is resolved end-to-end (`make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0
PERF_RUNS=3` produces ≥100 tok/s and a 5-token greedy reference
sequence is captured into this file).

## Update 2026-05-05 21:30 PT — `b54503497` is NOT the cause

The `wgpu::WaitAny` hypothesis above was **disproven**. Verification
trail:

1. Reverted `b54503497` on `webllm-browser-patches` via
   `git revert --no-edit b54503497` → revert commit
   `ffaa6ad4e Revert "ggml-webgpu: use wgpu::WaitAny under JSPI ..."`
   (non-destructive — restorable via `git revert HEAD` on the same
   branch).
2. Rebuilt both wasm32 + mem64 WASM artifacts (`make wasm-build` →
   ggml commit pin moved to `ffaa6ad4e` per CMake configure log).
3. Re-ran `make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0
   PERF_RUNS=3` → **timed out on Run 1/3 with the identical
   signature**.
4. Manual fresh navigation to `?v=postrevert-1&ingest=off` in the
   reused agentchrome tab progressed steps 1-6/8 cleanly, then
   stalled at "[6/8] Tokenizer ready" with no further progress
   for 7+ minutes (still pinned at the time of writing). Identical
   to the pre-revert behavior. Same single benign console message
   (`adapter_info: vendor: apple ...`). No errors.

Inspection of `b54503497`'s `get_tensor_map_pending` flag logic:
correct on paper (acquire → set true → WaitAny → reset to false at
line 642). Logic doesn't explain the hang.

## Updated root cause hypothesis

Tracing what changed since the last green tinyllama trace at
**2026-05-05 03:03:50 UTC** (run_id `1777950230832-irej74`,
79.8 tok/s greedy at 64 tokens per `eval/reports/smoke-runs.db`):

- llama.cpp `webllm-browser-patches` only added `b54503497` (now
  ruled out) — `fc1f81242` was already in tree at 03:03 UTC.
- webllm tree diff vs that timestamp: ~30 commits including the
  full **P2 v1 migration** (`4bb644c..bd7ae4b`, 11 commits that
  deleted `model-inference.ts` and routed everything through
  `LlamaDecodeWrapper`) **and its revert** at `0b57d41`
  (`revert(p2): roll back wrapper+dispatch+delete; keep bridge
  surface`, **4179 insertions / 1243 deletions across 25 files**).

The P2 v1 revert kept the C++ bridge expansion live
(`webllm_get_metadata`, `webllm_n_ctx_*`, `webllm_kv_seq_rm/clear`,
`webllm_state_seq_*`, `webllm_get_embeddings`, `webllm_perf_counter`
on `src/wasm/webgpu-bridge.cpp`; matching TS bindings in
`src/inference/llama-bridge.ts`). The hang therefore most plausibly
lives in **incomplete revert state** — either
`engine.adoptPreloadedModel` (smoke page line 743), the restored
`model-inference.ts` (2890 LOC re-introduced), or the `model-loader.ts`
delta (451 lines modified) — rather than in `b54503497`.

## Suggested next-session triage

1. **Quick bisect**: `git checkout 72cd44c` (the commit immediately
   before P2 v1 plan-write — `docs(TODO): close P0+P1, surface P2
   quickstart for next session`), rebuild WASM (still need to also
   `git checkout webllm-browser-patches~N` on llama.cpp to pre-`b4d4b48`
   JSPI pivot), run `make smoke-bench`. If green: confirms regression
   landed during P2 v1 work and survived the revert. If still hangs:
   look further back, possibly to the JSPI-pivot commit `b4d4b48`.
2. **Targeted diff**: `git diff 72cd44c...0b57d41 -- src/core/engine.ts
   src/inference/model-inference.ts src/inference/generation.ts
   src/models/model-loader.ts` to surface the post-revert deltas
   most likely to affect the chat-decode hot path.
3. **Decide whether to keep `ffaa6ad4e` (the WaitAny revert)** — it
   pollutes the patch stack one commit deep and was based on a
   wrong hypothesis. Cleanup: `git revert HEAD` on
   `webllm-browser-patches` (restores `b54503497` via a second
   revert) **or** force-rebuild back to `b54503497` via
   `git reset --hard b54503497` (destructive — flagged in CLAUDE.md
   as needing explicit user authorization). The reflog still
   contains `ffaa6ad4e` for ≥30 days regardless.

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
