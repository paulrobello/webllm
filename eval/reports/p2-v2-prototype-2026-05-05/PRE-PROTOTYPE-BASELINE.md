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

## Bisect (2026-05-05 22:00 PT) — `b4d4b48` JSPI pivot is the regression

User authorized "do 1 and 2": (1) drop `ffaa6ad4e` revert via
`git reset --hard HEAD~1` on `webllm-browser-patches` (llama.cpp
back at `b54503497`); (2) bisect through P2 v1 work.

Bisect table (llama.cpp held at `b54503497` throughout):

| webllm commit | Description | smoke-bench | tok/s | Notes |
|---|---|---|---|---|
| `72cd44c` | close P0+P1 doc | **HANG** | — | Rules out P2 v1 revert |
| `b4d4b48` | **JSPI pivot** | **HANG** | — | First JSPI commit |
| `b8cada0` | last ASYNCIFY (Reapply detokenize) | **GREEN** | 88.1 | 450 disp/tok, matmul 4.04 ms |
| `9b5b362` | post-greedy-cutover docs | **GREEN** | 86.6 | Same kernel shape |

**Single-commit regression at `b4d4b48 feat(wasm): pivot from
-sASYNCIFY to -sJSPI to unblock P1 wasm exports`** (Tue May 5
15:53 PT). The pivot only touches:
`smoke-test/p0-spike.{src.ts,js}`,
`smoke-test/p1-tokenizer-parity.js`,
`src/inference/llama-bridge.ts`,
`src/wasm/CMakeLists.txt`.
**It does NOT touch the legacy chat-decode path
(`src/inference/ggml-wasm.ts`,
`src/inference/model-inference.ts`).** The build-flag flip
alone — `GGML_WEBGPU_JSPI ON`, drop `-sASYNCIFY`, add
`-sJSPI -fwasm-exceptions` — breaks legacy decode.

## Refined root cause hypothesis

Most likely cause: at `b4d4b48`, the in-tree
`ggml-webgpu.cpp:wait_queue` and `map_buffer` were on the
`emscripten_sleep(1)` polling loop (the WaitAny replacement at
llama.cpp `b54503497` did not land until ~24 h later). Under
`-sJSPI` **without** `-sASYNCIFY`, `emscripten_sleep` does not
yield to the JS event loop the way Asyncify enables — the polling
spin prevents the WebGPU `OnSubmittedWorkDone` callback from
firing, so `done` is never set → infinite loop → hang.

`b54503497` was supposed to fix this by switching to
`wgpu::Instance::WaitAny` under JSPI, but the test at the
current branch tip (with `b54503497` active) **also hangs**, so
either:
1. The `#if defined(__EMSCRIPTEN__) && !defined(GGML_WEBGPU_JSPI)`
   guard at `ggml-webgpu.cpp:560` does not select the WaitAny
   branch under our build configuration, OR
2. `WaitAny` under JSPI has its own integration gap (e.g. the
   wgpu-internal waitable's promise is not exposed to a
   `JSPI_EXPORTS`-listed export).

The TS-side `callWithAsyncify` at `ggml-wasm.ts:151` checks
`this.m.Asyncify?.currData` — under JSPI, `Module.Asyncify` does
not exist; the optional-chaining returns undefined and the check
is a no-op. The wrapped fn() returns a JSPI-promised value
which the async function auto-unwraps. That path is correct on
inspection — the hang is likely in C++, not TS.

## Suggested fix paths (user decision needed)

**Path A — minimal: confirm `b54503497` actually engages.** Add
a runtime trace to `wait_queue` / `map_buffer` to verify which
branch executes under our JSPI build. If polling: the `#if`
guard is wrong; fix to also exclude the JSPI build. If WaitAny
and still hangs: escalate to Path B.

**Path B — JSPI-promising-wrap the wgpu wait.** May require
exposing `wgpu::Instance::WaitAny` (or an equivalent waitable
trampoline) as a `JSPI_EXPORTS`-listed export so the suspend
crosses the wasm/JS boundary correctly. Likely needs an upstream
ggml-webgpu patch (count toward Phase 2 patch budget — currently
0/3 used).

**Path C — restore ASYNCIFY for legacy.** Build two WASM
artifacts:
- `webllm-wasm.{js,wasm}` (legacy, `-sASYNCIFY`) — current chat
  path until P3 ports it.
- `webllm-wasm-jsep.{js,wasm}` (P2-v2 prototype, `-sJSPI`) — new
  jsep backend.

Heavyweight but cleanly decouples the pivot's blast radius.
P2-v2 Phase 2 is the natural vehicle since it already introduces
the dual-artifact pattern (`wasm-build-jsep` Makefile target).
However the pivot was justified by the
`__wasm_call_ctors`-trap problem under ASYNCIFY + new exports;
restoring ASYNCIFY for legacy reintroduces that constraint
unless the new bridge exports are excluded from the legacy build.

## Implication for Task 0 + Phase 2

Legacy baseline cannot be captured on `main` until JSPI legacy
decode works. **Workaround for the gate:** capture the baseline
at `b8cada0` (pre-pivot, 88.1 tok/s) as a stale-but-known-good
reference. Phase 2's gate becomes "jsep tok/s vs the b8cada0
baseline" with explicit caveat that the JSPI build's intrinsic
overhead (relative to ASYNCIFY) is folded into the comparison
(both legacy-`b8cada0`-ASYNCIFY and jsep-JSPI carry their build's
async-mechanism cost). This is acceptable for the prototype
gate — the jsep test is "is the EM_ASM-per-node architecture
viable", not "is JSPI overhead reasonable".

`ffaa6ad4e` (WaitAny revert) was dropped from
`webllm-browser-patches` per user authorization. llama.cpp tip
restored to `b54503497`.

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
