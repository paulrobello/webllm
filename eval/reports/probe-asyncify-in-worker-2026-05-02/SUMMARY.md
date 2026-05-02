# Probe: ASYNCIFY-in-worker re-confirmation against the production engine path

**Date:** 2026-05-02
**Tip:** `d59abedf37029d67c55187cc8a69dc54286fd60d` (smoke bundle rebuilt at this tip prior to probe run)
**Plan task:** Task 1 of `docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md`
**Predecessor:** probe 9d (closed 2026-05-01) — verified ASYNCIFY-in-worker with a tiny model via `loadModelFromBuffer` *factory*. This probe re-verifies with the **registered-model engine path** (`WebLLM.init()` → `engine.loadModelFromBuffer()` → `engine.chatCompletion()`).

## Setup

- Static server: `make smoke-serve` on `:8031` (already running, PID 32130).
- Browser: existing agentchrome session on port 56353, tab `A423CCEBDDF7D5A6355DB203EC344A8A` (reused per CLAUDE.md rule — no new Chrome window).
- Bundle: `smoke-test/webllm-bundle.js` rebuilt via `bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser` (244.25 KB).
- Model: `smoke-test/models/qwen3-0.6b-q4f16.gguf` (~610 MB, Q4_F16, fetched in-worker).
- Probe URL: `http://localhost:8031/probe-asyncify-in-worker.html?v=3`.

## Files added

- `smoke-test/probe-asyncify-in-worker.html` — boots a `DedicatedWorker`, surfaces results in `#log`.
- `smoke-test/probe-asyncify-in-worker-worker.js` — runs `WebLLM.init()` → `loadModelFromBuffer()` → `chatCompletion()` inside the worker; posts wall times, token count, decoded text, and the first 16 sampled tokenIds back to the page.

## Result

| Metric | Value |
|---|---|
| `tInitMs` (init + fetch + parse + GPU upload) | **854 ms** |
| `tGenMs` (16-token chat completion) | **390 ms** |
| `nTokens` | **16** |
| Generated text (post-think-mask) | `""` (empty — see note below) |
| First 16 tokenIds | `[151667, 198, 32313, 11, 279, 1196, 6801, 264, 2805, 21646, 13, 6771, 752, 1744, 315, 2494]` |
| Console errors | **none** (only `[info] booting worker…` and 4 result lines) |

### TokenIds decode

Decoded against the Qwen3 vocab, the 16 sampled tokens are:

```
<think>\nOkay, the user wants a short joke. Let me think of something
```

i.e. token 151667 is the `<think>` opener, then a coherent thinking-mode prelude. The visible-text field is empty because `</think>` (token 151668) was not reached within 16 tokens — visible-answer text is masked while the model is in thinking mode (canonical Qwen3 behaviour, mirrored in `smoke-test/real-model-smoke.js`). **The worker really ran inference end-to-end** — the tokenIds are non-zero, monotonically varying, and lexically coherent.

### Init-time sanity check

854 ms for `WebLLM.init()` + 600 MB GGUF fetch (HTTP-cache warm) + parse + WebGPU upload + KV-cache init is consistent with a fresh in-worker load on M-series silicon. `WebLLM.init()` itself is lightweight (no model load); the cost concentrates in `loadModelFromBuffer`.

## Verdict: **PASS**

- 16 tokens generated coherently inside a `DedicatedWorker` against the registered-model engine path (not the convenience factory).
- Zero runtime console errors. Zero ASYNCIFY-related stack errors. Zero `import.meta.url`-resolution surprises (the worker successfully resolved `./webllm-wasm.js` relative to the worker script).
- Confirms ASYNCIFY-driven `graphCompute` survives in a worker context for the production engine path. Precondition for the dual-mode (main+worker) deployment work is satisfied — Task 2 (proxy plumbing) can proceed.

## Implementation note

The task script (Step 2) had `loadModelFromBuffer` arguments in the order `(buf, name, options, wasmUrl)`, but the actual signature is `(buf, name, wasmUrl?, options?)`. The first probe run (`?v=1`) failed with `Failed to resolve module specifier '[object Object]'` — the options object was being passed where the wasmUrl string was expected. Worker corrected to the real signature and the rerun (`?v=2`, `?v=3`) PASSED. No source-code changes needed; just the probe-side argument order. Task description should be updated for future readers.

## Follow-ups

- None blocking. Proceed to Task 2.
- (cosmetic) Update Task 1's literal worker-script snippet in `docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md` to use the correct `loadModelFromBuffer` arg order if the plan is referenced again.
