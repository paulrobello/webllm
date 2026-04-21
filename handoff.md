# WebLLM Handoff — Browser Smoke Test Fixed

## Current Status

The browser smoke test now passes end-to-end in headed Chrome with the stricter pass criteria:
- all visible steps pass
- no relevant backend/runtime errors are emitted

Validated in the existing `agentchrome` browser session by reusing the same smoke-test tab.

## What Was Actually Fixed

### 1. Attention tensor ordering in `src/inference/model-inference.ts`
The original blocker from the previous handoff was real:
- K/V attention tensor layout for 3D `ggml_mul_mat` was wrong
- V cache now uses `[maxCtx, headDim, nKvHeads]`
- attention reads no longer permute K/V into an invalid broadcast shape
- GQA uses `ggml_mul_mat` broadcast instead of manual repeat

### 2. Synthetic smoke-test GGUF tensor shapes in `smoke-test/index.html`
The smoke test itself had incorrect tensor dimension ordering for linear weights.
That masked some issues and created new ones once attention was fixed.

The synthetic model now uses the correct GGUF-style dimensions for:
- `attn_q.weight`
- `attn_k.weight`
- `attn_v.weight`
- `attn_output.weight`
- `ffn_gate.weight`
- `ffn_up.weight`
- `ffn_down.weight`

### 3. Async browser readback path in `src/inference/ggml-wasm.ts`
A major follow-on bug was caused by async GPU tensor readback being treated like a synchronous operation.

Important fixes:
- `downloadFromTensor()` is async
- async tensor readback uses heap allocation, not `stackAlloc` across `await`
- `graphCompute()` is async-aware
- `webgpu_init()` and readback operations use explicit Asyncify-aware handling

### 4. Missing await on graph compute
`ModelInference.forward()` was starting tensor readback before async graph compute had fully completed.

Fixed by awaiting:
```ts
await wasm.graphCompute(graph)
```

### 5. Browser-side ggml-webgpu wait behavior in local `llama.cpp`
The local dependency at:
- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`

was patched so browser waits use an Emscripten-friendly callback + sleep flow instead of the problematic `WaitAny` path in this environment.

This was necessary to stabilize WebGPU queue/map completion in browser WASM.

### 6. Smoke-test asset cache busting
The page cache-busted HTML with `?v=...` but not imported assets.
That caused stale `webllm-bundle.js` / `webllm-wasm.js` behavior during debugging.

`smoke-test/index.html` now versions both imported assets using `window.location.search`.

### 7. Stronger smoke-test success criteria
The smoke test now explicitly fails if relevant backend/runtime errors are emitted.
It also checks:
- no relevant errors before cleanup
- no relevant errors after cleanup

## Files Changed In This Repo

- `CLAUDE.md`
- `handoff.md`
- `smoke-test/index.html`
- `src/core/engine.ts`
- `src/inference/ggml-wasm.ts`
- `src/inference/model-inference.ts`
- `src/wasm/CMakeLists.txt`

## Local Dependency Patched During Debugging

- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`

## Verification

### Local verification
- `make checkall` passes

### Browser verification
Use the existing browser session/tab when possible.

Preferred flow:
1. Start the smoke-test server:
   ```bash
   cd smoke-test && python3 -m http.server 8031
   ```
2. Reuse the existing `agentchrome` session/tab
3. Navigate the same tab to a cache-busted URL such as:
   ```
   http://localhost:8031/?v=33
   ```
4. Confirm the page reports:
   - `ALL TESTS PASSED`
   - `15 passed, 0 failed`

## Important Notes

### agentchrome usage
Do not keep launching fresh Chrome windows during the same debugging task.
Reuse the existing session and tab whenever possible.
See `CLAUDE.md` for the project rule.

### WASM build settings
Temporary debug `-sASSERTIONS=2` was removed after debugging.
Functional setting retained:
- `-sASYNCIFY_STACK_SIZE=1048576`

### ASYNCIFY assumptions
The earlier assumption that only `webgpu_init()` was async was wrong once browser-side ggml-webgpu waits were made Emscripten-friendly.
In practice, browser WebGPU operations like graph compute and tensor readback must be treated as async-capable in this integration.

## Recommended Next Work

If more cleanup is desired:
1. upstream or document the local `llama.cpp` browser wait patch
2. add a dedicated browser smoke-test automation target
3. add regression coverage around async graph compute + readback sequencing
