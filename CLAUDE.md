# CLAUDE.md

## agentchrome usage

When using `agentchrome` in this repository:

1. **Reuse the existing browser session first.**
   - Run `agentchrome connect --status` to find the current live session.
   - Prefer the existing connected port/session over launching a new browser.

2. **Reuse the existing tab when possible.**
   - Run `agentchrome --port <PORT> tabs list`.
   - If the needed page is already open, reuse that tab with `--tab <TAB_ID>`.
   - Prefer navigating the existing smoke-test tab to a cache-busted URL such as `http://localhost:8031/?v=<N>` instead of opening a new tab/window.

3. **Do not launch a new Chrome window unless necessary.**
   - Only use `agentchrome connect --launch ...` if:
     - there is no reachable existing session, or
     - the user explicitly requests a new browser/window/session.

4. **Preserve debugging continuity.**
   - Reusing the same session/tab keeps console history, page state, and reproducibility intact.
   - Avoid creating multiple parallel browser sessions during a single debugging task.

5. **Preferred workflow for this project.**
   - Check current session: `agentchrome connect --status`
   - List tabs: `agentchrome --port <PORT> tabs list`
   - Reuse the smoke-test tab: `agentchrome --port <PORT> --tab <TAB_ID> navigate 'http://localhost:8031/?v=<N>'`
   - Inspect page text / console on that same tab.

## smoke test policy

The browser smoke test should only be considered passing when:

- all visible smoke-test steps pass, and
- no relevant backend/runtime console errors are emitted.

### Browser regression workflow

When validating browser fixes in this repo:

1. Start the static server from `smoke-test/`:
   - `cd smoke-test && python3 -m http.server 8031`
2. Reuse the existing `agentchrome` session and the existing smoke-test tab.
3. Navigate the same tab to a cache-busted URL:
   - `http://localhost:8031/?v=<N>`
4. Check both:
   - page text/results
   - console errors on that same tab
5. Do not call the smoke test fixed unless both the page and console are clean under the policy above.

### Important implementation notes

These issues already caused real regressions here and should be preserved as project guidance:

- `graphCompute()` must be treated as async-capable in the browser integration and must be awaited before tensor readback.
- Async tensor readback must not use `stackAlloc` across `await` boundaries; use heap allocation for async readback buffers.
- The smoke test page must cache-bust imported assets too, not just the HTML URL. `webllm-bundle.js` and `webllm-wasm.js` should inherit the page query suffix.
- Browser-side smoke-test success requires checking for runtime/backend console failures, not just visible step output.
- Keep `-sASYNCIFY_STACK_SIZE=1048576` in the WASM build unless there is a verified replacement strategy.

### Local dependency note

This repo currently depends on a local browser-specific fix in:
- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`

If browser regressions reappear, inspect that local dependency before assuming the bug is entirely in this repo.
