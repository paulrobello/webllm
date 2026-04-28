# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`@paulrobello/webllm` — browser-side LLM inference over WebGPU. TypeScript
orchestration over two interchangeable inference backends: a patched
`llama.cpp` `ggml-webgpu` compiled to WASM (quantized production path),
and a pure-WGSL path for tiny models. Bun for tooling and tests; Chrome
for browser regressions; a Bun-backed SSE + SQLite dashboard
(`eval/live-server.ts`) aggregates runs in real time.

For architecture, the public API, and the full benchmark surface, see
[`README.md`](README.md). For eval methodology, see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Workflow policies (set 2026-04-28)

These apply to all work on this project — perf cycles, infra, refactors,
docs, and bug fixes alike.

- **Model-size ceiling: 30B parameters.** Anything larger (Llama-3-70B,
  DeepSeek-V2 236B, etc.) is out of scope. Levers that justify themselves
  via 70B+ targets must be **deferred** (with the ceiling cited), not
  silently dropped. 8-30B remains in scope.
- **Quick-wins override on YAGNI.** Speculative or YAGNI-flagged work is
  allowed when **(a)** there is measured gain (or a cheap probe phase
  that produces one) and **(b)** the gain outweighs the implementation
  / maintenance complexity. The §27 free-win sweep is the canonical
  pattern.
- **Probe-first is the default.** When a lever's gain is unmeasured,
  start with a probe. Probes are effectively free (time is not a factor)
  and they produce the data that drives every subsequent decision. Each
  probe declares up-front what it measures, the pass/fail thresholds,
  and which downstream decision it gates. Run probes proactively even
  when intuition says the answer is obvious — the measurement is the
  artifact. Templates: §29 verify-cost probe, §27 free-win sweep, §31
  MEMORY64 cap probe.
- **Complexity ≠ implementation time.** Time estimates are chronically
  overestimated and **do not factor** into whether work is worth doing.
  Don't reach for "multi-day", "couple of weeks", etc. as a deterrent.
  Score levers on **maintenance burden**, **surface area**, **risk
  surface to load-bearing invariants** (ASYNCIFY, JS↔WASM ABI, async
  readback, patch stack), **reversibility**, and **external-dependency
  exposure** — not on duration.
- **Always commit before work.** Commit pending state — specs, plans,
  TODO updates, policy changes — **before** starting the next
  implementation chunk. Reason: docs commits carry the load-bearing
  reasoning behind code changes; bundling them into one big commit
  destroys revertability (a `git revert` of the implementation also
  nukes the spec that justified it). Use the established cadence:
  `docs(spec):`, `docs(plan):`, `docs(TODO):`, `feat(...)`,
  `refactor(...)`, `fix(...)` as separate commits. The
  `docs/superpowers/` directory is gitignored; specs/plans in it must
  be force-added (`git add -f`) — see commits `ae68bbe`, `b23ccc9`,
  `66bc603` for the convention.

## Workflows

**Ship gate:** `make checkall` (fmt + lint + typecheck + test) must pass
before a change is "done". `make help` lists every target.

**Single test:** `bun test tests/<file>.test.ts` or `bun test -t "<pattern>"`.

**Browser regression workflow:**

1. Start the static server: `make smoke-serve` (or `bun run eval/smoke-serve.ts --port 8031`).
2. Reuse the existing agentchrome session and tab (see rules below).
3. Navigate the same tab to a cache-busted URL like `http://localhost:8031/?v=<N>`.
4. Check **both** the page `#log` text and the browser console.
5. The smoke test passes only when visible steps succeed AND no relevant runtime console errors are emitted. `adapter_info:` from the WebGPU backend is benign informational output and is not a failure.

**Live dashboard:** `make dashboard-serve` on port 8033. Two ingest paths:

- **Browser smoke runs** (`smoke-test/real-model.html`) auto-post `run_complete`
  to `http://localhost:8033` by default — no flag required. Override with
  `?ingest=<url>`. **Disable per-run** with `?ingest=off` (use this for
  throwaway sanity checks you don't want polluting the dashboard, e.g. when
  iterating on a patched WASM build before committing).
- **Bun harnesses** (`eval/chat-smoke.ts`, `eval/bench.ts`, etc.) require
  `WEBLLM_LIVE_BENCH_URL=http://localhost:8033` to publish events.

**Backfill missed runs:** `make import-reports` walks `eval/reports/` and
imports any speed-run / eval-report JSONs the dashboard hasn't already
recorded. Idempotent (skips by `runId` / `evalId`); safe to re-run after
the dashboard comes back online.

## Ports

- **8031** — smoke-test static site (`make smoke-serve`; override `SMOKE_PORT`).
- **8033** — live dashboard + SSE backend (`make dashboard-serve`; override `DASHBOARD_PORT`).

Both are reserved in `~/.claude/used_ports.md`.

## agentchrome usage

1. **Reuse the existing browser session first.** Run `agentchrome connect --status` and prefer that port over launching a new browser.
2. **Reuse the existing tab when possible.** `agentchrome --port <PORT> tabs list`, then navigate the existing smoke-test tab to a cache-busted URL with `--tab <TAB_ID>` rather than opening a new tab.
3. **Do not launch a new Chrome window** unless there is no reachable session or the user explicitly requests one.
4. **Preserve debugging continuity** — reusing the same session/tab keeps console history, page state, and reproducibility intact.

## Regression lessons — do not repeat these bugs

These issues caused real regressions here. The fixes are load-bearing:

- `graphCompute()` must be treated as async-capable in the browser integration and awaited before tensor readback.
- Async tensor readback must not use `stackAlloc` across `await` boundaries; use heap allocation for async readback buffers.
- The smoke-test page must cache-bust imported assets too, not just the HTML URL — `webllm-bundle.js` and `webllm-wasm.js` inherit the page query suffix.
- Smoke-test success requires checking for runtime/backend console failures, not just visible step output.
- Keep `-sASYNCIFY_STACK_SIZE=1048576` in the WASM build unless there is a verified replacement strategy.
- Qwen3 chat has **two** valid end-of-turn tokens: `<|im_end|>` (151645) and `<|endoftext|>` (151643). Both must be masked during the post-`</think>` "waiting for visible answer" window and treated as stop tokens. Canonical masking logic lives in `smoke-test/real-model-smoke.js`.
- **Custom-scorer registrations must be mirrored on both sides.** `eval/tasks/scorer-registrations.ts` (Bun) and `smoke-test/scorer-registrations.js` (browser) carry the same 13 scorer functions registered under the same names. The browser file intentionally imports `registerCustomScorer` from `./webllm-bundle.js` so both sides share the same registry instance — do not let a separate bundling step re-inline the `custom-scorers` module, or browser registrations become invisible to `score()`. When adding or editing a scorer, edit both files.

## Local llama.cpp dependency

`make wasm-build` compiles against a **local patched** `llama.cpp` at
`~/Repos/llama.cpp/` on branch **`webllm-browser-patches`**:

```bash
cd ~/Repos/llama.cpp && git checkout webllm-browser-patches
```

See [`docs/LLAMA_CPP_PATCHES.md`](docs/LLAMA_CPP_PATCHES.md) for the patch
inventory, rebase procedure, and troubleshooting. If a browser regression
reappears after a rebase, inspect that local branch before assuming the
bug is in this repo.
