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

- **Model-size ceiling: 8B parameters** (revised 2026-04-29 from prior
  30B). Rationale: the project's load-bearing use case is **agent +
  Three.js coexistence** in a single tab (browser-side LLM driving 3D
  agents alongside a renderer). Hardware baseline is **16 GB unified
  memory floor / 32 GB recommended / 128 GB development**. On the 16 GB
  floor, WebGPU sees ~10-11 GB; Three.js mid-complexity scenes take
  0.5-1 GB; KV cache + browser overhead another 1-2 GB → ~7-8 GB for
  the model. 8B Q4_K_M (~5 GB) fits comfortably with headroom. 14B Q4
  (~9 GB) fits on 32 GB+ but is too tight for the floor and is **not**
  load-bearing for project planning. Levers that justify themselves
  only via models >8B must be **deferred** (with the ceiling cited),
  not silently dropped. The prior 30B framing in archived items
  (`§C-v2-A`, `§22`, MEMORY64-bound resurrection paths) is retained
  for historical context but the practical retire-threshold is now 8B.
- **Hardware baseline doctrine.** All registrations, defaults, and
  workflow policies are sized for the **16 GB minimum / 32 GB
  recommended / 128 GB dev** tier. Three.js is assumed resident in
  the same tab; budget ~0.5-1 GB for scene + frame buffers + post-
  processing. The chat model + embedder + KV cache + scratch buffers
  must coexist with that. Default `contextLength` for chat models
  registered in `eval/models.ts` should reflect this — 4-8K is fine
  for agent dialogue, 32K is fine for embedders (no KV cache).
- **Single-model-active deployment.** Project ships **at most one
  chat model and one embedder loaded simultaneously**, or a single
  model used for both via tap-point self-embedding (bucket D —
  `ModelInference.embed(tokenIds)` taps the post-`output_norm`
  hidden state, same architecture truth source as the dedicated
  embedder path). Multi-model hot-loading is out of scope. KV-cache-
  per-conversation-on-shared-weights multiplexing is **deferred** —
  current `engine.ts` keeps one KV cache per loaded model, which is
  fine for single-active-conversation agents but doesn't support
  concurrent independent agent conversations on the same chat model
  weights. **Bucket D shipped 2026-04-30** — `engine.embed()` now
  dispatches encoder → causal-embedder → chat-model (gated per-model
  via `embeddingCapable: true` in the registration entry). Closure
  report and parity data:
  `eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`.
- **Per-binding 128 MiB cap doctrine** (lesson from bucket C
  Phase 4). WebGPU `maxStorageBufferBindingSize` is 128 MiB on
  Chrome/Apple regardless of total VRAM. The patched `ggml-webgpu`
  rejects ops with src/dst tensors exceeding this cap
  (`ggml-webgpu.cpp:4302-4307`). For models with vocab > ~65K
  (Qwen3 family, most modern models with token_embd > 128 MiB at
  f16), the canonical fix is **hybrid quant: only `token_embd.weight`
  quantized to Q4_K, all other weights f16**. One-line
  `llama-quantize --token-embedding-type Q4_K` invocation. Preserves
  parity ≥0.999 against f16 reference vectors (token_embd is a pure
  row lookup; per-row dequant error doesn't compound). Ship hybrid as
  the default for any embedder/chat model whose token_embd would
  exceed the cap; fall back to full Q4_K_M only if the agent use
  case explicitly trades quality for VRAM.
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
- **Rebase + sweep cycle doctrine.** When upstream `ggml-webgpu` moves
  and a rebase fires, every cycle classifies into one of three
  documented templates — pick the matching one and apply its decision
  rule:
  - **§27 (free win):** broad upside (e.g. +70-80% on IQ3_M from
    upstream's #22344 fast i-quant mat-vec). Adopt baseline; update
    canonical pins; close cycle. No follow-up needed.
  - **§28 (negative result):** the lever a prior cycle bet on closes
    *harder* (e.g. §C-v2-A gates moved 0.42× → 0.34× post-§27 because
    drafter Q8 didn't benefit from #22344). Document, retire the
    lever's resurrection paths, close cycle.
  - **§32 (small regression, accepted):** 5-of-6-models neutral, 1
    held a -6% regression. Don't revert the rebase — staying current
    has option value (next cycle's free wins land cleanly). Document
    and accept; pin the new canonical baseline.
- **Cap-probe doctrine — bump first, characterize second**
  (§31b lesson). When a measurement hits a cap at a configurable
  value, immediately try bumping the configuration to confirm whether
  the cap is configuration-bound or toolchain/runtime-bound. The bump
  is cheap (1 line + 1 rebuild attempt). §31a missed this step and
  landed a "configured-ceiling-bound, not hardware-bound" framing
  that understated the constraint by one layer (the configuration
  ceiling *was* the toolchain ceiling — Emscripten 5.0.6 wasm-ld
  hard-caps `--max-memory` at 16 GiB).
- **Pre-rebase baseline doctrine** (§32a lesson). When a rebase is the
  planned probe trigger and the previous outcome classified as §32
  template (small regression, accepted), capture pre-rebase
  profile-mode (`make smoke-bench PERF_MODEL=<m> PERF_RUNS=3`) on the
  canonical 6 *before* the rebase. Cost: ~3 min wall per model.
  Pay-off: a §32a-style follow-on probe gets a same-model baseline
  for diagnosis (would have diagnosed conclusively here rather than
  via the cross-model proxy that §32a had to use). Pre-rebase
  baselines are pinned at
  `eval/reports/pre-rebase-baselines-<DATE>/SUMMARY.md` with a
  ~1-month freshness window.
- **Greedy by default for accuracy bench (set 2026-05-04).** The
  `make bench-*` accuracy pass now pins **temperature 0** unless the
  caller explicitly overrides via `--eval-temperature <n>`. Rationale:
  small/weak models scored with temperature ≥ 0.4 had per-task
  variance large enough to flip the dimension mean by 25-33% absolute
  on n=1 reruns (tinyllama 04-26 0.35 → 04-28 0.24 was the trigger,
  emb-001 flipping 1→0→1 across identical code being the smoking
  gun). Greedy eliminates the sampling noise so a single pass per
  model gives usable signal. Speed pass is unaffected — `chat-smoke.ts`
  still uses the profile's native temperature so tok/s headlines stay
  comparable to pre-cutover history. **Pre-cutover canonical
  baselines (April 2026 and earlier) are a separate series**: they
  were captured at profile-native temps (mostly 0.6) and must not be
  cross-compared with greedy scores on the same dashboard chart.
  Tag sessions accordingly via `bench_session_started.evalTemperature`.
- **Bench session envelope (set 2026-05-04).** `eval/bench.ts` mints
  a `sessionId` per invocation and emits `bench_session_started` /
  `bench_session_complete` live events bracketing the multi-model
  loop. Each per-model `eval_started` carries the parent `sessionId`
  so the dashboard can roll up overall progress (model X/Y · task
  A/B). Threading: `WEBLLM_BENCH_SESSION_ID` env var → child harness
  → URL `?session=` → smoke page → `eval_started` payload. Sessions
  are in-memory only on the live-server (no SQLite table); a backend
  restart mid-session loses the rollup until the next session starts.
- **TODO archival cadence.** When a top-level TODO block closes
  (all sub-items resolved, follow-ups landed, child probes filed),
  move the block out of `TODO.md` to `TODO_ARCHIVE.md` and leave a
  4-8-line closure stub in its place. The stub stays inline in
  `TODO.md` and links to:
  - the canonical closure / validation report under
    `eval/reports/<area>/`, and
  - the archived block (`TODO_ARCHIVE.md`).

  **Trigger:** all listed sub-items + follow-ups carry
  **CLOSED `<DATE>`** markers and no new probes are queued under
  the heading. **Don't archive prematurely** — partially-closed
  blocks stay in `TODO.md` so the queued sub-items remain visible
  in the active surface.

  **What stays in `TODO.md`:** a watch-list-style closure stub
  (status, headline metric, links) so future sessions reading the
  watch list see the closed lever at a glance. **What moves to
  archive:** the full block (rationale, phasing, risk register,
  per-phase commit table, probe artifacts list — the load-bearing
  reasoning).

  See commits `41b964c` (close MEMORY64 follow-ups) and the
  follow-on archive commit (replaces ~270 lines of closed migration
  detail with a 12-line closure stub) as the canonical example.
  TODO archival is its own commit with `docs(TODO):` prefix —
  don't bundle with the closures themselves so a `git revert`
  of the archive is reversible without touching the closures.

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
  `?ingest=<url>`. **For throwaway/diagnostic runs, use `?ingest=off` rather
  than killing the dashboard.** Examples: iterating on a patched WASM build
  before committing, sanity-checking a fix that may not land, or running a
  diagnostic sweep where the gate hasn't yet been adjudicated. Killing
  `dashboard-serve` for throwaway runs is heavier-handed than necessary and
  takes the dashboard offline for any concurrent committed-bench traffic;
  prefer the per-URL `?ingest=off` toggle.
- **Bun harnesses** (`eval/chat-smoke.ts`, `eval/bench.ts`, etc.) require
  `WEBLLM_LIVE_BENCH_URL=http://localhost:8033` to publish events.

  Note: `make smoke-bench` (via `eval/perf.ts`) does **not** currently
  thread `?ingest=off` through to the smoke URL — diagnostic sweeps run via
  `make smoke-bench` will hit the dashboard if it's up. If the runs are
  throwaway and you want to keep the live DB clean without taking the
  dashboard down, plumb `ingest=off` through `eval/perf.ts`'s
  `extraParams` block (one-line addition gated on a `--no-ingest` flag or
  `WEBLLM_NO_INGEST` env var) rather than killing port 8033.

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

## HuggingFace downloads — always use `hfdownloader`

Always use the `hfdownloader` CLI (`/Users/probello/.local/bin/hfdownloader`)
for HuggingFace model / dataset fetches. **Do not rely on**
`huggingface_hub.snapshot_download`, `huggingface-cli download`, or
transformers' implicit `from_pretrained()` fetch when the goal is
acquiring weights for a downstream consumer.

**Why:** `hfdownloader` is fast and resumable (parallel range
requests, 8 connections per file by default, 3 concurrent files).
The Python paths are slower, single-stream, and easier to wedge —
observed 2026-04-30 during bucket D ref capture: an in-flight
`transformers.from_pretrained()` died after ~10 minutes mid-fetch
with no error, leaving 4 GB of partial shards on disk. Restart
resumed but lost wall time.

**How to apply:**

- Pre-fetch with `hfdownloader download <owner>/<name>` *before*
  running any Python script that loads weights (transformers /
  sentence-transformers will then load from the warm HF cache
  instantly).
- For datasets, add `--dataset`.
- Filter LFS artifacts (e.g., GGUF quants):
  `hfdownloader download <repo> --filters q4_0,q5_k_m`.
- Pin a revision: `-b <branch_or_sha>`.
- Use `--dry-run` first to see the file list / total size.
- Tokens: `HF_TOKEN` env or `--token`.
- Files land in `~/.cache/huggingface/` by default — same layout
  the Python tooling expects, so cache-warmth is shared.

This applies to ref-capture scripts under `eval/reports/<probe>/`
(see `bucket-d-probe-2026-04-29/capture-refs.py` for the canonical
pattern: `hfdownloader download Qwen/Qwen3-8B` first, then
`uv run --no-project --with-requirements ...` to run the script
that calls `from_pretrained()`).

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
