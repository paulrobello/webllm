# Contributing to WebLLM

How to set up a development environment, run the ship gate, exercise the
browser regression workflow, and land changes that match the project's
conventions. Audience: contributors preparing their first PR.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [The ship gate](#the-ship-gate)
- [Browser regression workflow](#browser-regression-workflow)
- [Commit conventions](#commit-conventions)
- [Working on the WASM backend](#working-on-the-wasm-backend)
- [Adding a model, quant, or chat template](#adding-a-model-quant-or-chat-template)
- [Related Documentation](#related-documentation)

## Prerequisites

**For library/eval work:**

- **Bun** — tooling, tests, and the eval harnesses all run on Bun. See
  `package.json` for the pinned version.
- **Chrome with WebGPU** — required for the browser regression workflow
  and any GPU-path change. The project is developed and regression-tested
  against Chrome; the WASM build assumes JSPI support and a 128 MiB
  `maxStorageBufferBindingSize` (Chrome-shaped constraints). Other
  WebGPU-shaped browsers may work but are not tested.

**Only for WASM backend work (`make wasm-build`):**

- **Emscripten SDK** sourced at `~/emsdk/emsdk_env.sh`.
- **A local patched `llama.cpp` checkout** at `~/Repos/llama.cpp/` on branch
  `webllm-browser-patches`. See
  [`docs/LLAMA_CPP_PATCHES.md`](docs/LLAMA_CPP_PATCHES.md) for the patch
  inventory and rebase procedure. Most contributions do not need this — it
  is only required when changing the C/C++ backend or rebuilding the WASM
  artifacts.

## Setup

```bash
git clone <this repo>
cd webllm
make install          # bun install
make checkall         # verify the clean-tree gate is green before you start
```

`make help` lists every target with descriptions and is the single source
of truth for tooling.

## The ship gate

`make checkall` runs format + lint + typecheck + test. A change is not done
until it is green:

```bash
make checkall         # fmt + lint + typecheck + typecheck:tests + test
```

Run a single test file or pattern with:

```bash
bun test tests/<file>.test.ts
bun test -t "<pattern>"
```

The Bun suite covers the orchestration layer, tokenizers, sampling,
persistence round-trips, and error-codec paths. It cannot exercise the
GPU/WASM forward path — that is the browser workflow below.

> **Note:** A pre-commit hook (pinned `gitleaks` + `detect-private-key` +
> language gates wired to the Make targets) runs on every push. Install it
> with `pre-commit install` if it is not already active in your checkout.

## Browser regression workflow

GPU-path changes (forward graph, RoPE, masks, sampling, stop-token
handling, WASM ABI) are gated by a manual browser smoke run, not the Bun
suite. The canonical workflow (see `CLAUDE.md` for the long version):

1. Start the static server: `make smoke-serve` (port 8031 by default).
2. Reuse the existing agentchrome session and tab rather than launching a
   new browser window.
3. Navigate the same tab to a cache-busted URL like
   `http://localhost:8031/?v=<N>` (the page and its imported assets inherit
   the suffix).
4. Check **both** the page `#log` text **and** the browser console. The
   smoke run passes only when the visible steps succeed **and** no relevant
   runtime console errors are emitted. `adapter_info:` lines from the
   WebGPU backend are benign informational output, not failures.
5. For throwaway/diagnostic runs, use `?ingest=off` on the smoke URL
   rather than killing the dashboard — this keeps the live DB clean for
   concurrent committed-bench traffic.

The canonical home of the Qwen3 dual stop-token masking logic and the
ship/no-ship harness is `smoke-test/real-model-smoke.js` — read it before
touching stop-token or end-of-turn handling.

## Commit conventions

This repo follows the **separate-commits doctrine**: specs, plans, TODO
updates, and policy changes land in their own commit **before** the
implementation chunk they justify, so a `git revert` of the implementation
does not nuke the reasoning. Use the conventional prefix that matches the
content:

- `feat(...)` — new functionality
- `fix(...)` — bug fix
- `refactor(...)` — structural change with no behavior delta
- `docs(spec):`, `docs(plan):`, `docs(TODO):`, `docs(rebase-...)` —
  documentation / specs / plans / TODO / rebase-closure commits
- `chore(...)`, `style(fmt):`, `test(...)` — tooling, formatting, tests

Examples mined from `git log --oneline`:

```text
docs(rebase-2026-07-14): closure SUMMARY — §27 perf-neutral maintenance
feat(stage5.2): add gemma-4-e2b-warm to bench-full profile set
fix(wasm): drop stranded JSEP-probe export from CMakeLists
refactor(architecture): wire MemoryPool, trim exports
```

## Working on the WASM backend

Only relevant if you are changing the patched `ggml-webgpu` C/C++ or
rebuilding the WASM artifacts. Checkout the local patched branch first:

```bash
cd ~/Repos/llama.cpp && git checkout webllm-browser-patches
```

Then back in this repo:

```bash
make wasm-build         # production build
make wasm-build-debug   # -sASSERTIONS=1, preserves abort messages
```

Every rebase onto a newer upstream `llama.cpp` master is classified into
one of three documented templates (§27 free win, §28 negative result, §32
small-regression-accepted) — read `docs/LLAMA_CPP_PATCHES.md` and
`CLAUDE.md`'s "Rebase + sweep cycle doctrine" before rebasing. Regression
lessons that have shipped real bugs (ASYNCIFY stack size, JSPI export
mirroring across WASM targets, async-readback allocation) are codified in
`CLAUDE.md` — do not repeat them.

## Adding a model, quant, or chat template

- **New model registration** — add an entry to `eval/models.ts`, then
  follow [`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md) "How to Add
  Support". Run `make bench-eval-models` to confirm the registration
  parses.
- **New quantization type** — requires a WGSL kernel in the local
  `llama.cpp` patch branch plus a WASM rebuild; see `docs/MODEL_SUPPORT.md`
  "Adding a Quantization Type".
- **New chat template** — add detection + a formatter in
  `src/inference/chat-template.ts`; see `docs/MODEL_SUPPORT.md` "Adding a
  Chat Template".

> **Note:** `docs/superpowers/` (specs and plans) is gitignored. Specs and
> plans that land there must be force-added (`git add -f`); see commits
> `ae68bbe`, `b23ccc9`, `66bc603` for the convention. Specs live there so
> the load-bearing reasoning behind a code change is version-controlled
> alongside the change without cluttering the default doc surface.

## Related Documentation

- [`README.md`](README.md) — project overview, Quick Start, API surface
- [`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md) — supported models, quants, architectures, embeddings
- [`docs/LLAMA_CPP_PATCHES.md`](docs/LLAMA_CPP_PATCHES.md) — local patch inventory and rebase procedure
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — benchmark methodology and metrics
- [`docs/reference/environment.md`](docs/reference/environment.md) — benchmark and harness environment variables
- [`CLAUDE.md`](CLAUDE.md) — repo guidance, regression lessons, workflow policies
