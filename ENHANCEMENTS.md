# Enhancement Ideas

> **Date**: 2026-07-14 (commit `eccf6e6`) · **Companion to**: `AUDIT.md` (defects) — these are
> opportunities *beyond* the audit findings. Each idea has a full implementation plan at
> `docs/fable/ENH-XXX-<slug>.md`, written to be executable by a smaller model without re-analysis.
>
> Graph context (par-mem, repo_id `webllm`): 235 files (197 TS / 21 JS / 4 py / 1 cpp), 8,739 call
> edges, 1,515 functions + 1,190 methods. Bridge analysis ranks `loadAndTest` (out-degree 137),
> `live-server fetch` (out 47), and `chatCompletionWithConversation` (out 47) as the fragile
> connectors; `ConversationPool.set` (in-degree 144) and `GgufParser.parse` are articulation
> points. Churn×complexity hotspots are dominated by the committed p0/p1 Emscripten bundles
> (remediated by ARC-009) — after that cleanup, the live hotspots are `real-model-page`'s
> `runChat` and `WebLLM.loadModelFromBuffer`.
>
> All ideas respect the project's workflow policies: 8B model ceiling, single-model-active,
> probe-first for unmeasured levers, complexity scored on maintenance burden/risk surface (not
> implementation time). Effort ratings below are **maintenance-burden + risk** ratings per that
> doctrine.

| ID | Title | Expected impact | Effort/risk | Status | Plan |
|----|-------|-----------------|-------------|--------|------|
| ENH-001 | Automated browser regression lane (Playwright) | High — converts the manual ship gate into a scriptable one; prerequisite safety net for ARC-001/QA-005 | Medium | ✅ Done — 2026-07-15 (`9c59fa7`) | [plan](docs/fable/ENH-001-playwright-browser-lane.md) |
| ENH-002 | Streaming token API through `chat()` and the worker proxy | High — user-visible latency win for agent/Three.js UX | Medium | ✅ Done — 2026-07-15 (`a652447`) | [plan](docs/fable/ENH-002-streaming-chat-api.md) |
| ENH-003 | KV-cache + scratch accounting in MemoryPool, with pressure events | Medium-High — makes the 16 GB-floor doctrine enforceable at runtime | Medium | 📋 Open | [plan](docs/fable/ENH-003-kv-memory-accounting.md) |
| ENH-004 | Extract `ConversationTurnRunner` from `chatCompletionWithConversation` | Medium — de-risks the engine's highest out-degree bridge (47) for future features | Medium | 📋 Open | [plan](docs/fable/ENH-004-conversation-turn-runner.md) |
| ENH-005 | Probe: per-conversation KV multiplexing cost for multi-NPC agents | High if it lands (unlocks concurrent NPC conversations); probe itself is cheap | Probe: Low · Follow-on: High | 📋 Open | [plan](docs/fable/ENH-005-kv-multiplex-probe.md) |
| ENH-006 | Persist bench sessions to SQLite on the live dashboard | Low-Medium — closes the documented "restart loses session rollup" gap | Low | 📋 Open | [plan](docs/fable/ENH-006-bench-session-persistence.md) |
| ENH-007 | Thread `--no-ingest` through `make smoke-bench` | Low — implements the one-liner CLAUDE.md already specifies for throwaway sweeps | Low | 📋 Open | [plan](docs/fable/ENH-007-no-ingest-flag.md) |
| ENH-008 | Verified dead-helper sweep of the eval harness | Low — trims real orphans surfaced by graph analysis, with the false-positive discipline the audit established | Low | 📋 Open | [plan](docs/fable/ENH-008-eval-dead-code-sweep.md) |

## Status tracking

The **Status** column is the source of truth for each idea's lifecycle. Keep it
current as work happens — an out-of-date status column is worse than none.

- **📋 Open** — not started.
- **🚧 In progress** — actively being worked; set this when implementation
  begins (not when planning begins).
- **✅ Done** — shipped **and** verified through the project gate
  (`make checkall`, plus the item's own plan-defined verification). When an item
  completes, update its Status cell to `✅ Done — <YYYY-MM-DD> (<commit-sha>)`
  *in the same commit* that lands or finalizes the work (a separate
  `docs(enh): mark ENH-XXX done` commit is fine if the implementation commit
  already shipped). **Also add a `> **Status**:` banner at the top of the
  item's section** so the done/in-progress/reverted state is visible when
  reading the detail, not only in the table. **Completed items stay in this
  file as a record** — do not delete the row, its section below, or its plan
  link; they are the durable artifact for what was built and why.
- **↩️ Reverted / 🔄 Superseded** — if a done item is later rolled back or
  replaced, change the status to one of these with a one-line reason (and a link
  to the follow-up) rather than removing the row.

## ENH-001 — Automated browser regression lane (Playwright)

> **Status**: ✅ Done — shipped 2026-07-15 (`9c59fa7`, with follow-up fixes `133173c` / `1f5fa90`).
> Verified via `make test-browser` (1 test passes) + `make checkall` (799 pass / 0 fail). Lane lives
> in `playwright.config.ts` + `tests-browser/smoke.spec.ts`; documented in `CLAUDE.md`.

The GPU inference paths — the code ARC-001 must refactor — are verified today only by the manual
agentchrome smoke workflow; Bun tests cannot touch them (36 permanently-skipped tests are the
symptom, QA-004). A Playwright + Chrome lane that loads the existing smoke page headless with the
smallest registered model, asserts the `#log` step sequence, and fails on console errors converts
the project's real ship gate into a one-command check (`make test-browser`). This directly
implements the user's standing rule ("use Playwright with Chrome to create tests for web apps";
agentchrome stays for interactive debugging) and is the single highest-leverage safety
investment before the Phase-2 refactors. **Impact**: high (every future GPU change gets an
automated gate; ARC-001's per-port verification becomes cheap). **Effort/risk**: medium — new
devDep and a CI-incompatible local target (needs a real GPU), but zero changes to shipped code.

## ENH-002 — Streaming token API

> **Status**: ✅ Done — shipped 2026-07-15 (`a652447` + follow-ups `517c7a0`, `7b427e4`, `28b5a88`).
> Verified via `make checkall` (green) + browser regression on `qwen3-0.6b-q4f16` in worker mode:
> visible answer streams progressively, `<think>` reasoning never flashes (confirmed across a
> 511-token think-only turn + a turn with a visible answer), and the stop button halts mid-stream.
> **Design note**: discovery showed streaming generators already existed inline + via the worker
> stream protocol; the shipped work makes deltas + `result.text` **visible-only** (the one
> intentional behavior change — `<think>` excluded) and adds additive `onToken` / `onThinking`
> callbacks (inline + worker). `chat()` returns visible-only text but takes no callbacks (drains).

`Generator.generate()` is already an AsyncGenerator yielding token IDs, and the tokenizer already
ships a `StreamingDecoder` (`tokenizer.ts:954`) implementing the full-redecode delta pattern —
but the public chat surface drains the generator and decodes once, so consumers get nothing until
generation completes. Exposing an `onToken` callback (or async-iterable variant) through
`chatCompletion` options, the conversation path, and the worker proxy (per-delta `postMessage`)
turns multi-second generations into progressive output — the difference between a frozen NPC and
a talking one at the 2-4 s/decision budget the NPC analysis established. **Impact**: high,
user-visible. **Effort/risk**: medium — the plumbing exists at both ends; the work is the middle
(engine option, worker RPC streaming events, proxy mirror) plus stop-token/steering interaction
care (suppressed-then-flushed text must not be double-emitted).

## ENH-003 — KV + scratch memory accounting and pressure events

ARC-002 (audit) wires MemoryPool for model *weights*. The doctrine budget on the 16 GB floor also
charges KV cache and scratch buffers (~1-2 GB) against the same envelope, and Three.js coexists in
the tab. This enhancement completes the accounting: charge KV allocation at model registration
(computable from `contextLength × layers × kvHeads × headDim × 2 × f16`), charge the WASM/GPU
scratch arenas, and emit the pressure/eviction events the README already promises when the budget
tightens. **Impact**: medium-high — the load-bearing coexistence scenario becomes observable and
enforceable instead of aspirational. **Effort/risk**: medium — arithmetic + event plumbing, no
inference-path changes; risk is overcounting (double-charging WASM-side allocations), handled in
the plan.

## ENH-004 — `ConversationTurnRunner` extraction

`chatCompletionWithConversation` (`engine.ts:842-1195`, CC 44, out-degree 47 — the engine's widest
bridge) single-handedly does lock chaining, KV snapshot load/save, prefill, decode, streaming, and
stop handling. ARC-004 deliberately deferred this split. Extracting a `ConversationTurnRunner`
with named phases makes ENH-002 (streaming) and ENH-005 (multiplexing) land as localized changes
instead of further growth of a 350-line method. **Impact**: medium — pure maintainability, but on
the exact code path every future conversation feature touches. **Effort/risk**: medium — behavior-
preserving extraction with strong existing tests around conversation flows; sequence after
ARC-004 (same file).

## ENH-005 — Probe: per-conversation KV multiplexing for multi-NPC agents

CLAUDE.md defers KV-per-conversation-on-shared-weights multiplexing; the NPC-control analysis
(2026-05-01) wants multiple concurrent NPC conversations on one chat model. The infrastructure
for a cheap probe already exists: `ConversationPool` KV snapshots serialize/restore per
conversation today. The probe measures snapshot-swap cost (save + load) at realistic NPC scales
(2/4/8 conversations, 4-8K contexts) on a canonical 8B model, producing the number that decides
between "swap is fine — document the batched-tick pattern" and "build true multi-slot KV". Per
probe-first doctrine the probe declares thresholds up front (plan sets them). **Impact**: high if
multiplexing lands; even a negative result retires speculation with data. **Effort/risk**: probe
is low (harness + report); the multi-slot follow-on is high and only proceeds on a failed-swap
verdict + explicit user opt-in.

## ENH-006 — Bench-session persistence

CLAUDE.md documents the gap: bench sessions are in-memory only on the live server, so a backend
restart mid-session loses the dashboard's model-X-of-Y rollup until the next session. A small
`bench_sessions` SQLite table (the DB layer and idempotent-import pattern already exist in
`eval/live-db.ts`) plus rehydration on startup closes it. **Impact**: low-medium — quality-of-life
for long multi-model bench runs. **Effort/risk**: low.

## ENH-007 — `--no-ingest` for `make smoke-bench`

CLAUDE.md explicitly specifies this: `make smoke-bench` doesn't thread `?ingest=off`, so
diagnostic sweeps hit the live dashboard DB unless the dashboard is taken down. The fix it
prescribes: plumb `ingest=off` through `eval/perf.ts`'s `extraParams` block behind a
`--no-ingest` flag or `WEBLLM_NO_INGEST` env var, then document it in CLAUDE.md and the (new,
DOC-006) env-var reference. **Impact**: low but directly requested by the project's own docs.
**Effort/risk**: low — one flag, one URL param, one doc line.

## ENH-008 — Verified dead-helper sweep of the eval harness

par-mem `find_dead_code` surfaces ~40 zero-caller candidates. Most are false positives (CLI
`main()` functions invoked at module top level; the audit filed two false-positive classes as
par-mem feedback). The residue worth acting on after manual verification: unused count helpers in
`eval/live-db.ts` (`countRuns`, `countSystemProfiles`, `countEvals`), possible orphan scorer
helpers (`fishLine`, `vegLine` in `eval/tasks/scorer-registrations.ts` — verify against the
browser mirror before touching), and bench fixtures (`bench/*.bench.ts` helpers). The plan
encodes the verification discipline so a smaller model doesn't delete live code. **Impact**: low —
hygiene; keeps future graph analytics clean post-ARC-009. **Effort/risk**: low with the
verification steps followed exactly.

---

*Not proposed*: speculative-decode resurrection (retired at 0.34× gate, §28 negative closure),
JSEP+MEMORY64 integration paths (negative closure 2026-05-08, re-evaluation triggers documented),
>8B-model levers (policy ceiling), tokenizer micro-optimization (heap `push` in-degree 192 is BPE
merge-loop internal fan-in, not a measured bottleneck — fails the measured-gain bar of the
quick-wins override).
