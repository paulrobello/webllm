# ENH-008 — Verified Dead-Helper Sweep of the Eval Harness

> **Status**: proposed · **Effort/risk**: Low (with the verification protocol followed exactly)
> **Depends on**: ARC-009 (bundle gitignore) first — it removes the analytics noise; run this sweep against a **reindexed** par-mem afterward.

## Goal

Remove the genuinely-orphaned helpers that par-mem's graph reachability surfaced, without touching
the many false positives, so the codebase and future graph analytics stay clean.

## Current state — candidate classification (par-mem `find_dead_code`, 2026-07-14)

The raw list has ~40 entries. Classified:

**DO NOT TOUCH — false positives by construction:**
- Every `main()` in `eval/*.ts`, `eval/probes/*.ts`, `scripts/*.ts`, and the Python parity tools —
  these are CLI entry points invoked at module top level (`main()` call at file bottom) or via
  Make targets. Zero *graph* callers ≠ dead.
- `log_receiver.py::do_POST` — the whole file is deleted by SEC-001; out of scope here.
- Nested closures flagged individually (`bench::main::cases`, `browser-smoke::findSmokeTab::any`,
  `runner::readLine::onData`, probe-internal helpers) — call-graph resolution artifacts inside
  live functions. The audit filed these false-positive classes as par-mem feedback already.
- `eval/live-server.ts::createLiveServer` / `parseJsonBody` — the server IS the entry point;
  verify trivially (grep their names in the file) and skip.

**REAL CANDIDATES — verify then remove:**
1. `eval/live-db.ts`: `countRuns` (:143), `countSystemProfiles` (:215), `countEvals` (:240)
2. `eval/tasks/scorer-registrations.ts`: `fishLine` (:155), `vegLine` (:161)
3. `bench/generation.bench.ts`: `mockForwardPass` (:19); `bench/scheduler.bench.ts`: `makeTask` (:4)
4. `eval/causal-embedder-parity.ts`: `UNRELATED_PAIRS` (:411), `embedSentence` (:431)
5. `eval/live-events.ts`: `LiveEventStore::constructor` flagged — constructors get flagged
   spuriously (known par-mem false-positive class); verify `new LiveEventStore` exists and skip.

## Verification protocol (run per candidate — no deletion without ALL steps clean)

1. `grep -rn "<name>" --include="*.ts" --include="*.js" --include="*.md" .` from repo root
   (excluding `node_modules`, `dist`) — catches direct calls, re-exports, string references,
   test-name mentions, and the **browser mirrors** under `smoke-test/`.
2. par-mem `get_symbol_context` (repo_id `webllm`) on the symbol — confirms zero callers in the
   graph AND surfaces type-level references.
3. `git log --oneline -3 -- <file>` — a helper added in the last ~2 weeks may be staged for
   upcoming work; if recent, leave it and note why.
4. Special cases:
   - `fishLine`/`vegLine`: these live in the **mirrored** scorer file. Check
     `smoke-test/scorer-registrations.js` for the same helpers: if the browser side HAS them and
     *uses* them inside a scorer, the Bun side not using them is **mirror drift — a bug, not dead
     code**: fix the Bun scorer to match the browser behavior instead of deleting (and note that
     the ARC-011 parity test would have caught it). If both sides define-and-don't-use, delete on
     both sides together.
   - `bench/` helpers: check how benches run (`grep -rn "bench" Makefile package.json`); if the
     bench files themselves are stale (referencing the inert `Scheduler` per ARC-002 — decide
     whether `bench/scheduler.bench.ts` should be deleted whole rather than de-linted), flag the
     file-level question in the fix report instead of surgically removing one helper from a dead
     file.
   - `UNRELATED_PAIRS`/`embedSentence`: parity harnesses run rarely; check the file's own
     invocations (a self-contained `main()` may reference them conditionally) before removing.
5. After each file's removals: run that file's consumers — `bun test` for anything imported by
   tests, the harness's own `--help`/dry invocation for CLIs (`bun run eval/<file>.ts --help` or
   the Make target that drives it).

## Implementation steps

1. Execute the protocol above per candidate; collect the surviving removals into one commit-sized
   change per directory (`eval/`, `bench/`).
2. Where a removal empties a now-unused import, remove that import too (orphans created by your
   change — in scope; pre-existing unrelated dead code — out of scope, note it instead).
3. Reindex par-mem (`index_directory`, incremental) and re-run `find_dead_code`: the report should
   shrink by exactly the removed entries; any *new* entries are fallout from your removals —
   investigate before finishing.
4. Append any newly-observed false-positive classes to `~/Repos/PAR-MEM-FEEDBACK.md` (read first,
   dedupe, evidence bar per the standing instructions).

## Files to touch

- `eval/live-db.ts`, `eval/tasks/scorer-registrations.ts` (± `smoke-test/scorer-registrations.js`),
  `bench/generation.bench.ts`, `bench/scheduler.bench.ts`, `eval/causal-embedder-parity.ts`
- Never: anything under **DO NOT TOUCH** above.

## Verification

1. `make checkall`.
2. Each touched harness still runs (step-5 spot checks recorded in the fix report).
3. Post-reindex `find_dead_code` delta matches the removal list exactly.

## Rollback

Deletions only, in small per-directory commits — `git revert` restores any helper a future probe
turns out to need. No API surface involved (eval tooling is not published).
