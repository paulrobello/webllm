# ENH-007 — Thread `--no-ingest` Through `make smoke-bench`

> **Status**: proposed · **Effort/risk**: Low · **Depends on**: nothing (coordinate trivially with anything else editing `eval/perf.ts`)

## Goal

`make smoke-bench NO_INGEST=1 ...` (and `bun run eval/perf.ts --no-ingest ...`) runs diagnostic
sweeps whose browser smoke runs do NOT post `run_complete` to the live dashboard — without taking
the dashboard down. This implements exactly the change CLAUDE.md already specifies.

## Current state

- CLAUDE.md, Live dashboard section: browser smoke runs auto-post to `http://localhost:8033` by
  default; `?ingest=off` is the documented per-URL off switch — but "`make smoke-bench` (via
  `eval/perf.ts`) does **not** currently thread `?ingest=off` through to the smoke URL… plumb
  `ingest=off` through `eval/perf.ts`'s `extraParams` block (one-line addition gated on a
  `--no-ingest` flag or `WEBLLM_NO_INGEST` env var) rather than killing port 8033."
- So: the URL parameter exists and works; only the perf-harness plumbing and the Make passthrough
  are missing.

## Implementation steps

1. **Locate the block**: in `eval/perf.ts` (~507 lines), find the `extraParams` construction the
   CLAUDE.md note references (grep `extraParams`). Note how existing boolean-ish params are
   appended to the smoke URL.
2. **Flag + env**: in `perf.ts`'s arg parsing (find the existing pattern — it already parses
   `PERF_MODEL`-adjacent options), accept `--no-ingest`; also honor
   `process.env.WEBLLM_NO_INGEST` (any non-empty value). Either source → append `ingest=off` to
   `extraParams`.
3. **Make passthrough**: in the `Makefile`'s `smoke-bench` target, mirror the existing
   `PERF_MODEL`/`PERF_RUNS` variable pattern: `NO_INGEST` var that, when set, adds `--no-ingest`
   to the perf.ts invocation.
4. **Docs — three touches**:
   - CLAUDE.md: rewrite the "does **not** currently thread" note to describe the new flag (keep it
     short; the note exists to route agents, and the gap it warns about is now closed).
   - `make help` line for the variable if the help target enumerates variables (check).
   - The env-var reference table (DOC-006, if landed): add `WEBLLM_NO_INGEST`.
5. **Do not** change the smoke page's default-on ingest behavior — committed bench runs keep
   posting by default; this is opt-out plumbing only.

## Files to touch

- `eval/perf.ts`, `Makefile`, `CLAUDE.md`, (`docs/reference/environment.md` if it exists by then)

## Verification

1. `make checkall`.
2. Behavioral: with `make dashboard-serve` up, note the dashboard run count; run
   `make smoke-bench PERF_MODEL=<smallest registered model> PERF_RUNS=1 NO_INGEST=1`; confirm the
   count is unchanged. Re-run without `NO_INGEST=1`; confirm the count increments (default
   behavior intact).
3. Grep check: the constructed smoke URL (perf.ts logs it, or add a temporary log during
   verification and remove it) contains `ingest=off` exactly when the flag/env is set.

## Rollback

Revert the flag plumbing; behavior returns to always-ingest. One-commit rollback, no data or
schema involvement.
