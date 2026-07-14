# Evaluation Reports

What lives under `eval/reports/`, how reports are named, what a closure
report must contain, and how `make import-reports` consumes the tree.

> **Note:** This directory is gitignored as a whole (`eval/reports/` in
> `.gitignore`) because it accumulates runtime-generated JSON, SQLite
> databases, and large artifacts. Closure reports (`SUMMARY.md`) and
> durable reference data are force-added (`git add -f`) so they are
> version-controlled; throwaway run JSONs and `.db` files stay local.

## Directory layout

```text
eval/reports/
├── README.md                          (this file)
├── archive/                           historical/superseded artifacts
│   ├── smoke-runs-*.db                dashboard SQLite snapshots
│   ├── pre-greedy-stragglers-*.json   pre-cutover run records
│   ├── wedged-run-*.json              captured wedge reproductions
│   └── 2026-04-24T*-*.json/html       early loose run records
├── smoke-runs/                        per-run JSON archive (live, not tracked)
├── <area>-<date>/                     closure reports (force-added)
│   └── SUMMARY.md
├── pre-rebase-baselines-<date>/       perf baselines captured before a rebase
└── llama-cpp-rebase-<date>/           rebase sweep matrices + conflict logs
```

## The `<area>-<date>` convention

A probe, campaign, or rebase closes by writing a directory named
`<area>-<date>/` where:

- `<area>` is a short kebab-case slug for the work (`bucket-d-parity`,
  `gemma-4-e2b-validation`, `llama-cpp-rebase-2026-07-14`,
  `encoder-parity-2026-04-28`, etc.).
- `<date>` is the ISO closure date (`YYYY-MM-DD`).

The directory holds the sweep matrices, parity JSONs, conflict logs, and
raw probe output, plus exactly one `SUMMARY.md` that a future session
reads first.

## What a SUMMARY.md contains

Every closure report's `SUMMARY.md` carries, at a minimum:

- **Status** — closed / negative / deferred / blocked, with a one-line
  headline metric (e.g. "70.8% eval at greedy temp=0", "cos >= 0.999 vs
  PyTorch f16 reference", "§27 perf-neutral, 6/6 within ±1.0%").
- **What was measured** — the probe or sweep setup, the models/quant
  matrix, the pass/fail thresholds declared up front.
- **The decision** — ship / demote / defer / revert, and which downstream
  levers it gates.
- **Links** — to the canonical pre-rebase baseline (for rebase sweeps), to
  related reports, and to the `TODO.md` / `TODO_ARCHIVE.md` entry that
  tracks the work.

This convention (58 closure `SUMMARY.md` reports at the time of writing)
plus the `TODO.md` archival stubs form the project's navigable decision
record. See `CLAUDE.md`'s "TODO archival cadence" for the closure →
archive handoff.

## Archive policy

Move artifacts into `archive/` when they are:

- Superseded by a newer baseline (old `smoke-runs-*.db` snapshots kept for
  cross-day audits, then archived).
- Throwaway reproductions that have served their diagnostic purpose
  (`wedged-run-*.json`, `pre-greedy-stragglers-*.json`).
- Early or pre-convention run records that no longer match the
  `<area>-<date>` shape (the 2026-04-24 timestamped root files).

`archive/` is still walked by `make import-reports` (the walker recurses),
so moving a run record into `archive/` does not lose its data from the
dashboard — the import is idempotent by `runId` / `evalId`, so re-importing
a moved file is a no-op.

## How `make import-reports` consumes the tree

`eval/import-reports.ts` walks `eval/reports/` recursively (stack-based
`readdirSync` over every subdirectory, including `archive/`), classifies
each JSON by shape (speed run vs eval report vs smoke run), and POSTs any
the dashboard has not already recorded (idempotent by `runId` / `evalId`).
It is safe to re-run after the dashboard comes back online or after moving
files between subdirectories.

```bash
make dashboard-serve     # start the backend on port 8033
make import-reports      # backfill every JSON under eval/reports/ the DB is missing
```

The walker deliberately skips non-JSON files (HTML reports, `.db` /
`.db-shm` / `.db-wal` SQLite sidecars, Markdown) — only JSON run/eval
records are ingested.

## Related Documentation

- [`CLAUDE.md`](../../CLAUDE.md) — TODO archival cadence; rebase + sweep cycle doctrine
- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — benchmark methodology and metrics
- [`docs/reference/environment.md`](../../docs/reference/environment.md) — `WEBLLM_SMOKE_RUNS_DIR`, dashboard ports
