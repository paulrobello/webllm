# ENH-006 — Persist Bench Sessions to SQLite on the Live Dashboard

> **Status**: proposed · **Effort/risk**: Low · **Depends on**: land after Phase 1 security edits and ARC-008 (route table) to avoid triple-editing `eval/live-server.ts`

## Goal

A dashboard-server restart mid-bench-session no longer loses the session rollup (model X/Y ·
task A/B). Sessions survive in a `bench_sessions` SQLite table and rehydrate on startup.

## Current state

- CLAUDE.md documents the gap verbatim: "Sessions are in-memory only on the live-server (no SQLite
  table); a backend restart mid-session loses the rollup until the next session starts."
- The envelope: `eval/bench.ts` mints a `sessionId`, emits `bench_session_started` /
  `bench_session_complete` live events; each per-model `eval_started` carries the parent
  `sessionId` (threading: `WEBLLM_BENCH_SESSION_ID` → child harness → URL `?session=` → smoke page
  → payload).
- Storage layer: `eval/live-db.ts` owns the SQLite schema and the idempotent-import discipline
  (skip by `runId`/`evalId`). Session events flow through `eval/live-events.ts`
  (`LiveEventStore`) and/or the server's event handling — **discovery step**: grep
  `bench_session_started` in `eval/` to find exactly where the in-memory session object lives and
  what fields it carries; those fields are the table columns.

## Implementation steps

1. **Schema** in `eval/live-db.ts`, following the existing table-creation pattern (CREATE TABLE IF
   NOT EXISTS in the same place the other tables are declared):
   `bench_sessions(session_id TEXT PRIMARY KEY, started_at INTEGER, total_models INTEGER,
   completed_models INTEGER, eval_temperature REAL, status TEXT, completed_at INTEGER,
   payload_json TEXT)` — adjust columns to match the actual in-memory session fields from the
   discovery grep (`payload_json` catches the remainder so the schema doesn't chase every field).
2. **Write path**: where `bench_session_started` is handled, INSERT OR REPLACE the row; where
   per-model progress updates the in-memory object, UPDATE `completed_models`; on
   `bench_session_complete`, set `status`/`completed_at`. Keep the in-memory map as the hot copy —
   SQLite is the durability layer, not the read path.
3. **Rehydrate**: at server startup (where the DB opens), SELECT sessions with
   `status = 'active'` and reload them into the in-memory map. Add a staleness guard: an active
   session older than 24 h is marked `abandoned` instead of rehydrated (a crashed run should not
   pin the dashboard forever).
4. **SSE catch-up**: if the dashboard front-end learns about sessions only via live SSE events
   (check how `smoke-test/dashboard.js` builds the rollup — grep `session` in it), add the current
   active sessions to whatever initial-state payload the server already sends on SSE connect (or
   the existing state endpoint) so a freshly-loaded dashboard page sees the restored session.
5. **Idempotency**: `make import-reports` does not touch sessions — confirm and leave it that way
   (sessions are ephemeral orchestration state, not report data).

## Files to touch

- `eval/live-db.ts` (schema + queries), `eval/live-server.ts` and/or `eval/live-events.ts`
  (write/rehydrate — wherever discovery finds the handlers), `smoke-test/dashboard.js` (only if
  step 4 requires the initial-state addition)

## Verification

1. `make checkall`.
2. End-to-end: `make dashboard-serve`; start a short 2-model bench with
   `WEBLLM_LIVE_BENCH_URL=http://localhost:8033`; after model 1 completes, kill and restart the
   dashboard server; the dashboard page shows the session rollup continuing (model 2/2), and
   `bench_session_complete` closes it normally. Then `sqlite3 <db path> "select * from
   bench_sessions"` shows the completed row. (Find the DB path in `eval/live-db.ts`.)
3. Staleness guard: hand-insert an `active` row with `started_at` 25 h old, restart, confirm it
   flips to `abandoned` and does not appear as live.

## Rollback

Drop the table usage by reverting; the table itself is harmless if left (additive schema). No
report-import or event-wire format changes.
