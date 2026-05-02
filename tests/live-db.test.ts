import type { Database } from "bun:sqlite";
import { afterEach, beforeEach, expect, test } from "bun:test";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	clearEvals,
	clearRuns,
	countEvals,
	countRuns,
	LIVE_DB_SCHEMA_VERSION,
	loadEvals,
	loadRuns,
	openLiveDb,
	type PersistedEvalReport,
	type PersistedRun,
	upsertEval,
	upsertRun,
} from "../eval/live-db.js";
import { LiveEventStore } from "../eval/live-events.js";

let tmpDir: string;
let dbPath: string;
let db: Database;

function makeRun(overrides: Partial<PersistedRun> = {}): PersistedRun {
	return {
		runId: "r1",
		schemaVersion: 1,
		timestamp: "2026-04-23T22:34:19.000Z",
		model: "qwen3-0.6b-q4f16",
		page: "smoke",
		thinking: "off",
		prompt: "hi",
		params: { contextLength: 4096 },
		...overrides,
	};
}

beforeEach(() => {
	tmpDir = join(tmpdir(), `webllm-live-db-${Date.now()}-${Math.random()}`);
	dbPath = join(tmpDir, "runs.db");
	db = openLiveDb(dbPath);
});

afterEach(() => {
	db.close();
	rmSync(tmpDir, { recursive: true, force: true });
});

test("openLiveDb creates the runs + evals tables at the current schema version", () => {
	const row = db.prepare("PRAGMA user_version").get() as {
		user_version: number;
	};
	expect(row.user_version).toBe(LIVE_DB_SCHEMA_VERSION);
	expect(countRuns(db)).toBe(0);
	expect(countEvals(db)).toBe(0);
});

test("upsertRun inserts and updates by run_id", () => {
	upsertRun(db, makeRun({ runId: "r-a", model: "llama-3.2-1b-q4f16" }));
	upsertRun(db, makeRun({ runId: "r-b", model: "qwen3-0.6b-q4f16" }));
	expect(countRuns(db)).toBe(2);

	// upsert same run_id overwrites in place, not a duplicate row
	upsertRun(
		db,
		makeRun({ runId: "r-a", model: "llama-3.2-1b-q4f16", prompt: "updated" }),
	);
	expect(countRuns(db)).toBe(2);
	const reloaded = loadRuns(db);
	const rA = reloaded.find((r) => r.runId === "r-a");
	expect(rA?.prompt).toBe("updated");
});

test("loadRuns orders by timestamp and respects limit/order args", () => {
	upsertRun(
		db,
		makeRun({ runId: "r1", timestamp: "2026-04-23T22:00:00.000Z" }),
	);
	upsertRun(
		db,
		makeRun({ runId: "r2", timestamp: "2026-04-23T22:30:00.000Z" }),
	);
	upsertRun(
		db,
		makeRun({ runId: "r3", timestamp: "2026-04-23T22:15:00.000Z" }),
	);

	const asc = loadRuns(db);
	expect(asc.map((r) => r.runId)).toEqual(["r1", "r3", "r2"]);

	const desc = loadRuns(db, { order: "desc" });
	expect(desc.map((r) => r.runId)).toEqual(["r2", "r3", "r1"]);

	const limited = loadRuns(db, { limit: 2, order: "desc" });
	expect(limited).toHaveLength(2);
});

test("upsertRun persists mode column ('main' default, 'worker' when set)", () => {
	// Pre-Task-9 record (no mode field) should land as 'main' via the
	// column DEFAULT, and round-trip without a `mode` key on the JSON
	// (since we only persist what was passed in).
	upsertRun(db, makeRun({ runId: "r-legacy" }));
	const legacyRow = db
		.prepare("SELECT mode FROM runs WHERE run_id = ?")
		.get("r-legacy") as { mode: string };
	expect(legacyRow.mode).toBe("main");

	// Explicit 'worker' record should round-trip through the column AND
	// the embedded record_json (since the JSON snapshot is the canonical
	// payload — the column exists for SQL slicing on cross-mode A/Bs).
	upsertRun(db, makeRun({ runId: "r-worker", mode: "worker" }));
	const workerRow = db
		.prepare("SELECT mode FROM runs WHERE run_id = ?")
		.get("r-worker") as { mode: string };
	expect(workerRow.mode).toBe("worker");
	const reloaded = loadRuns(db).find((r) => r.runId === "r-worker");
	expect(reloaded?.mode).toBe("worker");

	// Upsert overwrite swaps the column too (worker → main re-classification).
	upsertRun(db, makeRun({ runId: "r-worker", mode: "main" }));
	const reclassified = db
		.prepare("SELECT mode FROM runs WHERE run_id = ?")
		.get("r-worker") as { mode: string };
	expect(reclassified.mode).toBe("main");
});

test("clearRuns removes everything and returns the count", () => {
	upsertRun(db, makeRun({ runId: "r-a" }));
	upsertRun(db, makeRun({ runId: "r-b" }));
	const removed = clearRuns(db);
	expect(removed).toBe(2);
	expect(countRuns(db)).toBe(0);
});

test("LiveEventStore hydrates completed runs from the DB on construction", () => {
	upsertRun(
		db,
		makeRun({
			runId: "r-old",
			profile: "qwen3-0.6b-off-warm",
			timestamp: "2026-04-23T21:00:00.000Z",
		}),
	);
	const store = new LiveEventStore(100, db);
	expect(store.completedRuns().map((r) => r.runId)).toEqual(["r-old"]);
	const snap = store.snapshot();
	expect(snap.payload.runs.map((r) => r.runId)).toEqual(["r-old"]);
});

test("stampEvent run_complete mirrors to DB; reset clears DB", () => {
	const store = new LiveEventStore(100, db);
	store.stampEvent({
		kind: "run_complete",
		payload: { ...makeRun({ runId: "r-x" }), runId: "r-x" },
	});
	expect(countRuns(db)).toBe(1);

	store.stampEvent({ kind: "reset", payload: {} });
	expect(countRuns(db)).toBe(0);
	expect(store.completedRuns()).toHaveLength(0);
});

function makeEval(
	overrides: Partial<PersistedEvalReport> = {},
): PersistedEvalReport {
	return {
		evalId: "e1",
		timestamp: "2026-04-23T22:34:19.000Z",
		modelId: "qwen3-0.6b-q4f16",
		totalTasks: 2,
		overall: 0.75,
		results: [],
		dimensions: {
			"tool-calling": { total: 2, passed: 1, score: 0.75, avgLatencyMs: 120 },
		} as never,
		...overrides,
	};
}

test("evals table: upsert / load / clear round-trip", () => {
	upsertEval(db, makeEval({ evalId: "e-a", modelId: "llama-3.2-1b-q4f16" }));
	upsertEval(
		db,
		makeEval({ evalId: "e-b", modelId: "qwen3-0.6b-q4f16", overall: 0.9 }),
	);
	expect(countEvals(db)).toBe(2);

	upsertEval(
		db,
		makeEval({ evalId: "e-a", modelId: "llama-3.2-1b-q4f16", overall: 0.4 }),
	);
	expect(countEvals(db)).toBe(2);

	const all = loadEvals(db);
	const rA = all.find((r) => r.evalId === "e-a");
	expect(rA?.overall).toBe(0.4);

	const removed = clearEvals(db);
	expect(removed).toBe(2);
	expect(countEvals(db)).toBe(0);
});

test("LiveEventStore hydrates evals from DB and mirrors eval_complete back", () => {
	upsertEval(db, makeEval({ evalId: "e-old" }));
	const store = new LiveEventStore(100, db);
	expect(store.completedEvals().map((e) => e.evalId)).toEqual(["e-old"]);

	store.stampEvent({
		kind: "eval_complete",
		payload: makeEval({ evalId: "e-new", modelId: "qwen3-0.6b-q4f16" }),
	});
	expect(countEvals(db)).toBe(2);
	expect(
		store
			.completedEvals()
			.map((e) => e.evalId)
			.sort(),
	).toEqual(["e-new", "e-old"]);
});

test("hydration survives a full close-and-reopen cycle", () => {
	const s1 = new LiveEventStore(100, db);
	s1.stampEvent({
		kind: "run_complete",
		payload: { ...makeRun({ runId: "r-keep" }), runId: "r-keep" },
	});
	db.close();

	const reopened = openLiveDb(dbPath);
	const s2 = new LiveEventStore(100, reopened);
	expect(s2.completedRuns().map((r) => r.runId)).toEqual(["r-keep"]);
	db = reopened; // so afterEach closes the right one
});
