import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { SystemProfile } from "../src/evaluation/system-profile.ts";
import type { SmokeRunRecord } from "./smoke-runs.ts";
import type { EvalReport } from "./types.ts";

export const LIVE_DB_SCHEMA_VERSION = 3;

export type PersistedRun = SmokeRunRecord & {
	runId: string;
	systemId?: string;
};
export type PersistedEvalReport = EvalReport & {
	evalId: string;
	systemId?: string;
};
export type PersistedSystemProfile = SystemProfile;

export function openLiveDb(path: string): Database {
	if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
	const db = new Database(path, { create: true });
	db.exec("PRAGMA journal_mode = WAL");
	db.exec("PRAGMA synchronous = NORMAL");
	db.exec("PRAGMA foreign_keys = ON");
	db.exec(`
		CREATE TABLE IF NOT EXISTS runs (
			run_id TEXT PRIMARY KEY,
			timestamp TEXT NOT NULL,
			profile TEXT,
			model TEXT NOT NULL,
			page TEXT NOT NULL,
			thinking TEXT NOT NULL,
			system_id TEXT,
			record_json TEXT NOT NULL,
			inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
		);
	`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_runs_profile ON runs(profile);`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model);`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_runs_system ON runs(system_id);`);
	db.exec(`
		CREATE TABLE IF NOT EXISTS evals (
			eval_id TEXT PRIMARY KEY,
			timestamp TEXT NOT NULL,
			model_id TEXT NOT NULL,
			total_tasks INTEGER NOT NULL,
			overall_score REAL NOT NULL,
			system_id TEXT,
			report_json TEXT NOT NULL,
			inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
		);
	`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_evals_timestamp ON evals(timestamp);`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_evals_model ON evals(model_id);`);
	db.exec(`CREATE INDEX IF NOT EXISTS idx_evals_system ON evals(system_id);`);
	db.exec(`
		CREATE TABLE IF NOT EXISTS system_profiles (
			system_id TEXT PRIMARY KEY,
			collected_at TEXT NOT NULL,
			profile_json TEXT NOT NULL,
			first_seen TEXT NOT NULL DEFAULT (datetime('now')),
			last_seen TEXT NOT NULL DEFAULT (datetime('now'))
		);
	`);

	// Best-effort migration: pre-v3 dbs lack the system_id column. ALTER
	// TABLE ADD COLUMN is idempotent only via try/catch in SQLite.
	for (const table of ["runs", "evals"]) {
		try {
			db.exec(`ALTER TABLE ${table} ADD COLUMN system_id TEXT;`);
		} catch {
			// column already exists — fresh DB or prior migration ran
		}
	}

	db.exec(`PRAGMA user_version = ${LIVE_DB_SCHEMA_VERSION};`);
	return db;
}

export function upsertRun(db: Database, record: PersistedRun): void {
	db.prepare(
		`INSERT INTO runs (run_id, timestamp, profile, model, page, thinking, system_id, record_json)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(run_id) DO UPDATE SET
		   timestamp = excluded.timestamp,
		   profile = excluded.profile,
		   model = excluded.model,
		   page = excluded.page,
		   thinking = excluded.thinking,
		   system_id = excluded.system_id,
		   record_json = excluded.record_json`,
	).run(
		record.runId,
		record.timestamp,
		record.profile ?? null,
		record.model,
		record.page,
		record.thinking,
		record.systemId ?? null,
		JSON.stringify(record),
	);
}

export interface LoadRunsOptions {
	limit?: number;
	order?: "asc" | "desc";
}

export function loadRuns(
	db: Database,
	options: LoadRunsOptions = {},
): PersistedRun[] {
	const { limit = 5000, order = "asc" } = options;
	const rows = db
		.prepare(
			`SELECT record_json FROM runs ORDER BY timestamp ${order === "asc" ? "ASC" : "DESC"} LIMIT ?`,
		)
		.all(limit) as Array<{ record_json: string }>;
	return rows.map((r) => JSON.parse(r.record_json) as PersistedRun);
}

export function clearRuns(db: Database): number {
	const res = db.prepare("DELETE FROM runs").run();
	return res.changes;
}

export function countRuns(db: Database): number {
	const row = db.prepare("SELECT COUNT(*) AS c FROM runs").get() as {
		c: number;
	};
	return row.c;
}

export function upsertEval(db: Database, report: PersistedEvalReport): void {
	db.prepare(
		`INSERT INTO evals (eval_id, timestamp, model_id, total_tasks, overall_score, system_id, report_json)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(eval_id) DO UPDATE SET
		   timestamp = excluded.timestamp,
		   model_id = excluded.model_id,
		   total_tasks = excluded.total_tasks,
		   overall_score = excluded.overall_score,
		   system_id = excluded.system_id,
		   report_json = excluded.report_json`,
	).run(
		report.evalId,
		report.timestamp,
		report.modelId,
		report.totalTasks,
		report.overall,
		report.systemId ?? null,
		JSON.stringify(report),
	);
}

// ── System profiles ────────────────────────────────────────────────

export function upsertSystemProfile(
	db: Database,
	profile: PersistedSystemProfile,
): void {
	db.prepare(
		`INSERT INTO system_profiles (system_id, collected_at, profile_json)
		 VALUES (?, ?, ?)
		 ON CONFLICT(system_id) DO UPDATE SET
		   collected_at = excluded.collected_at,
		   profile_json = excluded.profile_json,
		   last_seen = datetime('now')`,
	).run(profile.systemId, profile.collectedAt, JSON.stringify(profile));
}

export function loadSystemProfiles(
	db: Database,
	options: LoadRunsOptions = {},
): PersistedSystemProfile[] {
	const { limit = 5000, order = "asc" } = options;
	const rows = db
		.prepare(
			`SELECT profile_json FROM system_profiles ORDER BY first_seen ${order === "asc" ? "ASC" : "DESC"} LIMIT ?`,
		)
		.all(limit) as Array<{ profile_json: string }>;
	return rows.map((r) => JSON.parse(r.profile_json) as PersistedSystemProfile);
}

export function getSystemProfile(
	db: Database,
	systemId: string,
): PersistedSystemProfile | null {
	const row = db
		.prepare("SELECT profile_json FROM system_profiles WHERE system_id = ?")
		.get(systemId) as { profile_json: string } | undefined;
	return row ? (JSON.parse(row.profile_json) as PersistedSystemProfile) : null;
}

export function clearSystemProfiles(db: Database): number {
	return db.prepare("DELETE FROM system_profiles").run().changes;
}

export function countSystemProfiles(db: Database): number {
	const row = db
		.prepare("SELECT COUNT(*) AS c FROM system_profiles")
		.get() as { c: number };
	return row.c;
}

export function loadEvals(
	db: Database,
	options: LoadRunsOptions = {},
): PersistedEvalReport[] {
	const { limit = 5000, order = "asc" } = options;
	const rows = db
		.prepare(
			`SELECT report_json FROM evals ORDER BY timestamp ${order === "asc" ? "ASC" : "DESC"} LIMIT ?`,
		)
		.all(limit) as Array<{ report_json: string }>;
	return rows.map((r) => JSON.parse(r.report_json) as PersistedEvalReport);
}

export function clearEvals(db: Database): number {
	const res = db.prepare("DELETE FROM evals").run();
	return res.changes;
}

export function countEvals(db: Database): number {
	const row = db.prepare("SELECT COUNT(*) AS c FROM evals").get() as {
		c: number;
	};
	return row.c;
}

export interface EvalSeriesPoint {
	evalId: string;
	timestamp: string;
	profile: string | null;
	modelId: string;
	overall: number;
	thinking: string | null;
}

export function loadEvalSeries(db: Database): EvalSeriesPoint[] {
	const rows = db
		.prepare(
			`SELECT eval_id, timestamp, report_json FROM evals ORDER BY timestamp ASC`,
		)
		.all() as Array<{ eval_id: string; timestamp: string; report_json: string }>;
	return rows.map((r) => {
		const report = JSON.parse(r.report_json) as PersistedEvalReport;
		return {
			evalId: report.evalId,
			timestamp: report.timestamp,
			profile: report.profile ?? null,
			modelId: report.modelId,
			overall: report.overall,
			thinking: report.thinking ?? null,
		};
	});
}
