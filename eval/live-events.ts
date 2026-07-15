import type { Database } from "bun:sqlite";
import type { SystemProfile } from "../src/evaluation/system-profile.ts";
import {
	clearEvals,
	clearRuns,
	clearSystemProfiles,
	loadEvals,
	loadRuns,
	loadSystemProfiles,
	upsertEval,
	upsertRun,
	upsertSystemProfile,
} from "./live-db.ts";
import type { SmokeRunRecord } from "./smoke-runs.ts";
import type { EvalReport } from "./types.ts";

export const LIVE_EVENT_SCHEMA_VERSION = 3;

export type LiveEventKind =
	| "snapshot"
	| "run_started"
	| "run_complete"
	| "run_failed"
	| "eval_started"
	| "eval_task_complete"
	| "eval_complete"
	| "eval_failed"
	| "bench_session_started"
	| "bench_session_complete"
	| "reset";

export interface LiveEventBase {
	seq: number;
	kind: LiveEventKind;
	timestamp: string;
}

export interface RunStartedPayload {
	runId: string;
	profile?: string | undefined;
	model: string;
	page: "smoke" | "debug";
	thinking: "off" | "on";
	prompt: string;
}

export interface RunFailedPayload {
	runId: string;
	profile?: string | undefined;
	model: string;
	error: string;
}

export interface LiveRunStartedEvent extends LiveEventBase {
	kind: "run_started";
	payload: RunStartedPayload;
}

export interface LiveRunCompleteEvent extends LiveEventBase {
	kind: "run_complete";
	payload: SmokeRunRecord & { runId: string };
}

export interface LiveRunFailedEvent extends LiveEventBase {
	kind: "run_failed";
	payload: RunFailedPayload;
}

export interface LiveResetEvent extends LiveEventBase {
	kind: "reset";
	payload: { reason?: string | undefined };
}

export interface EvalStartedPayload {
	evalId: string;
	modelId: string;
	totalTasks: number;
	dimensions: string[];
	label?: string | undefined;
	/**
	 * Set when this eval is part of a multi-model bench session driven by
	 * `eval/bench.ts`. Lets the dashboard aggregate per-model `eval_*`
	 * events into an overall progress view across the whole session.
	 */
	sessionId?: string | undefined;
}

/**
 * Bracket events for a multi-model bench session. `bench.ts` mints a
 * sessionId once at the top of the loop, posts `bench_session_started`
 * before any model runs, threads the sessionId into each per-model
 * `eval_started`, and posts `bench_session_complete` when the loop exits
 * (success or failure). The dashboard uses these to compute overall
 * progress (model X of Y, task A of B) without having to guess where a
 * session ends.
 */
export interface BenchSessionStartedPayload {
	sessionId: string;
	startedAt: string;
	totalModels: number;
	/**
	 * Best-effort total task count across all models. The bench harness
	 * doesn't always know this up front (it depends on per-model dimension
	 * filtering), so leave undefined when uncertain — the dashboard then
	 * falls back to summing `totalTasks` from each `eval_started`.
	 */
	totalTasks?: number | undefined;
	modelIds: string[];
	profileNames?: string[] | undefined;
	/**
	 * Sampling temperature pinned for the accuracy pass. `0` denotes
	 * greedy. Surfaced so the dashboard can tag the session and prevent
	 * accidental cross-temperature comparisons in score-over-time charts.
	 */
	evalTemperature?: number | undefined;
	label?: string | undefined;
}

export interface BenchSessionCompletePayload {
	sessionId: string;
	completedAt: string;
	totalModels: number;
	completedModels: number;
	failedModels: number;
	/** Optional aggregate summary; dashboard treats absence as "not reported". */
	overall?: number | undefined;
}

export interface EvalTaskCompletePayload {
	evalId: string;
	taskId: string;
	dimension: string;
	difficulty: string;
	score: number;
	latencyMs: number;
	tokensPerSecond: number;
	toolCallsCount: number;
	error?: string | undefined;
}

export interface EvalFailedPayload {
	evalId: string;
	modelId: string;
	error: string;
}

export interface PersistedEvalReport extends EvalReport {
	evalId: string;
}

export interface LiveEvalStartedEvent extends LiveEventBase {
	kind: "eval_started";
	payload: EvalStartedPayload;
}

export interface LiveEvalTaskCompleteEvent extends LiveEventBase {
	kind: "eval_task_complete";
	payload: EvalTaskCompletePayload;
}

export interface LiveEvalCompleteEvent extends LiveEventBase {
	kind: "eval_complete";
	payload: PersistedEvalReport;
}

export interface LiveEvalFailedEvent extends LiveEventBase {
	kind: "eval_failed";
	payload: EvalFailedPayload;
}

export interface LiveBenchSessionStartedEvent extends LiveEventBase {
	kind: "bench_session_started";
	payload: BenchSessionStartedPayload;
}

export interface LiveBenchSessionCompleteEvent extends LiveEventBase {
	kind: "bench_session_complete";
	payload: BenchSessionCompletePayload;
}

export interface LiveSnapshotEvent extends LiveEventBase {
	kind: "snapshot";
	payload: {
		schemaVersion: typeof LIVE_EVENT_SCHEMA_VERSION;
		runs: Array<SmokeRunRecord & { runId: string }>;
		evals: PersistedEvalReport[];
		systemProfiles: SystemProfile[];
	};
}

export type LiveEvent =
	| LiveRunStartedEvent
	| LiveRunCompleteEvent
	| LiveRunFailedEvent
	| LiveEvalStartedEvent
	| LiveEvalTaskCompleteEvent
	| LiveEvalCompleteEvent
	| LiveEvalFailedEvent
	| LiveBenchSessionStartedEvent
	| LiveBenchSessionCompleteEvent
	| LiveResetEvent
	| LiveSnapshotEvent;

/**
 * In-memory store of run records keyed by runId. Keeps a ring buffer of
 * the most recent events for replay on reconnect (via Last-Event-ID) and
 * a latest-per-profile map so the snapshot on connect is compact rather
 * than a full event log.
 *
 * If constructed with a SQLite database, completed runs are hydrated from
 * the DB on startup (so a restart doesn't lose history) and mirrored back
 * on every `run_complete` / `reset`. `seq` numbering is in-memory only —
 * a browser reconnecting with a `Last-Event-ID` from a previous process
 * lifetime falls back to the snapshot rather than delta replay.
 */
export class LiveEventStore {
	private nextSeq = 1;
	private readonly eventBuffer: LiveEvent[] = [];
	private readonly completeByRunId = new Map<
		string,
		SmokeRunRecord & { runId: string }
	>();
	private readonly completeByProfile = new Map<
		string,
		SmokeRunRecord & { runId: string }
	>();
	private readonly completeEvalsByEvalId = new Map<
		string,
		PersistedEvalReport
	>();
	private readonly systemProfilesById = new Map<string, SystemProfile>();

	constructor(
		private readonly maxEventBuffer = 500,
		private readonly db?: Database,
	) {
		if (this.db) {
			for (const run of loadRuns(this.db, { order: "asc" })) {
				this.completeByRunId.set(run.runId, run);
				const key = run.profile ?? run.model;
				this.completeByProfile.set(key, run);
			}
			for (const report of loadEvals(this.db, { order: "asc" })) {
				this.completeEvalsByEvalId.set(report.evalId, report);
			}
			for (const sys of loadSystemProfiles(this.db, { order: "asc" })) {
				this.systemProfilesById.set(sys.systemId, sys);
			}
		}
	}

	registerSystemProfile(profile: SystemProfile): void {
		this.systemProfilesById.set(profile.systemId, profile);
		if (this.db) upsertSystemProfile(this.db, profile);
	}

	getSystemProfile(systemId: string): SystemProfile | undefined {
		return this.systemProfilesById.get(systemId);
	}

	allSystemProfiles(): SystemProfile[] {
		return Array.from(this.systemProfilesById.values());
	}

	stampEvent<E extends Omit<LiveEvent, "seq" | "timestamp">>(event: E): E & {
		seq: number;
		timestamp: string;
	} {
		const stamped = {
			...event,
			seq: this.nextSeq++,
			timestamp: new Date().toISOString(),
		};
		this.eventBuffer.push(stamped as unknown as LiveEvent);
		if (this.eventBuffer.length > this.maxEventBuffer) {
			this.eventBuffer.shift();
		}
		if (stamped.kind === "run_complete") {
			const payload = (stamped as LiveRunCompleteEvent).payload;
			this.completeByRunId.set(payload.runId, payload);
			const key = payload.profile ?? payload.model;
			this.completeByProfile.set(key, payload);
			if (this.db) upsertRun(this.db, payload);
		} else if (stamped.kind === "eval_complete") {
			const payload = (stamped as LiveEvalCompleteEvent).payload;
			this.completeEvalsByEvalId.set(payload.evalId, payload);
			if (this.db) upsertEval(this.db, payload);
		} else if (stamped.kind === "reset") {
			this.completeByRunId.clear();
			this.completeByProfile.clear();
			this.completeEvalsByEvalId.clear();
			this.systemProfilesById.clear();
			if (this.db) {
				clearRuns(this.db);
				clearEvals(this.db);
				clearSystemProfiles(this.db);
			}
		}
		return stamped as E & { seq: number; timestamp: string };
	}

	snapshot(): LiveSnapshotEvent {
		return this.stampEvent({
			kind: "snapshot",
			payload: {
				schemaVersion: LIVE_EVENT_SCHEMA_VERSION,
				runs: Array.from(this.completeByRunId.values()),
				evals: Array.from(this.completeEvalsByEvalId.values()),
				systemProfiles: Array.from(this.systemProfilesById.values()),
			},
		}) as LiveSnapshotEvent;
	}

	eventsSince(lastSeq: number): LiveEvent[] {
		return this.eventBuffer.filter((e) => e.seq > lastSeq);
	}

	currentSeq(): number {
		return this.nextSeq - 1;
	}

	completedRuns(): Array<SmokeRunRecord & { runId: string }> {
		return Array.from(this.completeByRunId.values());
	}

	completedEvals(): PersistedEvalReport[] {
		return Array.from(this.completeEvalsByEvalId.values());
	}
}
