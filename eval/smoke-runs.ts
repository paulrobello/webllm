import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { SmokeTestPage, SmokeTestResult } from "./browser-smoke.js";
import type { SmokeProfile } from "./smoke-profiles.js";

export const SMOKE_RUNS_DIR_ENV = "WEBLLM_SMOKE_RUNS_DIR";
export const DEFAULT_SMOKE_RUNS_DIR = "eval/reports/smoke-runs";
export const SMOKE_RUN_SCHEMA_VERSION = 1;

export interface SmokeRunOneShot {
	assistantText: string;
	finishReason?: string | undefined;
	genTokens: number;
	prefillMs: number;
	decodeMs: number;
	totalMs: number;
	tokensPerSecond: number;
}

export interface SmokeRunInteractive {
	assistantText: string;
	finishReason: string;
	genTokens?: number | undefined;
	totalMs?: number | undefined;
	tokensPerSecond?: number | undefined;
}

export interface SmokeRunParams {
	contextLength?: number;
	maxTokens?: number;
	temperature?: number;
	topK?: number;
	topP?: number;
	repetitionPenalty?: number;
	seed?: number;
}

export type SmokeRunMode = "main" | "worker";

export interface SmokeRunRecord {
	schemaVersion: typeof SMOKE_RUN_SCHEMA_VERSION;
	timestamp: string;
	profile?: string | undefined;
	model: string;
	page: SmokeTestPage;
	thinking: "off" | "on";
	/**
	 * Engine host context — `'main'` (legacy default) when the engine runs
	 * on the page main thread, `'worker'` when wrapped by `WebLLMProxy` and
	 * driven via a `DedicatedWorker` (Task 8). Optional for backward-compat
	 * with run records persisted before this field was introduced.
	 */
	mode?: SmokeRunMode;
	prompt: string;
	params: SmokeRunParams;
	oneShot?: SmokeRunOneShot;
	interactive?: SmokeRunInteractive;
}

/**
 * Pick the directory where run records should be written. Honors an env
 * var override so the matrix runner (which shells out per case) can pin
 * a single directory across its children without every invocation needing
 * a CLI flag.
 */
export function resolveSmokeRunsDir(override?: string): string {
	if (override && override.length > 0) return override;
	const fromEnv = process.env[SMOKE_RUNS_DIR_ENV];
	if (fromEnv && fromEnv.length > 0) return fromEnv;
	return DEFAULT_SMOKE_RUNS_DIR;
}

/**
 * Turn a timestamp + record identity into a filesystem-safe file name.
 * Millisecond-epoch prefix keeps lexicographic order = chronological order.
 */
export function smokeRunFileName(
	record: Pick<SmokeRunRecord, "timestamp" | "profile" | "model">,
): string {
	const epoch = Date.parse(record.timestamp);
	const prefix = Number.isFinite(epoch)
		? String(epoch).padStart(14, "0")
		: record.timestamp.replace(/[^a-zA-Z0-9]/g, "-");
	const id = record.profile ?? record.model;
	const safeId = id.replace(/[^a-zA-Z0-9._-]/g, "-");
	return `${prefix}-${safeId}.json`;
}

export function buildSmokeRunRecord(input: {
	profile: SmokeProfile | null;
	modelId: string;
	page: SmokeTestPage;
	thinking: "off" | "on";
	mode?: SmokeRunMode;
	prompt: string;
	contextLength: number;
	params: SmokeRunParams;
	oneShotResult?: SmokeTestResult;
	oneShotAssistantText?: string;
	oneShotFinishReason?: string;
	interactiveResult?: {
		assistantText: string;
		finishReason: string;
		metrics?: {
			genTokens?: number;
			tokensPerSecond?: number;
			totalMs?: number;
		};
	};
	timestamp?: Date;
}): SmokeRunRecord {
	const timestamp = (input.timestamp ?? new Date()).toISOString();
	const record: SmokeRunRecord = {
		schemaVersion: SMOKE_RUN_SCHEMA_VERSION,
		timestamp,
		model: input.modelId,
		page: input.page,
		thinking: input.thinking,
		prompt: input.prompt,
		params: {
			contextLength: input.contextLength,
			...input.params,
		},
	};
	if (input.profile) record.profile = input.profile.name;
	if (input.mode) record.mode = input.mode;
	if (input.oneShotResult) {
		record.oneShot = {
			assistantText: input.oneShotAssistantText ?? "",
			finishReason: input.oneShotFinishReason,
			genTokens: input.oneShotResult.tokensGenerated,
			prefillMs: input.oneShotResult.prefillMs,
			decodeMs: input.oneShotResult.decodeMs,
			totalMs: input.oneShotResult.totalMs,
			tokensPerSecond: input.oneShotResult.tokensPerSecond,
		};
	}
	if (input.interactiveResult) {
		record.interactive = {
			assistantText: input.interactiveResult.assistantText,
			finishReason: input.interactiveResult.finishReason,
			genTokens: input.interactiveResult.metrics?.genTokens,
			tokensPerSecond: input.interactiveResult.metrics?.tokensPerSecond,
			totalMs: input.interactiveResult.metrics?.totalMs,
		};
	}
	return record;
}

export function writeSmokeRunRecord(
	record: SmokeRunRecord,
	dir: string = resolveSmokeRunsDir(),
): string {
	const fileName = smokeRunFileName(record);
	const filePath = join(dir, fileName);
	mkdirSync(dirname(filePath), { recursive: true });
	writeFileSync(filePath, `${JSON.stringify(record, null, 2)}\n`, "utf-8");
	return filePath;
}
