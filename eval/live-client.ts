import type {
	BenchSessionCompletePayload,
	BenchSessionStartedPayload,
	EvalFailedPayload,
	EvalStartedPayload,
	EvalTaskCompletePayload,
	PersistedEvalReport,
	RunFailedPayload,
	RunStartedPayload,
} from "./live-events.ts";
import type { SmokeRunRecord } from "./smoke-runs.ts";

export const LIVE_BENCH_URL_ENV = "WEBLLM_LIVE_BENCH_URL";

export function resolveLiveBenchUrl(override?: string): string | null {
	if (override && override.length > 0) return override;
	const fromEnv = process.env[LIVE_BENCH_URL_ENV];
	if (fromEnv && fromEnv.length > 0) return fromEnv;
	return null;
}

type IngestKind =
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

async function postIngest(
	baseUrl: string,
	kind: IngestKind,
	body: unknown,
	timeoutMs = 5000,
): Promise<boolean> {
	const controller = new AbortController();
	const to = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const url = new URL(`/ingest?kind=${kind}`, baseUrl).toString();
		const res = await fetch(url, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
			signal: controller.signal,
		});
		if (!res.ok) {
			console.warn(
				`live-bench ingest (${kind}) failed: HTTP ${res.status} ${res.statusText}`,
			);
			return false;
		}
		return true;
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		console.warn(`live-bench ingest (${kind}) error: ${msg}`);
		return false;
	} finally {
		clearTimeout(to);
	}
}

export async function publishRunStarted(
	baseUrl: string,
	payload: RunStartedPayload,
): Promise<boolean> {
	return postIngest(baseUrl, "run_started", payload);
}

export async function publishRunComplete(
	baseUrl: string,
	record: SmokeRunRecord & { runId: string },
): Promise<boolean> {
	return postIngest(baseUrl, "run_complete", record);
}

export async function publishRunFailed(
	baseUrl: string,
	payload: RunFailedPayload,
): Promise<boolean> {
	return postIngest(baseUrl, "run_failed", payload);
}

export async function publishEvalStarted(
	baseUrl: string,
	payload: EvalStartedPayload,
): Promise<boolean> {
	return postIngest(baseUrl, "eval_started", payload);
}

export async function publishEvalTaskComplete(
	baseUrl: string,
	payload: EvalTaskCompletePayload,
): Promise<boolean> {
	return postIngest(baseUrl, "eval_task_complete", payload);
}

export async function publishEvalComplete(
	baseUrl: string,
	report: PersistedEvalReport,
): Promise<boolean> {
	return postIngest(baseUrl, "eval_complete", report);
}

export async function publishEvalFailed(
	baseUrl: string,
	payload: EvalFailedPayload,
): Promise<boolean> {
	return postIngest(baseUrl, "eval_failed", payload);
}

export async function publishBenchSessionStarted(
	baseUrl: string,
	payload: BenchSessionStartedPayload,
): Promise<boolean> {
	return postIngest(baseUrl, "bench_session_started", payload);
}

export async function publishBenchSessionComplete(
	baseUrl: string,
	payload: BenchSessionCompletePayload,
): Promise<boolean> {
	return postIngest(baseUrl, "bench_session_complete", payload);
}
