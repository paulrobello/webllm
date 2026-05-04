import type { Database } from "bun:sqlite";
import { readFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import {
	type BenchSessionCompletePayload,
	type BenchSessionStartedPayload,
	type EvalFailedPayload,
	type EvalStartedPayload,
	type EvalTaskCompletePayload,
	type LiveEvent,
	LiveEventStore,
	type PersistedEvalReport,
	type RunFailedPayload,
	type RunStartedPayload,
} from "./live-events.ts";
import {
	computeSystemId,
	type SystemProfile,
	type SystemProfileInput,
} from "../src/evaluation/system-profile.ts";
import { openLiveDb } from "./live-db.ts";
import type { SmokeRunRecord } from "./smoke-runs.ts";
import { loadEvalSeries } from "./live-db.ts";
import { BENCHMARK_MODELS } from "./models.ts";

const DEFAULT_PORT = 8033;
const DEFAULT_HOST = "0.0.0.0";
const DEFAULT_DASHBOARD_ROOT = "smoke-test";
const DEFAULT_DB_PATH = "eval/reports/smoke-runs.db";

interface ServerOptions {
	port: number;
	host: string;
	dashboardRoot: string;
	dbPath: string | null;
}

function parseServerArgs(): ServerOptions {
	const { values } = parseArgs({
		options: {
			port: { type: "string" },
			host: { type: "string" },
			"dashboard-root": { type: "string" },
			db: { type: "string" },
			"no-db": { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});
	if (values.help) {
		console.log(`Usage: bun run eval/live-server.ts [options]

Options:
      --port <num>            Bind port (default: ${DEFAULT_PORT})
      --host <addr>           Bind host (default: ${DEFAULT_HOST})
      --dashboard-root <dir>  Serve static dashboard assets from this dir
                              (default: ${DEFAULT_DASHBOARD_ROOT})
      --db <path>             SQLite file for persisted run records
                              (default: ${DEFAULT_DB_PATH})
      --no-db                 In-memory only (disable SQLite persistence)
  -h, --help                  Show this help
`);
		process.exit(0);
	}
	const port = values.port ? Number(values.port) : DEFAULT_PORT;
	if (!Number.isFinite(port) || port <= 0 || port > 65535) {
		console.error(`Invalid --port: ${values.port}`);
		process.exit(1);
	}
	return {
		port,
		host: values.host ?? DEFAULT_HOST,
		dashboardRoot: resolve(values["dashboard-root"] ?? DEFAULT_DASHBOARD_ROOT),
		dbPath: values["no-db"] ? null : (values.db ?? DEFAULT_DB_PATH),
	};
}

const CONTENT_TYPES: Record<string, string> = {
	".html": "text/html; charset=utf-8",
	".js": "text/javascript; charset=utf-8",
	".mjs": "text/javascript; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".json": "application/json; charset=utf-8",
	".svg": "image/svg+xml",
	".png": "image/png",
	".ico": "image/x-icon",
};

function contentTypeFor(path: string): string {
	return CONTENT_TYPES[extname(path)] ?? "application/octet-stream";
}

function corsHeaders(): Record<string, string> {
	return {
		"access-control-allow-origin": "*",
		"access-control-allow-methods": "GET, POST, OPTIONS",
		"access-control-allow-headers": "content-type, last-event-id",
	};
}

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json", ...corsHeaders() },
	});
}

function errorResponse(code: string, message: string, status: number): Response {
	return jsonResponse({ code, message }, status);
}

function encodeSseFrame(event: LiveEvent): string {
	return `id: ${event.seq}\nevent: ${event.kind}\ndata: ${JSON.stringify(event)}\n\n`;
}

/**
 * Hub of active SSE subscribers. Every ingest broadcast goes through here.
 */
class Broadcaster {
	private readonly subscribers = new Set<ReadableStreamDefaultController<Uint8Array>>();
	private readonly encoder = new TextEncoder();

	addSubscriber(
		controller: ReadableStreamDefaultController<Uint8Array>,
	): void {
		this.subscribers.add(controller);
	}

	removeSubscriber(
		controller: ReadableStreamDefaultController<Uint8Array>,
	): void {
		this.subscribers.delete(controller);
	}

	broadcast(event: LiveEvent): void {
		const payload = this.encoder.encode(encodeSseFrame(event));
		for (const controller of this.subscribers) {
			try {
				controller.enqueue(payload);
			} catch {
				// Subscriber's stream is closed; removal happens in the cancel path.
			}
		}
	}

	subscriberCount(): number {
		return this.subscribers.size;
	}
}

async function parseJsonBody<T>(req: Request): Promise<T> {
	try {
		return (await req.json()) as T;
	} catch {
		throw new SyntaxError("invalid_json");
	}
}

function validateRunStarted(body: unknown): RunStartedPayload {
	const b = body as Partial<RunStartedPayload> & Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("run_started requires an object body");
	for (const key of ["runId", "model", "page", "thinking", "prompt"] as const) {
		if (typeof b[key] !== "string" || !(b[key] as string).length) {
			throw new Error(`run_started.${key} must be a non-empty string`);
		}
	}
	if (b.page !== "smoke" && b.page !== "debug") {
		throw new Error('run_started.page must be "smoke" or "debug"');
	}
	if (b.thinking !== "off" && b.thinking !== "on") {
		throw new Error('run_started.thinking must be "off" or "on"');
	}
	return {
		runId: b.runId as string,
		profile: typeof b.profile === "string" ? b.profile : undefined,
		model: b.model as string,
		page: b.page,
		thinking: b.thinking,
		prompt: b.prompt as string,
	};
}

function validateRunComplete(
	body: unknown,
): SmokeRunRecord & { runId: string; systemId?: string } {
	const b = body as Partial<SmokeRunRecord & { runId: string }> &
		Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("run_complete requires an object body");
	if (typeof b.runId !== "string" || !b.runId.length) {
		throw new Error("run_complete.runId required");
	}
	if (typeof b.schemaVersion !== "number") {
		throw new Error("run_complete.schemaVersion required");
	}
	if (typeof b.model !== "string" || !b.model.length) {
		throw new Error("run_complete.model required");
	}
	// `mode` is optional for backward-compat with pre-Task-9 ingesters; when
	// absent the persistence layer defaults the column to 'main'. If the
	// caller does send a value it must be one of the two valid host
	// contexts so cross-mode A/B slices stay clean.
	if (b.mode !== undefined && b.mode !== "main" && b.mode !== "worker") {
		throw new Error('run_complete.mode must be "main" or "worker" when set');
	}
	return b as SmokeRunRecord & { runId: string; systemId?: string };
}

function validateRunFailed(body: unknown): RunFailedPayload {
	const b = body as Partial<RunFailedPayload> & Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("run_failed requires an object body");
	if (typeof b.runId !== "string" || !b.runId.length)
		throw new Error("run_failed.runId required");
	if (typeof b.model !== "string" || !b.model.length)
		throw new Error("run_failed.model required");
	if (typeof b.error !== "string")
		throw new Error("run_failed.error required");
	return {
		runId: b.runId,
		profile: typeof b.profile === "string" ? b.profile : undefined,
		model: b.model,
		error: b.error,
	};
}

function validateEvalStarted(body: unknown): EvalStartedPayload {
	const b = body as Partial<EvalStartedPayload> & Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("eval_started requires an object body");
	if (typeof b.evalId !== "string" || !b.evalId.length)
		throw new Error("eval_started.evalId required");
	if (typeof b.modelId !== "string" || !b.modelId.length)
		throw new Error("eval_started.modelId required");
	if (typeof b.totalTasks !== "number")
		throw new Error("eval_started.totalTasks required");
	if (!Array.isArray(b.dimensions))
		throw new Error("eval_started.dimensions must be an array");
	return {
		evalId: b.evalId,
		modelId: b.modelId,
		totalTasks: b.totalTasks,
		dimensions: b.dimensions.map(String),
		label: typeof b.label === "string" ? b.label : undefined,
		sessionId:
			typeof b.sessionId === "string" && b.sessionId.length > 0
				? b.sessionId
				: undefined,
	};
}

function validateBenchSessionStarted(
	body: unknown,
): BenchSessionStartedPayload {
	const b = body as Partial<BenchSessionStartedPayload> &
		Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("bench_session_started requires an object body");
	if (typeof b.sessionId !== "string" || !b.sessionId.length)
		throw new Error("bench_session_started.sessionId required");
	if (typeof b.startedAt !== "string" || !b.startedAt.length)
		throw new Error("bench_session_started.startedAt required");
	if (typeof b.totalModels !== "number" || !Number.isFinite(b.totalModels))
		throw new Error("bench_session_started.totalModels required");
	if (!Array.isArray(b.modelIds))
		throw new Error("bench_session_started.modelIds must be an array");
	const evalTemperature =
		typeof b.evalTemperature === "number" && Number.isFinite(b.evalTemperature)
			? b.evalTemperature
			: undefined;
	const totalTasks =
		typeof b.totalTasks === "number" && Number.isFinite(b.totalTasks)
			? b.totalTasks
			: undefined;
	return {
		sessionId: b.sessionId,
		startedAt: b.startedAt,
		totalModels: b.totalModels,
		totalTasks,
		modelIds: b.modelIds.map(String),
		profileNames: Array.isArray(b.profileNames)
			? b.profileNames.map(String)
			: undefined,
		evalTemperature,
		label: typeof b.label === "string" ? b.label : undefined,
	};
}

function validateBenchSessionComplete(
	body: unknown,
): BenchSessionCompletePayload {
	const b = body as Partial<BenchSessionCompletePayload> &
		Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("bench_session_complete requires an object body");
	if (typeof b.sessionId !== "string" || !b.sessionId.length)
		throw new Error("bench_session_complete.sessionId required");
	if (typeof b.completedAt !== "string" || !b.completedAt.length)
		throw new Error("bench_session_complete.completedAt required");
	for (const key of [
		"totalModels",
		"completedModels",
		"failedModels",
	] as const) {
		if (typeof b[key] !== "number" || !Number.isFinite(b[key] as number))
			throw new Error(`bench_session_complete.${key} required`);
	}
	const overall =
		typeof b.overall === "number" && Number.isFinite(b.overall)
			? b.overall
			: undefined;
	return {
		sessionId: b.sessionId,
		completedAt: b.completedAt,
		totalModels: b.totalModels as number,
		completedModels: b.completedModels as number,
		failedModels: b.failedModels as number,
		overall,
	};
}

function validateEvalTaskComplete(body: unknown): EvalTaskCompletePayload {
	const b = body as Partial<EvalTaskCompletePayload> &
		Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("eval_task_complete requires an object body");
	for (const key of ["evalId", "taskId", "dimension", "difficulty"] as const) {
		if (typeof b[key] !== "string" || !(b[key] as string).length) {
			throw new Error(`eval_task_complete.${key} required`);
		}
	}
	for (const key of [
		"score",
		"latencyMs",
		"tokensPerSecond",
		"toolCallsCount",
	] as const) {
		if (typeof b[key] !== "number")
			throw new Error(`eval_task_complete.${key} required`);
	}
	return {
		evalId: b.evalId as string,
		taskId: b.taskId as string,
		dimension: b.dimension as string,
		difficulty: b.difficulty as string,
		score: b.score as number,
		latencyMs: b.latencyMs as number,
		tokensPerSecond: b.tokensPerSecond as number,
		toolCallsCount: b.toolCallsCount as number,
		error: typeof b.error === "string" ? b.error : undefined,
	};
}

function validateEvalComplete(body: unknown): PersistedEvalReport {
	const b = body as Partial<PersistedEvalReport> & Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("eval_complete requires an object body");
	if (typeof b.evalId !== "string" || !b.evalId.length)
		throw new Error("eval_complete.evalId required");
	if (typeof b.modelId !== "string" || !b.modelId.length)
		throw new Error("eval_complete.modelId required");
	if (typeof b.timestamp !== "string")
		throw new Error("eval_complete.timestamp required");
	if (typeof b.totalTasks !== "number")
		throw new Error("eval_complete.totalTasks required");
	if (typeof b.overall !== "number")
		throw new Error("eval_complete.overall required");
	if (!Array.isArray(b.results))
		throw new Error("eval_complete.results must be an array");
	return b as PersistedEvalReport;
}

function validateEvalFailed(body: unknown): EvalFailedPayload {
	const b = body as Partial<EvalFailedPayload> & Record<string, unknown>;
	if (!b || typeof b !== "object")
		throw new Error("eval_failed requires an object body");
	if (typeof b.evalId !== "string" || !b.evalId.length)
		throw new Error("eval_failed.evalId required");
	if (typeof b.modelId !== "string" || !b.modelId.length)
		throw new Error("eval_failed.modelId required");
	if (typeof b.error !== "string")
		throw new Error("eval_failed.error required");
	return { evalId: b.evalId, modelId: b.modelId, error: b.error };
}

function tryServeStatic(
	root: string,
	pathname: string,
): Response | null {
	const rel = pathname === "/" ? "/dashboard.html" : pathname;
	const safeRel = rel.replace(/\.\./g, "").replace(/^\/+/, "");
	const filePath = join(root, safeRel);
	try {
		const data = readFileSync(filePath);
		return new Response(data, {
			status: 200,
			headers: {
				"content-type": contentTypeFor(filePath),
				// `no-store, max-age=0` — never cache the dashboard's HTML/JS/CSS
				// in the browser. Soft reload (Cmd-R) reliably picks up code
				// changes without needing Cmd-Shift-R or a manual cache-bust.
				// `no-cache` (the previous setting) is weaker: it permits cached
				// reuse with revalidation, and Chrome was observed serving stale
				// dashboard.js after live-server restarts because the response
				// lacks ETag/Last-Modified validators.
				"cache-control": "no-store, max-age=0",
				...corsHeaders(),
			},
		});
	} catch {
		return null;
	}
}

export function createLiveServer(options: ServerOptions) {
	const db: Database | undefined = options.dbPath
		? openLiveDb(options.dbPath)
		: undefined;
	const store = new LiveEventStore(500, db);
	const broadcaster = new Broadcaster();
	const encoder = new TextEncoder();
	// In-memory staging for task lists the browser bench mode fetches.
	// Intentionally memory-only — task lists are ephemeral (single run) and
	// can be large (the full task catalogue is ~100 KB).
	const taskLists = new Map<string, unknown[]>();

	const server = Bun.serve({
		hostname: options.host,
		port: options.port,
		idleTimeout: 0,
		async fetch(req) {
			const url = new URL(req.url);

			if (req.method === "OPTIONS") {
				return new Response(null, { status: 204, headers: corsHeaders() });
			}

			if (url.pathname === "/health") {
				return jsonResponse({
					ok: true,
					subscribers: broadcaster.subscriberCount(),
					seq: store.currentSeq(),
					runs: store.completedRuns().length,
					evals: store.completedEvals().length,
					systemProfiles: store.allSystemProfiles().length,
				});
			}

			if (url.pathname === "/runs" && req.method === "GET") {
				return jsonResponse({ runs: store.completedRuns() });
			}

			if (url.pathname === "/evals" && req.method === "GET") {
				return jsonResponse({ evals: store.completedEvals() });
			}

			if (url.pathname === "/evals/series" && req.method === "GET") {
				if (!db) return jsonResponse({ series: [] });
				return jsonResponse({ series: loadEvalSeries(db) });
			}

			// ── Model registry ─────────────────────────────────────────
			// Serves the full BenchmarkModel registry from eval/models.ts
			// so the dashboard can drive encoder / param-count filters
			// from `architecture` and `paramsB` instead of hand-maintained
			// id-prefix maps. Sorted by id for stable client-side hashing.
			if (url.pathname === "/models" && req.method === "GET") {
				const models = [...BENCHMARK_MODELS].sort((a, b) =>
					a.id.localeCompare(b.id),
				);
				return jsonResponse({ models });
			}

			// ── System profile registry ────────────────────────────────
			if (url.pathname === "/system-profiles" && req.method === "POST") {
				try {
					const body = await parseJsonBody<
						SystemProfileInput & { systemId?: string }
					>(req);
					if (!body || typeof body.userAgent !== "string") {
						return errorResponse(
							"bad_request",
							"system profile must include at least userAgent",
							400,
						);
					}
					// Recompute the id server-side to keep dedup honest.
					const systemId = await computeSystemId(body);
					const profile: SystemProfile = {
						...(body as SystemProfileInput),
						systemId,
						collectedAt: new Date().toISOString(),
					};
					store.registerSystemProfile(profile);
					return jsonResponse({ ok: true, systemId });
				} catch (err) {
					return errorResponse(
						"bad_request",
						err instanceof Error ? err.message : String(err),
						400,
					);
				}
			}
			if (url.pathname === "/system-profiles" && req.method === "GET") {
				return jsonResponse({ systemProfiles: store.allSystemProfiles() });
			}
			if (
				url.pathname.startsWith("/system-profiles/") &&
				req.method === "GET"
			) {
				const id = url.pathname.slice("/system-profiles/".length);
				const profile = store.getSystemProfile(id);
				if (!profile) {
					return errorResponse(
						"not_found",
						`unknown system profile: ${id}`,
						404,
					);
				}
				return jsonResponse(profile);
			}

			// ── Task list staging (browser bench mode) ─────────────────
			if (url.pathname === "/tasks" && req.method === "POST") {
				try {
					const body = (await parseJsonBody<{
						id?: string;
						tasks: unknown[];
					}>(req)) as { id?: string; tasks: unknown[] };
					if (!Array.isArray(body.tasks) || body.tasks.length === 0) {
						return errorResponse(
							"bad_request",
							"tasks must be a non-empty array",
							400,
						);
					}
					const id =
						body.id ??
						`${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
					taskLists.set(id, body.tasks);
					return jsonResponse({ ok: true, id, count: body.tasks.length });
				} catch (err) {
					return errorResponse(
						"bad_request",
						err instanceof Error ? err.message : String(err),
						400,
					);
				}
			}
			if (url.pathname.startsWith("/tasks/") && req.method === "GET") {
				const id = url.pathname.slice("/tasks/".length);
				const tasks = taskLists.get(id);
				if (!tasks) {
					return errorResponse("not_found", `unknown task list: ${id}`, 404);
				}
				return jsonResponse({ id, tasks });
			}

			if (url.pathname === "/ingest" && req.method === "POST") {
				const kind = url.searchParams.get("kind");
				try {
					const body = await parseJsonBody<unknown>(req);
					if (kind === "run_started") {
						const payload = validateRunStarted(body);
						const stamped = store.stampEvent({ kind: "run_started", payload });
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "run_complete") {
						const payload = validateRunComplete(body);
						const stamped = store.stampEvent({
							kind: "run_complete",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "run_failed") {
						const payload = validateRunFailed(body);
						const stamped = store.stampEvent({ kind: "run_failed", payload });
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "eval_started") {
						const payload = validateEvalStarted(body);
						const stamped = store.stampEvent({
							kind: "eval_started",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "eval_task_complete") {
						const payload = validateEvalTaskComplete(body);
						const stamped = store.stampEvent({
							kind: "eval_task_complete",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "eval_complete") {
						const payload = validateEvalComplete(body);
						const stamped = store.stampEvent({
							kind: "eval_complete",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "eval_failed") {
						const payload = validateEvalFailed(body);
						const stamped = store.stampEvent({
							kind: "eval_failed",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "bench_session_started") {
						const payload = validateBenchSessionStarted(body);
						const stamped = store.stampEvent({
							kind: "bench_session_started",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "bench_session_complete") {
						const payload = validateBenchSessionComplete(body);
						const stamped = store.stampEvent({
							kind: "bench_session_complete",
							payload,
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					if (kind === "reset") {
						const reason =
							typeof (body as { reason?: unknown })?.reason === "string"
								? ((body as { reason?: string }).reason as string)
								: undefined;
						const stamped = store.stampEvent({
							kind: "reset",
							payload: { reason },
						});
						broadcaster.broadcast(stamped);
						return jsonResponse({ ok: true, seq: stamped.seq });
					}
					return errorResponse(
						"unknown_kind",
						`?kind= must be run_started | run_complete | run_failed | eval_started | eval_task_complete | eval_complete | eval_failed | bench_session_started | bench_session_complete | reset (got ${kind ?? "none"})`,
						400,
					);
				} catch (err) {
					const message = err instanceof Error ? err.message : String(err);
					return errorResponse("bad_request", message, 400);
				}
			}

			if (url.pathname === "/stream" && req.method === "GET") {
				const lastEventIdHeader = req.headers.get("last-event-id");
				const lastSeq = lastEventIdHeader
					? Number.parseInt(lastEventIdHeader, 10) || 0
					: 0;
				let assignedController:
					| ReadableStreamDefaultController<Uint8Array>
					| null = null;
				let heartbeat: ReturnType<typeof setInterval> | null = null;

				const body = new ReadableStream<Uint8Array>({
					start: (controller) => {
						assignedController = controller;
						broadcaster.addSubscriber(controller);

						controller.enqueue(
							encoder.encode(
								`retry: 3000\n: connected seq=${store.currentSeq()}\n\n`,
							),
						);

						const snapshot = store.snapshot();
						controller.enqueue(encoder.encode(encodeSseFrame(snapshot)));

						if (lastSeq > 0) {
							for (const event of store.eventsSince(lastSeq)) {
								controller.enqueue(encoder.encode(encodeSseFrame(event)));
							}
						}

						heartbeat = setInterval(() => {
							try {
								controller.enqueue(
									encoder.encode(`: heartbeat ${Date.now()}\n\n`),
								);
							} catch {
								// stream torn down
							}
						}, 15_000);
					},
					cancel: () => {
						if (heartbeat) clearInterval(heartbeat);
						if (assignedController)
							broadcaster.removeSubscriber(assignedController);
					},
				});

				return new Response(body, {
					status: 200,
					headers: {
						"content-type": "text/event-stream; charset=utf-8",
						"cache-control": "no-cache, no-transform",
						connection: "keep-alive",
						"x-accel-buffering": "no",
						...corsHeaders(),
					},
				});
			}

			if (req.method === "GET") {
				const served = tryServeStatic(options.dashboardRoot, url.pathname);
				if (served) return served;
			}

			return new Response("Not found", { status: 404 });
		},
	});

	return {
		server,
		store,
		broadcaster,
		db,
	};
}

if (import.meta.main) {
	const options = parseServerArgs();
	const { server, store, db } = createLiveServer(options);
	console.log(
		`Live dashboard server listening on http://${options.host}:${options.port}`,
	);
	console.log(`  dashboard  → http://localhost:${options.port}/`);
	console.log(`  SSE stream → http://localhost:${options.port}/stream`);
	console.log(`  ingest     → POST http://localhost:${options.port}/ingest?kind=…`);
	console.log(`  health     → http://localhost:${options.port}/health`);
	if (options.dbPath) {
		console.log(
			`  db         → ${options.dbPath} (hydrated ${store.completedRuns().length} runs)`,
		);
	} else {
		console.log(`  db         → in-memory only`);
	}
	const shutdown = () => {
		server.stop();
		if (db) db.close();
		process.exit(0);
	};
	process.on("SIGINT", shutdown);
	process.on("SIGTERM", shutdown);
}
