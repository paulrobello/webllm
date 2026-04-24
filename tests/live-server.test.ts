import { afterAll, beforeAll, expect, test } from "bun:test";
import { createLiveServer } from "../eval/live-server.js";

const PORT = 18033;
let handle: ReturnType<typeof createLiveServer>;

beforeAll(() => {
	handle = createLiveServer({
		port: PORT,
		host: "127.0.0.1",
		dashboardRoot: "smoke-test",
		dbPath: ":memory:",
	});
});

afterAll(() => {
	handle.server.stop(true);
	handle.db?.close();
});

test("GET /health returns liveness info", async () => {
	const res = await fetch(`http://127.0.0.1:${PORT}/health`);
	expect(res.ok).toBe(true);
	const body = await res.json();
	expect(body.ok).toBe(true);
	expect(typeof body.seq).toBe("number");
	expect(typeof body.runs).toBe("number");
});

test("POST /ingest?kind=run_started validates payload", async () => {
	const bad = await fetch(`http://127.0.0.1:${PORT}/ingest?kind=run_started`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ runId: "r1" }),
	});
	expect(bad.status).toBe(400);
	const err = await bad.json();
	expect(err.code).toBe("bad_request");

	const good = await fetch(`http://127.0.0.1:${PORT}/ingest?kind=run_started`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			runId: "r-smoke-1",
			profile: "qwen3-0.6b-thinking-warm",
			model: "qwen3-0.6b-q4f16",
			page: "smoke",
			thinking: "on",
			prompt: "hi",
		}),
	});
	expect(good.ok).toBe(true);
	const body = await good.json();
	expect(body.ok).toBe(true);
	expect(typeof body.seq).toBe("number");
});

test("POST /ingest?kind=run_complete persists and surfaces via /runs", async () => {
	const record = {
		runId: "r-complete-1",
		schemaVersion: 1,
		timestamp: new Date().toISOString(),
		model: "qwen3-0.6b-q4f16",
		profile: "qwen3-0.6b-off-warm",
		page: "smoke",
		thinking: "off",
		prompt: "hi",
		params: { contextLength: 4096 },
		oneShot: {
			assistantText: "ha",
			finishReason: "eos",
			genTokens: 10,
			prefillMs: 50,
			decodeMs: 200,
			totalMs: 250,
			tokensPerSecond: 40,
		},
	};
	const res = await fetch(`http://127.0.0.1:${PORT}/ingest?kind=run_complete`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(record),
	});
	expect(res.ok).toBe(true);

	const listing = await fetch(`http://127.0.0.1:${PORT}/runs`);
	const body = await listing.json();
	expect(
		body.runs.some((r: { runId: string }) => r.runId === "r-complete-1"),
	).toBe(true);
});

test("POST /ingest rejects unknown kinds", async () => {
	const res = await fetch(`http://127.0.0.1:${PORT}/ingest?kind=mystery`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: "{}",
	});
	expect(res.status).toBe(400);
	const err = await res.json();
	expect(err.code).toBe("unknown_kind");
});

test("POST /ingest?kind=eval_* surfaces completed evals via /evals", async () => {
	const start = await fetch(
		`http://127.0.0.1:${PORT}/ingest?kind=eval_started`,
		{
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				evalId: "e-1",
				modelId: "qwen3-0.6b-q4f16",
				totalTasks: 1,
				dimensions: ["tool-calling"],
			}),
		},
	);
	expect(start.ok).toBe(true);

	const task = await fetch(
		`http://127.0.0.1:${PORT}/ingest?kind=eval_task_complete`,
		{
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				evalId: "e-1",
				taskId: "t1",
				dimension: "tool-calling",
				difficulty: "easy",
				score: 1,
				latencyMs: 100,
				tokensPerSecond: 25,
				toolCallsCount: 1,
			}),
		},
	);
	expect(task.ok).toBe(true);

	const done = await fetch(
		`http://127.0.0.1:${PORT}/ingest?kind=eval_complete`,
		{
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				evalId: "e-1",
				timestamp: new Date().toISOString(),
				modelId: "qwen3-0.6b-q4f16",
				totalTasks: 1,
				overall: 1,
				results: [
					{
						taskId: "t1",
						dimension: "tool-calling",
						difficulty: "easy",
						score: 1,
						modelOutput: "ok",
						toolCalls: [],
						latencyMs: 100,
						tokensPerSecond: 25,
					},
				],
				dimensions: {
					"tool-calling": { total: 1, passed: 1, score: 1, avgLatencyMs: 100 },
				},
			}),
		},
	);
	expect(done.ok).toBe(true);

	const listing = await (await fetch(`http://127.0.0.1:${PORT}/evals`)).json();
	expect(
		listing.evals.some((e: { evalId: string }) => e.evalId === "e-1"),
	).toBe(true);
});

test("POST /tasks stages a task list and GET /tasks/:id reads it back", async () => {
	const tasks = [
		{ id: "t-a", dimension: "reasoning", difficulty: "easy" },
		{ id: "t-b", dimension: "tool-calling", difficulty: "medium" },
	];
	const post = await fetch(`http://127.0.0.1:${PORT}/tasks`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ tasks }),
	});
	expect(post.ok).toBe(true);
	const postBody = (await post.json()) as {
		ok: boolean;
		id: string;
		count: number;
	};
	expect(postBody.ok).toBe(true);
	expect(postBody.count).toBe(2);
	expect(typeof postBody.id).toBe("string");

	const get = await fetch(`http://127.0.0.1:${PORT}/tasks/${postBody.id}`);
	expect(get.ok).toBe(true);
	const getBody = (await get.json()) as { id: string; tasks: unknown[] };
	expect(getBody.id).toBe(postBody.id);
	expect(getBody.tasks).toHaveLength(2);
});

test("POST /tasks with custom id round-trips", async () => {
	const post = await fetch(`http://127.0.0.1:${PORT}/tasks`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ id: "pinned-id", tasks: [{ id: "only" }] }),
	});
	expect(post.ok).toBe(true);
	const postBody = (await post.json()) as { id: string };
	expect(postBody.id).toBe("pinned-id");

	const get = await fetch(`http://127.0.0.1:${PORT}/tasks/pinned-id`);
	expect(get.ok).toBe(true);
});

test("POST /tasks rejects empty arrays", async () => {
	const res = await fetch(`http://127.0.0.1:${PORT}/tasks`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ tasks: [] }),
	});
	expect(res.status).toBe(400);
});

test("GET /tasks/:unknown-id returns 404", async () => {
	const res = await fetch(`http://127.0.0.1:${PORT}/tasks/does-not-exist`);
	expect(res.status).toBe(404);
});

test("POST /system-profiles registers and returns a stable id, GET fetches it", async () => {
	const profile = {
		userAgent: "Mozilla/5.0 ... Chrome/130.0.0.0 Safari/537.36",
		chromeVersion: "130.0.0.0",
		gpuVendor: "Apple",
		gpuArchitecture: "metal-3",
		gpuDevice: "Apple M5 Max",
		hardwareConcurrency: 16,
		gpuMaxBufferSize: 268435456,
	};
	const post1 = await fetch(`http://127.0.0.1:${PORT}/system-profiles`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(profile),
	});
	expect(post1.ok).toBe(true);
	const { systemId } = (await post1.json()) as {
		ok: boolean;
		systemId: string;
	};
	expect(systemId).toMatch(/^[0-9a-f]{16}$/);

	// Re-posting the same profile yields the same id (dedup).
	const post2 = await fetch(`http://127.0.0.1:${PORT}/system-profiles`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(profile),
	});
	const { systemId: id2 } = (await post2.json()) as { systemId: string };
	expect(id2).toBe(systemId);

	// GET by id returns the full profile.
	const get = await fetch(
		`http://127.0.0.1:${PORT}/system-profiles/${systemId}`,
	);
	expect(get.ok).toBe(true);
	const fetched = (await get.json()) as { systemId: string; gpuVendor: string };
	expect(fetched.systemId).toBe(systemId);
	expect(fetched.gpuVendor).toBe("Apple");
});

test("POST /system-profiles rejects payloads without userAgent", async () => {
	const res = await fetch(`http://127.0.0.1:${PORT}/system-profiles`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ gpuVendor: "Apple" }),
	});
	expect(res.status).toBe(400);
});

test("GET /system-profiles/:unknown returns 404", async () => {
	const res = await fetch(
		`http://127.0.0.1:${PORT}/system-profiles/0000000000000000`,
	);
	expect(res.status).toBe(404);
});

test("POST /ingest?kind=eval_started validates payload", async () => {
	const bad = await fetch(`http://127.0.0.1:${PORT}/ingest?kind=eval_started`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ evalId: "e", modelId: "m" }),
	});
	expect(bad.status).toBe(400);
	const err = await bad.json();
	expect(err.code).toBe("bad_request");
});
