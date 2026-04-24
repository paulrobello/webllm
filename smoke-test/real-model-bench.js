/**
 * Browser bench mode. Runs an eval-task list against the already-loaded
 * smoke-test pipeline via the library's `runTasks` primitive. Each task
 * hits the real WebGPU inference path, so speed metrics are the genuine
 * article — no Bun-side Character shortcut.
 *
 * Called from real-model-page.js after the [7/7] smoke generation step
 * succeeds, gated by the `?bench=<taskListId>` URL param.
 */

export async function runBenchMode({
	WebLLM,
	runTasks,
	score,
	collectBrowserSystemProfile,
	wasm,
	inference,
	parsed,
	modelId,
	taskListId,
	ingestUrl,
	log,
	setProgress,
	profileName,
	thinking,
	params,
}) {
	// Load the custom-scorer registrations before any task is scored.
	// Side-effect import; safe to call multiple times.
	log("running", "[bench] loading custom-scorer registrations…");
	await import(`./scorer-registrations.js${window.location.search || ""}`);
	log("running", "[bench] requesting GPU device for library engine…");
	if (!navigator.gpu) {
		throw new Error("navigator.gpu not available; bench mode needs WebGPU");
	}
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("no WebGPU adapter; bench mode needs a GPU");
	}

	// System profile is normally collected once by real-model-page.js
	// when an ingest URL is present. If the page didn't get a chance to
	// (e.g. ingest URL only known at bench-mode time), do it here.
	let systemId = window.__webllmSystemId;
	if (!systemId) {
		try {
			const profile = await collectBrowserSystemProfile(adapter);
			systemId = profile.systemId;
			log(
				"running",
				`[bench] system: ${profile.gpuVendor ?? "?"} · ${profile.gpuArchitecture ?? "?"} · Chrome ${profile.chromeVersion ?? "?"} (id ${systemId})`,
			);
			await fetch(`${ingestUrl}/system-profiles`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify(profile),
			}).catch((err) =>
				console.warn(`[bench] system-profile registration failed: ${err}`),
			);
			window.__webllmSystemId = systemId;
		} catch (err) {
			console.warn(`[bench] system-profile collection failed: ${err}`);
		}
	}

	const device = await adapter.requestDevice();

	log("running", "[bench] constructing WebLLM engine and adopting pipeline…");
	const engine = await WebLLM.init({
		device,
		memoryBudget: 2_000_000_000,
	});
	const handle = await engine.adoptPreloadedModel(modelId, {
		wasm,
		inference,
		parsed,
	});

	log("running", `[bench] fetching task list ${taskListId} from ${ingestUrl}…`);
	const tasksRes = await fetch(`${ingestUrl}/tasks/${taskListId}`);
	if (!tasksRes.ok) {
		throw new Error(
			`task list fetch failed: HTTP ${tasksRes.status} ${tasksRes.statusText}`,
		);
	}
	const { tasks } = await tasksRes.json();
	if (!Array.isArray(tasks) || tasks.length === 0) {
		throw new Error("task list is empty");
	}

	const evalId = `bench-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
	const modelArchLabel = parsed?.hyperparams?.architecture ?? "";
	const dimensions = Array.from(new Set(tasks.map((t) => t.dimension)));

	await postIngest(ingestUrl, "eval_started", {
		evalId,
		modelId,
		totalTasks: tasks.length,
		dimensions,
		label: `${modelId} (${modelArchLabel}) · browser`,
	});

	log("running", `[bench] running ${tasks.length} tasks through real WebGPU…`);
	window.__benchStatus = {
		done: false,
		totalTasks: tasks.length,
		completedTasks: 0,
		passedTasks: 0,
		evalId,
	};

	const results = [];
	try {
		await runTasks(engine, handle.id, tasks, {
			onTaskStart: (task) => {
				log("running", `[bench]   ${task.id} (${task.dimension}/${task.difficulty})…`);
			},
			onTaskComplete: async (result) => {
				// Score was computed inside runTasks, but re-run with the library
				// scorer for any tasks that might have thrown. Keep authoritative.
				// (In practice runTasks already called score(); this is defensive.)
				if (result.score === undefined || Number.isNaN(result.score)) {
					result.score = score(result.modelOutput, tasks.find((t) => t.id === result.taskId));
				}
				results.push(result);
				window.__benchStatus.completedTasks++;
				if (result.score >= 0.5) window.__benchStatus.passedTasks++;
				setProgress(
					Math.min(
						99,
						(window.__benchStatus.completedTasks / tasks.length) * 100,
					),
				);
				const pct = Math.round(result.score * 100);
				log(
					result.score >= 0.5 ? "pass" : "fail",
					`[bench]   ${result.taskId}: ${pct}% (${Math.round(result.latencyMs)}ms, ${result.tokensPerSecond} tok/s)${result.error ? ` — ${result.error}` : ""}`,
				);
				await postIngest(ingestUrl, "eval_task_complete", {
					evalId,
					taskId: result.taskId,
					dimension: result.dimension,
					difficulty: result.difficulty,
					score: result.score,
					latencyMs: result.latencyMs,
					tokensPerSecond: result.tokensPerSecond,
					toolCallsCount: result.toolCalls?.length ?? 0,
					error: result.error,
				});
			},
		});
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		log("fail", `[bench] runTasks threw: ${msg}`);
		await postIngest(ingestUrl, "eval_failed", {
			evalId,
			modelId,
			error: msg,
		});
		window.__benchStatus.done = true;
		window.__benchStatus.error = msg;
		throw err;
	}

	// Build the final report (matches EvalReport shape).
	const report = buildReport(modelId, results, evalId);
	if (systemId) report.systemId = systemId;
	if (profileName) report.profile = profileName;
	report.thinking = thinking ? "on" : "off";
	if (params) {
		const prunedParams = {};
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null) prunedParams[k] = v;
		}
		if (Object.keys(prunedParams).length > 0) report.params = prunedParams;
	}
	await postIngest(ingestUrl, "eval_complete", report);

	window.__benchStatus.done = true;
	window.__benchStatus.overall = report.overall;
	setProgress(100);
	log(
		"pass",
		`[bench] done: ${window.__benchStatus.passedTasks}/${tasks.length} passing · overall ${Math.round(report.overall * 100)}%`,
	);
	return report;
}

async function postIngest(baseUrl, kind, body) {
	try {
		const res = await fetch(`${baseUrl}/ingest?kind=${kind}`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
		});
		if (!res.ok) {
			console.warn(`[bench] ingest ${kind} failed: HTTP ${res.status}`);
		}
	} catch (err) {
		console.warn(
			`[bench] ingest ${kind} threw: ${err?.message ?? String(err)}`,
		);
	}
}

function buildReport(modelId, results, evalId) {
	const byDim = new Map();
	for (const r of results) {
		const list = byDim.get(r.dimension) ?? [];
		list.push(r);
		byDim.set(r.dimension, list);
	}
	const dimensions = {};
	let totalScore = 0;
	for (const [dim, list] of byDim) {
		const passed = list.filter((r) => r.score >= 0.5).length;
		const avgScore = list.reduce((s, r) => s + r.score, 0) / list.length;
		const avgLatency = list.reduce((s, r) => s + r.latencyMs, 0) / list.length;
		dimensions[dim] = {
			total: list.length,
			passed,
			score: Math.round(avgScore * 100) / 100,
			avgLatencyMs: Math.round(avgLatency),
		};
		totalScore += avgScore * list.length;
	}
	const overall =
		results.length > 0 ? Math.round((totalScore / results.length) * 100) / 100 : 0;
	return {
		schemaVersion: 1,
		timestamp: new Date().toISOString(),
		modelId,
		evalId,
		totalTasks: results.length,
		results,
		dimensions,
		overall,
	};
}
