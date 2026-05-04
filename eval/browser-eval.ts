/**
 * Drive an accuracy eval through the browser's real WebGPU pipeline.
 *
 * Given a profile and a task list, this:
 *  1. POSTs the tasks to the live-server's /tasks staging endpoint
 *  2. Navigates the smoke page to `?bench=<taskListId>&ingest=<live-url>`
 *  3. Polls `window.__benchStatus` until `done: true`
 *  4. Reports the final score + per-dimension passed/total
 *
 * The browser itself streams `eval_started` / `eval_task_complete` /
 * `eval_complete` events to the live dashboard as it runs — so results
 * appear live without this driver intermediating.
 */
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
	type SmokeTestPage,
} from "./browser-smoke.js";
import { resolveLiveBenchUrl } from "./live-client.js";
import { getModelById } from "./models.js";
import {
	getSmokeProfile,
	listSmokeProfiles,
	profileToUrlParams,
	resolveProfileModel,
} from "./smoke-profiles.js";
import { embeddingTasks } from "./tasks/embedding.js";
import { instructionTasks } from "./tasks/instruction.js";
import { reasoningTasks } from "./tasks/reasoning.js";
// Side-effect import so the Bun driver is aware of the same scorer names
// the browser will register — useful for future Bun-side scoring.
import "./tasks/scorer-registrations.js";
import { semanticReasoningTasks } from "./tasks/semantic-reasoning.js";
import { toolCallingTasks } from "./tasks/tool-calling.js";
import type { EvalDimension, EvalTask } from "./types.js";

const ALL_TASKS: EvalTask[] = [
	...toolCallingTasks,
	...reasoningTasks,
	...instructionTasks,
	...semanticReasoningTasks,
	...embeddingTasks,
];

const DIMS: EvalDimension[] = [
	"tool-calling",
	"reasoning",
	"instruction-following",
	"semantic-reasoning",
	"embedding",
];

const POLL_INTERVAL_MS = 5000;
// 10 min hard ceiling — accuracy run on a 1B+ model is slow. Override via
// WEBLLM_HARD_TIMEOUT_MS for >4 GiB MEMORY64 targets where cold-cache fetch
// + 36 prompts can total 15-20 min.
const DEFAULT_TIMEOUT_MS = (() => {
	const raw = process.env.WEBLLM_HARD_TIMEOUT_MS;
	const parsed = raw !== undefined ? Number.parseInt(raw, 10) : NaN;
	return Number.isFinite(parsed) && parsed > 0 ? parsed : 10 * 60 * 1000;
})();
// Bail this much sooner if the page stops making progress. Covers two modes:
//   (a) `__benchStatus` is never published (model-load hang, CDP wedged, page
//       crashed before bench entry); pollBenchStatus keeps returning null.
//   (b) `__benchStatus` is published but `completedTasks` stops advancing —
//       a WASM abort leaves the module dead and subsequent tasks hang.
// The clock starts at loop entry and resets on (first non-null status) or
// (advance in `completedTasks`). 180s is generous enough for a cold 1–3B
// model download + weight load, yet still far below the hard ceiling.
// Override via WEBLLM_STALL_TIMEOUT_MS for >4 GiB MEMORY64 targets where
// a cold GGUF fetch can exceed 3 min before the bench loop publishes
// its first __benchStatus.
const STALL_TIMEOUT_MS = (() => {
	const raw = process.env.WEBLLM_STALL_TIMEOUT_MS;
	const parsed = raw !== undefined ? Number.parseInt(raw, 10) : NaN;
	return Number.isFinite(parsed) && parsed > 0 ? parsed : 180_000;
})();

function main(): void {
	const { values } = parseArgs({
		options: {
			profile: { type: "string" },
			model: { type: "string", short: "m" },
			dimension: { type: "string", short: "d" },
			"live-bench-url": { type: "string" },
			"timeout-ms": { type: "string" },
			port: { type: "string" },
			tab: { type: "string" },
			worker: { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	const liveUrl = resolveLiveBenchUrl(values["live-bench-url"]);
	if (!liveUrl) {
		console.error(
			"Error: --live-bench-url or WEBLLM_LIVE_BENCH_URL is required — the browser needs a URL to fetch tasks from and POST events to.",
		);
		process.exit(1);
	}

	const profile = values.profile ? getSmokeProfile(values.profile) : null;
	if (values.profile && !profile) {
		console.error(
			`Error: unknown profile "${values.profile}". Available: ${listSmokeProfiles().join(", ")}`,
		);
		process.exit(1);
	}

	const modelId = profile?.model ?? values.model;
	if (!modelId) {
		console.error("Error: --profile or --model is required");
		process.exit(1);
	}

	const model = profile ? resolveProfileModel(profile) : getModelById(modelId);
	if (!model) {
		console.error(`Error: model "${modelId}" not found in eval/models.ts`);
		process.exit(1);
	}

	const dimension = values.dimension as EvalDimension | undefined;
	if (dimension && !DIMS.includes(dimension)) {
		console.error(
			`Error: invalid dimension "${dimension}". Valid: ${DIMS.join(", ")}`,
		);
		process.exit(1);
	}

	// Tool-calling tasks demand rigid JSON output. Sampling above ~0.2
	// makes names and arg keys drift off-schema, which is noise rather
	// than signal when comparing models. When the whole suite is being
	// run (no explicit --dimension), restrict tool-calling to cold
	// profiles so warm/hot runs focus on dimensions where sampling
	// diversity actually matters. An explicit --dimension tool-calling
	// always runs, on the assumption the caller knows what they want.
	const COLD_TEMP_CEILING = 0.4;
	// When the parent bench harness pins an `--eval-temperature`, it wins
	// over the profile's native temperature (the speed pass keeps native;
	// only the accuracy pass is pinned). Use the effective temperature for
	// the cold-vs-warm gate so a greedy override unlocks tool-calling on
	// warm profiles instead of pointlessly skipping it.
	const evalTempOverrideRaw = process.env.WEBLLM_BENCH_EVAL_TEMPERATURE;
	const evalTempOverride =
		evalTempOverrideRaw !== undefined
			? Number.parseFloat(evalTempOverrideRaw)
			: NaN;
	const effectiveTemp = Number.isFinite(evalTempOverride)
		? evalTempOverride
		: profile?.temperature;
	const isCold =
		typeof effectiveTemp === "number" && effectiveTemp <= COLD_TEMP_CEILING;

	let tasks: EvalTask[];
	if (dimension) {
		tasks = ALL_TASKS.filter((t) => t.dimension === dimension);
	} else if (isCold) {
		tasks = ALL_TASKS;
	} else {
		tasks = ALL_TASKS.filter((t) => t.dimension !== "tool-calling");
		console.log(
			`Skipping tool-calling tasks (effective temperature ${effectiveTemp ?? "default"} > ${COLD_TEMP_CEILING}; rerun with --dimension tool-calling to force).`,
		);
	}

	// Embedding-track tasks need a working encoder (`engine.embed`). No
	// generative model has that wired up yet. Auto-skip unless the model
	// declares the capability and the caller didn't explicitly request
	// the dimension. Explicit `--dimension embedding` runs it regardless
	// so the "not yet implemented" error surfaces for debugging.
	if (dimension !== "embedding" && !model.capabilities?.embedding) {
		const before = tasks.length;
		tasks = tasks.filter((t) => t.dimension !== "embedding");
		if (before !== tasks.length) {
			console.log(
				`Skipping embedding tasks (model "${model.id}" has capabilities.embedding=false; rerun with --dimension embedding to force).`,
			);
		}
	}

	// Mirror of the above: embedding-only models (BERT-style encoders) can't
	// run generative dimensions. When no explicit `--dimension` is set,
	// auto-restrict to the embedding dimension so the bench harness can
	// drive arctic-embed without hard-failing on tool-calling / reasoning
	// / instruction-following / semantic-reasoning tasks.
	if (
		!dimension &&
		model.capabilities?.embedding &&
		!model.capabilities.toolCalling &&
		!model.capabilities.structuredOutput
	) {
		const before = tasks.length;
		tasks = tasks.filter((t) => t.dimension === "embedding");
		if (before !== tasks.length) {
			console.log(
				`Restricting to embedding tasks (model "${model.id}" is embedding-only).`,
			);
		}
	}

	if (tasks.length === 0) {
		console.error("Error: no tasks to run");
		process.exit(1);
	}

	const timeoutMs = values["timeout-ms"]
		? Number.parseInt(values["timeout-ms"], 10)
		: DEFAULT_TIMEOUT_MS;

	run(model, tasks, liveUrl, {
		profileName: profile?.name,
		contextLength: profile?.contextLength ?? model.contextLength,
		thinking: profile?.thinking === "on",
		port: values.port,
		tab: values.tab,
		timeoutMs,
		extraProfileParams: profile ? profileToUrlParams(profile) : {},
		worker: values.worker ?? false,
	}).catch((err) => {
		console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
		process.exit(1);
	});
}

async function run(
	model: NonNullable<ReturnType<typeof getModelById>>,
	tasks: EvalTask[],
	liveUrl: string,
	opts: {
		profileName?: string;
		contextLength: number;
		thinking: boolean;
		port?: string;
		tab?: string;
		timeoutMs: number;
		extraProfileParams: Record<string, string | number>;
		worker: boolean;
	},
): Promise<void> {
	await ensureSmokeServerReachable();
	await ensureModelDownloaded(model);

	// Stage the task list with live-server
	console.log(`Staging ${tasks.length} tasks at ${liveUrl}/tasks…`);
	const postRes = await fetch(`${liveUrl}/tasks`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ tasks }),
	});
	if (!postRes.ok) {
		throw new Error(
			`POST /tasks failed: HTTP ${postRes.status} ${postRes.statusText}`,
		);
	}
	const { id: taskListId } = (await postRes.json()) as { id: string };
	console.log(`Task list id: ${taskListId}`);

	const page: SmokeTestPage = "smoke";
	const { port, tab } = await resolveAgentchromeSession(opts.port, opts.tab, page);

	const extraParams: Record<string, string | number | boolean> = {
		browserEval: Date.now(),
		bench: taskListId,
		ingest: liveUrl,
		...opts.extraProfileParams,
	};
	if (opts.profileName) extraParams.profile = opts.profileName;
	if (opts.thinking) extraParams.thinking = 1;
	if (opts.worker) extraParams.worker = 1;

	// Tag this per-model eval with the parent bench session so the dashboard
	// can aggregate progress across all models in the run.
	const sessionId = process.env.WEBLLM_BENCH_SESSION_ID;
	if (sessionId) extraParams.session = sessionId;

	// Override profile temperature for the accuracy pass when the parent
	// bench harness has pinned one (default: greedy). Spread *after*
	// extraProfileParams so this wins; the speed pass is unaffected because
	// it never sees this env var (chat-smoke.ts reads profile temp directly).
	// The smoke page reads URL param `temp`, matching `profileToUrlParams`
	// — using the long form `temperature` here would silently no-op.
	const evalTemperatureRaw = process.env.WEBLLM_BENCH_EVAL_TEMPERATURE;
	if (evalTemperatureRaw !== undefined) {
		const t = Number.parseFloat(evalTemperatureRaw);
		if (Number.isFinite(t) && t >= 0) {
			extraParams.temp = t;
		}
	}

	const url = buildSmokeTestUrl(model.id, opts.contextLength, {
		page,
		extraParams,
	});
	console.log(`Navigating to ${url}`);
	agentchrome(port, tab, ["navigate", url]);

	const deadline = Date.now() + opts.timeoutMs;
	let lastCompleted = -1;
	let seenStatus = false;
	// Clock starts immediately. Any signal from the page — first observation
	// of `__benchStatus`, or an advance in `completedTasks` — resets it.
	let lastProgressAt = Date.now();
	while (Date.now() < deadline) {
		await sleep(POLL_INTERVAL_MS);
		const status = pollBenchStatus(port, tab);
		if (status) {
			if (!seenStatus) {
				seenStatus = true;
				lastProgressAt = Date.now();
			}
			if (status.completedTasks !== lastCompleted) {
				lastCompleted = status.completedTasks;
				lastProgressAt = Date.now();
				console.log(
					`  ${status.completedTasks}/${status.totalTasks} tasks (${status.passedTasks} passing)`,
				);
			}
			if (status.done) {
				if (status.error) throw new Error(`browser bench failed: ${status.error}`);
				console.log(
					`\nDone: ${status.passedTasks}/${status.totalTasks} passing · overall ${status.overall !== undefined ? `${Math.round(status.overall * 100)}%` : "?"}`,
				);
				return;
			}
		}
		if (Date.now() - lastProgressAt > STALL_TIMEOUT_MS) {
			const stalledSecs = Math.round((Date.now() - lastProgressAt) / 1000);
			const phase = seenStatus
				? `no task progress for ${stalledSecs}s — last seen ${Math.max(lastCompleted, 0)}/${tasks.length} tasks (likely WASM/page abort; check browser console)`
				: `no signal from page for ${stalledSecs}s — __benchStatus never published (model-load hang, CDP wedged, or page crashed before bench entry)`;
			throw new Error(`browser bench stalled: ${phase}`);
		}
	}
	throw new Error(
		`browser bench timed out after ${Math.round(opts.timeoutMs / 1000)}s — last seen ${Math.max(lastCompleted, 0)}/${tasks.length} tasks`,
	);
}

interface BenchStatus {
	done: boolean;
	totalTasks: number;
	completedTasks: number;
	passedTasks: number;
	overall?: number;
	error?: string;
}

function pollBenchStatus(port: string, tab: string): BenchStatus | null {
	// Suppress agentchrome's CDP-timeout chatter while the browser is busy
	// with a long inference. Those failures are expected mid-run and the
	// next poll succeeds once the browser's event loop services CDP again.
	try {
		const out = execFileSync(
			"agentchrome",
			[
				"--port",
				port,
				"--tab",
				tab,
				"js",
				"exec",
				"JSON.stringify(window.__benchStatus ?? null)",
			],
			{ encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] },
		);
		const resp = JSON.parse(out) as { result?: string };
		if (typeof resp.result !== "string") return null;
		const parsed = JSON.parse(resp.result) as BenchStatus | null;
		return parsed ?? null;
	} catch {
		return null;
	}
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function printUsage(): void {
	console.log(`Usage: bun run eval/browser-eval.ts [options]

Runs the accuracy eval entirely in the browser through real WebGPU. Streams
per-task events to the live dashboard; requires a running dashboard backend.

Options:
      --profile <name>        Profile (sets model + thinking + sampling)
  -m, --model <id>            Model id (alternative to --profile)
  -d, --dimension <name>      Limit to one dimension (tool-calling, reasoning,
                              instruction-following, embedding)
      --live-bench-url <url>  Live dashboard backend (env: WEBLLM_LIVE_BENCH_URL) — required
      --timeout-ms <num>      Overall timeout (default: ${DEFAULT_TIMEOUT_MS})
      --port <cdp-port>       agentchrome CDP port
      --tab <tab-id>          Specific Chrome tab id
      --worker                Run engine inside a DedicatedWorker via WebLLMProxy
  -h, --help                  Show this help

Examples:
  WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \\
    bun run eval/browser-eval.ts --profile qwen3-0.6b-off-warm

  WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \\
    bun run eval/browser-eval.ts --profile qwen3-0.6b-thinking-warm --dimension tool-calling
`);
}

main();
