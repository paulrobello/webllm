import { parseArgs } from "node:util";
import {
	publishEvalComplete,
	publishEvalFailed,
	publishEvalStarted,
	publishEvalTaskComplete,
	resolveLiveBenchUrl,
} from "./live-client.js";
import { getModelsByTier, TIER_ORDER, BROWSER_VRAM_LIMITS } from "./models.js";
import type { BenchmarkModel } from "./models.js";
import { generateHtmlReport } from "./report-html.js";
import { writeReport, formatTable } from "./report.js";
import { EvalRunner } from "./runner.js";
import {
	getSmokeProfile,
	listSmokeProfiles,
	type SmokeProfile,
} from "./smoke-profiles.js";
import { embeddingTasks } from "./tasks/embedding.js";
import { instructionTasks } from "./tasks/instruction.js";
import { reasoningTasks } from "./tasks/reasoning.js";
// Register the 13 custom scorers for side effect — must import BEFORE any
// task list that uses them crosses into the scorer.
import "./tasks/scorer-registrations.js";
import { toolCallingTasks } from "./tasks/tool-calling.js";
import type { EvalDimension, EvalReport } from "./types.js";

const allTasks = [...toolCallingTasks, ...reasoningTasks, ...instructionTasks, ...embeddingTasks];

const DIMENSION_LABELS: Record<string, string> = {
	"tool-calling": "TOOL CALLING",
	reasoning: "REASONING",
	"instruction-following": "INSTRUCTION FOLLOWING",
	embedding: "EMBEDDING",
};

const VALID_DIMENSIONS: string[] = [
	"tool-calling",
	"reasoning",
	"instruction-following",
	"semantic-reasoning",
	"embedding",
];

function main(): void {
	const { values } = parseArgs({
		options: {
			model: { type: "string", short: "m" },
			profile: { type: "string" },
			dimension: { type: "string", short: "d" },
			interactive: { type: "boolean", short: "i" },
			output: { type: "string", short: "o" },
			temperature: { type: "string", short: "t" },
			"max-tokens": { type: "string" },
			timeout: { type: "string" },
			models: { type: "boolean" },
			html: { type: "boolean" },
			list: { type: "boolean" },
			"live-bench-url": { type: "string" },
			"eval-id": { type: "string" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	if (values.models) {
		printModelList();
		process.exit(0);
	}

	if (values.list) {
		printTaskList();
		process.exit(0);
	}

	let profile: SmokeProfile | null = null;
	if (values.profile) {
		profile = getSmokeProfile(values.profile) ?? null;
		if (!profile) {
			console.error(
				`Error: unknown profile "${values.profile}". Available: ${listSmokeProfiles().join(", ")}`,
			);
			process.exit(1);
		}
	}

	const effectiveModelId = values.model ?? profile?.model;
	if (!effectiveModelId) {
		console.error(
			"Error: --model or --profile is required (unless using --list or --models)",
		);
		printUsage();
		process.exit(1);
	}

	if (values.dimension && !VALID_DIMENSIONS.includes(values.dimension)) {
		console.error(
			`Error: invalid dimension "${values.dimension}". Valid: ${VALID_DIMENSIONS.join(", ")}`,
		);
		process.exit(1);
	}

	const modelId = effectiveModelId;
	const runner = new EvalRunner(allTasks);
	// Precedence: CLI flag > parent bench-harness env override > profile > default.
	// `WEBLLM_BENCH_EVAL_TEMPERATURE` (set by `eval/bench.ts`) pins greedy
	// (or whatever the parent chose) for the accuracy pass without
	// requiring every profile to be re-defined.
	const evalTempEnvRaw = process.env.WEBLLM_BENCH_EVAL_TEMPERATURE;
	const evalTempFromEnv =
		evalTempEnvRaw !== undefined ? Number.parseFloat(evalTempEnvRaw) : NaN;
	const options = {
		temperature:
			values.temperature !== undefined
				? Number.parseFloat(values.temperature)
				: Number.isFinite(evalTempFromEnv)
					? evalTempFromEnv
					: profile?.temperature,
		maxTokens:
			values["max-tokens"] !== undefined
				? Number.parseInt(values["max-tokens"], 10)
				: profile?.maxTokens,
		timeout: values.timeout ? Number.parseInt(values.timeout, 10) : 30_000,
	};

	if (values.interactive) {
		runner
			.runInteractive(modelId, options)
			.then((report) => {
				writeReport(report, values.output);
				if (values.html) {
					const htmlPath = writeHtmlReport(report, values.output);
					console.log(`HTML report: ${htmlPath}`);
				}
			})
			.catch((err) => {
				console.error(
					`Fatal: ${err instanceof Error ? err.message : err}`,
				);
				process.exit(1);
			});
		return;
	}

	// Normal mode
	const filteredTasks = values.dimension
		? allTasks.filter((t) => t.dimension === values.dimension)
		: allTasks;

	const taskRunner = new EvalRunner(filteredTasks);
	const liveBenchUrl = resolveLiveBenchUrl(values["live-bench-url"]);
	const evalId =
		values["eval-id"] ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

	console.log(
		`Running ${filteredTasks.length} tasks against model "${modelId}"...`,
	);

	const startTime = Date.now();

	const runOptions = {
		...options,
		onTaskComplete: async (result: import("./types.js").EvalResult) => {
			process.stdout.write(".");
			if (liveBenchUrl) {
				await publishEvalTaskComplete(liveBenchUrl, {
					evalId,
					taskId: result.taskId,
					dimension: result.dimension,
					difficulty: result.difficulty,
					score: result.score,
					latencyMs: result.latencyMs,
					tokensPerSecond: result.tokensPerSecond,
					toolCallsCount: result.toolCalls.length,
					error: result.error,
				});
			}
		},
	};

	const sessionId = process.env.WEBLLM_BENCH_SESSION_ID;
	const startPromise = liveBenchUrl
		? publishEvalStarted(liveBenchUrl, {
				evalId,
				modelId,
				totalTasks: filteredTasks.length,
				dimensions: Array.from(
					new Set(filteredTasks.map((t) => t.dimension as string)),
				),
				label:
					profile?.name ?? (values.dimension ? `${values.dimension}` : "all"),
				...(sessionId ? { sessionId } : {}),
			})
		: Promise.resolve(true);

	startPromise
		.then(() =>
			values.dimension
				? taskRunner.runDimension(
						values.dimension as EvalDimension,
						modelId,
						runOptions,
					)
				: taskRunner.runAll(modelId, runOptions),
		)
		.then(async (report) => {
			const elapsed = Date.now() - startTime;
			console.log(`\nCompleted in ${elapsed}ms\n`);
			const annotated = {
				...report,
				thinking: (profile?.thinking ?? "off") as "off" | "on",
				profile: profile?.name,
				params: pruneUndefined({
					contextLength: profile?.contextLength,
					maxTokens: options.maxTokens,
					temperature: options.temperature,
					topK: profile?.topK,
					topP: profile?.topP,
					repetitionPenalty: profile?.repetitionPenalty,
					seed: profile?.seed,
				}),
			};
			writeReport(annotated, values.output);
			if (values.html) {
				const htmlPath = writeHtmlReport(annotated, values.output);
				console.log(`HTML report: ${htmlPath}`);
			}
			if (liveBenchUrl) {
				await publishEvalComplete(liveBenchUrl, { ...annotated, evalId });
			}
		})
		.catch(async (err) => {
			const message = err instanceof Error ? err.message : String(err);
			if (liveBenchUrl) {
				await publishEvalFailed(liveBenchUrl, {
					evalId,
					modelId,
					error: message,
				});
			}
			console.error(`\nFatal: ${message}`);
			process.exit(1);
		});
}

function printUsage(): void {
	console.log(`Usage: bun run eval/cli.ts [options]

Options:
  -m, --model <string>       Model ID to evaluate (required unless --list or --profile)
      --profile <name>       Named profile from eval/smoke-profiles.ts (sets model + sampling defaults)
      --eval-id <id>         Pin the eval run id (used by eval/bench.ts to correlate with a speed pass)
  -d, --dimension <string>   Filter to one dimension (tool-calling, reasoning, instruction-following, embedding)
  -i, --interactive          Interactive mode
  -o, --output <string>      Report output directory (default: eval/reports)
  -t, --temperature <number> Override sampling temperature
  --max-tokens <number>      Override max tokens per task
  --timeout <number>         Per-task timeout in ms (default: 30000)
  --html                     Generate HTML report alongside JSON
  --models                   List available benchmark models and exit
  --list                     List all tasks and exit
      --live-bench-url <url> Stream progress + final report to live dashboard backend (env: WEBLLM_LIVE_BENCH_URL)
  -h, --help                 Show this help`);
}

function pruneUndefined<T extends Record<string, unknown>>(obj: T): T {
	return Object.fromEntries(
		Object.entries(obj).filter(([, v]) => v !== undefined),
	) as T;
}

function writeHtmlReport(report: EvalReport, outputDir?: string): string {
	const dir = outputDir ?? "eval/reports";
	const filename = `${report.timestamp.replace(/[:.]/g, "-")}-${report.modelId}.html`;
	const path = `${dir}/${filename}`;
	generateHtmlReport(report, path);
	return path;
}

function printModelList(): void {
	console.log("\nBenchmark Models:\n");

	const byTier = getModelsByTier();
	const tierOrder: BenchmarkModel["tier"][] = ["ultrafast", "fast", "balanced", "quality"];

	for (const tier of tierOrder) {
		const models = byTier.get(tier);
		if (!models || models.length === 0) continue;

		const info = TIER_ORDER[tier];
		console.log(`  ${info.label} (${info.speedTarget}):`);

		for (const m of models) {
			const caps: string[] = [];
			if (m.capabilities.toolCalling) caps.push("tool-use");
			if (m.capabilities.structuredOutput) caps.push("structured");
			if (m.capabilities.embedding) caps.push("embedding");
			if (m.capabilities.vision) caps.push("vision");
			const capStr = caps.length > 0 ? ` [${caps.join(", ")}]` : "";
			const sizeStr = m.paramsB < 1 ? `${Math.round(m.paramsB * 1000)}M` : `${m.paramsB}B`;

			console.log(`    ${m.id.padEnd(35)} ${sizeStr.padEnd(6)} ${String(m.vramMB).padStart(5)} MB  ${m.license.padEnd(14)}${capStr}`);
		}
		console.log("");
	}

	console.log("Browser VRAM limits:");
	for (const [tier, info] of Object.entries(BROWSER_VRAM_LIMITS)) {
		console.log(`  ${tier.padEnd(12)} ${info.vramGB} GB  max ~${info.maxParamsB}B params  ${info.description}`);
	}
	console.log("");
}

function printTaskList(): void {
	const counts = new EvalRunner(allTasks).getTaskCount();

	console.log(`\nAvailable evaluation tasks (${counts.total} total):\n`);

	const grouped = new Map<string, typeof allTasks>();
	for (const task of allTasks) {
		const list = grouped.get(task.dimension) ?? [];
		list.push(task);
		grouped.set(task.dimension, list);
	}

	for (const [dim, tasks] of grouped) {
		const label = DIMENSION_LABELS[dim] ?? dim;
		console.log(`${label} (${tasks.length}):`);

		for (const task of tasks) {
			const diff = `[${task.difficulty}]`.padEnd(8);
			console.log(`  ${task.id}  ${diff}${task.description}`);
		}
		console.log("");
	}
}

main();
