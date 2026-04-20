import { parseArgs } from "node:util";
import { EvalRunner } from "./runner.js";
import { writeReport, formatTable } from "./report.js";
import type { EvalDimension } from "./types.js";
import { toolCallingTasks } from "./tasks/tool-calling.js";
import { reasoningTasks } from "./tasks/reasoning.js";
import { instructionTasks } from "./tasks/instruction.js";

const allTasks = [...toolCallingTasks, ...reasoningTasks, ...instructionTasks];

const DIMENSION_LABELS: Record<string, string> = {
	"tool-calling": "TOOL CALLING",
	reasoning: "REASONING",
	"instruction-following": "INSTRUCTION FOLLOWING",
};

const VALID_DIMENSIONS: string[] = [
	"tool-calling",
	"reasoning",
	"instruction-following",
];

function main(): void {
	const { values } = parseArgs({
		options: {
			model: { type: "string", short: "m" },
			dimension: { type: "string", short: "d" },
			interactive: { type: "boolean", short: "i" },
			output: { type: "string", short: "o" },
			temperature: { type: "string", short: "t" },
			"max-tokens": { type: "string" },
			timeout: { type: "string" },
			list: { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	if (values.list) {
		printTaskList();
		process.exit(0);
	}

	if (!values.model) {
		console.error("Error: --model is required (unless using --list)");
		printUsage();
		process.exit(1);
	}

	if (values.dimension && !VALID_DIMENSIONS.includes(values.dimension)) {
		console.error(
			`Error: invalid dimension "${values.dimension}". Valid: ${VALID_DIMENSIONS.join(", ")}`,
		);
		process.exit(1);
	}

	const modelId = values.model;
	const runner = new EvalRunner(allTasks);
	const options = {
		temperature: values.temperature
			? Number.parseFloat(values.temperature)
			: undefined,
		maxTokens: values["max-tokens"]
			? Number.parseInt(values["max-tokens"], 10)
			: undefined,
		timeout: values.timeout
			? Number.parseInt(values.timeout, 10)
			: 30_000,
	};

	if (values.interactive) {
		runner
			.runInteractive(modelId, options)
			.then((report) => {
				writeReport(report, values.output);
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

	console.log(
		`Running ${filteredTasks.length} tasks against model "${modelId}"...`,
	);

	const startTime = Date.now();

	const runPromise = values.dimension
		? taskRunner.runDimension(
				values.dimension as EvalDimension,
				modelId,
				{
					...options,
					onTaskComplete: () => {
						process.stdout.write(".");
					},
				},
			)
		: taskRunner.runAll(modelId, {
				...options,
				onTaskComplete: () => {
					process.stdout.write(".");
				},
			});

	runPromise
		.then((report) => {
			const elapsed = Date.now() - startTime;
			console.log(`\nCompleted in ${elapsed}ms\n`);
			writeReport(report, values.output);
		})
		.catch((err) => {
			console.error(
				`\nFatal: ${err instanceof Error ? err.message : err}`,
			);
			process.exit(1);
		});
}

function printUsage(): void {
	console.log(`
Usage: bun run eval/cli.ts [options]

Options:
  -m, --model <string>       Model ID to evaluate (required unless --list)
  -d, --dimension <string>   Filter to one dimension (tool-calling, reasoning, instruction-following)
  -i, --interactive          Interactive mode
  -o, --output <string>      Report output directory (default: eval/reports)
  -t, --temperature <number> Override sampling temperature
  --max-tokens <number>      Override max tokens per task
  --timeout <number>         Per-task timeout in ms (default: 30000)
  --list                     List all tasks and exit
  -h, --help                 Show this help
`);
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
