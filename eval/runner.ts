import type { ChatEngine } from "../src/characters/character.js";
import { runTask as libRunTask } from "../src/evaluation/runner.js";
import type {
	DimensionScore,
	EvalDimension,
	EvalReport,
	EvalResult,
	EvalTask,
} from "./types.js";

export interface EvalOptions {
	maxTokens?: number;
	temperature?: number;
	timeout?: number;
	/**
	 * Engine that drives real model inference. Without one, every task
	 * returns `error: "Character has no engine attached..."` — expected
	 * when the harness runs from Bun without a loaded model; real runs
	 * happen via the browser bench mode.
	 */
	engine?: ChatEngine;
	enableThinking?: boolean;
	onTaskStart?: (task: EvalTask) => void;
	onTaskComplete?: (result: EvalResult) => void;
}

/** Build an EvalReport from a model ID and an array of results. */
function buildReport(modelId: string, results: EvalResult[]): EvalReport {
	const byDimension = new Map<EvalDimension, EvalResult[]>();
	for (const r of results) {
		const list = byDimension.get(r.dimension) ?? [];
		list.push(r);
		byDimension.set(r.dimension, list);
	}

	const dimensions: Record<string, DimensionScore> = {};
	let totalScore = 0;
	for (const [dim, dimResults] of byDimension) {
		const total = dimResults.length;
		const passed = dimResults.filter((r) => r.score >= 0.5).length;
		const dimScore =
			total > 0 ? dimResults.reduce((s, r) => s + r.score, 0) / total : 0;
		const avgLatencyMs =
			total > 0
				? dimResults.reduce((s, r) => s + r.latencyMs, 0) / total
				: 0;
		dimensions[dim] = {
			total,
			passed,
			score: Math.round(dimScore * 100) / 100,
			avgLatencyMs: Math.round(avgLatencyMs),
		};
		totalScore += dimScore * total;
	}

	const overall =
		results.length > 0 ? Math.round((totalScore / results.length) * 100) / 100 : 0;

	return {
		timestamp: new Date().toISOString(),
		modelId,
		totalTasks: results.length,
		results,
		dimensions: dimensions as Record<EvalDimension, DimensionScore>,
		overall,
	};
}

/**
 * Core evaluation runner that executes eval tasks against a model via Character.
 *
 * Supports running individual tasks, all tasks, filtered by dimension,
 * and an interactive mode with a terminal menu.
 */
export class EvalRunner {
	private tasks: EvalTask[];

	constructor(tasks: EvalTask[]) {
		this.tasks = tasks;
	}

	/**
	 * Run a single task and return its result. Requires an engine in
	 * `options.engine` — without one, every task short-circuits with
	 * `error: "Character has no engine attached..."`. This is intentional:
	 * the harness is thin, the execution is the library's primitive.
	 */
	async runTask(task: EvalTask, options?: EvalOptions): Promise<EvalResult> {
		if (!options?.engine) {
			return {
				taskId: task.id,
				dimension: task.dimension,
				difficulty: task.difficulty,
				score: 0,
				modelOutput: "",
				toolCalls: [],
				latencyMs: 0,
				tokensPerSecond: 0,
				error:
					"no engine provided — accuracy benches must run through the browser bench mode against a loaded model",
			};
		}
		return libRunTask(options.engine, "eval", task, {
			timeoutMs: options.timeout,
			maxTokens: options.maxTokens,
			temperature: options.temperature,
			enableThinking: options.enableThinking,
		});
	}

	/** Run all tasks (optionally filtered) and return a full report. */
	async runAll(
		modelId: string,
		options?: EvalOptions,
	): Promise<EvalReport> {
		const results: EvalResult[] = [];

		for (const task of this.tasks) {
			options?.onTaskStart?.(task);
			const result = await this.runTask(task, options);
			results.push(result);
			options?.onTaskComplete?.(result);
		}

		return buildReport(modelId, results);
	}

	/** Run only tasks matching a specific dimension. */
	async runDimension(
		dimension: EvalDimension,
		modelId: string,
		options?: EvalOptions,
	): Promise<EvalReport> {
		const filtered = this.tasks.filter((t) => t.dimension === dimension);
		const results: EvalResult[] = [];

		for (const task of filtered) {
			options?.onTaskStart?.(task);
			const result = await this.runTask(task, options);
			results.push(result);
			options?.onTaskComplete?.(result);
		}

		return buildReport(modelId, results);
	}

	/** Interactive mode: show a menu, run selected tasks, display results. */
	async runInteractive(
		modelId: string,
		options?: EvalOptions,
	): Promise<EvalReport> {
		const dims = this.getTaskCount();

		console.log("\n=== WebLLM Eval Runner - Interactive Mode ===\n");
		console.log("Available dimensions:\n");

		const dimNames = Object.keys(dims.byDimension) as EvalDimension[];
		for (const dim of dimNames) {
			console.log(`  ${dim}: ${dims.byDimension[dim]} tasks`);
		}
		console.log(`\n  Total: ${dims.total} tasks`);
		console.log(`\n  0. Run all tasks`);

		for (let i = 0; i < dimNames.length; i++) {
			console.log(`  ${i + 1}. Run ${dimNames[i]} tasks`);
		}
		console.log(`  q. Quit\n`);

		const choice = await readLine("Choose option: ");
		if (choice === "q") {
			process.exit(0);
		}

		const choiceNum = Number.parseInt(choice, 10);
		let report: EvalReport;

		if (choiceNum === 0) {
			report = await this.runWithProgress(modelId, this.tasks, options);
		} else if (
			choiceNum >= 1 &&
			choiceNum <= dimNames.length
		) {
			const dim = dimNames[choiceNum - 1];
			const filtered = this.tasks.filter((t) => t.dimension === dim);
			report = await this.runWithProgress(modelId, filtered, options);
		} else {
			console.log("Invalid choice. Running all tasks.");
			report = await this.runWithProgress(modelId, this.tasks, options);
		}

		return report;
	}

	/** Run tasks with progress output, printing each result. */
	private async runWithProgress(
		modelId: string,
		tasks: EvalTask[],
		options?: EvalOptions,
	): Promise<EvalReport> {
		const results: EvalResult[] = [];

		for (let i = 0; i < tasks.length; i++) {
			const task = tasks[i];
			console.log(
				`\n[${i + 1}/${tasks.length}] ${task.id} (${task.difficulty}): ${task.description}`,
			);
			options?.onTaskStart?.(task);

			const result = await this.runTask(task, options);
			results.push(result);
			options?.onTaskComplete?.(result);

			if (result.error) {
				console.log(`  ERROR: ${result.error}`);
			} else {
				console.log(
					`  Score: ${result.score.toFixed(2)} | Latency: ${result.latencyMs}ms`,
				);
			}
		}

		return buildReport(modelId, results);
	}

	/** Get task counts by dimension. */
	getTaskCount(): { total: number; byDimension: Record<EvalDimension, number> } {
		const byDimension = {} as Record<EvalDimension, number>;
		for (const task of this.tasks) {
			byDimension[task.dimension] = (byDimension[task.dimension] ?? 0) + 1;
		}
		return { total: this.tasks.length, byDimension };
	}

	/** List all tasks. */
	listTasks(): EvalTask[] {
		return [...this.tasks];
	}
}

/** Read a single line from stdin (Bun-compatible). */
function readLine(prompt: string): Promise<string> {
	return new Promise((resolve) => {
		process.stdout.write(prompt);
		const chunks: Buffer[] = [];

		const onData = (chunk: Buffer) => {
			const str = chunk.toString();
			if (str.includes("\n")) {
				process.stdin.removeListener("data", onData);
				chunks.push(
					Buffer.from(str.slice(0, str.indexOf("\n")), "utf-8"),
				);
				resolve(Buffer.concat(chunks).toString("utf-8").trim());
			} else {
				chunks.push(chunk);
			}
		};

		process.stdin.resume();
		process.stdin.on("data", onData);
	});
}
