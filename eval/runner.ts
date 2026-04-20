import type {
	EvalTask,
	EvalResult,
	EvalReport,
	EvalDimension,
	DimensionScore,
	EvalToolDef,
} from "./types.js";
import { score } from "./scorer.js";
import { Character } from "../src/characters/character.js";
import type { ToolDefinition } from "../src/characters/tool-system.js";
import { ToolSystem } from "../src/characters/tool-system.js";

export interface EvalOptions {
	maxTokens?: number;
	temperature?: number;
	timeout?: number;
	onTaskStart?: (task: EvalTask) => void;
	onTaskComplete?: (result: EvalResult) => void;
}

const DEFAULT_TIMEOUT = 30_000;

/** Convert eval tool definitions to real ToolDefinitions with canned handlers. */
function evalToolsToToolDefs(tools: EvalToolDef[]): ToolDefinition[] {
	return tools.map((t) => ({
		name: t.name,
		description: t.description,
		parameters: Object.fromEntries(
			Object.entries(t.parameters).map(([k, v]) => [
				k,
				{
					type: v.type as
						| "string"
						| "number"
						| "boolean"
						| "array"
						| "object",
					description: v.description,
					required: v.required,
				},
			]),
		),
		handler: async () => t.response ?? "ok",
	}));
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

	/** Run a single task and return its result. */
	async runTask(task: EvalTask, options?: EvalOptions): Promise<EvalResult> {
		const timeout = options?.timeout ?? DEFAULT_TIMEOUT;

		try {
			const tools = task.tools ? evalToolsToToolDefs(task.tools) : undefined;
			const character = new Character({
				modelId: "eval",
				systemPrompt: task.systemPrompt,
				maxTokens: task.maxTokens ?? options?.maxTokens,
				temperature: options?.temperature,
				tools,
			});

			const start = Date.now();
			let output = "";

			const gen = character.chat(task.input);
			const iterator = gen[Symbol.asyncIterator]();

			// Collect response with timeout
			const timer = setTimeout(() => {
				character.stop();
			}, timeout);

			try {
				while (true) {
					const result = await Promise.race([
						iterator.next(),
						new Promise<IteratorResult<string>>((_, reject) =>
							setTimeout(
								() => reject(new Error(`Task timed out after ${timeout}ms`)),
								timeout,
							),
						),
					]);
					if (result.done) break;
					output += result.value;
				}
			} finally {
				clearTimeout(timer);
			}

			const end = Date.now();
			const latencyMs = end - start;

			// Parse tool calls from output
			const toolCalls: EvalResult["toolCalls"] = [];
			if (tools && tools.length > 0) {
				const toolSystem = new ToolSystem(tools);
				const call = toolSystem.parseToolCall(output);
				if (call) {
					const toolResult = await toolSystem.execute(call);
					toolCalls.push({
						name: call.name,
						arguments: call.arguments,
						result: toolResult.error ? toolResult.error : toolResult.result,
					});
				}
			}

			// Score the output
			const taskScore = score(output, task);

			// Estimate tokens/second using character length as a rough proxy
			const tokensPerSecond =
				latencyMs > 0 ? Math.round((output.length / latencyMs) * 1000) : 0;

			return {
				taskId: task.id,
				dimension: task.dimension,
				difficulty: task.difficulty,
				score: taskScore,
				modelOutput: output,
				toolCalls,
				latencyMs,
				tokensPerSecond,
			};
		} catch (err) {
			return {
				taskId: task.id,
				dimension: task.dimension,
				difficulty: task.difficulty,
				score: 0,
				modelOutput: "",
				toolCalls: [],
				latencyMs: 0,
				tokensPerSecond: 0,
				error: err instanceof Error ? err.message : String(err),
			};
		}
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
