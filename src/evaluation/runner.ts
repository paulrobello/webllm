/**
 * Reusable evaluation runner primitives. Drives a real `Character` against
 * an `EvalTask`, captures latency + tool calls, scores via the shared
 * scorer, and returns an `EvalResult`.
 *
 * Usage (library consumer):
 *
 *   const { engine, handle } = await WebLLM.loadModelFromBuffer(buf, name, {...});
 *   const result = await runTask(engine, handle.id, myTask);
 *
 * No I/O beyond what the engine's inference pipeline does — works in any
 * runtime that can host WebLLM.
 */
import type {
	Character,
	CharacterConfig,
	ChatEngine,
} from "../characters/character.js";
import { Character as CharacterClass } from "../characters/character.js";
import type { ToolDefinition } from "../characters/tool-system.js";
import { ToolSystem } from "../characters/tool-system.js";
import { score, scoreCosineSimilarityDetails } from "./scorer.js";
import type { EvalResult, EvalTask, EvalToolDef } from "./types.js";

export interface RunTaskOptions {
	/** Soft timeout; `character.stop()` is called when exceeded. Default: 30s. */
	timeoutMs?: number;
	/** Override the task's own maxTokens. */
	maxTokens?: number;
	/** Override sampling temperature. */
	temperature?: number;
	/** Thinking mode (Qwen3). */
	enableThinking?: boolean;
	/**
	 * Optional per-chunk callback — lets UIs stream tokens as they arrive.
	 * Return `false` to abort (calls `character.stop()`); any other return
	 * value (including `undefined`) keeps streaming.
	 */
	onToken?: (token: string) => boolean | undefined;
}

const DEFAULT_TIMEOUT = 30_000;

function evalToolsToToolDefs(tools: EvalToolDef[]): ToolDefinition[] {
	return tools.map((t) => ({
		name: t.name,
		description: t.description,
		parameters: Object.fromEntries(
			Object.entries(t.parameters).map(([k, v]) => [
				k,
				{
					type: v.type as "string" | "number" | "boolean" | "array" | "object",
					description: v.description,
					required: v.required,
				},
			]),
		),
		handler: async () => t.response ?? "ok",
	}));
}

/**
 * Run a single task against a freshly-constructed Character bound to the
 * given engine + model. Returns the scored `EvalResult` including per-task
 * latency and tool-call records.
 */
export async function runTask(
	engine: ChatEngine,
	modelId: string,
	task: EvalTask,
	options: RunTaskOptions = {},
): Promise<EvalResult> {
	const timeout = options.timeoutMs ?? DEFAULT_TIMEOUT;

	// Embedding-dimension tasks bypass chat entirely: there is no prompt
	// to stream, no tools to wire, no scorer to invoke on text. We embed
	// the input + reference and score by cosine. Throws explicitly if the
	// engine has no `embed` — this is a configuration bug, not a soft
	// fallback case.
	if (task.dimension === "embedding") {
		if (typeof engine.embed !== "function") {
			throw new Error(
				`runTask: embedding dimension requires engine.embed (model "${modelId}")`,
			);
		}
		if (task.scoring.type !== "cosine_similarity") {
			throw new Error(
				`runTask: embedding-dimension task "${task.id}" must use cosine_similarity scoring`,
			);
		}
		engine.resetConversation?.(modelId);
		const start = Date.now();
		let error: string | undefined;
		let cosine = 0;
		let score = 0;
		try {
			const a = await engine.embed(modelId, task.input);
			const b = await engine.embed(modelId, task.scoring.reference);
			const r = scoreCosineSimilarityDetails(a, b);
			cosine = r.cosine;
			score = r.score;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		const latencyMs = Date.now() - start;
		return {
			taskId: task.id,
			dimension: task.dimension,
			difficulty: task.difficulty,
			score,
			modelOutput: "",
			toolCalls: [],
			latencyMs,
			tokensPerSecond: 0,
			error,
			embeddingCosine: cosine,
		};
	}

	const tools = task.tools ? evalToolsToToolDefs(task.tools) : undefined;

	// Each task has its own system prompt and tools — there's no shared
	// prefix to preserve, and a stale KV cache from the previous task
	// would either bloat the prompt or (worse) collide with the new
	// position-0 writes and abort the WASM module.
	engine.resetConversation?.(modelId);

	const config: CharacterConfig = {
		modelId,
		systemPrompt: task.systemPrompt,
		maxTokens: options.maxTokens ?? task.maxTokens,
		temperature: options.temperature,
		enableThinking: options.enableThinking,
		tools,
		engine,
	};
	const character: Character = new CharacterClass(config);

	const start = Date.now();
	let output = "";
	let error: string | undefined;

	const timer = setTimeout(() => {
		character.stop();
	}, timeout);

	try {
		for await (const token of character.chat(task.input)) {
			if (options.onToken && options.onToken(token) === false) {
				character.stop();
				break;
			}
			output += token;
		}
	} catch (e) {
		error = e instanceof Error ? e.message : String(e);
	} finally {
		clearTimeout(timer);
	}

	const latencyMs = Date.now() - start;

	// Parse tool calls from the assembled output. The Character already
	// records these in its message history, but EvalResult wants them as
	// a plain list regardless of whether the scoring cared.
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

	const taskScore = error ? 0 : score(output, task);
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
		error,
	};
}

export interface RunTasksOptions extends RunTaskOptions {
	/** Fires before each task starts. */
	onTaskStart?: (task: EvalTask) => void;
	/** Fires after each task completes (pass or fail). */
	onTaskComplete?: (result: EvalResult) => void | Promise<void>;
}

/**
 * Run a batch of tasks sequentially. Sequential (not parallel) because a
 * single Character holds shared KV-cache state per model; running two at
 * once would corrupt it.
 */
export async function runTasks(
	engine: ChatEngine,
	modelId: string,
	tasks: EvalTask[],
	options: RunTasksOptions = {},
): Promise<EvalResult[]> {
	const results: EvalResult[] = [];
	for (const task of tasks) {
		options.onTaskStart?.(task);
		const result = await runTask(engine, modelId, task, options);
		results.push(result);
		await options.onTaskComplete?.(result);
	}
	return results;
}
