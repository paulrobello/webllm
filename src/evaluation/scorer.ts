import { ToolSystem } from "../characters/tool-system.js";
import { getCustomScorer } from "./custom-scorers.js";
import type { EvalTask } from "./types.js";

/**
 * Score a model's output against a task's expected answer / scoring rule.
 *
 * Pure function — takes a string output and the task definition, returns
 * a score in [0, 1]. 1 = full credit, 0 = no credit, values between are
 * partial (used by tool-call chain matching and JSON schema coverage).
 *
 * Works in any runtime (Node, Bun, browser) — no I/O.
 */
export function score(output: string, task: EvalTask): number {
	const { scoring } = task;

	switch (scoring.type) {
		case "exact":
			return scoreExact(output, task.expected);
		case "contains":
			return scoreContains(output, scoring.value);
		case "regex":
			return scoreRegex(output, scoring.pattern);
		case "json_schema":
			return scoreJsonSchema(output, scoring.schema);
		case "tool_call":
			return scoreToolCall(
				output,
				scoring.expectedName,
				scoring.expectedArgs,
				task,
			);
		case "tool_call_chain":
			return scoreToolCallChain(output, scoring.steps, task);
		case "no_tool_call":
			return scoreNoToolCall(output, task);
		case "custom": {
			const fn = getCustomScorer(scoring.name);
			if (!fn) {
				console.warn(
					`[scorer] no custom scorer registered for "${scoring.name}" — scoring 0. Did you import the scorer registrations module?`,
				);
				return 0;
			}
			return fn(output, task.expected);
		}
		case "cosine_similarity":
			// Cosine tasks require vectors, not text. The embedding-track
			// runner computes the cosine score directly via
			// `scoreCosineSimilarity` and bypasses this function. Reaching
			// this branch means a cosine task was fed into the text-score
			// path by mistake — return 0 rather than silently succeed.
			console.warn(
				`[scorer] cosine_similarity task "${task.id}" reached text score() — use engine.embed() + scoreCosineSimilarity() instead.`,
			);
			return 0;
		default:
			return 0;
	}
}

/**
 * Map a cosine similarity ∈ [-1, 1] to a score ∈ [0, 1]. The bench
 * harness uses the resulting score the same way as any other task score
 * (pass threshold 0.5 for dashboard counts, continuous value for
 * per-dim averaging).
 */
export function scoreCosineSimilarity(
	a: Float32Array,
	b: Float32Array,
): number {
	if (a.length !== b.length) {
		throw new Error(
			`scoreCosineSimilarity: vector length mismatch (${a.length} vs ${b.length})`,
		);
	}
	let dot = 0;
	let normA = 0;
	let normB = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	if (normA === 0 || normB === 0) return 0;
	const cos = dot / (Math.sqrt(normA) * Math.sqrt(normB));
	// Clamp to guard against floating-point overrun on near-identical
	// vectors, then map [-1, 1] → [0, 1].
	const clamped = Math.max(-1, Math.min(1, cos));
	return (clamped + 1) / 2;
}

function scoreExact(output: string, expected: string): number {
	return output.trim().toLowerCase() === expected.trim().toLowerCase() ? 1 : 0;
}

function scoreContains(output: string, value: string): number {
	return output.toLowerCase().includes(value.toLowerCase()) ? 1 : 0;
}

function scoreRegex(output: string, pattern: string): number {
	try {
		return new RegExp(pattern).test(output) ? 1 : 0;
	} catch {
		return 0;
	}
}

function scoreJsonSchema(
	output: string,
	schema: Record<string, string>,
): number {
	try {
		const json = extractJson(output);
		if (!json) return 0;
		const parsed = JSON.parse(json) as Record<string, unknown>;
		let matched = 0;
		const total = Object.keys(schema).length;
		for (const [field, expectedType] of Object.entries(schema)) {
			const val = parsed[field];
			if (val === undefined) continue;
			if (checkType(val, expectedType)) matched++;
		}
		return total > 0 ? matched / total : 0;
	} catch {
		return 0;
	}
}

function scoreToolCall(
	output: string,
	expectedName: string,
	expectedArgs: Record<string, unknown> | undefined,
	task: EvalTask,
): number {
	const system = makeToolSystem(task.tools ?? []);
	const call = system.parseToolCall(output);
	if (!call) return 0;
	if (call.name !== expectedName) return 0;
	if (!expectedArgs) return 1;

	let matched = 0;
	const total = Object.keys(expectedArgs).length;
	for (const [key, val] of Object.entries(expectedArgs)) {
		if (call.arguments[key] === val) matched++;
	}
	return total > 0 ? 0.5 + (0.5 * matched) / total : 1;
}

function scoreToolCallChain(
	output: string,
	steps: Array<{ name: string; args?: Record<string, unknown> }>,
	task: EvalTask,
): number {
	const system = makeToolSystem(task.tools ?? []);
	const calls = extractAllToolCalls(output, system);

	let matched = 0;
	for (let i = 0; i < steps.length; i++) {
		const step = steps[i];
		const call = calls[i];
		if (!call) break;
		if (call.name !== step.name) continue;
		if (!step.args) {
			matched++;
			continue;
		}
		let argMatch = 0;
		const argTotal = Object.keys(step.args).length;
		for (const [key, val] of Object.entries(step.args)) {
			if (call.arguments[key] === val) argMatch++;
		}
		matched += argTotal > 0 ? argMatch / argTotal : 1;
	}

	return steps.length > 0 ? matched / steps.length : 0;
}

function scoreNoToolCall(output: string, task: EvalTask): number {
	const system = makeToolSystem(task.tools ?? []);
	return system.parseToolCall(output) === null ? 1 : 0;
}

function extractJson(text: string): string | null {
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start === -1 || end === -1 || end < start) return null;
	return text.slice(start, end + 1);
}

function extractAllToolCalls(
	text: string,
	system: ToolSystem,
): Array<{ name: string; arguments: Record<string, unknown> }> {
	const calls: Array<{ name: string; arguments: Record<string, unknown> }> = [];
	let remaining = text;
	while (remaining.length > 0) {
		const call = system.parseToolCall(remaining);
		if (!call) break;
		calls.push(call);
		const matchIdx = remaining.indexOf(call.name);
		remaining =
			matchIdx >= 0 ? remaining.slice(matchIdx + call.name.length) : "";
	}
	return calls;
}

function makeToolSystem(
	tools: Array<{
		name: string;
		description: string;
		parameters: Record<
			string,
			{ type: string; description?: string; required?: boolean }
		>;
		response?: unknown;
	}>,
): ToolSystem {
	return new ToolSystem(
		tools.map((t) => ({
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
		})),
	);
}

function checkType(val: unknown, expectedType: string): boolean {
	if (expectedType === "string") return typeof val === "string";
	if (expectedType === "number") return typeof val === "number";
	if (expectedType === "boolean") return typeof val === "boolean";
	if (expectedType === "array") return Array.isArray(val);
	if (expectedType === "object")
		return typeof val === "object" && val !== null && !Array.isArray(val);
	return false;
}
