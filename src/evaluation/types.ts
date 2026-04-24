/**
 * Shared types for the evaluation harness. These define the contract between
 * task authors (who describe what to test and how to score it) and runners
 * (who execute the task against a Character and produce results).
 *
 * This module is the public library surface for evaluation; external
 * consumers can import from `@paulrobello/webllm` without pulling in the
 * Bun-side bench harness under `eval/`.
 */

export type EvalDimension =
	| "tool-calling"
	| "reasoning"
	| "instruction-following"
	| "embedding";

export type ScoringMethod =
	| { type: "exact" }
	| { type: "contains"; value: string }
	| { type: "regex"; pattern: string }
	| { type: "json_schema"; schema: Record<string, string> }
	| {
			type: "tool_call";
			expectedName: string;
			expectedArgs?: Record<string, unknown>;
	  }
	| {
			type: "tool_call_chain";
			steps: Array<{ name: string; args?: Record<string, unknown> }>;
	  }
	| { type: "no_tool_call" }
	| {
			/**
			 * Looks up `name` in the custom-scorer registry (see
			 * `custom-scorers.ts`). Tasks cross JSON boundaries, so scorer
			 * functions live in code registered on both sides by the same
			 * name — never inline in the task definition.
			 */
			type: "custom";
			name: string;
	  };

export interface EvalToolDef {
	name: string;
	description: string;
	parameters: Record<
		string,
		{ type: string; description?: string; required?: boolean }
	>;
	response: unknown;
}

export interface EvalTask {
	id: string;
	dimension: EvalDimension;
	description: string;
	systemPrompt: string;
	input: string;
	expected: string;
	scoring: ScoringMethod;
	tools?: EvalToolDef[];
	maxTokens?: number;
	difficulty: "easy" | "medium" | "hard";
}

export interface ToolCallRecord {
	name: string;
	arguments: Record<string, unknown>;
	result: unknown;
}

export interface EvalResult {
	taskId: string;
	dimension: EvalDimension;
	difficulty: string;
	score: number;
	modelOutput: string;
	toolCalls: ToolCallRecord[];
	latencyMs: number;
	tokensPerSecond: number;
	error?: string;
}

export interface DimensionScore {
	total: number;
	passed: number;
	score: number;
	avgLatencyMs: number;
}

export interface EvalReport {
	timestamp: string;
	modelId: string;
	totalTasks: number;
	results: EvalResult[];
	dimensions: Record<EvalDimension, DimensionScore>;
	overall: number;
	/**
	 * Intended thinking mode for the run. Recorded so consumers can
	 * differentiate thinking-on vs thinking-off runs of the same model.
	 */
	thinking?: "off" | "on";
	/** Profile name if the run was profile-driven. */
	profile?: string;
}
