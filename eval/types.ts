export type EvalDimension =
	| "tool-calling"
	| "reasoning"
	| "instruction-following";

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
			type: "custom";
			scorer: (output: string, expected: string) => number;
		};

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

export interface EvalToolDef {
	name: string;
	description: string;
	parameters: Record<string, { type: string; description?: string; required?: boolean }>;
	response: unknown;
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

export interface ToolCallRecord {
	name: string;
	arguments: Record<string, unknown>;
	result: unknown;
}

export interface EvalReport {
	timestamp: string;
	modelId: string;
	totalTasks: number;
	results: EvalResult[];
	dimensions: Record<EvalDimension, DimensionScore>;
	overall: number;
}

export interface DimensionScore {
	total: number;
	passed: number;
	score: number;
	avgLatencyMs: number;
}
