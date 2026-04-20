/** Schema for a single tool parameter. */
export interface ToolParameter {
	/** JSON Schema type of the parameter. */
	type: "string" | "number" | "boolean" | "array" | "object";
	/** Human-readable description of the parameter. */
	description?: string;
	/** Whether the parameter must be provided. */
	required?: boolean;
	/** Allowed values when the parameter is an enum-like string. */
	enum?: string[];
}

/** Describes a callable tool with its name, description, parameter schema, and handler. */
export interface ToolDefinition {
	/** Unique tool name used in call parsing. */
	name: string;
	/** Human-readable description of what the tool does. */
	description: string;
	/** Parameter schemas keyed by parameter name. */
	parameters: Record<string, ToolParameter>;
	/** Async function that executes the tool with parsed arguments. */
	handler: (args: Record<string, unknown>) => Promise<unknown>;
}

/** Parsed tool invocation extracted from model output. */
export interface ToolCall {
	/** Name of the tool to invoke. */
	name: string;
	/** Key-value arguments for the tool. */
	arguments: Record<string, unknown>;
}

/** Outcome of executing a tool call, including result or error. */
export interface ToolResult {
	/** The original tool call that was executed. */
	call: ToolCall;
	/** Return value from the handler on success. */
	result: unknown;
	/** Error message if execution failed. */
	error?: string;
}

const XML_TOOL_CALL_RE = /<tool_call=(\{.*?\})>/;
const JSON_OBJECT_RE =
	/\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}/;

/**
 * Registers, parses, and executes tool/function calls from model output.
 *
 * Supports two formats: XML-wrapped JSON (`<tool_call={...}>`) and bare JSON objects
 * with "name" and "arguments" fields.
 */
export class ToolSystem {
	private tools: Map<string, ToolDefinition> = new Map();

	/**
	 * @param tools - Tool definitions to register at construction time.
	 */
	constructor(tools: ToolDefinition[]) {
		for (const tool of tools) {
			this.tools.set(tool.name, tool);
		}
	}

	/**
	 * Extract a tool call from raw model output text.
	 *
	 * @param text - Model-generated text that may contain a tool call.
	 * @returns Parsed ToolCall, or null if no valid call found.
	 */
	parseToolCall(text: string): ToolCall | null {
		const xmlMatch = text.match(XML_TOOL_CALL_RE);
		if (xmlMatch?.[1]) {
			return this.parseJson(xmlMatch[1]);
		}

		const jsonMatch = text.match(JSON_OBJECT_RE);
		if (jsonMatch?.[0]) {
			return this.parseJson(jsonMatch[0]);
		}

		return null;
	}

	/**
	 * Execute a parsed tool call via its registered handler.
	 *
	 * @param call - Tool call with name and arguments.
	 * @returns ToolResult with the handler return value or error message.
	 */
	async execute(call: ToolCall): Promise<ToolResult> {
		const tool = this.tools.get(call.name);
		if (!tool) {
			return {
				call,
				result: undefined,
				error: `Unknown tool: ${call.name}`,
			};
		}

		try {
			const result = await tool.handler(call.arguments);
			return { call, result };
		} catch (err) {
			return {
				call,
				result: undefined,
				error: err instanceof Error ? err.message : String(err),
			};
		}
	}

	/**
	 * Check whether a tool with the given name is registered.
	 *
	 * @param name - Tool name.
	 * @returns True if the tool is registered.
	 */
	has(name: string): boolean {
		return this.tools.has(name);
	}

	/**
	 * @returns All registered tool definitions.
	 */
	getDefinitions(): ToolDefinition[] {
		return [...this.tools.values()];
	}

	/**
	 * Render all tool definitions as a human-readable prompt section for the model.
	 *
	 * @returns Formatted tool descriptions, or empty string if no tools registered.
	 */
	formatForPrompt(): string {
		if (this.tools.size === 0) return "";

		const lines: string[] = ["Available tools:"];

		for (const tool of this.tools.values()) {
			lines.push(`- name: ${tool.name}`);
			lines.push(`  description: ${tool.description}`);
			lines.push("  parameters:");
			for (const [paramName, param] of Object.entries(tool.parameters)) {
				const required = param.required ? ", required" : "";
				const enumPart = param.enum ? `, enum: ${param.enum.join(", ")}` : "";
				const desc = param.description ? `: ${param.description}` : "";
				lines.push(
					`    - ${paramName} (${param.type}${required}${enumPart})${desc}`,
				);
			}
		}

		return lines.join("\n");
	}

	/**
	 * Format a tool execution result as XML for inclusion in the conversation.
	 *
	 * @param result - ToolResult to format.
	 * @returns XML string with either <tool_result> or <tool_error> tag.
	 */
	formatResult(result: ToolResult): string {
		if (result.error) {
			return `<tool_error name="${result.call.name}">${result.error}</tool_error>`;
		}
		return `<tool_result name="${result.call.name}">${JSON.stringify(result.result)}</tool_result>`;
	}

	get size(): number {
		return this.tools.size;
	}

	private parseJson(json: string): ToolCall | null {
		try {
			const parsed = JSON.parse(json);
			if (
				typeof parsed === "object" &&
				parsed !== null &&
				typeof parsed.name === "string" &&
				typeof parsed.arguments === "object" &&
				parsed.arguments !== null
			) {
				return {
					name: parsed.name,
					arguments: parsed.arguments,
				};
			}
		} catch {
			// Not valid JSON
		}
		return null;
	}
}
