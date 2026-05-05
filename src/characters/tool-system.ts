import type { JsonSchemaParameterType } from "../core/chat-types.js";

/** Schema for a single tool parameter. */
export interface ToolParameter {
	/** JSON Schema type of the parameter. Sourced from the canonical union. */
	type: JsonSchemaParameterType;
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

// Qwen3 / Hermes / Llama-3.x common convention:
//   <tool_call>
//   {"name": "...", "arguments": {...}}
//   </tool_call>
// The JSON body may span newlines and nest braces, so we match the
// balanced content between the tags non-greedily with the `s` flag to
// let `.` cross line breaks.
const XML_TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/;
// Sub-8B Llama-3 family (notably Llama-3.2-3B) interpret the
// `<tool_call>` wrapper as XML and lift `name` / `arguments` into
// child elements rather than emitting JSON inside the wrapper:
//   <tool_call>
//     <name>get_weather</name>
//     <arguments>{"city": "Tokyo"}</arguments>
//   </tool_call>
// JSON-inside-XML is mechanically parseable; pure-XML arguments
// (`<arguments><to>John</to>...</arguments>`) fall through the JSON
// parse and return null gracefully.
const XML_NESTED_RE =
	/<tool_call>\s*<name>\s*([^<]+?)\s*<\/name>\s*<arguments>\s*([\s\S]*?)\s*<\/arguments>\s*<\/tool_call>/;
// Legacy one-liner form we used to produce internally: `<tool_call={...}>`.
// Keep parsing for back-compat with any stored outputs.
const XML_LEGACY_TOOL_CALL_RE = /<tool_call=(\{.*?\})>/;
const JSON_OBJECT_RE =
	/\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}/;

/**
 * Registers, parses, and executes tool/function calls from model output.
 *
 * Supports four formats:
 *   1. `<tool_call>\n{"name":...,"arguments":...}\n</tool_call>` — Qwen3/Hermes style.
 *   2. `<tool_call><name>X</name><arguments>{...}</arguments></tool_call>` —
 *      sub-8B Llama-3 family malformation (JSON-inside-XML).
 *   3. `<tool_call={...}>` — legacy internal form (kept for back-compat).
 *   4. A bare JSON object with "name" and "arguments" fields.
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
			const parsed = this.parseJson(xmlMatch[1].trim());
			if (parsed) return parsed;
		}

		const nestedMatch = text.match(XML_NESTED_RE);
		if (nestedMatch?.[1] && nestedMatch[2] !== undefined) {
			const name = nestedMatch[1].trim();
			const argsBody = nestedMatch[2].trim();
			try {
				const args: unknown = JSON.parse(argsBody);
				if (typeof args === "object" && args !== null && !Array.isArray(args)) {
					return { name, arguments: args as Record<string, unknown> };
				}
			} catch {
				// arguments body is not JSON (e.g. pure-XML child elements);
				// fall through to legacy/JSON branches and ultimately null.
			}
		}

		const legacyMatch = text.match(XML_LEGACY_TOOL_CALL_RE);
		if (legacyMatch?.[1]) {
			const parsed = this.parseJson(legacyMatch[1]);
			if (parsed) return parsed;
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
