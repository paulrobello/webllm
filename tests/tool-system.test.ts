import { describe, expect, test } from "bun:test";
import {
	type ToolCall,
	type ToolDefinition,
	type ToolResult,
	ToolSystem,
} from "../src/characters/tool-system.js";

function makeTool(
	name: string,
	handler?: (args: Record<string, unknown>) => Promise<unknown>,
): ToolDefinition {
	return {
		name,
		description: `Description for ${name}`,
		parameters: {
			item: {
				type: "string",
				description: "The item to check",
				required: true,
			},
			quantity: {
				type: "number",
				description: "How many to check",
			},
		},
		handler:
			handler ??
			(async (args: Record<string, unknown>) => ({
				available: true,
				item: args.item,
			})),
	};
}

describe("ToolSystem", () => {
	test("parseToolCall extracts standalone JSON tool call", () => {
		const system = new ToolSystem([]);
		const result = system.parseToolCall(
			'{"name": "check_stock", "arguments": {"item": "sword"}}',
		);
		expect(result).not.toBeNull();
		expect(result?.name).toBe("check_stock");
		expect(result?.arguments).toEqual({ item: "sword" });
	});

	test("parseToolCall extracts legacy <tool_call={...}> form", () => {
		const system = new ToolSystem([]);
		const result = system.parseToolCall(
			'<tool_call={"name": "check_stock", "arguments": {"item": "sword"}}>',
		);
		expect(result).not.toBeNull();
		expect(result?.name).toBe("check_stock");
		expect(result?.arguments).toEqual({ item: "sword" });
	});

	test("parseToolCall extracts Qwen3/Hermes <tool_call>...</tool_call> form", () => {
		const system = new ToolSystem([]);
		const result = system.parseToolCall(
			'I will help.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>',
		);
		expect(result).not.toBeNull();
		expect(result?.name).toBe("get_weather");
		expect(result?.arguments).toEqual({ city: "Tokyo" });
	});

	test("parseToolCall handles nested arguments across newlines", () => {
		const system = new ToolSystem([]);
		const result = system.parseToolCall(
			'<tool_call>\n{\n  "name": "search",\n  "arguments": {\n    "query": "pizza",\n    "filters": {"min_stars": 4}\n  }\n}\n</tool_call>',
		);
		expect(result).not.toBeNull();
		expect(result?.name).toBe("search");
		expect(result?.arguments).toEqual({
			query: "pizza",
			filters: { min_stars: 4 },
		});
	});

	test("parseToolCall returns null for non-tool text", () => {
		const system = new ToolSystem([]);
		expect(system.parseToolCall("Hello, how are you?")).toBeNull();
		expect(system.parseToolCall("")).toBeNull();
		expect(system.parseToolCall("Just regular text with no JSON")).toBeNull();
	});

	test("parseToolCall returns null for JSON without required fields", () => {
		const system = new ToolSystem([]);
		expect(system.parseToolCall('{"name": "check_stock"}')).toBeNull();
		expect(system.parseToolCall('{"arguments": {"item": "sword"}}')).toBeNull();
		expect(system.parseToolCall('{"other": "data"}')).toBeNull();
	});

	test("parseToolCall prefers XML-tagged format over standalone JSON", () => {
		const system = new ToolSystem([]);
		const text =
			'<tool_call={"name": "xml_tool", "arguments": {"a": 1}}> some text {"name": "json_tool", "arguments": {"b": 2}}';
		const result = system.parseToolCall(text);
		expect(result).not.toBeNull();
		expect(result?.name).toBe("xml_tool");
		expect(result?.arguments).toEqual({ a: 1 });
	});

	test("execute calls handler with arguments", async () => {
		const receivedArgs: Record<string, unknown>[] = [];
		const tool = makeTool("check_stock", async (args) => {
			receivedArgs.push(args);
			return { available: true };
		});
		const system = new ToolSystem([tool]);
		const call: ToolCall = {
			name: "check_stock",
			arguments: { item: "sword", quantity: 5 },
		};
		const result = await system.execute(call);
		expect(receivedArgs).toEqual([{ item: "sword", quantity: 5 }]);
		expect(result.result).toEqual({ available: true });
		expect(result.error).toBeUndefined();
		expect(result.call).toBe(call);
	});

	test("execute returns error for unknown tool", async () => {
		const system = new ToolSystem([makeTool("known_tool")]);
		const call: ToolCall = {
			name: "unknown_tool",
			arguments: { item: "sword" },
		};
		const result = await system.execute(call);
		expect(result.error).toBeDefined();
		expect(result.result).toBeUndefined();
		expect(result.call.name).toBe("unknown_tool");
	});

	test("execute catches handler errors", async () => {
		const tool = makeTool("failing_tool", async () => {
			throw new Error("Handler exploded");
		});
		const system = new ToolSystem([tool]);
		const call: ToolCall = {
			name: "failing_tool",
			arguments: {},
		};
		const result = await system.execute(call);
		expect(result.error).toBe("Handler exploded");
		expect(result.result).toBeUndefined();
	});

	test("formatForPrompt generates tool documentation", () => {
		const tool: ToolDefinition = {
			name: "check_inventory",
			description: "Check if an item is in stock",
			parameters: {
				item: {
					type: "string",
					description: "The item to check",
					required: true,
				},
				quantity: {
					type: "number",
					description: "How many to check",
				},
			},
			handler: async () => null,
		};
		const system = new ToolSystem([tool]);
		const formatted = system.formatForPrompt();
		expect(formatted).toContain("check_inventory");
		expect(formatted).toContain("Check if an item is in stock");
		expect(formatted).toContain("item (string, required)");
		expect(formatted).toContain("The item to check");
		expect(formatted).toContain("quantity (number)");
		expect(formatted).toContain("How many to check");
	});

	test("formatForPrompt returns empty string when no tools", () => {
		const system = new ToolSystem([]);
		expect(system.formatForPrompt()).toBe("");
	});

	test("formatForPrompt includes enum values", () => {
		const tool: ToolDefinition = {
			name: "sort_items",
			description: "Sort items by field",
			parameters: {
				field: {
					type: "string",
					required: true,
					enum: ["name", "price", "date"],
				},
			},
			handler: async () => null,
		};
		const system = new ToolSystem([tool]);
		const formatted = system.formatForPrompt();
		expect(formatted).toContain("enum: name, price, date");
	});

	test("formatResult formats successful result", () => {
		const system = new ToolSystem([]);
		const call: ToolCall = {
			name: "check_stock",
			arguments: { item: "sword" },
		};
		const result: ToolResult = {
			call,
			result: { available: true, count: 5 },
		};
		const formatted = system.formatResult(result);
		expect(formatted).toBe(
			'<tool_result name="check_stock">{"available":true,"count":5}</tool_result>',
		);
	});

	test("formatResult formats error result", () => {
		const system = new ToolSystem([]);
		const call: ToolCall = {
			name: "check_stock",
			arguments: { item: "sword" },
		};
		const result: ToolResult = {
			call,
			result: undefined,
			error: "Item not found",
		};
		const formatted = system.formatResult(result);
		expect(formatted).toBe(
			'<tool_error name="check_stock">Item not found</tool_error>',
		);
	});

	test("has checks tool registration", () => {
		const system = new ToolSystem([makeTool("tool_a"), makeTool("tool_b")]);
		expect(system.has("tool_a")).toBe(true);
		expect(system.has("tool_b")).toBe(true);
		expect(system.has("tool_c")).toBe(false);
	});

	test("size returns tool count", () => {
		const system = new ToolSystem([
			makeTool("a"),
			makeTool("b"),
			makeTool("c"),
		]);
		expect(system.size).toBe(3);
	});

	test("size is zero for empty system", () => {
		const system = new ToolSystem([]);
		expect(system.size).toBe(0);
	});

	test("getDefinitions returns all tool definitions", () => {
		const toolA = makeTool("tool_a");
		const toolB = makeTool("tool_b");
		const system = new ToolSystem([toolA, toolB]);
		const defs = system.getDefinitions();
		expect(defs).toHaveLength(2);
		expect(defs.map((d) => d.name).sort()).toEqual(["tool_a", "tool_b"]);
	});
});
