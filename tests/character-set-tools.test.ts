import { describe, expect, test } from "bun:test";
import { Character } from "../src/characters/character.js";
import type { ToolDefinition } from "../src/characters/tool-system.js";

const toolA: ToolDefinition = {
	name: "tool_a",
	description: "first tool",
	parameters: { input: { type: "string", description: "x" } },
	handler: async () => "result-a",
};

const toolB: ToolDefinition = {
	name: "tool_b",
	description: "second tool",
	parameters: { input: { type: "string", description: "y" } },
	handler: async () => "result-b",
};

describe("Character.setTools", () => {
	test("replaces tools list and recreates ToolSystem", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
			tools: [toolA],
		});
		expect(ch.config.tools).toEqual([toolA]);

		ch.setTools([toolB]);
		expect(ch.config.tools).toEqual([toolB]);
		// Access the private field via cast for the assertion. The
		// behavior assertion is "non-null toolSystem after setTools",
		// not a cross-module API contract.
		const toolSystem = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(toolSystem).not.toBeNull();
	});

	test("clears tools when called with empty array", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
			tools: [toolA],
		});
		ch.setTools([]);
		expect(ch.config.tools).toEqual([]);
		const toolSystem = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(toolSystem).toBeNull();
	});

	test("starts from no tools and adds them via setTools", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
		});
		const initial = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(initial).toBeNull();

		ch.setTools([toolA, toolB]);
		expect(ch.config.tools).toEqual([toolA, toolB]);
		const after = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(after).not.toBeNull();
	});
});
