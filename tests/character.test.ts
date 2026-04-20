import { describe, expect, test } from "bun:test";
import { Character } from "../src/characters/character.js";

describe("Character", () => {
	test("construction sets config", () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "You are a test assistant.",
			maxTokens: 512,
			temperature: 0.5,
		});
		expect(char.modelId).toBe("test-model");
		expect(char.systemPrompt).toBe("You are a test assistant.");
		expect(char.config.maxTokens).toBe(512);
		expect(char.config.temperature).toBe(0.5);
		expect(char.id).toBeTruthy();
	});

	test("chat adds user message to history", async () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		const tokens: string[] = [];
		for await (const token of char.chat("Hello")) {
			tokens.push(token);
		}
		const history = char.getHistory();
		expect(
			history.some((m) => m.role === "user" && m.content === "Hello"),
		).toBe(true);
		expect(history.some((m) => m.role === "assistant")).toBe(true);
	});

	test("clearHistory keeps system prompt", () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "System prompt",
		});
		char.clearHistory();
		const history = char.getHistory();
		expect(history.length).toBe(1);
		expect(history[0].role).toBe("system");
		expect(history[0].content).toBe("System prompt");
	});

	test("messageCount tracks messages", async () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		expect(char.messageCount).toBe(1);
		for await (const _ of char.chat("Hi")) {
			/* consume */
		}
		expect(char.messageCount).toBe(3);
	});

	test("setTemperature updates config", () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		char.setTemperature(0.3);
		expect(char.config.temperature).toBe(0.3);
	});

	test("stop sets inactive", async () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		const gen = char.chat("Hello");
		char.stop();
		for await (const _ of gen) {
			break;
		}
		expect(char.isActive).toBe(false);
	});

	test("isActive reflects generation state", async () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		expect(char.isActive).toBe(false);
	});

	test("tools are registered in ToolSystem", () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
			tools: [
				{
					name: "test_tool",
					description: "A test tool",
					parameters: {},
					handler: async () => "result",
				},
			],
		});
		expect(char.config.tools.length).toBe(1);
		expect(char.config.tools[0].name).toBe("test_tool");
	});

	test("custom id is used when provided", () => {
		const char = new Character({
			id: "my-char",
			modelId: "test-model",
			systemPrompt: "Test",
		});
		expect(char.id).toBe("my-char");
	});
});
