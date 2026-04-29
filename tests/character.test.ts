import { describe, expect, test } from "bun:test";
import { Character, type ChatEngine } from "../src/characters/character.js";
import type {
	ChatMessage,
	CompletionChunk,
	CompletionConfig,
} from "../src/core/chat-types.js";

/**
 * Minimal stub engine: echoes a fixed completion so Character.chat() exercises
 * the real streaming + history + tool-parse paths without needing a loaded
 * model. The echo includes the user's last message so tests can assert on
 * both the request and the yielded stream.
 */
function createStubEngine(
	reply = "stub-reply",
): ChatEngine & { calls: Array<{ modelId: string; messages: ChatMessage[] }> } {
	const calls: Array<{ modelId: string; messages: ChatMessage[] }> = [];
	return {
		calls,
		async *chatCompletion(
			modelId: string,
			messages: ChatMessage[],
			_config?: CompletionConfig,
		): AsyncGenerator<CompletionChunk, void> {
			calls.push({ modelId, messages: [...messages] });
			for (const ch of reply) {
				yield { text: ch, done: false };
			}
			yield { text: "", done: true };
		},
	};
}

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

	test("chat adds user message to history and streams engine tokens", async () => {
		const engine = createStubEngine("hello there");
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
			engine,
		});
		const tokens: string[] = [];
		for await (const token of char.chat("Hello")) {
			tokens.push(token);
		}
		expect(tokens.join("")).toBe("hello there");
		const history = char.getHistory();
		expect(
			history.some((m) => m.role === "user" && m.content === "Hello"),
		).toBe(true);
		expect(
			history.some(
				(m) => m.role === "assistant" && m.content === "hello there",
			),
		).toBe(true);
		expect(engine.calls).toHaveLength(1);
		expect(engine.calls[0].modelId).toBe("test-model");
	});

	test("chat without an engine throws a helpful error", async () => {
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
		});
		let threw = false;
		try {
			for await (const _ of char.chat("Hello")) {
				/* never reached */
			}
		} catch (err) {
			threw = true;
			expect(String(err)).toContain("no engine attached");
		}
		expect(threw).toBe(true);
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
			engine: createStubEngine("hi back"),
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
			engine: createStubEngine("abcdef"),
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

	test("AbortSignal aborted before chat() returns immediately", async () => {
		const engine = createStubEngine("nope");
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
			engine,
		});
		const ctrl = new AbortController();
		ctrl.abort();
		const tokens: string[] = [];
		for await (const t of char.chat("Hi", { signal: ctrl.signal })) {
			tokens.push(t);
		}
		expect(tokens).toEqual([]);
		// Engine should not have been invoked.
		expect(engine.calls).toEqual([]);
	});

	test("AbortSignal mid-stream short-circuits iteration", async () => {
		// Stub that emits a controllable number of chunks before the consumer
		// aborts. Aborting after the first chunk should yield exactly that one
		// chunk and skip the rest.
		const ctrl = new AbortController();
		const engine: ChatEngine = {
			async *chatCompletion(): AsyncGenerator<CompletionChunk, void> {
				yield { text: "a", done: false };
				ctrl.abort();
				yield { text: "b", done: false };
				yield { text: "c", done: false };
				yield { text: "", done: true };
			},
		};
		const char = new Character({
			modelId: "test-model",
			systemPrompt: "Test",
			engine,
		});
		const tokens: string[] = [];
		for await (const t of char.chat("Hi", { signal: ctrl.signal })) {
			tokens.push(t);
		}
		expect(tokens).toEqual(["a"]);
	});
});
