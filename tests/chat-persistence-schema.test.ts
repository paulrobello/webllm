import { expect, test } from "bun:test";
import {
	buildMetadata,
	CHAT_META_KEY,
	isCompatibleMeta,
} from "../smoke-test/chat-persistence.js";

test("buildMetadata captures modelId, systemPrompt, settings, messages, savedAt", () => {
	const meta = buildMetadata({
		modelId: "tinyllama-1.1b-chat-q4_0",
		systemPrompt: "You answer in one sentence.",
		settings: { temperature: 0.7, topK: 0 },
		messages: [
			{ role: "user", content: "hi" },
			{ role: "assistant", content: "hello" },
		],
	});
	expect(meta.modelId).toBe("tinyllama-1.1b-chat-q4_0");
	expect(meta.systemPrompt).toBe("You answer in one sentence.");
	expect(meta.settings.temperature).toBe(0.7);
	expect(meta.messages.length).toBe(2);
	expect(typeof meta.savedAtMs).toBe("number");
	expect(meta.savedAtMs).toBeGreaterThan(0);
	expect(meta.schemaVersion).toBe(1);
});

test("isCompatibleMeta accepts schemaVersion 1 with a known modelId", () => {
	const meta = buildMetadata({
		modelId: "tinyllama-1.1b-chat-q4_0",
		systemPrompt: "",
		settings: {},
		messages: [],
	});
	const known = new Set(["tinyllama-1.1b-chat-q4_0"]);
	expect(isCompatibleMeta(meta, known)).toBe(true);
});

test("isCompatibleMeta rejects unknown modelId", () => {
	const meta = buildMetadata({
		modelId: "ghost-model",
		systemPrompt: "",
		settings: {},
		messages: [],
	});
	expect(isCompatibleMeta(meta, new Set())).toBe(false);
});

test("isCompatibleMeta rejects unknown schemaVersion", () => {
	const meta = {
		...buildMetadata({
			modelId: "x",
			systemPrompt: "",
			settings: {},
			messages: [],
		}),
		schemaVersion: 999,
	};
	// biome-ignore lint/suspicious/noExplicitAny: test intentionally constructs an invalid schemaVersion
	expect(isCompatibleMeta(meta as any, new Set(["x"]))).toBe(false);
});

test("CHAT_META_KEY is a non-empty string constant", () => {
	expect(typeof CHAT_META_KEY).toBe("string");
	expect(CHAT_META_KEY.length).toBeGreaterThan(0);
});
