import { describe, expect, test } from "bun:test";
import type { CompletionConfig } from "../src/core/chat-types.js";
import {
	QWEN_NON_THINKING_DEFAULTS,
	QWEN_THINKING_DEFAULTS,
} from "../src/index.js";

describe("Sampling profile constants", () => {
	test("QWEN_THINKING_DEFAULTS matches expected values", () => {
		expect(QWEN_THINKING_DEFAULTS).toEqual({
			temperature: 0.6,
			topK: 20,
			topP: 0.95,
			repetitionPenalty: 1.05,
		});
	});

	test("QWEN_NON_THINKING_DEFAULTS matches expected values", () => {
		expect(QWEN_NON_THINKING_DEFAULTS).toEqual({
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
		});
	});

	test("constants are readonly at the type level", () => {
		// Compile-time check: `as const` produces readonly narrow types.
		// @ts-expect-error — readonly property cannot be assigned.
		QWEN_THINKING_DEFAULTS.temperature = 0.99;
		// @ts-expect-error — readonly property cannot be assigned.
		QWEN_NON_THINKING_DEFAULTS.topK = 99;
		// (Behavior already covered by `as const` literal types.)
	});
});

describe("CompletionConfig.sampling field", () => {
	test("union accepts all four mode strings; rejects unknown strings", () => {
		const a: CompletionConfig = { sampling: "auto" };
		const b: CompletionConfig = { sampling: "qwen-thinking" };
		const c: CompletionConfig = { sampling: "qwen-default" };
		const d: CompletionConfig = { sampling: "raw" };

		// @ts-expect-error — "off" is not a member of the sampling union.
		const e: CompletionConfig = { sampling: "off" };

		expect([a, b, c, d, e]).toHaveLength(5);
	});
});
