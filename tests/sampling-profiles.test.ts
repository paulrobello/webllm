import { describe, expect, test } from "bun:test";
import type { CompletionConfig } from "../src/core/chat-types.js";
import {
	PHI3_DEFAULTS,
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

	test("PHI3_DEFAULTS matches the upstream Phi-3 model card recommendation", () => {
		expect(PHI3_DEFAULTS).toEqual({
			temperature: 0.7,
			topK: 50,
			topP: 0.9,
			repetitionPenalty: 1.1,
		});
	});

	test("constants are frozen at runtime (assignment throws in strict mode)", () => {
		expect(() => {
			// @ts-expect-error — readonly property cannot be assigned.
			QWEN_THINKING_DEFAULTS.temperature = 0.99;
		}).toThrow(TypeError);
		expect(() => {
			// @ts-expect-error — readonly property cannot be assigned.
			QWEN_NON_THINKING_DEFAULTS.topK = 99;
		}).toThrow(TypeError);

		expect(QWEN_THINKING_DEFAULTS.temperature).toBe(0.6);
		expect(QWEN_NON_THINKING_DEFAULTS.topK).toBe(20);
	});
});

describe("CompletionConfig.sampling field", () => {
	test("union accepts all five mode strings; rejects unknown strings", () => {
		const a: CompletionConfig = { sampling: "auto" };
		const b: CompletionConfig = { sampling: "qwen-thinking" };
		const c: CompletionConfig = { sampling: "qwen-default" };
		const d: CompletionConfig = { sampling: "raw" };
		const e: CompletionConfig = { sampling: "phi3" };

		// @ts-expect-error — "off" is not a member of the sampling union.
		const f: CompletionConfig = { sampling: "off" };

		expect([a, b, c, d, e, f]).toHaveLength(6);
	});
});
