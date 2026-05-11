import { describe, expect, test } from "bun:test";
import {
	GEMMA4_DEFAULTS,
	MISTRAL_DEFAULTS,
	PHI3_DEFAULTS,
	QWEN_NON_THINKING_DEFAULTS,
	QWEN_THINKING_DEFAULTS,
	resolveSamplingParams,
} from "../src/core/sampling-profiles.js";

const ENGINE_FALLBACK = {
	temperature: 1.0,
	topK: 0,
	topP: 1.0,
	repetitionPenalty: 1.0,
};

describe("resolveSamplingParams — auto mode", () => {
	test("Qwen+ChatML, thinking=undefined → THINKING profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: true,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});

	test("Qwen+ChatML, thinking=true → THINKING profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: true,
				enableThinking: true,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});

	test("Qwen+ChatML, thinking=false → NON_THINKING profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: true,
				enableThinking: false,
				consumer: {},
			}),
		).toEqual({ ...QWEN_NON_THINKING_DEFAULTS });
	});

	test("non-Qwen, non-Phi3 → engine fallback (no profile applied)", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: false,
				consumer: {},
			}),
		).toEqual(ENGINE_FALLBACK);
	});

	test("Phi3 → PHI3 profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: false,
				isPhi3: true,
				consumer: {},
			}),
		).toEqual({ ...PHI3_DEFAULTS });
	});

	test("Qwen takes precedence over Phi3 when both signal true", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: true,
				isPhi3: true,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});

	test("Mistral → MISTRAL profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: false,
				isMistral: true,
				consumer: {},
			}),
		).toEqual({ ...MISTRAL_DEFAULTS });
	});

	test("Qwen takes precedence over Mistral when both signal true", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: true,
				isMistral: true,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});

	test("Phi3 takes precedence over Mistral when both signal true", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: false,
				isPhi3: true,
				isMistral: true,
				consumer: {},
			}),
		).toEqual({ ...PHI3_DEFAULTS });
	});

	test("non-Qwen + thinking=false still falls back (auto only fires on Qwen+ChatML)", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "auto",
				isQwenChatml: false,
				enableThinking: false,
				consumer: {},
			}),
		).toEqual(ENGINE_FALLBACK);
	});
});

describe("resolveSamplingParams — forced profile modes", () => {
	test("qwen-thinking on non-Qwen still applies THINKING profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "qwen-thinking",
				isQwenChatml: false,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});

	test("qwen-default on non-Qwen still applies NON_THINKING profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "qwen-default",
				isQwenChatml: false,
				consumer: {},
			}),
		).toEqual({ ...QWEN_NON_THINKING_DEFAULTS });
	});

	test("qwen-thinking ignores enableThinking=false (forced wins over auto-flip)", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "qwen-thinking",
				isQwenChatml: true,
				enableThinking: false,
				consumer: {},
			}),
		).toEqual({ ...QWEN_THINKING_DEFAULTS });
	});
});

describe("resolveSamplingParams — raw mode", () => {
	test("raw on Qwen+ChatML skips auto profile", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "raw",
				isQwenChatml: true,
				enableThinking: true,
				consumer: {},
			}),
		).toEqual(ENGINE_FALLBACK);
	});

	test("raw passes consumer values through unchanged", () => {
		expect(
			resolveSamplingParams({
				samplingMode: "raw",
				isQwenChatml: true,
				consumer: {
					temperature: 0.42,
					topK: 7,
					topP: 0.85,
					repetitionPenalty: 1.2,
				},
			}),
		).toEqual({
			temperature: 0.42,
			topK: 7,
			topP: 0.85,
			repetitionPenalty: 1.2,
		});
	});
});

describe("Gemma 4 auto dispatch", () => {
	test("returns GEMMA4_DEFAULTS when isGemma4 is true and mode is auto", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: false,
			isPhi3: false,
			isGemma4: true,
			consumer: {},
		});
		expect(out.temperature).toBe(1.0);
		expect(out.topP).toBe(0.95);
		expect(out.topK).toBe(64);
		expect(out.repetitionPenalty).toBe(1.0);
	});

	test("explicit mode 'gemma4' selects the profile regardless of arch flags", () => {
		const out = resolveSamplingParams({
			samplingMode: "gemma4",
			isQwenChatml: true,
			isPhi3: false,
			isGemma4: false,
			consumer: {},
		});
		expect(out.temperature).toBe(1.0);
	});

	test("consumer override beats profile defaults", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: false,
			isGemma4: true,
			consumer: { temperature: 0 },
		});
		expect(out.temperature).toBe(0);
		expect(out.topP).toBe(GEMMA4_DEFAULTS.topP); // unchanged
	});
});

describe("resolveSamplingParams — consumer override precedence", () => {
	test("consumer temperature wins over auto Qwen profile", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: true,
			consumer: { temperature: 0.99 },
		});
		expect(out.temperature).toBe(0.99);
		expect(out.topK).toBe(QWEN_THINKING_DEFAULTS.topK);
		expect(out.topP).toBe(QWEN_THINKING_DEFAULTS.topP);
		expect(out.repetitionPenalty).toBe(
			QWEN_THINKING_DEFAULTS.repetitionPenalty,
		);
	});

	test("consumer wins over forced qwen-default profile per-field", () => {
		const out = resolveSamplingParams({
			samplingMode: "qwen-default",
			isQwenChatml: false,
			consumer: { topK: 50, repetitionPenalty: 1.3 },
		});
		expect(out.temperature).toBe(QWEN_NON_THINKING_DEFAULTS.temperature);
		expect(out.topK).toBe(50);
		expect(out.topP).toBe(QWEN_NON_THINKING_DEFAULTS.topP);
		expect(out.repetitionPenalty).toBe(1.3);
	});

	test("consumer 0 is honored as a real override (not coerced to fallback)", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: true,
			consumer: { temperature: 0, topK: 0 },
		});
		expect(out.temperature).toBe(0);
		expect(out.topK).toBe(0);
	});
});
