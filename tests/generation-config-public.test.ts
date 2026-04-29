import { describe, test } from "bun:test";
import type { GenerationConfig } from "../src/index.js";

describe("Public GenerationConfig surface", () => {
	test("rejects internal steering fields at the type level", () => {
		// These assignments must NOT compile. If any does, the public type
		// is leaking internal steering fields and the split has regressed.

		// @ts-expect-error — thinkingOpenTokenId is internal-only
		const _a: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 42,
		};

		// @ts-expect-error — maskedTokensWhileThinking is internal-only
		const _b: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			maskedTokensWhileThinking: [1, 2, 3],
		};

		// @ts-expect-error — requireVisibleAnswerAfterThinking is internal-only
		const _c: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			requireVisibleAnswerAfterThinking: true,
		};

		// @ts-expect-error — prompt was dropped entirely
		const _d: GenerationConfig = {
			prompt: "hi",
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
		};

		// Sanity: a fully-populated public config must compile.
		const _ok: GenerationConfig = {
			maxTokens: 10,
			temperature: 0.7,
			topK: 40,
			topP: 0.9,
			repetitionPenalty: 1.05,
			stopTokens: [2],
			signal: new AbortController().signal,
		};

		void _a;
		void _b;
		void _c;
		void _d;
		void _ok;
	});
});
