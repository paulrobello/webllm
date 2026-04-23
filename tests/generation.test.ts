import { describe, expect, test } from "bun:test";
import {
	type GenerationConfig,
	Generator,
} from "../src/inference/generation.js";
import { Sampler } from "../src/inference/sampler.js";
import {
	TokenAttribute,
	type TokenData,
	Tokenizer,
	type TokenizerConfig,
	TokenizerType,
} from "../src/inference/tokenizer.js";
import {
	InferenceSession,
	type InferenceSessionConfig,
} from "../src/models/inference-session.js";

function mockForwardPass(
	_tokenIds: number[],
	_positions: number[],
): Promise<Float32Array> {
	const vocabSize = 10;
	const logits = new Float32Array(vocabSize);
	logits[3] = 10.0; // token 3 has highest logit
	logits[2] = -100; // EOS very low
	return Promise.resolve(logits);
}

const BASE_SESSION_CONFIG: InferenceSessionConfig = {
	maxTokens: 100,
	temperature: 0,
	topK: 40,
	topP: 1,
	repetitionPenalty: 1,
	contextOverflowPolicy: "stop",
};

function createThinkingTokenizer(): Tokenizer {
	const tokens: TokenData[] = [
		{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_start|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_end|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "\n", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "answer", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "<tool_call>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</tool_call>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "<tool_response>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</tool_response>", score: 0, attr: TokenAttribute.USER_DEFINED },
	];
	const config: TokenizerConfig = {
		type: TokenizerType.BPE,
		tokens,
		bpeRanks: new Map(),
		addedTokens: new Map([
			["<|im_start|>", 3],
			["<|im_end|>", 4],
			["<think>", 5],
			["</think>", 6],
			["<tool_call>", 9],
			["</tool_call>", 10],
			["<tool_response>", 11],
			["</tool_response>", 12],
		]),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
	return new Tokenizer(config);
}

describe("Generator", () => {
	test("yields tokens from generation", async () => {
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 5,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBeGreaterThan(0);
		expect(tokens.every((t) => t === 3)).toBe(true);
	});

	test("stops on EOS token", async () => {
		async function eosForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			logits[2] = 100.0;
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			eosForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([2]);
	});

	test("stops on maxTokens", async () => {
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 10 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 3,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBe(3);
	});

	test("returns generation stats", async () => {
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 10 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 3,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		let result: Awaited<ReturnType<typeof Generator.generate>["return"]>;
		const gen = Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		);
		while (true) {
			const { value, done } = await gen.next();
			if (done) {
				result = value;
				break;
			}
		}
		expect(result).toBeDefined();
		expect(result?.tokenCount).toBe(3);
		expect(result?.tokensPerSecond).toBeGreaterThan(0);
		expect(result?.timeToFirstTokenMs).toBeGreaterThanOrEqual(0);
	});

	test("stops on custom stop tokens", async () => {
		let callCount = 0;
		async function stopTokenForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else {
				logits[7] = 100.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			stopTokens: [7],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			stopTokenForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toContain(5);
		expect(tokens).not.toContain(7);
	});

	test("stops on forbidden control-token reentry after the first token", async () => {
		let callCount = 0;
		async function qwenReentryForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else {
				logits[7] = 100.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			forbiddenReentryTokens: [7],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			qwenReentryForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5]);
	});

	test("stops on repeated <think> before </think>", async () => {
		let callCount = 0;
		async function repeatedThinkForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else {
				logits[5] = 100.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			repeatedThinkForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5]);
	});

	test("masks repeated <think> while a think block is open", async () => {
		let callCount = 0;
		async function maskedThinkForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else {
				logits[5] = 100.0;
				logits[6] = 90.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 2,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
			maskedTokensWhileThinking: [5],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			maskedThinkForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5, 6]);
	});

	test("masks chat and tool control tokens after </think> until answer text starts", async () => {
		const tokenizer = createThinkingTokenizer();
		let callCount = 0;
		async function maskedPostThinkForward(): Promise<Float32Array> {
			const logits = new Float32Array(tokenizer.vocabSize);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else if (callCount === 2) {
				logits[7] = 100.0;
			} else if (callCount === 3) {
				logits[6] = 100.0;
			} else {
				logits[5] = 100.0;
				logits[3] = 100.0;
				logits[11] = 100.0;
				logits[8] = 90.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 4,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			tokenizer,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
			maskedTokensWhileThinking: [5, 3, 4],
			maskedTokensAfterThinkingUntilAnswer: [5, 3, 4, 9, 10, 11, 12],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			maskedPostThinkForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5, 7, 6, 8]);
	});

	test("requires visible answer text after </think> before allowing eos", async () => {
		const tokenizer = createThinkingTokenizer();
		let callCount = 0;
		async function postThinkEosForward(): Promise<Float32Array> {
			const logits = new Float32Array(tokenizer.vocabSize);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else if (callCount === 2) {
				logits[7] = 100.0;
			} else if (callCount === 3) {
				logits[6] = 100.0;
			} else {
				logits[2] = 100.0;
				logits[8] = 90.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 4,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			tokenizer,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
			maskedTokensWhileThinking: [5, 3, 4],
			maskedTokensAfterThinkingUntilAnswer: [5, 3, 4, 9, 10, 11, 12],
			requireVisibleAnswerAfterThinking: true,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			postThinkEosForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5, 7, 6, 8]);
	});

	test("suppresses whitespace-only tokens after </think> until answer text starts", async () => {
		const tokenizer = createThinkingTokenizer();
		let callCount = 0;
		async function postThinkWhitespaceForward(): Promise<Float32Array> {
			const logits = new Float32Array(tokenizer.vocabSize);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else if (callCount === 2) {
				logits[7] = 100.0;
			} else if (callCount === 3) {
				logits[6] = 100.0;
			} else {
				logits[7] = 100.0;
				logits[8] = 90.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 4,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			tokenizer,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
			maskedTokensWhileThinking: [5, 3, 4],
			maskedTokensAfterThinkingUntilAnswer: [5, 3, 4, 9, 10, 11, 12],
			requireVisibleAnswerAfterThinking: true,
			suppressWhitespaceOnlyAfterThinking: true,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			postThinkWhitespaceForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5, 7, 6, 8]);
	});

	test("suppresses whitespace-only first token until visible answer text starts", async () => {
		const tokenizer = createThinkingTokenizer();
		let callCount = 0;
		async function whitespaceFirstForward(): Promise<Float32Array> {
			const logits = new Float32Array(tokenizer.vocabSize);
			callCount++;
			if (callCount === 1) {
				logits[7] = 100.0;
				logits[8] = 90.0;
			} else {
				logits[2] = 100.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 4,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			tokenizer,
			requireVisibleAnswerBeforeStop: true,
			suppressWhitespaceOnlyUntilAnswer: true,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			whitespaceFirstForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([8]);
	});

	test("masks control-token relapse after visible answer text starts", async () => {
		const tokenizer = createThinkingTokenizer();
		let callCount = 0;
		async function postAnswerControlRelapseForward(): Promise<Float32Array> {
			const logits = new Float32Array(tokenizer.vocabSize);
			callCount++;
			if (callCount === 1) {
				logits[5] = 100.0;
			} else if (callCount === 2) {
				logits[7] = 100.0;
			} else if (callCount === 3) {
				logits[6] = 100.0;
			} else if (callCount === 4) {
				logits[8] = 100.0;
			} else {
				logits[5] = 100.0;
				logits[3] = 100.0;
				logits[8] = 90.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 5,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			tokenizer,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
			maskedTokensWhileThinking: [5, 3, 4],
			maskedTokensAfterThinkingUntilAnswer: [5, 3, 4, 9, 10, 11, 12],
			requireVisibleAnswerAfterThinking: true,
			suppressWhitespaceOnlyAfterThinking: true,
			maskedTokensAfterAnswerStarts: [5, 3, 4, 9, 10, 11, 12],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			postAnswerControlRelapseForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([5, 7, 6, 8, 8]);
	});

	test("stops on stray </think> without an open block", async () => {
		async function strayThinkCloseForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			logits[6] = 100.0;
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			enforceSingleThinkBlock: true,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			strayThinkCloseForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([]);
	});

	test("stops when AbortSignal is already aborted", async () => {
		const controller = new AbortController();
		controller.abort();
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 10,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
			controller.signal,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBeLessThanOrEqual(1);
	});

	test("stops when AbortSignal fires during generation", async () => {
		const controller = new AbortController();
		let callCount = 0;
		async function delayedForward(): Promise<Float32Array> {
			callCount++;
			if (callCount >= 3) controller.abort();
			const logits = new Float32Array(10);
			logits[3] = 10.0;
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 50 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 50,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			delayedForward,
			config,
			controller.signal,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBeLessThan(50);
	});

	test("applies repetition penalty to recent tokens", async () => {
		async function trackForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			logits[3] = 10.0;
			return logits;
		}
		const sampler = new Sampler({ temperature: 0, repetitionPenalty: 2.0 });
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 10 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 5,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 2.0,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			trackForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBe(5);
	});
});
