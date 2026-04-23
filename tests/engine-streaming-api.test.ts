import { afterEach, describe, expect, test } from "bun:test";
import { WebLLM } from "../src/index.js";
import { formatChatPrompt } from "../src/inference/chat-template.js";
import { Sampler } from "../src/inference/sampler.js";
import {
	TokenAttribute,
	type TokenData,
	Tokenizer,
	type TokenizerConfig,
	TokenizerType,
} from "../src/inference/tokenizer.js";

const TOKENS: TokenData[] = [
	{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "p", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "h", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "i", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "!", score: -1, attr: TokenAttribute.NORMAL },
];

function createTokenizer(): Tokenizer {
	const config: TokenizerConfig = {
		type: TokenizerType.BPE,
		tokens: TOKENS,
		bpeRanks: new Map(),
		addedTokens: new Map(),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: TOKENS.length,
	};
	return new Tokenizer(config);
}

function createThinkingTokenizer(): Tokenizer {
	const tokens: TokenData[] = [
		{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "p", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "h", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "i", score: -1, attr: TokenAttribute.NORMAL },
	];
	const config: TokenizerConfig = {
		type: TokenizerType.BPE,
		tokens,
		bpeRanks: new Map(),
		addedTokens: new Map([
			["<think>", 3],
			["</think>", 4],
		]),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
	return new Tokenizer(config);
}

function createLogits(tokenId: number, vocabSize: number): Float32Array {
	const logits = new Float32Array(vocabSize);
	logits[tokenId] = 100;
	return logits;
}

const QWEN_TMPL = `{%- for message in messages %}{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- elif message.role == "assistant" %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}`;
const originalSample = Sampler.prototype.sample;

function createTestEngine(sequence: number[]): WebLLM {
	return createTestEngineWithTokenizer(createTokenizer(), sequence);
}

function createTestEngineWithTokenizer(
	tokenizer: Tokenizer,
	sequence: number[],
): WebLLM {
	let step = 0;
	const engine = Object.create(WebLLM.prototype) as WebLLM &
		Record<string, unknown>;
	engine.modelManager = {
		get: () => ({
			loaded: true,
			tokenizer,
		}),
	};
	engine.inferenceEngines = new Map([
		[
			"model",
			{
				forward: async () => {
					const tokenId = sequence[Math.min(step, sequence.length - 1)];
					step++;
					return createLogits(tokenId, tokenizer.vocabSize);
				},
				cachedTokenCount: 0,
				resetKVCache: () => {},
			},
		],
	]);
	engine.sessions = new Map();
	return engine;
}

describe("WebLLM.generateStream", () => {
	afterEach(() => {
		Sampler.prototype.sample = originalSample;
	});

	test("streams decoded chunks and final completion metadata", async () => {
		const engine = createTestEngine([4, 5, 2]);

		const chunks = [] as Array<{
			text: string;
			tokenId?: number;
			done: boolean;
		}>;
		let finalStats: Record<string, unknown> | undefined;
		for await (const chunk of engine.generateStream("model", "p", {
			maxTokens: 8,
			temperature: 0,
		})) {
			chunks.push({
				text: chunk.text,
				tokenId: chunk.tokenId,
				done: chunk.done,
			});
			if (chunk.done) finalStats = chunk.stats as Record<string, unknown>;
		}

		expect(chunks).toEqual([
			{ text: "h", tokenId: 4, done: false },
			{ text: "i", tokenId: 5, done: false },
			{ text: "", tokenId: undefined, done: true },
		]);
		expect(finalStats).toMatchObject({
			text: "hi",
			tokenCount: 2,
			finishReason: "eos",
		});
	});

	test("supports AbortSignal cancellation and reports aborted completion", async () => {
		const engine = createTestEngine([4, 5, 6, 6]);
		const controller = new AbortController();
		const chunks: Array<{ text: string; done: boolean }> = [];
		let finishReason: string | undefined;

		for await (const chunk of engine.generateStream("model", "p", {
			maxTokens: 8,
			temperature: 0,
			signal: controller.signal,
		})) {
			chunks.push({ text: chunk.text, done: chunk.done });
			if (!chunk.done) controller.abort();
			if (chunk.done) {
				finishReason = chunk.stats?.finishReason;
			}
		}

		expect(chunks).toEqual([
			{ text: "h", done: false },
			{ text: "", done: true },
		]);
		expect(finishReason).toBe("aborted");
	});

	test("honors stop tokens without exposing internal decode routing", async () => {
		const engine = createTestEngine([4, 6, 6]);
		const emittedTokenIds: number[] = [];
		let finalStats: Record<string, unknown> | undefined;

		for await (const chunk of engine.generateStream("model", "p", {
			maxTokens: 8,
			temperature: 0,
			stopTokenIds: [6],
		})) {
			if (!chunk.done && chunk.tokenId !== undefined) {
				emittedTokenIds.push(chunk.tokenId);
			}
			if (chunk.done) finalStats = chunk.stats as Record<string, unknown>;
		}

		expect(emittedTokenIds).toEqual([4]);
		expect(finalStats).toMatchObject({
			text: "h",
			tokenCount: 1,
			finishReason: "stop-token",
		});
	});

	test("waits for </think> before exposing visible answer text", async () => {
		const engine = createTestEngineWithTokenizer(
			createThinkingTokenizer(),
			[3, 6, 4, 7, 2],
		);

		const chunks = [] as Array<{
			text: string;
			tokenId?: number;
			done: boolean;
		}>;
		let finalStats: Record<string, unknown> | undefined;
		for await (const chunk of engine.generateStream("model", "p", {
			maxTokens: 8,
			temperature: 0,
		})) {
			chunks.push({
				text: chunk.text,
				tokenId: chunk.tokenId,
				done: chunk.done,
			});
			if (chunk.done) finalStats = chunk.stats as Record<string, unknown>;
		}

		expect(chunks).toEqual([
			{ text: "", tokenId: 3, done: false },
			{ text: "", tokenId: 6, done: false },
			{ text: "", tokenId: 4, done: false },
			{ text: "i", tokenId: 7, done: false },
			{ text: "", tokenId: undefined, done: true },
		]);
		expect(finalStats).toMatchObject({
			text: "i",
			tokenCount: 4,
			finishReason: "eos",
		});
	});

	test("chat input uses qwen chat prompt without adding BOS when disabled by tokenizer", async () => {
		let seenPrompt = "";
		let seenIds: number[] = [];
		let seenPositions: number[] = [];
		const tokenizer = {
			options: {
				chatTemplate: QWEN_TMPL,
				addBosToken: false,
			},
			bosId: 1,
			eosId: 2,
			vocabSize: 32,
			encode: (prompt: string): number[] => {
				seenPrompt = prompt;
				return [21, 22, 23];
			},
			getId: () => undefined,
			decode: () => "",
		} as unknown as Tokenizer;
		const engine = Object.create(WebLLM.prototype) as WebLLM &
			Record<string, unknown>;
		engine.modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		engine.inferenceEngines = new Map([
			[
				"model",
				{
					forward: async (ids: Int32Array, positions: Int32Array) => {
						seenIds = Array.from(ids);
						seenPositions = Array.from(positions);
						return createLogits(2, tokenizer.vocabSize);
					},
					cachedTokenCount: 0,
					resetKVCache: () => {},
				},
			],
		]);
		engine.sessions = new Map();

		for await (const _chunk of engine.generateStream(
			"model",
			[{ role: "user", content: "Hello" }],
			{ maxTokens: 8, temperature: 0 },
		)) {
			// consume stream
		}

		expect(seenPrompt).toBe(
			formatChatPrompt([{ role: "user", content: "Hello" }], QWEN_TMPL),
		);
		expect(seenIds).toEqual([21, 22, 23]);
		expect(seenPositions).toEqual([0, 1, 2]);
	});

	test("chat input can disable qwen thinking in the rendered prompt", async () => {
		let seenPrompt = "";
		const tokenizer = {
			options: {
				chatTemplate: QWEN_TMPL,
				addBosToken: false,
			},
			bosId: 1,
			eosId: 2,
			vocabSize: 32,
			encode: (prompt: string): number[] => {
				seenPrompt = prompt;
				return [21, 22, 23];
			},
			getId: () => undefined,
			decode: () => "",
		} as unknown as Tokenizer;
		const engine = Object.create(WebLLM.prototype) as WebLLM &
			Record<string, unknown>;
		engine.modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		engine.inferenceEngines = new Map([
			[
				"model",
				{
					forward: async () => createLogits(2, tokenizer.vocabSize),
					cachedTokenCount: 0,
					resetKVCache: () => {},
				},
			],
		]);
		engine.sessions = new Map();

		for await (const _chunk of engine.generateStream(
			"model",
			[{ role: "user", content: "Hello" }],
			{ maxTokens: 8, temperature: 0, enableThinking: false },
		)) {
			// consume stream
		}

		expect(seenPrompt).toBe(
			formatChatPrompt([{ role: "user", content: "Hello" }], QWEN_TMPL, {
				enableThinking: false,
			}),
		);
	});

	test("qwen chat uses sampled thinking defaults when caller leaves them unset", async () => {
		let capturedSamplerConfig:
			| {
					temperature: number;
					topK: number;
					topP: number;
					repetitionPenalty: number;
			  }
			| undefined;
		Sampler.prototype.sample = function sampleForTest(): number {
			capturedSamplerConfig = {
				temperature: this.temperature,
				topK: this.topK,
				topP: this.topP,
				repetitionPenalty: this.repetitionPenalty,
			};
			return 2;
		};

		const tokenizer = {
			options: {
				chatTemplate: QWEN_TMPL,
				addBosToken: false,
			},
			bosId: 1,
			eosId: 2,
			vocabSize: 32,
			encode: (): number[] => [21, 22, 23],
			getId: () => undefined,
			decode: () => "",
		} as unknown as Tokenizer;
		const engine = Object.create(WebLLM.prototype) as WebLLM &
			Record<string, unknown>;
		engine.modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		engine.inferenceEngines = new Map([
			[
				"model",
				{
					forward: async () => createLogits(4, tokenizer.vocabSize),
					cachedTokenCount: 0,
					resetKVCache: () => {},
				},
			],
		]);
		engine.sessions = new Map();

		for await (const _chunk of engine.generateStream("model", [
			{ role: "user", content: "Hello" },
		])) {
			// consume stream
		}

		expect(capturedSamplerConfig).toEqual({
			temperature: 0.6,
			topK: 20,
			topP: 0.95,
			repetitionPenalty: 1.05,
		});
	});

	test("qwen chat uses sampled non-thinking defaults when thinking is disabled", async () => {
		let capturedSamplerConfig:
			| {
					temperature: number;
					topK: number;
					topP: number;
					repetitionPenalty: number;
			  }
			| undefined;
		Sampler.prototype.sample = function sampleForTest(): number {
			capturedSamplerConfig = {
				temperature: this.temperature,
				topK: this.topK,
				topP: this.topP,
				repetitionPenalty: this.repetitionPenalty,
			};
			return 2;
		};

		const tokenizer = {
			options: {
				chatTemplate: QWEN_TMPL,
				addBosToken: false,
			},
			bosId: 1,
			eosId: 2,
			vocabSize: 32,
			encode: (): number[] => [21, 22, 23],
			getId: () => undefined,
			decode: () => "",
		} as unknown as Tokenizer;
		const engine = Object.create(WebLLM.prototype) as WebLLM &
			Record<string, unknown>;
		engine.modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		engine.inferenceEngines = new Map([
			[
				"model",
				{
					forward: async () => createLogits(4, tokenizer.vocabSize),
					cachedTokenCount: 0,
					resetKVCache: () => {},
				},
			],
		]);
		engine.sessions = new Map();

		for await (const _chunk of engine.generateStream(
			"model",
			[{ role: "user", content: "Hello" }],
			{ enableThinking: false },
		)) {
			// consume stream
		}

		expect(capturedSamplerConfig).toEqual({
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
		});
	});
});
