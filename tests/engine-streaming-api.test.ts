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

function createQwenChatTokenizer(): Tokenizer {
	const tokens: TokenData[] = [
		{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_start|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_end|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: " Hello", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "!", score: -1, attr: TokenAttribute.NORMAL },
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
		]),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
		addBosToken: false,
		chatTemplate: QWEN_TMPL,
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

// Tests intentionally inject minimal internal state into a WebLLM
// instance. This shape narrows the unsafe cast to a single boundary so
// the rest of each test stays type-safe. Post-ARC-004 the per-model state
// lives in one `models: Map<string, ModelRecord>` row; tests populate the
// `inference` field of that record. `unknown` values keep us from
// reimplementing the full ModelInference / ModelEntry surface in fixtures.
type EngineInternals = {
	_modelManager: { get(id: string): unknown };
	models: Map<string, unknown>;
};

function asInternals(engine: WebLLM): EngineInternals {
	return engine as unknown as EngineInternals;
}

function createTestEngine(sequence: number[]): WebLLM {
	return createTestEngineWithTokenizer(createTokenizer(), sequence);
}

function createTestEngineWithTokenizer(
	tokenizer: Tokenizer,
	sequence: number[],
	architecture = "llama",
): WebLLM {
	let step = 0;
	const engine = Object.create(WebLLM.prototype) as WebLLM;
	const internals = asInternals(engine);
	internals._modelManager = {
		get: () => ({
			loaded: true,
			tokenizer,
			hyperparams: {
				architecture,
			},
		}),
	};
	internals.models = new Map<string, unknown>([
		[
			"model",
			{
				inference: {
					forward: async () => {
						const tokenId = sequence[Math.min(step, sequence.length - 1)];
						step++;
						return createLogits(tokenId, tokenizer.vocabSize);
					},
					cachedTokenCount: 0,
					resetKVCache: () => {},
				},
			},
		],
	]);
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
			tokenId?: number | undefined;
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
			if (chunk.done)
				finalStats = chunk.stats as unknown as Record<string, unknown>;
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
			if (chunk.done)
				finalStats = chunk.stats as unknown as Record<string, unknown>;
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
			tokenId?: number | undefined;
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
			if (chunk.done)
				finalStats = chunk.stats as unknown as Record<string, unknown>;
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
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			[
				"model",
				{
					inference: {
						forward: async (ids: Int32Array, positions: Int32Array) => {
							seenIds = Array.from(ids);
							seenPositions = Array.from(positions);
							return createLogits(2, tokenizer.vocabSize);
						},
						cachedTokenCount: 0,
						resetKVCache: () => {},
					},
				},
			],
		]);

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
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			[
				"model",
				{
					inference: {
						forward: async () => createLogits(2, tokenizer.vocabSize),
						cachedTokenCount: 0,
						resetKVCache: () => {},
					},
				},
			],
		]);

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
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			[
				"model",
				{
					inference: {
						forward: async () => createLogits(4, tokenizer.vocabSize),
						cachedTokenCount: 0,
						resetKVCache: () => {},
					},
				},
			],
		]);

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
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			[
				"model",
				{
					inference: {
						forward: async () => createLogits(4, tokenizer.vocabSize),
						cachedTokenCount: 0,
						resetKVCache: () => {},
					},
				},
			],
		]);

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

	test("CompletionConfig.seed propagates to the sampler PRNG", async () => {
		let usedSeededRng = false;
		Sampler.prototype.sample = function sampleForTest(this: Sampler): number {
			usedSeededRng = this.rng !== Math.random;
			return 2;
		};

		const tokenizer = {
			options: { chatTemplate: QWEN_TMPL, addBosToken: false },
			bosId: 1,
			eosId: 2,
			vocabSize: 32,
			encode: (): number[] => [21, 22, 23],
			getId: () => undefined,
			decode: () => "",
		} as unknown as Tokenizer;
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			[
				"model",
				{
					inference: {
						forward: async () => createLogits(2, tokenizer.vocabSize),
						cachedTokenCount: 0,
						resetKVCache: () => {},
					},
				},
			],
		]);

		for await (const _chunk of engine.generateStream(
			"model",
			[{ role: "user", content: "Hello" }],
			{ seed: 12345 },
		)) {
			// consume stream
		}

		expect(usedSeededRng).toBe(true);
	});

	test("session-tracker delta path is skipped when leading messages diverge", async () => {
		// Regression for the bug surfaced by
		// `eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md`:
		// the delta-encoding fast-path at `engine.ts:prepareChatPrompt`
		// trusted `promptMessages.length > prevMsgCount` as a continuation
		// signal without verifying the cached prompt's leading messages
		// match. In interleaved cross-conversation use, the model would
		// silently reason against the previous conversation's KV with the
		// new tail appended.
		//
		// Setup: drive three calls on the modelId path —
		//   1. system_a + user_a   (cold start, full prefill, length=2)
		//   2. system_b + user_b   (different leading system, length=2 — should reset)
		//   3. system_a + user_a + assistant_a + user_a2  (extends call 1's
		//      prompt to length=4, but cached state is from call 2 — must
		//      reset, NOT take the delta path)
		//
		// We instrument `forward` to record (ids, positions) and assert that
		// call 3's first forward starts at position 0 (full reset). Before
		// the fix, it incorrectly started past position 0 because the engine
		// trusted the message-count growth.
		//
		// The default `createTestEngineWithTokenizer` fake doesn't track
		// `cachedTokenCount`, which would short-circuit the buggy condition
		// (`session.currentPosition === inf.cachedTokenCount` becomes false
		// because cachedTokenCount stays at 0). Build a realistic fake here
		// that mirrors `ModelInference`'s positional bookkeeping so the bug
		// can manifest.
		const tokenizer = createQwenChatTokenizer();
		const engine = Object.create(WebLLM.prototype) as WebLLM;
		const internals = asInternals(engine);
		type RealisticInfer = {
			forward: (
				ids: Int32Array,
				positions: Int32Array,
			) => Promise<Float32Array>;
			cachedTokenCount: number;
			resetKVCache: () => void;
		};
		const sequence = [7, 2];
		let step = 0;
		const fake: RealisticInfer = {
			forward: async (_ids: Int32Array, positions: Int32Array) => {
				const lastPos = positions[positions.length - 1] ?? -1;
				fake.cachedTokenCount = lastPos + 1;
				const tokenId = sequence[Math.min(step, sequence.length - 1)];
				step++;
				return createLogits(tokenId, tokenizer.vocabSize);
			},
			cachedTokenCount: 0,
			resetKVCache: () => {
				fake.cachedTokenCount = 0;
			},
		};
		internals._modelManager = {
			get: () => ({
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "qwen3" },
			}),
		};
		internals.models = new Map<string, unknown>([
			["model", { inference: fake }],
		]);

		const calls: Array<{ firstPosition: number; resetCount: number }> = [];
		let resetCount = 0;
		const origReset = fake.resetKVCache;
		fake.resetKVCache = () => {
			resetCount++;
			origReset.call(fake);
		};
		const origForward = fake.forward;
		let perCallFirstSeen = false;
		let pendingFirstPos = -1;
		fake.forward = async (ids: Int32Array, positions: Int32Array) => {
			if (!perCallFirstSeen) {
				pendingFirstPos = positions[0] ?? -1;
				perCallFirstSeen = true;
			}
			return origForward(ids, positions);
		};

		const drainCall = async (
			messages: {
				role: "system" | "user" | "assistant";
				content: string;
			}[],
		) => {
			perCallFirstSeen = false;
			pendingFirstPos = -1;
			step = 0;
			const beforeResets = resetCount;
			for await (const _chunk of engine.chatCompletion("model", messages, {
				maxTokens: 1,
				temperature: 0,
				enableThinking: false,
			})) {
				// drain
			}
			calls.push({
				firstPosition: pendingFirstPos,
				resetCount: resetCount - beforeResets,
			});
		};

		await drainCall([
			{ role: "system", content: "alpha" },
			{ role: "user", content: "p" },
		]);
		await drainCall([
			{ role: "system", content: "beta" },
			{ role: "user", content: "p" },
		]);
		await drainCall([
			{ role: "system", content: "alpha" },
			{ role: "user", content: "p" },
			{ role: "assistant", content: "h" },
			{ role: "user", content: "p" },
		]);

		// Call 1 is the cold start — position should be 0.
		expect(calls[0].firstPosition).toBe(0);
		// Call 2 has a different leading system message. Length didn't grow,
		// so the delta path is skipped purely on length grounds.
		expect(calls[1].firstPosition).toBe(0);
		expect(calls[1].resetCount).toBeGreaterThanOrEqual(1);
		// Call 3 is the bug repro: length grew (2 → 4) and cachedTokenCount
		// matches session.currentPosition (the realistic fake tracks both),
		// but the cached state's leading system is "beta" (call 2), not
		// "alpha" (call 3). The fix must detect the mismatch and full-reset
		// rather than take the delta path.
		expect(calls[2].firstPosition).toBe(0);
		expect(calls[2].resetCount).toBeGreaterThanOrEqual(1);
	});

	test("qwen chatCompletion yields visible assistant text through think tokens", async () => {
		const engine = createTestEngineWithTokenizer(
			createQwenChatTokenizer(),
			[5, 6, 7, 8, 2],
			"qwen3",
		);
		const chunks: Array<{
			text: string;
			tokenId?: number | undefined;
			done: boolean;
		}> = [];
		let finalStats: Record<string, unknown> | undefined;

		for await (const chunk of engine.chatCompletion(
			"model",
			[{ role: "user", content: "Hello" }],
			{ maxTokens: 8, temperature: 0, enableThinking: false },
		)) {
			chunks.push({
				text: chunk.text,
				tokenId: chunk.tokenId,
				done: chunk.done,
			});
			if (chunk.done)
				finalStats = chunk.stats as unknown as Record<string, unknown>;
		}

		expect(chunks).toEqual([
			{ text: "", tokenId: 5, done: false },
			{ text: "", tokenId: 6, done: false },
			{ text: " Hello", tokenId: 7, done: false },
			{ text: "!", tokenId: 8, done: false },
			{ text: "", tokenId: undefined, done: true },
		]);
		expect(finalStats).toMatchObject({
			text: " Hello!",
			tokenCount: 4,
			finishReason: "eos",
		});
	});
});
