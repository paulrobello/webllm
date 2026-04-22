import { describe, expect, test } from "bun:test";
import { WebLLM } from "../src/index.js";
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

function createLogits(tokenId: number): Float32Array {
	const logits = new Float32Array(TOKENS.length);
	logits[tokenId] = 100;
	return logits;
}

function createTestEngine(sequence: number[]): WebLLM {
	const tokenizer = createTokenizer();
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
					return createLogits(tokenId);
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
});
