import { describe, expect, test } from "bun:test";
import {
	type GenerationStreamChunk,
	type GenerationStreamResult,
	generateTextStream,
	type InternalGenerationOptions,
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

/**
 * Tokenizer whose vocab is laid out for deterministic steering tests.
 *
 *   0 <pad>            CONTROL
 *   1 <s>              CONTROL
 *   2 </s>             CONTROL  (eos)
 *   3 <|im_start|>     CONTROL
 *   4 <|im_end|>       CONTROL
 *   5 <think>          USER_DEFINED
 *   6 </think>         USER_DEFINED
 *   7 "reasoning"      NORMAL
 *   8 " here"          NORMAL   (leading space)
 *   9 "visible"        NORMAL
 *  10 " answer"        NORMAL   (leading space)
 *
 * A `<think>` -> "reasoning" -> " here" -> `</think>` -> "visible" -> " answer"
 * sequence therefore decodes to:
 *   visible    = "visible answer"
 *   thinking   = "reasoning here"
 */
function createStreamingTokenizer(): Tokenizer {
	const tokens: TokenData[] = [
		{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_start|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<|im_end|>", score: 0, attr: TokenAttribute.CONTROL },
		{ text: "<think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "reasoning", score: -1, attr: TokenAttribute.NORMAL },
		{ text: " here", score: -2, attr: TokenAttribute.NORMAL },
		{ text: "visible", score: -3, attr: TokenAttribute.NORMAL },
		{ text: " answer", score: -4, attr: TokenAttribute.NORMAL },
	];
	const config: TokenizerConfig = {
		type: TokenizerType.BPE,
		tokens,
		bpeRanks: new Map(),
		addedTokens: new Map([
			["<pad>", 0],
			["<s>", 1],
			["</s>", 2],
			["<|im_start|>", 3],
			["<|im_end|>", 4],
			["<think>", 5],
			["</think>", 6],
		]),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
	return new Tokenizer(config);
}

const BASE_SESSION_CONFIG: InferenceSessionConfig = {
	maxTokens: 100,
	temperature: 0,
	topK: 0,
	topP: 1,
	repetitionPenalty: 1,
	contextOverflowPolicy: "stop",
};

/**
 * Build a forwardPass that emits `sequence` token-by-token (one token per
 * call, greedy argmax). The prefill call (call 1) consumes sequence[0];
 * each subsequent call consumes the next entry. Tokens past the end of the
 * sequence keep emitting the final entry so the loop only stops via
 * maxTokens / eos / abort.
 */
function sequenceForwardPass(
	sequence: number[],
	vocabSize: number,
	abortAfter?: number,
): {
	forwardPass: (ids: number[], pos: number[]) => Promise<Float32Array>;
	controller: AbortController;
} {
	const controller = new AbortController();
	let callCount = 0;
	const forwardPass = (
		_ids: number[],
		_pos: number[],
	): Promise<Float32Array> => {
		callCount++;
		if (abortAfter !== undefined && callCount >= abortAfter) {
			controller.abort();
		}
		const target = sequence[Math.min(callCount - 1, sequence.length - 1)];
		const logits = new Float32Array(vocabSize);
		logits[target] = 100.0;
		return Promise.resolve(logits);
	};
	return { forwardPass, controller };
}

async function drainStream(
	stream: AsyncGenerator<GenerationStreamChunk, GenerationStreamResult>,
	onChunk?: (chunk: GenerationStreamChunk) => void,
): Promise<GenerationStreamResult | undefined> {
	let finalStats: GenerationStreamResult | undefined;
	for await (const chunk of stream) {
		if (onChunk) onChunk(chunk);
		if (chunk.done && chunk.stats) {
			finalStats = chunk.stats;
		}
	}
	return finalStats;
}

describe("generateTextStream (ENH-002 visible-only streaming)", () => {
	test("think-block sequence: deltas are visible-only, callbacks fire, join invariant holds", async () => {
		const tokenizer = createStreamingTokenizer();
		const sequence = [5, 7, 8, 6, 9, 10]; // <think>reasoning here</think>visible answer
		const { forwardPass } = sequenceForwardPass(sequence, tokenizer.vocabSize);

		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);

		const onTokenDeltas: string[] = [];
		const onThinkingDeltas: string[] = [];
		const config: InternalGenerationOptions = {
			maxTokens: sequence.length,
			temperature: 0,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			onToken: (d) => onTokenDeltas.push(d),
			onThinking: (d) => onThinkingDeltas.push(d),
		};

		const chunks: GenerationStreamChunk[] = [];
		const finalStats = await drainStream(
			generateTextStream({
				promptTokenIds: [1],
				sampler,
				session,
				eosTokenId: tokenizer.eosId,
				tokenizer,
				forwardPass,
				config,
			}),
			(c) => chunks.push(c),
		);

		// 1. Join invariant: concatenated onToken deltas === final result.text.
		expect(finalStats).toBeDefined();
		const finalText = finalStats?.text ?? "";
		const joinedCallbacks = onTokenDeltas.join("");
		const joinedChunks = chunks
			.filter((c) => !c.done)
			.map((c) => c.text)
			.join("");
		expect(joinedCallbacks).toBe(finalText);
		expect(joinedChunks).toBe(finalText);

		// 2. result.text is visible-only.
		expect(finalText).toBe("visible answer");

		// 3. No onToken delta contains think tags or reasoning content.
		for (const d of onTokenDeltas) {
			expect(d).not.toContain("<think>");
			expect(d).not.toContain("</think>");
			expect(d).not.toContain("reasoning");
		}

		// 4. onThinking received the inner reasoning content (no tag wrappers).
		expect(onThinkingDeltas.join("")).toBe("reasoning here");
		for (const d of onThinkingDeltas) {
			expect(d).not.toContain("<think>");
			expect(d).not.toContain("</think>");
		}
	});

	test("no-think sequence: onThinking never fires, onToken joins to full text", async () => {
		const tokenizer = createStreamingTokenizer();
		const sequence = [9, 10]; // "visible" " answer"
		const { forwardPass } = sequenceForwardPass(sequence, tokenizer.vocabSize);

		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);

		const onTokenDeltas: string[] = [];
		let onThinkingCalls = 0;
		const config: InternalGenerationOptions = {
			maxTokens: sequence.length,
			temperature: 0,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			// Deliberately NOT setting thinkingOpenTokenId/thinkingCloseTokenId —
			// a no-think model configuration. onThinking must never fire.
			onToken: (d) => onTokenDeltas.push(d),
			onThinking: () => {
				onThinkingCalls++;
			},
		};

		const finalStats = await drainStream(
			generateTextStream({
				promptTokenIds: [1],
				sampler,
				session,
				eosTokenId: tokenizer.eosId,
				tokenizer,
				forwardPass,
				config,
			}),
		);

		expect(onThinkingCalls).toBe(0);
		expect(onTokenDeltas.join("")).toBe("visible answer");
		expect(finalStats?.text).toBe("visible answer");
	});

	test("abort mid-stream: no onToken fires after the stream resolves", async () => {
		const tokenizer = createStreamingTokenizer();
		const sequence = [5, 7, 8, 6, 9, 10];
		// Abort on the 3rd forward call — mid-think-block.
		const { forwardPass, controller } = sequenceForwardPass(
			sequence,
			tokenizer.vocabSize,
			3,
		);

		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 50 },
			0,
		);

		const onTokenDeltas: string[] = [];
		const config: InternalGenerationOptions = {
			maxTokens: 50,
			temperature: 0,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			signal: controller.signal,
			thinkingOpenTokenId: 5,
			thinkingCloseTokenId: 6,
			onToken: (d) => onTokenDeltas.push(d),
		};

		const finalStats = await drainStream(
			generateTextStream({
				promptTokenIds: [1],
				sampler,
				session,
				eosTokenId: tokenizer.eosId,
				tokenizer,
				forwardPass,
				config,
			}),
		);

		// Generation stopped (finishReason reflects the abort) and the loop
		// did not run away to maxTokens.
		expect(finalStats?.finishReason).toBe("aborted");
		expect(finalStats?.tokenCount).toBeLessThan(config.maxTokens);

		// Snapshot the callback log immediately after the stream resolves,
		// wait a beat, then confirm nothing fired late.
		const countAtResolve = onTokenDeltas.length;
		await new Promise((r) => setTimeout(r, 10));
		expect(onTokenDeltas.length).toBe(countAtResolve);
	});
});
