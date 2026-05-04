import { describe, expect, test } from "bun:test";
import type { TokenData, TokenizerConfig } from "../src/inference/tokenizer.js";
import {
	StreamingDecoder,
	TokenAttribute,
	Tokenizer,
	TokenizerType,
} from "../src/inference/tokenizer.js";

function makeTokenizer(tokens: string[]): Tokenizer {
	const tokenData: TokenData[] = tokens.map((text, i) => ({
		text,
		score: -i,
		attr: TokenAttribute.NORMAL,
	}));
	const config: TokenizerConfig = {
		type: TokenizerType.SPM,
		tokens: tokenData,
		bpeRanks: new Map(),
		addedTokens: new Map(),
		eosTokenId: tokens.length - 1,
		bosTokenId: 0,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
	return new Tokenizer(config);
}

function makeThinkingTokenizer(): Tokenizer {
	const tokens: TokenData[] = [
		{ text: "<think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "</think>", score: 0, attr: TokenAttribute.USER_DEFINED },
		{ text: "▁plan", score: -1, attr: TokenAttribute.NORMAL },
		{ text: "▁answer", score: -2, attr: TokenAttribute.NORMAL },
		{ text: "</s>", score: -3, attr: TokenAttribute.CONTROL },
	];
	const config: TokenizerConfig = {
		type: TokenizerType.SPM,
		tokens,
		bpeRanks: new Map(),
		addedTokens: new Map([
			["<think>", 0],
			["</think>", 1],
			["</s>", 4],
		]),
		eosTokenId: 4,
		bosTokenId: 4,
		padTokenId: 4,
		vocabSize: tokens.length,
		addPrefixSpace: false,
	};
	return new Tokenizer(config);
}

describe("StreamingDecoder", () => {
	test("push returns incremental text for simple tokens", () => {
		const tokenizer = makeTokenizer(["<s>", "▁Hello", "▁world", "▁!", "</s>"]);
		const decoder = new StreamingDecoder(tokenizer);

		expect(decoder.push(1)).toBe("Hello");
		expect(decoder.push(2)).toBe(" world");
		expect(decoder.push(3)).toBe(" !");
		expect(decoder.text).toBe("Hello world !");
	});

	test("push accumulates to full text matching decode()", () => {
		const tokenizer = makeTokenizer(["<s>", "▁foo", "▁bar", "▁baz", "</s>"]);
		const decoder = new StreamingDecoder(tokenizer);
		const ids = [1, 2, 3];

		for (const id of ids) {
			decoder.push(id);
		}

		expect(decoder.text).toBe(tokenizer.decode(ids));
	});

	test("reset clears state", () => {
		const tokenizer = makeTokenizer(["<s>", "▁test", "</s>"]);
		const decoder = new StreamingDecoder(tokenizer);

		decoder.push(1);
		expect(decoder.text).toBe("test");

		decoder.reset();
		expect(decoder.text).toBe("");
		expect(decoder.tokens).toEqual([]);
	});

	test("tokens getter returns accumulated IDs", () => {
		const tokenizer = makeTokenizer(["<s>", "▁a", "▁b", "</s>"]);
		const decoder = new StreamingDecoder(tokenizer);

		decoder.push(1);
		decoder.push(2);

		expect(decoder.tokens).toEqual([1, 2]);
	});

	test("can preserve special tokens while streaming", () => {
		const tokenizer = makeTokenizer(["<s>", "▁thinking", "</s>"]);
		const decoder = new StreamingDecoder(tokenizer, {
			includeSpecialTokens: true,
		});

		expect(decoder.push(0)).toBe("<s>");
		expect(decoder.push(1)).toBe(" thinking");
		expect(decoder.push(2)).toBe("</s>");
		expect(decoder.text).toBe("<s> thinking</s>");
	});

	test("holds back trailing U+FFFD until the multi-byte sequence completes", () => {
		// Simulate BPE byte fallback splitting a 4-byte emoji (🌞 = F0 9F 8C 9E)
		// across two tokens: first token decodes to one or two replacement chars,
		// second token completes the sequence and decode() returns the real
		// emoji. The streaming decoder must NOT emit `�` chunks the consumer
		// would accumulate forever — it should hold them until decode resolves.
		const tokens = ["<s>", "first-half", "second-half", "</s>"];
		const tokenizer = makeTokenizer(tokens);
		// Override decode to mimic the partial-UTF-8 → replacement-char shape
		// real BPE byte fallback produces.
		const orig = tokenizer.decode.bind(tokenizer);
		(tokenizer as { decode: typeof tokenizer.decode }).decode = (
			ids: number[],
		) => {
			if (ids.length === 1 && ids[0] === 1) return "Hi �";
			if (ids.length === 2 && ids[0] === 1 && ids[1] === 2) return "Hi 🌞";
			return orig(ids);
		};

		const decoder = new StreamingDecoder(tokenizer);
		expect(decoder.push(1)).toBe("Hi "); // ← `�` held back
		expect(decoder.text).toBe("Hi ");
		expect(decoder.push(2)).toBe("🌞");
		expect(decoder.text).toBe("Hi 🌞");
	});

	test("requires </think> before continuing visible answer text", () => {
		const tokenizer = makeThinkingTokenizer();
		const decoder = new StreamingDecoder(tokenizer);

		expect(decoder.push(0)).toBe("");
		expect(decoder.push(2)).toBe("");
		expect(decoder.text).toBe("");
		expect(decoder.push(1)).toBe("");
		expect(decoder.push(3)).toBe("answer");
		expect(decoder.text).toBe("answer");
	});
});
