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
});
