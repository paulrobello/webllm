import { describe, expect, test } from "bun:test";
import {
	TokenAttribute,
	type TokenData,
	Tokenizer,
	type TokenizerConfig,
	TokenizerType,
} from "../src/inference/tokenizer.js";

function makeBpeConfig(
	tokens: TokenData[],
	bpeRanks: Map<string, number>,
): TokenizerConfig {
	return {
		type: TokenizerType.BPE,
		tokens,
		bpeRanks,
		addedTokens: new Map(),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
}

function makeSpmConfig(tokens: TokenData[]): TokenizerConfig {
	return {
		type: TokenizerType.SPM,
		tokens,
		bpeRanks: new Map(),
		addedTokens: new Map(),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: tokens.length,
	};
}

const BASIC_TOKENS: TokenData[] = [
	{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "a", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "b", score: -2, attr: TokenAttribute.NORMAL },
	{ text: "c", score: -3, attr: TokenAttribute.NORMAL },
	{ text: "ab", score: -0.5, attr: TokenAttribute.NORMAL },
	{ text: "abc", score: -0.1, attr: TokenAttribute.NORMAL },
	{ text: " hello", score: -0.2, attr: TokenAttribute.NORMAL },
	{ text: " world", score: -0.3, attr: TokenAttribute.NORMAL },
];

describe("Tokenizer", () => {
	test("encodes single characters via BPE", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		const ids = tok.encode("a");
		expect(ids).toEqual([3]);
	});

	test("encodes merged tokens via BPE", () => {
		const ranks = new Map<string, number>([["a b", 0]]);
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, ranks));
		const ids = tok.encode("ab");
		expect(ids).toEqual([6]); // "ab" merged = index 6
	});

	test("encodes via SPM with score-based merging", () => {
		const tok = new Tokenizer(makeSpmConfig(BASIC_TOKENS));
		const ids = tok.encode("abc");
		expect(ids).toEqual([7]); // "abc" has highest score (-0.1)
	});

	test("decodes token IDs back to text", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		const text = tok.decode([3, 4, 5]);
		expect(text).toBe("abc");
	});

	test("handles special tokens in decode", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		const text = tok.decode([1, 3, 2]); // <s> a </s>
		expect(text).toBe("a");
	});

	test("vocabSize returns correct count", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		expect(tok.vocabSize).toBe(10);
	});

	test("eosId/bosId return correct values", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		expect(tok.eosId).toBe(2);
		expect(tok.bosId).toBe(1);
	});

	test("getId lookup returns token ID", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		expect(tok.getId("a")).toBe(3);
		expect(tok.getId("nonexistent")).toBeUndefined();
	});

	test("getToken lookup returns token data", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		const t = tok.getToken(3);
		expect(t?.text).toBe("a");
	});
});
