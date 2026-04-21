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
		// Synthetic vocab has no ▁-prefixed entries — skip LLaMA-style normalization.
		addPrefixSpace: false,
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

	test("SPM encode prepends ▁ and replaces spaces (LLaMA-style)", () => {
		// Mini vocab with intermediate merges so the pairwise algorithm can chain
		// single chars up to ▁hi / ▁world. Score ordering (higher score = preferred)
		// drives the greedy merge.
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL }, // 0
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL }, // 1
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL }, // 2
			{ text: "▁", score: -20, attr: TokenAttribute.NORMAL }, // 3
			{ text: "h", score: -20, attr: TokenAttribute.NORMAL }, // 4
			{ text: "i", score: -20, attr: TokenAttribute.NORMAL }, // 5
			{ text: "w", score: -20, attr: TokenAttribute.NORMAL }, // 6
			{ text: "o", score: -20, attr: TokenAttribute.NORMAL }, // 7
			{ text: "r", score: -20, attr: TokenAttribute.NORMAL }, // 8
			{ text: "l", score: -20, attr: TokenAttribute.NORMAL }, // 9
			{ text: "d", score: -20, attr: TokenAttribute.NORMAL }, // 10
			{ text: "▁h", score: -5, attr: TokenAttribute.NORMAL }, // 11
			{ text: "▁w", score: -5, attr: TokenAttribute.NORMAL }, // 12
			{ text: "▁hi", score: -1, attr: TokenAttribute.NORMAL }, // 13
			{ text: "▁world", score: -1, attr: TokenAttribute.NORMAL }, // 14
			{ text: "▁wo", score: -3, attr: TokenAttribute.NORMAL }, // 15
			{ text: "▁wor", score: -2, attr: TokenAttribute.NORMAL }, // 16
			{ text: "▁worl", score: -2, attr: TokenAttribute.NORMAL }, // 17
		];
		const cfg: TokenizerConfig = {
			type: TokenizerType.SPM,
			tokens,
			bpeRanks: new Map(),
			addedTokens: new Map(),
			eosTokenId: 2,
			bosTokenId: 1,
			padTokenId: 0,
			vocabSize: tokens.length,
			// addPrefixSpace defaults to true
		};
		const tok = new Tokenizer(cfg);
		expect(tok.encode("hi world")).toEqual([13, 14]);
	});

	test("SPM byte fallback emits <0xHH> tokens for unknown UTF-8 bytes", () => {
		// Build a vocab that has no multi-char tokens for 'é' (0xC3 0xA9) —
		// so it must fall back to <0xC3> and <0xA9> byte tokens.
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL }, // 0
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL }, // 1
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL }, // 2
			{ text: "▁", score: -20, attr: TokenAttribute.NORMAL }, // 3
			{ text: "c", score: -10, attr: TokenAttribute.NORMAL }, // 4
			{ text: "a", score: -10, attr: TokenAttribute.NORMAL }, // 5
			{ text: "f", score: -10, attr: TokenAttribute.NORMAL }, // 6
			{ text: "<0xC3>", score: 0, attr: TokenAttribute.BYTE }, // 7
			{ text: "<0xA9>", score: 0, attr: TokenAttribute.BYTE }, // 8
		];
		const cfg: TokenizerConfig = {
			type: TokenizerType.SPM,
			tokens,
			bpeRanks: new Map(),
			addedTokens: new Map(),
			eosTokenId: 2,
			bosTokenId: 1,
			padTokenId: 0,
			vocabSize: tokens.length,
		};
		const tok = new Tokenizer(cfg);
		// "café" -> after ▁ prefix "▁café" -> [▁, c, a, f, <0xC3>, <0xA9>]
		expect(tok.encode("café")).toEqual([3, 4, 5, 6, 7, 8]);
	});

	test("SPM decode reassembles byte-fallback tokens into UTF-8 characters", () => {
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "▁hi", score: -1, attr: TokenAttribute.NORMAL }, // 3
			{ text: "<0xC3>", score: 0, attr: TokenAttribute.BYTE }, // 4
			{ text: "<0xA9>", score: 0, attr: TokenAttribute.BYTE }, // 5
		];
		const cfg: TokenizerConfig = {
			type: TokenizerType.SPM,
			tokens,
			bpeRanks: new Map(),
			addedTokens: new Map(),
			eosTokenId: 2,
			bosTokenId: 1,
			padTokenId: 0,
			vocabSize: tokens.length,
		};
		const tok = new Tokenizer(cfg);
		// "▁hi" + "é" (as two bytes) -> decode to "hié"
		expect(tok.decode([3, 4, 5])).toBe("hié");
	});

	test("SPM decode converts ▁ back to space and strips leading prefix", () => {
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "▁hi", score: -1, attr: TokenAttribute.NORMAL },
			{ text: "▁world", score: -1, attr: TokenAttribute.NORMAL },
		];
		const cfg: TokenizerConfig = {
			type: TokenizerType.SPM,
			tokens,
			bpeRanks: new Map(),
			addedTokens: new Map(),
			eosTokenId: 2,
			bosTokenId: 1,
			padTokenId: 0,
			vocabSize: tokens.length,
		};
		const tok = new Tokenizer(cfg);
		expect(tok.decode([3, 4])).toBe("hi world");
	});
});
