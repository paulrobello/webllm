import { describe, expect, test } from "bun:test";
import { Sampler } from "../src/inference/sampler.js";
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
	overrides: Partial<TokenizerConfig> = {},
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
		...overrides,
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

function makeWordPieceTokenizer(
	extraTokens: string[] = [],
	overrides: Partial<TokenizerConfig> = {},
): Tokenizer {
	const base = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"];
	const tokens: TokenData[] = [...base, ...extraTokens].map((t) => ({
		text: t,
		score: 0,
		attr: base.includes(t) ? TokenAttribute.CONTROL : TokenAttribute.NORMAL,
	}));
	const addedTokens = new Map<string, number>();
	base.forEach((t, i) => {
		addedTokens.set(t, i);
	});
	return new Tokenizer({
		type: TokenizerType.WORDPIECE,
		tokens,
		bpeRanks: new Map(),
		addedTokens,
		eosTokenId: 3,
		bosTokenId: 2,
		padTokenId: 0,
		vocabSize: tokens.length,
		clsTokenId: 2,
		sepTokenId: 3,
		unkTokenId: 1,
		...overrides,
	});
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

	test("decode can preserve special tokens when requested", () => {
		const tok = new Tokenizer(makeBpeConfig(BASIC_TOKENS, new Map()));
		const text = tok.decode([1, 3, 2], { includeSpecialTokens: true });
		expect(text).toBe("<s>a</s>");
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

	test("BPE encode preserves GPT-2 byte-encoded whitespace and newlines for qwen-style tokenizers", () => {
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "Ċ", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "Ġ", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "a", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "b", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "Ġa", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "Ġab", score: 0, attr: TokenAttribute.NORMAL },
		];
		const ranks = new Map<string, number>([
			["Ġ a", 0],
			["Ġa b", 1],
		]);
		const tok = new Tokenizer(
			makeBpeConfig(tokens, ranks, { preTokenizer: "qwen2" }),
		);
		expect(tok.encode("\n a\n")).toEqual([3, 7, 3]);
	});

	test("BPE decode reverses GPT-2 byte encoding back to plain text", () => {
		const tokens: TokenData[] = [
			{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
			{ text: "Hello", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "Ġworld", score: 0, attr: TokenAttribute.NORMAL },
			{ text: "Ċ", score: 0, attr: TokenAttribute.NORMAL },
		];
		const tok = new Tokenizer(
			makeBpeConfig(tokens, new Map(), { preTokenizer: "qwen2" }),
		);
		expect(tok.decode([3, 4, 5])).toBe("Hello world\n");
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
			addPrefixSpace: true,
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
		// "café" -> no ▁ prefix -> [c, a, f, <0xC3>, <0xA9>]
		expect(tok.encode("café")).toEqual([4, 5, 6, 7, 8]);
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

describe("Sampler", () => {
	test("samples from large vocab without stack overflow", () => {
		const sampler = new Sampler({
			temperature: 0.7,
			topK: 40,
			topP: 0.95,
			seed: 123,
		});
		const logits = new Float32Array(151_936);
		logits[100] = 10;
		logits[200] = 11;
		logits[300] = 12;
		expect(() => sampler.sample(logits)).not.toThrow();
	});
});

describe("Tokenizer WordPiece config", () => {
	test("accepts WORDPIECE type with bert-style special token ids", () => {
		const tok = new Tokenizer({
			type: TokenizerType.WORDPIECE,
			tokens: [
				{ text: "[PAD]", score: 0, attr: TokenAttribute.CONTROL },
				{ text: "[UNK]", score: 0, attr: TokenAttribute.CONTROL },
				{ text: "[CLS]", score: 0, attr: TokenAttribute.CONTROL },
				{ text: "[SEP]", score: 0, attr: TokenAttribute.CONTROL },
				{ text: "hello", score: -1, attr: TokenAttribute.NORMAL },
			],
			bpeRanks: new Map(),
			addedTokens: new Map([
				["[PAD]", 0],
				["[UNK]", 1],
				["[CLS]", 2],
				["[SEP]", 3],
			]),
			eosTokenId: 3,
			bosTokenId: 2,
			padTokenId: 0,
			vocabSize: 5,
			clsTokenId: 2,
			sepTokenId: 3,
			unkTokenId: 1,
			maskTokenId: undefined,
		});
		expect(tok.options.type).toBe(TokenizerType.WORDPIECE);
		expect(tok.options.clsTokenId).toBe(2);
		expect(tok.options.sepTokenId).toBe(3);
		expect(tok.options.unkTokenId).toBe(1);
		expect(tok.options.maskTokenId).toBeUndefined();
	});
});

describe("WordPiece basic tokenize", () => {
	test("lowercases and splits on whitespace", () => {
		const tok = makeWordPieceTokenizer(["▁hello", "▁world"]);
		// [CLS] ▁hello ▁world [SEP] -> [2, 4, 5, 3]
		expect(tok.encode("Hello World")).toEqual([2, 4, 5, 3]);
	});

	test("splits punctuation off tokens", () => {
		const tok = makeWordPieceTokenizer(["▁hello", "▁.", "▁,", "▁world"]);
		// vocab: [CLS]=2 [SEP]=3 ▁hello=4 ▁.=5 ▁,=6 ▁world=7
		// [CLS] ▁hello ▁, ▁world ▁. [SEP]
		expect(tok.encode("hello, world.")).toEqual([2, 4, 6, 7, 5, 3]);
	});

	test("strips accents", () => {
		const tok = makeWordPieceTokenizer(["▁cafe"]);
		expect(tok.encode("café")).toEqual([2, 4, 3]);
	});

	test("emits UNK for unknown words", () => {
		const tok = makeWordPieceTokenizer([]);
		expect(tok.encode("unknown")).toEqual([2, 1, 3]); // [CLS] [UNK] [SEP]
	});

	test("throws when cls/sep/unk IDs are missing from config", () => {
		expect(() =>
			new Tokenizer({
				type: TokenizerType.WORDPIECE,
				tokens: [{ text: "[PAD]", score: 0, attr: TokenAttribute.CONTROL }],
				bpeRanks: new Map(),
				addedTokens: new Map(),
				eosTokenId: 0,
				bosTokenId: 0,
				padTokenId: 0,
				vocabSize: 1,
			}).encode("hello"),
		).toThrow(/clsTokenId/);
	});
});

describe("WordPiece subword + decode", () => {
	test("splits into subwords using phantom-space word-start", () => {
		// llama.cpp BERT vocab: "▁un" is word-start, "known" is continuation
		// (originally "##known" in HF, stripped during GGUF conversion).
		const tok = makeWordPieceTokenizer(["▁un", "known"]);
		const ids = tok.encode("unknown");
		expect(ids).toEqual([2, 4, 5, 3]);
	});

	test("falls back to UNK when no subword matches", () => {
		// Vocab has "▁un" but no continuation for "known".
		const tok = makeWordPieceTokenizer(["▁un"]);
		expect(tok.encode("unknown")).toEqual([2, 1, 3]);
	});

	test("truncates over-length input keeping CLS and SEP", () => {
		const extras = Array.from({ length: 600 }, (_, i) => `▁w${i}`);
		const tok = makeWordPieceTokenizer(extras, { contextLength: 10 });
		const text = extras
			.slice(0, 20)
			.map((t) => t.slice(1))
			.join(" ");
		const ids = tok.encode(text);
		expect(ids.length).toBe(10);
		expect(ids[0]).toBe(2); // [CLS]
		expect(ids[ids.length - 1]).toBe(3); // [SEP]
	});

	test("decode rejoins phantom-space word starts with spaces", () => {
		const tok = makeWordPieceTokenizer(["▁un", "known", "▁dog"]);
		// ids: [CLS, ▁un, known, ▁dog, SEP] -> "unknown dog"
		const decoded = tok.decode([2, 4, 5, 6, 3]);
		expect(decoded).toBe("unknown dog");
	});

	test("decode preserves special tokens when requested", () => {
		const tok = makeWordPieceTokenizer(["▁un", "known"]);
		const decoded = tok.decode([2, 4, 5, 3], { includeSpecialTokens: true });
		expect(decoded).toContain("[CLS]");
		expect(decoded).toContain("[SEP]");
	});

	test("emits UNK for chunks longer than max_input_chars_per_word (100)", () => {
		const tok = makeWordPieceTokenizer(["▁a", "a"]);
		const longChunk = "a".repeat(101);
		expect(tok.encode(longChunk)).toEqual([2, 1, 3]); // [CLS] [UNK] [SEP]
	});

	test("does NOT trigger UNK fallback for chunks exactly at max_input_chars (100)", () => {
		const tok = makeWordPieceTokenizer(["▁a", "a"]);
		const exactChunk = "a".repeat(100);
		const ids = tok.encode(exactChunk);
		// 1 [CLS] + 1 ("▁a") + 99 ("a") + 1 [SEP] = 102 ids
		expect(ids[0]).toBe(2);
		expect(ids[ids.length - 1]).toBe(3);
		expect(ids.length).toBe(102);
		// All middle ids are non-UNK
		for (let i = 1; i < ids.length - 1; i++) expect(ids[i]).not.toBe(1);
	});
});
