import { describe, expect, test } from "bun:test";
import { computeTokenizerHash } from "../src/core/persistence.js";

describe("computeTokenizerHash", () => {
	const baseConfig = {
		type: "BPE" as const,
		vocab: { hello: 1, world: 2 },
		merges: ["h e", "l l"],
		specialTokens: { bos: 0, eos: 3 },
	};

	test("identical input yields identical hash", async () => {
		const a = await computeTokenizerHash(baseConfig);
		const b = await computeTokenizerHash(baseConfig);
		expect(a).toBe(b);
		expect(a).toMatch(/^[0-9a-f]{64}$/); // sha256 hex
	});

	test("key-permuted input yields identical hash", async () => {
		const reordered = {
			specialTokens: { eos: 3, bos: 0 },
			merges: ["h e", "l l"],
			vocab: { world: 2, hello: 1 },
			type: "BPE" as const,
		};
		expect(await computeTokenizerHash(baseConfig)).toBe(
			await computeTokenizerHash(reordered),
		);
	});

	test("vocab change yields different hash", async () => {
		const changed = {
			...baseConfig,
			vocab: {
				...baseConfig.vocab,
				foo: 4,
			},
		};
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});

	test("specialTokens change yields different hash", async () => {
		const changed = {
			...baseConfig,
			specialTokens: {
				...baseConfig.specialTokens,
				eos: 99,
			},
		};
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});

	test("Map-valued field: hash differs when Map contents differ", async () => {
		const baseWithMap = {
			type: "BPE" as const,
			bpeRanks: new Map([
				["h e", 1],
				["l l", 2],
			]),
		};
		const changedMap = {
			type: "BPE" as const,
			bpeRanks: new Map([
				["h e", 1],
				["l l", 99], // different rank
			]),
		};
		expect(await computeTokenizerHash(baseWithMap)).not.toBe(
			await computeTokenizerHash(changedMap),
		);
	});

	test("Map-valued field: insertion-order-permuted Map yields identical hash", async () => {
		const a = {
			bpeRanks: new Map([
				["a", 1],
				["b", 2],
				["c", 3],
			]),
		};
		const b = {
			bpeRanks: new Map([
				["c", 3],
				["a", 1],
				["b", 2],
			]),
		};
		expect(await computeTokenizerHash(a)).toBe(await computeTokenizerHash(b));
	});

	test("Uint8Array field: hash differs when bytes differ", async () => {
		const a = { precompiledCharsmap: new Uint8Array([1, 2, 3, 4]) };
		const b = { precompiledCharsmap: new Uint8Array([1, 2, 3, 5]) };
		expect(await computeTokenizerHash(a)).not.toBe(
			await computeTokenizerHash(b),
		);
	});

	test("undefined-valued keys are dropped (matches JSON.stringify semantics)", async () => {
		const withUndef = { a: 1, b: undefined };
		const withoutB = { a: 1 };
		expect(await computeTokenizerHash(withUndef)).toBe(
			await computeTokenizerHash(withoutB),
		);
	});

	test("nested-object key permutation also yields identical hash", async () => {
		const a = { outer: { x: 1, y: 2, nested: { p: "a", q: "b" } } };
		const b = { outer: { nested: { q: "b", p: "a" }, y: 2, x: 1 } };
		expect(await computeTokenizerHash(a)).toBe(await computeTokenizerHash(b));
	});
});
