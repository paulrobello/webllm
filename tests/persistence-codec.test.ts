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
});
