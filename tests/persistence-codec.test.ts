import { describe, expect, test } from "bun:test";
import { computeTokenizerHash } from "../src/core/persistence.js";
import type { TokenizerConfig } from "../src/inference/tokenizer.js";

describe("computeTokenizerHash", () => {
	const baseConfig = {
		type: "BPE" as const,
		vocab: { hello: 1, world: 2 },
		merges: ["h e", "l l"],
		specialTokens: { bos: 0, eos: 3 },
	} as unknown as TokenizerConfig;

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
		} as unknown as TokenizerConfig;
		expect(await computeTokenizerHash(baseConfig)).toBe(
			await computeTokenizerHash(reordered),
		);
	});

	test("vocab change yields different hash", async () => {
		const changed = {
			...baseConfig,
			vocab: {
				...(baseConfig as unknown as { vocab: Record<string, number> }).vocab,
				foo: 4,
			},
		} as unknown as TokenizerConfig;
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});

	test("specialTokens change yields different hash", async () => {
		const changed = {
			...baseConfig,
			specialTokens: {
				...(baseConfig as unknown as { specialTokens: Record<string, number> })
					.specialTokens,
				eos: 99,
			},
		} as unknown as TokenizerConfig;
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});
});
