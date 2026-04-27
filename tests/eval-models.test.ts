import { describe, expect, test } from "bun:test";
import { BENCHMARK_MODELS } from "../eval/models.js";

describe("arctic-embed registration", () => {
	test("architecture is 'bert' for all snowflake-arctic-embed-* entries", () => {
		const arctic = BENCHMARK_MODELS.filter((m) =>
			m.id.startsWith("snowflake-arctic-embed"),
		);
		expect(arctic.length).toBeGreaterThan(0);
		for (const m of arctic) {
			expect(m.architecture).toBe("bert");
			expect(m.capabilities.embedding).toBe(true);
		}
	});
});

describe("recommendedPrefillTile auto-default", () => {
	const EXPECTED_7B_PLUS_TILE = 128;
	const EXPECTED_7B_PLUS_IDS = [
		"mistral-7b-instruct-v0.3-q4ks",
		"mistral-7b-instruct-v0.3-q3km",
		"mistral-7b-instruct-v0.3-iq4xs",
		"llama-3.1-8b-instruct-iq3m",
		"qwen3-8b-iq3m",
	];

	test("all 7B+ entries default to tile=128", () => {
		for (const id of EXPECTED_7B_PLUS_IDS) {
			const m = BENCHMARK_MODELS.find((x) => x.id === id);
			expect(m, `model ${id} missing from registry`).toBeDefined();
			expect(m?.recommendedPrefillTile).toBe(EXPECTED_7B_PLUS_TILE);
		}
	});

	test("sub-7B entries leave recommendedPrefillTile unset", () => {
		const subSeven = BENCHMARK_MODELS.filter((m) => m.paramsB < 7);
		expect(subSeven.length).toBeGreaterThan(0);
		for (const m of subSeven) {
			expect(
				m.recommendedPrefillTile,
				`${m.id} (paramsB=${m.paramsB}) should not have a recommendedPrefillTile`,
			).toBeUndefined();
		}
	});
});
