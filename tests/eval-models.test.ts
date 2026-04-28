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
