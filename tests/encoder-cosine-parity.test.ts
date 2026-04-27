import { describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";

const BASELINE_PATH = "eval/reports/embed-perf-baseline-cosine.json";

describe("encoder cosine parity (G3 guard)", () => {
	it("baseline file exists and is well-formed", () => {
		expect(existsSync(BASELINE_PATH)).toBe(true);
		const baseline = JSON.parse(readFileSync(BASELINE_PATH, "utf-8")) as {
			cosine: number;
			tolerance: number;
			model: string;
			pair: string[];
		};
		expect(baseline.cosine).toBeGreaterThan(0.5);
		expect(baseline.cosine).toBeLessThan(1.0);
		expect(baseline.tolerance).toBe(0.005);
		expect(baseline.model).toBe("snowflake-arctic-embed-s-q0f32-b4");
		expect(baseline.pair).toEqual(["happy", "joyful"]);
	});

	// The actual cosine measurement runs in the browser via [8/8] —
	// this test only enforces that the baseline file is present so the
	// browser-side check has something to compare against. Bun has no
	// navigator.gpu; the live bench-full run is the real G3 gate.
});
