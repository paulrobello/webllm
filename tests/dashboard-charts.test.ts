import { expect, test } from "bun:test";
import { buildTempSweepChartData } from "../smoke-test/dashboard-charts.js";

function makeEval(
	evalId: string,
	temperature: number,
	score: number,
	timestamp: string,
) {
	return {
		evalId,
		timestamp,
		modelId: "qwen3-0.6b-q4f16",
		params: { temperature },
		dimensions: {
			reasoning: { total: 4, passed: 3, score, avgLatencyMs: 10 },
		},
	};
}

test("temperature sweep chart includes hot bucket data and styling", () => {
	const data = buildTempSweepChartData([
		makeEval("cold", 0.3, 0.25, "2026-04-24T10:00:00.000Z"),
		makeEval("warm", 0.6, 0.5, "2026-04-24T10:01:00.000Z"),
		makeEval("hot", 0.9, 0.9, "2026-04-24T10:02:00.000Z"),
	]);

	expect(data.labels).toEqual(["qwen3-0.6b-q4f16 · reasoning"]);
	const hot = data.datasets.find((dataset) => dataset.label === "hot");
	expect(hot?.data).toEqual([90]);
	expect(hot?.backgroundColor).toBe("#f85149");
});
