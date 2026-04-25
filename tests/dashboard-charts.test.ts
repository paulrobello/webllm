import { expect, test } from "bun:test";
import {
	buildEmbeddingCosineChartData,
	buildTempSweepChartData,
	DIM_NAMES,
} from "../smoke-test/dashboard-charts.js";

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

test("DIM_NAMES includes embedding alongside the existing dimensions", () => {
	expect(DIM_NAMES).toEqual([
		"tool-calling",
		"reasoning",
		"instruction-following",
		"semantic-reasoning",
		"embedding",
	]);
});

test("buildEmbeddingCosineChartData groups latest cosine per (model, task)", () => {
	const evals = [
		{
			evalId: "e1",
			timestamp: "2026-04-24T10:00:00.000Z",
			modelId: "arctic-embed-s",
			results: [
				{
					taskId: "synonyms-1",
					dimension: "embedding",
					embeddingCosine: 0.42,
				},
				{
					// non-embedding dimension result must be ignored.
					taskId: "math-1",
					dimension: "reasoning",
					embeddingCosine: 0.99,
				},
				{
					// missing cosine must be skipped, not crash.
					taskId: "synonyms-2",
					dimension: "embedding",
				},
			],
		},
		{
			// Newer eval for the same (model, task) must overwrite.
			evalId: "e2",
			timestamp: "2026-04-24T11:00:00.000Z",
			modelId: "arctic-embed-s",
			results: [
				{
					taskId: "synonyms-1",
					dimension: "embedding",
					embeddingCosine: 0.81,
				},
			],
		},
		{
			evalId: "e3",
			timestamp: "2026-04-24T12:00:00.000Z",
			modelId: "bge-small",
			results: [
				{
					taskId: "synonyms-1",
					dimension: "embedding",
					embeddingCosine: 0.77,
				},
			],
		},
	];

	const data = buildEmbeddingCosineChartData(evals);

	expect(data.labels).toEqual(["synonyms-1"]);
	expect(data.datasets.map((d) => d.label)).toEqual([
		"arctic-embed-s",
		"bge-small",
	]);
	const arctic = data.datasets.find((d) => d.label === "arctic-embed-s");
	const bge = data.datasets.find((d) => d.label === "bge-small");
	// latest eval (e2) wins for arctic-embed-s, not the older 0.42 in e1.
	expect(arctic?.data).toEqual([0.81]);
	expect(bge?.data).toEqual([0.77]);
});

test("buildEmbeddingCosineChartData returns empty when no embedding cosines present", () => {
	const data = buildEmbeddingCosineChartData([
		{
			evalId: "e1",
			timestamp: "2026-04-24T10:00:00.000Z",
			modelId: "qwen3-0.6b",
			results: [{ taskId: "math-1", dimension: "reasoning" }],
		},
	]);
	expect(data.labels).toEqual([]);
	expect(data.datasets).toEqual([]);
});

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
