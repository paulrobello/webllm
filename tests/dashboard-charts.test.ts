import { expect, test } from "bun:test";
import {
	buildEmbeddingCosineChartData,
	buildEmbeddingLatencyChartData,
	buildEmbeddingThroughputChartData,
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

test("buildEmbeddingLatencyChartData returns median ms per text per model, latest eval wins", () => {
	const evals = [
		// older eval — should be overwritten by the newer one for the same model.
		{
			evalId: "old",
			timestamp: "2026-04-25T09:00:00.000Z",
			modelId: "arctic-embed-s",
			results: [
				{ taskId: "t1", dimension: "embedding", latencyMs: 200 },
				{ taskId: "t2", dimension: "embedding", latencyMs: 200 },
			],
		},
		{
			evalId: "fresh",
			timestamp: "2026-04-25T10:00:00.000Z",
			modelId: "arctic-embed-s",
			results: [
				{ taskId: "t1", dimension: "embedding", latencyMs: 10 },
				{ taskId: "t2", dimension: "embedding", latencyMs: 20 },
				{ taskId: "t3", dimension: "embedding", latencyMs: 30 },
				// non-embedding row must be ignored.
				{ taskId: "t4", dimension: "reasoning", latencyMs: 9999 },
			],
		},
		{
			evalId: "m",
			timestamp: "2026-04-25T10:30:00.000Z",
			modelId: "arctic-embed-m",
			results: [
				{ taskId: "t1", dimension: "embedding", latencyMs: 40 },
				{ taskId: "t2", dimension: "embedding", latencyMs: 60 },
			],
		},
	];

	const data = buildEmbeddingLatencyChartData(evals);
	// Sorted by latency ascending: arctic-embed-s (median 20) before
	// arctic-embed-m (median 50).
	expect(data.labels).toEqual(["arctic-embed-s", "arctic-embed-m"]);
	expect(data.datasets[0].data).toEqual([20, 50]);
});

test("buildEmbeddingThroughputChartData inverts latency to texts/sec", () => {
	const evals = [
		{
			evalId: "fresh",
			timestamp: "2026-04-25T10:00:00.000Z",
			modelId: "arctic-embed-s",
			results: [
				{ taskId: "t1", dimension: "embedding", latencyMs: 10 },
				{ taskId: "t2", dimension: "embedding", latencyMs: 20 },
				{ taskId: "t3", dimension: "embedding", latencyMs: 30 },
			],
		},
	];
	const data = buildEmbeddingThroughputChartData(evals);
	expect(data.labels).toEqual(["arctic-embed-s"]);
	// 1000 / median(10,20,30) = 1000 / 20 = 50 texts/sec.
	expect(data.datasets[0].data).toEqual([50]);
});

test("temperature sweep chart splits thinking-on and thinking-off into separate series", () => {
	// Same modelId across both rows; only the `thinking` field differs.
	// Pre-fix this would collapse to a single series with the later
	// timestamp silently overwriting the earlier per bucket.
	const make = (
		evalId: string,
		thinking: "on" | "off",
		bucket: "cold" | "warm" | "hot",
		score: number,
		timestamp: string,
	) => ({
		evalId,
		timestamp,
		modelId: "qwen3-0.6b-q4f16",
		thinking,
		params: {
			temperature: bucket === "cold" ? 0.1 : bucket === "warm" ? 0.6 : 0.9,
		},
		dimensions: {
			reasoning: { total: 4, passed: 3, score, avgLatencyMs: 10 },
		},
	});

	const data = buildTempSweepChartData([
		make("a", "off", "cold", 0.4, "2026-04-25T10:00:00.000Z"),
		make("b", "off", "warm", 0.5, "2026-04-25T10:01:00.000Z"),
		make("c", "on", "cold", 0.7, "2026-04-25T10:02:00.000Z"),
		make("d", "on", "warm", 0.8, "2026-04-25T10:03:00.000Z"),
	]);

	expect(data.labels.sort()).toEqual([
		"qwen3-0.6b-q4f16 (think) · reasoning",
		"qwen3-0.6b-q4f16 · reasoning",
	]);
	const cold = data.datasets.find((d) => d.label === "cold");
	const warm = data.datasets.find((d) => d.label === "warm");
	// Both thinking modes contribute their own cold + warm; nothing was
	// overwritten by the latest-wins logic.
	expect(cold?.data.filter((v) => v != null).length).toBe(2);
	expect(warm?.data.filter((v) => v != null).length).toBe(2);
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
