/**
 * Canonical eval-dimension names rendered in dashboard panels. The
 * order is the row order in the per-dimension grouped bar chart.
 * Mirrors `EvalDimension` in src/evaluation/types.ts.
 */
export const DIM_NAMES = [
	"tool-calling",
	"reasoning",
	"instruction-following",
	"semantic-reasoning",
	"embedding",
];

export const TEMP_SWEEP_BUCKETS = ["cold", "warm", "hot"];

export const TEMP_SWEEP_BUCKET_COLORS = {
	cold: "#58a6ff",
	warm: "#d29922",
	hot: "#f85149",
};

/**
 * Bucket temperature into hot / warm / cold bands for the evals table.
 * Matches the smoke-profile convention: ≤0.4 cold, ≥0.8 hot, otherwise warm.
 * Returns null when temperature wasn't recorded.
 */
export function tempBucket(temperature) {
	if (typeof temperature !== "number" || !Number.isFinite(temperature)) {
		return null;
	}
	if (temperature <= 0.4) return "cold";
	if (temperature >= 0.8) return "hot";
	return "warm";
}

export function buildTempSweepChartData(evals) {
	// Group evals by (modelId, thinking, dimension, tempBucket).
	//
	// `thinking` is part of the key because Qwen-style models share a single
	// modelId across thinking-on and thinking-off runs but produce
	// substantively different scores. Without this split the latest-per-
	// bucket logic below silently overwrites one mode with the other.
	const byModelDim = new Map();
	for (const ev of evals) {
		const bucket = tempBucket(ev.params?.temperature);
		if (!bucket) continue;
		const thinking = ev.thinking === "on" ? "on" : "off";
		for (const [dim, ds] of Object.entries(ev.dimensions ?? {})) {
			if (!ds || !ds.total) continue;
			const k = `${ev.modelId}::${thinking}::${dim}`;
			if (!byModelDim.has(k)) {
				byModelDim.set(k, {
					model: ev.modelId,
					thinking,
					dim,
					points: {},
				});
			}
			const entry = byModelDim.get(k);
			// Latest per bucket wins.
			if (!entry.points[bucket] || entry.points[bucket].ts < ev.timestamp) {
				entry.points[bucket] = {
					score: Math.round((ds.score ?? 0) * 100),
					ts: ev.timestamp,
				};
			}
		}
	}

	// Only include series that have data at >1 temp bucket.
	const series = Array.from(byModelDim.values()).filter(
		(s) => Object.keys(s.points).length > 1,
	);
	// Surface the thinking mode in the label so thinking-on and thinking-off
	// rows are distinguishable; thinking-off carries no suffix to keep
	// non-thinking-capable models (Llama, TinyLlama) labelled the same way
	// they were before.
	const labels = series.map(
		(s) =>
			`${s.model}${s.thinking === "on" ? " (think)" : ""} · ${s.dim}`,
	);
	const datasets = TEMP_SWEEP_BUCKETS.map((bucket) => ({
		label: bucket,
		data: series.map((s) => s.points[bucket]?.score ?? null),
		backgroundColor: TEMP_SWEEP_BUCKET_COLORS[bucket],
		borderColor: TEMP_SWEEP_BUCKET_COLORS[bucket],
		borderRadius: 3,
		barPercentage: 0.7,
		categoryPercentage: 0.85,
	}));

	return { labels, datasets };
}

/**
 * Pluck the embedding-dimension `EvalResult`s out of an eval's results
 * array. Reused by latency / throughput / cosine builders so the
 * embedding-only filter and finite-number guard live in one place.
 */
function embeddingResultsOf(ev) {
	const results = Array.isArray(ev?.results) ? ev.results : [];
	return results.filter(
		(r) =>
			r?.dimension === "embedding" &&
			typeof r.latencyMs === "number" &&
			Number.isFinite(r.latencyMs) &&
			r.latencyMs > 0,
	);
}

function median(values) {
	if (values.length === 0) return null;
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

/**
 * Median embedding latency (ms per text) per model. Pulls per-task
 * latencies from `EvalResult.latencyMs` for embedding-dim results in the
 * latest eval per model. Latest-wins is keyed by `modelId` only —
 * encoder models don't have a `thinking` mode so there's no
 * Qwen-style collision risk to split on.
 */
export function buildEmbeddingLatencyChartData(evals) {
	const latestByModel = new Map();
	for (const ev of evals ?? []) {
		const rs = embeddingResultsOf(ev);
		if (rs.length === 0) continue;
		const prev = latestByModel.get(ev.modelId);
		if (!prev || prev.timestamp < ev.timestamp) {
			latestByModel.set(ev.modelId, ev);
		}
	}
	const rows = Array.from(latestByModel.values())
		.map((ev) => ({
			modelId: ev.modelId,
			medianMs: median(embeddingResultsOf(ev).map((r) => r.latencyMs)),
		}))
		.filter((r) => r.medianMs != null)
		.sort((a, b) => (a.medianMs ?? 0) - (b.medianMs ?? 0));

	return {
		labels: rows.map((r) => r.modelId),
		datasets: [
			{
				label: "median ms / text",
				data: rows.map((r) => Number((r.medianMs ?? 0).toFixed(1))),
			},
		],
	};
}

/**
 * Embedding throughput (texts per second) per model, derived as
 * `1000 / medianLatencyMs`. Same latest-eval-per-modelId rules as the
 * latency builder — both surface the same underlying data, just with
 * the axis flipped to give an at-a-glance "higher is better" view.
 */
export function buildEmbeddingThroughputChartData(evals) {
	const latency = buildEmbeddingLatencyChartData(evals);
	return {
		labels: latency.labels,
		datasets: [
			{
				label: "texts / sec",
				data: latency.datasets[0].data.map((ms) =>
					ms > 0 ? Number((1000 / ms).toFixed(2)) : 0,
				),
			},
		],
	};
}

/**
 * Build a grouped bar chart of per-task embedding cosine values.
 *
 * Group key is `(modelId, taskId)` so each task's cosine across models
 * sits on a shared label. We surface the *raw* cosine in [-1, 1] from
 * `EvalResult.embeddingCosine` (set only when `dimension === "embedding"`),
 * not the post-threshold pass/fail `score`. Latest eval per (model, task)
 * wins.
 *
 * Returns `{ labels, datasets }` ready for Chart.js bar config; callers
 * supply colors and chart options.
 */
export function buildEmbeddingCosineChartData(evals) {
	const byModelTask = new Map();
	for (const ev of evals ?? []) {
		const results = Array.isArray(ev.results) ? ev.results : [];
		for (const r of results) {
			if (r?.dimension !== "embedding") continue;
			if (typeof r.embeddingCosine !== "number") continue;
			if (!Number.isFinite(r.embeddingCosine)) continue;
			const key = `${ev.modelId}::${r.taskId}`;
			const prev = byModelTask.get(key);
			if (!prev || prev.timestamp < ev.timestamp) {
				byModelTask.set(key, {
					modelId: ev.modelId,
					taskId: r.taskId,
					cosine: r.embeddingCosine,
					timestamp: ev.timestamp,
				});
			}
		}
	}

	// Tasks become labels (X axis), one dataset per model. This puts every
	// model's cosine for the same task next to each other so eyeballing
	// regressions across models is easy.
	const taskIds = Array.from(
		new Set(Array.from(byModelTask.values()).map((p) => p.taskId)),
	).sort();
	const modelIds = Array.from(
		new Set(Array.from(byModelTask.values()).map((p) => p.modelId)),
	).sort();

	const datasets = modelIds.map((modelId) => ({
		label: modelId,
		data: taskIds.map((tid) => {
			const p = byModelTask.get(`${modelId}::${tid}`);
			return p ? Number(p.cosine.toFixed(4)) : null;
		}),
	}));

	return { labels: taskIds, datasets };
}
