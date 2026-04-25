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
	// Group evals by (modelId, dimension, tempBucket).
	const byModelDim = new Map();
	for (const ev of evals) {
		const bucket = tempBucket(ev.params?.temperature);
		if (!bucket) continue;
		for (const [dim, ds] of Object.entries(ev.dimensions ?? {})) {
			if (!ds || !ds.total) continue;
			const k = `${ev.modelId}::${dim}`;
			if (!byModelDim.has(k)) {
				byModelDim.set(k, { model: ev.modelId, dim, points: {} });
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
	const labels = series.map((s) => `${s.model} · ${s.dim}`);
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
