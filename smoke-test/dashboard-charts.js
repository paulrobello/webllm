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
