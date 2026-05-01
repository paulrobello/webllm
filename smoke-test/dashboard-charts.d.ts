// Ambient declaration for the dashboard-charts JS helpers used by tests.

export const DIM_NAMES: readonly string[];
export const TEMP_SWEEP_BUCKETS: readonly string[];
export const TEMP_SWEEP_BUCKET_COLORS: Record<string, string>;

interface ChartData {
	labels: string[];
	datasets: Array<{
		label: string;
		data: Array<number | null>;
		[key: string]: unknown;
	}>;
}

export function tempBucket(temperature: number): string;
export function buildTempSweepChartData(evals: unknown): ChartData;
export function buildEmbeddingLatencyChartData(evals: unknown): ChartData;
export function buildEmbeddingThroughputChartData(evals: unknown): ChartData;
export function buildEmbeddingCosineChartData(evals: unknown): ChartData;
