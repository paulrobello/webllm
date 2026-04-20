import type { EvalReport, EvalResult, EvalDimension } from "./types.js";
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const DIMENSION_WIDTH = 22;
const SCORE_WIDTH = 8;
const PASSED_WIDTH = 8;
const LATENCY_LABEL = "AVG LATENCY";

/** Serialize the report to formatted JSON. */
export function formatJson(report: EvalReport): string {
	return JSON.stringify(report, null, 2);
}

/** Format the report as a human-readable terminal table. */
export function formatTable(report: EvalReport): string {
	const lines: string[] = [];

	lines.push("Model Evaluation Report");
	lines.push(
		`Model: ${report.modelId} | ${report.timestamp}`,
	);
	lines.push("");

	// Header
	const header =
		"DIMENSION".padEnd(DIMENSION_WIDTH) +
		"SCORE".padEnd(SCORE_WIDTH) +
		"PASSED".padEnd(PASSED_WIDTH) +
		LATENCY_LABEL;
	lines.push(header);
	lines.push("-".repeat(header.length));

	// Dimension rows
	for (const [dim, ds] of Object.entries(report.dimensions)) {
		const dimLabel = dim.padEnd(DIMENSION_WIDTH);
		const scoreLabel = ds.score.toFixed(2).padEnd(SCORE_WIDTH);
		const passedLabel = `${ds.passed}/${ds.total}`.padEnd(PASSED_WIDTH);
		const latencyLabel = `${ds.avgLatencyMs}ms`;
		lines.push(dimLabel + scoreLabel + passedLabel + latencyLabel);
	}

	lines.push("");

	// Overall line
	const totalPassed = report.results.filter((r) => r.score >= 0.5).length;
	lines.push(
		`Overall: ${report.overall.toFixed(2)} (${totalPassed}/${report.totalTasks} tasks)`,
	);

	// Failures
	const failures = report.results.filter((r) => r.score < 0.5);
	if (failures.length > 0) {
		lines.push("");
		lines.push("FAILURES:");
		for (const f of failures) {
			const reason = f.error ?? truncate(f.modelOutput, 40);
			lines.push(
				`  ${f.taskId}  score: ${f.score.toFixed(2)}  "${reason}"`,
			);
		}
	}

	return lines.join("\n");
}

/** Write the report to disk as JSON and print the table to stdout. Returns the file path. */
export function writeReport(report: EvalReport, dir?: string): string {
	const reportDir = dir ?? "eval/reports";
	mkdirSync(reportDir, { recursive: true });

	const timestamp = report.timestamp.replace(/[:.]/g, "-");
	const filename = `${timestamp}-${report.modelId}.json`;
	const filepath = join(reportDir, filename);

	writeFileSync(filepath, formatJson(report), "utf-8");

	// Print table to stdout
	console.log(formatTable(report));

	return filepath;
}

/** Truncate a string to maxLength, appending ellipsis if needed. */
function truncate(str: string, maxLength: number): string {
	if (str.length <= maxLength) return str;
	return `${str.slice(0, maxLength - 3)}...`;
}
