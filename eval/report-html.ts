import type { EvalReport } from "./types.js";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Generate a self-contained HTML report from an EvalReport, embedding the
 * JSON data directly into the template for offline viewing.
 */
export function generateHtmlReport(report: EvalReport, outputPath?: string): string {
	const templatePath = resolve(__dirname, "report-template.html");
	const template = readFileSync(templatePath, "utf-8");
	const jsonStr = JSON.stringify(report)
		.replace(/</g, "\\u003c")
		.replace(/>/g, "\\u003e")
		.replace(/&/g, "\\u0026");

	const injected = template.replace(
		"<script>",
		`<script>window.EVAL_DATA = ${jsonStr};`,
	);

	if (outputPath) {
		mkdirSync(dirname(outputPath), { recursive: true });
		writeFileSync(outputPath, injected, "utf-8");
	}

	return injected;
}
