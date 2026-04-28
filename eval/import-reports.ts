/**
 * Backfill report files under eval/reports/ into the live dashboard SQLite DB.
 *
 * Wired for two cases the live ingest does NOT handle on its own:
 *   1. Browser/Bun runs that completed while the dashboard was offline.
 *   2. Older JSON reports that predate the dashboard or that were written
 *      with WEBLLM_LIVE_BENCH_URL unset.
 *
 * Idempotent: existing run_id / eval_id rows are fetched first and skipped.
 *
 * Importable shapes (auto-classified by file content):
 *   • eval/reports/smoke-runs/*.json — SmokeRunRecord (speed run)
 *   • eval/reports/<top-level>.json  — EvalReport (eval run with results[])
 *
 * Skipped (no run/eval semantics):
 *   • eval/reports/embed-perf-*\/*.json (raw timing traces)
 *   • eval/reports/embed-perf-baseline-cosine.json, perf-baseline.json
 *   • eval/reports/<bench-dir>/*.{txt,log,md} (bench-harness scratch output)
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { basename, extname, join, sep } from "node:path";
import { parseArgs } from "node:util";
import { resolveLiveBenchUrl } from "./live-client.js";
import type { PersistedEvalReport } from "./live-events.js";
import type { SmokeRunRecord } from "./smoke-runs.js";

const DEFAULT_BASE_URL = "http://localhost:8033";
const DEFAULT_REPORTS_DIR = "eval/reports";

interface Summary {
	scanned: number;
	importedRuns: number;
	importedEvals: number;
	skippedExisting: number;
	skippedShape: number;
	failed: number;
}

function printUsage(): void {
	console.log(`Usage: bun run eval/import-reports.ts [options]

Walks a reports directory, classifies each JSON file as a speed run or eval
report, and POSTs new ones to the live dashboard. Already-imported records
(matched by runId / evalId) are skipped.

Options:
  --dir <path>          Reports directory (default: ${DEFAULT_REPORTS_DIR})
  --url <base-url>      Dashboard base URL (default: $WEBLLM_LIVE_BENCH_URL or ${DEFAULT_BASE_URL})
  --dry-run             Classify and dedup, but do not POST
  --verbose             Print one line per skip/import
  --help, -h            This message
`);
}

async function main(): Promise<void> {
	const { values } = parseArgs({
		options: {
			dir: { type: "string" },
			url: { type: "string" },
			"dry-run": { type: "boolean" },
			verbose: { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	const reportsDir = values.dir ?? DEFAULT_REPORTS_DIR;
	const baseUrl = values.url ?? resolveLiveBenchUrl() ?? DEFAULT_BASE_URL;
	const dryRun = !!values["dry-run"];
	const verbose = !!values.verbose;

	if (!isDirectory(reportsDir)) {
		console.error(`reports directory not found: ${reportsDir}`);
		process.exit(1);
	}

	const reachable = await checkReachable(baseUrl);
	if (!reachable.ok) {
		console.error(
			`dashboard not reachable at ${baseUrl}: ${reachable.reason}\n` +
				`start it with \`make dashboard-serve\` (or pass --url <base-url>).`,
		);
		process.exit(1);
	}
	console.log(`dashboard ok at ${baseUrl}`);

	const existing = await fetchExistingIds(baseUrl);
	console.log(
		`existing in dashboard: ${existing.runIds.size} runs, ${existing.evalIds.size} evals`,
	);

	const files = walkJson(reportsDir);
	console.log(`scanned ${files.length} JSON files under ${reportsDir}`);

	const summary: Summary = {
		scanned: files.length,
		importedRuns: 0,
		importedEvals: 0,
		skippedExisting: 0,
		skippedShape: 0,
		failed: 0,
	};

	for (const file of files) {
		const rel = file.startsWith(reportsDir)
			? file.slice(reportsDir.length).replace(/^[\\/]+/, "")
			: file;
		let parsed: unknown;
		try {
			parsed = JSON.parse(readFileSync(file, "utf8"));
		} catch (err) {
			summary.failed++;
			console.warn(
				`  parse-fail ${rel}: ${err instanceof Error ? err.message : String(err)}`,
			);
			continue;
		}

		const classified = classify(parsed, file);
		if (classified.kind === "skip") {
			summary.skippedShape++;
			if (verbose) console.log(`  skip-shape ${rel} (${classified.reason})`);
			continue;
		}

		if (classified.kind === "run") {
			const id = classified.record.runId;
			if (existing.runIds.has(id)) {
				summary.skippedExisting++;
				if (verbose) console.log(`  skip-existing run ${rel} (${id})`);
				continue;
			}
			if (dryRun) {
				summary.importedRuns++;
				console.log(`  [dry-run] would import run ${rel} (${id})`);
				continue;
			}
			const ok = await postIngest(baseUrl, "run_complete", classified.record);
			if (ok) {
				summary.importedRuns++;
				existing.runIds.add(id);
				console.log(`  + run  ${rel} → ${id}`);
			} else {
				summary.failed++;
			}
			continue;
		}

		// classified.kind === "eval"
		const id = classified.report.evalId;
		if (existing.evalIds.has(id)) {
			summary.skippedExisting++;
			if (verbose) console.log(`  skip-existing eval ${rel} (${id})`);
			continue;
		}
		if (dryRun) {
			summary.importedEvals++;
			console.log(`  [dry-run] would import eval ${rel} (${id})`);
			continue;
		}
		const ok = await postIngest(baseUrl, "eval_complete", classified.report);
		if (ok) {
			summary.importedEvals++;
			existing.evalIds.add(id);
			console.log(`  + eval ${rel} → ${id}`);
		} else {
			summary.failed++;
		}
	}

	console.log("");
	console.log("import summary:");
	console.log(`  scanned          ${summary.scanned}`);
	console.log(`  imported runs    ${summary.importedRuns}`);
	console.log(`  imported evals   ${summary.importedEvals}`);
	console.log(`  skipped existing ${summary.skippedExisting}`);
	console.log(`  skipped shape    ${summary.skippedShape}`);
	console.log(`  failed           ${summary.failed}`);

	if (summary.failed > 0) process.exit(2);
}

function isDirectory(path: string): boolean {
	try {
		return statSync(path).isDirectory();
	} catch {
		return false;
	}
}

async function checkReachable(
	baseUrl: string,
): Promise<{ ok: true } | { ok: false; reason: string }> {
	try {
		const url = new URL("/health", baseUrl).toString();
		const controller = new AbortController();
		const to = setTimeout(() => controller.abort(), 3000);
		try {
			const res = await fetch(url, { signal: controller.signal });
			if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
			return { ok: true };
		} finally {
			clearTimeout(to);
		}
	} catch (err) {
		return {
			ok: false,
			reason: err instanceof Error ? err.message : String(err),
		};
	}
}

interface ExistingIds {
	runIds: Set<string>;
	evalIds: Set<string>;
}

async function fetchExistingIds(baseUrl: string): Promise<ExistingIds> {
	const runIds = new Set<string>();
	const evalIds = new Set<string>();
	try {
		const r = await fetch(new URL("/runs", baseUrl).toString());
		if (r.ok) {
			const body = (await r.json()) as { runs?: Array<{ runId?: string }> };
			for (const row of body.runs ?? []) {
				if (typeof row.runId === "string") runIds.add(row.runId);
			}
		}
	} catch (err) {
		console.warn(
			`fetch /runs failed: ${err instanceof Error ? err.message : String(err)}`,
		);
	}
	try {
		const r = await fetch(new URL("/evals", baseUrl).toString());
		if (r.ok) {
			const body = (await r.json()) as {
				evals?: Array<{ evalId?: string }>;
			};
			for (const row of body.evals ?? []) {
				if (typeof row.evalId === "string") evalIds.add(row.evalId);
			}
		}
	} catch (err) {
		console.warn(
			`fetch /evals failed: ${err instanceof Error ? err.message : String(err)}`,
		);
	}
	return { runIds, evalIds };
}

type Classified =
	| { kind: "run"; record: SmokeRunRecord & { runId: string } }
	| { kind: "eval"; report: PersistedEvalReport }
	| { kind: "skip"; reason: string };

/**
 * Decide what an arbitrary JSON file represents. Two real shapes survive
 * here; everything else is shape-skipped.
 *
 *   • SmokeRunRecord — has `schemaVersion` and `oneShot`/`interactive`.
 *   • EvalReport     — has `results[]` plus `modelId` + `totalTasks`.
 */
function classify(parsed: unknown, file: string): Classified {
	if (!parsed || typeof parsed !== "object") {
		return { kind: "skip", reason: "not an object" };
	}
	const obj = parsed as Record<string, unknown>;

	const stem = basename(file, extname(file));
	const isInSmokeRunsDir = file.split(sep).includes("smoke-runs");

	// Speed run? smoke-runs JSONs always have a SmokeRunRecord shape.
	const looksLikeRun =
		typeof obj.schemaVersion === "number" &&
		typeof obj.model === "string" &&
		typeof obj.timestamp === "string" &&
		(obj.oneShot !== undefined || obj.interactive !== undefined);
	if (looksLikeRun) {
		const record = obj as Record<string, unknown> & SmokeRunRecord;
		const runId =
			typeof obj.runId === "string" && obj.runId.length > 0
				? (obj.runId as string)
				: stem;
		return {
			kind: "run",
			record: { ...(record as SmokeRunRecord), runId },
		};
	}

	// Eval report?
	const looksLikeEval =
		typeof obj.modelId === "string" &&
		typeof obj.timestamp === "string" &&
		typeof obj.totalTasks === "number" &&
		Array.isArray(obj.results);
	if (looksLikeEval) {
		const evalId =
			typeof obj.evalId === "string" && obj.evalId.length > 0
				? (obj.evalId as string)
				: stem;
		const overall = typeof obj.overall === "number" ? obj.overall : 0;
		const dimensions =
			obj.dimensions && typeof obj.dimensions === "object"
				? (obj.dimensions as Record<string, unknown>)
				: {};
		const report = {
			...(obj as Record<string, unknown>),
			evalId,
			overall,
			dimensions,
		} as unknown as PersistedEvalReport;
		return { kind: "eval", report };
	}

	if (isInSmokeRunsDir) {
		return { kind: "skip", reason: "smoke-runs file with non-run shape" };
	}
	return { kind: "skip", reason: "unrecognized shape" };
}

function walkJson(root: string): string[] {
	const out: string[] = [];
	const stack = [root];
	while (stack.length > 0) {
		const dir = stack.pop();
		if (!dir) break;
		let entries: string[];
		try {
			entries = readdirSync(dir);
		} catch {
			continue;
		}
		for (const name of entries) {
			const full = join(dir, name);
			let st: ReturnType<typeof statSync>;
			try {
				st = statSync(full);
			} catch {
				continue;
			}
			if (st.isDirectory()) {
				stack.push(full);
				continue;
			}
			if (st.isFile() && name.endsWith(".json")) {
				out.push(full);
			}
		}
	}
	out.sort();
	return out;
}

async function postIngest(
	baseUrl: string,
	kind: "run_complete" | "eval_complete",
	body: unknown,
): Promise<boolean> {
	try {
		const url = new URL(`/ingest?kind=${kind}`, baseUrl).toString();
		const res = await fetch(url, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
		});
		if (!res.ok) {
			const text = await res.text().catch(() => "");
			console.warn(
				`  ingest ${kind} HTTP ${res.status} ${res.statusText}: ${text.slice(0, 200)}`,
			);
			return false;
		}
		return true;
	} catch (err) {
		console.warn(
			`  ingest ${kind} threw: ${err instanceof Error ? err.message : String(err)}`,
		);
		return false;
	}
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
