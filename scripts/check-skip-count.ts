#!/usr/bin/env bun
// QA-004 — Static skip-count ratchet.
//
// Counts `skipIf(` occurrences across tests/ and fails if the count
// exceeds the pinned ceiling. The ceiling is the post-QA-004 count:
// every remaining skip is a genuine environment gate (WebGPU under Bun,
// missing optional fixture file) that fake-indexeddb / local Bun cannot
// satisfy. The IndexedDB-gated suites were un-skipped in QA-004 via a
// fake-indexeddb polyfill; those skips are gone and must stay gone.
//
// Direction of travel: LOWER the ceiling when skips are removed (e.g. by
// gaining a WebGPU-in-CI path, deleting a dead suite, or polyfilling
// another environment dependency). NEVER raise it without an explicit
// justification — a rising skip count is the exact erosion signal this
// ratchet exists to catch.
//
// See AUDIT.md QA-004 and the remediation playbook in
// AUDIT-REMEDIATION-PLAN.md for context.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const TESTS_DIR = join(import.meta.dir, "..", "tests");
const PINNED_CEILING = 13;

function listTestFiles(dir: string): string[] {
	const out: string[] = [];
	for (const name of readdirSync(dir)) {
		const path = join(dir, name);
		const st = statSync(path);
		if (st.isDirectory()) {
			out.push(...listTestFiles(path));
		} else if (name.endsWith(".test.ts") || name.endsWith(".test.tsx")) {
			out.push(path);
		}
	}
	return out;
}

function countSkipIf(files: string[]): { count: number; hits: string[] } {
	let count = 0;
	const hits: string[] = [];
	for (const file of files) {
		const src = readFileSync(file, "utf8");
		const lines = src.split("\n");
		for (let i = 0; i < lines.length; i++) {
			if (lines[i].includes("skipIf(")) {
				count++;
				hits.push(`${file}:${i + 1}: ${lines[i].trim()}`);
			}
		}
	}
	return { count, hits };
}

const files = listTestFiles(TESTS_DIR);
const { count, hits } = countSkipIf(files);

if (count > PINNED_CEILING) {
	console.error(
		`[check-skip-count] FAIL: ${count} skipIf( occurrences in tests/ exceed the pinned ceiling of ${PINNED_CEILING}.`,
	);
	console.error(
		"[check-skip-count] Skip-count ratchet tripped — see scripts/check-skip-count.ts header for direction-of-travel.",
	);
	console.error("[check-skip-count] Current skips:");
	for (const h of hits) {
		console.error(`  ${h}`);
	}
	console.error(
		"[check-skip-count] Either remove the new skip (polyfill the env dep, delete the dead suite) or, if a genuine new env gate landed, lower this ceiling never raise it — justify explicitly in the commit message.",
	);
	process.exit(1);
}

console.log(
	`[check-skip-count] OK: ${count} skipIf( occurrences in tests/ (ceiling ${PINNED_CEILING}).`,
);
