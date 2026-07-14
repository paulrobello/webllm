#!/usr/bin/env bun
// ARC-012 — JSPI/ABI build-time invariant check.
//
// Two shipped regressions are documented in CLAUDE.md's "Regression
// lessons" list, both caused by the JSPI promising-wrap invariant being
// enforced by comments rather than by construction:
//
//   #1 — A WASM build target shipped without `-sJSPI_EXPORTS` in its
//        link options (the MEM64 target was once missed). Every >3.5 GiB
//        model then failed at WebGPU init with
//        "SuspendError: trying to suspend without WebAssembly.promising".
//
//   #2 — A JSPI-promising-wrapped export whose TS binding consumes the
//        return value synchronously. Promising-wrap makes the export
//        always return Promise<T>; the wasm32 binding `result >>> 0`
//        silently coerces the unawaited Promise to 0 (leaked buffer
//        pointer); the wasm64 binding `Number(result)` coerces to NaN,
//        then `BigInt(NaN)` throws RangeError in the matching free path.
//        The historical culprit was `backend_alloc_ctx_tensors`.
//
// This script catches both at build time:
//
//   (a) Parses `src/wasm/CMakeLists.txt` for the `JSPI_EXPORTS` list.
//   (b) Regression #1 guard — asserts EVERY `add_executable(webllm-wasm*)`
//       target in that file links with `-sJSPI_EXPORTS`.
//   (c) Regression #2 guard — for each export name, finds its binding
//       call site(s) in the inference layer and asserts the call is
//       `await`ed, `.then`-chained, or wrapped in `callWithAsyncify`.
//       Unawaited call sites that are not on the pinned allowlist fail.
//       Additionally — and regardless of the allowlist — any call site
//       whose return is synchronously coerced via `Number(...)`, `BigInt
//       (...)`, or `>>>` (the exact regression-#2 pointer-marshalling
//       signature) ALWAYS fails.
//
// INVERSE-RULE CAVEAT (not mechanically checkable): this script catches
// "JSPI export consumed as a synchronous value" but CANNOT catch the
// inverse — an export on the list whose binding does not actually
// suspend (the `backend_alloc_ctx_tensors` removal rationale). That
// inverse remains a human-audit responsibility; CLAUDE.md's JSPI lesson
// is the governing doc. To support that human audit, the script prints
// the full checked export list on success so a reviewer can eyeball
// each name's binding for the inverse.
//
// See AUDIT.md ARC-012 and the remediation playbook in
// AUDIT-REMEDIATION-PLAN.md for context. Mirrors the pinned-ratchet
// shape of scripts/check-skip-count.ts (allowlist below is the
// accepted-fire-and-forget surface; shrink it, never silently grow it).

import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..");
const CMAKE_PATH =
	process.env.WEBLLM_JSPI_CMAKE ??
	join(ROOT, "src/wasm/CMakeLists.txt");
// ggml-wasm.ts owns the low-level bindings (webgpu_init, ctx_*, backend_*,
// graph_compute, async-tensor-get); llama-bridge.ts owns the webllm_*
// family. Both files must be scanned to cover the full JSPI surface.
const DEFAULT_BINDINGS = [
	join(ROOT, "src/inference/ggml-wasm.ts"),
	join(ROOT, "src/inference/llama-bridge.ts"),
];
const BINDING_PATHS =
	process.env.WEBLLM_JSPI_BINDINGS?.split(":").filter(Boolean) ??
	DEFAULT_BINDINGS;

// Pinned allowlist of JSPI exports whose TS bindings are intentionally
// NOT awaited at the call site. Every entry must carry a one-line
// rationale. Direction of travel: SHRINK this list (await the call),
// never grow it without an explicit justification — a growing fire-and-
// forget surface is the exact erosion this check exists to catch.
// NOTE: even allowlisted names hard-fail if a call site ever coerces the
// return via Number()/BigInt()/>>> (the regression-#2 signature).
const FIRE_AND_FORGET_ALLOWLIST: Record<string, string> = {
	ctx_free: "void context teardown; no return value consumed",
	backend_buffer_free: "void GPUBuffer free; no return value consumed",
	backend_tensor_set: "void memcpy into backend buffer; no return value",
	backend_tensor_set3: "void 3-tensor memcpy; no return value",
	webllm_free_model: "void model teardown; no return value consumed",
	webllm_free_context: "void context teardown; no return value consumed",
	ctx_create:
		"returns int32 error code, compared via `rc < 0` (NOT pointer-coerced). Re-audit if it ever becomes pointer-return.",
};

interface Violation {
	exportName: string;
	file: string;
	line: number; // 1-based
	text: string;
	reason: string;
	hard: boolean;
}

// ── (a) Parse JSPI_EXPORTS + (b) verify per-target link flag ──────────────

function parseCmake(path: string): {
	jspiExports: string[];
	targets: { name: string; hasJspiFlag: boolean; line: number }[];
} {
	const src = readFileSync(path, "utf8");
	const lines = src.split("\n");

	// string(CONCAT JSPI_EXPORTS  "a,"  "b,c,"  ... )  — join the quoted
	// pieces between the header line and the closing paren, then split on
	// comma/whitespace.
	const jspiExports: string[] = [];
	let inJspi = false;
	let buf = "";
	for (const ln of lines) {
		if (/string\s*\(\s*CONCAT\s+JSPI_EXPORTS\b/.test(ln)) {
			inJspi = true;
			buf += ln;
			continue;
		}
		if (inJspi) {
			buf += `\n${ln}`;
			// The string(CONCAT ...) closes at the first ")" at the end of
			// a line following the opening. Heuristic: a line that is just
			// ")" (possibly trailing whitespace) ends the block.
			if (/^\s*\)\s*$/.test(ln)) {
				break;
			}
		}
	}
	// Extract every double-quoted segment from the captured block and
	// concatenate (mirrors CMake string(CONCAT ...) semantics).
	const quoted = buf.match(/"([^"]*)"/g) ?? [];
	const joined = quoted.map((s) => s.slice(1, -1)).join("");
	for (const piece of joined.split(/[,\s]+/)) {
		const name = piece.trim();
		if (name.length > 0) jspiExports.push(name);
	}

	// Find every add_executable(webllm-wasm* ...) and, for each, scan
	// forward ~40 lines for a target_link_options(...) block that
	// contains -sJSPI_EXPORTS. This is a small file (286 lines) so the
	// windowed scan is robust.
	const targets: { name: string; hasJspiFlag: boolean; line: number }[] =
		[];
	for (let i = 0; i < lines.length; i++) {
		const m =
			/add_executable\s*\(\s*(webllm-wasm[A-Za-z0-9_-]*)\b/.exec(
				lines[i],
			);
		if (!m) continue;
		const name = m[1];
		const window = lines.slice(i, Math.min(lines.length, i + 40));
		// Require the flag inside a target_link_options block for THIS
		// target (not the EXPORTED_FUNCTIONS line, which also lives
		// nearby). We scan from the `target_link_options(<name> ...)` that
		// appears after this add_executable.
		let hasJspiFlag = false;
		let inLinkOpts = false;
		for (const wl of window) {
			if (/target_link_options\s*\(/.test(wl)) inLinkOpts = true;
			if (inLinkOpts && /-sJSPI_EXPORTS=/.test(wl)) {
				hasJspiFlag = true;
				break;
			}
			if (inLinkOpts && /^\s*\)\s*$/.test(wl)) break;
		}
		targets.push({ name, hasJspiFlag, line: i + 1 });
	}

	return { jspiExports, targets };
}

// ── (c) Per-export binding call-site check ────────────────────────────────

// Context window (lines above/below a call site) scanned for the
// `await` / `callWithAsyncify` / `.then` markers that mark a call as
// properly async. Multi-line `callWithAsyncify(() =>\n  this.m._x())`
// needs a few lines of lookback.
const CTX_LINES = 3;

function isCallProperlyAsync(
	lines: string[],
	idx: number,
): boolean {
	const lo = Math.max(0, idx - CTX_LINES);
	const hi = Math.min(lines.length, idx + CTX_LINES + 1);
	const ctx = lines.slice(lo, hi).join("\n");
	return (
		/\bawait\b/.test(ctx) ||
		/\bcallWithAsyncify\b/.test(ctx) ||
		/\.then\s*\(/.test(ctx)
	);
}

// The exact regression-#2 signature: the JSPI call's return value is
// synchronously coerced to a number/bigint. Two shapes:
//   Number(this.m._x(...)) / BigInt(this.m._x(...))  — coercion wraps call
//   this.m._x(...) >>> N                              — call result shifted
// `BigInt`/`Number` appearing as a CALL ARGUMENT (e.g. `this.m._ctx_create(BigInt(n))`)
// is intentionally NOT matched — the coercion must wrap the call result.
function syncCoercionOnCall(
	line: string,
	exportName: string,
): boolean {
	const callRe = `(?:this\\.m|mod)\\._${escapeRe(exportName)}\\s*\\(`;
	// Shape 1: coercion wraps the call — `Number(` or `BigInt(` precedes
	// `this.m._name(` (allowing whitespace) on the same line.
	const wraps = new RegExp(
		`(?:Number|BigInt)\\s*\\(\\s*${callRe}`,
	).test(line);
	// Shape 2: call's closing paren is followed by ` >>> `.
	const shifted = new RegExp(
		`${callRe.replace("\\(", "\\(")}[^)]*\\)\\s*>>>`,
	).test(line);
	return wraps || shifted;
}

function escapeRe(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findCallSites(
	file: string,
	exportName: string,
): { line: number; text: string }[] {
	const src = readFileSync(file, "utf8");
	const lines = src.split("\n");
	// Match `_<name>(` — the leading underscore + immediate paren
	// distinguishes a call from a type-decl line like `_name: (...)`
	// (which has `: ` between the name and paren, so no match here).
	const callRe = new RegExp(`_${escapeRe(exportName)}\\(`);
	const hits: { line: number; text: string }[] = [];
	for (let i = 0; i < lines.length; i++) {
		if (callRe.test(lines[i])) {
			hits.push({ line: i + 1, text: lines[i].trim() });
		}
	}
	return hits;
}

// ── Main ──────────────────────────────────────────────────────────────────

const { jspiExports, targets } = parseCmake(CMAKE_PATH);

const violations: Violation[] = [];
const reviewNotes: Violation[] = [];
const perExport: {
	name: string;
	sites: number;
	awaited: number;
}[] = [];

for (const name of jspiExports) {
	let siteCount = 0;
	let awaitedCount = 0;
	for (const file of BINDING_PATHS) {
		let src: string;
		try {
			src = readFileSync(file, "utf8");
		} catch {
			// A configured scratch run may point at a single file; skip
			// missing ones silently.
			continue;
		}
		const lines = src.split("\n");
		const callRe = new RegExp(`_${escapeRe(name)}\\(`);
		for (let i = 0; i < lines.length; i++) {
			if (!callRe.test(lines[i])) continue;
			// Skip type-declaration lines (`_name:`) — defensive; the
			// `_<name>(` anchor already excludes them, but the WasmPtr
			// interface blocks use `_name: (...)` which won't match.
			siteCount++;
			const properlyAsync = isCallProperlyAsync(lines, i);
			if (properlyAsync) {
				awaitedCount++;
				continue;
			}
			// Hard-fail regression-#2 signature regardless of allowlist.
			if (syncCoercionOnCall(lines[i], name)) {
				violations.push({
					exportName: name,
					file,
					line: i + 1,
					text: lines[i].trim(),
					reason:
						"JSPI export return synchronously coerced (Number/BigInt/>>>) without await — regression #2 signature (cf. historical backend_alloc_ctx_tensors)",
					hard: true,
				});
				continue;
			}
			// Unawaited, not coerced. Allowlisted fire-and-forget → review
			// note (printed); otherwise hard violation.
			if (FIRE_AND_FORGET_ALLOWLIST[name] !== undefined) {
				reviewNotes.push({
					exportName: name,
					file,
					line: i + 1,
					text: lines[i].trim(),
					reason: `allowlisted fire-and-forget: ${FIRE_AND_FORGET_ALLOWLIST[name]}`,
					hard: false,
				});
			} else {
				violations.push({
					exportName: name,
					file,
					line: i + 1,
					text: lines[i].trim(),
					reason:
						"JSPI export call is not awaited / .then-chained / wrapped in callWithAsyncify, and is not on the fire-and-forget allowlist",
					hard: true,
				});
			}
		}
	}
	perExport.push({ name, sites: siteCount, awaited: awaitedCount });
}

// Regression #1 guard: every webllm-wasm* executable must link -sJSPI_EXPORTS.
const missingFlag = targets.filter((t) => !t.hasJspiFlag);

// ── Report ────────────────────────────────────────────────────────────────

const tag = "[check-jspi]";

if (targets.length === 0) {
	console.error(
		`${tag} FAIL: no \`add_executable(webllm-wasm*)\` targets found in ${CMAKE_PATH}.`,
	);
	console.error(
		`${tag} Either CMakeLists.txt restructured or the parser is stale.`,
	);
	process.exit(1);
}

if (missingFlag.length > 0) {
	console.error(
		`${tag} FAIL: regression #1 — webllm-wasm* target(s) missing \`-sJSPI_EXPORTS\` link flag:`,
	);
	for (const t of missingFlag) {
		console.error(`  ${t.name} (add_executable at line ${t.line})`);
	}
	console.error(
		`${tag} Each WASM build target must carry \`-sJSPI_EXPORTS=\${JSPI_EXPORTS}\` whenever GGML_WEBGPU_JSPI is on (CLAUDE.md regression lesson).`,
	);
	process.exit(1);
}

if (violations.length > 0) {
	console.error(
		`${tag} FAIL: ${violations.length} JSPI binding invariant violation(s):`,
	);
	for (const v of violations) {
		console.error(
			`  ${v.file}:${v.line}  (_${v.exportName})  ${v.reason}`,
		);
		console.error(`    > ${v.text}`);
	}
	console.error(
		`${tag} Either await the call (preferred), wrap it in callWithAsyncify, or — if it is a genuine void fire-and-forget — add it to FIRE_AND_FORGET_ALLOWLIST in scripts/check-jspi-exports.ts with a one-line rationale.`,
	);
	process.exit(1);
}

// Success.
const targetsVerified = targets.map((t) => t.name).join(", ");
console.log(
	`${tag} OK: ${jspiExports.length} JSPI exports checked across ${targets.length} target(s): ${targetsVerified}.`,
);
console.log(
	`${tag} Targets verified to link -sJSPI_EXPORTS: ${targets.map((t) => t.name).join(", ")}.`,
);

// Inverse-audit listing: print each export + its call-site coverage so a
// human can spot (a) an export with zero bindings (dead list entry?) and
// (b) an export whose binding does not actually suspend (the inverse rule
// this script cannot check mechanically).
console.log(`${tag} JSPI_EXPORTS list (for human inverse-audit):`);
for (const e of perExport) {
	const allow = FIRE_AND_FORGET_ALLOWLIST[e.name];
	const suffix =
		e.sites === 0
			? "no binding call site in scanned files (may be dynamic / out-of-scope)"
			: `${e.awaited}/${e.sites} call site(s) properly async`;
	const allowSuffix = allow ? ` · ALLOWLISTED (${allow})` : "";
	console.log(`  ${e.name.padEnd(36)} ${suffix}${allowSuffix}`);
}

if (reviewNotes.length > 0) {
	console.log(
		`${tag} ${reviewNotes.length} allowlisted fire-and-forget call site(s) (review, not failing):`,
	);
	for (const v of reviewNotes) {
		console.log(`  ${v.file}:${v.line}  (_${v.exportName})  ${v.reason}`);
		console.log(`    > ${v.text}`);
	}
}

console.log(
	`${tag} Inverse-rule reminder: this check cannot catch a non-suspending export wrongly on the list (the historical backend_alloc_ctx_tensors case). Audit the list above manually; CLAUDE.md's JSPI lesson remains governing.`,
);
