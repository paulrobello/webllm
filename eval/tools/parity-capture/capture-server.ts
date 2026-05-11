#!/usr/bin/env bun
/**
 * Capture server for the webllm vs HF parity-capture pipeline.
 *
 * Accepts POST /capture with a JSON body matching the schema in
 * eval/tools/parity-capture/README.md and writes it to
 * <run-dir>/webllm.json. Port 8035 by default (registered in
 * ~/.claude/used_ports.md). Single-file, Bun-native HTTP only.
 *
 *   bun run eval/tools/parity-capture/capture-server.ts \
 *     --run-dir eval/reports/parity-tinyllama-2026-05-11
 *
 * The browser harness (smoke-test/parity-capture.html) POSTs to
 * http://localhost:8035/capture with `{ runDir?, payload }`. If the
 * request body carries `runDir`, it must be a sibling slug of the
 * configured base (allows multi-model runs against one server). The
 * default is to write `webllm.json` under the CLI-configured run-dir.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { resolve as resolvePath, join as joinPath, dirname } from "node:path";

interface CliArgs {
	port: number;
	runDir: string;
}

function parseArgs(argv: string[]): CliArgs {
	let port = 8035;
	let runDir = "";
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === "--port") {
			port = Number(argv[++i]);
			if (!Number.isFinite(port) || port <= 0) {
				throw new Error(`invalid --port: ${argv[i]}`);
			}
		} else if (a === "--run-dir") {
			runDir = argv[++i] ?? "";
		} else if (a === "--help" || a === "-h") {
			console.log(
				"Usage: bun run capture-server.ts --run-dir <dir> [--port 8035]",
			);
			process.exit(0);
		} else {
			throw new Error(`unknown arg: ${a}`);
		}
	}
	if (!runDir) {
		throw new Error("--run-dir is required");
	}
	return { port, runDir: resolvePath(runDir) };
}

const { port, runDir } = parseArgs(process.argv.slice(2));

mkdirSync(runDir, { recursive: true });
console.log(`[capture-server] run-dir: ${runDir}`);
console.log(`[capture-server] listening on http://localhost:${port}`);
console.log(
	`[capture-server] POST /capture  → writes <run-dir>/webllm.json (or override via body.outFile)`,
);

const corsHeaders: Record<string, string> = {
	"access-control-allow-origin": "*",
	"access-control-allow-methods": "POST, OPTIONS",
	"access-control-allow-headers": "content-type",
};

Bun.serve({
	port,
	async fetch(req: Request) {
		const url = new URL(req.url);
		if (req.method === "OPTIONS") {
			return new Response(null, { status: 204, headers: corsHeaders });
		}
		if (url.pathname === "/health" && req.method === "GET") {
			return new Response("ok", { status: 200, headers: corsHeaders });
		}
		if (url.pathname !== "/capture" || req.method !== "POST") {
			return new Response("not found", { status: 404, headers: corsHeaders });
		}

		let body: unknown;
		try {
			body = await req.json();
		} catch (err) {
			return new Response(`invalid json: ${(err as Error).message}`, {
				status: 400,
				headers: corsHeaders,
			});
		}

		const obj = body as { outFile?: string; payload?: unknown } & Record<
			string,
			unknown
		>;
		// Accept either { payload: {...} } or a flat capture object.
		const payload = obj.payload ?? obj;
		const outFile = typeof obj.outFile === "string" ? obj.outFile : "webllm.json";

		// outFile is restricted to a filename under run-dir (no path traversal).
		if (outFile.includes("..") || outFile.includes("/")) {
			return new Response(
				`invalid outFile (must be a bare filename): ${outFile}`,
				{ status: 400, headers: corsHeaders },
			);
		}
		const dest = joinPath(runDir, outFile);
		mkdirSync(dirname(dest), { recursive: true });
		writeFileSync(dest, JSON.stringify(payload, null, 2));
		const size = Bun.file(dest).size;
		console.log(`[capture-server] wrote ${dest} (${size} bytes)`);
		return new Response(JSON.stringify({ ok: true, path: dest, bytes: size }), {
			status: 200,
			headers: { "content-type": "application/json", ...corsHeaders },
		});
	},
});
