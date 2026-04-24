import { existsSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const DEFAULT_PORT = 8031;
const DEFAULT_HOST = "0.0.0.0";
const DEFAULT_ROOT = "smoke-test";

const CONTENT_TYPES: Record<string, string> = {
	".html": "text/html; charset=utf-8",
	".js": "text/javascript; charset=utf-8",
	".mjs": "text/javascript; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".json": "application/json; charset=utf-8",
	".svg": "image/svg+xml",
	".png": "image/png",
	".ico": "image/x-icon",
	".wasm": "application/wasm",
	".gguf": "application/octet-stream",
};

function contentTypeFor(path: string): string {
	return CONTENT_TYPES[extname(path).toLowerCase()] ?? "application/octet-stream";
}

function parseServerArgs(): { port: number; host: string; root: string } {
	const { values } = parseArgs({
		options: {
			port: { type: "string" },
			host: { type: "string" },
			root: { type: "string" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});
	if (values.help) {
		console.log(`Usage: bun run eval/smoke-serve.ts [options]

Options:
      --port <num>   Bind port (default: ${DEFAULT_PORT})
      --host <addr>  Bind host (default: ${DEFAULT_HOST})
      --root <dir>   Static root (default: ${DEFAULT_ROOT})
  -h, --help         Show this help
`);
		process.exit(0);
	}
	const port = values.port ? Number(values.port) : DEFAULT_PORT;
	if (!Number.isFinite(port) || port <= 0 || port > 65535) {
		console.error(`Invalid --port: ${values.port}`);
		process.exit(1);
	}
	return {
		port,
		host: values.host ?? DEFAULT_HOST,
		root: resolve(values.root ?? DEFAULT_ROOT),
	};
}

const options = parseServerArgs();

if (!existsSync(options.root) || !statSync(options.root).isDirectory()) {
	console.error(`Static root not found: ${options.root}`);
	process.exit(1);
}

const server = Bun.serve({
	hostname: options.host,
	port: options.port,
	idleTimeout: 120,
	async fetch(req) {
		const url = new URL(req.url);
		// Strip query string, prevent path escape, default to index
		const rel = url.pathname === "/" ? "/index.html" : url.pathname;
		const safe = rel.replace(/\.\./g, "").replace(/^\/+/, "");
		const filePath = join(options.root, safe);
		if (!existsSync(filePath)) {
			return new Response("Not found", { status: 404 });
		}
		try {
			const stat = statSync(filePath);
			if (stat.isDirectory()) {
				return new Response("Directory listing disabled", { status: 403 });
			}
			return new Response(Bun.file(filePath), {
				status: 200,
				headers: {
					"content-type": contentTypeFor(filePath),
					"cache-control": "no-cache",
				},
			});
		} catch (err) {
			return new Response(
				`server error: ${err instanceof Error ? err.message : String(err)}`,
				{ status: 500 },
			);
		}
	},
	// Bun.serve already swallows connection-close errors silently;
	// the explicit error handler keeps any genuine server faults visible.
	error(err) {
		return new Response(`server error: ${err.message}`, { status: 500 });
	},
});

console.log(
	`smoke static server on http://${options.host}:${options.port} (root=${options.root})`,
);

process.on("SIGINT", () => {
	server.stop();
	process.exit(0);
});
process.on("SIGTERM", () => {
	server.stop();
	process.exit(0);
});
