import { existsSync, statSync, writeFileSync } from "node:fs";
import { extname, resolve, sep } from "node:path";
import { parseArgs } from "node:util";

const DEFAULT_PORT = 8031;
const DEFAULT_HOST = "127.0.0.1";
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
	".glb": "model/gltf-binary",
	".gltf": "model/gltf+json",
	".bin": "application/octet-stream",
};

function contentTypeFor(path: string): string {
	return (
		CONTENT_TYPES[extname(path).toLowerCase()] ?? "application/octet-stream"
	);
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

// Route aliases — map a virtual URL onto an absolute path outside the
// static root. Used to expose canonical artifacts (e.g. parity fixtures
// under eval/reports/) without requiring a symlink in smoke-test/.
const ROUTE_ALIASES: Record<string, string> = {
	"/parity-fixture.json": resolve(
		"eval/reports/p1-tokenizer-2026-05-05/parity-fixture.json",
	),
};

const server = Bun.serve({
	hostname: options.host,
	port: options.port,
	idleTimeout: 120,
	async fetch(req) {
		const url = new URL(req.url);

		// POST endpoint — accept JSON-serialized regenerated parity fixture
		// and write to the canonical eval/reports path. Browser harness
		// uses this to round-trip canonical llama_tokenize output back to
		// disk without going through the OS download dialog. Dev-only;
		// the smoke server isn't exposed beyond localhost.
		if (
			req.method === "POST" &&
			url.pathname === "/save-parity-fixture" &&
			ROUTE_ALIASES["/parity-fixture.json"]
		) {
			try {
				const body = await req.text();
				writeFileSync(ROUTE_ALIASES["/parity-fixture.json"], body);
				return new Response(`ok ${body.length} bytes\n`, { status: 200 });
			} catch (err) {
				console.error("save-parity-fixture:", err);
				return new Response("internal error", { status: 500 });
			}
		}

		const aliasTarget = ROUTE_ALIASES[url.pathname];
		if (aliasTarget) {
			if (!existsSync(aliasTarget)) {
				return new Response("Not found", { status: 404 });
			}
			return new Response(Bun.file(aliasTarget), {
				status: 200,
				headers: {
					"content-type": contentTypeFor(aliasTarget),
					"cache-control": "no-cache",
				},
			});
		}
		// Containment check: resolve and verify the path stays under root.
		// `resolve` with a leading-slash rel would ignore the root, so strip
		// leading slashes first (kept from the original sanitizer).
		const rel = url.pathname === "/" ? "/index.html" : url.pathname;
		const rootAbs = resolve(options.root);
		const candidate = resolve(rootAbs, rel.replace(/^\/+/, ""));
		if (candidate !== rootAbs && !candidate.startsWith(rootAbs + sep)) {
			return new Response("Not found", { status: 404 });
		}
		const filePath = candidate;
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
			console.error("static-serve:", err);
			return new Response("internal error", { status: 500 });
		}
	},
	// Bun.serve already swallows connection-close errors silently;
	// the explicit error handler keeps any genuine server faults visible.
	error(err) {
		console.error("server:", err);
		return new Response("internal error", { status: 500 });
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
