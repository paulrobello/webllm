import { existsSync } from "node:fs";
import { copyFile, mkdir, readdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { $ } from "bun";

interface BuildAsset {
	readonly source: string;
	readonly target: string;
}

const wasmAssets: readonly BuildAsset[] = [
	{
		source: "src/wasm/build/webllm-wasm.js",
		target: "dist/webllm-wasm.js",
	},
	{
		source: "src/wasm/build/webllm-wasm.wasm",
		target: "dist/webllm-wasm.wasm",
	},
	{
		source: "src/wasm/build-mem64/webllm-wasm-mem64.js",
		target: "dist/webllm-wasm-mem64.js",
	},
	{
		source: "src/wasm/build-mem64/webllm-wasm-mem64.wasm",
		target: "dist/webllm-wasm-mem64.wasm",
	},
];

const missingAssets = wasmAssets.filter((asset) => !existsSync(asset.source));
if (missingAssets.length > 0) {
	throw new Error(
		[
			"Missing WASM runtime assets required for package build.",
			"Run `make wasm-build` first, then rerun `bun run build`.",
			...missingAssets.map((asset) => `- ${asset.source}`),
		].join("\n"),
	);
}

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });

await $`bun build src/index.ts src/internal.ts src/persistence/indexeddb-store.ts --outdir dist --target browser --minify --root src`;
await $`tsc --emitDeclarationOnly --declaration --declarationMap --outDir dist`;

// ARC-007: exclude prototype declarations from the published type surface.
// The JSEP backend (P2-v2 prototype) and the Tier-3 llama-decode spike are
// active R&D and ship without semver guarantees; their .d.ts files are
// stripped after declaration emit so npm consumers see only the canonical
// `"default"` backend contract. Sources remain in `src/` for in-tree and
// smoke-bundle consumers.
const prototypeDeclGlobs: readonly string[] = [
	"inference/jsep",
	"index-jsep.d.ts",
	"inference/llama-bridge.d.ts",
	"inference/llama-tokenizer.d.ts",
];

async function removeProtoDecls(
	root: string,
	globs: readonly string[],
): Promise<string[]> {
	const removed: string[] = [];
	for (const glob of globs) {
		const target = join(root, glob);
		const exists = await stat(target).then(
			(s) => s.isDirectory() || s.isFile(),
			() => false,
		);
		if (!exists) continue;
		if (glob.endsWith(".d.ts")) {
			await rm(target, { force: true });
			removed.push(target);
			continue;
		}
		// Directory: walk and collect .d.ts files, then remove the tree.
		const stack: string[] = [target];
		while (stack.length > 0) {
			const dir = stack.pop()!;
			let entries: string[] = [];
			try {
				entries = await readdir(dir);
			} catch {
				continue;
			}
			for (const entry of entries) {
				const full = join(dir, entry);
				const s = await stat(full).catch(() => null);
				if (!s) continue;
				if (s.isDirectory()) {
					stack.push(full);
				} else if (entry.endsWith(".d.ts")) {
					await rm(full, { force: true });
					removed.push(full);
				}
			}
			await rm(dir, { recursive: true, force: true }).catch(() => {});
		}
	}
	return removed;
}

const removedDecls = await removeProtoDecls("dist", prototypeDeclGlobs);
if (removedDecls.length > 0) {
	console.log(
		`[build-package] stripped ${removedDecls.length} prototype declaration(s) from dist/:`,
	);
	for (const path of removedDecls) console.log(`  - ${path}`);
} else {
	console.log(
		"[build-package] no prototype declarations found to strip (expected after a clean tsc emit if no jsep/llama-bridge/llama-tokenizer sources are present).",
	);
}

for (const asset of wasmAssets) {
	await copyFile(asset.source, asset.target);
}
