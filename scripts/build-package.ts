import { existsSync } from "node:fs";
import { copyFile, mkdir, rm } from "node:fs/promises";
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

await $`bun build src/index.ts src/persistence/indexeddb-store.ts --outdir dist --target browser --minify --root src`;
await $`tsc --emitDeclarationOnly --declaration --declarationMap --outDir dist`;

for (const asset of wasmAssets) {
	await copyFile(asset.source, asset.target);
}
