#!/usr/bin/env bun
/**
 * Bucket C Phase 0 probe — download and inspect Qwen3-Embedding-0.6B GGUF.
 * Writes 00-gguf-discovery.txt summarizing metadata keys + tensor list.
 * Run: bun eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GgufParser } from "../../../src/models/gguf-parser.js";

const OUT_DIR = "eval/reports/bucket-c-probe-2026-04-29";
const CACHE_DIR = join(OUT_DIR, "cache");
mkdirSync(CACHE_DIR, { recursive: true });

interface Candidate {
	label: string;
	url: string;
}

const candidates: Candidate[] = [
	{
		label: "qwen3-embedding-0.6b",
		url: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf",
	},
];

const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	console.log(s);
};

for (const c of candidates) {
	log(`\n=== ${c.label} ===`);
	log(`URL: ${c.url}`);
	const localPath = join(CACHE_DIR, `${c.label}.gguf`);
	if (!existsSync(localPath)) {
		log(`Downloading…`);
		const res = await fetch(c.url);
		if (!res.ok) {
			log(`FAILED: HTTP ${res.status} ${res.statusText}`);
			continue;
		}
		const buf = await res.arrayBuffer();
		writeFileSync(localPath, new Uint8Array(buf));
		log(`Saved ${buf.byteLength} bytes to ${localPath}`);
	} else {
		log(`Reusing cached ${localPath}`);
	}

	const bytes = new Uint8Array(await Bun.file(localPath).arrayBuffer());
	const ctx = GgufParser.parse(bytes);

	log(
		`general.architecture = ${JSON.stringify(ctx.metadata.get("general.architecture")?.value)}`,
	);

	const archKeys: string[] = [];
	for (const k of ctx.metadata.keys()) {
		if (
			k.startsWith("qwen") ||
			k.startsWith("general.") ||
			k.startsWith("tokenizer.") ||
			k.includes("pool") ||
			k.includes("embed") ||
			k.includes("norm")
		) {
			archKeys.push(k);
		}
	}
	archKeys.sort();
	log(`Metadata keys (${archKeys.length}):`);
	for (const k of archKeys) {
		const v = ctx.metadata.get(k)?.value;
		const repr =
			typeof v === "string" || typeof v === "number" || typeof v === "boolean"
				? String(v)
				: `<${typeof v}>`;
		log(`  ${k} = ${repr}`);
	}

	const tensorNames = ctx.tensors.map((t) => t.name).sort();
	log(`Tensors (${tensorNames.length}):`);
	for (const n of tensorNames.slice(0, 80)) log(`  ${n}`);
	if (tensorNames.length > 80) log(`  … (${tensorNames.length - 80} more)`);

	const projTensors = tensorNames.filter(
		(n) =>
			n.includes("proj") ||
			n.includes("pool") ||
			n.includes("output_norm") ||
			n.includes("output.weight"),
	);
	log(`Projection / pooling / output tensors (${projTensors.length}):`);
	for (const n of projTensors) log(`  ${n}`);
}

writeFileSync(join(OUT_DIR, "00-gguf-discovery.txt"), `${lines.join("\n")}\n`);
log(`\nWrote ${OUT_DIR}/00-gguf-discovery.txt`);
