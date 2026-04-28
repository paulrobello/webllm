#!/usr/bin/env bun
/**
 * Phase 0 probe — download and inspect both candidate GGUFs.
 * Writes 00-gguf-discovery.txt summarizing metadata keys + tensor list.
 * Run: bun eval/reports/encoder-parity-2026-04-28/probe-gguf.ts
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GgufParser } from "../../../src/models/gguf-parser.js";

const OUT_DIR = "eval/reports/encoder-parity-2026-04-28";
const CACHE_DIR = join(OUT_DIR, "cache");
mkdirSync(CACHE_DIR, { recursive: true });

interface Candidate {
	label: string;
	url: string;
}

const candidates: Candidate[] = [
	{
		label: "nomic-embed-text-v1.5",
		url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf",
	},
	{
		label: "jina-embeddings-v2-base-en",
		url: "https://huggingface.co/gaianet/jina-embeddings-v2-base-en-GGUF/resolve/main/jina-embeddings-v2-base-en-f16.gguf",
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
			k.startsWith("nomic-bert.") ||
			k.startsWith("jina-bert-v2.") ||
			k.startsWith("bert.") ||
			k.startsWith("general.") ||
			k.startsWith("tokenizer.")
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
	for (const n of tensorNames.slice(0, 50)) log(`  ${n}`);
	if (tensorNames.length > 50) log(`  … (${tensorNames.length - 50} more)`);

	const hasPosEmb = tensorNames.includes("position_embd.weight");
	log(`HAS position_embd.weight: ${hasPosEmb}`);
}

writeFileSync(join(OUT_DIR, "00-gguf-discovery.txt"), `${lines.join("\n")}\n`);
log(`\nWrote ${OUT_DIR}/00-gguf-discovery.txt`);
