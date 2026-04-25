#!/usr/bin/env bun
// Extract a BERT GGUF's vocabulary metadata into a JSON file the golden
// fixture generator (scripts/generate-bert-golden.py) and the Bun-side
// regression test (tests/wordpiece-golden.test.ts) can consume without
// re-parsing the binary.
//
// Usage:
//   bun run scripts/extract-bert-vocab.ts <gguf-path> [output-path]
//
// Writes JSON of the form:
//   {
//     "model": "bert",
//     "tokens": ["[PAD]", "[unused0]", ..., "▁happy", ...],
//     "clsId": 101, "sepId": 102, "unkId": 100, "padId": 0
//   }
//
// llama.cpp's BERT GGUF converter rewrites the HF vocab so word-initial
// tokens carry a "▁" (U+2581) prefix and "##xyz" continuations have the
// "##" stripped. The exported JSON preserves that exact representation
// — both the TS Tokenizer and the HF reference must agree on it.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { GgufParser } from "../src/models/gguf-parser.js";

function getStringArray(ctx: ReturnType<typeof GgufParser.parse>, key: string): string[] {
	const kv = ctx.metadata.get(key);
	if (!kv || !Array.isArray(kv.value)) return [];
	return kv.value as string[];
}

function getNumber(
	ctx: ReturnType<typeof GgufParser.parse>,
	key: string,
): number | undefined {
	const kv = ctx.metadata.get(key);
	if (!kv || typeof kv.value !== "number") return undefined;
	return kv.value;
}

function getString(
	ctx: ReturnType<typeof GgufParser.parse>,
	key: string,
): string | undefined {
	const kv = ctx.metadata.get(key);
	if (!kv || typeof kv.value !== "string") return undefined;
	return kv.value;
}

function main(): void {
	const args = process.argv.slice(2);
	if (args.length < 1) {
		console.error(
			"usage: bun run scripts/extract-bert-vocab.ts <gguf-path> [output-path]",
		);
		process.exit(1);
	}
	const ggufPath = resolve(args[0]);
	const outPath = resolve(
		args[1] ?? "tests/fixtures/bert-wordpiece-vocab.json",
	);

	const buf = readFileSync(ggufPath);
	const ab = buf.buffer.slice(
		buf.byteOffset,
		buf.byteOffset + buf.byteLength,
	) as ArrayBuffer;
	const ctx = GgufParser.parse(ab);

	const model = getString(ctx, "tokenizer.ggml.model");
	if (model !== "bert") {
		throw new Error(
			`expected tokenizer.ggml.model="bert", got "${model ?? "<unset>"}"`,
		);
	}
	const tokens = getStringArray(ctx, "tokenizer.ggml.tokens");
	if (tokens.length === 0) {
		throw new Error("tokenizer.ggml.tokens is empty or missing");
	}

	const out = {
		model,
		tokens,
		clsId: getNumber(ctx, "tokenizer.ggml.cls_token_id"),
		// llama.cpp + Arctic-Embed GGUFs misspell this key as "seperator".
		// Do NOT correct to "separator" or the read returns undefined.
		sepId: getNumber(ctx, "tokenizer.ggml.seperator_token_id"),
		unkId: getNumber(ctx, "tokenizer.ggml.unknown_token_id"),
		padId: getNumber(ctx, "tokenizer.ggml.padding_token_id"),
	};

	writeFileSync(outPath, `${JSON.stringify(out)}\n`);
	console.error(
		`wrote ${tokens.length} tokens (cls=${out.clsId} sep=${out.sepId} unk=${out.unkId}) to ${outPath}`,
	);
}

main();
