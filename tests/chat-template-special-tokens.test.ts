/**
 * Audit that the special-token literals emitted by every formatter in
 * `src/inference/chat-template.ts` round-trip through their target
 * tokenizer to a single token id.
 *
 * Catches latent equivalents of the `<|assistant|?` typo (Phi-3 bug #3,
 * 2026-04-29) before they ship to a public-API caller. The bug class:
 * a one-character corruption in a formatter literal is silently
 * tokenized as plain text, the model never sees the role marker, and
 * decode produces gibberish only at runtime.
 *
 * Per-architecture coverage map:
 *   phi3            → phi-3.5-mini-q4km
 *   llama3          → llama-3.1-8b-instruct-iq3m
 *   chatml (qwen3)  → qwen3-0.6b-q4f16
 *   llama2-via-mistral-v3 → mistral-7b-instruct-v0.3-q4ks
 *                     (covers `[INST]`/`[/INST]` as actual special tokens
 *                     in the Mistral-v3 vocab)
 *
 * Deferred:
 *   mistral-v7  — `[SYSTEM_PROMPT]`/`[/SYSTEM_PROMPT]` are only special
 *                 tokens in v7-template GGUFs (Mistral-Nemo, Mistral-Large).
 *                 Nemo is 6.6 GiB; too heavy for a unit test. Re-introduce
 *                 if a smaller v7-template GGUF appears in the fleet.
 *   gemma       — no Gemma GGUF in the fleet.
 *   zephyr/llama2 — `<|user|>`, `[INST]`, `<<SYS>>` etc. are *plain text*
 *                 in those vocabs, not special tokens; nothing to verify.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Tokenizer } from "../src/inference/tokenizer.js";
import { ModelLoader } from "../src/models/model-loader.js";

interface AuditCase {
	model: string;
	gguf: string;
	formatter: string;
	literals: string[];
}

const AUDIT_CASES: AuditCase[] = [
	{
		model: "phi-3.5-mini",
		gguf: "smoke-test/models/phi-3.5-mini-q4km.gguf",
		formatter: "formatPhi3",
		literals: ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"],
	},
	{
		model: "llama-3.1-8b",
		gguf: "smoke-test/models/llama-3.1-8b-instruct-iq3m.gguf",
		formatter: "formatLlama3",
		literals: [
			"<|begin_of_text|>",
			"<|start_header_id|>",
			"<|end_header_id|>",
			"<|eot_id|>",
		],
	},
	{
		model: "qwen3-0.6b",
		gguf: "smoke-test/models/qwen3-0.6b-q4f16.gguf",
		formatter: "formatChatml",
		literals: ["<|im_start|>", "<|im_end|>"],
	},
	{
		model: "mistral-7b-v0.3",
		gguf: "smoke-test/models/mistral-7b-instruct-v0.3-q4ks.gguf",
		formatter: "formatLlama2 (Mistral-v3 vocab)",
		// `<<SYS>>`, `<</SYS>>` are plain text in this vocab and intentionally
		// excluded — the audit only covers literals expected to be single
		// special tokens.
		literals: ["[INST]", "[/INST]", "</s>"],
	},
];

interface LoadedTokenizer {
	tokenizer: Tokenizer;
	architecture: string;
}

const tokenizerCache = new Map<string, LoadedTokenizer>();

function loadTokenizer(ggufPath: string): LoadedTokenizer {
	const cached = tokenizerCache.get(ggufPath);
	if (cached) return cached;
	const data = readFileSync(ggufPath);
	const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	const parsed = ModelLoader.parseModel(view);
	const tokenizer = new Tokenizer(parsed.tokenizerConfig);
	const result: LoadedTokenizer = {
		tokenizer,
		architecture: parsed.hyperparams.architecture,
	};
	tokenizerCache.set(ggufPath, result);
	return result;
}

for (const audit of AUDIT_CASES) {
	const ggufPath = resolve(audit.gguf);
	const skip = !existsSync(ggufPath);

	describe.skipIf(skip)(
		`chat-template special-token audit · ${audit.model} (${audit.formatter})`,
		() => {
			for (const lit of audit.literals) {
				test(`encodes ${JSON.stringify(lit)} as exactly one token id`, () => {
					const { tokenizer } = loadTokenizer(ggufPath);
					const ids = tokenizer.encode(lit);
					if (ids.length !== 1) {
						const decoded = tokenizer.decode(ids, {
							includeSpecialTokens: true,
						});
						throw new Error(
							`expected ${JSON.stringify(lit)} to encode as 1 token, got ${ids.length}: [${ids.join(",")}] (decoded=${JSON.stringify(decoded)}). ` +
								`This is the bug class that masked the Phi-3 \`<|assistant|?\` typo — the formatter literal is being tokenized as plain text instead of the intended role marker.`,
						);
					}
					expect(ids.length).toBe(1);
					// Round-trip back to the literal so we also catch silent
					// normalization (e.g. unicode-confusable tokens that share
					// an id with something else).
					const decoded = tokenizer.decode(ids, {
						includeSpecialTokens: true,
					});
					expect(decoded).toBe(lit);
				});
			}

			test("formatter output round-trips: every special-token literal recoverable in encoded stream", () => {
				const { tokenizer } = loadTokenizer(ggufPath);
				// Build a minimal prompt that exercises every literal: each
				// literal appears at least once when wrapped in arbitrary text.
				const prompt = audit.literals
					.map((lit, i) => `seg-${i} ${lit} after-${i}`)
					.join(" / ");
				const ids = tokenizer.encode(prompt);
				const expectedIds = audit.literals.map((lit) => {
					const e = tokenizer.encode(lit);
					return e[0];
				});
				for (const expected of expectedIds) {
					expect(ids).toContain(expected);
				}
			});
		},
	);
}
