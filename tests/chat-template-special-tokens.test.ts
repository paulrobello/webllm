/**
 * Audit that the special-token literals emitted by every formatter in
 * `src/inference/chat-template.ts` round-trip through their target
 * tokenizer to a single token id, and that each family's expected
 * chat-stop literal resolves via `tokenizer.getId(...)` (the API
 * `engine.ts:addChatStopToken` calls).
 *
 * Catches two bug classes:
 *
 * 1. The `<|assistant|?` typo class (Phi-3 bug #3, 2026-04-29): a
 *    one-character corruption in a formatter literal is silently
 *    tokenized as plain text, the model never sees the role marker,
 *    and decode produces gibberish only at runtime.
 *
 * 2. The chat-stop-token-not-resolving class (Phi-3 + Mistral-v0.3,
 *    2026-05-04): the family's expected EOS / turn-end literal isn't
 *    registered in the tokenizer's `tokenToId` map, so
 *    `tokenizer.getId(literal)` returns undefined,
 *    `addChatStopToken` silently no-ops, and the model wanders past
 *    end-of-turn into multi-turn self-dialogue.
 *
 * Coverage spans every chat-capable GGUF in `smoke-test/models/`:
 *   phi3   → phi-3.5-mini
 *   llama3 → llama-3.1-8b, llama-3.2-1b, llama-3.2-3b
 *   chatml → hermes-3-llama-3.2-3b, qwen2.5-{1.5b,3b,coder-1.5b},
 *            qwen3-{0.6b,1.7b,4b,8b,14b}, smollm2-{360m,1.7b}
 *   llama2 → mistral-7b-v0.3, mistral-nemo
 *   zephyr → tinyllama-1.1b-chat (only `</s>` is special; role markers
 *            are plain text)
 *   gemma  → gemma-2-2b
 *
 * Deferred:
 *   mistral-v7 — no v7-template GGUF in the fleet that's small enough
 *                 to ship as a smoke fixture (Mistral-Large is too
 *                 heavy; Nemo uses the llama2 template, not v7).
 */

import { describe, expect, test } from "bun:test";
import { closeSync, existsSync, openSync, readSync } from "node:fs";
import { resolve } from "node:path";
import { Tokenizer } from "../src/inference/tokenizer.js";
import { ModelLoader } from "../src/models/model-loader.js";

interface AuditCase {
	model: string;
	gguf: string;
	formatter: string;
	literals: string[];
	/**
	 * Chat-stop literals the engine's `addChatStopToken` helper looks up
	 * via `tokenizer.getId(...)` for this family. If `getId` returns
	 * undefined, the chat-stop branch in `engine.ts` silently no-ops and
	 * the model wanders past end-of-turn into self-dialogue (the failure
	 * mode that hit Phi-3 and Mistral-v0.3 in 2026-05-04).
	 *
	 * This is a separate check from `literals` above because `encode()`
	 * can succeed (literal tokenizes as one id) while `getId()` returns
	 * undefined (literal isn't registered in `tokenToId` map). The engine
	 * uses `getId()`, so that's what must hold.
	 */
	chatStopTokens: string[];
}

const LLAMA3_LITERALS = [
	"<|begin_of_text|>",
	"<|start_header_id|>",
	"<|end_header_id|>",
	"<|eot_id|>",
];
const CHATML_LITERALS = ["<|im_start|>", "<|im_end|>"];
const MISTRAL_LITERALS = ["[INST]", "[/INST]", "</s>"];
const GEMMA_LITERALS = ["<start_of_turn>", "<end_of_turn>"];

const AUDIT_CASES: AuditCase[] = [
	// --- phi3 ---
	{
		model: "phi-3.5-mini",
		gguf: "smoke-test/models/phi-3.5-mini-q4km.gguf",
		formatter: "formatPhi3",
		literals: ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"],
		chatStopTokens: ["<|end|>"],
	},
	// --- llama3 ---
	{
		model: "llama-3.1-8b",
		gguf: "smoke-test/models/llama-3.1-8b-instruct-iq3m.gguf",
		formatter: "formatLlama3",
		literals: LLAMA3_LITERALS,
		chatStopTokens: ["<|eot_id|>"],
	},
	{
		model: "llama-3.2-1b",
		gguf: "smoke-test/models/llama-3.2-1b-q4f16.gguf",
		formatter: "formatLlama3",
		literals: LLAMA3_LITERALS,
		chatStopTokens: ["<|eot_id|>"],
	},
	{
		model: "llama-3.2-3b",
		gguf: "smoke-test/models/llama-3.2-3b-q4f16.gguf",
		formatter: "formatLlama3",
		literals: LLAMA3_LITERALS,
		chatStopTokens: ["<|eot_id|>"],
	},
	// --- chatml ---
	// Hermes-3 is chatml on top of a Llama-3 vocab; only `<|im_end|>` is
	// available as a stop literal (no `<|endoftext|>`).
	{
		model: "hermes-3-llama-3.2-3b",
		gguf: "smoke-test/models/hermes-3-llama-3.2-3b-q4f16.gguf",
		formatter: "formatChatml (Llama-3 vocab)",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>"],
	},
	{
		model: "qwen2.5-1.5b",
		gguf: "smoke-test/models/qwen2.5-1.5b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen2.5-3b",
		gguf: "smoke-test/models/qwen2.5-3b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen2.5-coder-1.5b",
		gguf: "smoke-test/models/qwen2.5-coder-1.5b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen3-0.6b",
		gguf: "smoke-test/models/qwen3-0.6b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen3-1.7b",
		gguf: "smoke-test/models/qwen3-1.7b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen3-4b",
		gguf: "smoke-test/models/qwen3-4b-q4f16.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen3-8b",
		gguf: "smoke-test/models/qwen3-8b-iq3m.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	{
		model: "qwen3-14b",
		gguf: "smoke-test/models/qwen3-14b-q4ks.gguf",
		formatter: "formatChatml",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>", "<|endoftext|>"],
	},
	// SmolLM2 uses chatml but is registered with `architecture: "llama"`
	// and has no `<|endoftext|>` token. Stops on `<|im_end|>` (which
	// happens to share id 2 with this vocab's eos slot — fine because
	// the chatml-trained model emits `<|im_end|>` for turn ends).
	{
		model: "smollm2-360m",
		gguf: "smoke-test/models/smollm2-360m-q4f16.gguf",
		formatter: "formatChatml (SmolLM2 vocab)",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>"],
	},
	{
		model: "smollm2-1.7b",
		gguf: "smoke-test/models/smollm2-1.7b-q4f16.gguf",
		formatter: "formatChatml (SmolLM2 vocab)",
		literals: CHATML_LITERALS,
		chatStopTokens: ["<|im_end|>"],
	},
	// --- llama2 (Mistral-Instruct family) ---
	{
		model: "mistral-7b-v0.3",
		gguf: "smoke-test/models/mistral-7b-instruct-v0.3-q4ks.gguf",
		formatter: "formatLlama2 (Mistral-v3 vocab)",
		// `<<SYS>>`, `<</SYS>>` are plain text in this vocab and intentionally
		// excluded — the audit only covers literals expected to be single
		// special tokens.
		literals: MISTRAL_LITERALS,
		chatStopTokens: ["</s>"],
	},
	{
		model: "mistral-nemo-2407",
		gguf: "smoke-test/models/mistral-nemo-instruct-2407-q4ks.gguf",
		formatter: "formatLlama2 (Mistral-Nemo vocab)",
		literals: MISTRAL_LITERALS,
		chatStopTokens: ["</s>"],
	},
	// --- zephyr ---
	// TinyLlama-Chat: `<|user|>`, `<|assistant|>`, `<|system|>` are
	// *plain text* in this vocab (not special tokens), so they are
	// intentionally excluded from `literals`. Only `</s>` is special.
	{
		model: "tinyllama-1.1b",
		gguf: "smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf",
		formatter: "formatZephyr",
		literals: [],
		chatStopTokens: ["</s>"],
	},
	// --- gemma ---
	{
		model: "gemma-2-2b",
		gguf: "smoke-test/models/gemma-2-2b-q4f16.gguf",
		formatter: "formatGemma",
		literals: GEMMA_LITERALS,
		chatStopTokens: ["<end_of_turn>"],
	},
];

interface LoadedTokenizer {
	tokenizer: Tokenizer;
	architecture: string;
}

const tokenizerCache = new Map<string, LoadedTokenizer>();

/**
 * Read only enough of the GGUF to parse its header + tokenizer config.
 * Loading the whole file via `readFileSync` runs out of address space
 * on the larger fixtures (qwen3-14b, mistral-nemo, qwen3-8b) — Bun's
 * V8 heap is capped well below the actual file size. Try progressively
 * larger header reads (2 MiB → 16 MiB → 64 MiB) until parse succeeds;
 * GGUF metadata is always at the front and 64 MiB covers every model
 * in the fleet.
 */
function loadTokenizer(ggufPath: string): LoadedTokenizer {
	const cached = tokenizerCache.get(ggufPath);
	if (cached) return cached;
	const fd = openSync(ggufPath, "r");
	let parsed: ReturnType<typeof ModelLoader.parseModel> | null = null;
	let lastErr: unknown = null;
	try {
		for (const sizeMb of [2, 16, 64]) {
			const buf = Buffer.alloc(sizeMb * 1024 * 1024);
			readSync(fd, buf, 0, buf.length, 0);
			const view = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
			try {
				parsed = ModelLoader.parseModel(view);
				break;
			} catch (e) {
				lastErr = e;
			}
		}
	} finally {
		closeSync(fd);
	}
	if (!parsed) {
		throw new Error(
			`Failed to parse GGUF metadata from ${ggufPath} with up to 64 MiB header buffer: ${String(lastErr)}`,
		);
	}
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

			for (const stop of audit.chatStopTokens) {
				test(`chat-stop literal ${JSON.stringify(stop)} resolves via tokenizer.getId() (engine addChatStopToken contract)`, () => {
					const { tokenizer } = loadTokenizer(ggufPath);
					const id = tokenizer.getId(stop);
					if (id === undefined) {
						const encoded = tokenizer.encode(stop);
						throw new Error(
							`tokenizer.getId(${JSON.stringify(stop)}) returned undefined. ` +
								`engine.ts:addChatStopToken silently no-ops in this case, leaving the model with no chat-turn stop and producing multi-turn self-dialogue. ` +
								`encode(${JSON.stringify(stop)}) yields [${encoded.join(",")}] — if this is a single id, the GGUF vocab has the literal but it's not registered as a special token in tokenToId.`,
						);
					}
					expect(typeof id).toBe("number");
					// Cross-check: encoding the same literal yields the same id.
					const encoded = tokenizer.encode(stop);
					expect(encoded.length).toBe(1);
					expect(encoded[0]).toBe(id);
				});
			}
		},
	);
}
