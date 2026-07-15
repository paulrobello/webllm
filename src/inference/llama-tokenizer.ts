// Tier 3 P1 — LlamaTokenizer wraps a llama_model* handle and the
// LlamaBridge to provide the same public surface as the legacy
// Tokenizer (encode/decode/getId/bosId/eosId/vocabSize/options).
// Models loaded via the new path (webllm_load_model) build a
// LlamaTokenizer; legacy callers continue to construct Tokenizer
// from a TokenizerConfig until P2 deletes the legacy path.
//
// @experimental This module is part of the Tier-3 llama-decode spike
// and is NOT part of the published package's semver contract. Declaration
// emit is excluded from npm types by `scripts/build-package.ts`. See ARC-007.
//
// The streaming detokenizer in tokenizer.ts (StreamingDecoder)
// stays — its prevText differential decode is project-specific and
// is not exposed by upstream. P1 only swaps the encode/decode
// implementation; streaming logic is unchanged.

import type { LlamaBridge } from "./llama-bridge.js";

// Minimal options surface — only fields engine.ts actually reads from
// the legacy Tokenizer.options. P2 may extend this if more legacy
// fields turn out to be load-bearing.
export interface LlamaTokenizerOptions {
	chatTemplate?: string;
	/**
	 * Encoder-only mode (BERT-family). When true, `encode()` calls
	 * `llama_tokenize` with `add_bos=true`, which for BERT-family
	 * vocabs prepends `[CLS]` and appends `[SEP]` (BOS for BERT IS
	 * `[CLS]`). For causal LMs (the default), BOS is supplied via
	 * the chat template, not at tokenize time, so this stays false.
	 */
	encoderOnly?: boolean;
}

export class LlamaTokenizer {
	readonly bridge: LlamaBridge;
	readonly model: number;
	private readonly _options: LlamaTokenizerOptions;
	private readonly addedTokenCache = new Map<string, number>();

	constructor(
		bridge: LlamaBridge,
		model: number,
		options: LlamaTokenizerOptions = {},
	) {
		this.bridge = bridge;
		this.model = model;
		this._options = options;
	}

	encode(text: string): number[] {
		// add_bos default false: engine.ts adds BOS via chat template
		// for causal LMs. Match legacy Tokenizer.encode behavior
		// (no implicit BOS) for that case. Encoder-only vocabs flip
		// add_bos=true so [CLS]/[SEP] (BERT) or <s>/</s> (encoder
		// SPM) are prepended/appended automatically by llama_tokenize.
		// parse_special=true: chat-template tokens like <|im_start|>
		// must encode as single ids (matches legacy
		// encodeWithSpecialTokens path).
		const ids = this.bridge.tokenize(this.model, text, {
			addBos: this._options.encoderOnly === true,
			parseSpecial: true,
		});
		return Array.from(ids);
	}

	decode(ids: number[]): string {
		if (ids.length === 0) return "";
		return this.bridge.detokenize(this.model, new Int32Array(ids));
	}

	getId(token: string): number | undefined {
		// Cache lookups so repeated stop-token resolution doesn't
		// re-tokenize. Round-trip via tokenize() with parse_special=1
		// — single-token specials encode to exactly one id.
		const cached = this.addedTokenCache.get(token);
		if (cached !== undefined) return cached;
		const ids = this.bridge.tokenize(this.model, token, {
			addBos: false,
			parseSpecial: true,
		});
		if (ids.length !== 1) return undefined;
		const id = ids[0];
		this.addedTokenCache.set(token, id);
		return id;
	}

	get bosId(): number {
		return this.bridge.tokenBos(this.model);
	}

	get eosId(): number {
		return this.bridge.tokenEos(this.model);
	}

	get vocabSize(): number {
		return this.bridge.nVocab(this.model);
	}

	get options(): LlamaTokenizerOptions {
		return this._options;
	}
}
