// Tier 3 P0 spike — TypeScript bindings for the webllm_* bridge exports
// that wrap upstream llama.cpp's llama_model / llama_context / llama_decode.
//
// @experimental This module is a Tier-3 prototype spike and is NOT part of
// the published package's semver contract. Declaration emit is excluded
// from npm types by `scripts/build-package.ts`. See ARC-007.
//
// See:
//   docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md
//   docs/superpowers/plans/2026-05-05-tier3-p0-spike.md
//
// Pointer ABI: webllm ships two binaries — wasm32 (production default,
// pointers fit in JS Number) and wasm64 with `-sWASM_BIGINT=1` (>4 GiB
// models, pointers cross the boundary as bigint). The public LlamaBridge
// API uses `number` everywhere; the constructor probes `_bridge_malloc(0)`
// at init and translates to/from bigint at the cwrap boundary on wasm64.
// Number.MAX_SAFE_INTEGER (2^53) ≫ 16 GiB (2^34), so the conversion is
// safe under all webllm-supported memory budgets.
//
// This mirrors the legacy GgmlWasm.malloc/free pattern in ggml-wasm.ts —
// the same codebase invariant applies here.

export interface LlamaContextParams {
	/** KV cache size in tokens. 0 = use the model's training-time default. */
	nCtx: number;
	/** False = causal LM (logits output), true = embedder mode (pooled). */
	embeddings?: boolean;
	/** 0=NONE, 1=MEAN, 2=CLS, 3=LAST. Ignored when embeddings is false. */
	poolingType?: 0 | 1 | 2 | 3;
	/** Enable flash attention (replaces per-arch FA gating). */
	flashAttn?: boolean;
}

export interface LlamaBridge {
	/** Load a GGUF buffer into a llama_model. Throws on failure. */
	loadModel(buf: Uint8Array): Promise<number>;
	/** Free a llama_model handle. Idempotent on null. */
	freeModel(handle: number): void;
	/** Create a llama_context for the given model. Throws on failure. */
	createContext(model: number, params: LlamaContextParams): Promise<number>;
	/** Free a llama_context handle. Idempotent on null. */
	freeContext(ctx: number): void;
	/**
	 * Decode tokens at sequence positions [pastLen, pastLen+tokens.length).
	 * Logits are computed for the last token only (greedy single-step).
	 * Returns the llama_decode status (0 = success).
	 */
	decode(ctx: number, tokens: Int32Array, pastLen: number): Promise<number>;
	/**
	 * Get logits for the i-th token of the last decode. ith=-1 → last
	 * logits-flagged token. Returns a Float32Array view INTO ctx-owned
	 * memory — valid until the next decode call. Do not retain.
	 */
	getLogits(ctx: number, model: number, ith?: number): Promise<Float32Array>;
	/** Returns the model's vocab size. Used to size logits views. */
	nVocab(model: number): number;
	/**
	 * Tokenize text. Returns id list. Throws on bridge_malloc failure.
	 * Internally retries with a larger buffer if the first attempt was
	 * too small (mirrors upstream llama_tokenize's negative-count
	 * semantics).
	 */
	tokenize(
		model: number,
		text: string,
		options?: { addBos?: boolean; parseSpecial?: boolean },
	): Int32Array;
	/**
	 * Detokenize id list back to a UTF-8 string. Throws on
	 * bridge_malloc failure. Buffer-too-small triggers a retry
	 * with the upstream-reported required size.
	 */
	detokenize(model: number, tokens: Int32Array): string;
	/** BOS token id, or -1 if the vocab doesn't define one. */
	tokenBos(model: number): number;
	/** EOS token id, or -1 if the vocab doesn't define one. */
	tokenEos(model: number): number;
	/** Read a string metadata value by key. Returns null if missing. */
	getMetadata(model: number, key: string): string | null;
	/** Hyperparam accessors. Negative return = missing model handle. */
	nCtxTrain(model: number): number;
	nEmbd(model: number): number;
	nLayer(model: number): number;
	nHead(model: number): number;
	nHeadKv(model: number): number;
	/** Per-context KV size in tokens. */
	nCtx(ctx: number): number;

	/** Drop tokens [p0, p1) for seq_id. p1=-1 means "to the end". */
	kvSeqRm(ctx: number, seqId: number, p0: number, p1: number): void;
	/** Clear all sequences in this context's KV cache. */
	kvClear(ctx: number): void;

	/** Bytes needed to serialize seq_id's KV state. */
	stateSeqGetSize(ctx: number, seqId: number): number;
	/** Copy seq_id's KV state into a freshly-allocated Uint8Array. */
	stateSeqGetData(ctx: number, seqId: number): Uint8Array;
	/**
	 * Restore seq_id's KV state from a previously-captured blob.
	 * Returns true on success. The blob must come from a context
	 * with the SAME model + n_ctx + flash_attn flag — restoring
	 * across mismatched configs is undefined behavior.
	 */
	stateSeqSetData(ctx: number, blob: Uint8Array, destSeqId: number): boolean;

	/**
	 * Read embeddings for the i-th token of the last decode.
	 * ith=-1 → pooled (or last-position when pooling is NONE).
	 * Returns a Float32Array view INTO ctx-owned memory — valid
	 * until the next decode call. Length = nEmbd(model).
	 */
	getEmbeddings(
		ctx: number,
		model: number,
		ith?: number,
	): Promise<Float32Array>;
}

/**
 * ABI-polymorphic pointer: `number` on wasm32, `bigint` on
 * wasm64+WASM_BIGINT. Typed as a union so tsc catches the historical
 * bug class where a JSPI-wrapped `Promise<WasmPtr>` is coerced by
 * `>>> 0` (silent leak on wasm32) or `BigInt(NaN)` (RangeError on
 * wasm64) — a `Promise` is assignable to `any` but NOT to this union.
 */
type WasmPtr = number | bigint;

/**
 * Minimum subset of the Emscripten module surface this bridge needs.
 * Pointer parameters and returns are typed `WasmPtr` (number on wasm32,
 * bigint on wasm64+WASM_BIGINT); the adapter below probes the ABI and
 * translates via `to64`/`from64`. Exports listed in JSPI_EXPORTS
 * (CMakeLists.txt) are promising-wrapped and return `Promise<WasmPtr>` —
 * the TS binding MUST await them before unwrapping (see regression
 * lessons in CLAUDE.md).
 */
interface RawLlamaModule {
	// JSPI-wrapped (see JSPI_EXPORTS in src/wasm/CMakeLists.txt) — await required.
	_webllm_load_model: (bufPtr: WasmPtr, nBytes: WasmPtr) => Promise<WasmPtr>;
	_webllm_free_model: (handle: WasmPtr) => void;
	// JSPI-wrapped — await required.
	_webllm_create_context: (
		model: WasmPtr,
		nCtx: number,
		embeddings: number,
		poolingType: number,
		flashAttn: number,
	) => Promise<WasmPtr>;
	_webllm_free_context: (ctx: WasmPtr) => void;
	// JSPI-wrapped; returns a status code (0 = success), not a pointer.
	_webllm_decode: (
		ctx: WasmPtr,
		tokensPtr: WasmPtr,
		nTokens: number,
		pastLen: number,
	) => Promise<number>;
	// JSPI-wrapped — await required.
	_webllm_get_logits: (ctx: WasmPtr, ith: number) => Promise<WasmPtr>;
	_webllm_n_vocab: (model: WasmPtr) => number;
	_webllm_tokenize: (
		model: WasmPtr,
		textPtr: WasmPtr,
		nText: number,
		tokensOut: WasmPtr,
		nTokensMax: number,
		addBos: number,
		parseSpecial: number,
	) => number;
	_webllm_detokenize: (
		model: WasmPtr,
		tokensPtr: WasmPtr,
		nTokens: number,
		textOut: WasmPtr,
		nTextMax: number,
	) => number;
	_webllm_token_bos: (model: WasmPtr) => number;
	_webllm_token_eos: (model: WasmPtr) => number;
	_webllm_get_metadata: (
		model: WasmPtr,
		keyPtr: WasmPtr,
		bufPtr: WasmPtr,
		bufSize: number,
	) => number;
	_webllm_n_ctx_train: (model: WasmPtr) => number;
	_webllm_n_embd: (model: WasmPtr) => number;
	_webllm_n_layer: (model: WasmPtr) => number;
	_webllm_n_head: (model: WasmPtr) => number;
	_webllm_n_head_kv: (model: WasmPtr) => number;
	_webllm_n_ctx: (ctx: WasmPtr) => number;

	_webllm_kv_seq_rm: (
		ctx: WasmPtr,
		seqId: number,
		p0: number,
		p1: number,
	) => void;
	_webllm_kv_clear: (ctx: WasmPtr) => void;

	_webllm_state_seq_get_size: (ctx: WasmPtr, seqId: number) => number;
	_webllm_state_seq_get_data: (
		ctx: WasmPtr,
		dst: WasmPtr,
		size: number,
		seqId: number,
	) => number;
	_webllm_state_seq_set_data: (
		ctx: WasmPtr,
		src: WasmPtr,
		size: number,
		destSeqId: number,
	) => number;

	// JSPI-wrapped — await required.
	_webllm_get_embeddings: (ctx: WasmPtr, ith: number) => Promise<WasmPtr>;
	_bridge_malloc: (size: WasmPtr) => WasmPtr;
	_bridge_free: (ptr: WasmPtr) => void;
	HEAPU8: Uint8Array;
}

/**
 * Construct a typed bridge over the raw Emscripten module. The bridge
 * owns no state of its own beyond the wasm32/wasm64 ABI flag detected
 * at construction time. All lifecycle responsibility (free model /
 * context, manage logits view lifetime) lives with the caller.
 */
export function createLlamaBridge(mod: RawLlamaModule): LlamaBridge {
	// Probe ABI shape: try wasm32 (Number) first; on wasm64+WASM_BIGINT
	// the call throws TypeError because the i64 arg can't accept a JS
	// Number. Mirrors the GgmlWasm.init() probe in ggml-wasm.ts.
	let is64 = false;
	try {
		const probe = mod._bridge_malloc(0);
		is64 = typeof probe === "bigint";
		mod._bridge_free(probe);
	} catch {
		const probe = mod._bridge_malloc(0n);
		is64 = true;
		mod._bridge_free(probe);
	}

	// Translation helpers — `to64`/`from64` are no-ops on wasm32, BigInt
	// conversions on wasm64. All public-API pointer values are JS Number;
	// translation happens at the cwrap call boundary only.
	const to64 = is64
		? (n: number): bigint => BigInt(n)
		: (n: number): number => n;
	const from64 = is64
		? (v: number | bigint): number => Number(v)
		: (v: number | bigint): number => (v as number) >>> 0;

	const malloc = (size: number): number =>
		from64(mod._bridge_malloc(to64(size)));
	const free = (ptr: number): void => {
		mod._bridge_free(to64(ptr));
	};

	return {
		async loadModel(buf: Uint8Array): Promise<number> {
			const ptr = malloc(buf.byteLength);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for GGUF buffer");
			}
			try {
				mod.HEAPU8.set(buf, ptr);
				// Under JSPI, webllm_load_model is promising-wrapped — must
				// await before unwrapping the i32/i64 return value.
				const handle = from64(
					await mod._webllm_load_model(to64(ptr), to64(buf.byteLength)),
				);
				if (handle === 0) {
					throw new Error("webllm: webllm_load_model returned null");
				}
				return handle;
			} finally {
				free(ptr);
			}
		},

		freeModel(handle: number): void {
			mod._webllm_free_model(to64(handle));
		},

		async createContext(
			model: number,
			params: LlamaContextParams,
		): Promise<number> {
			// Under JSPI, webllm_create_context is promising-wrapped — must
			// await before unwrapping the i32/i64 return value.
			const handle = from64(
				await mod._webllm_create_context(
					to64(model),
					params.nCtx,
					params.embeddings ? 1 : 0,
					params.poolingType ?? 0,
					params.flashAttn ? 1 : 0,
				),
			);
			if (handle === 0) {
				throw new Error("webllm: webllm_create_context returned null");
			}
			return handle;
		},

		freeContext(ctx: number): void {
			mod._webllm_free_context(to64(ctx));
		},

		async decode(
			ctx: number,
			tokens: Int32Array,
			pastLen: number,
		): Promise<number> {
			const ptr = malloc(tokens.byteLength);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for decode tokens");
			}
			try {
				new Int32Array(mod.HEAPU8.buffer, ptr, tokens.length).set(tokens);
				return await mod._webllm_decode(
					to64(ctx),
					to64(ptr),
					tokens.length,
					pastLen,
				);
			} finally {
				free(ptr);
			}
		},

		async getLogits(
			ctx: number,
			model: number,
			ith = -1,
		): Promise<Float32Array> {
			// Under JSPI, webllm_get_logits is promising-wrapped — must
			// await before unwrapping the i32/i64 pointer return.
			const ptr = from64(await mod._webllm_get_logits(to64(ctx), ith));
			if (ptr === 0) {
				throw new Error("webllm: webllm_get_logits returned null");
			}
			const nVocab = mod._webllm_n_vocab(to64(model));
			return new Float32Array(mod.HEAPU8.buffer, ptr, nVocab);
		},

		nVocab(model: number): number {
			return mod._webllm_n_vocab(to64(model));
		},

		tokenize(
			model: number,
			text: string,
			options?: { addBos?: boolean; parseSpecial?: boolean },
		): Int32Array {
			const addBos = options?.addBos ? 1 : 0;
			const parseSpecial = options?.parseSpecial !== false ? 1 : 0;
			const utf8 = new TextEncoder().encode(text);
			const textPtr = malloc(utf8.byteLength);
			if (textPtr === 0) {
				throw new Error("webllm: bridge_malloc failed for tokenize text");
			}
			try {
				mod.HEAPU8.set(utf8, textPtr);

				let cap = Math.max(16, utf8.byteLength + 8);
				let tokensPtr = malloc(cap * 4);
				if (tokensPtr === 0) {
					throw new Error("webllm: bridge_malloc failed for tokenize tokens");
				}
				try {
					let n = mod._webllm_tokenize(
						to64(model),
						to64(textPtr),
						utf8.byteLength,
						to64(tokensPtr),
						cap,
						addBos,
						parseSpecial,
					);
					if (n < 0) {
						const required = -n;
						free(tokensPtr);
						cap = required;
						tokensPtr = malloc(cap * 4);
						if (tokensPtr === 0) {
							throw new Error(
								"webllm: bridge_malloc failed for tokenize retry",
							);
						}
						n = mod._webllm_tokenize(
							to64(model),
							to64(textPtr),
							utf8.byteLength,
							to64(tokensPtr),
							cap,
							addBos,
							parseSpecial,
						);
						if (n < 0) {
							throw new Error(
								`webllm: tokenize returned ${n} after retry (required ${required})`,
							);
						}
					}
					return new Int32Array(
						mod.HEAPU8.buffer.slice(tokensPtr, tokensPtr + n * 4),
					);
				} finally {
					free(tokensPtr);
				}
			} finally {
				free(textPtr);
			}
		},

		detokenize(model: number, tokens: Int32Array): string {
			const tokensPtr = malloc(tokens.byteLength);
			if (tokensPtr === 0) {
				throw new Error("webllm: bridge_malloc failed for detokenize tokens");
			}
			try {
				new Int32Array(mod.HEAPU8.buffer, tokensPtr, tokens.length).set(tokens);

				let cap = Math.max(64, tokens.length * 4 + 8);
				let textPtr = malloc(cap);
				if (textPtr === 0) {
					throw new Error("webllm: bridge_malloc failed for detokenize text");
				}
				try {
					let n = mod._webllm_detokenize(
						to64(model),
						to64(tokensPtr),
						tokens.length,
						to64(textPtr),
						cap,
					);
					if (n < 0) {
						const required = -n;
						free(textPtr);
						cap = required;
						textPtr = malloc(cap);
						if (textPtr === 0) {
							throw new Error(
								"webllm: bridge_malloc failed for detokenize retry",
							);
						}
						n = mod._webllm_detokenize(
							to64(model),
							to64(tokensPtr),
							tokens.length,
							to64(textPtr),
							cap,
						);
						if (n < 0) {
							throw new Error(
								`webllm: detokenize returned ${n} after retry (required ${required})`,
							);
						}
					}
					const bytes = new Uint8Array(
						mod.HEAPU8.buffer.slice(textPtr, textPtr + n),
					);
					return new TextDecoder().decode(bytes);
				} finally {
					free(textPtr);
				}
			} finally {
				free(tokensPtr);
			}
		},

		tokenBos(model: number): number {
			return mod._webllm_token_bos(to64(model));
		},

		tokenEos(model: number): number {
			return mod._webllm_token_eos(to64(model));
		},

		getMetadata(model: number, key: string): string | null {
			const utf8 = new TextEncoder().encode(`${key}\0`);
			const keyPtr = malloc(utf8.byteLength);
			if (keyPtr === 0) {
				throw new Error("webllm: bridge_malloc failed for metadata key");
			}
			try {
				mod.HEAPU8.set(utf8, keyPtr);
				// First call sized to 0 → returns required size or -1 if missing.
				const required = mod._webllm_get_metadata(
					to64(model),
					to64(keyPtr),
					to64(0),
					0,
				);
				if (required < 0) return null;
				const cap = required + 1;
				const bufPtr = malloc(cap);
				if (bufPtr === 0) {
					throw new Error("webllm: bridge_malloc failed for metadata buf");
				}
				try {
					const n = mod._webllm_get_metadata(
						to64(model),
						to64(keyPtr),
						to64(bufPtr),
						cap,
					);
					if (n < 0) return null;
					return new TextDecoder().decode(
						new Uint8Array(mod.HEAPU8.buffer.slice(bufPtr, bufPtr + n)),
					);
				} finally {
					free(bufPtr);
				}
			} finally {
				free(keyPtr);
			}
		},

		nCtxTrain(model: number): number {
			return mod._webllm_n_ctx_train(to64(model));
		},
		nEmbd(model: number): number {
			return mod._webllm_n_embd(to64(model));
		},
		nLayer(model: number): number {
			return mod._webllm_n_layer(to64(model));
		},
		nHead(model: number): number {
			return mod._webllm_n_head(to64(model));
		},
		nHeadKv(model: number): number {
			return mod._webllm_n_head_kv(to64(model));
		},
		nCtx(ctx: number): number {
			return mod._webllm_n_ctx(to64(ctx));
		},

		kvSeqRm(ctx: number, seqId: number, p0: number, p1: number): void {
			mod._webllm_kv_seq_rm(to64(ctx), seqId, p0, p1);
		},
		kvClear(ctx: number): void {
			mod._webllm_kv_clear(to64(ctx));
		},

		stateSeqGetSize(ctx: number, seqId: number): number {
			return mod._webllm_state_seq_get_size(to64(ctx), seqId);
		},
		stateSeqGetData(ctx: number, seqId: number): Uint8Array {
			const size = mod._webllm_state_seq_get_size(to64(ctx), seqId);
			if (size === 0) return new Uint8Array(0);
			const ptr = malloc(size);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for state-seq blob");
			}
			try {
				const n = mod._webllm_state_seq_get_data(
					to64(ctx),
					to64(ptr),
					size,
					seqId,
				);
				if (n === 0) {
					throw new Error("webllm: state_seq_get_data returned 0 bytes");
				}
				return new Uint8Array(mod.HEAPU8.buffer.slice(ptr, ptr + n));
			} finally {
				free(ptr);
			}
		},
		stateSeqSetData(ctx: number, blob: Uint8Array, destSeqId: number): boolean {
			if (blob.byteLength === 0) return true;
			const ptr = malloc(blob.byteLength);
			if (ptr === 0) {
				throw new Error("webllm: bridge_malloc failed for state-seq restore");
			}
			try {
				mod.HEAPU8.set(blob, ptr);
				const n = mod._webllm_state_seq_set_data(
					to64(ctx),
					to64(ptr),
					blob.byteLength,
					destSeqId,
				);
				return n > 0;
			} finally {
				free(ptr);
			}
		},

		async getEmbeddings(
			ctx: number,
			model: number,
			ith = -1,
		): Promise<Float32Array> {
			const ptr = from64(await mod._webllm_get_embeddings(to64(ctx), ith));
			if (ptr === 0) {
				throw new Error("webllm: webllm_get_embeddings returned null");
			}
			const nEmbd = mod._webllm_n_embd(to64(model));
			return new Float32Array(mod.HEAPU8.buffer, ptr, nEmbd);
		},
	};
}
