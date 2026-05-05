// Tier 3 P0 spike — TypeScript bindings for the webllm_* bridge exports
// that wrap upstream llama.cpp's llama_model / llama_context / llama_decode.
//
// See:
//   docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md
//   docs/superpowers/plans/2026-05-05-tier3-p0-spike.md
//
// Pointer semantics: this build uses -sWASM_BIGINT=1, so all pointers
// crossing the JS↔WASM boundary are bigint. Float32Array / Int32Array
// views over mod.HEAPU8.buffer use Number(ptr) for byte offsets.

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
	loadModel(buf: Uint8Array): Promise<bigint>;
	/** Free a llama_model handle. Idempotent on null. */
	freeModel(handle: bigint): void;
	/** Create a llama_context for the given model. Throws on failure. */
	createContext(model: bigint, params: LlamaContextParams): bigint;
	/** Free a llama_context handle. Idempotent on null. */
	freeContext(ctx: bigint): void;
	/**
	 * Decode tokens at sequence positions [pastLen, pastLen+tokens.length).
	 * Logits are computed for the last token only (greedy single-step).
	 * Returns the llama_decode status (0 = success).
	 */
	decode(ctx: bigint, tokens: Int32Array, pastLen: number): Promise<number>;
	/**
	 * Get logits for the i-th token of the last decode. ith=-1 → last
	 * logits-flagged token. Returns a Float32Array view INTO ctx-owned
	 * memory — valid until the next decode call. Do not retain.
	 */
	getLogits(ctx: bigint, model: bigint, ith?: number): Float32Array;
	/** Returns the model's vocab size. Used to size logits views. */
	nVocab(model: bigint): number;
}

/** Minimum subset of the Emscripten module surface this bridge needs. */
interface RawLlamaModule {
	_webllm_load_model: (bufPtr: bigint, nBytes: bigint) => bigint;
	_webllm_free_model: (handle: bigint) => void;
	_webllm_create_context: (
		model: bigint,
		nCtx: number,
		embeddings: number,
		poolingType: number,
		flashAttn: number,
	) => bigint;
	_webllm_free_context: (ctx: bigint) => void;
	_webllm_decode: (
		ctx: bigint,
		tokensPtr: bigint,
		nTokens: number,
		pastLen: number,
	) => Promise<number>;
	_webllm_get_logits: (ctx: bigint, ith: number) => bigint;
	_webllm_n_vocab: (model: bigint) => number;
	_bridge_malloc: (size: bigint) => bigint;
	_bridge_free: (ptr: bigint) => void;
	HEAPU8: Uint8Array;
}

/**
 * Construct a typed bridge over the raw Emscripten module. The bridge
 * owns no state of its own — it's a thin marshalling layer. All
 * lifecycle responsibility (free model / context, manage logits view
 * lifetime) lives with the caller.
 */
export function createLlamaBridge(mod: RawLlamaModule): LlamaBridge {
	return {
		async loadModel(buf: Uint8Array): Promise<bigint> {
			const ptr = mod._bridge_malloc(BigInt(buf.byteLength));
			if (ptr === 0n) {
				throw new Error("webllm: bridge_malloc failed for GGUF buffer");
			}
			try {
				mod.HEAPU8.set(buf, Number(ptr));
				const handle = mod._webllm_load_model(ptr, BigInt(buf.byteLength));
				if (handle === 0n) {
					throw new Error("webllm: webllm_load_model returned null");
				}
				return handle;
			} finally {
				mod._bridge_free(ptr);
			}
		},

		freeModel(handle: bigint): void {
			mod._webllm_free_model(handle);
		},

		createContext(model: bigint, params: LlamaContextParams): bigint {
			const handle = mod._webllm_create_context(
				model,
				params.nCtx,
				params.embeddings ? 1 : 0,
				params.poolingType ?? 0,
				params.flashAttn ? 1 : 0,
			);
			if (handle === 0n) {
				throw new Error("webllm: webllm_create_context returned null");
			}
			return handle;
		},

		freeContext(ctx: bigint): void {
			mod._webllm_free_context(ctx);
		},

		async decode(
			ctx: bigint,
			tokens: Int32Array,
			pastLen: number,
		): Promise<number> {
			const nBytes = BigInt(tokens.byteLength);
			const ptr = mod._bridge_malloc(nBytes);
			if (ptr === 0n) {
				throw new Error("webllm: bridge_malloc failed for decode tokens");
			}
			try {
				new Int32Array(mod.HEAPU8.buffer, Number(ptr), tokens.length).set(
					tokens,
				);
				return await mod._webllm_decode(ctx, ptr, tokens.length, pastLen);
			} finally {
				mod._bridge_free(ptr);
			}
		},

		getLogits(ctx: bigint, model: bigint, ith = -1): Float32Array {
			const ptr = mod._webllm_get_logits(ctx, ith);
			if (ptr === 0n) {
				throw new Error("webllm: webllm_get_logits returned null");
			}
			const nVocab = mod._webllm_n_vocab(model);
			return new Float32Array(mod.HEAPU8.buffer, Number(ptr), nVocab);
		},

		nVocab(model: bigint): number {
			return mod._webllm_n_vocab(model);
		},
	};
}
