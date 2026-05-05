// Tier 3 P0 spike — TypeScript bindings for the webllm_* bridge exports
// that wrap upstream llama.cpp's llama_model / llama_context / llama_decode.
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
	createContext(model: number, params: LlamaContextParams): number;
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
	getLogits(ctx: number, model: number, ith?: number): Float32Array;
	/** Returns the model's vocab size. Used to size logits views. */
	nVocab(model: number): number;
}

/**
 * Minimum subset of the Emscripten module surface this bridge needs.
 * Pointer parameters and returns are typed `any` because the concrete
 * type is `number` on wasm32 and `bigint` on wasm64+WASM_BIGINT — the
 * adapter below probes and translates.
 */
interface RawLlamaModule {
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_load_model: (bufPtr: any, nBytes: any) => any;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_free_model: (handle: any) => void;
	_webllm_create_context: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		model: any,
		nCtx: number,
		embeddings: number,
		poolingType: number,
		flashAttn: number,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	) => any;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_free_context: (ctx: any) => void;
	_webllm_decode: (
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		ctx: any,
		// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
		tokensPtr: any,
		nTokens: number,
		pastLen: number,
	) => Promise<number>;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_get_logits: (ctx: any, ith: number) => any;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_webllm_n_vocab: (model: any) => number;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_bridge_malloc: (size: any) => any;
	// biome-ignore lint/suspicious/noExplicitAny: ABI-polymorphic pointer types
	_bridge_free: (ptr: any) => void;
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
				const handle = from64(
					mod._webllm_load_model(to64(ptr), to64(buf.byteLength)),
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

		createContext(model: number, params: LlamaContextParams): number {
			const handle = from64(
				mod._webllm_create_context(
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

		getLogits(ctx: number, model: number, ith = -1): Float32Array {
			const ptr = from64(mod._webllm_get_logits(to64(ctx), ith));
			if (ptr === 0) {
				throw new Error("webllm: webllm_get_logits returned null");
			}
			const nVocab = mod._webllm_n_vocab(to64(model));
			return new Float32Array(mod.HEAPU8.buffer, ptr, nVocab);
		},

		nVocab(model: number): number {
			return mod._webllm_n_vocab(to64(model));
		},
	};
}
