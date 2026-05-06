// Tier 3 P2 — LlamaDecodeWrapper replaces ModelInference for the
// causal-LM forward path. The wrapper owns a (model, ctx) pair from
// the bridge and exposes the same public surface engine.ts read on
// the legacy class.
//
// Lifetime: one wrapper per loaded causal-LM model, one main context
// for chat decode (created in initKVCache), and ONE lazily-created
// embedder context for Bucket-D self-embed (configured with
// embeddings=true + pooling=LAST). Both contexts share the same
// llama_model handle.
//
// See:
//   docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md §P2
//   docs/superpowers/plans/2026-05-05-tier3-p2-causal-lm.md

import type { LlamaBridge, LlamaContextParams } from "./llama-bridge.js";

export interface LlamaDecodeWrapperOptions {
	/** Enable flash attention for the main causal context. */
	flashAttn?: boolean;
}

export interface EmbedOptions {
	/** Pooling strategy for Bucket-D self-embed. Default: "last-token". */
	pooling?: "last-token" | "mean";
}

export class LlamaDecodeWrapper {
	readonly bridge: LlamaBridge;
	readonly model: number;
	readonly flashAttn: boolean;

	private mainCtx = 0;
	private embedCtx = 0;
	private nCached = 0;
	private mainCtxTokens = 0;

	constructor(
		bridge: LlamaBridge,
		model: number,
		opts: LlamaDecodeWrapperOptions = {},
	) {
		this.bridge = bridge;
		this.model = model;
		this.flashAttn = opts.flashAttn ?? false;
	}

	/**
	 * No-op. Legacy ModelInference.loadWeights uploaded weights into a
	 * ggml graph; in P2 the weights are already on GPU after
	 * webllm_load_model. The method exists for API symmetry — engine.ts
	 * calls it unconditionally between `new LlamaDecodeWrapper(...)` and
	 * `initKVCache(...)`.
	 */
	loadWeights(): void {
		/* no-op */
	}

	/**
	 * Allocate the main causal context with `nCtx` tokens of KV. Idempotent:
	 * a second call replaces the first context (frees the old one). After
	 * this, `forward()` is callable.
	 */
	async initKVCache(nCtx: number): Promise<void> {
		if (nCtx <= 0) {
			throw new Error(`initKVCache: nCtx must be > 0; got ${nCtx}`);
		}
		if (this.mainCtx) {
			this.bridge.freeContext(this.mainCtx);
			this.mainCtx = 0;
		}
		const params: LlamaContextParams = {
			nCtx,
			embeddings: false,
			poolingType: 0,
			flashAttn: this.flashAttn,
		};
		this.mainCtx = await this.bridge.createContext(this.model, params);
		this.mainCtxTokens = this.bridge.nCtx(this.mainCtx);
		this.nCached = 0;
	}

	/** KV cache tokens currently materialized for the main context. */
	get cachedTokenCount(): number {
		return this.nCached;
	}

	/** Effective context window of the main context (post-clamp). */
	get maxContextLength(): number {
		return this.mainCtxTokens;
	}

	/**
	 * Run a forward pass over `tokenIds` at `positions`. Returns the LAST
	 * position's logits as a Float32Array view INTO ctx-owned memory —
	 * valid until the next forward / embed call. Mirrors the legacy
	 * ModelInference.forward contract (which returned the last
	 * position's logits regardless of input length).
	 *
	 * `positions` must be sequential starting at `cachedTokenCount`; any
	 * other layout indicates a session-tracker bug upstream and throws.
	 * The legacy class enforced the same invariant via its internal
	 * nCached counter; surfacing it as a precondition makes the contract
	 * explicit.
	 */
	async forward(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.mainCtx) {
			throw new Error("forward called before initKVCache");
		}
		if (tokenIds.length === 0) {
			throw new Error("forward called with empty tokenIds");
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forward: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		const pastLen = this.nCached;
		for (let i = 0; i < positions.length; i++) {
			if (positions[i] !== pastLen + i) {
				throw new Error(
					`forward: positions must be sequential from cachedTokenCount=${pastLen}; positions[${i}]=${positions[i]} expected ${pastLen + i}`,
				);
			}
		}
		const status = await this.bridge.decode(this.mainCtx, tokenIds, pastLen);
		if (status !== 0) {
			throw new Error(`forward: webllm_decode returned status ${status}`);
		}
		this.nCached = pastLen + tokenIds.length;
		return await this.bridge.getLogits(this.mainCtx, this.model, -1);
	}

	/** Drop all KV cache state on the main context. */
	resetKVCache(): void {
		if (!this.mainCtx) return;
		this.bridge.kvClear(this.mainCtx);
		this.nCached = 0;
	}

	/**
	 * Drop tokens [keepLen, cachedTokenCount). Used by spec-decode
	 * rollback and prefix-cache mid-conversation truncation.
	 */
	truncateKVCache(keepLen: number): void {
		if (!this.mainCtx) return;
		if (keepLen < 0 || keepLen > this.nCached) {
			throw new Error(
				`truncateKVCache: keepLen=${keepLen} out of range [0, ${this.nCached}]`,
			);
		}
		this.bridge.kvSeqRm(this.mainCtx, 0, keepLen, -1);
		this.nCached = keepLen;
	}

	/**
	 * Serialize the main context's seq=0 KV state for prefix-cache
	 * persistence. The blob is opaque (arch + flash_attn-internal
	 * format); restore it via {@link loadKVCache} on a context with
	 * the same model + n_ctx + flash_attn config.
	 *
	 * Note vs legacy: the `nTokens` argument is retained for API
	 * symmetry with engine.ts callers, but is informational only —
	 * the upstream serializer always captures the full materialized
	 * KV state. Callers that pass `nTokens < cachedTokenCount` should
	 * `truncateKVCache(nTokens)` first if they want a shorter blob.
	 */
	async serializeKVCache(nTokens: number): Promise<Uint8Array> {
		if (!this.mainCtx) {
			throw new Error("serializeKVCache called before initKVCache");
		}
		if (nTokens > this.nCached) {
			throw new Error(
				`serializeKVCache: nTokens=${nTokens} > cachedTokenCount=${this.nCached}`,
			);
		}
		return this.bridge.stateSeqGetData(this.mainCtx, 0);
	}

	/**
	 * Restore main context's seq=0 KV state from a blob produced by
	 * {@link serializeKVCache}, then truncate to `nTokens`. The
	 * `snapshotLen` parameter (default = `nTokens`) records the length
	 * the blob was serialized at — it must be >= nTokens. Truncating
	 * a longer-stored snapshot to a shorter prefix matches the legacy
	 * loadKVCache contract used by engine.ts (prefix-cache reload of
	 * a shared prefix from a longer-stored snapshot).
	 */
	async loadKVCache(
		bytes: Uint8Array,
		nTokens: number,
		snapshotLen?: number,
	): Promise<void> {
		if (!this.mainCtx) {
			throw new Error("loadKVCache called before initKVCache");
		}
		const sl = snapshotLen ?? nTokens;
		if (sl < nTokens) {
			throw new Error(`loadKVCache: snapshotLen=${sl} < nTokens=${nTokens}`);
		}
		const ok = this.bridge.stateSeqSetData(this.mainCtx, bytes, 0);
		if (!ok) {
			throw new Error("loadKVCache: state_seq_set_data failed");
		}
		// State now holds `sl` tokens. Truncate down to nTokens via
		// kv_seq_rm so the consumer's first forward pass extends from
		// position nTokens.
		if (nTokens < sl) {
			this.bridge.kvSeqRm(this.mainCtx, 0, nTokens, -1);
		}
		this.nCached = nTokens;
	}

	/**
	 * Bucket-D self-embed: build a pooled embedding for `tokenIds`
	 * using a side context configured with embeddings=true. The
	 * embedder context is allocated lazily on first call and reused
	 * for the lifetime of this wrapper.
	 *
	 * Pooling: `"last-token"` (default, matches Bucket-D doctrine
	 * 2026-04-30) maps to llama_pooling_type LAST=3. `"mean"` maps to
	 * MEAN=1. The Bucket-D distinguishability gate tests last-token
	 * — `"mean"` is preserved for parity with the legacy embed()
	 * surface but is not exercised by the current canonical fleet.
	 */
	async embed(
		tokenIds: Int32Array,
		opts: EmbedOptions = {},
	): Promise<Float32Array> {
		if (tokenIds.length === 0) {
			throw new Error("embed called with empty tokenIds");
		}
		const pooling = opts.pooling ?? "last-token";
		const poolingType = pooling === "mean" ? 1 : 3;
		if (!this.embedCtx) {
			// Use the model's training-time n_ctx for the embedder context;
			// embedder requests are typically <2K tokens so the budget is
			// generous. Match flash_attn to the main context.
			const nCtxTrain = this.bridge.nCtxTrain(this.model);
			this.embedCtx = await this.bridge.createContext(this.model, {
				nCtx: nCtxTrain > 0 ? nCtxTrain : 4096,
				embeddings: true,
				poolingType: poolingType as 0 | 1 | 2 | 3,
				flashAttn: this.flashAttn,
			});
		}
		// Embedder is single-shot: clear KV before each request so
		// prior embedding's state doesn't leak into this one.
		this.bridge.kvClear(this.embedCtx);
		const status = await this.bridge.decode(this.embedCtx, tokenIds, 0);
		if (status !== 0) {
			throw new Error(`embed: webllm_decode returned status ${status}`);
		}
		// ith=-1 returns the pooled embedding when pooling_type != NONE.
		// Copy out of ctx-owned memory before returning so callers can
		// retain the result across subsequent decode calls.
		const view = await this.bridge.getEmbeddings(this.embedCtx, this.model, -1);
		return new Float32Array(view);
	}

	/** Drop both contexts. Idempotent. */
	dispose(): void {
		if (this.embedCtx) {
			this.bridge.freeContext(this.embedCtx);
			this.embedCtx = 0;
		}
		if (this.mainCtx) {
			this.bridge.freeContext(this.mainCtx);
			this.mainCtx = 0;
		}
		this.nCached = 0;
	}
}
