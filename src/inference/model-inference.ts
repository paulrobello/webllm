import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	F32_BYTES,
	GgmlType,
	type GgmlWasm,
	type GraphComputeProfile,
	RopeMode,
	type TensorPtr,
} from "./ggml-wasm.js";

interface LayerWeights {
	attnNorm: TensorPtr;
	// Optional Phi-3 attn-norm bias. RMSNorm + bias add. Null on llama /
	// qwen / mistral GGUFs (which lack norm biases) and on Phi-3.5-mini
	// (which also has no norm biases) — the field stays in the interface
	// so future Phi-3 variants that ship norm biases can light up the
	// path without an interface change.
	attnNormBias: TensorPtr | null;
	// Phi-3 fused QKV: single [3*n_embd, n_embd] matrix (or [E + 2*kvDim,
	// n_embd] under GQA). When non-null, qProj/kProj/vProj are null and
	// the forward graph takes the fused matmul + view-slice path
	// (mirrors encoder-inference.ts:263-296).
	qkvFused: TensorPtr | null;
	qProj: TensorPtr | null;
	kProj: TensorPtr | null;
	vProj: TensorPtr | null;
	// Qwen2 / Qwen2.5 use biased Q/K/V projections. Llama, Qwen3, Mistral,
	// and Phi-3 (whose biases — if any — are baked into qkvFused) don't —
	// these stay null and the forward graph skips the add. Without this,
	// qwen2 GGUFs produce garbage (random-token) output because Q/K/V are
	// off by the bias shift.
	qBias: TensorPtr | null;
	kBias: TensorPtr | null;
	vBias: TensorPtr | null;
	qNorm: TensorPtr | null;
	kNorm: TensorPtr | null;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	ffnNormBias: TensorPtr | null;
	// Phi-3 fused gate-up: single [2*n_ff, n_embd] matrix. When non-null,
	// gateProj/upProj are null and the forward graph splits the matmul
	// output into gate/up halves before SwiGLU.
	gateUpFused: TensorPtr | null;
	gateProj: TensorPtr | null;
	upProj: TensorPtr | null;
	downProj: TensorPtr;
}

interface WeightTensors {
	tokEmb: TensorPtr;
	norm: TensorPtr;
	// Optional Phi-3 final-norm bias.
	normBias: TensorPtr | null;
	output: TensorPtr | null;
	// Phi-3 lm_head bias. Null for all other architectures and for
	// Phi-3.5-mini (its lm_head is biasless).
	outputBias: TensorPtr | null;
	layers: LayerWeights[];
}

interface LayerKVCache {
	k: TensorPtr;
	v: TensorPtr;
}

/**
 * Per-phase wall-clock timing for one `forward()` call, in milliseconds.
 * Captured only when `traceEnabled` is true — the default is off so the
 * hot path pays nothing. Used by the profile harness in `eval/perf.ts`.
 */
export type DecodeMode = "full" | "greedy" | "topk" | "verify";

export interface DecodeResult {
	/** Full logits (mode='full'). */
	logits?: Float32Array;
	/** Greedy argmax token ID (mode='greedy'). */
	tokenId?: number;
	/** Top-K indices (mode='topk'). */
	topKIndices?: Int32Array;
	/** Top-K logit values (mode='topk'). */
	topKValues?: Float32Array;
}

export interface ForwardTrace {
	mode: DecodeMode;
	nTokens: number;
	pastLen: number;
	ctxCreateMs: number;
	buildGraphMs: number;
	backendAllocMs: number;
	uploadLeavesMs: number;
	graphComputeMs: number;
	downloadResultMs: number;
	teardownMs: number;
	totalMs: number;
	backendProfileTotalMs?: number;
	backendMatmulMs?: number | null;
	backendAttentionMs?: number | null;
	backendEncodeOverheadMs?: number;
	backendDispatchCount?: number;
	backendBreakdownAvailable?: boolean;
}

export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	// nomic-bert uses GPT-NeoX-style RoPE (split-halves), per
	// llama.cpp/src/llama-model.cpp:9266 (LLM_ARCH_NOMIC_BERT →
	// LLAMA_ROPE_TYPE_NEOX). The HF config also surfaces this as
	// `rotary_emb_interleaved: false`.
	if (architecture === "nomic-bert") return RopeMode.NEOX;
	// Phi-3 / Phi-3.5 / PhiMoE all use NEOX-style RoPE (split-halves)
	// per llama.cpp's llama-model.cpp:9282 — phi3 sits in the same
	// case-list as PLAMO, GEMMA, STARCODER2, GPTNEOX, etc., all
	// returning LLAMA_ROPE_TYPE_NEOX. Without this, Phi-3 forward
	// produces fluent-but-meaningless output (interleaved RoPE
	// rotates the wrong feature pairs and the model decodes to
	// nonsense like "IMDbSidenoteSidenotepisode...").
	if (architecture === "phi3") return RopeMode.NEOX;
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}

/**
 * Compute the default `prefillTileSize` for a model based on hyperparameters.
 *
 * Rule: `layerCount >= 32 AND embeddingLength >= 4096` → 128, else 0.
 *
 * Maps directly to the §22 abort signature observed in
 * `eval/reports/prefill-tiling-2026-04-27/00-phase0-diagnostic.txt`:
 * "32 layers × seq=512 of F32 intermediates" exceeds the host-side ggml graph
 * allocator budget at `ggml-alloc.c:82`. Either gate alone keeps the per-tile
 * working set below the budget on every currently-registered model.
 *
 * Override surface (ctor opt / `?prefillTile=` / `--prefill-tile`) wins
 * unconditionally — including the explicit-zero force-disable path.
 *
 * Spec: `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md`.
 */
export function computeDefaultPrefillTileSize(hp: ModelHyperparams): number {
	return hp.layerCount >= 32 && hp.embeddingLength >= 4096 ? 128 : 0;
}

/**
 * Verify a tensor is laid out contiguously as F32. The fused-projection
 * helpers `buildQKV` / `buildFFNGateUp` produce strided `opView*` slices
 * that they immediately wrap in `opCont()` to materialize a contiguous
 * f32 copy — required because downstream rope/permute ops assume
 * contiguity. If a future refactor inserts an op between the helper and
 * the rope/permute chain that re-introduces strided derivatives, this
 * assertion fires loud at graph build time instead of producing silent
 * gibberish at decode (the failure mode of bug #2 in the Phi-3 closure,
 * commit `7c85a2a`).
 *
 * Implements the same predicate as ggml's `ggml_is_contiguous`, scoped
 * to F32 (the dtype of every fused-projection output here): tightly
 * packed strides nb[0]=4, nb[i]=nb[i-1]*ne[i-1] for i=1..3.
 */
export function assertContiguousF32(
	wasm: Pick<GgmlWasm, "tensorType" | "tensorNe" | "tensorNb">,
	tensor: TensorPtr,
	label: string,
): void {
	const type = wasm.tensorType(tensor);
	if (type !== GgmlType.F32) {
		throw new Error(
			`${label}: expected F32 (type ${GgmlType.F32}), got type ${type}`,
		);
	}
	const ne0 = wasm.tensorNe(tensor, 0);
	const ne1 = wasm.tensorNe(tensor, 1);
	const ne2 = wasm.tensorNe(tensor, 2);
	const nb0 = wasm.tensorNb(tensor, 0);
	const nb1 = wasm.tensorNb(tensor, 1);
	const nb2 = wasm.tensorNb(tensor, 2);
	const nb3 = wasm.tensorNb(tensor, 3);
	if (
		nb0 !== F32_BYTES ||
		nb1 !== nb0 * ne0 ||
		nb2 !== nb1 * ne1 ||
		nb3 !== nb2 * ne2
	) {
		throw new Error(
			`${label}: tensor not F32-contiguous (ne=[${ne0},${ne1},${ne2}], nb=[${nb0},${nb1},${nb2},${nb3}])`,
		);
	}
}

/**
 * Manages model weights loaded into WASM/ggml tensors and runs forward passes.
 *
 * Loads GGUF weight data into ggml tensors allocated on the WebGPU backend,
 * then builds and executes computation graphs for transformer forward passes.
 * Supports incremental decoding via KV cache for O(n) autoregressive generation.
 *
 * V cache is stored as [maxCtx, headDim, nKvHeads] so that mul_mat(V, attn)
 * produces [headDim, nTokens, nHeads] with built-in GQA broadcast.
 */
export class ModelInference {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: WeightTensors | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();

	private kvLayers: LayerKVCache[] | null = null;
	private kvBuf: BufferPtr = 0;
	private nCached = 0;
	/** When true, `forward()` populates `lastTrace` on every call. */
	traceEnabled = false;
	/** Timing of the most recent `forward()` call, or null if never traced. */
	lastTrace: ForwardTrace | null = null;
	/**
	 * When true, attention call sites use ggml_flash_attn_ext + an FA-ready
	 * V cache layout ([headDim, maxCtx, nKvHeads]). When false, they use the
	 * manual opMulMat(K,Q) → opSoftMaxExt → opMulMat(V,attn) chain + the
	 * legacy V layout ([maxCtx, headDim, nKvHeads]).
	 *
	 * Pinned at construction. Switching at runtime would require either two
	 * cached V tensors (memory bloat) or a per-step transpose copy
	 * (regression). Set once and treat as immutable.
	 */
	readonly flashAttn: boolean;
	/**
	 * When > 0, `forward()` automatically chunks long inputs into tiles of
	 * this many tokens. Each tile is a separate ctxCreate → graph build →
	 * graph compute → ctxFree cycle; KV cache accumulates across tiles via
	 * the existing `nCached = totalLen` advance.
	 *
	 * Default 0 = disabled (single-call forward, bit-identical to pre-§22).
	 *
	 * Used to unblock long-prefill on 7B+ models that otherwise abort in
	 * the host-side ggml graph allocator at `backend_alloc_ctx_tensors`.
	 * See TODO §22 for the per-model recommended tile size and rationale.
	 */
	readonly prefillTileSize: number;
	/**
	 * When true (default), `buildQKV` / `buildFFNGateUp` verify that every
	 * fused-projection output (Q/K/V/gate/up) materialized via `opCont` is
	 * F32-contiguous. The check costs ~7 wasm-bridge round trips per output
	 * × 5 outputs/layer × layerCount per forward — measured at <1% of
	 * graph-build wall time on Phi-3.5-mini, paid only when
	 * `LayerWeights.qkvFused` / `gateUpFused` are non-null (Phi-3 family
	 * today). Set false to disable for benchmarking; the assertion is
	 * defense-in-depth against silent regression of the contiguous-view
	 * invariant fixed in commit `7c85a2a`.
	 */
	assertFusedContiguity: boolean;

	constructor(
		wasm: GgmlWasm,
		hyperparams: ModelHyperparams,
		opts: {
			flashAttn?: boolean;
			prefillTileSize?: number;
			assertFusedContiguity?: boolean;
		} = {},
	) {
		this.wasm = wasm;
		this.hp = hyperparams;
		this.flashAttn = opts.flashAttn ?? false;
		this.prefillTileSize =
			opts.prefillTileSize ?? computeDefaultPrefillTileSize(hyperparams);
		if (this.prefillTileSize < 0) {
			throw new Error(
				`prefillTileSize must be >= 0; got ${this.prefillTileSize}`,
			);
		}
		this.assertFusedContiguity = opts.assertFusedContiguity ?? true;
	}

	loadWeights(
		ggufCtx: GgufContext,
		ggufData: Uint8Array | ((offset: number, byteLength: number) => Uint8Array),
	): void {
		// Callback form is required when the source bytes live in the WASM
		// heap: ctxCreate / backendAllocCtxTensors / per-chunk scratch
		// mallocs below may grow memory, which detaches any pre-existing
		// JS view of HEAPU8. uploadRangeChunked re-derives a fresh view
		// from the live heap once per chunk *after* its scratch malloc,
		// so the upload path stays valid across grow events.
		const isCallback = typeof ggufData === "function";
		const dataAt = isCallback
			? ggufData
			: (off: number, len: number) =>
					new Uint8Array(ggufData.buffer, ggufData.byteOffset + off, len);
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) {
			tensorMap.set(t.name, t);
		}

		// ctxCreate uses no_alloc=true (see webgpu-bridge.cpp::ctx_create);
		// tensor data lives in GPU buffers assigned by backendAllocCtxTensors,
		// not in this mempool. We only need room for ggml_tensor + ggml_object
		// metadata per tensor — adding ggufCtx.totalDataSize here would
		// allocate a multi-GB unused buffer and push 4B-class models over
		// the WASM 4 GB cap.
		const memSize = ggufCtx.tensors.length * 16384 + (1 << 20);
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const norm = this.makeTensor(tensorMap, "output_norm.weight");
		const output = tensorMap.has("output.weight")
			? this.makeTensor(tensorMap, "output.weight")
			: null;
		const outputBias = tensorMap.has("output.bias")
			? this.makeTensor(tensorMap, "output.bias")
			: null;
		const normBias = tensorMap.has("output_norm.bias")
			? this.makeTensor(tensorMap, "output_norm.bias")
			: null;

		const isPhi3 = hp.architecture === "phi3";
		const layers: LayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			const opt = (s: string) =>
				tensorMap.has(p(s)) ? this.makeTensor(tensorMap, p(s)) : null;
			if (isPhi3) {
				// Phi-3: fused QKV + fused gate-up. The forward graph slices
				// the fused outputs via opView3d / opView2d. Per-layer norms
				// may carry an optional bias (RMSNorm + bias) on some Phi-3
				// variants; Phi-3.5-mini does not have norm biases so the
				// opt() calls return null for this specific model.
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: opt("attn_norm.bias"),
					qkvFused: this.makeTensor(tensorMap, p("attn_qkv.weight")),
					qProj: null,
					kProj: null,
					vProj: null,
					qBias: null,
					kBias: null,
					vBias: null,
					qNorm: null,
					kNorm: null,
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: opt("ffn_norm.bias"),
					gateUpFused: this.makeTensor(tensorMap, p("ffn_up.weight")),
					gateProj: null,
					upProj: null,
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
				});
			} else {
				// Default split-QKV / split-gate-up path used by llama / qwen* /
				// mistral / etc. Existing behaviour preserved exactly; only
				// nulls were added for phi3-only fields.
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: null,
					qkvFused: null,
					qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
					kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
					vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
					qBias: opt("attn_q.bias"),
					kBias: opt("attn_k.bias"),
					vBias: opt("attn_v.bias"),
					qNorm: opt("attn_q_norm.weight"),
					kNorm: opt("attn_k_norm.weight"),
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: null,
					gateUpFused: null,
					gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
					upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
				});
			}
		}

		this.weights = { tokEmb, norm, normBias, output, outputBias, layers };
		this.weightBuf = wasm.backendAllocCtxTensors();

		for (const t of ggufCtx.tensors) {
			const tensor = this.nameToTensor.get(t.name);
			if (!tensor) continue;
			const srcOffset = ggufCtx.dataOffset + t.offset;
			const nbytes = wasm.tensorNbytes(tensor);
			if (isCallback) {
				wasm.uploadRangeChunked(
					tensor,
					(off, len) => dataAt(srcOffset + off, len),
					nbytes,
				);
			} else {
				wasm.uploadToTensorChunked(tensor, dataAt(srcOffset, nbytes));
			}
		}
	}

	initKVCache(maxContextLength: number): void {
		if (this.kvLayers) return;

		const { hp, wasm } = this;
		// KV cache dtype:
		// - FA mode: F16 (2 bytes/elem). Required for ggml-webgpu's
		//   FLASH_ATTN_EXT VEC/TILE paths to engage — see
		//   flash_attn_get_decisions::kv_vec_type_supported in
		//   ggml-webgpu-shader-lib.hpp. opCpy handles F32→F16 on KV writes;
		//   FA reads F16 directly. The §6 -7.7% short-context measurement
		//   was made BEFORE FA could engage and no longer applies.
		// - Manual mode: F32 (4 bytes/elem). Legacy mul_mat-friendly default.
		const kvElemBytes = this.flashAttn ? 2 : 4;
		const perLayerBytes =
			hp.embeddingHeadLength * maxContextLength * kvElemBytes;
		const totalBytes = hp.layerCount * 2 * perLayerBytes;
		const memSize = hp.layerCount * 2 * 16384 + totalBytes + (1 << 20);

		wasm.ctxCreate(memSize);

		// FA mode requires F16 K + F16 V (per §18 commit baad612 +
		// flash_attn_get_decisions::kv_vec_type_supported). Manual mode
		// keeps the legacy F32 layout. opCpy converts F32→F16 on writes.
		const kvType = this.flashAttn ? GgmlType.F16 : GgmlType.F32;
		this.kvLayers = [];
		for (let i = 0; i < hp.layerCount; i++) {
			this.kvLayers.push({
				// K: [headDim, maxCtx, nKvHeads] — same layout in both modes;
				// dtype tracks FA mode.
				k: wasm.tensorNew3d(
					kvType,
					hp.embeddingHeadLength,
					maxContextLength,
					hp.headCountKv,
				),
				// V layout depends on FA mode:
				// - FA mode:     [headDim, maxCtx, nKvHeads] (FA-ready)
				// - Manual mode: [maxCtx, headDim, nKvHeads] (mul_mat compat)
				v: this.flashAttn
					? wasm.tensorNew3d(
							kvType,
							hp.embeddingHeadLength,
							maxContextLength,
							hp.headCountKv,
						)
					: wasm.tensorNew3d(
							kvType,
							maxContextLength,
							hp.embeddingHeadLength,
							hp.headCountKv,
						),
			});
		}

		this.kvBuf = wasm.backendAllocCtxTensors();
		this.nCached = 0;
	}

	resetKVCache(): void {
		this.nCached = 0;
	}

	/**
	 * Roll the logical KV cache size back to `n`. Used after partial accept
	 * in speculative decoding when only some of K drafted tokens are kept.
	 *
	 * The physical KV-buffer slots remain in place — they're overwritten on
	 * the next forward at the rolled-back position.
	 */
	truncateKVCache(n: number): void {
		if (n < 0) throw new Error(`truncateKVCache: n=${n} must be >= 0`);
		if (n > this.nCached) {
			throw new Error(
				`truncateKVCache: n=${n} > current nCached=${this.nCached}; cannot grow via truncate.`,
			);
		}
		this.nCached = n;
	}

	get cachedTokenCount(): number {
		return this.nCached;
	}

	/**
	 * Run a forward pass for the given tokens. When `prefillTileSize > 0`
	 * and `tokenIds.length > prefillTileSize`, the call is automatically
	 * chunked into tiles. Returns the **last** position's logits (matching
	 * the legacy single-call contract).
	 *
	 * KV cache accumulates across tiles via the existing
	 * `this.nCached = totalLen` advance at the end of `forwardSingle()`.
	 * Intermediate-tile readbacks are tiny (`vocabSize * 4 B` per tile,
	 * last position only) and are discarded.
	 */
	async forward(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		const tileSize = this.prefillTileSize;
		if (tileSize === 0 || tokenIds.length <= tileSize) {
			return await this.forwardSingle(tokenIds, positions);
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forward: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		let lastLogits: Float32Array | undefined;
		for (let off = 0; off < tokenIds.length; off += tileSize) {
			const end = Math.min(off + tileSize, tokenIds.length);
			const tileIds = tokenIds.subarray(off, end);
			const tilePos = positions.subarray(off, end);
			lastLogits = await this.forwardSingle(tileIds, tilePos);
		}
		// `lastLogits` is defined because the loop runs at least once
		// (tokenIds.length > tileSize >= 1).
		return lastLogits as Float32Array;
	}

	/**
	 * Build the per-layer QKV projection stage and return reshaped
	 * Q/K/V tensors ready for RoPE.
	 *
	 * Two paths share this helper:
	 * - **Split-QKV** (llama / qwen* / mistral / phi): three matmuls,
	 *   optional bias add, reshape to [headDim, nHeads_q_or_kv, nTokens].
	 * - **Fused-QKV** (phi3): one matmul on `lw.qkvFused`, then 3
	 *   `opView3d` slices over the [E + 2*kvDim, nTokens] output.
	 *   Mirrors the encoder fused path at
	 *   `src/inference/encoder-inference.ts:263-296` and llama.cpp's
	 *   `build_qkv` in `src/llama-graph.cpp`.
	 *
	 * After the split/fused branch the qNorm / kNorm gain is applied
	 * uniformly (Qwen3 family uses these; phi3 doesn't but the field
	 * is null and the branch skips).
	 *
	 * Used by `forwardSingle`, `forwardAllPositions`, `forwardDecode`.
	 * The `debugLayerOutput` path interleaves checkpoints between Q,
	 * K, and V so it cannot share this helper — it has its own inline
	 * split-QKV code that throws if the model is fused.
	 */
	private buildQKV(
		lw: LayerWeights,
		normed: TensorPtr,
		nTokens: number,
		headDim: number,
		nHeads: number,
	): { qReady: TensorPtr; kReady: TensorPtr; v3: TensorPtr } {
		const { wasm, hp } = this;
		let q3: TensorPtr;
		let k3: TensorPtr;
		let v3: TensorPtr;
		if (lw.qkvFused) {
			// Phi-3 fused QKV: one matmul → 3 opView3d slices. No GQA on
			// Phi-3.5-mini (kvDim == E); the math handles GQA-shrunk K/V
			// correctly for any future Phi-3 variants that ship GQA.
			const E = hp.embeddingLength;
			const kvDim = headDim * hp.headCountKv;
			const fusedRowDim = E + 2 * kvDim;
			const qkv = wasm.opMulMat(lw.qkvFused, normed); // [fusedRowDim, nTokens]
			const headBytes = F32_BYTES * headDim;
			const tokenBytes = F32_BYTES * fusedRowDim;
			q3 = wasm.opCont(
				wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, 0),
			);
			k3 = wasm.opCont(
				wasm.opView3d(
					qkv,
					headDim,
					hp.headCountKv,
					nTokens,
					headBytes,
					tokenBytes,
					F32_BYTES * E,
				),
			);
			v3 = wasm.opCont(
				wasm.opView3d(
					qkv,
					headDim,
					hp.headCountKv,
					nTokens,
					headBytes,
					tokenBytes,
					F32_BYTES * (E + kvDim),
				),
			);
			if (this.assertFusedContiguity) {
				assertContiguousF32(wasm, q3, "buildQKV.fused.q3");
				assertContiguousF32(wasm, k3, "buildQKV.fused.k3");
				assertContiguousF32(wasm, v3, "buildQKV.fused.v3");
			}
		} else {
			if (!lw.qProj || !lw.kProj || !lw.vProj) {
				throw new Error(
					`split-QKV path requires qProj/kProj/vProj for ${hp.architecture}`,
				);
			}
			const qRaw = wasm.opMulMat(lw.qProj, normed);
			const kRaw = wasm.opMulMat(lw.kProj, normed);
			const vRaw = wasm.opMulMat(lw.vProj, normed);
			const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
			const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
			const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;
			q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			k3 = wasm.opReshape3d(k, headDim, hp.headCountKv, nTokens);
			v3 = wasm.opReshape3d(v, headDim, hp.headCountKv, nTokens);
		}
		const qReady = lw.qNorm
			? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
			: q3;
		const kReady = lw.kNorm
			? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
			: k3;
		return { qReady, kReady, v3 };
	}

	/**
	 * Build the per-layer FFN gate / up projections.
	 *
	 * - **Split** (llama / qwen* / mistral / phi): two matmuls.
	 * - **Fused gate-up** (phi3): one matmul on `lw.gateUpFused`,
	 *   then 2 `opView2d` slices over the [2*ffSize, nTokens] output.
	 *   Mirrors llama.cpp's `LLM_FFN_SWIGLU` mode where the up
	 *   tensor is [2*ffSize, n_embd] and gets split into halves
	 *   before the SwiGLU multiply.
	 */
	private buildFFNGateUp(
		lw: LayerWeights,
		ffnNormed: TensorPtr,
		nTokens: number,
	): { gate: TensorPtr; up: TensorPtr } {
		const { wasm, hp } = this;
		if (lw.gateUpFused) {
			// llama.cpp's swiglu kernel (ggml-cpu/ops.cpp:3170-3179)
			// computes y = silu(first_half) * second_half when swapped=0.
			// HF Phi3MLP forward is `up * silu(gate)` with `chunk(2, dim=-1)`,
			// so HF puts gate first / up second along the output dim, and
			// llama.cpp's convert_hf_to_gguf.py Phi3MiniModel preserves
			// that order — gate is the FIRST half, up is the SECOND.
			const ffSize = hp.feedForwardLength;
			const fused = wasm.opMulMat(lw.gateUpFused, ffnNormed); // [2*ffSize, nTokens]
			const tokenBytes = F32_BYTES * 2 * ffSize;
			const gate = wasm.opCont(
				wasm.opView2d(fused, ffSize, nTokens, tokenBytes, 0),
			);
			const up = wasm.opCont(
				wasm.opView2d(fused, ffSize, nTokens, tokenBytes, F32_BYTES * ffSize),
			);
			if (this.assertFusedContiguity) {
				assertContiguousF32(wasm, gate, "buildFFNGateUp.fused.gate");
				assertContiguousF32(wasm, up, "buildFFNGateUp.fused.up");
			}
			return { gate, up };
		}
		if (!lw.gateProj || !lw.upProj) {
			throw new Error(
				`split-gate-up path requires gateProj/upProj for ${hp.architecture}`,
			);
		}
		return {
			gate: wasm.opMulMat(lw.gateProj, ffnNormed),
			up: wasm.opMulMat(lw.upProj, ffnNormed),
		};
	}

	/**
	 * Single-call forward pass: one ctxCreate → graph build → graph compute →
	 * ctxFree cycle for the given tokens. **Do not call directly** — public
	 * callers go through `forward()`, which dispatches to either this or the
	 * tiled loop based on `prefillTileSize`.
	 *
	 * The graph build invariants documented inline below are load-bearing —
	 * see comments around `no_alloc=true` rationale, V-cache permute axes,
	 * and last-position-only readback. Do not refactor the body.
	 */
	private async forwardSingle(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// 64 bytes/elem covers long-prefill metadata on 7B+ (32x prior was
		// too tight at seq=512 on Mistral 7B).
		const graphMem = hp.layerCount * 32768 + totalLen * hp.embeddingLength * 64;
		wasm.ctxCreate(graphMem);

		const t1 = trace ? performance.now() : 0;

		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		// Leaf input tensors — data is uploaded below, AFTER backendAllocCtxTensors
		// assigns real GPU buffers. ctxCreate uses no_alloc=true so tensor->data
		// is null until then; tensorSetData (memcpy to tensor->data) would be a no-op.
		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);

		// Causal attention mask tensor: [totalLen, nTokens]. Values are 0 for
		// visible positions and -Infinity for masked. ggml_soft_max_ext consumes
		// this alongside the pre-scale multiplier.
		//
		// For single-token decode (nTokens == 1) the mask is trivially all
		// zeros — the sole query sees every prior key — so we can skip creating
		// and uploading the tensor entirely and pass a null mask to
		// soft_max_ext. Saves one tensor alloc + one `backendTensorSet` per
		// decode step.
		const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		// FA requires F16 mask (ggml.c:5330); opSoftMaxExt accepts F16 too,
		// so this works for both attention paths. Causal mask values are
		// written as F16 bit patterns: 0x0000 = 0.0, 0xFC00 = -Inf.
		const maskTensor = needsMask
			? wasm.tensorNew2d(GgmlType.F16, totalLen, maskPaddedCols)
			: 0;

		// Embedding lookup: get_rows handles Q4_0→F32 dequant (opCpy does not)
		const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);

		// Create the graph up front so each layer can expand its KV-cache writes
		// into it *before* attention ops that read kv.k / kv.v are added. Without
		// this ordering, the cpy (write) and the view (read) have no dependency
		// edge, so attention reads stale data (zeros).
		const graph = wasm.graphNew(hp.layerCount * 64 + 128);

		let cur = x;
		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];
			const kv = this.kvLayers[il];

			// LLaMA RMSNorm: (x / rms(x)) * gamma. ggml_rms_norm only does the
			// normalize step — the per-dim gain `attn_norm.weight` must be applied
			// separately. Same for `ffn_norm.weight` and the final `output_norm.weight`.
			let normed = wasm.opMul(wasm.opRmsNorm(cur, hp.normEpsilon), lw.attnNorm);
			if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

			const { qReady, kReady, v3 } = this.buildQKV(
				lw,
				normed,
				nTokens,
				headDim,
				nHeads,
			);

			const qRope = wasm.opRope(
				qReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);
			const kRope = wasm.opRope(
				kReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);

			// Write K to cache: permute kRope(0,2,1,3) -> [headDim, nTokens, nKvHeads]
			const kNb1 = wasm.tensorNb(kv.k, 1);
			const kNb2 = wasm.tensorNb(kv.k, 2);
			const kWriteView = wasm.opView3d(
				kv.k,
				headDim,
				nTokens,
				hp.headCountKv,
				kNb1,
				kNb2,
				pastLen * kNb1,
			);
			const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
			// ggml_cpy handles non-contiguous src via strides, so the previously
			// mandatory opCont is redundant — dropping it saves a GPU dispatch per
			// layer per forward.
			const kWrite = wasm.opCpy(kRopeP, kWriteView);
			// Expand into the graph NOW so the cpy node precedes attention reads.
			wasm.graphBuildForwardExpand(graph, kWrite);

			// V cache layout depends on this.flashAttn:
			// - FA mode:     [headDim, maxCtx, nKvHeads]  (FA-ready)
			// - Manual mode: [maxCtx, headDim, nKvHeads]  (mul_mat compat)
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);
			let v3P: TensorPtr;
			let vWriteView: TensorPtr;
			if (this.flashAttn) {
				// src(headDim=0, nKvHeads=1, nTokens=2) -> dst(headDim=0, nTokens=1, nKvHeads=2): permute (0, 2, 1, 3).
				v3P = wasm.opPermute(v3, 0, 2, 1, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					headDim,
					nTokens,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb1,
				);
			} else {
				// src(headDim=0, nKvHeads=1, nTokens=2) -> dst(nTokens=0, headDim=1, nKvHeads=2): permute (1, 2, 0, 3).
				const vNb0 = wasm.tensorNb(kv.v, 0);
				v3P = wasm.opPermute(v3, 1, 2, 0, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					nTokens,
					headDim,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb0,
				);
			}
			// opCpy handles strided src — opCont skipped (see K write above).
			const vWrite = wasm.opCpy(v3P, vWriteView);
			wasm.graphBuildForwardExpand(graph, vWrite);

			// Read K from cache: [headDim, totalLen, nKvHeads]
			const fullK = wasm.opView3d(
				kv.k,
				headDim,
				totalLen,
				hp.headCountKv,
				kNb1,
				kNb2,
				0,
			);
			// Read V from cache. FA mode wants [headDim, totalLen, nKvHeads]
			// with stride_v1 = headDim (in elements). Manual mode wants
			// [totalLen, headDim, nKvHeads] for opMulMat(V, attn).
			const fullV = this.flashAttn
				? wasm.opView3d(kv.v, headDim, totalLen, hp.headCountKv, vNb1, vNb2, 0)
				: wasm.opView3d(kv.v, totalLen, headDim, hp.headCountKv, vNb1, vNb2, 0);

			// Permute Q: [headDim, nHeads, nTokens] -> [headDim, nTokens, nHeads]
			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);

			let merged: TensorPtr;
			if (this.flashAttn) {
				// Fused FA: Q=[headDim, nTokens, nHeads], K/V=[headDim, totalLen, nKvHeads]
				// -> [headDim, nHeads, nTokens]. The ggml-webgpu backend picks
				// VEC/TILE shader if F16 K/V (KV is F32 in this plan; backend
				// supports F32 K/V too — see flash_attn_get_decisions).
				const attnOut = wasm.opFlashAttn(
					qp,
					fullK,
					fullV,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0, // max_bias (ALiBi disabled)
					0.0, // logit_softcap (Gemma; not used by Llama/Qwen/Mistral)
				);
				// FA returns contiguous [headDim, nHeads, nTokens] — reshape
				// directly to [embDim, nTokens] for oProj. No permute or
				// opCont needed (this is one of FA's wins).
				merged = wasm.opReshape2d(attnOut, nHeads * headDim, nTokens);
			} else {
				// Manual chain: QK^T -> scaled+masked softmax -> V * attn.
				// QK^T: K=[headDim, totalLen, nKvHeads], Q=[headDim, nTokens, nHeads]
				//       -> [totalLen, nTokens, nHeads].
				const qk = wasm.opMulMat(fullK, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				// V * attn: V=[totalLen, headDim, nKvHeads], attn=[totalLen, nTokens, nHeads]
				//           -> [headDim, nTokens, nHeads].
				const attnOut = wasm.opMulMat(fullV, attnW);
				// Merge heads: [headDim, nTokens, nHeads] -> [headDim, nHeads, nTokens] -> [embDim, nTokens].
				merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);
			}

			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			let ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);
			const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, nTokens);
			// Fused silu(gate) * up — single GPU op instead of silu+mul.
			const ffnHidden = wasm.opSwigluSplit(gate, up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
		}

		let finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		if (weights.normBias) finalNorm = wasm.opAdd(finalNorm, weights.normBias);
		let logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
		if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);

		wasm.graphBuildForwardExpand(graph, logits);

		const t2 = trace ? performance.now() : 0;

		const graphBuf = wasm.backendAllocCtxTensors();

		const t3 = trace ? performance.now() : 0;

		// Upload leaf input data AFTER backend buffers are assigned.
		// All three buffers (pos / tokenIds / mask) are written in a single
		// WASM call via backendTensorSet3 to avoid 2–3 separate FFI hops per
		// forward. Mask slot is skipped entirely (tensor = 0) when !needsMask.
		{
			const posBytes = nTokens * 4;
			const idsBytes = nTokens * 4;
			const maskBytes = needsMask ? totalLen * maskPaddedCols * 2 : 0;
			const totalBytes = posBytes + idsBytes + maskBytes;

			const heap = wasm.malloc(totalBytes);
			try {
				const posPtr = heap;
				const idsPtr = heap + posBytes;
				const maskPtr = heap + posBytes + idsBytes;

				const posView = new Int32Array(wasm.heapU8.buffer, posPtr, nTokens);
				const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, nTokens);
				for (let i = 0; i < nTokens; i++) {
					posView[i] = positions[i];
					idsView[i] = tokenIds[i];
				}

				if (needsMask) {
					// Causal mask: mask[key, query] = -Infinity if key > pastLen + query,
					// else 0. Shape [totalLen, nTokensPadded] stored row-major.
					const mask = new Uint16Array(
						wasm.heapU8.buffer,
						maskPtr,
						totalLen * maskPaddedCols,
					);
					const F16_NEG_INF = 0xfc00;
					for (let q = 0; q < nTokens; q++) {
						const rowBase = q * totalLen;
						const visibleUpTo = pastLen + q;
						for (let k = 0; k < totalLen; k++) {
							mask[rowBase + k] = k <= visibleUpTo ? 0 : F16_NEG_INF;
						}
					}
					// Padding rows past nTokens: zero (unused but keeps buffer defined).
					for (let q = nTokens; q < maskPaddedCols; q++) {
						const rowBase = q * totalLen;
						for (let k = 0; k < totalLen; k++) mask[rowBase + k] = 0;
					}
				}

				wasm.backendTensorSet3(
					posTensor,
					posPtr,
					posBytes,
					tokenIdsTensor,
					idsPtr,
					idsBytes,
					needsMask ? maskTensor : 0,
					maskPtr,
					maskBytes,
				);
			} finally {
				wasm.free(heap);
			}
		}

		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);

		const t5 = trace ? performance.now() : 0;

		const logitsBytes = hp.vocabularySize * 4;
		const offset = nTokens > 1 ? (nTokens - 1) * logitsBytes : 0;
		const resultBuf = await wasm.downloadFromTensor(
			logits,
			logitsBytes,
			offset,
		);
		const result = new Float32Array(
			resultBuf.buffer,
			resultBuf.byteOffset,
			hp.vocabularySize,
		).slice();

		const t6 = trace ? performance.now() : 0;

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		const t7 = trace ? performance.now() : 0;

		if (trace) {
			this.recordTrace(
				"full",
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}

		return result;
	}

	/**
	 * Build a single-pass causal forward over `tokenIds` and return the
	 * post-`output_norm` hidden state as a flat `[E * N]` Float32Array
	 * (row-major-reversed `[E, N]`). **Does not touch the KV cache.**
	 *
	 * Mirrors `CausalLMEmbedder.forwardEmbed` but uses this class's
	 * `buildQKV` / `buildFFNGateUp` helpers so it handles every chat
	 * architecture in the fleet (split + fused QKV; split + fused
	 * gate-up). The chat session's `nCached`, `kvLayers`, and
	 * conversation transcript are unchanged after this call.
	 *
	 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp`
	 * (`res->t_embd = cur` after the final RMSNorm, before `lm_head`).
	 *
	 * Concurrency: the caller must serialize this against any
	 * `forward()` / `forwardSingle()` on the same engine; both paths
	 * share the global WASM ctx-stack.
	 */
	private async forwardForEmbedding(
		tokenIds: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { hp, wasm, weights } = this;
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const graphMem = hp.layerCount * 32768 + N * E * 24;
		wasm.ctxCreate(graphMem);

		try {
			const posTensor = wasm.tensorNew1d(GgmlType.I32, N);
			const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, N);

			const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
			const maskPaddedRows = padTo(N, 32);
			const maskTensor = wasm.tensorNew2d(GgmlType.F16, N, maskPaddedRows);

			const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
			const graph = wasm.graphNew(hp.layerCount * 32 + 128);

			let cur = x;
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				let normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

				const { qReady, kReady, v3 } = this.buildQKV(
					lw,
					normed,
					N,
					headDim,
					nHeads,
				);

				const qRope = wasm.opRope(
					qReady,
					posTensor,
					headDim,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0.0,
					1.0,
					0.0,
					0.0,
				);
				const kRope = wasm.opRope(
					kReady,
					posTensor,
					headDim,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0.0,
					1.0,
					0.0,
					0.0,
				);

				// Manual attention chain — mirrors CausalLMEmbedder.
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				const qk = wasm.opMulMat(kp, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					N,
				);

				const oProj = wasm.opMulMat(lw.oProj, merged);
				const attnResidual = wasm.opAdd(oProj, cur);

				let ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);

				const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, N);
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

				cur = wasm.opAdd(ffnOut, attnResidual);
			}

			// Final output_norm — TAP POINT. No lm_head; no sampling.
			let finalHidden = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				weights.norm,
			);
			if (weights.normBias)
				finalHidden = wasm.opAdd(finalHidden, weights.normBias);

			wasm.graphBuildForwardExpand(graph, finalHidden);
			const graphBuf = wasm.backendAllocCtxTensors();

			try {
				const idsBytes = N * 4;
				const posBytes = N * 4;
				const maskBytes = N * maskPaddedRows * 2;
				const totalBytes = idsBytes + posBytes + maskBytes;
				const heap = wasm.malloc(totalBytes);
				try {
					const idsPtr = heap;
					const posPtr = heap + idsBytes;
					const maskPtr = heap + idsBytes + posBytes;

					const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, N);
					const posView = new Int32Array(wasm.heapU8.buffer, posPtr, N);
					// Positions always start at 0 — embed() processes an isolated
					// sequence with no KV history; applying an offset would shift
					// RoPE relative to the training distribution.
					for (let i = 0; i < N; i++) {
						idsView[i] = tokenIds[i];
						posView[i] = i;
					}

					const F16_NEG_INF = 0xfc00;
					const mask = new Uint16Array(
						wasm.heapU8.buffer,
						maskPtr,
						N * maskPaddedRows,
					);
					for (let q = 0; q < N; q++) {
						const rowBase = q * N;
						for (let k = 0; k < N; k++) {
							mask[rowBase + k] = k <= q ? 0 : F16_NEG_INF;
						}
					}
					for (let q = N; q < maskPaddedRows; q++) {
						const rowBase = q * N;
						for (let k = 0; k < N; k++) mask[rowBase + k] = 0;
					}

					wasm.backendTensorSet3(
						tokenIdsTensor,
						idsPtr,
						idsBytes,
						posTensor,
						posPtr,
						posBytes,
						maskTensor,
						maskPtr,
						maskBytes,
					);
				} finally {
					wasm.free(heap);
				}

				await wasm.graphCompute(graph);

				const totalFloats = E * N;
				const bytes = await wasm.downloadFromTensor(
					finalHidden,
					totalFloats * 4,
				);
				const hidden = new Float32Array(
					bytes.buffer,
					bytes.byteOffset,
					totalFloats,
				);
				return new Float32Array(hidden);
			} finally {
				wasm.backendBufferFree(graphBuf);
			}
		} finally {
			wasm.ctxFree();
		}
	}

	/**
	 * Compute an L2-normalized sentence embedding by running a single-
	 * pass causal forward over `tokenIds`, tapping the post-
	 * `output_norm` hidden state, pooling, and L2-normalizing.
	 * **Does not write to the KV cache** — the chat session's state is
	 * unchanged.
	 *
	 * Pooling (`opts.pooling`, default `"last-token"`):
	 *   - `"last-token"`: take the hidden state at column `N - 1`.
	 *     Canonical bucket D pool; matches qwen3-8b-iq3m ref capture.
	 *   - `"mean"`: average all `N` columns, then L2-normalize. Use on
	 *     models with high last-token anisotropy (e.g., Phi-3.5-mini).
	 *
	 * Concurrency: the caller (typically `engine.embed`) must serialize
	 * this against any concurrent `forward()` / `generate()` on the
	 * same engine. The two paths share the global WASM ctx-stack.
	 */
	async embed(
		tokenIds: Int32Array,
		opts?: { pooling?: "last-token" | "mean" },
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const pooling = opts?.pooling ?? "last-token";
		const hidden = await this.forwardForEmbedding(tokenIds);
		const E = this.hp.embeddingLength;
		const N = tokenIds.length;
		const pooled = new Float32Array(E);
		if (pooling === "mean") {
			for (let col = 0; col < N; col++) {
				const base = col * E;
				for (let i = 0; i < E; i++) pooled[i] += hidden[base + i];
			}
			const invN = 1 / N;
			for (let i = 0; i < E; i++) pooled[i] *= invN;
		} else {
			const lastCol = (N - 1) * E;
			for (let i = 0; i < E; i++) pooled[i] = hidden[lastCol + i];
		}
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}

	/**
	 * Multi-token forward pass that returns logits at **all** input positions.
	 *
	 * Same graph as `forward()` for `nTokens > 1`, but downloads the full logits
	 * tensor instead of slicing the last row. Output is row-major:
	 * `[k * vocabSize + i]` is the logit for the token-after-input-at-position-k,
	 * vocab index i.
	 *
	 * Used by speculative decoding to verify K drafted tokens in one forward.
	 * Caller must `truncateKVCache(pastLen + accepted)` after partial accept.
	 */
	async forwardVerify(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		if (tokenIds.length < 2) {
			throw new Error(
				"forwardVerify requires nTokens >= 2; use forward() for prefill or forwardDecode() for single-step decode.",
			);
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forwardVerify: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		return this.forwardAllPositions(tokenIds, positions);
	}

	/**
	 * Internal worker: same graph as `forward()` for `nTokens > 1`, but the
	 * logits readback returns all `nTokens` rows instead of only the last.
	 *
	 * **This is a verbatim copy of `forward()`'s body** with one targeted
	 * change at the readback. The graph construction (V-permute axes 1,2,0,3,
	 * KV `opCpy` strided-source handling, `graphBuildForwardExpand` ordering
	 * before the attention reads, `tensorSetData` no-op rationale, mask
	 * padding to multiple of 32, etc.) carries load-bearing invariants
	 * documented inline in `forward()`. Read `forward()` for the rationale
	 * before changing anything here. If you need to fix a graph-shape bug,
	 * fix it in both methods or extract a shared helper.
	 */
	private async forwardAllPositions(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// 64 bytes/elem covers long-prefill metadata on 7B+ (32x prior was
		// too tight at seq=512 on Mistral 7B).
		const graphMem = hp.layerCount * 32768 + totalLen * hp.embeddingLength * 64;
		wasm.ctxCreate(graphMem);

		const t1 = trace ? performance.now() : 0;

		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);

		const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		const maskTensor = needsMask
			? wasm.tensorNew2d(GgmlType.F16, totalLen, maskPaddedCols)
			: 0;

		const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);

		const graph = wasm.graphNew(hp.layerCount * 64 + 128);

		let cur = x;
		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];
			const kv = this.kvLayers[il];

			let normed = wasm.opMul(wasm.opRmsNorm(cur, hp.normEpsilon), lw.attnNorm);
			if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

			const { qReady, kReady, v3 } = this.buildQKV(
				lw,
				normed,
				nTokens,
				headDim,
				nHeads,
			);

			const qRope = wasm.opRope(
				qReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);
			const kRope = wasm.opRope(
				kReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);

			const kNb1 = wasm.tensorNb(kv.k, 1);
			const kNb2 = wasm.tensorNb(kv.k, 2);
			const kWriteView = wasm.opView3d(
				kv.k,
				headDim,
				nTokens,
				hp.headCountKv,
				kNb1,
				kNb2,
				pastLen * kNb1,
			);
			const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
			const kWrite = wasm.opCpy(kRopeP, kWriteView);
			wasm.graphBuildForwardExpand(graph, kWrite);

			// V cache write — same dual-layout pattern as forward(); see Task 3.
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);
			let v3P: TensorPtr;
			let vWriteView: TensorPtr;
			if (this.flashAttn) {
				v3P = wasm.opPermute(v3, 0, 2, 1, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					headDim,
					nTokens,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb1,
				);
			} else {
				const vNb0 = wasm.tensorNb(kv.v, 0);
				v3P = wasm.opPermute(v3, 1, 2, 0, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					nTokens,
					headDim,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb0,
				);
			}
			const vWrite = wasm.opCpy(v3P, vWriteView);
			wasm.graphBuildForwardExpand(graph, vWrite);

			const fullK = wasm.opView3d(
				kv.k,
				headDim,
				totalLen,
				hp.headCountKv,
				kNb1,
				kNb2,
				0,
			);
			const fullV = this.flashAttn
				? wasm.opView3d(kv.v, headDim, totalLen, hp.headCountKv, vNb1, vNb2, 0)
				: wasm.opView3d(kv.v, totalLen, headDim, hp.headCountKv, vNb1, vNb2, 0);

			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
			let merged: TensorPtr;
			if (this.flashAttn) {
				const attnOut = wasm.opFlashAttn(
					qp,
					fullK,
					fullV,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
					0.0,
				);
				merged = wasm.opReshape2d(attnOut, nHeads * headDim, nTokens);
			} else {
				const qk = wasm.opMulMat(fullK, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(fullV, attnW);
				merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);
			}

			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			let ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);
			const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, nTokens);
			const ffnHidden = wasm.opSwigluSplit(gate, up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
		}

		let finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		if (weights.normBias) finalNorm = wasm.opAdd(finalNorm, weights.normBias);
		let logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
		if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);

		wasm.graphBuildForwardExpand(graph, logits);

		const t2 = trace ? performance.now() : 0;

		const graphBuf = wasm.backendAllocCtxTensors();

		const t3 = trace ? performance.now() : 0;

		{
			const posBytes = nTokens * 4;
			const idsBytes = nTokens * 4;
			const maskBytes = needsMask ? totalLen * maskPaddedCols * 2 : 0;
			const totalBytes = posBytes + idsBytes + maskBytes;

			const heap = wasm.malloc(totalBytes);
			try {
				const posPtr = heap;
				const idsPtr = heap + posBytes;
				const maskPtr = heap + posBytes + idsBytes;

				const posView = new Int32Array(wasm.heapU8.buffer, posPtr, nTokens);
				const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, nTokens);
				for (let i = 0; i < nTokens; i++) {
					posView[i] = positions[i];
					idsView[i] = tokenIds[i];
				}

				if (needsMask) {
					const mask = new Uint16Array(
						wasm.heapU8.buffer,
						maskPtr,
						totalLen * maskPaddedCols,
					);
					const F16_NEG_INF = 0xfc00;
					for (let q = 0; q < nTokens; q++) {
						const rowBase = q * totalLen;
						const visibleUpTo = pastLen + q;
						for (let k = 0; k < totalLen; k++) {
							mask[rowBase + k] = k <= visibleUpTo ? 0 : F16_NEG_INF;
						}
					}
					for (let q = nTokens; q < maskPaddedCols; q++) {
						const rowBase = q * totalLen;
						for (let k = 0; k < totalLen; k++) mask[rowBase + k] = 0;
					}
				}

				wasm.backendTensorSet3(
					posTensor,
					posPtr,
					posBytes,
					tokenIdsTensor,
					idsPtr,
					idsBytes,
					needsMask ? maskTensor : 0,
					maskPtr,
					maskBytes,
				);
			} finally {
				wasm.free(heap);
			}
		}

		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);

		const t5 = trace ? performance.now() : 0;

		// All-positions readback: download full logits tensor (nTokens rows of
		// vocabSize floats each) starting at offset 0, instead of slicing the
		// last row like forward() does.
		const logitsBytes = hp.vocabularySize * 4;
		const totalBytes = nTokens * logitsBytes;
		const resultBuf = await wasm.downloadFromTensor(logits, totalBytes, 0);
		const result = new Float32Array(
			resultBuf.buffer,
			resultBuf.byteOffset,
			nTokens * hp.vocabularySize,
		).slice();

		const t6 = trace ? performance.now() : 0;

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		const t7 = trace ? performance.now() : 0;

		if (trace) {
			this.recordTrace(
				"verify",
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}

		return result;
	}

	/**
	 * Decode a single token using GPU-side reduction (ARGMAX or TOP_K).
	 * Avoids downloading the full vocabulary logits - returns either a single
	 * token ID (greedy) or a reduced set of indices + values (topk).
	 *
	 * @param tokenIds  - Input token IDs (must be length 1 for decode).
	 * @param positions - Corresponding positions.
	 * @param mode      - 'greedy' for ARGMAX, 'topk' for TOP_K + GET_ROWS.
	 * @param topK      - K value when mode is 'topk'.
	 */
	async forwardDecode(
		tokenIds: Int32Array,
		positions: Int32Array,
		mode: DecodeMode,
		topK?: number,
	): Promise<DecodeResult> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// 64 bytes/elem covers long-prefill metadata on 7B+ (32x prior was
		// too tight at seq=512 on Mistral 7B).
		const graphMem = hp.layerCount * 32768 + totalLen * hp.embeddingLength * 64;
		wasm.ctxCreate(graphMem);

		const t1 = trace ? performance.now() : 0;

		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);

		const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		const maskTensor = needsMask
			? wasm.tensorNew2d(GgmlType.F16, totalLen, maskPaddedCols)
			: 0;

		const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
		const graph = wasm.graphNew(hp.layerCount * 64 + 128);

		let cur = x;
		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];
			const kv = this.kvLayers[il];

			let normed = wasm.opMul(wasm.opRmsNorm(cur, hp.normEpsilon), lw.attnNorm);
			if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

			const { qReady, kReady, v3 } = this.buildQKV(
				lw,
				normed,
				nTokens,
				headDim,
				nHeads,
			);

			const qRope = wasm.opRope(
				qReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);
			const kRope = wasm.opRope(
				kReady,
				posTensor,
				headDim,
				ropeMode,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);

			const kNb1 = wasm.tensorNb(kv.k, 1);
			const kNb2 = wasm.tensorNb(kv.k, 2);
			const kWriteView = wasm.opView3d(
				kv.k,
				headDim,
				nTokens,
				hp.headCountKv,
				kNb1,
				kNb2,
				pastLen * kNb1,
			);
			const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
			wasm.graphBuildForwardExpand(graph, wasm.opCpy(kRopeP, kWriteView));

			// V cache write — same dual-layout pattern as forward(); see Task 3.
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);
			let v3P: TensorPtr;
			let vWriteView: TensorPtr;
			if (this.flashAttn) {
				v3P = wasm.opPermute(v3, 0, 2, 1, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					headDim,
					nTokens,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb1,
				);
			} else {
				const vNb0 = wasm.tensorNb(kv.v, 0);
				v3P = wasm.opPermute(v3, 1, 2, 0, 3);
				vWriteView = wasm.opView3d(
					kv.v,
					nTokens,
					headDim,
					hp.headCountKv,
					vNb1,
					vNb2,
					pastLen * vNb0,
				);
			}
			wasm.graphBuildForwardExpand(graph, wasm.opCpy(v3P, vWriteView));

			const fullK = wasm.opView3d(
				kv.k,
				headDim,
				totalLen,
				hp.headCountKv,
				kNb1,
				kNb2,
				0,
			);
			const fullV = this.flashAttn
				? wasm.opView3d(kv.v, headDim, totalLen, hp.headCountKv, vNb1, vNb2, 0)
				: wasm.opView3d(kv.v, totalLen, headDim, hp.headCountKv, vNb1, vNb2, 0);

			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
			let merged: TensorPtr;
			if (this.flashAttn) {
				const attnOut = wasm.opFlashAttn(
					qp,
					fullK,
					fullV,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
					0.0,
				);
				merged = wasm.opReshape2d(attnOut, nHeads * headDim, nTokens);
			} else {
				const qk = wasm.opMulMat(fullK, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(fullV, attnW);
				merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);
			}
			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			let ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);
			const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, nTokens);
			const ffnHidden = wasm.opSwigluSplit(gate, up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);
			cur = wasm.opAdd(ffnOut, attnResidual);
		}

		let finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		if (weights.normBias) finalNorm = wasm.opAdd(finalNorm, weights.normBias);
		let logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
		if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);

		const t2 = trace ? performance.now() : 0;

		if (mode === "greedy") {
			const argmaxResult = wasm.opArgmax(logits);
			wasm.graphBuildForwardExpand(graph, argmaxResult);

			const graphBuf = wasm.backendAllocCtxTensors();
			const t3 = trace ? performance.now() : 0;
			this.uploadLeaves(
				wasm,
				tokenIds,
				positions,
				nTokens,
				pastLen,
				totalLen,
				needsMask,
				maskPaddedCols,
				posTensor,
				tokenIdsTensor,
				maskTensor,
			);
			const t4 = trace ? performance.now() : 0;

			const graphProfile = await this.computeGraphWithOptionalProfile(
				trace,
				graph,
			);
			const t5 = trace ? performance.now() : 0;

			const buf = await wasm.downloadFromTensor(argmaxResult, 4, 0);
			const tokenId = new Int32Array(buf.buffer, buf.byteOffset, 1)[0];
			const t6 = trace ? performance.now() : 0;

			wasm.backendBufferFree(graphBuf);
			wasm.ctxFree();
			this.nCached = totalLen;
			const t7 = trace ? performance.now() : 0;
			if (trace) {
				this.recordTrace(
					"greedy",
					nTokens,
					pastLen,
					t0,
					t1,
					t2,
					t3,
					t4,
					t5,
					t6,
					t7,
					graphProfile,
				);
			}
			return { tokenId };
		}

		if (mode === "topk" && topK && topK > 0) {
			const topKIndices = wasm.opTopK(logits, topK);
			// `ggml_get_rows(a, b)` produces `[a.ne[0], b.ne[0], ...]` and
			// gathers along `a.ne[1]` (rows). `logits` is `[vocab, 1]` so
			// naively reshaping to `[vocab, 1]` leaves vocab on the inner
			// dim where get_rows can't reach it — the result would be
			// `[vocab, topK]` and the subsequent reshape to `[topK, 1]`
			// trips `GGML_ASSERT(nelements == ne0*ne1)`. Flip so vocab is
			// the row dim: each "row" holds one logit, and get_rows picks
			// the top-K rows.
			const logitsRow = wasm.opReshape2d(logits, 1, hp.vocabularySize);
			const topKValues2D = wasm.opGetRows(logitsRow, topKIndices);
			const topKValues = wasm.opReshape2d(topKValues2D, topK, 1);

			wasm.graphBuildForwardExpand(graph, topKValues);

			const graphBuf = wasm.backendAllocCtxTensors();
			const t3 = trace ? performance.now() : 0;
			this.uploadLeaves(
				wasm,
				tokenIds,
				positions,
				nTokens,
				pastLen,
				totalLen,
				needsMask,
				maskPaddedCols,
				posTensor,
				tokenIdsTensor,
				maskTensor,
			);
			const t4 = trace ? performance.now() : 0;

			const graphProfile = await this.computeGraphWithOptionalProfile(
				trace,
				graph,
			);
			const t5 = trace ? performance.now() : 0;

			const kBytes = topK * 4;
			const idxBuf = await wasm.downloadFromTensor(topKIndices, kBytes, 0);
			const valBuf = await wasm.downloadFromTensor(topKValues, kBytes, 0);
			const t6 = trace ? performance.now() : 0;

			const indices = new Int32Array(
				idxBuf.buffer,
				idxBuf.byteOffset,
				topK,
			).slice();
			const values = new Float32Array(
				valBuf.buffer,
				valBuf.byteOffset,
				topK,
			).slice();

			wasm.backendBufferFree(graphBuf);
			wasm.ctxFree();
			this.nCached = totalLen;
			const t7 = trace ? performance.now() : 0;
			if (trace) {
				this.recordTrace(
					"topk",
					nTokens,
					pastLen,
					t0,
					t1,
					t2,
					t3,
					t4,
					t5,
					t6,
					t7,
					graphProfile,
				);
			}
			return { topKIndices: indices, topKValues: values };
		}

		// Fallback: full logits
		wasm.graphBuildForwardExpand(graph, logits);
		const graphBuf = wasm.backendAllocCtxTensors();
		const t3 = trace ? performance.now() : 0;
		this.uploadLeaves(
			wasm,
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			needsMask,
			maskPaddedCols,
			posTensor,
			tokenIdsTensor,
			maskTensor,
		);
		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);
		const t5 = trace ? performance.now() : 0;

		const logitsBytes = hp.vocabularySize * 4;
		const offset = nTokens > 1 ? (nTokens - 1) * logitsBytes : 0;
		const resultBuf = await wasm.downloadFromTensor(
			logits,
			logitsBytes,
			offset,
		);
		const t6 = trace ? performance.now() : 0;
		const fullLogits = new Float32Array(
			resultBuf.buffer,
			resultBuf.byteOffset,
			hp.vocabularySize,
		).slice();

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;
		const t7 = trace ? performance.now() : 0;
		if (trace) {
			this.recordTrace(
				"full",
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}
		return { logits: fullLogits };
	}

	private async computeGraphWithOptionalProfile(
		trace: boolean,
		graph: number,
	): Promise<GraphComputeProfile | null> {
		// Detailed backend profiling is gated behind trace mode so normal
		// inference and non-profile perf runs keep using the representative
		// graphCompute() path.
		if (!trace) {
			await this.wasm.graphCompute(graph);
			return null;
		}
		await this.wasm.graphComputeWithDetailedProfile(graph);
		return this.wasm.getLastGraphComputeProfile();
	}

	private recordTrace(
		mode: DecodeMode,
		nTokens: number,
		pastLen: number,
		t0: number,
		t1: number,
		t2: number,
		t3: number,
		t4: number,
		t5: number,
		t6: number,
		t7: number,
		graphProfile?: GraphComputeProfile | null,
	): void {
		this.lastTrace = {
			mode,
			nTokens,
			pastLen,
			ctxCreateMs: t1 - t0,
			buildGraphMs: t2 - t1,
			backendAllocMs: t3 - t2,
			uploadLeavesMs: t4 - t3,
			graphComputeMs: t5 - t4,
			downloadResultMs: t6 - t5,
			teardownMs: t7 - t6,
			totalMs: t7 - t0,
			backendProfileTotalMs: graphProfile?.totalMs,
			backendMatmulMs: graphProfile?.matmulMs,
			backendAttentionMs: graphProfile?.attentionMs,
			backendEncodeOverheadMs: graphProfile?.encodeOverheadMs,
			backendDispatchCount: graphProfile?.dispatchCount,
			backendBreakdownAvailable: graphProfile?.breakdownAvailable,
		};
	}

	/** Upload position, token ID, and mask data after backend alloc. */
	private uploadLeaves(
		wasm: GgmlWasm,
		tokenIds: Int32Array,
		positions: Int32Array,
		nTokens: number,
		pastLen: number,
		totalLen: number,
		needsMask: boolean,
		maskPaddedCols: number,
		posTensor: TensorPtr,
		tokenIdsTensor: TensorPtr,
		maskTensor: TensorPtr,
	): void {
		const posBytes = nTokens * 4;
		const idsBytes = nTokens * 4;
		const maskBytes = needsMask ? totalLen * maskPaddedCols * 2 : 0;
		const totalBytes = posBytes + idsBytes + maskBytes;

		const heap = wasm.malloc(totalBytes);
		try {
			const posPtr = heap;
			const idsPtr = heap + posBytes;
			const maskPtr = heap + posBytes + idsBytes;

			const posView = new Int32Array(wasm.heapU8.buffer, posPtr, nTokens);
			const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, nTokens);
			for (let i = 0; i < nTokens; i++) {
				posView[i] = positions[i];
				idsView[i] = tokenIds[i];
			}

			if (needsMask) {
				const mask = new Uint16Array(
					wasm.heapU8.buffer,
					maskPtr,
					totalLen * maskPaddedCols,
				);
				const F16_NEG_INF = 0xfc00;
				for (let q = 0; q < nTokens; q++) {
					const rowBase = q * totalLen;
					const visibleUpTo = pastLen + q;
					for (let k = 0; k < totalLen; k++) {
						mask[rowBase + k] = k <= visibleUpTo ? 0 : F16_NEG_INF;
					}
				}
				for (let q = nTokens; q < maskPaddedCols; q++) {
					const rowBase = q * totalLen;
					for (let k = 0; k < totalLen; k++) mask[rowBase + k] = 0;
				}
			}

			wasm.backendTensorSet3(
				posTensor,
				posPtr,
				posBytes,
				tokenIdsTensor,
				idsPtr,
				idsBytes,
				needsMask ? maskTensor : 0,
				maskPtr,
				maskBytes,
			);
		} finally {
			wasm.free(heap);
		}
	}

	// ── Debug tooling ────────────────────────────────────────────────────
	//
	// These helpers sidestep the normal forward() path to read intermediate
	// tensors directly from the GPU. They exist because the inference pipeline
	// has a long, subtle chain of ops where any one broken step poisons every
	// subsequent prediction, and it's very hard to tell from the final logits
	// alone which step broke. Keep them around — the next bug will need them too.
	//
	// Usage from the browser console (see `smoke-test/real-model.html` for the
	// scaffolding that exposes `window.inference` / `window.tokenizer`):
	//
	//   // Raw dequantized embedding row for a token
	//   const x = await window.inference.debugReadEmbeddingRow(1);  // BOS
	//
	//   // K / V cache contents after a forward pass (use a probe prefill first)
	//   window.inference.resetKVCache();
	//   await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
	//   const k = await window.inference.debugReadKCache(0, 64*4, 0);
	//   const v = await window.inference.debugReadVCache(0, 64*4, 0);
	//
	//   // First N floats of any F32 norm weight after loadWeights()
	//   const g = await window.inference.debugReadNormWeight("attn0", 8);
	//
	//   // Fully run the transformer up through layer L for a single token
	//   // and return the hidden state. Great for "where does it diverge?".
	//   const h = await window.inference.debugLayerOutput(1, /* layerIdx= */ 0);

	/** DEBUG: read back a slice of kv.k for a given layer. */
	async debugReadKCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.kvLayers[layerIdx].k;
		const bytes = await this.wasm.downloadFromTensor(tensor, nBytes, offset);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nBytes / 4);
	}

	/** DEBUG: read back a slice of kv.v for a given layer. */
	async debugReadVCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.kvLayers[layerIdx].v;
		const bytes = await this.wasm.downloadFromTensor(tensor, nBytes, offset);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nBytes / 4);
	}

	/**
	 * DEBUG: run the transformer stack for a single token and return the
	 * hidden state at a specific checkpoint inside a specific layer.
	 *
	 * checkpoint:
	 *   "layer_output" — cur after layer L (attn residual + ffn residual)
	 *   "post_attn"    — cur after layer L's attn residual add, before FFN
	 *   "pre_attn"     — cur entering layer L (= output of layer L-1)
	 *   "attn_out"     — just oProj(...) for layer L, before residual add
	 *   "ffn_out"      — just ffn_down(...) for layer L, before residual add
	 *   "ffn_gate"     — raw ffn_gate_proj output (pre-silu)
	 *   "ffn_up"       — raw ffn_up_proj output
	 *   "ffn_hidden"   — silu(gate) * up, i.e. the input to ffn_down_proj
	 *   "ffn_normed"   — input to FFN after RMSNorm + ffn_norm gain
	 *   "attn_normed"  — input to attention after RMSNorm + attn_norm gain
	 *   "attn_q"       — raw Q projection output (pre-reshape, pre-rope)
	 *   "attn_k"       — raw K projection output
	 *   "attn_v"       — raw V projection output
	 *
	 * The difference (post_attn - pre_attn) == attn_out, and
	 * (layer_output - post_attn) == ffn_out — so these together pinpoint
	 * which branch of a layer is under- or over-shooting.
	 */
	async debugLayerOutput(
		tokenId: number,
		layerIdx: number,
		checkpoint:
			| "layer_output"
			| "post_attn"
			| "pre_attn"
			| "attn_out"
			| "ffn_out"
			| "ffn_gate"
			| "ffn_up"
			| "ffn_hidden"
			| "ffn_normed"
			| "attn_normed"
			| "attn_q"
			| "attn_k"
			| "attn_v" = "layer_output",
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { wasm, hp, weights } = this;
		// debugLayerOutput interleaves checkpoint breaks between
		// individual Q / K / V matmuls and between gate / up matmuls.
		// Phi-3-class architectures fuse QKV and gate-up into single
		// matmuls so those checkpoints aren't addressable. The split
		// path below assumes per-layer split projections exist.
		if (
			weights.layers.length > 0 &&
			(weights.layers[0].qkvFused !== null ||
				weights.layers[0].gateUpFused !== null)
		) {
			throw new Error(
				`debugLayerOutput is split-QKV / split-gate-up only; ${hp.architecture} uses fused projections so per-Q/K/V or per-gate/up checkpoints are not addressable. Use forwardSingle for end-to-end output instead.`,
			);
		}
		const isFfnIntermediate =
			checkpoint === "ffn_gate" ||
			checkpoint === "ffn_up" ||
			checkpoint === "ffn_hidden";
		const kvDim = hp.embeddingHeadLength * hp.headCountKv;
		const nbytes =
			(isFfnIntermediate
				? hp.feedForwardLength
				: checkpoint === "attn_k" || checkpoint === "attn_v"
					? kvDim
					: hp.embeddingLength) * 4;

		wasm.ctxCreate(16 * 1024 * 1024);
		try {
			const posTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const idsTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const maskTensor = wasm.tensorNew2d(GgmlType.F16, 1, 32);
			const ropeMode = getRopeModeForArchitecture(hp.architecture);

			let cur = wasm.opGetRows(weights.tokEmb, idsTensor);
			const graph = wasm.graphNew(2048);

			let returnTensor: TensorPtr = cur;

			for (let il = 0; il <= layerIdx; il++) {
				if (il === layerIdx && checkpoint === "pre_attn") {
					returnTensor = cur;
					break;
				}
				const lw = weights.layers[il];
				const kv = this.kvLayers[il];
				const normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (il === layerIdx && checkpoint === "attn_normed") {
					returnTensor = normed;
					break;
				}
				const qRaw = wasm.opMulMat(lw.qProj as number, normed);
				const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
				if (il === layerIdx && checkpoint === "attn_q") {
					returnTensor = q;
					break;
				}
				const kRaw = wasm.opMulMat(lw.kProj as number, normed);
				const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
				if (il === layerIdx && checkpoint === "attn_k") {
					returnTensor = k;
					break;
				}
				const vRaw = wasm.opMulMat(lw.vProj as number, normed);
				const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;
				if (il === layerIdx && checkpoint === "attn_v") {
					returnTensor = v;
					break;
				}
				const q3 = wasm.opReshape3d(q, hp.embeddingHeadLength, hp.headCount, 1);
				const k3 = wasm.opReshape3d(
					k,
					hp.embeddingHeadLength,
					hp.headCountKv,
					1,
				);
				const v3 = wasm.opReshape3d(
					v,
					hp.embeddingHeadLength,
					hp.headCountKv,
					1,
				);
				const qRope = wasm.opRope(
					q3,
					posTensor,
					hp.embeddingHeadLength,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0,
					1,
					0,
					0,
				);
				const kRope = wasm.opRope(
					k3,
					posTensor,
					hp.embeddingHeadLength,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0,
					1,
					0,
					0,
				);
				const kNb1 = wasm.tensorNb(kv.k, 1);
				const kNb2 = wasm.tensorNb(kv.k, 2);
				const kWriteView = wasm.opView3d(
					kv.k,
					hp.embeddingHeadLength,
					1,
					hp.headCountKv,
					kNb1,
					kNb2,
					0,
				);
				const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
				wasm.graphBuildForwardExpand(
					graph,
					wasm.opCpy(wasm.opCont(kRopeP), kWriteView),
				);
				// V cache write — same dual-layout pattern as forward(); see Task 3.
				// Single-token at offset 0 (no pastLen). Manual mode wraps in
				// opCont because debugLayerOutput's opCpy semantics here differ
				// from forward's (legacy: opCont preserved for shape safety).
				const vNb1 = wasm.tensorNb(kv.v, 1);
				const vNb2 = wasm.tensorNb(kv.v, 2);
				let v3P: TensorPtr;
				let vWriteView: TensorPtr;
				if (this.flashAttn) {
					v3P = wasm.opPermute(v3, 0, 2, 1, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						hp.embeddingHeadLength,
						1,
						hp.headCountKv,
						vNb1,
						vNb2,
						0,
					);
				} else {
					v3P = wasm.opPermute(v3, 1, 2, 0, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						1,
						hp.embeddingHeadLength,
						hp.headCountKv,
						vNb1,
						vNb2,
						0,
					);
				}
				wasm.graphBuildForwardExpand(
					graph,
					wasm.opCpy(wasm.opCont(v3P), vWriteView),
				);
				const fullK = wasm.opView3d(
					kv.k,
					hp.embeddingHeadLength,
					1,
					hp.headCountKv,
					kNb1,
					kNb2,
					0,
				);
				const fullV = this.flashAttn
					? wasm.opView3d(
							kv.v,
							hp.embeddingHeadLength,
							1,
							hp.headCountKv,
							vNb1,
							vNb2,
							0,
						)
					: wasm.opView3d(
							kv.v,
							1,
							hp.embeddingHeadLength,
							hp.headCountKv,
							vNb1,
							vNb2,
							0,
						);
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				let merged: TensorPtr;
				if (this.flashAttn) {
					const attnOut = wasm.opFlashAttn(
						qp,
						fullK,
						fullV,
						maskTensor,
						1 / Math.sqrt(hp.embeddingHeadLength),
						0,
						0,
					);
					merged = wasm.opReshape2d(
						attnOut,
						hp.headCount * hp.embeddingHeadLength,
						1,
					);
				} else {
					const qk = wasm.opMulMat(fullK, qp);
					const attnW = wasm.opSoftMaxExt(
						qk,
						maskTensor,
						1 / Math.sqrt(hp.embeddingHeadLength),
						0,
					);
					const attnOut = wasm.opMulMat(fullV, attnW);
					merged = wasm.opReshape2d(
						wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
						hp.headCount * hp.embeddingHeadLength,
						1,
					);
				}
				const oProj = wasm.opMulMat(lw.oProj, merged);
				if (il === layerIdx && checkpoint === "attn_out") {
					returnTensor = oProj;
					break;
				}
				const attnResidual = wasm.opAdd(oProj, cur);
				if (il === layerIdx && checkpoint === "post_attn") {
					returnTensor = attnResidual;
					break;
				}
				const ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (il === layerIdx && checkpoint === "ffn_normed") {
					returnTensor = ffnNormed;
					break;
				}
				const gate = wasm.opMulMat(lw.gateProj as number, ffnNormed);
				if (il === layerIdx && checkpoint === "ffn_gate") {
					returnTensor = gate;
					break;
				}
				const up = wasm.opMulMat(lw.upProj as number, ffnNormed);
				if (il === layerIdx && checkpoint === "ffn_up") {
					returnTensor = up;
					break;
				}
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				if (il === layerIdx && checkpoint === "ffn_hidden") {
					returnTensor = ffnHidden;
					break;
				}
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);
				if (il === layerIdx && checkpoint === "ffn_out") {
					returnTensor = ffnOut;
					break;
				}
				cur = wasm.opAdd(ffnOut, attnResidual);
				if (il === layerIdx && checkpoint === "layer_output") {
					returnTensor = cur;
				}
			}

			wasm.graphBuildForwardExpand(graph, returnTensor);
			const graphBuf = wasm.backendAllocCtxTensors();

			// Upload inputs
			const sp = wasm.stackSave();
			try {
				const posPtr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, posPtr, 1)[0] = 0;
				wasm.backendTensorSet(posTensor, posPtr, 0, 4);
				const idsPtr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, idsPtr, 1)[0] = tokenId;
				wasm.backendTensorSet(idsTensor, idsPtr, 0, 4);
			} finally {
				wasm.stackRestore(sp);
			}
			const maskHeap = wasm.malloc(32 * 2);
			try {
				new Uint16Array(wasm.heapU8.buffer, maskHeap, 32).fill(0);
				wasm.backendTensorSet(maskTensor, maskHeap, 0, 32 * 2);
			} finally {
				wasm.free(maskHeap);
			}
			await wasm.graphCompute(graph);

			const bytes = await wasm.downloadFromTensor(returnTensor, nbytes, 0);
			wasm.backendBufferFree(graphBuf);
			return new Float32Array(bytes.buffer, bytes.byteOffset, nbytes / 4);
		} finally {
			wasm.ctxFree();
		}
	}

	/**
	 * DEBUG: dequantize and return the embedding row for a single token by
	 * running a one-op graph: `opGetRows(tokEmb, [tokenId])`. Requires
	 * `initKVCache` to have been called first (needs a ctx on the stack).
	 */
	async debugReadEmbeddingRow(tokenId: number): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { wasm, hp, weights } = this;
		const nbytes = hp.embeddingLength * 4;

		wasm.ctxCreate(64 * 1024);
		try {
			const idsTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const out = wasm.opGetRows(weights.tokEmb, idsTensor);
			const graph = wasm.graphNew(8);
			wasm.graphBuildForwardExpand(graph, out);
			const graphBuf = wasm.backendAllocCtxTensors();

			// Upload the token id.
			const sp = wasm.stackSave();
			try {
				const ptr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, ptr, 1)[0] = tokenId;
				wasm.backendTensorSet(idsTensor, ptr, 0, 4);
			} finally {
				wasm.stackRestore(sp);
			}

			await wasm.graphCompute(graph);

			const bytes = await wasm.downloadFromTensor(out, nbytes, 0);
			wasm.backendBufferFree(graphBuf);
			return new Float32Array(bytes.buffer, bytes.byteOffset, nbytes / 4);
		} finally {
			wasm.ctxFree();
		}
	}

	/** DEBUG: read back first N floats of an F32 norm weight. */
	async debugReadNormWeight(
		which: "output" | "attn0" | "ffn0",
		nFloats = 8,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		let tensor: TensorPtr;
		if (which === "output") tensor = this.weights.norm;
		else if (which === "attn0") tensor = this.weights.layers[0].attnNorm;
		else tensor = this.weights.layers[0].ffnNorm;
		const bytes = await this.wasm.downloadFromTensor(tensor, nFloats * 4, 0);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nFloats);
	}

	async dispose(): Promise<void> {
		if (this.kvBuf) {
			this.wasm.backendBufferFree(this.kvBuf);
			this.kvBuf = 0;
		}
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		if (this.kvLayers) {
			this.wasm.ctxFree();
			this.kvLayers = null;
		}
		this.wasm.ctxFree();
		this.weights = null;
		this.nameToTensor.clear();
		this.nCached = 0;
	}

	private makeTensor(
		tensorMap: Map<string, GgufTensorInfo>,
		name: string,
	): TensorPtr {
		const info = tensorMap.get(name);
		if (!info) throw new Error(`Weight "${name}" not found in GGUF`);

		const d = info.dimensions;
		const t = info.type;
		let tensor: TensorPtr;

		if (d.length === 1) tensor = this.wasm.tensorNew1d(t, d[0]);
		else if (d.length === 2) tensor = this.wasm.tensorNew2d(t, d[0], d[1]);
		else if (d.length === 3)
			tensor = this.wasm.tensorNew3d(t, d[0], d[1], d[2]);
		else tensor = this.wasm.tensorNew4d(t, d[0], d[1], d[2], d[3]);

		this.wasm.tensorSetName(tensor, name);
		this.nameToTensor.set(name, tensor);
		return tensor;
	}
}
