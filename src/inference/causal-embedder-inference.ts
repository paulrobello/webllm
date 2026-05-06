import {
	CAUSAL_EMBEDDER_ARCHITECTURES,
	isCausalEmbedderArchitecture,
	type ModelHyperparams,
} from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	GgmlType,
	type GgmlWasm,
	type TensorPtr,
} from "./ggml-wasm.js";
import { getRopeModeForArchitecture } from "./model-inference.js";

/**
 * Per-layer weight tensors for a Qwen3-style causal LM. Mirrors the chat
 * Qwen3 layout but `output.weight` is intentionally NOT loaded — the
 * embedder bypasses the `lm_head` matmul.
 */
interface CausalEmbedderLayerWeights {
	attnNorm: TensorPtr;
	qProj: TensorPtr;
	kProj: TensorPtr;
	vProj: TensorPtr;
	/** Per-head Q-norm; present in Qwen3, optional in this loader. */
	qNorm: TensorPtr | null;
	/** Per-head K-norm; present in Qwen3, optional in this loader. */
	kNorm: TensorPtr | null;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	gateProj: TensorPtr;
	upProj: TensorPtr;
	downProj: TensorPtr;
}

interface CausalEmbedderWeights {
	tokEmb: TensorPtr;
	/** Final pre-`lm_head` norm gain (`output_norm.weight`). */
	norm: TensorPtr;
	layers: CausalEmbedderLayerWeights[];
}

/**
 * Causal-LM-derived embedder. Runs the standard causal forward graph through
 * all input tokens in one pass, taps the hidden state at post-`output_norm`
 * (before `lm_head`), pools last-token, L2-normalizes, returns Float32Array.
 *
 * Sibling to `EncoderInference` and `ModelInference`. Owns its own weight
 * buffer and ctx. No KV cache.
 *
 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp:91-104`
 * (`res->t_embd = cur` after `output_norm`, before `lm_head`).
 *
 * The `embed()` method and `forwardEmbed()` helper land in Task 6.
 */
export class CausalLMEmbedder {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: CausalEmbedderWeights | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();

	constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
		if (!isCausalEmbedderArchitecture(hyperparams.architecture)) {
			throw new Error(
				`CausalLMEmbedder does not support architecture "${hyperparams.architecture}"; supported: ${CAUSAL_EMBEDDER_ARCHITECTURES.join(", ")}`,
			);
		}
		this.wasm = wasm;
		this.hp = hyperparams;
	}

	loadWeights(
		ggufCtx: GgufContext,
		ggufData: Uint8Array | ((offset: number, byteLength: number) => Uint8Array),
	): void {
		// Mirror EncoderInference.loadWeights exactly. Callback form is
		// required when the source bytes live in the WASM heap; see
		// ModelInference.loadWeights for the full rationale.
		const isCallback = typeof ggufData === "function";
		const dataAt = isCallback
			? ggufData
			: (off: number, len: number) =>
					new Uint8Array(ggufData.buffer, ggufData.byteOffset + off, len);
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) tensorMap.set(t.name, t);

		// ctxCreate uses no_alloc=true; tensor data lives in GPU buffers,
		// not in the ggml mempool. Only metadata budget is needed.
		const memSize = ggufCtx.tensors.length * 16384 + (1 << 20);
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const norm = this.makeTensor(tensorMap, "output_norm.weight");

		const layers: CausalEmbedderLayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			layers.push({
				attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
				qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
				kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
				vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
				// Qwen3 carries per-head Q/K norms; absent on Qwen2/older.
				qNorm: this.makeTensorOptional(tensorMap, p("attn_q_norm.weight")),
				kNorm: this.makeTensorOptional(tensorMap, p("attn_k_norm.weight")),
				oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
				ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
				gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
				upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
				downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
			});
		}

		this.weights = { tokEmb, norm, layers };
		this.weightBuf = wasm.backendAllocCtxTensors();

		// Upload tensor bytes via the WASM heap. Lifted from
		// EncoderInference.loadWeights — uses the chunked uploaders, which
		// transit data through the heap (malloc + backendTensorSet + free)
		// internally. Tensors not requested by this loader (e.g.
		// `output.weight`) are skipped.
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

	private makeTensorOptional(
		tensorMap: Map<string, GgufTensorInfo>,
		name: string,
	): TensorPtr | null {
		return tensorMap.has(name) ? this.makeTensor(tensorMap, name) : null;
	}

	async dispose(): Promise<void> {
		// This class owns the weight buffer and the ctx that holds the weight
		// tensors, so it must free both. Mirrors EncoderInference.dispose.
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		if (this.weights) {
			this.wasm.ctxFree();
			this.weights = null;
		}
		this.nameToTensor.clear();
	}

	/**
	 * Run causal forward + last-token pool + L2 normalize. Returns the
	 * embedding as a Float32Array of length `embeddingLength` (= 1024 for
	 * Qwen3-Embedding-0.6B).
	 *
	 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp:91-104`
	 * — `res->t_embd = cur` after `output_norm` and before `lm_head`.
	 */
	async embed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const hidden = await this.forwardEmbed(tokenIds);
		// Last-token pool: select column N-1 from row-major-reversed [E, N].
		// Column n occupies bytes [n*E, (n+1)*E).
		const E = this.hp.embeddingLength;
		const N = tokenIds.length;
		const lastCol = (N - 1) * E;
		const pooled = new Float32Array(E);
		for (let i = 0; i < E; i++) pooled[i] = hidden[lastCol + i];
		// L2-normalize.
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}

	/**
	 * Build the causal forward graph over all `tokenIds` in one pass and
	 * return the hidden state at `cur` AFTER `output_norm` and BEFORE the
	 * (omitted) `lm_head` matmul. Shape: row-major-reversed `[E, N]`.
	 *
	 * Mirrors `ModelInference.forwardSingle` minus KV cache + lm_head +
	 * sampling. Causal mask is built over `[N, N]` since the embedder
	 * always processes >=1 token in a single pass with no past state.
	 */
	private async forwardEmbed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("weights not loaded");
		const { hp, wasm, weights } = this;
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const nKvHeads = hp.headCountKv;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		// Graph memory budget — modeled on encoder-inference.ts `embed`. No KV
		// cache so the totalLen multiplier is just N (not pastLen + N).
		const graphMem = hp.layerCount * 32768 + N * E * 24;
		wasm.ctxCreate(graphMem);

		try {
			// Leaf inputs.
			const posTensor = wasm.tensorNew1d(GgmlType.I32, N);
			const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, N);

			// Causal mask. Shape mirrors the canonical model-inference.ts
			// orientation: ne[0] = N (key dim), ne[1] = padded query dim
			// (multiple of 32). F16 bit patterns: 0x0000 = +0.0, 0xFC00 = -Inf.
			// ggml_soft_max_ext accepts F16 mask.
			const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
			const maskPaddedRows = padTo(N, 32);
			const maskTensor = wasm.tensorNew2d(GgmlType.F16, N, maskPaddedRows);

			// Token embedding lookup.
			const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);

			// Build the graph up-front so layer ops expand into it as we go.
			const graph = wasm.graphNew(hp.layerCount * 32 + 128);

			let cur = x;
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				// LLaMA RMSNorm: (x / rms(x)) * gamma.
				const normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);

				// Q/K/V projections — split (no fused QKV in Qwen3-Embedding-0.6B).
				const q = wasm.opMulMat(lw.qProj, normed);
				const k = wasm.opMulMat(lw.kProj, normed);
				const v = wasm.opMulMat(lw.vProj, normed);

				// Reshape Q/K/V to [headDim, nHeads, N] / [headDim, nKvHeads, N].
				const q3 = wasm.opReshape3d(q, headDim, nHeads, N);
				const k3 = wasm.opReshape3d(k, headDim, nKvHeads, N);
				const v3 = wasm.opReshape3d(v, headDim, nKvHeads, N);

				// Per-head Q/K RMSNorm (Qwen3-specific).
				const qNormed = lw.qNorm
					? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
					: q3;
				const kNormed = lw.kNorm
					? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
					: k3;

				// RoPE on Q and K.
				const qRope = wasm.opRope(
					qNormed,
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
					kNormed,
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

				// Permute Q to [headDim, N, nHeads], K to [headDim, N, nKvHeads],
				// V to [N, headDim, nKvHeads] for the manual attention chain.
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				// Attention: QK^T -> scaled+masked softmax -> V * attn.
				const qk = wasm.opMulMat(kp, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				// Merge heads: [headDim, N, nHeads] -> [E, N].
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					N,
				);

				const oProj = wasm.opMulMat(lw.oProj, merged);
				const attnResidual = wasm.opAdd(oProj, cur);

				const ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				const gate = wasm.opMulMat(lw.gateProj, ffnNormed);
				const up = wasm.opMulMat(lw.upProj, ffnNormed);
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

				cur = wasm.opAdd(ffnOut, attnResidual);
			}

			// Final output_norm — TAP POINT (qwen3.cpp:98 res->t_embd = cur).
			const finalHidden = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				weights.norm,
			);

			wasm.graphBuildForwardExpand(graph, finalHidden);
			const graphBuf = wasm.backendAllocCtxTensors();

			try {
				// Upload leaf inputs in one bundled FFI call (mirrors model-
				// inference.ts forwardSingle): pos + tokenIds + mask via
				// backendTensorSet3.
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
					for (let i = 0; i < N; i++) {
						idsView[i] = tokenIds[i];
						posView[i] = i;
					}

					// Causal mask: mask[q, k] = -Inf if k > q else 0.
					// Layout: row-major over (query, key) with stride N per row.
					// Rows past N (padding rows up to maskPaddedRows) stay zero.
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

				// Download the [E, N] hidden state.
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
				// Copy into a stable Float32Array since the heap-backed view is
				// invalidated when the next malloc/grow happens.
				return new Float32Array(hidden);
			} finally {
				wasm.backendBufferFree(graphBuf);
			}
		} finally {
			wasm.ctxFree();
		}
	}
}
