import { GgmlWasm, type TensorPtr, type BufferPtr, GgmlType, RopeMode } from "./ggml-wasm.js";
import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";

interface LayerWeights {
	attnNorm: TensorPtr;
	qProj: TensorPtr;
	kProj: TensorPtr;
	vProj: TensorPtr;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	gateProj: TensorPtr;
	upProj: TensorPtr;
	downProj: TensorPtr;
}

interface WeightTensors {
	tokEmb: TensorPtr;
	norm: TensorPtr;
	output: TensorPtr | null;
	layers: LayerWeights[];
}

interface LayerKVCache {
	k: TensorPtr;
	v: TensorPtr;
}

/**
 * Manages model weights loaded into WASM/ggml tensors and runs forward passes.
 *
 * Loads GGUF weight data into ggml tensors allocated on the WebGPU backend,
 * then builds and executes computation graphs for transformer forward passes.
 * Supports incremental decoding via KV cache for O(n) autoregressive generation.
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

	constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
		this.wasm = wasm;
		this.hp = hyperparams;
	}

	/**
	 * Create ggml tensors for all model weights and upload data to GPU.
	 *
	 * Two-phase approach:
	 * 1. Create tensor metadata in ggml context (no GPU allocation yet)
	 * 2. Allocate all tensors on GPU at once via backend_alloc_ctx_tensors
	 * 3. Upload weight data from GGUF ArrayBuffer to GPU tensors
	 */
	loadWeights(ggufCtx: GgufContext, ggufData: ArrayBuffer): void {
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) {
			tensorMap.set(t.name, t);
		}

		const memSize = ggufCtx.tensors.length * 4096 + ggufCtx.totalDataSize;
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const norm = this.makeTensor(tensorMap, "output_norm.weight");
		const output = tensorMap.has("output.weight")
			? this.makeTensor(tensorMap, "output.weight")
			: null;

		const layers: LayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			layers.push({
				attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
				qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
				kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
				vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
				oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
				ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
				gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
				upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
				downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
			});
		}

		this.weights = { tokEmb, norm, output, layers };

		// Allocate all weight tensors on GPU at once
		this.weightBuf = wasm.backendAllocCtxTensors();

		// Upload each tensor's data from GGUF to GPU
		for (const t of ggufCtx.tensors) {
			const tensor = this.nameToTensor.get(t.name);
			if (!tensor) continue;

			const srcOffset = ggufCtx.dataOffset + t.offset;
			const nbytes = wasm.tensorNbytes(tensor);

			// Copy GGUF data to WASM heap, then upload to GPU
			const heapPtr = wasm.malloc(nbytes);
			wasm.heapU8.set(new Uint8Array(ggufData, srcOffset, nbytes), heapPtr);
			wasm.backendTensorSet(tensor, heapPtr, 0, nbytes);
			wasm.free(heapPtr);
		}

		// Weight context stays on the stack; do NOT call ctxFree here.
	}

	/**
	 * Allocate GPU tensors for the KV cache.
	 *
	 * Must be called after loadWeights(). Creates per-layer K and V tensors
	 * sized for the maximum context length, in a separate ggml context that
	 * persists across forward passes.
	 */
	initKVCache(maxContextLength: number): void {
		if (this.kvLayers) return; // already initialized

		const { hp, wasm } = this;
		const perLayerBytes = hp.embeddingHeadLength * maxContextLength * 4; // f32
		const totalBytes = hp.layerCount * 2 * perLayerBytes;
		const memSize = hp.layerCount * 2 * 4096 + totalBytes;

		wasm.ctxCreate(memSize);

		this.kvLayers = [];
		for (let i = 0; i < hp.layerCount; i++) {
			this.kvLayers.push({
				k: wasm.tensorNew3d(GgmlType.F32, hp.embeddingHeadLength, maxContextLength, hp.headCountKv),
				v: wasm.tensorNew3d(GgmlType.F32, hp.embeddingHeadLength, maxContextLength, hp.headCountKv),
			});
		}

		this.kvBuf = wasm.backendAllocCtxTensors();
		// KV context stays on the stack; do NOT call ctxFree here.
		this.nCached = 0;
	}

	/** Reset the KV cache position counter without reallocating GPU memory. */
	resetKVCache(): void {
		this.nCached = 0;
	}

	/** Number of tokens currently stored in the KV cache. */
	get cachedTokenCount(): number {
		return this.nCached;
	}

	/**
	 * Run a forward pass producing logits for the given tokens at the given positions.
	 *
	 * On the first call (prefill), processes all prompt tokens and populates the KV cache.
	 * On subsequent calls (decode), processes only the new token(s) and reuses cached K/V.
	 *
	 * @param tokenIds - Token IDs to process (prefill: all prompt tokens; decode: single token)
	 * @param positions - Position indices for each token
	 * @returns Float32Array of logits, length = vocabularySize
	 */
	forward(tokenIds: Int32Array, _positions: Int32Array): Float32Array {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		// Fresh context for the compute graph (weights and KV cache stay in their own contexts)
		const graphMem = hp.layerCount * 8192 + totalLen * hp.embeddingLength * 8;
		wasm.ctxCreate(graphMem);

		const embDim = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const useGQA = hp.headCountKv !== hp.headCount;

		// Input: embedding lookup for each token
		const x = wasm.tensorNew2d(GgmlType.F32, embDim, nTokens);
		wasm.tensorSetName(x, "input_embeddings");

		for (let t = 0; t < nTokens; t++) {
			const embRow = wasm.opView2d(
				weights.tokEmb, embDim, 1,
				wasm.tensorNb(weights.tokEmb, 1),
				tokenIds[t] * wasm.tensorNb(weights.tokEmb, 1),
			);
			const xRow = wasm.opView2d(
				x, embDim, 1,
				wasm.tensorNb(x, 1),
				t * wasm.tensorNb(x, 1),
			);
			wasm.opCpy(embRow, xRow);
		}

		// Transformer layers
		let cur = x;
		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];
			const kv = this.kvLayers[il];

			// Pre-attention RMS norm
			const normed = wasm.opRmsNorm(cur, hp.normEpsilon);

			// QKV projections
			const q = wasm.opMulMat(lw.qProj, normed);
			const k = wasm.opMulMat(lw.kProj, normed);
			const v = wasm.opMulMat(lw.vProj, normed);

			// Reshape to [headDim, nHeads, nTokens] for RoPE
			const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			const k3 = wasm.opReshape3d(k, headDim, hp.headCountKv, nTokens);
			const v3 = wasm.opReshape3d(v, headDim, hp.headCountKv, nTokens);

			// RoPE on Q and K
			const qRope = wasm.opRope(q3, headDim, RopeMode.NORMAL, hp.contextLength,
				hp.ropeFreqBase, hp.ropeScale, 0.0, 1.0, 0.0, 0.0);
			const kRope = wasm.opRope(k3, headDim, RopeMode.NORMAL, hp.contextLength,
				hp.ropeFreqBase, hp.ropeScale, 0.0, 1.0, 0.0, 0.0);

			// Store new K and V into cache at positions [pastLen..pastLen+nTokens-1]
			const kNb1 = wasm.tensorNb(kv.k, 1);
			const kNb2 = wasm.tensorNb(kv.k, 2);
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);

			const kWriteView = wasm.opView3d(kv.k, headDim, nTokens, hp.headCountKv,
				kNb1, kNb2, pastLen * kNb1);
			const vWriteView = wasm.opView3d(kv.v, headDim, nTokens, hp.headCountKv,
				vNb1, vNb2, pastLen * vNb1);
			wasm.opCpy(kRope, kWriteView);
			wasm.opCpy(v3, vWriteView);

			// Read full K and V from cache [0..totalLen-1]
			const fullK = wasm.opView3d(kv.k, headDim, totalLen, hp.headCountKv,
				kNb1, kNb2, 0);
			const fullV = wasm.opView3d(kv.v, headDim, totalLen, hp.headCountKv,
				vNb1, vNb2, 0);

			// Permute for attention: [headDim, nTokens/totalLen, nHeads/nKvHeads]
			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
			let kp = wasm.opPermute(fullK, 0, 2, 1, 3);
			let vp = wasm.opPermute(fullV, 0, 2, 1, 3);

			// GQA: repeat K and V from nKvHeads to nHeads if needed
			if (useGQA) {
				kp = wasm.opRepeat(kp, qp);
				vp = wasm.opRepeat(vp, qp);
			}

			// QK^T: [totalLen, nTokens, nHeads]
			const qk = wasm.opMulMat(kp, qp);
			const qkScaled = wasm.opScale(qk, 1.0 / Math.sqrt(headDim));
			const qkMasked = wasm.opDiagMaskInf(qkScaled, pastLen);
			const attnW = wasm.opSoftMax(qkMasked);

			// Attention * V: [nTokens, nHeads, totalLen] * [headDim, totalLen, nHeads] -> [headDim, nTokens, nHeads]
			// After permute back and merge: [nHeads*headDim, nTokens]
			const attnOut = wasm.opMulMat(vp, attnW);

			// Merge heads: permute back then reshape
			const merged = wasm.opReshape2d(
				wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
				nHeads * headDim, nTokens,
			);

			// Output projection + residual
			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			// FFN: gate + up -> SiLU(gate) * up -> down
			const ffnNormed = wasm.opRmsNorm(attnResidual, hp.normEpsilon);
			const gate = wasm.opMulMat(lw.gateProj, ffnNormed);
			const up = wasm.opMulMat(lw.upProj, ffnNormed);
			const ffnHidden = wasm.opMul(wasm.opSilu(gate), up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
		}

		// Final norm + LM head
		const finalNorm = wasm.opRmsNorm(cur, hp.normEpsilon);
		const logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);

		// Allocate graph tensors on GPU and compute
		const graphBuf = wasm.backendAllocCtxTensors();
		const graph = wasm.graphNew(hp.layerCount * 32 + 64);
		wasm.graphBuildForwardExpand(graph, logits);
		wasm.graphCompute(graph);

		// Read logits for the last token position
		const logitsBytes = hp.vocabularySize * 4;
		const heapPtr = wasm.malloc(logitsBytes);

		if (nTokens > 1) {
			// For prefill, read only the last token's logits
			const offset = (nTokens - 1) * logitsBytes;
			wasm.backendTensorGet(logits, heapPtr, offset, logitsBytes);
		} else {
			wasm.backendTensorGet(logits, heapPtr, 0, logitsBytes);
		}

		const result = new Float32Array(wasm.heapU8.buffer, heapPtr, hp.vocabularySize).slice();
		wasm.free(heapPtr);
		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();

		// Update cache position
		this.nCached = totalLen;

		return result;
	}

	/** Free all GPU resources. */
	dispose(): void {
		if (this.kvBuf) {
			this.wasm.backendBufferFree(this.kvBuf);
			this.kvBuf = 0;
		}
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		// Free contexts in reverse order (stack discipline)
		if (this.kvLayers) {
			this.wasm.ctxFree();
			this.kvLayers = null;
		}
		this.wasm.ctxFree();
		this.weights = null;
		this.nameToTensor.clear();
		this.nCached = 0;
	}

	/** Create a ggml tensor from GGUF info and register in name map. */
	private makeTensor(tensorMap: Map<string, GgufTensorInfo>, name: string): TensorPtr {
		const info = tensorMap.get(name);
		if (!info) throw new Error(`Weight "${name}" not found in GGUF`);

		const d = info.dimensions;
		const t = info.type;
		let tensor: TensorPtr;

		if (d.length === 1) tensor = this.wasm.tensorNew1d(t, d[0]);
		else if (d.length === 2) tensor = this.wasm.tensorNew2d(t, d[0], d[1]);
		else if (d.length === 3) tensor = this.wasm.tensorNew3d(t, d[0], d[1], d[2]);
		else tensor = this.wasm.tensorNew4d(t, d[0], d[1], d[2], d[3]);

		this.wasm.tensorSetName(tensor, name);
		this.nameToTensor.set(name, tensor);
		return tensor;
	}
}
