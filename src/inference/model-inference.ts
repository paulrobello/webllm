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

/**
 * Manages model weights loaded into WASM/ggml tensors and runs forward passes.
 *
 * Loads GGUF weight data into ggml tensors allocated on the WebGPU backend,
 * then builds and executes computation graphs for transformer forward passes.
 */
export class ModelInference {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: WeightTensors | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();

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
	}

	/**
	 * Run a forward pass producing logits for the given tokens at the given positions.
	 *
	 * @param tokenIds - Token IDs to process (prefill: all prompt tokens; decode: single token)
	 * @param positions - Position indices for each token
	 * @returns Float32Array of logits, length = vocabularySize (last token's logits for decode)
	 */
	forward(tokenIds: Int32Array, positions: Int32Array): Float32Array {
		if (!this.weights) throw new Error("Weights not loaded");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;

		// Fresh context for the compute graph (weights live in their own context)
		const graphMem = hp.layerCount * 8192 + nTokens * hp.embeddingLength * 8;
		wasm.ctxCreate(graphMem);

		const embDim = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;

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

			// Permute for attention: [headDim, nTokens, nHeads]
			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
			const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
			const vp = wasm.opPermute(v3, 0, 2, 1, 3);

			// QK^T
			const qk = wasm.opMulMat(kp, qp);
			const qkScaled = wasm.opScale(qk, 1.0 / Math.sqrt(headDim));
			const qkMasked = wasm.opDiagMaskInf(qkScaled, positions[0]);
			const attnW = wasm.opSoftMax(qkMasked);

			// Attention * V
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

		return result;
	}

	/** Free all GPU resources. */
	dispose(): void {
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		this.wasm.ctxFree();
		this.weights = null;
		this.nameToTensor.clear();
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
