import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	GgmlType,
	type GgmlWasm,
	RopeMode,
	type TensorPtr,
} from "./ggml-wasm.js";

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
 *
 * V cache is stored as [maxCtx, headDim, nKvHeads] so that mul_mat(V, attn)
 * produces [headDim, nTokens, nHeads] with built-in GQA broadcast.
 */
export class ModelInference {
	private wasm: GgmlWasm;
	// biome-ignore lint/correctness/noUnusedPrivateClassMembers: accessed via destructuring in methods
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

	loadWeights(ggufCtx: GgufContext, ggufData: ArrayBuffer): void {
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) {
			tensorMap.set(t.name, t);
		}

		const memSize =
			ggufCtx.tensors.length * 16384 + ggufCtx.totalDataSize + (1 << 20);
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
		this.weightBuf = wasm.backendAllocCtxTensors();

		for (const t of ggufCtx.tensors) {
			const tensor = this.nameToTensor.get(t.name);
			if (!tensor) continue;
			const srcOffset = ggufCtx.dataOffset + t.offset;
			const nbytes = wasm.tensorNbytes(tensor);
			wasm.uploadToTensorChunked(
				tensor,
				new Uint8Array(ggufData, srcOffset, nbytes),
			);
		}
	}

	initKVCache(maxContextLength: number): void {
		if (this.kvLayers) return;

		const { hp, wasm } = this;
		const perLayerBytes = hp.embeddingHeadLength * maxContextLength * 4;
		const totalBytes = hp.layerCount * 2 * perLayerBytes;
		const memSize = hp.layerCount * 2 * 16384 + totalBytes + (1 << 20);

		wasm.ctxCreate(memSize);

		this.kvLayers = [];
		for (let i = 0; i < hp.layerCount; i++) {
			this.kvLayers.push({
				// K: [headDim, maxCtx, nKvHeads]
				k: wasm.tensorNew3d(
					GgmlType.F32,
					hp.embeddingHeadLength,
					maxContextLength,
					hp.headCountKv,
				),
				// V: [maxCtx, headDim, nKvHeads] for ggml_mul_mat compatibility
				v: wasm.tensorNew3d(
					GgmlType.F32,
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

	get cachedTokenCount(): number {
		return this.nCached;
	}

	async forward(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const graphMem = hp.layerCount * 32768 + totalLen * hp.embeddingLength * 32;
		wasm.ctxCreate(graphMem);

		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;

		// Leaf input tensors — data is uploaded below, AFTER backendAllocCtxTensors
		// assigns real GPU buffers. ctxCreate uses no_alloc=true so tensor->data
		// is null until then; tensorSetData (memcpy to tensor->data) would be a no-op.
		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);

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
			const normed = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				lw.attnNorm,
			);

			const q = wasm.opMulMat(lw.qProj, normed);
			const k = wasm.opMulMat(lw.kProj, normed);
			const v = wasm.opMulMat(lw.vProj, normed);

			const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			const k3 = wasm.opReshape3d(k, headDim, hp.headCountKv, nTokens);
			const v3 = wasm.opReshape3d(v, headDim, hp.headCountKv, nTokens);

			const qRope = wasm.opRope(
				q3,
				posTensor,
				headDim,
				RopeMode.NORMAL,
				hp.contextLength,
				hp.ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);
			const kRope = wasm.opRope(
				k3,
				posTensor,
				headDim,
				RopeMode.NORMAL,
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
			const kWrite = wasm.opCpy(wasm.opCont(kRopeP), kWriteView);
			// Expand into the graph NOW so the cpy node precedes attention reads.
			wasm.graphBuildForwardExpand(graph, kWrite);

			// Write V to cache: permute v3(2,0,1,3) -> [nTokens, headDim, nKvHeads]
			const vNb0 = wasm.tensorNb(kv.v, 0);
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);
			const v3P = wasm.opPermute(v3, 2, 0, 1, 3);
			const vWriteView = wasm.opView3d(
				kv.v,
				nTokens,
				headDim,
				hp.headCountKv,
				vNb1,
				vNb2,
				pastLen * vNb0,
			);
			const vWrite = wasm.opCpy(wasm.opCont(v3P), vWriteView);
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
			// Read V from cache: [totalLen, headDim, nKvHeads]
			const fullV = wasm.opView3d(
				kv.v,
				totalLen,
				headDim,
				hp.headCountKv,
				vNb1,
				vNb2,
				0,
			);

			// Permute Q: [headDim, nHeads, nTokens] -> [headDim, nTokens, nHeads]
			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);

			// QK^T: K=[headDim, totalLen, nKvHeads], Q=[headDim, nTokens, nHeads] -> [totalLen, nTokens, nHeads]
			const qk = wasm.opMulMat(fullK, qp);
			const qkScaled = wasm.opScale(qk, 1.0 / Math.sqrt(headDim));
			const qkMasked = wasm.opDiagMaskInf(qkScaled, pastLen);
			const attnW = wasm.opSoftMax(qkMasked);

			// V * attn: V=[totalLen, headDim, nKvHeads], attn=[totalLen, nTokens, nHeads] -> [headDim, nTokens, nHeads]
			const attnOut = wasm.opMulMat(fullV, attnW);

			// Merge heads: [headDim, nTokens, nHeads] -> permute -> [headDim, nHeads, nTokens] -> [embDim, nTokens]
			const merged = wasm.opReshape2d(
				wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
				nHeads * headDim,
				nTokens,
			);

			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			const ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			const gate = wasm.opMulMat(lw.gateProj, ffnNormed);
			const up = wasm.opMulMat(lw.upProj, ffnNormed);
			const ffnHidden = wasm.opMul(wasm.opSilu(gate), up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
		}

		const finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		const logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);

		const graphBuf = wasm.backendAllocCtxTensors();

		// Upload leaf input data AFTER backend buffers are assigned.
		// backendTensorSet writes into the real GPU buffer; tensorSetData
		// (memcpy via tensor->data) is invalid for backend-allocated tensors.
		{
			const sp = wasm.stackSave();
			try {
				const posPtr = wasm.stackAlloc(nTokens * 4);
				const posView = new Int32Array(wasm.heapU8.buffer, posPtr, nTokens);
				for (let i = 0; i < nTokens; i++) posView[i] = positions[i];
				wasm.backendTensorSet(posTensor, posPtr, 0, nTokens * 4);

				const idsPtr = wasm.stackAlloc(nTokens * 4);
				const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, nTokens);
				for (let i = 0; i < nTokens; i++) idsView[i] = tokenIds[i];
				wasm.backendTensorSet(tokenIdsTensor, idsPtr, 0, nTokens * 4);
			} finally {
				wasm.stackRestore(sp);
			}
		}

		// KV cache writes are already expanded into the graph per layer (above),
		// so they precede attention reads. Finally add logits + its dependency
		// chain — nodes already in the graph are deduped by build_forward_expand.
		wasm.graphBuildForwardExpand(graph, logits);

		await wasm.graphCompute(graph);

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
		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		return result;
	}

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
