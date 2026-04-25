import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	GgmlType,
	type GgmlWasm,
	type TensorPtr,
} from "./ggml-wasm.js";

interface EncoderLayerWeights {
	qProj: TensorPtr;
	qBias: TensorPtr;
	kProj: TensorPtr;
	kBias: TensorPtr;
	vProj: TensorPtr;
	vBias: TensorPtr;
	oProj: TensorPtr;
	oBias: TensorPtr;
	attnNormW: TensorPtr;
	attnNormB: TensorPtr;
	ffnUp: TensorPtr;
	ffnUpBias: TensorPtr;
	ffnDown: TensorPtr;
	ffnDownBias: TensorPtr;
	ffnNormW: TensorPtr;
	ffnNormB: TensorPtr;
}

interface EncoderWeights {
	tokEmb: TensorPtr;
	positionEmb: TensorPtr;
	tokenTypes: TensorPtr;
	inputNormW: TensorPtr;
	inputNormB: TensorPtr;
	layers: EncoderLayerWeights[];
}

/**
 * BERT-style bidirectional encoder. Produces a single L2-normalized
 * sentence embedding via forward + pool + normalize. No KV cache.
 */
export class EncoderInference {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: EncoderWeights | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();
	private lastLeaves: {
		tokenIdsTensor: TensorPtr;
		posTensor: TensorPtr;
		segTensor: TensorPtr;
	} | null = null;

	constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
		if (hyperparams.architecture !== "bert") {
			throw new Error(
				`EncoderInference requires architecture "bert", got "${hyperparams.architecture}"`,
			);
		}
		this.wasm = wasm;
		this.hp = hyperparams;
	}

	loadWeights(ggufCtx: GgufContext, ggufData: ArrayBuffer): void {
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) tensorMap.set(t.name, t);

		const memSize =
			ggufCtx.tensors.length * 16384 + ggufCtx.totalDataSize + (1 << 20);
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const positionEmb = this.makeTensor(tensorMap, "position_embd.weight");
		const tokenTypes = this.makeTensor(tensorMap, "token_types.weight");
		const inputNormW = this.makeTensor(tensorMap, "token_embd_norm.weight");
		const inputNormB = this.makeTensor(tensorMap, "token_embd_norm.bias");

		const layers: EncoderLayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			layers.push({
				qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
				qBias: this.makeTensor(tensorMap, p("attn_q.bias")),
				kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
				kBias: this.makeTensor(tensorMap, p("attn_k.bias")),
				vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
				vBias: this.makeTensor(tensorMap, p("attn_v.bias")),
				oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
				oBias: this.makeTensor(tensorMap, p("attn_output.bias")),
				attnNormW: this.makeTensor(tensorMap, p("attn_output_norm.weight")),
				attnNormB: this.makeTensor(tensorMap, p("attn_output_norm.bias")),
				ffnUp: this.makeTensor(tensorMap, p("ffn_up.weight")),
				ffnUpBias: this.makeTensor(tensorMap, p("ffn_up.bias")),
				ffnDown: this.makeTensor(tensorMap, p("ffn_down.weight")),
				ffnDownBias: this.makeTensor(tensorMap, p("ffn_down.bias")),
				ffnNormW: this.makeTensor(tensorMap, p("layer_output_norm.weight")),
				ffnNormB: this.makeTensor(tensorMap, p("layer_output_norm.bias")),
			});
		}

		this.weights = {
			tokEmb,
			positionEmb,
			tokenTypes,
			inputNormW,
			inputNormB,
			layers,
		};
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

	private layerNorm(
		x: TensorPtr,
		gamma: TensorPtr,
		beta: TensorPtr,
	): TensorPtr {
		const normed = this.wasm.opNorm(x, this.hp.normEpsilon);
		return this.wasm.opAdd(this.wasm.opMul(normed, gamma), beta);
	}

	private buildGraph(nTokens: number): TensorPtr {
		if (!this.weights) throw new Error("weights not loaded");
		const { wasm, weights } = this;

		// Leaf inputs — uploaded after backendAllocCtxTensors. Mirrors the
		// causal-LM pattern in model-inference.ts.
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const segTensor = wasm.tensorNew1d(GgmlType.I32, 1);
		this.lastLeaves = { tokenIdsTensor, posTensor, segTensor };

		// Token + position + segment embeddings. Position and segment are added
		// to the per-token embedding before the input LayerNorm. Single-text
		// usage hard-codes segment 0.
		let x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
		x = wasm.opAdd(x, wasm.opGetRows(weights.positionEmb, posTensor));
		x = wasm.opAdd(x, wasm.opGetRows(weights.tokenTypes, segTensor));
		x = this.layerNorm(x, weights.inputNormW, weights.inputNormB);

		const { hp } = this;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const E = hp.embeddingLength;
		const invSqrtHd = 1.0 / Math.sqrt(headDim);

		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];

			// Self-attention with biases, bidirectional (null mask).
			const q = wasm.opAdd(wasm.opMulMat(lw.qProj, x), lw.qBias);
			const k = wasm.opAdd(wasm.opMulMat(lw.kProj, x), lw.kBias);
			const v = wasm.opAdd(wasm.opMulMat(lw.vProj, x), lw.vBias);

			const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			const k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
			const v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);

			const qp = wasm.opPermute(q3, 0, 2, 1, 3);
			const kp = wasm.opPermute(k3, 0, 2, 1, 3);
			// V permute [1,2,0,3] yields [N, headDim, nHeads] logically but
			// leaves nb[0] > nb[1], which ggml_mul_mat asserts against. Match
			// llama.cpp's no-KV-cache BERT path: make V contiguous before the
			// mul_mat that consumes it.
			const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

			const qk = wasm.opMulMat(kp, qp);
			const aw = wasm.opSoftMaxExt(qk, 0, invSqrtHd, 0.0);
			const out = wasm.opMulMat(vp, aw);
			const merged = wasm.opReshape2d(
				wasm.opCont(wasm.opPermute(out, 0, 2, 1, 3)),
				E,
				nTokens,
			);
			const attnProj = wasm.opAdd(wasm.opMulMat(lw.oProj, merged), lw.oBias);

			// Post-attention LayerNorm (BERT post-norm: residual then LN).
			x = this.layerNorm(wasm.opAdd(x, attnProj), lw.attnNormW, lw.attnNormB);

			// FFN: up-project + bias, GeLU, down-project + bias.
			let h = wasm.opAdd(wasm.opMulMat(lw.ffnUp, x), lw.ffnUpBias);
			h = wasm.opGelu(h);
			const ffnProj = wasm.opAdd(wasm.opMulMat(lw.ffnDown, h), lw.ffnDownBias);

			// Post-FFN LayerNorm (BERT post-norm: residual then LN).
			x = this.layerNorm(wasm.opAdd(x, ffnProj), lw.ffnNormW, lw.ffnNormB);
		}

		return x;
	}

	/**
	 * Pool a contiguous `[E, N]` row-major hidden-state buffer down to a
	 * single L2-normalized embedding vector of length `E`. Pure function;
	 * exposed for unit tests and for any caller that has its own forward
	 * compute path.
	 *
	 * Layout matches ggml's row-major-reversed convention used elsewhere
	 * in this codebase: column n occupies bytes `[n*E, (n+1)*E)`.
	 */
	static poolAndNormalize(
		hidden: Float32Array,
		E: number,
		N: number,
		poolingType: "cls" | "mean",
	): Float32Array {
		const pooled = new Float32Array(E);
		if (poolingType === "cls") {
			for (let i = 0; i < E; i++) pooled[i] = hidden[i]; // column 0
		} else {
			for (let n = 0; n < N; n++) {
				const base = n * E;
				for (let i = 0; i < E; i++) pooled[i] += hidden[base + i];
			}
			for (let i = 0; i < E; i++) pooled[i] /= N;
		}
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}

	/**
	 * Run the encoder forward pass over the provided token ids and return
	 * a single L2-normalized embedding. The token id array must already
	 * include any model-specific framing (e.g. [CLS] ... [SEP] for BERT)
	 * — typically produced by the model's WordPiece `Tokenizer.encode`.
	 */
	async embed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const { hp, wasm } = this;
		const N = tokenIds.length;

		// Graph memory budget — sized off layer count + token count, similar
		// to the causal-LM pattern in model-inference.ts but without KV cache
		// allocations.
		const graphMem = hp.layerCount * 32768 + N * hp.embeddingLength * 24;
		wasm.ctxCreate(graphMem);

		const finalHidden = this.buildGraph(N);
		const graph = wasm.graphNew(hp.layerCount * 32 + 128);
		wasm.graphBuildForwardExpand(graph, finalHidden);
		const graphBuf = wasm.backendAllocCtxTensors();

		try {
			// Upload leaf inputs in one bundled FFI call. Mirrors the causal-LM
			// pattern; mask slot is unused for the encoder, so segTensor takes
			// its place.
			const leaves = this.lastLeaves;
			if (!leaves) throw new Error("buildGraph did not set leaves");

			const idsBytes = N * 4;
			const posBytes = N * 4;
			const segBytes = 4;
			const totalBytes = idsBytes + posBytes + segBytes;
			const heap = wasm.malloc(totalBytes);
			try {
				const idsPtr = heap;
				const posPtr = heap + idsBytes;
				const segPtr = heap + idsBytes + posBytes;

				const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, N);
				const posView = new Int32Array(wasm.heapU8.buffer, posPtr, N);
				const segView = new Int32Array(wasm.heapU8.buffer, segPtr, 1);
				for (let i = 0; i < N; i++) {
					idsView[i] = tokenIds[i];
					posView[i] = i;
				}
				segView[0] = 0;

				wasm.backendTensorSet3(
					leaves.tokenIdsTensor,
					idsPtr,
					idsBytes,
					leaves.posTensor,
					posPtr,
					posBytes,
					leaves.segTensor,
					segPtr,
					segBytes,
				);
			} finally {
				wasm.free(heap);
			}

			await wasm.graphCompute(graph);

			// Download `[E, N]` row-major hidden state.
			const E = hp.embeddingLength;
			const totalFloats = E * N;
			const bytes = await wasm.downloadFromTensor(finalHidden, totalFloats * 4);
			const hidden = new Float32Array(
				bytes.buffer,
				bytes.byteOffset,
				totalFloats,
			);

			const pooling = hp.poolingType ?? "cls";
			return EncoderInference.poolAndNormalize(hidden, E, N, pooling);
		} finally {
			wasm.backendBufferFree(graphBuf);
			wasm.ctxFree();
		}
	}

	async dispose(): Promise<void> {
		// Mirror ModelInference.dispose: this class owns the weight buffer and
		// the ctx that holds the weight tensors, so it must free both. (Forward
		// graph teardown will land in Tasks 7-10 alongside the graph itself.)
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
}
