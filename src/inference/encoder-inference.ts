import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import type { BufferPtr, GgmlWasm, TensorPtr } from "./ggml-wasm.js";

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
	// biome-ignore lint/correctness/noUnusedPrivateClassMembers: accessed via destructuring in methods
	private hp: ModelHyperparams;
	private weights: EncoderWeights | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();

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
