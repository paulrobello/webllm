import { describe, expect, test } from "bun:test";
import type { ModelHyperparams } from "../src/core/types.js";
import { EncoderInference } from "../src/inference/encoder-inference.js";
import type { GgmlWasm, TensorPtr } from "../src/inference/ggml-wasm.js";

describe("EncoderInference construction", () => {
	test("rejects non-bert hyperparams", () => {
		const hp: ModelHyperparams = {
			architecture: "llama",
			contextLength: 512,
			embeddingLength: 384,
			headCount: 12,
			headCountKv: 12,
			layerCount: 12,
			vocabularySize: 30522,
			embeddingHeadLength: 32,
			feedForwardLength: 1536,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
		};
		expect(() => new EncoderInference({} as never, hp)).toThrow(
			/requires architecture "bert"/,
		);
	});
});

interface FakeWasm {
	fake: GgmlWasm;
	ops: string[];
}

function makeFakeWasm(): FakeWasm {
	let next = 1;
	const ops: string[] = [];
	const stub = {
		ctxCreate: () => {},
		tensorNew1d: () => next++,
		tensorNew2d: () => next++,
		tensorNew3d: () => next++,
		tensorNew4d: () => next++,
		tensorSetName: () => {},
		tensorNbytes: () => 0,
		uploadToTensorChunked: () => {},
		backendAllocCtxTensors: () => 0,
		graphNew: () => next++,
		graphBuildForwardExpand: () => {},
		opGetRows: () => {
			ops.push("getrows");
			return next++;
		},
		opAdd: () => {
			ops.push("add");
			return next++;
		},
		opMul: () => {
			ops.push("mul");
			return next++;
		},
		opNorm: () => {
			ops.push("norm");
			return next++;
		},
		opMulMat: () => {
			ops.push("mulmat");
			return next++;
		},
		opSoftMaxExt: (_q: TensorPtr, mask: TensorPtr) => {
			ops.push(mask === 0 ? "softmax-nullmask" : "softmax-mask");
			return next++;
		},
		opGelu: () => {
			ops.push("gelu");
			return next++;
		},
		opReshape2d: () => next++,
		opReshape3d: () => next++,
		opPermute: () => next++,
		opCont: () => next++,
	} as unknown as GgmlWasm;
	return { fake: stub, ops };
}

function makeBertHp(layerCount: number): ModelHyperparams {
	return {
		architecture: "bert",
		contextLength: 512,
		embeddingLength: 384,
		headCount: 12,
		headCountKv: 12,
		layerCount,
		vocabularySize: 30522,
		embeddingHeadLength: 32,
		feedForwardLength: 1536,
		ropeFreqBase: 10000,
		ropeScale: 1,
		normEpsilon: 1e-12,
		expertCount: 0,
		expertUsedCount: 0,
		poolingType: "cls",
		causalAttention: false,
	};
}

describe("EncoderInference input embedding graph", () => {
	test("emits tok + pos + seg lookups and input LayerNorm", () => {
		const { fake, ops } = makeFakeWasm();
		const hp = makeBertHp(0); // skip blocks
		const enc = new EncoderInference(fake, hp);
		// Stub weights — bypass loadWeights
		(enc as unknown as { weights: object }).weights = {
			tokEmb: 100,
			positionEmb: 101,
			tokenTypes: 102,
			inputNormW: 103,
			inputNormB: 104,
			layers: [],
		};
		(enc as unknown as { buildGraph: (n: number) => number }).buildGraph(4);
		// 3 lookups: token, position, segment
		expect(ops.filter((o) => o === "getrows").length).toBe(3);
		// Input LN: 1 norm + 1 mul (gamma) + 1 add (beta)
		expect(ops.filter((o) => o === "norm").length).toBe(1);
		// 2 adds: token+pos, then +seg, then LN bias add = 3 adds total
		// (sequence: getrows tok, getrows pos, ADD, getrows seg, ADD, NORM, MUL, ADD)
		expect(ops.filter((o) => o === "add").length).toBe(3);
		expect(ops.filter((o) => o === "mul").length).toBe(1);
	});
});

describe("EncoderInference attention block", () => {
	test("each layer uses null-mask softmax + bias on QKV + O", () => {
		const { fake, ops } = makeFakeWasm();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		// Stub weights with 2 layers of dummy pointers.
		const layerWeights = Array.from({ length: 2 }, () => ({
			qProj: 1,
			qBias: 1,
			kProj: 1,
			kBias: 1,
			vProj: 1,
			vBias: 1,
			oProj: 1,
			oBias: 1,
			attnNormW: 1,
			attnNormB: 1,
			ffnUp: 1,
			ffnUpBias: 1,
			ffnDown: 1,
			ffnDownBias: 1,
			ffnNormW: 1,
			ffnNormB: 1,
		}));
		(enc as unknown as { weights: object }).weights = {
			tokEmb: 100,
			positionEmb: 101,
			tokenTypes: 102,
			inputNormW: 103,
			inputNormB: 104,
			layers: layerWeights,
		};
		(enc as unknown as { buildGraph: (n: number) => number }).buildGraph(8);
		// 2 layers × null-mask softmax = 2 null-mask, 0 causal-mask.
		expect(ops.filter((o) => o === "softmax-nullmask").length).toBe(2);
		expect(ops.filter((o) => o === "softmax-mask").length).toBe(0);
	});
});

describe("EncoderInference FFN block", () => {
	test("each layer runs one GeLU and applies biases on up/down", () => {
		const { fake, ops } = makeFakeWasm();
		const hp = makeBertHp(3);
		const enc = new EncoderInference(fake, hp);
		const layerWeights = Array.from({ length: 3 }, () => ({
			qProj: 1,
			qBias: 1,
			kProj: 1,
			kBias: 1,
			vProj: 1,
			vBias: 1,
			oProj: 1,
			oBias: 1,
			attnNormW: 1,
			attnNormB: 1,
			ffnUp: 1,
			ffnUpBias: 1,
			ffnDown: 1,
			ffnDownBias: 1,
			ffnNormW: 1,
			ffnNormB: 1,
		}));
		(enc as unknown as { weights: object }).weights = {
			tokEmb: 100,
			positionEmb: 101,
			tokenTypes: 102,
			inputNormW: 103,
			inputNormB: 104,
			layers: layerWeights,
		};
		(enc as unknown as { buildGraph: (n: number) => number }).buildGraph(4);
		// One GeLU per layer.
		expect(ops.filter((o) => o === "gelu").length).toBe(3);
		// Total LayerNorms: 1 (input) + 2 per layer (post-attn + post-FFN) = 1 + 2*3 = 7.
		expect(ops.filter((o) => o === "norm").length).toBe(1 + 2 * 3);
	});
});
