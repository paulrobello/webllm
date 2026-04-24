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
