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

describe("EncoderInference pool + normalize", () => {
	test("CLS pool picks column 0 and L2 normalizes", () => {
		// Hidden state E=4, N=3, column 0 = [3, 4, 0, 0]
		// → pooled [3, 4, 0, 0], norm=5, normalized [0.6, 0.8, 0, 0]
		const E = 4;
		const N = 3;
		const hidden = new Float32Array([
			3,
			4,
			0,
			0, // col 0
			9,
			9,
			9,
			9, // col 1
			7,
			7,
			7,
			7, // col 2
		]);
		const out = EncoderInference.poolAndNormalize(hidden, E, N, "cls");
		expect(out[0]).toBeCloseTo(0.6);
		expect(out[1]).toBeCloseTo(0.8);
		expect(out[2]).toBeCloseTo(0);
		expect(out[3]).toBeCloseTo(0);
	});

	test("MEAN pool averages across N and L2 normalizes", () => {
		const E = 2;
		const N = 2;
		const hidden = new Float32Array([1, 0, 3, 0]); // col0=[1,0], col1=[3,0]
		// mean = [2, 0]; normalized = [1, 0]
		const out = EncoderInference.poolAndNormalize(hidden, E, N, "mean");
		expect(out[0]).toBeCloseTo(1);
		expect(out[1]).toBeCloseTo(0);
	});

	test("handles zero-norm pooled vector by returning zeros", () => {
		const E = 2;
		const N = 1;
		const hidden = new Float32Array([0, 0]);
		const out = EncoderInference.poolAndNormalize(hidden, E, N, "cls");
		expect(out[0]).toBe(0);
		expect(out[1]).toBe(0);
	});

	test("output length always equals E", () => {
		const out = EncoderInference.poolAndNormalize(
			new Float32Array(8),
			4,
			2,
			"mean",
		);
		expect(out.length).toBe(4);
	});
});

describe("EncoderInference embed() validation", () => {
	test("throws on empty token id array", async () => {
		const { fake } = makeFakeWasm();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		// Set weights stub so the empty-input check fires before graph build.
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
		await expect(enc.embed(new Int32Array(0))).rejects.toThrow(/empty input/);
	});

	test("throws when weights not loaded", async () => {
		const { fake } = makeFakeWasm();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		await expect(enc.embed(new Int32Array([1, 2, 3]))).rejects.toThrow(
			/weights not loaded/,
		);
	});
});

interface FakeEmbedWasm {
	fake: GgmlWasm;
	counters: {
		ctxCreate: number;
		ctxFree: number;
		backendAllocCtxTensors: number;
		backendBufferFree: number;
		graphNew: number;
		graphBuildForwardExpand: number;
	};
	callOrder: string[];
}

function makeFakeWasmForEmbed(): FakeEmbedWasm {
	let next = 1;
	const counters = {
		ctxCreate: 0,
		ctxFree: 0,
		backendAllocCtxTensors: 0,
		backendBufferFree: 0,
		graphNew: 0,
		graphBuildForwardExpand: 0,
	};
	const callOrder: string[] = [];
	// Real heap-backed buffer so `new Int32Array(heapU8.buffer, ptr, N)` works.
	const heap = new Uint8Array(new ArrayBuffer(4096));
	const stub = {
		ctxCreate: () => {
			counters.ctxCreate += 1;
			callOrder.push("ctxCreate");
			return 0;
		},
		ctxFree: () => {
			counters.ctxFree += 1;
			callOrder.push("ctxFree");
		},
		tensorNew1d: () => next++,
		tensorNew2d: () => next++,
		tensorNew3d: () => next++,
		tensorNew4d: () => next++,
		tensorSetName: () => {},
		tensorNbytes: () => 0,
		uploadToTensorChunked: () => {},
		backendAllocCtxTensors: () => {
			counters.backendAllocCtxTensors += 1;
			callOrder.push("backendAllocCtxTensors");
			return next++;
		},
		backendBufferFree: (_buf: number) => {
			counters.backendBufferFree += 1;
			callOrder.push(`backendBufferFree:${_buf}`);
		},
		graphNew: () => {
			counters.graphNew += 1;
			callOrder.push("graphNew");
			return next++;
		},
		graphBuildForwardExpand: () => {
			counters.graphBuildForwardExpand += 1;
		},
		opGetRows: () => next++,
		opAdd: () => next++,
		opMul: () => next++,
		opNorm: () => next++,
		opMulMat: () => next++,
		opSoftMaxExt: () => next++,
		opGelu: () => next++,
		opReshape2d: () => next++,
		opReshape3d: () => next++,
		opPermute: () => next++,
		opCont: () => next++,
		malloc: (_size: number) => 0,
		free: (_ptr: number) => {},
		get heapU8() {
			return heap;
		},
		backendTensorSet3: () => {},
		graphCompute: async (_g: number) => {
			return 0;
		},
		downloadFromTensor: async (_t: number, byteLen: number) => {
			return new Uint8Array(byteLen);
		},
	} as unknown as GgmlWasm;
	return { fake: stub, counters, callOrder };
}

function stubEncoderWeights(enc: EncoderInference, layerCount: number): void {
	const layerWeights = Array.from({ length: layerCount }, () => ({
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
}

describe("EncoderInference graph-cache", () => {
	test("reuses ctx + buffer across same-N embeds", async () => {
		const { fake, counters } = makeFakeWasmForEmbed();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		stubEncoderWeights(enc, 2);

		for (let i = 0; i < 5; i++) {
			await enc.embed(new Int32Array([1, 2, 3]));
		}

		expect(counters.ctxCreate).toBe(1);
		expect(counters.backendAllocCtxTensors).toBe(1);
		expect(counters.backendBufferFree).toBe(0);
		expect(counters.graphNew).toBe(1);
		expect(counters.graphBuildForwardExpand).toBe(1);
		expect(counters.ctxFree).toBe(0);
	});

	test("rebuilds when N changes", async () => {
		const { fake, counters } = makeFakeWasmForEmbed();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		stubEncoderWeights(enc, 2);

		for (let i = 0; i < 3; i++) {
			await enc.embed(new Int32Array([1, 2, 3]));
		}
		for (let i = 0; i < 2; i++) {
			await enc.embed(new Int32Array([1, 2, 3, 4, 5]));
		}

		expect(counters.ctxCreate).toBe(2);
		expect(counters.backendAllocCtxTensors).toBe(2);
		expect(counters.backendBufferFree).toBe(1);
		expect(counters.graphNew).toBe(2);
		expect(counters.graphBuildForwardExpand).toBe(2);
		// Old graph ctx popped exactly once when N changed.
		expect(counters.ctxFree).toBe(1);
	});

	test("dispose pops graph then weight in order", async () => {
		const { fake, counters, callOrder } = makeFakeWasmForEmbed();
		const hp = makeBertHp(2);
		const enc = new EncoderInference(fake, hp);
		stubEncoderWeights(enc, 2);
		// Simulate loadWeights having allocated the weight buffer too.
		(enc as unknown as { weightBuf: number }).weightBuf = 1234;

		await enc.embed(new Int32Array([1, 2, 3]));
		await enc.dispose();

		expect(counters.ctxFree).toBe(2);
		expect(counters.backendBufferFree).toBe(2);

		// Verify graph buf freed BEFORE weight buf in dispose.
		const freeEvents = callOrder.filter((e) =>
			e.startsWith("backendBufferFree"),
		);
		expect(freeEvents.length).toBe(2);
		// Graph buf was allocated first (returned by backendAllocCtxTensors during
		// embed); weight buf is the explicit 1234 we set above.
		expect(freeEvents[1]).toBe("backendBufferFree:1234");
		expect(freeEvents[0]).not.toBe("backendBufferFree:1234");
	});
});
