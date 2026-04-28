import { describe, expect, test } from "bun:test";
import {
	ENCODER_ARCHITECTURES,
	isEncoderArchitecture,
	type ModelHyperparams,
} from "../src/core/types.js";
import { EncoderInference } from "../src/inference/encoder-inference.js";
import type { GgmlWasm, TensorPtr } from "../src/inference/ggml-wasm.js";
import type { GgufContext, GgufTensorInfo } from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

describe("isEncoderArchitecture", () => {
	test("returns true for encoder archs", () => {
		expect(isEncoderArchitecture("bert")).toBe(true);
		expect(isEncoderArchitecture("nomic-bert")).toBe(true);
		expect(isEncoderArchitecture("jina-bert-v2")).toBe(true);
	});
	test("returns false for causal archs", () => {
		for (const a of [
			"llama",
			"mistral",
			"qwen",
			"qwen2",
			"qwen3",
			"phi",
			"gemma",
			"mixtral",
			"deepseek",
		] as const) {
			expect(isEncoderArchitecture(a)).toBe(false);
		}
	});
	test("ENCODER_ARCHITECTURES tuple matches helper truth-table", () => {
		expect(ENCODER_ARCHITECTURES).toEqual([
			"bert",
			"nomic-bert",
			"jina-bert-v2",
		]);
	});
});

function fakeCtx(
	arch: string,
	extras: Record<string, unknown> = {},
): GgufContext {
	const meta = new Map<string, { value: unknown }>();
	meta.set("general.architecture", { value: arch });
	meta.set(`${arch}.embedding_length`, { value: 768 });
	meta.set(`${arch}.block_count`, { value: 12 });
	meta.set(`${arch}.attention.head_count`, { value: 12 });
	meta.set(`${arch}.feed_forward_length`, { value: 3072 });
	meta.set(`${arch}.attention.layer_norm_epsilon`, { value: 1e-12 });
	meta.set(`${arch}.context_length`, { value: 8192 });
	meta.set(`${arch}.attention.causal`, { value: false });
	meta.set(`${arch}.pooling_type`, { value: 1 });
	for (const [k, v] of Object.entries(extras)) meta.set(k, { value: v });
	return {
		metadata: meta,
		tensors: [],
		dataOffset: 0,
		totalDataSize: 0,
	} as unknown as GgufContext;
}

describe("ModelLoader.extractHyperparams non-BERT encoder branches", () => {
	test("nomic-bert produces RoPE-ready hyperparams", () => {
		const ctx = fakeCtx("nomic-bert", {
			"nomic-bert.rope.freq_base": 1000.0,
		});
		const hp = (
			ModelLoader as unknown as {
				extractHyperparams: (c: unknown) => unknown;
			}
		).extractHyperparams(ctx) as Record<string, unknown>;
		expect(hp.architecture).toBe("nomic-bert");
		expect(hp.causalAttention).toBe(false);
		expect(hp.poolingType).toBe("mean");
		expect(hp.ropeFreqBase).toBe(1000.0);
		expect(hp.normEpsilon).toBeCloseTo(1e-12);
		expect(hp.alibiMaxBias).toBeUndefined();
	});
	test("jina-bert-v2 falls back to alibiMaxBias=8.0 when metadata absent", () => {
		const ctx = fakeCtx("jina-bert-v2");
		const hp = (
			ModelLoader as unknown as {
				extractHyperparams: (c: unknown) => unknown;
			}
		).extractHyperparams(ctx) as Record<string, unknown>;
		expect(hp.architecture).toBe("jina-bert-v2");
		expect(hp.alibiMaxBias).toBe(8.0);
	});
	test("jina-bert-v2 honors alibi_bias_max metadata when present", () => {
		const ctx = fakeCtx("jina-bert-v2", {
			"jina-bert-v2.attention.alibi_bias_max": 16.0,
		});
		const hp = (
			ModelLoader as unknown as {
				extractHyperparams: (c: unknown) => unknown;
			}
		).extractHyperparams(ctx) as Record<string, unknown>;
		expect(hp.alibiMaxBias).toBe(16.0);
	});
});

describe("EncoderInference construction", () => {
	function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
		return {
			architecture: arch,
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
	}
	test("rejects causal LM hyperparams", () => {
		expect(() => new EncoderInference({} as never, makeHp("llama"))).toThrow(
			/not yet support architecture "llama"/,
		);
	});
	test("accepts bert", () => {
		expect(
			() => new EncoderInference({} as never, makeHp("bert")),
		).not.toThrow();
	});
	test("accepts nomic-bert", () => {
		expect(
			() => new EncoderInference({} as never, makeHp("nomic-bert")),
		).not.toThrow();
	});
	test("accepts jina-bert-v2", () => {
		expect(
			() => new EncoderInference({} as never, makeHp("jina-bert-v2")),
		).not.toThrow();
	});
});

interface FakeWasm {
	fake: GgmlWasm;
	ops: string[];
	softmaxMaxBias: number[];
	view3dCalls: Array<{ nb1: number; nb2: number; offset: number }>;
}

function makeFakeWasm(): FakeWasm {
	let next = 1;
	const ops: string[] = [];
	const softmaxMaxBias: number[] = [];
	const view3dCalls: Array<{ nb1: number; nb2: number; offset: number }> = [];
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
		opSoftMaxExt: (
			_q: TensorPtr,
			mask: TensorPtr,
			_scale: number,
			maxBias: number,
		) => {
			ops.push("softmaxext");
			ops.push(mask === 0 ? "softmax-nullmask" : "softmax-mask");
			softmaxMaxBias.push(maxBias);
			return next++;
		},
		opGelu: () => {
			ops.push("gelu");
			return next++;
		},
		opSilu: () => {
			ops.push("silu");
			return next++;
		},
		opRope: () => {
			ops.push("rope");
			return next++;
		},
		opView3d: (
			_x: TensorPtr,
			_ne0: number,
			_ne1: number,
			_ne2: number,
			nb1: number,
			nb2: number,
			offset: number,
		) => {
			ops.push("view3d");
			view3dCalls.push({ nb1, nb2, offset });
			return next++;
		},
		opReshape2d: () => next++,
		opReshape3d: () => next++,
		opPermute: () => next++,
		opCont: () => next++,
	} as unknown as GgmlWasm;
	return { fake: stub, ops, softmaxMaxBias, view3dCalls };
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

describe("EncoderInference.makeTensorOptional", () => {
	test("returns null when tensor name absent", () => {
		const fake = makeFakeWasm();
		const enc = new EncoderInference(fake.fake, {
			architecture: "bert",
			contextLength: 512,
			embeddingLength: 384,
			headCount: 12,
			headCountKv: 12,
			layerCount: 1,
			vocabularySize: 30522,
			embeddingHeadLength: 32,
			feedForwardLength: 1536,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
		});
		const empty = new Map();
		const result = (
			enc as unknown as {
				makeTensorOptional: (m: unknown, n: string) => unknown;
			}
		).makeTensorOptional(empty, "missing.weight");
		expect(result).toBeNull();
	});
});

interface EncoderLayerWeightsForTest {
	qkvFused: TensorPtr | null;
	qProj: TensorPtr | null;
	qBias: TensorPtr | null;
	kProj: TensorPtr | null;
	kBias: TensorPtr | null;
	vProj: TensorPtr | null;
	vBias: TensorPtr | null;
	oProj: TensorPtr;
	oBias: TensorPtr | null;
	attnNormW: TensorPtr;
	attnNormB: TensorPtr;
	ffnGate: TensorPtr | null;
	ffnUp: TensorPtr;
	ffnUpBias: TensorPtr | null;
	ffnDown: TensorPtr;
	ffnDownBias: TensorPtr | null;
	ffnNormW: TensorPtr;
	ffnNormB: TensorPtr;
}

describe("EncoderInference.loadWeights arch dispatch", () => {
	function dim(name: string): GgufTensorInfo {
		return {
			name,
			dimensions: [768, 768],
			type: 0,
			offset: 0,
		} as GgufTensorInfo;
	}
	function makeCtx(names: string[]): GgufContext {
		return {
			metadata: new Map(),
			tensors: names.map(dim),
			dataOffset: 0,
			totalDataSize: 0,
		} as unknown as GgufContext;
	}
	function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
		return {
			architecture: arch,
			contextLength: 512,
			embeddingLength: 768,
			headCount: 12,
			headCountKv: 12,
			layerCount: 1,
			vocabularySize: 30522,
			embeddingHeadLength: 64,
			feedForwardLength: 3072,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
			poolingType: "mean",
			causalAttention: false,
			...(arch === "jina-bert-v2" ? { alibiMaxBias: 8.0 } : {}),
		};
	}

	const bertNames = [
		"token_embd.weight",
		"token_embd_norm.weight",
		"token_embd_norm.bias",
		"token_types.weight",
		"position_embd.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_q.bias",
		"blk.0.attn_k.weight",
		"blk.0.attn_k.bias",
		"blk.0.attn_v.weight",
		"blk.0.attn_v.bias",
		"blk.0.attn_output.weight",
		"blk.0.attn_output.bias",
		"blk.0.attn_output_norm.weight",
		"blk.0.attn_output_norm.bias",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_up.bias",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_down.bias",
		"blk.0.layer_output_norm.weight",
		"blk.0.layer_output_norm.bias",
	];
	const jinaNames = [
		"token_embd.weight",
		"token_embd_norm.weight",
		"token_embd_norm.bias",
		"token_types.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_q.bias",
		"blk.0.attn_k.weight",
		"blk.0.attn_k.bias",
		"blk.0.attn_v.weight",
		"blk.0.attn_v.bias",
		"blk.0.attn_output.weight",
		"blk.0.attn_output.bias",
		"blk.0.attn_output_norm.weight",
		"blk.0.attn_output_norm.bias",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_down.bias",
		"blk.0.layer_output_norm.weight",
		"blk.0.layer_output_norm.bias",
	];

	test("bert: full-bias path; ffnGate null", () => {
		const fake = makeFakeWasm();
		const enc = new EncoderInference(fake.fake, makeHp("bert"));
		enc.loadWeights(makeCtx(bertNames), new Uint8Array(0));
		const layers = (
			enc as unknown as {
				weights: { layers: EncoderLayerWeightsForTest[] };
			}
		).weights.layers;
		expect(layers[0].qProj).not.toBeNull();
		expect(layers[0].qBias).not.toBeNull();
		expect(layers[0].oBias).not.toBeNull();
		expect(layers[0].ffnUpBias).not.toBeNull();
		expect(layers[0].ffnDownBias).not.toBeNull();
		expect(layers[0].ffnGate).toBeNull();
		expect(layers[0].qkvFused).toBeNull();
	});

	test("jina-bert-v2: split QKV + biases; SwiGLU gate; mixed FFN biases", () => {
		const fake = makeFakeWasm();
		const enc = new EncoderInference(fake.fake, makeHp("jina-bert-v2"));
		enc.loadWeights(makeCtx(jinaNames), new Uint8Array(0));
		const layers = (
			enc as unknown as {
				weights: { layers: EncoderLayerWeightsForTest[] };
			}
		).weights.layers;
		expect(layers[0].qProj).not.toBeNull();
		expect(layers[0].qBias).not.toBeNull();
		expect(layers[0].oBias).not.toBeNull();
		expect(layers[0].ffnGate).not.toBeNull();
		expect(layers[0].ffnUpBias).toBeNull();
		expect(layers[0].ffnDownBias).not.toBeNull();
		expect(layers[0].qkvFused).toBeNull();
	});

	test("nomic-bert: fused QKV; no biases; SwiGLU gate", () => {
		const fake = makeFakeWasm();
		const enc = new EncoderInference(fake.fake, makeHp("nomic-bert"));
		enc.loadWeights(
			makeCtx([
				"token_embd.weight",
				"token_embd_norm.weight",
				"token_embd_norm.bias",
				"token_types.weight",
				"blk.0.attn_qkv.weight",
				"blk.0.attn_output.weight",
				"blk.0.attn_output_norm.weight",
				"blk.0.attn_output_norm.bias",
				"blk.0.ffn_gate.weight",
				"blk.0.ffn_up.weight",
				"blk.0.ffn_down.weight",
				"blk.0.layer_output_norm.weight",
				"blk.0.layer_output_norm.bias",
			]),
			new Uint8Array(0),
		);
		const layers = (
			enc as unknown as {
				weights: { layers: EncoderLayerWeightsForTest[] };
			}
		).weights.layers;
		expect(layers[0].qkvFused).not.toBeNull();
		expect(layers[0].qProj).toBeNull();
		expect(layers[0].qBias).toBeNull();
		expect(layers[0].kProj).toBeNull();
		expect(layers[0].kBias).toBeNull();
		expect(layers[0].vProj).toBeNull();
		expect(layers[0].vBias).toBeNull();
		expect(layers[0].oBias).toBeNull();
		expect(layers[0].ffnGate).not.toBeNull();
		expect(layers[0].ffnUpBias).toBeNull();
		expect(layers[0].ffnDownBias).toBeNull();
	});
});

describe("EncoderInference.buildGraph arch dispatch", () => {
	function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
		return {
			architecture: arch,
			contextLength: 512,
			embeddingLength: 384,
			headCount: 12,
			headCountKv: 12,
			layerCount: 2,
			vocabularySize: 30522,
			embeddingHeadLength: 32,
			feedForwardLength: 1536,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
			poolingType: "mean",
			causalAttention: false,
			...(arch === "jina-bert-v2" ? { alibiMaxBias: 8.0 } : {}),
		};
	}
	function buildAndCount(arch: ModelHyperparams["architecture"]): {
		getrows: number;
		rope: number;
		silu: number;
		gelu: number;
		view3d: number;
		softmax_max_bias: number[];
		view3dCalls: Array<{ nb1: number; nb2: number; offset: number }>;
	} {
		const fake = makeFakeWasm();
		const hp = makeHp(arch);
		const enc = new EncoderInference(fake.fake, hp);
		(enc as unknown as { weights: unknown }).weights = {
			tokEmb: 1,
			positionEmb: arch === "bert" ? 2 : null,
			tokenTypes: 3,
			inputNormW: 4,
			inputNormB: 5,
			layers: Array.from({ length: hp.layerCount }, () => ({
				qkvFused: arch === "nomic-bert" ? 100 : null,
				qProj: arch === "nomic-bert" ? null : 10,
				qBias: arch === "bert" || arch === "jina-bert-v2" ? 11 : null,
				kProj: arch === "nomic-bert" ? null : 12,
				kBias: arch === "bert" || arch === "jina-bert-v2" ? 13 : null,
				vProj: arch === "nomic-bert" ? null : 14,
				vBias: arch === "bert" || arch === "jina-bert-v2" ? 15 : null,
				oProj: 16,
				oBias: arch === "bert" || arch === "jina-bert-v2" ? 17 : null,
				attnNormW: 18,
				attnNormB: 19,
				ffnGate: arch === "nomic-bert" || arch === "jina-bert-v2" ? 20 : null,
				ffnUp: 21,
				ffnUpBias: arch === "bert" ? 22 : null,
				ffnDown: 23,
				ffnDownBias: arch === "bert" || arch === "jina-bert-v2" ? 24 : null,
				ffnNormW: 25,
				ffnNormB: 26,
			})),
		};
		(enc as unknown as { buildGraph: (n: number) => unknown }).buildGraph(4);
		return {
			getrows: fake.ops.filter((o) => o === "getrows").length,
			rope: fake.ops.filter((o) => o === "rope").length,
			silu: fake.ops.filter((o) => o === "silu").length,
			gelu: fake.ops.filter((o) => o === "gelu").length,
			view3d: fake.ops.filter((o) => o === "view3d").length,
			softmax_max_bias: fake.softmaxMaxBias,
			view3dCalls: fake.view3dCalls,
		};
	}

	test("bert: pos-embedding + GeLU FFN, no rope, no silu, max_bias=0", () => {
		const r = buildAndCount("bert");
		expect(r.getrows).toBe(3);
		expect(r.rope).toBe(0);
		expect(r.silu).toBe(0);
		expect(r.gelu).toBe(2);
		expect(r.softmax_max_bias).toEqual([0, 0]);
	});

	test("jina-bert-v2: no pos-embedding, SwiGLU FFN, no rope, max_bias=8.0", () => {
		const r = buildAndCount("jina-bert-v2");
		expect(r.getrows).toBe(2);
		expect(r.rope).toBe(0);
		expect(r.silu).toBe(2);
		expect(r.gelu).toBe(0);
		expect(r.softmax_max_bias).toEqual([8.0, 8.0]);
	});

	test("nomic-bert: fused QKV (3 view3d/layer) + RoPE (Q+K/layer) + SwiGLU, max_bias=0", () => {
		const r = buildAndCount("nomic-bert");
		expect(r.getrows).toBe(2);
		expect(r.view3d).toBe(6); // 2 layers × 3 slices (Q, K, V)
		expect(r.rope).toBe(4); // 2 layers × (Q + K)
		expect(r.silu).toBe(2);
		expect(r.gelu).toBe(0);
		expect(r.softmax_max_bias).toEqual([0, 0]);

		// Verify fused-QKV view3d byte arithmetic per layer. Spec-flagged
		// High-risk failure mode "Fused-QKV view_3d offsets miscomputed":
		// without these assertions, an aliased (0, 0, 0) call sequence would
		// pass the count check above.
		expect(r.view3dCalls.length).toBe(6); // 2 layers × 3 slices
		const hp = makeHp("nomic-bert");
		const elemSize = 4; // F32_BYTES — production uses F32_BYTES
		const E = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const expectedOffsets = [0, elemSize * E, 2 * elemSize * E];
		const expectedNb1 = elemSize * headDim;
		const expectedNb2 = elemSize * 3 * E;
		for (let layer = 0; layer < 2; layer++) {
			for (let slice = 0; slice < 3; slice++) {
				const call = r.view3dCalls[layer * 3 + slice];
				expect(call.offset).toBe(expectedOffsets[slice]);
				expect(call.nb1).toBe(expectedNb1);
				expect(call.nb2).toBe(expectedNb2);
			}
		}
	});
});
