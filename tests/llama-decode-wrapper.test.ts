import { describe, expect, it } from "bun:test";
import type { LlamaBridge } from "../src/inference/llama-bridge.js";
import { LlamaDecodeWrapper } from "../src/inference/llama-decode-wrapper.js";

interface DecodeCall {
	ctx: number;
	tokens: number[];
	pastLen: number;
}

function makeFakeBridge(): {
	bridge: LlamaBridge;
	calls: { decode: DecodeCall[]; kvClear: number[]; kvSeqRm: unknown[] };
	nextCtx: { value: number };
} {
	const decode: DecodeCall[] = [];
	const kvClear: number[] = [];
	const kvSeqRm: unknown[] = [];
	const nextCtx = { value: 1000 };
	const fakeLogits = new Float32Array(8);
	const fakeEmbeddings = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
	const bridge: LlamaBridge = {
		loadModel: async () => 999,
		freeModel: () => {},
		createContext: async () => {
			nextCtx.value += 1;
			return nextCtx.value;
		},
		freeContext: () => {},
		decode: async (ctx, tokens, pastLen) => {
			decode.push({ ctx, tokens: Array.from(tokens), pastLen });
			return 0;
		},
		getLogits: async () => fakeLogits,
		nVocab: () => 8,
		tokenize: () => new Int32Array(0),
		detokenize: () => "",
		tokenBos: () => 1,
		tokenEos: () => 2,
		getMetadata: () => null,
		nCtxTrain: () => 4096,
		nEmbd: () => 8,
		nLayer: () => 4,
		nHead: () => 4,
		nHeadKv: () => 4,
		nCtx: () => 4096,
		kvSeqRm: (ctx, seqId, p0, p1) => {
			kvSeqRm.push({ ctx, seqId, p0, p1 });
		},
		kvClear: (ctx) => {
			kvClear.push(ctx);
		},
		stateSeqGetSize: () => 1024,
		stateSeqGetData: () => new Uint8Array(1024),
		stateSeqSetData: () => true,
		getEmbeddings: async () => fakeEmbeddings,
	};
	return { bridge, calls: { decode, kvClear, kvSeqRm }, nextCtx };
}

describe("LlamaDecodeWrapper", () => {
	it("forwards sequential positions and tracks cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		w.loadWeights();
		await w.initKVCache(2048);

		expect(w.cachedTokenCount).toBe(0);
		expect(w.maxContextLength).toBe(4096);

		await w.forward(new Int32Array([10, 20, 30]), new Int32Array([0, 1, 2]));
		expect(w.cachedTokenCount).toBe(3);
		expect(calls.decode).toHaveLength(1);
		expect(calls.decode[0].pastLen).toBe(0);
		expect(calls.decode[0].tokens).toEqual([10, 20, 30]);

		await w.forward(new Int32Array([40]), new Int32Array([3]));
		expect(w.cachedTokenCount).toBe(4);
		expect(calls.decode[1].pastLen).toBe(3);
	});

	it("rejects non-sequential positions", async () => {
		const { bridge } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		expect(
			w.forward(new Int32Array([10, 20]), new Int32Array([0, 5])),
		).rejects.toThrow(/sequential from cachedTokenCount=0/);
	});

	it("resetKVCache calls kvClear and zeros cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.forward(new Int32Array([10, 20]), new Int32Array([0, 1]));
		expect(w.cachedTokenCount).toBe(2);
		w.resetKVCache();
		expect(w.cachedTokenCount).toBe(0);
		expect(calls.kvClear).toHaveLength(1);
	});

	it("truncateKVCache calls kvSeqRm and updates cachedTokenCount", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.forward(
			new Int32Array([10, 20, 30, 40]),
			new Int32Array([0, 1, 2, 3]),
		);
		w.truncateKVCache(2);
		expect(w.cachedTokenCount).toBe(2);
		expect(calls.kvSeqRm).toHaveLength(1);
		expect(calls.kvSeqRm[0]).toMatchObject({ seqId: 0, p0: 2, p1: -1 });
	});

	it("loadKVCache truncates when snapshotLen > nTokens", async () => {
		const { bridge, calls } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		await w.loadKVCache(new Uint8Array(1024), 5, 10);
		expect(w.cachedTokenCount).toBe(5);
		// One kvSeqRm call to drop tokens [5, 10).
		expect(calls.kvSeqRm).toHaveLength(1);
		expect(calls.kvSeqRm[0]).toMatchObject({ seqId: 0, p0: 5, p1: -1 });
	});

	it("embed creates a side context once and clears KV between calls", async () => {
		const { bridge, calls, nextCtx } = makeFakeBridge();
		const w = new LlamaDecodeWrapper(bridge, 999);
		await w.initKVCache(2048);
		const e1 = await w.embed(new Int32Array([10, 20, 30]));
		const e2 = await w.embed(new Int32Array([40, 50]));
		// 1 main ctx + 1 embed ctx.
		expect(nextCtx.value).toBe(1002);
		// Two clears: each embed call resets the side context.
		expect(calls.kvClear).toHaveLength(2);
		// Both calls returned the fake embeddings.
		expect(Array.from(e1)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
		expect(Array.from(e2)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
	});
});
