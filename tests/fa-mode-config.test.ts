import { describe, expect, it } from "bun:test";
import type { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";
import type { ModelHyperparams } from "../src/inference/types.js";

// We never call any wasm method here — we only inspect the constructor
// option behavior, which doesn't touch the WASM module. A null-typed cast
// is sufficient because the field is read-only and writes happen only in
// methods that aren't exercised in this test.
const stubWasm = {} as GgmlWasm;
const stubHp = {
	layerCount: 1,
	embeddingHeadLength: 8,
	headCountKv: 1,
} as ModelHyperparams;

describe("ModelInference flashAttn option", () => {
	it("defaults to false when no option is passed", () => {
		const inf = new ModelInference(stubWasm, stubHp);
		expect(inf.flashAttn).toBe(false);
	});

	it("defaults to false when an empty options object is passed", () => {
		const inf = new ModelInference(stubWasm, stubHp, {});
		expect(inf.flashAttn).toBe(false);
	});

	it("respects flashAttn: true", () => {
		const inf = new ModelInference(stubWasm, stubHp, { flashAttn: true });
		expect(inf.flashAttn).toBe(true);
	});

	it("respects flashAttn: false explicitly", () => {
		const inf = new ModelInference(stubWasm, stubHp, { flashAttn: false });
		expect(inf.flashAttn).toBe(false);
	});

	it("is read-only after construction", () => {
		const inf = new ModelInference(stubWasm, stubHp, { flashAttn: true });
		// `readonly` is compile-time only; verify the value just doesn't shift
		// through any method we're likely to call. resetKVCache is the
		// closest the public surface gets to "reset state" — confirm it
		// doesn't reset the mode.
		inf.resetKVCache();
		expect(inf.flashAttn).toBe(true);
	});
});
