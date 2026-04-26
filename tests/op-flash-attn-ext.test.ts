import { describe, expect, it } from "bun:test";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";

describe("opFlashAttn bridge surface", () => {
	it("exposes opFlashAttn, opFlashAttnSetPrec, opFlashAttnAddSinks on the class prototype", () => {
		const proto = GgmlWasm.prototype as Record<string, unknown>;
		expect(typeof proto.opFlashAttn).toBe("function");
		expect(typeof proto.opFlashAttnSetPrec).toBe("function");
		expect(typeof proto.opFlashAttnAddSinks).toBe("function");
	});
});
