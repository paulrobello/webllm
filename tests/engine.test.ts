import { describe, expect, test } from "bun:test";
import { pickWasmUrl } from "../src/core/engine.js";

describe("pickWasmUrl", () => {
	const GIB = 1024 * 1024 * 1024;
	const MARGIN = 3.5 * GIB;

	test("routes 2 GiB models through the wasm32 binary", () => {
		expect(pickWasmUrl(2_000_000_000)).toBe("./webllm-wasm.js");
	});

	test("routes 5 GiB models through the wasm64 binary", () => {
		expect(pickWasmUrl(5_000_000_000)).toBe("./webllm-wasm-mem64.js");
	});

	test("override wins regardless of size", () => {
		expect(pickWasmUrl(7_500_000_000, "custom.js")).toBe("custom.js");
	});

	test("3.5 GiB boundary is inclusive of wasm32", () => {
		expect(pickWasmUrl(MARGIN)).toBe("./webllm-wasm.js");
		expect(pickWasmUrl(MARGIN + 1)).toBe("./webllm-wasm-mem64.js");
	});
});
