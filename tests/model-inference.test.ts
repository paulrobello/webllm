import { describe, expect, test } from "bun:test";

import { RopeMode } from "../src/inference/ggml-wasm.js";
import { getRopeModeForArchitecture } from "../src/inference/rope/rope-mode.js";

describe("getRopeModeForArchitecture", () => {
	test("uses NEOX rope for qwen-family architectures", () => {
		expect(getRopeModeForArchitecture("qwen")).toBe(RopeMode.NEOX);
		expect(getRopeModeForArchitecture("qwen2")).toBe(RopeMode.NEOX);
		expect(getRopeModeForArchitecture("qwen3")).toBe(RopeMode.NEOX);
	});

	test("keeps normal rope for llama-family architectures", () => {
		expect(getRopeModeForArchitecture("llama")).toBe(RopeMode.NORMAL);
	});
});
