import { describe, expect, test } from "bun:test";
import { WebLLM } from "../src/core/engine.js";

describe("WebLLM.embed API surface", () => {
	test("rejects unknown modelId with descriptive error", async () => {
		const engine = await WebLLM.init({
			device: {} as GPUDevice,
			memoryBudget: 1 << 28,
		});
		await expect(engine.embed("nonexistent", "hi")).rejects.toThrow(
			/not found/,
		);
	});
});
