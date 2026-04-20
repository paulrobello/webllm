import { describe, expect, test } from "bun:test";
import {
	LightweightModel,
	type LightweightModelConfig,
} from "../src/inference/lightweight.js";

// Minimal mock GPUDevice — enough for construction and init
function createMockDevice(): Record<string, unknown> {
	return {
		createBuffer: (_desc: GPUBufferDescriptor) => ({
			size: _desc.size,
			usage: _desc.usage,
			mapAsync: async () => {},
			getMappedRange: () => new ArrayBuffer(_desc.size),
			unmap: () => {},
			destroy: () => {},
		}),
		createShaderModule: () => ({}),
		createComputePipeline: () => ({
			getBindGroupLayout: () => ({}),
		}),
		queue: {
			writeBuffer: () => {},
			submit: () => {},
		},
		createCommandEncoder: () => ({
			beginComputePass: () => ({
				setPipeline: () => {},
				setBindGroup: () => {},
				dispatchWorkgroups: () => {},
				end: () => {},
			}),
			copyBufferToBuffer: () => {},
			finish: () => ({}),
		}),
	};
}

const MOCK_CONFIG: LightweightModelConfig = {
	device: createMockDevice() as unknown as GPUDevice,
	embeddingDim: 64,
	vocabularySize: 1000,
	hiddenDim: 64,
	layerCount: 2,
	maxSequenceLength: 128,
};

describe("LightweightModel", () => {
	test("construction sets config", () => {
		const model = new LightweightModel(MOCK_CONFIG);
		expect(model.embeddingDim).toBe(64);
		expect(model.vocabularySize).toBe(1000);
		expect(model.hiddenDim).toBe(64);
		expect(model.layerCount).toBe(2);
	});

	test("isLoaded is false before init", () => {
		const model = new LightweightModel(MOCK_CONFIG);
		expect(model.isLoaded).toBe(false);
	});

	test("init creates pipelines and sets loaded", async () => {
		const model = new LightweightModel(MOCK_CONFIG);
		await model.init();
		expect(model.isLoaded).toBe(true);
	});

	test("destroy resets state", async () => {
		const model = new LightweightModel(MOCK_CONFIG);
		await model.init();
		model.destroy();
		expect(model.isLoaded).toBe(false);
	});

	test("init can be called after destroy", async () => {
		const model = new LightweightModel(MOCK_CONFIG);
		await model.init();
		model.destroy();
		await model.init();
		expect(model.isLoaded).toBe(true);
		model.destroy();
	});
});
