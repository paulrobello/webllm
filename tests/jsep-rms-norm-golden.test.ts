/**
 * JSEP backend — rms_norm golden tests.
 *
 * Hand-builds JSEP runtime + descriptor blobs in JS, runs the rms_norm
 * dispatch end-to-end against a real `GPUDevice`, and compares results
 * to a CPU reference rms_norm.
 *
 * Bun has no WebGPU; without it the suite skips. Browser smoke covers
 * the WebGPU path against the patched WASM build.
 *
 * RMS norm semantics (unary; ggml RMS_NORM is `out[j] = x[j]/rms` only.
 * The per-channel weight multiply is a separate GGML_OP_MUL node):
 *   src0: [last_dim, n_rows, ...]   input.
 *   dst:  same shape as src0.
 *   eps:  op_params[0] (f32).
 *
 *   per-row: out[j] = x[j] / sqrt(mean(x²) + eps)
 */

import { describe, expect, test } from "bun:test";
import { CommandEncoderBatcher } from "../src/inference/jsep/command-encoder.js";
import { GpuDataManager } from "../src/inference/jsep/gpu-data-manager.js";
import { GGML_OP_RMS_NORM } from "../src/inference/jsep/index.js";
import {
	GGML_TYPE_F32,
	type JsepOpDescriptor,
	type JsepTensorMeta,
} from "../src/inference/jsep/ops/matmul.js";
import { dispatchRmsNorm } from "../src/inference/jsep/ops/rms-norm.js";
import { JsepPipelineCache } from "../src/inference/jsep/pipeline-cache.js";

const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as Navigator & { gpu?: unknown }).gpu !== "undefined";

async function getDevice(): Promise<GPUDevice | null> {
	if (!HAS_WEBGPU) return null;
	try {
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return null;
		return await adapter.requestDevice();
	} catch {
		return null;
	}
}

interface Ctx {
	device: GPUDevice;
	dataManager: GpuDataManager;
	encoderBatcher: CommandEncoderBatcher;
	pipelineCache: JsepPipelineCache;
	bindGroupLayoutCache: Map<string, GPUBindGroupLayout>;
}

function makeCtx(device: GPUDevice): Ctx {
	return {
		device,
		dataManager: new GpuDataManager(device),
		encoderBatcher: new CommandEncoderBatcher(device),
		pipelineCache: new JsepPipelineCache(device),
		bindGroupLayoutCache: new Map<string, GPUBindGroupLayout>(),
	};
}

// CPU reference rms_norm: per-row independently. Unary — no weight.
//   x:  [rows, cols] flat row-major (row r, col c → x[r*cols + c]).
function cpuRmsNorm(
	x: Float32Array,
	rows: number,
	cols: number,
	eps: number,
): Float32Array {
	const out = new Float32Array(rows * cols);
	for (let r = 0; r < rows; r++) {
		let sumSq = 0;
		for (let c = 0; c < cols; c++) {
			const v = x[r * cols + c];
			sumSq += v * v;
		}
		const inv = 1 / Math.sqrt(sumSq / cols + eps);
		for (let c = 0; c < cols; c++) {
			out[r * cols + c] = x[r * cols + c] * inv;
		}
	}
	return out;
}

function maxAbsDelta(a: Float32Array, b: Float32Array): number {
	let max = 0;
	for (let i = 0; i < a.length; i++) {
		const d = Math.abs(a[i] - b[i]);
		if (d > max) max = d;
	}
	return max;
}

// Build a meta for a contiguous F32 tensor.
function makeF32Meta(
	handle: number,
	ne: [number, number, number, number],
): JsepTensorMeta {
	const nb: [number, number, number, number] = [0, 0, 0, 0];
	nb[0] = 4;
	nb[1] = nb[0] * ne[0];
	nb[2] = nb[1] * ne[1];
	nb[3] = nb[2] * ne[2];
	// Stage 2 ABI: each test tensor lives at offset 0 in its own dataManager
	// allocation, so bufHandle=handle and offset=0.
	return { bufHandle: handle, offset: 0, type: GGML_TYPE_F32, ne, nb };
}

// Run a single rms_norm test for given (rows, cols, eps).
async function runRmsNormGolden(
	rows: number,
	cols: number,
	eps: number,
	tolerance: number,
): Promise<void> {
	const device = await getDevice();
	if (!device) {
		console.warn("WebGPU adapter unavailable; skipping");
		return;
	}
	const ctx = makeCtx(device);

	// Deterministic input; spread across a sensible range so the rms is
	// not tiny (which would inflate the post-scale error).
	const x = new Float32Array(rows * cols);
	for (let i = 0; i < x.length; i++) x[i] = ((i * 7) % 31) * 0.05 - 0.7;

	const reference = cpuRmsNorm(x, rows, cols, eps);

	const xBytes = new Uint8Array(x.buffer.slice(0));
	const dstBytes = rows * cols * 4;

	const hX = ctx.dataManager.alloc(xBytes.byteLength);
	const hD = ctx.dataManager.alloc(dstBytes);

	device.queue.writeBuffer(
		ctx.dataManager.get(hX).buffer,
		0,
		xBytes,
		0,
		xBytes.byteLength,
	);

	// op_params block: 16 f32 in ggml. Lay it down in a host-side buffer
	// at a known byte offset; pass that offset as the `opParamsPtr` and
	// the buffer as the heap.
	const opParamsHost = new ArrayBuffer(64);
	new Float32Array(opParamsHost)[0] = eps;

	const desc: JsepOpDescriptor = {
		op: GGML_OP_RMS_NORM,
		nSrc: 1,
		dst: makeF32Meta(hD, [cols, rows, 1, 1]),
		srcs: [makeF32Meta(hX, [cols, rows, 1, 1])],
	};

	const status = dispatchRmsNorm(ctx, desc, 0, opParamsHost);
	expect(status).toBe(0);

	ctx.encoderBatcher.flush();

	const heap = new ArrayBuffer(dstBytes);
	await ctx.dataManager.readAsync(hD, 0, 0, dstBytes, heap);
	const got = new Float32Array(heap.slice(0));

	const delta = maxAbsDelta(got, reference);
	expect(delta).toBeLessThan(tolerance);

	ctx.dataManager.destroy();
	device.destroy();
}

describe("JSEP rms_norm golden", () => {
	if (!HAS_WEBGPU) {
		test.skip("requires WebGPU; covered by browser smoke", () => {});
		return;
	}

	test("rms_norm 1×2048 typical attention width, eps=1e-5", async () => {
		await runRmsNormGolden(1, 2048, 1e-5, 1e-4);
	});

	test("rms_norm 1×64 small width, eps=1e-6", async () => {
		await runRmsNormGolden(1, 64, 1e-6, 1e-4);
	});
});
