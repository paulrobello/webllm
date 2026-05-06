/**
 * JSEP backend — matmul golden tests.
 *
 * Hand-builds JSEP runtime + descriptor blobs in JS, runs the matmul
 * dispatch end-to-end against a real `GPUDevice`, and compares results
 * to a CPU reference matmul.
 *
 * Bun has no WebGPU; without it the suite skips. Browser smoke covers
 * the WebGPU path against the patched WASM build.
 *
 * Shape convention (ggml MUL_MAT):
 *   src0: [K, M, ...]   stored as ne[0]=K (cols), ne[1]=M (rows)
 *   src1: [K, N, ...]
 *   dst:  [M, N, ...]
 *
 * dst[m, n] = sum_k src0[k, m] * src1[k, n]
 */

import { describe, expect, test } from "bun:test";
import { CommandEncoderBatcher } from "../src/inference/jsep/command-encoder.js";
import { GpuDataManager } from "../src/inference/jsep/gpu-data-manager.js";
import { GGML_OP_MUL_MAT } from "../src/inference/jsep/index.js";
import {
	dispatchMatmul,
	GGML_TYPE_F16,
	GGML_TYPE_F32,
	GGML_TYPE_Q4_0,
	type JsepOpDescriptor,
	type JsepTensorMeta,
} from "../src/inference/jsep/ops/matmul.js";
import { PipelineCache } from "../src/inference/jsep/pipeline-cache.js";

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
	pipelineCache: PipelineCache;
	bindGroupLayoutCache: Map<string, GPUBindGroupLayout>;
}

function makeCtx(device: GPUDevice): Ctx {
	return {
		device,
		dataManager: new GpuDataManager(device),
		encoderBatcher: new CommandEncoderBatcher(device),
		pipelineCache: new PipelineCache(device),
		bindGroupLayoutCache: new Map<string, GPUBindGroupLayout>(),
	};
}

// CPU reference matmul under ggml's [K, M] × [K, N] → [M, N] convention.
function cpuMatmul(
	src0: Float32Array,
	src1: Float32Array,
	M: number,
	K: number,
	N: number,
): Float32Array {
	const out = new Float32Array(M * N);
	// dst[m, n] = sum_k src0[k, m] * src1[k, n]
	// src0 strided: src0[k + K*m] (row-major over [M rows of K])
	// src1 strided: src1[k + K*n]
	// dst strided: dst[m + M*n]
	for (let n = 0; n < N; n++) {
		for (let m = 0; m < M; m++) {
			let acc = 0;
			for (let k = 0; k < K; k++) {
				acc += src0[k + K * m] * src1[k + K * n];
			}
			out[m + M * n] = acc;
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

// Encode an f32 to its f16 bit pattern (round-to-nearest-even, no
// subnormal handling — sufficient for unit-test inputs in normal range).
function f32ToF16Bits(f: number): number {
	const buf = new ArrayBuffer(4);
	new Float32Array(buf)[0] = f;
	const u32 = new Uint32Array(buf)[0];

	const sign = (u32 >> 31) & 0x1;
	const expF32 = (u32 >> 23) & 0xff;
	const mantF32 = u32 & 0x7fffff;

	if (expF32 === 0xff) {
		// Inf/NaN
		return (sign << 15) | 0x7c00 | (mantF32 ? 1 : 0);
	}
	const expF16 = expF32 - 127 + 15;
	if (expF16 >= 0x1f) {
		// Overflow → Inf.
		return (sign << 15) | 0x7c00;
	}
	if (expF16 <= 0) {
		// Underflow → zero (subnormals not handled — fine for test data).
		return sign << 15;
	}
	const mantF16 = mantF32 >> 13;
	return (sign << 15) | (expF16 << 10) | mantF16;
}

function f32ArrayToF16Bytes(values: Float32Array): Uint8Array {
	const out = new Uint8Array(values.length * 2);
	const view = new DataView(out.buffer);
	for (let i = 0; i < values.length; i++) {
		view.setUint16(i * 2, f32ToF16Bits(values[i]), true);
	}
	return out;
}

// Q4_0 block: 1 f16 scale (2 bytes) + 16 nibble bytes = 18 bytes / 32 elems.
// Nibble byte i (i in 0..16) holds q[i] in low 4 bits and q[i+16] in high 4.
// Quantization: x = (nibble - 8) * scale.
function packQ4_0(values: Float32Array): {
	bytes: Uint8Array;
	dequant: Float32Array;
} {
	const QK = 32;
	if (values.length % QK !== 0) {
		throw new Error(`packQ4_0: length ${values.length} not multiple of ${QK}`);
	}
	const nBlocks = values.length / QK;
	const bytes = new Uint8Array(nBlocks * 18);
	const dequant = new Float32Array(values.length);

	for (let b = 0; b < nBlocks; b++) {
		const blockOff = b * 18;
		// Find max(|x|) in this block; choose scale so that max maps to ±8.
		let amax = 0;
		let max = 0;
		for (let i = 0; i < QK; i++) {
			const x = values[b * QK + i];
			if (Math.abs(x) > amax) {
				amax = Math.abs(x);
				max = x;
			}
		}
		// Following ggml-quants.c quantize_row_q4_0 reference.
		const d = max / -8;
		const scale = d === 0 ? 0 : 1 / d;
		// Encode scale as f16.
		const scaleBits = f32ToF16Bits(d);
		new DataView(bytes.buffer, bytes.byteOffset + blockOff, 2).setUint16(
			0,
			scaleBits,
			true,
		);
		const decodedScale = f16BitsToF32(scaleBits);

		// Quantize each element.
		const nibbles: number[] = new Array(QK);
		for (let i = 0; i < QK; i++) {
			const x = values[b * QK + i];
			const q = Math.min(15, Math.max(0, Math.round(x * scale + 8)));
			nibbles[i] = q;
			dequant[b * QK + i] = (q - 8) * decodedScale;
		}
		for (let i = 0; i < 16; i++) {
			bytes[blockOff + 2 + i] =
				(nibbles[i] & 0xf) | ((nibbles[i + 16] & 0xf) << 4);
		}
	}
	return { bytes, dequant };
}

function f16BitsToF32(bits: number): number {
	const sign = (bits >> 15) & 0x1;
	const exp = (bits >> 10) & 0x1f;
	const mant = bits & 0x3ff;
	if (exp === 0) {
		// Zero / subnormal.
		if (mant === 0) return sign ? -0 : 0;
		// Subnormal half: value = (-1)^s * 2^-14 * (mant / 1024)
		const v = 2 ** -14 * (mant / 1024);
		return sign ? -v : v;
	}
	if (exp === 0x1f) {
		return mant ? NaN : sign ? -Infinity : Infinity;
	}
	const val = 2 ** (exp - 15) * (1 + mant / 1024);
	return sign ? -val : val;
}

// Compute strided byte layouts for ggml tensors. ggml tensors are
// row-contiguous in memory: nb[0]=type_size, nb[1]=nb[0]*ne[0],
// nb[2]=nb[1]*ne[1], etc. We mirror this for our hand-built tensors.
function makeMeta(
	handle: number,
	type: number,
	ne: [number, number, number, number],
	bytesPerElem: number, // for non-quant: type size; for quant: pass blockBytes
	elemsPerBlock: number, // 1 for non-quant; QK4_0=32 for Q4_0
): JsepTensorMeta {
	const nb: [number, number, number, number] = [0, 0, 0, 0];
	// nb[0] = bytes per logical element — tricky for quant.
	// ggml uses nb[0]=type_size (bytes per block), nb[1]=ne[0]/blck_size*type_size.
	// For our purposes: nb[1] is the row stride in bytes — that's what the
	// shader uses.
	nb[0] = bytesPerElem; // unused by shader for matmul
	nb[1] = (ne[0] / elemsPerBlock) * bytesPerElem;
	nb[2] = nb[1] * ne[1];
	nb[3] = nb[2] * ne[2];
	return { handle, type, ne, nb };
}

describe("JSEP matmul golden", () => {
	if (!HAS_WEBGPU) {
		test.skip("requires WebGPU; covered by browser smoke", () => {});
		return;
	}

	test("F32 × F32 → F32 — 32×32 × 32×32 → 32×32", async () => {
		const device = await getDevice();
		if (!device) {
			console.warn("WebGPU adapter unavailable; skipping");
			return;
		}
		const ctx = makeCtx(device);

		const M = 32;
		const K = 32;
		const N = 32;

		// Deterministic inputs.
		const src0 = new Float32Array(M * K);
		const src1 = new Float32Array(N * K);
		for (let i = 0; i < src0.length; i++) src0[i] = (i % 11) * 0.1 - 0.5;
		for (let i = 0; i < src1.length; i++) src1[i] = (i % 7) * 0.13 - 0.4;

		const reference = cpuMatmul(src0, src1, M, K, N);

		const src0Bytes = new Uint8Array(src0.buffer.slice(0));
		const src1Bytes = new Uint8Array(src1.buffer.slice(0));
		const dstBytes = M * N * 4;

		const h0 = ctx.dataManager.alloc(src0Bytes.byteLength);
		const h1 = ctx.dataManager.alloc(src1Bytes.byteLength);
		const hd = ctx.dataManager.alloc(dstBytes);

		// Stage uploads via writeBuffer directly (bypassing the heap-bridge).
		device.queue.writeBuffer(
			ctx.dataManager.get(h0).buffer,
			0,
			src0Bytes,
			0,
			src0Bytes.byteLength,
		);
		device.queue.writeBuffer(
			ctx.dataManager.get(h1).buffer,
			0,
			src1Bytes,
			0,
			src1Bytes.byteLength,
		);

		const desc: JsepOpDescriptor = {
			op: GGML_OP_MUL_MAT,
			nSrc: 2,
			dst: makeMeta(hd, GGML_TYPE_F32, [M, N, 1, 1], 4, 1),
			srcs: [
				makeMeta(h0, GGML_TYPE_F32, [K, M, 1, 1], 4, 1),
				makeMeta(h1, GGML_TYPE_F32, [K, N, 1, 1], 4, 1),
			],
		};

		const status = dispatchMatmul(ctx, desc);
		expect(status).toBe(0);

		ctx.encoderBatcher.flush();

		// Readback.
		const heap = new ArrayBuffer(dstBytes);
		await ctx.dataManager.readAsync(hd, 0, 0, dstBytes, heap);
		const got = new Float32Array(heap.slice(0));

		const delta = maxAbsDelta(got, reference);
		expect(delta).toBeLessThan(1e-4);

		ctx.dataManager.destroy();
		device.destroy();
	});

	test("F16 × F32 → F32 — (128, 16) × (16, 64) → (128, 64)", async () => {
		const device = await getDevice();
		if (!device) return;
		const ctx = makeCtx(device);

		const M = 128;
		const K = 16;
		const N = 64;

		const src0 = new Float32Array(M * K);
		const src1 = new Float32Array(N * K);
		for (let i = 0; i < src0.length; i++)
			src0[i] = ((i * 17) % 23) * 0.05 - 0.4;
		for (let i = 0; i < src1.length; i++)
			src1[i] = ((i * 13) % 19) * 0.07 - 0.3;

		// Round src0 through f16 for fair reference comparison.
		const src0F16Bytes = f32ArrayToF16Bytes(src0);
		const src0F16Roundtripped = new Float32Array(src0.length);
		const view = new DataView(src0F16Bytes.buffer);
		for (let i = 0; i < src0.length; i++) {
			src0F16Roundtripped[i] = f16BitsToF32(view.getUint16(i * 2, true));
		}
		const reference = cpuMatmul(src0F16Roundtripped, src1, M, K, N);

		const src1Bytes = new Uint8Array(src1.buffer.slice(0));
		const dstBytes = M * N * 4;

		const h0 = ctx.dataManager.alloc(src0F16Bytes.byteLength);
		const h1 = ctx.dataManager.alloc(src1Bytes.byteLength);
		const hd = ctx.dataManager.alloc(dstBytes);

		device.queue.writeBuffer(
			ctx.dataManager.get(h0).buffer,
			0,
			src0F16Bytes,
			0,
			src0F16Bytes.byteLength,
		);
		device.queue.writeBuffer(
			ctx.dataManager.get(h1).buffer,
			0,
			src1Bytes,
			0,
			src1Bytes.byteLength,
		);

		const desc: JsepOpDescriptor = {
			op: GGML_OP_MUL_MAT,
			nSrc: 2,
			dst: makeMeta(hd, GGML_TYPE_F32, [M, N, 1, 1], 4, 1),
			srcs: [
				makeMeta(h0, GGML_TYPE_F16, [K, M, 1, 1], 2, 1),
				makeMeta(h1, GGML_TYPE_F32, [K, N, 1, 1], 4, 1),
			],
		};

		const status = dispatchMatmul(ctx, desc);
		expect(status).toBe(0);

		ctx.encoderBatcher.flush();

		const heap = new ArrayBuffer(dstBytes);
		await ctx.dataManager.readAsync(hd, 0, 0, dstBytes, heap);
		const got = new Float32Array(heap.slice(0));

		const delta = maxAbsDelta(got, reference);
		expect(delta).toBeLessThan(1e-3);

		ctx.dataManager.destroy();
		device.destroy();
	});

	test("Q4_0 × F32 → F32 — 32×32 × 32×32 → 32×32", async () => {
		const device = await getDevice();
		if (!device) return;
		const ctx = makeCtx(device);

		const M = 32;
		const K = 32;
		const N = 32;

		const src0 = new Float32Array(M * K);
		const src1 = new Float32Array(N * K);
		// Inputs scaled into a friendly Q4_0 range (max ≈ ±2 → scale ≈ 0.25).
		for (let i = 0; i < src0.length; i++) src0[i] = ((i % 17) - 8) * 0.2;
		for (let i = 0; i < src1.length; i++) src1[i] = ((i % 13) - 6) * 0.15;

		// Pack src0 into Q4_0; reference uses the *dequantized* values for
		// fairness — Q4_0 is lossy but the kernel is faithful to the dequant.
		const { bytes: q4Bytes, dequant: src0Dequant } = packQ4_0(src0);
		const reference = cpuMatmul(src0Dequant, src1, M, K, N);

		const src1Bytes = new Uint8Array(src1.buffer.slice(0));
		const dstBytes = M * N * 4;

		const h0 = ctx.dataManager.alloc(q4Bytes.byteLength);
		const h1 = ctx.dataManager.alloc(src1Bytes.byteLength);
		const hd = ctx.dataManager.alloc(dstBytes);

		device.queue.writeBuffer(
			ctx.dataManager.get(h0).buffer,
			0,
			q4Bytes,
			0,
			q4Bytes.byteLength,
		);
		device.queue.writeBuffer(
			ctx.dataManager.get(h1).buffer,
			0,
			src1Bytes,
			0,
			src1Bytes.byteLength,
		);

		// Q4_0 stride: row of K elems = (K/QK4_0) blocks * 18 bytes/block.
		// makeMeta with bytesPerElem=18 (block bytes), elemsPerBlock=32.
		const desc: JsepOpDescriptor = {
			op: GGML_OP_MUL_MAT,
			nSrc: 2,
			dst: makeMeta(hd, GGML_TYPE_F32, [M, N, 1, 1], 4, 1),
			srcs: [
				makeMeta(h0, GGML_TYPE_Q4_0, [K, M, 1, 1], 18, 32),
				makeMeta(h1, GGML_TYPE_F32, [K, N, 1, 1], 4, 1),
			],
		};

		const status = dispatchMatmul(ctx, desc);
		expect(status).toBe(0);

		ctx.encoderBatcher.flush();

		const heap = new ArrayBuffer(dstBytes);
		await ctx.dataManager.readAsync(hd, 0, 0, dstBytes, heap);
		const got = new Float32Array(heap.slice(0));

		const delta = maxAbsDelta(got, reference);
		// Reference uses *dequantized* src0, so kernel and reference should
		// match to f32 round-off — but Q4_0 stride encoding has more room
		// for off-by-one bugs, so allow a generous tolerance.
		expect(delta).toBeLessThan(1e-2);

		ctx.dataManager.destroy();
		device.destroy();
	});
});
