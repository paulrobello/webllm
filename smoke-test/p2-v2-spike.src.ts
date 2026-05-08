// P2-v2 Phase 2 follow-on spike harness — exercise the JSEP backend
// end-to-end via webllm_decode and measure the T3 gate metrics.
//
// Loads tinyllama-1.1b-chat-q4_0.gguf through the JSEP-built WASM
// (`webllm-wasm-jsep.js`), installs JSEP callbacks, decodes the
// "The capital of France is" 6-token prompt, then greedy-decodes 5
// continuation tokens. After the loop, snapshots `module.__jsep.counters`
// and reports per-token wall + EM_ASM crossings/token.
//
// Bundled to smoke-test/p2-v2-spike.js via:
//   bun build smoke-test/p2-v2-spike.src.ts --outfile smoke-test/p2-v2-spike.js \
//     --target browser

import { installJsepCallbacks } from "../src/inference/jsep/index.js";
import { createLlamaBridge } from "../src/inference/llama-bridge.js";
import { GgufParser } from "../src/models/gguf-parser.js";
import {
	dispatchMatmul,
	GGML_TYPE_F16,
	GGML_TYPE_F32,
	GGML_TYPE_Q4_0,
	GGML_TYPE_Q4_K,
	type JsepOpDescriptor,
} from "../src/inference/jsep/ops/matmul.js";
import {
	GGML_OP_MUL_MAT,
	GGML_OP_RMS_NORM,
	GGML_OP_SET_ROWS,
} from "../src/inference/jsep/index.js";
import { dispatchRmsNorm } from "../src/inference/jsep/ops/rms-norm.js";
import { dispatchSetRows } from "../src/inference/jsep/ops/set-rows.js";

// I64 indices type for SET_ROWS. Mirrors the ggml internal enum value
// (`GGML_TYPE_I64 = 27`) — we don't import the constant from set-rows.ts
// because it lives in a `__setRowsInternals` re-export not on the
// public path.
const GGML_TYPE_I64 = 27;

// Same fixture as p0-spike.src.ts — "The capital of France is".
const PROMPT_TOKEN_IDS = [1, 450, 7483, 310, 3444, 338];
const N_GENERATE = 5;
const GGUF_URL = "/models/tinyllama-1.1b-chat-q4_0.gguf";

// ----- Q4_K kernel self-test ------------------------------------------------

function f32ToF16Bits(f: number): number {
	const buf = new ArrayBuffer(4);
	new Float32Array(buf)[0] = f;
	const u32 = new Uint32Array(buf)[0];
	const sign = (u32 >> 31) & 0x1;
	const expF32 = (u32 >> 23) & 0xff;
	const mantF32 = u32 & 0x7fffff;
	if (expF32 === 0xff) return (sign << 15) | 0x7c00 | (mantF32 ? 1 : 0);
	const expF16 = expF32 - 127 + 15;
	if (expF16 >= 0x1f) return (sign << 15) | 0x7c00;
	if (expF16 <= 0) return sign << 15;
	return (sign << 15) | (expF16 << 10) | (mantF32 >> 13);
}

function f16BitsToF32(bits: number): number {
	const sign = (bits >> 15) & 0x1;
	const exp = (bits >> 10) & 0x1f;
	const mant = bits & 0x3ff;
	if (exp === 0) {
		if (mant === 0) return sign ? -0 : 0;
		const v = 2 ** -14 * (mant / 1024);
		return sign ? -v : v;
	}
	if (exp === 0x1f) return mant ? NaN : sign ? -Infinity : Infinity;
	const val = 2 ** (exp - 15) * (1 + mant / 1024);
	return sign ? -val : val;
}

// Pack a single Q4_K super-block (256 elems / 144 bytes) given explicit
// (d, dmin, sc[8], m[8], nibbles[256]).
function packQ4_KSingle(
	d: number,
	dmin: number,
	sc: Uint8Array,
	m: Uint8Array,
	nibbles: Uint8Array,
): { bytes: Uint8Array; dequant: Float32Array } {
	const bytes = new Uint8Array(144);
	const view = new DataView(bytes.buffer);
	view.setUint16(0, f32ToF16Bits(d), true);
	view.setUint16(2, f32ToF16Bits(dmin), true);
	for (let j = 0; j < 4; j++) {
		bytes[4 + j] = (sc[j] & 0x3f) | (((sc[j + 4] >> 4) & 0x3) << 6);
		bytes[4 + j + 4] = (m[j] & 0x3f) | (((m[j + 4] >> 4) & 0x3) << 6);
		bytes[4 + j + 8] = (sc[j + 4] & 0xf) | ((m[j + 4] & 0xf) << 4);
	}
	for (let pair = 0; pair < 4; pair++) {
		for (let l = 0; l < 32; l++) {
			const lo = nibbles[pair * 64 + l] & 0xf;
			const hi = nibbles[pair * 64 + 32 + l] & 0xf;
			bytes[16 + pair * 32 + l] = lo | (hi << 4);
		}
	}
	const dequant = new Float32Array(256);
	const decodedD = f16BitsToF32(f32ToF16Bits(d));
	const decodedDmin = f16BitsToF32(f32ToF16Bits(dmin));
	for (let pair = 0; pair < 4; pair++) {
		const is0 = pair * 2;
		const is1 = pair * 2 + 1;
		const d1 = decodedD * sc[is0];
		const m1 = decodedDmin * m[is0];
		const d2 = decodedD * sc[is1];
		const m2 = decodedDmin * m[is1];
		for (let l = 0; l < 32; l++) {
			dequant[pair * 64 + l] = d1 * nibbles[pair * 64 + l] - m1;
			dequant[pair * 64 + 32 + l] = d2 * nibbles[pair * 64 + 32 + l] - m2;
		}
	}
	return { bytes, dequant };
}

async function runQ4KSelfTest(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
): Promise<void> {
	const M = 1;
	const K = 256;
	const N = 1;
	const sc = new Uint8Array([5, 12, 19, 26, 33, 40, 47, 54]);
	const m = new Uint8Array([3, 7, 11, 15, 19, 23, 27, 31]);
	const nibbles = new Uint8Array(256);
	for (let i = 0; i < 256; i++) nibbles[i] = i & 0xf;
	const { bytes: q4Bytes, dequant: src0Dequant } = packQ4_KSingle(
		0.05,
		0.01,
		sc,
		m,
		nibbles,
	);
	const src1 = new Float32Array(N * K);
	for (let i = 0; i < src1.length; i++) src1[i] = ((i % 13) - 6) * 0.15;

	let reference = 0;
	for (let k = 0; k < K; k++) reference += src0Dequant[k] * src1[k];

	const h0 = runtime.dataManager.alloc(q4Bytes.byteLength);
	const h1 = runtime.dataManager.alloc(src1.byteLength);
	const hd = runtime.dataManager.alloc(M * N * 4);
	const rec0 = runtime.dataManager.get(h0);
	const rec1 = runtime.dataManager.get(h1);
	runtime.device.queue.writeBuffer(
		rec0.buffer,
		0,
		q4Bytes,
		0,
		q4Bytes.byteLength,
	);
	runtime.device.queue.writeBuffer(
		rec1.buffer,
		0,
		new Uint8Array(src1.buffer),
		0,
		src1.byteLength,
	);

	const desc: JsepOpDescriptor = {
		op: GGML_OP_MUL_MAT,
		nSrc: 2,
		dst: {
			bufHandle: hd,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [M, N, 1, 1],
			nb: [4, 4, 4 * M, 4 * M * N],
		},
		srcs: [
			{
				bufHandle: h0,
				offset: 0,
				type: GGML_TYPE_Q4_K,
				ne: [K, M, 1, 1],
				nb: [144, 144 * (K / 256), 144 * (K / 256) * M, 0],
			},
			{
				bufHandle: h1,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [K, N, 1, 1],
				nb: [4, 4 * K, 4 * K * N, 0],
			},
		],
	};
	const status = dispatchMatmul(runtime, desc);
	runtime.encoderBatcher.flush();

	const recd = runtime.dataManager.get(hd);
	const staging = runtime.device.createBuffer({
		size: M * N * 4,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(recd.buffer, 0, staging, 0, M * N * 4);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, M * N * 4);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();
	const passLog = {
		status,
		got: got[0],
		reference,
		delta: Math.abs(got[0] - reference),
		dequantFirst4: Array.from(src0Dequant.slice(0, 4)),
		dequantLast4: Array.from(src0Dequant.slice(-4)),
	};
	log(`Q4K_SELFTEST = ${JSON.stringify(passLog)}`);
	(window as any).__q4kSelfTest = passLog;

	runtime.dataManager.free(h0);
	runtime.dataManager.free(h1);
	runtime.dataManager.free(hd);
}

// Stage 4.3a: production-shape Q4_K MUL_MAT self-test --------------------
//
// Existing Q4K_SELFTEST exercises K=256 (single super-block per row). Real
// MUL_MATs in the prefill graph use K=2048 (8 super-blocks per row) and the
// divert path (dst aliases src1's buffer). This selftest covers both:
//   B1 (no divert, M=64, K=2048, N=6)  — kernel correctness at production K
//   B2 (divert,    M=64, K=2048, N=6)  — divert path correctness
//
// Outcome triage:
//   B1 fails               → Q4_K kernel buggy at large K (per-super-block bug)
//   B1 passes, B2 fails    → divert path buggy (tempDst lifecycle? cpyBuf2Buf?)
//   both pass              → Bug A is upstream of MUL_MAT; pivot to 4.3b
//
// CPU reference uses the same packQ4_KSingle helper as Q4K_SELFTEST so any
// dequant divergence with the WGSL kernel surfaces as cross-test inconsistency.

interface MatmulSelfTestResult {
	mode: "no-divert" | "divert";
	M: number;
	K: number;
	N: number;
	status: number;
	maxAbsDelta: number;
	hasNaN: boolean;
	hasInf: boolean;
	zeroCount: number;
	first4Got: number[];
	first4Ref: number[];
	last4Got: number[];
	last4Ref: number[];
}

function buildSyntheticQ4KMatrix(M: number, K: number): {
	bytes: Uint8Array;
	dequant: Float32Array;
	rowBytes: number;
} {
	if (K % 256 !== 0) throw new Error(`K=${K} must be multiple of 256`);
	const superBlocksPerRow = K / 256;
	const rowBytes = superBlocksPerRow * 144;
	const bytes = new Uint8Array(M * rowBytes);
	const dequant = new Float32Array(M * K);

	// Per-super-block parameters: deterministic, varied across rows + blocks.
	for (let r = 0; r < M; r++) {
		for (let sb = 0; sb < superBlocksPerRow; sb++) {
			// Pick d/dmin/sc/m so dequant magnitudes stay in roughly the
			// same range as the existing Q4K_SELFTEST (max ~40), but vary
			// per (r, sb) so a stride bug shows up as cross-block leakage.
			const d = 0.04 + 0.0001 * ((r + sb * 3) % 17);
			const dmin = 0.008 + 0.0001 * ((r * 5 + sb) % 11);
			const sc = new Uint8Array(8);
			const m = new Uint8Array(8);
			for (let i = 0; i < 8; i++) {
				sc[i] = (4 + i * 7 + r + sb * 2) & 0x3f;
				m[i] = (2 + i * 5 + r * 3 + sb) & 0x3f;
			}
			const nibbles = new Uint8Array(256);
			for (let i = 0; i < 256; i++) {
				nibbles[i] = (i + r + sb * 11) & 0xf;
			}
			const { bytes: blkBytes, dequant: blkDeq } = packQ4_KSingle(
				d,
				dmin,
				sc,
				m,
				nibbles,
			);
			bytes.set(blkBytes, r * rowBytes + sb * 144);
			dequant.set(blkDeq, r * K + sb * 256);
		}
	}

	return { bytes, dequant, rowBytes };
}

async function runMatmulProductionSelfTest(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
	mode: "no-divert" | "divert",
): Promise<MatmulSelfTestResult> {
	const M = 64;
	const K = 2048;
	const N = 6;

	const {
		bytes: q4Bytes,
		dequant: src0Dequant,
		rowBytes,
	} = buildSyntheticQ4KMatrix(M, K);

	// src1 in roughly the same range as RMSNORM output (~[-1.6, 1.6]).
	const src1 = new Float32Array(N * K);
	for (let n = 0; n < N; n++) {
		for (let k = 0; k < K; k++) {
			src1[n * K + k] = (((k * 13 + n * 7) % 31) - 15) * 0.1;
		}
	}

	// CPU reference: dst[m,n] = sum_k src0_dequant[m,k] * src1[n,k]
	// Output layout: [M, N] with nb[1] = M*4 (col-major in m for given n).
	const reference = new Float32Array(M * N);
	for (let n = 0; n < N; n++) {
		for (let m = 0; m < M; m++) {
			let acc = 0;
			for (let k = 0; k < K; k++) {
				acc += src0Dequant[m * K + k] * src1[n * K + k];
			}
			reference[n * M + m] = acc;
		}
	}

	// In the divert variant we put src1 and dst into the same GPU buffer
	// (different offsets) so dispatchMatmul fires the divert path.
	const dstBytes = M * N * 4;
	let h0: number;
	let hSrc1: number;
	let hDst: number;
	let src1Offset = 0;
	let dstOffset = 0;
	if (mode === "no-divert") {
		h0 = runtime.dataManager.alloc(q4Bytes.byteLength);
		hSrc1 = runtime.dataManager.alloc(src1.byteLength);
		hDst = runtime.dataManager.alloc(dstBytes);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(h0).buffer,
			0,
			q4Bytes,
			0,
			q4Bytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			0,
			new Uint8Array(src1.buffer),
			0,
			src1.byteLength,
		);
	} else {
		// One shared buffer holds src1 at offset 0 and dst at offset
		// aligned past src1. 256-byte alignment matches the production
		// allocator's GGML_JSEP_BUFFER_ALIGN.
		h0 = runtime.dataManager.alloc(q4Bytes.byteLength);
		const ALIGN = 256;
		const src1Aligned = (src1.byteLength + ALIGN - 1) & ~(ALIGN - 1);
		const sharedBytes = src1Aligned + dstBytes;
		hSrc1 = runtime.dataManager.alloc(sharedBytes);
		hDst = hSrc1; // alias — triggers divert
		src1Offset = 0;
		dstOffset = src1Aligned;
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(h0).buffer,
			0,
			q4Bytes,
			0,
			q4Bytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			0,
			new Uint8Array(src1.buffer),
			0,
			src1.byteLength,
		);
		// Pre-fill the dst region with a sentinel (-7777) so a no-write
		// failure shows up as that value rather than zero.
		const sentinel = new Float32Array(dstBytes / 4);
		sentinel.fill(-7777);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			dstOffset,
			new Uint8Array(sentinel.buffer),
			0,
			dstBytes,
		);
	}

	const desc: JsepOpDescriptor = {
		op: GGML_OP_MUL_MAT,
		nSrc: 2,
		dst: {
			bufHandle: hDst,
			offset: dstOffset,
			type: GGML_TYPE_F32,
			ne: [M, N, 1, 1],
			nb: [4, 4 * M, 4 * M * N, 4 * M * N],
		},
		srcs: [
			{
				bufHandle: h0,
				offset: 0,
				type: GGML_TYPE_Q4_K,
				ne: [K, M, 1, 1],
				nb: [144, rowBytes, rowBytes * M, 0],
			},
			{
				bufHandle: hSrc1,
				offset: src1Offset,
				type: GGML_TYPE_F32,
				ne: [K, N, 1, 1],
				nb: [4, 4 * K, 4 * K * N, 0],
			},
		],
	};
	const status = dispatchMatmul(runtime, desc);
	runtime.encoderBatcher.flush();

	const dstRec = runtime.dataManager.get(hDst);
	const staging = runtime.device.createBuffer({
		size: dstBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(dstRec.buffer, dstOffset, staging, 0, dstBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, dstBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDelta = 0;
	let hasNaN = false;
	let hasInf = false;
	let zeroCount = 0;
	for (let i = 0; i < got.length; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		else if (v === 0) zeroCount++;
		const d = Math.abs(v - reference[i]);
		if (d > maxAbsDelta) maxAbsDelta = d;
	}

	const result: MatmulSelfTestResult = {
		mode,
		M,
		K,
		N,
		status,
		maxAbsDelta,
		hasNaN,
		hasInf,
		zeroCount,
		first4Got: Array.from(got.slice(0, 4)),
		first4Ref: Array.from(reference.slice(0, 4)),
		last4Got: Array.from(got.slice(-4)),
		last4Ref: Array.from(reference.slice(-4)),
	};

	runtime.dataManager.free(h0);
	runtime.dataManager.free(hSrc1);
	if (mode === "no-divert") runtime.dataManager.free(hDst);

	return result;
}

// ----- Stage 4.18 Probe 8a: Q4_0 production-shape matmul sweep -------------
//
// Stage 4.17 confirmed Outcome B (kernel-precision divergence) at the very
// first compute node (Qcur-0, ne=[2048,6,1,1], Q4_0 weights). The 5.24e-4
// max-abs first8 delta vs the CPU-fallback reference compounds across 22
// layers into a +6 magnitude logits delta. This sweep characterizes the
// JSEP Q4_0 matmul kernel's precision profile across all production shapes
// used in TinyLlama-Q4_0:
//   (2048, 2048, 6) — Q-proj / out-proj          (verified on JSEP per Probe 8b)
//   (256,  2048, 6) — K-proj                       (V-proj routes to CPU)
//   (5632, 2048, 6) — FFN gate / up               (NB: FFN routes to CPU per Probe 8b — measured for completeness)
//   (2048, 5632, 6) — FFN down                    (CPU per 8b)
//   (32000, 2048, 1) — lm_head                    (CPU per 8b)
//
// For each shape the harness reports BOTH:
//   - delta vs f64 ground truth (sums in JS f64, "exact" floating-point)
//   - delta vs f32 element-wise reference (sums in JS f32 with the same
//     k-major order as the JSEP kernel; if JSEP matches this to ULP it
//     means the kernel is mathematically identical to a CPU f32 single-
//     pass loop and the precision delta is purely an f32-summation
//     non-associativity artifact, not a kernel bug)
//
// Output magnitudes are NOT bounded — random Q4_0 weights × random src1
// produce sums with normal magnitudes ~sqrt(K) × per-term-magnitude. We
// report `outputMaxAbs` so callers can compute relative error.

const QK4_0 = 32;
const Q4_0_BYTES_PER_BLOCK = 18;

// Pack one Q4_0 block (32 elements). Returns 18 bytes + 32 dequantized
// values (post-roundtrip-through-f16 scale, matching the WGSL kernel's
// dequant exactly).
function packQ4_0Block(
	d: number,
	nibbles: Uint8Array,
): { bytes: Uint8Array; dequant: Float32Array } {
	if (nibbles.length !== QK4_0) throw new Error("Q4_0 block needs 32 nibbles");
	const bytes = new Uint8Array(Q4_0_BYTES_PER_BLOCK);
	const view = new DataView(bytes.buffer);
	view.setUint16(0, f32ToF16Bits(d), true);
	for (let i = 0; i < 16; i++) {
		bytes[2 + i] = (nibbles[i] & 0xf) | ((nibbles[i + 16] & 0xf) << 4);
	}
	const decodedD = f16BitsToF32(f32ToF16Bits(d));
	const dequant = new Float32Array(QK4_0);
	for (let i = 0; i < QK4_0; i++) {
		dequant[i] = (nibbles[i] - 8) * decodedD;
	}
	return { bytes, dequant };
}

function buildSyntheticQ4_0Matrix(
	M: number,
	K: number,
	seed: number,
): { bytes: Uint8Array; dequant: Float32Array; rowBytes: number } {
	if (K % QK4_0 !== 0) throw new Error(`K=${K} must be multiple of 32`);
	const blocksPerRow = K / QK4_0;
	const rowBytes = blocksPerRow * Q4_0_BYTES_PER_BLOCK;
	const bytes = new Uint8Array(M * rowBytes);
	const dequant = new Float32Array(M * K);

	// Mid-layer weight scales for TinyLlama Q4_0 land around 0.01-0.05;
	// pick deterministic per-(row, block) values in that range so a
	// stride bug shows up as cross-block leakage rather than uniform
	// scaling.
	const nibbles = new Uint8Array(QK4_0);
	for (let r = 0; r < M; r++) {
		for (let b = 0; b < blocksPerRow; b++) {
			const d = 0.012 + 0.0008 * (((r * 7 + b * 13 + seed) % 31) / 31);
			for (let i = 0; i < QK4_0; i++) {
				nibbles[i] = (i * 5 + r * 3 + b * 11 + seed) & 0xf;
			}
			const { bytes: blkBytes, dequant: blkDeq } = packQ4_0Block(d, nibbles);
			bytes.set(blkBytes, r * rowBytes + b * Q4_0_BYTES_PER_BLOCK);
			dequant.set(blkDeq, r * K + b * QK4_0);
		}
	}
	return { bytes, dequant, rowBytes };
}

interface MatmulQ4_0SweepResult {
	M: number;
	K: number;
	N: number;
	status: number;
	maxAbsDeltaVsF64: number;
	maxAbsDeltaVsF32Loop: number;
	outputMaxAbs: number;
	hasNaN: boolean;
	hasInf: boolean;
	first4Got: number[];
	first4F64: number[];
	first4F32Loop: number[];
}

async function runMatmulQ4_0Sweep(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
	M: number,
	K: number,
	N: number,
): Promise<MatmulQ4_0SweepResult> {
	const seed = (M * 31) ^ (K * 7) ^ (N * 13);
	const { bytes: q4Bytes, dequant: src0Dequant, rowBytes } =
		buildSyntheticQ4_0Matrix(M, K, seed);
	const src1 = new Float32Array(N * K);
	for (let n = 0; n < N; n++) {
		for (let k = 0; k < K; k++) {
			src1[n * K + k] = (((k * 13 + n * 7 + seed) % 31) - 15) * 0.1;
		}
	}

	// Reference 1 — f64 ground truth (JS Number is f64).
	const refF64 = new Float64Array(M * N);
	// Reference 2 — f32 element-wise k-major loop (matches JSEP kernel
	// summation order; promote each accumulate via Math.fround to round-
	// trip through f32).
	const refF32Loop = new Float32Array(M * N);
	for (let n = 0; n < N; n++) {
		for (let m = 0; m < M; m++) {
			let acc64 = 0;
			let acc32 = Math.fround(0);
			for (let k = 0; k < K; k++) {
				const a = src0Dequant[m * K + k];
				const b = src1[n * K + k];
				acc64 += a * b;
				acc32 = Math.fround(acc32 + Math.fround(a * b));
			}
			refF64[n * M + m] = acc64;
			refF32Loop[n * M + m] = acc32;
		}
	}

	const dstBytes = M * N * 4;
	const h0 = runtime.dataManager.alloc(q4Bytes.byteLength);
	const hSrc1 = runtime.dataManager.alloc(src1.byteLength);
	const hDst = runtime.dataManager.alloc(dstBytes);
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(h0).buffer,
		0,
		q4Bytes,
		0,
		q4Bytes.byteLength,
	);
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(hSrc1).buffer,
		0,
		new Uint8Array(src1.buffer),
		0,
		src1.byteLength,
	);

	const desc: JsepOpDescriptor = {
		op: GGML_OP_MUL_MAT,
		nSrc: 2,
		dst: {
			bufHandle: hDst,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [M, N, 1, 1],
			nb: [4, 4 * M, 4 * M * N, 4 * M * N],
		},
		srcs: [
			{
				bufHandle: h0,
				offset: 0,
				type: GGML_TYPE_Q4_0,
				ne: [K, M, 1, 1],
				nb: [Q4_0_BYTES_PER_BLOCK, rowBytes, rowBytes * M, 0],
			},
			{
				bufHandle: hSrc1,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [K, N, 1, 1],
				nb: [4, 4 * K, 4 * K * N, 0],
			},
		],
	};
	const status = dispatchMatmul(runtime, desc);
	runtime.encoderBatcher.flush();

	const dstRec = runtime.dataManager.get(hDst);
	const staging = runtime.device.createBuffer({
		size: dstBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(dstRec.buffer, 0, staging, 0, dstBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, dstBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDeltaVsF64 = 0;
	let maxAbsDeltaVsF32Loop = 0;
	let outputMaxAbs = 0;
	let hasNaN = false;
	let hasInf = false;
	for (let i = 0; i < got.length; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		const d64 = Math.abs(v - refF64[i]);
		const d32 = Math.abs(v - refF32Loop[i]);
		if (d64 > maxAbsDeltaVsF64) maxAbsDeltaVsF64 = d64;
		if (d32 > maxAbsDeltaVsF32Loop) maxAbsDeltaVsF32Loop = d32;
		const a = Math.abs(v);
		if (a > outputMaxAbs) outputMaxAbs = a;
	}

	runtime.dataManager.free(h0);
	runtime.dataManager.free(hSrc1);
	runtime.dataManager.free(hDst);

	return {
		M,
		K,
		N,
		status,
		maxAbsDeltaVsF64,
		maxAbsDeltaVsF32Loop,
		outputMaxAbs,
		hasNaN,
		hasInf,
		first4Got: Array.from(got.slice(0, 4)),
		first4F64: Array.from(refF64.slice(0, 4)).map(Number),
		first4F32Loop: Array.from(refF32Loop.slice(0, 4)),
	};
}

// ----- Stage 4.22 Probe 10: production-dispatch kernel-input replay --------
//
// Stage 4.21 closed Outcome F-1: the host→GPU upload chain is byte-identical
// end-to-end. The 5.24e-4 production Qcur-0 delta vs the 1.68e-6 Stage 4.18
// synthetic-sweep delta (312× gap) must therefore originate inside the
// dispatch / kernel-execution boundary at production conditions. Probe 10
// captures the actual src0 (Q4_0 weight) + src1 (f32 activation) + dst
// bytes the kernel sees at the first production MUL_MAT (Qcur-0,
// M=2048/K=2048/N=6) and replays them through the same dispatchMatmul
// entry point used by Stage 4.18. Verdict:
//   G-1 (synthetic reproduces 5.24e-4 on captured bytes)  → Stage 4.18
//       sweep missed an input distribution / tile geometry case at M=2048.
//   G-2 (synthetic ≤1e-5 on the same bytes)               → bug between
//       dispatch site and shader execution (pipeline cache, bind-group
//       offsets, workgroup count, src0/src1 swap).

// Q4_K row geometry: 144 bytes per 256-element super-block. Mirrors the
// WGSL kernel in `src/inference/jsep/ops/matmul.ts::load_q4_K` and the
// libllama `block_q4_K` layout.
const Q4_K_BYTES_PER_BLOCK_SPIKE = 144;

// Port of `q4k_unpack_scale_min` (matmul.ts WGSL). Returns (sc, m) for
// sub-block index `is` in [0, 8) given the scales[12] byte region.
function q4kUnpackScaleMin(
	bytes: Uint8Array,
	scalesByteBase: number,
	is: number,
): [number, number] {
	const at = (off: number) => bytes[scalesByteBase + off];
	if (is < 4) {
		return [at(is) & 63, at(is + 4) & 63];
	}
	const qA = at(is + 4);
	const qB = at(is - 4);
	const qC = at(is);
	const sc = (qA & 0xf) | ((qB >> 6) << 4);
	const m = (qA >> 4) | ((qC >> 6) << 4);
	return [sc, m];
}

// Dequantize a Q4_K weight tile (M rows × K cols) into f32, matching the
// WGSL kernel's element-by-element formula. Used as the CPU reference for
// Probe 10 replay.
function dequantQ4_KTile(
	bytes: Uint8Array,
	M: number,
	K: number,
): Float32Array {
	if (K % 256 !== 0) throw new Error(`Q4_K K=${K} must be multiple of 256`);
	const blocksPerRow = K / 256;
	const rowBytes = blocksPerRow * Q4_K_BYTES_PER_BLOCK_SPIKE;
	if (bytes.byteLength < M * rowBytes) {
		throw new Error(
			`dequantQ4_KTile: bytes=${bytes.byteLength} < M*rowBytes=${M * rowBytes}`,
		);
	}
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const out = new Float32Array(M * K);
	for (let m = 0; m < M; m++) {
		for (let sb = 0; sb < blocksPerRow; sb++) {
			const blockBase = m * rowBytes + sb * Q4_K_BYTES_PER_BLOCK_SPIKE;
			const d = f16BitsToF32(view.getUint16(blockBase, true));
			const dmin = f16BitsToF32(view.getUint16(blockBase + 2, true));
			const scalesBase = blockBase + 4;
			const qsBase = blockBase + 16;
			for (let inSuper = 0; inSuper < 256; inSuper++) {
				const pair = (inSuper / 64) | 0;
				const withinPair = inSuper % 64;
				const l = withinPair % 32;
				const is = pair * 2 + (withinPair >= 32 ? 1 : 0);
				const qByteIdx = pair * 32 + l;
				const rawByte = bytes[qsBase + qByteIdx];
				const nibble = withinPair < 32 ? rawByte & 0xf : (rawByte >> 4) & 0xf;
				const [sc, mMin] = q4kUnpackScaleMin(bytes, scalesBase, is);
				out[m * K + sb * 256 + inSuper] = d * sc * nibble - dmin * mMin;
			}
		}
	}
	return out;
}

// Dequantize a Q4_0 weight tile (M rows × K cols) into f32, matching the
// WGSL kernel's nibble unpacking (in_block / 16 selects high vs low
// nibble; scale is f16 at block start).
function dequantQ4_0Tile(
	bytes: Uint8Array,
	M: number,
	K: number,
): Float32Array {
	if (K % QK4_0 !== 0) throw new Error(`K=${K} must be multiple of 32`);
	const blocksPerRow = K / QK4_0;
	const rowBytes = blocksPerRow * Q4_0_BYTES_PER_BLOCK;
	if (bytes.byteLength < M * rowBytes) {
		throw new Error(
			`dequantQ4_0Tile: bytes=${bytes.byteLength} < M*rowBytes=${M * rowBytes}`,
		);
	}
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const out = new Float32Array(M * K);
	for (let m = 0; m < M; m++) {
		for (let b = 0; b < blocksPerRow; b++) {
			const base = m * rowBytes + b * Q4_0_BYTES_PER_BLOCK;
			const dF16 = view.getUint16(base, true);
			const d = f16BitsToF32(dF16);
			for (let i = 0; i < QK4_0; i++) {
				const byteIdx = base + 2 + (i % 16);
				const raw = bytes[byteIdx];
				const nibble = i < 16 ? raw & 0xf : (raw >> 4) & 0xf;
				out[m * K + b * QK4_0 + i] = (nibble - 8) * d;
			}
		}
	}
	return out;
}

interface MatmulFromBytesResult {
	M: number;
	K: number;
	N: number;
	src0Type: number;
	status: number;
	maxAbsDeltaVsF32Loop: number;
	maxAbsDeltaVsF64: number;
	outputMaxAbs: number;
	hasNaN: boolean;
	hasInf: boolean;
	first8Got: number[];
	first8F32Loop: number[];
}

// Run a synthetic quantized matmul through the same dispatchMatmul entry
// point used in production, using EXACT bytes captured at the production
// dispatch site. Supports Q4_0 (type 2) and Q4_K (type 12) src0. Returns
// the max-abs delta vs an f32 element-wise k-major reference (matches the
// WGSL kernel's accumulation order).
async function runMatmulFromBytes(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
	M: number,
	K: number,
	N: number,
	src0Type: number,
	q4Bytes: Uint8Array,
	src1Bytes: Uint8Array,
): Promise<MatmulFromBytesResult> {
	let blockBytes: number;
	let elemsPerBlock: number;
	if (src0Type === GGML_TYPE_Q4_0) {
		if (K % QK4_0 !== 0) throw new Error(`Q4_0 K=${K} must be multiple of 32`);
		blockBytes = Q4_0_BYTES_PER_BLOCK;
		elemsPerBlock = QK4_0;
	} else if (src0Type === GGML_TYPE_Q4_K) {
		if (K % 256 !== 0) throw new Error(`Q4_K K=${K} must be multiple of 256`);
		blockBytes = Q4_K_BYTES_PER_BLOCK_SPIKE;
		elemsPerBlock = 256;
	} else {
		throw new Error(`runMatmulFromBytes: unsupported src0Type=${src0Type}`);
	}
	const blocksPerRow = K / elemsPerBlock;
	const rowBytes = blocksPerRow * blockBytes;
	const expectedSrc0Bytes = M * rowBytes;
	const expectedSrc1Bytes = N * K * 4;
	if (q4Bytes.byteLength < expectedSrc0Bytes) {
		throw new Error(
			`runMatmulFromBytes: q4Bytes=${q4Bytes.byteLength} < ` +
				`expected=${expectedSrc0Bytes}`,
		);
	}
	if (src1Bytes.byteLength < expectedSrc1Bytes) {
		throw new Error(
			`runMatmulFromBytes: src1Bytes=${src1Bytes.byteLength} < ` +
				`expected=${expectedSrc1Bytes}`,
		);
	}

	const src0Dequant =
		src0Type === GGML_TYPE_Q4_0
			? dequantQ4_0Tile(q4Bytes, M, K)
			: dequantQ4_KTile(q4Bytes, M, K);
	const src1View = new Float32Array(
		src1Bytes.buffer,
		src1Bytes.byteOffset,
		expectedSrc1Bytes / 4,
	);

	// Reference — f32 element-wise loop in k-major order.
	const refF32Loop = new Float32Array(M * N);
	const refF64 = new Float64Array(M * N);
	for (let n = 0; n < N; n++) {
		for (let m = 0; m < M; m++) {
			let acc32 = Math.fround(0);
			let acc64 = 0;
			for (let k = 0; k < K; k++) {
				const a = src0Dequant[m * K + k];
				const b = src1View[n * K + k];
				acc32 = Math.fround(acc32 + Math.fround(a * b));
				acc64 += a * b;
			}
			refF32Loop[n * M + m] = acc32;
			refF64[n * M + m] = acc64;
		}
	}

	const dstBytes = M * N * 4;
	const h0 = runtime.dataManager.alloc(expectedSrc0Bytes);
	const hSrc1 = runtime.dataManager.alloc(expectedSrc1Bytes);
	const hDst = runtime.dataManager.alloc(dstBytes);
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(h0).buffer,
		0,
		q4Bytes,
		0,
		expectedSrc0Bytes,
	);
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(hSrc1).buffer,
		0,
		src1Bytes,
		0,
		expectedSrc1Bytes,
	);

	const desc: JsepOpDescriptor = {
		op: GGML_OP_MUL_MAT,
		nSrc: 2,
		dst: {
			bufHandle: hDst,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [M, N, 1, 1],
			nb: [4, 4 * M, 4 * M * N, 4 * M * N],
		},
		srcs: [
			{
				bufHandle: h0,
				offset: 0,
				type: src0Type,
				ne: [K, M, 1, 1],
				nb: [blockBytes, rowBytes, rowBytes * M, 0],
			},
			{
				bufHandle: hSrc1,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [K, N, 1, 1],
				nb: [4, 4 * K, 4 * K * N, 0],
			},
		],
	};
	const status = dispatchMatmul(runtime, desc);
	runtime.encoderBatcher.flush();

	const dstRec = runtime.dataManager.get(hDst);
	const staging = runtime.device.createBuffer({
		size: dstBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(dstRec.buffer, 0, staging, 0, dstBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, dstBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDeltaVsF32Loop = 0;
	let maxAbsDeltaVsF64 = 0;
	let outputMaxAbs = 0;
	let hasNaN = false;
	let hasInf = false;
	for (let i = 0; i < got.length; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		const d32 = Math.abs(v - refF32Loop[i]);
		const d64 = Math.abs(v - refF64[i]);
		if (d32 > maxAbsDeltaVsF32Loop) maxAbsDeltaVsF32Loop = d32;
		if (d64 > maxAbsDeltaVsF64) maxAbsDeltaVsF64 = d64;
		const a = Math.abs(v);
		if (a > outputMaxAbs) outputMaxAbs = a;
	}

	runtime.dataManager.free(h0);
	runtime.dataManager.free(hSrc1);
	runtime.dataManager.free(hDst);

	return {
		M,
		K,
		N,
		src0Type,
		status,
		maxAbsDeltaVsF32Loop,
		maxAbsDeltaVsF64,
		outputMaxAbs,
		hasNaN,
		hasInf,
		first8Got: Array.from(got.slice(0, 8)),
		first8F32Loop: Array.from(refF32Loop.slice(0, 8)),
	};
}

// Compare two f32 buffers (provided as Uint8Array views over the same
// f32 layout) elementwise and report max abs delta + first 8 vals from
// both. Used to compare captured production dst-after vs CPU f32 ref.
function compareF32Buffers(
	gotBytes: Uint8Array,
	refF32: Float32Array,
): {
	maxAbsDelta: number;
	hasNaN: boolean;
	hasInf: boolean;
	first8Got: number[];
	first8Ref: number[];
} {
	const got = new Float32Array(
		gotBytes.buffer,
		gotBytes.byteOffset,
		Math.min(gotBytes.byteLength / 4, refF32.length),
	);
	let maxAbsDelta = 0;
	let hasNaN = false;
	let hasInf = false;
	for (let i = 0; i < got.length; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		const d = Math.abs(v - refF32[i]);
		if (d > maxAbsDelta) maxAbsDelta = d;
	}
	return {
		maxAbsDelta,
		hasNaN,
		hasInf,
		first8Got: Array.from(got.slice(0, 8)),
		first8Ref: Array.from(refF32.slice(0, 8)),
	};
}

// ----- SET_ROWS V-cache transpose self-test (Stage 4.6 D1) -----------------
//
// TinyLlama (no FA) writes the V cache via the transposed layout at
// `llama-kv-cache.cpp:1281`:
//
//     v_view = ggml_reshape_2d(v, 1, ggml_nelements(v))
//     SET_ROWS dst:  F16 [1, N_total_cells]   (ne[0]=1)
//     SET_ROWS src0: F32 [1, N_rows]          (one f32 per row)
//     SET_ROWS src1: I64 [N_rows]             (each maps row → cell idx)
//
// The kernel writes one F16 cell per source row, addressing dst as
// `array<atomic<u32>>` and merging into the appropriate halfword via
// CAS (since two adjacent F16 cells share a u32 word and concurrent
// writes to the pair would otherwise race). Existing kernel selftests
// only exercise multi-element rows (RMSNORM/MATMUL); SET_ROWS at the
// V-cache shape has never been verified against a CPU reference. If
// the kernel writes the wrong cell, leaks bits across pair-mates, or
// drops writes under contention, V-cache decode produces real-but-
// wrong tokens — exactly the Stage 4.5 partial-flip symptom.
//
// This selftest covers:
//   1. Indices that DO share a u32 word (0&1, 6&7) — atomic CAS race
//   2. Indices in distinct words (2 alone in word 1, 4 alone in word 2)
//   3. Pre-fill dst with a sentinel pattern; verify untargeted cells
//      are PRESERVED (the divert path's pre-copy is load-bearing —
//      drop of pre-copy would zero out untargeted cells)
//
// Outcome triage:
//   FAIL with maxAbsDelta > 1e-3 on targeted cells → CAS encoding bug
//   FAIL with non-target cells corrupted             → divert pre-copy bug
//   FAIL with off-by-one cell offsets                → indices/stride bug
//   PASS on both no-divert + divert                  → kernel is correct
//                                                      (move to D2 or D3)

interface SetRowsSelfTestResult {
	mode: "no-divert" | "divert";
	status: number;
	N_CELLS: number;
	N_ROWS: number;
	indices: number[];
	srcF32: number[];
	preF16: number[];   // dst F16 cells before dispatch (decoded to f32)
	postF16: number[];  // dst F16 cells after dispatch (decoded to f32)
	expectedF16: number[]; // CPU reference: pre with target cells overwritten
	maxAbsDeltaTargeted: number;
	maxAbsDeltaUntargeted: number;
	hasNaN: boolean;
	hasInf: boolean;
}

async function runSetRowsVCacheSelfTest(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
	mode: "no-divert" | "divert",
): Promise<SetRowsSelfTestResult> {
	// 16 F16 cells = 32 bytes = 8 u32 words. Adjacent index pairs that share
	// a word: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15).
	// We target {0, 1, 6, 7} so two pair-mates collide on word 0 and word 3,
	// while the rest of the buffer is left untargeted (must remain
	// sentinel-valued post-dispatch).
	const N_CELLS = 16;
	const N_ROWS = 4;
	const indices = [0n, 1n, 6n, 7n] as readonly bigint[];

	// Source rows have distinguishable F32 values that survive F16 round-trip.
	// f16(x) loses precision below ~1e-4 / above ~65504 / for denormals; pick
	// values that f16 represents exactly (small integers + halves).
	const srcF32 = new Float32Array(N_ROWS);
	srcF32[0] = 0.5; // → f16 0x3800
	srcF32[1] = -0.25; // → f16 0xb400
	srcF32[2] = 1.5; // → f16 0x3e00
	srcF32[3] = -3.0; // → f16 0xc200

	// Sentinel: each cell pre-loaded with a distinct decodable F16 so we can
	// detect (a) which cells were written and (b) whether untargeted cells
	// were preserved. F16 representation of 100 + i*0.5 is exact for i<200.
	const sentinelF32 = new Float32Array(N_CELLS);
	for (let i = 0; i < N_CELLS; i++) sentinelF32[i] = 100 + i * 0.5;
	const sentinelF16Bytes = new Uint8Array(N_CELLS * 2);
	const sv = new DataView(sentinelF16Bytes.buffer);
	for (let i = 0; i < N_CELLS; i++) {
		sv.setUint16(i * 2, f32ToF16Bits(sentinelF32[i]), true);
	}

	// CPU reference: take the sentinel state, then overwrite each indexed
	// cell with f32→f16(srcF32[row]). Compare the read-back F16 cells
	// (decoded to f32) against this.
	const expectedF16 = new Float32Array(N_CELLS);
	expectedF16.set(sentinelF32);
	for (let r = 0; r < N_ROWS; r++) {
		const cellIdx = Number(indices[r]);
		expectedF16[cellIdx] = f16BitsToF32(f32ToF16Bits(srcF32[r]));
	}

	// Indices buffer: I64 = 8 bytes per index. WGSL kernel reads only the
	// low 32 bits (idx_pair_off * 2u + 0). We zero-fill the high half.
	const indicesBytes = new Uint8Array(N_ROWS * 8);
	const iv = new DataView(indicesBytes.buffer);
	for (let r = 0; r < N_ROWS; r++) {
		iv.setUint32(r * 8, Number(indices[r]) >>> 0, true); // low half
		iv.setUint32(r * 8 + 4, 0, true); // high half
	}

	// Allocate buffers. In divert mode we co-locate src1 and dst into a
	// single buffer so dst.bufHandle === src[2].bufHandle (the structural
	// alias that ggml SET_ROWS always produces). The non-divert mode
	// keeps them in separate buffers.
	const srcF32Bytes = new Uint8Array(srcF32.buffer);
	const dstByteSize = N_CELLS * 2;
	const ALIGN = 256;
	const srcF32Aligned = (srcF32Bytes.byteLength + ALIGN - 1) & ~(ALIGN - 1);
	const indicesAligned = (indicesBytes.byteLength + ALIGN - 1) & ~(ALIGN - 1);

	let hSrc0: number;
	let hSrc1: number;
	let hDst: number;
	let dstOffset = 0;

	if (mode === "no-divert") {
		hSrc0 = runtime.dataManager.alloc(srcF32Bytes.byteLength);
		hSrc1 = runtime.dataManager.alloc(indicesBytes.byteLength);
		hDst = runtime.dataManager.alloc(dstByteSize);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc0).buffer,
			0,
			srcF32Bytes,
			0,
			srcF32Bytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			0,
			indicesBytes,
			0,
			indicesBytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hDst).buffer,
			0,
			sentinelF16Bytes,
			0,
			sentinelF16Bytes.byteLength,
		);
	} else {
		// Shared buffer holds [src0 | indices | dst] at aligned offsets.
		// Setting hDst = hSrc1 (with src1 at offset 0, dst at indicesAligned)
		// and passing hSrc1 as src[2] in the descriptor triggers the divert
		// path, mirroring the production case where v_view shares storage
		// with the cache_v_l buffer.
		hSrc0 = runtime.dataManager.alloc(srcF32Aligned);
		const sharedSize = indicesAligned + dstByteSize;
		hSrc1 = runtime.dataManager.alloc(sharedSize);
		hDst = hSrc1;
		dstOffset = indicesAligned;
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc0).buffer,
			0,
			srcF32Bytes,
			0,
			srcF32Bytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			0,
			indicesBytes,
			0,
			indicesBytes.byteLength,
		);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hSrc1).buffer,
			dstOffset,
			sentinelF16Bytes,
			0,
			sentinelF16Bytes.byteLength,
		);
	}

	// Pre-dispatch readback (sanity check: did the sentinel land?).
	const preStaging = runtime.device.createBuffer({
		size: dstByteSize,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	{
		const enc = runtime.device.createCommandEncoder();
		enc.copyBufferToBuffer(
			runtime.dataManager.get(hDst).buffer,
			dstOffset,
			preStaging,
			0,
			dstByteSize,
		);
		runtime.device.queue.submit([enc.finish()]);
	}
	await preStaging.mapAsync(GPUMapMode.READ, 0, dstByteSize);
	const preBytes = new Uint8Array(preStaging.getMappedRange().slice(0));
	preStaging.unmap();
	preStaging.destroy();
	const preF16 = new Float32Array(N_CELLS);
	{
		const dv = new DataView(preBytes.buffer);
		for (let i = 0; i < N_CELLS; i++) {
			preF16[i] = f16BitsToF32(dv.getUint16(i * 2, true));
		}
	}

	// SET_ROWS descriptor — V-cache transpose layout (ne[0]=1).
	// Per ggml semantics dst.bufHandle === src[2].bufHandle. We pass src[2]
	// as a descriptor-only entry so the divert path's structural-alias
	// detection fires.
	const desc: JsepOpDescriptor = {
		op: GGML_OP_SET_ROWS,
		nSrc: 3,
		dst: {
			bufHandle: hDst,
			offset: dstOffset,
			type: GGML_TYPE_F16,
			ne: [1, N_CELLS, 1, 1],
			nb: [2, 2, 2 * N_CELLS, 2 * N_CELLS],
		},
		srcs: [
			{
				bufHandle: hSrc0,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [1, N_ROWS, 1, 1],
				nb: [4, 4, 4 * N_ROWS, 4 * N_ROWS],
			},
			{
				bufHandle: hSrc1,
				offset: 0,
				type: GGML_TYPE_I64,
				ne: [N_ROWS, 1, 1, 1],
				nb: [8, 8 * N_ROWS, 8 * N_ROWS, 8 * N_ROWS],
			},
			{
				// src[2] = a, the destination buffer. Same handle as dst
				// (dst is view_tensor(a)) — this is what triggers divert.
				bufHandle: hDst,
				offset: dstOffset,
				type: GGML_TYPE_F16,
				ne: [1, N_CELLS, 1, 1],
				nb: [2, 2, 2 * N_CELLS, 2 * N_CELLS],
			},
		],
	};

	const status = dispatchSetRows(runtime, desc);
	runtime.encoderBatcher.flush();

	// Read back the dst F16 cells.
	const postStaging = runtime.device.createBuffer({
		size: dstByteSize,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	{
		const enc = runtime.device.createCommandEncoder();
		enc.copyBufferToBuffer(
			runtime.dataManager.get(hDst).buffer,
			dstOffset,
			postStaging,
			0,
			dstByteSize,
		);
		runtime.device.queue.submit([enc.finish()]);
	}
	await postStaging.mapAsync(GPUMapMode.READ, 0, dstByteSize);
	const postBytes = new Uint8Array(postStaging.getMappedRange().slice(0));
	postStaging.unmap();
	postStaging.destroy();
	const postF16 = new Float32Array(N_CELLS);
	{
		const dv = new DataView(postBytes.buffer);
		for (let i = 0; i < N_CELLS; i++) {
			postF16[i] = f16BitsToF32(dv.getUint16(i * 2, true));
		}
	}

	const targetSet = new Set<number>(indices.map((b) => Number(b)));
	let maxAbsDeltaTargeted = 0;
	let maxAbsDeltaUntargeted = 0;
	let hasNaN = false;
	let hasInf = false;
	for (let i = 0; i < N_CELLS; i++) {
		const v = postF16[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		const d = Math.abs(v - expectedF16[i]);
		if (targetSet.has(i)) {
			if (d > maxAbsDeltaTargeted) maxAbsDeltaTargeted = d;
		} else if (d > maxAbsDeltaUntargeted) maxAbsDeltaUntargeted = d;
	}

	const result: SetRowsSelfTestResult = {
		mode,
		status,
		N_CELLS,
		N_ROWS,
		indices: indices.map((b) => Number(b)),
		srcF32: Array.from(srcF32),
		preF16: Array.from(preF16),
		postF16: Array.from(postF16),
		expectedF16: Array.from(expectedF16),
		maxAbsDeltaTargeted,
		maxAbsDeltaUntargeted,
		hasNaN,
		hasInf,
	};

	runtime.dataManager.free(hSrc0);
	runtime.dataManager.free(hSrc1);
	if (mode === "no-divert") runtime.dataManager.free(hDst);

	return result;
}

// ----- RMS_NORM real-shape self-test ---------------------------------------
//
// Stage 3.5 Step 1: drive `dispatchRmsNorm` with a row whose column count
// matches TinyLlama's first model RMS_NORM (cols=2048). The kernel was
// written and validated only via the synthetic golden-test harness in
// Phase 2. If RMS_NORM produces zero (or NaN, or wrong-magnitude) output
// for non-zero input on real-model shapes, the residual stream collapses
// to zero on the first layer's first norm — fully consistent with the
// observed Outcome C all-zero logits.

interface MallocModule {
	_malloc: (n: number) => number;
	_free: (p: number) => void;
	HEAPU8: Uint8Array;
	HEAPF32: Float32Array;
}

async function runRmsNormSelfTest(
	mod: MallocModule,
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
): Promise<void> {
	const rows = 1;
	const cols = 2048;
	const eps = 1e-5;

	// Non-zero input pattern that exercises both signs and a non-trivial
	// magnitude. Mean(x²) is ~0.21 for this pattern, so inv_rms is near 2.18.
	const x = new Float32Array(rows * cols);
	for (let i = 0; i < x.length; i++) x[i] = ((i % 17) - 8) * 0.1;

	// CPU reference.
	let sumSq = 0;
	for (let i = 0; i < cols; i++) sumSq += x[i] * x[i];
	const invRms = 1 / Math.sqrt(sumSq / cols + eps);
	const reference = new Float32Array(cols);
	for (let i = 0; i < cols; i++) reference[i] = x[i] * invRms;

	const totalBytes = x.byteLength;
	const hX = runtime.dataManager.alloc(totalBytes);
	const hOut = runtime.dataManager.alloc(totalBytes);

	// Upload x into the JSEP-side input buffer.
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(hX).buffer,
		0,
		new Uint8Array(x.buffer),
		0,
		totalBytes,
	);

	// `dispatchRmsNorm` reads eps from `heapBuffer[opParamsPtr]` as f32, so
	// allocate 4 bytes in the wasm heap and write eps there. This mirrors
	// the production path where `op_params[0]` is a node field on the
	// wasm side.
	const opParamsPtr = mod._malloc(4);
	mod.HEAPF32[opParamsPtr >>> 2] = eps;

	const desc: JsepOpDescriptor = {
		op: GGML_OP_RMS_NORM,
		nSrc: 1,
		dst: {
			bufHandle: hOut,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [cols, rows, 1, 1],
			nb: [4, 4 * cols, 4 * cols * rows, 0],
		},
		srcs: [
			{
				bufHandle: hX,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [cols, rows, 1, 1],
				nb: [4, 4 * cols, 4 * cols * rows, 0],
			},
		],
	};

	const status = dispatchRmsNorm(
		{
			device: runtime.device,
			dataManager: runtime.dataManager,
			encoderBatcher: runtime.encoderBatcher,
			pipelineCache: runtime.pipelineCache,
			bindGroupLayoutCache: runtime.bindGroupLayoutCache,
		},
		desc,
		opParamsPtr,
		mod.HEAPU8.buffer,
	);
	runtime.encoderBatcher.flush();

	const recOut = runtime.dataManager.get(hOut);
	const staging = runtime.device.createBuffer({
		size: totalBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(recOut.buffer, 0, staging, 0, totalBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, totalBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDelta = 0;
	let hasNaN = false;
	let hasInf = false;
	let zeroCount = 0;
	for (let i = 0; i < cols; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		else if (v === 0) zeroCount++;
		const d = Math.abs(v - reference[i]);
		if (d > maxAbsDelta) maxAbsDelta = d;
	}

	const passLog = {
		status,
		invRms,
		first4Got: Array.from(got.slice(0, 4)),
		first4Ref: Array.from(reference.slice(0, 4)),
		last4Got: Array.from(got.slice(-4)),
		last4Ref: Array.from(reference.slice(-4)),
		maxAbsDelta,
		hasNaN,
		hasInf,
		zeroCount,
	};
	log(`RMSNORM_SELFTEST = ${JSON.stringify(passLog)}`);
	(window as any).__rmsNormSelfTest = passLog;

	mod._free(opParamsPtr);
	runtime.dataManager.free(hX);
	runtime.dataManager.free(hOut);
}

// Stage 4.3a: multi-row RMS_NORM self-test --------------------------------
//
// Existing RMSNORM_SELFTEST drives a single row (rows=1). Real prefill ops
// push rows=6 (the prompt batch) through this kernel. With workgroup_size
// (1, 256), gid.x indexes the row; a per-thread row-sum bug or row-stride
// bug would be invisible at rows=1 but produce cross-row contamination at
// rows=6. This selftest exercises both the non-divert and divert variants.
//
// Outcome triage:
//   A1 fails               → RMS_NORM kernel buggy at multi-row (gid.x indexing)
//   A1 passes, A2 fails    → divert path buggy (separate from matmul divert)
//   both pass              → RMS_NORM is fine; bug is downstream

interface RmsNormMultiRowResult {
	mode: "no-divert" | "divert";
	rows: number;
	cols: number;
	status: number;
	maxAbsDelta: number;
	hasNaN: boolean;
	hasInf: boolean;
	zeroCount: number;
	perRowMaxDelta: number[];
}

async function runRmsNormMultiRowSelfTest(
	mod: MallocModule,
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
	mode: "no-divert" | "divert",
): Promise<RmsNormMultiRowResult> {
	const rows = 6;
	const cols = 2048;
	const eps = 1e-5;

	// Distinct per-row patterns so a row-mix bug shows as cross-row leakage.
	const x = new Float32Array(rows * cols);
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			x[r * cols + c] = ((c % 17) - 8) * 0.1 + r * 0.01;
		}
	}

	// CPU reference: per-row inv_rms × x.
	const reference = new Float32Array(rows * cols);
	for (let r = 0; r < rows; r++) {
		let sumSq = 0;
		for (let c = 0; c < cols; c++) sumSq += x[r * cols + c] ** 2;
		const inv = 1 / Math.sqrt(sumSq / cols + eps);
		for (let c = 0; c < cols; c++) {
			reference[r * cols + c] = x[r * cols + c] * inv;
		}
	}

	const totalBytes = x.byteLength;
	let hX: number;
	let hOut: number;
	let xOffset = 0;
	let outOffset = 0;
	if (mode === "no-divert") {
		hX = runtime.dataManager.alloc(totalBytes);
		hOut = runtime.dataManager.alloc(totalBytes);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hX).buffer,
			0,
			new Uint8Array(x.buffer),
			0,
			totalBytes,
		);
	} else {
		// Shared buffer with x at offset 0, out at aligned offset past x.
		const ALIGN = 256;
		const xAligned = (totalBytes + ALIGN - 1) & ~(ALIGN - 1);
		const sharedBytes = xAligned + totalBytes;
		hX = runtime.dataManager.alloc(sharedBytes);
		hOut = hX;
		xOffset = 0;
		outOffset = xAligned;
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hX).buffer,
			0,
			new Uint8Array(x.buffer),
			0,
			totalBytes,
		);
		const sentinel = new Float32Array(totalBytes / 4);
		sentinel.fill(-7777);
		runtime.device.queue.writeBuffer(
			runtime.dataManager.get(hX).buffer,
			outOffset,
			new Uint8Array(sentinel.buffer),
			0,
			totalBytes,
		);
	}

	const opParamsPtr = mod._malloc(4);
	mod.HEAPF32[opParamsPtr >>> 2] = eps;

	const desc: JsepOpDescriptor = {
		op: GGML_OP_RMS_NORM,
		nSrc: 1,
		dst: {
			bufHandle: hOut,
			offset: outOffset,
			type: GGML_TYPE_F32,
			ne: [cols, rows, 1, 1],
			nb: [4, 4 * cols, 4 * cols * rows, 0],
		},
		srcs: [
			{
				bufHandle: hX,
				offset: xOffset,
				type: GGML_TYPE_F32,
				ne: [cols, rows, 1, 1],
				nb: [4, 4 * cols, 4 * cols * rows, 0],
			},
		],
	};

	const status = dispatchRmsNorm(
		{
			device: runtime.device,
			dataManager: runtime.dataManager,
			encoderBatcher: runtime.encoderBatcher,
			pipelineCache: runtime.pipelineCache,
			bindGroupLayoutCache: runtime.bindGroupLayoutCache,
		},
		desc,
		opParamsPtr,
		mod.HEAPU8.buffer,
	);
	runtime.encoderBatcher.flush();

	const outRec = runtime.dataManager.get(hOut);
	const staging = runtime.device.createBuffer({
		size: totalBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(outRec.buffer, outOffset, staging, 0, totalBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, totalBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDelta = 0;
	let hasNaN = false;
	let hasInf = false;
	let zeroCount = 0;
	const perRowMaxDelta = new Array<number>(rows).fill(0);
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const v = got[r * cols + c];
			if (Number.isNaN(v)) hasNaN = true;
			else if (!Number.isFinite(v)) hasInf = true;
			else if (v === 0) zeroCount++;
			const d = Math.abs(v - reference[r * cols + c]);
			if (d > maxAbsDelta) maxAbsDelta = d;
			if (d > perRowMaxDelta[r]) perRowMaxDelta[r] = d;
		}
	}

	mod._free(opParamsPtr);
	runtime.dataManager.free(hX);
	if (mode === "no-divert") runtime.dataManager.free(hOut);

	return {
		mode,
		rows,
		cols,
		status,
		maxAbsDelta,
		hasNaN,
		hasInf,
		zeroCount,
		perRowMaxDelta,
	};
}

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	console.log(msg);
}

async function runSpike(): Promise<void> {
	try {
		log("[1/8] Initializing JSEP WASM module...");
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm-jsep.js")).default;
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});

		log("[2/8] Acquiring WebGPU device...");
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) {
			log("no WebGPU adapter", "fail");
			return;
		}
		const device = await adapter.requestDevice();
		// Stage 4.2 — capture WebGPU validation/internal errors so silent
		// dispatch failures show up in the log instead of surfacing only
		// as zero/NaN dst.
		const gpuErrLog: Array<{ kind: string; msg: string }> = [];
		device.addEventListener("uncapturederror", (ev) => {
			const e = ev as unknown as { error: GPUError };
			const kind =
				e.error instanceof GPUValidationError
					? "validation"
					: e.error instanceof GPUOutOfMemoryError
						? "oom"
						: "internal";
			gpuErrLog.push({ kind, msg: e.error.message });
		});
		(window as any).__gpuErrLog = gpuErrLog;

		// Stage 4.15 — gate per-divert-dispatch readback in
		// `dispatchMatmul` (src/inference/jsep/ops/matmul.ts). Must be set
		// before the first MUL_MAT divert fires (model load + decode).
		// Self-capped at 32 entries; deferred mapAsync drains during the
		// post-DONE inspection window.
		(globalThis as any).__stage415DivertProbe = true;
		(globalThis as any).__stage415DivertLog = [];

		// Stage 4.8 — initialize dispatchSetRows entry/exit log before
		// installJsepCallbacks runs so the warmup + every selftest +
		// production graph dispatch is captured.
		const stage48SetRowsLog: Array<{ phase: string; data?: unknown }> = [];
		(globalThis as any).__stage48SetRowsLog = stage48SetRowsLog;
		(window as any).__stage48SetRowsLog = stage48SetRowsLog;

		// Stage 4.8 — temp-dst capture state (buffers allocated post-device).
		const stage48Captures = {
			preKernelFirst8U16: null as number[] | null,
			postKernelFirst8U16: null as number[] | null,
			postCopyBackFirst8U16: null as number[] | null,
			src0AtKernelTimeF32: null as number[] | null,
			err: null as string | null,
		};
		(window as any).__stage48Captures = stage48Captures;

		// Stage 4.8 — capture every console.error into a buffer so we
		// don't miss dispatchSetRows validation failures past CDP's
		// console-buffer cap.
		const consoleErrors: string[] = [];
		const origConsoleError = console.error.bind(console);
		console.error = (...args: unknown[]) => {
			try {
				consoleErrors.push(args.map((a) => String(a)).join(" "));
			} catch {
				/* swallow */
			}
			origConsoleError(...args);
		};
		(window as any).__consoleErrors = consoleErrors;

		// Stage 4.9 diagnostic — H1-inverse host_mirror peek. Capture first
		// 16 bytes / 8 F32 of host_mirror at hostPtr for the distinctive
		// (handle=26, offset=0, size=6144) signature: this is i=3 SET_ROWS
		// src0 (K-projection-after-ROPE, F32 [256,6,1,1]) under H1-inverse.
		// If first8F32 is all zeros, host_mirror is itself stale at H1-inverse
		// time → CPU op chain didn't write h26o0 yet. If non-zero, then GPU
		// writeBuffer is failing to land before the divert hook reads.
		const h1invDiag = {
			callIdx: 0,
			match: { handle: 26, offset: 0, size: 6144 },
			captures: [] as Array<{
				callIdx: number;
				handle: number;
				offset: number;
				size: number;
				first16: number[];
				first8F32: number[];
			}>,
		};
		(globalThis as any).__h1invDiag = h1invDiag;
		(window as any).__h1invDiag = h1invDiag;

		log("[3/8] Installing JSEP callbacks (must precede webllm_load_model)...");
		const runtime = installJsepCallbacks(mod, device);

		// Stage 3 self-test: hand-crafted Q4_K matmul against a CPU reference
		// to localize whether the all-zeros logits chain comes from the Q4_K
		// kernel or from elsewhere (RMS_NORM, splits, CPY).
		await runQ4KSelfTest(runtime);

		// Stage 3.5 Step 1 self-test: real-model RMS_NORM shape (cols=2048,
		// matches TinyLlama first norm). Disambiguates whether the residual
		// stream is being zeroed at the first norm vs further downstream.
		await runRmsNormSelfTest(mod as MallocModule, runtime);

		// Stage 4.3a self-tests: production-shape multi-row RMS_NORM and
		// production-shape Q4_K MUL_MAT, in both non-divert and divert
		// configurations. The post-Stage-4.2 baseline confirmed Bug A
		// (POSTPREFILL_BUF11 = canonical NaN at MUL_MAT dst offsets); these
		// selftests localize whether the bug is in the kernel or the
		// divert path.
		const rmsMulti1 = await runRmsNormMultiRowSelfTest(
			mod as MallocModule,
			runtime,
			"no-divert",
		);
		log(`RMSNORM_MULTIROW_NODIVERT = ${JSON.stringify(rmsMulti1)}`);
		(window as any).__rmsNormMultiNoDivert = rmsMulti1;

		const rmsMulti2 = await runRmsNormMultiRowSelfTest(
			mod as MallocModule,
			runtime,
			"divert",
		);
		log(`RMSNORM_MULTIROW_DIVERT = ${JSON.stringify(rmsMulti2)}`);
		(window as any).__rmsNormMultiDivert = rmsMulti2;

		const matProd1 = await runMatmulProductionSelfTest(runtime, "no-divert");
		log(`MATMUL_PROD_NODIVERT = ${JSON.stringify(matProd1)}`);
		(window as any).__matmulProdNoDivert = matProd1;

		const matProd2 = await runMatmulProductionSelfTest(runtime, "divert");
		log(`MATMUL_PROD_DIVERT = ${JSON.stringify(matProd2)}`);
		(window as any).__matmulProdDivert = matProd2;

		// Stage 4.18 Probe 8a: per-shape Q4_0 matmul precision sweep.
		// All 5 production shapes used in TinyLlama-Q4_0. Reports delta
		// vs both f64 ground truth and f32 element-wise loop reference;
		// the second tells us whether JSEP matches a CPU f32 single-pass
		// loop bit-for-bit (i.e. kernel is mathematically equivalent and
		// the precision is purely an f32-non-associativity feature, not
		// a bug).
		const q4_0SweepShapes: Array<[number, number, number, string]> = [
			[2048, 2048, 6, "q-out-proj"],
			[256, 2048, 6, "k-v-proj"],
			[5632, 2048, 6, "ffn-gate-up"],
			[2048, 5632, 6, "ffn-down"],
			[32000, 2048, 1, "lm-head"],
		];
		const q4_0SweepResults: Array<MatmulQ4_0SweepResult & { tag: string }> =
			[];
		for (const [M, K, N, tag] of q4_0SweepShapes) {
			const r = await runMatmulQ4_0Sweep(runtime, M, K, N);
			const tagged = { tag, ...r };
			q4_0SweepResults.push(tagged);
			log(`MATMUL_Q4_0_SWEEP[${tag}] = ${JSON.stringify(tagged)}`);
		}
		(window as any).__matmulQ4_0Sweep = q4_0SweepResults;

		// Stage 4.6 D1: SET_ROWS V-cache transpose self-test. Drives the
		// F32→F16 atomic-CAS path with adjacent indices that share u32
		// words ({0,1} and {6,7}); CPU reference is the sentinel state
		// with target cells overwritten by f32→f16(srcF32). PASS proves
		// the kernel is correct; FAIL points at CAS encoding,
		// indices/strides, or divert pre-copy.
		const setRows1 = await runSetRowsVCacheSelfTest(runtime, "no-divert");
		log(`SETROWS_VCACHE_NODIVERT = ${JSON.stringify(setRows1)}`);
		(window as any).__setRowsNoDivert = setRows1;

		const setRows2 = await runSetRowsVCacheSelfTest(runtime, "divert");
		log(`SETROWS_VCACHE_DIVERT = ${JSON.stringify(setRows2)}`);
		(window as any).__setRowsDivert = setRows2;

		log("[4/8] Initializing ggml-webgpu backend...");
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}

		log(`[5/8] Fetching GGUF from ${GGUF_URL}...`);
		const resp = await fetch(GGUF_URL);
		if (!resp.ok) {
			log(`fetch failed: ${resp.status} ${resp.statusText}`, "fail");
			return;
		}
		const buf = new Uint8Array(await resp.arrayBuffer());
		log(`     loaded ${(buf.byteLength / 1024 / 1024).toFixed(1)} MiB`);

		log("[6/8] Loading model + creating context...");
		const bridge = createLlamaBridge(mod);

		// Stage 4.20 Probe 9b — arm the JSEP set_tensor weight-hash probe
		// before loadModel so it captures FNV-1a-32 of the bytes set_tensor
		// sees for blk.0.attn_q.weight + blk.0.attn_k.weight. Compared
		// downstream against an independent JS-side hash of the same
		// tensors parsed out of `buf` directly. Probe is a no-op outside
		// the JSEP build (the export only exists when WEBLLM_BACKEND_JSEP
		// is on); guard with `?.` so the spike still loads under default
		// builds without throwing.
		(globalThis as any).__weightHashLog = [];
		const setWeightHashProbe = (mod as any)._ggml_jsep_set_weight_hash_probe as
			| ((enable: number) => void)
			| undefined;
		if (typeof setWeightHashProbe === "function") {
			setWeightHashProbe(1);
			log("     [probe9b] weight-hash probe armed");
		} else {
			log("     [probe9b] weight-hash export missing — non-JSEP build?", "fail");
		}

		// Stage 4.29 Probe 16 — arm the CPU-side set_tensor weight-hash
		// probe alongside the JSEP one. Stage 4.28 (P-15-jsep-bypass)
		// confirmed `blk.0.ffn_norm.weight` (F32) and `blk.0.ffn_down.weight`
		// (Q6_K) bypass the JSEP set_tensor hook entirely — they live on
		// the CPU buft, so we need a parallel CPU-side probe to test
		// suspect 2 (gain-vector mis-load). The export ships in every
		// build (CPU backend is unconditional), unlike the JSEP probe.
		(globalThis as any).__cpuWeightHashLog = [];
		const setCpuWeightHashProbe = (mod as any)._ggml_cpu_set_weight_hash_probe as
			| ((enable: number) => void)
			| undefined;
		if (typeof setCpuWeightHashProbe === "function") {
			setCpuWeightHashProbe(1);
			log("     [probe16] CPU weight-hash probe armed");
		} else {
			log("     [probe16] CPU weight-hash export missing — old build?", "fail");
		}

		const t0 = performance.now();
		const model = await bridge.loadModel(buf);
		const tLoadMs = performance.now() - t0;
		const vocab = bridge.nVocab(model);
		log(`     model loaded in ${tLoadMs.toFixed(0)} ms; vocab = ${vocab}`);

		// Stage 4.20 Probe 9b — disarm probe + emit JS-side reference hash
		// for the same two tensors, parsed independently out of `buf`. Any
		// mismatch ⇒ Outcome E (corruption between GGUF parser and
		// set_tensor — fix lives in libllama). Match ⇒ Outcome F (the
		// upload preserves bytes; investigate kernel or GPU upload via a
		// follow-on probe).
		if (typeof setWeightHashProbe === "function") {
			setWeightHashProbe(0);
		}
		// Stage 4.29 Probe 16 — disarm CPU probe.
		if (typeof setCpuWeightHashProbe === "function") {
			setCpuWeightHashProbe(0);
		}
		try {
			const gguf = GgufParser.parse(buf);
			// Stage 4.28 Probe 15 — extended from 2 weights (Stage 4.20) to
			// 7 to localize the cascade source after Stage 4.27 confirmed
			// `attn_out-0` Δ=4.77e-3 → `ffn_norm-0` Δ=0.183 reproduces
			// bit-for-bit. `ffn_norm.weight` is highest-prior (38× jump
			// suggests gain-vector mis-load); `attn_output.weight` is the
			// other primary suspect (never directly probed). The three
			// FFN matmul weights round out the layer-0 weight surface so
			// any FFN-block upload bug is caught simultaneously.
			const targetNames = [
				"blk.0.attn_q.weight",
				"blk.0.attn_k.weight",
				"blk.0.attn_output.weight",
				"blk.0.ffn_norm.weight",
				"blk.0.ffn_gate.weight",
				"blk.0.ffn_up.weight",
				"blk.0.ffn_down.weight",
			];
			// GgufParser.tensors entries don't carry an explicit byte size;
			// compute it from element count × per-element bytes-per-element.
			// For Q4_0 (type=2) blocks are 32 elements in 18 bytes ⇒ 18/32
			// bytes per element. Matches GgufParser.calculateTotalDataSize
			// internal logic.
			const elemBytes = (type: number): number => {
				switch (type) {
					case 0: return 4;          // F32
					case 1: return 2;          // F16
					case 2: return 18 / 32;    // Q4_0
					case 8: return 34 / 32;    // Q8_0
					case 12: return 144 / 256; // Q4_K
					case 14: return 210 / 256; // Q6_K
				}
				throw new Error(`probe9b: unsupported tensor type ${type}`);
			};
			const ref: Record<string, { fnv1a: number; offset: number; size: number }> = {};
			for (const name of targetNames) {
				const t = gguf.tensors.find((ti) => ti.name === name);
				if (!t) {
					log(`     [probe9b] tensor ${name} not in GGUF metadata`, "fail");
					continue;
				}
				const elemCount = t.dimensions.reduce((a, b) => a * b, 1);
				const tSize = Math.round(elemCount * elemBytes(t.type));
				const start = gguf.dataOffset + t.offset;
				const end = start + tSize;
				if (end > buf.byteLength) {
					log(`     [probe9b] tensor ${name} extends past buf end`, "fail");
					continue;
				}
				let h = 2166136261 >>> 0;
				for (let i = start; i < end; ++i) {
					h ^= buf[i];
					h = Math.imul(h, 16777619) >>> 0;
				}
				ref[name] = { fnv1a: h >>> 0, offset: t.offset, size: tSize };
			}
			(globalThis as any).__weightHashRef = ref;

			const log4 = (globalThis as any).__weightHashLog as Array<{
				name: string;
				bufHandle: number;
				offset: number;
				size: number;
				fnv1a_pre: number;
			}>;
			const verdict: Array<{
				name: string;
				match: boolean;
				fnv1a_pre: string;
				fnv1a_ref: string;
				size_pre: number;
				size_ref: number;
			}> = [];
			for (const name of targetNames) {
				const pre = log4.find((e) => e.name === name);
				const r = ref[name];
				if (!pre || !r) {
					verdict.push({
						name,
						match: false,
						fnv1a_pre: pre ? `0x${pre.fnv1a_pre.toString(16).padStart(8, "0")}` : "<missing>",
						fnv1a_ref: r ? `0x${r.fnv1a.toString(16).padStart(8, "0")}` : "<missing>",
						size_pre: pre?.size ?? -1,
						size_ref: r?.size ?? -1,
					});
					continue;
				}
				const match = (pre.fnv1a_pre >>> 0) === (r.fnv1a >>> 0) && pre.size === r.size;
				verdict.push({
					name,
					match,
					fnv1a_pre: `0x${(pre.fnv1a_pre >>> 0).toString(16).padStart(8, "0")}`,
					fnv1a_ref: `0x${(r.fnv1a >>> 0).toString(16).padStart(8, "0")}`,
					size_pre: pre.size,
					size_ref: r.size,
				});
			}
			(globalThis as any).__weightHashVerdict = verdict;
			for (const v of verdict) {
				log(
					`     [probe9b] ${v.name}: pre=${v.fnv1a_pre} ref=${v.fnv1a_ref} ` +
						`size_pre=${v.size_pre} size_ref=${v.size_ref} match=${v.match}`,
					v.match ? "ok" : "fail",
				);
			}
			const allMatch = verdict.length === targetNames.length && verdict.every((v) => v.match);
			log(
				`     [probe9b] OUTCOME: ${allMatch ? "F (hashes match — upload preserves bytes)" : "E (hash mismatch — upload corruption)"}`,
				allMatch ? "ok" : "fail",
			);
		} catch (err) {
			log(`     [probe9b] ref-hash computation threw: ${(err as Error).message}`, "fail");
		}

		// Stage 4.21 Probe 9c — GPU-side post-upload mapAsync readback hash.
		// Walk __weightHashLog and, for each entry, compute FNV-1a-32 over
		// the bytes that actually live on the GPU buffer at (bufHandle,
		// offset, size). Compare against entry.fnv1a_pre (set_tensor's
		// pre-upload host-side hash from Stage 4.20). Match ⇒ Outcome F-1
		// (upload through GPU is bit-clean — kernel re-investigation in
		// production conditions). Differ ⇒ Outcome F-2 (host→GPU corruption
		// in the writeBuffer path; bisect Module.jsepWrite ➜
		// device.queue.writeBuffer).
		try {
			const log4c = (globalThis as any).__weightHashLog as Array<{
				name: string;
				bufHandle: number;
				offset: number;
				size: number;
				fnv1a_pre: number;
			}>;
			const verdict9c: Array<{
				name: string;
				match: boolean;
				fnv1a_pre: string;
				fnv1a_gpu: string;
				size: number;
			}> = [];
			for (const entry of log4c) {
				const rec = runtime.dataManager.get(entry.bufHandle);
				if (entry.offset + entry.size > rec.size) {
					log(
						`     [probe9c] ${entry.name}: range out of bounds ` +
							`(off=${entry.offset} size=${entry.size} buf_size=${rec.size})`,
						"fail",
					);
					continue;
				}
				const staging = runtime.device.createBuffer({
					size: entry.size,
					usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
				});
				const enc = runtime.device.createCommandEncoder();
				enc.copyBufferToBuffer(rec.buffer, entry.offset, staging, 0, entry.size);
				runtime.device.queue.submit([enc.finish()]);
				await staging.mapAsync(GPUMapMode.READ, 0, entry.size);
				const mapped = new Uint8Array(staging.getMappedRange(0, entry.size));
				let h = 2166136261 >>> 0;
				for (let i = 0; i < entry.size; ++i) {
					h ^= mapped[i];
					h = Math.imul(h, 16777619) >>> 0;
				}
				staging.unmap();
				staging.destroy();
				const match = (h >>> 0) === (entry.fnv1a_pre >>> 0);
				verdict9c.push({
					name: entry.name,
					match,
					fnv1a_pre: `0x${(entry.fnv1a_pre >>> 0).toString(16).padStart(8, "0")}`,
					fnv1a_gpu: `0x${(h >>> 0).toString(16).padStart(8, "0")}`,
					size: entry.size,
				});
			}
			(globalThis as any).__weightHashGpuVerdict = verdict9c;
			for (const v of verdict9c) {
				log(
					`     [probe9c] ${v.name}: pre=${v.fnv1a_pre} gpu=${v.fnv1a_gpu} ` +
						`size=${v.size} match=${v.match}`,
					v.match ? "ok" : "fail",
				);
			}
			const allMatch9c =
				verdict9c.length > 0 && verdict9c.every((v) => v.match);
			log(
				`     [probe9c] OUTCOME: ${allMatch9c ? "F-1 (GPU bytes match — upload chain bit-clean; kernel re-investigation)" : "F-2 (GPU bytes differ — host→GPU corruption in writeBuffer path)"}`,
				allMatch9c ? "ok" : "fail",
			);

			// Stage 4.28 Probe 15 — synthesize per-weight verdict across the
			// extended 7-tensor allowlist into a single P-15-* outcome line.
			// Each weight's verdict line is in the format the brief specifies:
			//   [STAGE-4.28] <name> ref_hash=0x<H1> gpu_readback_hash=0x<H2> match=<bool>
			// where ref_hash is the JS-side GgufParser hash (probe9b.ref) and
			// gpu_readback_hash is the GPU mapAsync readback hash (probe9c).
			// "match" is the AND of (ref==pre) AND (pre==gpu) — i.e., the
			// full chain GGUF→set_tensor→GPU is bit-clean for that weight.
			const stage428Names = [
				"blk.0.attn_q.weight",
				"blk.0.attn_k.weight",
				"blk.0.attn_output.weight",
				"blk.0.ffn_norm.weight",
				"blk.0.ffn_gate.weight",
				"blk.0.ffn_up.weight",
				"blk.0.ffn_down.weight",
			];
			const refMap = (globalThis as any).__weightHashRef as
				| Record<string, { fnv1a: number; offset: number; size: number }>
				| undefined;
			const gpuMap: Record<string, { fnv1a_pre: string; fnv1a_gpu: string; match: boolean }> = {};
			for (const v of verdict9c) {
				gpuMap[v.name] = { fnv1a_pre: v.fnv1a_pre, fnv1a_gpu: v.fnv1a_gpu, match: v.match };
			}
			const stage428Lines: string[] = [];
			let firstMismatch: string | null = null;
			for (const name of stage428Names) {
				const r = refMap?.[name];
				const g = gpuMap[name];
				const refHex = r ? `0x${(r.fnv1a >>> 0).toString(16).padStart(8, "0")}` : "<missing>";
				const gpuHex = g ? g.fnv1a_gpu : "<missing>";
				const refMatchesPre = !!(r && g && refHex === g.fnv1a_pre);
				const fullMatch = !!(r && g && g.match && refMatchesPre);
				stage428Lines.push(
					`[STAGE-4.28] ${name} ref_hash=${refHex} gpu_readback_hash=${gpuHex} match=${fullMatch}`,
				);
				if (!fullMatch && firstMismatch === null) {
					firstMismatch = name;
				}
			}
			(globalThis as any).__stage428Lines = stage428Lines;
			(globalThis as any).__stage428FirstMismatch = firstMismatch;
			for (const line of stage428Lines) {
				log(`     ${line}`, line.endsWith("match=true") ? "ok" : "fail");
			}
			let outcome: string;
			if (firstMismatch === null) {
				outcome = "P-15-clean (all 7 weights byte-exact end-to-end; suspects 1+2 close; pivot to suspect 3 first8-window blindness on kqv_out-0)";
			} else if (firstMismatch === "blk.0.ffn_norm.weight") {
				outcome = "P-15-gain (ffn_norm.weight gain-vector mis-load CONFIRMED; trace upload path)";
			} else if (firstMismatch === "blk.0.attn_output.weight") {
				outcome = "P-15-output-proj (attn_output.weight mis-uploaded despite attn_q/k bit-clean; diff upload call sites)";
			} else if (firstMismatch.startsWith("blk.0.ffn_")) {
				outcome = `P-15-ffn (${firstMismatch} mis-uploaded; deep-dive FFN-block upload path)`;
			} else {
				outcome = `P-15-other (${firstMismatch} first mismatch — unexpected; revisit upload path for that tensor)`;
			}
			log(`     [STAGE-4.28] OUTCOME: ${outcome}`, firstMismatch === null ? "ok" : "fail");

			// Stage 4.29 Probe 16 — unify the JSEP and CPU set_tensor logs
			// into a single per-weight verdict line. For each of the 7
			// targetNames, look up:
			//   ref_hash      = JS-side GgufParser FNV-1a-32 over the same
			//                   bytes (already in __weightHashRef from
			//                   Stage 4.20/4.28).
			//   jsep_pre_hash = JSEP set_tensor pre-upload hash, if the
			//                   tensor was JSEP-resident (refMap from
			//                   __weightHashLog).
			//   jsep_gpu_hash = post-writeBuffer GPU readback hash, if
			//                   JSEP-resident (gpuMap from probe9c).
			//   cpu_pre_hash  = CPU set_tensor post-memcpy hash, if the
			//                   tensor was CPU-buft-resident
			//                   (__cpuWeightHashLog from this stage).
			// match = at least one captured hash matches ref AND every
			// captured hash matches ref. A weight that appears in neither
			// log emits match=false with all captured hashes "<missing>"
			// — that's the P-16-silent escalation case.
			const stage429Names = stage428Names; // identical 7-name allowlist
			const cpuLog = (globalThis as any).__cpuWeightHashLog as Array<{
				name: string;
				offset: number;
				size: number;
				fnv1a_pre: number;
			}>;
			const cpuMap: Record<string, { fnv1a_pre: string; size: number }> = {};
			for (const e of cpuLog) {
				cpuMap[e.name] = {
					fnv1a_pre: `0x${(e.fnv1a_pre >>> 0).toString(16).padStart(8, "0")}`,
					size: e.size,
				};
			}
			const stage429Lines: string[] = [];
			let firstMismatch429: string | null = null;
			let cpuFiredCount = 0;
			let cpuByteCleanCount = 0;
			let cpuDirtyNames: string[] = [];
			for (const name of stage429Names) {
				const r = refMap?.[name];
				const jsepPre = (globalThis as any).__weightHashLog
					.find((e: { name: string; fnv1a_pre: number }) => e.name === name) as
					| { name: string; fnv1a_pre: number }
					| undefined;
				const jsepPreHex = jsepPre
					? `0x${(jsepPre.fnv1a_pre >>> 0).toString(16).padStart(8, "0")}`
					: "<missing>";
				const jsepGpuHex = gpuMap[name]?.fnv1a_gpu ?? "<missing>";
				const cpuPreHex = cpuMap[name]?.fnv1a_pre ?? "<missing>";
				const refHex = r ? `0x${(r.fnv1a >>> 0).toString(16).padStart(8, "0")}` : "<missing>";
				if (cpuMap[name]) {
					cpuFiredCount += 1;
					if (cpuPreHex === refHex) {
						cpuByteCleanCount += 1;
					} else {
						cpuDirtyNames.push(name);
					}
				}
				const captured = [jsepPreHex, jsepGpuHex, cpuPreHex].filter((h) => h !== "<missing>");
				const anyCaptured = captured.length > 0;
				const allMatchRef = anyCaptured && captured.every((h) => h === refHex);
				const match = !!r && anyCaptured && allMatchRef;
				stage429Lines.push(
					`[STAGE-4.29] ${name} ref_hash=${refHex} ` +
						`jsep_pre_hash=${jsepPreHex} jsep_gpu_hash=${jsepGpuHex} ` +
						`cpu_pre_hash=${cpuPreHex} match=${match}`,
				);
				if (!match && firstMismatch429 === null) {
					firstMismatch429 = name;
				}
			}
			(globalThis as any).__stage429Lines = stage429Lines;
			(globalThis as any).__stage429FirstMismatch = firstMismatch429;
			(globalThis as any).__stage429CpuFiredCount = cpuFiredCount;
			(globalThis as any).__stage429CpuByteCleanCount = cpuByteCleanCount;
			(globalThis as any).__stage429CpuDirtyNames = cpuDirtyNames;
			for (const line of stage429Lines) {
				log(`     ${line}`, line.endsWith("match=true") ? "ok" : "fail");
			}
			log(
				`     [STAGE-4.29] CPU hook fired on ${cpuFiredCount}/${stage429Names.length} weights ` +
					`(${cpuByteCleanCount} byte-clean, ${cpuDirtyNames.length} dirty)`,
				cpuFiredCount > 0 ? "ok" : "fail",
			);
			let outcome429: string;
			if (cpuFiredCount === 0) {
				// CPU hook fired zero times — the bypass weights don't live
				// on CPU buft either. Brief's risk-register #1 escalation
				// path: instrument untargeted set_tensor logging to identify
				// which buft owns ffn_norm / ffn_down, or fall back to
				// Shape B (cb_eval weight-tap).
				outcome429 = "P-16-silent (CPU hook never fired — bypass weights are not on CPU buft; escalate to untargeted set_tensor logging or Shape B)";
			} else if (firstMismatch429 === null) {
				// All 7 weights byte-clean across whichever buft owns each
				// (5 JSEP + 2 CPU, or some other split). Suspect 2
				// (ffn_norm.weight gain-vector mis-load) is dead.
				outcome429 = "P-16-clean (all 7 layer-0 weights byte-exact across JSEP + CPU bufts; suspect 2 gain-vector mis-load DEAD; pivot Stage 4.30 to suspect 3 first8-window blindness on kqv_out-0)";
			} else if (firstMismatch429 === "blk.0.ffn_norm.weight") {
				// CPU pre-upload hash differs from GGUF reference — the
				// gain vector was corrupted before reaching the CPU op.
				outcome429 = "P-16-gain (ffn_norm.weight CPU buft hash mismatch — gain-vector mis-load CONFIRMED; trace upload byte trajectory in Stage 4.30)";
			} else if (firstMismatch429 === "blk.0.ffn_down.weight") {
				// Q6_K weight upload path corruption. The Q6_K-specific
				// upload + dispatch path is the suspect.
				outcome429 = "P-16-ffn-down (ffn_down.weight CPU buft hash mismatch — Q6_K upload path corruption; deep-dive Q6_K-specific path in Stage 4.30)";
			} else {
				outcome429 = `P-16-other (${firstMismatch429} first mismatch — unexpected; revisit upload path for that tensor)`;
			}
			log(
				`     [STAGE-4.29] OUTCOME: ${outcome429}`,
				firstMismatch429 === null && cpuFiredCount > 0 ? "ok" : "fail",
			);
		} catch (err) {
			log(`     [probe9c] GPU-readback hash threw: ${(err as Error).message}`, "fail");
		}

		// Stage 4.30 Probe 17 — post-load tensor->data byte-hash peek.
		//
		// Stage 4.29 (Outcome P-16-silent) confirmed neither the JSEP
		// set_tensor hook (5/7 fire) nor the default CPU set_tensor hook
		// (0/7 fire) ever touches blk.0.ffn_norm.weight or
		// blk.0.ffn_down.weight during model load. With GGML_CPU=OFF in
		// the JSEP build, the most plausible owner is the GGUF mmap-
		// direct host buft — tensor->data points straight into MEMFS
		// bytes with no upload step. This block reads tensor->data
		// post-loadModel via the new `webllm_get_tensor_data_hash`
		// export and FNV-1a-32-hashes ggml_nbytes(t) bytes for each of
		// the 7 layer-0 weights, then compares against the JS-side
		// GgufParser reference already computed into __weightHashRef in
		// Stage 4.20.
		//
		// CLEAN ⇒ suspect 2 (ffn_norm.weight gain-vector mis-load) DEAD
		//          by direct measurement; pivot Stage 4.31 to suspect 3
		//          (kqv_out-0 first8-window blindness) or upstream
		//          cascade source at Qcur-0 inputs.
		// DIRTY ⇒ gain-vector mis-load CONFIRMED via a non-set_tensor
		//          pathway; Stage 4.31 traces the buggy upload byte
		//          trajectory back to mmap → CPU-op-read corruption.
		try {
			const stage430Names = [
				"blk.0.attn_q.weight",
				"blk.0.attn_k.weight",
				"blk.0.attn_output.weight",
				"blk.0.ffn_norm.weight",
				"blk.0.ffn_gate.weight",
				"blk.0.ffn_up.weight",
				"blk.0.ffn_down.weight",
			];
			const refMap430 = (globalThis as any).__weightHashRef as
				| Record<string, { fnv1a: number; offset: number; size: number }>
				| undefined;
			const getHash = (mod as any)._webllm_get_tensor_data_hash as
				| ((m: number, namePtr: number, outSizePtr: number) => number)
				| undefined;
			const stage430Lines: string[] = [];
			let firstMismatch430: string | null = null;
			let peekFiredCount = 0;
			let peekByteCleanCount = 0;
			const peekDirtyNames: string[] = [];
			if (typeof getHash !== "function") {
				log(
					"     [STAGE-4.30] _webllm_get_tensor_data_hash export missing — old build?",
					"fail",
				);
			} else {
				const mAny = mod as {
					_malloc: (n: number) => number;
					_free: (p: number) => void;
					stringToUTF8: (s: string, ptr: number, max: number) => void;
					lengthBytesUTF8: (s: string) => number;
					HEAPU8: Uint8Array;
				};
				for (const name of stage430Names) {
					const nameLen = mAny.lengthBytesUTF8(name) + 1;
					const namePtr = mAny._malloc(nameLen);
					const outSizePtr = mAny._malloc(4);
					if (!namePtr || !outSizePtr) {
						log(
							`     [STAGE-4.30] _malloc failed (namePtr=${namePtr} outSizePtr=${outSizePtr})`,
							"fail",
						);
						if (namePtr) mAny._free(namePtr);
						if (outSizePtr) mAny._free(outSizePtr);
						continue;
					}
					mAny.stringToUTF8(name, namePtr, nameLen);
					// Re-derive the U32 view AFTER each malloc — heap-grow
					// detaches prior ArrayBuffers; mAny.HEAPU8 is re-bound
					// by Emscripten on growth.
					new Uint32Array(mAny.HEAPU8.buffer)[outSizePtr >>> 2] = 0;
					const h = getHash(model, namePtr, outSizePtr) >>> 0;
					const sz =
						new Uint32Array(mAny.HEAPU8.buffer)[outSizePtr >>> 2] >>> 0;
					mAny._free(namePtr);
					mAny._free(outSizePtr);
					const r = refMap430?.[name];
					const refHex = r
						? `0x${(r.fnv1a >>> 0).toString(16).padStart(8, "0")}`
						: "<missing>";
					const exportFired = !(sz === 0 && h === 0);
					const dataHex = exportFired
						? `0x${h.toString(16).padStart(8, "0")}`
						: "<missing>";
					const sizeMatch = !!r && sz === r.size;
					const hashMatch = !!r && (h >>> 0) === (r.fnv1a >>> 0);
					const match = exportFired && sizeMatch && hashMatch;
					if (exportFired) {
						peekFiredCount += 1;
						if (match) peekByteCleanCount += 1;
						else peekDirtyNames.push(name);
					}
					stage430Lines.push(
						`[STAGE-4.30] ${name} ref=${refHex} data_peek=${dataHex} ` +
							`size_data=${sz} size_ref=${r?.size ?? -1} match=${match}`,
					);
					if (!match && firstMismatch430 === null) {
						firstMismatch430 = name;
					}
				}
			}
			(globalThis as any).__stage430Lines = stage430Lines;
			(globalThis as any).__stage430FirstMismatch = firstMismatch430;
			(globalThis as any).__stage430PeekFiredCount = peekFiredCount;
			(globalThis as any).__stage430PeekByteCleanCount = peekByteCleanCount;
			(globalThis as any).__stage430PeekDirtyNames = peekDirtyNames;
			for (const line of stage430Lines) {
				log(`     ${line}`, line.endsWith("match=true") ? "ok" : "fail");
			}
			log(
				`     [STAGE-4.30] tensor->data peek fired on ${peekFiredCount}/${stage430Names.length} weights ` +
					`(${peekByteCleanCount} byte-clean, ${peekDirtyNames.length} dirty)`,
				peekFiredCount > 0 ? "ok" : "fail",
			);
			let outcome430: string;
			if (peekFiredCount === 0) {
				outcome430 =
					"P-17-other (export never returned non-zero — model handle bad, name resolution failed, or tensor->data null)";
			} else if (firstMismatch430 === null) {
				outcome430 =
					"P-17-clean (all 7 layer-0 weights byte-exact at tensor->data; suspect 2 gain-vector mis-load DEAD by direct measurement; pivot Stage 4.31 to suspect 3 first8-window blindness on kqv_out-0 OR upstream cascade source at Qcur-0 inputs)";
			} else if (firstMismatch430 === "blk.0.ffn_norm.weight") {
				outcome430 =
					"P-17-gain (ffn_norm.weight tensor->data hash mismatch — gain-vector mis-load CONFIRMED via non-set_tensor pathway; trace mmap → CPU-op-read corruption in Stage 4.31)";
			} else if (firstMismatch430 === "blk.0.ffn_down.weight") {
				outcome430 =
					"P-17-ffn-down (ffn_down.weight Q6_K tensor->data hash mismatch — Q6_K upload + dispatch path corruption; deep-dive Q6_K-specific path in Stage 4.31)";
			} else {
				outcome430 = `P-17-jsep-deep (${firstMismatch430} mismatches — JSEP-resident weight host_mirror out of sync with GPU buffer; reopens suspect 1 via different mechanism)`;
			}
			log(
				`     [STAGE-4.30] OUTCOME: ${outcome430}`,
				firstMismatch430 === null && peekFiredCount > 0 ? "ok" : "fail",
			);
			(globalThis as any).__stage430Outcome = outcome430;
		} catch (err) {
			log(
				`     [STAGE-4.30] tensor->data peek threw: ${(err as Error).message}`,
				"fail",
			);
		}

		const ctx = await bridge.createContext(model, { nCtx: 512 });

		// Snapshot counters AFTER model load so model-load JSEP traffic
		// (alloc bursts, weight uploads) doesn't pollute per-token rate.
		const counter0: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		log(`     counters@load = ${JSON.stringify(counter0)}`);

		// Find the JSEP activations buffer dynamically. The Stage 4.2 brief
		// hardcoded handle 11 because that was the scratch buffer's handle
		// in the post-model-load LIVE_BUFFERS list (4×128 MiB weights at
		// handles 6-9, 16 MiB at 10, 64 MiB at 11). Stage 4.3a's selftests
		// alloc/free buffers before model load, advancing the handle
		// counter, so the activations buffer's handle drifts. Pick it as
		// the smallest-handle live buffer that is NOT one of the 128 MiB
		// weight buffers — i.e. bucket ≤ 8.
		const dmAnyEarly = runtime.dataManager as unknown as {
			handles: Map<number, { size: number; bucket: number }>;
		};
		let actHandle = -1;
		let actSize = 0;
		for (const [h, rec] of dmAnyEarly.handles.entries()) {
			if (rec.bucket >= 9) continue; // skip 128 MiB weight buffers
			if (rec.size >= 32 * 1024 * 1024) {
				// 64 MiB scratch beats 16 MiB if both present.
				if (rec.size > actSize) {
					actHandle = h;
					actSize = rec.size;
				}
			}
		}
		log(`ACT_BUF_HANDLE = ${actHandle} size=${actSize}`);

		// Stage 4.2 — pre-prefill GPU buffer dump. Establishes the
		// "initial state" of the JSEP activations buffer at known offsets
		// BEFORE any JSEP ops dispatch, so we can compare to the
		// post-prefill state.
		async function dumpBuf11Pre(off: number, size: number): Promise<number[]> {
			const rec = runtime.dataManager.get(actHandle);
			const staging = runtime.device.createBuffer({
				size,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
			const enc = runtime.device.createCommandEncoder();
			enc.copyBufferToBuffer(rec.buffer, off, staging, 0, size);
			runtime.device.queue.submit([enc.finish()]);
			await staging.mapAsync(GPUMapMode.READ, 0, size);
			const data = new Float32Array(staging.getMappedRange().slice(0));
			staging.unmap();
			staging.destroy();
			return Array.from(data.slice(0, 8));
		}
		const preProbeOffsets = [0, 524288, 2101248, 4194304, 6295552];
		const preProbe: Record<string, number[]> = {};
		for (const off of preProbeOffsets) {
			preProbe[String(off)] = await dumpBuf11Pre(off, 32);
		}
		log(`PREPREFILL_BUF11 = ${JSON.stringify(preProbe)}`);

		// Stage 4.3b — wrap jsepWrite/jsepRead/jsepRunOp with a UNIFIED event
		// sequence (`evtSeq`) so the three streams can be interleaved in the
		// order JSEP actually called them. Stage 4.2's per-stream indices made
		// it impossible to tell whether jsepWrite[i=1] happened before or after
		// jsepRunOp[i=1]; with `seq` the answer is one comparison.
		//
		// Caps bumped from 30 → enough to cover the full prefill graph
		// (1602 runOps, ~2400 writes, ~2400 reads). Full logs live on
		// `window.__jsep*Log`; the page log emits a summary + a curated slice
		// (entries that hit the activations buffer at offsets of interest).
		const MAX_WRITE = 3000;
		const MAX_READ = 3000;
		const RUN_MAX = 1700;
		// Per-runOp deferred dst readback: capture the first
		// DST_PROBE_COUNT runOps' dst contents (first 8 f32 of dst at
		// [dstO, dstO+32)). Probes scheduled via Promise.resolve().then(...)
		// so they run AFTER the wasm sync chain returns, by which time the
		// dispatch has been submitted (and for diverted ops, the destroy()
		// has fired). For runOps writing to the same dst offset that gets
		// overwritten by a LATER runOp before our probe runs, we lose
		// isolation — that's fine for the first ~20 ops where dst offsets
		// in buf19 mostly distinct (the runLog confirms that).
		const DST_PROBE_COUNT = 30;

		let evtSeq = 0;
		type WrEntry = {
			seq: number;
			i: number;
			handle: number;
			offset: number;
			size: number;
			first8: number[];
			firstBytes: number[];
		};
		const writeLog: WrEntry[] = [];
		const readLog: WrEntry[] = [];
		const writeOrig = mod.jsepWrite as (
			h: number,
			o: number,
			p: number,
			s: number,
		) => void;
		const readOrig = mod.jsepRead as (
			h: number,
			o: number,
			p: number,
			s: number,
		) => Promise<void>;
		mod.jsepWrite = (h: number, o: number, p: number, s: number): void => {
			if (writeLog.length < MAX_WRITE) {
				const n = Math.min(8, Math.floor(s / 4));
				const first8 =
					n > 0
						? Array.from(new Float32Array(mod.HEAPU8.buffer, p, n))
						: [];
				const firstBytes = Array.from(
					new Uint8Array(mod.HEAPU8.buffer, p, Math.min(16, s)),
				);
				writeLog.push({
					seq: evtSeq++,
					i: writeLog.length,
					handle: h,
					offset: o,
					size: s,
					first8,
					firstBytes,
				});
			}
			writeOrig(h, o, p, s);
		};
		mod.jsepRead = (
			h: number,
			o: number,
			p: number,
			s: number,
		): Promise<void> => {
			const idx = readLog.length < MAX_READ ? readLog.length : -1;
			if (idx >= 0) {
				readLog.push({
					seq: evtSeq++,
					i: idx,
					handle: h,
					offset: o,
					size: s,
					first8: [-1] /* sentinel: pre-await placeholder */,
					firstBytes: [-1],
				});
			}
			return readOrig(h, o, p, s).then(() => {
				if (idx >= 0) {
					try {
						const heap = mod.HEAPU8.buffer as ArrayBuffer;
						const n = Math.min(8, Math.floor(s / 4));
						const first8 =
							n > 0
								? Array.from(new Float32Array(heap, p, n))
								: [];
						const firstBytes = Array.from(
							new Uint8Array(heap, p, Math.min(16, s)),
						);
						readLog[idx].first8 = first8;
						readLog[idx].firstBytes = firstBytes;
					} catch (e) {
						readLog[idx].first8 = [-99];
						readLog[idx].firstBytes = [
							-99,
							(e as Error).message.length,
						];
					}
				}
			});
		};
		(window as any).__jsepWriteLog = writeLog;
		(window as any).__jsepReadLog = readLog;

		// Instrument jsepRunOp: record (op, dst.handle, dst.offset, src
		// offsets/types) per dispatch so we can correlate "buffer at offset X
		// is NaN" with "what JSEP op (if any) dispatched into X". The
		// descriptor word layout matches readDescriptor in matmul.ts —
		// op at [0], n_src at [1], 19-i32 dst block at [2..20], 19-i32 src
		// blocks starting at [21], [40], etc.
		const TBLK = 19;
		type RunOpEntry = {
			seq: number;
			i: number;
			op: number;
			nSrc: number;
			dstH: number;
			dstO: number;
			srcs: Array<{ h: number; o: number; t: number }>;
			status: number;
			divert: boolean;
		};
		const runLog: RunOpEntry[] = [];
		const runOpOrig = mod.jsepRunOp as (
			d: number,
			dw: number,
			pp: number,
			pl: number,
		) => number;
		// Stage 4.3b: per-runOp deferred dst readback. For the first
		// DST_PROBE_COUNT runOps we schedule a read of dst[dstO..+32) via
		// Promise.resolve().then(...). The .then runs on the next
		// microtask — by then jsepRunOp has returned and the dispatch (or
		// diverted submit) has been queued. The first runOp in the prefill
		// graph that reads back canonical NaN at its dst is the first NaN
		// producer.
		const dstProbes: Array<{
			i: number;
			seq: number;
			op: number;
			dstH: number;
			dstO: number;
			divert: boolean;
			first8: number[] | null;
			err?: string;
		}> = [];
		const dstProbePromises: Promise<void>[] = [];
		(window as any).__dstProbes = dstProbes;

		// Stage 4.6 D2-lite: per-SET_ROWS source/indices probe. For the
		// first SET_ROWS_DIAG_COUNT SET_ROWS dispatches (op=42), capture
		// src[0] first 8 F32 values (the K/V data being written), src[1]
		// first 8 I64 indices (cells targeted in the cache), dst pre and
		// post at the first targeted cell. If src[0] has NaN/Inf/garbage
		// values, the bug is upstream of SET_ROWS (H-source). If indices
		// are wildly out of range, it's H-indices. If both look fine but
		// post differs from f16(src[0]@indices), it's a dispatcher bug
		// in production graph context (D1 wouldn't have caught it).
		const SET_ROWS_DIAG_COUNT = 10;
		type SetRowsDiagEntry = {
			i: number;
			seq: number;
			dstH: number;
			dstO: number;
			divert: boolean;
			src0H: number;
			src0O: number;
			src0Type: number;
			src1H: number;
			src1O: number;
			src1Type: number;
			dstNe: number[];
			src0Ne: number[];
			src1Ne: number[];
			// Stage 4.8: full nb arrays so we can validate stride integrality
			// when status=-1 fires (dispatchSetRows uses console.error which
			// may be lost past CDP's buffer cap).
			dstNb?: number[];
			src0Nb?: number[];
			src1Nb?: number[];
			src0First8F32: number[] | null;
			src1First8Idx: number[] | null;
			dstPreFirst8U16: number[] | null;
			dstPostFirst8U16: number[] | null;
			// Stage 4.7 D2-tight: synchronous readback captured before
			// jsepRunOp returns. JSPI suspends the wasm-side caller while
			// the readback awaits, so no later ops have run yet.
			dstImmediateFirst8U16: number[] | null;
			errImmediate?: string;
			immediateMs?: number;
			err?: string;
		};
		const setRowsDiag: SetRowsDiagEntry[] = [];
		const setRowsDiagPromises: Promise<void>[] = [];
		(window as any).__setRowsDiag = setRowsDiag;
		const GGML_OP_SET_ROWS_VAL = 42; // mirrors GGML_OP_SET_ROWS

		// Stage 4.8 Step B sentinel probe — fires once on the FIRST
		// SET_ROWS divert dispatch (Stage 4.7 R1: i=3, K-cache layer 0,
		// dstO=0 silently no-ops). Writes a distinct u16 pattern into
		// real-dst BEFORE the dispatch runs, then reads after. The
		// pattern post-dispatch tells us which step inside the divert
		// path (pre-copy / kernel / copy-back) failed.
		let sentinelProbeDone = false;
		const SENTINEL_PROBE_U16: number[] = [
			0xbe01, 0xbe02, 0xbe03, 0xbe04, 0xbe05, 0xbe06, 0xbe07, 0xbe08,
		];
		const stage48Probe = {
			triggered: false,
			i: -1,
			preWriteFirst8U16: null as number[] | null,
			postWriteFirst8U16: null as number[] | null,
			postDispatchFirst8U16: null as number[] | null,
			err: null as string | null,
		};
		(window as any).__stage48Probe = stage48Probe;

		// Stage 4.7 D2-tight: jsepRunOp is now async. graph_compute is in
		// JSPI_EXPORTS (src/wasm/CMakeLists.txt) so the wasm side awaits the
		// returned Promise<number>. JSPI suspends the wasm caller while any
		// readback inside this wrapper runs, so reads here see the immediate
		// post-dispatch state — no later ops have run yet.
		mod.jsepRunOp = (async (
			descriptorPtr: number,
			descriptorWords: number,
			opParamsPtr: number,
			opParamsLen: number,
		) => {
			let entry: RunOpEntry | null = null;
			let probeI = -1;
			let setRowsDiagEntry: SetRowsDiagEntry | null = null;
			if (runLog.length < RUN_MAX) {
				const buf = mod.HEAPU8.buffer as ArrayBuffer;
				const heap32 = new Int32Array(buf, 0, buf.byteLength >>> 2);
				const op = heap32[descriptorPtr >>> 2];
				const nSrc = heap32[(descriptorPtr >>> 2) + 1];
				const baseW = descriptorPtr >>> 2;
				const dstH = heap32[baseW + 2 + 0];
				const dstO = heap32[baseW + 2 + 1];
				const srcs: Array<{ h: number; o: number; t: number }> = [];
				for (let s = 0; s < nSrc; s++) {
					const sw = baseW + 2 + TBLK + s * TBLK;
					srcs.push({
						h: heap32[sw + 0],
						o: heap32[sw + 1],
						t: heap32[sw + 2],
					});
				}
				const divert = srcs.some(
					(sr) => sr.h === dstH && srcs.length > 0,
				);
				entry = {
					seq: evtSeq++,
					i: runLog.length,
					op,
					nSrc,
					dstH,
					dstO,
					srcs,
					status: -777,
					divert,
				};
				runLog.push(entry);
				if (runLog.length <= DST_PROBE_COUNT) {
					probeI = runLog.length - 1;
					dstProbes.push({
						i: probeI,
						seq: entry.seq,
						op,
						dstH,
						dstO,
						divert,
						first8: null,
					});
				}

				// Stage 4.6 D2-lite: per-SET_ROWS source/indices probe.
				// Capture full ne arrays for src[0], src[1], dst so we can
				// interpret the readback dimensions correctly.
				if (
					op === GGML_OP_SET_ROWS_VAL &&
					nSrc >= 2 &&
					setRowsDiag.length < SET_ROWS_DIAG_COUNT
				) {
					const dstNe = [
						heap32[baseW + 2 + 3 + 0],
						heap32[baseW + 2 + 3 + 1],
						heap32[baseW + 2 + 3 + 2],
						heap32[baseW + 2 + 3 + 3],
					];
					const src0W = baseW + 2 + TBLK;
					const src1W = baseW + 2 + 2 * TBLK;
					const src0Ne = [
						heap32[src0W + 3 + 0],
						heap32[src0W + 3 + 1],
						heap32[src0W + 3 + 2],
						heap32[src0W + 3 + 3],
					];
					const src1Ne = [
						heap32[src1W + 3 + 0],
						heap32[src1W + 3 + 1],
						heap32[src1W + 3 + 2],
						heap32[src1W + 3 + 3],
					];
					// Stage 4.8: capture nb arrays (low halves of i64).
					const dstNb = [
						heap32[baseW + 2 + 11 + 0],
						heap32[baseW + 2 + 11 + 1],
						heap32[baseW + 2 + 11 + 2],
						heap32[baseW + 2 + 11 + 3],
					];
					const src0Nb = [
						heap32[src0W + 11 + 0],
						heap32[src0W + 11 + 1],
						heap32[src0W + 11 + 2],
						heap32[src0W + 11 + 3],
					];
					const src1Nb = [
						heap32[src1W + 11 + 0],
						heap32[src1W + 11 + 1],
						heap32[src1W + 11 + 2],
						heap32[src1W + 11 + 3],
					];
					setRowsDiagEntry = {
						i: runLog.length - 1,
						seq: entry.seq,
						dstH,
						dstO,
						divert,
						src0H: srcs[0].h,
						src0O: srcs[0].o,
						src0Type: srcs[0].t,
						src1H: srcs[1].h,
						src1O: srcs[1].o,
						src1Type: srcs[1].t,
						dstNe,
						src0Ne,
						src1Ne,
						dstNb,
						src0Nb,
						src1Nb,
						src0First8F32: null,
						src1First8Idx: null,
						dstPreFirst8U16: null,
						dstPostFirst8U16: null,
						dstImmediateFirst8U16: null,
					};
					setRowsDiag.push(setRowsDiagEntry);
				}
			}
			const status = runOpOrig(
				descriptorPtr,
				descriptorWords,
				opParamsPtr,
				opParamsLen,
			);
			if (entry) entry.status = status;

			// Stage 4.8 Step B sentinel probe — post-dispatch capture.
			// Read real-dst at dstO right after the divert path returns to
			// see what landed where the sentinel was written.
			if (
				stage48Probe.triggered &&
				stage48Probe.postDispatchFirst8U16 === null &&
				setRowsDiagEntry &&
				setRowsDiagEntry.i === stage48Probe.i
			) {
				try {
					const dev = runtime.device;
					const READ_U16 = 16;
					runtime.encoderBatcher.flush();
					const dstRec = runtime.dataManager.get(setRowsDiagEntry.dstH);
					const staging = dev.createBuffer({
						size: READ_U16,
						usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
					});
					const enc = dev.createCommandEncoder();
					enc.copyBufferToBuffer(
						dstRec.buffer,
						setRowsDiagEntry.dstO,
						staging,
						0,
						READ_U16,
					);
					dev.queue.submit([enc.finish()]);
					await staging.mapAsync(GPUMapMode.READ, 0, READ_U16);
					const u16 = new Uint16Array(staging.getMappedRange().slice(0));
					staging.unmap();
					staging.destroy();
					stage48Probe.postDispatchFirst8U16 = Array.from(u16);
				} catch (e) {
					stage48Probe.err = (e as Error).message;
				}
			}
			// Schedule deferred dst readback. We capture dstH / dstO in
			// closure because runLog/dstProbes get mutated by later calls.
			if (probeI >= 0 && entry) {
				const probe = dstProbes[probeI];
				const captureDstH = entry.dstH;
				const captureDstO = entry.dstO;
				dstProbePromises.push(
					Promise.resolve().then(async () => {
						try {
							const rec = runtime.dataManager.get(captureDstH);
							const READ = 32;
							const staging = runtime.device.createBuffer({
								size: READ,
								usage:
									GPUBufferUsage.MAP_READ |
									GPUBufferUsage.COPY_DST,
							});
							const enc = runtime.device.createCommandEncoder();
							enc.copyBufferToBuffer(
								rec.buffer,
								captureDstO,
								staging,
								0,
								READ,
							);
							runtime.device.queue.submit([enc.finish()]);
							await staging.mapAsync(GPUMapMode.READ, 0, READ);
							const data = new Float32Array(
								staging.getMappedRange().slice(0),
							);
							staging.unmap();
							staging.destroy();
							probe.first8 = Array.from(data.slice(0, 8));
						} catch (e) {
							probe.err = (e as Error).message;
						}
					}),
				);
			}
			// Stage 4.6 D2-lite: schedule SET_ROWS source / indices / dst
			// readbacks. SET_ROWS src[0] is read-only, so reading after
			// dispatch returns the same value the kernel saw. dst-post is
			// what the kernel actually wrote. The first targeted dst cell
			// (src1[0]) tells us whether the write landed at the right
			// offset with the right value.
			if (setRowsDiagEntry) {
				const diag = setRowsDiagEntry;
				setRowsDiagPromises.push(
					Promise.resolve().then(async () => {
						try {
							const dev = runtime.device;
							const READ_F32 = 32; // 8 F32 values = 32 bytes
							const READ_I64 = 64; // 8 I64 values = 64 bytes (low+high halves)
							const READ_U16 = 16; // 8 F16 cells = 16 bytes

							// src[0] — F32 K/V data being written
							const src0Rec = runtime.dataManager.get(diag.src0H);
							const src0Staging = dev.createBuffer({
								size: READ_F32,
								usage:
									GPUBufferUsage.MAP_READ |
									GPUBufferUsage.COPY_DST,
							});

							// src[1] — I64/I32 indices
							const src1Rec = runtime.dataManager.get(diag.src1H);
							const src1Staging = dev.createBuffer({
								size: READ_I64,
								usage:
									GPUBufferUsage.MAP_READ |
									GPUBufferUsage.COPY_DST,
							});

							// dst — F16 cells at the buffer offset (we read
							// the start of the buffer view, which may or may
							// not contain the targeted cells; comparing
							// post against pre at the same window shows
							// whether the SET_ROWS dispatch wrote here).
							const dstRec = runtime.dataManager.get(diag.dstH);
							const dstStaging = dev.createBuffer({
								size: READ_U16,
								usage:
									GPUBufferUsage.MAP_READ |
									GPUBufferUsage.COPY_DST,
							});

							const enc = dev.createCommandEncoder();
							enc.copyBufferToBuffer(
								src0Rec.buffer,
								diag.src0O,
								src0Staging,
								0,
								READ_F32,
							);
							enc.copyBufferToBuffer(
								src1Rec.buffer,
								diag.src1O,
								src1Staging,
								0,
								READ_I64,
							);
							enc.copyBufferToBuffer(
								dstRec.buffer,
								diag.dstO,
								dstStaging,
								0,
								READ_U16,
							);
							dev.queue.submit([enc.finish()]);

							await src0Staging.mapAsync(
								GPUMapMode.READ,
								0,
								READ_F32,
							);
							const src0F32 = new Float32Array(
								src0Staging.getMappedRange().slice(0),
							);
							src0Staging.unmap();
							src0Staging.destroy();
							diag.src0First8F32 = Array.from(src0F32.slice(0, 8));

							await src1Staging.mapAsync(
								GPUMapMode.READ,
								0,
								READ_I64,
							);
							const src1U32 = new Uint32Array(
								src1Staging.getMappedRange().slice(0),
							);
							src1Staging.unmap();
							src1Staging.destroy();
							// I64 is 8 bytes = 2 u32; index value is low half
							// (high half should be 0 for typical indices).
							const idx: number[] = [];
							for (let k = 0; k < 8; k++) {
								idx.push(src1U32[k * 2]);
							}
							diag.src1First8Idx = idx;

							await dstStaging.mapAsync(
								GPUMapMode.READ,
								0,
								READ_U16,
							);
							const dstU16 = new Uint16Array(
								dstStaging.getMappedRange().slice(0),
							);
							dstStaging.unmap();
							dstStaging.destroy();
							diag.dstPostFirst8U16 = Array.from(dstU16);
						} catch (e) {
							diag.err = (e as Error).message;
						}
					}),
				);
			}
			// Stage 4.7 D2-tight — synchronous post-dispatch readback for the
			// first SET_ROWS_DIAG_COUNT SET_ROWS dispatches. Flush the
			// encoder batcher so the dispatch lands on the GPU, then read 16
			// bytes (8 F16 cells) from dst[dstO..+16). JSPI awaits this
			// promise before resuming the wasm-side caller, so when wasm
			// proceeds to the NEXT jsepRunOp the read here is guaranteed to
			// reflect ONLY this op's output (no later ops have run yet).
			if (setRowsDiagEntry) {
				const diag = setRowsDiagEntry;
				const tImmStart = performance.now();
				try {
					const dev = runtime.device;
					const READ_U16 = 16;
					runtime.encoderBatcher.flush();
					const dstRec = runtime.dataManager.get(diag.dstH);
					const staging = dev.createBuffer({
						size: READ_U16,
						usage:
							GPUBufferUsage.MAP_READ |
							GPUBufferUsage.COPY_DST,
					});
					const enc = dev.createCommandEncoder();
					enc.copyBufferToBuffer(
						dstRec.buffer,
						diag.dstO,
						staging,
						0,
						READ_U16,
					);
					dev.queue.submit([enc.finish()]);
					await staging.mapAsync(GPUMapMode.READ, 0, READ_U16);
					const u16 = new Uint16Array(
						staging.getMappedRange().slice(0),
					);
					staging.unmap();
					staging.destroy();
					diag.dstImmediateFirst8U16 = Array.from(u16);
				} catch (e) {
					diag.errImmediate = (e as Error).message;
				}
				diag.immediateMs = performance.now() - tImmStart;
			}
			return status;
		}) as any;
		(window as any).__jsepRunLog = runLog;

		// Stage 4.8 — install divert hook RIGHT BEFORE decode. The hook
		// captures pre-kernel/post-kernel/post-copy-back snapshots of the
		// FIRST production divert dispatch (i=3, K-cache layer 0). All
		// prior divert calls (warmup + selftests) have already run, so the
		// "triggered" flag will fire on the production i=3.
		// 8 rows × 16 bytes per row = 128 bytes per capture buffer.
		const STAGE48_CAPTURE_SIZE = 128;
		const __stage48PreKernelBuf = device.createBuffer({
			size: STAGE48_CAPTURE_SIZE,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const __stage48PostKernelBuf = device.createBuffer({
			size: STAGE48_CAPTURE_SIZE,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const __stage48PostCopyBackBuf = device.createBuffer({
			size: STAGE48_CAPTURE_SIZE,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const __stage48Src0CaptureBuf = device.createBuffer({
			size: STAGE48_CAPTURE_SIZE,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		(globalThis as any).__stage48DivertHook = {
			triggered: false,
			preKernelBuf: __stage48PreKernelBuf,
			postKernelBuf: __stage48PostKernelBuf,
			postCopyBackBuf: __stage48PostCopyBackBuf,
			src0CaptureBuf: __stage48Src0CaptureBuf,
			capturedDstSize: 0,
			capturedNe0: 0,
			capturedNr: 0,
		};

		// Stage 4.17 Probe 7 — arm per-node first8 dump for the
		// allowlisted layer-0 + final tensors. ~11 names × 6 forward
		// passes (1 prefill + 5 decode) ≈ 66 entries; cap at 200 for
		// headroom. Dump prints on stderr → __stderrLines via printErr.
		mod._webllm_enable_node_dump(200);

		// Stage 4.22 Probe 10 — arm one-shot capture of the first
		// production Q4_0 MUL_MAT dispatch (Qcur-0). The selftests above
		// exercise dispatchMatmul with synthetic Q4_0 inputs; they all
		// completed and `await staging.mapAsync` drained the queue before
		// we reach this point, so arming here guarantees the next eligible
		// dispatch is a production prefill op.
		(globalThis as unknown as {
			__probe10Capture: { armed: boolean; result: unknown };
		}).__probe10Capture = { armed: true, result: null };
		log("     [probe10] capture armed (first Q4_0 MUL_MAT in prefill)");

		// Stage 4.25 Probe 13 — arm Kahan-summed accumulator path on the
		// SAME first-eligible Q4_K MUL_MAT (M=2048, K=2048, N=6) dispatch.
		// dispatchMatmul auto-disarms after first fire so only Qcur-0
		// layer 0 takes the Kahan kernel; every other Q4_K MUL_MAT in
		// the 22-layer prefill stays on the production pipeline.
		(globalThis as unknown as { __stage425KahanArm: boolean }).__stage425KahanArm =
			true;
		log("     [probe13] kahan accumulator armed (Qcur-0 layer 0 only)");

		log(`[7/8] Decoding prompt (${PROMPT_TOKEN_IDS.length} tokens)...`);
		const promptTokens = new Int32Array(PROMPT_TOKEN_IDS);
		const tPrefillStart = performance.now();
		let status = await bridge.decode(ctx, promptTokens, 0);
		const tPrefillMs = performance.now() - tPrefillStart;
		if (status !== 0) {
			log(`     prefill failed status=${status}`, "fail");
			bridge.freeContext(ctx);
			bridge.freeModel(model);
			return;
		}
		log(`     prefill ${tPrefillMs.toFixed(0)} ms`);

		// Stage 4.22 Probe 10 — wait for the capture's mapAsync promises
		// to resolve, then replay the captured src0+src1 bytes through the
		// synthetic Q4_0 matmul harness. Verdict G-1 (synthetic reproduces)
		// vs G-2 (synthetic ≤1e-5).
		try {
			await runtime.device.queue.onSubmittedWorkDone();
			// Yield once to let the mapAsync .then() callbacks run.
			await new Promise((resolve) => setTimeout(resolve, 50));
			const probe10 = (globalThis as unknown as {
				__probe10Capture?: {
					armed: boolean;
					result: {
						M: number;
						K: number;
						N: number;
						src0Type: number;
						src0Bytes: Uint8Array | null;
						src1Bytes: Uint8Array | null;
						dstBeforeBytes: Uint8Array | null;
						dstAfterBytes: Uint8Array | null;
					} | null;
				};
			}).__probe10Capture;
			const cap = probe10?.result;
			if (!cap) {
				log("     [probe10] no capture recorded — gate did not fire", "fail");
			} else if (
				!cap.src0Bytes ||
				!cap.src1Bytes ||
				!cap.dstBeforeBytes ||
				!cap.dstAfterBytes
			) {
				log(
					`     [probe10] capture incomplete: src0=${!!cap.src0Bytes} ` +
						`src1=${!!cap.src1Bytes} dstBefore=${!!cap.dstBeforeBytes} ` +
						`dstAfter=${!!cap.dstAfterBytes}`,
					"fail",
				);
			} else {
				log(
					`     [probe10] captured M=${cap.M} K=${cap.K} N=${cap.N} ` +
						`src0=${cap.src0Bytes.byteLength}B ` +
						`src1=${cap.src1Bytes.byteLength}B ` +
						`dstBefore=${cap.dstBeforeBytes.byteLength}B ` +
						`dstAfter=${cap.dstAfterBytes.byteLength}B`,
				);

				// Replay the captured bytes through the synthetic harness.
				const replay = await runMatmulFromBytes(
					runtime,
					cap.M,
					cap.K,
					cap.N,
					cap.src0Type,
					cap.src0Bytes,
					cap.src1Bytes,
				);
				log(`MATMUL_PROBE10_REPLAY = ${JSON.stringify(replay)}`);

				// Build the same f32 reference the replay used so we can
				// score the captured production dst-after directly.
				const src0Dequant =
					cap.src0Type === GGML_TYPE_Q4_0
						? dequantQ4_0Tile(cap.src0Bytes, cap.M, cap.K)
						: dequantQ4_KTile(cap.src0Bytes, cap.M, cap.K);
				const src1View = new Float32Array(
					cap.src1Bytes.buffer,
					cap.src1Bytes.byteOffset,
					(cap.N * cap.K * 4) / 4,
				);
				const refF32Loop = new Float32Array(cap.M * cap.N);
				for (let n = 0; n < cap.N; n++) {
					for (let m = 0; m < cap.M; m++) {
						let acc32 = Math.fround(0);
						for (let k = 0; k < cap.K; k++) {
							const a = src0Dequant[m * cap.K + k];
							const b = src1View[n * cap.K + k];
							acc32 = Math.fround(acc32 + Math.fround(a * b));
						}
						refF32Loop[n * cap.M + m] = acc32;
					}
				}
				const captured = compareF32Buffers(cap.dstAfterBytes, refF32Loop);
				log(
					`MATMUL_PROBE10_CAPTURED_DELTA = ${JSON.stringify({
						M: cap.M,
						K: cap.K,
						N: cap.N,
						maxAbsDelta: captured.maxAbsDelta,
						hasNaN: captured.hasNaN,
						hasInf: captured.hasInf,
						first8Got: captured.first8Got,
						first8Ref: captured.first8Ref,
					})}`,
				);

				// Verdict line. G-1: synthetic reproduces ≥ 1e-4 (within
				// 2× of capturedDelta) ⇒ Stage 4.18 sweep missed an
				// input/tile case. G-2: synthetic ≤ 1e-5 on the same
				// bytes ⇒ dispatch-boundary bug.
				const capturedDelta = captured.maxAbsDelta;
				const syntheticDelta = replay.maxAbsDeltaVsF32Loop;
				log(
					`     [probe10] M=${cap.M} K=${cap.K} N=${cap.N} ` +
						`capturedDelta=${capturedDelta.toExponential(3)} ` +
						`syntheticDelta=${syntheticDelta.toExponential(3)}`,
				);
				let outcome: string;
				if (syntheticDelta >= 1e-4 && syntheticDelta >= capturedDelta * 0.5) {
					outcome =
						"G-1 (synthetic reproduces — Stage 4.18 sweep missed input/tile case)";
				} else if (syntheticDelta <= 1e-5) {
					outcome =
						"G-2 (synthetic ≤1e-5 — bug between dispatch site and shader execution)";
				} else {
					outcome =
						`G-INDETERMINATE (capturedDelta=${capturedDelta.toExponential(3)} ` +
						`syntheticDelta=${syntheticDelta.toExponential(3)} — neither boundary)`;
				}
				log(`     [probe10] OUTCOME: ${outcome}`);

				// Stage 4.24 Probe 12 — WGSL-equivalent dequant (Path A,
				// `dequantQ4_KTile`) vs libllama dequant (Path B, the
				// `webllm_dequantize_q4_K` shim wrapping
				// `ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float` =
				// `dequantize_row_q4_K`). Diffs the resulting f32 weight
				// tiles element-wise on the production layer-0 wq Q4_K
				// bytes (`cap.src0Bytes`). H-3a if maxAbs > 1e-5 (WGSL
				// dequant disagreement); H-3b if ≤ 1e-5 (matmul
				// accumulation-order disagreement). Q4_0 capture path is
				// not exercised here — the production capture in TinyLlama
				// is Q4_K (verified by Stage 4.22's surprise finding that
				// "Q4_0" in the GGUF filename is the HF tier label, not
				// the on-disk tensor type for projections).
				if (cap.src0Type === GGML_TYPE_Q4_K) {
					const totalElems = cap.M * cap.K;
					const wgslDequant = dequantQ4_KTile(
						cap.src0Bytes,
						cap.M,
						cap.K,
					);

					const srcBytes = cap.src0Bytes.byteLength;
					const dstBytes = totalElems * 4;
					const srcPtr = (mod as { _malloc: (n: number) => number })._malloc(srcBytes);
					const dstPtr = (mod as { _malloc: (n: number) => number })._malloc(dstBytes);
					if (!srcPtr || !dstPtr) {
						log(
							`     [probe12] _malloc failed (srcPtr=${srcPtr} dstPtr=${dstPtr})`,
							"fail",
						);
					} else {
						const heap = (mod as { HEAPU8: Uint8Array }).HEAPU8;
						heap.set(cap.src0Bytes, srcPtr);
						const status = (mod as {
							_webllm_dequantize_q4_K: (
								s: number,
								d: number,
								k: number,
							) => number;
						})._webllm_dequantize_q4_K(srcPtr, dstPtr, totalElems);
						if (status !== 0) {
							log(`     [probe12] dequant shim status=${status}`, "fail");
						} else {
							const heapF32 = (mod as { HEAPF32: Float32Array }).HEAPF32;
							const llamaDequant = new Float32Array(totalElems);
							llamaDequant.set(
								heapF32.subarray(dstPtr / 4, dstPtr / 4 + totalElems),
							);

							let maxAbs = 0;
							let maxIdx = -1;
							let nNaN = 0;
							let nInf = 0;
							for (let i = 0; i < totalElems; i++) {
								const a = wgslDequant[i];
								const b = llamaDequant[i];
								if (Number.isNaN(a) || Number.isNaN(b)) {
									nNaN++;
									continue;
								}
								if (!Number.isFinite(a) || !Number.isFinite(b)) {
									nInf++;
									continue;
								}
								const d = Math.abs(a - b);
								if (d > maxAbs) {
									maxAbs = d;
									maxIdx = i;
								}
							}
							const verdict = maxAbs > 1e-5 ? "H-3a" : "H-3b";
							const first8Wgsl = Array.from(wgslDequant.subarray(0, 8));
							const first8Llama = Array.from(llamaDequant.subarray(0, 8));
							log(
								`PROBE12_DEQUANT_DELTA = ${JSON.stringify({
									M: cap.M,
									K: cap.K,
									totalElems,
									maxAbsDelta: maxAbs,
									maxIdx,
									nNaN,
									nInf,
									first8Wgsl,
									first8Llama,
									verdict,
								})}`,
							);
							log(
								`     [probe12] dequantDeltaMax=${maxAbs.toExponential(3)} ` +
									`maxIdx=${maxIdx} OUTCOME: ${verdict}`,
							);
						}
						(mod as { _free: (p: number) => void })._free(srcPtr);
						(mod as { _free: (p: number) => void })._free(dstPtr);
					}
				} else {
					log(
						`     [probe12] skipped — cap.src0Type=${cap.src0Type} ≠ Q4_K`,
					);
				}

				// Stage 4.25 Probe 13 — diff captured Kahan dst-after first8
				// against Stage 4.24's hard-coded non-Kahan baseline first8Got
				// (recorded in `STAGE-4.24-RESULT.md`). The captured `first8Got`
				// from the current run reflects the Kahan kernel output (the
				// dispatch gate triggered on this same first-eligible Q4_K
				// MUL_MAT). Build a Kahan-summed JS reference too — the Kahan
				// WGSL output should match a Kahan JS loop to ~ULP and a plain
				// f32 loop to ~5e-5 (single-ULP recovery) if Kahan worked.
				const STAGE424_BASELINE_FIRST8: readonly number[] = [
					-0.01618947833776474, 0.004848937503993511,
					-0.015738369897007942, -0.02449355274438858,
					-0.007620065473020077, 0.04053414985537529,
					-0.009678085334599018, 0.04543862119317055,
				];
				const kahanFirst8 = captured.first8Got;
				let kahanVsBaselineMax = 0;
				let kahanVsBaselineIdx = -1;
				const perElem: number[] = [];
				for (let i = 0; i < 8; i++) {
					const d = Math.abs(kahanFirst8[i] - STAGE424_BASELINE_FIRST8[i]);
					perElem.push(d);
					if (d > kahanVsBaselineMax) {
						kahanVsBaselineMax = d;
						kahanVsBaselineIdx = i;
					}
				}

				// Kahan-summed JS reference over the same captured src0/src1.
				// Mirrors the WGSL Neumaier accumulator (always-keep-larger).
				const refKahan = new Float32Array(cap.M * cap.N);
				for (let n = 0; n < cap.N; n++) {
					for (let m = 0; m < cap.M; m++) {
						let acc = Math.fround(0);
						let comp = Math.fround(0);
						for (let k = 0; k < cap.K; k++) {
							const a = src0Dequant[m * cap.K + k];
							const b = src1View[n * cap.K + k];
							const term = Math.fround(a * b);
							const t = Math.fround(acc + term);
							if (Math.abs(acc) >= Math.abs(term)) {
								comp = Math.fround(comp + Math.fround(Math.fround(acc - t) + term));
							} else {
								comp = Math.fround(comp + Math.fround(Math.fround(term - t) + acc));
							}
							acc = t;
						}
						refKahan[n * cap.M + m] = Math.fround(acc + comp);
					}
				}
				const kahanRef = compareF32Buffers(cap.dstAfterBytes, refKahan);

				// Verdict thresholds (per Stage 4.25 brief):
				//   capturedDelta = max delta between WGSL-Kahan output and
				//   the Stage 4.24 non-Kahan WGSL baseline. Measures how
				//   much enabling Kahan moved the kernel. The historical
				//   cross-module disagreement (libllama vs WGSL Qcur-0)
				//   was 5.242e-4 — if Kahan moves the WGSL output by that
				//   magnitude in the right direction, Kahan IS the fix.
				let verdict: string;
				if (kahanVsBaselineMax <= 1e-5) {
					verdict =
						"H-3b-structural (Kahan ≈ baseline — accumulation order is not the disagreement source; cascade mitigation needed)";
				} else if (kahanVsBaselineMax > 1e-4) {
					verdict =
						"H-3b-Kahan (Kahan moved kernel ≥1e-4 — investigate alignment with libllama; ship Kahan if production prefill output stabilizes)";
				} else {
					verdict =
						"H-3b-partial (Kahan moved kernel between 1e-5 and 1e-4 — investigate FMA + Kahan combination)";
				}
				log(
					`MATMUL_PROBE13_DELTA = ${JSON.stringify({
						M: cap.M,
						K: cap.K,
						N: cap.N,
						kahanVsBaselineMax,
						kahanVsBaselineIdx,
						perElem,
						kahanFirst8,
						stage424BaselineFirst8: STAGE424_BASELINE_FIRST8,
						kahanVsKahanRefMax: kahanRef.maxAbsDelta,
					})}`,
				);
				const kahanFired =
					(globalThis as unknown as { __stage425KahanFired?: boolean })
						.__stage425KahanFired === true;
				log(
					`     [probe13] kahanArm=true kahanFired=${kahanFired} ` +
						`capturedDelta=${kahanVsBaselineMax.toExponential(3)} ` +
						`baseline=5.242e-04 ` +
						`kahanVsKahanRef=${kahanRef.maxAbsDelta.toExponential(3)} ` +
						`verdict: ${verdict}`,
				);

				// Stage 4.26 Probe 14 — score libllama's CPU Q4_K × Q8_K
				// matmul against an f64 reference on the same captured
				// production Q-projection inputs (cap.src0Bytes / cap.src1Bytes
				// / src1View / src0Dequant). Mirrors Stage 4.24's
				// `webllm_dequantize_q4_K` shim pattern: malloc src0/src1/dst,
				// HEAPU8.set the captured bytes, call the shim, slice the
				// result, free. The f64 reference accumulates in pure double
				// precision over the JS-side dequant (already in
				// `src0Dequant`). Verdict thresholds per Stage 4.26 brief:
				//   ≥1e-4 → H-4-libllama-imprecise (libllama IS the imprecise
				//          side; close matmul-precision investigation)
				//   ≤1e-5 → H-4-libllama-precise   (both sides agree with
				//          truth; bug is upstream src1 / RMSNorm)
				//   else  → H-4-libllama-mid       (multi-source contributor)
				if (cap.src0Type === GGML_TYPE_Q4_K) {
					const dstShimSize = cap.M * cap.N * 4;
					const src0Ptr = (mod as { _malloc: (n: number) => number })._malloc(
						cap.src0Bytes.byteLength,
					);
					const src1Ptr = (mod as { _malloc: (n: number) => number })._malloc(
						cap.src1Bytes.byteLength,
					);
					const dstShimPtr = (mod as { _malloc: (n: number) => number })._malloc(
						dstShimSize,
					);
					if (!src0Ptr || !src1Ptr || !dstShimPtr) {
						log(
							`     [probe14] _malloc failed (src0=${src0Ptr} src1=${src1Ptr} dst=${dstShimPtr})`,
							"fail",
						);
						if (src0Ptr) (mod as { _free: (p: number) => void })._free(src0Ptr);
						if (src1Ptr) (mod as { _free: (p: number) => void })._free(src1Ptr);
						if (dstShimPtr)
							(mod as { _free: (p: number) => void })._free(dstShimPtr);
					} else {
						// Re-derive heap views after malloc (heap may have grown
						// — same pattern as Probe 12).
						const heapU8 = (mod as { HEAPU8: Uint8Array }).HEAPU8;
						heapU8.set(cap.src0Bytes, src0Ptr);
						heapU8.set(cap.src1Bytes, src1Ptr);
						const status = (
							mod as {
								_webllm_q4k_q8k_matmul: (
									s0: number,
									s1: number,
									d: number,
									M: number,
									K: number,
									N: number,
								) => number;
							}
						)._webllm_q4k_q8k_matmul(
							src0Ptr,
							src1Ptr,
							dstShimPtr,
							cap.M,
							cap.K,
							cap.N,
						);
						if (status !== 0) {
							log(
								`     [probe14] matmul shim status=${status} ` +
									`(-1=bad-args -2=missing-cpu-traits -3=malloc-fail)`,
								"fail",
							);
						} else {
							const heapF32 = (mod as { HEAPF32: Float32Array }).HEAPF32;
							const llamaOutput = new Float32Array(cap.M * cap.N);
							llamaOutput.set(
								heapF32.subarray(
									dstShimPtr / 4,
									dstShimPtr / 4 + cap.M * cap.N,
								),
							);

							// f64 reference — pure double accumulation over
							// the same JS-side dequant + raw src1 f32 values.
							// No Math.fround; this is the truth oracle both
							// modules are scored against.
							const refF64 = new Float64Array(cap.M * cap.N);
							for (let n = 0; n < cap.N; n++) {
								for (let m = 0; m < cap.M; m++) {
									let acc = 0;
									const aRow = m * cap.K;
									const bRow = n * cap.K;
									for (let k = 0; k < cap.K; k++) {
										acc += src0Dequant[aRow + k] * src1View[bRow + k];
									}
									refF64[n * cap.M + m] = acc;
								}
							}

							let llamaVsF64Max = 0;
							let llamaVsF64Idx = -1;
							let nNaN = 0;
							let nInf = 0;
							for (let i = 0; i < cap.M * cap.N; i++) {
								const v = llamaOutput[i];
								if (Number.isNaN(v)) {
									nNaN++;
									continue;
								}
								if (!Number.isFinite(v)) {
									nInf++;
									continue;
								}
								const d = Math.abs(v - refF64[i]);
								if (d > llamaVsF64Max) {
									llamaVsF64Max = d;
									llamaVsF64Idx = i;
								}
							}

							// Cross-reference: re-score the WGSL captured dst
							// against the same f64 reference for an apples-to-
							// apples comparison (Probe 10 used an f32-loop
							// reference; refresh against f64 here so both
							// numbers in the closure share an oracle).
							const wgslOutput = new Float32Array(
								cap.dstAfterBytes.buffer,
								cap.dstAfterBytes.byteOffset,
								cap.M * cap.N,
							);
							let wgslVsF64Max = 0;
							for (let i = 0; i < cap.M * cap.N; i++) {
								const d = Math.abs(wgslOutput[i] - refF64[i]);
								if (d > wgslVsF64Max) wgslVsF64Max = d;
							}

							let llamaVsWgslMax = 0;
							let llamaVsWgslIdx = -1;
							for (let i = 0; i < cap.M * cap.N; i++) {
								const d = Math.abs(llamaOutput[i] - wgslOutput[i]);
								if (d > llamaVsWgslMax) {
									llamaVsWgslMax = d;
									llamaVsWgslIdx = i;
								}
							}

							let probe14Verdict: string;
							if (llamaVsF64Max >= 1e-4) {
								probe14Verdict =
									"H-4-libllama-imprecise (libllama ≥1e-4 from f64 truth — close matmul-precision investigation; pivot to other ops in cascade)";
							} else if (llamaVsF64Max <= 1e-5) {
								probe14Verdict =
									"H-4-libllama-precise (libllama ≤1e-5 from f64 truth — both sides accurate; cross-module gap must come from upstream src1 / RMSNorm divergence)";
							} else {
								probe14Verdict =
									"H-4-libllama-mid (libllama between 1e-5 and 1e-4 from f64 truth — multi-source contribution; queue both upstream src1 re-capture AND libllama precision quantification)";
							}

							log(
								`PROBE14_LLAMA_MATMUL_VS_F64 = ${JSON.stringify({
									M: cap.M,
									K: cap.K,
									N: cap.N,
									llamaVsF64Max,
									llamaVsF64Idx,
									wgslVsF64Max,
									llamaVsWgslMax,
									llamaVsWgslIdx,
									nNaN,
									nInf,
									first8Llama: Array.from(llamaOutput.subarray(0, 8)),
									first8Wgsl: Array.from(wgslOutput.subarray(0, 8)),
									first8RefF64: Array.from(refF64.subarray(0, 8)),
								})}`,
							);
							log(
								`     [probe14] llamaVsF64=${llamaVsF64Max.toExponential(3)} ` +
									`wgslVsF64=${wgslVsF64Max.toExponential(3)} ` +
									`llamaVsWgsl=${llamaVsWgslMax.toExponential(3)} ` +
									`verdict: ${probe14Verdict}`,
							);
						}
						(mod as { _free: (p: number) => void })._free(src0Ptr);
						(mod as { _free: (p: number) => void })._free(src1Ptr);
						(mod as { _free: (p: number) => void })._free(dstShimPtr);
					}
				} else {
					log(
						`     [probe14] skipped — cap.src0Type=${cap.src0Type} ≠ Q4_K`,
					);
				}
			}
		} catch (err) {
			log(
				`     [probe10] replay threw: ${(err as Error).message}`,
				"fail",
			);
		}

		// Stage 4.2 Step 2 — manual GPU read of buf 11 at the offsets that
		// jsepRead reported as NaN. If these reads return the SAME NaN, the
		// JSEP-supported ops (MUL_MAT/RMS_NORM/SET_ROWS) genuinely failed
		// to write valid data. If they return real data, the bug is in
		// jsepRead synchronization (e.g., reading before the producing op
		// dispatched).
		async function dumpBuf11(off: number, size: number): Promise<number[]> {
			const rec = runtime.dataManager.get(actHandle);
			const staging = runtime.device.createBuffer({
				size,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
			const enc = runtime.device.createCommandEncoder();
			enc.copyBufferToBuffer(rec.buffer, off, staging, 0, size);
			runtime.device.queue.submit([enc.finish()]);
			await staging.mapAsync(GPUMapMode.READ, 0, size);
			const data = new Float32Array(staging.getMappedRange().slice(0));
			staging.unmap();
			staging.destroy();
			return Array.from(data.slice(0, 8));
		}
		const probeOffsets = [
			0, 524288, 528384, 1052672, 2101248, 4194304, 6295552, 17829888,
			35655680,
		];
		const probeResults: Record<string, number[]> = {};
		for (const off of probeOffsets) {
			probeResults[String(off)] = await dumpBuf11(off, 32);
		}
		log(`POSTPREFILL_BUF11 = ${JSON.stringify(probeResults)}`);
		// Dump all live buffers + their sizes (private fields via reflection)
		const dmAny = runtime.dataManager as unknown as {
			handles: Map<number, { size: number; bucket: number }>;
		};
		const liveBufs: Array<{ h: number; size: number; bucket: number }> = [];
		for (const [h, rec] of dmAny.handles.entries()) {
			liveBufs.push({ h, size: rec.size, bucket: rec.bucket });
		}
		log(`LIVE_BUFFERS = ${JSON.stringify(liveBufs)}`);
		log(`GPU_ERR_LOG = ${JSON.stringify(gpuErrLog.slice(0, 8))}`);
		log(`GPU_ERR_COUNT = ${gpuErrLog.length}`);

		log(`[8/8] Greedy decoding ${N_GENERATE} tokens...`);
		const generatedIds: number[] = [];
		let nPast = PROMPT_TOKEN_IDS.length;
		const tDecodeStart = performance.now();
		const logitStats: Array<{
			step: number;
			first8: number[];
			topVal: number;
			topId: number;
			hasNaN: boolean;
			hasInf: boolean;
			finiteCount: number;
			minFinite: number;
			maxFinite: number;
		}> = [];
		for (let step = 0; step < N_GENERATE; step++) {
			const logits = await bridge.getLogits(ctx, model);
			let topId = 0;
			let topVal = -Infinity;
			let hasNaN = false;
			let hasInf = false;
			let finiteCount = 0;
			let minFinite = Infinity;
			let maxFinite = -Infinity;
			for (let i = 0; i < logits.length; i++) {
				const v = logits[i];
				if (Number.isNaN(v)) {
					hasNaN = true;
				} else if (!Number.isFinite(v)) {
					hasInf = true;
				} else {
					finiteCount++;
					if (v < minFinite) minFinite = v;
					if (v > maxFinite) maxFinite = v;
				}
				if (v > topVal) {
					topVal = v;
					topId = i;
				}
			}
			if (step === 0) {
				logitStats.push({
					step,
					first8: Array.from(logits.slice(0, 8)),
					topVal,
					topId,
					hasNaN,
					hasInf,
					finiteCount,
					minFinite,
					maxFinite,
				});
			}
			generatedIds.push(topId);
			const single = new Int32Array([topId]);
			status = await bridge.decode(ctx, single, nPast);
			nPast++;
			if (status !== 0) {
				log(`decode step ${step} failed status=${status}`, "fail");
				break;
			}
		}
		const tDecodeMs = performance.now() - tDecodeStart;

		const counter1: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		const deltas: Record<string, number> = {};
		for (const k of Object.keys(counter1)) {
			deltas[k] = (counter1[k] ?? 0) - (counter0[k] ?? 0);
		}

		const perTokenMs = tDecodeMs / N_GENERATE;
		const totalCrossings =
			(deltas.alloc ?? 0) +
			(deltas.free ?? 0) +
			(deltas.write ?? 0) +
			(deltas.read ?? 0) +
			(deltas.clear ?? 0) +
			(deltas.runOp ?? 0) +
			(deltas.sync ?? 0);
		const crossingsPerToken = totalCrossings / N_GENERATE;

		// Detokenize generated ids into a readable string.
		let generatedText = "";
		try {
			generatedText = bridge.detokenize(model, new Int32Array(generatedIds));
		} catch (e) {
			generatedText = `<detokenize failed: ${(e as Error).message}>`;
		}

		// Stage 4.3b — wait for all per-runOp dst probes to complete
		// before emitting the summary. Probe promises are pushed onto
		// `dstProbePromises` as runOps fire; awaiting all settles every
		// staging copy + mapAsync.
		await Promise.allSettled(dstProbePromises);
		// Stage 4.6 D2-lite — wait for SET_ROWS source/indices/dst
		// readbacks to complete.
		await Promise.allSettled(setRowsDiagPromises);

		// Emit summaries + first-30 slices on the page log (full data
		// stays on `window.__jsep*Log` and `window.__dstProbes` for
		// targeted `js exec` fetches). Inlining the full 1700-entry
		// arrays would blow past agentchrome's 16 KB inline limit.
		log(
			`LOG_COUNTS = ${JSON.stringify({
				runOps: runLog.length,
				writes: writeLog.length,
				reads: readLog.length,
				dstProbes: dstProbes.length,
			})}`,
		);
		log(`JSEPRUN_LOG_FIRST30 = ${JSON.stringify(runLog.slice(0, 30))}`);
		log(`JSEPWRITE_LOG_FIRST30 = ${JSON.stringify(writeLog.slice(0, 30))}`);
		log(`JSEPREAD_LOG_FIRST30 = ${JSON.stringify(readLog.slice(0, 30))}`);
		log(`DST_PROBES = ${JSON.stringify(dstProbes)}`);
		// Filter: all writes that hit the activations buffer at offsets
		// the POSTPREFILL probe found NaN at (0, 528384, 1052672,
		// 2101248, 4194304, 6295552, 17829888, 35655680). These are the
		// CPU writebacks that pollute scratch.
		const NAN_OFFSETS = new Set([
			0, 524288, 528384, 1052672, 2101248, 4194304, 6295552, 17829888,
			35655680,
		]);
		const writesToNanOffsets = writeLog.filter(
			(w) => w.handle === actHandle && NAN_OFFSETS.has(w.offset),
		);
		log(
			`WRITES_TO_NAN_OFFSETS_COUNT = ${writesToNanOffsets.length}`,
		);
		log(
			`WRITES_TO_NAN_OFFSETS_FIRST20 = ${JSON.stringify(writesToNanOffsets.slice(0, 20))}`,
		);
		// First runOp whose dst probe came back as NaN (any of first8 NaN).
		const firstNanProbe = dstProbes.find(
			(p) => p.first8 !== null && p.first8.some((v) => Number.isNaN(v)),
		);
		log(`FIRST_NAN_DST_PROBE = ${JSON.stringify(firstNanProbe ?? null)}`);
		// First runOp whose dst probe came back as all-zero (Bug B candidate
		// — if lm_head's dst stays zero post-dispatch this surfaces here).
		const firstZeroProbe = dstProbes.find(
			(p) =>
				p.first8 !== null &&
				p.first8.every((v) => v === 0),
		);
		log(`FIRST_ALLZERO_DST_PROBE = ${JSON.stringify(firstZeroProbe ?? null)}`);
		// Stage 4.6 D2-lite — first SET_ROWS dispatches' source / indices /
		// dst readback. Tells us whether the K/V data fed into SET_ROWS is
		// sensible (small floats), garbage (NaN/Inf/denormals), or all-zero
		// (Stage 4.4 H1-pre-fix host_mirror state). The first SET_ROWS in a
		// TinyLlama prefill graph writes the K cache for the prompt's first
		// layer; src[0] should contain F32 K-projection outputs after ROPE
		// rotation — typical magnitudes are |x| < 5.
		// Stage 4.8 — read back temp-dst captures from the FIRST production
		// divert dispatch.
		try {
			await __stage48PreKernelBuf.mapAsync(GPUMapMode.READ, 0, 128);
			stage48Captures.preKernelFirst8U16 = Array.from(
				new Uint16Array(__stage48PreKernelBuf.getMappedRange().slice(0)),
			);
			__stage48PreKernelBuf.unmap();
		} catch (e) {
			stage48Captures.err = (e as Error).message;
		}
		try {
			await __stage48PostKernelBuf.mapAsync(GPUMapMode.READ, 0, 128);
			stage48Captures.postKernelFirst8U16 = Array.from(
				new Uint16Array(__stage48PostKernelBuf.getMappedRange().slice(0)),
			);
			__stage48PostKernelBuf.unmap();
		} catch (e) {
			stage48Captures.err = (e as Error).message;
		}
		try {
			await __stage48PostCopyBackBuf.mapAsync(GPUMapMode.READ, 0, 128);
			stage48Captures.postCopyBackFirst8U16 = Array.from(
				new Uint16Array(__stage48PostCopyBackBuf.getMappedRange().slice(0)),
			);
			__stage48PostCopyBackBuf.unmap();
		} catch (e) {
			stage48Captures.err = (e as Error).message;
		}
		try {
			await __stage48Src0CaptureBuf.mapAsync(GPUMapMode.READ, 0, 128);
			stage48Captures.src0AtKernelTimeF32 = Array.from(
				new Float32Array(__stage48Src0CaptureBuf.getMappedRange().slice(0)),
			);
			__stage48Src0CaptureBuf.unmap();
		} catch (e) {
			stage48Captures.err = (e as Error).message;
		}
		log(`STAGE48_CAPTURES = ${JSON.stringify(stage48Captures)}`);

		log(`SETROWS_DIAG_FIRST5 = ${JSON.stringify(setRowsDiag.slice(0, 5))}`);
		// Stage 4.17 Probe 7 — emit captured CHECKPOINT lines from stderr
		// for diff against the non-JSEP reference run.
		const checkpointLines = ((window as any).__stderrLines as string[])
			.filter((s) => s.includes("[CHECKPOINT"));
		log(`CHECKPOINT_COUNT = ${checkpointLines.length}`);
		(window as any).__stage417Checkpoints = checkpointLines;
		for (const line of checkpointLines) log(line);

		// Stage 4.31 Probe 18 Shape A — parse the `[CHECKPOINT-FULL ...]`
		// lines emitted by node_dump_cb for `kqv_out-0` (mean / abs_max /
		// abs_min / NaN / Inf over the FULL tensor, defeating the
		// first8-window blindness Stage 4.27 flagged on this op). Both
		// the JSEP spike and the non-JSEP ref-probe emit these — diff is
		// computed offline by a Python helper that compares
		// `__stage431Stats` from the two runs and synthesises
		// P-18-{first8-blind, full-clean, full-dirty}.
		const stage431Pat =
			/\[CHECKPOINT-FULL idx=(\d+) name=(\S+) n_elements=(\d+) finite=(\d+) mean=(\S+) abs_max=(\S+) abs_min=(\S+) nan=(\d+) inf=(\d+)\]/;
		const stage431Stats: Array<{
			idx: number;
			name: string;
			n_elements: number;
			finite: number;
			mean: number;
			abs_max: number;
			abs_min: number;
			nan: number;
			inf: number;
		}> = [];
		for (const line of (window as any).__stderrLines as string[]) {
			const m = line.match(stage431Pat);
			if (!m) continue;
			stage431Stats.push({
				idx: +m[1],
				name: m[2],
				n_elements: +m[3],
				finite: +m[4],
				mean: Number(m[5]),
				abs_max: Number(m[6]),
				abs_min: Number(m[7]),
				nan: +m[8],
				inf: +m[9],
			});
		}
		(window as any).__stage431Stats = stage431Stats;
		log(`STAGE431_STATS_COUNT = ${stage431Stats.length}`);
		for (const s of stage431Stats) {
			log(
				`[STAGE-4.31] idx=${s.idx} name=${s.name} n=${s.n_elements} ` +
					`finite=${s.finite} mean=${s.mean} abs_max=${s.abs_max} ` +
					`abs_min=${s.abs_min} nan=${s.nan} inf=${s.inf}`,
			);
		}

		log(`LOGIT_STATS_STEP0 = ${JSON.stringify(logitStats[0])}`);
		log(`GENERATED_TOKENS = ${JSON.stringify(generatedIds)}`);
		log(`GENERATED_TEXT = ${JSON.stringify(generatedText)}`);
		log(`PER_TOKEN_MS = ${perTokenMs.toFixed(2)}`);
		log(`COUNTER_DELTAS = ${JSON.stringify(deltas)}`);
		log(`CROSSINGS_PER_TOKEN = ${crossingsPerToken.toFixed(1)}`);
		log(`TOTAL_DECODE_MS = ${tDecodeMs.toFixed(0)}`);
		log(`TOTAL_PREFILL_MS = ${tPrefillMs.toFixed(0)}`);
		log(`MODEL_LOAD_MS = ${tLoadMs.toFixed(0)}`);
		log("DONE", "pass");

		// Expose for agentchrome `js exec` readout.
		(window as any).__spikeResult = {
			generatedIds,
			generatedText,
			perTokenMs,
			deltas,
			crossingsPerToken,
			totalDecodeMs: tDecodeMs,
			totalPrefillMs: tPrefillMs,
			modelLoadMs: tLoadMs,
			counter0,
			counter1,
		};

		bridge.freeContext(ctx);
		bridge.freeModel(model);
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runSpike();
