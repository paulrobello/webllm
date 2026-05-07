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
import {
	dispatchMatmul,
	GGML_TYPE_F16,
	GGML_TYPE_F32,
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
		const t0 = performance.now();
		const model = await bridge.loadModel(buf);
		const tLoadMs = performance.now() - t0;
		const vocab = bridge.nVocab(model);
		log(`     model loaded in ${tLoadMs.toFixed(0)} ms; vocab = ${vocab}`);
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
		log(`SETROWS_DIAG_FIRST5 = ${JSON.stringify(setRowsDiag.slice(0, 5))}`);
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
