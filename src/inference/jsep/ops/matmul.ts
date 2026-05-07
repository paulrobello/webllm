/**
 * JSEP matmul kernel — F32, F16, Q4_0, Q4_K dispatch paths.
 *
 * Reads the per-op descriptor packed C++-side in
 * `ggml/src/ggml-jsep/ggml-jsep.cpp::ggml_backend_jsep_graph_compute`
 * and dispatches one or more compute passes via the
 * `CommandEncoderBatcher`. Pipelines are cached on
 * (src0_type, src1_type, dst_type, ndim).
 *
 * ggml MUL_MAT semantics (the canonical definition):
 *   src0: [K, M, ...] — rows × cols stored as ne[0]=K (cols), ne[1]=M (rows).
 *   src1: [K, N, ...] — same K for inner-product axis.
 *   dst:  [M, N, ...] — ne[0]=M, ne[1]=N.
 *
 * In conventional matrix terms: dst = src0^T @ src1, where src0 carries
 * the "weights" (M rows × K columns) and src1 carries the "activations"
 * (N columns of K-dim activations).
 *
 * Phase 2 covers the 2D (B = batch product = 1) cases plus a defensive
 * batch loop for 3D when ne[2] > 1.
 *
 * Descriptor layout (Phase 3 Stage 2 — must match `ggml-jsep.cpp` — see
 * top of that file):
 *
 *   per-tensor block (19 i32 = 76 bytes):
 *     [0]      buf_handle         (i32; jsep buffer handle — keys dataManager.get)
 *     [1]      offset             (i32; byte offset of tensor->data within buffer)
 *     [2]      type               (i32; ggml_type enum value)
 *     [3..6]   ne[0..3] low half  (i32; ne is int64_t)
 *     [7..10]  ne[0..3] high half (i32; zero on wasm32 with sub-2GB tensors)
 *     [11..14] nb[0..3] low half  (i32; nb is size_t)
 *     [15..18] nb[0..3] high half (i32; zero on wasm32)
 *
 *   descriptor (offset 0 in i32 words):
 *     [0]      op                 (i32; ggml_op enum)
 *     [1]      n_src              (i32)
 *     [2..20]  dst block          (19 i32)
 *     [21..39] src[0] block       (19 i32)
 *     [40..58] src[1] block       (19 i32)
 *     ...
 */

import type { CommandEncoderBatcher } from "../command-encoder.js";
import type { GpuDataManager } from "../gpu-data-manager.js";
import type { PipelineCache } from "../pipeline-cache.js";

// ggml_type enum values (subset; see ggml/include/ggml.h:389).
export const GGML_TYPE_F32 = 0;
export const GGML_TYPE_F16 = 1;
export const GGML_TYPE_Q4_0 = 2;
export const GGML_TYPE_Q4_K = 12;

// Per-tensor block size in i32 slots (must match C++ side).
const TENSOR_BLOCK_I32 = 19;

// Workgroup tile size for the matmul kernel: TILE_M output rows × TILE_N
// output cols per workgroup. Each invocation computes one output element.
const TILE_M = 16;
const TILE_N = 16;

// Q4_0 block geometry (must match ggml-common.h; QK4_0 == 32).
const QK4_0 = 32;

// Q4_K super-block geometry (must match ggml-common.h::block_q4_K):
//   - QK_K = 256 elements per super-block
//   - 144 bytes per super-block:
//       d (f16, 2B) + dmin (f16, 2B) + scales[12] (6-bit packed) + qs[128]
//   - 8 sub-blocks of 32 elements each; sub-blocks come in pairs sharing a
//     32-byte qs region (low nibble = first 32, high nibble = second 32).
const QK_K = 256;
const Q4_K_BYTES_PER_BLOCK = 144;

export interface JsepTensorMeta {
	// JSEP buffer-level handle (resolves to a GPUBuffer via dataManager.get).
	bufHandle: number;
	// Byte offset of tensor->data within the buffer. Aligned to
	// GGML_JSEP_BUFFER_ALIGN (256B) by ggml allocation, which is also
	// ≥ minStorageBufferOffsetAlignment so it can be used directly as a
	// bind-group entry `offset` without further rounding.
	offset: number;
	type: number;
	ne: [number, number, number, number];
	nb: [number, number, number, number];
}

export interface JsepOpDescriptor {
	op: number;
	nSrc: number;
	dst: JsepTensorMeta;
	srcs: JsepTensorMeta[];
}

// Stage 4.15 Probe 5 — captured per-divert-dispatch readback entry.
// Populated when `globalThis.__stage415DivertProbe` is truthy. Lives in
// `globalThis.__stage415DivertLog` (cap 32). The mapAsyncs resolve after
// `dispatchMatmul` returns, so consumers must wait ~200ms after DONE
// before inspecting `tempBytes` / `dstBytes`.
interface Stage415DivertEntry {
	divertIdx: number;
	dstHandle: number;
	dstOffset: number;
	dstNe: [number, number, number, number];
	src0Ne: [number, number, number, number];
	src1Ne: [number, number, number, number];
	tempBytes: Uint8Array | null;
	dstBytes: Uint8Array | null;
}

// Stage 4.22 Probe 10 — one-shot capture of the actual src0/src1/dst bytes
// the kernel sees at the first eligible JSEP MUL_MAT dispatch in production
// prefill. Armed by the spike harness (`globalThis.__probe10Capture.armed
// = true`) just before `bridge.decode`, fires only once (auto-disarms),
// and only matches Q4_0 src0 dispatches in the divert path. Captures into
// staging buffers via separate command encoders submitted before/after the
// kernel encoder so dst-before / dst-after are unambiguous. The mapAsync
// promises resolve after `dispatchMatmul` returns; consumers must `await
// runtime.device.queue.onSubmittedWorkDone()` then poll the result fields.
export interface Probe10CaptureResult {
	M: number;
	K: number;
	N: number;
	src0Type: number;
	src0Bytes: Uint8Array | null;
	src1Bytes: Uint8Array | null;
	dstBeforeBytes: Uint8Array | null;
	dstAfterBytes: Uint8Array | null;
}

interface Probe10Global {
	armed: boolean;
	result: Probe10CaptureResult | null;
}

export interface JsepOpContext {
	device: GPUDevice;
	dataManager: GpuDataManager;
	encoderBatcher: CommandEncoderBatcher;
	pipelineCache: PipelineCache;
	// Bind-group layouts, memoized per pipeline cache key. Owned by the
	// runtime (not module scope) so each `JsepRuntime` / `GPUDevice` gets
	// its own cache; reusing a layout across devices is a WebGPU error.
	bindGroupLayoutCache: Map<string, GPUBindGroupLayout>;
}

/**
 * Decode the descriptor at `byteOffset` into a structured form. The
 * caller passes a fresh Int32Array view of `module.HEAPU8.buffer` for
 * heap-grow safety — never cache HEAP32 across awaits.
 */
export function readDescriptor(
	heap32: Int32Array,
	byteOffset: number,
): JsepOpDescriptor {
	const base = byteOffset >> 2;
	const op = heap32[base];
	const nSrc = heap32[base + 1];

	const readBlock = (slot: number): JsepTensorMeta => {
		const off = base + slot;
		return {
			bufHandle: heap32[off],
			offset: heap32[off + 1],
			type: heap32[off + 2],
			// We use the low half; high half should be zero on wasm32.
			ne: [heap32[off + 3], heap32[off + 4], heap32[off + 5], heap32[off + 6]],
			nb: [
				heap32[off + 11],
				heap32[off + 12],
				heap32[off + 13],
				heap32[off + 14],
			],
		};
	};

	const dst = readBlock(2);
	const srcs: JsepTensorMeta[] = [];
	for (let i = 0; i < nSrc; ++i) {
		srcs.push(readBlock(2 + TENSOR_BLOCK_I32 * (1 + i)));
	}
	return { op, nSrc, dst, srcs };
}

// Human-readable dtype name for cache-key construction. Unknown types
// are stamped with their numeric id so misroutes surface as a missing
// kernel rather than a silent collision.
function typeName(t: number): string {
	switch (t) {
		case GGML_TYPE_F32:
			return "f32";
		case GGML_TYPE_F16:
			return "f16";
		case GGML_TYPE_Q4_0:
			return "q4_0";
		case GGML_TYPE_Q4_K:
			return "q4_k";
		default:
			return `t${t}`;
	}
}

/**
 * WGSL kernel selector. Returns a self-contained compute shader for the
 * (src0_type, src1_type, dst_type) combination.
 *
 * Bind layout (all kernels share):
 *   @binding(0) src0 — storage<read>  (u32 array; reinterpret per dtype)
 *   @binding(1) src1 — storage<read>  f32 array
 *   @binding(2) dst  — storage<read_write> f32 array
 *   @binding(3) shape — uniform { M, K, N, batch_count, src0_stride1_bytes,
 *                                  src1_stride1_bytes, dst_stride1_bytes,
 *                                  src0_stride2_bytes, src1_stride2_bytes,
 *                                  dst_stride2_bytes, _pad0, _pad1 } (12 u32)
 *
 * Stage 4.25 Probe 13 — when `kahan=true` (only honored for Q4_K), the
 * intra-thread K-loop accumulator uses Neumaier-corrected summation. The
 * dispatch gate (`__stage425KahanArm` global, shape match Qcur-0 only)
 * keeps the variant kernel out of every other dispatch's pipeline cache.
 */
function buildMatmulShader(
	src0Type: number,
	src1Type: number,
	dstType: number,
	kahan = false,
): string {
	if (src1Type !== GGML_TYPE_F32 || dstType !== GGML_TYPE_F32) {
		throw new Error(
			`buildMatmulShader: unsupported (src1=${src1Type}, dst=${dstType}); ` +
				`Phase 2 requires src1=F32, dst=F32`,
		);
	}

	const HEADER = `
struct Shape {
    M: u32,
    K: u32,
    N: u32,
    batch_count: u32,
    src0_row_bytes: u32,
    src1_row_bytes: u32,
    dst_row_bytes: u32,
    src0_batch_bytes: u32,
    src1_batch_bytes: u32,
    dst_batch_bytes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;
`;

	switch (src0Type) {
		case GGML_TYPE_F32:
			// src0 stored as f32; reinterpret u32 → f32.
			return `${HEADER}
fn load_src0(m: u32, k: u32, batch: u32) -> f32 {
    let bytes_per_elem: u32 = 4u;
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let byte_off: u32 = batch * batch_bytes + m * row_bytes + k * bytes_per_elem;
    let word_idx: u32 = byte_off / 4u;
    return bitcast<f32>(src0[word_idx]);
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_src0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;

		case GGML_TYPE_F16:
			// src0 stored as f16; pack 2 f16 per u32. Use unpack2x16float.
			return `${HEADER}
fn load_src0(m: u32, k: u32, batch: u32) -> f32 {
    let bytes_per_elem: u32 = 2u;
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let byte_off: u32 = batch * batch_bytes + m * row_bytes + k * bytes_per_elem;
    let word_idx: u32 = byte_off / 4u;
    let pair: vec2<f32> = unpack2x16float(src0[word_idx]);
    // Lane within the u32 word (0 or 1) — 2 f16 per u32.
    let lane: u32 = (byte_off / 2u) & 1u;
    if (lane == 0u) {
        return pair.x;
    } else {
        return pair.y;
    }
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_src0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;

		case GGML_TYPE_Q4_0:
			// Q4_0: 32-elem blocks; 18 bytes per block (1 f16 scale + 16
			// nibble bytes). One block stores 32 quantized values; nibbles
			// are stored in-place: byte[i] = (q[i+16] << 4) | q[i] for
			// i in 0..16. Dequant: x = (nibble - 8) * scale.
			return `${HEADER}
const QK4_0: u32 = ${QK4_0}u;
const Q4_0_BYTES_PER_BLOCK: u32 = 18u;

fn load_q4_0(m: u32, k: u32, batch: u32) -> f32 {
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let row_byte_base: u32 = batch * batch_bytes + m * row_bytes;
    let block_idx: u32 = k / QK4_0;
    let in_block: u32 = k % QK4_0;
    let block_byte_base: u32 = row_byte_base + block_idx * Q4_0_BYTES_PER_BLOCK;

    // Scale: f16 at block_byte_base..+2.
    let scale_word_idx: u32 = block_byte_base / 4u;
    let scale_byte_lane: u32 = (block_byte_base / 2u) & 1u;
    let scale_pair: vec2<f32> = unpack2x16float(src0[scale_word_idx]);
    let scale: f32 = select(scale_pair.y, scale_pair.x, scale_byte_lane == 0u);

    // Nibble byte at offset (2 + (in_block % 16)); nibble lane is
    // (0 = low, 1 = high) selected by in_block / 16.
    let nibble_byte_off: u32 = block_byte_base + 2u + (in_block % 16u);
    let nibble_word_idx: u32 = nibble_byte_off / 4u;
    let nibble_byte_lane: u32 = nibble_byte_off & 3u;
    let raw_byte: u32 = (src0[nibble_word_idx] >> (nibble_byte_lane * 8u)) & 0xffu;
    let nibble: u32 = select(raw_byte >> 4u, raw_byte & 0xfu, in_block < 16u);
    return (f32(nibble) - 8.0) * scale;
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_q4_0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;

		case GGML_TYPE_Q4_K: {
			// Q4_K: 256-elem super-blocks; 144 bytes per super-block layout
			// (matches ggml-common.h::block_q4_K):
			//   bytes [0..2]   d     (f16 super-block scale)
			//   bytes [2..4]   dmin  (f16 super-block min)
			//   bytes [4..16]  scales[12] (6-bit packed sc + m for 8 sub-blocks)
			//   bytes [16..144] qs[128]   (4-bit nibbles, 2/byte)
			// 8 sub-blocks of 32 elements; pairs share a 32-byte qs slice
			// (low nibble = first 32, high nibble = second 32). Per element
			//   value = d * sc[is] * nibble - dmin * m[is]
			// where (sc[is], m[is]) come from get_scale_min_k4 (6-bit unpack).
			return `${HEADER}
const QK_K: u32 = ${QK_K}u;
const Q4_K_BYTES_PER_BLOCK: u32 = ${Q4_K_BYTES_PER_BLOCK}u;

fn q4k_byte_at(byte_off: u32) -> u32 {
    let word_idx: u32 = byte_off / 4u;
    let lane: u32 = byte_off & 3u;
    return (src0[word_idx] >> (lane * 8u)) & 0xffu;
}

// Mirrors ggml-quants.c::get_scale_min_k4. Returns (sc, m) for sub-block
// index "is" in [0, 8). "scales_byte_base" points to scales[0].
fn q4k_unpack_scale_min(scales_byte_base: u32, is: u32) -> vec2<u32> {
    if (is < 4u) {
        let sc: u32 = q4k_byte_at(scales_byte_base + is) & 63u;
        let m: u32 = q4k_byte_at(scales_byte_base + is + 4u) & 63u;
        return vec2<u32>(sc, m);
    }
    // is in [4, 8): pull lower 4 bits from q[is+4]/q[is+4]>>4 and high 2
    // bits from q[is-4]/q[is] respectively.
    let q_a: u32 = q4k_byte_at(scales_byte_base + is + 4u);
    let q_b: u32 = q4k_byte_at(scales_byte_base + is - 4u);
    let q_c: u32 = q4k_byte_at(scales_byte_base + is);
    let sc: u32 = (q_a & 0xfu) | ((q_b >> 6u) << 4u);
    let m: u32 = (q_a >> 4u) | ((q_c >> 6u) << 4u);
    return vec2<u32>(sc, m);
}

fn load_q4_K(m: u32, k: u32, batch: u32) -> f32 {
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let row_byte_base: u32 = batch * batch_bytes + m * row_bytes;
    let super_block_idx: u32 = k / QK_K;
    let in_super: u32 = k % QK_K;
    let block_byte_base: u32 = row_byte_base + super_block_idx * Q4_K_BYTES_PER_BLOCK;

    // d (f16) + dmin (f16) live in the first 4 bytes of the super-block.
    // block_byte_base is 4-aligned (rows hold whole 144-byte blocks; row
    // and batch strides are byte-multiples of 144, hence of 4).
    let dm: vec2<f32> = unpack2x16float(src0[block_byte_base / 4u]);
    let d: f32 = dm.x;
    let dmin: f32 = dm.y;

    let scales_byte_base: u32 = block_byte_base + 4u;
    let qs_byte_base: u32 = block_byte_base + 16u;

    // Sub-block addressing — see ggml-quants.c::dequantize_row_q4_K.
    let pair: u32 = in_super / 64u;            // 0..3
    let within_pair: u32 = in_super % 64u;     // 0..63
    let l: u32 = within_pair % 32u;            // element within half-pair
    let is: u32 = pair * 2u + select(0u, 1u, within_pair >= 32u);
    let q_byte_idx: u32 = pair * 32u + l;

    let raw_byte: u32 = q4k_byte_at(qs_byte_base + q_byte_idx);
    let nibble: u32 = select(raw_byte >> 4u, raw_byte & 0xfu, within_pair < 32u);

    let scm: vec2<u32> = q4k_unpack_scale_min(scales_byte_base, is);
    let sc: f32 = f32(scm.x);
    let m_min: f32 = f32(scm.y);

    return d * sc * f32(nibble) - dmin * m_min;
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    ${kahan ? "var compensation: f32 = 0.0;" : ""}
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_q4_K(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        ${
					kahan
						? `// Neumaier-Kahan compensated summation. Recovers ~1 extra
        // f32 mantissa bit per step over a length-K reduction by tracking
        // the lost-low-order term in 'compensation'. WebGPU has no f64,
        // so the correction lives entirely in f32 — single-step Kahan
        // recovers the per-add rounding loss; cascading multiple
        // consecutive losses (e.g. matched magnitudes) needs Neumaier's
        // variant which we use here (always-keep-larger).
        let term: f32 = a * b;
        let t: f32 = acc + term;
        if (abs(acc) >= abs(term)) {
            compensation = compensation + ((acc - t) + term);
        } else {
            compensation = compensation + ((term - t) + acc);
        }
        acc = t;`
						: "acc = acc + a * b;"
				}
    }
    ${kahan ? "acc = acc + compensation;" : ""}
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;
		}

		default:
			throw new Error(`matmul: unsupported src0 type ${src0Type}`);
	}
}

const SHAPE_UNIFORM_BYTES = 12 * 4; // 12 u32 slots

function buildPipeline(
	device: GPUDevice,
	src0Type: number,
	src1Type: number,
	dstType: number,
	kahan = false,
): GPUComputePipeline {
	const wgsl = buildMatmulShader(src0Type, src1Type, dstType, kahan);
	const shaderModule = device.createShaderModule({ code: wgsl });
	const layout = device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: "read-only-storage" },
			},
			{
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: "read-only-storage" },
			},
			{
				binding: 2,
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: "storage" },
			},
			{
				binding: 3,
				visibility: GPUShaderStage.COMPUTE,
				buffer: { type: "uniform" },
			},
		],
	});
	const pipelineLayout = device.createPipelineLayout({
		bindGroupLayouts: [layout],
	});
	return device.createComputePipeline({
		layout: pipelineLayout,
		compute: { module: shaderModule, entryPoint: "main" },
	});
}

/**
 * Dispatch a MUL_MAT op described by `desc`. Returns 0 on success,
 * negative on validation failure.
 */
export function dispatchMatmul(
	ctx: JsepOpContext,
	desc: JsepOpDescriptor,
): number {
	if (desc.nSrc !== 2) {
		console.error(`dispatchMatmul: expected 2 srcs, got ${desc.nSrc}`);
		return -1;
	}

	const src0 = desc.srcs[0];
	const src1 = desc.srcs[1];
	const dst = desc.dst;

	if (src1.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
		console.error(
			`dispatchMatmul: only src1=F32, dst=F32 in Phase 2 ` +
				`(got src1=${src1.type}, dst=${dst.type})`,
		);
		return -1;
	}

	// ggml MUL_MAT shape: src0=[K, M], src1=[K, N], dst=[M, N].
	const K = src0.ne[0];
	const M = src0.ne[1];
	const N = src1.ne[1];
	if (src1.ne[0] !== K || dst.ne[0] !== M || dst.ne[1] !== N) {
		console.error(
			`dispatchMatmul: shape mismatch — src0=[${src0.ne.join(",")}], ` +
				`src1=[${src1.ne.join(",")}], dst=[${dst.ne.join(",")}]`,
		);
		return -1;
	}

	// Batch product over ne[2] * ne[3]. For Phase 2 most workloads are
	// batch=1; we still compute it generally.
	const batchCount = Math.max(1, src1.ne[2]) * Math.max(1, src1.ne[3]);

	const ndim = batchCount > 1 ? 3 : 2;

	// Stage 4.25 Probe 13 — gated Kahan-summed accumulator path. Armed by
	// the spike via `globalThis.__stage425KahanArm`; auto-disarms after
	// first eligible dispatch so only Qcur-0 layer 0 (M=2048, K=2048, N=6,
	// Q4_K) takes the variant kernel. The variant kernel uses a separate
	// pipeline cache key (`-kahan` suffix) so the production kernel cache
	// is unchanged for every other dispatch.
	const kahanGlobal = globalThis as unknown as {
		__stage425KahanArm?: boolean;
	};
	let useKahan = false;
	if (
		kahanGlobal.__stage425KahanArm &&
		src0.type === GGML_TYPE_Q4_K &&
		M === 2048 &&
		K === 2048 &&
		N === 6
	) {
		useKahan = true;
		kahanGlobal.__stage425KahanArm = false;
		// One-shot confirmation that the Kahan dispatch fired. Read by the
		// spike's Probe 13 verdict block to disambiguate "Kahan ran and
		// produced identical output" from "Kahan gate never fired".
		(
			globalThis as unknown as { __stage425KahanFired?: boolean }
		).__stage425KahanFired = true;
	}

	const cacheKey = `mat-${typeName(src0.type)}-${typeName(src1.type)}-${typeName(dst.type)}-${ndim}${useKahan ? "-kahan" : ""}`;

	const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
		const p = buildPipeline(device, src0.type, src1.type, dst.type, useKahan);
		// Recover the bindGroupLayout from the pipeline (createComputePipeline
		// owns it via the explicit layout we set).
		ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
		return p;
	});
	const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
	if (!bindGroupLayout) {
		console.error(`dispatchMatmul: missing bindGroupLayout for ${cacheKey}`);
		return -1;
	}

	// FIXME(phase 3): the shape uniform buffer is allocated fresh per
	// dispatch and never destroyed — `shapeBuffer` leaks until the
	// `GPUDevice` is destroyed. Intended fix: cache shape uniforms by
	// literal (M,K,N,batch,strides) tuple, or ring-buffer a fixed pool of
	// uniforms keyed by submit index. Tracked in the Task 7 closure
	// follow-ups / next-session brief.
	const shapeBuffer = ctx.device.createBuffer({
		size: SHAPE_UNIFORM_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	const shapeData = new Uint32Array(SHAPE_UNIFORM_BYTES / 4);
	shapeData[0] = M;
	shapeData[1] = K;
	shapeData[2] = N;
	shapeData[3] = batchCount;
	// Row stride in bytes — for src0 this is nb[1]; for src1 the *column*
	// stride is nb[1] (each "row" of src1 is K f32s on axis 0). Same for dst.
	shapeData[4] = src0.nb[1];
	shapeData[5] = src1.nb[1];
	shapeData[6] = dst.nb[1];
	// Batch stride in bytes (axis 2). When ne[2]=1 this is unused but we
	// still pass nb[2] for consistency.
	shapeData[7] = src0.nb[2];
	shapeData[8] = src1.nb[2];
	shapeData[9] = dst.nb[2];
	shapeData[10] = 0;
	shapeData[11] = 0;
	ctx.device.queue.writeBuffer(shapeBuffer, 0, shapeData);

	const src0Rec = ctx.dataManager.get(src0.bufHandle);
	const src1Rec = ctx.dataManager.get(src1.bufHandle);
	const dstRec = ctx.dataManager.get(dst.bufHandle);

	const dispatchX = Math.ceil(M / TILE_M);
	const dispatchY = Math.ceil(N / TILE_N);
	const dispatchZ = batchCount;

	// WebGPU sync-scope rule: a single GPUBuffer bound as both read-only
	// storage and writable storage in the same compute pass is rejected
	// ("[Buffer] usage (Storage(read-write)|Storage(read-only)) includes
	// writable usage and another usage in the same synchronization
	// scope"). Even non-overlapping byte ranges trip the rule — the
	// validator works at buffer granularity.
	//
	// The libllama scheduler packs activation tensors and matmul
	// intermediates into a single jsep_buf to save memory, so dst
	// routinely aliases src1 (sometimes src0). Without the divert below
	// every aliased dispatch is silently dropped at encoder.finish() and
	// dst stays at its previous (often zero) state — the Outcome C
	// "all-zero logits" failure mode that gated Stage 3.
	//
	// Divert: when dst.bufHandle equals any src bufHandle, allocate a
	// fresh temp GPUBuffer of size batchCount*N*M*4 (f32 contiguous
	// matmul output), bind that as the kernel's dst at offset 0, then
	// copyBufferToBuffer back into dstRec at dst.offset after the
	// dispatch. The diverted dispatch lives in its own command-encoder
	// (flush the batcher first) so unrelated batched dispatches can't
	// claim the same buffer pair in the same pass.
	const dstAliasesSrc =
		dst.bufHandle === src0.bufHandle || dst.bufHandle === src1.bufHandle;

	if (dstAliasesSrc) {
		// Contiguity assertion — divert assumes dst is f32-contiguous
		// (nb[1]=M*4, nb[2]=N*M*4). matmul output from libllama's
		// scheduler is always contiguous, but guard against future
		// callers that might pass strided dst.
		const expectedRowBytes = M * 4;
		const expectedBatchBytes = N * M * 4;
		if (
			dst.nb[1] !== expectedRowBytes ||
			(batchCount > 1 && dst.nb[2] !== expectedBatchBytes)
		) {
			console.error(
				`dispatchMatmul: aliased dst is non-contiguous (nb=[${dst.nb.join(",")}], ` +
					`expected row=${expectedRowBytes}, batch=${expectedBatchBytes}); ` +
					`divert path requires contiguous dst.`,
			);
			return -1;
		}

		const dstBytesNeeded = batchCount * N * M * 4;
		const tempDst = ctx.device.createBuffer({
			size: dstBytesNeeded,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		// Stage 4.22 Probe 10 — gated one-shot capture of the actual bytes
		// the kernel sees. Gates on quantized (Q4_0 or Q4_K) src0 to avoid
		// matching f32 MUL_MAT selftests but cover the actual production
		// weight type — TinyLlama-Q4_0.gguf misleadingly stores Q4_K
		// (type 12) tensors despite the filename. The auto-disarm ensures
		// only the first eligible dispatch fires.
		const probe10Global = globalThis as unknown as {
			__probe10Capture?: Probe10Global;
		};
		const probe10 = probe10Global.__probe10Capture;
		let probe10DstAfterStaging: GPUBuffer | null = null;
		let probe10Result: Probe10CaptureResult | null = null;
		let probe10DstSize = 0;
		if (
			probe10?.armed &&
			(src0.type === GGML_TYPE_Q4_0 || src0.type === GGML_TYPE_Q4_K)
		) {
			probe10.armed = false;
			const src0RowBytes = src0.nb[1];
			const src0Size = M * src0RowBytes;
			const src1Size = batchCount * N * K * 4;
			probe10DstSize = dstBytesNeeded;

			const src0Staging = ctx.device.createBuffer({
				size: src0Size,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
			const src1Staging = ctx.device.createBuffer({
				size: src1Size,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
			const dstBeforeStaging = ctx.device.createBuffer({
				size: probe10DstSize,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
			probe10Result = {
				M,
				K,
				N,
				src0Type: src0.type,
				src0Bytes: null,
				src1Bytes: null,
				dstBeforeBytes: null,
				dstAfterBytes: null,
			};
			probe10.result = probe10Result;

			// PRE-encoder — copy src0/src1/dst-before into staging BEFORE
			// the kernel encoder runs. Submission order on a single queue
			// is FIFO, so the kernel encoder submitted next will execute
			// after these copies.
			const preEnc = ctx.device.createCommandEncoder();
			preEnc.copyBufferToBuffer(
				src0Rec.buffer,
				src0.offset,
				src0Staging,
				0,
				src0Size,
			);
			preEnc.copyBufferToBuffer(
				src1Rec.buffer,
				src1.offset,
				src1Staging,
				0,
				src1Size,
			);
			preEnc.copyBufferToBuffer(
				dstRec.buffer,
				dst.offset,
				dstBeforeStaging,
				0,
				probe10DstSize,
			);
			ctx.device.queue.submit([preEnc.finish()]);
			const result = probe10Result;
			void Promise.all([
				src0Staging.mapAsync(GPUMapMode.READ, 0, src0Size),
				src1Staging.mapAsync(GPUMapMode.READ, 0, src1Size),
				dstBeforeStaging.mapAsync(GPUMapMode.READ, 0, probe10DstSize),
			]).then(() => {
				result.src0Bytes = new Uint8Array(
					src0Staging.getMappedRange().slice(0),
				);
				result.src1Bytes = new Uint8Array(
					src1Staging.getMappedRange().slice(0),
				);
				result.dstBeforeBytes = new Uint8Array(
					dstBeforeStaging.getMappedRange().slice(0),
				);
				src0Staging.unmap();
				src1Staging.unmap();
				dstBeforeStaging.unmap();
				src0Staging.destroy();
				src1Staging.destroy();
				dstBeforeStaging.destroy();
			});

			probe10DstAfterStaging = ctx.device.createBuffer({
				size: probe10DstSize,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			});
		}
		const divertBindGroup = ctx.device.createBindGroup({
			layout: bindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: {
						buffer: src0Rec.buffer,
						offset: src0.offset,
						size: src0Rec.size - src0.offset,
					},
				},
				{
					binding: 1,
					resource: {
						buffer: src1Rec.buffer,
						offset: src1.offset,
						size: src1Rec.size - src1.offset,
					},
				},
				{
					binding: 2,
					resource: { buffer: tempDst, offset: 0, size: dstBytesNeeded },
				},
				{ binding: 3, resource: { buffer: shapeBuffer } },
			],
		});

		// Flush any pending batched dispatches — the diverted matmul +
		// its CPY-back must live in a self-contained encoder.
		ctx.encoderBatcher.flush();

		const enc = ctx.device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, divertBindGroup);
		pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
		pass.end();
		enc.copyBufferToBuffer(
			tempDst,
			0,
			dstRec.buffer,
			dst.offset,
			dstBytesNeeded,
		);
		ctx.device.queue.submit([enc.finish()]);
		// Stage 4.22 Probe 10 — POST-kernel: encode + submit dst-after
		// copy. Submission order is FIFO so this runs after the kernel
		// encoder's copyBufferToBuffer landed dst back into dstRec.buffer.
		if (probe10DstAfterStaging && probe10Result) {
			const postEnc = ctx.device.createCommandEncoder();
			postEnc.copyBufferToBuffer(
				dstRec.buffer,
				dst.offset,
				probe10DstAfterStaging,
				0,
				probe10DstSize,
			);
			ctx.device.queue.submit([postEnc.finish()]);
			const stagingAfter = probe10DstAfterStaging;
			const resultAfter = probe10Result;
			void stagingAfter
				.mapAsync(GPUMapMode.READ, 0, probe10DstSize)
				.then(() => {
					resultAfter.dstAfterBytes = new Uint8Array(
						stagingAfter.getMappedRange().slice(0),
					);
					stagingAfter.unmap();
					stagingAfter.destroy();
				});
		}
		// Stage 4.15 Probe 5 — gated per-divert-dispatch readback of
		// (a) tempDst[0..16) and (b) dstRec.buffer[dst.offset..+16) to
		// disambiguate kernel-bug vs copy-bug vs handle-mismatch.
		const probeGlobal = globalThis as unknown as {
			__stage415DivertProbe?: boolean;
			__stage415DivertLog?: Stage415DivertEntry[];
		};
		if (probeGlobal.__stage415DivertProbe) {
			if (!probeGlobal.__stage415DivertLog) {
				probeGlobal.__stage415DivertLog = [];
			}
			const dbg = probeGlobal.__stage415DivertLog;
			if (dbg.length < 32) {
				const tempStaging = ctx.device.createBuffer({
					size: 16,
					usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
				});
				const dstStaging = ctx.device.createBuffer({
					size: 16,
					usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
				});
				const probeEnc = ctx.device.createCommandEncoder();
				probeEnc.copyBufferToBuffer(tempDst, 0, tempStaging, 0, 16);
				probeEnc.copyBufferToBuffer(
					dstRec.buffer,
					dst.offset,
					dstStaging,
					0,
					16,
				);
				ctx.device.queue.submit([probeEnc.finish()]);
				const meta: Stage415DivertEntry = {
					divertIdx: dbg.length,
					dstHandle: dst.bufHandle,
					dstOffset: dst.offset,
					dstNe: [dst.ne[0], dst.ne[1], dst.ne[2], dst.ne[3]],
					src0Ne: [src0.ne[0], src0.ne[1], src0.ne[2], src0.ne[3]],
					src1Ne: [src1.ne[0], src1.ne[1], src1.ne[2], src1.ne[3]],
					tempBytes: null,
					dstBytes: null,
				};
				dbg.push(meta);
				void Promise.all([
					tempStaging.mapAsync(GPUMapMode.READ),
					dstStaging.mapAsync(GPUMapMode.READ),
				]).then(() => {
					meta.tempBytes = new Uint8Array(
						tempStaging.getMappedRange().slice(0),
					);
					meta.dstBytes = new Uint8Array(dstStaging.getMappedRange().slice(0));
					tempStaging.unmap();
					dstStaging.unmap();
					tempStaging.destroy();
					dstStaging.destroy();
				});
			}
		}
		// destroy() after submit is documented-safe — pending GPU work is
		// allowed to complete using the destroyed buffer's underlying
		// memory.
		tempDst.destroy();
		return 0;
	}

	const bindGroup = ctx.device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: {
					buffer: src0Rec.buffer,
					offset: src0.offset,
					size: src0Rec.size - src0.offset,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: src1Rec.buffer,
					offset: src1.offset,
					size: src1Rec.size - src1.offset,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: dstRec.buffer,
					offset: dst.offset,
					size: dstRec.size - dst.offset,
				},
			},
			{ binding: 3, resource: { buffer: shapeBuffer } },
		],
	});

	ctx.encoderBatcher.record({
		pipeline,
		bindGroup,
		dispatchX,
		dispatchY,
		dispatchZ,
	});

	return 0;
}
