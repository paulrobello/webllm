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
 * Descriptor layout (must match `ggml-jsep.cpp` — see top of that file):
 *
 *   per-tensor block (18 i32 = 72 bytes):
 *     [0]      handle             (i32; from jsep_tensor_handle())
 *     [1]      type               (i32; ggml_type enum value)
 *     [2..5]   ne[0..3] low half  (i32; ne is int64_t)
 *     [6..9]   ne[0..3] high half (i32; zero on wasm32 with sub-2GB tensors)
 *     [10..13] nb[0..3] low half  (i32; nb is size_t)
 *     [14..17] nb[0..3] high half (i32; zero on wasm32)
 *
 *   descriptor (offset 0 in i32 words):
 *     [0]      op                 (i32; ggml_op enum)
 *     [1]      n_src              (i32)
 *     [2..19]  dst block          (18 i32)
 *     [20..37] src[0] block       (18 i32)
 *     [38..55] src[1] block       (18 i32)
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
const TENSOR_BLOCK_I32 = 18;

// Workgroup tile size for the matmul kernel: TILE_M output rows × TILE_N
// output cols per workgroup. Each invocation computes one output element.
const TILE_M = 16;
const TILE_N = 16;

// Q4_0 block geometry (must match ggml-common.h; QK4_0 == 32).
const QK4_0 = 32;

export interface JsepTensorMeta {
	handle: number;
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
			handle: heap32[off],
			type: heap32[off + 1],
			// We use the low half; high half should be zero on wasm32.
			ne: [heap32[off + 2], heap32[off + 3], heap32[off + 4], heap32[off + 5]],
			nb: [
				heap32[off + 10],
				heap32[off + 11],
				heap32[off + 12],
				heap32[off + 13],
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
 */
function buildMatmulShader(
	src0Type: number,
	src1Type: number,
	dstType: number,
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

		case GGML_TYPE_Q4_K:
			// Q4_K hand-packing is significantly more involved
			// (super-block of 256 elems with 6-bit quantized scales + mins
			// per sub-block). Phase 2 ships F32 + F16 + Q4_0 goldens; Q4_K
			// will be exercised via the real-model browser smoke in Task 7.
			throw new Error(
				"matmul Q4_K kernel: deferred to Task 7 (browser smoke covers via real weights)",
			);

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
): GPUComputePipeline {
	const wgsl = buildMatmulShader(src0Type, src1Type, dstType);
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
	const cacheKey = `mat-${typeName(src0.type)}-${typeName(src1.type)}-${typeName(dst.type)}-${ndim}`;

	const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
		const p = buildPipeline(device, src0.type, src1.type, dst.type);
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

	const src0Buf = ctx.dataManager.get(src0.handle).buffer;
	const src1Buf = ctx.dataManager.get(src1.handle).buffer;
	const dstBuf = ctx.dataManager.get(dst.handle).buffer;

	const bindGroup = ctx.device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: src0Buf } },
			{ binding: 1, resource: { buffer: src1Buf } },
			{ binding: 2, resource: { buffer: dstBuf } },
			{ binding: 3, resource: { buffer: shapeBuffer } },
		],
	});

	const dispatchX = Math.ceil(M / TILE_M);
	const dispatchY = Math.ceil(N / TILE_N);
	const dispatchZ = batchCount;

	ctx.encoderBatcher.record({
		pipeline,
		bindGroup,
		dispatchX,
		dispatchY,
		dispatchZ,
	});

	return 0;
}
