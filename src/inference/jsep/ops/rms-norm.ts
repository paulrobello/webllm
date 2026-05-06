/**
 * JSEP rms_norm kernel — F32 path.
 *
 * Reads the per-op descriptor packed C++-side in
 * `ggml/src/ggml-jsep/ggml-jsep.cpp::ggml_backend_jsep_graph_compute`
 * and dispatches a single compute pass via the `CommandEncoderBatcher`.
 * Pipelines are cached on dtype (single key for Phase 2 since only F32
 * is supported).
 *
 * ggml RMS_NORM semantics:
 *   src0:    [last_dim, n_rows, ...]  — input.
 *   src1:    [last_dim]               — per-channel weight (gain).
 *   dst:     same shape as src0.
 *   op_params[0] (f32): eps.
 *
 *   per-row formula: out[j] = (x[j] / sqrt(mean(x²) + eps)) * w[j]
 *                    where mean(x²) = sum_j(x[j]²) / last_dim
 *
 * Phase 2 ships the F32→F32 path only. The kernel mirrors
 * `SHADER_RMS_NORM` from `wgsl-shaders.ts` faithfully (each thread
 * recomputes the row sum independently — wasteful but correct, matches
 * the existing canonical implementation). A future phase can add a
 * shared-memory tree reduction.
 */

import {
	GGML_TYPE_F32,
	type JsepOpContext,
	type JsepOpDescriptor,
} from "./matmul.js";

// Workgroup geometry: 1 row × 256 cols per workgroup. Mirrors
// `SHADER_RMS_NORM` in `src/inference/wgsl-shaders.ts`. For typical
// last_dim values (≤ 8192) this dispatches at most 32 workgroups along
// the column axis per row — fine.
const WG_X = 1;
const WG_Y = 256;

const SHAPE_UNIFORM_BYTES = 4 * 4; // rows (u32), cols (u32), eps (f32), _pad (u32)

const RMS_NORM_WGSL = /* wgsl */ `
struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row: u32 = gid.x;
    let col: u32 = gid.y;
    if (row >= params.rows || col >= params.cols) { return; }

    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
        let v: f32 = x[row * params.cols + i];
        sum_sq = sum_sq + v * v;
    }
    let inv_rms: f32 = inverseSqrt(sum_sq / f32(params.cols) + params.eps);
    let idx: u32 = row * params.cols + col;
    out[idx] = x[idx] * weight[col] * inv_rms;
}
`;

function buildPipeline(device: GPUDevice): GPUComputePipeline {
	const shaderModule = device.createShaderModule({ code: RMS_NORM_WGSL });
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
 * Dispatch an RMS_NORM op described by `desc`. Returns 0 on success,
 * negative on validation failure.
 *
 * `opParamsPtr` points at `node->op_params` in the WASM heap; the first
 * f32 (4 bytes) is `eps`.
 */
export function dispatchRmsNorm(
	ctx: JsepOpContext,
	desc: JsepOpDescriptor,
	opParamsPtr: number,
	heapBuffer: ArrayBufferLike,
): number {
	if (desc.nSrc !== 2) {
		console.error(`dispatchRmsNorm: expected 2 srcs, got ${desc.nSrc}`);
		return -1;
	}

	const src0 = desc.srcs[0];
	const src1 = desc.srcs[1];
	const dst = desc.dst;

	if (
		src0.type !== GGML_TYPE_F32 ||
		src1.type !== GGML_TYPE_F32 ||
		dst.type !== GGML_TYPE_F32
	) {
		console.error(
			`dispatchRmsNorm: only F32 path supported in Phase 2 ` +
				`(got src0=${src0.type}, src1=${src1.type}, dst=${dst.type})`,
		);
		return -1;
	}

	// Read eps as f32 from op_params[0]. Re-derive the heap view each
	// call — the WASM heap may have grown between EM_ASM frames.
	const eps = new Float32Array(heapBuffer, opParamsPtr, 1)[0];

	const cols = src0.ne[0];
	const rows = src0.ne[1] * Math.max(1, src0.ne[2]) * Math.max(1, src0.ne[3]);

	if (src1.ne[0] !== cols) {
		console.error(
			`dispatchRmsNorm: weight length ${src1.ne[0]} != cols ${cols}`,
		);
		return -1;
	}
	if (
		dst.ne[0] !== src0.ne[0] ||
		dst.ne[1] !== src0.ne[1] ||
		dst.ne[2] !== src0.ne[2] ||
		dst.ne[3] !== src0.ne[3]
	) {
		console.error(
			`dispatchRmsNorm: shape mismatch — src0=[${src0.ne.join(",")}], ` +
				`dst=[${dst.ne.join(",")}]`,
		);
		return -1;
	}

	const cacheKey = "rms-norm-f32";

	const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
		const p = buildPipeline(device);
		// Recover the bindGroupLayout from the pipeline (createComputePipeline
		// owns it via the explicit layout we set).
		ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
		return p;
	});
	const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
	if (!bindGroupLayout) {
		console.error(`dispatchRmsNorm: missing bindGroupLayout for ${cacheKey}`);
		return -1;
	}

	// FIXME(phase 3): like dispatchMatmul, this allocates a fresh uniform
	// buffer per dispatch and never destroys it — `shapeBuffer` leaks
	// until the `GPUDevice` is destroyed. Same fix as matmul: cache by
	// (rows, cols, eps) tuple or ring-buffer a fixed pool keyed by
	// submit index. Tracked in the Task 7 closure follow-ups.
	const shapeBuffer = ctx.device.createBuffer({
		size: SHAPE_UNIFORM_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	// Pack (u32, u32, f32, u32) — write u32 slots first, then bitcast eps
	// in via Float32Array view sharing the underlying buffer.
	const shapeU32 = new Uint32Array(SHAPE_UNIFORM_BYTES / 4);
	shapeU32[0] = rows;
	shapeU32[1] = cols;
	shapeU32[3] = 0;
	new Float32Array(shapeU32.buffer)[2] = eps;
	ctx.device.queue.writeBuffer(shapeBuffer, 0, shapeU32);

	const xBuf = ctx.dataManager.get(src0.handle).buffer;
	const wBuf = ctx.dataManager.get(src1.handle).buffer;
	const outBuf = ctx.dataManager.get(dst.handle).buffer;

	const bindGroup = ctx.device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: xBuf } },
			{ binding: 1, resource: { buffer: wBuf } },
			{ binding: 2, resource: { buffer: outBuf } },
			{ binding: 3, resource: { buffer: shapeBuffer } },
		],
	});

	const dispatchX = Math.ceil(rows / WG_X);
	const dispatchY = Math.ceil(cols / WG_Y);
	const dispatchZ = 1;

	ctx.encoderBatcher.record({
		pipeline,
		bindGroup,
		dispatchX,
		dispatchY,
		dispatchZ,
	});

	return 0;
}
