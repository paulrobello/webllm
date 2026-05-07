/**
 * JSEP set_rows kernel — F32 source + I64/I32 indices → F16/F32 dest.
 *
 * Phase 3 / Option A-prime Stage 1. First write op the libllama scheduler
 * hits when KV cache lives in jsep_buf. Without this kernel, decode aborts
 * at sched_reserve with "pre-allocated tensor (cache_k_l0 (view)) in a
 * buffer (jsep_buf) that cannot run the operation (SET_ROWS)".
 *
 * ggml SET_ROWS semantics (`ggml/src/ggml.c::ggml_set_rows`):
 *
 *   src[0] = b   F32 source data,    shape [nc, nr, ne02, ne03]
 *   src[1] = c   I64/I32 indices,    shape [nr, ne11, ne12, 1]
 *   src[2] = a   destination buffer  (the result tensor is `view_tensor(a)`)
 *   dst    = view of a              same shape as a   [ne0, ne1, ne2, ne3]
 *                                    where ne0 == nc, ne2 == ne02, ne3 == ne03
 *
 *   Per-row write (CPU reference in ops.cpp:4904-4951):
 *     i12 = i03 % ne12
 *     i11 = i02 % ne11
 *     i10 = i  (∈ [0, nr))
 *     i1  = c[i10, i11, i12]
 *     dst[*, i1, i02, i03] = from_float(b[*, i, i02, i03])     // nc elements per row
 *
 * KV cache call sites (`llama-kv-cache.cpp:1229,1264,1285`):
 *   1. cache_k row write (every decode step):
 *        src0: F32 [head_dim*n_kv_heads, n_tokens]
 *        src1: I64 [n_tokens]
 *        dst:  F16 [head_dim*n_kv_heads, n_ctx] (view of cache_k_l)
 *   2. cache_v row write (non-transposed, FA-enabled): identical shape.
 *   3. cache_v cell write (transposed, FA-disabled — line 1281):
 *        v_view = ggml_reshape_2d(v, 1, ggml_nelements(v))
 *        src0: F32 [1, n_tokens*n_embd_v_gqa]
 *        src1: I64 [n_tokens*n_embd_v_gqa]
 *        dst:  F16 [1, ggml_nelements(v)]            (view, ne[0]=1)
 *      Indices for this branch (line 1397) target individual F16 cells
 *      that may land in the same u32 word as another thread's target —
 *      adjacent indices i1 and i1+1 share a u32 pair. Naive pair-pack
 *      writes would race. Atomic CAS on each half-word read-modify-
 *      write fixes the race at the cost of a ~1.5x kernel slowdown for
 *      KV writes (negligible vs the model forward).
 *
 * F16 storage: WGSL's `f16` extension requires the `shader-f16` device
 * feature, which the spike device does not request. We declare F16 dst
 * as `array<atomic<u32>>` and pack each F32 source element via
 * `pack2x16float(vec2<f32>(v, 0)) & 0xFFFF`, then atomically merge into
 * the appropriate halfword via a CAS loop.
 *
 * Strides: element-space (in source-type units for src0, in 64-bit
 * indices for I64 / 32-bit for I32, F16 cell units for F16 dst, F32
 * for F32 dst) derived from descriptor nb (byte) values. The kernel
 * assumes stride[0] == element-size (ggml_is_contiguous_rows guarantee).
 */

import {
	GGML_TYPE_F16,
	GGML_TYPE_F32,
	type JsepOpContext,
	type JsepOpDescriptor,
} from "./matmul.js";

const GGML_TYPE_I32 = 26;
const GGML_TYPE_I64 = 27;

// Workgroup size: 64 threads in X across the inner dim. Keeps occupancy
// reasonable on modern WebGPU devices without overrunning subgroup
// counts on small (n_kv_heads*head_dim = 256-ish) rows.
const WG_X = 64;

// Params layout: 16 u32 slots (64 bytes; meets WebGPU's 16-byte
// alignment for vec4 boundaries). Same shape across F16 and F32 dst.
const PARAMS_U32 = 16;
const PARAMS_BYTES = PARAMS_U32 * 4;

// Common shader header — params struct + index resolution helpers
// shared between F16 and F32 dst variants. Strides on the dst are in
// units of the dst element type (f16 cells for F16, f32 cells for F32).
function shaderCommon(): string {
	return /* wgsl */ `
struct Params {
    ne0:        u32, // dst inner dim (== src0 ne[0])
    nr:         u32, // src0 ne[1] = number of source rows
    ne02:       u32,
    ne03:       u32,
    ne11:       u32,
    ne12:       u32,
    src0_s1_f:  u32, // src0 stride[1] in F32 elements
    src0_s2_f:  u32,
    src0_s3_f:  u32,
    src1_s0_p:  u32, // src1 stride[0] in u32 pairs (i64) / u32 (i32)
    src1_s1_p:  u32,
    src1_s2_p:  u32,
    dst_s1_e:   u32, // dst stride[1] in dst-element units
    dst_s2_e:   u32,
    dst_s3_e:   u32,
    idx_is_i64: u32, // 1 if src1 is i64 (read low half of pairs), 0 if i32
};
`;
}

const SET_ROWS_F32_TO_F16_WGSL = `${shaderCommon()}
@group(0) @binding(0) var<storage, read> src0_f: array<f32>;
@group(0) @binding(1) var<storage, read> src1_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_a: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: Params;

fn write_f16_atomic(cell_idx: u32, value_f16_lo: u32) {
    let word_idx: u32 = cell_idx >> 1u;
    let half: u32 = cell_idx & 1u;
    var mask: u32 = 0xFFFF0000u;
    var shifted: u32 = value_f16_lo;
    if (half == 1u) {
        mask = 0x0000FFFFu;
        shifted = value_f16_lo << 16u;
    }
    loop {
        let old: u32 = atomicLoad(&dst_a[word_idx]);
        let nv: u32 = (old & mask) | shifted;
        let r = atomicCompareExchangeWeak(&dst_a[word_idx], old, nv);
        if (r.exchanged) { break; }
    }
}

@compute @workgroup_size(${WG_X}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col: u32       = gid.x; // ∈ [0, ne0)
    let row_idx: u32   = gid.y; // i (∈ [0, nr))
    let batch_lin: u32 = gid.z; // i02 + i03 * ne02

    if (col >= p.ne0) { return; }
    if (row_idx >= p.nr) { return; }
    let total_batches: u32 = p.ne02 * p.ne03;
    if (batch_lin >= total_batches) { return; }

    let i02: u32 = batch_lin % p.ne02;
    let i03: u32 = batch_lin / p.ne02;
    let i11: u32 = i02 % p.ne11;
    let i12: u32 = i03 % p.ne12;

    let idx_pair_off: u32 = row_idx * p.src1_s0_p
                          + i11     * p.src1_s1_p
                          + i12     * p.src1_s2_p;
    var i1: u32;
    if (p.idx_is_i64 == 1u) {
        i1 = src1_u[idx_pair_off * 2u];
    } else {
        i1 = src1_u[idx_pair_off];
    }

    let s_off: u32 = col
                   + row_idx * p.src0_s1_f
                   + i02     * p.src0_s2_f
                   + i03     * p.src0_s3_f;
    let v: f32 = src0_f[s_off];

    // pack2x16float(v, 0) → low 16 bits = f16(v); high 16 bits = f16(0).
    let packed_pair: u32 = pack2x16float(vec2<f32>(v, 0.0));
    let v_f16_lo: u32 = packed_pair & 0xFFFFu;

    let cell_idx: u32 = col
                      + i1  * p.dst_s1_e
                      + i02 * p.dst_s2_e
                      + i03 * p.dst_s3_e;
    write_f16_atomic(cell_idx, v_f16_lo);
}
`;

const SET_ROWS_F32_TO_F32_WGSL = `${shaderCommon()}
@group(0) @binding(0) var<storage, read> src0_f: array<f32>;
@group(0) @binding(1) var<storage, read> src1_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_f: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(${WG_X}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col: u32       = gid.x;
    let row_idx: u32   = gid.y;
    let batch_lin: u32 = gid.z;

    if (col >= p.ne0) { return; }
    if (row_idx >= p.nr) { return; }
    let total_batches: u32 = p.ne02 * p.ne03;
    if (batch_lin >= total_batches) { return; }

    let i02: u32 = batch_lin % p.ne02;
    let i03: u32 = batch_lin / p.ne02;
    let i11: u32 = i02 % p.ne11;
    let i12: u32 = i03 % p.ne12;

    let idx_pair_off: u32 = row_idx * p.src1_s0_p
                          + i11     * p.src1_s1_p
                          + i12     * p.src1_s2_p;
    var i1: u32;
    if (p.idx_is_i64 == 1u) {
        i1 = src1_u[idx_pair_off * 2u];
    } else {
        i1 = src1_u[idx_pair_off];
    }

    let s_off: u32 = col
                   + row_idx * p.src0_s1_f
                   + i02     * p.src0_s2_f
                   + i03     * p.src0_s3_f;
    let v: f32 = src0_f[s_off];

    let d_off: u32 = col
                   + i1  * p.dst_s1_e
                   + i02 * p.dst_s2_e
                   + i03 * p.dst_s3_e;
    dst_f[d_off] = v;
}
`;

function buildPipeline(device: GPUDevice, wgsl: string): GPUComputePipeline {
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
 * Dispatch a SET_ROWS op. Returns 0 on success, negative on validation
 * failure (causing the C++ side to mark graph_compute FAILED).
 *
 * Descriptor expectations (n_src = 3 from `ggml_set_rows`):
 *   srcs[0] = b   F32 source data
 *   srcs[1] = c   I64 or I32 row indices
 *   srcs[2] = a   destination buffer (same handle as desc.dst — desc.dst
 *                 is `view_tensor(a)` so they share storage)
 *   desc.dst.type ∈ {F16, F32}
 */
export function dispatchSetRows(
	ctx: JsepOpContext,
	desc: JsepOpDescriptor,
): number {
	// Stage 4.8 instrumentation — temporary entry log to trace which path
	// returns -1 silently for the first production divert call.
	const __stage48Log = (
		globalThis as unknown as {
			__stage48SetRowsLog?: Array<{ phase: string; data?: unknown }>;
		}
	).__stage48SetRowsLog;
	if (__stage48Log)
		__stage48Log.push({
			phase: "entry",
			data: {
				nSrc: desc.nSrc,
				op: desc.op,
				dstH: desc.dst.bufHandle,
				dstO: desc.dst.offset,
			},
		});
	if (desc.nSrc !== 3) {
		console.error(`dispatchSetRows: expected 3 srcs, got ${desc.nSrc}`);
		if (__stage48Log)
			__stage48Log.push({ phase: "ret-1-nsrc", data: desc.nSrc });
		return -1;
	}

	const src0 = desc.srcs[0];
	const src1 = desc.srcs[1];
	const dst = desc.dst;

	if (src0.type !== GGML_TYPE_F32) {
		console.error(`dispatchSetRows: src0 must be F32 (got type=${src0.type})`);
		if (__stage48Log)
			__stage48Log.push({ phase: "ret-1-src0type", data: src0.type });
		return -1;
	}
	if (src1.type !== GGML_TYPE_I64 && src1.type !== GGML_TYPE_I32) {
		console.error(
			`dispatchSetRows: src1 must be I64 or I32 (got type=${src1.type})`,
		);
		if (__stage48Log)
			__stage48Log.push({ phase: "ret-1-src1type", data: src1.type });
		return -1;
	}
	if (dst.type !== GGML_TYPE_F16 && dst.type !== GGML_TYPE_F32) {
		console.error(
			`dispatchSetRows: dst must be F16 or F32 (got type=${dst.type})`,
		);
		if (__stage48Log)
			__stage48Log.push({ phase: "ret-1-dsttype", data: dst.type });
		return -1;
	}

	const ne0 = src0.ne[0];
	const nr = src0.ne[1];
	const ne02 = Math.max(1, src0.ne[2]);
	const ne03 = Math.max(1, src0.ne[3]);
	const ne11 = Math.max(1, src1.ne[1]);
	const ne12 = Math.max(1, src1.ne[2]);

	if (dst.ne[0] !== ne0) {
		console.error(
			`dispatchSetRows: dst ne[0]=${dst.ne[0]} != src0 ne[0]=${ne0}`,
		);
		if (__stage48Log)
			__stage48Log.push({
				phase: "ret-1-ne0mismatch",
				data: { dst: dst.ne[0], src0: ne0 },
			});
		return -1;
	}

	// Element strides. ggml_is_contiguous_rows guarantees nb[0] == elem_size,
	// so cross-row addressing is governed by nb[1..3] only.
	const f32Bytes = 4;
	const idxStride0Bytes = src1.type === GGML_TYPE_I64 ? 8 : 4;
	const dstElemBytes = dst.type === GGML_TYPE_F16 ? 2 : 4;

	const isI64 = src1.type === GGML_TYPE_I64 ? 1 : 0;
	const isF16Dst = dst.type === GGML_TYPE_F16;

	const src0_s1_f = src0.nb[1] / f32Bytes;
	const src0_s2_f = src0.nb[2] / f32Bytes;
	const src0_s3_f = src0.nb[3] / f32Bytes;

	const src1_s0_p = src1.nb[0] / idxStride0Bytes;
	const src1_s1_p = src1.nb[1] / idxStride0Bytes;
	const src1_s2_p = src1.nb[2] / idxStride0Bytes;

	const dst_s1_e = dst.nb[1] / dstElemBytes;
	const dst_s2_e = dst.nb[2] / dstElemBytes;
	const dst_s3_e = dst.nb[3] / dstElemBytes;

	for (const [name, val] of [
		["src0_s1_f", src0_s1_f],
		["src0_s2_f", src0_s2_f],
		["src0_s3_f", src0_s3_f],
		["src1_s0_p", src1_s0_p],
		["src1_s1_p", src1_s1_p],
		["src1_s2_p", src1_s2_p],
		["dst_s1_e", dst_s1_e],
		["dst_s2_e", dst_s2_e],
		["dst_s3_e", dst_s3_e],
	] as const) {
		if (!Number.isInteger(val) || val < 0) {
			console.error(
				`dispatchSetRows: non-integer or negative stride ${name}=${val} ` +
					`(src0.nb=[${src0.nb.join(",")}] src1.nb=[${src1.nb.join(",")}] ` +
					`dst.nb=[${dst.nb.join(",")}] dstElemBytes=${dstElemBytes})`,
			);
			if (__stage48Log)
				__stage48Log.push({
					phase: "ret-1-stride",
					data: { name, val },
				});
			return -1;
		}
	}

	const cacheKey = isF16Dst ? "set-rows-f32-to-f16" : "set-rows-f32-to-f32";
	const wgsl = isF16Dst ? SET_ROWS_F32_TO_F16_WGSL : SET_ROWS_F32_TO_F32_WGSL;

	const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
		const p = buildPipeline(device, wgsl);
		ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
		return p;
	});
	const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
	if (!bindGroupLayout) {
		console.error(`dispatchSetRows: missing bindGroupLayout for ${cacheKey}`);
		if (__stage48Log) __stage48Log.push({ phase: "ret-1-bgl", data: cacheKey });
		return -1;
	}

	// FIXME(phase 3): like dispatchMatmul + dispatchRmsNorm, this allocates
	// a fresh uniform buffer per dispatch and never destroys it. Decode KV
	// writes happen every step so the leak grows fast — same fix-shape as
	// the existing FIXMEs: cache or ring-buffer keyed by stride tuple.
	const paramsBuf = ctx.device.createBuffer({
		size: PARAMS_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	const params = new Uint32Array(PARAMS_U32);
	params[0] = ne0;
	params[1] = nr;
	params[2] = ne02;
	params[3] = ne03;
	params[4] = ne11;
	params[5] = ne12;
	params[6] = src0_s1_f;
	params[7] = src0_s2_f;
	params[8] = src0_s3_f;
	params[9] = src1_s0_p;
	params[10] = src1_s1_p;
	params[11] = src1_s2_p;
	params[12] = dst_s1_e;
	params[13] = dst_s2_e;
	params[14] = dst_s3_e;
	params[15] = isI64;
	ctx.device.queue.writeBuffer(paramsBuf, 0, params);

	const src0Rec = ctx.dataManager.get(src0.bufHandle);
	const src1Rec = ctx.dataManager.get(src1.bufHandle);
	const dstRec = ctx.dataManager.get(dst.bufHandle);

	const dispatchX = Math.ceil(ne0 / WG_X);
	const dispatchY = nr;
	const dispatchZ = ne02 * ne03;

	// WebGPU sync-scope rule: a single GPUBuffer bound as both read-only
	// storage and writable storage in the same compute pass is rejected
	// at buffer granularity (see matmul.ts:564 for the long form).
	//
	// SET_ROWS structurally aliases dst with src[2] — `dst = view(src[2])`
	// is the ggml semantics of the op, so dstRec.buffer always equals
	// the GPUBuffer that holds src[2]. src[2] is descriptor-only (not
	// bound to this kernel's bind group), so the within-kernel bind group
	// is conflict-free. The conflict instead happens within the batched
	// pass: neighbouring batched dispatches read other tensors that share
	// dstRec.buffer (cache_k / cache_v lives in jsep_buf alongside many
	// activations), so SET_ROWS' read_write binding of dstRec.buffer
	// trips the validator at encoder.finish().
	//
	// Divert (read-modify-write): allocate a temp dst, pre-copy real dst
	// into temp so unwritten rows + the F16 atomic-CAS path see the
	// correct prior state, dispatch into temp, then copy temp back into
	// real dst. Live in own command encoder (flush batcher first) so the
	// diverted dispatch can't conflict with batched neighbours.
	//
	// Aliasing rate measured 2026-05-06 on TinyLlama Q4_0 prefill+5
	// decode: 264/264 = 100% on src[2] (0% on src0/src1). The check
	// below covers all three to remain robust if a future caller ever
	// passes a kernel-bound aliased src.
	const src2BufHandle = desc.srcs.length > 2 ? desc.srcs[2].bufHandle : -1;
	const dstAliasesSrc =
		dst.bufHandle === src0.bufHandle ||
		dst.bufHandle === src1.bufHandle ||
		(src2BufHandle >= 0 && dst.bufHandle === src2BufHandle);

	if (dstAliasesSrc) {
		// Total bytes spanned by dst — handles the strided cache_v
		// transposed view (ne[0]=1, indices land in adjacent f16 cells)
		// by taking the max over all dims of (ne[d] * nb[d]).
		let dstSize = dst.ne[0] * dst.nb[0];
		for (let d = 1; d < 4; d++) {
			if (dst.ne[d] > 0) dstSize = Math.max(dstSize, dst.ne[d] * dst.nb[d]);
		}
		if (dst.offset + dstSize > dstRec.size) {
			console.error(
				`dispatchSetRows: divert would read past dst buffer end ` +
					`(offset=${dst.offset} + size=${dstSize} > ${dstRec.size})`,
			);
			if (__stage48Log)
				__stage48Log.push({
					phase: "ret-1-divertOOB",
					data: {
						offset: dst.offset,
						dstSize,
						dstRecSize: dstRec.size,
					},
				});
			return -1;
		}

		const tempDst = ctx.device.createBuffer({
			size: dstSize,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});
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
					resource: { buffer: tempDst, offset: 0, size: dstSize },
				},
				{ binding: 3, resource: { buffer: paramsBuf } },
			],
		});

		// Flush any pending batched dispatches — the diverted SET_ROWS +
		// its pre/post copies must live in a self-contained encoder.
		ctx.encoderBatcher.flush();

		const __stage48Hook = (
			globalThis as unknown as {
				__stage48DivertHook?: {
					triggered: boolean;
					// Capture buffers — 128 bytes each (64 F16 cells / 32 F32 cells),
					// 16 bytes per row across nr rows.
					preKernelBuf: GPUBuffer;
					postKernelBuf: GPUBuffer;
					postCopyBackBuf: GPUBuffer;
					src0CaptureBuf: GPUBuffer;
					capturedDstSize: number;
					capturedNe0: number;
					capturedNr: number;
				};
			}
		).__stage48DivertHook;
		const __stage48Capture = __stage48Hook && !__stage48Hook.triggered;
		if (__stage48Capture && __stage48Hook) {
			__stage48Hook.triggered = true;
			__stage48Hook.capturedDstSize = dstSize;
			__stage48Hook.capturedNe0 = ne0;
			__stage48Hook.capturedNr = nr;
		}

		const enc = ctx.device.createCommandEncoder();
		// Pre-copy real dst → temp so unwritten rows + the F16 atomic CAS
		// path both see the correct prior state.
		enc.copyBufferToBuffer(dstRec.buffer, dst.offset, tempDst, 0, dstSize);
		// Stage 4.8 — pre-kernel snapshot: tempDst rows 0..nr-1 first 8 cells
		// (8 F16 cells = 16 bytes per row, spaced ne0*2 bytes apart).
		// Also capture src0 (the K data the kernel will read) at the SAME
		// encoder timeline so we can see what the kernel saw.
		if (__stage48Capture && __stage48Hook) {
			const rowStrideBytes = ne0 * 2; // F16 cell stride
			const src0RowStrideBytes = ne0 * 4; // F32 cell stride for src0
			const rowsToCapture = Math.min(nr, 8);
			for (let r = 0; r < rowsToCapture; r++) {
				enc.copyBufferToBuffer(
					tempDst,
					r * rowStrideBytes,
					__stage48Hook.preKernelBuf,
					r * 16,
					16,
				);
				// Capture 16 bytes (4 F32) per src0 row at start.
				enc.copyBufferToBuffer(
					src0Rec.buffer,
					src0.offset + r * src0RowStrideBytes,
					__stage48Hook.src0CaptureBuf,
					r * 16,
					16,
				);
			}
		}
		const pass = enc.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, divertBindGroup);
		pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
		pass.end();
		// Stage 4.8 — post-kernel snapshot: rows 0..nr-1 first 8 cells.
		if (__stage48Capture && __stage48Hook) {
			const rowStrideBytes = ne0 * 2;
			const rowsToCapture = Math.min(nr, 8);
			for (let r = 0; r < rowsToCapture; r++) {
				enc.copyBufferToBuffer(
					tempDst,
					r * rowStrideBytes,
					__stage48Hook.postKernelBuf,
					r * 16,
					16,
				);
			}
		}
		// Post-copy temp → real dst.
		enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstSize);
		// Stage 4.8 — post-copy-back snapshot: dst rows 0..nr-1.
		if (__stage48Capture && __stage48Hook) {
			const rowStrideBytes = ne0 * 2;
			const rowsToCapture = Math.min(nr, 8);
			for (let r = 0; r < rowsToCapture; r++) {
				enc.copyBufferToBuffer(
					dstRec.buffer,
					dst.offset + r * rowStrideBytes,
					__stage48Hook.postCopyBackBuf,
					r * 16,
					16,
				);
			}
		}
		ctx.device.queue.submit([enc.finish()]);
		// destroy() after submit is documented-safe — pending GPU work is
		// allowed to complete using the destroyed buffer's underlying
		// memory.
		tempDst.destroy();
		if (__stage48Log)
			__stage48Log.push({
				phase: "ret-0-divert",
				data: { dstSize, dispatchX, dispatchY, dispatchZ },
			});
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
			{ binding: 3, resource: { buffer: paramsBuf } },
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

// Exported for test access without affecting the runtime path.
export const __setRowsInternals = {
	GGML_TYPE_I32,
	GGML_TYPE_I64,
	WG_X,
	PARAMS_U32,
};
