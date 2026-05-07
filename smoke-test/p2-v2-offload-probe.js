// src/inference/jsep/command-encoder.ts
class CommandEncoderBatcher {
  device;
  maxDispatch;
  commandEncoder = null;
  passEncoder = null;
  pendingDispatchCount = 0;
  constructor(device, options) {
    this.device = device;
    this.maxDispatch = options?.maxDispatch ?? 16;
  }
  record(dispatch) {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
    if (!this.passEncoder) {
      this.passEncoder = this.commandEncoder.beginComputePass();
    }
    this.passEncoder.setPipeline(dispatch.pipeline);
    this.passEncoder.setBindGroup(0, dispatch.bindGroup);
    this.passEncoder.dispatchWorkgroups(dispatch.dispatchX, dispatch.dispatchY, dispatch.dispatchZ);
    this.pendingDispatchCount++;
    if (this.pendingDispatchCount >= this.maxDispatch) {
      this.flush();
    }
  }
  flush() {
    if (!this.commandEncoder)
      return;
    if (this.passEncoder) {
      this.passEncoder.end();
      this.passEncoder = null;
    }
    const commands = this.commandEncoder.finish();
    this.device.queue.submit([commands]);
    this.commandEncoder = null;
    this.pendingDispatchCount = 0;
  }
  pendingCount() {
    return this.pendingDispatchCount;
  }
}

// src/inference/jsep/gpu-data-manager.ts
var SIZE_BUCKETS = [
  1 << 10,
  4 << 10,
  16 << 10,
  64 << 10,
  256 << 10,
  1 << 20,
  4 << 20,
  16 << 20,
  64 << 20,
  128 << 20
];
function pickBucket(size) {
  for (let i = 0;i < SIZE_BUCKETS.length; i++) {
    if (size <= SIZE_BUCKETS[i]) {
      return { bucketIndex: i, capacity: SIZE_BUCKETS[i] };
    }
  }
  return { bucketIndex: -1, capacity: size };
}

class GpuDataManager {
  device;
  handles = new Map;
  freeBuckets = SIZE_BUCKETS.map(() => []);
  nextHandle = 1;
  constructor(device) {
    this.device = device;
  }
  alloc(size) {
    if (size <= 0) {
      throw new Error(`GpuDataManager.alloc: invalid size ${size}`);
    }
    const { bucketIndex, capacity } = pickBucket(size);
    let record;
    if (bucketIndex >= 0) {
      const pool = this.freeBuckets[bucketIndex];
      const reused = pool.pop();
      if (reused) {
        record = reused;
      } else {
        record = {
          buffer: this.device.createBuffer({
            size: capacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
          }),
          size: capacity,
          bucket: bucketIndex
        };
      }
    } else {
      record = {
        buffer: this.device.createBuffer({
          size: capacity,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        }),
        size: capacity,
        bucket: -1
      };
    }
    const handle = this.nextHandle++;
    this.handles.set(handle, record);
    return handle;
  }
  free(handle) {
    const record = this.handles.get(handle);
    if (!record)
      return;
    this.handles.delete(handle);
    if (record.bucket >= 0) {
      this.freeBuckets[record.bucket].push(record);
    } else {
      record.buffer.destroy();
    }
  }
  get(handle) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.get: invalid handle ${handle}`);
    }
    return { buffer: record.buffer, size: record.size };
  }
  write(handle, offset, hostPtr, size, wasmHeap) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.write: invalid handle ${handle}`);
    }
    const view = new Uint8Array(wasmHeap, hostPtr, size);
    this.device.queue.writeBuffer(record.buffer, offset, view, 0, size);
  }
  async readAsync(handle, offset, hostPtr, size, wasmHeap) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.readAsync: invalid handle ${handle}`);
    }
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(record.buffer, offset, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, size);
    const mapped = new Uint8Array(staging.getMappedRange(0, size));
    const dest = new Uint8Array(wasmHeap, hostPtr, size);
    dest.set(mapped);
    staging.unmap();
    staging.destroy();
  }
  clear(handle, value, offset, size) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.clear: invalid handle ${handle}`);
    }
    const scratch = new Uint8Array(size);
    if (value !== 0)
      scratch.fill(value);
    this.device.queue.writeBuffer(record.buffer, offset, scratch, 0, size);
  }
  liveHandleCount() {
    return this.handles.size;
  }
  destroy() {
    for (const record of this.handles.values()) {
      record.buffer.destroy();
    }
    this.handles.clear();
    for (const bucket of this.freeBuckets) {
      for (const record of bucket) {
        record.buffer.destroy();
      }
      bucket.length = 0;
    }
  }
}

// src/inference/jsep/ops/matmul.ts
var GGML_TYPE_F32 = 0;
var GGML_TYPE_F16 = 1;
var GGML_TYPE_Q4_0 = 2;
var GGML_TYPE_Q4_K = 12;
var TENSOR_BLOCK_I32 = 19;
var TILE_M = 16;
var TILE_N = 16;
var QK4_0 = 32;
var QK_K = 256;
var Q4_K_BYTES_PER_BLOCK = 144;
function readDescriptor(heap32, byteOffset) {
  const base = byteOffset >> 2;
  const op = heap32[base];
  const nSrc = heap32[base + 1];
  const readBlock = (slot) => {
    const off = base + slot;
    return {
      bufHandle: heap32[off],
      offset: heap32[off + 1],
      type: heap32[off + 2],
      ne: [heap32[off + 3], heap32[off + 4], heap32[off + 5], heap32[off + 6]],
      nb: [
        heap32[off + 11],
        heap32[off + 12],
        heap32[off + 13],
        heap32[off + 14]
      ]
    };
  };
  const dst = readBlock(2);
  const srcs = [];
  for (let i = 0;i < nSrc; ++i) {
    srcs.push(readBlock(2 + TENSOR_BLOCK_I32 * (1 + i)));
  }
  return { op, nSrc, dst, srcs };
}
function typeName(t) {
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
function buildMatmulShader(src0Type, src1Type, dstType) {
  if (src1Type !== GGML_TYPE_F32 || dstType !== GGML_TYPE_F32) {
    throw new Error(`buildMatmulShader: unsupported (src1=${src1Type}, dst=${dstType}); ` + `Phase 2 requires src1=F32, dst=F32`);
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
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_q4_K(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;
    default:
      throw new Error(`matmul: unsupported src0 type ${src0Type}`);
  }
}
var SHAPE_UNIFORM_BYTES = 12 * 4;
function buildPipeline(device, src0Type, src1Type, dstType) {
  const wgsl = buildMatmulShader(src0Type, src1Type, dstType);
  const shaderModule = device.createShaderModule({ code: wgsl });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout]
  });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" }
  });
}
function dispatchMatmul(ctx, desc) {
  if (desc.nSrc !== 2) {
    console.error(`dispatchMatmul: expected 2 srcs, got ${desc.nSrc}`);
    return -1;
  }
  const src0 = desc.srcs[0];
  const src1 = desc.srcs[1];
  const dst = desc.dst;
  if (src1.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchMatmul: only src1=F32, dst=F32 in Phase 2 ` + `(got src1=${src1.type}, dst=${dst.type})`);
    return -1;
  }
  const K = src0.ne[0];
  const M = src0.ne[1];
  const N = src1.ne[1];
  if (src1.ne[0] !== K || dst.ne[0] !== M || dst.ne[1] !== N) {
    console.error(`dispatchMatmul: shape mismatch — src0=[${src0.ne.join(",")}], ` + `src1=[${src1.ne.join(",")}], dst=[${dst.ne.join(",")}]`);
    return -1;
  }
  const batchCount = Math.max(1, src1.ne[2]) * Math.max(1, src1.ne[3]);
  const ndim = batchCount > 1 ? 3 : 2;
  const cacheKey = `mat-${typeName(src0.type)}-${typeName(src1.type)}-${typeName(dst.type)}-${ndim}`;
  const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
    const p = buildPipeline(device, src0.type, src1.type, dst.type);
    ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
    return p;
  });
  const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
  if (!bindGroupLayout) {
    console.error(`dispatchMatmul: missing bindGroupLayout for ${cacheKey}`);
    return -1;
  }
  const shapeBuffer = ctx.device.createBuffer({
    size: SHAPE_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const shapeData = new Uint32Array(SHAPE_UNIFORM_BYTES / 4);
  shapeData[0] = M;
  shapeData[1] = K;
  shapeData[2] = N;
  shapeData[3] = batchCount;
  shapeData[4] = src0.nb[1];
  shapeData[5] = src1.nb[1];
  shapeData[6] = dst.nb[1];
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
  const dstAliasesSrc = dst.bufHandle === src0.bufHandle || dst.bufHandle === src1.bufHandle;
  if (dstAliasesSrc) {
    const expectedRowBytes = M * 4;
    const expectedBatchBytes = N * M * 4;
    if (dst.nb[1] !== expectedRowBytes || batchCount > 1 && dst.nb[2] !== expectedBatchBytes) {
      console.error(`dispatchMatmul: aliased dst is non-contiguous (nb=[${dst.nb.join(",")}], ` + `expected row=${expectedRowBytes}, batch=${expectedBatchBytes}); ` + `divert path requires contiguous dst.`);
      return -1;
    }
    const dstBytesNeeded = batchCount * N * M * 4;
    const tempDst = ctx.device.createBuffer({
      size: dstBytesNeeded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const divertBindGroup = ctx.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: src0Rec.buffer,
            offset: src0.offset,
            size: src0Rec.size - src0.offset
          }
        },
        {
          binding: 1,
          resource: {
            buffer: src1Rec.buffer,
            offset: src1.offset,
            size: src1Rec.size - src1.offset
          }
        },
        {
          binding: 2,
          resource: { buffer: tempDst, offset: 0, size: dstBytesNeeded }
        },
        { binding: 3, resource: { buffer: shapeBuffer } }
      ]
    });
    ctx.encoderBatcher.flush();
    const enc = ctx.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, divertBindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    pass.end();
    enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstBytesNeeded);
    ctx.device.queue.submit([enc.finish()]);
    const probeGlobal = globalThis;
    if (probeGlobal.__stage415DivertProbe) {
      if (!probeGlobal.__stage415DivertLog) {
        probeGlobal.__stage415DivertLog = [];
      }
      const dbg = probeGlobal.__stage415DivertLog;
      if (dbg.length < 32) {
        const tempStaging = ctx.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const dstStaging = ctx.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const probeEnc = ctx.device.createCommandEncoder();
        probeEnc.copyBufferToBuffer(tempDst, 0, tempStaging, 0, 16);
        probeEnc.copyBufferToBuffer(dstRec.buffer, dst.offset, dstStaging, 0, 16);
        ctx.device.queue.submit([probeEnc.finish()]);
        const meta = {
          divertIdx: dbg.length,
          dstHandle: dst.bufHandle,
          dstOffset: dst.offset,
          dstNe: [dst.ne[0], dst.ne[1], dst.ne[2], dst.ne[3]],
          src0Ne: [src0.ne[0], src0.ne[1], src0.ne[2], src0.ne[3]],
          src1Ne: [src1.ne[0], src1.ne[1], src1.ne[2], src1.ne[3]],
          tempBytes: null,
          dstBytes: null
        };
        dbg.push(meta);
        Promise.all([
          tempStaging.mapAsync(GPUMapMode.READ),
          dstStaging.mapAsync(GPUMapMode.READ)
        ]).then(() => {
          meta.tempBytes = new Uint8Array(tempStaging.getMappedRange().slice(0));
          meta.dstBytes = new Uint8Array(dstStaging.getMappedRange().slice(0));
          tempStaging.unmap();
          dstStaging.unmap();
          tempStaging.destroy();
          dstStaging.destroy();
        });
      }
    }
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
          size: src0Rec.size - src0.offset
        }
      },
      {
        binding: 1,
        resource: {
          buffer: src1Rec.buffer,
          offset: src1.offset,
          size: src1Rec.size - src1.offset
        }
      },
      {
        binding: 2,
        resource: {
          buffer: dstRec.buffer,
          offset: dst.offset,
          size: dstRec.size - dst.offset
        }
      },
      { binding: 3, resource: { buffer: shapeBuffer } }
    ]
  });
  ctx.encoderBatcher.record({
    pipeline,
    bindGroup,
    dispatchX,
    dispatchY,
    dispatchZ
  });
  return 0;
}

// src/inference/jsep/ops/rms-norm.ts
var WG_X = 1;
var WG_Y = 256;
var SHAPE_UNIFORM_BYTES2 = 4 * 4;
var RMS_NORM_WGSL = `
struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

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
    out[idx] = x[idx] * inv_rms;
}
`;
function buildPipeline2(device) {
  const shaderModule = device.createShaderModule({ code: RMS_NORM_WGSL });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout]
  });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" }
  });
}
function dispatchRmsNorm(ctx, desc, opParamsPtr, heapBuffer) {
  if (desc.nSrc !== 1) {
    console.error(`dispatchRmsNorm: expected 1 src, got ${desc.nSrc}`);
    return -1;
  }
  const src0 = desc.srcs[0];
  const dst = desc.dst;
  if (src0.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchRmsNorm: only F32 path supported in Phase 2 ` + `(got src0=${src0.type}, dst=${dst.type})`);
    return -1;
  }
  const eps = new Float32Array(heapBuffer, opParamsPtr, 1)[0];
  const cols = src0.ne[0];
  const rows = src0.ne[1] * Math.max(1, src0.ne[2]) * Math.max(1, src0.ne[3]);
  if (dst.ne[0] !== src0.ne[0] || dst.ne[1] !== src0.ne[1] || dst.ne[2] !== src0.ne[2] || dst.ne[3] !== src0.ne[3]) {
    console.error(`dispatchRmsNorm: shape mismatch — src0=[${src0.ne.join(",")}], ` + `dst=[${dst.ne.join(",")}]`);
    return -1;
  }
  const cacheKey = "rms-norm-f32";
  const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
    const p = buildPipeline2(device);
    ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
    return p;
  });
  const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
  if (!bindGroupLayout) {
    console.error(`dispatchRmsNorm: missing bindGroupLayout for ${cacheKey}`);
    return -1;
  }
  const shapeBuffer = ctx.device.createBuffer({
    size: SHAPE_UNIFORM_BYTES2,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const shapeU32 = new Uint32Array(SHAPE_UNIFORM_BYTES2 / 4);
  shapeU32[0] = rows;
  shapeU32[1] = cols;
  shapeU32[3] = 0;
  new Float32Array(shapeU32.buffer)[2] = eps;
  ctx.device.queue.writeBuffer(shapeBuffer, 0, shapeU32);
  const src0Rec = ctx.dataManager.get(src0.bufHandle);
  const dstRec = ctx.dataManager.get(dst.bufHandle);
  const dispatchX = Math.ceil(rows / WG_X);
  const dispatchY = Math.ceil(cols / WG_Y);
  const dispatchZ = 1;
  const dstAliasesSrc = dst.bufHandle === src0.bufHandle;
  if (dstAliasesSrc) {
    const dstBytesNeeded = rows * cols * 4;
    const tempDst = ctx.device.createBuffer({
      size: dstBytesNeeded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const divertBindGroup = ctx.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: src0Rec.buffer,
            offset: src0.offset,
            size: src0Rec.size - src0.offset
          }
        },
        {
          binding: 1,
          resource: { buffer: tempDst, offset: 0, size: dstBytesNeeded }
        },
        { binding: 2, resource: { buffer: shapeBuffer } }
      ]
    });
    ctx.encoderBatcher.flush();
    const enc = ctx.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, divertBindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    pass.end();
    enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstBytesNeeded);
    ctx.device.queue.submit([enc.finish()]);
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
          size: src0Rec.size - src0.offset
        }
      },
      {
        binding: 1,
        resource: {
          buffer: dstRec.buffer,
          offset: dst.offset,
          size: dstRec.size - dst.offset
        }
      },
      { binding: 2, resource: { buffer: shapeBuffer } }
    ]
  });
  ctx.encoderBatcher.record({
    pipeline,
    bindGroup,
    dispatchX,
    dispatchY,
    dispatchZ
  });
  return 0;
}

// src/inference/jsep/ops/set-rows.ts
var GGML_TYPE_I32 = 26;
var GGML_TYPE_I64 = 27;
var WG_X2 = 64;
var PARAMS_U32 = 16;
var PARAMS_BYTES = PARAMS_U32 * 4;
function shaderCommon() {
  return `
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
var SET_ROWS_F32_TO_F16_WGSL = `${shaderCommon()}
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

@compute @workgroup_size(${WG_X2}, 1, 1)
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
var SET_ROWS_F32_TO_F32_WGSL = `${shaderCommon()}
@group(0) @binding(0) var<storage, read> src0_f: array<f32>;
@group(0) @binding(1) var<storage, read> src1_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_f: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(${WG_X2}, 1, 1)
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
function buildPipeline3(device, wgsl) {
  const shaderModule = device.createShaderModule({ code: wgsl });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout]
  });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" }
  });
}
function dispatchSetRows(ctx, desc) {
  const __stage48Log = globalThis.__stage48SetRowsLog;
  if (__stage48Log)
    __stage48Log.push({
      phase: "entry",
      data: {
        nSrc: desc.nSrc,
        op: desc.op,
        dstH: desc.dst.bufHandle,
        dstO: desc.dst.offset
      }
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
    console.error(`dispatchSetRows: src1 must be I64 or I32 (got type=${src1.type})`);
    if (__stage48Log)
      __stage48Log.push({ phase: "ret-1-src1type", data: src1.type });
    return -1;
  }
  if (dst.type !== GGML_TYPE_F16 && dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchSetRows: dst must be F16 or F32 (got type=${dst.type})`);
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
    console.error(`dispatchSetRows: dst ne[0]=${dst.ne[0]} != src0 ne[0]=${ne0}`);
    if (__stage48Log)
      __stage48Log.push({
        phase: "ret-1-ne0mismatch",
        data: { dst: dst.ne[0], src0: ne0 }
      });
    return -1;
  }
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
    ["dst_s3_e", dst_s3_e]
  ]) {
    if (!Number.isInteger(val) || val < 0) {
      console.error(`dispatchSetRows: non-integer or negative stride ${name}=${val} ` + `(src0.nb=[${src0.nb.join(",")}] src1.nb=[${src1.nb.join(",")}] ` + `dst.nb=[${dst.nb.join(",")}] dstElemBytes=${dstElemBytes})`);
      if (__stage48Log)
        __stage48Log.push({
          phase: "ret-1-stride",
          data: { name, val }
        });
      return -1;
    }
  }
  const cacheKey = isF16Dst ? "set-rows-f32-to-f16" : "set-rows-f32-to-f32";
  const wgsl = isF16Dst ? SET_ROWS_F32_TO_F16_WGSL : SET_ROWS_F32_TO_F32_WGSL;
  const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
    const p = buildPipeline3(device, wgsl);
    ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
    return p;
  });
  const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
  if (!bindGroupLayout) {
    console.error(`dispatchSetRows: missing bindGroupLayout for ${cacheKey}`);
    if (__stage48Log)
      __stage48Log.push({ phase: "ret-1-bgl", data: cacheKey });
    return -1;
  }
  const paramsBuf = ctx.device.createBuffer({
    size: PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
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
  const dispatchX = Math.ceil(ne0 / WG_X2);
  const dispatchY = nr;
  const dispatchZ = ne02 * ne03;
  const src2BufHandle = desc.srcs.length > 2 ? desc.srcs[2].bufHandle : -1;
  const dstAliasesSrc = dst.bufHandle === src0.bufHandle || dst.bufHandle === src1.bufHandle || src2BufHandle >= 0 && dst.bufHandle === src2BufHandle;
  if (dstAliasesSrc) {
    let dstSize = dst.ne[0] * dst.nb[0];
    for (let d = 1;d < 4; d++) {
      if (dst.ne[d] > 0)
        dstSize = Math.max(dstSize, dst.ne[d] * dst.nb[d]);
    }
    if (dst.offset + dstSize > dstRec.size) {
      console.error(`dispatchSetRows: divert would read past dst buffer end ` + `(offset=${dst.offset} + size=${dstSize} > ${dstRec.size})`);
      if (__stage48Log)
        __stage48Log.push({
          phase: "ret-1-divertOOB",
          data: {
            offset: dst.offset,
            dstSize,
            dstRecSize: dstRec.size
          }
        });
      return -1;
    }
    const tempDst = ctx.device.createBuffer({
      size: dstSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const divertBindGroup = ctx.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: src0Rec.buffer,
            offset: src0.offset,
            size: src0Rec.size - src0.offset
          }
        },
        {
          binding: 1,
          resource: {
            buffer: src1Rec.buffer,
            offset: src1.offset,
            size: src1Rec.size - src1.offset
          }
        },
        {
          binding: 2,
          resource: { buffer: tempDst, offset: 0, size: dstSize }
        },
        { binding: 3, resource: { buffer: paramsBuf } }
      ]
    });
    ctx.encoderBatcher.flush();
    const __stage48Hook = globalThis.__stage48DivertHook;
    const __stage48Capture = __stage48Hook && !__stage48Hook.triggered;
    if (__stage48Capture && __stage48Hook) {
      __stage48Hook.triggered = true;
      __stage48Hook.capturedDstSize = dstSize;
      __stage48Hook.capturedNe0 = ne0;
      __stage48Hook.capturedNr = nr;
    }
    const enc = ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(dstRec.buffer, dst.offset, tempDst, 0, dstSize);
    if (__stage48Capture && __stage48Hook) {
      const rowStrideBytes = ne0 * 2;
      const src0RowStrideBytes = ne0 * 4;
      const rowsToCapture = Math.min(nr, 8);
      for (let r = 0;r < rowsToCapture; r++) {
        enc.copyBufferToBuffer(tempDst, r * rowStrideBytes, __stage48Hook.preKernelBuf, r * 16, 16);
        enc.copyBufferToBuffer(src0Rec.buffer, src0.offset + r * src0RowStrideBytes, __stage48Hook.src0CaptureBuf, r * 16, 16);
      }
    }
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, divertBindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    pass.end();
    if (__stage48Capture && __stage48Hook) {
      const rowStrideBytes = ne0 * 2;
      const rowsToCapture = Math.min(nr, 8);
      for (let r = 0;r < rowsToCapture; r++) {
        enc.copyBufferToBuffer(tempDst, r * rowStrideBytes, __stage48Hook.postKernelBuf, r * 16, 16);
      }
    }
    enc.copyBufferToBuffer(tempDst, 0, dstRec.buffer, dst.offset, dstSize);
    if (__stage48Capture && __stage48Hook) {
      const rowStrideBytes = ne0 * 2;
      const rowsToCapture = Math.min(nr, 8);
      for (let r = 0;r < rowsToCapture; r++) {
        enc.copyBufferToBuffer(dstRec.buffer, dst.offset + r * rowStrideBytes, __stage48Hook.postCopyBackBuf, r * 16, 16);
      }
    }
    ctx.device.queue.submit([enc.finish()]);
    tempDst.destroy();
    if (__stage48Log)
      __stage48Log.push({
        phase: "ret-0-divert",
        data: { dstSize, dispatchX, dispatchY, dispatchZ }
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
          size: src0Rec.size - src0.offset
        }
      },
      {
        binding: 1,
        resource: {
          buffer: src1Rec.buffer,
          offset: src1.offset,
          size: src1Rec.size - src1.offset
        }
      },
      {
        binding: 2,
        resource: {
          buffer: dstRec.buffer,
          offset: dst.offset,
          size: dstRec.size - dst.offset
        }
      },
      { binding: 3, resource: { buffer: paramsBuf } }
    ]
  });
  ctx.encoderBatcher.record({
    pipeline,
    bindGroup,
    dispatchX,
    dispatchY,
    dispatchZ
  });
  return 0;
}

// src/inference/jsep/pipeline-cache.ts
class PipelineCache {
  device;
  cache = new Map;
  constructor(device) {
    this.device = device;
  }
  getOrCreate(key, builder) {
    const existing = this.cache.get(key);
    if (existing)
      return existing;
    const created = builder(this.device);
    this.cache.set(key, created);
    return created;
  }
  size() {
    return this.cache.size;
  }
}

// src/inference/jsep/index.ts
var STATUS_NOT_IMPLEMENTED = 1;
var GGML_OP_RMS_NORM = 25;
var GGML_OP_MUL_MAT = 29;
var GGML_OP_SET_ROWS = 42;
function installJsepCallbacks(module, device) {
  if (module.__jsep) {
    throw new Error("installJsepCallbacks: callbacks already installed on this module. " + "Call destroyJsepCallbacks(module) first if you need to re-install.");
  }
  const dataManager = new GpuDataManager(device);
  const encoderBatcher = new CommandEncoderBatcher(device);
  const pipelineCache = new PipelineCache(device);
  const bindGroupLayoutCache = new Map;
  const counters = {
    alloc: 0,
    free: 0,
    write: 0,
    read: 0,
    clear: 0,
    runOp: 0,
    sync: 0
  };
  const runtime = {
    device,
    dataManager,
    encoderBatcher,
    pipelineCache,
    bindGroupLayoutCache,
    counters
  };
  module.__jsep = runtime;
  module.jsepAlloc = (size) => {
    counters.alloc++;
    return dataManager.alloc(size);
  };
  module.jsepFree = (handle) => {
    counters.free++;
    dataManager.free(handle);
  };
  module.jsepWrite = (handle, offset, hostPtr, size) => {
    counters.write++;
    encoderBatcher.flush();
    const __h1invDiag = globalThis.__h1invDiag;
    if (__h1invDiag && __h1invDiag.captures.length < 8 && handle === __h1invDiag.match.handle && offset === __h1invDiag.match.offset && size === __h1invDiag.match.size) {
      const heap8 = new Uint8Array(module.HEAPU8.buffer, hostPtr, 16);
      const heap32 = new Float32Array(module.HEAPU8.buffer, hostPtr, 8);
      __h1invDiag.captures.push({
        callIdx: __h1invDiag.callIdx++,
        handle,
        offset,
        size,
        first16: Array.from(heap8),
        first8F32: Array.from(heap32)
      });
    }
    dataManager.write(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepRead = (handle, offset, hostPtr, size) => {
    counters.read++;
    encoderBatcher.flush();
    return dataManager.readAsync(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepClear = (handle, value, offset, size) => {
    counters.clear++;
    encoderBatcher.flush();
    dataManager.clear(handle, value, offset, size);
  };
  module.jsepRunOp = (descriptorPtr, _descriptorWords, opParamsPtr, _opParamsLen) => {
    counters.runOp++;
    const buf = module.HEAPU8.buffer;
    const heap32 = new Int32Array(buf, 0, buf.byteLength >>> 2);
    const desc = readDescriptor(heap32, descriptorPtr);
    const ctx = {
      device,
      dataManager,
      encoderBatcher,
      pipelineCache,
      bindGroupLayoutCache
    };
    if (desc.op === GGML_OP_MUL_MAT) {
      return dispatchMatmul(ctx, desc);
    }
    if (desc.op === GGML_OP_RMS_NORM) {
      return dispatchRmsNorm(ctx, desc, opParamsPtr, buf);
    }
    if (desc.op === GGML_OP_SET_ROWS) {
      return dispatchSetRows(ctx, desc);
    }
    return STATUS_NOT_IMPLEMENTED;
  };
  module.jsepSync = () => {
    counters.sync++;
    encoderBatcher.flush();
  };
  {
    const F32_BYTES = 4;
    const I64_BYTES = 8;
    const F16_BYTES = 2;
    const NE0 = 256;
    const NR = 6;
    const DST_ROWS = 512;
    const srcAllocSize = NE0 * NR * F32_BYTES + NR * I64_BYTES;
    const dstAllocSize = NE0 * DST_ROWS * F16_BYTES;
    const srcHandle = dataManager.alloc(srcAllocSize);
    const dstHandle = dataManager.alloc(dstAllocSize);
    const desc = {
      op: GGML_OP_SET_ROWS,
      nSrc: 3,
      dst: {
        bufHandle: dstHandle,
        offset: 0,
        type: 1,
        ne: [NE0, DST_ROWS, 1, 1],
        nb: [
          F16_BYTES,
          NE0 * F16_BYTES,
          NE0 * DST_ROWS * F16_BYTES,
          NE0 * DST_ROWS * F16_BYTES
        ]
      },
      srcs: [
        {
          bufHandle: srcHandle,
          offset: 0,
          type: 0,
          ne: [NE0, NR, 1, 1],
          nb: [
            F32_BYTES,
            NE0 * F32_BYTES,
            NE0 * NR * F32_BYTES,
            NE0 * NR * F32_BYTES
          ]
        },
        {
          bufHandle: srcHandle,
          offset: NE0 * F32_BYTES,
          type: 27,
          ne: [NR, 1, 1, 1],
          nb: [I64_BYTES, NR * I64_BYTES, NR * I64_BYTES, NR * I64_BYTES]
        },
        {
          bufHandle: dstHandle,
          offset: 0,
          type: 1,
          ne: [NE0, DST_ROWS, 1, 1],
          nb: [
            F16_BYTES,
            NE0 * F16_BYTES,
            NE0 * DST_ROWS * F16_BYTES,
            NE0 * DST_ROWS * F16_BYTES
          ]
        }
      ]
    };
    const ctx = {
      device,
      dataManager,
      encoderBatcher,
      pipelineCache,
      bindGroupLayoutCache
    };
    dispatchSetRows(ctx, desc);
    dataManager.free(srcHandle);
    dataManager.free(dstHandle);
  }
  return runtime;
}

// smoke-test/p2-v2-offload-probe.src.ts
function log(msg, cls = "") {
  const el = document.getElementById("log");
  if (!el)
    return;
  const line = document.createElement("div");
  if (cls)
    line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  console.log(msg);
}
async function runProbe() {
  try {
    log("[1/6] Initializing JSEP WASM module...");
    const wasmGlueUrl = `./webllm-wasm-jsep.js${window.location.search || "?v=fresh"}`;
    const createModule = (await import(wasmGlueUrl)).default;
    window.__lastModuleRef = null;
    window.__stderrLines = [];
    const mod = await createModule({
      printErr: (s) => {
        window.__stderrLines.push(s);
        console.error(s);
      }
    });
    window.__lastModuleRef = mod;
    log("[2/6] Acquiring WebGPU device...");
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      log("no WebGPU adapter", "fail");
      return;
    }
    const device = await adapter.requestDevice();
    log("[3/6] Installing JSEP callbacks (must precede webgpu_init)...");
    installJsepCallbacks(mod, device);
    log("[4/6] Initializing ggml-webgpu backend...");
    const initStatus = await mod._webgpu_init();
    if (initStatus !== 0) {
      log(`webgpu_init returned ${initStatus}`, "fail");
      return;
    }
    log("[5/6] Snapshotting JSEP counters before probe...");
    const counter0 = {
      ...mod.__jsep?.counters ?? {}
    };
    log(`     counters@pre = ${JSON.stringify(counter0)}`);
    log("[6/6] Running webllm_synthetic_offload_probe...");
    const status = await mod._webllm_synthetic_offload_probe();
    const logPtr = mod._webllm_synthetic_probe_log();
    const probeLog = mod.UTF8ToString(logPtr);
    const counter1 = {
      ...mod.__jsep?.counters ?? {}
    };
    const deltas = {};
    for (const k of Object.keys(counter1)) {
      deltas[k] = (counter1[k] ?? 0) - (counter0[k] ?? 0);
    }
    const runOpDelta = deltas.runOp ?? 0;
    log(`PROBE_STATUS = ${status}`);
    log(`PROBE_LOG =
${probeLog}`);
    log(`COUNTER_DELTAS = ${JSON.stringify(deltas)}`);
    log(`RUN_OP_DELTA = ${runOpDelta}`);
    const pass = status === 0 && runOpDelta >= 1;
    if (pass) {
      log("VERDICT: PASS — JSEP fired ≥1 MUL_MAT via offload_op.", "pass");
    } else if (status === 2) {
      log("VERDICT: NOT-APPLICABLE — JSEP not registered in this build.", "fail");
    } else if (status !== 0) {
      log(`VERDICT: FAIL — probe returned non-zero status ${status}.`, "fail");
    } else {
      log(`VERDICT: FAIL — scheduler ran but JSEP runOp delta = ${runOpDelta} (expected ≥ 1).`, "fail");
    }
    window.__probeResult = {
      status,
      log: probeLog,
      runOpDelta,
      deltas,
      counter0,
      counter1,
      pass
    };
  } catch (err) {
    const e = err;
    log(`FAIL — ${e.message}
${e.stack ?? ""}`, "fail");
    try {
      const modAny = window.__lastModuleRef;
      if (modAny?.__jsep?.counters) {
        log(`POST_THROW_COUNTERS = ${JSON.stringify(modAny.__jsep.counters)}`);
        window.__probeResult = {
          status: -1,
          error: e.message,
          stack: e.stack ?? "",
          countersOnThrow: { ...modAny.__jsep.counters }
        };
      }
    } catch {}
  }
}
runProbe();
