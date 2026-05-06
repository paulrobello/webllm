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
var TENSOR_BLOCK_I32 = 18;
var TILE_M = 16;
var TILE_N = 16;
var QK4_0 = 32;
function readDescriptor(heap32, byteOffset) {
  const base = byteOffset >> 2;
  const op = heap32[base];
  const nSrc = heap32[base + 1];
  const readBlock = (slot) => {
    const off = base + slot;
    return {
      handle: heap32[off],
      type: heap32[off + 1],
      ne: [heap32[off + 2], heap32[off + 3], heap32[off + 4], heap32[off + 5]],
      nb: [
        heap32[off + 10],
        heap32[off + 11],
        heap32[off + 12],
        heap32[off + 13]
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
      throw new Error("matmul Q4_K kernel: deferred to Task 7 (browser smoke covers via real weights)");
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
  const src0Buf = ctx.dataManager.get(src0.handle).buffer;
  const src1Buf = ctx.dataManager.get(src1.handle).buffer;
  const dstBuf = ctx.dataManager.get(dst.handle).buffer;
  const bindGroup = ctx.device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: src0Buf } },
      { binding: 1, resource: { buffer: src1Buf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: shapeBuffer } }
    ]
  });
  const dispatchX = Math.ceil(M / TILE_M);
  const dispatchY = Math.ceil(N / TILE_N);
  const dispatchZ = batchCount;
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
function dispatchRmsNorm(ctx, desc, opParamsPtr, heapBuffer) {
  if (desc.nSrc !== 2) {
    console.error(`dispatchRmsNorm: expected 2 srcs, got ${desc.nSrc}`);
    return -1;
  }
  const src0 = desc.srcs[0];
  const src1 = desc.srcs[1];
  const dst = desc.dst;
  if (src0.type !== GGML_TYPE_F32 || src1.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchRmsNorm: only F32 path supported in Phase 2 ` + `(got src0=${src0.type}, src1=${src1.type}, dst=${dst.type})`);
    return -1;
  }
  const eps = new Float32Array(heapBuffer, opParamsPtr, 1)[0];
  const cols = src0.ne[0];
  const rows = src0.ne[1] * Math.max(1, src0.ne[2]) * Math.max(1, src0.ne[3]);
  if (src1.ne[0] !== cols) {
    console.error(`dispatchRmsNorm: weight length ${src1.ne[0]} != cols ${cols}`);
    return -1;
  }
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
  const xBuf = ctx.dataManager.get(src0.handle).buffer;
  const wBuf = ctx.dataManager.get(src1.handle).buffer;
  const outBuf = ctx.dataManager.get(dst.handle).buffer;
  const bindGroup = ctx.device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: shapeBuffer } }
    ]
  });
  const dispatchX = Math.ceil(rows / WG_X);
  const dispatchY = Math.ceil(cols / WG_Y);
  const dispatchZ = 1;
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
    dataManager.write(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepRead = (handle, offset, hostPtr, size) => {
    counters.read++;
    return dataManager.readAsync(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepClear = (handle, value, offset, size) => {
    counters.clear++;
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
    return STATUS_NOT_IMPLEMENTED;
  };
  module.jsepSync = () => {
    counters.sync++;
    encoderBatcher.flush();
  };
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
