/**
 * WGSL compute shaders for WebGPU-accelerated LLM inference operations.
 *
 * Each shader is a self-contained WGSL compute pipeline that operates on
 * storage buffers bound at group 0. The uniform params struct provides
 * dimension information to each kernel.
 */

/** Matrix multiplication: C[M,N] = A[M,K] * B[K,N] */
export const SHADER_MATMUL_F32 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params { m: u32, n: u32, k: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.m || col >= params.n) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.k; i = i + 1u) {
    sum += a[row * params.k + i] * b[i * params.n + col];
  }
  c[row * params.n + col] = sum;
}
`;

/** RMS normalization: out = x * weight / sqrt(mean(x^2) + eps) */
export const SHADER_RMS_NORM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params { rows: u32, cols: u32, eps: f32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.rows || col >= params.cols) { return; }

  // Compute sum of squares for the entire row
  var sum_sq: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    let val = x[row * params.cols + i];
    sum_sq += val * val;
  }
  let rms = inversesqrt(sum_sq / f32(params.cols) + params.eps);
  out[row * params.cols + col] = x[row * params.cols + col] * weight[col] * rms;
}
`;

/** Layer normalization: out = (x - mean) / sqrt(var + eps) * weight + bias */
export const SHADER_LAYER_NORM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { rows: u32, cols: u32, eps: f32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.rows || col >= params.cols) { return; }

  // Compute mean
  var mean: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    mean += x[row * params.cols + i];
  }
  mean = mean / f32(params.cols);

  // Compute variance
  var variance: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    let diff = x[row * params.cols + i] - mean;
    variance += diff * diff;
  }
  variance = variance / f32(params.cols);

  let inv_std = inversesqrt(variance + params.eps);
  let normalized = (x[row * params.cols + col] - mean) * inv_std;
  out[row * params.cols + col] = normalized * weight[col] + bias[col];
}
`;

/** Softmax with scale: out[i] = exp(x[i]*scale - max) / sum(exp(x*scale - max)) */
export const SHADER_SOFTMAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params { rows: u32, cols: u32, scale: f32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.rows || col >= params.cols) { return; }

  // Find max value in the row for numerical stability
  var max_val: f32 = x[row * params.cols];
  for (var i: u32 = 1u; i < params.cols; i = i + 1u) {
    let val = x[row * params.cols + i] * params.scale;
    if (val > max_val) {
      max_val = val;
    }
  }

  // Compute sum of exp(x*scale - max)
  var sum_exp: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    sum_exp += exp(x[row * params.cols + i] * params.scale - max_val);
  }

  out[row * params.cols + col] = exp(x[row * params.cols + col] * params.scale - max_val) / sum_exp;
}
`;

/** GELU activation using tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
export const SHADER_GELU = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params { n: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let val = x[idx];
  let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2.0 / pi)
  let inner = sqrt_2_over_pi * (val + 0.044715 * val * val * val);
  out[idx] = 0.5 * val * (1.0 + tanh(inner));
}
`;

/** SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)) */
export const SHADER_SILU = /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params { n: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let val = x[idx];
  out[idx] = val / (1.0 + exp(-val));
}
`;

/** Gather embedding vectors by token ID */
export const SHADER_EMBEDDING_LOOKUP = /* wgsl */ `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params { vocab_size: u32, embed_dim: u32, n_tokens: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_tokens * params.embed_dim;
  if (idx >= total) { return; }
  let token_idx = idx / params.embed_dim;
  let dim_idx = idx % params.embed_dim;
  let token_id = token_ids[token_idx];
  out[idx] = weights[token_id * params.embed_dim + dim_idx];
}
`;

/** Registry of all WGSL compute shaders keyed by operation name. */
export const ALL_SHADERS = {
	matmul_f32: SHADER_MATMUL_F32,
	rms_norm: SHADER_RMS_NORM,
	layer_norm: SHADER_LAYER_NORM,
	softmax: SHADER_SOFTMAX,
	gelu: SHADER_GELU,
	silu: SHADER_SILU,
	embedding_lookup: SHADER_EMBEDDING_LOOKUP,
} as const;

/** Union type of all shader names in the registry. */
export type ShaderName = keyof typeof ALL_SHADERS;
