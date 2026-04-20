#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-webgpu.h"
#include <cstdio>
#include <cstring>

static ggml_backend_t g_backend = nullptr;
static struct ggml_context* g_ctx = nullptr;

extern "C" {

// ── Backend lifecycle ───────────────────────────────────────────────────

int32_t webgpu_init() {
    g_backend = ggml_backend_webgpu_init();
    if (!g_backend) {
        fprintf(stderr, "webgpu_init: failed to create WebGPU backend\n");
        return -1;
    }
    return 0;
}

void webgpu_shutdown() {
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
}

// ── Context management ──────────────────────────────────────────────────

int32_t ctx_create(int64_t mem_size) {
    struct ggml_init_params params = {
        .mem_size   = (size_t)mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    g_ctx = ggml_init(params);
    return g_ctx ? 0 : -1;
}

void ctx_free() {
    if (g_ctx) {
        ggml_free(g_ctx);
        g_ctx = nullptr;
    }
}

// ── Tensor creation ─────────────────────────────────────────────────────

void* tensor_new_1d(int32_t type, int64_t ne0) {
    return ggml_new_tensor_1d(g_ctx, (ggml_type)type, ne0);
}

void* tensor_new_2d(int32_t type, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(g_ctx, (ggml_type)type, ne0, ne1);
}

void* tensor_new_3d(int32_t type, int64_t ne0, int64_t ne1, int64_t ne2) {
    return ggml_new_tensor_3d(g_ctx, (ggml_type)type, ne0, ne1, ne2);
}

void* tensor_new_4d(int32_t type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return ggml_new_tensor_4d(g_ctx, (ggml_type)type, ne0, ne1, ne2, ne3);
}

void tensor_set_name(void* tensor, const char* name) {
    ggml_set_name((struct ggml_tensor*)tensor, name);
}

// ── Tensor properties ───────────────────────────────────────────────────

int64_t tensor_nelements(void* tensor) {
    return ggml_nelements((const struct ggml_tensor*)tensor);
}

size_t tensor_nbytes(void* tensor) {
    return ggml_nbytes((const struct ggml_tensor*)tensor);
}

int32_t tensor_type(void* tensor) {
    return (int32_t)((const struct ggml_tensor*)tensor)->type;
}

int64_t tensor_ne(void* tensor, int32_t dim) {
    return ((const struct ggml_tensor*)tensor)->ne[dim];
}

size_t tensor_nb(void* tensor, int32_t dim) {
    return ((const struct ggml_tensor*)tensor)->nb[dim];
}

void* tensor_data(void* tensor) {
    return ((struct ggml_tensor*)tensor)->data;
}

// ── Tensor data I/O (direct memory access via WASM heap) ────────────────

void tensor_set_data(void* tensor, const void* data, size_t size) {
    memcpy(((struct ggml_tensor*)tensor)->data, data, size);
}

void tensor_get_data(void* tensor, void* out, size_t size) {
    memcpy(out, ((struct ggml_tensor*)tensor)->data, size);
}

// ── Graph operations ────────────────────────────────────────────────────

void* op_mul_mat(void* a, void* b) {
    return ggml_mul_mat(g_ctx, (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_add(void* a, void* b) {
    return ggml_add(g_ctx, (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_mul(void* a, void* b) {
    return ggml_mul(g_ctx, (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_rms_norm(void* x, float eps) {
    return ggml_rms_norm(g_ctx, (struct ggml_tensor*)x, eps);
}

void* op_silu(void* x) {
    return ggml_silu(g_ctx, (struct ggml_tensor*)x);
}

void* op_gelu(void* x) {
    return ggml_gelu(g_ctx, (struct ggml_tensor*)x);
}

void* op_rope(void* x, int32_t n_dims, int32_t mode, int32_t n_ctx_orig,
              float freq_base, float freq_scale, float ext_factor,
              float attn_factor, float beta_fast, float beta_slow) {
    return ggml_rope_ext(g_ctx, (struct ggml_tensor*)x, nullptr, nullptr,
                         n_dims, mode, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
}

void* op_reshape_2d(void* x, int64_t ne0, int64_t ne1) {
    return ggml_reshape_2d(g_ctx, (struct ggml_tensor*)x, ne0, ne1);
}

void* op_reshape_3d(void* x, int64_t ne0, int64_t ne1, int64_t ne2) {
    return ggml_reshape_3d(g_ctx, (struct ggml_tensor*)x, ne0, ne1, ne2);
}

void* op_permute(void* x, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3) {
    return ggml_permute(g_ctx, (struct ggml_tensor*)x, dim0, dim1, dim2, dim3);
}

void* op_cont(void* x) {
    return ggml_cont(g_ctx, (struct ggml_tensor*)x);
}

void* op_view_2d(void* x, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
    return ggml_view_2d(g_ctx, (struct ggml_tensor*)x, ne0, ne1, nb1, offset);
}

void* op_view_3d(void* x, int64_t ne0, int64_t ne1, int64_t ne2,
                 size_t nb1, size_t nb2, size_t offset) {
    return ggml_view_3d(g_ctx, (struct ggml_tensor*)x, ne0, ne1, ne2, nb1, nb2, offset);
}

void* op_cpy(void* src, void* dst) {
    return ggml_cpy(g_ctx, (struct ggml_tensor*)src, (struct ggml_tensor*)dst);
}

void* op_soft_max(void* x) {
    return ggml_soft_max(g_ctx, (struct ggml_tensor*)x);
}

void* op_scale(void* x, float s) {
    return ggml_scale(g_ctx, (struct ggml_tensor*)x, s);
}

void* op_repeat(void* x, void* y) {
    return ggml_repeat(g_ctx, (struct ggml_tensor*)x, (struct ggml_tensor*)y);
}

void* op_diag_mask_inf(void* x, int32_t n_past) {
    return ggml_diag_mask_inf(g_ctx, (struct ggml_tensor*)x, n_past);
}

void* op_norm(void* x, float eps) {
    return ggml_norm(g_ctx, (struct ggml_tensor*)x, eps);
}

// ── Graph compute ───────────────────────────────────────────────────────

void* graph_new(int32_t size) {
    return ggml_new_graph_custom(g_ctx, (size_t)size, false);
}

void graph_build_forward_expand(void* graph, void* tensor) {
    ggml_build_forward_expand((struct ggml_cgraph*)graph,
                              (struct ggml_tensor*)tensor);
}

int32_t graph_compute(void* graph) {
    return (int32_t)ggml_backend_graph_compute(g_backend,
                                                (struct ggml_cgraph*)graph);
}

// ── Backend buffer ──────────────────────────────────────────────────────

void* backend_alloc_ctx_tensors() {
    return ggml_backend_alloc_ctx_tensors(g_ctx, g_backend);
}

void backend_buffer_free(void* buffer) {
    if (buffer) ggml_backend_buffer_free((ggml_backend_buffer_t)buffer);
}

void backend_tensor_set(void* tensor, const void* data, size_t offset, size_t size) {
    ggml_backend_tensor_set((struct ggml_tensor*)tensor, data, offset, size);
}

void backend_tensor_get(void* tensor, void* out, size_t offset, size_t size) {
    ggml_backend_tensor_get((struct ggml_tensor*)tensor, out, offset, size);
}

size_t backend_tensor_alignment() {
    return ggml_backend_get_alignment(g_backend);
}

} // extern "C"
