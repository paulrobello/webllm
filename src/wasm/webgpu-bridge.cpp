#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-webgpu.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static ggml_backend_t g_backend = nullptr;
static std::vector<struct ggml_context*> g_ctx_stack;

static struct ggml_context* current_ctx() {
    return g_ctx_stack.empty() ? nullptr : g_ctx_stack.back();
}

extern "C" {

// ── Heap allocator wrappers ────────────────────────────────────────────
// Thin shims around stdlib malloc/free with explicit signatures the
// Emscripten linker can see during EXPORTED_FUNCTIONS marshaling. Under
// -sMEMORY64=1 -sWASM_BIGINT=1, custom exports correctly return BigInt
// pointers to JS, while stdlib _malloc/_free in Emscripten 5.0.6 return
// JS Number (truncated). Used by the §31a sub-probe to measure the
// MEMORY64 heap cap; see eval/reports/memory64-probe-2026-04-28/SUMMARY.md.
void* bridge_malloc(size_t size) {
    return std::malloc(size);
}

void bridge_free(void* ptr) {
    std::free(ptr);
}

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
    // Free all contexts (bottom-up)
    for (auto it = g_ctx_stack.rbegin(); it != g_ctx_stack.rend(); ++it) {
        if (*it) ggml_free(*it);
    }
    g_ctx_stack.clear();
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
}

// ── Context management (stack-based) ────────────────────────────────────

int32_t ctx_create(size_t mem_size) {
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    auto* ctx = ggml_init(params);
    if (!ctx) return -1;
    g_ctx_stack.push_back(ctx);
    return (int32_t)(g_ctx_stack.size() - 1);
}

void ctx_free() {
    if (!g_ctx_stack.empty()) {
        auto* ctx = g_ctx_stack.back();
        g_ctx_stack.pop_back();
        if (ctx) ggml_free(ctx);
    }
}

// ── Tensor creation ─────────────────────────────────────────────────────

void* tensor_new_1d(int32_t type, int32_t ne0) {
    return ggml_new_tensor_1d(current_ctx(), (ggml_type)type, ne0);
}

void* tensor_new_2d(int32_t type, int32_t ne0, int32_t ne1) {
    return ggml_new_tensor_2d(current_ctx(), (ggml_type)type, ne0, ne1);
}

void* tensor_new_3d(int32_t type, int32_t ne0, int32_t ne1, int32_t ne2) {
    return ggml_new_tensor_3d(current_ctx(), (ggml_type)type, ne0, ne1, ne2);
}

void* tensor_new_4d(int32_t type, int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3) {
    return ggml_new_tensor_4d(current_ctx(), (ggml_type)type, ne0, ne1, ne2, ne3);
}

void tensor_set_name(void* tensor, const char* name) {
    ggml_set_name((struct ggml_tensor*)tensor, name);
}

// ── Tensor properties ───────────────────────────────────────────────────

int32_t tensor_nelements(void* tensor) {
    return ggml_nelements((const struct ggml_tensor*)tensor);
}

int32_t tensor_nbytes(void* tensor) {
    return ggml_nbytes((const struct ggml_tensor*)tensor);
}

int32_t tensor_type(void* tensor) {
    return (int32_t)((const struct ggml_tensor*)tensor)->type;
}

int32_t tensor_ne(void* tensor, int32_t dim) {
    return ((const struct ggml_tensor*)tensor)->ne[dim];
}

int32_t tensor_nb(void* tensor, int32_t dim) {
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
    return ggml_mul_mat(current_ctx(), (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_add(void* a, void* b) {
    return ggml_add(current_ctx(), (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_mul(void* a, void* b) {
    return ggml_mul(current_ctx(), (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_rms_norm(void* x, float eps) {
    return ggml_rms_norm(current_ctx(), (struct ggml_tensor*)x, eps);
}

void* op_silu(void* x) {
    return ggml_silu(current_ctx(), (struct ggml_tensor*)x);
}

void* op_gelu(void* x) {
    return ggml_gelu(current_ctx(), (struct ggml_tensor*)x);
}

void* op_rope(void* x, void* pos, int32_t n_dims, int32_t mode, int32_t n_ctx_orig,
              float freq_base, float freq_scale, float ext_factor,
              float attn_factor, float beta_fast, float beta_slow) {
    return ggml_rope_ext(current_ctx(), (struct ggml_tensor*)x, (struct ggml_tensor*)pos, nullptr,
                         n_dims, mode, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
}

void* op_reshape_2d(void* x, int32_t ne0, int32_t ne1) {
    return ggml_reshape_2d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1);
}

void* op_reshape_3d(void* x, int32_t ne0, int32_t ne1, int32_t ne2) {
    return ggml_reshape_3d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, ne2);
}

void* op_permute(void* x, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3) {
    return ggml_permute(current_ctx(), (struct ggml_tensor*)x, dim0, dim1, dim2, dim3);
}

void* op_cont(void* x) {
    return ggml_cont(current_ctx(), (struct ggml_tensor*)x);
}

void* op_view_2d(void* x, int32_t ne0, int32_t ne1, int32_t nb1, size_t offset) {
    return ggml_view_2d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, nb1, offset);
}

void* op_view_3d(void* x, int32_t ne0, int32_t ne1, int32_t ne2,
                 int32_t nb1, int32_t nb2, size_t offset) {
    return ggml_view_3d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, ne2, nb1, nb2, offset);
}

void* op_cpy(void* src, void* dst) {
    return ggml_cpy(current_ctx(), (struct ggml_tensor*)src, (struct ggml_tensor*)dst);
}

void* op_soft_max(void* x) {
    return ggml_soft_max(current_ctx(), (struct ggml_tensor*)x);
}

void* op_soft_max_ext(void* x, void* mask, float scale, float max_bias) {
    return ggml_soft_max_ext(current_ctx(), (struct ggml_tensor*)x,
                             (struct ggml_tensor*)mask, scale, max_bias);
}

// Fused scaled-dot-product attention. Replaces opMulMat(K,Q) + opSoftMaxExt + opMulMat(V,attn).
// Q: [head_dim, n_tokens, n_head]
// K: [head_dim, n_kv,     n_head_kv]   (must match V layout; F16/Q4_0/Q8_0 for VEC/TILE paths)
// V: [head_dim, n_kv,     n_head_kv]   (n.b. matches K layout — different from old V cache layout)
// mask: [n_kv_padded, n_tokens] F32, broadcast over heads. -inf masked, 0 visible. May be nullptr.
// scale: typically 1/sqrt(head_dim).
// max_bias: ALiBi max bias; pass 0 for standard causal attention.
// logit_softcap: Gemma-style logit soft-cap; pass 0 for standard.
// Returns: [head_dim, n_head, n_tokens] (note the dim order — caller must reshape/permute).
void* op_flash_attn_ext(void* q, void* k, void* v, void* mask,
                        float scale, float max_bias, float logit_softcap) {
    return ggml_flash_attn_ext(current_ctx(),
                               (struct ggml_tensor*)q,
                               (struct ggml_tensor*)k,
                               (struct ggml_tensor*)v,
                               (struct ggml_tensor*)mask,
                               scale, max_bias, logit_softcap);
}

void op_flash_attn_ext_set_prec(void* a, int32_t prec) {
    ggml_flash_attn_ext_set_prec((struct ggml_tensor*)a, (enum ggml_prec)prec);
}

void op_flash_attn_ext_add_sinks(void* a, void* sinks) {
    ggml_flash_attn_ext_add_sinks((struct ggml_tensor*)a, (struct ggml_tensor*)sinks);
}

// Fused silu(a) * b (LLaMA SwiGLU FFN), avoids three separate dispatches.
void* op_swiglu_split(void* a, void* b) {
    return ggml_glu_split(current_ctx(), (struct ggml_tensor*)a,
                          (struct ggml_tensor*)b, GGML_GLU_OP_SWIGLU);
}

void* op_scale(void* x, float s) {
    return ggml_scale(current_ctx(), (struct ggml_tensor*)x, s);
}

void* op_repeat(void* x, void* y) {
    return ggml_repeat(current_ctx(), (struct ggml_tensor*)x, (struct ggml_tensor*)y);
}

void* op_get_rows(void* a, void* b) {
    return ggml_get_rows(current_ctx(), (struct ggml_tensor*)a, (struct ggml_tensor*)b);
}

void* op_argmax(void* src) {
    return ggml_argmax(current_ctx(), (struct ggml_tensor*)src);
}

void* op_top_k(void* src, int32_t k) {
    return ggml_top_k(current_ctx(), (struct ggml_tensor*)src, k);
}

void* op_diag_mask_inf(void* x, int32_t n_past) {
    return ggml_diag_mask_inf(current_ctx(), (struct ggml_tensor*)x, n_past);
}

void* op_norm(void* x, float eps) {
    return ggml_norm(current_ctx(), (struct ggml_tensor*)x, eps);
}

// ── Graph compute ───────────────────────────────────────────────────────

void* graph_new(size_t size) {
    return ggml_new_graph_custom(current_ctx(), size, false);
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
    return ggml_backend_alloc_ctx_tensors(current_ctx(), g_backend);
}

void backend_buffer_free(void* buffer) {
    if (buffer) ggml_backend_buffer_free((ggml_backend_buffer_t)buffer);
}

void backend_tensor_set(void* tensor, const void* data, size_t offset, size_t size) {
    ggml_backend_tensor_set((struct ggml_tensor*)tensor, data, offset, size);
}

// Batched upload for per-forward leaf inputs (pos / token ids / mask).
// Saves 1-2 JS->WASM FFI hops per forward. A null tensor pointer skips
// that slot (for the common "no mask on single-token decode" case).
void backend_tensor_set3(
    void* t1, const void* d1, size_t sz1,
    void* t2, const void* d2, size_t sz2,
    void* t3, const void* d3, size_t sz3
) {
    if (t1) ggml_backend_tensor_set((struct ggml_tensor*)t1, d1, 0, sz1);
    if (t2) ggml_backend_tensor_set((struct ggml_tensor*)t2, d2, 0, sz2);
    if (t3) ggml_backend_tensor_set((struct ggml_tensor*)t3, d3, 0, sz3);
}

void backend_tensor_get(void* tensor, void* out, size_t offset, size_t size) {
    ggml_backend_tensor_get((struct ggml_tensor*)tensor, out, offset, size);
}

int32_t backend_tensor_get_async_begin(void* tensor, size_t offset, size_t size) {
    return ggml_backend_webgpu_tensor_get_async_begin(
        (const struct ggml_tensor*)tensor,
        offset,
        size);
}

int32_t backend_tensor_get_async_poll(int32_t request_id) {
    return ggml_backend_webgpu_tensor_get_async_poll(request_id);
}

void backend_tensor_get_async_finish(int32_t request_id, void* out, size_t size) {
    ggml_backend_webgpu_tensor_get_async_finish(request_id, out, size);
}

void backend_tensor_get_async_cancel(int32_t request_id) {
    ggml_backend_webgpu_tensor_get_async_cancel(request_id);
}

int32_t backend_tensor_get_async_callback_support() {
    return 1;
}

int32_t backend_tensor_alignment() {
    return ggml_backend_get_alignment(g_backend);
}

void webgpu_set_graph_profiling_enabled(int32_t enabled) {
    ggml_backend_webgpu_set_graph_profiling_enabled(enabled);
}

int32_t webgpu_last_graph_profile_valid() {
    return ggml_backend_webgpu_last_graph_profile_valid();
}

int32_t webgpu_last_graph_profile_breakdown_available() {
    return ggml_backend_webgpu_last_graph_profile_breakdown_available();
}

double webgpu_last_graph_profile_total_ms() {
    return ggml_backend_webgpu_last_graph_profile_total_ms();
}

double webgpu_last_graph_profile_matmul_ms() {
    return ggml_backend_webgpu_last_graph_profile_matmul_ms();
}

double webgpu_last_graph_profile_attention_ms() {
    return ggml_backend_webgpu_last_graph_profile_attention_ms();
}

double webgpu_last_graph_profile_encode_overhead_ms() {
    return ggml_backend_webgpu_last_graph_profile_encode_overhead_ms();
}

int32_t webgpu_last_graph_profile_dispatch_count() {
    return ggml_backend_webgpu_last_graph_profile_dispatch_count();
}

// ── llama.cpp model lifecycle ────────────────────────────────────────────

// Load a llama.cpp model from a buffer in JS heap memory.
//
// Strategy: write the GGUF bytes to Emscripten MEMFS at a fixed virtual
// path, then call llama_model_load_from_file. This avoids exposing the
// FS runtime method to JS — the buffer ↔ MEMFS round trip happens
// entirely inside the bridge.
//
// The buffer is COPIED into MEMFS — caller may free `buf` after this
// returns. Returns model handle on success, nullptr on failure.
void* webllm_load_model(const void* buf, size_t n_bytes) {
    const char* path = "/tmp/webllm-model.gguf";
    FILE* f = std::fopen(path, "wb");
    if (!f) return nullptr;
    size_t wrote = std::fwrite(buf, 1, n_bytes, f);
    std::fclose(f);
    if (wrote != n_bytes) return nullptr;

    llama_model_params mparams = llama_model_default_params();
    // n_gpu_layers default is 0; we want all layers on the (only)
    // backend. ggml-webgpu registers itself as the only available
    // backend under Emscripten so layers route to it regardless of
    // this setting, but setting 999 is the canonical pattern.
    mparams.n_gpu_layers = 999;
    return llama_model_load_from_file(path, mparams);
}

void webllm_free_model(void* model) {
    if (model) llama_model_free(static_cast<llama_model*>(model));
}

// Create a llama_context for the given model.
//
//   n_ctx        — KV cache size in tokens. 0 = use the model's
//                  training-time default (n_ctx_train).
//   embeddings   — 0 = causal LM (logits output), 1 = embedder mode
//                  (pooled embeddings output).
//   pooling_type — 0=NONE, 1=MEAN, 2=CLS, 3=LAST. Ignored when
//                  embeddings=0; required for BERT-family encoders
//                  and for Bucket-D self-embedding (P4).
//   flash_attn   — 0=DISABLED, 1=ENABLED (maps to llama_flash_attn_type).
//                  Replaces the per-arch FA gating that lived in TS
//                  ModelInference under the legacy path.
//
// Returns context handle on success, nullptr on failure.
//
// Threading is pinned to 1 — pthreads are not enabled in this
// Emscripten build (CMakeLists.txt does not set
// USE_PTHREADS / -pthread).
void* webllm_create_context(void* model, int32_t n_ctx, int32_t embeddings,
                            int32_t pooling_type, int32_t flash_attn) {
    if (!model) return nullptr;
    llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0) cparams.n_ctx = (uint32_t) n_ctx;
    cparams.embeddings = embeddings != 0;
    cparams.pooling_type = (enum llama_pooling_type) pooling_type;
    cparams.flash_attn_type = (enum llama_flash_attn_type) flash_attn;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    return llama_init_from_model(static_cast<llama_model*>(model), cparams);
}

void webllm_free_context(void* ctx) {
    if (ctx) llama_free(static_cast<llama_context*>(ctx));
}

// Decode n_tokens tokens at sequence positions [past_len, past_len+n_tokens).
// All tokens go to seq_id=0. Logits are computed for the LAST token only
// (sufficient for greedy / single-step decode; P5 spec-decode will need
// a richer logits-mask API).
//
// Returns 0 on success, non-zero llama.cpp status on failure.
//
// The caller-side ABI uses int32_t past_len and int32_t n_tokens — these
// are llama_pos and (logically) batch size. Cast happens at the boundary.
int32_t webllm_decode(void* ctx, const int32_t* token_ids, int32_t n_tokens,
                      int32_t past_len) {
    if (!ctx || n_tokens <= 0) return -1;
    llama_context* lctx = static_cast<llama_context*>(ctx);

    llama_batch batch = llama_batch_init(n_tokens, /*embd*/ 0, /*n_seq_max*/ 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = token_ids[i];
        batch.pos[i] = past_len + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    int32_t status = llama_decode(lctx, batch);
    llama_batch_free(batch);
    return status;
}

// Returns pointer to logits for the i-th token of the last decode batch.
// ith=-1 → use llama_get_logits (returns logits for the most recent
// logits-flagged token). Otherwise llama_get_logits_ith(ith) which
// indexes into the batch's logits-flagged tokens.
//
// The pointer is into ctx-owned memory; valid until the next decode
// call. Caller must NOT free it.
const float* webllm_get_logits(void* ctx, int32_t ith) {
    if (!ctx) return nullptr;
    llama_context* lctx = static_cast<llama_context*>(ctx);
    return ith < 0 ? llama_get_logits(lctx) : llama_get_logits_ith(lctx, ith);
}

} // extern "C"
