#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-webgpu.h"
#include "llama.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Stage 4.30 Probe 17 forward decl — internal llama.cpp helper exposing
// llama_model::tensors_by_name without pulling in src/llama-model.h.
// Defined in llama-model.cpp; declared in src/llama-model.h. Used by
// `webllm_get_tensor_data_hash` below to resolve a tensor by name and
// FNV-1a-32-hash the bytes the kernel actually reads from t->data
// post-load. Has C++ linkage (returns std::vector<...> &) so it must
// be declared outside the extern "C" block.
const std::vector<std::pair<std::string, ggml_tensor *>> &
llama_internal_get_tensor_map(const struct llama_model * model);

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

// ── Synthetic offload probe (P2-v2 Phase 2 follow-on, Task 13) ──────────
//
// Validates the Task 10 patch (ggml-jsep `supports_buft` + `offload_op`)
// in its native habitat: a tiny ggml graph whose MUL_MAT inputs live on
// `ggml_backend_cpu_buffer_type()` (a host buffer). The scheduler's
// `offload_op` path is gated on `ggml_backend_buffer_is_host(src->buffer)`
// — under the OUTCOME-D production chat path weights+KV land in
// `webgpu_buf` so the path never fires. This probe builds the graph in
// raw ggml (no libllama, no model load), enabling the scheduler with
// `op_offload=true`, and lets the JS-side caller observe whether
// `module.__jsep.counters.runOp` increments — i.e. whether MUL_MAT got
// routed to JSEP via offload.
//
// Status semantics (used in conjunction with JS-side counter snapshot —
// "Option α" from Task 13 design):
//   0  = scheduler ran graph successfully — JS must verify runOp delta
//   2  = JSEP not registered (probe inapplicable; non-JSEP build)
//   3  = scheduler init failed
//   4  = compute failed
//   5  = WebGPU not initialized (call webgpu_init first)
//
// The probe is intentionally exposed unconditionally (no #ifdef
// WEBLLM_BACKEND_JSEP). On a non-JSEP build the JSEP-name lookup
// returns false and the probe returns 2 cleanly — caller handles it.
//
// See: eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md
//      ("TL;DR (Task 11 + Task 12 update)" — Outcome D framing)

static char g_probe_log[2048] = {0};

static void probe_log_reset(void) {
    g_probe_log[0] = '\0';
}

static void probe_log_append(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    size_t cur = std::strlen(g_probe_log);
    if (cur >= sizeof(g_probe_log) - 1) { va_end(ap); return; }
    vsnprintf(g_probe_log + cur, sizeof(g_probe_log) - cur, fmt, ap);
    va_end(ap);
}

int32_t webllm_synthetic_offload_probe(void) {
    probe_log_reset();

    if (!g_backend) {
        probe_log_append("[probe] webgpu_init must be called first\n");
        return 5;
    }

    // 1. Locate JSEP and WebGPU registrations.
    ggml_backend_reg_t jsep_reg = nullptr;
    ggml_backend_reg_t webgpu_reg = nullptr;
    const size_t n_reg = ggml_backend_reg_count();
    probe_log_append("[probe] backend reg count=%zu\n", n_reg);
    for (size_t i = 0; i < n_reg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        const char* name = ggml_backend_reg_name(reg);
        if (!name) continue;
        probe_log_append("[probe]   reg[%zu] = %s\n", i, name);
        if (std::strcmp(name, "JSEP") == 0) jsep_reg = reg;
        else if (std::strcmp(name, "WebGPU") == 0) webgpu_reg = reg;
    }

    if (!jsep_reg) {
        probe_log_append("[probe] JSEP not registered — probe inapplicable\n");
        return 2;
    }
    if (!webgpu_reg) {
        probe_log_append("[probe] WebGPU reg missing\n");
        return 3;
    }

    // 2. Acquire JSEP backend (a fresh handle is fine — JSEP shares
    //    JS-side state via module.__jsep, so creating a new backend
    //    handle still routes to the same callback table). The existing
    //    g_backend is the WebGPU backend used by the rest of the
    //    bridge; reuse it directly.
    ggml_backend_dev_t jsep_dev = ggml_backend_reg_dev_get(jsep_reg, 0);
    if (!jsep_dev) {
        probe_log_append("[probe] JSEP reg has no device at index 0\n");
        return 3;
    }
    ggml_backend_t jsep_backend = ggml_backend_dev_init(jsep_dev, nullptr);
    if (!jsep_backend) {
        probe_log_append("[probe] ggml_backend_dev_init(JSEP) failed\n");
        return 3;
    }

    // 3. CPU backend for host-buffer-typed leaves and as scheduler fallback.
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        probe_log_append("[probe] ggml_backend_cpu_init failed\n");
        ggml_backend_free(jsep_backend);
        return 3;
    }

    // 4. Build a tiny ggml graph: A [K=128, M=64] × B [K=128] -> C [M=64].
    //    Tensors get F32, allocated on the CPU host_buft below.
    const int K = 128;
    const int M = 64;
    // Generous mem_size: tensor headers + graph. Data is in a separate
    // backend buffer so we just need ggml header overhead here.
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* probe_ctx = ggml_init(iparams);
    if (!probe_ctx) {
        probe_log_append("[probe] ggml_init failed\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(jsep_backend);
        return 3;
    }

    struct ggml_tensor* A = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* B = ggml_new_tensor_1d(probe_ctx, GGML_TYPE_F32, K);
    ggml_set_name(A, "probe_A");
    ggml_set_name(B, "probe_B");
    struct ggml_tensor* C = ggml_mul_mat(probe_ctx, A, B);
    ggml_set_name(C, "probe_C");

    // 5. Allocate A and B on CPU host_buft. The scheduler's offload_op
    //    check fires only when src->buffer is host memory.
    ggml_backend_buffer_type_t host_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t leaves_buf =
        ggml_backend_alloc_ctx_tensors_from_buft(probe_ctx, host_buft);
    if (!leaves_buf) {
        probe_log_append("[probe] alloc_ctx_tensors_from_buft(host) failed\n");
        ggml_free(probe_ctx);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(jsep_backend);
        return 3;
    }

    // Fill A with 1.0f and B with 1/K so each output of C ≈ 1.0
    // (loose correctness check; not the gate metric).
    {
        std::vector<float> a_data((size_t)K * M, 1.0f);
        std::vector<float> b_data((size_t)K, 1.0f / (float)K);
        ggml_backend_tensor_set(A, a_data.data(), 0, a_data.size() * sizeof(float));
        ggml_backend_tensor_set(B, b_data.data(), 0, b_data.size() * sizeof(float));
    }

    // 6. Create scheduler with [JSEP, WebGPU, CPU] and op_offload=true.
    //    op_offload=true is what enables the scheduler to consider
    //    `offload_op` candidates for ops whose src lives on host_buft.
    ggml_backend_t backends[3] = { jsep_backend, g_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, /*bufts=*/nullptr, /*n_backends=*/3,
        GGML_DEFAULT_GRAPH_SIZE, /*parallel=*/false, /*op_offload=*/true);
    if (!sched) {
        probe_log_append("[probe] ggml_backend_sched_new failed\n");
        ggml_backend_buffer_free(leaves_buf);
        ggml_free(probe_ctx);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(jsep_backend);
        return 3;
    }

    // 7. Build and compute the graph.
    struct ggml_cgraph* graph = ggml_new_graph(probe_ctx);
    ggml_build_forward_expand(graph, C);
    probe_log_append("[probe] graph built: 1 MUL_MAT, A=[%d,%d] B=[%d] C=[%d]\n",
                     K, M, K, M);
    probe_log_append("[probe] backends: [JSEP, WebGPU, CPU], op_offload=true\n");

    enum ggml_status st = ggml_backend_sched_graph_compute(sched, graph);
    probe_log_append("[probe] sched_graph_compute status=%d\n", (int)st);

    // 8. Cleanup. (The graph itself lives in probe_ctx and is freed
    //    via ggml_free below — no explicit graph free.)
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(leaves_buf);
    ggml_free(probe_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(jsep_backend);

    if (st != GGML_STATUS_SUCCESS) {
        probe_log_append("[probe] FAIL: compute returned %d\n", (int)st);
        return 4;
    }

    probe_log_append("[probe] PASS (status=0): JS must snapshot "
                     "module.__jsep.counters.runOp delta to confirm "
                     "JSEP fired the dispatch.\n");
    return 0;
}

const char* webllm_synthetic_probe_log(void) {
    return g_probe_log;
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
//
// Memory-pressure notes (cross-model loads on the wasm32 4 GiB cap):
//   - use_mmap=false: under Emscripten MEMFS, mmap keeps the entire
//     file's Uint8Array region pinned for the lifetime of the
//     llama_model. Peak heap then reaches ~2× model size (MEMFS
//     copy + mmap view + decode scratch) before weights are on GPU
//     and the host copy is released. Direct fread frees the host
//     copy as soon as ggml-webgpu finishes the upload.
//   - std::remove() after load drops the MEMFS file once the
//     weights are on GPU. Without this, sequential loads hit the
//     wasm32 cap because the prior model's GGUF bytes (770 MiB,
//     1017 MiB, ...) accumulate in MEMFS until the runtime exits.
void* webllm_load_model(const void* buf, size_t n_bytes) {
    const char* path = "/tmp/webllm-model.gguf";
    FILE* f = std::fopen(path, "wb");
    if (!f) return nullptr;
    size_t wrote = std::fwrite(buf, 1, n_bytes, f);
    std::fclose(f);
    if (wrote != n_bytes) {
        std::remove(path);
        return nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    // n_gpu_layers default is 0; we want all layers on the (only)
    // backend. ggml-webgpu registers itself as the only available
    // backend under Emscripten so layers route to it regardless of
    // this setting, but setting 999 is the canonical pattern.
    mparams.n_gpu_layers = 999;
    mparams.use_mmap = false;

    // Phase 3 (P2-v2 Option A-prime) JSEP-build device-hint.
    //
    // When BOTH WebGPU and JSEP are registered (the wasm-build-jsep
    // build), libllama enumerates only ONE GPU device by default. The
    // Phase 2 Task 11 hint pinned to WebGPU — weights+KV landed in
    // webgpu_buf, JSEP stayed structurally dormant (offload_op gates
    // on host buffers, never fires for webgpu_buf sources). Phase 3
    // (Option A-prime) inverts the choice: pin to JSEP so weights+KV
    // land in jsep_buf and every consumer op naturally targets JSEP
    // via the existing GpuDataManager-based dispatch path. Toggled
    // by the WEBLLM_PIN_TO_JSEP compile-time define (default 1 in the
    // jsep build via CMakeLists.txt).
    //
    // The non-JSEP build (only WebGPU registered) is unchanged —
    // mparams.devices stays NULL and libllama uses default discovery.
    //
    // See: TODO.md "Phase 3 entry: Option A-prime";
    //      eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md
    //      ("TL;DR (Task 11 + Task 12 update)" — Outcome D framing).
    {
        ggml_backend_reg_t webgpu_reg = nullptr;
        ggml_backend_reg_t jsep_reg   = nullptr;
        const size_t n_reg = ggml_backend_reg_count();
        for (size_t i = 0; i < n_reg; ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            if (!reg) continue;
            const char* name = ggml_backend_reg_name(reg);
            if (!name) continue;
            if (std::strcmp(name, "WebGPU") == 0) {
                webgpu_reg = reg;
            } else if (std::strcmp(name, "JSEP") == 0) {
                jsep_reg = reg;
            }
        }

        if (webgpu_reg && jsep_reg) {
#if defined(WEBLLM_PIN_TO_JSEP) && WEBLLM_PIN_TO_JSEP
            ggml_backend_reg_t selected_reg = jsep_reg;
            const char* selected_name = "JSEP";
            const char* excluded_name = "WebGPU";
#else
            ggml_backend_reg_t selected_reg = webgpu_reg;
            const char* selected_name = "WebGPU";
            const char* excluded_name = "JSEP";
#endif
            ggml_backend_dev_t selected_dev =
                ggml_backend_reg_dev_get(selected_reg, 0);
            if (selected_dev) {
                static ggml_backend_dev_t selected_devs[2];
                selected_devs[0] = selected_dev;
                selected_devs[1] = nullptr;
                mparams.devices = selected_devs;
                fprintf(stderr,
                        "[webllm] JSEP build detected: pinning libllama "
                        "devices to %s only (%s excluded from "
                        "model->devices)\n",
                        selected_name, excluded_name);
            } else {
                fprintf(stderr,
                        "[webllm] warning: %s reg found but no device "
                        "at index 0; falling back to default device "
                        "discovery\n",
                        selected_name);
            }
        } else if (!webgpu_reg) {
            fprintf(stderr,
                    "[webllm] warning: WebGPU backend not registered "
                    "(jsep_present=%d); falling back to default device "
                    "discovery\n",
                    jsep_reg ? 1 : 0);
        }
        // else: only WebGPU registered (non-JSEP build) — leave
        // mparams.devices = NULL for default behavior.
    }

    llama_model* model = llama_model_load_from_file(path, mparams);
    // Drop the MEMFS file regardless of load outcome — weights are
    // either on GPU (model != nullptr) or the load failed and the
    // bytes are no longer needed.
    std::remove(path);
    return model;
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
// ── Stage 4.17 Probe 7: per-node checkpoint dump ────────────────────────
// Backend-agnostic post-compute tensor inspection, gated on a runtime
// counter armed by JS via webllm_enable_node_dump(N). Wired into
// llama_context_params.cb_eval below; ggml-backend-sched copies tensor
// data back to host buffer before the ask=false call so t->data is
// host-readable regardless of underlying backend (CPU / WebGPU / JSEP).
//
// Allowlist focuses on layer-0 attention chain + final logits — the
// minimum surface to localize where JSEP diverges from the non-JSEP
// reference (`make wasm-build` produces "Paris" correctly; JSEP
// produces "in" / "inonic boso-" gibberish post-Stage-4.16).
static const char* const NODE_DUMP_ALLOWLIST[] = {
    // Stage 4.19 Probe 9a: upstream-of-Q-projection inputs to localize
    // where the 5.24e-4 Qcur-0 production delta originates. inp_embd
    // is the embedding lookup output (cb il=-1, no suffix); attn_norm-0
    // is the RMSNorm output that feeds Q/K/V projection at layer 0;
    // l_out-0 is the residual-stream output after layer 0.
    "inp_embd", "attn_norm-0", "l_out-0",
    "Qcur-0", "Kcur-0", "Vcur-0",
    "kq-0", "kq_soft_max-0", "kqv_out-0",
    "attn_out-0", "ffn_norm-0", "ffn_out-0",
    "result_norm", "result_output",
    nullptr,
};

static int g_node_dump_remaining = 0;
static int g_node_dump_idx = 0;

static bool node_dump_name_match(const char* name) {
    if (!name || !*name) return false;
    for (const char* const* p = NODE_DUMP_ALLOWLIST; *p; ++p) {
        if (std::strcmp(name, *p) == 0) return true;
    }
    return false;
}

static bool node_dump_cb(struct ggml_tensor* t, bool ask, void* /*user_data*/) {
    if (g_node_dump_remaining <= 0) return false;
    if (!t || !node_dump_name_match(t->name)) return false;
    if (ask) return true;  // request that backend copy data back to host

    // ask=false: t->data points to host-readable copy. ggml_get_f32_1d
    // handles dequant for F16 / BF16 / quant types. Some intermediate
    // tensors are non-contiguous (views) which ggml_get_f32_1d does NOT
    // support — skip those rather than abort the decode.
    float v[8] = {0};
    int n = (int)ggml_nelements(t);
    if (n > 8) n = 8;
    if (ggml_is_contiguous(t)) {
        for (int i = 0; i < n; ++i) v[i] = ggml_get_f32_1d(t, i);
    }
    // Stage 4.18 Probe 8b: log the producing backend (CPU vs JSEP/WebGPU)
    // so we can confirm or refute the "V runs on CPU" hypothesis from
    // Stage 4.17. ggml_backend_buffer_name returns "CPU", "JSEP", or
    // similar; that tells us which backend allocated the buffer that
    // holds this tensor's data — i.e., which backend the scheduler
    // routed the producing op to.
    const char* buf_name = (t->buffer ? ggml_backend_buffer_name(t->buffer) : "<null>");
    fprintf(stderr,
            "[CHECKPOINT idx=%d name=%s type=%d backend=%s ne=[%lld,%lld,%lld,%lld] "
            "contig=%d first8=[%g,%g,%g,%g,%g,%g,%g,%g]]\n",
            g_node_dump_idx, t->name, (int)t->type, buf_name,
            (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3],
            ggml_is_contiguous(t) ? 1 : 0,
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

    // Stage 4.31 Probe 18 Shape A: widen the readback for `kqv_out-0`
    // from the 8-element first8 window to the full tensor and emit
    // mean / abs-max / abs-min / NaN-count / Inf-count over every
    // element. Targets the suspected first8-window blindness — the
    // post-attention output's first8 is dominated by V[pos=0] (which
    // multiplies the position-0 softmax row that the causal mask pins
    // to [1, 0, 0, …]); positions 1+ are unmonitored by the existing
    // first8 dump. Only fires for `kqv_out-0` and only when contiguous;
    // every other allowlisted tensor keeps the cheap first8-only path.
    //
    // Stage 4.32 Probe 19b: extend aggregate stats to `kq-0` and
    // `kq_soft_max-0` to localize if the divergence originates upstream
    // of the V x softmax mat-mul.
    if (ggml_is_contiguous(t) && (
        std::strcmp(t->name, "kqv_out-0") == 0 ||
        std::strcmp(t->name, "kq-0") == 0 ||
        std::strcmp(t->name, "kq_soft_max-0") == 0)) {

        int total = (int) ggml_nelements(t);
        double sum_v = 0.0;
        double abs_max = 0.0;
        double abs_min = 0.0;
        bool abs_min_set = false;
        int nan_count = 0;
        int inf_count = 0;
        for (int i = 0; i < total; ++i) {
            float x = ggml_get_f32_1d(t, i);
            if (std::isnan(x)) {
                nan_count++;
                continue;
            }
            if (std::isinf(x)) {
                inf_count++;
                continue;
            }
            sum_v += (double) x;
            double ax = std::fabs((double) x);
            if (ax > abs_max) abs_max = ax;
            if (!abs_min_set || ax < abs_min) {
                abs_min = ax;
                abs_min_set = true;
            }
        }
        int finite = total - nan_count - inf_count;
        double mean = finite > 0 ? sum_v / (double) finite : 0.0;
        if (!abs_min_set) abs_min = 0.0;
        fprintf(stderr,
                "[CHECKPOINT-FULL idx=%d name=%s n_elements=%d finite=%d "
                "mean=%.9g abs_max=%.9g abs_min=%.9g nan=%d inf=%d]\n",
                g_node_dump_idx, t->name, total, finite,
                mean, abs_max, abs_min, nan_count, inf_count);

        // Stage 4.32 Probe 19: Element-wise dump for `kqv_out-0`.
        // Emit one line per row of 16 elements to keep stderr volume
        // manageable (12288 elements = 768 lines).
        if (std::strcmp(t->name, "kqv_out-0") == 0) {
            for (int i = 0; i < total; i += 16) {
                int count = std::min(16, total - i);
                fprintf(stderr, "[CHECKPOINT-IDX-DUMP idx=%d name=%s start=%d count=%d values=[",
                        g_node_dump_idx, t->name, i, count);
                for (int j = 0; j < count; ++j) {
                    fprintf(stderr, "%s%.9g", (j == 0 ? "" : ","), ggml_get_f32_1d(t, i + j));
                }
                fprintf(stderr, "]]\n");
            }
        }
    }

    g_node_dump_idx++;
    g_node_dump_remaining--;
    return true;
}

// Arm the per-node dump for the next `max_dumps` matched-name nodes.
// Idempotent: each call resets the counter and the printed idx.
void webllm_enable_node_dump(int32_t max_dumps) {
    g_node_dump_remaining = max_dumps > 0 ? max_dumps : 0;
    g_node_dump_idx = 0;
}

// Stage 4.24 Probe 12: libllama Q4_K dequant shim. Calls
// ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float (= dequantize_row_q4_K
// in ggml-quants.c). Used by the spike harness to cross-check the WGSL
// `load_q4_K` reconstruction against libllama's reference path on
// captured production weight bytes (`__probe10Capture.result.src0Bytes`).
//
// `src` is k/QK_K Q4_K super-blocks (144 bytes each); `dst` receives k
// f32 outputs. Caller is responsible for src/dst lifetime + alignment.
// `k` MUST be a multiple of QK_K (256). Returns 0 on success, -1 on
// alignment violation.
int32_t webllm_dequantize_q4_K(const void* src, float* dst, int32_t k) {
    if (!src || !dst || k <= 0 || (k % 256) != 0) return -1;
    const struct ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (!traits || !traits->to_float) return -1;
    traits->to_float(src, dst, (int64_t) k);
    return 0;
}

// Stage 4.26 Probe 14: libllama Q4_K × Q8_K matmul shim. Mirrors the path
// inside ggml_compute_forward_mul_mat for type=Q4_K: each src1 row is
// quantized f32 → block_q8_K via the CPU type traits' from_float, then
// vec_dot_q4_K_q8_K is invoked once per (m, n) output element. dst layout
// matches libllama: dst[n*M + m] (column-major within each output row).
//
// Used by the spike harness to score libllama's CPU matmul against an f64
// reference on the same captured Q-projection inputs that Probe 10 already
// scores the WGSL kernel against. Tells us whether libllama is the
// imprecise side of the historical 5.24e-4 cross-module disagreement.
//
// Inputs:
//   src0_q4k: M*K Q4_K weights (M rows × K cols, packed Q4_K super-blocks)
//   src1_f32: N*K f32 activations (N rows × K cols)
//   dst_f32:  M*N f32 outputs (libllama [M] = ne[0])
// Returns 0 on success, -1 on bad args, -2 on missing type traits, -3 on
// malloc failure.
int32_t webllm_q4k_q8k_matmul(const void* src0_q4k, const float* src1_f32,
                              float* dst_f32, int32_t M, int32_t K, int32_t N) {
    if (!src0_q4k || !src1_f32 || !dst_f32) return -1;
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) return -1;

    const struct ggml_type_traits_cpu* q8_traits =
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const struct ggml_type_traits_cpu* q4_traits =
        ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float) return -2;
    if (!q4_traits || !q4_traits->vec_dot) return -2;

    const size_t nb_q4k_row = ggml_row_size(GGML_TYPE_Q4_K, (int64_t) K);
    const size_t nb_q8k_row = ggml_row_size(GGML_TYPE_Q8_K, (int64_t) K);

    void* src1_q8k = std::malloc((size_t) N * nb_q8k_row);
    if (!src1_q8k) return -3;

    for (int32_t n = 0; n < N; ++n) {
        q8_traits->from_float(src1_f32 + (size_t) n * (size_t) K,
                              (char*) src1_q8k + (size_t) n * nb_q8k_row,
                              (int64_t) K);
    }

    for (int32_t n = 0; n < N; ++n) {
        const char* vy = (const char*) src1_q8k + (size_t) n * nb_q8k_row;
        for (int32_t m = 0; m < M; ++m) {
            const char* vx = (const char*) src0_q4k + (size_t) m * nb_q4k_row;
            q4_traits->vec_dot((int) K,
                               &dst_f32[(size_t) n * (size_t) M + (size_t) m],
                               /*bs=*/0,
                               vx, /*bx=*/0,
                               vy, /*by=*/0,
                               /*nrc=*/1);
        }
    }
    std::free(src1_q8k);
    return 0;
}

// Stage 4.30 Probe 17: post-load tensor->data byte-hash peek.
//
// Resolves `name` against llama_model::tensors_by_name (via the internal
// `llama_internal_get_tensor_map` helper) and FNV-1a-32-hashes
// `ggml_nbytes(t)` bytes from `t->data`. Writes the byte count to
// `*out_size_ptr` if non-null, and returns the hash (0 on miss).
//
// Used by the spike harness to compare against a JS-side `GgufParser`
// reference hash of the same tensor's GGUF bytes. Closes suspect 2
// (`ffn_norm.weight` gain-vector mis-load) by direct measurement on the
// non-set_tensor pathway that Stage 4.29's CPU-side hook did not see
// (Outcome P-16-silent: 0/7 fires for the bypass-weights, GGUF mmap-
// direct host buft owns them).
//
//   CLEAN (hash matches): suspect 2 dies; pivot to suspect 3 / pre-Qcur.
//   DIRTY (hash mismatch): mis-load CONFIRMED via mmap → CPU-op path.
uint32_t webllm_get_tensor_data_hash(void* model_handle, const char* name,
                                     uint32_t* out_size_ptr) {
    if (out_size_ptr) *out_size_ptr = 0;
    if (!model_handle || !name) return 0;
    const llama_model* model = static_cast<const llama_model*>(model_handle);
    const auto& tensors = llama_internal_get_tensor_map(model);
    for (const auto& kv : tensors) {
        if (kv.first != name) continue;
        const ggml_tensor* t = kv.second;
        if (!t || !t->data) return 0;
        const size_t nbytes = ggml_nbytes(t);
        const uint8_t* bytes = static_cast<const uint8_t*>(t->data);
        uint32_t h = 2166136261u;
        for (size_t i = 0; i < nbytes; ++i) {
            h ^= bytes[i];
            h *= 16777619u;
        }
        if (out_size_ptr) *out_size_ptr = (uint32_t) nbytes;
        return h;
    }
    return 0;
}

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
    cparams.cb_eval = node_dump_cb;
    cparams.cb_eval_user_data = nullptr;
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

// Returns vocab size (number of tokens in the model's tokenizer).
// Used by JS to size the logits Float32Array view created from
// the pointer returned by webllm_get_logits.
int32_t webllm_n_vocab(void* model) {
    if (!model) return 0;
    const llama_vocab* v = llama_model_get_vocab(static_cast<const llama_model*>(model));
    return llama_vocab_n_tokens(v);
}

// Tokenize text into model vocab IDs. Returns the number of tokens written
// to tokens_out, OR a negative number whose absolute value is the required
// buffer size if n_tokens_max was too small (mirrors upstream llama_tokenize
// semantics — JS-side caller grows the buffer and retries). add_bos=1 to
// prepend BOS, parse_special=1 to recognize <|...|>-style added tokens.
// Export is via -sEXPORTED_FUNCTIONS in CMakeLists.txt (matches the
// rest of the webllm_* surface in this block — no KEEPALIVE macro needed).
int32_t webllm_tokenize(
    void* model_handle,
    const char* text,
    int32_t n_text,
    int32_t* tokens_out,
    int32_t n_tokens_max,
    int32_t add_bos,
    int32_t parse_special)
{
    if (!model_handle || !text || !tokens_out) return 0;
    auto* model = static_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return 0;
    return llama_tokenize(
        vocab, text, n_text,
        tokens_out, n_tokens_max,
        add_bos != 0,
        parse_special != 0);
}

// Detokenize ids back to a UTF-8 byte buffer. Returns the number of bytes
// written, or a negative count whose absolute value is the required buffer
// size if n_text_max was too small. Mirrors upstream llama_detokenize.
// remove_special=0, unparse_special=0 — sensible defaults for the wrapper;
// the streaming detokenizer in tokenizer.ts handles special-token control.
int32_t webllm_detokenize(
    void* model_handle,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text_out,
    int32_t n_text_max)
{
    if (!model_handle || !tokens || !text_out) return 0;
    auto* model = static_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return 0;
    return llama_detokenize(
        vocab, tokens, n_tokens,
        text_out, n_text_max,
        /*remove_special=*/false,
        /*unparse_special=*/false);
}

int32_t webllm_token_bos(void* model_handle) {
    if (!model_handle) return -1;
    auto* model = static_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return -1;
    return llama_vocab_bos(vocab);
}

int32_t webllm_token_eos(void* model_handle) {
    if (!model_handle) return -1;
    auto* model = static_cast<llama_model*>(model_handle);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return -1;
    return llama_vocab_eos(vocab);
}

// Get a metadata string by key. Mirrors llama_model_meta_val_str:
// returns the length of the string on success, -1 on missing key.
// The output buffer must be sized for the value + null terminator.
// Caller-side retry-on-truncation uses the upstream pattern (call
// once with a small buffer to read the required length, then again
// with the right size).
int32_t webllm_get_metadata(
    void* model_handle,
    const char* key,
    char* buf,
    int32_t buf_size)
{
    if (!model_handle || !key || !buf || buf_size <= 0) return -1;
    auto* model = static_cast<llama_model*>(model_handle);
    return llama_model_meta_val_str(model, key, buf, (size_t) buf_size);
}

// Typed hyperparam accessors. Each returns an int32_t; -1 on
// missing model handle. These read from llama.cpp's parsed
// hyperparams (faster than meta_val_str, no string round-trip).
int32_t webllm_n_ctx_train(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_ctx_train(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_embd(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_embd(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_layer(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_layer(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_head(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_head(static_cast<llama_model*>(model_handle));
}
int32_t webllm_n_head_kv(void* model_handle) {
    if (!model_handle) return -1;
    return llama_model_n_head_kv(static_cast<llama_model*>(model_handle));
}
// Per-context KV-cache size in tokens (= the n_ctx the wrapper passed
// to webllm_create_context, possibly clamped to model's n_ctx_train).
int32_t webllm_n_ctx(void* ctx_handle) {
    if (!ctx_handle) return 0;
    return (int32_t) llama_n_ctx(static_cast<llama_context*>(ctx_handle));
}

// Drop tokens [p0, p1) for seq_id from the context's KV cache.
// p1 = -1 means "to the end". Used by truncateKVCache and by
// loadKVCache after a full state-set followed by truncation.
// The bool return of llama_memory_seq_rm is intentionally discarded:
// partial-removal failure is non-fatal for the wrapper's truncation
// use-case (a partially-removed sequence still satisfies "tokens >=
// keepLen are gone").
void webllm_kv_seq_rm(void* ctx_handle, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx_handle) return;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, (llama_seq_id) seq_id, (llama_pos) p0, (llama_pos) p1);
}

// Clear all sequences. Equivalent to a full resetKVCache.
void webllm_kv_clear(void* ctx_handle) {
    if (!ctx_handle) return;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, /*data=*/true);
}

// Get the byte size needed to serialize seq_id's KV state.
// Returns 0 on failure (or empty seq).
int32_t webllm_state_seq_get_size(void* ctx_handle, int32_t seq_id) {
    if (!ctx_handle) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_get_size(ctx, (llama_seq_id) seq_id);
}

// Copy seq_id's KV state into a caller-provided buffer.
// Returns the number of bytes written, 0 on failure.
// Caller must size dst to at least webllm_state_seq_get_size.
int32_t webllm_state_seq_get_data(
    void* ctx_handle,
    void* dst,
    int32_t size,
    int32_t seq_id)
{
    if (!ctx_handle || !dst || size <= 0) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_get_data(
        ctx, (uint8_t*) dst, (size_t) size, (llama_seq_id) seq_id);
}

// Restore seq_id's KV state from a caller-provided buffer.
// Returns positive on success, 0 on failure (per upstream contract).
// The buffer must have been produced by webllm_state_seq_get_data
// from a context with the SAME model + n_ctx + flash_attn flag.
int32_t webllm_state_seq_set_data(
    void* ctx_handle,
    const void* src,
    int32_t size,
    int32_t dest_seq_id)
{
    if (!ctx_handle || !src || size <= 0) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return (int32_t) llama_state_seq_set_data(
        ctx, (const uint8_t*) src, (size_t) size, (llama_seq_id) dest_seq_id);
}

// Get pointer to embeddings for the i-th token of the last decode.
// ith=-1 → use llama_get_embeddings (returns the pooled embedding
// when pooling_type != NONE, or the last-position embedding when
// pooling_type == NONE). Otherwise llama_get_embeddings_ith(ith)
// for per-position embeddings (Bucket-D pre-pool tap).
//
// Returns a pointer into ctx-owned memory; valid until the next
// decode call. JS-side caller wraps it as a Float32Array view of
// length n_embd. Caller must NOT free.
const float* webllm_get_embeddings(void* ctx_handle, int32_t ith) {
    if (!ctx_handle) return nullptr;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    return ith < 0 ? llama_get_embeddings(ctx) : llama_get_embeddings_ith(ctx, ith);
}

// Diagnostic: read the perf context counters directly. Used by P2.1.A
// to verify whether upstream's graph-reuse fast path is actually
// engaging on our single-token decode loop. Returns:
//   field=0  → n_decode (total decode batches)
//   field=1  → n_reused (graphs reused; should ≈ n_decode if reuse is on)
//   field=2  → n_p_eval (prompt-eval token count, prefill)
//   field=3  → n_eval   (decode token count, ≈ n_decode for batch=1)
// Other fields return 0. Cheap; no async work — pure counter read.
int32_t webllm_perf_counter(void* ctx_handle, int32_t field) {
    if (!ctx_handle) return 0;
    auto* ctx = static_cast<llama_context*>(ctx_handle);
    llama_perf_context_data d = llama_perf_context(ctx);
    switch (field) {
        case 0: return d.n_eval; // close enough; ggml builds a graph per decode call
        case 1: return d.n_reused;
        case 2: return d.n_p_eval;
        case 3: return d.n_eval;
        default: return 0;
    }
}

} // extern "C"
