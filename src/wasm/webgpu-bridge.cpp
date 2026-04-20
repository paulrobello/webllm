#include <cstdint>
#include <cstring>
#include <webgpu/webgpu_cpp.h>

extern "C" {

int32_t webgpu_init() { return 0; }
void webgpu_shutdown() {}

static constexpr int32_t MAX_BUFFERS = 256;
static wgpu::Buffer buffers[MAX_BUFFERS];
static int32_t next_buffer_id = 0;

int32_t webgpu_create_buffer(uint64_t size, uint32_t usage) {
    if (next_buffer_id >= MAX_BUFFERS) return -1;
    int32_t id = next_buffer_id++;
    return id;
}

void webgpu_write_buffer(int32_t id, const void* data, uint64_t size) {}
void webgpu_read_buffer(int32_t id, void* out, uint64_t size) {}

void webgpu_destroy_buffer(int32_t id) {
    if (id >= 0 && id < MAX_BUFFERS && buffers[id]) {
        buffers[id].Destroy();
        buffers[id] = nullptr;
    }
}

void webgpu_mul_mat(int32_t a, int32_t b, int32_t out, int32_t m, int32_t n, int32_t k, int32_t type_a, int32_t type_b) {}
void webgpu_flash_attn(int32_t q, int32_t k, int32_t v, int32_t out, int32_t head_dim, int32_t n_heads, int32_t seq_len, float scale) {}
void webgpu_rope(int32_t tensor, int32_t freqs, int32_t out, int32_t dim, float freq_base, float freq_scale) {}
void webgpu_rms_norm(int32_t x, int32_t weight, int32_t out, int32_t rows, int32_t cols, float eps) {}
void webgpu_soft_max(int32_t x, int32_t out, int32_t rows, int32_t cols, float scale) {}
void webgpu_cpy(int32_t src, int32_t dst, int32_t size) {}
void webgpu_get_rows(int32_t x, int32_t indices, int32_t out, int32_t n_rows, int32_t row_size) {}
void webgpu_set_rows(int32_t x, int32_t indices, int32_t out, int32_t n_rows, int32_t row_size) {}

} // extern "C"
