cmake_minimum_required(VERSION 3.13)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../llama.cpp" CACHE PATH "Path to llama.cpp repo")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}")
endif()

project(webllm-wasm C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Debug build toggle. With assertions on, Emscripten includes runtime
# checks and GGML_ASSERT messages are preserved with file:line context —
# invaluable for diagnosing WASM aborts. Assertions off is the shipping
# configuration; it's significantly faster (often 2×+ for decode).
option(WEBLLM_ASSERTIONS "Enable Emscripten -sASSERTIONS=1 in the WASM build" OFF)

# Disable everything we don't need
set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
set(GGML_WEBGPU_JSPI OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU_GPU_PROFILE ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

# Upstream bug: ggml-webgpu CMakeLists.txt line 80 links DawnWebGPU_TARGET
# unconditionally. Under Emscripten this variable is never set. Pre-set to empty.
set(DawnWebGPU_TARGET "" CACHE STRING "WebGPU target" FORCE)

# Pull in ggml build system (builds ggml-base + enabled backends)
# For wasm32 (default target)
add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

# For MEMORY64 target: Reuse wasm32 GGML libraries as a starting point.
# A full separate mem64 GGML build is complex due to CMake environment setup.
# For the probe phase, we attempt linking wasm32 libraries with wasm64 flags.
# If this fails (expected), it's a probe finding.
if(EMSCRIPTEN)
    set(GGML_MEM64_BUILD_DIR "${CMAKE_BINARY_DIR}/ggml")  # Use the wasm32 GGML build
endif()

# Emscripten needs an executable target to produce .js + .wasm output.
# add_library only produces an archive; add_executable triggers the linker.
if(EMSCRIPTEN)
    string(CONCAT EXPORTED_FUNCTIONS
        "_webgpu_init,_webgpu_shutdown,_malloc,_free,"
        "_ctx_create,_ctx_free,"
        "_tensor_new_1d,_tensor_new_2d,_tensor_new_3d,_tensor_new_4d,"
        "_tensor_set_name,_tensor_nelements,_tensor_nbytes,_tensor_type,"
        "_tensor_ne,_tensor_nb,_tensor_data,"
        "_tensor_set_data,_tensor_get_data,"
        "_op_mul_mat,_op_add,_op_mul,_op_rms_norm,_op_silu,_op_gelu,"
        "_op_rope,_op_reshape_2d,_op_reshape_3d,_op_permute,_op_cont,"
        "_op_view_2d,_op_view_3d,_op_cpy,_op_soft_max,_op_soft_max_ext,_op_swiglu_split,_op_scale,"
        "_op_flash_attn_ext,_op_flash_attn_ext_set_prec,_op_flash_attn_ext_add_sinks,"
        "_op_repeat,_op_diag_mask_inf,_op_norm,_op_get_rows,_op_argmax,_op_top_k,"
        "_graph_new,_graph_build_forward_expand,_graph_compute,"
        "_backend_alloc_ctx_tensors,_backend_buffer_free,"
        "_backend_tensor_set,_backend_tensor_set3,_backend_tensor_get,"
        "_backend_tensor_get_async_begin,_backend_tensor_get_async_poll,"
        "_backend_tensor_get_async_finish,_backend_tensor_get_async_cancel,"
        "_backend_tensor_get_async_callback_support,_backend_tensor_alignment,"
        "_webgpu_set_graph_profiling_enabled,"
        "_webgpu_last_graph_profile_valid,_webgpu_last_graph_profile_breakdown_available,_webgpu_last_graph_profile_total_ms,"
        "_webgpu_last_graph_profile_matmul_ms,_webgpu_last_graph_profile_attention_ms,"
        "_webgpu_last_graph_profile_encode_overhead_ms,_webgpu_last_graph_profile_dispatch_count"
    )
    add_executable(webllm-wasm webgpu-bridge.cpp)
    target_link_libraries(webllm-wasm PRIVATE ggml-base ggml-webgpu)
    target_include_directories(webllm-wasm PRIVATE
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/ggml/src
    )
    target_link_options(webllm-wasm PRIVATE
        "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
        "-sEXPORTED_RUNTIME_METHODS=['stackAlloc','stackSave','stackRestore','stringToUTF8','lengthBytesUTF8','HEAPU8','HEAPF32','cwrap','Asyncify']"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sUSE_ES6_IMPORT_META=1"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMAXIMUM_MEMORY=4GB"
        "-sSTACK_SIZE=8388608"
        "-sENVIRONMENT='web'"
        "-sASYNCIFY_STACK_SIZE=1048576"
        "-O3"
        "-sASSERTIONS=$<IF:$<BOOL:${WEBLLM_ASSERTIONS}>,1,0>"
    )
    set_target_properties(webllm-wasm PROPERTIES
        PREFIX ""
    )

    # ── MEMORY64 cap probe target (separate build, dead code on `main`
    # until promoted; spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md).
    # Reuses the same source + same exports list as `webllm-wasm`; deltas:
    #   -sMEMORY64=1            64-bit pointers on the WASM side
    #   -sWASM_BIGINT=1         BigInt at the JS↔WASM boundary
    #   -sMAXIMUM_MEMORY=16GB   probes the upper bound of Chrome's heap grant
    # Links against mem64-compiled GGML libraries.
    # Note: For now, reuses wasm32 GGML libraries. The linker will fail if wasm64
    # object files are truly required. This is expected as a probe finding.
    add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)
    target_link_libraries(webllm-wasm-mem64 PRIVATE
        ${GGML_MEM64_BUILD_DIR}/src/libggml-base.a
        ${GGML_MEM64_BUILD_DIR}/src/ggml-webgpu/libggml-webgpu.a
    )
    target_include_directories(webllm-wasm-mem64 PRIVATE
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/ggml/src
    )
    target_compile_options(webllm-wasm-mem64 PRIVATE
        "-sMEMORY64=1"
        "-sWASM_BIGINT=1"
    )
    target_link_options(webllm-wasm-mem64 PRIVATE
        "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
        "-sEXPORTED_RUNTIME_METHODS=['stackAlloc','stackSave','stackRestore','stringToUTF8','lengthBytesUTF8','HEAPU8','HEAPF32','cwrap','Asyncify']"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sUSE_ES6_IMPORT_META=1"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMEMORY64=1"
        "-sWASM_BIGINT=1"
        "-sMAXIMUM_MEMORY=16GB"
        "-sSTACK_SIZE=8388608"
        "-sENVIRONMENT='web'"
        "-sASYNCIFY_STACK_SIZE=1048576"
        "-O3"
        "-sASSERTIONS=$<IF:$<BOOL:${WEBLLM_ASSERTIONS}>,1,0>"
    )
    set_target_properties(webllm-wasm-mem64 PROPERTIES
        PREFIX ""
    )
else()
    add_library(webllm-bridge webgpu-bridge.cpp)
    target_link_libraries(webllm-bridge PRIVATE ggml-base ggml-webgpu)
    target_include_directories(webllm-bridge PRIVATE
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/ggml/src
    )
endif()
