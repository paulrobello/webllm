cmake_minimum_required(VERSION 3.13)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../llama.cpp" CACHE PATH "Path to llama.cpp repo")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}")
endif()

project(webllm-wasm C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Disable everything we don't need
set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
set(GGML_WEBGPU_JSPI OFF CACHE BOOL "" FORCE)
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
add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

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
        "_op_view_2d,_op_view_3d,_op_cpy,_op_soft_max,_op_scale,"
        "_op_repeat,_op_diag_mask_inf,_op_norm,_op_get_rows,"
        "_graph_new,_graph_build_forward_expand,_graph_compute,"
        "_backend_alloc_ctx_tensors,_backend_buffer_free,"
        "_backend_tensor_set,_backend_tensor_get,_backend_tensor_alignment"
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
        "-sASSERTIONS=0"
    )
    set_target_properties(webllm-wasm PROPERTIES
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
