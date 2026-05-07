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

option(WEBLLM_BUILD_MEM64 "Build the wasm64 webllm-wasm-mem64 target instead of webllm-wasm. Set ON in src/wasm/build-mem64/ for the MEMORY64 cap probe (see docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md). Must be combined with -DCMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1 so ggml is also built as wasm64." OFF)

# WEBLLM_BACKEND selects the ggml backend mix wired into the WASM artifact.
#   "default" (or unset): ggml-webgpu only — the canonical Tier 3 build.
#   "jsep":               ggml-webgpu + ggml-jsep — the P2-v2 JSEP-style
#                         prototype. Output basename gains a "-jsep" suffix
#                         (webllm-wasm-jsep.{js,wasm}). Spec:
#                         docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md
set(WEBLLM_BACKEND "default" CACHE STRING "Backend variant: 'default' or 'jsep'")
set_property(CACHE WEBLLM_BACKEND PROPERTY STRINGS "default" "jsep")

if(WEBLLM_BACKEND STREQUAL "jsep")
    set(WEBLLM_BACKEND_JSEP ON)
else()
    set(WEBLLM_BACKEND_JSEP OFF)
endif()

# Disable everything we don't need
set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
if(WEBLLM_BACKEND_JSEP)
    set(GGML_JSEP ON CACHE BOOL "" FORCE)
endif()
# JSPI replaces ASYNCIFY for WebGPU async readback. Flipped ON 2026-05-05
# to unblock Tier 3 P1 — adding any new wasm export under -sASYNCIFY +
# -exceptions + libllama static initializers traps in __wasm_call_ctors
# (related upstream issues #18045, #18254, #25551 — all OPEN). JSPI uses
# native browser Promise integration; sidesteps the wasm-ld
# static-init transformation that ASYNCIFY exposes.
# Closure report: eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md
set(GGML_WEBGPU_JSPI ON CACHE BOOL "" FORCE)
set(GGML_WEBGPU_GPU_PROFILE ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
# GGML_CPU was OFF in the legacy graph-builder path: the project hand-
# rolled forward graphs in TS and dispatched directly to ggml-webgpu, so
# no scheduler fallback was needed. The Tier 3 migration uses libllama's
# llama_model_load_from_file, which throws "no CPU backend found" at
# llama-model.cpp:852 when use_extra_bufts is enabled. ON re-enables the
# CPU backend so libllama's loader finds the device it expects for extra
# buffer types and host-side staging. Layers still route to ggml-webgpu
# via n_gpu_layers=999 — CPU code paths are present in the binary but
# are not on the hot decode path.
set(GGML_CPU ON CACHE BOOL "" FORCE)
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

# Disable llama.cpp extras we don't need (tests/tools/examples default OFF
# when LLAMA_STANDALONE is OFF, but set explicitly to be safe).
# Force LLAMA_WASM_MEM64=OFF: its default is ON which adds -sMEMORY64=1
# globally, making libllama.a a wasm64 archive incompatible with the wasm32
# webllm-wasm target. The mem64 build goes through a separate build dir.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_MEM64     OFF CACHE BOOL "" FORCE)

# Upstream bug: ggml-webgpu CMakeLists.txt line 80 links DawnWebGPU_TARGET
# unconditionally. Under Emscripten this variable is never set. Pre-set to empty.
set(DawnWebGPU_TARGET "" CACHE STRING "WebGPU target" FORCE)

# Pull in ggml build system (builds ggml-base + enabled backends)
add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

# Pull in llama.cpp top-level (EXCLUDE_FROM_ALL so only explicitly linked
# targets are built). The ggml target already exists at this point, so
# llama.cpp's CMakeLists.txt will skip its own ggml add_subdirectory.
add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama-cpp EXCLUDE_FROM_ALL)

# Emscripten needs an executable target to produce .js + .wasm output.
# add_library only produces an archive; add_executable triggers the linker.
if(EMSCRIPTEN)
    string(CONCAT EXPORTED_FUNCTIONS
        "_webgpu_init,_webgpu_shutdown,_malloc,_free,"
        "_bridge_malloc,_bridge_free,"
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
        "_webgpu_last_graph_profile_encode_overhead_ms,_webgpu_last_graph_profile_dispatch_count,"
        "_webllm_load_model,_webllm_free_model,"
        "_webllm_create_context,_webllm_free_context,"
        "_webllm_decode,_webllm_get_logits,"
        "_webllm_n_vocab,"
        "_webllm_tokenize,_webllm_detokenize,"
        "_webllm_token_bos,_webllm_token_eos,"
        "_webllm_get_metadata,"
        "_webllm_n_ctx_train,_webllm_n_embd,_webllm_n_layer,"
        "_webllm_n_head,_webllm_n_head_kv,_webllm_n_ctx,"
        "_webllm_kv_seq_rm,_webllm_kv_clear,"
        "_webllm_state_seq_get_size,_webllm_state_seq_get_data,_webllm_state_seq_set_data,"
        "_webllm_get_embeddings,"
        "_webllm_perf_counter,"
        "_webllm_synthetic_offload_probe,_webllm_synthetic_probe_log,"
        "_webllm_enable_node_dump"
    )

    # JSPI requires explicit listing of exports that may transitively call
    # async imports (WebGPU async readback, queue.onSubmittedWorkDone, etc).
    # Each listed export is wrapped in WebAssembly.promising at module init,
    # so the JS caller gets a Promise back. Anything that touches the WebGPU
    # backend belongs here. Functions that don't touch WebGPU (pure C
    # tokenization, tensor inspection, malloc) are intentionally excluded —
    # promising-wrapping them adds a per-call suspend-frame allocation.
    # NOTE: JSPI_EXPORTS uses bare wasm export names (no leading
    # underscore), unlike EXPORTED_FUNCTIONS. Emscripten puts these names
    # into a regex against the wasm-side export keys (e.g. "webgpu_init"),
    # not the Module-side aliases (e.g. "_webgpu_init"). Mismatch ⇒
    # silent no-op ⇒ "trying to suspend without WebAssembly.promising".
    string(CONCAT JSPI_EXPORTS
        "webgpu_init,"
        "ctx_create,ctx_free,"
        "backend_alloc_ctx_tensors,backend_buffer_free,"
        "backend_tensor_set,backend_tensor_set3,backend_tensor_get,"
        "backend_tensor_get_async_begin,backend_tensor_get_async_poll,"
        "backend_tensor_get_async_finish,backend_tensor_get_async_cancel,"
        "graph_compute,"
        "webllm_load_model,webllm_free_model,"
        "webllm_create_context,webllm_free_context,"
        "webllm_decode,webllm_get_logits,"
        "webllm_get_embeddings,"
        "webllm_synthetic_offload_probe"
    )

    if(WEBLLM_BUILD_MEM64)
        # ── MEMORY64 cap probe target.
        # Built only when -DWEBLLM_BUILD_MEM64=ON is passed at cmake configure
        # (driven by `make mem64-probe` in src/wasm/build-mem64/). The probe
        # binary links wasm64-against-wasm64 archives because the global
        # CMAKE_C_FLAGS / CMAKE_CXX_FLAGS include `-sMEMORY64=1`, which
        # propagates to the ggml add_subdirectory above and produces wasm64
        # ggml-base + ggml-webgpu archives.
        # Spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md
        add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)
        target_link_libraries(webllm-wasm-mem64 PRIVATE llama ggml-base ggml-webgpu)
        target_include_directories(webllm-wasm-mem64 PRIVATE
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include
            ${LLAMA_CPP_DIR}/ggml/src
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
        add_executable(webllm-wasm webgpu-bridge.cpp)
        if(WEBLLM_BACKEND_JSEP)
            target_link_libraries(webllm-wasm PRIVATE llama ggml-base ggml-webgpu ggml-jsep)
            set_target_properties(webllm-wasm PROPERTIES OUTPUT_NAME "webllm-wasm-jsep")
            # Phase 3 / Option A-prime — pin libllama to JSEP only so weights+KV
            # land in jsep_buf; every consumer op naturally targets JSEP via the
            # scheduler's GpuDataManager-based dispatch path. Inversion of the
            # Phase 2 Task 11 device-hint (which pinned to WebGPU only and left
            # JSEP structurally dormant). See TODO.md "Phase 3 entry: Option
            # A-prime" + eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md.
            target_compile_definitions(webllm-wasm PRIVATE WEBLLM_PIN_TO_JSEP=1)
        else()
            target_link_libraries(webllm-wasm PRIVATE llama ggml-base ggml-webgpu)
        endif()
        target_include_directories(webllm-wasm PRIVATE
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include
            ${LLAMA_CPP_DIR}/ggml/src
        )
        target_link_options(webllm-wasm PRIVATE
            "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
            "-sJSPI_EXPORTS=${JSPI_EXPORTS}"
            "-sEXPORTED_RUNTIME_METHODS=['stackAlloc','stackSave','stackRestore','stringToUTF8','lengthBytesUTF8','HEAPU8','HEAPF32','cwrap','Asyncify']"
            "-sMODULARIZE=1"
            "-sEXPORT_ES6=1"
            "-sUSE_ES6_IMPORT_META=1"
            "-sALLOW_MEMORY_GROWTH=1"
            "-sMAXIMUM_MEMORY=4GB"
            "-sSTACK_SIZE=8388608"
            "-sENVIRONMENT='web'"
            "-sASYNCIFY_STACK_SIZE=1048576"
            # WASM_BIGINT on wasm32: pointers stay i32 (size_t is uint32_t),
            # but i64 args at the JS↔WASM boundary (e.g. emwgpuWaitAny's
            # timeoutNS) need BigInt to round-trip correctly. The Tier 3
            # libllama load path triggers WebGPU async waits that fail with
            # "Cannot convert undefined to a BigInt" without this flag.
            # Browser support is universal as of Chrome 119+ / Safari 15+ /
            # Firefox 78+. The legacy graph-builder path didn't hit this
            # because it never engaged WebGPU's promise-await machinery
            # through libllama-style scheduler calls.
            "-sWASM_BIGINT=1"
            "-O3"
            "-sASSERTIONS=$<IF:$<BOOL:${WEBLLM_ASSERTIONS}>,1,0>"
        )
        set_target_properties(webllm-wasm PROPERTIES
            PREFIX ""
        )
    endif()
else()
    add_library(webllm-bridge webgpu-bridge.cpp)
    target_link_libraries(webllm-bridge PRIVATE llama ggml-base ggml-webgpu)
    target_include_directories(webllm-bridge PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/ggml/src
    )
endif()
