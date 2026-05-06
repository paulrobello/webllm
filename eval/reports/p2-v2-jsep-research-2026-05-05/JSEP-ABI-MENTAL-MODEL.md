# JSEP ABI Mental Model
**Date:** 2026-05-05
**Purpose:** Inform `ggml-jsep` design for webllm P2-v2.
**ORT ref:** `microsoft/onnxruntime` main @ `3e217610` (2026-05-06)

Files cited use repo-relative paths from `microsoft/onnxruntime`. JS side
under `js/web/lib/wasm/jsep/`, C++ side under
`onnxruntime/core/providers/js/`.

## 1. JS↔WASM call mechanism

The boundary is a small fixed table of JS callbacks injected into the
Emscripten `Module` at startup, and a few `EMSCRIPTEN_KEEPALIVE` C++
exports called *back* from JS during a kernel run.

- JS-side hand-off: `init.ts:194` looks up `module.jsepInit` (an exported
  Emscripten function) and calls it with the EP name and a fixed-shape
  array of callbacks: `[backend, jsepAlloc, jsepFree, jsepCopy,
  jsepCopyAsync, jsepCreateKernel, jsepReleaseKernel, jsepRun,
  jsepCaptureBegin, jsepCaptureEnd, jsepReplay]`
  (`init.ts:205-274`). `jsepInit` (defined in the Emscripten JS-library
  glue, not in the public source tree) stores them on `Module` as
  `Module.jsepAlloc`, `Module.jsepRun`, etc.
- C++-side calls: `EM_ASM` / `EM_ASM_INT` / `EM_ASM_PTR` macros are used
  inline at every call site. Examples:
  `WebGpuAllocator::Alloc` calls `Module.jsepAlloc($0)`
  (`onnxruntime/core/providers/js/allocator.cc:14`);
  `JsKernel::ComputeInternal` calls `Module.jsepRunKernel(...)`
  (`onnxruntime/core/providers/js/js_kernel.h:203-205`); kernel
  construction calls `Module.jsepCreateKernel(...)`
  (`js_kernel.h:28-29`); destructor calls `Module.jsepReleaseKernel(...)`
  (`js_kernel.h:83`).
- Synchronicity: `jsepRunKernel` is **synchronous from C++'s
  point of view** — `EM_ASM_INT` blocks WASM until the JS function
  returns an integer status (`js_kernel.h:203`). The JS implementation
  does *not* await GPU work; it only records command-encoder dispatches
  (`program-manager.ts:73-77`) and returns. Async-only paths
  (`jsepCopyAsync` for GPU→CPU readback, kernel error promises pushed
  into `errors[]` `init.ts:259`) require Asyncify or JSPI to actually
  block WASM. The async-readback path is the one that historically
  forces ASYNCIFY in ORT-Web's WASM build.
- Round trips per op: **one** `EM_ASM_INT(jsepRunKernel)` call per ONNX
  graph node. The kernel may internally call back into C++ via
  `JsepOutput` (one `_JsepOutput!` per output tensor — `init.ts:151`) to
  let ORT allocate the output tensor on the C++ side. So per op:
  1 JS→WASM result + N WASM→JS `JsepOutput` minor crossings (N =
  output count, usually 1).

## 2. Per-kernel descriptor shape

Two phases: **kernel creation (once)** and **kernel run (per node-step)**.

**Creation** (`jsepCreateKernel(kernelType: string, kernelId: number,
attribute: unknown)` — `init.ts:247-253`):

- `kernelType`: ONNX op name string ("MatMul", "Add", …).
- `kernelId`: opaque number — actually the C++ `OpKernel*` cast to
  `uintptr_t` (`js_kernel.h:28` passes `this`).
- `attribute`: the *parsed* attribute object, marshalled inline by the
  `JSEP_INIT_KERNEL_ATTRIBUTE` macro using `EM_ASM`'s JS-literal-object
  builder, e.g. `({alpha:$1, beta:$2}, valA, valB)` —
  `js_kernel.h:29, 57-69`. Numbers cross as f64; strings are not used
  here.

The JS side looks up the kernel run function in
`WEBGPU_OP_RESOLVE_RULES` (a `Map<string, [RunFunction,
ParseAttributeFunction?]>`, `op-resolve-rules.ts:63-162`) and stores a
`KernelInfo { kernelType, kernelName, kernelEntry, attributes }`
keyed by `kernelId` (`backend-webgpu.ts:708-721`).

**Run** (`jsepRunKernel(kernelId, contextDataOffset, sessionHandle,
errors)` — `init.ts:259-267`):

- The descriptor is a **flat `uintptr_t` array laid out in WASM linear
  memory**, allocated by `JsKernel::SerializeKernelContext`
  (`js_kernel.h:86-154`). Layout:
  ```
  [0] context_ptr          // OpKernelContext*, opaque to JS
  [1] input_count
  [2] output_count
  [3] custom_data_ptr
  [4] custom_data_size
  // then per input:
  [..] dataType            // ONNX TensorProto_DataType enum
  [..] data_ptr            // raw pointer; in JSEP-WebGPU mode this is a GpuDataId
  [..] dim_count
  [..] dim[0] ... dim[N-1]
  // placeholder inputs are 3 zeros
  ```
- JS reconstructs this with `module.getValue` reads in a loop
  (`init.ts:91-113`) into a `ComputeContextImpl` whose `inputs` are
  `TensorViewImpl{dataType, data, dims}` records. No copies — JS reads
  directly out of `module.HEAP8.buffer`.
- Outputs are *not* in the descriptor up front; JS calls back into C++
  via `Module._JsepOutput(opKernelContext, index, dimsPtr)` to ask ORT
  to allocate the output and return its `data` pointer (which is again
  a GpuDataId) — `init.ts:141-161`, `js_export.cc:8-24`.

## 3. GPU buffer lifecycle

GPU buffers are referenced **by integer handle**, never by pointer.
There is no copy on the boundary; the WASM heap holds the handle, the
JS side resolves it to a `GPUBuffer`.

- Allocator: ORT calls `WebGpuAllocator::Alloc(size)` for any tensor
  that lives on the WebGPU EP. That path is `EM_ASM_PTR { return
  Module.jsepAlloc($0); }` (`allocator.cc:14`) → `backend.alloc(size)`
  (`init.ts:210`) → `gpuDataManager.create(size).id`
  (`backend-webgpu.ts:700-702`). The integer ID is what ORT stores as
  the tensor's `DataRaw()` pointer. `jsepFree` is the symmetric path
  (`allocator.cc:24-27`, `init.ts:213`).
- ID → `GPUBuffer` resolution: `GpuDataManager` keeps a
  `Map<GpuDataId, {gpuData: {buffer: GPUBuffer, …}, originalSize}>`
  with bucketed-size free-lists for reuse
  (`gpu-data-manager.ts:84-113` for bucket table; the `GpuData`
  struct at `types.ts:25-29` is `{type, id, buffer}`). Resolution
  happens at run time inside `WebGpuBackend.run` via
  `this.gpuDataManager.get(data)` for each input
  (`backend-webgpu.ts:479-484`).
- CPU↔GPU transfers: `jsepCopy(src, dst, size, isSourceGpu)` covers
  CPU→GPU (`writeBuffer`) and GPU→GPU (`memcpy`); GPU→CPU is the
  separate async path `jsepCopyAsync` that does
  `device.queue.submit` + `mapAsync` (`init.ts:215-244`).
- Lifetime: ORT owns the *logical* tensor lifetime via its allocator;
  JS owns the *physical* `GPUBuffer` lifetime. Releasing an ID returns
  the buffer to a bucket free-list; outputs marked `-1` (temporary) are
  released at end-of-kernel and `-2` (persistent) are released at
  `releaseKernel` (`backend-webgpu.ts:526-535, 723-734`).
- IO binding: external `GPUBuffer`s can be registered to a GpuDataId
  without copying via
  `gpuDataManager.registerExternalBuffer` (`gpu-data-manager.ts:55`).

## 4. Scheduler / dispatch granularity

**The graph walker stays in C++/WASM.** ORT's existing scheduler walks
the ONNX graph node-by-node and invokes each `JsKernel::Compute`, which
fires one `jsepRunKernel` EM_ASM. JS receives one node at a time — there
is no whole-subgraph descriptor crossing the boundary.

- One JS↔WASM crossing per node (Q1). For an N-node graph, that's N
  EM_ASM trips per `session.run`.
- WebGPU command batching is *internal to JS*: `WebGpuBackend` keeps a
  single `GPUCommandEncoder` open and accumulates dispatches across
  multiple kernels. `flush()` (`backend-webgpu.ts:341-378`) ends the
  pass and submits when `pendingDispatchNumber >= maxDispatchNumber`
  (default 16, `:200`) or at session-end. This is how JSEP amortizes
  GPU submit cost without changing the per-kernel ABI.
- Pipeline cache: `ProgramManager.repo` is a `Map<string, Artifact>`
  where `Artifact = { programInfo, computePipeline,
  uniformVariablesInfo }` (`program-manager.ts:21-34`,
  `types.ts:126-130`). The cache key is a string built from program
  name + custom hint + per-input dependency descriptor
  (`'none'|'type'|'rank'|'dims'|'data'`,
  `backend-webgpu.ts:46-115`). Cache hits skip
  `device.createShaderModule` + `createComputePipeline`
  (`backend-webgpu.ts:626-631`).
- Inside a kernel, the kernel run function may issue multiple
  `context.compute(programInfo)` calls (e.g. a kernel that decomposes
  into several shaders). Each `compute` invocation goes through the
  same key lookup → potential pipeline build → bind group create →
  dispatch (`init.ts:115-139`, `program-manager.ts:35-89`,
  `backend-webgpu.ts:462-684`). Bind groups are *not* cached — they
  are rebuilt per dispatch (`program-manager.ts:56-60`).
- Graph-capture mode (`captureBegin`/`captureEnd`/`replay`,
  `init.ts:269-273`) records pipeline+bindGroup+dispatchGroup tuples
  per session, so a static-shape session can replay GPU commands
  without re-entering JS at all (`program-manager.ts:62-71`). This is
  the path SD-Turbo / static-shape models take.

## 5. Implications for ggml-jsep

**What ports cleanly:**

- The "callback table on Module" pattern (Q1) maps directly to the
  ggml-backend interface. ggml already expects a function-pointer
  vtable per backend (`init_tensor`, `set_tensor`, `cpy_tensor_async`,
  `graph_compute`, …). A `ggml-jsep` backend stub in C provides those
  vtable entries, each implemented as one EM_ASM_* call into a JS
  callback registered at startup. This is the minimum-viable ABI.
- Handle-based GPU buffer ownership (Q3) is exactly what
  `ggml_backend_buffer` wants: ggml allocates/frees buffers via the
  backend, and only stores an opaque `void*` per `ggml_tensor`. We
  pass an integer handle through it. No memory-layout coupling.
- WGSL pipeline cache keyed by `(op, dtype, shape-deps)`
  (`backend-webgpu.ts:95-115`) ports to ggml verbatim — same key
  ingredients (`op`, `src0->type`, shape if dynamic).
- The internal command-encoder batching (Q4, `flush()` after 16
  dispatches) is exactly the lever P2-v2 needs: it amortizes the
  per-dispatch JS-side cost without changing the ABI.

**What's different / awkward:**

- ORT's scheduler walks the graph node-by-node from C++ and each node
  becomes one EM_ASM. ggml has its own
  `ggml_backend_graph_compute(cgraph)` entrypoint that receives the
  *entire* `cgraph` — we can choose between (a) JSEP-style: have the
  ggml-jsep `graph_compute` walk the cgraph in WASM and EM_ASM per
  node (matches JSEP, simplest port), or (b) descriptor-once: serialize
  the whole cgraph into a flat buffer and EM_ASM once. (b) is the
  "real" P2-v2 win — it's strictly better than ORT here. Recommend
  starting with (a) for the matmul prototype, measure, then move to
  (b) if per-node EM_ASM is still hot.
- ORT's per-kernel `attribute` is **parsed at session-create** and
  stored on the JS side per-kernelId (`backend-webgpu.ts:714-721`).
  ggml's analogue is `tensor->op_params` — small fixed bytes per
  tensor, re-passed every graph compute. We don't get JSEP's
  amortize-once benefit unless we also introduce a "kernel
  registration" step at session/model-load time. For matmul/add/etc.
  the params are tiny so this is fine; for ops with large baked
  attributes (Conv kernel sizes etc.) it matters. Not load-bearing
  for LLM decode.
- ORT's tensor descriptor (Q2) carries `dataType` from
  `TensorProto_DataType`. ggml has its own `GGML_TYPE_*` enum and many
  more quantized types (Q4_K, Q6_K, IQ-quants). The JS dispatch table
  needs a `(op, ggml_type)` keying, not a single op-name lookup.
  `op-resolve-rules.ts` becomes a 2D map.
- ORT's `ComputeContext.compute(programInfo)` lets a single kernel
  emit multiple sub-shader dispatches inside one EM_ASM
  (`init.ts:115-139`). Useful for matmul that splits into pack →
  matmul → unpack. ggml ops are typically one-shader-per-op so this
  flexibility is nice-to-have, not needed for the prototype.
- **No embedded graph optimizer in JSEP** — ORT does its graph
  optimization C++-side before JSEP sees nodes. So JSEP doesn't
  compete with ggml's scheduler. ✓
- Async readback (`jsepCopyAsync`, `init.ts:235`) is the same shape
  as the patched-llama.cpp async readback we already have. The JSEP
  pattern doesn't help us avoid ASYNCIFY/JSPI for that path.

**Open questions for Phase 2 prototype:**

1. **Graph-once vs node-per-EM_ASM?** The whole P2-v2 thesis is "JS↔WASM
   crossings are the bottleneck." ORT pays N crossings per `session.run`
   and is fast enough. We've measured 18× regression at the per-WebGPU-
   command level (much finer than per-op). Per-op should be fine, but
   prototype both and measure on a single matmul-heavy decode loop.
2. **Descriptor encoding cost.** ORT's flat `uintptr_t[]` descriptor +
   `module.getValue` loop in JS (`init.ts:91-113`) is non-trivial per
   call. For ggml's per-tensor `ne[4]`/`nb[4]`/`op_params` we want a
   leaner encoding — possibly a fixed-size struct view directly into
   WASM heap, not a marshalled JS object. Probe this.
3. **Pipeline cache key.** ggml has many more dtypes; `inputDependencies`
   = `'type'` is mandatory (not `'dims'` default).
4. **Custom data path.** ORT's `kernelCustomData` / `customDataBuffer`
   (`types.ts:177-181`) lets a kernel cache JS-side state across calls
   (e.g. precomputed shader strings). For ggml this maps to per-op
   pipeline state. Needed for any kernel with a non-trivial JS-side
   build cost (matmul-packed has one). Worth porting.
5. **C++ exports back from JS.** JSEP needs `_JsepOutput` and
   `_JsepGetNodeName` exported from WASM (`init.ts:151,252`). For
   ggml-jsep, output allocation already happens C++-side at graph-
   compute setup, so we may not need a `JsepOutput` analogue — JS only
   needs read access to existing output tensor handles, not the ability
   to ask C++ to allocate. Simpler.
