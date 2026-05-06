# ggml Backend Contract for P2-v2
**Date:** 2026-05-05
**Purpose:** Identify minimum surface a `ggml-jsep` backend must implement.
**ggml ref:** `webllm-browser-patches @ b54503497` (head of local llama.cpp branch).

All citations are repo-relative paths under `/Users/probello/Repos/llama.cpp/`.
A ggml backend is built out of four nested vtables — **buffer-type**,
**buffer**, **device**, **backend (stream)** — plus a **registry**. The
buffer & buffer-type vtables own *storage*; the device & backend vtables
own *capabilities + execution*; the registry advertises the device(s) to
the rest of ggml.

## 1. Vtable contract

Per `ggml/src/ggml-backend-impl.h`:

- **`ggml_backend_i`** (l. 105-140 — execution / streaming):
  required: `get_name`, `free`, `graph_compute`. All seven of
  `set_tensor_async`, `get_tensor_async`, `set_tensor_2d_async`,
  `get_tensor_2d_async`, `cpy_tensor_async`, `synchronize`,
  `event_record`/`event_wait`, `graph_optimize`,
  `graph_plan_{create,free,update,compute}` are **optional** and may be
  `NULL`. `synchronize` becomes required *if* any async op is supplied
  (l. 117 comment). The CPU backend `ggml_backend_cpu_i`
  (`ggml/src/ggml-cpu/ggml-cpu.cpp:193-210`) leaves all but
  `get_name`/`free`/`graph_plan_*`/`graph_compute` as `NULL`. The WebGPU
  backend `ggml_backend_webgpu_i`
  (`ggml/src/ggml-webgpu/ggml-webgpu.cpp:3550-3567`) additionally
  implements `set_tensor_async`, `synchronize`, and event hooks; sets
  `graph_plan_*` to NULL.
- **`ggml_backend_device_i`** (l. 160-202 — capabilities):
  required: `get_name`, `get_description`, `get_memory`, `get_type`,
  `get_props`, `init_backend`, `get_buffer_type`, `supports_op`,
  `supports_buft`. Optional: `get_host_buffer_type`,
  `buffer_from_host_ptr`, `offload_op`, `event_*`.
- **`ggml_backend_buffer_type_i`** (l. 17-29 — allocator):
  required: `get_name`, `alloc_buffer`, `get_alignment`. Optional:
  `get_max_size` (default `SIZE_MAX`), `get_alloc_size` (default
  `ggml_nbytes`), `is_host` (default `false`).
- **`ggml_backend_buffer_i`** (l. 41-62 — per-buffer ops):
  required: `get_base`, `set_tensor`, `get_tensor`, `memset_tensor`,
  `clear`. Optional and may be `NULL`: `free_buffer`, `init_tensor`
  (only needed for tensor-extras), `set_tensor_2d`, `get_tensor_2d`,
  `cpy_tensor` (return false → fallback to staging copy), `reset`.
- **`ggml_backend_reg_i`** (l. 214-224 — registry):
  required: `get_name`, `get_device_count`, `get_device`. Optional:
  `get_proc_address`. Top-level struct must set
  `api_version = GGML_BACKEND_API_VERSION` (= 2, l. 11).

## 2. Buffer / allocator contract

`ggml_backend_buffer_t` is a thin handle: struct
`{ iface, buft, context, size, usage }` (l. 64-70). The backend supplies
its own **`context`** void* — handle into native storage (an integer ID,
a `wgpu::Buffer`, etc.). ggml never dereferences `context`.

Crucially, `tensor->data` is **not** a real pointer for non-host
backends. It's a per-buffer-type sentinel offset that the backend uses to
recover an in-buffer offset. WebGPU does this with
`webgpu_ptr_base = 0x1000` (`ggml-webgpu.cpp:110`), and
`ggml_webgpu_tensor_offset(t) = (uintptr_t)t->data - webgpu_ptr_base + t->view_offs`
(`:112-114`). `get_base` returns the sentinel (`:3582-3585`); the
allocator hands out `tensor->data = base + cumulative_offset`. So the
contract is: **`(buffer, tensor->data − get_base())` is the in-buffer
byte offset for that tensor**. `set_tensor`/`get_tensor` receive that
buffer + tensor + a `(offset, size)` byte range and must move bytes
to/from the supplied host pointer.

Routing: `ggml_backend_sched` decides which backend a tensor is on by
the **buffer type** of `tensor->buffer`. `supports_buft(dev, buft)`
returns true iff the device can execute on tensors stored in `buft`
(`ggml-backend.cpp:846-865`). For `ggml-jsep` the natural answer is
"only my own buft", same as WebGPU.

`is_host = false` for any GPU-style backend; ggml then knows it must
route reads/writes through `set_tensor`/`get_tensor` instead of
`memcpy` on the raw pointer.

## 3. `graph_compute` contract

Signature (from `ggml-backend-impl.h:130`):
`enum ggml_status (*graph_compute)(ggml_backend_t, struct ggml_cgraph *)`.

The cgraph passed in is the layout from `ggml-impl.h:329-347`:
`{ size, n_nodes, n_leafs, nodes**, grads**, grad_accs**, leafs**,
use_counts*, visited_hash_set, order, uid }`. The backend iterates
`cgraph->nodes[0..n_nodes)` in order; each node is a `ggml_tensor *`
with `op`, `op_params[16]`, `src[GGML_MAX_SRC]` already wired up.

What the backend **may assume** at entry:
- All input/output tensors are pre-allocated (pointers/handles already
  set on `tensor->buffer` and `tensor->data`).
- All weights are already uploaded — split-handling and host→device
  copies happen *before* `graph_compute` is invoked, in
  `ggml_backend_sched_compute_splits`
  (`ggml-backend.cpp:1554-1675`). Per-split inputs are explicitly
  copied via `cpy_tensor_async` or `set_tensor` before the split's
  `graph_compute` call (`:1664-1672`).
- Output tensors are also pre-allocated; the backend just writes to
  them in place. There's no JSEP-style `JsepOutput` callback.

Return: `GGML_STATUS_SUCCESS`, `GGML_STATUS_ALLOC_FAILED`,
`GGML_STATUS_FAILED`, or `GGML_STATUS_ABORTED`.

Sync semantics: `graph_compute` is **always async if the backend says
so**. The public sync entrypoint
`ggml_backend_graph_compute` (`ggml-backend.cpp:444-448`) calls
`graph_compute_async` then `synchronize`. The scheduler uses the async
form internally (`:1678, :1700`). For a JS-callback backend, returning
immediately after recording is fine; `synchronize` does the GPU wait.
The patched WebGPU backend's `synchronize`
(`ggml-webgpu.cpp:3545-3548`) drives the queue under JSPI.

**ggml hands off the whole graph, not per-node.** `graph_compute` is
called once per *split* (a contiguous sub-graph routed to the same
backend). Inside, the backend is free to walk node-by-node (ggml-cpu,
`ggml-cpu.cpp:170-191`, just calls `ggml_graph_compute(cgraph, &cplan)`
into the threadpool) or to do whole-graph passes (ggml-webgpu does
fusion + multi-op encoding before any submit,
`ggml-webgpu.cpp:3347-3450`). This is the structural lever JSEP-on-ggml
gets that JSEP-on-ORT does not.

## 4. Scheduler interaction

`ggml_backend_sched` is the multi-backend dispatcher
(`ggml-backend.cpp:820-1714`). Backends are passed at `sched_new` time
in **priority order** (lowest index = highest priority). Per node,
assignment runs in `ggml_backend_sched_split_graph`
(`:1014`); the per-node decision is in
`ggml_backend_sched_backend_id_from_cur` (`:878-933`):

1. If the tensor already lives in some backend's buffer
   (`tensor->buffer` set), the buffer type pins it — choose the
   highest-priority backend whose `supports_buft(buft)` AND
   `supports_op(op)` are true (`:845-865, :880-884`).
2. Else if it's a graph input (`GGML_TENSOR_FLAG_INPUT`), assign
   the **last** (lowest-priority, assumed CPU) backend (`:902-906`).
3. Else, route to where the weights live (`:909-929`); allow
   `offload_op`-capable backends to steal hot ops from CPU (`:919-925`).
4. Pass-2/pass-3 expand assignments adjacent-wise, gated by
   `supports_op` (`:1006-1011, :1072-1110`).

`supports_op(op)` is the **per-op feasibility check**: given a fully-
typed `ggml_tensor` (op, type, src types/shapes), can this backend run
it? The CPU backend's version returns true for almost everything
(`ggml-cpu.cpp:423-440` — only checks for buffer extras like AMX);
WebGPU's enumerates supported `(op, dtype)` combos
(`ggml-webgpu.cpp:4409-4600+`). `supports_buft(buft)` is the same
question for *storage*: can the backend execute on tensors in this
buffer-type? CPU says yes to host buffers; WebGPU says yes only to its
own `buft`.

CPU fallback for a partial backend: when ggml-jsep returns false for an
op, the scheduler's expand passes (`:1072-1110`) refuse to extend the
GPU-backend assignment over that node, leaving it `-1`; then the
unassigned-node pass picks the next backend down the priority list — in
practice the CPU backend (its `supports_op` is permissive). The split
boundary at the unsupported op forces a `set_tensor`/`get_tensor`
copy through `compute_splits` (`:1554-1675`). For a matmul-only stub:
just register CPU at lower priority, return `true` from
`supports_op` only for `GGML_OP_MUL_MAT` with the supported type
combos, return `false` for everything else, and the scheduler will
peel the rest off to CPU automatically. **No need to implement the
other ops as stubs that abort.**

## 5. CPU reference walk

`ggml-cpu/ggml-cpu.cpp:170-191`:

```c
static enum ggml_status ggml_backend_cpu_graph_compute(
        ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = backend->context;
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, ...);
    if (cpu_ctx->work_size < cplan.work_size) { /* realloc work_data */ }
    cplan.work_data = cpu_ctx->work_data;
    cplan.abort_callback = cpu_ctx->abort_callback;
    return ggml_graph_compute(cgraph, &cplan);
}
```

So the CPU backend *delegates back into the core ggml graph executor*
that ships in `ggml-cpu/ggml-cpu.c`. The vtable's `graph_compute` is a
thin wrapper that allocates scratch + runs `ggml_graph_compute` on the
CPU threadpool. There's no event/synchronize machinery
(`:201, :207-208` are NULL) because compute is fully synchronous from
the caller's POV. Buffer plumbing is just `malloc/free` + memcpy
(`ggml-backend.cpp:2213-2293`); `is_host = true`
(`:2322-2326`) so `tensor->data` is a real `void*`. **This is the
contract floor**: a backend that implements only `get_name`, `free`,
`graph_compute`, and a buffer-type that is `is_host=false` with
`set_tensor`/`get_tensor` callbacks is a complete backend. Everything
else is opt-in optimization.

By contrast `ggml-webgpu`'s `graph_compute`
(`ggml-webgpu.cpp:3347-3450`) does the work P2-v2 wants to relocate:
it walks `cgraph->nodes`, builds bind groups, **issues
`device.CreateCommandEncoder()` / `BeginComputePass()` / `Dispatch` /
`Submit` calls from C++**. Under emdawnwebgpu, every one of those wgpu
API calls becomes a per-call shim crossing into JS to drive Dawn-on-
WebGPU. That is a **per-WebGPU-API-call** crossing rate, not per-op —
and is what the §22 `webllm_perf_counter` instrumentation surfaced as
the 18× decode regression. The JSEP redirect collapses all of those
crossings to **one call per node** (per-node JSEP) or **one call per
graph** (graph-once JSEP).

## 6. Minimum surface for matmul-stub prototype

**Required `ggml_backend_i` entries:**

- `get_name` → return `"jsep"`.
- `free` → release backend context.
- `graph_compute` → walk `cgraph->nodes`, emit one `EM_ASM_INT` per
  node into JS (`Module.jsepRunNode(opcode, src_handles[], dst_handle,
  op_params_ptr)`); JS records WebGPU dispatches and returns status.
  Initial cut: per-node EM_ASM (matches JSEP). Later: serialize whole
  cgraph and EM_ASM once.
- `synchronize` → EM_ASM into JS to flush command encoder + `mapAsync`
  any pending readback. Required because `set_tensor_async` exists.
- `set_tensor_async` (optional but useful) → `Module.jsepCopy(srcHostPtr,
  dstHandle, size, /*isSourceGpu=*/false)`. Mirrors WebGPU's
  `WriteBuffer` path.
- All other entries → `NULL`. (No `graph_plan_*`, no `event_*` for v0.)

**Required `ggml_backend_buffer_type_i`:**

- `get_name` → `"jsep_buf"`.
- `alloc_buffer` → EM_ASM `Module.jsepAlloc(size)` returning an integer
  handle, store it in `buffer->context`. Wrap with
  `ggml_backend_buffer_init`.
- `get_alignment` → 16 or 256 (storage-buffer offset alignment).
- `get_max_size` → 128 MiB (per-binding cap doctrine in CLAUDE.md).
- `is_host` → false.

**Required `ggml_backend_buffer_i`:**

- `get_base` → return a fixed sentinel (e.g. `(void*)0x1000`).
  Same trick as WebGPU.
- `set_tensor` → EM_ASM `Module.jsepWriteTensor(handle, offset, hostPtr,
  size)`.
- `get_tensor` → EM_ASM `Module.jsepReadTensor(handle, offset, hostPtr,
  size)` — async readback path; under ASYNCIFY this becomes the same
  shape as the existing `webllm-browser-patches` async readback
  (`ff362d4ae`, `846e0685e`, `702d40ee9`).
- `memset_tensor`, `clear` → EM_ASM `Module.jsepClear(handle, value,
  offset, size)`.
- `free_buffer` → EM_ASM `Module.jsepFree(handle)`.
- All other entries → `NULL`.

**Required `ggml_backend_device_i`:** `get_name`, `get_description`,
`get_memory`, `get_type` (return `GGML_BACKEND_DEVICE_TYPE_GPU`),
`get_props`, `init_backend`, `get_buffer_type`,
**`supports_op`** (return `true` only for `GGML_OP_MUL_MAT` with
`(src0->type ∈ {F16, Q4_K, ...}, src1->type == F32, op->type == F32)`;
false otherwise — scheduler peels other ops to CPU, §4),
**`supports_buft`** (true iff `buft == jsep_buft`).

**Required `ggml_backend_reg_i`:** `get_name`, `get_device_count` (=1),
`get_device`. `api_version = 2`. Wire it via
`ggml_backend_register(reg)` from `ggml_backend_load_all` analog.

**Ops the stub handles:** `GGML_OP_MUL_MAT` only.
**Ops with CPU fallback** (via scheduler): everything else
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_SOFT_MAX`,
`GGML_OP_FLASH_ATTN_EXT`, etc.). Expect significant tensor-roundtrip
traffic per token in the prototype — that's expected and is the
regression we measure against in Phase 3.

**llama.cpp patch budget (band B):** small. ~1 file added
(`ggml/src/ggml-jsep/ggml-jsep.cpp`, ~600 LoC), CMake + emcmake hook,
1 line to the backend-loader to register it. No changes to
`ggml-backend.cpp`, `ggml-impl.h`, or `ggml.c`. The existing
`webllm-browser-patches` async-readback bundle (`846e0685e`,
`702d40ee9`, `55fba3670`) is **directly reusable** — that's the
template for `get_tensor`'s host-readback path.

## 7. Open questions / risks

1. **`tensor->data` sentinel collisions across backends.** If both
   ggml-jsep and ggml-webgpu are registered (during a transition
   window), their `webgpu_ptr_base = 0x1000` sentinels collide in any
   process where a tensor moves between them. They never do in
   practice (different buffer-types), but make the `jsep_ptr_base`
   distinct (e.g. `0x2000`) for safety.
2. **`get_tensor` *sync* signature, async reality.** The buffer
   vtable's `get_tensor` is a void-returning sync function
   (`ggml-backend-impl.h:51`). On WebGPU it has to block on
   `mapAsync`. The `webllm-browser-patches` ASYNCIFY hooks fix this for
   ggml-webgpu; ggml-jsep inherits the same constraint. Risk: the
   per-tensor crossing happens *inside* a sync function, so JSPI
   yielding has to be propagated through the EM_ASM. Same path the
   existing patch stack already validated, so reusable.
3. **`init_tensor` may be needed.** WebGPU sets it to `NULL`
   (`ggml-webgpu.cpp:3897`) but tensor-extras (e.g., per-tensor
   pre-baked WGSL pipeline cache key) might need it. Defer; the
   prototype does the lookup at run time.
4. **Scheduler's `cpy_tensor_async` fast path bypassed.** Without
   `cpy_tensor_async`, every cross-backend split boundary serializes
   through `set_tensor`/`get_tensor` host-staging
   (`ggml-backend.cpp:1664-1672`). For a matmul-only stub on a 22-layer
   model, this is *every layer*, *every token*, twice. Expect the
   prototype to be slow on absolute tok/s; the gate is whether the
   *per-node JS↔WASM crossing rate* drops vs. ggml-webgpu, not whether
   the prototype beats ggml-webgpu end-to-end.
5. **`supports_op`'s argument is the full tensor.** It can inspect
   shapes — important for matmul, where some shape combinations have
   no efficient WGSL path. Mirror webgpu's
   (`:4409-4600`) acceptance logic for the supported subset.
6. **Whole-graph EM_ASM vs per-node.** The §5 finding (ggml hands the
   *whole* cgraph to `graph_compute`) means graph-once is a real
   option from day 1. Prototype both shapes; per-node is cheaper to
   build, graph-once is the structural P2-v2 win. Decide on Phase 2
   measurements, not up-front.
7. **`ggml_visit_parents_graph` patch.** The existing
   `webllm-browser-patches` patch (`17517488a`) makes graph traversal
   iterative for WASM stack safety. ggml-jsep does **not** call
   `ggml_visit_parents_graph` itself (it walks `cgraph->nodes`
   linearly), but ggml-side graph construction does. No new patch
   needed in this dimension.
