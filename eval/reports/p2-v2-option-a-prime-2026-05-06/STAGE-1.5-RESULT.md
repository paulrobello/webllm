# Phase 3 / Option A-prime — Stage 1.5 Result

**Date:** 2026-05-06
**Outcome:** **MIXED** — RMS_NORM signature fixed; new bug surfaced
that's larger than Stage 1.5 scope: the **JSEP descriptor ABI conflates
"offset within buffer" with "buffer handle"**. The Phase 2 synthetic
offload probe never tripped this because each test tensor got its own
isolated buffer (offset 0). Under Option A-prime with a real model
loaded, the bug manifests as `GpuDataManager.get: invalid handle 0`
(or some non-zero offset value).

## What Stage 1.5 fixed

1. **`dispatchRmsNorm` — unary signature** (`src/inference/jsep/ops/rms-norm.ts`).
   Phase 2 wrote the kernel against a fused RMS_NORM-then-multiply
   signature with a weight binding. ggml's `ggml_rms_norm`
   (`ggml/src/ggml.c:3117`) is **unary** — the per-channel weight
   multiply is a separate `GGML_OP_MUL` node. Fix:
   - `nSrc !== 1` validation (was `!== 2`).
   - Drop the weight binding from bind-group layout + WGSL.
   - WGSL `out[idx] = x[idx] * inv_rms` (was `* w[col] * inv_rms`).
   - Test fixture (`tests/jsep-rms-norm-golden.test.ts`) updated to
     match unary signature.

2. **`supports_buft` narrowed to JSEP-only** (`ggml-jsep.cpp`).
   Phase 2 Task 10 widened `supports_buft` to advertise JSEP as
   compatible with **host buft** + **WebGPU buft** + jsep_buft. Under
   Option A-prime that's wrong: when JSEP is the lone GPU device, the
   scheduler should insert CPY-to-jsep_buf at split boundaries for
   any host-resident src. Returning TRUE for host buft caused the
   scheduler to dispatch RMS_NORM nodes with CPU-resident sources to
   JSEP without inserting the migration CPY — leaving graph_compute
   to dereference CPU heap pointers as JSEP handles (the original
   "invalid handle 44100352" symptom).
   - The Phase 2 offload_op flow (synthetic probe Outcome E) is now
     dormant. Deliberate trade — Option A-prime supersedes that path.

## Stage 1.5's hidden discovery

After both fixes, the spike's symptom changed:

```
FAIL — GpuDataManager.get: invalid handle 0
  at GpuDataManager.get
  at dispatchRmsNorm
```

Handle 0 means `t->data == GGML_JSEP_PTR_BASE` (the sentinel base) —
i.e., `jsep_tensor_handle(t) == 0`. Looking at the JSEP buffer
implementation (`ggml-jsep.cpp:208-237`):

- `ggml_backend_jsep_buffer_set_tensor` calls
  `ggml_jsep_write(ctx->handle, total_offset, ...)` where `ctx->handle`
  is the JSEP **buffer's** handle and `total_offset` is the absolute
  byte offset within that buffer.
- BUT `ggml_jsep_write_tensor_block` packs only
  `out[0] = jsep_tensor_handle(t)` — which is `t->data - PTR_BASE` —
  the **offset within the buffer**, not the buffer's handle.

The descriptor format docs claim `out[0]` is "handle (i32; from
`jsep_tensor_handle()`)" and the Phase 2 JS dispatchers consume it as
a `GpuDataManager` handle. This works ONLY if every tensor is at
offset 0 within its buffer — which holds for the synthetic offload
probe (each test tensor gets its own `ggml_jsep_alloc`) but **fails
in production**.

`alloc` counter at model load was **6** for tinyllama:
- 1 jsep_buf for model weights (455.06 MiB, hundreds of tensors)
- 1 jsep_buf for KV cache (11.00 MiB, ~22 K + 22 V tensors as views)
- 1 jsep_buf for compute scratch (32.00 MiB, hundreds of intermediates)
- 1 CPU host buf (181.11 MiB, token_embd spill + various)
- ~2 more for output / staging

Each big buffer hosts 100+ tensors at distinct offsets. The descriptor
must carry **(buf_handle, offset)** pairs per tensor, not a single
"handle" field that lies about its identity.

## Decision: descriptor ABI extension required

Stage 1.5 consciously stops short of fixing this. The fix is bigger
than a Stage and touches:

- **C++ side** (`ggml-jsep.cpp`): bump `GGML_JSEP_TENSOR_BLOCK_I32`
  from 18 to 19; emit `(buf_handle, offset)` pair at slot [0..1];
  shift type/ne/nb to [2..18]. Update header docs.
- **JS side** (`ops/matmul.ts`): bump `TENSOR_BLOCK_I32`; update
  `readDescriptor`; bind buffers at offsets via WebGPU's
  `{buffer, offset, size}` resource form.
- **All dispatchers** (matmul, rms-norm, set-rows): switch from
  `dataManager.get(handle).buffer` → `dataManager.get(buf_handle)`
  with explicit offset on every binding entry.
- **Tests**: `jsep-rms-norm-golden.test.ts`,
  `jsep-set-rows.test.ts` (if added), `jsep-matmul.test.ts` need
  the new descriptor shape.
- **Synthetic offload probe** (`webgpu-bridge.cpp:webllm_synthetic_offload_probe`)
  needs to either populate the new fields or be retired (deferred —
  Outcome E was already banked).

Patch budget: descriptor ABI changes amend the existing JSEP skeleton
patch (`48acb658d`), so patch stack count is unchanged at +4.

## Patch stack at end of Stage 1.5

`webllm-browser-patches`:
1. `48acb658d` Phase 2 — JSEP skeleton + MUL_MAT/RMS_NORM
2. `7919d1839` Task 9 — metadata-op allowlist
3. `49413d8e9` Task 10 — supports_buft + offload_op (will be partially
   reverted in Stage 2: supports_buft narrowed to JSEP-only)
4. `d8b80dee2` Stage 1 — SET_ROWS supports_op
5. *(pending Stage 1.5 commit)* — supports_buft narrowing
   (Option A-prime; supersedes Task 10's host-buft acceptance for
   the production path)

webllm:
- `b640d17` Stage 0 — device-hint inversion
- `e60a39e` Stage 1 — SET_ROWS kernel
- *(pending Stage 1.5 commit)* — RMS_NORM unary fix +
  `supports_buft` narrowing in companion llama.cpp commit

## Next session pickup (Stage 2 = descriptor ABI fix)

1. Extend descriptor block to 19 i32 with `(buf_handle, offset)` at
   slots [0..1]. Update all sites (C++ packer + JS readDescriptor +
   3 dispatchers + 2-3 test files).
2. Re-run spike at `?v=A-prime-stage2`. Expected outcome: spike either
   completes (greedy-decodes "Paris" — matches Outcome D's 81 tok/s
   hypothesis, validated in real chat path) or surfaces the next
   missing op kernel (likely MUL or ADD per the TODO Phase 3
   prediction).
3. Document outcome in `STAGE-2-RESULT.md`.

The ABI-fix is a clean 1-stage piece of work: zero new ops, just
better descriptor packing. After it lands, ops added in subsequent
stages benefit from the corrected ABI for free.

## Files touched (Stage 1.5)

webllm:
- `src/inference/jsep/ops/rms-norm.ts` — unary signature.
- `tests/jsep-rms-norm-golden.test.ts` — drop weight from fixture.
- `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-1.5-RESULT.md` (new).

llama.cpp `webllm-browser-patches`:
- `ggml/src/ggml-jsep/ggml-jsep.cpp` — `supports_buft` narrowed to
  jsep_buft only.
