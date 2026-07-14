# JSEP prototype backend — error-signaling convention

> **@experimental** — Active Phase-3 R&D. The JSEP backend is retained per
> audit decision ARC-007 (2026-07-14), which marks it as an experimental
> boundary rather than retiring it. No semver guarantee covers this
> directory; declaration emit excludes it from the published type surface
> (`scripts/build-package.ts`). See audit issue ARC-007 for the decision
> record and QA-003 / QA-013 for the open quality items scoped here.

This directory implements the pure-TS WebGPU backend ("JSEP") that mirrors
the patched `ggml-webgpu` C call convention. The three op dispatchers —
`dispatchMatmul`, `dispatchRmsNorm`, `dispatchSetRows` — signal failure
differently from the rest of the library, and that difference is
load-bearing. This document records why so the seam is not a silent
surprise (audit issue QA-013).

## The convention

Each dispatcher returns a numeric status code and emits `console.error`
on failure. **It does not throw `WebLLMError`** — or any member of the
library's typed error hierarchy — on the dispatch-failure path.

Status codes (defined in [`index.ts`](./index.ts)):

| Constant                 | Value | Meaning                                          |
|--------------------------|-------|--------------------------------------------------|
| `STATUS_OK`              | `0`   | Dispatch succeeded.                              |
| `STATUS_NOT_IMPLEMENTED` | `1`   | Op kind not handled here; C++ falls back to CPU. |
| `STATUS_FAILED`          | `-1`  | Validation failure (shape, type, missing layout).|

Evidence — every dispatcher follows the same shape:

- [`ops/rms-norm.ts`](./ops/rms-norm.ts) states the contract in its
  docstring (lines 102–108): *"Returns 0 on success, negative on
  validation failure."* Failure sites at :115–118, :123–129, :143–149,
  :161–164 pair `console.error(...)` with `return -1`.
- [`ops/matmul.ts`](./ops/matmul.ts) — `dispatchMatmul` returns `-1`
  alongside `console.error` at :633–634 (missing bind-group layout) and
  :710–715 (aliased dst is non-contiguous); success returns `0`.
- [`ops/set-rows.ts`](./ops/set-rows.ts) — `dispatchSetRows` follows
  the same pattern at :281–330, :366–376, :390–392, :469–482.

The few `throw new Error(...)` sites under `jsep/` (e.g.
[`gpu-data-manager.ts:67,123`](./gpu-data-manager.ts),
[`ops/matmul.ts:234,503`](./ops/matmul.ts)) guard **programmer-error
preconditions** (invalid handle, unsupported tensor type) and throw bare
`Error` — not `WebLLMError`. They are not part of the dispatch-failure
path and do not participate in the library's typed error hierarchy.

## Why it works this way

The dispatchers have exactly one production caller: `module.jsepRunOp` in
[`index.ts`](./index.ts) (lines 270–300). `jsepRunOp` is an **EM_ASM
callback** — a JS function invoked from C++ through Emscripten's bridge,
whose `number` return is consumed as a C `int` status code by
`ggml/src/ggml-jsep/ggml-jsep.cpp::ggml_backend_jsep_graph_compute` (the
C++ consumer is named in the module header at the top of each op file).
The body is a thin pass-through:

```ts
if (desc.op === GGML_OP_MUL_MAT)  return dispatchMatmul(ctx, desc);
if (desc.op === GGML_OP_RMS_NORM) return dispatchRmsNorm(ctx, desc, opParamsPtr, buf);
if (desc.op === GGML_OP_SET_ROWS) return dispatchSetRows(ctx, desc);
return STATUS_NOT_IMPLEMENTED;
```

A thrown `WebLLMError` cannot cross an EM_ASM boundary as a typed object
— it would surface in C++ as an uncaught Emscripten exception with no
payload. The sentinel return mirrors ggml's C call convention so the
status is legible at the C++ consumer without any marshalling.

## Conversion boundary — none exists today

This is the honest part: **no TS-layer code converts these sentinels into
`WebLLMError`.** Concretely:

- `jsepRunOp` ([`index.ts:290–296`](./index.ts)) returns each dispatch
  result **unchanged** across the EM_ASM boundary; the C++ graph-compute
  routine is the consumer.
- The only TS-side callers of the dispatchers are the golden tests, and
  they check the sentinel themselves:
  - `tests/jsep-matmul-golden.test.ts:358–587` —
    `expect(status).toBe(0)` after each `const status = dispatchMatmul(ctx, desc)`.
  - `tests/jsep-rms-norm-golden.test.ts:160–161` — the same shape for
    `dispatchRmsNorm`.
- `src/index-jsep.ts:39` re-exports `dispatchMatmul` **unchanged** for
  the Stage 3 spike harness; the adjacent comment labels it *"Probe-only
  … Not part of the public API."*

**Implication for callers:** any TS code that invokes these dispatchers
directly must inspect the return value — `status === 0` for success,
`status === STATUS_NOT_IMPLEMENTED` for "C++ should fall back to CPU",
`status < 0` for hard failure. Relying on a `try`/`catch` will miss every
failure mode this layer can produce.

If a future change routes JSEP failures through the library's typed error
hierarchy (e.g. a wrapper at the engine boundary that converts
`STATUS_FAILED` to a `WebLLMError`), update this section to name the
conversion site explicitly. Until then, the sentinel is the contract.

## Related

- **QA-003** (audit) — the per-dispatch uniform-buffer leak in all three
  ops. Its fix intentionally leaves this `return -1` convention in place;
  changing the convention would touch every caller and is out of scope
  for that issue.
- **ARC-007** (audit) — the "keep JSEP" decision that gates this
  document. If JSEP is ever retired, this convention is retired with it.
- **CLAUDE.md "Regression lessons"** — the broader C-ABI seam
  (`graphCompute()` async capability, JSPI export rules) this boundary
  participates in.
