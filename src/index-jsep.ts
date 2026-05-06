/**
 * Bundle entry for the JSEP-style backend variant (P2-v2 prototype).
 *
 * Mirrors `src/index.ts` (the canonical bundle entry) but additionally
 * eagerly imports `src/inference/jsep/index.ts` so `bun build` does
 * not tree-shake the JSEP runtime out of the resulting
 * `webllm-bundle-jsep.js`. Without that side-effect import, the runtime
 * is only reachable through the engine's lazy `await import(...)` path
 * inside `GgmlWasm.installJsepCallbacks`, and a static bundler that
 * follows static imports first may still emit the runtime as a
 * separate chunk — bundlers often inline dynamic imports too, but
 * eagerly importing the module here makes the inclusion explicit and
 * audit-able.
 *
 * Spec: `docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md`
 * Plan: `docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md` Task 6.
 */

// Side-effect import — landed in the bundle even when nothing references
// the named exports.
import "./inference/jsep/index.js";

// Re-export the canonical public surface unchanged so consumers that
// switch to `webllm-bundle-jsep.js` see the same API as the default
// bundle.
export * from "./index.js";

// Probe-only re-exports so the Stage 3 spike can drive the matmul
// dispatcher with hand-crafted descriptors. Not part of the public API.
export { GGML_OP_MUL_MAT as JSEP_GGML_OP_MUL_MAT } from "./inference/jsep/index.js";
export {
	dispatchMatmul,
	GGML_TYPE_F32 as JSEP_GGML_TYPE_F32,
	GGML_TYPE_Q4_K as JSEP_GGML_TYPE_Q4_K,
	type JsepOpDescriptor as JSEP_JsepOpDescriptor,
} from "./inference/jsep/ops/matmul.js";
