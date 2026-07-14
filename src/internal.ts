/**
 * Unstable internal surface for the smoke harness and power users — no
 * semver guarantees.
 *
 * This barrel re-exports the deep inference internals that the canonical
 * `src/index.ts` public barrel deliberately omits (ARC-003). The split
 * exists so that:
 *
 * - **Public consumers** (`@paulrobello/webllm`) get a stable, narrow
 *   surface (`WebLLM` + error classes + sampling profiles + types) and
 *   internal refactors (including the planned `ModelInference` graph-
 *   builder consolidation ARC-001) do not register as breaking changes.
 * - **In-repo harnesses** (the browser smoke pages under `smoke-test/`,
 *   `eval/` shims) and power users that opt into the unstable surface
 *   can reach the deep internals via `@paulrobello/webllm/internal`.
 *
 * Anything re-exported here may move, rename, or disappear in any release.
 * The `./internal` subpath in `package.json#exports` is the contract for
 * opting in; consumers who do so accept the breakage risk.
 *
 * See `AUDIT.md` ARC-003 for the rationale and the stabilization
 * (graduation) criteria for each symbol.
 */

export type { GameLoopCallback, GameLoopConfig } from "./core/game-loop.js";
export { GameLoop } from "./core/game-loop.js";
export { PipelineCache } from "./core/pipeline-cache.js";
export type { ScheduledTask } from "./core/scheduler.js";
// ── Inert / R&D orchestration modules (ARC-002 demotions) ─────
// These are wired into the engine for bookkeeping or staged for future
// use, but are NOT load-bearing for the canonical chat/embed path. Kept
// here for power users who want to drive them directly; the public barrel
// omits them so the README's API table can stay honest about what's
// canonical vs. what's experimental.
export { Scheduler } from "./core/scheduler.js";
// ── Chat-template helpers used by the smoke harness ───────────
// The file-level comment in `chat-template.ts` admits these are internal
// helpers — the smoke bundle inspects template family and builds prompts
// without going through `engine.chatCompletion`. Exposed via `./internal`
// so the smoke bundle can keep doing so without polluting the npm surface.
export {
	detectChatTemplate,
	encodeChatPrompt,
} from "./inference/chat-template.js";
export type {
	GenerationConfig,
	GenerationFinishReason,
	GenerationResult,
	GenerationStreamChunk,
	GenerationStreamResult,
} from "./inference/generation.js";
export { Generator } from "./inference/generation.js";
export { GgmlWasm } from "./inference/ggml-wasm.js";
export type {
	LightweightModelConfig,
	LightweightWeights,
} from "./inference/lightweight.js";
export { LightweightModel } from "./inference/lightweight.js";
// ── Deep inference internals (engine plumbing) ─────────────────
export { ModelInference } from "./inference/model-inference.js";
export type { SamplerConfig } from "./inference/sampler.js";
export { Sampler } from "./inference/sampler.js";
export { StreamRouter } from "./inference/stream-router.js";
// ── GGUF / inference-session internals ────────────────────────
export { GgufParser } from "./models/gguf-parser.js";
export type {
	GgufContext,
	GgufHeader,
	GgufKv,
	GgufTensorInfo,
} from "./models/gguf-types.js";
export type { InferenceSessionConfig } from "./models/inference-session.js";
export { InferenceSession } from "./models/inference-session.js";
