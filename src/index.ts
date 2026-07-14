export type {
	CharacterConfig,
	CharacterMessage,
	ChatEngine,
} from "./characters/character.js";
export { Character } from "./characters/character.js";
export { CharacterManager } from "./characters/character-manager.js";
export type {
	ToolCall,
	ToolDefinition,
	ToolParameter,
	ToolResult,
} from "./characters/tool-system.js";
export { ToolSystem } from "./characters/tool-system.js";
export type {
	ChatMessage,
	CompletionChunk,
	CompletionConfig,
	CompletionStats,
	StreamChunk,
	StreamConfig,
	StreamInput,
	StreamStats,
} from "./core/chat-types.js";
export type {
	ConversationHandle,
	ConversationOptions,
	KVSnapshot,
} from "./core/conversation-pool.js";
export { WebLLM } from "./core/engine.js";
export type { WebLLMErrorCode } from "./core/errors.js";
export {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
	WebLLMError,
} from "./core/errors.js";
export type { GameLoopCallback, GameLoopConfig } from "./core/game-loop.js";
export type {
	BufferAllocation,
	MemoryEvent,
	MemoryEventHandler,
} from "./core/memory-pool.js";
export { MemoryPool } from "./core/memory-pool.js";
export { ModelManager } from "./core/model-manager.js";
export {
	MISTRAL_DEFAULTS,
	PHI3_DEFAULTS,
	QWEN_NON_THINKING_DEFAULTS,
	QWEN_THINKING_DEFAULTS,
} from "./core/sampling-profiles.js";
export type {
	Backend,
	EventHandler,
	GgmlType,
	LoadedModelMetadata,
	MemoryPressureEvent,
	ModelArchitecture,
	ModelEntry,
	ModelHandle,
	ModelHyperparams,
	ModelLoadOptions,
	ModelMetadata,
	ModelWeights,
	TensorInfo,
	WebLLMConfig,
} from "./core/types.js";
export type { CustomScorer } from "./evaluation/custom-scorers.js";
export {
	getCustomScorer,
	hasCustomScorer,
	listCustomScorers,
	registerCustomScorer,
} from "./evaluation/custom-scorers.js";
export type {
	RunTaskOptions,
	RunTasksOptions,
} from "./evaluation/runner.js";
export { EngineDeadError, runTask, runTasks } from "./evaluation/runner.js";
export { score } from "./evaluation/scorer.js";
export type {
	SystemProfile,
	SystemProfileInput,
} from "./evaluation/system-profile.js";
export {
	collectBrowserSystemProfile,
	computeSystemId,
} from "./evaluation/system-profile.js";
// ── Evaluation (public library primitives) ────────────────────
export type {
	DimensionScore,
	EvalDimension,
	EvalReport,
	EvalResult,
	EvalTask,
	EvalToolDef,
	ScoringMethod,
	ToolCallRecord,
} from "./evaluation/types.js";
export { CausalLMEmbedder } from "./inference/causal-embedder-inference.js";
// `detectChatTemplate` / `encodeChatPrompt` were removed from this public
// barrel in ARC-003 — they are internal helpers and now live under the
// unstable `./internal` subpath (`src/internal.ts`). The smoke harness
// imports them from `webllm-internal.js`; see Makefile `smoke-test` target.
export { EncoderInference } from "./inference/encoder-inference.js";
export type {
	GenerationConfig,
	GenerationFinishReason,
	GenerationResult,
	GenerationStreamChunk,
	GenerationStreamResult,
} from "./inference/generation.js";
export type {
	LightweightModelConfig,
	LightweightWeights,
} from "./inference/lightweight.js";
export type { SamplerConfig } from "./inference/sampler.js";
export type { TokenData, TokenizerConfig } from "./inference/tokenizer.js";
export {
	StreamingDecoder,
	Tokenizer,
	TokenizerType,
} from "./inference/tokenizer.js";
export type {
	GgufContext,
	GgufHeader,
	GgufKv,
	GgufTensorInfo,
} from "./models/gguf-types.js";
export type { InferenceSessionConfig } from "./models/inference-session.js";
export type { KVCacheConfig, KVCell } from "./models/kv-cache.js";
export { KVCache } from "./models/kv-cache.js";
export type { ParsedModel } from "./models/model-loader.js";
export { ModelLoader } from "./models/model-loader.js";

// ─── Worker bundle re-entry ──────────────────────────────────
//
// When the same bundle module is loaded inside a DedicatedWorker
// (via `new Worker(import.meta.url, { type: "module" })`), boot the
// message-handler host instead of just exposing the public exports.
// Main-thread bundle loads see the typeof check fail and skip this.
// biome-ignore lint/suspicious/noExplicitAny: globalThis narrowing for WebWorker types not in tsconfig lib
const _workerGlobals = globalThis as any;
if (
	typeof _workerGlobals.DedicatedWorkerGlobalScope !== "undefined" &&
	_workerGlobals instanceof _workerGlobals.DedicatedWorkerGlobalScope
) {
	type WebLLMType = import("./core/engine.js").WebLLM;
	const { WebLLM } = await import("./core/engine.js");
	const { startWorkerHost } = await import("./core/webllm-worker-host.js");
	const { serializeError } = await import("./core/webllm-error-codec.js");
	let engine: WebLLMType | null = null;
	startWorkerHost({
		// The host stores `engine` once init lands; method-calls
		// before init result in "unknown engine method" which becomes
		// a GENERIC error main-thread.
		get engine() {
			if (!engine) throw new Error("worker engine not initialized");
			return engine;
		},
		postMessage: (m) => (self as unknown as Worker).postMessage(m),
		receive: (handler) => {
			self.addEventListener("message", (e) => {
				const msg = (e as MessageEvent).data;
				// Intercept init: the host treats it as a no-op so the
				// bundle entry owns engine construction.
				if (msg?.type === "init") {
					void (async () => {
						try {
							engine = await WebLLM.init({
								...msg.config,
								worker: false,
							});
							(self as unknown as Worker).postMessage({
								type: "init-done",
								id: msg.id,
							});
						} catch (err) {
							(self as unknown as Worker).postMessage({
								type: "method-error",
								id: msg.id,
								error: serializeError(err),
							});
						}
					})();
					return;
				}
				// Intercept dispose: the host treats it as a no-op too,
				// but production needs to call engine.dispose() before
				// the proxy's terminate() drops the worker.
				if (msg?.type === "dispose") {
					void (async () => {
						try {
							if (engine) await engine.dispose();
							(self as unknown as Worker).postMessage({
								type: "method-result",
								id: msg.id,
								value: undefined,
							});
						} catch (err) {
							(self as unknown as Worker).postMessage({
								type: "method-error",
								id: msg.id,
								error: serializeError(err),
							});
						}
					})();
					return;
				}
				handler(msg);
			});
		},
	});
}
