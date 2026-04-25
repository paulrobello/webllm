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
export { WebLLM } from "./core/engine.js";
export type { GameLoopCallback, GameLoopConfig } from "./core/game-loop.js";
export { GameLoop } from "./core/game-loop.js";
export type {
	BufferAllocation,
	MemoryEvent,
	MemoryEventHandler,
} from "./core/memory-pool.js";
export { MemoryPool } from "./core/memory-pool.js";
export { ModelManager } from "./core/model-manager.js";
export { PipelineCache } from "./core/pipeline-cache.js";
export { Scheduler } from "./core/scheduler.js";
export type {
	EventHandler,
	GgmlType,
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
export { runTask, runTasks } from "./evaluation/runner.js";
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
export type { ChatTemplateType } from "./inference/chat-template.js";
export {
	detectChatTemplate,
	encodeChatPrompt,
	formatChatDelta,
	formatChatPrompt,
} from "./inference/chat-template.js";
export { EncoderInference } from "./inference/encoder-inference.js";
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
export { ModelInference } from "./inference/model-inference.js";
export type { SamplerConfig } from "./inference/sampler.js";
export { Sampler } from "./inference/sampler.js";
export { StreamRouter } from "./inference/stream-router.js";
export type { TokenData, TokenizerConfig } from "./inference/tokenizer.js";
export {
	StreamingDecoder,
	TokenAttribute,
	Tokenizer,
	TokenizerType,
} from "./inference/tokenizer.js";
export type { ShaderName } from "./inference/wgsl-shaders.js";
export {
	ALL_SHADERS,
	SHADER_EMBEDDING_LOOKUP,
	SHADER_GELU,
	SHADER_LAYER_NORM,
	SHADER_MATMUL_F32,
	SHADER_RMS_NORM,
	SHADER_SILU,
	SHADER_SOFTMAX,
} from "./inference/wgsl-shaders.js";
export { GgufParser } from "./models/gguf-parser.js";
export type {
	GgufContext,
	GgufHeader,
	GgufKv,
	GgufTensorInfo,
} from "./models/gguf-types.js";
export type { InferenceSessionConfig } from "./models/inference-session.js";
export { InferenceSession } from "./models/inference-session.js";
export type { KVCacheConfig, KVCell } from "./models/kv-cache.js";
export { KVCache } from "./models/kv-cache.js";
export type { ParsedModel } from "./models/model-loader.js";
export { ModelLoader } from "./models/model-loader.js";
