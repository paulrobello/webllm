/** Configuration for initializing a WebLLM engine instance. */
export interface WebLLMConfig {
	cacheDir?: string;
	memoryBudget: number;
	frameBudgetMs?: number;
	/** Maximum concurrent conversations per loaded model. Default: 4. */
	maxConversations?: number;
	/**
	 * Run engine in a DedicatedWorker. Default false.
	 *
	 * When true, WebGPU + ggml-wasm execute off-main-thread; the returned
	 * WebLLM is a proxy. All public methods retain their signatures.
	 */
	worker?: boolean;
}

/** Options passed when loading a model into the engine. */
export interface ModelLoadOptions {
	priority: number;
	contextLength?: number;
	gpuLayers?: number;
	lightweight?: boolean;
	/**
	 * When true, the model is constructed in Flash-Attention KV layout
	 * (F16 K+V, FA-ready V transpose). REQUIRED for `engine.createConversation`,
	 * `engine.exportConversation` / `importConversation`, and the prefix-cache
	 * persistence path. Default `false` (manual-attention mode — F16 K, F32 V,
	 * compatible with the legacy `chatCompletion(modelId, ...)` path only).
	 */
	flashAttn?: boolean;
	/**
	 * When true, the model's hidden state (post-`output_norm`, last-token-pooled
	 * and L2-normalized) is exposed for embedding via `engine.embed()`. Quality
	 * trades 5-15% vs dedicated retrieval-tuned embedders; acceptable for in-domain
	 * retrieval tasks. Set only on models that have passed the bucket D parity
	 * gate (cos >= 0.999 vs PyTorch reference).
	 */
	embeddingCapable?: boolean;
	/**
	 * Pooling strategy for `engine.embed()` when `embeddingCapable: true`.
	 * Default `"last-token"`. Use `"mean"` for chat models with high
	 * last-token anisotropy (e.g., Phi-3.5-mini); the mean of all token
	 * hidden states preserves more semantic spread.
	 */
	embeddingPooling?: "last-token" | "mean";
}

/** Read-only handle returned after successfully loading a model. */
export interface ModelHandle {
	readonly id: string;
	readonly name: string;
	readonly priority: number;
	readonly lightweight: boolean;
}

/** Supported GGML tensor data types for quantized and full-precision weights. */
export type GgmlType =
	| "f32"
	| "f16"
	| "q4_0"
	| "q4_1"
	| "q5_0"
	| "q5_1"
	| "q8_0"
	| "q2_k"
	| "q3_k"
	| "q4_k"
	| "q5_k"
	| "q6_k"
	| "iq2_xxs"
	| "iq2_xs"
	| "iq2_s"
	| "iq3_xxs"
	| "iq3_s"
	| "iq1_s"
	| "iq1_m"
	| "iq4_nl"
	| "iq4_xs";

/** Supported model architectures for inference dispatch. */
export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "phi3"
	| "gemma"
	| "qwen"
	| "qwen2"
	| "qwen3"
	| "qwen3-embedding"
	| "mixtral"
	| "deepseek"
	| "bert"
	| "nomic-bert"
	| "jina-bert-v2";

/** All architectures handled by EncoderInference (bidirectional, no KV cache). */
export const ENCODER_ARCHITECTURES = [
	"bert",
	"nomic-bert",
	"jina-bert-v2",
] as const;

export function isEncoderArchitecture(a: ModelArchitecture): boolean {
	return (ENCODER_ARCHITECTURES as readonly string[]).includes(a);
}

/** All architectures handled by CausalLMEmbedder (causal LM with last-token pooling, no KV cache). */
export const CAUSAL_EMBEDDER_ARCHITECTURES = ["qwen3-embedding"] as const;

export function isCausalEmbedderArchitecture(a: ModelArchitecture): boolean {
	return (CAUSAL_EMBEDDER_ARCHITECTURES as readonly string[]).includes(a);
}

/** Metadata for a single tensor within a GGUF model file. */
export interface TensorInfo {
	name: string;
	nDimensions: number;
	dimensions: number[];
	type: GgmlType;
	offset: number;
	size: number;
}

/** High-level model metadata parsed from the GGUF header. */
export interface ModelMetadata {
	architecture: ModelArchitecture;
	contextLength: number;
	embeddingLength: number;
	headCount: number;
	layerCount: number;
	vocabularySize: number;
	ropeFreqBase: number;
	ropeScale: number;
}

/**
 * Serializable subset of the parsed GGUF model returned alongside the
 * model handle by `loadModelFromBuffer` / `loadModelFromUrl`. Contains
 * the same three fields as `ParsedModel` (the loader-internal type),
 * shaped to be plain data so it can cross the worker boundary via
 * `postMessage`'s structured clone.
 *
 * Lets the worker-mode caller obtain the parsed metadata from the engine
 * itself (which has to parse GGUF anyway to build inference) rather than
 * doing a separate main-side parse against a bounded header-prefix Range
 * fetch — the architectural fix that retired the smoke page's HEAD-Range
 * + main-side `GgufParser.parse()` two-step.
 */
export interface LoadedModelMetadata {
	hyperparams: ModelHyperparams;
	tokenizerConfig: import("../inference/tokenizer.js").TokenizerConfig;
	kvCacheConfig: import("../models/kv-cache.js").KVCacheConfig;
}

/** Generic event handler callback. */
export type EventHandler<T = void> = (event: T) => void;

/** Emitted when GPU memory usage crosses the pressure threshold. */
export interface MemoryPressureEvent {
	used: number;
	total: number;
	modelId: string;
}

/** Full hyperparameters for a loaded model used during inference. */
export interface ModelHyperparams {
	architecture: ModelArchitecture;
	contextLength: number;
	embeddingLength: number;
	headCount: number;
	headCountKv: number;
	layerCount: number;
	vocabularySize: number;
	embeddingHeadLength: number;
	feedForwardLength: number;
	ropeFreqBase: number;
	ropeScale: number;
	normEpsilon: number;
	expertCount: number;
	expertUsedCount: number;
	/**
	 * Canonical quantization name derived from GGUF `general.file_type`
	 * (e.g., `"F16"`, `"Q4_K_M"`, `"IQ3_M"`). Defaults to `"unknown"` when
	 * the metadata key is absent or carries an unmapped enum value. Used by
	 * the persistence fingerprint to refuse loading a KV blob captured
	 * against a different quant of the same architecture.
	 */
	quantType: string;
	/** Pooling strategy for `embed()`. CLS/MEAN for BERT-family encoders; LAST-TOKEN for causal-LM-derived embedders (e.g., Qwen3-Embedding). */
	poolingType?: "cls" | "mean" | "last-token";
	/**
	 * When false, attention is bidirectional (BERT-style encoders).
	 * Only encoder architectures populate this field; `undefined` means causal
	 * attention (the decoder default).
	 */
	causalAttention?: boolean;
	/**
	 * Only jina-bert-v2 populates this; the value is passed straight to
	 * `opSoftMaxExt`'s `max_bias` arg, which causes ggml's softmax to apply
	 * the standard ALiBi linear bias (slopes derived as `2^(-8/n_head * h)`).
	 * Defaults to 8.0 when GGUF metadata omits the key (gaianet mirror).
	 */
	alibiMaxBias?: number;
}

/** GPU buffer mappings and tensor metadata for a loaded model's weights. */
export interface ModelWeights {
	/** Tensor name -> GPU buffer ID. */
	tensorBuffers: Map<string, number>;
	/** Tensor name -> tensor metadata. */
	tensorInfos: Map<string, TensorInfo>;
}

/** Internal tracked state for a loaded model within the ModelManager. */
export interface ModelEntry {
	readonly id: string;
	readonly name: string;
	readonly priority: number;
	readonly lightweight: boolean;
	hyperparams: ModelHyperparams;
	kvCache: import("../models/kv-cache.js").KVCache;
	tokenizer: import("../inference/tokenizer.js").Tokenizer;
	memoryAllocations: number[];
	loaded: boolean;
	activeSessions: number;
	embeddingCapable?: boolean;
	embeddingPooling?: "last-token" | "mean";
	/** SHA-256 of canonical-key-sorted tokenizerConfig JSON; computed once at load. */
	tokenizerHash?: string;
	/** Cached fingerprint for persistence validation; computed once at load. */
	fingerprint?: import("./persistence.js").ModelFingerprint;
}
