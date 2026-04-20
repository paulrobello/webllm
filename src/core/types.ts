export interface WebLLMConfig {
	device: GPUDevice;
	cacheDir?: string;
	memoryBudget: number;
	frameBudgetMs?: number;
}

export interface ModelLoadOptions {
	priority: number;
	contextLength?: number;
	gpuLayers?: number;
	lightweight?: boolean;
}

export interface ModelHandle {
	readonly id: string;
	readonly name: string;
	readonly priority: number;
	readonly lightweight: boolean;
}

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

export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "gemma"
	| "qwen"
	| "mixtral"
	| "deepseek";

export interface TensorInfo {
	name: string;
	nDimensions: number;
	dimensions: number[];
	type: GgmlType;
	offset: number;
	size: number;
}

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

export type EventHandler<T = void> = (event: T) => void;

export interface MemoryPressureEvent {
	used: number;
	total: number;
	modelId: string;
}

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
}

export interface ModelWeights {
	/** Tensor name -> GPU buffer ID. */
	tensorBuffers: Map<string, number>;
	/** Tensor name -> tensor metadata. */
	tensorInfos: Map<string, TensorInfo>;
}
