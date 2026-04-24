import type { ModelArchitecture } from "../src/core/types.js";

/** Quantization formats supported by the benchmark suite. */
export type QuantFormat = "q4f16_1" | "q4f32_1" | "q0f16" | "q0f32";

/** Model capability flags. */
export interface ModelCapabilities {
	toolCalling: boolean;
	structuredOutput: boolean;
	vision: boolean;
	embedding: boolean;
}

/** A benchmark model definition with all metadata needed for evaluation. */
export interface BenchmarkModel {
	/** Unique identifier used in reports and CLI args. */
	id: string;
	/** Human-readable model name. */
	name: string;
	/** Model family for grouping in reports. */
	family: string;
	/** llama.cpp architecture identifier. */
	architecture: ModelArchitecture;
	/** Parameter count in billions. */
	paramsB: number;
	/** Approximate VRAM required in MB at default quantization. */
	vramMB: number;
	/** Default quantization format for benchmarks. */
	defaultQuant: QuantFormat;
	/** Available quantization variants for sweep benchmarks. */
	availableQuants: QuantFormat[];
	/** Model capabilities. */
	capabilities: ModelCapabilities;
	/** License identifier. */
	license: string;
	/** Minimum context window for benchmarks. */
	contextLength: number;
	/** Recommended benchmark tier. */
	tier: "ultrafast" | "fast" | "balanced" | "quality";
	/** Whether this model requires shader-f16 GPU feature. */
	requiresShaderF16: boolean;
	/** HuggingFace download URL. */
	downloadUrl: string;
	/** GGUF HuggingFace URL (for native llama.cpp). */
	ggufUrl: string;
}

/**
 * All benchmark models organized by tier. Each model has verified GGUF
 * availability and WebGPU-compatible parameter counts.
 */
export const BENCHMARK_MODELS: BenchmarkModel[] = [
	// --- Ultrafast tier (sub-1GB VRAM, 100+ tok/s) ---

	{
		id: "smollm2-360m-q4f16",
		name: "SmolLM2 360M Instruct",
		family: "SmolLM2",
		architecture: "llama",
		paramsB: 0.36,
		vramMB: 376,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1", "q0f16", "q0f32"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "ultrafast",
		requiresShaderF16: true,
		downloadUrl: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/huggingface-quants/SmolLM2-360M-Instruct-GGUF",
	},

	// --- Fast tier (1-2GB VRAM, 60+ tok/s) ---

	{
		id: "qwen3-0.6b-q4f16",
		name: "Qwen3 0.6B",
		family: "Qwen3",
		architecture: "qwen",
		paramsB: 0.6,
		vramMB: 1403,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1", "q0f16"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
	},

	{
		id: "llama-3.2-1b-q4f16",
		name: "Llama 3.2 1B Instruct",
		family: "Llama 3.2",
		architecture: "llama",
		paramsB: 1.23,
		vramMB: 879,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1", "q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Llama-3.2",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
	},

	{
		id: "qwen2.5-1.5b-q4f16",
		name: "Qwen2.5 1.5B Instruct",
		family: "Qwen2.5",
		architecture: "qwen",
		paramsB: 1.54,
		vramMB: 1630,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
	},

	{
		id: "smollm2-1.7b-q4f16",
		name: "SmolLM2 1.7B Instruct",
		family: "SmolLM2",
		architecture: "llama",
		paramsB: 1.71,
		vramMB: 1774,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: true,
		downloadUrl: "https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Mungert/SmolLM2-1.7B-Instruct-GGUF",
	},

	// --- Balanced tier (2-3GB VRAM, 30+ tok/s) ---

	{
		id: "qwen3-1.7b-q4f16",
		name: "Qwen3 1.7B",
		family: "Qwen3",
		architecture: "qwen",
		paramsB: 1.7,
		vramMB: 2037,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF",
	},

	{
		id: "hermes-3-llama-3.2-3b-q4f16",
		name: "Hermes 3 Llama 3.2 3B",
		family: "Hermes 3",
		architecture: "llama",
		paramsB: 3.21,
		vramMB: 2264,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Llama-3.2",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF",
	},

	{
		id: "llama-3.2-3b-q4f16",
		name: "Llama 3.2 3B Instruct",
		family: "Llama 3.2",
		architecture: "llama",
		paramsB: 3.21,
		vramMB: 2264,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Llama-3.2",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
	},

	{
		id: "qwen2.5-3b-q4f16",
		name: "Qwen2.5 3B Instruct",
		family: "Qwen2.5",
		architecture: "qwen",
		paramsB: 3.09,
		vramMB: 2505,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen2.5-3B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
	},

	{
		id: "phi-3.5-mini-q4f16",
		name: "Phi-3.5 Mini Instruct",
		family: "Phi",
		architecture: "phi",
		paramsB: 3.82,
		vramMB: 2520,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "MIT",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF",
	},

	// --- Quality tier (3-4GB VRAM, 20-30 tok/s) ---

	{
		id: "qwen3-4b-q4f16",
		name: "Qwen3 4B",
		family: "Qwen3",
		architecture: "qwen",
		paramsB: 4.0,
		vramMB: 3432,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen3-4B-GGUF",
	},

	// --- Specialized models ---

	{
		id: "tinyllama-1.1b-chat-q4_0",
		name: "TinyLlama 1.1B Chat v1.0",
		family: "TinyLlama",
		architecture: "llama",
		paramsB: 1.1,
		vramMB: 760,
		defaultQuant: "q0f32",
		availableQuants: ["q0f32"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 2048,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
		// Actually Q4_0 but no q4_0 in our enum — this is the Q4_0 GGUF we use
		// as the smoke-test reference model. See smoke-test/models/.
		ggufUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
	},

	// --- Embedding models ---

	{
		id: "snowflake-arctic-embed-s-q0f32-b4",
		name: "Snowflake Arctic Embed S",
		family: "Arctic Embed",
		architecture: "bert",
		paramsB: 0.033,
		vramMB: 239,
		defaultQuant: "q0f32",
		availableQuants: ["q0f32"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 512,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-s-q0f32-MLC",
		ggufUrl: "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-s-gguf",
	},

	{
		id: "snowflake-arctic-embed-m-q0f32-b4",
		name: "Snowflake Arctic Embed M",
		family: "Arctic Embed",
		architecture: "bert",
		paramsB: 0.109,
		vramMB: 539,
		defaultQuant: "q0f32",
		availableQuants: ["q0f32"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 512,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-m-q0f32-MLC",
		ggufUrl: "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-m-gguf",
	},

	// --- Specialized models ---

	{
		id: "qwen2.5-coder-1.5b-q4f16",
		name: "Qwen2.5 Coder 1.5B Instruct",
		family: "Qwen2.5 Coder",
		architecture: "qwen",
		paramsB: 1.54,
		vramMB: 1630,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
	},

	{
		id: "gemma-2-2b-q4f16",
		name: "Gemma 2 2B IT",
		family: "Gemma 2",
		architecture: "gemma",
		paramsB: 2.61,
		vramMB: 1584,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1", "q4f32_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Gemma",
		contextLength: 4096,
		tier: "fast",
		requiresShaderF16: true,
		downloadUrl: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
		ggufUrl: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
	},
];

/** Tier display order and labels. */
export const TIER_ORDER: Record<BenchmarkModel["tier"], { order: number; label: string; speedTarget: string }> = {
	ultrafast: { order: 0, label: "Ultrafast", speedTarget: "100+ tok/s" },
	fast: { order: 1, label: "Fast", speedTarget: "60+ tok/s" },
	balanced: { order: 2, label: "Balanced", speedTarget: "30+ tok/s" },
	quality: { order: 3, label: "Quality", speedTarget: "20-30 tok/s" },
};

/** Get models grouped by tier, sorted by VRAM. */
export function getModelsByTier(): Map<BenchmarkModel["tier"], BenchmarkModel[]> {
	const tiers = new Map<BenchmarkModel["tier"], BenchmarkModel[]>();
	for (const model of BENCHMARK_MODELS) {
		const list = tiers.get(model.tier) ?? [];
		list.push(model);
		list.sort((a, b) => a.vramMB - b.vramMB);
		tiers.set(model.tier, list);
	}
	return tiers;
}

/** Find a benchmark model by its ID. */
export function getModelById(id: string): BenchmarkModel | undefined {
	return BENCHMARK_MODELS.find((m) => m.id === id);
}

/** Get all models that support tool calling. */
export function getToolCallingModels(): BenchmarkModel[] {
	return BENCHMARK_MODELS.filter((m) => m.capabilities.toolCalling);
}

/** Get all embedding models. */
export function getEmbeddingModels(): BenchmarkModel[] {
	return BENCHMARK_MODELS.filter((m) => m.capabilities.embedding);
}

/** Browser VRAM limits by device tier for model selection guidance. */
export const BROWSER_VRAM_LIMITS: Record<string, { vramGB: number; maxParamsB: number; description: string }> = {
	"low-end": { vramGB: 2, maxParamsB: 1, description: "Mobile, old integrated GPUs" },
	midrange: { vramGB: 4, maxParamsB: 3, description: "M1 base, RTX 3060, mid-range laptops" },
	"high-end": { vramGB: 8, maxParamsB: 7, description: "M3/M4 Pro, RTX 4070+" },
	enthusiast: { vramGB: 16, maxParamsB: 14, description: "M4 Max, RTX 4090, desktop workstations" },
};
