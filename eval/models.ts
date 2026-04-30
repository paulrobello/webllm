import type { ModelArchitecture } from "../src/core/types.js";

/** Quantization formats supported by the benchmark suite. */
export type QuantFormat = "q4f16_1" | "q4f32_1" | "q0f16" | "q0f32" | "hyb" | "iq3m" | "q4km";

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
	/**
	 * Optional case-insensitive substring used by the bench downloader
	 * (`ensureModelDownloaded` in `eval/browser-smoke.ts`) to pick a
	 * specific GGUF file from the repo when the MLC `defaultQuant`
	 * naming doesn't match the GGUF naming. Example: BERT embedding
	 * GGUFs are commonly published as `<name>-F16.GGUF` while MLC uses
	 * `q0f32`; setting `ggufFilePattern: "f16"` pins the verified file.
	 */
	ggufFilePattern?: string;
	/**
	 * When true, the bench downloader treats any cached file at
	 * `smoke-test/models/<id>.gguf` as authoritative and skips the
	 * HuggingFace tree-fetch + size-verify entirely. Used for models
	 * whose canonical GGUF is built locally rather than published
	 * (e.g. the bucket C hybrid quant: `token_embd` Q4_K, rest f16,
	 * not in any upstream mirror — see CLAUDE.md "Per-binding 128
	 * MiB cap doctrine"). Falls back to the normal download path
	 * if the local cache is missing.
	 */
	localGGUFOnly?: boolean;
	/**
	 * When true, `engine.embed(id, text)` is allowed to dispatch through
	 * `inferenceEngines` (the chat-model self-embedding path / "bucket D").
	 * The chat model produces an embedding by tapping the post-`output_norm`
	 * hidden state, last-token-pooling, and L2-normalizing. Quality drops
	 * 5-15% on MTEB benchmarks vs dedicated retrieval-tuned embedders;
	 * acceptable for in-domain retrieval (agent memory, dialogue history).
	 *
	 * Parity gate against a PyTorch HF f16 reference:
	 * - `q4f16_1` / `q0f16` (4-bit or f16 GGUF): `cos >= 0.999`
	 * - `hyb` (hybrid Q4_K token_embd + f16): `cos >= 0.995`
	 * - `iq3m` (IQ3_M i-quant GGUF): `cos >= 0.90` (quant-induced noise
	 *   accumulates across 36+ layers at 8B params; semantic quality
	 *   confirmed via 4-pair cosine-distinguishability check)
	 */
	embeddingCapable?: boolean;
	/**
	 * Pooling strategy for `engine.embed()` when `embeddingCapable: true`.
	 * Meaningful only on chat models routed through bucket D
	 * (`ModelInference.embed`); ignored for encoder / causal-embedder
	 * registrations whose pooling is fixed by their architecture.
	 *
	 * - `"last-token"` (default): take the post-`output_norm` hidden state
	 *   at the final token position. Matches the canonical bucket D ref
	 *   (qwen3-8b-iq3m).
	 * - `"mean"`: average the post-`output_norm` hidden state across all
	 *   token positions. Use on models with high last-token anisotropy
	 *   that compresses semantic separation between paraphrases and
	 *   unrelated text (e.g., Phi-3.5-mini).
	 */
	embeddingPooling?: "last-token" | "mean";
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
		// huggingface-quants/SmolLM2-360M-Instruct-GGUF returns 401 (gated/missing
		// as of 2026-04-26). bartowski's mirror is open and includes Q4_0 plus
		// the K-quant ladder. Pinning Q4_0 keeps the cross-family GEMV comparison
		// honest against tinyllama-1.1b-chat-q4_0's Q4_0 baseline.
		ggufUrl: "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF",
		ggufFilePattern: "Q4_0",
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
		// Picker would otherwise fall through to q4_k_m (1066 MB). §9 on
		// Qwen3-1.7B showed Q4_K_M is a slight tok/s regression vs Q8 due
		// to K-quant compute overhead clawing back bandwidth savings; Q4_0
		// (1017 MB) keeps wave-1 cross-family GEMV comparisons clean
		// against tinyllama-1.1b-chat-q4_0 and smollm2-360m-q4f16.
		ggufUrl: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
		ggufFilePattern: "Q4_0",
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
		// Mungert mirror is open; Q4_0 (920 MB) keeps the cross-family
		// GEMV comparison clean against tinyllama-1.1b-chat-q4_0,
		// smollm2-360m-q4f16, and qwen2.5-1.5b-q4f16 — all wave-1 entries
		// pinned to Q4_0 so absolute matmul ms are honest cross-family.
		ggufUrl: "https://huggingface.co/Mungert/SmolLM2-1.7B-Instruct-GGUF",
		ggufFilePattern: "Q4_0",
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
		// Resolves to Qwen3-1.7B-Q8_0.gguf via the picker fallback (only file
		// in the repo). TODO §9 (2026-04-26) tested Q4_0 (-11.8% matmul,
		// +0.7% tok/s — in noise, ~58% smaller download) and Q4_K_M
		// (-5.8% matmul, -4% tok/s) as alternatives via unsloth's mirror;
		// neither delivered Stub B's predicted ~40% matmul drop. Keeping Q8
		// as the canonical baseline. To re-enable the alternative-quant
		// experiment, switch ggufUrl to unsloth/Qwen3-1.7B-GGUF and pin
		// ggufFilePattern (e.g. "Q4_0", "Q4_K_M").
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
		// NousResearch's mirror has only K-quants + Q8_0, no Q4_0; switched
		// to bartowski for wave-1 quant parity. Pin "Q4_0." with trailing
		// dot to skip the ARM repack variants (same as llama-3.2-3b).
		ggufUrl: "https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF",
		ggufFilePattern: "Q4_0.",
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
		// Bartowski lists Q4_0 plus ARM-repack variants (Q4_0_4_4, _4_8,
		// _8_8) that use a SVE/dot-product layout our shader doesn't
		// handle. Pin the trailing dot to match the plain Q4_0.gguf only.
		ggufUrl: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
		ggufFilePattern: "Q4_0.",
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
		// Q4_0 (1.9 GB) is the wave-1 cross-family default. Qwen2 arch
		// requires the attn_{q,k,v}.bias support that landed 2026-04-26
		// (bug-fix #25); without it this entry produces gibberish like
		// qwen2.5-1.5b did pre-fix.
		ggufUrl: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
		ggufFilePattern: "Q4_0",
	},

	// Phi-3.5-mini-instruct — 3.82B, fused QKV + fused gate-up FFN.
	// Re-registered 2026-04-29 (Path B: fused-forward, phi3-gated). 32
	// layers, hidden 3072, 32 heads (no GQA), intermediate 8192, vocab
	// 32064. MIT license. Sliding window present in HF config but at
	// sliding_window=262144 (effectively no SWA at our ctx=4096). The
	// GGUF reports general.architecture="phi3" (per llama.cpp arch
	// table); the older "phi" entry is reserved for Phi-1 / Phi-2.
	// 197 tensors total: 6 per layer (attn_norm, attn_qkv, attn_output,
	// ffn_norm, ffn_up [fused gate-up], ffn_down) × 32 + 5 globals.
	// No norm biases, no lm_head bias for this specific model — the
	// loader's optional-bias paths stay null and inert.
	{
		id: "phi-3.5-mini-q4km",
		name: "Phi-3.5 Mini Instruct (Q4_K_M, fused-forward)",
		family: "Phi",
		architecture: "phi3",
		paramsB: 3.82,
		vramMB: 2520,
		defaultQuant: "q4km",
		availableQuants: ["q4km"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "MIT",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		// Bucket D self-embedding NOT enabled. Parity gates 10/10 at cos >= 0.91
		// (last-token, vs PyTorch f16 ref) but the 16+16 mean-margin
		// distinguishability harness measures margin = -0.006 (last-token)
		// and -0.027 (mean-pool) — paraphrase cosines are not separated from
		// unrelated cosines, so self-embedding produces semantically random
		// vectors. Probed and demoted 2026-04-30; closure report at
		// eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md.
		// Do not flip `embeddingCapable: true` without re-running the harness
		// against a different quant tier (Q5_K_M / Q6_K / f16) — Q4_K_M
		// quant noise compounded with the model's high last-token anisotropy
		// is the load-bearing failure.
		downloadUrl: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
		ggufUrl: "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF",
		ggufFilePattern: "Q4_K_M",
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
		// Qwen/Qwen3-4B-GGUF only carries K-quants + Q5/Q6/Q8 (no Q4_0).
		// Switched to unsloth's mirror to match wave-1 cross-family Q4_0
		// quant convention. Pin "Q4_0." with trailing dot to skip any
		// future ARM-repack variants (mirrors llama-3.2-3b/hermes-3 style).
		ggufUrl: "https://huggingface.co/unsloth/Qwen3-4B-GGUF",
		ggufFilePattern: "Q4_0.",
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
		ggufFilePattern: "f16",
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
		ggufFilePattern: "f16",
	},

	{
		id: "bge-small-en-v1.5-q0f16",
		name: "BGE Small EN v1.5",
		family: "BGE",
		architecture: "bert",
		paramsB: 0.033,
		vramMB: 240,
		defaultQuant: "q0f16",
		availableQuants: ["q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "MIT",
		contextLength: 512,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/BAAI/bge-small-en-v1.5",
		ggufUrl: "https://huggingface.co/ChristianAzinn/bge-small-en-v1.5-gguf",
		// File on the mirror is `bge-small-en-v1.5_fp16.gguf`; substring
		// `fp16` matches uniquely (won't collide with `fp32`).
		ggufFilePattern: "fp16",
	},

	{
		id: "bge-large-en-v1.5-q0f16",
		name: "BGE Large EN v1.5",
		family: "BGE",
		architecture: "bert",
		paramsB: 0.335,
		vramMB: 1500,
		defaultQuant: "q0f16",
		availableQuants: ["q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "MIT",
		contextLength: 512,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/BAAI/bge-large-en-v1.5",
		ggufUrl: "https://huggingface.co/ChristianAzinn/bge-large-en-v1.5-gguf",
		ggufFilePattern: "fp16",
	},

	{
		id: "jina-embeddings-v2-base-en-q0f16",
		name: "Jina Embeddings v2 Base EN",
		family: "Jina Embeddings",
		architecture: "jina-bert-v2",
		paramsB: 0.137,
		vramMB: 320,
		defaultQuant: "q0f16",
		availableQuants: ["q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 8192,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
		ggufUrl: "https://huggingface.co/gaianet/jina-embeddings-v2-base-en-GGUF",
		// Mirror publishes `jina-embeddings-v2-base-en-f16.gguf`; the `f16`
		// substring matches uniquely (no q*-quant variants on this mirror).
		ggufFilePattern: "f16",
	},

	{
		id: "nomic-embed-text-v1.5-q0f16",
		name: "Nomic Embed Text v1.5",
		family: "Nomic Embed",
		architecture: "nomic-bert",
		paramsB: 0.137,
		vramMB: 320,
		defaultQuant: "q0f16",
		availableQuants: ["q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 2048,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
		ggufUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
		// Matches `nomic-embed-text-v1.5.f16.gguf` on the mirror.
		ggufFilePattern: "f16",
	},

	{
		id: "qwen3-embedding-0.6b-hyb",
		name: "Qwen3 Embedding 0.6B (hybrid)",
		family: "Qwen3-Embedding",
		architecture: "qwen3-embedding",
		paramsB: 0.6,
		vramMB: 1100,
		// Hybrid quant: token_embd Q4_K (83 MiB; fits the 128 MiB WebGPU
		// per-binding cap), all other weights f16. Per-row dequant on the
		// embedding lookup doesn't compound through the forward, so parity
		// against f16 sentence-transformers refs holds at >=0.999. See
		// CLAUDE.md "Per-binding 128 MiB cap doctrine" for the recipe.
		// Build locally:
		//   llama-quantize --token-embedding-type Q4_K --allow-requantize \
		//     <upstream-f16>.gguf qwen3-embedding-0.6b-hyb.gguf F16
		defaultQuant: "hyb",
		availableQuants: ["hyb"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 32768,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
		// Hybrid GGUF is not published upstream; build locally per the
		// recipe above. `localGGUFOnly: true` makes the bench downloader
		// treat the local cache at `smoke-test/models/<id>.gguf` as
		// authoritative and skip the HF tree-fetch + size-verify.
		ggufUrl: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF",
		ggufFilePattern: "hyb",
		localGGUFOnly: true,
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

	// --- Wave 2 (7B+) ---
	// Q4_0 7B = ~4.46 GB exceeds the WASM 4 GiB streaming cap. Q3_K_M
	// (3.36 GB) was tried first and produced gibberish output — the Q3_K
	// matmul kernel has a correctness bug not exercised by wave-1
	// (which pinned to Q4_0 / Q8_0 across the fleet; §9 tested Q4_K_M
	// briefly). Q4_K_S (3.95 GB) fits with margin and uses the same
	// Q4_K shader family that §9 verified works. Mistral architecture
	// reports as "llama" in GGUF and uses the [INST]/[/INST] llama2-
	// style chat template. Bartowski mirror is open.
	{
		id: "mistral-7b-instruct-v0.3-q4ks",
		name: "Mistral 7B Instruct v0.3",
		family: "Mistral",
		architecture: "mistral",
		paramsB: 7.25,
		vramMB: 4400,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
		ggufUrl: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF",
		ggufFilePattern: "Q4_K_S",
	},

	// >4 GiB MEMORY64 reproducer (Phase 7 of the MEMORY64 full
	// migration, 2026-04-28). Q5_K_M (~5.1 GB) is the smallest
	// Mistral-7B quant that exceeds the wasm32 4 GiB streaming cap.
	// `pickWasmUrl` (src/core/engine.ts) auto-routes this entry to
	// webllm-wasm-mem64.{js,wasm} since file size > 3.5 GiB.
	//
	// Originally the standing wasm64 reproducer for the
	// `_wgpuDeviceCreateBindGroup` HEAPU32-truncation bug. Now decodes
	// end-to-end on wasm64 (34.6 tok/s greedy single-pass) under the
	// vendored Dawn `v20260423.175430` port (post-`8d78be5`); kept as
	// a Q5_K-family probe for future regressions until the canonical-6
	// parity sweep grows a Q5_K row.
	{
		id: "mistral-7b-instruct-v0.3-q5km",
		name: "Mistral 7B Instruct v0.3 (Q5_K_M)",
		family: "Mistral",
		architecture: "mistral",
		paramsB: 7.25,
		vramMB: 5400,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
		ggufUrl: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF",
		ggufFilePattern: "Q5_K_M",
	},

	// Q3_K_M wave-2 fleet entry — the UB-safe u32 loader fix
	// (llama.cpp `webllm-browser-patches` patch 11) restored Q3_K
	// correctness on Tint/Dawn. Original optimized Q3_K mul_mat_vec
	// / get_rows kernels are unchanged. Provides a Q3_K vs Q4_K_S
	// vs IQ4_XS three-way at the same Mistral-7B param count.
	{
		id: "mistral-7b-instruct-v0.3-q3km",
		name: "Mistral 7B Instruct v0.3 (Q3_K_M)",
		family: "Mistral",
		architecture: "mistral",
		paramsB: 7.25,
		vramMB: 3500,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
		ggufUrl: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF",
		ggufFilePattern: "Q3_K_M",
	},

	// IQ-family probe (kept as a working data point alongside the
	// Q4_K_S canonical entry above). IQ4_XS verified coherent
	// output on 2026-04-26 after the Q3_K shader (#28) blocked
	// the cleaner Q3_K_M path; this confirmed the IQ-family code
	// path is intact and unblocks 8B+ candidates via IQ3_M /
	// IQ3_XS quants.
	{
		id: "mistral-7b-instruct-v0.3-iq4xs",
		name: "Mistral 7B Instruct v0.3 (IQ4_XS)",
		family: "Mistral",
		architecture: "mistral",
		paramsB: 7.25,
		vramMB: 4200,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
		ggufUrl: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF",
		ggufFilePattern: "IQ4_XS",
	},

	// Mistral-Nemo-Instruct-2407 — 12B params, the >4 GiB MEMORY64
	// validation target (Phase 7, 2026-04-28..29). Real Mistral release
	// (2024-07), commonly mirrored. Q4_K_S (~6.63 GiB) is firmly above
	// the wasm32 4 GiB streaming cap. `pickWasmUrl` auto-routes via
	// vramMB > 3500.
	//
	// First Phase 7 attempt registered mistral-7b-q5km (5.1 GiB) and
	// blocked on what looked like a Q5_K-kernel wasm64 bug. Pivoting
	// to this 12B Q4_K_S target proved the bug was kernel-family-
	// independent — same _wgpuDeviceCreateBindGroup failure under a
	// Phase-5-validated kernel. Probe (PHASE-7-BLOCKED.md + FINDINGS)
	// pinned the bug to the Emscripten wasm64 shim reading 8-byte
	// handle pointers as HEAPU32 (low 32 only). Fix lives in
	// scripts/fix-mem64-bindgroup-shim.py and is applied after every
	// `make wasm-build-mem64`.
	{
		id: "mistral-nemo-instruct-2407-q4ks",
		name: "Mistral Nemo Instruct 2407 (Q4_K_S, >4 GiB validation)",
		family: "Mistral",
		architecture: "mistral",
		paramsB: 12.25,
		vramMB: 7000,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
		ggufUrl: "https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF",
		ggufFilePattern: "Q4_K_S",
	},

	// First 8B candidate (wave 2 model 2). Q4_K_S 8B = 4475 MB
	// exceeds the 4 GiB WASM cap; Q3_K_S would fit but routes
	// through the broken Q3_K shader (#28). IQ3_M (3609 MB) uses
	// GGML_TYPE_IQ3_S tensors which `supports_op` covers and the
	// IQ4_XS Mistral probe verified working in this branch on
	// 2026-04-26. Bartowski mirror is open with the full IQ
	// ladder.
	{
		id: "llama-3.1-8b-instruct-iq3m",
		name: "Llama 3.1 8B Instruct",
		family: "Llama 3.1",
		architecture: "llama",
		paramsB: 8.03,
		vramMB: 4500,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Llama-3.1",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
		ggufUrl: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
		ggufFilePattern: "IQ3_M",
	},

	// Wave 2 model 4 — Qwen3-8B at IQ3_M to round out the cross-
	// family 8B comparison alongside Llama-3.1-8B IQ3_M. Q4_K_S
	// 8B (4580 MB) exceeds the 4 GiB WASM cap; IQ3_M (3897 MB)
	// fits with margin and reuses the same GGML_TYPE_IQ3_S code
	// path verified working on Mistral IQ4_XS / Llama-3.1-8B
	// IQ3_M in §13. Bartowski mirror is open with the full IQ
	// ladder.
	{
		id: "qwen3-8b-iq3m",
		name: "Qwen3 8B",
		family: "Qwen3",
		architecture: "qwen",
		paramsB: 8.19,
		vramMB: 4500,
		defaultQuant: "iq3m",
		availableQuants: ["iq3m"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		embeddingCapable: true,
		downloadUrl: "https://huggingface.co/Qwen/Qwen3-8B",
		ggufUrl: "https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF",
		ggufFilePattern: "IQ3_M",
	},

	// Qwen3-14B Q4_K_S — 13B-class registration target (queued
	// 2026-04-29 from the MEMORY64 closure stub). 40 layers,
	// hidden 5120, GQA 5:1 (40 Q heads / 8 KV heads), SwiGLU,
	// vocab 151936. Disk filesize 7.99 GiB (HEAD-verified
	// 8,573,475,872 bytes on bartowski/Qwen_Qwen3-14B-GGUF);
	// well above the 4 GiB wasm32 streaming cap and well under
	// the 16 GiB Emscripten 5.0.6 wasm-ld --max-memory ceiling.
	// `pickWasmUrl(byteLength)` auto-routes to wasm64 since
	// vramMB > 3500. Same Qwen3 kernel surface already validated
	// at 0.6B / 1.7B / 4B / 8B; this row exercises the next
	// param-count band inside the 30B project ceiling. Expected
	// smoke-bench band: 15-19 tok/s (extrapolated from
	// Mistral-Nemo 12B Q4_K_S 19.3 tok/s and Mistral-7B Q4_K_S
	// 28.2 tok/s); hard floor 12 tok/s.
	{
		id: "qwen3-14b-q4ks",
		name: "Qwen3 14B (Q4_K_S, 13B-class registration)",
		family: "Qwen3",
		architecture: "qwen",
		paramsB: 14.77,
		vramMB: 8800,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: true, structuredOutput: true, vision: false, embedding: false },
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "quality",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/Qwen/Qwen3-14B",
		ggufUrl: "https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF",
		ggufFilePattern: "Q4_K_S",
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
