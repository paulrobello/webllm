import type { BenchmarkModel } from "./models.js";
import { getModelById } from "./models.js";

export type ThinkingMode = "off" | "on";

/**
 * A named test profile pinning a model plus all the knobs we want to vary
 * between runs. Fields are optional so profiles can be as narrow or wide
 * as needed; anything unset falls back to the page / sampler defaults for
 * the model's architecture.
 */
export interface SmokeProfile {
	name: string;
	model: string;
	thinking?: ThinkingMode;
	contextLength?: number;
	maxTokens?: number;
	temperature?: number;
	topK?: number;
	topP?: number;
	repetitionPenalty?: number;
	seed?: number;
	prompt?: string;
	/**
	 * Marks the profile as targeting an embedding-only model (no text
	 * generation). Bench harnesses skip the speed phase (`chat-smoke`) and
	 * accuracy harnesses default to the `embedding` dimension.
	 */
	embedding?: boolean;
}

const DEFAULT_PROMPT = "Tell one short joke.";

/**
 * Profile set for dashboard-style comparison. Naming convention:
 *   <model-short>-<mode>-<temp>   where <temp> is hot (0.9), warm (0.6),
 *                                 or cold (0.1). Thinking variants are
 *                                 qwen3-only since llama has no think tags.
 *
 * Tool-calling tasks are restricted to cold profiles (see browser-eval.ts) —
 * rigid JSON output degrades quickly above ~0.2.
 */
export const SMOKE_PROFILES: readonly SmokeProfile[] = [
	// ── Qwen3 0.6B — non-thinking ──────────────────────────────
	{
		name: "qwen3-0.6b-off-cold",
		model: "qwen3-0.6b-q4f16",
		thinking: "off",
		temperature: 0.1,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-off-warm",
		model: "qwen3-0.6b-q4f16",
		thinking: "off",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-off-hot",
		model: "qwen3-0.6b-q4f16",
		thinking: "off",
		temperature: 0.9,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen3 0.6B — thinking ──────────────────────────────────
	{
		name: "qwen3-0.6b-thinking-cold",
		model: "qwen3-0.6b-q4f16",
		thinking: "on",
		temperature: 0.1,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-thinking-warm",
		model: "qwen3-0.6b-q4f16",
		thinking: "on",
		temperature: 0.6,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-thinking-hot",
		model: "qwen3-0.6b-q4f16",
		thinking: "on",
		temperature: 0.9,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen3 1.7B — thinking off ──────────────────────────────
	{
		name: "qwen3-1.7b-off-cold",
		model: "qwen3-1.7b-q4f16",
		temperature: 0.1,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-1.7b-off-warm",
		model: "qwen3-1.7b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-1.7b-off-hot",
		model: "qwen3-1.7b-q4f16",
		temperature: 0.9,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen3 1.7B — thinking ──────────────────────────────────
	{
		name: "qwen3-1.7b-thinking-cold",
		model: "qwen3-1.7b-q4f16",
		thinking: "on",
		temperature: 0.1,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-1.7b-thinking-warm",
		model: "qwen3-1.7b-q4f16",
		thinking: "on",
		temperature: 0.6,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-1.7b-thinking-hot",
		model: "qwen3-1.7b-q4f16",
		thinking: "on",
		temperature: 0.9,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	// ── Llama 3.2 1B ───────────────────────────────────────────
	{
		name: "llama-3.2-1b-cold",
		model: "llama-3.2-1b-q4f16",
		temperature: 0.1,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "llama-3.2-1b-warm",
		model: "llama-3.2-1b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "llama-3.2-1b-hot",
		model: "llama-3.2-1b-q4f16",
		temperature: 0.9,
		prompt: DEFAULT_PROMPT,
	},
	// ── TinyLlama ──────────────────────────────────────────────
	{
		name: "tinyllama-warm",
		model: "tinyllama-1.1b-chat-q4_0",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── SmolLM2 360M ───────────────────────────────────────────
	{
		name: "smollm2-360m-warm",
		model: "smollm2-360m-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen2.5 1.5B ───────────────────────────────────────────
	{
		name: "qwen2.5-1.5b-warm",
		model: "qwen2.5-1.5b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── SmolLM2 1.7B ───────────────────────────────────────────
	{
		name: "smollm2-1.7b-warm",
		model: "smollm2-1.7b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen2.5 3B ─────────────────────────────────────────────
	{
		name: "qwen2.5-3b-warm",
		model: "qwen2.5-3b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Llama 3.2 3B ───────────────────────────────────────────
	{
		name: "llama-3.2-3b-warm",
		model: "llama-3.2-3b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Hermes 3 (Llama 3.2 3B fine-tune) ─────────────────────
	{
		name: "hermes-3-llama-3.2-3b-warm",
		model: "hermes-3-llama-3.2-3b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen3 4B ───────────────────────────────────────────────
	{
		name: "qwen3-4b-warm",
		model: "qwen3-4b-q4f16",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-4b-thinking-warm",
		model: "qwen3-4b-q4f16",
		thinking: "on",
		temperature: 0.6,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	// ── Mistral 7B Instruct v0.3 (wave 2) ─────────────────────
	{
		name: "mistral-7b-v0.3-warm",
		model: "mistral-7b-instruct-v0.3-q4ks",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "mistral-7b-v0.3-q3km-warm",
		model: "mistral-7b-instruct-v0.3-q3km",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Llama 3.1 8B Instruct (wave 2 model 2) ────────────────
	{
		name: "llama-3.1-8b-warm",
		model: "llama-3.1-8b-instruct-iq3m",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	// ── Qwen3 8B (wave 2 model 4) ─────────────────────────────
	{
		name: "qwen3-8b-warm",
		model: "qwen3-8b-iq3m",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-8b-thinking-warm",
		model: "qwen3-8b-iq3m",
		thinking: "on",
		temperature: 0.6,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	// ── Snowflake Arctic Embed (encoder-only) ──────────────────
	// Embedding profiles don't generate; the temperature / thinking /
	// prompt fields are intentionally absent. Bench harnesses key on
	// `embedding: true` to skip speed phases and route accuracy runs to
	// the embedding dimension only.
	{
		name: "arctic-embed-s",
		model: "snowflake-arctic-embed-s-q0f32-b4",
		embedding: true,
	},
	{
		name: "arctic-embed-m",
		model: "snowflake-arctic-embed-m-q0f32-b4",
		embedding: true,
	},
	{
		name: "bge-small",
		model: "bge-small-en-v1.5-q0f16",
		embedding: true,
	},
	{
		name: "bge-large",
		model: "bge-large-en-v1.5-q0f16",
		embedding: true,
	},
	{
		name: "jina-embeddings-v2-base-en",
		model: "jina-embeddings-v2-base-en-q0f16",
		embedding: true,
	},
];

/**
 * Pre-built comparison sets you can pass via `--profiles <name>` in the
 * matrix runner without spelling out every profile name.
 */
export const SMOKE_PROFILE_SETS: Readonly<Record<string, readonly string[]>> = {
	"llama-vs-qwen": [
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-thinking-warm",
		"qwen3-1.7b-off-warm",
		"qwen3-1.7b-thinking-warm",
		"llama-3.2-1b-warm",
	],
	"qwen3-sizes": [
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-thinking-warm",
		"qwen3-1.7b-off-warm",
		"qwen3-1.7b-thinking-warm",
		"qwen3-4b-warm",
		"qwen3-4b-thinking-warm",
		"qwen3-8b-warm",
		"qwen3-8b-thinking-warm",
	],
	"temperature-sweep": [
		"qwen3-0.6b-off-cold",
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-off-hot",
		"llama-3.2-1b-cold",
		"llama-3.2-1b-warm",
		"llama-3.2-1b-hot",
	],
	"thinking-modes": [
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-thinking-cold",
		"qwen3-0.6b-thinking-warm",
		"qwen3-0.6b-thinking-hot",
		"qwen3-1.7b-off-warm",
		"qwen3-1.7b-thinking-cold",
		"qwen3-1.7b-thinking-warm",
		"qwen3-1.7b-thinking-hot",
	],
	full: [
		"qwen3-0.6b-off-cold",
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-off-hot",
		"qwen3-0.6b-thinking-cold",
		"qwen3-0.6b-thinking-warm",
		"qwen3-0.6b-thinking-hot",
		"qwen3-1.7b-off-cold",
		"qwen3-1.7b-off-warm",
		"qwen3-1.7b-off-hot",
		"qwen3-1.7b-thinking-cold",
		"qwen3-1.7b-thinking-warm",
		"qwen3-1.7b-thinking-hot",
		"llama-3.2-1b-cold",
		"llama-3.2-1b-warm",
		"llama-3.2-1b-hot",
		"tinyllama-warm",
		"smollm2-360m-warm",
		"qwen2.5-1.5b-warm",
		"smollm2-1.7b-warm",
		"qwen2.5-3b-warm",
		"llama-3.2-3b-warm",
		"hermes-3-llama-3.2-3b-warm",
		"qwen3-4b-warm",
		"qwen3-4b-thinking-warm",
		"mistral-7b-v0.3-warm",
		"mistral-7b-v0.3-q3km-warm",
		"llama-3.1-8b-warm",
		"qwen3-8b-warm",
		"qwen3-8b-thinking-warm",
		"arctic-embed-s",
		"arctic-embed-m",
		"bge-small",
		"bge-large",
		"jina-embeddings-v2-base-en",
	],
	embeddings: [
		"arctic-embed-s",
		"arctic-embed-m",
		"bge-small",
		"bge-large",
		"jina-embeddings-v2-base-en",
	],
};

export function getSmokeProfileSet(name: string): readonly string[] | undefined {
	return SMOKE_PROFILE_SETS[name];
}

export function listSmokeProfileSets(): string[] {
	return Object.keys(SMOKE_PROFILE_SETS);
}

export function listSmokeProfiles(): string[] {
	return SMOKE_PROFILES.map((p) => p.name);
}

export function getSmokeProfile(name: string): SmokeProfile | undefined {
	return SMOKE_PROFILES.find((p) => p.name === name);
}

export function resolveProfileModel(
	profile: SmokeProfile,
): BenchmarkModel | undefined {
	return getModelById(profile.model);
}

/**
 * Convert a profile to the URL query params the smoke page reads. Returned
 * values are strings/numbers only, never undefined, so callers can spread
 * them into an `extraParams` record directly.
 */
export function profileToUrlParams(
	profile: SmokeProfile,
): Record<string, string | number> {
	const params: Record<string, string | number> = {};
	if (profile.thinking === "on") params.thinking = 1;
	if (profile.contextLength !== undefined) params.ctx = profile.contextLength;
	if (profile.maxTokens !== undefined) params.max = profile.maxTokens;
	if (profile.temperature !== undefined) params.temp = profile.temperature;
	if (profile.topK !== undefined) params.topK = profile.topK;
	if (profile.topP !== undefined) params.topP = profile.topP;
	if (profile.repetitionPenalty !== undefined)
		params.rep = profile.repetitionPenalty;
	if (profile.seed !== undefined) params.seed = profile.seed;
	if (profile.prompt !== undefined) params.prompt = profile.prompt;
	return params;
}
