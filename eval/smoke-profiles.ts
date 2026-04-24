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
}

const DEFAULT_PROMPT = "Tell one short joke.";

/**
 * Profile set for dashboard-style comparison. Naming convention:
 *   <model-short>-<mode>-<temp>   where <temp> is hot (~0.9), warm (default),
 *                                 or cold (~0.3). Thinking variants are
 *                                 qwen3-only since llama has no think tags.
 */
export const SMOKE_PROFILES: readonly SmokeProfile[] = [
	// ── Qwen3 0.6B — non-thinking ──────────────────────────────
	{
		name: "qwen3-0.6b-off-cold",
		model: "qwen3-0.6b-q4f16",
		thinking: "off",
		temperature: 0.3,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-off-warm",
		model: "qwen3-0.6b-q4f16",
		thinking: "off",
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
		temperature: 0.3,
		maxTokens: 1024,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "qwen3-0.6b-thinking-warm",
		model: "qwen3-0.6b-q4f16",
		thinking: "on",
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
	// ── Llama 3.2 1B ───────────────────────────────────────────
	{
		name: "llama-3.2-1b-cold",
		model: "llama-3.2-1b-q4f16",
		temperature: 0.3,
		prompt: DEFAULT_PROMPT,
	},
	{
		name: "llama-3.2-1b-warm",
		model: "llama-3.2-1b-q4f16",
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
		prompt: DEFAULT_PROMPT,
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
		"llama-3.2-1b-warm",
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
	],
	full: [
		"qwen3-0.6b-off-cold",
		"qwen3-0.6b-off-warm",
		"qwen3-0.6b-off-hot",
		"qwen3-0.6b-thinking-cold",
		"qwen3-0.6b-thinking-warm",
		"qwen3-0.6b-thinking-hot",
		"llama-3.2-1b-cold",
		"llama-3.2-1b-warm",
		"llama-3.2-1b-hot",
		"tinyllama-warm",
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
