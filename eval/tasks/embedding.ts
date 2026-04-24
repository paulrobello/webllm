import type { EvalTask } from "../types.js";

/**
 * True embedding-track tasks. Unlike the chat-driven `semantic-reasoning`
 * dimension, these are scored by cosine similarity between two vectors
 * produced by `engine.embed(modelId, text)` — no generation, no chat
 * template. They are only run against models whose
 * `capabilities.embedding` is `true` (see `eval/browser-eval.ts` for the
 * auto-skip). Running them requires a working encoder forward pass in
 * `ModelInference` for the model's architecture; until that lands the
 * runner surfaces a clear "embedding not implemented" error.
 *
 * The `expected` field is only used for logging. The scoring compares
 * `input` (or the model's generated text, for chat-capable models that
 * reuse this suite) against `scoring.reference`.
 *
 * Pass threshold is tuned per task — pairs that should be near-synonyms
 * require higher similarity than loosely related pairs.
 */
export const embeddingTasks: EvalTask[] = [
	// ── Synonym / near-synonym pairs (high cosine expected) ──────────────
	{
		id: "emb-sim-001",
		dimension: "embedding",
		description: "Near-synonym pair: happy / joyful",
		systemPrompt: "",
		input: "happy",
		expected: "joyful",
		scoring: {
			type: "cosine_similarity",
			reference: "joyful",
			threshold: 0.75,
		},
		difficulty: "easy",
	},
	{
		id: "emb-sim-002",
		dimension: "embedding",
		description: "Near-synonym pair: car / automobile",
		systemPrompt: "",
		input: "car",
		expected: "automobile",
		scoring: {
			type: "cosine_similarity",
			reference: "automobile",
			threshold: 0.75,
		},
		difficulty: "easy",
	},
	{
		id: "emb-sim-003",
		dimension: "embedding",
		description: "Near-synonym pair: big / large",
		systemPrompt: "",
		input: "big",
		expected: "large",
		scoring: {
			type: "cosine_similarity",
			reference: "large",
			threshold: 0.75,
		},
		difficulty: "easy",
	},
	// ── Same-topic sentence pairs (mid–high cosine expected) ─────────────
	{
		id: "emb-sent-001",
		dimension: "embedding",
		description: "Same-topic sentences about dogs",
		systemPrompt: "",
		input: "The dog chased the ball across the park.",
		expected: "A puppy ran after a tennis ball in the backyard.",
		scoring: {
			type: "cosine_similarity",
			reference: "A puppy ran after a tennis ball in the backyard.",
			threshold: 0.6,
		},
		difficulty: "medium",
	},
	{
		id: "emb-sent-002",
		dimension: "embedding",
		description: "Same-topic sentences about cooking",
		systemPrompt: "",
		input: "She seasoned the steak with salt and pepper before grilling.",
		expected: "He added black pepper and sea salt to the meat before cooking it over a flame.",
		scoring: {
			type: "cosine_similarity",
			reference:
				"He added black pepper and sea salt to the meat before cooking it over a flame.",
			threshold: 0.6,
		},
		difficulty: "medium",
	},
	// ── Paraphrase pairs (high cosine expected) ──────────────────────────
	{
		id: "emb-para-001",
		dimension: "embedding",
		description: "Paraphrase: passive vs. active voice",
		systemPrompt: "",
		input: "The cat was chased by the dog.",
		expected: "The dog chased the cat.",
		scoring: {
			type: "cosine_similarity",
			reference: "The dog chased the cat.",
			threshold: 0.7,
		},
		difficulty: "medium",
	},
	{
		id: "emb-para-002",
		dimension: "embedding",
		description: "Paraphrase: reordered clauses",
		systemPrompt: "",
		input: "Because it was raining, we stayed inside.",
		expected: "We stayed inside because it was raining.",
		scoring: {
			type: "cosine_similarity",
			reference: "We stayed inside because it was raining.",
			threshold: 0.8,
		},
		difficulty: "easy",
	},
	// ── Unrelated pairs (low cosine expected — pass by being LOW) ────────
	// For these, threshold is the max acceptable similarity — we invert
	// in the scorer when the task description demands "should be different".
	// Kept minimal for now; can grow as the encoder lands.
	{
		id: "emb-neg-001",
		dimension: "embedding",
		description: "Unrelated topic sanity check: weather vs. cooking",
		systemPrompt: "",
		input: "The forecast predicts thunderstorms this afternoon.",
		expected: "I boiled the pasta for nine minutes.",
		scoring: {
			type: "cosine_similarity",
			reference: "I boiled the pasta for nine minutes.",
			// Low threshold — we expect low similarity. The score maps
			// cosine to [0, 1]; a pass at 0.4 means the encoder
			// correctly kept these apart. If cosine >= 0.4 the encoder
			// is probably collapsing sentences to a shared "some topic"
			// region, which is a bug worth flagging.
			threshold: 0.4,
		},
		difficulty: "easy",
	},
];
