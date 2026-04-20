import type { EvalTask } from "../types.js";

export const embeddingTasks: EvalTask[] = [
	// ── Easy ──────────────────────────────────────────────────────────────
	{
		id: "emb-001",
		dimension: "embedding",
		description: "Semantic similarity — pick the most similar word",
		systemPrompt:
			"You are a semantic similarity assistant. Answer with exactly one word from the given choices. Do not add explanation or punctuation.",
		input:
			"Which word is most similar in meaning to 'happy': sad, joyful, tired, heavy?",
		expected: "joyful",
		scoring: { type: "contains", value: "joyful" },
		difficulty: "easy",
	},
	{
		id: "emb-002",
		dimension: "embedding",
		description: "Basic categorization — assign item to a category",
		systemPrompt:
			"You are a categorization assistant. Respond with exactly one of the provided category names. Do not add explanation or punctuation.",
		input:
			"Which category does 'apple' belong to: animals, vehicles, food, furniture?",
		expected: "food",
		scoring: { type: "contains", value: "food" },
		difficulty: "easy",
	},
	{
		id: "emb-003",
		dimension: "embedding",
		description: "Synonym detection — identify a synonym",
		systemPrompt:
			"You are a vocabulary assistant. Provide a single-word synonym for the given word. Do not add explanation or punctuation.",
		input: "What is a synonym for 'fast'?",
		expected: "quick",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const synonyms = [
					"quick",
					"rapid",
					"swift",
					"speedy",
					"brisk",
					"hasty",
				];
				const lower = output.toLowerCase().trim();
				if (synonyms.some((s) => lower.includes(s))) return 1;
				return 0;
			},
		},
		difficulty: "easy",
	},
	{
		id: "emb-004",
		dimension: "embedding",
		description: "Antonym detection — identify an opposite",
		systemPrompt:
			"You are a vocabulary assistant. Provide a single-word antonym for the given word. Do not add explanation or punctuation.",
		input: "What is the opposite of 'hot'?",
		expected: "cold",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const antonyms = [
					"cold",
					"cool",
					"freezing",
					"frigid",
					"icy",
					"chilly",
				];
				const lower = output.toLowerCase().trim();
				if (antonyms.some((a) => lower.includes(a))) return 1;
				return 0;
			},
		},
		difficulty: "easy",
	},

	// ── Medium ───────────────────────────────────────────────────────────
	{
		id: "emb-005",
		dimension: "embedding",
		description: "Analogy completion — A is to B as C is to ?",
		systemPrompt:
			"You are an analogy assistant. Complete the analogy by responding with exactly one word. Do not add explanation or punctuation.",
		input: "Hand is to glove as foot is to what?",
		expected: "sock",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const validAnswers = ["sock", "shoe", "boot"];
				const lower = output.toLowerCase().trim();
				if (validAnswers.some((a) => lower.includes(a))) return 1;
				return 0;
			},
		},
		difficulty: "medium",
	},
	{
		id: "emb-006",
		dimension: "embedding",
		description: "Semantic grouping — categorize items into groups",
		systemPrompt:
			"You are a categorization assistant. Group the provided items by category. Output each group on a separate line in the format 'Category: item1, item2'. Use only the items given.",
		input:
			"Group these items by category: salmon, carrot, trout, broccoli, tuna",
		expected:
			"Fish: salmon, trout, tuna | Vegetable: carrot, broccoli",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const lower = output.toLowerCase();
				const fish = ["salmon", "trout", "tuna"];
				const vegetables = ["carrot", "broccoli"];
				let groupedCorrectly = 0;

				// Check that fish are grouped together (appear on the same line)
				for (const f of fish) {
					if (lower.includes(f)) groupedCorrectly++;
				}
				// Check that vegetables are grouped together
				for (const v of vegetables) {
					if (lower.includes(v)) groupedCorrectly++;
				}

				// All 5 items must appear
				if (groupedCorrectly < 5) return 0;

				// Check that fish appear on the same line and vegetables on the same line
				const lines = output.split("\n");
				const fishLine = lines.find(
					(l) =>
						l.toLowerCase().includes("salmon") ||
						l.toLowerCase().includes("trout") ||
						l.toLowerCase().includes("tuna"),
				);
				const vegLine = lines.find(
					(l) =>
						l.toLowerCase().includes("carrot") ||
						l.toLowerCase().includes("broccoli"),
				);

				if (!fishLine || !vegLine) return 0.5;

				const fishTogether =
					fishLine.toLowerCase().includes("salmon") &&
					fishLine.toLowerCase().includes("trout") &&
					fishLine.toLowerCase().includes("tuna");
				const vegTogether =
					vegLine.toLowerCase().includes("carrot") &&
					vegLine.toLowerCase().includes("broccoli");

				if (fishTogether && vegTogether) return 1;
				if (fishTogether || vegTogether) return 0.75;
				return 0.5;
			},
		},
		difficulty: "medium",
	},
	{
		id: "emb-007",
		dimension: "embedding",
		description:
			"Contextual similarity — determine if a word means the same thing in context",
		systemPrompt:
			"You are a semantic analysis assistant. Answer with exactly 'yes' or 'no'. Do not add explanation.",
		input:
			"In the sentence 'The bank was steep and muddy', does the word 'bank' mean the same as in 'She deposited money at the bank'?",
		expected: "no",
		scoring: {
			type: "regex",
			pattern: "(?i)^\\s*no\\b",
		},
		difficulty: "medium",
	},
	{
		id: "emb-008",
		dimension: "embedding",
		description:
			"Cross-domain mapping — find analogous concepts across domains",
		systemPrompt:
			"You are an analogy assistant. Identify the concept that best matches. Respond with exactly one of the given choices. Do not add explanation.",
		input:
			"In a computer, a 'firewall' blocks unauthorized network traffic. Which everyday object is most analogous: umbrella, security guard, flashlight, thermostat?",
		expected: "security guard",
		scoring: { type: "contains", value: "security guard" },
		difficulty: "medium",
	},

	// ── Hard ─────────────────────────────────────────────────────────────
	{
		id: "emb-009",
		dimension: "embedding",
		description:
			"Multi-hop reasoning about semantic relationships",
		systemPrompt:
			"You are a semantic reasoning assistant. Answer with exactly one word. Do not add explanation or punctuation.",
		input:
			"If 'puppy' is related to 'dog', and 'dog' is related to 'canine', then what is 'puppy' related to through both 'dog' and the young of cats?",
		expected: "kitten",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const lower = output.toLowerCase().trim();
				if (lower.includes("kitten")) return 1;
				if (lower.includes("cub")) return 0.75;
				if (lower.includes("cat") && !lower.includes("category"))
					return 0.5;
				return 0;
			},
		},
		difficulty: "hard",
	},
	{
		id: "emb-010",
		dimension: "embedding",
		description:
			"Fine-grained semantic distinction between near-synonyms",
		systemPrompt:
			"You are a precise language assistant. Choose the word that best fits the described nuance. Respond with exactly one of the given choices. Do not add explanation.",
		input:
			"Which word best describes a low, continuous sound made by a person expressing dissatisfaction: 'murmur', 'grumble', 'whisper', 'shout'?",
		expected: "grumble",
		scoring: { type: "contains", value: "grumble" },
		difficulty: "hard",
	},
	{
		id: "emb-011",
		dimension: "embedding",
		description: "Embedding space arithmetic — word analogy via vector math",
		systemPrompt:
			"You are a word embedding assistant. Given a word equation, solve for the missing word. Respond with exactly one word. Do not add explanation or punctuation.",
		input: "king - man + woman = ?",
		expected: "queen",
		scoring: { type: "contains", value: "queen" },
		difficulty: "hard",
	},
	{
		id: "emb-012",
		dimension: "embedding",
		description:
			"Context-dependent similarity across multiple word senses",
		systemPrompt:
			"You are a word-sense disambiguation assistant. For each given sentence, identify whether the target word is used in the same sense. Respond with only 'same' or 'different' for each pair, one per line.",
		input:
			"Compare the word 'light' in these pairs:\n" +
			"1. 'The room has good light' vs 'This box is very light'\n" +
			"2. 'Turn on the light' vs 'The light was blinding'\n" +
			"3. 'She wore a light jacket' vs 'The feather is light'",
		expected: "different\nsame\nsame",
		scoring: {
			type: "custom",
			scorer: (output: string, _expected: string): number => {
				const expectedAnswers = ["different", "same", "same"];
				const lines = output
					.split("\n")
					.map((l) => l.toLowerCase().trim())
					.filter((l) => l.length > 0);

				let correct = 0;
				for (let i = 0; i < expectedAnswers.length; i++) {
					const target = expectedAnswers[i];
					if (lines[i] && lines[i].includes(target)) {
						correct++;
					} else {
						// Also check if the answer appears anywhere in output near the right context
						const allAnswers = lines.join(" ");
						// Count occurrences of each expected answer
						const sameCount = (
							allAnswers.match(/\bsame\b/g) || []
						).length;
						const diffCount = (
							allAnswers.match(/\bdifferent\b/g) || []
						).length;
						// If we have 2 "same" and 1 "different" total, give partial credit
						if (sameCount === 2 && diffCount === 1) {
							return 0.75;
						}
					}
				}
				return correct / 3;
			},
		},
		difficulty: "hard",
	},
];
