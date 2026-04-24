import type { EvalTask } from "../types.js";

export const reasoningTasks: EvalTask[] = [
	// ── Easy ──────────────────────────────────────────────────────────────
	{
		id: "rs-001",
		dimension: "reasoning",
		description: "Basic addition",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "What is 2 + 2?",
		expected: "4",
		scoring: { type: "contains", value: "4" },
		difficulty: "easy",
	},
	{
		id: "rs-002",
		dimension: "reasoning",
		description: "Basic geography fact",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "What is the capital of France?",
		expected: "Paris",
		scoring: { type: "contains", value: "Paris" },
		difficulty: "easy",
	},
	{
		id: "rs-003",
		dimension: "reasoning",
		description: "Basic count of days in a week",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "How many days are in a week?",
		expected: "7",
		scoring: { type: "contains", value: "7" },
		difficulty: "easy",
	},
	{
		id: "rs-004",
		dimension: "reasoning",
		description: "Basic observation about sky color",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "What color is the sky on a clear day?",
		expected: "blue",
		scoring: { type: "contains", value: "blue" },
		difficulty: "easy",
	},

	// ── Medium ───────────────────────────────────────────────────────────
	{
		id: "rs-005",
		dimension: "reasoning",
		description: "Syllogistic reasoning — basic deductive logic",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "If all cats are animals, and Whiskers is a cat, is Whiskers an animal?",
		expected: "Yes, Whiskers is an animal.",
		scoring: { type: "contains", value: "yes" },
		difficulty: "medium",
	},
	{
		id: "rs-006",
		dimension: "reasoning",
		description: "Percentage calculation — sale price",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "A shirt costs $25. It's on sale for 20% off. What is the sale price?",
		expected: "$20",
		scoring: { type: "contains", value: "20" },
		difficulty: "medium",
	},
	{
		id: "rs-007",
		dimension: "reasoning",
		description: "Decimal comparison",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "Which is larger: 0.3 or 0.29?",
		expected: "0.3",
		scoring: { type: "contains", value: "0.3" },
		difficulty: "medium",
	},
	{
		id: "rs-008",
		dimension: "reasoning",
		description: "Physical cause-and-effect reasoning",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "If you drop a glass on a concrete floor, what will likely happen?",
		expected: "It will break.",
		scoring: { type: "regex", pattern: "\\b(break|shatter)\\b" },
		difficulty: "medium",
	},

	// ── Hard ─────────────────────────────────────────────────────────────
	{
		id: "rs-009",
		dimension: "reasoning",
		description: "Rate-distance calculation with decimal multiplier",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "A train travels 60 miles per hour. How far will it travel in 2.5 hours?",
		expected: "150 miles",
		scoring: { type: "contains", value: "150" },
		difficulty: "hard",
	},
	{
		id: "rs-010",
		dimension: "reasoning",
		description: "Day-of-week arithmetic — 10 days from Wednesday",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "If today is Wednesday, what day of the week will it be in 10 days?",
		expected: "Saturday",
		scoring: { type: "regex", pattern: "\\b[Ss]aturday\\b" },
		difficulty: "hard",
	},
	{
		id: "rs-011",
		dimension: "reasoning",
		description: "Trick question — sheep counting",
		systemPrompt:
			"You are a helpful assistant. Answer questions directly and concisely.",
		input: "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
		expected: "9",
		scoring: { type: "contains", value: "9" },
		difficulty: "hard",
	},
	{
		id: "rs-012",
		dimension: "reasoning",
		description:
			"Water jug problem — measure exactly 4 gallons using 5 and 3 gallon jugs",
		systemPrompt:
			"You are a helpful assistant. Think step by step and explain your reasoning clearly.",
		input: "I have a 5-gallon jug and a 3-gallon jug. How can I measure exactly 4 gallons?",
		expected:
			"Fill the 5-gallon jug, pour into the 3-gallon jug (leaving 2 in the 5), empty the 3, pour the 2 into the 3, fill the 5 again, pour into the 3 (which has 2, so takes 1 more), leaving exactly 4 in the 5-gallon jug.",
		scoring: { type: "custom", name: "rs-012-water-jug" },
		difficulty: "hard",
	},
];
