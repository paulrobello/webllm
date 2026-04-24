import type { EvalTask } from "../types.js";

export const instructionTasks: EvalTask[] = [
	// ── Easy ──────────────────────────────────────────────────────────────
	{
		id: "in-001",
		dimension: "instruction-following",
		description: "Format output with bullet points",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "List 3 colors. Use bullet points.",
		expected: "A list of 3 colors with bullet points",
		scoring: { type: "regex", pattern: "^\\s*[-*•]" },
		difficulty: "easy",
	},
	{
		id: "in-002",
		dimension: "instruction-following",
		description: "Answer in exactly one sentence",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "What is water? Answer in exactly one sentence.",
		expected: "A single sentence defining water",
		scoring: { type: "custom", name: "in-002-one-sentence" },
		difficulty: "easy",
	},
	{
		id: "in-003",
		dimension: "instruction-following",
		description: "Start response with specific phrase",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "Tell me about cats. Start your response with 'Cats are'.",
		expected: "Response starts with 'Cats are'",
		scoring: { type: "regex", pattern: "^Cats are" },
		difficulty: "easy",
	},
	{
		id: "in-004",
		dimension: "instruction-following",
		description: "Answer with just a number",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "What is 2+2? Answer with just the number.",
		expected: "4",
		scoring: { type: "regex", pattern: "^\\s*4\\s*$" },
		difficulty: "easy",
	},

	// ── Medium ───────────────────────────────────────────────────────────
	{
		id: "in-005",
		dimension: "instruction-following",
		description: "Respond in JSON format with specific schema",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input:
			"What are the primary colors? Respond in JSON format with a 'colors' array field.",
		expected: '{"colors":["red","yellow","blue"]}',
		scoring: {
			type: "json_schema",
			schema: { colors: "array" },
		},
		difficulty: "medium",
	},
	{
		id: "in-006",
		dimension: "instruction-following",
		description: "Numbered list with exactly 5 items",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "Name the first 5 planets from the sun as a numbered list.",
		expected:
			"A numbered list with 5 planets: Mercury, Venus, Earth, Mars, Jupiter",
		scoring: { type: "custom", name: "in-006-numbered-5-items" },
		difficulty: "medium",
	},
	{
		id: "in-007",
		dimension: "instruction-following",
		description: "Include a required word in the explanation",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "Explain gravity. Your explanation must include the word 'mass'.",
		expected:
			"An explanation of gravity that includes the word 'mass'",
		scoring: { type: "contains", value: "mass" },
		difficulty: "medium",
	},
	{
		id: "in-008",
		dimension: "instruction-following",
		description: "Describe something while avoiding specific words",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input: "Describe a sunset without using the words 'beautiful', 'pretty', or 'nice'.",
		expected:
			"A sunset description that avoids the forbidden words",
		scoring: { type: "custom", name: "in-008-avoid-forbidden-words" },
		difficulty: "medium",
	},

	// ── Hard ─────────────────────────────────────────────────────────────
	{
		id: "in-009",
		dimension: "instruction-following",
		description:
			"Explain photosynthesis with 3 constraints: 3 bullet points, each starting with a capital, includes 'chlorophyll'",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input:
			"Explain photosynthesis in exactly 3 bullet points, each starting with a capital letter, and include the word 'chlorophyll' somewhere.",
		expected:
			"3 bullet points, each capitalized, containing 'chlorophyll'",
		scoring: { type: "custom", name: "in-009-photosynthesis-3-bullets" },
		difficulty: "hard",
	},
	{
		id: "in-010",
		dimension: "instruction-following",
		description: "Conditional response — respond based on even/odd",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input:
			"If the number I'm thinking of is even, respond with 'EVEN'. If odd, respond with 'ODD'. I'm thinking of 7.",
		expected: "ODD",
		scoring: { type: "exact" },
		difficulty: "hard",
	},
	{
		id: "in-011",
		dimension: "instruction-following",
		description: "JSON output with specific field types and values",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input:
			"Respond with a JSON object with fields: name (string), age (number), hobbies (array). Use these values: name is Alice, age is 30, hobbies are reading and hiking.",
		expected:
			'{"name":"Alice","age":30,"hobbies":["reading","hiking"]}',
		scoring: { type: "custom", name: "in-011-alice-json" },
		difficulty: "hard",
	},
	{
		id: "in-012",
		dimension: "instruction-following",
		description: "Answer 3 questions in exact specified order",
		systemPrompt:
			"You are a helpful assistant. Follow the user's formatting instructions exactly.",
		input:
			"First, tell me what 3+4 equals. Then tell me what color the sky is. Then tell me the capital of Japan. Do them in this exact order.",
		expected:
			"7 appears first, then blue, then Tokyo — in that order",
		scoring: { type: "custom", name: "in-012-three-questions-order" },
		difficulty: "hard",
	},
];
