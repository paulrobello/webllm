/**
 * Long prompts for §4 FA revisit prefill TTFT measurement.
 *
 * Three deterministic prompts with target token counts measured against
 * the LLaMA / Qwen / Mistral tokenizer family (BPE, ~1 token per ~4
 * characters of English prose). Counts are approximate — within ±15%
 * across the four §4-baseline tokenizers. Treat the IDs as workload
 * labels, not exact token counts.
 *
 * Each prompt ends with an open question that drives the model to
 * generate a long-form answer, exercising the long-decode workload
 * point too.
 */

const FILLER_PARAGRAPH =
	"Software engineering is the systematic application of engineering principles to the design, development, maintenance, testing, and evaluation of software. It involves understanding requirements, designing systems, writing code, performing tests, and ensuring quality through processes that are repeatable and predictable. Modern software engineering relies on tools, frameworks, and methodologies that have evolved over decades of industrial practice. Version control, automated testing, continuous integration, and code review are common pillars of disciplined development workflows.";

function repeat(p: string, times: number): string {
	return Array.from({ length: times }, () => p).join(" ");
}

export const LONG_PROMPTS: Record<string, string> = {
	// ~256 tokens
	"prefill-256": `${repeat(FILLER_PARAGRAPH, 1)} Given that context, briefly explain what makes a software engineering team effective.`,
	// ~512 tokens
	"prefill-512": `${repeat(FILLER_PARAGRAPH, 2)} Given that context, briefly explain what makes a software engineering team effective.`,
	// ~1024 tokens
	"prefill-1024": `${repeat(FILLER_PARAGRAPH, 4)} Given that context, briefly explain what makes a software engineering team effective.`,
};
