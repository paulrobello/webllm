/**
 * Pinned text fixtures for §D encoder perf cycle. Single source of
 * truth — eval/embed-perf.ts and tests/encoder-cosine-parity.test.ts
 * both import from here.
 *
 * `short`     — minimal real text; exercises the per-call fixed-overhead path.
 * `long`      — ~200 token English paragraph; exercises the matmul/encode path.
 * `batchMixed` — 64 entries (32 short + 32 long); exercises batch throughput.
 *
 * The smoke page (smoke-test/real-model-page.js) intentionally inlines
 * the same text values rather than importing this Bun-only file. Keep
 * the inlined copies in sync with the values defined here.
 */

const LONG_TEXT =
	"Compilers translate human-readable source code into instructions a " +
	"computer can execute. The translation usually runs in several stages: " +
	"a lexer breaks the input into tokens, a parser assembles those tokens " +
	"into a syntax tree, a semantic analyser checks the tree for meaning, " +
	"and a code generator emits machine code or bytecode for some target " +
	"architecture. Modern compilers add an optimiser between the analyser " +
	"and the generator that reorders, inlines, and rewrites the program in " +
	"ways that preserve its observable behaviour while reducing its runtime " +
	"or code size.";

function buildBatchMixed(): string[] {
	const arr: string[] = [];
	for (let i = 0; i < 32; i++) arr.push("happy");
	for (let i = 0; i < 32; i++) arr.push(LONG_TEXT);
	return arr;
}

export const EMBED_PROMPTS = {
	short: "happy",
	long: LONG_TEXT,
	batchMixed: buildBatchMixed(),
} as const;

export type EmbedFixtureKey = keyof typeof EMBED_PROMPTS;
