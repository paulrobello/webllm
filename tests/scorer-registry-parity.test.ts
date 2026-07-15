import { expect, test } from "bun:test";

/**
 * Parity test for the Bun-side and browser-side custom-scorer registries.
 *
 * CLAUDE.md mandates that the shipped custom scorers be mirrored across
 * `eval/tasks/scorer-registrations.ts` (Bun) and
 * `smoke-test/scorer-registrations.js` (browser). The existing
 * `tests/custom-scorers.test.ts` only asserts the Bun side registers the
 * expected names — nothing compares the browser twin, so a scorer edited on
 * one side silently skews browser-eval accuracy. This test closes that gap.
 *
 * We read both files as TEXT (via `Bun.file(...).text()`) rather than importing
 * them: the browser file imports `registerCustomScorer` from `./webllm-bundle.js`,
 * a build artifact that does not exist under Bun, so importing the browser file
 * would throw at resolution time.
 */
const BUN_PATH = "eval/tasks/scorer-registrations.ts";
const BROWSER_PATH = "smoke-test/scorer-registrations.js";

const NAME_RE = /registerCustomScorer\(\s*["']([^"']+)["']/g;

function extractNames(text: string): Set<string> {
	const names = new Set<string>();
	for (const m of text.matchAll(NAME_RE)) {
		names.add(m[1]);
	}
	return names;
}

/**
 * Extract the body text for each registered scorer: the span from the end of
 * `registerCustomScorer("name",` up to and including the first `});` that
 * closes the arrow-fn block and the `registerCustomScorer(...)` paren. Stopping
 * at `});` (rather than at the start of the next registration) keeps trailing
 * section-header comments out of the comparison — those comments differ
 * cosmetically between the twins (dash counts, occasional rewording) and would
 * otherwise produce false-positive drift. The scorer bodies themselves never
 * contain a top-level `});` (nested arrow callbacks use expression bodies), so
 * the terminator is unambiguous for the shipped set.
 */
function extractBodies(text: string): Map<string, string> {
	const bodies = new Map<string, string>();
	const re = /registerCustomScorer\(\s*["']([^"']+)["']\s*,/g;
	const matches = [...text.matchAll(re)];
	for (const m of matches) {
		const name = m[1];
		const start = (m.index ?? 0) + m[0].length;
		const endRel = text.indexOf("});", start);
		const end = endRel === -1 ? text.length : endRel + 3;
		bodies.set(name, text.slice(start, end));
	}
	return bodies;
}

/**
 * Normalize scorer body text for cross-file comparison. Conservative — only
 * syntactic / formatter differences that preserve semantics are smoothed over:
 *   - strip TS parameter type annotations (`(h: unknown)` → `(h)`) so the TS
 *     and JS twins can align
 *   - drop trailing commas inside call/arrow parens (Bun's formatter inserts
 *     them; the browser file does not)
 *   - collapse whitespace runs to a single space
 * Any real logic drift (different conditions, return values, control flow)
 * survives normalization and will fail the body-parity assertion.
 */
function normalizeBody(s: string): string {
	return s
		.replace(/:\s*[A-Za-z_]\w*\s*(?=[,)])/g, "")
		.replace(/,\s*\)/g, ")")
		.replace(/\s+/g, " ")
		.trim();
}

test("scorer registry parity: Bun and browser twins register identical name-sets", async () => {
	const bunText = await Bun.file(BUN_PATH).text();
	const browserText = await Bun.file(BROWSER_PATH).text();

	const bunNames = extractNames(bunText);
	const browserNames = extractNames(browserText);

	// Count is derived from the Bun side — adding a scorer on both sides must
	// pass without touching this test. The `> 0` sanity check guards against
	// a regex change silently zeroing the set and making the equality trivially
	// true.
	expect(bunNames.size).toBeGreaterThan(0);
	expect(browserNames.size).toBe(bunNames.size);
	for (const name of bunNames) {
		expect(browserNames.has(name), `browser registry missing "${name}"`).toBe(
			true,
		);
	}
});

test("scorer registry parity: per-scorer body text matches after normalization", async () => {
	const bunText = await Bun.file(BUN_PATH).text();
	const browserText = await Bun.file(BROWSER_PATH).text();

	const bunBodies = extractBodies(bunText);
	const browserBodies = extractBodies(browserText);

	// Known TS/JS idiomatic drift that survives normalization. These are
	// semantically-equivalent rewrites the text parity check cannot smooth over.
	// Listed explicitly so the test fails loudly if the set grows — adding an
	// entry must be a deliberate, documented choice, not silent drift.
	const KNOWN_BODY_DRIFT = new Set<string>([
		// Browser uses optional chaining (`lines[i]?.includes(target)`); Bun
		// uses logical-AND (`lines[i] && lines[i].includes(target)`).
		// Semantically equivalent for the array-bounds case this scorer
		// exercises.
		"emb-012-light-sense-disambiguation",
	]);

	const mismatches: string[] = [];
	for (const name of bunBodies.keys()) {
		if (KNOWN_BODY_DRIFT.has(name)) continue;
		const bun = normalizeBody(bunBodies.get(name) ?? "");
		const browser = normalizeBody(browserBodies.get(name) ?? "");
		if (bun !== browser) mismatches.push(name);
	}

	expect(
		mismatches,
		`scorer body drift after normalization: ${mismatches.join(", ")}`,
	).toEqual([]);
});
