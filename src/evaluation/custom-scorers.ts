/**
 * Named registry for custom scorers. Scoring functions don't survive a
 * JSON serialisation boundary (e.g. Bun → live-server → browser bench
 * mode), so tasks carry `{ type: "custom", name: "..." }` and the scorer
 * looks the function up here.
 *
 * Both sides (the Bun harness and the browser bench page) must register
 * the same scorer functions under the same names before running tasks.
 * The default set used by the shipped `eval/tasks/*` lives in
 * `eval/tasks/scorer-registrations.ts`.
 */

export type CustomScorer = (output: string, expected: string) => number;

const registry = new Map<string, CustomScorer>();

export function registerCustomScorer(name: string, fn: CustomScorer): void {
	registry.set(name, fn);
}

export function getCustomScorer(name: string): CustomScorer | undefined {
	return registry.get(name);
}

export function listCustomScorers(): string[] {
	return Array.from(registry.keys());
}

export function hasCustomScorer(name: string): boolean {
	return registry.has(name);
}
