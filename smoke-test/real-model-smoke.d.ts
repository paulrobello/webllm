// Ambient declaration for the smoke-test JS module so TS-checked tests can
// import it. Signatures use `unknown` so the .js stays the source of truth.

export function shouldAutoInsertBos(tokenizerConfig: unknown): boolean;
export function shouldRunSmokeDiagnostics(params: URLSearchParams): boolean;
export function getSmokePageCopy(debugMode: boolean): {
	title: string;
	subtitle: string;
};
export function getSmokePageShellMarkup(): string;
export function getSmokeChatOptions(...args: unknown[]): unknown;
export function getThinkingModeFromParams(params: URLSearchParams): unknown;
export function modelSupportsThinking(parsed: unknown): boolean;
export function getSmokeSamplingConfig(...args: unknown[]): unknown;
export function createSmokeSamplerFactory(opts: unknown): (
	chatTemplate: unknown,
	chatOptions: unknown,
) => unknown;
export function getSmokeSamplingOverridesFromParams(
	params: URLSearchParams,
): unknown;
export function sanitizeDisplayText(text: string): string;
export function buildSmokePrompt(
	prompt: string,
	chatOptions: unknown,
	encodeChat: (
		messages: unknown,
		tokenizer: unknown,
		chatOptions: unknown,
	) => number[],
	tokenizer: unknown,
): { mode: string; tokens: number[] };
export function findSingleTokenProbe(
	tokenizer: unknown,
	candidates: unknown,
): unknown;
export function createSmokeCompletionRunner(opts: unknown): unknown;
export function createPrefillComparisonRunner(opts: unknown): (
	...args: unknown[]
) => Promise<unknown>;
