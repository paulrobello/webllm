// Ambient declaration for the smoke-test JS module so TS-checked tests can
// import it. Signatures use minimal typing so the .js stays the source of truth.

export function splitThinking(raw: string): { thinking: string; answer: string };
export function renderMarkdown(text: string): string;
export function renderAssistantInto(el: unknown, raw: string): Promise<void>;
