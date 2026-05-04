// Ambient declaration for the smoke-test JS module so TS-checked tests can
// import it. Signatures use minimal typing so the .js stays the source of truth.

export interface TurnMetrics {
  ttftMs: number;
  decodeMs: number;
  totalMs: number;
  outputTokens: number;
  finishReason: string;
  text: string;
  stopped: boolean;
}

export interface SessionTotals {
  turns: number;
  totalOutputTokens: number;
  totalDecodeMs: number;
  history: number[];
}

export type ContextBarState = "neutral" | "amber" | "red";

export function contextBarState(used: number, max: number): ContextBarState;
export function formatContext(used: number, max: number): string;
export function formatLastTurn(m: TurnMetrics): string;
export function formatSession(s: SessionTotals): string;
export function newSessionTotals(): SessionTotals;
export function addTurn(session: SessionTotals, m: TurnMetrics): void;
export function applyContextBar(barInner: HTMLElement, labelEl: HTMLElement, used: number, max: number): void;
export function renderSparkline(canvas: HTMLCanvasElement, history: number[]): void;
