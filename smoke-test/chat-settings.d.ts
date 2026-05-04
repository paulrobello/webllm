// Ambient declaration for the smoke-test JS module so TS-checked tests can
// import it. Signatures use minimal typing so the .js stays the source of truth.

export interface ChatSettingsModel {
  architecture: string;
  contextLength: number;
  [key: string]: unknown;
}

export interface ChatSamplerDefaults {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
}

export interface ChatSettingsConfig {
  temperature?: number;
  topK?: number;
  topP?: number;
  maxTokens?: number;
  seed?: number;
  enableThinking?: boolean;
}

export interface ChatSettingsApi {
  getConfig(): ChatSettingsConfig;
  close(): void;
}

export function defaultSettings(model: ChatSettingsModel, enableThinking: boolean): ChatSamplerDefaults;
export function isThinkingCapable(model: ChatSettingsModel): boolean;
export function renderSettingsPanel(
  panelEl: HTMLElement,
  model: ChatSettingsModel,
  onChange?: () => void,
): ChatSettingsApi;
