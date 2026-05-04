export interface RestoreContext {
  restoreCard: HTMLElement;
  modelSelect: HTMLSelectElement;
  systemPromptEl: HTMLTextAreaElement;
  transcript: HTMLElement;
  clearBtn: HTMLButtonElement;
  exportBtn: HTMLButtonElement;
  findModel: (id: string) => { id: string; name: string } | undefined;
  listChatModels: () => Array<{ id: string }>;
  loadModelById: (id: string) => Promise<{ success: boolean; error?: Error }>;
  getEngine: () => unknown;
  getLoadedModel: () => unknown;
  appendBubble: (role: string, text: string) => HTMLElement;
  renderAssistantInto: (el: HTMLElement, text: string) => Promise<void>;
  createChatConversation: (engine: unknown, model: unknown, systemPrompt: string) => Promise<unknown>;
  setConv: (conv: unknown) => void;
  refreshContext: () => void;
}

export function maybeOfferRestore(ctx: RestoreContext): Promise<void>;
