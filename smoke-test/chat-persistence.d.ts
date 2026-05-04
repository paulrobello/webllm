// Ambient declaration for the smoke-test JS module so TS-checked tests can
// import it. Signatures use minimal typing so the .js stays the source of truth.

export const CHAT_META_KEY: string;
export const CHAT_BLOB_KEY: string;
export const CHAT_DB_NAME: string;

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatPersistedMetadata {
  schemaVersion: 1;
  modelId: string;
  systemPrompt: string;
  settings: Record<string, unknown>;
  messages: ChatMessage[];
  savedAtMs: number;
}

export interface BuildMetadataInput {
  modelId: string;
  systemPrompt: string;
  settings: Record<string, unknown>;
  messages: ChatMessage[];
}

export function buildMetadata(input: BuildMetadataInput): ChatPersistedMetadata;
export function isCompatibleMeta(
  meta: ChatPersistedMetadata | null | undefined,
  knownModelIds: Set<string>,
): boolean;

export interface SaveCurrentInput {
  engine: unknown;
  conv: { handle: unknown; messages: ChatMessage[] };
  modelId: string;
  systemPrompt: string;
  settings: Record<string, unknown>;
}
export function saveCurrent(input: SaveCurrentInput): Promise<void>;
export function loadMetadata(): ChatPersistedMetadata | null;
export function loadBlob(): Promise<Uint8Array | undefined>;
export function clearCurrent(): Promise<void>;
export function relativeTime(savedAtMs: number): string;
