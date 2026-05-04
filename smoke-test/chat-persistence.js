// chat-persistence.js — IndexedDB blob + localStorage metadata sidecar
// for the page's single auto-saved conversation.

export const CHAT_META_KEY = "chat:current:meta";
export const CHAT_BLOB_KEY = "chat:current";
export const CHAT_DB_NAME = "webllm-chat";

const SCHEMA_VERSION = 1;

let _store = null;
let _IndexedDBConversationStore = null;
async function store() {
  if (!_store) {
    if (!_IndexedDBConversationStore) {
      const mod = await import("./webllm-persistence.js");
      _IndexedDBConversationStore = mod.IndexedDBConversationStore;
    }
    _store = new _IndexedDBConversationStore(CHAT_DB_NAME);
  }
  return _store;
}

export function buildMetadata({ modelId, systemPrompt, settings, messages }) {
  return {
    schemaVersion: SCHEMA_VERSION,
    modelId,
    systemPrompt,
    settings,
    messages,
    savedAtMs: Date.now(),
  };
}

export function isCompatibleMeta(meta, knownModelIds) {
  if (!meta || meta.schemaVersion !== SCHEMA_VERSION) return false;
  if (!meta.modelId || !knownModelIds.has(meta.modelId)) return false;
  return true;
}

/** Persist the engine blob + sidecar metadata in one step. */
export async function saveCurrent({ engine, conv, modelId, systemPrompt, settings }) {
  const meta = buildMetadata({ modelId, systemPrompt, settings, messages: conv.messages });
  localStorage.setItem(CHAT_META_KEY, JSON.stringify(meta));
  try {
    const blob = await engine.exportConversation(conv.handle);
    const s = await store();
    await s.put(CHAT_BLOB_KEY, blob);
  } catch (e) {
    // Metadata-only save is acceptable — message history alone is
    // useful even without KV reuse on restore.
    console.warn("[chat-persistence] blob save failed:", e?.message ?? e);
  }
}

/** Read the sidecar metadata, or null if absent / unparsable. */
export function loadMetadata() {
  const raw = localStorage.getItem(CHAT_META_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

/** Read the engine blob, or undefined if absent. */
export async function loadBlob() {
  const s = await store();
  return await s.get(CHAT_BLOB_KEY);
}

/** Clear both sidecar and blob. */
export async function clearCurrent() {
  localStorage.removeItem(CHAT_META_KEY);
  try {
    const s = await store();
    await s.delete(CHAT_BLOB_KEY);
  } catch { /* tolerate */ }
}

export function relativeTime(savedAtMs) {
  const dt = (Date.now() - savedAtMs) / 1000;
  if (dt < 60) return `${Math.floor(dt)}s ago`;
  if (dt < 3600) return `${Math.floor(dt / 60)}m ago`;
  if (dt < 86400) return `${Math.floor(dt / 3600)}h ago`;
  return `${Math.floor(dt / 86400)}d ago`;
}
