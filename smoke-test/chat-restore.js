// chat-restore.js — restore-card UI + Resume/Discard flow.
// Owns the saved-conversation restore experience on page load:
//   1. Read saved metadata.
//   2. If compatible, render the restore card.
//   3. On Resume: load the model (via injected loadModelById callback),
//      reinstate KV via importConversation (or fall back to a fresh
//      conversation with replayed messages), and repopulate the transcript.
//   4. On Discard: drop both stores.

import {
  clearCurrent,
  isCompatibleMeta,
  loadBlob,
  loadMetadata,
  relativeTime,
} from "./chat-persistence.js";

/**
 * @typedef {Object} RestoreContext
 * @property {HTMLElement} restoreCard
 * @property {HTMLSelectElement} modelSelect
 * @property {HTMLTextAreaElement} systemPromptEl
 * @property {HTMLElement} transcript
 * @property {HTMLButtonElement} clearBtn
 * @property {HTMLButtonElement} exportBtn
 * @property {() => { id: string, name: string } | undefined} findModel
 * @property {() => Array<{ id: string }>} listChatModels
 * @property {(id: string) => Promise<{ success: boolean, error?: Error }>} loadModelById
 * @property {() => unknown} getEngine
 * @property {() => unknown} getLoadedModel
 * @property {(role: string, text: string) => HTMLElement} appendBubble
 * @property {(el: HTMLElement, text: string) => Promise<void>} renderAssistantInto
 * @property {(engine: unknown, model: unknown, systemPrompt: string) => Promise<unknown>} createChatConversation
 * @property {(conv: unknown) => void} setConv
 * @property {() => void} refreshContext
 */

/**
 * Check for a saved compatible conversation; if present, render the
 * restore card with Resume / Discard. Wires both buttons.
 *
 * @param {RestoreContext} ctx
 */
export async function maybeOfferRestore(ctx) {
  const meta = loadMetadata();
  if (!meta) return;
  const known = new Set(ctx.listChatModels().map((m) => m.id));
  if (!isCompatibleMeta(meta, known)) {
    await clearCurrent();
    return;
  }
  const model = ctx.findModel(meta.modelId);
  if (!model) {
    await clearCurrent();
    return;
  }
  ctx.restoreCard.hidden = false;
  ctx.restoreCard.innerHTML = `
    Resume conversation with <strong>${escapeHtml(model.name)}</strong>
    (${meta.messages.length} turns, last active ${relativeTime(meta.savedAtMs)})?
    <button id="chat-restore-yes" type="button">Resume</button>
    <button id="chat-restore-no"  type="button">Discard</button>
  `;
  document.getElementById("chat-restore-no").onclick = async () => {
    await clearCurrent();
    ctx.restoreCard.hidden = true;
  };
  document.getElementById("chat-restore-yes").onclick = async () => {
    ctx.restoreCard.hidden = true;
    ctx.modelSelect.value = meta.modelId;
    const result = await ctx.loadModelById(meta.modelId);
    if (!result.success) {
      // Load card already shows the error from loadModelById.
      // Leave the saved data in place so the user can retry next reload.
      return;
    }
    const engine = ctx.getEngine();
    const loadedModel = ctx.getLoadedModel();
    let conv;
    const blob = await loadBlob();
    if (blob) {
      try {
        const handle = await engine.importConversation(meta.modelId, blob);
        conv = { handle, modelId: meta.modelId, systemPrompt: meta.systemPrompt, messages: meta.messages.slice() };
      } catch (e) {
        console.warn("[chat-restore] importConversation failed; falling back to metadata-only restore:", e);
        conv = await ctx.createChatConversation(engine, loadedModel, meta.systemPrompt);
        conv.messages = meta.messages.slice();
      }
    } else {
      conv = await ctx.createChatConversation(engine, loadedModel, meta.systemPrompt);
      conv.messages = meta.messages.slice();
    }
    ctx.setConv(conv);
    ctx.systemPromptEl.value = meta.systemPrompt;
    for (const msg of meta.messages) {
      if (msg.role === "user") ctx.appendBubble("user", msg.content);
      else if (msg.role === "assistant") {
        const div = document.createElement("div");
        div.className = "chat-msg assistant";
        ctx.transcript.appendChild(div);
        await ctx.renderAssistantInto(div, msg.content);
      }
    }
    ctx.clearBtn.disabled = false;
    ctx.exportBtn.disabled = false;
    ctx.refreshContext();
  };
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
