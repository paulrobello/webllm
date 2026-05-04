// chat-page.js — entry module for the dedicated chat page.
// Builds DOM, wires events, owns top-level state.
// Subsequent tasks wire in model load, conversation, metrics, persistence.

export async function runChatPage() {
  document.body.innerHTML = renderShell();
  const { populateDropdown, findModel } = await import("./chat-models.js");
  const modelSelect = document.getElementById("chat-model-select");
  populateDropdown(modelSelect);

  // `?model=<id>` URL param wins over the localStorage memory and also
  // auto-loads the model. Useful for sharing a deep-link or for
  // debugging — drive the failing case directly without dropdown clicks.
  const urlParams = new URLSearchParams(globalThis.location?.search ?? "");
  const urlModelId = urlParams.get("model");
  const autoLoadModelId = urlModelId && findModel(urlModelId) ? urlModelId : null;
  const lastModelId = localStorage.getItem("chat:lastModelId");
  const preselectId = autoLoadModelId ?? (lastModelId && findModel(lastModelId) ? lastModelId : null);
  if (preselectId) modelSelect.value = preselectId;

  const { WebLLM } = await import("./webllm-bundle.js");
  const { loadSelectedModel } = await import("./chat-models.js");

  const loadCard = document.getElementById("chat-load-card");
  const sendBtn = document.getElementById("chat-send");

  // State exposed to subsequent tasks via closure (Task 5+ will reference these).
  let engine = null;
  let loadedModel = null;
  // The engine assigns its own auto-generated handle id at load time
  // (see `WebLLM.registerModelHandle`); the catalog id from `model.id`
  // is just a display label. `createConversation` keys off the engine
  // handle id, so we keep the loaded handle here and pass `loadedHandleId`
  // (not `loadedModel.id`) to all engine APIs.
  let loadedHandleId = null;

  /**
   * Load the model with id `id`. Disposes any prior engine. Returns
   * `{ success: true }` on success, `{ success: false, error }` on
   * failure (the load card surfaces the error message either way).
   */
  async function loadModelById(id) {
    const model = findModel(id);
    if (!model) return { success: false, error: new Error(`unknown model id: ${id}`) };

    if (engine) {
      try { await engine.shutdown?.(); } catch (_e) { /* tolerate */ }
      engine = null;
      loadedModel = null;
      loadedHandleId = null;
      sendBtn.disabled = true;
    }
    loadCard.hidden = false;
    loadCard.textContent = `Loading ${model.name}…`;

    try {
      engine = await WebLLM.init({
        memoryBudget: 2_000_000_000,
        maxConversations: 4,
        worker: true,
      });
      const loadResult = await loadSelectedModel(model, engine, (pct, mb, totalMb) => {
        loadCard.textContent = `Loading ${model.name}: ${(pct * 100).toFixed(0)}% (${mb.toFixed(1)} / ${totalMb.toFixed(1)} MB)`;
      });
      loadedModel = model;
      loadedHandleId = loadResult?.handle?.id;
      if (!loadedHandleId) throw new Error("loadSelectedModel returned no handle.id");
      localStorage.setItem("chat:lastModelId", model.id);
      if (model.vramMB > 5000) {
        loadCard.innerHTML = `Loaded ${model.name} (${model.defaultQuant}, ctx ${model.contextLength}) <span class="pill amber">heavy: ${model.vramMB} MB — tight on 16 GB tier with Three.js</span>`;
      } else {
        loadCard.textContent = `Loaded ${model.name} (${model.defaultQuant}, ctx ${model.contextLength})`;
      }
      sendBtn.disabled = false;
      rebuildSettings();
      if (settingsApi) settingsApi.close();
      return { success: true };
    } catch (e) {
      console.error("[chat-page] load failed:", e);
      loadCard.textContent = `Load failed: ${e.message}`;
      engine = null;
      loadedModel = null;
      loadedHandleId = null;
      return { success: false, error: e };
    }
  }

  modelSelect.addEventListener("change", async () => {
    if (conv && conv.messages.length > 0 && !confirm("Discard current conversation?")) {
      modelSelect.value = loadedModel?.id ?? "";
      return;
    }
    const id = modelSelect.value;
    if (!id) return;
    await loadModelById(id);
  });

  const { createChatConversation, disposeChatConversation, sendTurn } =
    await import("./chat-conversation.js");
  const { renderAssistantInto } = await import("./chat-render.js");
  const { renderSettingsPanel } = await import("./chat-settings.js");
  const {
    applyContextBar, formatLastTurn, formatSession,
    newSessionTotals, addTurn, renderSparkline,
  } = await import("./chat-metrics.js");
  const { saveCurrent, clearCurrent } = await import("./chat-persistence.js");
  const { listChatModels } = await import("./chat-models.js");
  const { maybeOfferRestore } = await import("./chat-restore.js");

  const restoreCard = document.getElementById("chat-restore-card");

  const ctxBar = document.createElement("span");
  ctxBar.className = "chat-bar";
  const ctxBarInner = document.createElement("div");
  ctxBar.appendChild(ctxBarInner);
  const ctxPill = document.getElementById("chat-status-context");
  ctxPill.prepend(ctxBar, " ");

  const lastPill = document.getElementById("chat-status-last");
  const sessionPill = document.getElementById("chat-status-session");
  const sparklineEl = document.getElementById("chat-sparkline");
  const sparklineCanvas = sparklineEl.querySelector("canvas");
  const sparklineToggle = document.getElementById("chat-sparkline-toggle");
  sparklineToggle.addEventListener("click", (e) => {
    e.preventDefault();
    sparklineEl.classList.toggle("open");
    if (sparklineEl.classList.contains("open")) renderSparkline(sparklineCanvas, session.history);
  });

  let session = newSessionTotals();

  function refreshContext() {
    const used = conv ? estimateContextTokens(conv) : 0;
    const max = loadedModel?.contextLength ?? 0;
    applyContextBar(ctxBarInner, ctxPill, used, max);
  }

  function estimateContextTokens(c) {
    if (!engine || !loadedModel) return 0;
    try {
      const parts = [];
      if (c.systemPrompt) parts.push(c.systemPrompt);
      for (const m of c.messages) parts.push(m.content);
      return engine.tokenize(loadedHandleId, parts.join("\n")).length;
    } catch {
      // Fallback if tokenize isn't available (older bundle / model
      // not loaded yet): chars/4 is a reasonable order-of-magnitude
      // estimate for the bar.
      let chars = (c.systemPrompt || "").length;
      for (const m of c.messages) chars += m.content.length;
      return Math.ceil(chars / 4);
    }
  }

  const settingsPanel = document.getElementById("chat-settings-panel");
  const settingsToggle = document.getElementById("chat-settings-toggle");

  let settingsApi = null;

  function rebuildSettings() {
    if (!loadedModel) return;
    settingsApi = renderSettingsPanel(settingsPanel, loadedModel, () => {/* live config; nothing to invalidate */});
  }

  settingsToggle.addEventListener("click", () => {
    if (!settingsApi) { rebuildSettings(); return; }
    if (settingsPanel.classList.contains("open")) settingsApi.close();
    else { settingsPanel.classList.add("open"); settingsPanel.setAttribute("aria-hidden", "false"); }
  });

  const transcript = document.getElementById("chat-transcript");
  const input = document.getElementById("chat-input");
  const stopBtn = document.getElementById("chat-stop");
  const clearBtn = document.getElementById("chat-clear");
  const systemPromptEl = document.getElementById("chat-system-prompt");
  const systemApplyBtn = document.getElementById("chat-system-apply");

  // Restore + persist the system prompt independently of any saved
  // conversation. Stored separately from `chat:current:meta` (which only
  // exists while a conversation is active) so a user-typed prompt
  // survives reloads and "Discard" / "Clear conversation" actions.
  const SYSTEM_PROMPT_KEY = "chat:systemPrompt";
  const savedSystemPrompt = localStorage.getItem(SYSTEM_PROMPT_KEY);
  if (savedSystemPrompt !== null) systemPromptEl.value = savedSystemPrompt;
  systemPromptEl.addEventListener("input", () => {
    localStorage.setItem(SYSTEM_PROMPT_KEY, systemPromptEl.value);
  });
  const exportBtn = document.getElementById("chat-export");

  async function exportTranscript() {
    if (!conv) return;
    let blob;
    try { blob = await engine.exportConversation(conv.handle); } catch { /* fallback to messages-only */ }
    const meta = {
      modelId: loadedModel.id,
      systemPrompt: conv.systemPrompt,
      settings: settingsApi?.getConfig() ?? {},
      messages: conv.messages,
      blobBase64: blob ? btoa(String.fromCharCode(...blob)) : null,
      exportedAt: new Date().toISOString(),
    };
    const dl = new Blob([JSON.stringify(meta, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(dl);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${loadedModel.id}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
  exportBtn.addEventListener("click", () => void exportTranscript());

  let conv = null;
  let abortController = null;

  async function ensureConversation() {
    if (conv) return conv;
    conv = await createChatConversation(engine, loadedHandleId, loadedModel, systemPromptEl.value);
    return conv;
  }

  async function clearConversation() {
    if (!conv) return;
    await disposeChatConversation(engine, conv);
    conv = null;
    transcript.innerHTML = "";
    clearBtn.disabled = true;
    exportBtn.disabled = true;
    session = newSessionTotals();
    lastPill.textContent = "last —";
    sessionPill.textContent = "session —";
    refreshContext();
    await clearCurrent();
  }

  function appendBubble(role, text) {
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
    return div;
  }

  async function send() {
    const text = input.value.trim();
    if (!text || !engine || !loadedModel) return;
    input.value = "";
    appendBubble("user", text);
    const assistantBubble = document.createElement("div");
    assistantBubble.className = "chat-msg assistant";
    transcript.appendChild(assistantBubble);
    transcript.scrollTop = transcript.scrollHeight;

    let renderQueued = false;
    function scheduleRender(t) {
      if (renderQueued) return;
      renderQueued = true;
      requestAnimationFrame(async () => {
        renderQueued = false;
        await renderAssistantInto(assistantBubble, t);
        transcript.scrollTop = transcript.scrollHeight;
      });
    }
    await ensureConversation();

    abortController = new AbortController();
    stopBtn.hidden = false;
    sendBtn.disabled = true;

    await sendTurn({
      engine,
      conv,
      userText: text,
      config: settingsApi?.getConfig() ?? {},
      signal: abortController.signal,
      onChunk: (_t, totalText) => scheduleRender(totalText),
      onDone: (m) => {
        if (m.stopped && m.text === "") {
          assistantBubble.classList.add("stopped");
          assistantBubble.textContent = "[stopped, no reply]";
        } else {
          scheduleRender(m.text);
        }
        console.log("[chat-page] turn done", m);
        addTurn(session, m);
        lastPill.textContent = formatLastTurn(m);
        sessionPill.textContent = formatSession(session);
        refreshContext();
        if (sparklineEl.classList.contains("open")) renderSparkline(sparklineCanvas, session.history);
      },
      onError: (e) => {
        if (e?.name === "ConversationContextOverflowError" || /context.*(overflow|full)/i.test(String(e?.message))) {
          assistantBubble.classList.add("error");
          assistantBubble.innerHTML = `Context full. <button id="ovf-clear" type="button">Clear conversation</button> <button id="ovf-export-clear" type="button">Export &amp; clear</button>`;
          document.getElementById("ovf-clear").onclick = () => void clearConversation();
          document.getElementById("ovf-export-clear").onclick = async () => {
            await exportTranscript();
            await clearConversation();
          };
        } else {
          assistantBubble.classList.add("error");
          assistantBubble.textContent = `Error: ${e.message}`;
        }
      },
    });

    abortController = null;
    stopBtn.hidden = true;
    sendBtn.disabled = false;
    clearBtn.disabled = false;
    exportBtn.disabled = false;
    if (conv) {
      await saveCurrent({
        engine, conv,
        modelId: loadedModel.id,
        systemPrompt: conv.systemPrompt,
        settings: settingsApi?.getConfig() ?? {},
      });
    }
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  });
  sendBtn.addEventListener("click", () => void send());
  stopBtn.addEventListener("click", () => abortController?.abort());
  clearBtn.addEventListener("click", () => void clearConversation());
  systemApplyBtn.addEventListener("click", async () => {
    if (conv && conv.messages.length > 0 && !confirm("Discard current conversation?")) return;
    await clearConversation();
  });
  await maybeOfferRestore({
    restoreCard,
    modelSelect,
    systemPromptEl,
    transcript,
    clearBtn,
    exportBtn,
    findModel,
    listChatModels,
    loadModelById,
    getEngine: () => engine,
    getLoadedModel: () => loadedModel,
    getLoadedHandleId: () => loadedHandleId,
    appendBubble,
    renderAssistantInto,
    createChatConversation,
    setConv: (c) => { conv = c; },
    refreshContext,
  });
  console.log("[chat-page] shell mounted");

  // Auto-load priority:
  //   1. `?model=<id>` URL param wins — explicit intent, even when a
  //      saved conversation exists for a different model.
  //   2. Otherwise, if no restore card was offered (no compatible saved
  //      session) but localStorage remembers the last model, auto-load
  //      it so the user can type and send immediately. Without this the
  //      dropdown shows the right model but the engine isn't loaded and
  //      the Send button stays disabled — a stuck-state UX bug.
  //   3. If a restore card *was* offered, do nothing — user clicks
  //      Resume (which loads the model) or Discard (and picks fresh).
  const restoreOffered = !restoreCard.hidden;
  if (autoLoadModelId) {
    console.log("[chat-page] auto-loading model from URL param:", autoLoadModelId);
    await loadModelById(autoLoadModelId);
  } else if (!restoreOffered && lastModelId && findModel(lastModelId)) {
    console.log("[chat-page] auto-loading remembered last model:", lastModelId);
    await loadModelById(lastModelId);
  }
}

function renderShell() {
  return `
<div class="chat-app">
  <header class="chat-header">
    <h1>WebLLM Chat</h1>
    <select id="chat-model-select" aria-label="Model"></select>
    <button id="chat-settings-toggle" type="button">⚙ Settings</button>
  </header>
  <section class="chat-settings-panel" id="chat-settings-panel" aria-hidden="true"></section>
  <section class="chat-system">
    <label for="chat-system-prompt">System prompt</label>
    <textarea id="chat-system-prompt" placeholder="You are a helpful assistant."></textarea>
    <button id="chat-system-apply" type="button">Apply</button>
  </section>
  <div id="chat-load-card" class="chat-load-card" hidden></div>
  <div id="chat-restore-card" class="chat-restore-card" hidden></div>
  <main class="chat-transcript" id="chat-transcript" aria-live="polite"></main>
  <section class="chat-composer">
    <textarea id="chat-input" placeholder="Type a message — Enter to send, Shift+Enter for newline" rows="2"></textarea>
    <button id="chat-send" type="button" disabled>Send</button>
    <button id="chat-stop" type="button" hidden>Stop</button>
  </section>
  <section class="chat-status">
    <span class="pill" id="chat-status-context">context —</span>
    <span class="pill" id="chat-status-last">last —</span>
    <span class="pill" id="chat-status-session">session —</span>
    <span class="pill"><a href="#" id="chat-sparkline-toggle">chart ▾</a></span>
    <span class="chat-sparkline" id="chat-sparkline"><canvas width="120" height="30"></canvas></span>
  </section>
  <footer class="chat-actions">
    <button id="chat-clear" type="button" disabled>Clear conversation</button>
    <button id="chat-export" type="button" disabled>⤓ Export</button>
  </footer>
</div>`;
}
