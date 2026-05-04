// chat-page.js — entry module for the dedicated chat page.
// Builds DOM, wires events, owns top-level state.
// Subsequent tasks wire in model load, conversation, metrics, persistence.

export async function runChatPage() {
  document.body.innerHTML = renderShell();
  const { populateDropdown, findModel } = await import("./chat-models.js");
  const modelSelect = document.getElementById("chat-model-select");
  populateDropdown(modelSelect);

  const lastModelId = localStorage.getItem("chat:lastModelId");
  if (lastModelId && findModel(lastModelId)) modelSelect.value = lastModelId;

  const { WebLLM } = await import("./webllm-bundle.js");
  const { loadSelectedModel } = await import("./chat-models.js");

  const loadCard = document.getElementById("chat-load-card");
  const sendBtn = document.getElementById("chat-send");

  // State exposed to subsequent tasks via closure (Task 5+ will reference these).
  let engine = null;
  let loadedModel = null;

  modelSelect.addEventListener("change", async () => {
    const id = modelSelect.value;
    if (!id) return;
    const model = findModel(id);
    if (!model) return;

    if (engine) {
      try { await engine.shutdown?.(); } catch (_e) { /* tolerate */ }
      engine = null;
      loadedModel = null;
      sendBtn.disabled = true;
    }
    loadCard.hidden = false;
    loadCard.textContent = `Loading ${model.name}…`;

    try {
      engine = new WebLLM({ baseAssetUrl: "./" });
      await engine.init({});
      await loadSelectedModel(model, engine, (pct, mb, totalMb) => {
        loadCard.textContent = `Loading ${model.name}: ${(pct * 100).toFixed(0)}% (${mb.toFixed(1)} / ${totalMb.toFixed(1)} MB)`;
      });
      loadedModel = model;
      localStorage.setItem("chat:lastModelId", model.id);
      loadCard.textContent = `Loaded ${model.name} (${model.defaultQuant}, ctx ${model.contextLength})`;
      sendBtn.disabled = false;
    } catch (e) {
      loadCard.textContent = `Load failed: ${e.message}`;
      engine = null;
      loadedModel = null;
    }
  });
  console.log("[chat-page] shell mounted");
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
