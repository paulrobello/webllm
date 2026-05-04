// chat-settings.js — settings panel state + family-aware sampler defaults.

const ENGINE_FALLBACKS = { temperature: 1.0, topK: 0, topP: 1.0, repetitionPenalty: 1.0 };

const QWEN_THINKING = { temperature: 0.6, topK: 20, topP: 0.95, repetitionPenalty: 1.05 };
const QWEN_NON_THINKING = { temperature: 0.7, topK: 20, topP: 0.8, repetitionPenalty: 1.1 };
const PHI3 = { temperature: 0, topK: 0, topP: 1, repetitionPenalty: 1.1 };
const MISTRAL = { temperature: 0.7, topK: 0, topP: 0.95, repetitionPenalty: 1.0 };

/**
 * Compute the default settings for a given model. Mirrors the engine's
 * `resolveSamplingParams` profile selection so the UI shows the same
 * numbers the engine will use.
 */
export function defaultSettings(model, enableThinking) {
  const isQwenChatml = model.architecture === "qwen2" || model.architecture === "qwen3";
  if (isQwenChatml) {
    return enableThinking === false ? { ...QWEN_NON_THINKING } : { ...QWEN_THINKING };
  }
  if (model.architecture === "phi3") {
    return { ...PHI3 };
  }
  // Mistral-Instruct family — registered with `architecture: "llama"` but
  // its chat template is `[INST]…[/INST]` without `<<SYS>>`. Distinguish
  // by family name (registry uses "Mistral" / "Mistral Nemo" etc.).
  if (typeof model.family === "string" && /^mistral/i.test(model.family)) {
    return { ...MISTRAL };
  }
  return { ...ENGINE_FALLBACKS };
}

export function isThinkingCapable(model) {
  return model.architecture === "qwen3";
}

/**
 * Render the settings panel into `panelEl`. Returns a `getConfig()`
 * accessor that reads the current values into a `CompletionConfig`-shaped
 * object. Calls `onChange()` whenever a control changes.
 */
export function renderSettingsPanel(panelEl, model, onChange) {
  const defaults = defaultSettings(model, true);
  const thinkingCapable = isThinkingCapable(model);
  panelEl.innerHTML = `
    <label for="chat-temperature">Temperature</label>
    <input id="chat-temperature" type="number" min="0" max="2" step="0.05" value="${defaults.temperature}">
    <label for="chat-topk">Top-K</label>
    <input id="chat-topk" type="number" min="0" step="1" value="${defaults.topK}">
    <label for="chat-topp">Top-P</label>
    <input id="chat-topp" type="number" min="0" max="1" step="0.05" value="${defaults.topP}">
    <label for="chat-maxtok">Max output tokens</label>
    <input id="chat-maxtok" type="number" min="1" step="1" value="512">
    <label for="chat-seed">Seed (blank = random)</label>
    <input id="chat-seed" type="number" step="1" value="">
    ${thinkingCapable ? `
      <label for="chat-thinking">Thinking mode (Qwen3)</label>
      <input id="chat-thinking" type="checkbox" checked>
    ` : ""}
    <button id="chat-settings-reset" type="button" style="grid-column: 1 / -1; justify-self: start;">Reset to defaults</button>
  `;
  panelEl.classList.add("open");
  panelEl.setAttribute("aria-hidden", "false");

  const get = (id) => /** @type {HTMLInputElement} */ (panelEl.querySelector(id));
  const numOrUndef = (el) => (el.value === "" ? undefined : Number(el.value));

  function reset() {
    const enableThinking = thinkingCapable ? get("#chat-thinking").checked : true;
    const d = defaultSettings(model, enableThinking);
    get("#chat-temperature").value = String(d.temperature);
    get("#chat-topk").value = String(d.topK);
    get("#chat-topp").value = String(d.topP);
    get("#chat-maxtok").value = "512";
    get("#chat-seed").value = "";
  }
  panelEl.querySelector("#chat-settings-reset").addEventListener("click", () => { reset(); onChange?.(); });
  if (thinkingCapable) {
    get("#chat-thinking").addEventListener("change", () => { reset(); onChange?.(); });
  }
  for (const inputEl of panelEl.querySelectorAll("input")) {
    inputEl.addEventListener("input", () => onChange?.());
  }

  return {
    getConfig() {
      return {
        temperature: numOrUndef(get("#chat-temperature")),
        topK: numOrUndef(get("#chat-topk")),
        topP: numOrUndef(get("#chat-topp")),
        maxTokens: numOrUndef(get("#chat-maxtok")) ?? 512,
        seed: numOrUndef(get("#chat-seed")),
        enableThinking: thinkingCapable ? get("#chat-thinking").checked : undefined,
      };
    },
    close() {
      panelEl.classList.remove("open");
      panelEl.setAttribute("aria-hidden", "true");
    },
  };
}
