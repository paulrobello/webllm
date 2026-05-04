// chat-models.js — model catalog filter, dropdown render, load lifecycle.
import { BENCHMARK_MODELS } from "./webllm-models.js";

/**
 * Chat-eligible models from the registered fleet.
 * Excludes encoder + dedicated-embedder entries; chat models flagged
 * `embeddingCapable: true` (bucket D) remain in scope as chat models.
 */
export function listChatModels() {
  return BENCHMARK_MODELS
    .filter((m) => !m.capabilities.embedding && m.architecture !== "bert")
    .slice()
    .sort((a, b) => a.family.localeCompare(b.family) || a.paramsB - b.paramsB);
}

/**
 * Populate a `<select>` element grouped by family, sorted by paramsB
 * within each family. Each `<option>`'s value is the model id.
 */
export function populateDropdown(selectEl) {
  selectEl.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "— select a model —";
  selectEl.appendChild(placeholder);

  const byFamily = new Map();
  for (const m of listChatModels()) {
    if (!byFamily.has(m.family)) byFamily.set(m.family, []);
    byFamily.get(m.family).push(m);
  }
  for (const [family, models] of byFamily) {
    const group = document.createElement("optgroup");
    group.label = family;
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} · ${m.paramsB}B · ${m.defaultQuant} · ~${m.vramMB} MB`;
      group.appendChild(opt);
    }
    selectEl.appendChild(group);
  }
}

/** Lookup helper. */
export function findModel(id) {
  return BENCHMARK_MODELS.find((m) => m.id === id);
}
