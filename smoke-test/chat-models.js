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

/**
 * Fetch the GGUF with byte progress, then hand the buffer to the
 * engine. Mirrors the smoke page's progress pattern so we drive a
 * live load card. Returns the engine + handle ready for chat.
 *
 * @param {Object} model         A `BenchmarkModel` from `listChatModels()`.
 * @param {Object} engine        A constructed `WebLLM` instance.
 * @param {(pct:number, mb:number, totalMb:number)=>void} onProgress
 */
export async function loadSelectedModel(model, engine, onProgress) {
  const url = ggufUrl(model);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const total = Number(resp.headers.get("content-length") || 0);
  if (!total) throw new Error("missing content-length on model response");

  const buf = new Uint8Array(total);
  const reader = resp.body.getReader();
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, received);
    received += value.length;
    onProgress(received / total, received / 1e6, total / 1e6);
  }
  if (received !== total) {
    throw new Error(`short read: expected ${total} bytes, got ${received}`);
  }

  const result = await engine.loadModelFromBuffer(buf, model.id, undefined, {
    flashAttn: true, // required for createConversation
  });
  return result;
}

/**
 * Resolve a directly-fetchable GGUF URL for a registered model.
 * The smoke harness puts pre-downloaded GGUFs at `./models/<id>.gguf`;
 * fall back to that path. Custom hosting can be added later.
 */
function ggufUrl(model) {
  return `./models/${model.id}.gguf`;
}
