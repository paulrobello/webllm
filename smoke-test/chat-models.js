// chat-models.js — model catalog filter, dropdown render, load lifecycle.
import { BENCHMARK_MODELS } from "./webllm-models.js";

/**
 * Chat-eligible models from the registered fleet.
 * Excludes encoder + dedicated-embedder entries; chat models flagged
 * `embeddingCapable: true` (bucket D) remain in scope as chat models.
 */
export function listChatModels() {
	return BENCHMARK_MODELS.filter(
		(m) => !m.capabilities.embedding && m.architecture !== "bert",
	)
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
 * Stream the GGUF directly into the WASM heap with byte progress.
 * Goes through `engine.loadModelFromUrl`, which avoids the V8 ~4 GiB
 * ArrayBuffer cap that would trip on 7B+ Q4 / 8B IQ3_M models.
 * Returns the engine handle + inference + metadata.
 *
 * @param {Object} model A `BenchmarkModel` from `listChatModels()`.
 * @param {Object} engine A constructed `WebLLM` instance.
 * @param {(pct:number, mb:number, totalMb:number)=>void} onProgress
 */
export async function loadSelectedModel(model, engine, onProgress) {
	const url = ggufUrl(model);
	return await engine.loadModelFromUrl(
		url,
		model.id,
		undefined,
		{
			priority: 0,
			flashAttn: true, // FA required for createConversation
			// Clamp the KV cache to the chat-page-registered contextLength.
			// Without this, the engine sizes the cache to the GGUF max (which
			// for Qwen3 GGUFs is ~40960 with rope_scaling) — a single per-
			// layer K tensor at head_dim=128, n_kv_heads=8, F16 then becomes
			// 80 MiB and trips the `ggml-alloc.c:82` "not enough space in the
			// buffer" assertion against `maxStorageBufferBindingSize`.
			contextLength: model.contextLength,
		},
		(received, total) => {
			onProgress(received / total, received / 1e6, total / 1e6);
		},
	);
}

/**
 * Resolve a directly-fetchable GGUF URL for a registered model.
 * The smoke harness puts pre-downloaded GGUFs at `./models/<id>.gguf`;
 * fall back to that path. Custom hosting can be added later.
 */
function ggufUrl(model) {
	return `./models/${model.id}.gguf`;
}
