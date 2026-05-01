// Probe 9d worker — minimal Worker-resident inference path.
//
// Runs on `qwen3-0.6b-q4f16` (610 MB GGUF — fits in a single
// ArrayBuffer; uses the public `WebLLM.loadModelFromBuffer` factory
// rather than re-implementing the smoke page's heap-streaming
// loader). Spike scope: just enough to answer "does the per-call
// ~42 ms decode hitch from probes 9b/9c survive a thread move?"
//
// Protocol (postMessage):
//   { type: "init", modelId, modelUrl, contextLength, wasmUrl }
//     → posts { type: "init-done" }  or { type: "error", message }
//   { type: "chat", id, prompt, maxTokens }
//     → posts { type: "chat-done", id, wallMs, prefillMs, genTokens, output }
//
// Streaming chunks are NOT proxied to main thread — the probe only
// measures wall time, not per-chunk latency. Less message traffic
// means cleaner main-thread frame timing.

import { WebLLM } from "./webllm-bundle.js";

let engine = null;
let handle = null;

self.addEventListener("message", async (e) => {
	const msg = e.data;
	try {
		if (msg.type === "init") {
			const resp = await fetch(msg.modelUrl);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const buf = await resp.arrayBuffer();
			const result = await WebLLM.loadModelFromBuffer(
				buf,
				msg.modelId,
				{ memoryBudget: 8e9 },
				msg.wasmUrl,
				{ priority: 0, contextLength: msg.contextLength },
			);
			engine = result.engine;
			handle = result.handle;
			self.postMessage({ type: "init-done" });
			return;
		}
		if (msg.type === "chat") {
			if (!engine || !handle) {
				self.postMessage({
					type: "error",
					id: msg.id,
					message: "engine not initialized",
				});
				return;
			}
			const tStart = performance.now();
			let outputText = "";
			let genTokens = 0;
			let prefillMs = 0;
			for await (const chunk of engine.chatCompletion(
				handle.id,
				[{ role: "user", content: msg.prompt }],
				{ maxTokens: msg.maxTokens ?? 16 },
			)) {
				if (chunk.done) {
					prefillMs = chunk.stats?.timeToFirstTokenMs ?? 0;
					continue;
				}
				if (chunk.text) outputText += chunk.text;
				if (chunk.tokenId !== undefined) genTokens++;
			}
			const wallMs = performance.now() - tStart;
			self.postMessage({
				type: "chat-done",
				id: msg.id,
				wallMs,
				prefillMs,
				genTokens,
				output: outputText,
			});
			return;
		}
	} catch (err) {
		self.postMessage({
			type: "error",
			id: msg?.id,
			message: err instanceof Error ? err.message : String(err),
			stack: err instanceof Error ? err.stack : undefined,
		});
	}
});
