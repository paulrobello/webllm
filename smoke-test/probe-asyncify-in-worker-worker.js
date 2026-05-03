// Probe: confirm ASYNCIFY-driven graphCompute survives in a DedicatedWorker
// against the registered-model engine path (not the loadModelFromBuffer
// factory). Fires WebLLM.init() inside the worker, registers qwen3-0.6b
// via loadModelFromBuffer, runs a 16-token chatCompletion, posts result.
//
// The worker also emits coarse timeline events back to the page so a
// rAF-driven frame-probe (running on the page's main thread) can segment
// its samples by phase (prefill / decode / post). This is the validation
// surface for the chunk-coalescing change in src/core/webllm-worker-host.ts —
// median decode-window rAF delta is the gate that matters.

import { WebLLM } from "./webllm-bundle.js";

self.addEventListener("message", async (e) => {
	if (e.data?.type !== "run") return;
	try {
		const t0 = performance.now();
		const engine = await WebLLM.init({ memoryBudget: 8e9 });
		const resp = await fetch("./models/qwen3-0.6b-q4f16.gguf");
		if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
		const buf = await resp.arrayBuffer();
		const { handle } = await engine.loadModelFromBuffer(
			buf,
			"qwen3-0.6b-q4f16",
			"./webllm-wasm.js",
			{ priority: 0, contextLength: 4096 },
		);
		const tInit = performance.now() - t0;

		const tStart = performance.now();
		self.postMessage({ type: "phase", phase: "chat-start", t: tStart });
		let text = "";
		let nTokens = 0;
		let firstTokenAt = 0;
		const tokenIds = [];
		for await (const chunk of engine.chatCompletion(
			handle.id,
			[{ role: "user", content: "Tell one short joke." }],
			{ maxTokens: 16 },
		)) {
			if (chunk.text) text += chunk.text;
			if (chunk.tokenId !== undefined) {
				nTokens++;
				if (firstTokenAt === 0) {
					firstTokenAt = performance.now();
					self.postMessage({
						type: "phase",
						phase: "first-token",
						t: firstTokenAt,
					});
				}
				if (tokenIds.length < 16) tokenIds.push(chunk.tokenId);
			}
		}
		const tEnd = performance.now();
		const tGen = tEnd - tStart;
		self.postMessage({ type: "phase", phase: "chat-end", t: tEnd });

		self.postMessage({
			type: "done",
			tInitMs: tInit,
			tGenMs: tGen,
			nTokens,
			text,
			tokenIds,
			timeline: {
				tStart,
				firstTokenAt,
				tEnd,
				prefillMs: firstTokenAt > 0 ? firstTokenAt - tStart : 0,
			},
		});
	} catch (err) {
		self.postMessage({
			type: "error",
			message: err instanceof Error ? err.message : String(err),
			stack: err instanceof Error ? err.stack : undefined,
		});
	}
});
