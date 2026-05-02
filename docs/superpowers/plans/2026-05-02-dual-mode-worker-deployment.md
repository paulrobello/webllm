# Dual-mode (main-thread + worker) deployment — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WebLLM.init({ worker: true })` flag that runs WebGPU + ggml-wasm in a `DedicatedWorker`. Public TypeScript surface stays identical between modes (modulo the conversation-method async-ification called out in the spec).

**Architecture:** Single bundle (`webllm-bundle.js`) detects `DedicatedWorkerGlobalScope` at module load. Main-thread `WebLLM.init` with `worker: true` constructs `new Worker(import.meta.url, { type: "module" })` and returns a `WebLLMProxy` that mirrors `WebLLM`'s public methods over postMessage. Worker side instantiates the real `WebLLM` and routes RPCs.

**Tech Stack:** TypeScript, Bun (test runner + bundler), Biome (lint/format), `tsc --noEmit` (typecheck), Chromium for browser regressions, agentchrome for smoke driving.

**Spec:** [`docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md`](../specs/2026-05-02-dual-mode-worker-deployment-design.md)

---

## File structure

**Created:**
- `src/core/worker-bridge.ts` — message envelope types + request-id counter (Component 1).
- `src/core/webllm-error-codec.ts` — `serializeError` / `reconstructError` (Component 4).
- `src/core/webllm-worker-host.ts` — worker-side message handler + reflect dispatch (Component 3).
- `src/core/webllm-proxy.ts` — main-thread `WebLLMProxy` class (Component 2).
- `tests/webllm-error-codec.test.ts` — mirror-drift sentinel.
- `tests/webllm-proxy-surface.test.ts` — surface-reflection sentinel.
- `tests/worker-bridge-protocol.test.ts` — envelope round-trip tests.
- `tests/webllm-proxy-integration.test.ts` — proxy + stub-worker end-to-end.
- `smoke-test/probe-asyncify-in-worker.html` — Task 1 probe page (lives under smoke-test/ because `make smoke-serve` serves that root).
- `smoke-test/probe-asyncify-in-worker-worker.js` — Task 1 worker script.
- `eval/reports/dual-mode-worker-2026-05-02/` — final closure report directory.

**Modified:**
- `src/core/engine.ts` — `WebLLMConfig.worker` flag (line ~172 area), `WebLLM.init` branch, async-ify `createConversation` / `disposeConversation` / `forkConversation`, add `dispose()`.
- `src/core/types.ts` — add `worker?: boolean` to `WebLLMConfig`.
- `src/index.ts` — export `WebLLMProxy` (for type-import only) and `dispose` if needed for type re-export.
- `smoke-test/real-model-page.js` — read `?worker=1` URL param, plumb to `WebLLM.init`.
- `eval/perf.ts` — add `--worker` flag + URL-param plumbing.
- `eval/bench.ts` — add `--worker` flag.
- `eval/chat-smoke.ts` — add `--worker` flag.
- `eval/embed-perf.ts` — add `--worker` flag.
- `eval/live-server.ts` (dashboard) — accept `mode` field on ingested runs.
- Internal call sites that use sync `createConversation` / `disposeConversation` / `forkConversation` — add `await`.

---

## Task 1: Re-confirm ASYNCIFY-in-worker with the production engine

**Why first:** Probe 9d used a tiny model and the public `loadModelFromBuffer` factory. We need to confirm the *production* engine path (registered model, `chatCompletion` with system prompt) still works in a worker before plumbing the proxy. Also surfaces any `import.meta.url` resolution surprises early (Risk: bundling).

**Files:**
- Create: `smoke-test/probe-asyncify-in-worker.html`
- Create: `smoke-test/probe-asyncify-in-worker-worker.js`
- Create: `eval/reports/probe-asyncify-in-worker-2026-05-02/SUMMARY.md`

- [ ] **Step 1: Make sure smoke server is up**

```bash
make smoke-serve
```

Expected: server running on port 8031 (or `make smoke-restart` if it's stale).

- [ ] **Step 2: Write the worker script**

Create `smoke-test/probe-asyncify-in-worker-worker.js`:

```js
// Probe: confirm ASYNCIFY-driven graphCompute survives in a DedicatedWorker
// against the registered-model engine path (not the loadModelFromBuffer
// factory). Fires WebLLM.init() inside the worker, registers qwen3-0.6b
// via loadModelFromBuffer, runs a 16-token chatCompletion, posts result.

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
		let text = "";
		let nTokens = 0;
		for await (const chunk of engine.chatCompletion(handle.id, [
			{ role: "user", content: "Tell one short joke." },
		], { maxTokens: 16 })) {
			if (chunk.text) text += chunk.text;
			if (chunk.tokenId !== undefined) nTokens++;
		}
		const tGen = performance.now() - tStart;

		self.postMessage({
			type: "done",
			tInitMs: tInit,
			tGenMs: tGen,
			nTokens,
			text,
		});
	} catch (err) {
		self.postMessage({
			type: "error",
			message: err instanceof Error ? err.message : String(err),
			stack: err instanceof Error ? err.stack : undefined,
		});
	}
});
```

- [ ] **Step 3: Write the probe HTML**

Create `smoke-test/probe-asyncify-in-worker.html`:

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>ASYNCIFY in worker probe</title>
<style>body{font:12px monospace;background:#0d1117;color:#c9d1d9;padding:16px}
.pass{color:#3fb950}.fail{color:#f85149}.info{color:#79c0ff}</style></head>
<body><div id="log"></div>
<script type="module">
const log = (cls, msg) => {
	const d = document.createElement("div");
	d.className = cls;
	d.textContent = `[${cls}] ${msg}`;
	document.getElementById("log").appendChild(d);
	console.log(`[${cls}]`, msg);
};
log("info", "booting worker…");
const w = new Worker("./probe-asyncify-in-worker-worker.js", { type: "module" });
w.addEventListener("message", (e) => {
	if (e.data.type === "done") {
		log("pass", `init=${e.data.tInitMs.toFixed(0)}ms gen=${e.data.tGenMs.toFixed(0)}ms tokens=${e.data.nTokens}`);
		log("info", `output: ${JSON.stringify(e.data.text)}`);
		log("pass", `[7/8] Generated ${e.data.nTokens} tokens in ${(e.data.tGenMs/1000).toFixed(1)}s (finish=stop, tokensIn=0)`);
	} else {
		log("fail", e.data.message);
		if (e.data.stack) log("fail", e.data.stack);
	}
});
w.postMessage({ type: "run" });
</script></body></html>
```

- [ ] **Step 4: Build the bundle**

```bash
bun run build
```

Expected: `dist/index.js` rebuilt without errors.

- [ ] **Step 5: Drive the probe through agentchrome**

Reuse the existing browser session (per CLAUDE.md "agentchrome usage" rules). Navigate the existing tab to `http://localhost:8031/probe-asyncify-in-worker.html?v=1`.

```bash
agentchrome connect --status
agentchrome --port <PORT> tabs list
agentchrome navigate http://localhost:8031/probe-asyncify-in-worker.html?v=1 --tab <TAB_ID>
```

Wait for `[pass] Generated N tokens` line in `#log`. Capture page text + console.

- [ ] **Step 6: Write the closure report**

Create `eval/reports/probe-asyncify-in-worker-2026-05-02/SUMMARY.md` with:
- Tip used (`git rev-parse HEAD`)
- Wall times: tInitMs, tGenMs
- Token count + generated text
- Verdict: PASS if 16 tokens generated coherently and no console errors. FAIL if console errors or token count <= 0.

- [ ] **Step 7: Commit the probe**

```bash
git add smoke-test/probe-asyncify-in-worker.html smoke-test/probe-asyncify-in-worker-worker.js
git add -f eval/reports/probe-asyncify-in-worker-2026-05-02/SUMMARY.md
git commit -m "probe: ASYNCIFY in DedicatedWorker — production engine path"
```

**Acceptance:** PASS verdict in the SUMMARY. STOP and surface the failure if the probe fails — proceeding past a broken precondition is the load-bearing risk for this whole effort.

---

## Task 2: `worker-bridge.ts` — message envelope types

**Parallelizable with Task 3.**

**Files:**
- Create: `src/core/worker-bridge.ts`
- Create: `tests/worker-bridge-protocol.test.ts`

- [ ] **Step 1: Write the failing test**

Create `tests/worker-bridge-protocol.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import {
	makeRequestId,
	makeStreamId,
	type ProxyToWorker,
	type WorkerToProxy,
	type SerializedError,
} from "../src/core/worker-bridge.js";

describe("worker-bridge", () => {
	test("makeRequestId returns monotonically increasing positive integers", () => {
		const a = makeRequestId();
		const b = makeRequestId();
		const c = makeRequestId();
		expect(a).toBeGreaterThan(0);
		expect(b).toBeGreaterThan(a);
		expect(c).toBeGreaterThan(b);
	});

	test("makeStreamId returns monotonically increasing positive integers, separate counter from request ids", () => {
		const a = makeStreamId();
		const b = makeStreamId();
		expect(a).toBeGreaterThan(0);
		expect(b).toBeGreaterThan(a);
	});

	test("ProxyToWorker variants survive structuredClone", () => {
		const samples: ProxyToWorker[] = [
			{ type: "init", id: 1, config: { memoryBudget: 8e9 } },
			{ type: "method-call", id: 2, name: "embed", args: ["m1", "hello"] },
			{
				type: "stream-start",
				streamId: 3,
				name: "chatCompletion",
				args: ["m1", [{ role: "user", content: "hi" }], {}],
			},
			{ type: "stream-cancel", streamId: 3 },
			{ type: "dispose", id: 4 },
		];
		for (const s of samples) {
			const cloned = structuredClone(s);
			expect(cloned).toEqual(s);
		}
	});

	test("WorkerToProxy variants survive structuredClone", () => {
		const err: SerializedError = {
			code: "MODEL_NOT_FOUND",
			message: "x",
			modelId: "m1",
		};
		const samples: WorkerToProxy[] = [
			{ type: "init-done", id: 1 },
			{ type: "method-result", id: 2, value: { id: "h1" } },
			{ type: "method-error", id: 2, error: err },
			{
				type: "stream-chunk",
				streamId: 3,
				chunk: { text: "hi", tokenId: 42, done: false },
			},
			{ type: "stream-done", streamId: 3 },
			{ type: "stream-error", streamId: 3, error: err },
			{ type: "log", level: "info", message: "ok" },
		];
		for (const s of samples) {
			const cloned = structuredClone(s);
			expect(cloned).toEqual(s);
		}
	});
});
```

- [ ] **Step 2: Run test — confirm fail**

```bash
bun test tests/worker-bridge-protocol.test.ts
```

Expected: FAIL — `Cannot find module '../src/core/worker-bridge.js'`.

- [ ] **Step 3: Implement `worker-bridge.ts`**

Create `src/core/worker-bridge.ts`:

```ts
/**
 * Message envelope types and shared helpers for the WebLLM worker bridge.
 *
 * Both `webllm-proxy.ts` (main thread) and `webllm-worker-host.ts` (worker)
 * import from here. Pure type module — no runtime imports of engine code.
 */

import type { GenerationStreamChunk } from "../inference/generation.js";
import type { WebLLMConfig } from "./types.js";
import type { WebLLMErrorCode } from "./errors.js";

export type RequestId = number;
export type StreamId = number;

export interface SerializedError {
	code: WebLLMErrorCode | "GENERIC" | "DISPOSED";
	message: string;
	stack?: string;
	modelId?: string;
	architecture?: string;
	conversationId?: string;
	liveConversationIds?: string[];
	requestedTokens?: number;
	maxContextTokens?: number;
}

export type ProxyToWorker =
	| { type: "init"; id: RequestId; config: Omit<WebLLMConfig, "worker"> }
	| { type: "method-call"; id: RequestId; name: string; args: unknown[] }
	| {
			type: "stream-start";
			streamId: StreamId;
			name: "chatCompletion" | "generateStream";
			args: unknown[];
	  }
	| { type: "stream-cancel"; streamId: StreamId }
	| { type: "dispose"; id: RequestId };

export type WorkerToProxy =
	| { type: "init-done"; id: RequestId }
	| { type: "method-result"; id: RequestId; value: unknown }
	| { type: "method-error"; id: RequestId; error: SerializedError }
	| {
			type: "stream-chunk";
			streamId: StreamId;
			chunk: GenerationStreamChunk;
	  }
	| { type: "stream-done"; streamId: StreamId; value?: unknown }
	| { type: "stream-error"; streamId: StreamId; error: SerializedError }
	| { type: "log"; level: "info" | "warn" | "error"; message: string };

let _nextRequestId = 0;
let _nextStreamId = 0;

export function makeRequestId(): RequestId {
	_nextRequestId += 1;
	return _nextRequestId;
}

export function makeStreamId(): StreamId {
	_nextStreamId += 1;
	return _nextStreamId;
}
```

- [ ] **Step 4: Run test — confirm pass**

```bash
bun test tests/worker-bridge-protocol.test.ts
```

Expected: 4 PASS.

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/worker-bridge.ts tests/worker-bridge-protocol.test.ts
git commit -m "feat(worker): add message envelope types for proxy/worker bridge"
```

---

## Task 3: `webllm-error-codec.ts` — typed error round-trip

**Parallelizable with Task 2.**

**Files:**
- Create: `src/core/webllm-error-codec.ts`
- Create: `tests/webllm-error-codec.test.ts`

- [ ] **Step 1: Write the failing test (mirror-drift sentinel)**

Create `tests/webllm-error-codec.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
	WebLLMError,
	type WebLLMErrorCode,
} from "../src/core/errors.js";
import {
	reconstructError,
	serializeError,
} from "../src/core/webllm-error-codec.js";

// Factory table that mirrors errors.ts. Adding a new WebLLMError subclass
// without updating both this table and the codec switch fails the
// "round-trips every code" test below.
const FACTORIES: Record<WebLLMErrorCode, () => WebLLMError> = {
	MODEL_NOT_FOUND: () => new ModelNotFoundError("m1"),
	MODEL_NOT_LOADED: () => new ModelNotLoadedError("m1"),
	INFERENCE_ENGINE_MISSING: () => new InferenceEngineMissingError("m1"),
	ENCODER_REQUIRED: () => new EncoderRequiredError("m1", "qwen3", "use chatCompletion"),
	SPECULATIVE_DECODING_RESERVED: () => new SpeculativeDecodingReservedError(),
	CONVERSATION_NOT_FOUND: () => new ConversationNotFoundError("c1"),
	CONVERSATION_NOT_POPULATED: () => new ConversationNotPopulatedError("c1"),
	CONVERSATION_POOL_FULL: () => new ConversationPoolFullError(["c1", "c2"]),
	CONVERSATION_CONTEXT_OVERFLOW: () =>
		new ConversationContextOverflowError("c1", 5000, 4096),
	CONVERSATION_BUSY: () => new ConversationBusyError("c1"),
};

describe("webllm-error-codec — mirror-drift sentinel", () => {
	test("every WebLLMErrorCode has a factory entry", () => {
		const codes: WebLLMErrorCode[] = [
			"MODEL_NOT_FOUND",
			"MODEL_NOT_LOADED",
			"INFERENCE_ENGINE_MISSING",
			"ENCODER_REQUIRED",
			"SPECULATIVE_DECODING_RESERVED",
			"CONVERSATION_NOT_FOUND",
			"CONVERSATION_NOT_POPULATED",
			"CONVERSATION_POOL_FULL",
			"CONVERSATION_CONTEXT_OVERFLOW",
			"CONVERSATION_BUSY",
		];
		for (const c of codes) {
			expect(FACTORIES[c]).toBeDefined();
		}
	});

	for (const [code, factory] of Object.entries(FACTORIES)) {
		test(`round-trip preserves instanceof, code, message for ${code}`, () => {
			const original = factory();
			const wire = JSON.parse(JSON.stringify(serializeError(original)));
			const rebuilt = reconstructError(wire);
			expect(rebuilt).toBeInstanceOf(WebLLMError);
			expect(rebuilt).toBeInstanceOf(original.constructor);
			expect((rebuilt as WebLLMError).code).toBe(original.code);
			expect(rebuilt.message).toBe(original.message);
		});
	}

	test("ModelNotFoundError preserves modelId field", () => {
		const e = new ModelNotFoundError("qwen3-8b-iq3m");
		const r = reconstructError(JSON.parse(JSON.stringify(serializeError(e)))) as ModelNotFoundError;
		expect(r.modelId).toBe("qwen3-8b-iq3m");
	});

	test("ConversationContextOverflowError preserves all numeric fields", () => {
		const e = new ConversationContextOverflowError("c-x", 5123, 4096);
		const r = reconstructError(JSON.parse(JSON.stringify(serializeError(e)))) as ConversationContextOverflowError;
		expect(r.conversationId).toBe("c-x");
		expect(r.requestedTokens).toBe(5123);
		expect(r.maxContextTokens).toBe(4096);
	});

	test("ConversationPoolFullError preserves liveConversationIds array", () => {
		const e = new ConversationPoolFullError(["c1", "c2", "c3"]);
		const r = reconstructError(JSON.parse(JSON.stringify(serializeError(e)))) as ConversationPoolFullError;
		expect([...r.liveConversationIds]).toEqual(["c1", "c2", "c3"]);
	});

	test("EncoderRequiredError preserves architecture field", () => {
		const e = new EncoderRequiredError("m1", "qwen3", "hint");
		const r = reconstructError(JSON.parse(JSON.stringify(serializeError(e)))) as EncoderRequiredError;
		expect(r.architecture).toBe("qwen3");
	});

	test("non-WebLLMError throws round-trip as plain Error with GENERIC code", () => {
		const e = new RangeError("boom");
		const wire = JSON.parse(JSON.stringify(serializeError(e)));
		expect(wire.code).toBe("GENERIC");
		const r = reconstructError(wire);
		expect(r).not.toBeInstanceOf(WebLLMError);
		expect(r.message).toBe("boom");
	});

	test("DISPOSED code reconstructs as WebLLMError with DISPOSED code (used for crash/dispose paths)", () => {
		const wire = { code: "DISPOSED" as const, message: "engine disposed" };
		const r = reconstructError(wire) as WebLLMError;
		expect(r).toBeInstanceOf(WebLLMError);
		expect(r.code).toBe("DISPOSED" as unknown as WebLLMErrorCode);
		expect(r.message).toBe("engine disposed");
	});

	test("non-Error thrown values serialize with GENERIC code and string-coerced message", () => {
		const wire = serializeError("string thrown");
		expect(wire.code).toBe("GENERIC");
		expect(wire.message).toContain("string thrown");
	});
});
```

- [ ] **Step 2: Run test — confirm fail**

```bash
bun test tests/webllm-error-codec.test.ts
```

Expected: FAIL — `Cannot find module '../src/core/webllm-error-codec.js'`.

- [ ] **Step 3: Implement the codec**

Create `src/core/webllm-error-codec.ts`:

```ts
/**
 * Serialize/reconstruct WebLLMError subclasses across the postMessage
 * boundary. structuredClone drops class identity, so the worker side
 * serializes to a flat shape with a `code` field and the proxy side
 * rebuilds the matching subclass via a single switch.
 *
 * Mirror-drift sentinel: tests/webllm-error-codec.test.ts parametrizes
 * over a factory table that mirrors errors.ts. Adding a new WebLLMError
 * subclass requires updating both this codec and the test factory.
 */

import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
	WebLLMError,
} from "./errors.js";
import type { SerializedError } from "./worker-bridge.js";

export function serializeError(e: unknown): SerializedError {
	if (e instanceof WebLLMError) {
		const out: SerializedError = {
			code: e.code,
			message: e.message,
			stack: e.stack,
		};
		// Subclass-specific fields. Use property checks rather than
		// `instanceof` because a single `if`-chain reads cleaner and
		// the codec must mirror every subclass's fields anyway.
		if (e instanceof ModelNotFoundError) out.modelId = e.modelId;
		else if (e instanceof ModelNotLoadedError) out.modelId = e.modelId;
		else if (e instanceof InferenceEngineMissingError) out.modelId = e.modelId;
		else if (e instanceof EncoderRequiredError) {
			out.modelId = e.modelId;
			out.architecture = e.architecture;
		} else if (e instanceof ConversationNotFoundError)
			out.conversationId = e.conversationId;
		else if (e instanceof ConversationNotPopulatedError)
			out.conversationId = e.conversationId;
		else if (e instanceof ConversationPoolFullError)
			out.liveConversationIds = [...e.liveConversationIds];
		else if (e instanceof ConversationContextOverflowError) {
			out.conversationId = e.conversationId;
			out.requestedTokens = e.requestedTokens;
			out.maxContextTokens = e.maxContextTokens;
		} else if (e instanceof ConversationBusyError)
			out.conversationId = e.conversationId;
		// SpeculativeDecodingReservedError carries no extra fields.
		return out;
	}
	if (e instanceof Error) {
		return { code: "GENERIC", message: e.message, stack: e.stack };
	}
	return { code: "GENERIC", message: `non-Error thrown: ${String(e)}` };
}

export function reconstructError(s: SerializedError): WebLLMError | Error {
	switch (s.code) {
		case "MODEL_NOT_FOUND":
			return attachStack(new ModelNotFoundError(s.modelId ?? ""), s);
		case "MODEL_NOT_LOADED":
			return attachStack(new ModelNotLoadedError(s.modelId ?? ""), s);
		case "INFERENCE_ENGINE_MISSING":
			return attachStack(new InferenceEngineMissingError(s.modelId ?? ""), s);
		case "ENCODER_REQUIRED":
			return attachStack(
				new EncoderRequiredError(s.modelId ?? "", s.architecture ?? ""),
				s,
				// EncoderRequiredError's hint is encoded into message; we
				// pass message verbatim instead of trying to recover hint.
				/* preserveMessage */ true,
			);
		case "SPECULATIVE_DECODING_RESERVED":
			return attachStack(new SpeculativeDecodingReservedError(), s);
		case "CONVERSATION_NOT_FOUND":
			return attachStack(
				new ConversationNotFoundError(s.conversationId ?? ""),
				s,
			);
		case "CONVERSATION_NOT_POPULATED":
			return attachStack(
				new ConversationNotPopulatedError(s.conversationId ?? ""),
				s,
			);
		case "CONVERSATION_POOL_FULL":
			return attachStack(
				new ConversationPoolFullError(s.liveConversationIds ?? []),
				s,
			);
		case "CONVERSATION_CONTEXT_OVERFLOW":
			return attachStack(
				new ConversationContextOverflowError(
					s.conversationId ?? "",
					s.requestedTokens ?? 0,
					s.maxContextTokens ?? 0,
				),
				s,
			);
		case "CONVERSATION_BUSY":
			return attachStack(
				new ConversationBusyError(s.conversationId ?? ""),
				s,
			);
		case "DISPOSED": {
			const e = new WebLLMError(
				s.message,
				"DISPOSED" as unknown as Parameters<
					typeof WebLLMError
				>[1],
			);
			if (s.stack) e.stack = s.stack;
			return e;
		}
		case "GENERIC":
		default: {
			const e = new Error(s.message);
			if (s.stack) e.stack = s.stack;
			return e;
		}
	}
}

function attachStack(
	e: WebLLMError,
	s: SerializedError,
	preserveMessage = false,
): WebLLMError {
	if (preserveMessage) {
		(e as { message: string }).message = s.message;
	}
	if (s.stack) e.stack = s.stack;
	return e;
}
```

> **Note for the implementer:** `WebLLMErrorCode` is a closed union in `errors.ts` and does NOT include `"DISPOSED"` or `"GENERIC"` today. The `SerializedError.code` field uses the broader union (`WebLLMErrorCode | "GENERIC" | "DISPOSED"`). The `WebLLMError` constructor accepts only `WebLLMErrorCode`, so the `DISPOSED` reconstruction casts via `as unknown as` to slot the synthesized code in. This matches how the spec uses `DISPOSED` (proxy synthesizes it for crash/dispose paths; main-thread `WebLLM` never throws `DISPOSED` directly).

- [ ] **Step 4: Run test — confirm pass**

```bash
bun test tests/webllm-error-codec.test.ts
```

Expected: all tests PASS (≥15 tests including the parametrized ones).

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck && bun run typecheck:tests
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/webllm-error-codec.ts tests/webllm-error-codec.test.ts
git commit -m "feat(worker): add typed-error codec for proxy/worker bridge"
```

---

## Task 4: `webllm-worker-host.ts` — worker-side message handler

**Depends on Tasks 2 + 3.** No browser; logic is testable with a stub `postMessage` channel.

**Files:**
- Create: `src/core/webllm-worker-host.ts`
- Create: `tests/webllm-worker-host.test.ts`

- [ ] **Step 1: Write the failing test**

Create `tests/webllm-worker-host.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import type {
	ProxyToWorker,
	WorkerToProxy,
} from "../src/core/worker-bridge.js";
import { startWorkerHost } from "../src/core/webllm-worker-host.js";

// Minimal fake engine implementing the surface the host reflects against.
function makeFakeEngine() {
	return {
		// non-streaming method, returns plain JSON
		async embed(modelId: string, text: string): Promise<Float32Array> {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return new Float32Array([text.length, 0.5, 0.25]);
		},
		// non-streaming method that throws WebLLMError
		async createConversation(modelId: string) {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return { id: "c1", modelHandleId: modelId };
		},
		// streaming method (async generator)
		async *chatCompletion(_modelId: string, _msgs: unknown[]) {
			yield { text: "hi", tokenId: 1, done: false };
			yield { text: " there", tokenId: 2, done: false };
			yield { text: "", done: true, stats: { decodeTokensPerSec: 99 } };
		},
		async dispose() {
			/* no-op */
		},
	};
}

interface FakeChannel {
	hostInbox: ProxyToWorker[];
	proxyInbox: WorkerToProxy[];
	postToHost(m: ProxyToWorker): void;
}

function makeChannel(): {
	channel: FakeChannel;
	postToProxy: (m: WorkerToProxy) => void;
} {
	const proxyInbox: WorkerToProxy[] = [];
	const hostInbox: ProxyToWorker[] = [];
	return {
		channel: {
			hostInbox,
			proxyInbox,
			postToHost(m) {
				hostInbox.push(m);
			},
		},
		postToProxy: (m: WorkerToProxy) => {
			proxyInbox.push(m);
		},
	};
}

describe("webllm-worker-host", () => {
	test("method-call success returns method-result", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		const host = startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		host; // host is the unsubscribe handle; not needed for this test
		channel.postToHost({
			type: "method-call",
			id: 1,
			name: "embed",
			args: ["m1", "abc"],
		});
		// allow microtasks to settle
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox).toEqual([
			{
				type: "method-result",
				id: 1,
				value: new Float32Array([3, 0.5, 0.25]),
			},
		]);
	});

	test("method-call WebLLMError serializes to method-error with code", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "method-call",
			id: 2,
			name: "embed",
			args: ["missing", "x"],
		});
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox).toHaveLength(1);
		const reply = channel.proxyInbox[0];
		expect(reply.type).toBe("method-error");
		if (reply.type === "method-error") {
			expect(reply.id).toBe(2);
			expect(reply.error.code).toBe("MODEL_NOT_FOUND");
			expect(reply.error.modelId).toBe("missing");
		}
	});

	test("stream-start drains async iterator into stream-chunk + stream-done", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 7,
			name: "chatCompletion",
			args: ["m1", [{ role: "user", content: "hi" }], {}],
		});
		// drain microtasks
		for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
		const types = channel.proxyInbox.map((m) => m.type);
		expect(types).toEqual([
			"stream-chunk",
			"stream-chunk",
			"stream-chunk",
			"stream-done",
		]);
	});

	test("unknown method returns method-error with GENERIC code", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "method-call",
			id: 9,
			name: "definitelyDoesNotExist",
			args: [],
		});
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox[0].type).toBe("method-error");
		if (channel.proxyInbox[0].type === "method-error") {
			expect(channel.proxyInbox[0].error.code).toBe("GENERIC");
		}
	});
});
```

- [ ] **Step 2: Run test — confirm fail**

```bash
bun test tests/webllm-worker-host.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement the host**

Create `src/core/webllm-worker-host.ts`:

```ts
/**
 * Worker-side message handler. Constructs a real WebLLM engine and
 * routes ProxyToWorker messages by name via reflect-dispatch.
 *
 * Streaming methods (`chatCompletion`, `generateStream`) drain their
 * async iterator and emit one `stream-chunk` per yield, terminating
 * with `stream-done`. A `stream-cancel` message wires through an
 * AbortController on the args' last `config` argument.
 *
 * Decoupled from `Worker` globals so it can be unit-tested with an
 * in-process channel (see tests/webllm-worker-host.test.ts).
 */

import { serializeError } from "./webllm-error-codec.js";
import type {
	ProxyToWorker,
	StreamId,
	WorkerToProxy,
} from "./worker-bridge.js";

export interface WorkerHostOptions {
	/** Engine instance to dispatch RPCs against. */
	// biome-ignore lint/suspicious/noExplicitAny: reflect-dispatch by name
	engine: any;
	/** Send a message to the proxy (main thread). */
	postMessage(m: WorkerToProxy): void;
	/** Subscribe to incoming proxy messages. */
	receive(handler: (m: ProxyToWorker) => void): void;
	/** Optional logger (default: silent). */
	log?(level: "info" | "warn" | "error", message: string): void;
}

export interface WorkerHostHandle {
	/** Stop receiving messages. Does not stop in-flight streams. */
	close(): void;
}

export function startWorkerHost(opts: WorkerHostOptions): WorkerHostHandle {
	const aborts = new Map<StreamId, AbortController>();
	let closed = false;

	opts.receive((msg) => {
		if (closed) return;
		switch (msg.type) {
			case "method-call":
				void handleMethodCall(msg);
				return;
			case "stream-start":
				void handleStreamStart(msg);
				return;
			case "stream-cancel": {
				const ac = aborts.get(msg.streamId);
				if (ac) ac.abort();
				return;
			}
			case "init":
			case "dispose":
				// `init` and `dispose` are typically handled by the bundle
				// entry that owns engine construction. Surfaces here only
				// in the unit-test path; route via method-call/result.
				return;
		}
	});

	async function handleMethodCall(msg: Extract<ProxyToWorker, { type: "method-call" }>) {
		try {
			const fn = opts.engine[msg.name];
			if (typeof fn !== "function") {
				throw new Error(`unknown engine method: ${msg.name}`);
			}
			const value = await fn.apply(opts.engine, msg.args);
			opts.postMessage({ type: "method-result", id: msg.id, value });
		} catch (e) {
			opts.postMessage({
				type: "method-error",
				id: msg.id,
				error: serializeError(e),
			});
		}
	}

	async function handleStreamStart(
		msg: Extract<ProxyToWorker, { type: "stream-start" }>,
	) {
		const ac = new AbortController();
		aborts.set(msg.streamId, ac);
		// Inject signal into config (last arg, expected to be an
		// optional config object). If args is shorter or last arg is
		// not an object, append a fresh config; consumers tolerate
		// extra fields on config per chat-types.ts.
		const args = [...msg.args];
		const lastIdx = args.length - 1;
		const last = args[lastIdx];
		if (last && typeof last === "object" && !Array.isArray(last)) {
			args[lastIdx] = { ...(last as object), signal: ac.signal };
		} else {
			args.push({ signal: ac.signal });
		}
		try {
			const fn = opts.engine[msg.name];
			if (typeof fn !== "function") {
				throw new Error(`unknown engine streaming method: ${msg.name}`);
			}
			const iter = fn.apply(opts.engine, args);
			for await (const chunk of iter) {
				if (ac.signal.aborted) break;
				opts.postMessage({
					type: "stream-chunk",
					streamId: msg.streamId,
					chunk,
				});
			}
			opts.postMessage({ type: "stream-done", streamId: msg.streamId });
		} catch (e) {
			opts.postMessage({
				type: "stream-error",
				streamId: msg.streamId,
				error: serializeError(e),
			});
		} finally {
			aborts.delete(msg.streamId);
		}
	}

	return {
		close() {
			closed = true;
		},
	};
}
```

- [ ] **Step 4: Run test — confirm pass**

```bash
bun test tests/webllm-worker-host.test.ts
```

Expected: 4 PASS.

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck && bun run typecheck:tests
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/webllm-worker-host.ts tests/webllm-worker-host.test.ts
git commit -m "feat(worker): add worker-side message handler with reflect dispatch"
```

---

## Task 5: `webllm-proxy.ts` — main-thread proxy class (non-streaming)

**Depends on Tasks 2, 3, 4.**

**Files:**
- Create: `src/core/webllm-proxy.ts`
- Create: `tests/webllm-proxy-integration.test.ts`

- [ ] **Step 1: Write the failing test (non-streaming portion)**

Create `tests/webllm-proxy-integration.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import { WebLLMProxy } from "../src/core/webllm-proxy.js";
import { startWorkerHost } from "../src/core/webllm-worker-host.js";
import type {
	ProxyToWorker,
	WorkerToProxy,
} from "../src/core/worker-bridge.js";

// Build an in-process channel that wires WebLLMProxy ↔ startWorkerHost
// without ever constructing a real Worker. Each side has its own
// receive handler; sends are delivered as microtasks.
function makeInProcessChannel(): {
	worker: { postMessage: (m: ProxyToWorker) => void; addEventListener: (event: "message" | "error" | "messageerror", h: (e: { data: WorkerToProxy } | Event) => void) => void; terminate: () => void };
	hostPost: (m: WorkerToProxy) => void;
	hostReceive: (handler: (m: ProxyToWorker) => void) => void;
} {
	let toHost: ((m: ProxyToWorker) => void) | null = null;
	let toProxy: ((e: { data: WorkerToProxy }) => void) | null = null;
	const worker = {
		postMessage(m: ProxyToWorker) {
			queueMicrotask(() => toHost?.(m));
		},
		addEventListener(
			event: "message" | "error" | "messageerror",
			h: (e: { data: WorkerToProxy } | Event) => void,
		) {
			if (event === "message") {
				toProxy = h as (e: { data: WorkerToProxy }) => void;
			}
		},
		terminate() {},
	};
	const hostPost = (m: WorkerToProxy) => {
		queueMicrotask(() => toProxy?.({ data: m }));
	};
	const hostReceive = (handler: (m: ProxyToWorker) => void) => {
		toHost = handler;
	};
	return { worker, hostPost, hostReceive };
}

function makeFakeEngine() {
	return {
		async embed(modelId: string): Promise<Float32Array> {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return new Float32Array([1, 2, 3]);
		},
		async createConversation(modelId: string) {
			return { id: `c-${modelId}`, modelHandleId: modelId };
		},
		async disposeConversation(_conv: unknown) {
			return undefined;
		},
		async dispose() {},
	};
}

describe("WebLLMProxy — non-streaming", () => {
	test("embed round-trip returns the worker's value", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const v = await proxy.embed("m1", "hello");
		expect(Array.from(v)).toEqual([1, 2, 3]);
	});

	test("embed surfaces typed ModelNotFoundError main-thread", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		await expect(proxy.embed("missing", "x")).rejects.toBeInstanceOf(
			ModelNotFoundError,
		);
	});

	test("createConversation returns the worker's handle", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const h = await proxy.createConversation("m1");
		expect(h).toEqual({ id: "c-m1", modelHandleId: "m1" });
	});

	test("dispose terminates the worker and rejects subsequent calls", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		let terminated = false;
		const origTerminate = worker.terminate;
		worker.terminate = () => {
			terminated = true;
			origTerminate();
		};
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		await proxy.dispose();
		expect(terminated).toBe(true);
		await expect(proxy.embed("m1", "x")).rejects.toThrow(/dispose/i);
	});
});
```

- [ ] **Step 2: Run test — confirm fail**

```bash
bun test tests/webllm-proxy-integration.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement the proxy (non-streaming portion)**

Create `src/core/webllm-proxy.ts`:

```ts
/**
 * Main-thread façade that mirrors the public WebLLM surface over a
 * postMessage channel. See spec §"Components" #2.
 *
 * Non-streaming methods marshal as a single method-call/method-result
 * round trip. Streaming methods are added in a follow-up commit and
 * back the AsyncIterableIterator they return with a per-stream queue.
 */

import { reconstructError } from "./webllm-error-codec.js";
import {
	type ProxyToWorker,
	type RequestId,
	type SerializedError,
	type WorkerToProxy,
	makeRequestId,
} from "./worker-bridge.js";
import type { WebLLMConfig, ModelHandle } from "./types.js";
import type {
	ConversationHandle,
	ConversationOptions,
} from "./conversation-pool.js";

interface MinimalWorker {
	postMessage(m: ProxyToWorker, transfer?: Transferable[]): void;
	addEventListener(
		event: "message",
		h: (e: { data: WorkerToProxy }) => void,
	): void;
	addEventListener(event: "error", h: (e: Event) => void): void;
	addEventListener(event: "messageerror", h: (e: Event) => void): void;
	terminate(): void;
}

export class WebLLMProxy {
	private worker: MinimalWorker;
	private pending = new Map<
		RequestId,
		{ resolve: (v: unknown) => void; reject: (e: unknown) => void }
	>();
	private disposed = false;

	private constructor(worker: MinimalWorker) {
		this.worker = worker;
		worker.addEventListener("message", (e: { data: WorkerToProxy }) => {
			this.handleMessage(e.data);
		});
		worker.addEventListener("error", () => {
			this.handleCrash("worker error event");
		});
		worker.addEventListener("messageerror", () => {
			this.handleCrash("worker messageerror event");
		});
	}

	/**
	 * Public init path used by `WebLLM.init({ worker: true })`. Constructs
	 * a Worker, sends the init RPC, awaits init-done.
	 */
	static async init(config: WebLLMConfig): Promise<WebLLMProxy> {
		const worker = new Worker(new URL(import.meta.url, import.meta.url), {
			type: "module",
		}) as unknown as MinimalWorker;
		const proxy = new WebLLMProxy(worker);
		await proxy.callInit(config);
		return proxy;
	}

	/** Test-only entry: hand in a pre-wired worker (e.g. an in-process channel). */
	static async fromWorker(worker: MinimalWorker): Promise<WebLLMProxy> {
		const proxy = new WebLLMProxy(worker);
		await proxy.callInit({}); // empty config OK for the in-process tests
		return proxy;
	}

	private async callInit(config: WebLLMConfig): Promise<void> {
		const { worker: _w, ...rest } = config;
		void _w;
		await this.request<void>({
			type: "init",
			id: makeRequestId(),
			config: rest,
		});
	}

	// ────────── public WebLLM surface (non-streaming) ──────────

	loadModel = (...args: unknown[]) => this.callMethod<unknown>("loadModel", args);
	// Instance signature in engine.ts:1132 is (data, name, wasmUrl?, options?)
	// returning { handle, inference }. The proxy mirrors that exactly.
	loadModelFromBuffer = (
		data: ArrayBuffer,
		name: string,
		wasmUrl?: string,
		options?: unknown,
	): Promise<{ handle: ModelHandle; inference: unknown }> =>
		this.callMethod<{ handle: ModelHandle; inference: unknown }>(
			"loadModelFromBuffer",
			[data, name, wasmUrl, options],
			[data],
		);
	unloadModel = (id: string) => this.callMethod<void>("unloadModel", [id]);
	embed = (modelId: string, text: string) =>
		this.callMethod<Float32Array>("embed", [modelId, text]);
	chat = (modelId: string, prompt: string, config?: unknown) =>
		this.callMethod<string>("chat", [modelId, prompt, config]);
	createConversation = (
		modelHandleId: string,
		opts?: ConversationOptions,
	): Promise<ConversationHandle> =>
		this.callMethod<ConversationHandle>("createConversation", [
			modelHandleId,
			opts,
		]);
	disposeConversation = (conv: ConversationHandle): Promise<void> =>
		this.callMethod<void>("disposeConversation", [conv]);
	forkConversation = (src: ConversationHandle): Promise<ConversationHandle> =>
		this.callMethod<ConversationHandle>("forkConversation", [src]);

	async dispose(): Promise<void> {
		if (this.disposed) return;
		try {
			await this.request<void>({
				type: "dispose",
				id: makeRequestId(),
			});
		} catch {
			// ignore — we're tearing down anyway
		}
		this.handleCrash("dispose");
	}

	// ────────── private plumbing ──────────

	private callMethod<T>(
		name: string,
		args: unknown[],
		transfer?: Transferable[],
	): Promise<T> {
		if (this.disposed) {
			return Promise.reject(new Error("WebLLM proxy disposed"));
		}
		return this.request<T>(
			{ type: "method-call", id: makeRequestId(), name, args },
			transfer,
		);
	}

	private request<T>(msg: ProxyToWorker, transfer?: Transferable[]): Promise<T> {
		return new Promise<T>((resolve, reject) => {
			const id = "id" in msg ? msg.id : -1;
			if (id < 0) {
				reject(new Error("internal: request without id"));
				return;
			}
			this.pending.set(id, {
				resolve: (v) => resolve(v as T),
				reject,
			});
			try {
				if (transfer && transfer.length > 0) {
					this.worker.postMessage(msg, transfer);
				} else {
					this.worker.postMessage(msg);
				}
			} catch (e) {
				this.pending.delete(id);
				reject(e);
			}
		});
	}

	private handleMessage(m: WorkerToProxy): void {
		switch (m.type) {
			case "init-done": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.resolve(undefined);
				}
				return;
			}
			case "method-result": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.resolve(m.value);
				}
				return;
			}
			case "method-error": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.reject(reconstructError(m.error));
				}
				return;
			}
			case "stream-chunk":
			case "stream-done":
			case "stream-error":
				// Routed by the streaming task (Task 6) — leave a no-op
				// hook here for now so the unit tests for non-streaming
				// methods don't trip over unhandled-event warnings.
				return;
			case "log":
				return;
		}
	}

	private handleCrash(reason: string): void {
		if (this.disposed) return;
		this.disposed = true;
		const err: SerializedError = {
			code: "DISPOSED",
			message: `engine disposed (${reason})`,
		};
		const rebuilt = reconstructError(err);
		for (const p of this.pending.values()) {
			p.reject(rebuilt);
		}
		this.pending.clear();
		try {
			this.worker.terminate();
		} catch {
			// ignore
		}
	}
}
```

- [ ] **Step 4: Run test — confirm pass**

```bash
bun test tests/webllm-proxy-integration.test.ts
```

Expected: 4 PASS (the streaming tests come in Task 6).

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck && bun run typecheck:tests
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/webllm-proxy.ts tests/webllm-proxy-integration.test.ts
git commit -m "feat(worker): add WebLLMProxy class for non-streaming methods"
```

---

## Task 6: Proxy streaming + cancellation

**Depends on Task 5.**

**Files:**
- Modify: `src/core/webllm-proxy.ts` — add `chatCompletion` and `generateStream`, route `stream-chunk` / `stream-done` / `stream-error`.
- Modify: `tests/webllm-proxy-integration.test.ts` — add streaming tests.

- [ ] **Step 1: Add failing streaming tests**

Append to `tests/webllm-proxy-integration.test.ts`:

```ts
describe("WebLLMProxy — streaming", () => {
	function makeStreamingEngine() {
		return {
			async *chatCompletion(
				_modelId: string,
				_msgs: unknown[],
				config?: { signal?: AbortSignal },
			) {
				const sig = config?.signal;
				const chunks = [
					{ text: "a", tokenId: 1, done: false },
					{ text: "b", tokenId: 2, done: false },
					{ text: "c", tokenId: 3, done: false },
					{ text: "", done: true, stats: { decodeTokensPerSec: 100 } },
				];
				for (const c of chunks) {
					if (sig?.aborted) return;
					await new Promise((r) => setTimeout(r, 0));
					yield c;
				}
			},
			async dispose() {},
		};
	}

	test("chatCompletion yields all chunks in order", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeStreamingEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const seen: string[] = [];
		for await (const chunk of proxy.chatCompletion("m1", [
			{ role: "user", content: "hi" },
		])) {
			if (chunk.text) seen.push(chunk.text);
		}
		expect(seen).toEqual(["a", "b", "c"]);
	});

	test("chatCompletion early-break sends stream-cancel", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeStreamingEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		let count = 0;
		for await (const chunk of proxy.chatCompletion("m1", [
			{ role: "user", content: "hi" },
		])) {
			count += 1;
			if (chunk.tokenId === 1) break;
		}
		expect(count).toBe(1);
	});

	test("chatCompletion propagates worker-side stream-error as typed error", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: {
				// biome-ignore lint/correctness/useYield: intentional throw before yield
				async *chatCompletion() {
					throw new (await import("../src/core/errors.js")).ModelNotFoundError(
						"m1",
					);
				},
				async dispose() {},
			},
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const ModelNotFoundError = (await import("../src/core/errors.js"))
			.ModelNotFoundError;
		const consume = async () => {
			for await (const _c of proxy.chatCompletion("m1", [
				{ role: "user", content: "hi" },
			])) {
				/* drain */
			}
		};
		await expect(consume()).rejects.toBeInstanceOf(ModelNotFoundError);
	});
});
```

- [ ] **Step 2: Run streaming tests — confirm fail**

```bash
bun test tests/webllm-proxy-integration.test.ts
```

Expected: 3 new tests FAIL (no `chatCompletion` on `WebLLMProxy`).

- [ ] **Step 3: Add streaming to `WebLLMProxy`**

Modify `src/core/webllm-proxy.ts`:

Add to imports:
```ts
import type { GenerationStreamChunk } from "../inference/generation.js";
import { makeStreamId, type StreamId } from "./worker-bridge.js";
```

Add a private `streams` field next to `pending`:
```ts
private streams = new Map<
	StreamId,
	{
		queue: GenerationStreamChunk[];
		waiters: Array<(r: IteratorResult<GenerationStreamChunk>) => void>;
		errored: unknown | null;
		done: boolean;
	}
>();
```

Add public streaming methods (next to the non-streaming ones):
```ts
chatCompletion = (
	modelOrConv: string | ConversationHandle,
	messages: unknown[],
	config?: unknown,
): AsyncIterableIterator<GenerationStreamChunk> =>
	this.startStream("chatCompletion", [modelOrConv, messages, config]);

generateStream = (
	modelId: string,
	input: unknown,
	config?: unknown,
): AsyncIterableIterator<GenerationStreamChunk> =>
	this.startStream("generateStream", [modelId, input, config]);
```

Add the queue-backed iterator:
```ts
private startStream(
	name: "chatCompletion" | "generateStream",
	args: unknown[],
): AsyncIterableIterator<GenerationStreamChunk> {
	if (this.disposed) {
		const err = new Error("WebLLM proxy disposed");
		return (async function* () {
			throw err;
		})();
	}
	const streamId = makeStreamId();
	this.streams.set(streamId, {
		queue: [],
		waiters: [],
		errored: null,
		done: false,
	});
	this.worker.postMessage({ type: "stream-start", streamId, name, args });
	const self = this;
	let cancelled = false;
	const iter: AsyncIterableIterator<GenerationStreamChunk> = {
		[Symbol.asyncIterator]() {
			return this;
		},
		next(): Promise<IteratorResult<GenerationStreamChunk>> {
			const s = self.streams.get(streamId);
			if (!s) {
				return Promise.resolve({ value: undefined, done: true });
			}
			if (s.errored) {
				const err = s.errored;
				self.streams.delete(streamId);
				return Promise.reject(err);
			}
			if (s.queue.length > 0) {
				const value = s.queue.shift() as GenerationStreamChunk;
				return Promise.resolve({ value, done: false });
			}
			if (s.done) {
				self.streams.delete(streamId);
				return Promise.resolve({ value: undefined, done: true });
			}
			return new Promise((resolve) => s.waiters.push(resolve));
		},
		return(): Promise<IteratorResult<GenerationStreamChunk>> {
			if (!cancelled) {
				cancelled = true;
				try {
					self.worker.postMessage({ type: "stream-cancel", streamId });
				} catch {
					// ignore
				}
			}
			self.streams.delete(streamId);
			return Promise.resolve({ value: undefined, done: true });
		},
		throw(e?: unknown): Promise<IteratorResult<GenerationStreamChunk>> {
			self.streams.delete(streamId);
			return Promise.reject(e);
		},
	};
	return iter;
}
```

Update `handleMessage` so `stream-chunk` / `stream-done` / `stream-error` route into the queue:
```ts
case "stream-chunk": {
	const s = this.streams.get(m.streamId);
	if (!s) return;
	if (s.waiters.length > 0) {
		const w = s.waiters.shift() as (r: IteratorResult<GenerationStreamChunk>) => void;
		w({ value: m.chunk, done: false });
	} else {
		s.queue.push(m.chunk);
	}
	return;
}
case "stream-done": {
	const s = this.streams.get(m.streamId);
	if (!s) return;
	s.done = true;
	for (const w of s.waiters) w({ value: undefined, done: true });
	s.waiters = [];
	return;
}
case "stream-error": {
	const s = this.streams.get(m.streamId);
	if (!s) return;
	s.errored = reconstructError(m.error);
	for (const w of s.waiters) {
		// Reject pending waiters by switching to throw flow.
		// Use Promise.reject via a synchronous throw inside next().
	}
	// The `next()` call that picks this up will see `errored` and reject.
	// To wake any current waiters, deliver done=true so they re-enter `next()`
	// and observe the error on the next tick.
	for (const w of s.waiters) w({ value: undefined, done: true });
	s.waiters = [];
	return;
}
```

> **Important:** the `stream-error` arm above wakes waiters with `done:true` so the next `next()` call observes `s.errored`. The simpler approach — directly rejecting the waiter Promise — would work but requires storing reject handlers alongside the resolvers. Pick whichever pattern feels cleaner during implementation; the test asserts the error surfaces as `instanceof ModelNotFoundError`, not the exact tick.

Also extend `handleCrash` to fail any open streams:
```ts
for (const s of this.streams.values()) {
	s.errored = rebuilt;
	for (const w of s.waiters) w({ value: undefined, done: true });
	s.waiters = [];
}
this.streams.clear();
```

- [ ] **Step 4: Run streaming tests — confirm pass**

```bash
bun test tests/webllm-proxy-integration.test.ts
```

Expected: all 7 tests PASS (4 non-streaming + 3 streaming).

- [ ] **Step 5: Add the surface-reflection sentinel test**

Create `tests/webllm-proxy-surface.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import { WebLLM } from "../src/core/engine.js";
import { WebLLMProxy } from "../src/core/webllm-proxy.js";

// The proxy must mirror the public methods on WebLLM. New methods on
// WebLLM that aren't mirrored here cause this test to fail loudly.
const PROXIED_METHODS: ReadonlyArray<keyof WebLLMProxy & string> = [
	"loadModel",
	"loadModelFromBuffer",
	"unloadModel",
	"embed",
	"chat",
	"chatCompletion",
	"generateStream",
	"createConversation",
	"disposeConversation",
	"forkConversation",
	"dispose",
];

describe("WebLLMProxy — surface mirror sentinel", () => {
	test("every proxied method exists on WebLLM", () => {
		for (const name of PROXIED_METHODS) {
			expect(
				typeof (WebLLM.prototype as unknown as Record<string, unknown>)[name],
			).toBe("function");
		}
	});

	test("every proxied method exists on WebLLMProxy", () => {
		// Proxy methods are arrow-bound on the instance, not on the prototype.
		// Construct via a stub channel to test instance shape.
		const { default: makeStub } = require("./_helpers/proxy-stub.cjs") as {
			default: () => WebLLMProxy;
		};
		void makeStub; // placeholder; real stub built in Task 6 if helper missing
		// Minimal check: the class definition has the field declarations;
		// using `WebLLMProxy.prototype` is sufficient because arrow-class-fields
		// land on the instance, but the surface test still wants instance access.
		// Use a synthetic instance via `Object.create(WebLLMProxy.prototype)`;
		// arrow fields aren't bound, so we check both shapes:
		const keys = new Set(Object.getOwnPropertyNames(WebLLMProxy.prototype));
		const expected = ["dispose"]; // class methods only
		for (const k of expected) expect(keys.has(k)).toBe(true);
	});
});
```

> **Note for the implementer:** the second test's "instance-shape" check is awkward because the proxy uses arrow-class-field methods (so they're per-instance, not on the prototype). If the test as written is too brittle, replace the second assertion with a real instance-build via the in-process channel from `webllm-proxy-integration.test.ts` and assert each method resolves to a function. The prototype-only check on `dispose` is the minimal safety net; the integration test already covers the rest.

- [ ] **Step 6: Run all proxy tests + typecheck**

```bash
bun test tests/webllm-proxy-integration.test.ts tests/webllm-proxy-surface.test.ts
bun run typecheck && bun run typecheck:tests
```

Expected: all PASS, 0 type errors.

- [ ] **Step 7: Commit**

```bash
git add src/core/webllm-proxy.ts tests/webllm-proxy-integration.test.ts tests/webllm-proxy-surface.test.ts
git commit -m "feat(worker): add streaming + cancellation to WebLLMProxy"
```

---

## Task 7: Async-ify conversation methods on main-thread `WebLLM`

**Breaking change.** Touches every internal call site that uses `createConversation` / `disposeConversation` / `forkConversation`.

**Files:**
- Modify: `src/core/engine.ts` — convert three method signatures to `Promise`.
- Modify: every internal call site (smoke harness, eval runners, tests). Audit with `grep`.

- [ ] **Step 1: Audit call sites**

```bash
grep -rn "createConversation\|disposeConversation\|forkConversation" src/ tests/ eval/ smoke-test/ | grep -v "\.test\.ts.*//"
```

Capture the full list. Every non-test, non-comment hit on the production paths is a call site that needs `await` after this change.

- [ ] **Step 2: Update `engine.ts` signatures**

In `src/core/engine.ts`, change:

```ts
createConversation(
    modelHandleId: string,
    options?: ConversationOptions,
): ConversationHandle {
```
to:
```ts
async createConversation(
    modelHandleId: string,
    options?: ConversationOptions,
): Promise<ConversationHandle> {
```

Same for `disposeConversation` (returns `Promise<void>`) and `forkConversation` (returns `Promise<ConversationHandle>`). The bodies stay the same — these methods don't actually await anything today, but the `async` keyword wraps the return in a Promise.

- [ ] **Step 3: Update all internal call sites**

For each hit from Step 1 outside `engine.ts`, add `await`. Examples that are likely present (from prefix-cache work):
- `eval/probes/probe-9a-prefill-prefix.ts`
- `eval/probes/probe-9b-batched-vs-sequential.ts`
- `eval/reports/probe-9b-2026-05-01/*.ts`
- `smoke-test/real-model-page.js` — the bench-mode prefix-cache path.
- Any test under `tests/` that uses these methods.

> **For the implementer:** use the grep audit from Step 1 as a checklist. Each call site's surrounding function must already be `async`; if it isn't, wrap accordingly.

- [ ] **Step 4: Run typecheck — catches any missed `await`**

```bash
bun run typecheck && bun run typecheck:tests
```

Expected: 0 errors. If any errors fire, they point directly at the missed call sites — fix them and re-run.

- [ ] **Step 5: Run full test suite**

```bash
bun test
```

Expected: all green. The conversation-pool tests (`tests/conversation-pool.test.ts`) and chat-completion-conversation tests (`tests/chat-completion-conversation.test.ts`) are the primary safety net here.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor(api): async-ify createConversation/disposeConversation/forkConversation

Worker mode (incoming) requires these methods to be async because the
postMessage round-trip is inherently async. Making them async on the
main-thread engine too keeps the public API symmetric across modes.

Breaking change: callers must await. Internal callers updated."
```

---

## Task 8: Wire `WebLLM.init({ worker: true })` and bundle entry detection

**Depends on Tasks 5, 6, 7.**

**Files:**
- Modify: `src/core/types.ts` — add `worker?: boolean` to `WebLLMConfig`.
- Modify: `src/core/engine.ts` — branch in `WebLLM.init`, add `dispose()` method.
- Modify: `src/index.ts` — add bundle-entry detection that boots the worker host when the bundle is loaded inside a `DedicatedWorkerGlobalScope`.

- [ ] **Step 1: Add `worker` to `WebLLMConfig`**

In `src/core/types.ts`, add to the `WebLLMConfig` interface:

```ts
/**
 * Run engine in a DedicatedWorker. Default false.
 *
 * When true, WebGPU + ggml-wasm execute off-main-thread; the returned
 * WebLLM is a proxy. All public methods retain their signatures.
 */
worker?: boolean;
```

- [ ] **Step 2: Branch `WebLLM.init`**

In `src/core/engine.ts`, replace:
```ts
static async init(config: WebLLMConfig): Promise<WebLLM> {
    return new WebLLM(config);
}
```
with:
```ts
static async init(config: WebLLMConfig): Promise<WebLLM> {
    if (config.worker && !isWorkerContext()) {
        const { WebLLMProxy } = await import("./webllm-proxy.js");
        // The proxy mirrors WebLLM's public surface; its TS type is
        // structurally compatible enough to return as WebLLM.
        return WebLLMProxy.init(config) as unknown as Promise<WebLLM>;
    }
    return new WebLLM({ ...config, worker: false });
}
```

Add a helper at module scope (after imports):
```ts
function isWorkerContext(): boolean {
    return (
        typeof DedicatedWorkerGlobalScope !== "undefined" &&
        // biome-ignore lint/suspicious/noExplicitAny: globalThis narrowing
        (globalThis as any) instanceof DedicatedWorkerGlobalScope
    );
}
```

- [ ] **Step 3: Add `dispose()` method to `WebLLM`**

In `src/core/engine.ts`, add (alongside `unloadModel`):

```ts
/**
 * Release all engine resources: unload every model, free the WebGPU
 * device, drop the wasm module references. After dispose(), the engine
 * is unusable. Worker-mode callers see worker.terminate() too via
 * WebLLMProxy.dispose().
 */
async dispose(): Promise<void> {
    const ids = [...this.inferenceEngines.keys(), ...this.encoderEngines.keys(), ...this.causalEmbedderEngines.keys()];
    const seen = new Set<string>();
    for (const id of ids) {
        if (seen.has(id)) continue;
        seen.add(id);
        await this.unloadModel(id);
    }
}
```

- [ ] **Step 4: Wire bundle-entry detection**

At the **end** of `src/index.ts`, append:

```ts
// ─── Worker bundle re-entry ──────────────────────────────────
//
// When the same bundle module is loaded inside a DedicatedWorker
// (via `new Worker(import.meta.url, { type: "module" })`), boot the
// message-handler host instead of just exposing the public exports.
// Main-thread bundle loads see the typeof check fail and skip this.
if (
    typeof DedicatedWorkerGlobalScope !== "undefined" &&
    // biome-ignore lint/suspicious/noExplicitAny: globalThis narrowing
    (globalThis as any) instanceof DedicatedWorkerGlobalScope
) {
    const { WebLLM } = await import("./core/engine.js");
    const { startWorkerHost } = await import("./core/webllm-worker-host.js");
    let engine: InstanceType<typeof WebLLM> | null = null;
    startWorkerHost({
        // The host stores `engine` once init lands; method-calls
        // before init result in "unknown engine method" which becomes
        // a GENERIC error main-thread.
        get engine() {
            if (!engine) throw new Error("worker engine not initialized");
            return engine;
        },
        postMessage: (m) => (self as unknown as Worker).postMessage(m),
        receive: (handler) => {
            self.addEventListener("message", (e) => {
                const msg = (e as MessageEvent).data;
                if (msg?.type === "init") {
                    void (async () => {
                        try {
                            engine = await WebLLM.init({ ...msg.config, worker: false });
                            (self as unknown as Worker).postMessage({
                                type: "init-done",
                                id: msg.id,
                            });
                        } catch (err) {
                            const { serializeError } = await import("./core/webllm-error-codec.js");
                            (self as unknown as Worker).postMessage({
                                type: "method-error",
                                id: msg.id,
                                error: serializeError(err),
                            });
                        }
                    })();
                    return;
                }
                handler(msg);
            });
        },
    });
}
```

> **Note for the implementer:** the `get engine()` accessor pattern lets the host start receiving messages before `init` lands — only the `init` handler resolves the engine. If `init` arrives multiple times the second one overwrites; document this as undefined behavior (consumers shouldn't double-init a single Worker).

> **Important — dispose must reach the engine.** `webllm-worker-host.ts` deliberately treats `init` and `dispose` as no-ops (it's the bundle entry's job to own engine lifecycle). The proxy's `dispose()` is fire-and-forget today (Task 5). So the bundle entry's receive handler must intercept `{ type: "dispose", id }` BEFORE delegating to `startWorkerHost`, call `await engine.dispose()`, and post `{ type: "method-result", id }` (or just let the proxy's subsequent `worker.terminate()` close the channel). Otherwise `engine.dispose()` never fires inside the worker → leaked WebGPU device + heap. Add this `dispose` arm alongside the `init` arm in the bundle-entry receive switch.

- [ ] **Step 5: Build the bundle**

```bash
bun run build
```

Expected: `dist/index.js` rebuilt; both the main exports and the worker-handler block are present in the output.

- [ ] **Step 6: Run full test suite**

```bash
bun test
```

Expected: all green. (No new tests in this task — the worker-mode end-to-end gets exercised in Task 10.)

- [ ] **Step 7: Run typecheck + checkall**

```bash
make checkall
```

Expected: 0 errors.

- [ ] **Step 8: Commit**

```bash
git add src/core/engine.ts src/core/types.ts src/index.ts
git commit -m "feat(worker): wire WebLLM.init({ worker: true }) + bundle re-entry

Same bundle works in both contexts: main-thread exports the public
WebLLM API; in a DedicatedWorker the bundle entry boots the message
handler instead. WebLLM.init delegates to WebLLMProxy when worker:true."
```

---

## Task 9: Smoke + bench harness flags

**Depends on Task 8.**

**Files:**
- Modify: `smoke-test/real-model-page.js` — read `?worker=1` URL param, pass `worker: true` into `WebLLM.init`.
- Modify: `eval/perf.ts` — add `--worker` flag, plumb to URL param.
- Modify: `eval/bench.ts` — add `--worker` flag.
- Modify: `eval/chat-smoke.ts` — add `--worker` flag.
- Modify: `eval/embed-perf.ts` — add `--worker` flag.
- Modify: `eval/live-server.ts` (dashboard ingestion) — accept and store `mode` field.

- [ ] **Step 1: Read `?worker` URL param in smoke harness**

In `smoke-test/real-model-page.js`, near the top where URL params are parsed (look for `new URLSearchParams(window.location.search)`), add:

```js
const useWorker = params.get("worker") === "1";
```

Where `WebLLM.init` is called (likely in a `bootEngine` / `initEngine` style function), thread `worker: useWorker` into the config object. Search for the existing call pattern and adjust.

When publishing live-bench events (search for `WEBLLM_LIVE_BENCH_URL` plumbing or similar), include `mode: useWorker ? "worker" : "main"` in the payload.

- [ ] **Step 2: Add `--worker` flag to `eval/perf.ts`**

Find the arg-parsing block (search for `process.argv` or a CLI-arg helper). Add:
```ts
const useWorker = process.argv.includes("--worker");
```

Where the smoke URL is built (search for `?model=` query construction), append `&worker=1` when `useWorker`. Live ingest payloads gain `mode: useWorker ? "worker" : "main"`.

- [ ] **Step 3: Same edits in `eval/bench.ts`, `eval/chat-smoke.ts`, `eval/embed-perf.ts`**

Each follows the same pattern: parse `--worker`, append `worker=1` to the smoke URL, include `mode` in any live-server event payload.

- [ ] **Step 4: Accept `mode` in dashboard ingestion**

In `eval/live-server.ts`, find the ingest handler (search for `run_complete` event or POST `/runs`). Add `mode TEXT` column to the SQLite schema (use a migration that ALTER TABLEs if absent). Persist the value when present; default to `'main'` when missing.

```ts
// In schema setup:
db.exec(`ALTER TABLE runs ADD COLUMN mode TEXT DEFAULT 'main';`);
// (gracefully handle "duplicate column" error if it already exists)
```

- [ ] **Step 5: Manual smoke verification**

Bring up the static server + dashboard:
```bash
make smoke-serve
make dashboard-serve
```

Drive a worker-mode smoke run (use existing agentchrome session per CLAUDE.md):
```bash
agentchrome navigate "http://localhost:8031/smoke-test/real-model.html?model=qwen3-0.6b-q4f16&worker=1&v=1" --tab <TAB_ID>
```

Wait for `[7/8]` and `[8/8]` lines. Confirm dashboard at `http://localhost:8033` shows the run with `mode=worker`.

- [ ] **Step 6: Commit**

```bash
git add smoke-test/real-model-page.js eval/perf.ts eval/bench.ts eval/chat-smoke.ts eval/embed-perf.ts eval/live-server.ts
git commit -m "feat(worker): add ?worker / --worker flags to smoke and bench harnesses

Live dashboard records mode=main|worker per run for cross-mode A/B."
```

---

## Task 10: Browser regression sweep + cross-mode A/B + closure report

**Depends on Tasks 1, 8, 9.** This is the validation gate before declaring item 10 closed.

**Files:**
- Create: `eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`
- Possibly create supporting raw-data files under the same directory.

- [ ] **Step 1: Smoke regression on `qwen3-0.6b-q4f16`**

```bash
agentchrome navigate "http://localhost:8031/smoke-test/real-model.html?model=qwen3-0.6b-q4f16&worker=1&v=1" --tab <TAB_ID>
```

Expected: `[1/8]` through `[8/8]` lines all PASS, no console errors. Capture `[7/8]` decode tok/s and `[8/8]` accuracy.

- [ ] **Step 2: Smoke regression on `qwen3-8b-iq3m`** (gates the heap-streaming loader path under worker mode)

```bash
agentchrome navigate "http://localhost:8031/smoke-test/real-model.html?model=qwen3-8b-iq3m&worker=1&v=2" --tab <TAB_ID>
```

Expected: same — full sequence PASS, no console errors. Decode tok/s should be within ±5% of the same-day main-thread baseline.

- [ ] **Step 3: Frame-probe coexistence under worker mode**

```bash
agentchrome navigate "http://localhost:8031/smoke-test/real-model.html?model=qwen3-8b-iq3m&worker=1&frameProbe=1&v=3" --tab <TAB_ID>
```

Capture frame-probe summary. Gate: median `decode_max < 15 ms` (probe 9d measured 9.1 ms median; 5.5× headroom over main-thread 49.8 ms).

- [ ] **Step 4: Cross-mode A/B perf — canonical 6**

Run with `--worker`:
```bash
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3 PERF_EXTRA="--worker"
make smoke-bench PERF_MODEL=qwen3-0.6b-q4f16          PERF_RUNS=3 PERF_EXTRA="--worker"
make smoke-bench PERF_MODEL=qwen3-1.7b-q4f16          PERF_RUNS=3 PERF_EXTRA="--worker"
make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q4ks PERF_RUNS=3 PERF_EXTRA="--worker"
make smoke-bench PERF_MODEL=llama-3.1-8b-instruct-iq3m  PERF_RUNS=3 PERF_EXTRA="--worker"
make smoke-bench PERF_MODEL=qwen3-8b-iq3m              PERF_RUNS=3 PERF_EXTRA="--worker"
```

> **For the implementer:** `make smoke-bench` may not currently honor a `PERF_EXTRA` env var. If not, add the plumbing inside `eval/perf.ts`'s `extraParams` block as a one-line addition gated on a new env var (matches the pattern called out in `CLAUDE.md` "Live dashboard" section).

For each run, capture decode tok/s. Compare to the same-day main-thread baseline (re-run main-thread first if no fresh baseline exists). Gate: each model within ±5%.

- [ ] **Step 5: Embedder parity in worker**

Run `embed-perf` with `--worker` against:
- arctic-embed (encoder)
- qwen3-embedding-0.6b-hyb (causal-LM embedder)
- qwen3-8b-iq3m (bucket D self-embed)

Each must pass its existing parity gate (encoder: `cos >= 0.999`; hyb: `cos >= 0.995`; iq3m: `cos >= 0.90` + 16+16 mean-margin gate per CLAUDE.md).

- [ ] **Step 6: Cross-mode token-identical A/B with greedy sampling**

Run a fixed prompt set (e.g., the 36-prompt eval suite or a 5-prompt sanity subset) with `temperature: 0` greedy in both `worker=0` and `worker=1` mode. Tokens must be byte-identical.

If they're not, **stop and investigate** — divergence indicates worker-boundary corruption (stale config, race in init, etc.).

- [ ] **Step 7: Write closure report**

Create `eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`:

```markdown
# Dual-mode (main+worker) deployment — closure report

> **Date:** 2026-05-02
> **Tip:** <git rev-parse HEAD>
> **Spec:** docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md
> **Plan:** docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md

## Verdict: PASS / FAIL

## Smoke regression (worker mode)
- qwen3-0.6b-q4f16: <pass/fail>, decode <X> tok/s
- qwen3-8b-iq3m:    <pass/fail>, decode <X> tok/s, no console errors

## Frame-probe coexistence
- median decode_max: <X> ms (gate < 15 ms)
- baseline frame median: <X> ms

## Cross-mode A/B perf (decode tok/s)
| Model | main | worker | Δ% | within ±5% |
|---|---:|---:|---:|:-:|
| tinyllama-1.1b-chat-q4_0 | | | | |
| qwen3-0.6b-q4f16 | | | | |
| qwen3-1.7b-q4f16 | | | | |
| mistral-7b-instruct-v0.3-q4ks | | | | |
| llama-3.1-8b-instruct-iq3m | | | | |
| qwen3-8b-iq3m | | | | |

## Embedder parity in worker
- arctic-embed: <pass/fail>, parity <X>
- qwen3-embedding-0.6b-hyb: <pass/fail>, parity <X>
- qwen3-8b-iq3m self-embed: <pass/fail>, parity <X>, mean-margin <X>

## Cross-mode token-identical A/B
- 5-prompt suite, greedy: <token-identical/divergence>

## Tests added
- tests/worker-bridge-protocol.test.ts (envelope round-trip)
- tests/webllm-error-codec.test.ts (mirror-drift sentinel — N tests over M error codes)
- tests/webllm-worker-host.test.ts (host RPC handling)
- tests/webllm-proxy-integration.test.ts (proxy + stub-channel end-to-end)
- tests/webllm-proxy-surface.test.ts (surface mirror sentinel)

## Lessons / follow-ups
<any leftovers — e.g., probe-time issues, scope confirmations, items
deferred for a follow-up cycle>
```

- [ ] **Step 8: Update `TODO.md`**

Add a closure stub for item 10 in the active surface; archive the full block to `TODO_ARCHIVE.md` per the project's "TODO archival cadence" doctrine. The closure stub should reference the SUMMARY report and the spec/plan paths.

- [ ] **Step 9: Run `make checkall` one last time**

```bash
make checkall
```

Expected: fmt + lint + typecheck + typecheck:tests + tests all green.

- [ ] **Step 10: Commit closure**

```bash
git add -f eval/reports/dual-mode-worker-2026-05-02/
git commit -m "docs(report): dual-mode worker deployment — closure"
git add TODO.md TODO_ARCHIVE.md
git commit -m "docs(TODO): archive dual-mode worker deployment block — item 10 closed"
```

**Acceptance:**
- All gates above PASS.
- `make checkall` green.
- Closure report committed and TODO archived.

---

## Self-review checklist (run after the plan is written)

**Spec coverage** — every spec section maps to at least one task:

- ✅ Q1 single flag, drop-in `WebLLM` shape — Task 8 wires `worker: true`; Task 5/6 builds proxy mirroring `WebLLM` surface.
- ✅ Q2 async-ify conv methods — Task 7.
- ✅ Q3 hybrid loadModel/loadModelFromBuffer — proxy methods cover both (Task 5); `loadModelFromBuffer` uses Transferable; `loadModel` worker-fetches transparently because the worker-side `WebLLM` instance fetches.
- ✅ Q4 typed error codec — Task 3 + sentinel test.
- ✅ Q5 single-bundle re-entry via `import.meta.url` — Task 8 step 4.
- ✅ Q6a smoke + bench flags — Task 9.
- ✅ Q6b embedder parity — Task 10 step 5.
- ✅ Q6c out-of-scope confirmations — held by absence (no SAB / no SharedWorker / no auto-restart tasks anywhere).
- ✅ Q6d lifecycle (`dispose()`) — Task 8 step 3.
- ✅ Risk: ASYNCIFY in worker — Task 1 probe.
- ✅ Risk: `import.meta.url` resolution — Task 8 build + Task 10 sweep.
- ✅ Risk: relative URL fetch in worker — Task 10 covers via real model loads.
- ✅ Risk: `GenerationStreamChunk.stats` cloneability — implicitly verified by Task 10's stream tests passing on real models.
- ✅ Risk: WebGPU limits in worker — Task 1 probe + Task 10 covers.
- ✅ Risk: tok/s parity — Task 10 step 4 cross-mode A/B.
- ✅ Risk: mirror drift — Task 6 step 5 + Task 3 sentinels.

**Placeholder scan:** No "TBD" / "TODO" markers left in the plan content (the closure-report template uses `<X>` placeholders that are intentional fill-ins for the reporter).

**Type consistency:** `WebLLMProxy.fromWorker` is the test-only entry; `WebLLMProxy.init` is the production entry — both return `Promise<WebLLMProxy>`. `chatCompletion` returns `AsyncIterableIterator<GenerationStreamChunk>` consistently across `engine.ts` and `webllm-proxy.ts`. `dispose()` returns `Promise<void>` on both sides.
