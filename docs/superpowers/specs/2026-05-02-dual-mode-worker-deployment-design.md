# Dual-mode (main-thread + worker) deployment — design

> **Date:** 2026-05-02
> **Status:** Spec — awaiting plan author
> **Trigger:** Probe 9d (closed 2026-05-01) measured a 5.5× decode-hitch
> reduction (49.8 ms → 9.1 ms median) by running the engine inside a
> `DedicatedWorker`. Item 10 in `TODO.md` "Next session pickup" — gate
> PASSED, dual-mode work is justified by data.
> **Related artifacts:**
> - Probe 9d closure: [`eval/reports/probe-9d-2026-05-01/SUMMARY.md`](../../../eval/reports/probe-9d-2026-05-01/SUMMARY.md)
> - Probe 9d worker spike: [`smoke-test/probe-9d-worker.js`](../../../smoke-test/probe-9d-worker.js) + [`smoke-test/probe-9d.html`](../../../smoke-test/probe-9d.html)
> - Public surface inventory: [`src/index.ts`](../../../src/index.ts)
> - Engine: [`src/core/engine.ts`](../../../src/core/engine.ts)

## Goal

A single configuration flag (`WebLLM.init({ worker: true })`) flips engine
construction so all WebGPU + ggml-wasm work runs in a `DedicatedWorker`.
The public TypeScript surface is identical between modes; consumer code
(smoke harness, eval runners, app code, `Character`) makes zero call-site
changes other than `await`-ing the conversation methods that are being
async-ified per the Q2 decision below.

The load-bearing use case is **agent + Three.js coexistence**: an LLM
drives NPCs through tool calls inside the same tab as a Three.js render
loop. Probe 9d confirmed that moving the engine off-main-thread fully
absorbs the deterministic per-call decode hitch from the render-loop
perspective.

## Decisions log (Q1–Q6)

| # | Topic | Choice | Rationale |
|---|---|---|---|
| Q1 | API surface | **A — single flag, drop-in `WebLLM` shape** | Smoke harness and eval/perf/bench want one-line `worker:true` flip; `WebLLM` public surface is small enough to mirror; alternatives (separate type, separate import) push branching cost onto consumers. |
| Q2 | Sync→async migration | **A1 — async-ify on main thread too** | Conversation methods become `Promise<...>` in both modes; worker proxy is signature-identical. Breaking change is mechanical and timed well (TS API audit fresh, no external `ConversationHandle` consumers). |
| Q3 | Model loading | **B3 — hybrid (`loadModel` worker-fetches, `loadModelFromBuffer` transfers)** | Worker keeps the heap-streaming loader for >2 GiB models (qwen3-8b-iq3m); buffer path keeps the existing API for small-model cases (probe 9d, embedders). |
| Q4 | Typed errors across boundary | **E1 — reconstruct typed `WebLLMError` main-thread via codec** | Bounded cost (≈8 subclasses); preserves `instanceof` checks; mirror-drift sentinel covers it (same pattern as `JsonSchemaParameterType`). |
| Q5 | Worker construction | **W3 — same-bundle re-entry via `import.meta.url`** | Single file matches today's `webllm-bundle.js` mental model; no Blob URL CSP issues; no extra config. W1 (separate `webllm-worker.js`) held as escape hatch if `import.meta.url` resolution fights with the build. |
| Q6 | Harness + scope | Confirmed | Smoke `?worker=1`, bench `--worker`, dashboard `mode` field; embedder parity all paths; no SAB / no cross-worker WebGPU / no SharedWorker / no auto-restart on crash; ±5% decode tok/s parity target. |

## Architecture

```
┌─────────────────────── main thread ─────────────────────────┐
│                                                             │
│  consumer code (Character / harness / app)                  │
│         │                                                   │
│         ▼                                                   │
│   WebLLM (proxy)   ──── methods marshal via postMessage     │
│         │                                                   │
│         │            ┌─────── DedicatedWorker ────────────┐ │
│         └─────────►  │  WebLLM (real)                     │ │
│                      │  ├─ ggml-wasm                      │ │
│                      │  ├─ ModelInference / Encoder /     │ │
│                      │  │    CausalLMEmbedder / Generator │ │
│                      │  ├─ KV-cache pool / conversations  │ │
│                      │  └─ WebGPU adapter + device        │ │
│                      └────────────────────────────────────┘ │
│                                                             │
│  Three.js + render loop (untouched, main thread)            │
└─────────────────────────────────────────────────────────────┘
```

**Single-file bundle (Q5/W3).** `webllm-bundle.js` becomes context-aware:
at module load it inspects `typeof DedicatedWorkerGlobalScope !== "undefined"
&& self instanceof DedicatedWorkerGlobalScope`. On main thread it exports
the public API. Inside a worker it installs a `message`/`error` handler
that constructs a real `WebLLM` instance and routes RPCs.

**Worker construction (Q5/W3).** `WebLLM.init({ worker: true })` does:

```ts
const w = new Worker(new URL(import.meta.url, import.meta.url), { type: "module" });
```

The same module URL is the worker entry. Boot RPC sends
`{ type: "init", config }`; worker calls `WebLLM.init({ ...config, worker: false })`
against itself and posts `{ type: "init-done" }`.

**Resource ownership.**
- WebGPU device, ggml-wasm heap, KV-cache pool, conversation handles,
  model weights → **worker only**.
- ConversationHandle tokens, model handle IDs, error codecs → **shared
  (opaque IDs main-thread, real objects worker-side)**.
- Three.js scene, frame loop, tool handlers, `Character` instance →
  **main only**.

**Data-flow constraint.** No cross-thread WebGPU sharing, no SAB. The
worker boundary is pure postMessage with structured clone for chunks /
handles / errors and `Transferable` for the one-shot `loadModelFromBuffer`
ArrayBuffer.

## Components

Five new units, each with one clear purpose. Existing files
(`engine.ts`, `model-inference.ts`, etc.) are **not modified**
structurally — they continue to be the worker-side implementation.

### 1. `src/core/worker-bridge.ts` — message protocol & types

Shared by both sides. Defines the discriminated union of
request/response messages, request IDs, the `SerializedError` shape,
and the `MessageEnvelope<T>` wrapper. Pure type definitions plus a tiny
request-ID counter. ~150 LOC. Both `webllm-proxy.ts` and
`webllm-worker-host.ts` import from here.

### 2. `src/core/webllm-proxy.ts` — main-thread façade

Implements the public `WebLLM` shape (post-Q2 async-ification). One
class, `WebLLMProxy`, with the same method names as `WebLLM`. Internally:

- Owns the `Worker` instance and the pending-request `Map<RequestId, {resolve, reject}>`.
- Each plain method becomes `postMessage({type: "method-call", id, name, args})` + Promise wrapper.
- `chatCompletion` / `generateStream` return `AsyncIterableIterator<GenerationStreamChunk>`
  backed by a per-call queue; chunks arrive as `{type: "stream-chunk", streamId, chunk}`
  and the iterator yields/awaits them.
- `dispose()` posts shutdown, then `worker.terminate()` and rejects all pending.
- Catch-side: rebuilds typed `WebLLMError` subclasses from the `code`
  field via `webllm-error-codec`.

### 3. `src/core/webllm-worker-host.ts` — worker-side message handler

Boots when the bundle detects worker context. Constructs the real
`WebLLM` (the existing `engine.ts` class, untouched) and routes RPCs:

- `init` → `WebLLM.init({ ...config, worker: false })`, stash result.
- `method-call` → reflect-dispatch to the engine method by name with
  the deserialized args; serialize the result.
- `stream-start` for `chatCompletion`/`generateStream` → drive the real
  async-iterator; for each chunk post `{type: "stream-chunk", streamId, chunk}`;
  on iterator end post `{type: "stream-done", streamId, value}`.
- Error path: catch, serialize via the codec, post
  `{type: "method-error", id, error}` or `{type: "stream-error", streamId, error}`.

### 4. `src/core/webllm-error-codec.ts` — error round-trip

Pure functions:

```ts
export interface SerializedError {
    code: WebLLMErrorCode | "GENERIC" | "DISPOSED";
    message: string;
    stack?: string;
    modelId?: string;
    conversationId?: string;
    poolSize?: number;
    contextTokens?: number;
    maxContextTokens?: number;
}

export function serializeError(e: unknown): SerializedError;
export function reconstructError(s: SerializedError): WebLLMError | Error;
```

`reconstructError` is a single switch on `code` that builds the matching
subclass: `ModelNotFoundError(modelId)`, `ConversationBusyError(conversationId)`,
`ConversationContextOverflowError(conversationId, contextTokens, maxContextTokens)`,
etc. Stack carries both sides — the reconstructed error's `stack` field
is set to `worker_stack + "\n    at <postMessage boundary>\n" + main_stack`.

This is the **mirror-drift sentinel for Q4**.

### 5. `src/core/engine.ts` — touched only at the entry point

`WebLLM.init(config)` gains the `worker: boolean` branch:

```ts
if (config.worker && !isWorkerContext()) {
    return WebLLMProxy.init(config);
}
// existing main-thread path unchanged
```

The `createConversation` / `disposeConversation` / `forkConversation`
methods are async-ified (Q2 → A1) — these are the only public-API
breaking changes in `engine.ts`.

### What's deliberately NOT a new component

No `WorkerEngine` second class. No second bundle file. No tool-call
mid-stream RPC. No SharedArrayBuffer. No worker pool.

## Public API surface

### Change 1 — `WebLLMConfig` gains `worker?: boolean`

```ts
export interface WebLLMConfig {
    // ...existing fields...
    /**
     * Run engine in a DedicatedWorker. Default false.
     * When true, WebGPU + ggml-wasm execute off-main-thread; the
     * returned WebLLM is a proxy. All public methods retain their
     * signatures (see "async-ified conversation methods").
     */
    worker?: boolean;
}
```

`WebLLM.init(config)` returns `Promise<WebLLM>` regardless. When
`worker: true`, the returned object is a `WebLLMProxy` instance whose
TypeScript type is structurally compatible with `WebLLM`.

### Change 2 — async-ified conversation methods (Q2 → A1)

Breaking change to `WebLLM`:

```ts
// Before
createConversation(modelHandleId: string, opts?: ConversationOptions): ConversationHandle;
disposeConversation(conv: ConversationHandle): void;
forkConversation(src: ConversationHandle): ConversationHandle;

// After
createConversation(modelHandleId: string, opts?: ConversationOptions): Promise<ConversationHandle>;
disposeConversation(conv: ConversationHandle): Promise<void>;
forkConversation(src: ConversationHandle): Promise<ConversationHandle>;
```

`ConversationHandle` itself is unchanged — `{id, modelHandleId, ...}`.
In worker mode the handle is opaque main-thread (the `id` references a
worker-side object), but its shape is the same.

### Unchanged signatures (already async or stream)

- `WebLLM.init` (already `Promise`)
- `loadModel` / `loadModelFromBuffer` / `unloadModel` (all `Promise`)
- `chatCompletion` (already `AsyncGenerator<GenerationStreamChunk>`)
- `embed` (already `Promise<Float32Array>`)
- `chat` / `generateStream` (already async or async-iterable)

### New method — `dispose()`

```ts
/**
 * Releases all engine resources. In worker mode, also terminates
 * the worker. In main-thread mode, releases WebGPU device + frees
 * all loaded models. After dispose(), all subsequent method calls
 * throw WebLLMError("DISPOSED").
 */
dispose(): Promise<void>;
```

Required by worker mode; also useful main-thread for explicit cleanup.
`unloadModel(id)` keeps single-model semantics; `dispose()` is engine-wide.

### What does NOT change

- `Character`, `CharacterManager`, `ToolSystem`, `ToolDefinition` —
  untouched. Tools live main-thread; `Character` calls
  `engine.chatCompletion` which is already async-iterable.
- `ConversationHandle`, `KVSnapshot`, `WebLLMError` taxonomy — untouched.
- `eval/runner.ts`, `score`, custom-scorer registrations — untouched.
- `loadModel(url)` semantics — same. Worker fetches the URL and
  heap-streams.
- `loadModelFromBuffer(buf)` semantics — same. Worker mode transfers
  the buffer as a Transferable.

## Data flow / postMessage protocol

Single MessageChannel between proxy and worker. All envelopes carry a
`type` discriminator and (where applicable) an `id` for request/response
correlation.

### Message envelope (shared types in `worker-bridge.ts`)

```ts
type RequestId = number;
type StreamId = number;

type ProxyToWorker =
    | { type: "init"; id: RequestId; config: Omit<WebLLMConfig, "worker"> }
    | { type: "method-call"; id: RequestId; name: string; args: unknown[] }
    | { type: "stream-start"; streamId: StreamId; name: "chatCompletion" | "generateStream"; args: unknown[] }
    | { type: "stream-cancel"; streamId: StreamId }
    | { type: "dispose"; id: RequestId };

type WorkerToProxy =
    | { type: "init-done"; id: RequestId }
    | { type: "method-result"; id: RequestId; value: unknown }
    | { type: "method-error"; id: RequestId; error: SerializedError }
    | { type: "stream-chunk"; streamId: StreamId; chunk: GenerationStreamChunk }
    | { type: "stream-done"; streamId: StreamId; value?: unknown }
    | { type: "stream-error"; streamId: StreamId; error: SerializedError }
    | { type: "log"; level: "info" | "warn" | "error"; message: string };
```

### Lifecycles

**1. `WebLLM.init({ worker: true, ...config })`:**
- Main constructs `Worker(new URL(import.meta.url, import.meta.url), { type: "module" })`.
- Posts `{ type: "init", id: 1, config }`.
- Worker runs the same bundle; entry sees `DedicatedWorkerGlobalScope`,
  installs message handler, runs
  `WebLLM.init({ ...config, worker: false })`, stashes the engine,
  posts `{ type: "init-done", id: 1 }`.
- Main resolves the init Promise with a `WebLLMProxy` wrapping the worker.

**2. Plain method call — e.g., `await proxy.createConversation(modelId, opts)`:**
- Proxy posts `{ type: "method-call", id, name: "createConversation", args: [modelId, opts] }`.
- Worker reflect-dispatches `engine.createConversation(modelId, opts)`,
  awaits it.
- On success: posts `{ type: "method-result", id, value: handle }`. Proxy
  resolves with the handle (handle is plain JSON-serializable).
- On error: serialize via codec, post `{ type: "method-error", id, error }`.
  Proxy reconstructs typed error and rejects.

**3. Streamed method — `for await (const chunk of proxy.chatCompletion(...))`:**
- Proxy assigns a fresh `streamId`, posts
  `{ type: "stream-start", streamId, name: "chatCompletion", args }`.
- Proxy returns an `AsyncIterableIterator<GenerationStreamChunk>` whose
  `next()` awaits the next item from a per-stream queue.
- Worker drives the real `engine.chatCompletion(...)` async-iterator.
  For each chunk: post `{ type: "stream-chunk", streamId, chunk }`. On
  natural completion: post `{ type: "stream-done", streamId }`. On thrown
  error: post `{ type: "stream-error", streamId, error }`.
- Proxy queues incoming chunks; each `next()` either resolves immediately
  from the queue or registers a pending Promise.
- **Cancellation:** if the consumer breaks out of `for await` early, the
  proxy posts `{ type: "stream-cancel", streamId }`. Worker uses an
  `AbortController` (the `config.signal` already exists in
  `chatCompletion`) to stop generation. Worker still posts `stream-done`
  to release the proxy queue.

**4. `loadModelFromBuffer(buf, ...)`:**
- Proxy posts
  `{ type: "method-call", id, name: "loadModelFromBuffer", args: [buf, modelId, opts, wasmUrl, loadOpts] }`
  with `[buf]` in the `transferList` argument of `postMessage`. Buffer
  ownership moves to the worker; main-thread reference becomes detached
  (intentional — caller must not reuse).
- Result handle returns via `method-result` like any other call.

**5. `loadModel(url, ...)`:**
- Proxy posts `{ type: "method-call", id, name: "loadModel", args: [url, ...] }`.
- Worker fetches the URL itself (heap-streams into wasm64 for >2 GiB
  models, matching the existing main-thread loader). No buffer crosses
  the postMessage boundary.

**6. `dispose()`:**
- Proxy posts `{ type: "dispose", id }`. Worker awaits
  `engine.dispose()` (releases WebGPU device + frees models), posts
  `{ type: "method-result", id }`.
- Proxy then calls `worker.terminate()` and rejects all pending
  requests/streams with `WebLLMError("DISPOSED")`.

### Throughput sanity check

- Decode tok/s: 25–100 → 25–100 stream-chunk envelopes/sec, ~50 bytes
  structured-clone each. Lost in the noise.
- Per-token postMessage round-trip cost: ≤1 ms (Chrome structured clone
  of small object). Within the ±5% parity target.
- No backpressure mechanism. Consumer always drains via `for await`; if
  they don't, that's a consumer bug. Worker doesn't pause generation
  waiting for ack.

### Quirk worth flagging during implementation

`GenerationStreamChunk.stats` (final chunk) carries some non-cloneable
internals if any subsystem put a class instance there. Audit during
implementation: stats must be a plain object. If any field fails
structured clone, fix the source — don't add a serializer. Surface area
is `Generator` only.

## Error handling

### Worker-side untyped throws

Anything not a `WebLLMError` (e.g., a `RangeError` from a malformed
GGUF) serializes with `code: "GENERIC"` and reconstructs as a plain
`Error` with the worker-side stack preserved as `cause`. Consumer's
`instanceof WebLLMError` check returns false for these, matching today's
main-thread behavior.

### Worker crash / termination

If the worker emits `error` (uncaught exception) or `messageerror`
(un-cloneable message), the proxy:

1. Marks the engine as disposed (no auto-restart, per Q6c).
2. Rejects all pending requests with
   `new WebLLMError("DISPOSED", "worker crashed: " + workerErr.message)`.
3. Closes all active stream queues with the same error.
4. Calls `worker.terminate()`.

Consumer recovery is to `WebLLM.init` a fresh engine.

### Disposed-state guard

Every proxy method's first line is
`if (this.disposed) throw new WebLLMError("DISPOSED");` — covers
post-dispose and post-crash paths uniformly.

## Testing

### Unit tests (Bun, no browser)

- `tests/webllm-error-codec.test.ts` — mirror-drift sentinel:
  parametrizes over a `Map<WebLLMErrorCode, () => WebLLMError>` factory
  table that mirrors `errors.ts`, round-trips each subclass through
  `serializeError → JSON.parse(JSON.stringify(...)) → reconstructError`,
  asserts `instanceof OriginalSubclass`, `code`, `message`, and all
  subclass-specific fields preserved.
- `tests/webllm-proxy-surface.test.ts` — reflection test: every public
  method on `WebLLM.prototype` exists on `WebLLMProxy.prototype` with
  matching arity.
- `tests/worker-bridge-protocol.test.ts` — round-trip tests for the
  message envelope discriminated union: every `ProxyToWorker` and
  `WorkerToProxy` variant survives `structuredClone`.

### Integration tests (Bun + Worker mock, no browser)

- `tests/webllm-proxy-integration.test.ts` — instantiates a
  `WebLLMProxy` with a stub worker (in-process EventTarget on both
  sides), drives `init` → `createConversation` → `chatCompletion` →
  `disposeConversation` → `dispose`, asserts call sequencing, error
  propagation, and `for await` cancellation. Uses a fake engine on the
  "worker" side that returns canned chunk streams. **Does not touch
  WebGPU.**

### Browser smoke (real WebGPU, real worker)

- `smoke-test/real-model.html?worker=1` — boots
  `WebLLM.init({ worker: true, ... })`, runs the existing
  `[1/8]…[8/8]` chat regression sequence end-to-end on
  `qwen3-0.6b-q4f16` first, then `qwen3-8b-iq3m` (gates the
  heap-streaming loader path under worker mode).
- Same page, `?worker=1&frameProbe=1&scene=...` — coexistence test.
  Asserts decode_max stays in the 9.1 ms-median band probe 9d measured
  (gate: median decode_max < 15 ms; 5.5× headroom over main-thread
  49.8 ms).
- Cross-mode A/B: `?worker=0` and `?worker=1` runs of the same prompt
  produce identical token streams (deterministic decode, greedy
  sampling). Catches subtle worker-boundary corruption.

### Bench parity

- `eval/perf.ts --worker` and without — same canonical 6 fleet, decode
  tok/s within ±5%. Live dashboard records `mode` per run; perf table
  A/Bs them.
- `eval/embed-perf.ts --worker` — encoder, causal-LM embedder, bucket D
  self-embed all pass parity gates from worker context.
- `eval/bench.ts --worker` — `bench-full` 6 profiles × 3 thinking-on/off
  run identically in worker mode; accuracy delta within sampling noise.

### Tests deliberately NOT added

- No SharedArrayBuffer / cross-tab tests (out of scope).
- No SharedWorker tests (out of scope).
- No worker-pool / multi-engine tests (single-model-active per CLAUDE.md).
- No auto-restart tests (Q6c — crashes throw, consumer rebuilds).

## Out of scope

- **SharedArrayBuffer weight sharing across multiple workers.**
  Single-model-active per CLAUDE.md.
- **Cross-worker WebGPU resource handoff.** No consumer.
- **`SharedWorker` multi-tab inference.** No consumer.
- **Worker pool / multiple engines.** One Worker per `WebLLM` instance,
  period.
- **Auto-restart on worker crash.** Proxy throws `DISPOSED` on next
  call; consumer rebuilds.
- **Mid-stream tool-call RPC.** Tools live main-thread (`Character`);
  engine just emits `<tool_call>` tokens. No worker-side tool dispatch.
- **Backwards-compat shim for sync conversation methods.** Q2 → A1 is
  a clean break.
- **Persisting handles across page reload.** Out of scope; matches
  prefix-cache follow-up #5 status.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ASYNCIFY behaves differently in `DedicatedWorker` context | Low | High (would block the whole effort) | Probe 9d already booted ASYNCIFY in a worker successfully. First plan task: replicate the 1-token forward with the production engine config to lock in the result before plumbing the proxy. |
| `import.meta.url` doesn't resolve to a worker-loadable URL after bundling | Med | Med | Build emits an ES module (already does). If `import.meta.url` is wrong post-bundle, fall back to `WebLLMConfig.workerUrl` override (W1 escape hatch). Probe with a no-op worker init in CI. |
| `loadModel(url)` worker-fetch fails on relative URLs (`./models/foo.gguf`) | Med | Med | Worker `self.location.href` is the worker bundle URL. Resolve URLs against `import.meta.url` at the call site. Smoke test catches this. |
| `GenerationStreamChunk.stats` contains a non-cloneable field | Low | Low | Audit during component-4 implementation. If found, fix at source (make plain object). Don't add a serializer. Surface area is `Generator` only. |
| Worker WebGPU `maxStorageBufferBindingSize` differs from main thread | Low | High (would affect quant choice) | Re-run the existing `[diagnoseAlloc]` probe from worker context as part of init. Log mismatch. Per CLAUDE.md "per-binding 128 MiB cap doctrine" — no behavior change expected. |
| Decode tok/s parity worse than ±5% under postMessage | Low | Med | Probe 9d showed structural hitch reduction with no decode regression on `qwen3-0.6b-q4f16`. Plan adds a same-day same-tip A/B on canonical 6 before declaring done. If parity fails, investigate before shipping. |
| Mirror-drift between `WebLLM` and `WebLLMProxy` over time | Med (long-term) | Low | Surface-reflection test (`webllm-proxy-surface.test.ts`) fails CI when methods are added to `WebLLM` without updating the proxy. Same pattern as `JsonSchemaParameterType` mirror sentinel. |

## Rollout / phasing (sketch — full breakdown belongs in the plan)

Rough cuts so the plan author has a starting point:

1. **Probe**: ASYNCIFY-in-worker re-confirm with the production engine
   (~1 task).
2. **Foundation**: `worker-bridge.ts` types + `webllm-error-codec.ts` +
   tests (~1 task, parallelizable).
3. **Bundle entry context detection** + `WebLLM.init` worker branch
   (~1 task).
4. **Proxy class — non-streaming methods** (init, loadModel,
   loadModelFromBuffer, embed, createConversation, disposeConversation,
   forkConversation, unloadModel, dispose) + integration tests with
   stub worker (~1 task).
5. **Proxy class — streaming** (chatCompletion, generateStream) +
   cancellation + integration tests (~1 task).
6. **Async-ify conversation methods on main-thread `WebLLM`** + update
   internal call sites (smoke harness, eval runners) — clean break, no
   shim (~1 task).
7. **Smoke + bench harness flags** (`?worker=1`, `--worker`, dashboard
   `mode` field) (~1 task).
8. **Browser regression sweep**: `qwen3-0.6b-q4f16` then
   `qwen3-8b-iq3m` end-to-end, frame-probe coexistence, embedder parity
   (~1 task).
9. **Cross-mode A/B perf** on canonical 6 + closure report (~1 task).

Tasks 2 and 4–5 have natural parallelism (codec + bridge types are
independent of the proxy class shell).

## Success criteria

The work is "done" when **all** of these hold:

- `make checkall` passes (fmt + lint + typecheck + tests, including the
  new mirror-drift + surface-reflection sentinels).
- `smoke-test/real-model.html?worker=1` runs the `[1/8]…[8/8]` chat
  regression on both `qwen3-0.6b-q4f16` and `qwen3-8b-iq3m` with no
  console errors.
- `?worker=1&frameProbe=1` median decode_max < 15 ms (worker-mode hitch
  absorbed; 5.5× headroom over main-thread 49.8 ms baseline).
- `eval/perf.ts --worker` decode tok/s on canonical 6 within ±5% of the
  same-day main-thread baseline.
- All three embed paths (encoder, causal-LM embedder, bucket D
  self-embed) pass parity gates from worker context.
- Cross-mode A/B with greedy sampling produces token-identical outputs
  on a fixed prompt set.
- Closure report at `eval/reports/dual-mode-worker-<DATE>/SUMMARY.md`
  with all of the above as a single artifact.
