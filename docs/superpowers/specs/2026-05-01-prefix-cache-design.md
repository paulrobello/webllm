# Prefix-cache via per-conversation KV snapshots — design

> **Date:** 2026-05-01
> **Author:** brainstormed via `superpowers:brainstorming`
> **Driven by:** probe 9a (PASS, 89.7% prefix share) + probe 9b
> (PARTIAL — sequential canonical, hard-dependent on prefix caching)
> **Status:** design approved by user; awaiting implementation plan

## Problem

The NPC + Three.js coexistence target (per CLAUDE.md "agent + 3D
agent" use case) needs `chatCompletion` calls that re-use the
stable `[system, tools, ...]` prefix across ticks instead of
re-prefilling it on every call. Probe 9a measured prefix at
**89.7%** of canonical NPC-tick prefill cost (a = 12.31 ms /
prefix-token, b = 14.11 ms / tail-token at qwen3-8b-iq3m). At
~5500 ms / tick without prefix caching, the harness blows a
1 Hz NPC tick budget by 5.5×. With prefix caching, per-tick
collapses to ~150 ms (75 ms tail-only prefill + ~75 ms decode of
1-2 tokens) — 6+ Hz comfortably.

The current engine has **single-conversation** prefix detection
(`prepareChatPrompt`, `engine.ts:533`) that reuses KV when the
new messages array strictly extends the prior one. It does
**not** handle:

- **Diverging messages with shared token prefix** (e.g. switching
  NPCs mid-tick — same `[system, tools]` header, different
  `[ctx, observation]` tail).
- **Multiple concurrent conversations on the same loaded model**
  (CLAUDE.md "single-model-active" doctrine deferred this; probe
  9a's PASS verdict promotes it to load-bearing).

Both gaps are required for the agent + Three.js use case (NPCs
need their own dialog state, and tick scenarios need
cross-conversation prefix sharing of the system header).

## Use cases (per user, design Q1 = "c")

The design must cover both:

- **(a) Per-tick stateless reasoning across NPCs.** Each tick
  builds a fresh `[system, tools, world_state, npc_obs]` prompt.
  NPCs share the `[system, tools]` prefix; per-NPC tails differ.
- **(b) Multi-turn dialog per NPC.** Each NPC has its own dialog
  history that grows over time. NPCs share the system prefix;
  each NPC also has its own multi-turn message stream.

## Non-goals (v1)

- LRU eviction. v1 throws `ConversationPoolFullError`; caller
  manages lifecycle.
- Cross-conversation prefix sharing (zero-copy KV region for the
  shared `[system, tools]` header). Each conversation pays the
  prefix once on first tick.
- GPU-resident-per-conversation KV (Storage B). v1 ships
  copy-based snapshot-and-swap (Storage A); zero `ggml-webgpu`
  patches.
- Concurrent in-flight calls per conversation. v1 throws
  `ConversationBusyError`.
- Persistence across page reloads.
- Worker migration. v1 lives main-thread; item 10 wraps later.

These are sized for the **2-4 concurrent NPCs** target the user
specified, with "may scale larger" handled by adding LRU + Storage
B as v2 follow-ups once v1 ships and the API stabilizes.

## Architecture

### 1. Public API surface

Three additions to `WebLLM`, fully back-compatible. Existing
`chatCompletion(modelHandleId, ...)` keeps working unchanged.

```typescript
/**
 * Allocate a conversation handle backed by a per-conversation KV
 * snapshot. No KV memory is allocated until the first chatCompletion
 * call. Caller owns the lifecycle.
 */
createConversation(
  modelHandleId: string,
  options?: ConversationOptions,
): ConversationHandle;

/**
 * Release the conversation's KV snapshot. Idempotent.
 */
disposeConversation(conv: ConversationHandle): void;

/**
 * Stateful chatCompletion overload. Engine swaps conv's snapshot
 * into the model's working KV before prefill+decode, snapshots
 * the result back on completion. messages is the full canonical
 * history each call — engine finds the longest-shared-prefix vs
 * conv's prior tokens and only prefills the divergent tail.
 */
chatCompletion(
  conv: ConversationHandle,
  messages: ChatMessage[],
  config?: CompletionConfig,
): AsyncGenerator<CompletionChunk, void>;
```

```typescript
interface ConversationOptions {
  /** Maximum KV tokens for this conversation. Default: model contextLength. */
  maxContextTokens?: number;
}
interface ConversationHandle {
  readonly id: string;
  readonly modelHandleId: string;
}
```

The new `chatCompletion(conv, ...)` overload sits next to the
existing `chatCompletion(modelHandleId, ...)`. Discriminated by
argument shape (handle is structured object, modelHandleId is
string).

### 2. Internal architecture

New module `src/core/conversation-pool.ts`:

```typescript
interface KVSnapshot {
  conversationId: string;
  modelHandleId: string;
  tokenIds: number[];     // canonical tokens that produced this KV state
  kvBytes: Uint8Array;    // serialized per-layer K+V tensor data
  byteSize: number;
  lastAccessMs: number;   // for v2 LRU
}

class ConversationPool {
  private snapshots = new Map<string, KVSnapshot>();
  private locks = new Map<string, Promise<void>>();  // per-conversation in-flight guard
  private nextId = 1;

  create(modelHandleId: string, options?: ConversationOptions): ConversationHandle;
  dispose(conv: ConversationHandle): void;
  get(conv: ConversationHandle): KVSnapshot | undefined;
  set(conv: ConversationHandle, snapshot: KVSnapshot): void;
  has(conv: ConversationHandle): boolean;
  acquireLock(conv: ConversationHandle): Promise<() => void>;  // returns release fn
  disposeAllForModel(modelHandleId: string): void;  // model unload hook
}
```

### 3. Per-call protocol

`chatCompletion(conv, messages, opts)`:

1. Validate `conv` exists in pool; if not, throw
   `ConversationNotFoundError` (covers both never-created and
   auto-disposed-on-model-unload).
2. Acquire per-conversation lock. If already held, throw
   `ConversationBusyError` (no queueing in v1).
3. Acquire per-model lock (engine-wide; serializes calls across
   conversations on the same model since they share one working
   KV cache).
4. Tokenize the chat-template-formatted `messages` →
   `newTokens: number[]`.
5. Look up `conv`'s prior snapshot. If present, compute
   longest-shared-token-prefix between `snapshot.tokenIds` and
   `newTokens` → `sharedLen` (≤ both lengths).
6. **Load phase**: if `sharedLen > 0`, copy the first `sharedLen`
   tokens' worth of KV bytes from `snapshot.kvBytes` into the
   model's working KV cache via `inf.loadKVCache(...)`; set
   `inf.nCached = sharedLen`. Else `inf.resetKVCache()`.
7. **Prefill phase**: forward the divergent tail
   `newTokens.slice(sharedLen)`. KV cache grows to
   `newTokens.length`.
8. **Decode phase**: standard streaming generation. KV grows by
   each generated token.
9. **Save phase** (in the final-chunk handler): serialize the
   working KV's positions `[0, finalLen)` via
   `inf.serializeKVCache(finalLen)` into a fresh `Uint8Array`,
   replace `snapshot.kvBytes`, update `snapshot.tokenIds = newTokens
   ++ generatedIds`, update `snapshot.lastAccessMs`. Release locks.

If any phase throws, release locks in `finally`.

### 4. New `ModelInference` methods

Single source of truth for KV memory layout:

```typescript
/**
 * Serialize positions [0, nTokens) of every layer's K and V into a
 * flat Uint8Array. Layout: [layer0.K | layer0.V | layer1.K | ...].
 * The returned buffer is freshly allocated; caller owns it.
 */
serializeKVCache(nTokens: number): Uint8Array;

/**
 * Inverse of serializeKVCache. Writes positions [0, nTokens) into
 * every layer's K/V from the supplied buffer. Sets nCached = nTokens.
 * Throws if bytes.byteLength doesn't match the expected size for
 * (nTokens, nLayers, nHeadsKV, headDim, kvDtype).
 */
loadKVCache(bytes: Uint8Array, nTokens: number): void;
```

Both operate on the existing `LayerKVCache` (`model-inference.ts:63`)
memory layout — no new tensor allocation, just memcpy in/out of
the WASM heap.

Round-trip identity is the load-bearing correctness invariant: a
prefill, then `serializeKVCache(N)` → `resetKVCache()` →
`loadKVCache(bytes, N)`, then a forward pass at position N+1 must
produce **bit-identical logits** vs. running the same prefill +
forward without the round-trip (within fp tolerance, but
typically exact since memcpy is lossless and forward is
deterministic).

### 5. Storage layout (v1 = Storage A, copy-based)

Snapshots live as plain `Uint8Array` on the JS heap. The KV
tensors are WebGPU buffers (GPU-resident); `serializeKVCache`
issues a GPU→CPU readback via the existing `tensorGetData`
primitive (`ggml-wasm.ts:470` — blocks via ASYNCIFY until the
readback completes), and `loadKVCache` issues a CPU→GPU upload
via `uploadToTensor` (`ggml-wasm.ts:479`). Both stage through
the WASM heap.

Apple unified memory makes both directions cheap (~10 GB/s
effective; no PCIe). Discrete-GPU systems (Windows/Linux non-
Apple) pay PCIe transfer cost; rough estimate 5-8 GB/s. Either
way, pre-implementation per-call overhead estimates at qwen3-8b-iq3m
sizing:

- Stateless tick (~500-token KV, ~37 MB): ~5-10 ms each direction.
- Long dialog (~2000-token KV, ~150 MB): ~20-40 ms each direction.
- Worst-case (4096-token KV, ~302 MB): ~40-80 ms each direction.

These are estimates; the validation probe measures real numbers.
The 10-160 ms range is acceptable next to the ~1100 ms savings
(≥7× better than the cost). Storage B (rebind-based, zero-copy)
is queued as a v2 lever once v1 stabilizes and per-call overhead
is measured against real harness usage.

### 6. Pool sizing

Default `maxConversations = 4` (sized to user-specified target).
Override via `engine.config.maxConversations` (new field). Each
conversation's snapshot byte budget is bounded by
`min(options.maxContextTokens, model.contextLength) ×
kvBytesPerToken`. At qwen3-8b-iq3m / ctx=4096: 302 MB / conv. At
qwen3-0.6b-q4f16 / ctx=4096: 18 MB / conv.

Pool-full behavior: `createConversation` throws
`ConversationPoolFullError` with `context = {liveConversationIds:
[...]}` so the caller can choose what to dispose.

## Lifecycle, errors, edge cases

### Lifecycle

- `createConversation(modelHandleId, options?)` — allocates handle
  with `id = "conv_<nextId>"`, no KV bytes yet. Throws
  `ModelNotFoundError` if the model isn't loaded,
  `ConversationPoolFullError` if pool at cap.
- First `chatCompletion(conv, ...)` call populates the snapshot.
- Subsequent calls follow the per-call protocol above (load → run
  → save).
- `disposeConversation(conv)` — releases snapshot bytes, removes
  from pool. Idempotent. Subsequent `chatCompletion(conv, ...)`
  throws `ConversationNotFoundError`.
- `engine.unloadModel(modelHandleId)` auto-disposes all
  conversations attached to that model via
  `ConversationPool.disposeAllForModel`.

### Errors (extend existing `WebLLMError` taxonomy)

| Error | Trigger |
|---|---|
| `ConversationNotFoundError` | Handle disposed or never created. (Also fired when caller holds a handle whose model was unloaded — auto-dispose path.) |
| `ConversationPoolFullError` | `createConversation` at cap. Carries `liveConversationIds`. |
| `ConversationContextOverflowError` | Prefill or decode would push KV past `maxContextTokens`. v1 throws; v2 may add `engine.truncateConversation`. |
| `ConversationBusyError` | Concurrent `chatCompletion` on same handle. |

### Edge cases

- **Empty messages array**: reject with existing `InvalidPromptError`.
- **maxContextTokens > model.contextLength**: clamp to model max,
  log warning once per conversation.
- **Token-prefix mismatch**: `sharedLen < snapshot.tokenIds.length`
  is fine — engine truncates snapshot to `sharedLen` before
  loading. Discarded tail is just a length marker; no buffer
  shrink.
- **Snapshot byte-size growth**: each save reallocates `kvBytes`
  to `nTokens × kvBytesPerToken`. JS engine handles fragmentation;
  v1 doesn't pool buffers.
- **Concurrent calls across conversations**: serialized via the
  per-model lock. Conv B's call awaits conv A's save phase
  before loading. Throughput equivalent to current
  single-conversation behavior.

## Testing

### Unit tests (Bun, no browser)

- `tests/conversation-pool.test.ts` — pool create / dispose / get
  lifecycle; pool-full error; idempotent dispose; auto-dispose on
  model unload.
- `tests/kv-snapshot-roundtrip.test.ts` — `serializeKVCache` /
  `loadKVCache` round-trip identity. Setup: load TinyLlama, prefill
  a fixed token sequence, serialize, reset KV, load back, run
  forward at the next position, assert logit equality (fp
  tolerance) vs the no-snapshot path. **Load-bearing correctness
  test** for the whole feature.
- `tests/chat-completion-conversation.test.ts` — `chatCompletion(conv,
  messages, ...)` with two conversations on the same model.
  Assertions:
  - **Isolation**: `conv1` then `conv2` then `conv1` returns the
    same logits for `conv1`'s second turn as a fresh `conv1` would
    have produced (no cross-talk via shared working KV cache).
  - **Prefix detection**: consecutive calls sharing a prefix
    trigger the truncate-and-prefill-tail path (assert via spy on
    `inf.resetKVCache` not being called when `sharedLen > 0`).
- Existing test suite (513/12/0) must keep passing — back-compat
  invariant for the existing `chatCompletion(modelHandleId, ...)`
  overload.

### Integration / probe

- New probe `eval/probes/probe-prefix-cache-validation-2026-05-DD.ts`:
  4-NPC sequential tick scenario on `qwen3-8b-iq3m`. Captures
  per-tick prefill latency for two patterns:
  1. Without conversation handles (current
     `chatCompletion(modelHandleId, ...)`, full re-prefill).
  2. With handles (new path).
  Pass: handles path lands per-tick prefill in the **75-150 ms band**
  that probe 9a's marginal-cost math projects (a · 40 + b · 40 ≈
  1100 ms baseline → 75-150 ms with cache).
  Writes
  `eval/reports/prefix-cache-validation-2026-05-DD/SUMMARY.md`.

The probe is the load-bearing real-world verification — unit
tests prove correctness, the probe proves the win is delivered.

## Known follow-ups (out of scope for v1; queued)

1. **LRU eviction.** Required when scaling beyond ~8 concurrent
   conversations. v1 throws `ConversationPoolFullError`; v2 evicts
   the oldest non-busy snapshot.
2. **Cross-conversation prefix sharing.** Each conversation pays
   the full `[system, tools]` prefix on first tick today.
   Optimization: detect identical prefix tokens across conversation
   snapshots and share a single KV region for the shared prefix.
   Estimated savings: 1100 ms × N on first-tick-per-NPC. Requires
   either C++ patches (zero-copy) or a "prefix prefill cache"
   with copy-from-prefix-store on conversation creation. Defer
   until probe data shows multi-NPC spawn cost is meaningful.
3. **Storage B (rebind-based).** Per-conversation GPU buffers,
   swap by rebinding rather than copying. Removes the ~14-60 ms
   per-call snapshot overhead. Requires `ggml-webgpu` patches
   (12th patch on the stack). Defer until per-call overhead is
   measured against real harness usage and the API + lifecycle
   have stabilized.
4. **Concurrent in-flight per conversation.** v1 throws
   `ConversationBusyError`. Some patterns (parallel speculative
   ticks, request-fanout) might want concurrent calls. v2 could
   clone the working KV at concurrency request time. Defer.
5. **Persistence across page reloads.** v1 conversation state
   lives in JS heap; reload loses it. v2 could serialize snapshots
   to IndexedDB with the app's opt-in. Defer until a consumer asks.
6. **Worker migration (item 10).** Conversation pool needs to
   live worker-side once dual-mode ships. v1 design doesn't cross
   thread boundaries; v2 wraps the public API in the worker
   bridge. Defer until item 10 starts.

## Cross-references

- Probe 9a (PASS): `eval/reports/probe-9a-2026-05-01/SUMMARY.md`
- Probe 9b (PARTIAL — sequential canonical): `eval/reports/probe-9b-2026-05-01/SUMMARY.md`
- TODO § Next session pickup item 10 (dual-mode worker; probe
  9d PASSED 2026-05-01) for the worker migration follow-up.
- Existing single-conversation prefix detection:
  `src/core/engine.ts:533` (`prepareChatPrompt`).
- KV cache primitives: `src/inference/model-inference.ts:209`
  (`kvLayers`), `:396` (`initKVCache`), `:454` (`resetKVCache`),
  `:465` (`truncateKVCache`).
