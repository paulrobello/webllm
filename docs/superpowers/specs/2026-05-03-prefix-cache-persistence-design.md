# Prefix-cache persistence across reloads — design

> **Date:** 2026-05-03
> **Author:** brainstormed via `superpowers:brainstorming`
> **Driven by:** TODO.md item 11 follow-up #5 (queued 2026-05-02 as
> "IndexedDB-backed snapshot store with app opt-in"); user-as-proxy
> consumer ask, scenario "end-user agent state" (durability matters,
> schema/version migration matters, quota errors must surface, opt-in
> per conversation).
> **Status:** design draft; awaiting user review and implementation
> plan.

## Problem

The `ConversationPool` (shipped 2026-05-02) lives in-memory engine-
side. Conversation handles, KV snapshots, and per-conv options
evaporate on page reload. Per item-11 closure, the prefix-cache
mechanism made interleaved-NPC tick wall-time savings load-bearing
(84% on `qwen3-8b-iq3m`); but every page reload pays the full
re-prefill on the user's next interaction, defeating the value
proposition for any app where the user closes the tab and returns
later (the canonical "end-user agent state" use case).

The follow-up needs to let an app round-trip a conversation's KV
state through any persistent storage the app chooses (IndexedDB,
OPFS, server-side sync, encrypted-at-rest). The engine should provide
the smallest defensible boundary — produce/consume opaque blobs —
and a thin optional helper for the IndexedDB plumbing most apps will
want.

## Use case (per Q1 = "b")

End-user agent state. Concretely:

- A user has a long-running NPC / chatbot built on top of WebLLM.
  They've spent N turns establishing context, persona, conversation
  history.
- They close the tab. They return tomorrow.
- The app calls `webllm.importConversation(modelId, blob)` against
  the previously-saved blob and continues from where the user left
  off — no re-prefill of the established context, no re-establishment
  of persona.

Failure modes the design must handle gracefully (because they all
arise in real deployments): user upgrades the app and the engine
bumps its KV wire format; user clears site data; user opens the
app on a different device; quota exhausted; private browsing mode.

## Non-goals (deferred until a real consumer asks)

- **LoRA / runtime adapter fingerprinting.** Project doesn't
  currently support adapters; if added, schema bumps to v2.
- **Cross-tab consistency.** No locking. Last writer wins. Apps
  needing sync use `BroadcastChannel` themselves.
- **Encryption at rest.** App responsibility. Blob is opaque to the
  helper; apps wrap with `crypto.subtle.encrypt` if needed.
- **Schema migrations.** Single integer `schemaVersion`, no migration
  table. Mismatch = refuse + re-prefill from app side. Add migrators
  only when long-lived persisted convs across engine versions are a
  named consumer requirement.
- **`maxContextTokens` compatibility checks** *as a gate*. The header
  records the user's original setting; surface as
  `ConversationContextOverflowError` on next `chatCompletion(conv,…)`
  if the loaded model can't accommodate. No extra import-time check.
- **Auto-save in the engine.** Pure-pull: engine produces blobs on
  demand; app drives when to write. Auto-save policy belongs in
  app code or in the optional helper, not in `chatCompletion`.
- **Smoke-page integration.** The existing `smoke-test/real-model.html`
  does not currently exercise persistence; adding it is a low-
  priority follow-up (mirrors the item-11 closure pattern).

## Architecture

Two strictly-separated tiers:

```
┌─────────────────────────────────────────────────────────┐
│  ENGINE (load-bearing, ships in core)                  │
│  • exportConversation(conv) → Uint8Array (header+kv)   │
│  • importConversation(modelId, blob) → ConversationHandle│
│  • IncompatibleConversationError + CorruptBlobError    │
│  Knows: GGUF metadata, schemaVersion, KV serialize/load│
│  Does NOT touch: IndexedDB, OPFS, fetch, postMessage   │
└────────────────────────────┬───────────────────────────┘
                             │ Uint8Array (opaque to consumer)
┌────────────────────────────┴───────────────────────────┐
│  HELPER (optional, separate import path)               │
│  @paulrobello/webllm/persistence                       │
│  • IndexedDBConversationStore                          │
│    .put(key, blob) .get(key) .list() .delete(key)      │
│  • PersistenceUnavailableError + PersistenceQuotaError │
│  Knows: IDB plumbing only. App composes engine + store.│
└─────────────────────────────────────────────────────────┘
```

Both tiers work identically in main-thread and worker modes. The
engine tier round-trips large blobs through the proxy via a small
worker-bridge extension (transfer-on-return for allowlisted
methods). The helper has zero `window` / `document` references and
runs in either thread.

### Canonical app composition

```ts
const webllm = await WebLLM.init({worker: true});
const {handle: model} = await webllm.loadModelFromUrl(url, "qwen3-8b");
const store = new IndexedDBConversationStore("my-app-conversations");

let conv: ConversationHandle;
const blob = await store.get("user-42-session");
try {
  conv = blob ? await webllm.importConversation(model.id, blob)
              : await webllm.createConversation(model.id);
} catch (e) {
  if (e instanceof IncompatibleConversationError ||
      e instanceof CorruptBlobError) {
    await store.delete("user-42-session");
    conv = await webllm.createConversation(model.id);
  } else throw e;
}

for await (const chunk of webllm.chatCompletion(conv, msgs)) emit(chunk);
const fresh = await webllm.exportConversation(conv);
await store.put("user-42-session", fresh);
```

### Why two tiers, not one

The engine boundary stays minimal (two methods, two error classes,
no IDB knowledge) and the helper is genuinely optional. Apps that
want OPFS, server sync, or encrypted-at-rest implement their own
`Store` against the same `Uint8Array` blob without wrapping or
working around an IDB-coupled engine.

## Wire format

The blob is a single `Uint8Array` so it round-trips the worker
bridge as one transferable. Layout:

```
offset  size       field
─────────────────────────────────────────────────────────────────
0       4          magic = 0x57 0x4C 0x4B 0x56  ("WLKV" ASCII)
4       4          headerLen (uint32 little-endian)
8       headerLen  headerJson (UTF-8 JSON of PersistedConversationHeader)
8+hL    rest       kvBytes (raw output of inf.serializeKVCache)
```

### Header schema

```ts
interface PersistedConversationHeader {
  schemaVersion: 1;
  fingerprint: ModelFingerprint;
  conversationOptions: { maxContextTokens?: number };
  tokenIds: number[];
  byteSize: number;            // sanity: must equal kvBytes.length
  savedAtMs: number;           // for store-side LRU / debugging
}

interface ModelFingerprint {
  architecture: ModelArchitecture;  // "qwen3" | "llama" | "phi3" | …
  vocabSize: number;
  nEmbd: number;
  nLayer: number;
  nHead: number;
  nHeadKV: number;
  ropeBase: number;
  quantType: string;                 // "Q4_K_M" | "IQ3_M" | "hyb-Q4_K-fp16" | …
  tokenizerHash: string;             // sha256 of canonical-key-sorted
                                     // tokenizerConfig JSON
}
```

### Where each field comes from at save time

| Field | Source |
|---|---|
| `schemaVersion` | Constant `KV_PERSISTENCE_SCHEMA_VERSION = 1` in `core/persistence.ts` |
| `fingerprint.architecture` / `nEmbd` / `nLayer` / `nHead` / `nHeadKV` / `vocabSize` / `ropeBase` | `ModelInference.metadata.hyperparams` (already parsed at load) |
| `fingerprint.quantType` | `ModelInference.metadata.quantType` (already parsed) |
| `fingerprint.tokenizerHash` | Computed once at `loadModelFromBuffer/Url`, cached on the model entry. SHA-256 over `tokenizerConfig` JSON serialized with sorted keys. |
| `conversationOptions` | `ConversationPool.options(conv)` |
| `tokenIds` / `byteSize` | `ConversationPool.get(conv)` (in-memory snapshot already has them) |

### Validation order at import time

1. Magic bytes match → else `CorruptBlobError("bad-magic")`.
2. `headerLen` fits in remaining blob length → else
   `CorruptBlobError("bad-header-len")`.
3. Header JSON.parse → else `CorruptBlobError("bad-header-json")`.
4. `header.schemaVersion === KV_PERSISTENCE_SCHEMA_VERSION` →
   else `IncompatibleConversationError("schema-mismatch", {got, want})`.
5. Compute current model fingerprint, deep-equal against header
   field-by-field in deterministic order →
   - First non-tokenizer field that differs:
     `IncompatibleConversationError("fingerprint-mismatch", {field, got, want})`.
   - Only `tokenizerHash` differs:
     `IncompatibleConversationError("tokenizer-mismatch", {got, want})`
     (called out separately because most-common false-positive).
6. `kvBytes.length === header.byteSize` → else
   `CorruptBlobError("byte-size-mismatch", {got, want})`.
7. Allocate pool entry, populate snapshot from header + kvBytes,
   return new `ConversationHandle`.

The handle's `id` is fresh — old `id` is **not** preserved. Handles
are scoped to the engine session; cross-session identity is the
app's `key` namespace at the helper layer.

### Why JSON-not-binary, magic-bytes-first

JSON header is debuggable in DevTools, has zero new deps, and the
header is small (≪ 1 KB) so encoding overhead is irrelevant. Magic
bytes distinguish a future format (e.g. multipart for incremental
save) from a corrupt stream — without magic, JSON+bytes is
indistinguishable from random data.

## Engine API

### Public additions to `WebLLM` (`src/core/engine.ts`)

```ts
async exportConversation(conv: ConversationHandle): Promise<Uint8Array> {
  this.conversationPool.assertExists(conv);
  const release = this.conversationPool.tryAcquireLock(conv);
  if (!release) throw new ConversationBusyError(conv.id);
  try {
    const snap = this.conversationPool.get(conv);
    if (!snap) throw new ConversationNotPopulatedError(conv.id);
    const inf = this.inferenceEngines.get(snap.modelHandleId);
    if (!inf) throw new InferenceEngineMissingError(snap.modelHandleId);
    return encodePersistedConversation({
      schemaVersion: KV_PERSISTENCE_SCHEMA_VERSION,
      fingerprint: this.modelFingerprint(snap.modelHandleId),
      conversationOptions: this.conversationPool.options(conv),
      tokenIds: snap.tokenIds,
      byteSize: snap.byteSize,
      savedAtMs: Date.now(),
    }, snap.kvBytes);
  } finally { release(); }
}

async importConversation(
  modelHandleId: string,
  blob: Uint8Array,
  options?: ConversationOptions,
): Promise<ConversationHandle> {
  const entry = this._modelManager.get(modelHandleId);
  if (!entry || !entry.loaded) throw new ModelNotLoadedError(modelHandleId);
  const inf = this.inferenceEngines.get(modelHandleId);
  if (!inf) throw new InferenceEngineMissingError(modelHandleId);
  if (!inf.flashAttn) {
    throw new Error(
      `importConversation requires FA mode; "${modelHandleId}" is in manual mode.`,
    );
  }
  const {header, kvBytes} = decodePersistedConversation(
    blob,
    this.modelFingerprint(modelHandleId),
  );
  const opts = options ?? header.conversationOptions;
  const conv = this.conversationPool.create(modelHandleId, opts);
  this.conversationPool.set(conv, {
    conversationId: conv.id,
    modelHandleId,
    tokenIds: header.tokenIds,
    kvBytes,
    byteSize: header.byteSize,
    lastAccessMs: Date.now(),
  });
  return conv;
}

private modelFingerprint(modelHandleId: string): ModelFingerprint {
  // Cached on the model entry, computed once at load time.
}
```

### Pure-format module (`src/core/persistence.ts`)

```ts
export const KV_PERSISTENCE_SCHEMA_VERSION = 1;
export const KV_PERSISTENCE_MAGIC = new Uint8Array([0x57, 0x4C, 0x4B, 0x56]);

export interface PersistedConversationHeader { /* per "Wire format" */ }
export interface ModelFingerprint { /* per "Wire format" */ }

export function encodePersistedConversation(
  header: PersistedConversationHeader,
  kvBytes: Uint8Array,
): Uint8Array;

export function decodePersistedConversation(
  blob: Uint8Array,
  expectedFingerprint: ModelFingerprint,
): { header: PersistedConversationHeader; kvBytes: Uint8Array };

export function computeTokenizerHash(
  tokenizerConfig: TokenizerConfig,
): string;
```

No I/O, no platform APIs, no proxy concerns — trivially unit-testable
in Bun without WebGPU/IDB stubs.

### Error classes (`src/core/errors.ts`)

```ts
export type IncompatibleConversationReason =
  | "schema-mismatch"
  | "fingerprint-mismatch"
  | "tokenizer-mismatch";

export class IncompatibleConversationError extends WebLLMError {
  readonly code = "INCOMPATIBLE_CONVERSATION" as const;
  constructor(
    public readonly reason: IncompatibleConversationReason,
    public readonly details: Record<string, unknown>,
  ) { super(`incompatible persisted conversation: ${reason}`); }
}

export type CorruptBlobReason =
  | "bad-magic"
  | "bad-header-len"
  | "bad-header-json"
  | "byte-size-mismatch";

export class CorruptBlobError extends WebLLMError {
  readonly code = "CORRUPT_BLOB" as const;
  constructor(
    public readonly reason: CorruptBlobReason,
    public readonly details: Record<string, unknown>,
  ) { super(`corrupt persisted-conversation blob: ${reason}`); }
}
```

Both extend `WebLLMError` and round-trip through
`webllm-error-codec.ts` (five entries total: these two plus the
helper's `PersistenceUnavailableError`, `PersistenceQuotaError`,
and `PersistenceIOError`).

### Concurrency guarantees

- `exportConversation` acquires the pool lock — concurrent with an
  in-flight `chatCompletion(conv,…)` it waits until the turn settles,
  then captures the post-turn snapshot. Throws `ConversationBusyError`
  if the lock is already held by another caller.
- `importConversation` does not interact with any existing conv
  (creates a new one); no lock contention.
- Both honor LRU eviction on pool create — if importing at capacity,
  the oldest non-locked entry is evicted, exactly like
  `createConversation` and `forkConversation`.

### Surface sentinel update

`tests/webllm-proxy-surface.test.ts` adds `"exportConversation"` and
`"importConversation"` to `PROXIED_METHODS`. That test forces the
proxy + worker host to gain matching mirrors before the rest of the
implementation lands.

## Worker bridge changes

Three small surgical changes; none touch streaming.

### Extend `method-result` with optional transfer list

```ts
// src/core/worker-bridge.ts
export type WorkerToProxy =
  | { type: "init-done"; id: RequestId }
  | { type: "method-result"; id: RequestId; value: unknown;
      transfer?: Transferable[] }   // ← new optional field
  | { type: "method-error"; ... }
  | { type: "stream-chunk"; ... }
  | …;
```

### Worker host: per-method transfer allowlist

```ts
// src/core/webllm-worker-host.ts (handleMethodCall)
const TRANSFERS_BUFFER_ON_RETURN = new Set(["exportConversation"]);

async function handleMethodCall(msg) {
  try {
    const fn = opts.engine[msg.name];
    if (typeof fn !== "function") throw new Error(`unknown engine method: ${msg.name}`);
    const value = await fn.apply(opts.engine, msg.args);

    const stripsInference = msg.name === "loadModelFromBuffer" || msg.name === "loadModelFromUrl";
    const sanitized = stripsInference && value && typeof value === "object" && "handle" in value
      ? { handle: value.handle, metadata: value.metadata }
      : value;

    let transfer: Transferable[] | undefined;
    if (TRANSFERS_BUFFER_ON_RETURN.has(msg.name)
        && sanitized instanceof Uint8Array) {
      transfer = [sanitized.buffer as ArrayBuffer];
    }

    opts.postMessage(
      { type: "method-result", id: msg.id, value: sanitized, transfer },
      transfer,
    );
  } catch (e) { … }
}
```

### Worker host: postMessage signature widening

```ts
postMessage(m: WorkerToProxy, transfer?: Transferable[]): void;
```

The host adapter at the worker boot script's main entry passes the
transfer list through to `globalThis.postMessage(m, transfer)`. The
in-process channel adapter used in `webllm-proxy-integration.test.ts`
ignores the transfer list (main-thread-internal — `queueMicrotask`
carries by reference, transfer is meaningless).

### Proxy side: zero changes

`callMethod<T>(name, args, transfer?)` already takes an outbound
transfer list. Inbound, the value arrives as a normal Uint8Array
reference; protocol-handling unchanged.

### Detached-buffer semantics, caller-visible

After `await proxy.importConversation(modelId, blob)`, the caller's
`blob.buffer` is detached in worker mode — `blob.length` reads as 0.
This matches the existing `loadModelFromBuffer` contract. Caller
guidance:

> *Callers passing a `Uint8Array` to `importConversation` in worker
> mode must treat the array as consumed after the call returns. If
> the caller needs to keep a copy (e.g. for retry on transient
> failure), they slice it before passing:
> `await webllm.importConversation(id, blob.slice())`. In main-thread
> mode the buffer is never detached, but the spec contract treats
> both modes uniformly.*

## Persistence helper module

Lives at `src/persistence/indexeddb-store.ts`. Re-exports as a
separate entry point (`@paulrobello/webllm/persistence`) so
tree-shakers drop it from bundles that don't use it. Mirrors the
existing `pipeline-cache.ts` IDB pattern.

### Public surface

```ts
export interface ConversationStoreEntry {
  key: string;
  byteLength: number;
  savedAtMs: number;
}

export class IndexedDBConversationStore {
  constructor(dbName: string, storeName?: string);
  async open(): Promise<void>;
  async put(key: string, blob: Uint8Array): Promise<void>;
  async get(key: string): Promise<Uint8Array | undefined>;
  async delete(key: string): Promise<void>;
  async list(): Promise<ConversationStoreEntry[]>;
  async clear(): Promise<void>;
  async close(): Promise<void>;
}

export class PersistenceUnavailableError extends WebLLMError {
  readonly code = "PERSISTENCE_UNAVAILABLE" as const;
  // reasons: "indexeddb-missing" | "indexeddb-blocked" | "open-failed"
}

export class PersistenceQuotaError extends WebLLMError {
  readonly code = "PERSISTENCE_QUOTA" as const;
  constructor(public readonly attemptedBytes: number) { ... }
}

export type PersistenceIOReason = "io-failure" | "transaction-aborted";

export class PersistenceIOError extends WebLLMError {
  readonly code = "PERSISTENCE_IO" as const;
  constructor(
    public readonly reason: PersistenceIOReason,
    public readonly cause: unknown,
  ) { super(`persistence IO error: ${reason}`); }
}
```

### Design choices

- **No automatic eviction.** Right policy is app-specific. Apps that
  want eviction call `list()`, sort by `savedAtMs`, and delete tail
  entries themselves. YAGNI for first ship.
- **`list()` reads only headers, not full blobs.** A side metadata
  object store (`conversation-meta`) is updated atomically alongside
  the payload store (`conversations`) so `list()` is cheap. Both are
  written in the same `readwrite` transaction.
- **`open()` is idempotent and lazy.** Constructor doesn't open;
  first `put`/`get` triggers `open()` automatically. Explicit
  `open()` provided for apps that want to surface init errors at
  startup.
- **`PersistenceUnavailableError` distinguishes three sub-causes**
  because UX response differs: `"indexeddb-missing"` vs
  `"indexeddb-blocked"` vs `"open-failed"`.
- **`PersistenceQuotaError` carries `attemptedBytes`** so the app
  can decide whether to evict + retry or surface to user.
- **Worker-compatible.** `IndexedDB` is available in
  `DedicatedWorkerGlobalScope`. Helper has zero `window` /
  `document` references.

### Bundle exports

`src/index.ts` does **not** re-export the helper. Two import paths:

```ts
import { WebLLM } from "@paulrobello/webllm";
import { IndexedDBConversationStore } from "@paulrobello/webllm/persistence";
```

`package.json`:

```json
{
  "exports": {
    ".": "./dist/index.js",
    "./persistence": "./dist/persistence/indexeddb-store.js"
  }
}
```

## Failure-mode matrix

| Stage | Failure | Surface | App-side recovery |
|---|---|---|---|
| `store.open()` | IndexedDB API absent | `PersistenceUnavailableError("indexeddb-missing")` | Skip persistence; ephemeral conv |
| | Browser blocked IDB | `PersistenceUnavailableError("indexeddb-blocked")` | Same; optional UI banner |
| | DB open fails | `PersistenceUnavailableError("open-failed")` | Offer "wipe and start fresh" |
| `store.put()` | Quota exceeded | `PersistenceQuotaError(attemptedBytes)` | Evict via `list()`+`delete()`, retry; or surface to user |
| | Other IO error | `PersistenceIOError("io-failure", cause)` | Retry once; fall back if persistent |
| `store.get()` | Key not found | `undefined` (not an error) | Caller falls back to `createConversation` |
| | Read transaction abort | `PersistenceIOError("io-failure", cause)` | Retry; treat as not-found if persistent |
| `exportConversation` | `conv` not in pool | `ConversationNotFoundError` (existing) | Bug — caller held a stale handle |
| | `conv` never had a turn | `ConversationNotPopulatedError` (existing) | Drive at least one turn before exporting |
| | `conv` in-flight | `ConversationBusyError` (existing) | Await in-flight turn |
| | Model unloaded | `InferenceEngineMissingError` (existing) | Bug — caller held stale handle |
| `importConversation` | `modelId` not loaded | `ModelNotLoadedError` (existing) | Load first |
| | Bad magic / short blob | `CorruptBlobError("bad-magic")` | `store.delete(key)`, fall back |
| | `headerLen` overflow | `CorruptBlobError("bad-header-len")` | Same |
| | Header JSON.parse fails | `CorruptBlobError("bad-header-json")` | Same |
| | byteSize mismatch | `CorruptBlobError("byte-size-mismatch")` | Same |
| | schemaVersion mismatch | `IncompatibleConversationError("schema-mismatch", {got, want})` | Engine bumped wire; `delete`+fallback |
| | Non-tokenizer fingerprint diff | `IncompatibleConversationError("fingerprint-mismatch", {field, got, want})` | Different model; `delete`+fallback |
| | Only tokenizer hash diff | `IncompatibleConversationError("tokenizer-mismatch", {got, want})` | Same recovery |
| | Pool at capacity, all locked | `ConversationPoolFullError` (existing) | Await + retry |

`store.get` returning `undefined` for a missing key (rather than
throwing) is intentional — "no saved conversation yet" is the
canonical first-load case, not an error. Matches `pipeline-cache.ts`.

## Testing strategy

Five test files; ≈ 25-35 new test cases; no new fixtures beyond
synthetic GGUF metadata.

### `tests/persistence-codec.test.ts` (pure-Bun, no skips)

Targets `src/core/persistence.ts`. Covers:
- Round-trip `decode(encode(header, kvBytes))` byte-equality.
- All `CorruptBlobError` sub-causes (bad magic / bad header-len /
  bad header-json / byte-size-mismatch).
- All `IncompatibleConversationError` sub-causes (schema /
  fingerprint per-field / tokenizer-only).
- `computeTokenizerHash`: same input → same hash; key-permuted
  JSON → same hash; semantic change → different hash.

### `tests/engine-conversation-persistence.test.ts` (skip on `!HAS_WEBGPU`)

Targets engine round-trip against a real loaded model (TinyLlama).
Covers:
- 1 turn → `exportConversation` produces non-empty blob with
  correct magic.
- Decode the blob → tokenIds + byteSize match.
- `importConversation` returns fresh handle (id ≠ original).
- `chatCompletion(importedConv, sameMessages, {temperature: 0})`
  greedy decodes byte-identically to a fresh-prefill control.
  (Highest-value end-to-end test.)
- Mismatch fingerprint: corrupt header → typed error.
- Mismatch schema: hand-roll v99 blob → typed error.
- Concurrency: parallel `exportConversation` waits for in-flight
  `chatCompletion(conv,…)` then captures post-turn snapshot.
- Second `exportConversation` overlapping first → `ConversationBusyError`.
- Forked-handle export round-trips like a primary handle.
- Import-then-export → byte-identical (modulo `savedAtMs`).

### `tests/webllm-proxy-integration.test.ts` (extends existing)

Adds a `describe("WebLLMProxy — persistence")` block that round-
trips both methods through the in-process channel:
- `proxy.exportConversation` returns Uint8Array with correct magic.
- `proxy.importConversation(modelId, blob)` returns
  `ConversationHandle`.
- Transfer-list code path exercised via spy on worker-host postMessage.
- Surface sentinel: methods exist on `WebLLMProxy.prototype` and
  `WebLLM.prototype`.
- Worker-side `CorruptBlobError` / `IncompatibleConversationError`
  reconstruct main-thread with `instanceof` preserved.

### `tests/persistence-indexeddb-store.test.ts` (skip on `!indexedDBAvailable`)

Targets the helper in isolation against fake-indexeddb. Covers:
- `put` / `get` / `delete` / `list` happy path.
- `list()` reads metadata side-store only (verified via spy).
- `put` overwrite updates metadata.
- `get(missing)` returns `undefined`.
- `clear()` empties both stores atomically.
- Quota error: stub `IDBRequest.error` →
  `PersistenceQuotaError(attemptedBytes)` with actual size.
- IDB-missing: `globalThis.indexedDB = undefined` →
  `PersistenceUnavailableError("indexeddb-missing")`.
- `open()` idempotent.
- `close()` releases cached DB; subsequent `get` re-opens.

### `tests/error-codec.test.ts` (extends existing)

Five new round-trip cases — `IncompatibleConversationError`,
`CorruptBlobError`, `PersistenceUnavailableError`,
`PersistenceQuotaError`, `PersistenceIOError`. Each: throw worker
side, reconstruct main side, assert `instanceof` + `code` +
`reason`/`details`.

### Skip-count delta

Adds ~9 skips on `!indexedDBAvailable` (matching the existing
`pipeline-cache.test.ts` pattern). Project's documented skip count
(currently 11) rises to ~20. This is a known-pattern continuation,
not a new pattern; flag in the watch-list note as expected.

### What's deliberately not tested

- **Real-IDB browser smoke** of full Worker + persistence path. The
  engine→proxy bridge is already smoke-validated by dual-mode
  worker; the IDB plumbing is identical to validated `pipeline-
  cache`. Browser smoke is good follow-up but not load-bearing for
  v1.
- **Bench targets.** Persistence is not on a perf-critical path.
  Export adds ≪ 1 ms over what `chatCompletion` already pays for
  `serializeKVCache`. IDB write cost is environmental and the
  consumer's tradeoff is qualitative.

## Out-of-scope items recap

- LoRA / runtime adapter fingerprints (schema v2 if added).
- Cross-tab consistency.
- Encryption at rest.
- Schema migration table.
- `maxContextTokens` import-time gate.
- Engine-side auto-save.
- Browser smoke for persistence path.
- Persistence bench targets.

Each will land as an additive change behind a real consumer ask.

## Plan transition

The implementation plan should phase per:

1. **Phase 0 (probe-style):** add `KV_PERSISTENCE_SCHEMA_VERSION`,
   `KV_PERSISTENCE_MAGIC`, `encodePersistedConversation`,
   `decodePersistedConversation`, `computeTokenizerHash`, and the
   pure-Bun codec test. Land in isolation; no engine changes.
2. **Phase 1:** add `IncompatibleConversationError` /
   `CorruptBlobError` to `core/errors.ts` + codec wiring; surface-
   sentinel additions; `modelFingerprint` helper on the model
   manager + cache field; `WebLLM.exportConversation` /
   `importConversation` methods; engine-level WebGPU test.
3. **Phase 2:** worker-bridge transfer-list extension; proxy method
   additions; worker-host allowlist; proxy-integration tests.
4. **Phase 3:** persistence helper module + IDB tests + errors +
   `package.json` exports field.
5. **Phase 4:** docs (README section under "Conversation persistence"),
   TODO closure with link to closure report.

Each phase is independently shippable and reviewable. Phase 0 is a
no-risk codec-only commit; Phases 1-3 each have an isolated test
gate.

**Touch surface estimate:**

- **New files (~7):** `src/core/persistence.ts`,
  `src/persistence/indexeddb-store.ts`,
  `tests/persistence-codec.test.ts`,
  `tests/engine-conversation-persistence.test.ts`,
  `tests/persistence-indexeddb-store.test.ts`, plus extensions to
  the test files listed below.
- **Modified files (~7):** `src/core/engine.ts` (two methods +
  `modelFingerprint` helper), `src/core/errors.ts` (five new
  classes), `src/core/webllm-error-codec.ts` (five new codec
  entries), `src/core/worker-bridge.ts` (`method-result.transfer?`
  field), `src/core/webllm-worker-host.ts` (allowlist + transfer
  send), `src/core/webllm-proxy.ts` (two new method mirrors),
  `package.json` (exports field for `./persistence` subpath).
- **Test extensions (~3):** `tests/webllm-proxy-surface.test.ts`
  (sentinel update), `tests/webllm-proxy-integration.test.ts`
  (persistence describe block), `tests/error-codec.test.ts` (five
  round-trip cases).
