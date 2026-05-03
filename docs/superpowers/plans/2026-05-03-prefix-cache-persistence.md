# Prefix-cache persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WebLLM.exportConversation(conv): Uint8Array` and `WebLLM.importConversation(modelId, blob): ConversationHandle` engine API plus an optional `IndexedDBConversationStore` helper, so apps can persist conversations across page reloads.

**Architecture:** Two strictly-separated tiers. Engine ships pure-blob primitives (load-bearing). Helper module ships IndexedDB plumbing (optional, separate import path). Worker bridge gains a per-method transfer-list allowlist on `method-result`. Wire format: 4-byte magic + 4-byte uint32 LE headerLen + JSON header + raw kvBytes.

**Tech Stack:** TypeScript, Bun (runtime + test), `biome` (fmt+lint), `tsc --noEmit` (typecheck), WebGPU (engine integration tests, gated `!HAS_WEBGPU`), `fake-indexeddb` already in tree (helper tests, gated `!indexedDBAvailable`).

**Spec:** [`docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`](../specs/2026-05-03-prefix-cache-persistence-design.md)

**Ship gate:** every task ends with `make checkall` clean and a single commit. `make checkall` runs `fmt → lint → typecheck → typecheck:tests → test`.

---

## File Structure

**New files:**
- `src/core/persistence.ts` — pure codec (encode/decode/computeTokenizerHash + types). No I/O, no platform APIs.
- `src/persistence/indexeddb-store.ts` — IDB helper. Uses globals only (`indexedDB`).
- `tests/persistence-codec.test.ts` — pure-Bun codec unit tests.
- `tests/engine-conversation-persistence.test.ts` — WebGPU end-to-end (skip on `!HAS_WEBGPU`).
- `tests/persistence-indexeddb-store.test.ts` — IDB helper unit tests (skip on `!indexedDBAvailable`).

**Modified core files:**
- `src/core/errors.ts` — five new error classes + five new `WebLLMErrorCode` literals.
- `src/core/webllm-error-codec.ts` — five new serialize/reconstruct entries.
- `src/core/worker-bridge.ts` — extend `WorkerToProxy.method-result` with `transfer?: Transferable[]`.
- `src/core/webllm-worker-host.ts` — transfer-on-return allowlist; widen `postMessage` signature.
- `src/core/webllm-proxy.ts` — add `exportConversation` / `importConversation` mirrors.
- `src/core/engine.ts` — `WebLLM.exportConversation`, `WebLLM.importConversation`, private `modelFingerprint`, `tokenizerHash` cache on `ModelEntry`.
- `src/core/types.ts` — add `tokenizerHash?: string` and `fingerprint?: ModelFingerprint` fields to `ModelEntry`.
- `package.json` — add `./persistence` to `exports` field.

**Modified test files:**
- `tests/webllm-proxy-surface.test.ts` — add two entries to `PROXIED_METHODS`.
- `tests/webllm-proxy-integration.test.ts` — add `describe("WebLLMProxy — persistence")` block.
- `tests/webllm-error-codec.test.ts` — add five entries to `FACTORIES` table and codes literal array; add fields-preserved tests.

---

## Phase 0 — Pure codec + errors (no engine, no IDB, no proxy)

**Outcome at end of phase:** `src/core/persistence.ts` exports `encodePersistedConversation`, `decodePersistedConversation`, `computeTokenizerHash`, `KV_PERSISTENCE_SCHEMA_VERSION`, `KV_PERSISTENCE_MAGIC`. Five new error classes shipped + codec round-trips them. All tests pure-Bun, no skips. Zero behavior change for existing engine.

### Task 0.1: Add five new error classes

**Files:**
- Modify: `src/core/errors.ts`

- [ ] **Step 1: Re-read `src/core/errors.ts` end-to-end** (per project doctrine "EDIT INTEGRITY").

- [ ] **Step 2: Add five codes to `WebLLMErrorCode` literal union.**

The union currently contains 10 codes (`MODEL_NOT_FOUND`, …, `CONVERSATION_BUSY`). Add five more:

```ts
// In src/core/errors.ts, extend the union (find existing block at top of file):
export type WebLLMErrorCode =
  | "MODEL_NOT_FOUND"
  | "MODEL_NOT_LOADED"
  | "INFERENCE_ENGINE_MISSING"
  | "ENCODER_REQUIRED"
  | "SPECULATIVE_DECODING_RESERVED"
  | "CONVERSATION_NOT_FOUND"
  | "CONVERSATION_NOT_POPULATED"
  | "CONVERSATION_POOL_FULL"
  | "CONVERSATION_CONTEXT_OVERFLOW"
  | "CONVERSATION_BUSY"
  | "INCOMPATIBLE_CONVERSATION"
  | "CORRUPT_BLOB"
  | "PERSISTENCE_UNAVAILABLE"
  | "PERSISTENCE_QUOTA"
  | "PERSISTENCE_IO";
```

- [ ] **Step 3: Append the five new error classes** at the end of `src/core/errors.ts`:

```ts
export type IncompatibleConversationReason =
  | "schema-mismatch"
  | "fingerprint-mismatch"
  | "tokenizer-mismatch";

export class IncompatibleConversationError extends WebLLMError {
  readonly reason: IncompatibleConversationReason;
  readonly details: Record<string, unknown>;
  constructor(
    reason: IncompatibleConversationReason,
    details: Record<string, unknown>,
  ) {
    super(
      `incompatible persisted conversation: ${reason}`,
      "INCOMPATIBLE_CONVERSATION",
    );
    this.name = "IncompatibleConversationError";
    this.reason = reason;
    this.details = details;
  }
}

export type CorruptBlobReason =
  | "bad-magic"
  | "bad-header-len"
  | "bad-header-json"
  | "byte-size-mismatch";

export class CorruptBlobError extends WebLLMError {
  readonly reason: CorruptBlobReason;
  readonly details: Record<string, unknown>;
  constructor(reason: CorruptBlobReason, details: Record<string, unknown>) {
    super(`corrupt persisted-conversation blob: ${reason}`, "CORRUPT_BLOB");
    this.name = "CorruptBlobError";
    this.reason = reason;
    this.details = details;
  }
}

export type PersistenceUnavailableReason =
  | "indexeddb-missing"
  | "indexeddb-blocked"
  | "open-failed";

export class PersistenceUnavailableError extends WebLLMError {
  readonly reason: PersistenceUnavailableReason;
  readonly cause: unknown;
  constructor(reason: PersistenceUnavailableReason, cause?: unknown) {
    super(`persistence unavailable: ${reason}`, "PERSISTENCE_UNAVAILABLE");
    this.name = "PersistenceUnavailableError";
    this.reason = reason;
    this.cause = cause;
  }
}

export class PersistenceQuotaError extends WebLLMError {
  readonly attemptedBytes: number;
  constructor(attemptedBytes: number) {
    super(
      `persistence quota exceeded (attempted ${attemptedBytes} bytes)`,
      "PERSISTENCE_QUOTA",
    );
    this.name = "PersistenceQuotaError";
    this.attemptedBytes = attemptedBytes;
  }
}

export type PersistenceIOReason = "io-failure" | "transaction-aborted";

export class PersistenceIOError extends WebLLMError {
  readonly reason: PersistenceIOReason;
  readonly cause: unknown;
  constructor(reason: PersistenceIOReason, cause: unknown) {
    super(`persistence IO error: ${reason}`, "PERSISTENCE_IO");
    this.name = "PersistenceIOError";
    this.reason = reason;
    this.cause = cause;
  }
}
```

- [ ] **Step 4: Run `bun run typecheck`.** Expect: pass. The codec test will fail (mirror-drift sentinel), but that's caught in Task 0.2.

- [ ] **Step 5: Run `bun test tests/errors.test.ts`** — must pass (file should be unaffected; this is just a smoke check).

- [ ] **Step 6: Commit.**

```bash
git add src/core/errors.ts
git commit -m "feat(errors): add 5 persistence error classes"
```

---

### Task 0.2: Wire codec serialize/reconstruct + extend factory test

**Files:**
- Modify: `src/core/webllm-error-codec.ts`
- Modify: `src/core/worker-bridge.ts` (add fields to `SerializedError`)
- Modify: `tests/webllm-error-codec.test.ts`

- [ ] **Step 1: Re-read `src/core/webllm-error-codec.ts` and `src/core/worker-bridge.ts`** (specifically the `SerializedError` type).

- [ ] **Step 2: Extend `SerializedError` in `src/core/worker-bridge.ts`** with the new optional fields:

```ts
// Add to the existing SerializedError type definition. Existing fields
// (modelId, conversationId, requestedTokens, etc.) remain.
export interface SerializedError {
  code: string;
  message: string;
  stack?: string;
  // Existing fields (modelId, conversationId, ...) here.
  modelId?: string;
  architecture?: string;
  conversationId?: string;
  requestedTokens?: number;
  maxContextTokens?: number;
  liveConversationIds?: string[];
  // NEW for persistence errors:
  reason?: string;
  details?: Record<string, unknown>;
  attemptedBytes?: number;
  cause?: { message: string; name?: string }; // serialized cause object
}
```

(The `cause` field is structured-cloneable; full Error objects are not. We capture only message + name for diagnostics.)

- [ ] **Step 3: Extend `serializeError` in `src/core/webllm-error-codec.ts`.**

Find the existing `if (e instanceof WebLLMError)` block. Add five new `else if` branches *before* the closing `return out;`:

```ts
import {
  // ... existing imports
  IncompatibleConversationError,
  CorruptBlobError,
  PersistenceUnavailableError,
  PersistenceQuotaError,
  PersistenceIOError,
} from "./errors.js";

// Inside serializeError, after the existing ConversationBusyError branch:
else if (e instanceof IncompatibleConversationError) {
  out.reason = e.reason;
  out.details = e.details;
}
else if (e instanceof CorruptBlobError) {
  out.reason = e.reason;
  out.details = e.details;
}
else if (e instanceof PersistenceUnavailableError) {
  out.reason = e.reason;
  if (e.cause !== undefined) out.cause = serializeCause(e.cause);
}
else if (e instanceof PersistenceQuotaError) {
  out.attemptedBytes = e.attemptedBytes;
}
else if (e instanceof PersistenceIOError) {
  out.reason = e.reason;
  out.cause = serializeCause(e.cause);
}
```

Add a `serializeCause` helper at module scope:

```ts
function serializeCause(c: unknown): { message: string; name?: string } | undefined {
  if (c instanceof Error) return { message: c.message, name: c.name };
  if (typeof c === "string") return { message: c };
  return undefined;
}
```

- [ ] **Step 4: Extend `reconstructError`** with five new switch cases:

```ts
import type {
  IncompatibleConversationReason,
  CorruptBlobReason,
  PersistenceUnavailableReason,
  PersistenceIOReason,
} from "./errors.js";

// Inside the switch (code) { ... }:
case "INCOMPATIBLE_CONVERSATION":
  return attachStack(
    new IncompatibleConversationError(
      (s.reason ?? "schema-mismatch") as IncompatibleConversationReason,
      s.details ?? {},
    ),
    s,
  );
case "CORRUPT_BLOB":
  return attachStack(
    new CorruptBlobError(
      (s.reason ?? "bad-magic") as CorruptBlobReason,
      s.details ?? {},
    ),
    s,
  );
case "PERSISTENCE_UNAVAILABLE":
  return attachStack(
    new PersistenceUnavailableError(
      (s.reason ?? "indexeddb-missing") as PersistenceUnavailableReason,
      s.cause,
    ),
    s,
  );
case "PERSISTENCE_QUOTA":
  return attachStack(new PersistenceQuotaError(s.attemptedBytes ?? 0), s);
case "PERSISTENCE_IO":
  return attachStack(
    new PersistenceIOError(
      (s.reason ?? "io-failure") as PersistenceIOReason,
      s.cause,
    ),
    s,
  );
```

- [ ] **Step 5: Update `tests/webllm-error-codec.test.ts`.**

Re-read the file first. Then:

(a) Add five imports to the top:

```ts
import {
  // ... existing imports
  CorruptBlobError,
  IncompatibleConversationError,
  PersistenceIOError,
  PersistenceQuotaError,
  PersistenceUnavailableError,
} from "../src/core/errors.js";
```

(b) Extend `FACTORIES`:

```ts
const FACTORIES: Record<WebLLMErrorCode, () => WebLLMError> = {
  // ... existing entries
  INCOMPATIBLE_CONVERSATION: () =>
    new IncompatibleConversationError("schema-mismatch", { got: 99, want: 1 }),
  CORRUPT_BLOB: () =>
    new CorruptBlobError("bad-magic", { firstFour: [0, 0, 0, 0] }),
  PERSISTENCE_UNAVAILABLE: () =>
    new PersistenceUnavailableError("indexeddb-missing"),
  PERSISTENCE_QUOTA: () => new PersistenceQuotaError(123_456_789),
  PERSISTENCE_IO: () =>
    new PersistenceIOError("io-failure", new Error("disk full")),
};
```

(c) Extend the codes literal array in the `every WebLLMErrorCode has a factory entry` test by appending:

```ts
"INCOMPATIBLE_CONVERSATION",
"CORRUPT_BLOB",
"PERSISTENCE_UNAVAILABLE",
"PERSISTENCE_QUOTA",
"PERSISTENCE_IO",
```

(d) Add five fields-preserved tests, mirroring the existing `ModelNotFoundError preserves modelId field` style:

```ts
test("IncompatibleConversationError preserves reason + details", () => {
  const err = new IncompatibleConversationError(
    "fingerprint-mismatch",
    { field: "nLayer", got: 28, want: 32 },
  );
  const round = reconstructError(serializeError(err));
  expect(round).toBeInstanceOf(IncompatibleConversationError);
  const ic = round as IncompatibleConversationError;
  expect(ic.reason).toBe("fingerprint-mismatch");
  expect(ic.details).toEqual({ field: "nLayer", got: 28, want: 32 });
});

test("CorruptBlobError preserves reason + details", () => {
  const err = new CorruptBlobError("byte-size-mismatch", { got: 99, want: 100 });
  const round = reconstructError(serializeError(err));
  expect(round).toBeInstanceOf(CorruptBlobError);
  const cb = round as CorruptBlobError;
  expect(cb.reason).toBe("byte-size-mismatch");
  expect(cb.details).toEqual({ got: 99, want: 100 });
});

test("PersistenceQuotaError preserves attemptedBytes", () => {
  const err = new PersistenceQuotaError(987_654_321);
  const round = reconstructError(serializeError(err));
  expect(round).toBeInstanceOf(PersistenceQuotaError);
  expect((round as PersistenceQuotaError).attemptedBytes).toBe(987_654_321);
});

test("PersistenceUnavailableError preserves reason", () => {
  const err = new PersistenceUnavailableError("indexeddb-blocked");
  const round = reconstructError(serializeError(err));
  expect(round).toBeInstanceOf(PersistenceUnavailableError);
  expect((round as PersistenceUnavailableError).reason).toBe("indexeddb-blocked");
});

test("PersistenceIOError preserves reason + cause message", () => {
  const err = new PersistenceIOError("transaction-aborted", new Error("abort"));
  const round = reconstructError(serializeError(err));
  expect(round).toBeInstanceOf(PersistenceIOError);
  const io = round as PersistenceIOError;
  expect(io.reason).toBe("transaction-aborted");
  expect((io.cause as { message: string }).message).toBe("abort");
});
```

- [ ] **Step 6: Run `bun test tests/webllm-error-codec.test.ts`** — expect: all pass (existing + 5 new + 5 fields-preserved).

- [ ] **Step 7: Run `make checkall`** — expect: clean.

- [ ] **Step 8: Commit.**

```bash
git add src/core/webllm-error-codec.ts src/core/worker-bridge.ts tests/webllm-error-codec.test.ts
git commit -m "feat(errors): wire 5 persistence error classes through codec"
```

---

### Task 0.3: Implement `computeTokenizerHash` (TDD)

**Files:**
- Create: `src/core/persistence.ts`
- Create: `tests/persistence-codec.test.ts`

- [ ] **Step 1: Write the failing test** for `computeTokenizerHash`.

Create `tests/persistence-codec.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import { computeTokenizerHash } from "../src/core/persistence.js";

describe("computeTokenizerHash", () => {
  const baseConfig = {
    type: "BPE" as const,
    vocab: { hello: 1, world: 2 },
    merges: ["h e", "l l"],
    specialTokens: { bos: 0, eos: 3 },
  };

  test("identical input yields identical hash", () => {
    const a = computeTokenizerHash(baseConfig);
    const b = computeTokenizerHash(baseConfig);
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{64}$/); // sha256 hex
  });

  test("key-permuted input yields identical hash", () => {
    const reordered = {
      specialTokens: { eos: 3, bos: 0 },
      merges: ["h e", "l l"],
      vocab: { world: 2, hello: 1 },
      type: "BPE" as const,
    };
    expect(computeTokenizerHash(baseConfig)).toBe(
      computeTokenizerHash(reordered),
    );
  });

  test("vocab change yields different hash", () => {
    const changed = {
      ...baseConfig,
      vocab: { ...baseConfig.vocab, foo: 4 },
    };
    expect(computeTokenizerHash(baseConfig)).not.toBe(
      computeTokenizerHash(changed),
    );
  });

  test("specialTokens change yields different hash", () => {
    const changed = {
      ...baseConfig,
      specialTokens: { ...baseConfig.specialTokens, eos: 99 },
    };
    expect(computeTokenizerHash(baseConfig)).not.toBe(
      computeTokenizerHash(changed),
    );
  });
});
```

- [ ] **Step 2: Run the test — must fail with module not found.**

```bash
bun test tests/persistence-codec.test.ts
```

Expected: error parsing the import; `src/core/persistence.ts` doesn't exist.

- [ ] **Step 3: Create `src/core/persistence.ts` with the minimum to pass.**

```ts
/**
 * Wire format and helpers for persisted-conversation blobs.
 *
 * Pure module: no I/O, no platform APIs, no proxy concerns. The engine
 * (`engine.ts`) and helper (`indexeddb-store.ts`) compose these
 * primitives. Spec: 2026-05-03-prefix-cache-persistence-design.md.
 */

import type { TokenizerConfig } from "../inference/tokenizer.js";
import { CorruptBlobError, IncompatibleConversationError } from "./errors.js";

export const KV_PERSISTENCE_SCHEMA_VERSION = 1;
// "WLKV" — magic bytes that mark a persisted-conversation blob.
export const KV_PERSISTENCE_MAGIC = new Uint8Array([0x57, 0x4c, 0x4b, 0x56]);

export interface ModelFingerprint {
  architecture: string;
  vocabSize: number;
  nEmbd: number;
  nLayer: number;
  nHead: number;
  nHeadKV: number;
  ropeBase: number;
  quantType: string;
  tokenizerHash: string;
}

export interface PersistedConversationHeader {
  schemaVersion: 1;
  fingerprint: ModelFingerprint;
  conversationOptions: { maxContextTokens?: number };
  tokenIds: number[];
  byteSize: number;
  savedAtMs: number;
}

/**
 * Deterministic hex-sha256 over the canonical-key-sorted JSON of a
 * tokenizer config. Used to fingerprint vocab pinning so blobs from a
 * subtly-different tokenizer (same arch, retrained) refuse to import.
 */
export async function computeTokenizerHash(
  cfg: TokenizerConfig,
): Promise<string> {
  const canonical = stableStringify(cfg as unknown);
  const bytes = new TextEncoder().encode(canonical);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function stableStringify(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) {
    return "[" + v.map(stableStringify).join(",") + "]";
  }
  const obj = v as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  return (
    "{" +
    keys.map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k])).join(",") +
    "}"
  );
}
```

Note: `computeTokenizerHash` is `async` because `crypto.subtle.digest` is async. Update the test imports accordingly:

```ts
test("identical input yields identical hash", async () => {
  const a = await computeTokenizerHash(baseConfig);
  const b = await computeTokenizerHash(baseConfig);
  expect(a).toBe(b);
  expect(a).toMatch(/^[0-9a-f]{64}$/);
});
```

— and apply `async`/`await` to the other three tests.

- [ ] **Step 4: Run the test — must pass.**

```bash
bun test tests/persistence-codec.test.ts
```

Expected: 4 pass.

- [ ] **Step 5: Run `make checkall`** — clean.

- [ ] **Step 6: Commit.**

```bash
git add src/core/persistence.ts tests/persistence-codec.test.ts
git commit -m "feat(persistence): add computeTokenizerHash with stable-key digest"
```

---

### Task 0.4: Implement `encodePersistedConversation` (TDD)

**Files:**
- Modify: `src/core/persistence.ts`
- Modify: `tests/persistence-codec.test.ts`

- [ ] **Step 1: Append a failing test** for `encodePersistedConversation`:

```ts
import {
  computeTokenizerHash,
  encodePersistedConversation,
  KV_PERSISTENCE_MAGIC,
  KV_PERSISTENCE_SCHEMA_VERSION,
  type PersistedConversationHeader,
} from "../src/core/persistence.js";

const SAMPLE_FINGERPRINT = {
  architecture: "qwen3",
  vocabSize: 151_936,
  nEmbd: 4096,
  nLayer: 28,
  nHead: 32,
  nHeadKV: 8,
  ropeBase: 10_000,
  quantType: "Q4_K_M",
  tokenizerHash: "a".repeat(64),
};

const SAMPLE_HEADER: PersistedConversationHeader = {
  schemaVersion: 1,
  fingerprint: SAMPLE_FINGERPRINT,
  conversationOptions: { maxContextTokens: 4096 },
  tokenIds: [1, 2, 3, 4, 5],
  byteSize: 16,
  savedAtMs: 1_700_000_000_000,
};

const SAMPLE_KV = new Uint8Array([0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
  0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f]);

describe("encodePersistedConversation", () => {
  test("layout: magic + uint32 LE headerLen + header JSON + kvBytes", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);

    // Magic.
    expect(blob.slice(0, 4)).toEqual(KV_PERSISTENCE_MAGIC);

    // headerLen (uint32 LE).
    const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    const headerLen = dv.getUint32(4, /* littleEndian */ true);
    expect(headerLen).toBeGreaterThan(0);

    // Header JSON.
    const headerJson = new TextDecoder().decode(
      blob.subarray(8, 8 + headerLen),
    );
    const header = JSON.parse(headerJson);
    expect(header.schemaVersion).toBe(KV_PERSISTENCE_SCHEMA_VERSION);
    expect(header.fingerprint).toEqual(SAMPLE_FINGERPRINT);
    expect(header.tokenIds).toEqual([1, 2, 3, 4, 5]);

    // KV bytes.
    expect(blob.subarray(8 + headerLen)).toEqual(SAMPLE_KV);
    expect(blob.byteLength).toBe(8 + headerLen + SAMPLE_KV.byteLength);
  });

  test("two encodes of identical input produce byte-identical blobs", () => {
    const a = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    const b = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    expect(a).toEqual(b);
  });
});
```

- [ ] **Step 2: Run test — must fail** (`encodePersistedConversation` not exported).

- [ ] **Step 3: Implement `encodePersistedConversation`** — append to `src/core/persistence.ts`:

```ts
export function encodePersistedConversation(
  header: PersistedConversationHeader,
  kvBytes: Uint8Array,
): Uint8Array {
  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);
  const out = new Uint8Array(4 + 4 + headerBytes.byteLength + kvBytes.byteLength);
  out.set(KV_PERSISTENCE_MAGIC, 0);
  new DataView(out.buffer).setUint32(4, headerBytes.byteLength, /* LE */ true);
  out.set(headerBytes, 8);
  out.set(kvBytes, 8 + headerBytes.byteLength);
  return out;
}
```

- [ ] **Step 4: Run test — must pass.**

- [ ] **Step 5: Run `make checkall`** — clean.

- [ ] **Step 6: Commit.**

```bash
git add src/core/persistence.ts tests/persistence-codec.test.ts
git commit -m "feat(persistence): add encodePersistedConversation"
```

---

### Task 0.5: Implement `decodePersistedConversation` happy path (TDD)

**Files:**
- Modify: `src/core/persistence.ts`
- Modify: `tests/persistence-codec.test.ts`

- [ ] **Step 1: Append happy-path test:**

```ts
import {
  decodePersistedConversation,
  // ... existing imports
} from "../src/core/persistence.js";

describe("decodePersistedConversation — happy path", () => {
  test("round-trip preserves header + kvBytes byte-equal", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    const { header, kvBytes } = decodePersistedConversation(
      blob,
      SAMPLE_FINGERPRINT,
    );
    expect(header).toEqual(SAMPLE_HEADER);
    expect(kvBytes).toEqual(SAMPLE_KV);
  });
});
```

- [ ] **Step 2: Run — must fail** (function not exported).

- [ ] **Step 3: Implement `decodePersistedConversation`** in `src/core/persistence.ts`. Add `import` of `CorruptBlobError`, `IncompatibleConversationError` if not already present:

```ts
export function decodePersistedConversation(
  blob: Uint8Array,
  expectedFingerprint: ModelFingerprint,
): { header: PersistedConversationHeader; kvBytes: Uint8Array } {
  // 1. Magic.
  if (blob.byteLength < 8) {
    throw new CorruptBlobError("bad-magic", { byteLength: blob.byteLength });
  }
  for (let i = 0; i < 4; i++) {
    if (blob[i] !== KV_PERSISTENCE_MAGIC[i]) {
      throw new CorruptBlobError("bad-magic", {
        firstFour: Array.from(blob.subarray(0, 4)),
      });
    }
  }
  // 2. headerLen.
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const headerLen = dv.getUint32(4, /* LE */ true);
  if (8 + headerLen > blob.byteLength) {
    throw new CorruptBlobError("bad-header-len", {
      headerLen,
      blobLength: blob.byteLength,
    });
  }
  // 3. Header JSON.
  let header: PersistedConversationHeader;
  try {
    const json = new TextDecoder().decode(blob.subarray(8, 8 + headerLen));
    header = JSON.parse(json) as PersistedConversationHeader;
  } catch (e) {
    throw new CorruptBlobError("bad-header-json", {
      cause: e instanceof Error ? e.message : String(e),
    });
  }
  // 4. Schema version.
  if (header.schemaVersion !== KV_PERSISTENCE_SCHEMA_VERSION) {
    throw new IncompatibleConversationError("schema-mismatch", {
      got: header.schemaVersion,
      want: KV_PERSISTENCE_SCHEMA_VERSION,
    });
  }
  // 5. Fingerprint (with tokenizer separated).
  validateFingerprint(header.fingerprint, expectedFingerprint);
  // 6. byteSize sanity.
  const kvBytes = blob.subarray(8 + headerLen);
  if (kvBytes.byteLength !== header.byteSize) {
    throw new CorruptBlobError("byte-size-mismatch", {
      got: kvBytes.byteLength,
      want: header.byteSize,
    });
  }
  return { header, kvBytes: new Uint8Array(kvBytes) };
}

function validateFingerprint(got: ModelFingerprint, want: ModelFingerprint): void {
  // Check non-tokenizer fields first in deterministic order; only after
  // those match do we check tokenizerHash, so a tokenizer-only mismatch
  // surfaces as the more-specific tokenizer-mismatch reason.
  const fields: Array<keyof ModelFingerprint> = [
    "architecture",
    "vocabSize",
    "nEmbd",
    "nLayer",
    "nHead",
    "nHeadKV",
    "ropeBase",
    "quantType",
  ];
  for (const f of fields) {
    if (got[f] !== want[f]) {
      throw new IncompatibleConversationError("fingerprint-mismatch", {
        field: f,
        got: got[f],
        want: want[f],
      });
    }
  }
  if (got.tokenizerHash !== want.tokenizerHash) {
    throw new IncompatibleConversationError("tokenizer-mismatch", {
      got: got.tokenizerHash,
      want: want.tokenizerHash,
    });
  }
}
```

The `new Uint8Array(kvBytes)` copy decouples the returned bytes from the input blob's underlying buffer (so a caller-side detach of the input blob doesn't poison the snapshot stored in the pool).

- [ ] **Step 4: Run — must pass.**

- [ ] **Step 5: Run `make checkall`** — clean.

- [ ] **Step 6: Commit.**

```bash
git add src/core/persistence.ts tests/persistence-codec.test.ts
git commit -m "feat(persistence): add decodePersistedConversation happy path"
```

---

### Task 0.6: Add codec validation tests for all error sub-causes

**Files:**
- Modify: `tests/persistence-codec.test.ts`

- [ ] **Step 1: Append eight failure-mode tests** to `tests/persistence-codec.test.ts`:

```ts
import {
  CorruptBlobError,
  IncompatibleConversationError,
} from "../src/core/errors.js";

describe("decodePersistedConversation — failure modes", () => {
  test("throws CorruptBlobError(bad-magic) on wrong magic bytes", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    blob[0] = 0xff; // corrupt magic
    expect(() =>
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT),
    ).toThrow(CorruptBlobError);
    try {
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
    } catch (e) {
      expect((e as CorruptBlobError).reason).toBe("bad-magic");
    }
  });

  test("throws CorruptBlobError(bad-header-len) on overflowing headerLen", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    new DataView(blob.buffer).setUint32(4, blob.byteLength * 2, true);
    expect(() =>
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT),
    ).toThrow(CorruptBlobError);
  });

  test("throws CorruptBlobError(bad-header-json) on broken JSON", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    const dv = new DataView(blob.buffer);
    const headerLen = dv.getUint32(4, true);
    blob[8] = 0x7b; // '{'
    blob[8 + 1] = 0x21; // '!' — invalid JSON character right after {
    blob[8 + headerLen - 1] = 0x21; // '!' — break trailing brace
    expect(() =>
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT),
    ).toThrow(CorruptBlobError);
    try {
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
    } catch (e) {
      expect((e as CorruptBlobError).reason).toBe("bad-header-json");
    }
  });

  test("throws IncompatibleConversationError(schema-mismatch)", () => {
    const wrong = { ...SAMPLE_HEADER, schemaVersion: 99 as 1 };
    const blob = encodePersistedConversation(wrong, SAMPLE_KV);
    expect(() =>
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT),
    ).toThrow(IncompatibleConversationError);
    try {
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
    } catch (e) {
      expect((e as IncompatibleConversationError).reason).toBe("schema-mismatch");
      expect((e as IncompatibleConversationError).details).toEqual({
        got: 99,
        want: 1,
      });
    }
  });

  test("throws IncompatibleConversationError(fingerprint-mismatch) on first differing field", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    const want = { ...SAMPLE_FINGERPRINT, nLayer: 32 };
    try {
      decodePersistedConversation(blob, want);
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(IncompatibleConversationError);
      const ic = e as IncompatibleConversationError;
      expect(ic.reason).toBe("fingerprint-mismatch");
      expect(ic.details).toEqual({ field: "nLayer", got: 28, want: 32 });
    }
  });

  test("throws IncompatibleConversationError(tokenizer-mismatch) when only tokenizer differs", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    const want = { ...SAMPLE_FINGERPRINT, tokenizerHash: "b".repeat(64) };
    try {
      decodePersistedConversation(blob, want);
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(IncompatibleConversationError);
      expect((e as IncompatibleConversationError).reason).toBe(
        "tokenizer-mismatch",
      );
    }
  });

  test("throws CorruptBlobError(byte-size-mismatch) when kvBytes length differs", () => {
    // Build a blob whose header.byteSize disagrees with actual trailing length.
    const truncatedKv = SAMPLE_KV.slice(0, 8); // header still says byteSize=16
    const blob = encodePersistedConversation(SAMPLE_HEADER, truncatedKv);
    expect(() =>
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT),
    ).toThrow(CorruptBlobError);
    try {
      decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
    } catch (e) {
      expect((e as CorruptBlobError).reason).toBe("byte-size-mismatch");
    }
  });

  test("first differing field is the one reported (deterministic order)", () => {
    const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
    // Differ on architecture AND nLayer; expect architecture (earlier in
    // the deterministic order list) to be reported.
    const want = { ...SAMPLE_FINGERPRINT, architecture: "llama", nLayer: 32 };
    try {
      decodePersistedConversation(blob, want);
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as IncompatibleConversationError).details).toMatchObject({
        field: "architecture",
      });
    }
  });
});
```

- [ ] **Step 2: Run — should already pass** (decodePersistedConversation already implements the validation).

- [ ] **Step 3: Run `make checkall`** — clean.

- [ ] **Step 4: Commit.**

```bash
git add tests/persistence-codec.test.ts
git commit -m "test(persistence): cover all decode failure modes"
```

---

## Phase 1 — Engine API (no proxy, no IDB helper)

**Outcome at end of phase:** `WebLLM.exportConversation(conv)` and `WebLLM.importConversation(modelId, blob, options?)` work main-thread. `_modelManager`'s `ModelEntry` carries a cached `tokenizerHash` + `fingerprint`. WebGPU end-to-end test exercises the full round-trip with greedy-identity verification. No proxy changes yet — worker mode pays a structured-clone cost (Phase 2 fixes that).

### Task 1.1: Cache `tokenizerHash` on `ModelEntry` at load time

**Files:**
- Modify: `src/core/types.ts`
- Modify: `src/core/engine.ts`

- [ ] **Step 1: Re-read `src/core/types.ts` for `ModelEntry` shape, and `src/core/engine.ts` lines 1080-1450 (model load paths).**

- [ ] **Step 2: Extend `ModelEntry`** in `src/core/types.ts` (find the existing interface at line ~200):

```ts
import type { ModelFingerprint } from "./persistence.js";

export interface ModelEntry {
  // ... existing fields
  /** SHA-256 of canonical-key-sorted tokenizerConfig JSON; computed once at load. */
  tokenizerHash?: string;
  /** Cached fingerprint for persistence validation; computed once at load. */
  fingerprint?: ModelFingerprint;
}
```

- [ ] **Step 3: Populate `tokenizerHash` and `fingerprint` at load time in `engine.ts`.**

Find the two load paths around `engine.ts:1121-1122` and `engine.ts:1421-1422`. Both set `entry.hyperparams = parsed.hyperparams; entry.tokenizer = new Tokenizer(parsed.tokenizerConfig);`. After each of those sites, add:

```ts
entry.tokenizerHash = await computeTokenizerHash(parsed.tokenizerConfig);
entry.fingerprint = {
  architecture: String(parsed.hyperparams.architecture),
  vocabSize: parsed.hyperparams.vocabSize,
  nEmbd: parsed.hyperparams.embeddingLength,
  nLayer: parsed.hyperparams.layerCount,
  nHead: parsed.hyperparams.headCount,
  nHeadKV: parsed.hyperparams.headCountKv,
  ropeBase: parsed.hyperparams.ropeFreqBase ?? 10_000,
  quantType: parsed.quantType ?? "unknown",
  tokenizerHash: entry.tokenizerHash,
};
```

(Field names follow the project's existing `ModelHyperparams` convention. If a name doesn't exist, search `src/core/types.ts` for the matching field — e.g. `embeddingLength` vs `nEmbd` — and adjust.)

Add the import at the top of `engine.ts` if not already present:

```ts
import { computeTokenizerHash } from "./persistence.js";
import type { ModelFingerprint } from "./persistence.js";
```

- [ ] **Step 4: Run `make checkall`** — should be clean. The new fields are optional, so no existing call site breaks.

- [ ] **Step 5: Commit.**

```bash
git add src/core/types.ts src/core/engine.ts
git commit -m "feat(engine): cache tokenizerHash + fingerprint on model entry at load"
```

---

### Task 1.2: Add `WebLLM.exportConversation` (TDD)

**Files:**
- Modify: `src/core/engine.ts`
- Create: `tests/engine-conversation-persistence.test.ts`

- [ ] **Step 1: Re-read `src/core/engine.ts` lines 525-605** (existing conversation methods).

- [ ] **Step 2: Write failing test.** Create `tests/engine-conversation-persistence.test.ts`:

```ts
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { WebLLM } from "../src/core/engine.js";
import {
  KV_PERSISTENCE_MAGIC,
  decodePersistedConversation,
} from "../src/core/persistence.js";
import {
  ConversationBusyError,
  ConversationNotPopulatedError,
  CorruptBlobError,
  IncompatibleConversationError,
} from "../src/core/errors.js";

const HAS_WEBGPU = typeof navigator !== "undefined" && "gpu" in navigator;
// Reuse the same fixture path the rest of the WebGPU tests use.
const TINYLLAMA = process.env.WEBLLM_TINYLLAMA_GGUF
  ?? "fixtures/tinyllama-1.1b-chat-v1.0-q4_0.gguf";

describe.skipIf(!HAS_WEBGPU || !existsSync(TINYLLAMA))(
  "WebLLM.exportConversation",
  () => {
    let webllm: WebLLM;
    let modelId: string;

    beforeAll(async () => {
      webllm = new WebLLM({});
      const buf = await Bun.file(TINYLLAMA).arrayBuffer();
      const result = await webllm.loadModelFromBuffer(buf, "tinyllama");
      modelId = result.handle.id;
    });

    afterAll(async () => {
      await webllm.dispose();
    });

    test("export after one chatCompletion produces a blob with correct magic + parseable header", async () => {
      const conv = await webllm.createConversation(modelId);
      for await (const _ of webllm.chatCompletion(
        conv,
        [{ role: "user", content: "Hi" }],
        { temperature: 0, maxTokens: 4 },
      )) {
        /* drain */
      }
      const blob = await webllm.exportConversation(conv);
      expect(blob).toBeInstanceOf(Uint8Array);
      expect(blob.byteLength).toBeGreaterThan(8);
      expect(blob.slice(0, 4)).toEqual(KV_PERSISTENCE_MAGIC);

      const fingerprint = (webllm as unknown as {
        _modelManager: { get(id: string): { fingerprint: unknown } };
      })._modelManager.get(modelId).fingerprint;
      const { header, kvBytes } = decodePersistedConversation(
        blob,
        fingerprint as never,
      );
      expect(header.schemaVersion).toBe(1);
      expect(header.tokenIds.length).toBeGreaterThan(0);
      expect(kvBytes.byteLength).toBe(header.byteSize);
      await webllm.disposeConversation(conv);
    });

    test("export of un-populated conv throws ConversationNotPopulatedError", async () => {
      const conv = await webllm.createConversation(modelId);
      await expect(webllm.exportConversation(conv)).rejects.toBeInstanceOf(
        ConversationNotPopulatedError,
      );
      await webllm.disposeConversation(conv);
    });

    test("concurrent export while chatCompletion is in-flight throws ConversationBusyError", async () => {
      const conv = await webllm.createConversation(modelId);
      const stream = webllm.chatCompletion(
        conv,
        [{ role: "user", content: "Test" }],
        { temperature: 0, maxTokens: 8 },
      );
      // Pull first chunk to ensure the lock is acquired.
      const iter = stream[Symbol.asyncIterator]();
      await iter.next();
      await expect(webllm.exportConversation(conv)).rejects.toBeInstanceOf(
        ConversationBusyError,
      );
      // Drain the rest.
      for (;;) {
        const r = await iter.next();
        if (r.done) break;
      }
      await webllm.disposeConversation(conv);
    });
  },
);
```

- [ ] **Step 3: Run — must fail** (`exportConversation` not on `WebLLM`).

```bash
bun test tests/engine-conversation-persistence.test.ts
```

(If WebGPU/fixture missing, the suite skips entirely — that's expected; this task assumes a developer with a configured WebGPU + fixture environment, but skip-passing is the gate this whole file targets.)

- [ ] **Step 4: Implement `exportConversation`** — append to the public methods on `WebLLM` in `src/core/engine.ts`, near the existing `forkConversation` method (~line 572):

```ts
import {
  decodePersistedConversation,
  encodePersistedConversation,
  KV_PERSISTENCE_SCHEMA_VERSION,
  type ModelFingerprint,
} from "./persistence.js";

// ... in class WebLLM:
async exportConversation(conv: ConversationHandle): Promise<Uint8Array> {
  this.conversationPool.assertExists(conv);
  const release = this.conversationPool.tryAcquireLock(conv);
  if (!release) throw new ConversationBusyError(conv.id);
  try {
    const snap = this.conversationPool.get(conv);
    if (!snap) throw new ConversationNotPopulatedError(conv.id);
    const entry = this._modelManager.get(snap.modelHandleId);
    if (!entry) throw new ModelNotFoundError(snap.modelHandleId);
    if (!entry.fingerprint) {
      throw new InferenceEngineMissingError(snap.modelHandleId);
    }
    const header = {
      schemaVersion: KV_PERSISTENCE_SCHEMA_VERSION as 1,
      fingerprint: entry.fingerprint,
      conversationOptions: this.conversationPool.options(conv),
      tokenIds: snap.tokenIds,
      byteSize: snap.byteSize,
      savedAtMs: Date.now(),
    };
    return encodePersistedConversation(header, snap.kvBytes);
  } finally {
    release();
  }
}
```

Add the missing imports (`ConversationBusyError`, `ConversationNotPopulatedError`, `ModelNotFoundError`, `InferenceEngineMissingError` if not already imported in `engine.ts`).

- [ ] **Step 5: Run test — must pass.**

- [ ] **Step 6: Run `make checkall`** — clean.

- [ ] **Step 7: Commit.**

```bash
git add src/core/engine.ts tests/engine-conversation-persistence.test.ts
git commit -m "feat(engine): WebLLM.exportConversation"
```

---

### Task 1.3: Add `WebLLM.importConversation` (TDD)

**Files:**
- Modify: `src/core/engine.ts`
- Modify: `tests/engine-conversation-persistence.test.ts`

- [ ] **Step 1: Append failing tests** to the existing describe block:

```ts
describe.skipIf(!HAS_WEBGPU || !existsSync(TINYLLAMA))(
  "WebLLM.importConversation",
  () => {
    let webllm: WebLLM;
    let modelId: string;

    beforeAll(async () => {
      webllm = new WebLLM({});
      const buf = await Bun.file(TINYLLAMA).arrayBuffer();
      const result = await webllm.loadModelFromBuffer(buf, "tinyllama-imp");
      modelId = result.handle.id;
    });

    afterAll(async () => {
      await webllm.dispose();
    });

    test("export → import round-trip yields a fresh handle that decodes byte-identically", async () => {
      const convA = await webllm.createConversation(modelId);
      const messages = [{ role: "user" as const, content: "Hello" }];
      // Drive convA through one turn under greedy decoding.
      const collectedA: number[] = [];
      for await (const chunk of webllm.chatCompletion(convA, messages, {
        temperature: 0,
        maxTokens: 8,
      })) {
        if (typeof (chunk as { tokenId?: number }).tokenId === "number") {
          collectedA.push((chunk as { tokenId: number }).tokenId);
        }
      }
      const blob = await webllm.exportConversation(convA);

      // Reset by disposing convA and importing the blob into a fresh handle.
      await webllm.disposeConversation(convA);
      const convB = await webllm.importConversation(modelId, blob);
      expect(convB.id).not.toBe(convA.id);
      expect(convB.modelHandleId).toBe(modelId);

      // The post-import next turn should produce token-identical output to a
      // freshly-prefilled control under greedy decoding.
      const followUp = [
        ...messages,
        { role: "assistant" as const, content: "" },
        { role: "user" as const, content: "And again" },
      ];
      const collectedB: number[] = [];
      for await (const chunk of webllm.chatCompletion(convB, followUp, {
        temperature: 0,
        maxTokens: 8,
      })) {
        if (typeof (chunk as { tokenId?: number }).tokenId === "number") {
          collectedB.push((chunk as { tokenId: number }).tokenId);
        }
      }

      // Compare against a fresh-prefill control (no import).
      const convCtrl = await webllm.createConversation(modelId);
      const collectedCtrl: number[] = [];
      for await (const chunk of webllm.chatCompletion(convCtrl, followUp, {
        temperature: 0,
        maxTokens: 8,
      })) {
        if (typeof (chunk as { tokenId?: number }).tokenId === "number") {
          collectedCtrl.push((chunk as { tokenId: number }).tokenId);
        }
      }
      expect(collectedB).toEqual(collectedCtrl);

      await webllm.disposeConversation(convB);
      await webllm.disposeConversation(convCtrl);
    });

    test("import of fingerprint-mismatched blob throws IncompatibleConversationError", async () => {
      const conv = await webllm.createConversation(modelId);
      for await (const _ of webllm.chatCompletion(
        conv,
        [{ role: "user", content: "Hi" }],
        { temperature: 0, maxTokens: 4 },
      )) {
        /* drain */
      }
      const blob = await webllm.exportConversation(conv);
      // Hand-corrupt the header's nLayer field so it differs from the model.
      const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
      const headerLen = dv.getUint32(4, true);
      const headerJson = new TextDecoder().decode(
        blob.subarray(8, 8 + headerLen),
      );
      const parsed = JSON.parse(headerJson);
      parsed.fingerprint.nLayer = 999;
      const newJson = new TextEncoder().encode(JSON.stringify(parsed));
      // Build a new blob with the patched header (length may differ).
      const kvBytes = blob.subarray(8 + headerLen);
      const out = new Uint8Array(8 + newJson.byteLength + kvBytes.byteLength);
      out.set(blob.subarray(0, 4), 0);
      new DataView(out.buffer).setUint32(4, newJson.byteLength, true);
      out.set(newJson, 8);
      out.set(kvBytes, 8 + newJson.byteLength);

      await expect(
        webllm.importConversation(modelId, out),
      ).rejects.toBeInstanceOf(IncompatibleConversationError);

      await webllm.disposeConversation(conv);
    });

    test("import of corrupt-magic blob throws CorruptBlobError", async () => {
      const blob = new Uint8Array([0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
      await expect(
        webllm.importConversation(modelId, blob),
      ).rejects.toBeInstanceOf(CorruptBlobError);
    });
  },
);
```

- [ ] **Step 2: Run — must fail** (`importConversation` not implemented).

- [ ] **Step 3: Implement `importConversation` in `engine.ts`**, after `exportConversation`:

```ts
async importConversation(
  modelHandleId: string,
  blob: Uint8Array,
  options?: ConversationOptions,
): Promise<ConversationHandle> {
  const entry = this._modelManager.get(modelHandleId);
  if (!entry) throw new ModelNotFoundError(modelHandleId);
  if (!entry.loaded || !entry.tokenizer) {
    throw new ModelNotLoadedError(modelHandleId);
  }
  const inf = this.inferenceEngines.get(modelHandleId);
  if (!inf) throw new InferenceEngineMissingError(modelHandleId);
  if (!inf.flashAttn) {
    throw new Error(
      `importConversation requires FA mode; "${modelHandleId}" is in manual mode.`,
    );
  }
  if (!entry.fingerprint) {
    throw new InferenceEngineMissingError(modelHandleId);
  }
  const { header, kvBytes } = decodePersistedConversation(
    blob,
    entry.fingerprint,
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
```

- [ ] **Step 4: Run test — must pass.**

- [ ] **Step 5: Run `make checkall`** — clean.

- [ ] **Step 6: Commit.**

```bash
git add src/core/engine.ts tests/engine-conversation-persistence.test.ts
git commit -m "feat(engine): WebLLM.importConversation"
```

---

## Phase 2 — Worker bridge: transfer-list extension + proxy mirrors

**Outcome at end of phase:** `WebLLMProxy.exportConversation` / `importConversation` round-trip through the worker bridge with transferable buffers in both directions. PROXIED_METHODS sentinel updated. Proxy-integration suite gains a persistence describe block.

### Task 2.1: Extend `WorkerToProxy.method-result` with optional transfer

**Files:**
- Modify: `src/core/worker-bridge.ts`

- [ ] **Step 1: Re-read `src/core/worker-bridge.ts`.**

- [ ] **Step 2: Add `transfer?: Transferable[]` to the `method-result` variant** of `WorkerToProxy`:

```ts
export type WorkerToProxy =
  | { type: "init-done"; id: RequestId }
  | {
      type: "method-result";
      id: RequestId;
      value: unknown;
      transfer?: Transferable[]; // NEW — informational; real transfer
                                  // is the second arg to postMessage.
    }
  // ... rest unchanged
;
```

- [ ] **Step 3: Run `make checkall`** — clean (no consumers yet; type widens compatibly).

- [ ] **Step 4: Commit.**

```bash
git add src/core/worker-bridge.ts
git commit -m "feat(worker-bridge): method-result accepts optional transfer list"
```

---

### Task 2.2: Worker host transfer allowlist + signature widening

**Files:**
- Modify: `src/core/webllm-worker-host.ts`
- Modify: `tests/webllm-worker-host.test.ts`

- [ ] **Step 1: Re-read `src/core/webllm-worker-host.ts`.**

- [ ] **Step 2: Widen `WorkerHostOptions.postMessage`** signature and add the allowlist:

```ts
export interface WorkerHostOptions {
  // biome-ignore lint/suspicious/noExplicitAny: reflect-dispatch by name
  engine: any;
  /** Send a message to the proxy. Optional second arg lists Transferables
   *  that should be moved (not copied) across the boundary. */
  postMessage(m: WorkerToProxy, transfer?: Transferable[]): void;
  receive(handler: (m: ProxyToWorker) => void): void;
  log?(level: "info" | "warn" | "error", message: string): void;
}

const TRANSFERS_BUFFER_ON_RETURN = new Set(["exportConversation"]);
```

- [ ] **Step 3: Update `handleMethodCall`** to populate the transfer list for allowlisted returns:

```ts
async function handleMethodCall(
  msg: Extract<ProxyToWorker, { type: "method-call" }>,
) {
  try {
    const fn = opts.engine[msg.name];
    if (typeof fn !== "function") {
      throw new Error(`unknown engine method: ${msg.name}`);
    }
    const value = await fn.apply(opts.engine, msg.args);

    const stripsInference =
      msg.name === "loadModelFromBuffer" || msg.name === "loadModelFromUrl";
    const sanitized =
      stripsInference &&
      value &&
      typeof value === "object" &&
      "handle" in (value as object)
        ? {
            handle: (value as { handle: unknown }).handle,
            metadata: (value as { metadata?: unknown }).metadata,
          }
        : value;

    let transfer: Transferable[] | undefined;
    if (
      TRANSFERS_BUFFER_ON_RETURN.has(msg.name) &&
      sanitized instanceof Uint8Array
    ) {
      transfer = [sanitized.buffer as ArrayBuffer];
    }

    opts.postMessage(
      { type: "method-result", id: msg.id, value: sanitized, transfer },
      transfer,
    );
  } catch (e) {
    opts.postMessage({
      type: "method-error",
      id: msg.id,
      error: serializeError(e),
    });
  }
}
```

- [ ] **Step 4: Add a worker-host unit test** — append to `tests/webllm-worker-host.test.ts`:

```ts
test("exportConversation result is transferred (envelope.transfer populated)", async () => {
  const sentMessages: Array<{ msg: WorkerToProxy; transfer?: Transferable[] }> = [];
  const handlers = { receive: null as ((m: ProxyToWorker) => void) | null };
  const engine = {
    async exportConversation() {
      return new Uint8Array([1, 2, 3, 4, 5]);
    },
    async dispose() {},
  };
  startWorkerHost({
    engine,
    postMessage: (msg, transfer) => sentMessages.push({ msg, transfer }),
    receive: (h) => { handlers.receive = h; },
  });
  handlers.receive?.({
    type: "method-call",
    id: 1 as RequestId,
    name: "exportConversation",
    args: [{ id: "c1", modelHandleId: "m1" }],
  });
  // Settle the microtask queue.
  await new Promise((r) => setTimeout(r, 0));
  const result = sentMessages.find(
    (e) => e.msg.type === "method-result",
  );
  expect(result).toBeDefined();
  expect(result?.transfer?.length).toBe(1);
  expect(result?.transfer?.[0]).toBeInstanceOf(ArrayBuffer);
});

test("non-allowlisted method does NOT populate transfer list", async () => {
  const sentMessages: Array<{ msg: WorkerToProxy; transfer?: Transferable[] }> = [];
  const handlers = { receive: null as ((m: ProxyToWorker) => void) | null };
  const engine = {
    async embed() { return new Float32Array([1, 2, 3]); },
    async dispose() {},
  };
  startWorkerHost({
    engine,
    postMessage: (msg, transfer) => sentMessages.push({ msg, transfer }),
    receive: (h) => { handlers.receive = h; },
  });
  handlers.receive?.({
    type: "method-call",
    id: 1 as RequestId,
    name: "embed",
    args: ["m1", "hi"],
  });
  await new Promise((r) => setTimeout(r, 0));
  const result = sentMessages.find(
    (e) => e.msg.type === "method-result",
  );
  expect(result?.transfer).toBeUndefined();
});
```

(Add necessary imports: `RequestId`, `WorkerToProxy`, `ProxyToWorker` from `../src/core/worker-bridge.js`.)

- [ ] **Step 5: Run `bun test tests/webllm-worker-host.test.ts`** — must pass (existing + 2 new).

- [ ] **Step 6: Run `make checkall`** — clean.

- [ ] **Step 7: Commit.**

```bash
git add src/core/webllm-worker-host.ts tests/webllm-worker-host.test.ts
git commit -m "feat(worker-host): transfer-on-return allowlist for exportConversation"
```

---

### Task 2.3: Add proxy mirrors + sentinel update

**Files:**
- Modify: `src/core/webllm-proxy.ts`
- Modify: `tests/webllm-proxy-surface.test.ts`

- [ ] **Step 1: Re-read `src/core/webllm-proxy.ts`** lines 110-180 (existing public surface).

- [ ] **Step 2: Add the two mirrors** to the `WebLLMProxy` class. After the existing `forkConversation` arrow-method:

```ts
exportConversation = (conv: ConversationHandle): Promise<Uint8Array> =>
  this.callMethod<Uint8Array>("exportConversation", [conv]);

importConversation = (
  modelHandleId: string,
  blob: Uint8Array,
  options?: ConversationOptions,
): Promise<ConversationHandle> =>
  this.callMethod<ConversationHandle>(
    "importConversation",
    [modelHandleId, blob, options],
    [blob.buffer as ArrayBuffer],
  );
```

- [ ] **Step 3: Update the surface sentinel.**

Edit `tests/webllm-proxy-surface.test.ts` and append two entries to `PROXIED_METHODS`:

```ts
const PROXIED_METHODS: ReadonlyArray<keyof WebLLMProxy & string> = [
  // ... existing entries
  "exportConversation",
  "importConversation",
];
```

- [ ] **Step 4: Run `bun test tests/webllm-proxy-surface.test.ts`** — must pass (TS catches missing proxy mirror at compile; runtime test confirms WebLLM has the methods, which Phase 1 added).

- [ ] **Step 5: Run `make checkall`** — clean.

- [ ] **Step 6: Commit.**

```bash
git add src/core/webllm-proxy.ts tests/webllm-proxy-surface.test.ts
git commit -m "feat(proxy): mirror exportConversation + importConversation"
```

---

### Task 2.4: Add proxy-integration tests for persistence round-trip

**Files:**
- Modify: `tests/webllm-proxy-integration.test.ts`

- [ ] **Step 1: Re-read** the existing `WebLLMProxy — conversation lifecycle` block at the bottom of `tests/webllm-proxy-integration.test.ts` (added by probe #6 closure).

- [ ] **Step 2: Append a `WebLLMProxy — persistence` describe block:**

```ts
describe("WebLLMProxy — persistence", () => {
  test("exportConversation round-trips a Uint8Array (magic preserved)", async () => {
    const KV_PERSISTENCE_MAGIC = new Uint8Array([0x57, 0x4c, 0x4b, 0x56]);
    const sampleBlob = new Uint8Array([
      ...KV_PERSISTENCE_MAGIC,
      ...Array(200).fill(0),
    ]);
    const engine = {
      async exportConversation(_conv: unknown) {
        return sampleBlob;
      },
      async dispose() {},
    };
    const { worker, hostPost, hostReceive } = makeInProcessChannel();
    startWorkerHost({ engine, postMessage: hostPost, receive: hostReceive });
    const proxy = await WebLLMProxy.fromWorker(worker);
    const result = await proxy.exportConversation({
      id: "c1",
      modelHandleId: "m1",
    });
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.slice(0, 4)).toEqual(KV_PERSISTENCE_MAGIC);
  });

  test("importConversation round-trips and returns the worker's handle", async () => {
    const seenBlobLen: number[] = [];
    const engine = {
      async importConversation(modelId: string, blob: Uint8Array) {
        seenBlobLen.push(blob.byteLength);
        return { id: `c-imp-${modelId}`, modelHandleId: modelId };
      },
      async dispose() {},
    };
    const { worker, hostPost, hostReceive } = makeInProcessChannel();
    startWorkerHost({ engine, postMessage: hostPost, receive: hostReceive });
    const proxy = await WebLLMProxy.fromWorker(worker);
    const blob = new Uint8Array([0x57, 0x4c, 0x4b, 0x56, 0, 0, 0, 0]);
    const conv = await proxy.importConversation("m1", blob);
    expect(conv).toEqual({ id: "c-imp-m1", modelHandleId: "m1" });
    expect(seenBlobLen).toEqual([8]);
  });

  test("CorruptBlobError thrown worker-side is reconstructed main-side as instanceof", async () => {
    const { CorruptBlobError } = await import("../src/core/errors.js");
    const engine = {
      async importConversation() {
        throw new CorruptBlobError("bad-magic", { firstFour: [0xff, 0xff, 0xff, 0xff] });
      },
      async dispose() {},
    };
    const { worker, hostPost, hostReceive } = makeInProcessChannel();
    startWorkerHost({ engine, postMessage: hostPost, receive: hostReceive });
    const proxy = await WebLLMProxy.fromWorker(worker);
    await expect(
      proxy.importConversation("m1", new Uint8Array([0])),
    ).rejects.toBeInstanceOf(CorruptBlobError);
  });

  test("IncompatibleConversationError preserves reason + details across boundary", async () => {
    const { IncompatibleConversationError } = await import(
      "../src/core/errors.js"
    );
    const engine = {
      async importConversation() {
        throw new IncompatibleConversationError("fingerprint-mismatch", {
          field: "nLayer",
          got: 28,
          want: 32,
        });
      },
      async dispose() {},
    };
    const { worker, hostPost, hostReceive } = makeInProcessChannel();
    startWorkerHost({ engine, postMessage: hostPost, receive: hostReceive });
    const proxy = await WebLLMProxy.fromWorker(worker);
    try {
      await proxy.importConversation("m1", new Uint8Array(8));
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(IncompatibleConversationError);
      expect((e as InstanceType<typeof IncompatibleConversationError>).reason)
        .toBe("fingerprint-mismatch");
      expect((e as InstanceType<typeof IncompatibleConversationError>).details)
        .toEqual({ field: "nLayer", got: 28, want: 32 });
    }
  });
});
```

- [ ] **Step 3: Run** `bun test tests/webllm-proxy-integration.test.ts` — must pass (existing 11 + 4 new).

- [ ] **Step 4: Run `make checkall`** — clean.

- [ ] **Step 5: Commit.**

```bash
git add tests/webllm-proxy-integration.test.ts
git commit -m "test(proxy): cover persistence round-trip + error codec"
```

---

## Phase 3 — IndexedDB helper module

**Outcome at end of phase:** `IndexedDBConversationStore` exported from `@paulrobello/webllm/persistence`. Apps that don't use persistence pay zero bytes for it.

### Task 3.1: Implement `IndexedDBConversationStore`

**Files:**
- Create: `src/persistence/indexeddb-store.ts`
- Create: `tests/persistence-indexeddb-store.test.ts`

- [ ] **Step 1: Re-read `src/core/pipeline-cache.ts`** for the IDB-pattern reference.

- [ ] **Step 2: Write the failing tests.** Create `tests/persistence-indexeddb-store.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  PersistenceQuotaError,
  PersistenceUnavailableError,
} from "../src/core/errors.js";
import { IndexedDBConversationStore } from "../src/persistence/indexeddb-store.js";

const indexedDBAvailable = typeof indexedDB !== "undefined";
const DB_NAME = "test-conv-store";

describe.skipIf(!indexedDBAvailable)("IndexedDBConversationStore", () => {
  let store: IndexedDBConversationStore;

  beforeEach(async () => {
    await new Promise<void>((resolve) => {
      const req = indexedDB.deleteDatabase(DB_NAME);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
      req.onblocked = () => resolve();
    });
    store = new IndexedDBConversationStore(DB_NAME);
  });

  afterEach(async () => {
    await store.close();
  });

  test("put + get round-trip", async () => {
    const blob = new Uint8Array([1, 2, 3, 4, 5]);
    await store.put("k1", blob);
    const got = await store.get("k1");
    expect(got).toEqual(blob);
  });

  test("get of missing key returns undefined", async () => {
    expect(await store.get("nope")).toBeUndefined();
  });

  test("delete removes the entry", async () => {
    await store.put("k1", new Uint8Array([7]));
    await store.delete("k1");
    expect(await store.get("k1")).toBeUndefined();
  });

  test("list returns metadata only (byteLength + savedAtMs)", async () => {
    await store.put("k1", new Uint8Array([1, 2, 3]));
    await store.put("k2", new Uint8Array(100));
    const entries = await store.list();
    expect(entries).toHaveLength(2);
    const k1 = entries.find((e) => e.key === "k1");
    expect(k1?.byteLength).toBe(3);
    expect(k1?.savedAtMs).toBeGreaterThan(0);
  });

  test("put overwrite replaces both blob and metadata", async () => {
    await store.put("k1", new Uint8Array([1, 2, 3]));
    await store.put("k1", new Uint8Array([1, 2, 3, 4, 5, 6]));
    const got = await store.get("k1");
    expect(got?.byteLength).toBe(6);
    const entries = await store.list();
    expect(entries.find((e) => e.key === "k1")?.byteLength).toBe(6);
  });

  test("clear removes all entries", async () => {
    await store.put("k1", new Uint8Array([1]));
    await store.put("k2", new Uint8Array([2]));
    await store.clear();
    expect(await store.list()).toEqual([]);
  });

  test("open() is idempotent", async () => {
    await store.open();
    await store.open();
    await store.put("k1", new Uint8Array([9]));
    expect((await store.get("k1"))?.[0]).toBe(9);
  });

  test("indexedDB-missing throws PersistenceUnavailableError", async () => {
    const original = (globalThis as { indexedDB?: unknown }).indexedDB;
    (globalThis as { indexedDB?: unknown }).indexedDB = undefined;
    try {
      const orphan = new IndexedDBConversationStore("no-idb");
      await expect(orphan.put("k", new Uint8Array([1]))).rejects.toBeInstanceOf(
        PersistenceUnavailableError,
      );
    } finally {
      (globalThis as { indexedDB?: unknown }).indexedDB = original;
    }
  });

  test("QuotaExceededError surfaces as PersistenceQuotaError", async () => {
    // Patch the `put` request path to simulate a quota error. We replace
    // the IDBObjectStore.put to throw a DOMException("QuotaExceededError").
    const QuotaError =
      typeof DOMException !== "undefined"
        ? new DOMException("quota exceeded", "QuotaExceededError")
        : Object.assign(new Error("quota exceeded"), {
            name: "QuotaExceededError",
          });
    await store.open();
    const db = (store as unknown as { db: IDBDatabase }).db;
    const origTransaction = db.transaction.bind(db);
    db.transaction = ((...args: Parameters<IDBDatabase["transaction"]>) => {
      const tx = origTransaction(...args);
      const origGetStore = tx.objectStore.bind(tx);
      tx.objectStore = (name: string) => {
        const s = origGetStore(name);
        const origPut = s.put.bind(s);
        s.put = ((...putArgs: Parameters<IDBObjectStore["put"]>) => {
          const req = origPut(...putArgs);
          // Force the request to error on next microtask.
          queueMicrotask(() => {
            Object.defineProperty(req, "error", {
              get() { return QuotaError; },
            });
            req.dispatchEvent(new Event("error"));
          });
          return req;
        }) as IDBObjectStore["put"];
        return s;
      };
      return tx;
    }) as IDBDatabase["transaction"];
    await expect(
      store.put("k1", new Uint8Array([1, 2, 3])),
    ).rejects.toBeInstanceOf(PersistenceQuotaError);
  });
});
```

(The QuotaExceededError test is intricate because faking it cleanly requires patching the request-level error path. If `fake-indexeddb` exposes a more direct mechanism, prefer that — search the project for prior usage.)

- [ ] **Step 3: Run** `bun test tests/persistence-indexeddb-store.test.ts` — must fail with module-not-found.

- [ ] **Step 4: Implement the helper.** Create `src/persistence/indexeddb-store.ts`:

```ts
/**
 * IndexedDB-backed store for persisted-conversation blobs.
 * Companion to the engine's `exportConversation` / `importConversation`
 * primitives. Optional: apps that want OPFS or server-side storage
 * implement their own store against the same Uint8Array contract.
 *
 * Spec: 2026-05-03-prefix-cache-persistence-design.md
 */

import {
  PersistenceIOError,
  PersistenceQuotaError,
  PersistenceUnavailableError,
} from "../core/errors.js";

export interface ConversationStoreEntry {
  key: string;
  byteLength: number;
  savedAtMs: number;
}

interface ConversationMetaRecord {
  byteLength: number;
  savedAtMs: number;
}

const PAYLOAD_STORE = "conversations";
const META_STORE = "conversation-meta";

export class IndexedDBConversationStore {
  private dbName: string;
  private db: IDBDatabase | null = null;

  constructor(dbName: string) {
    this.dbName = dbName;
  }

  async open(): Promise<void> {
    if (this.db) return;
    if (typeof indexedDB === "undefined") {
      throw new PersistenceUnavailableError("indexeddb-missing");
    }
    return new Promise<void>((resolve, reject) => {
      let req: IDBOpenDBRequest;
      try {
        req = indexedDB.open(this.dbName, 1);
      } catch (e) {
        reject(new PersistenceUnavailableError("indexeddb-blocked", e));
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(PAYLOAD_STORE)) {
          db.createObjectStore(PAYLOAD_STORE);
        }
        if (!db.objectStoreNames.contains(META_STORE)) {
          db.createObjectStore(META_STORE);
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve();
      };
      req.onerror = () =>
        reject(new PersistenceUnavailableError("open-failed", req.error));
      req.onblocked = () =>
        reject(new PersistenceUnavailableError("indexeddb-blocked"));
    });
  }

  async put(key: string, blob: Uint8Array): Promise<void> {
    await this.open();
    const db = this.db!;
    const meta: ConversationMetaRecord = {
      byteLength: blob.byteLength,
      savedAtMs: Date.now(),
    };
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onabort = () => {
        const err = tx.error;
        if (err && err.name === "QuotaExceededError") {
          reject(new PersistenceQuotaError(blob.byteLength));
        } else {
          reject(new PersistenceIOError("transaction-aborted", err));
        }
      };
      tx.onerror = () => {
        const err = tx.error;
        if (err && err.name === "QuotaExceededError") {
          reject(new PersistenceQuotaError(blob.byteLength));
        } else {
          reject(new PersistenceIOError("io-failure", err));
        }
      };
      try {
        const payloadReq = tx.objectStore(PAYLOAD_STORE).put(blob, key);
        payloadReq.onerror = () => {
          if (payloadReq.error?.name === "QuotaExceededError") {
            reject(new PersistenceQuotaError(blob.byteLength));
            tx.abort();
          }
        };
        tx.objectStore(META_STORE).put(meta, key);
      } catch (e) {
        reject(new PersistenceIOError("io-failure", e));
      }
    });
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    await this.open();
    const db = this.db!;
    return new Promise<Uint8Array | undefined>((resolve, reject) => {
      const req = db
        .transaction(PAYLOAD_STORE, "readonly")
        .objectStore(PAYLOAD_STORE)
        .get(key);
      req.onsuccess = () => resolve(req.result ?? undefined);
      req.onerror = () =>
        reject(new PersistenceIOError("io-failure", req.error));
    });
  }

  async delete(key: string): Promise<void> {
    await this.open();
    const db = this.db!;
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () =>
        reject(new PersistenceIOError("io-failure", tx.error));
      tx.onabort = () =>
        reject(new PersistenceIOError("transaction-aborted", tx.error));
      tx.objectStore(PAYLOAD_STORE).delete(key);
      tx.objectStore(META_STORE).delete(key);
    });
  }

  async list(): Promise<ConversationStoreEntry[]> {
    await this.open();
    const db = this.db!;
    return new Promise<ConversationStoreEntry[]>((resolve, reject) => {
      const tx = db.transaction(META_STORE, "readonly");
      const store = tx.objectStore(META_STORE);
      const out: ConversationStoreEntry[] = [];
      const req = store.openCursor();
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          const key = String(cursor.key);
          const meta = cursor.value as ConversationMetaRecord;
          out.push({
            key,
            byteLength: meta.byteLength,
            savedAtMs: meta.savedAtMs,
          });
          cursor.continue();
        } else {
          resolve(out);
        }
      };
      req.onerror = () =>
        reject(new PersistenceIOError("io-failure", req.error));
    });
  }

  async clear(): Promise<void> {
    await this.open();
    const db = this.db!;
    return new Promise<void>((resolve, reject) => {
      const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () =>
        reject(new PersistenceIOError("io-failure", tx.error));
      tx.objectStore(PAYLOAD_STORE).clear();
      tx.objectStore(META_STORE).clear();
    });
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}
```

- [ ] **Step 5: Run tests — must pass.**

If the QuotaExceededError test is flaky against `fake-indexeddb`, mark just that one test as `.skip` with a TODO comment and ship the rest. (Real-IDB browsers will exercise it; the unit-test-only scenario is brittle.)

- [ ] **Step 6: Run `make checkall`** — clean.

- [ ] **Step 7: Commit.**

```bash
git add src/persistence/indexeddb-store.ts tests/persistence-indexeddb-store.test.ts
git commit -m "feat(persistence): IndexedDBConversationStore helper"
```

---

### Task 3.2: Wire the `./persistence` subpath export

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Re-read `package.json`** for the existing `exports` field shape.

- [ ] **Step 2: Add the subpath export.** Find the existing `"exports"` block (or create one if absent) and add:

```json
{
  "exports": {
    ".": "./dist/index.js",
    "./persistence": "./dist/persistence/indexeddb-store.js"
  }
}
```

(If the project uses a different output path — check `tsconfig.json` `outDir` and existing build scripts — adjust accordingly. The path must match what the build emits.)

- [ ] **Step 3: Verify the build still emits `dist/persistence/indexeddb-store.js`.**

```bash
bun run build 2>&1 | tail -20
ls dist/persistence/ 2>&1 || echo "no persistence subdir"
```

If the build doesn't pick up `src/persistence/`, check the build script (`scripts/build.ts` or equivalent). The project uses `tsc` for type emission and possibly a bundler — match whichever convention is in place.

- [ ] **Step 4: Run `make checkall`** — clean.

- [ ] **Step 5: Commit.**

```bash
git add package.json
git commit -m "feat(package): expose @paulrobello/webllm/persistence subpath"
```

---

## Phase 4 — Docs and TODO closure

### Task 4.1: README persistence section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Re-read** the existing README.md for tone and section conventions.

- [ ] **Step 2: Add a "Conversation persistence" section** under the existing "Conversations" / "Prefix caching" area, with this content:

```markdown
### Conversation persistence

Conversations and their KV state evaporate on page reload. Apps that
want to preserve conversation state across reloads use the
`exportConversation` / `importConversation` primitives plus the
optional `IndexedDBConversationStore` helper.

```ts
import { WebLLM } from "@paulrobello/webllm";
import { IndexedDBConversationStore } from "@paulrobello/webllm/persistence";

const webllm = await WebLLM.init({ worker: true });
const { handle: model } = await webllm.loadModelFromUrl(url, "qwen3-8b");
const store = new IndexedDBConversationStore("my-app-conversations");

let conv;
const blob = await store.get("user-42-session");
try {
  conv = blob
    ? await webllm.importConversation(model.id, blob)
    : await webllm.createConversation(model.id);
} catch (e) {
  // IncompatibleConversationError or CorruptBlobError → discard, restart
  await store.delete("user-42-session");
  conv = await webllm.createConversation(model.id);
}

// Per turn, after chatCompletion settles:
const fresh = await webllm.exportConversation(conv);
await store.put("user-42-session", fresh);
```

Apps that need OPFS, server-side sync, or encrypted-at-rest implement
their own store against the same `Uint8Array` contract — the engine
primitives don't depend on any specific storage backend.

See [`docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`](docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md)
for the wire format, error taxonomy, and worker-mode marshaling
details.
```

- [ ] **Step 3: Run `make checkall`** — clean.

- [ ] **Step 4: Commit.**

```bash
git add README.md
git commit -m "docs(README): conversation persistence section"
```

---

### Task 4.2: TODO closure stub

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Re-read** the existing TODO.md item-11 follow-up #5 stub (around line 1091).

- [ ] **Step 2: Replace the stub** with a closure entry:

```markdown
    - **#5 Persistence across reloads — CLOSED 2026-05-03 (consumer
      ask honored).** Two-tier design: engine primitives
      `exportConversation(conv)` / `importConversation(modelId,
      blob, options?)` ship in core; `IndexedDBConversationStore`
      ships behind `@paulrobello/webllm/persistence` subpath
      (apps wanting OPFS / server sync / encrypted-at-rest
      implement their own store against the same Uint8Array
      contract). Five new error classes
      (`IncompatibleConversationError` / `CorruptBlobError` /
      `PersistenceUnavailableError` / `PersistenceQuotaError` /
      `PersistenceIOError`); model-fingerprint + tokenizer-hash
      gate; integer `schemaVersion` (no migrations); per-method
      transfer-allowlist on the worker bridge for fast large-blob
      returns. Spec
      [`docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`](docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md);
      plan
      [`docs/superpowers/plans/2026-05-03-prefix-cache-persistence.md`](docs/superpowers/plans/2026-05-03-prefix-cache-persistence.md).
```

- [ ] **Step 3: Run `make checkall`** — clean.

- [ ] **Step 4: Commit.**

```bash
git add TODO.md
git commit -m "docs(TODO): close item 11 follow-up #5 (persistence across reloads)"
```

---

## Self-Review

**Spec coverage check:**
- ✅ §"Use case (b)" — durability via Phase 3 helper + opt-in keying
- ✅ §"Architecture" two tiers — Phase 0/1/2 (engine) and Phase 3 (helper)
- ✅ §"Wire format" magic + uint32 LE headerLen + JSON header + kvBytes — Task 0.4
- ✅ §"Header schema" all fields — Task 0.4 + 1.1 + 1.2
- ✅ §"Validation order" 6-step list — Task 0.5 + 0.6
- ✅ §"Engine API" exportConversation + importConversation + modelFingerprint — Task 1.1 / 1.2 / 1.3
- ✅ §"Concurrency guarantees" lock acquisition — Task 1.2 + concurrency test
- ✅ §"Surface sentinel" — Task 2.3
- ✅ §"Worker bridge" three changes — Task 2.1 + 2.2
- ✅ §"Detached buffer semantics" — documented in Task 4.1 README
- ✅ §"Persistence helper" full surface — Task 3.1
- ✅ §"Bundle exports" subpath — Task 3.2
- ✅ §"Failure-mode matrix" all 18 rows — Task 0.6 + 1.2 + 1.3 + 3.1
- ✅ §"Error-codec wiring" 5 entries — Task 0.2
- ✅ §"Testing strategy" all 5 test files — Task 0.3-0.6, 1.2-1.3, 2.4, 3.1, 0.2

**Placeholder scan:** none found. Every code block is concrete; every command has an expected outcome.

**Type consistency:** verified `ModelFingerprint` field names match across persistence.ts (declaration), engine.ts (cache + use), and tests (fixture). `ConversationOptions` shape is `{maxContextTokens?: number}` everywhere. `ConversationHandle` is `{id: string; modelHandleId: string}` everywhere.

**Scope check:** 14 tasks across 5 phases. Each phase is independently shippable; each task is one commit. Total touch surface as estimated in the spec.

---

## Execution Handoff

Per global preference (CLAUDE.md): proceed with **superpowers:subagent-driven-development**. Plan approval is execution approval. Begin Task 0.1 immediately.
