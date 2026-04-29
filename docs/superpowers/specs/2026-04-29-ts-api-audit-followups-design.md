# TS API Audit Follow-ups — Design

**Date:** 2026-04-29
**Status:** Draft → User review
**Source:** TODO.md:926-1020 ("TS API audit follow-ups", items a-f)

## Goal

Land all six deferred TS API audit items in one work cycle as a coherent
"public API hygiene" pass. Each item was previously deferred behind a
"consumer ask" gate; the user has now triggered the full set.

## Scope

Six items from the TODO:
- **(a)** Split `GenerationConfig` into public/internal halves.
- **(b)** Path-specific `WebLLMConfig.device` plumbing.
- **(c)** Expose Qwen3 default-sampling profile + add `sampling` flag.
- **(d)** Engine accessor convention (properties everywhere).
- **(e)** `ToolSystem` swappability on `Character`.
- **(f)** Lower-priority polish (literal union, JSDoc, README).

## Phasing

Single spec, three implementation phases. Each phase ≤5 files (project
doctrine), `make checkall` ship-gate, separate commit before next phase.

| Phase | Items | Files (est.) | Theme |
|---|---|---|---|
| 1 | a, b, c | 6 | Type surface |
| 2 | d, e | ≤5 | Engine + Character |
| 3 | f | ≤3 | Polish |

If any phase hits the 5-file ceiling, split into sub-checkpoints
(e.g. 1a / 1b) on the writing-plans pass.

---

## Item (a) — `GenerationConfig` split

### Decision summary

- **Rename** current 22-field type → `InternalGenerationOptions`
  (engine-internal, unexported).
- **Introduce** new public 7-field `GenerationConfig`.
- **Drop** `prompt: string` from the type entirely (unused by
  `Generator.generate()`; it takes `promptTokenIds` positionally).
- **Move** `signal` from a separate parameter on `Generator.generate()`
  into the config object (matches `CompletionConfig` shape).

### New types

```ts
// src/inference/generation.ts

/** Configuration for a single generation request. */
export interface GenerationConfig {
  /** Maximum number of tokens to generate. */
  maxTokens: number;
  /** Sampling temperature. 0 = greedy. */
  temperature: number;
  /** Top-K sampling parameter. 0 = disabled. */
  topK: number;
  /** Top-P (nucleus) sampling parameter. 1.0 = disabled. */
  topP: number;
  /** Repetition penalty multiplier. 1.0 = disabled. */
  repetitionPenalty: number;
  /** Optional custom stop token IDs. */
  stopTokens?: number[];
  /** Optional AbortSignal to cancel mid-stream. */
  signal?: AbortSignal;
}

/** Internal-only options including Qwen3 chat-control steering fields. */
export interface InternalGenerationOptions extends GenerationConfig {
  forbiddenReentryTokens?: number[];
  thinkingOpenTokenId?: number;
  thinkingCloseTokenId?: number;
  enforceSingleThinkBlock?: boolean;
  maskedTokensWhileThinking?: number[];
  tokenizer?: Tokenizer;
  maskedTokensAfterThinkingUntilAnswer?: number[];
  requireVisibleAnswerAfterThinking?: boolean;
  requireVisibleAnswerBeforeStop?: boolean;
  suppressWhitespaceOnlyAfterThinking?: boolean;
  suppressWhitespaceOnlyUntilAnswer?: boolean;
  maskedTokensAfterAnswerStarts?: number[];
  requireLeadingWhitespaceAfterThinking?: boolean;
}
```

### `Generator.generate` signature change

Current:
```ts
static async *generate(
  promptTokenIds, sampler, session, eosTokenId,
  forwardPass, config: GenerationConfig,
  signal?: AbortSignal,
  forwardDecode?,
)
```

New:
```ts
static async *generate(
  promptTokenIds, sampler, session, eosTokenId,
  forwardPass, config: InternalGenerationOptions,
  forwardDecode?,
)
```

`signal` is read from `config.signal` inside the loop instead of via a
separate parameter.

### Public exports

`src/index.ts` exports only `GenerationConfig` (public). The
`InternalGenerationOptions` type stays unexported.

### Engine adaptations

- `engine.chat(modelId, prompt, config?: Partial<GenerationConfig>)` —
  consumer surface unchanged minus the dropped `prompt` field.
- `engine.chatCompletion` builds an `InternalGenerationOptions` by
  merging consumer's `GenerationConfig` with auto-applied steering
  (Qwen3 thinking-token IDs, masking, etc.).

### Risk register (a)

- `src/inference/speculative.ts:2,230` imports `GenerationConfig`
  and types `config: GenerationConfig`. After rename, switch to
  `InternalGenerationOptions`.
- Any test fixture constructing `GenerationConfig` with steering
  fields is retyped to `InternalGenerationOptions`.
- The `signal` migration adjusts one external-call signature; the
  `engine.chat`/`engine.chatCompletion` paths and the `speculative.ts`
  call site move signal into the config object.

---

## Item (b) — `WebLLMConfig.device` removal

### Decision summary

- **Remove** `device: GPUDevice` from `WebLLMConfig`.
- **Caller passes device** directly to
  `engine.loadLightweightModel(config: LightweightModelConfig)`.
- The wasm path is unaffected (acquires its own device internally).

### New types

```ts
// src/core/types.ts
export interface WebLLMConfig {
  cacheDir?: string;
  memoryBudget: number;
  frameBudgetMs?: number;
}
```

`LightweightModelConfig` (in `src/inference/lightweight.ts`) is unchanged
— `device: GPUDevice` is already a required field.

### Engine signature change

```ts
// before
async loadLightweightModel(
  config: Omit<LightweightModelConfig, "device">
): Promise<LightweightModel> { ... }

// after
async loadLightweightModel(
  config: LightweightModelConfig
): Promise<LightweightModel> { ... }
```

The `device: this._config.device, ...config` injection is dropped.

### Risk register (b)

- BC-break: any consumer doing `WebLLM.init({ device, memoryBudget, ... })`
  must drop `device` and instead pass it to `loadLightweightModel`.
  Mechanical migration. Smoke harnesses + tests get a grep sweep.
- The wasm path was never affected.

---

## Item (c) — `sampling` flag + Qwen profile export

### Decision summary

- **Move** `QWEN_THINKING_DEFAULTS` / `QWEN_NON_THINKING_DEFAULTS` from
  private constants in `engine.ts` to a new exported file
  `src/core/sampling-profiles.ts`.
- **Add** `sampling?: "auto" | "qwen-thinking" | "qwen-default" | "raw"`
  to `CompletionConfig` (default `"auto"`).
- **Default `"auto"`** preserves current magic behavior. Zero BC-break.

### New module

```ts
// src/core/sampling-profiles.ts
export const QWEN_THINKING_DEFAULTS = {
  temperature: 0.6,
  topK: 20,
  topP: 0.95,
  repetitionPenalty: 1.05,
} as const;

export const QWEN_NON_THINKING_DEFAULTS = {
  temperature: 0.7,
  topK: 20,
  topP: 0.8,
  repetitionPenalty: 1.1,
} as const;
```

### `CompletionConfig` addition

```ts
// src/core/chat-types.ts
export interface CompletionConfig {
  // ... existing fields ...

  /**
   * How to apply sampling-parameter defaults.
   * - `"auto"` (default): apply Qwen profiles when architecture starts
   *   with `"qwen"` and chat template is ChatML; otherwise use
   *   consumer-provided values.
   * - `"qwen-thinking"`: force `QWEN_THINKING_DEFAULTS` regardless of
   *   architecture.
   * - `"qwen-default"`: force `QWEN_NON_THINKING_DEFAULTS` regardless
   *   of architecture.
   * - `"raw"`: skip auto-application; use only consumer-provided
   *   values (with engine fallbacks for unspecified fields).
   *
   * Consumer-provided field values always override profile defaults.
   * Example: `sampling: "qwen-thinking", temperature: 0.9` applies
   *   the qwen profile then overrides temperature with 0.9.
   */
  sampling?: "auto" | "qwen-thinking" | "qwen-default" | "raw";
}
```

### Engine adaptation

`engine.chatCompletion` reads `config.sampling ?? "auto"` and dispatches:
- `"auto"`: existing magic (unchanged behavior).
- `"qwen-thinking"` / `"qwen-default"`: force the named profile.
- `"raw"`: skip profile application; pass consumer values straight to
  `Sampler` / `InternalGenerationOptions`.

### Public exports

`src/index.ts` exports `QWEN_THINKING_DEFAULTS`, `QWEN_NON_THINKING_DEFAULTS`
from the new `sampling-profiles.ts`.

### Risk register (c)

- Existing call sites that exercise the magic (Qwen3 smoke tests, eval
  harness) keep working under default `"auto"`.
- New test: `sampling: "raw"` on a Qwen3 architecture asserts the
  profile is NOT applied (e.g. consumer's `temperature: 1.0` survives).

---

## Item (d) — Engine accessor migration

### Decision summary

Migrate three `getX()` methods to `get x()` properties. No deprecation
aliases (project policy: "no BC-compat shims").

### Change

```ts
// before (engine.ts)
getMemoryPool(): MemoryPool { return this.memoryPool; }
getScheduler(): Scheduler { return this.scheduler; }
getModelManager(): ModelManager { return this.modelManager; }

// after
get memoryPool(): MemoryPool { return this._memoryPool; }
get scheduler(): Scheduler { return this._scheduler; }
get modelManager(): ModelManager { return this._modelManager; }
```

### Field rename

Underscore-prefix the private backing fields to avoid getter/field
collision:

- `memoryPool` → `_memoryPool`
- `scheduler` → `_scheduler`
- `modelManager` → `_modelManager`

Internal references (`this.memoryPool`, `this.scheduler`,
`this.modelManager`) sweep across `engine.ts` (~10 references).

### Risk register (d)

- BC-break: external callers using `engine.getMemoryPool()` etc. break.
  No known consumers; sweep tests + eval harness pre-edit.

---

## Item (e) — `Character.setTools`

### Decision summary

Add **one** method: `setTools(tools)`. No `attachToolSystem` (YAGNI;
no consumer ask for parser swap).

### Method

```ts
// src/characters/character.ts
setTools(tools: ToolDefinition[]): void {
  this.config.tools = tools;
  this.toolSystem = tools.length > 0 ? new ToolSystem(tools) : null;
}
```

### Behavior

- Replaces the tool list and recreates the internal `ToolSystem`.
- Empty array clears tools (mirrors constructor logic).
- Message history preserved.
- No effect on in-flight `chat()`; change applies to next call.

### Test plan

`tests/character-set-tools.test.ts`:
1. Replace tool list — `parseToolCall` finds new tools, not old.
2. Clear tools (empty array) — `parseToolCall` returns null.
3. Parsing works after swap — round-trip a tool call through new tools.

---

## Item (f) — Polish bundle

### (f.1) `ChatToolSchema` literal union

```ts
// src/core/chat-types.ts
export interface ChatToolSchema {
  name: string;
  description: string;
  parameters: Record<
    string,
    {
      type: "string" | "number" | "integer" | "boolean" | "array" | "object";
      description?: string;
      required?: boolean;
    }
  >;
}
```

Pre-edit: grep all `parameters[...].type` literals across tests/fixtures.
Any string outside the union is a pre-existing bug — fix the fixture,
do not widen the union.

### (f.2) `GenerationFinishReason` per-variant JSDoc

```ts
// src/inference/generation.ts
export type GenerationFinishReason =
  /** Generation cancelled via AbortSignal. */
  | "aborted"
  /** EOS token sampled. */
  | "eos"
  /** maxTokens budget exhausted. */
  | "max-tokens"
  /** A custom `stopTokens` entry was sampled. */
  | "stop-token";
```

### (f.3) README API table additions

Add rows for:
- `engine.removeCharacter(id)` — remove a registered character.
- `engine.shutdown()` — release GPU buffers and dispose engines.

### (f.4) README feature-claim trim

Tighten "Memory management" / "Scheduling" sections to reflect
**read-only consumer surface**:

- `engine.memoryPool` (getter, returns `MemoryPool`).
- `engine.scheduler` (getter, returns `Scheduler`).

Remove any wording that implies allocate/schedule operations are
consumer-callable. They are not.

---

## Testing strategy

| Phase | Test additions / changes |
|---|---|
| 1 (a, b, c) | Compile-fail assertions for steering fields on public `GenerationConfig` (`// @ts-expect-error`). New test for `sampling: "raw"` skipping qwen magic. Existing `chat-completion.test.ts` covers signal-on-config end-to-end. |
| 2 (d, e) | `tests/character-set-tools.test.ts` (3 cases). Sweep existing tests calling `getMemoryPool()` / `getScheduler()` / `getModelManager()` → property syntax. |
| 3 (f) | Existing `ChatToolSchema` typecheck tests should pass once any out-of-union fixture is fixed. |

**Ship gate per phase:** `make checkall` green before commit.

---

## Implementation commit cadence

Per project doctrine ("always commit before work"), each phase's
implementation lands as its own commit with `feat(api):` or
`refactor(api):` prefix. The plan + spec land in their own
`docs(spec):` / `docs(plan):` commits beforehand.

Tentative commit map:
- `docs(spec): TS API audit follow-ups (a-f)` — this file.
- `docs(plan): TS API audit follow-ups implementation plan` — from writing-plans.
- `feat(api): split GenerationConfig (item a)` — Phase 1a.
- `feat(api): drop WebLLMConfig.device (item b)` — Phase 1b.
- `feat(api): sampling flag + Qwen profile export (item c)` — Phase 1c.
- `refactor(api): engine accessor properties (item d)` — Phase 2a.
- `feat(api): Character.setTools (item e)` — Phase 2b.
- `docs(api): polish bundle (item f)` — Phase 3.

(Phase boundaries final form is decided by writing-plans.)

---

## Spec self-review

- **Placeholder scan:** No TBDs; all sub-decisions resolved during
  brainstorming.
- **Internal consistency:** Sections (a)-(f) reference shared types
  (`GenerationConfig`, `CompletionConfig`) without contradicting each
  other. Phase 1 lands type changes that Phase 2/3 depend on; ordering
  is correct.
- **Scope check:** Six items, three phases, ≤5 files per phase. Single
  spec is appropriate; doesn't need decomposition.
- **Ambiguity check:** `sampling: "raw"` semantics explicitly state
  "with engine fallbacks for unspecified fields" — clarifies how
  unspecified `temperature`/etc. are handled.
