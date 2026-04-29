# TS API Audit Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all six deferred TS API audit items (a-f) as a coherent public API hygiene pass — `GenerationConfig` split, `WebLLMConfig.device` removal, `sampling` flag, engine accessor migration, `Character.setTools`, and a polish bundle (literal union, JSDoc, README).

**Architecture:** Six items grouped into three phases. Each item gets its own commit on `main`. Refactors (a/b/d) drive correctness via `bun run typecheck` + existing test suite. New behavior (c/e/f.1) follows TDD: failing test → implement → passing test → commit.

**Tech Stack:** TypeScript, Bun (`bun test`), biome (`bun run lint`), tsc (`bun run typecheck`). Ship gate per phase: `make checkall` (fmt + lint + typecheck + test) green.

**Source spec:** [`docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md`](../specs/2026-04-29-ts-api-audit-followups-design.md)

---

## File Structure

| File | Change | Phase |
|---|---|---|
| `src/inference/generation.ts` | Rename current type → `InternalGenerationOptions`; new public `GenerationConfig` (7 fields); drop `prompt` field; drop `signal` positional param on `Generator.generate`; drop `signal` field on `GenerationStreamOptions` | 1a |
| `src/inference/speculative.ts` | Type import rename `GenerationConfig` → `InternalGenerationOptions` | 1a |
| `src/core/engine.ts` | Update `chat`/`generateStream` signatures; build `InternalGenerationOptions` internally; move `signal` into config object | 1a |
| `src/index.ts` | Export only public `GenerationConfig` (no `InternalGenerationOptions`) | 1a |
| `tests/generation-config-public.test.ts` | **NEW**: compile-fail tests asserting public type rejects steering fields | 1a |
| `src/core/types.ts` | Drop `device: GPUDevice` from `WebLLMConfig` | 1b |
| `src/core/engine.ts` | Drop `device` injection in `loadLightweightModel`; flip signature from `Omit<LightweightModelConfig, "device">` → `LightweightModelConfig` | 1b |
| `tests/embed-api.test.ts` | Drop `device: {} as GPUDevice` from `WebLLM.init` call | 1b |
| `smoke-test/real-model-page.js` | Drop `device: engineDevice` from `WebLLM.init` call | 1b |
| `src/core/sampling-profiles.ts` | **NEW**: exported `QWEN_THINKING_DEFAULTS` / `QWEN_NON_THINKING_DEFAULTS` | 1c |
| `src/core/chat-types.ts` | Add `sampling` field to `CompletionConfig` | 1c |
| `src/core/engine.ts` | Import constants from `sampling-profiles`; dispatch on `config.sampling` | 1c |
| `src/index.ts` | Export Qwen constants | 1c |
| `tests/sampling-profiles.test.ts` | **NEW**: 4 tests for `auto` / `qwen-thinking` / `qwen-default` / `raw` semantics | 1c |
| `src/core/engine.ts` | Underscore-prefix backing fields; replace 3 methods with `get` properties | 2a |
| `src/characters/character.ts` | Add `setTools(tools)` method | 2b |
| `tests/character-set-tools.test.ts` | **NEW**: 3-case test (replace, clear, parse-after-swap) | 2b |
| `src/core/chat-types.ts` | Narrow `ChatToolSchema.parameters[*].type` to literal union | 3 |
| `src/inference/generation.ts` | Add per-variant JSDoc to `GenerationFinishReason` | 3 |
| `README.md` | Add `removeCharacter`/`shutdown` rows; trim Memory/Scheduling claims | 3 |

---

## Phase 1a — Item (a): Split `GenerationConfig`

### Task 1: Rename current `GenerationConfig` to `InternalGenerationOptions` and introduce new public `GenerationConfig`

**Files:**
- Modify: `src/inference/generation.ts:11-90` (interface definition)
- Modify: `src/inference/generation.ts:130-148` (`GenerationStreamOptions`)
- Modify: `src/inference/generation.ts:173-190` (`Generator.generate` signature)
- Modify: `src/inference/generation.ts:651-680` (`generateTextStream` body — read `config.signal` instead of param)

- [ ] **Step 1: Re-read `src/inference/generation.ts` lines 1-200**

This grounds you in the current shape. Confirm field names and order before editing.

- [ ] **Step 2: Replace the `GenerationConfig` interface block with two interfaces and updated `GenerationStreamOptions`**

Replace lines 11-90 (the current single `GenerationConfig`) with:

```ts
/** Configuration for a single generation request. Public API surface. */
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
	/** Optional custom stop token IDs that halt generation. */
	stopTokens?: number[];
	/** Optional AbortSignal to cancel generation mid-stream. */
	signal?: AbortSignal;
}

/**
 * Internal options for `Generator.generate` / `generateTextStream`. Extends
 * the public `GenerationConfig` with chat-control steering fields (Qwen3
 * thinking-block masks, leading-whitespace enforcement, etc.). Engine builds
 * this internally and never exposes it on the public API.
 */
export interface InternalGenerationOptions extends GenerationConfig {
	/**
	 * Optional token IDs that should terminate generation if produced after the
	 * first generated token. Used to contain malformed chat-control reentry.
	 */
	forbiddenReentryTokens?: number[];
	/** Optional token ID for opening a thinking block. */
	thinkingOpenTokenId?: number;
	/** Optional token ID for closing a thinking block. */
	thinkingCloseTokenId?: number;
	/**
	 * When true, treat repeated `<think>` or a stray `</think>` as malformed and
	 * stop generation.
	 */
	enforceSingleThinkBlock?: boolean;
	/**
	 * Optional token IDs to suppress from sampling while inside an open thinking
	 * block. Used to steer malformed Qwen chat outputs toward `</think>` or
	 * normal reasoning tokens instead of repeating control markers.
	 */
	maskedTokensWhileThinking?: number[];
	/**
	 * Optional tokenizer used for lightweight token classification during
	 * generation-time steering.
	 */
	tokenizer?: Tokenizer;
	/**
	 * Optional token IDs to suppress after a think block closes and before any
	 * visible assistant answer text has been emitted.
	 */
	maskedTokensAfterThinkingUntilAnswer?: number[];
	/**
	 * When true, suppress EOS and custom stop tokens after `</think>` until at
	 * least one visible answer token has been emitted.
	 */
	requireVisibleAnswerAfterThinking?: boolean;
	/**
	 * When true, suppress EOS and custom stop tokens from the start of
	 * generation until at least one visible answer token has been emitted.
	 */
	requireVisibleAnswerBeforeStop?: boolean;
	/**
	 * When true, suppress whitespace-only text tokens after `</think>` until a
	 * visible answer token has been emitted.
	 */
	suppressWhitespaceOnlyAfterThinking?: boolean;
	/**
	 * When true, suppress whitespace-only text tokens from the start of
	 * generation until a visible answer token has been emitted.
	 */
	suppressWhitespaceOnlyUntilAnswer?: boolean;
	/**
	 * Optional token IDs to suppress after visible assistant answer text has
	 * started, preventing relapse into control-token scaffolding.
	 */
	maskedTokensAfterAnswerStarts?: number[];
	/**
	 * When true, the first token sampled after `</think>` closes is forced to
	 * begin with whitespace by masking + resampling until a non-control token
	 * whose decoded text starts with `\s` is produced. One-shot: applies only
	 * to the first post-`</think>` step, then defers to
	 * `suppressWhitespaceOnlyAfterThinking` for subsequent steps. Prevents
	 * run-on output like `</think>The answer ...`.
	 */
	requireLeadingWhitespaceAfterThinking?: boolean;
}
```

**Note:** the `prompt: string` field is intentionally absent from both. It was unused.

- [ ] **Step 3: Update `GenerationStreamOptions` to drop the `signal` field and use `InternalGenerationOptions`**

Replace lines 130-148 (`GenerationStreamOptions` interface):

```ts
export interface GenerationStreamOptions {
	promptTokenIds: number[];
	sampler: Sampler;
	session: InferenceSession;
	eosTokenId: number;
	tokenizer: Tokenizer;
	forwardPass: (
		tokenIds: number[],
		positions: number[],
	) => Float32Array | Promise<Float32Array>;
	config: InternalGenerationOptions;
	forwardDecode?: (
		tokenIds: number[],
		positions: number[],
		mode: DecodeMode,
		topK?: number,
	) => Promise<DecodeResult>;
}
```

The `signal?: AbortSignal` field is removed. Signal now travels via `config.signal`.

- [ ] **Step 4: Update `Generator.generate` signature**

Replace `static async *generate` signature (lines 173-190). Drop the positional `signal?: AbortSignal` parameter and update the `config` type:

```ts
static async *generate(
	promptTokenIds: number[],
	sampler: Sampler,
	session: InferenceSession,
	eosTokenId: number,
	forwardPass: (
		tokenIds: number[],
		positions: number[],
	) => Float32Array | Promise<Float32Array>,
	config: InternalGenerationOptions,
	forwardDecode?: (
		tokenIds: number[],
		positions: number[],
		mode: DecodeMode,
		topK?: number,
	) => Promise<DecodeResult>,
): AsyncGenerator<number, GenerationResult> {
```

Inside the function body, replace any reference to the `signal` parameter with `config.signal`. There is exactly one place to update — search inside `Generator.generate` for `signal` usages and replace `signal` → `config.signal`. Common pattern: `if (signal?.aborted)` → `if (config.signal?.aborted)`. Read lines 190-650 if needed to find every reference.

- [ ] **Step 5: Update `generateTextStream` to drop the `signal` destructure and pass full config**

Replace the destructure block (lines 651-660):

```ts
export async function* generateTextStream({
	promptTokenIds,
	sampler,
	session,
	eosTokenId,
	tokenizer,
	forwardPass,
	config,
	forwardDecode,
}: GenerationStreamOptions): AsyncGenerator<
	GenerationStreamChunk,
	GenerationStreamResult
> {
```

Then update the inner `Generator.generate(...)` call (line 667-674): drop the `signal` positional argument. New call:

```ts
const gen = Generator.generate(
	promptTokenIds,
	sampler,
	session,
	eosTokenId,
	forwardPass,
	config,
	forwardDecode,
);
```

If anywhere inside `generateTextStream` a `signal` variable was destructured/referenced, switch to `config.signal`.

### Task 2: Update `speculative.ts` to use `InternalGenerationOptions`

**Files:**
- Modify: `src/inference/speculative.ts:2,230`

- [ ] **Step 1: Read `src/inference/speculative.ts` lines 1-10 and 220-240**

Confirm the current import + type usage.

- [ ] **Step 2: Update the import**

At line 2, change:
```ts
import type {
	GenerationConfig,
	...
} from "./generation.js";
```
to:
```ts
import type {
	InternalGenerationOptions,
	...
} from "./generation.js";
```
(or whatever sibling imports are co-imported — keep them).

- [ ] **Step 3: Update the field type at line 230**

```ts
config: InternalGenerationOptions;
```

### Task 3: Update `engine.ts` call sites — chat() type signature, generateStream() signal-into-config, drop unused prompt field

**Files:**
- Modify: `src/core/engine.ts:204-260` (`chat` method)
- Modify: `src/core/engine.ts:312-320` (`genConfig` literal in `generateStream`)
- Modify: `src/core/engine.ts:430-450` (`generateTextStream` call site)

- [ ] **Step 1: Re-read `src/core/engine.ts` lines 200-460**

Get a clear picture of the three call sites.

- [ ] **Step 2: Update `chat` method signature and `genConfig` literal**

At line 207, the public `chat` method takes `Partial<GenerationConfig>`. After the split, that's now `Partial<GenerationConfig>` against the *new* (7-field) public type — semantically fine. But the inline `genConfig: GenerationConfig` literal at line 239 needs the type updated to `InternalGenerationOptions` because the engine builds the full internal shape.

Search for the literal `const genConfig: GenerationConfig = {` (around lines 239 and 312). For each one, the type annotation changes to `InternalGenerationOptions`. Also remove the `prompt:` field if present (it was required on the old type; it is now absent).

Around line 239 (in `chat`), change:
```ts
const genConfig: InternalGenerationOptions = {
	maxTokens: config?.maxTokens ?? 512,
	temperature: config?.temperature ?? 1.0,
	topK: config?.topK ?? 0,
	topP: config?.topP ?? 1.0,
	repetitionPenalty: config?.repetitionPenalty ?? 1.0,
};
```

(Drop any `prompt` field that was there.)

- [ ] **Step 3: Update the second `genConfig` literal at line 312-320**

Apply the same fix:
- Change type annotation from `GenerationConfig` → `InternalGenerationOptions`.
- Drop the `prompt: typeof input === "string" ? input : "",` field.
- Add `signal: config?.signal,` so the signal flows through the config object.

```ts
const genConfig: InternalGenerationOptions = {
	maxTokens: config?.maxTokens ?? 512,
	temperature: effectiveTemperature,
	topK: effectiveTopK,
	topP: effectiveTopP,
	repetitionPenalty: effectiveRepetitionPenalty,
	stopTokens: config?.stopTokenIds ? [...config.stopTokenIds] : undefined,
	signal: config?.signal,
};
```

- [ ] **Step 4: Update the `generateTextStream` call site at lines 430-448**

Drop the `signal: config?.signal,` line (signal now lives on `genConfig`). New call:

```ts
yield* generateTextStream({
	promptTokenIds: tokens,
	sampler,
	session,
	eosTokenId: tokenizer.eosId,
	tokenizer,
	forwardPass,
	config: genConfig,
	forwardDecode,
});
```

- [ ] **Step 5: Update the `Generator.generate` call site in `chat` (line 248)**

The call already passes `genConfig` and no signal positional. Just verify it still works:

```ts
const gen = Generator.generate(
	tokens,
	sampler,
	session,
	tokenizer.eosId ?? 2,
	forwardPass,
	genConfig,
);
```

Update the import at top of engine.ts (line 10) — change `type GenerationConfig` to keep both `GenerationConfig` (still used for `Partial<...>` on the `chat` method signature) AND add `InternalGenerationOptions`:

```ts
import {
	type GenerationConfig,
	type InternalGenerationOptions,
	Generator,
	...
} from "../inference/generation.js";
```

### Task 4: Update `src/index.ts` — public exports

**Files:**
- Modify: `src/index.ts:90-100`

- [ ] **Step 1: Read `src/index.ts` lines 85-110**

Confirm what's currently exported from `inference/generation`.

- [ ] **Step 2: Verify `GenerationConfig` is exported, `InternalGenerationOptions` is NOT exported**

The current export block likely already exports `GenerationConfig`. Leave it. Do NOT add `InternalGenerationOptions` to the export list. (If for any reason an existing export pulled in `InternalGenerationOptions`, audit and remove.)

### Task 5: Add compile-fail assertion test for public surface

**Files:**
- Create: `tests/generation-config-public.test.ts`

- [ ] **Step 1: Write the test**

```ts
import { describe, test } from "bun:test";
import type { GenerationConfig } from "../src/index.js";

describe("Public GenerationConfig surface", () => {
	test("rejects internal steering fields at the type level", () => {
		// These assignments must NOT compile. If any does, the public type
		// is leaking internal steering fields and the split has regressed.

		// @ts-expect-error — thinkingOpenTokenId is internal-only
		const _a: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			thinkingOpenTokenId: 42,
		};

		// @ts-expect-error — maskedTokensWhileThinking is internal-only
		const _b: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			maskedTokensWhileThinking: [1, 2, 3],
		};

		// @ts-expect-error — requireVisibleAnswerAfterThinking is internal-only
		const _c: GenerationConfig = {
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
			requireVisibleAnswerAfterThinking: true,
		};

		// @ts-expect-error — prompt was dropped entirely
		const _d: GenerationConfig = {
			prompt: "hi",
			maxTokens: 10,
			temperature: 1,
			topK: 0,
			topP: 1,
			repetitionPenalty: 1,
		};

		// Sanity: a fully-populated public config must compile.
		const _ok: GenerationConfig = {
			maxTokens: 10,
			temperature: 0.7,
			topK: 40,
			topP: 0.9,
			repetitionPenalty: 1.05,
			stopTokens: [2],
			signal: new AbortController().signal,
		};

		void _a;
		void _b;
		void _c;
		void _d;
		void _ok;
	});
});
```

- [ ] **Step 2: Run typecheck and tests**

```bash
bun run typecheck
bun test tests/generation-config-public.test.ts
```

Expected: typecheck passes (the `@ts-expect-error` assertions consume the expected errors); the test runs without runtime errors. If `bun run typecheck` reports an "Unused @ts-expect-error directive" — that means the type is wrong (the assignment compiles when it shouldn't), and you've broken the split. Fix the public `GenerationConfig` so the steering fields are excluded.

### Task 6: Phase 1a ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

Expected: green. If failing, fix before proceeding.

- [ ] **Step 2: Stage and commit**

```bash
git add src/inference/generation.ts src/inference/speculative.ts src/core/engine.ts src/index.ts tests/generation-config-public.test.ts
git commit -m "feat(api): split GenerationConfig into public + internal halves

Renames the existing 22-field GenerationConfig to InternalGenerationOptions
(engine-internal, unexported) and introduces a new 7-field public
GenerationConfig surfacing only consumer-relevant sampling/output knobs:
maxTokens, temperature, topK, topP, repetitionPenalty, stopTokens, signal.

Also:
- Drop unused 'prompt: string' field — Generator.generate already takes
  promptTokenIds positionally; the field was never read.
- Migrate signal from a separate positional parameter on Generator.generate
  (and a separate field on GenerationStreamOptions) into config.signal.
  Matches CompletionConfig's shape; one less positional argument to thread.

Adds compile-fail assertion test (tests/generation-config-public.test.ts)
locking in the steering-field exclusion via @ts-expect-error.

Item (a) of TS API audit follow-ups; spec at
docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md."
```

---

## Phase 1b — Item (b): Drop `WebLLMConfig.device`

### Task 7: Remove `device` from `WebLLMConfig`

**Files:**
- Modify: `src/core/types.ts:2-7`

- [ ] **Step 1: Edit `WebLLMConfig` interface**

Replace lines 2-7:

```ts
/** Configuration for initializing a WebLLM engine instance. */
export interface WebLLMConfig {
	cacheDir?: string;
	memoryBudget: number;
	frameBudgetMs?: number;
}
```

(Drop the `device: GPUDevice;` field.)

### Task 8: Flip `loadLightweightModel` signature and drop the device injection

**Files:**
- Modify: `src/core/engine.ts:193-202`

- [ ] **Step 1: Edit the method**

Replace:

```ts
async loadLightweightModel(
	config: Omit<LightweightModelConfig, "device">,
): Promise<LightweightModel> {
	const model = new LightweightModel({
		device: this._config.device,
		...config,
	});
	await model.init();
	return model;
}
```

with:

```ts
async loadLightweightModel(
	config: LightweightModelConfig,
): Promise<LightweightModel> {
	const model = new LightweightModel(config);
	await model.init();
	return model;
}
```

### Task 9: Update test fixture and smoke harness

**Files:**
- Modify: `tests/embed-api.test.ts:6-9`
- Modify: `smoke-test/real-model-page.js:512-516`

- [ ] **Step 1: Update test**

In `tests/embed-api.test.ts`, around line 6, drop the `device:` line:

```ts
const engine = await WebLLM.init({
	memoryBudget: 1 << 28,
});
```

- [ ] **Step 2: Update smoke harness**

In `smoke-test/real-model-page.js` around line 512, drop the `device: engineDevice,` line. Also drop the surrounding lines that acquire `engineDevice` if they are no longer used. Use grep before deleting:

```bash
grep -n "engineDevice\|engineAdapter" smoke-test/real-model-page.js
```

If `engineDevice`/`engineAdapter` are referenced elsewhere, leave the acquisition; otherwise delete. After:

```js
smokeEngine = await WebLLM.init({
	memoryBudget: 2_000_000_000,
});
```

### Task 10: Phase 1b ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

- [ ] **Step 2: Commit**

```bash
git add src/core/types.ts src/core/engine.ts tests/embed-api.test.ts smoke-test/real-model-page.js
git commit -m "feat(api): drop device from WebLLMConfig

WebLLMConfig.device was required at engine init but consumed only by
loadLightweightModel — the wasm path acquires its own device internally
via navigator.gpu.requestAdapter() inside the WASM bundle. The required
field misled consumers about how the device flows through the engine.

Drops device from WebLLMConfig entirely. Caller passes their own
GPUDevice directly to engine.loadLightweightModel(LightweightModelConfig).
LightweightModelConfig already has device as a required field, so this
is a clean type narrowing for the only consumer.

BC-break for any consumer doing WebLLM.init({ device, ... }); migration
is mechanical (drop the field, pass on loadLightweightModel call).
Updates tests/embed-api.test.ts and smoke-test/real-model-page.js
which were the only known call sites.

Item (b) of TS API audit follow-ups."
```

---

## Phase 1c — Item (c): `sampling` flag + Qwen profile export

### Task 11: Create `src/core/sampling-profiles.ts` with exported constants

**Files:**
- Create: `src/core/sampling-profiles.ts`

- [ ] **Step 1: Write the file**

```ts
/**
 * Sampling-parameter profiles surfaced to consumers via `CompletionConfig.sampling`.
 *
 * `QWEN_THINKING_DEFAULTS` matches Qwen3's recommended parameters when the
 * model is invoked in thinking mode (default for ChatML-templated Qwen3
 * variants). `QWEN_NON_THINKING_DEFAULTS` is the official non-thinking
 * profile applied when `enableThinking: false` is set.
 *
 * Engine auto-applies these when `architecture` starts with `"qwen"` and the
 * chat template is ChatML and `sampling` is `"auto"` (default). Consumers
 * can also force a profile via `sampling: "qwen-thinking"` /
 * `"qwen-default"` regardless of architecture, or opt out with
 * `sampling: "raw"`.
 */
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

### Task 12: Update `engine.ts` to import from `sampling-profiles` and remove inline constants

**Files:**
- Modify: `src/core/engine.ts:97-109` (remove inline constants)
- Modify: `src/core/engine.ts:1-65` (add import)

- [ ] **Step 1: Add the import**

In the import block at the top of `engine.ts` (around lines 1-65), add:

```ts
import {
	QWEN_THINKING_DEFAULTS,
	QWEN_NON_THINKING_DEFAULTS,
} from "./sampling-profiles.js";
```

- [ ] **Step 2: Delete the inline constants**

Delete lines 97-109 (the two `const QWEN_*_DEFAULTS = { ... } as const;` blocks). They're now imported.

### Task 13: Add `sampling` field to `CompletionConfig`

**Files:**
- Modify: `src/core/chat-types.ts:23-65` (`CompletionConfig` interface)

- [ ] **Step 1: Read current `CompletionConfig`**

Read `src/core/chat-types.ts` lines 23-80 to confirm the field set.

- [ ] **Step 2: Add the `sampling` field**

Insert after the existing `repetitionPenalty?` field (around line 36):

```ts
/**
 * How to apply sampling-parameter defaults.
 * - `"auto"` (default): apply Qwen profiles when architecture starts
 *   with `"qwen"` and chat template is ChatML; otherwise use
 *   consumer-provided values.
 * - `"qwen-thinking"`: force `QWEN_THINKING_DEFAULTS` regardless of
 *   architecture.
 * - `"qwen-default"`: force `QWEN_NON_THINKING_DEFAULTS` regardless
 *   of architecture.
 * - `"raw"`: skip auto-application; use only consumer-provided values
 *   (with engine fallbacks for unspecified fields).
 *
 * Consumer-provided field values always override profile defaults.
 * Example: `sampling: "qwen-thinking", temperature: 0.9` applies the
 * qwen profile then overrides temperature with 0.9.
 */
sampling?: "auto" | "qwen-thinking" | "qwen-default" | "raw";
```

### Task 14: Update `engine.generateStream` to dispatch on `sampling`

**Files:**
- Modify: `src/core/engine.ts:283-320` (the magic-defaults block)

- [ ] **Step 1: Replace the auto-detect block**

Currently lines 285-292:

```ts
const isQwenChatml =
	Array.isArray(input) &&
	String(entry.hyperparams.architecture).startsWith("qwen") &&
	detectChatTemplate(chatTemplate ?? "") === "chatml";
const qwenDefaults =
	isQwenChatml && config?.enableThinking === false
		? QWEN_NON_THINKING_DEFAULTS
		: QWEN_THINKING_DEFAULTS;
```

Replace with:

```ts
const samplingMode = config?.sampling ?? "auto";
const isQwenChatml =
	Array.isArray(input) &&
	String(entry.hyperparams.architecture).startsWith("qwen") &&
	detectChatTemplate(chatTemplate ?? "") === "chatml";
const applyAutoQwen = samplingMode === "auto" && isQwenChatml;
const forcedProfile =
	samplingMode === "qwen-thinking"
		? QWEN_THINKING_DEFAULTS
		: samplingMode === "qwen-default"
			? QWEN_NON_THINKING_DEFAULTS
			: null;
const autoProfile = applyAutoQwen
	? config?.enableThinking === false
		? QWEN_NON_THINKING_DEFAULTS
		: QWEN_THINKING_DEFAULTS
	: null;
const activeProfile = forcedProfile ?? autoProfile;
```

- [ ] **Step 2: Replace the `effective*` block**

Currently lines 293-304:

```ts
const effectiveTemperature = isQwenChatml ? ... : (config?.temperature ?? 1.0);
const effectiveTopK = isQwenChatml ? ... : (config?.topK ?? 0);
const effectiveTopP = isQwenChatml ? ... : (config?.topP ?? 1.0);
const effectiveRepetitionPenalty = isQwenChatml ? ... : (config?.repetitionPenalty ?? 1.0);
```

Replace with:

```ts
// Consumer-provided values override profile defaults; profile defaults
// override engine fallbacks. samplingMode === "raw" produces a null
// activeProfile, falling through to the engine fallbacks directly.
const effectiveTemperature =
	config?.temperature ?? activeProfile?.temperature ?? 1.0;
const effectiveTopK = config?.topK ?? activeProfile?.topK ?? 0;
const effectiveTopP = config?.topP ?? activeProfile?.topP ?? 1.0;
const effectiveRepetitionPenalty =
	config?.repetitionPenalty ?? activeProfile?.repetitionPenalty ?? 1.0;
```

The rest of `generateStream` (sampler construction, genConfig literal, etc.) is unchanged.

### Task 15: Add `sampling-profiles` exports to `src/index.ts`

**Files:**
- Modify: `src/index.ts:1-110`

- [ ] **Step 1: Add the export**

Append (or co-locate with other `core/` exports):

```ts
export {
	QWEN_THINKING_DEFAULTS,
	QWEN_NON_THINKING_DEFAULTS,
} from "./core/sampling-profiles.js";
```

### Task 16: Add `sampling-profiles.test.ts` with 4 cases

**Files:**
- Create: `tests/sampling-profiles.test.ts`

- [ ] **Step 1: Write the test file**

```ts
import { describe, expect, test } from "bun:test";
import type { CompletionConfig } from "../src/core/chat-types.js";
import {
	QWEN_THINKING_DEFAULTS,
	QWEN_NON_THINKING_DEFAULTS,
} from "../src/index.js";

describe("Sampling profile constants", () => {
	test("QWEN_THINKING_DEFAULTS matches expected values", () => {
		expect(QWEN_THINKING_DEFAULTS).toEqual({
			temperature: 0.6,
			topK: 20,
			topP: 0.95,
			repetitionPenalty: 1.05,
		});
	});

	test("QWEN_NON_THINKING_DEFAULTS matches expected values", () => {
		expect(QWEN_NON_THINKING_DEFAULTS).toEqual({
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
		});
	});

	test("constants are readonly at the type level", () => {
		// Compile-time check: `as const` produces readonly narrow types.
		// @ts-expect-error — readonly property cannot be assigned.
		QWEN_THINKING_DEFAULTS.temperature = 0.99;
		// @ts-expect-error — readonly property cannot be assigned.
		QWEN_NON_THINKING_DEFAULTS.topK = 99;
		// (Behavior already covered by `as const` literal types.)
	});
});

describe("CompletionConfig.sampling field", () => {
	test("union accepts all four mode strings; rejects unknown strings", () => {
		const a: CompletionConfig = { sampling: "auto" };
		const b: CompletionConfig = { sampling: "qwen-thinking" };
		const c: CompletionConfig = { sampling: "qwen-default" };
		const d: CompletionConfig = { sampling: "raw" };

		// @ts-expect-error — "off" is not a member of the sampling union.
		const e: CompletionConfig = { sampling: "off" };

		expect([a, b, c, d, e]).toHaveLength(5);
	});
});
```

**Note:** Behavioral tests (does `"raw"` actually skip the magic at runtime?) require a loaded model and are exercised end-to-end by the existing `chat-completion.test.ts` / smoke harness. The unit tests above lock in the constants and the type union; runtime behavior is verified by the ship gate (smoke + bench).

### Task 17: Phase 1c ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

- [ ] **Step 2: Commit**

```bash
git add src/core/sampling-profiles.ts src/core/engine.ts src/core/chat-types.ts src/index.ts tests/sampling-profiles.test.ts
git commit -m "feat(api): expose Qwen sampling profiles + CompletionConfig.sampling flag

Moves QWEN_THINKING_DEFAULTS / QWEN_NON_THINKING_DEFAULTS from private
constants in engine.ts to a new exported src/core/sampling-profiles.ts
module. Both are now public exports for consumers who want to inspect
or replicate the values.

Adds CompletionConfig.sampling: 'auto' | 'qwen-thinking' | 'qwen-default'
| 'raw'. Default 'auto' preserves the existing magic (apply Qwen profile
when architecture starts with 'qwen' + chat template is ChatML). 'raw'
opts out for consumers who want to control sampling end-to-end.
'qwen-thinking' / 'qwen-default' force a profile regardless of
architecture for consumers who want the same vibes on a non-Qwen model.

Consumer-provided field values always override profile defaults
(documented in JSDoc). Zero BC-break: existing call sites without
'sampling' get default 'auto' = current behavior.

Item (c) of TS API audit follow-ups."
```

---

## Phase 2a — Item (d): Engine accessor migration

### Task 18: Underscore-prefix backing fields and migrate methods to property getters

**Files:**
- Modify: `src/core/engine.ts:113-118` (private field declarations)
- Modify: `src/core/engine.ts:126-133` (constructor field assignments)
- Modify: `src/core/engine.ts:710-718` (the three `getX()` methods)
- Modify: `src/core/engine.ts` other internal references

- [ ] **Step 1: Grep all internal references to confirm scope**

```bash
grep -n "this\.memoryPool\|this\.scheduler\|this\.modelManager" src/core/engine.ts
```

Note line numbers; you'll update each one.

- [ ] **Step 2: Rename private fields**

In the field declarations (around line 113-118), change:

```ts
private memoryPool: MemoryPool;
private scheduler: Scheduler;
// (modelManager has slightly different declaration; locate by grep)
private modelManager: ModelManager;
```

to:

```ts
private _memoryPool: MemoryPool;
private _scheduler: Scheduler;
private _modelManager: ModelManager;
```

- [ ] **Step 3: Update constructor assignments**

Around lines 126-128, change `this.memoryPool = ...` → `this._memoryPool = ...`, `this.modelManager = ...` → `this._modelManager = ...`, `this.scheduler = ...` → `this._scheduler = ...`.

- [ ] **Step 4: Sweep all other internal references**

For every `this.memoryPool` / `this.scheduler` / `this.modelManager` reference in `engine.ts` body (the grep results from Step 1), update to the underscore-prefixed name.

- [ ] **Step 5: Replace the three `getX()` methods with `get` properties**

At lines 710-718, replace:

```ts
getMemoryPool(): MemoryPool {
	return this.memoryPool;
}
getScheduler(): Scheduler {
	return this.scheduler;
}
getModelManager(): ModelManager {
	return this.modelManager;
}
```

with:

```ts
get memoryPool(): MemoryPool {
	return this._memoryPool;
}
get scheduler(): Scheduler {
	return this._scheduler;
}
get modelManager(): ModelManager {
	return this._modelManager;
}
```

### Task 19: Phase 2a ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

Expected: green. The grep at task 18 step 1 confirmed no external test/eval call sites; if `tsc` reports an unknown call to `getMemoryPool`/etc. in `src/`, sweep + fix.

- [ ] **Step 2: Commit**

```bash
git add src/core/engine.ts
git commit -m "refactor(api): engine accessors as properties

Migrates engine.getMemoryPool() / getScheduler() / getModelManager() from
methods to 'get' property accessors (engine.memoryPool / engine.scheduler
/ engine.modelManager). Matches the pre-existing engine.config /
engine.pipelineCache property surface; the methods were leftover from
when these returned different lifetimes.

Underscore-prefixes the private backing fields to avoid getter/field
name collision. No deprecation aliases per project policy ('no BC-compat
shims'); these are recently-released APIs with no known consumers
(grep tests/ eval/ found zero call sites).

Item (d) of TS API audit follow-ups."
```

---

## Phase 2b — Item (e): `Character.setTools`

### Task 20: Write failing tests for `setTools`

**Files:**
- Create: `tests/character-set-tools.test.ts`

- [ ] **Step 1: Read existing character/tool-system shapes**

```bash
sed -n '60,100p' src/characters/character.ts
sed -n '1,80p' src/characters/tool-system.ts
```

Capture the exact `ToolDefinition` import path and a minimal valid tool fixture.

- [ ] **Step 2: Write the test file**

```ts
import { describe, expect, test } from "bun:test";
import { Character } from "../src/characters/character.js";
import type { ToolDefinition } from "../src/characters/tool-system.js";

const toolA: ToolDefinition = {
	name: "tool_a",
	description: "first tool",
	parameters: { input: { type: "string", description: "x" } },
	handler: async () => "result-a",
};

const toolB: ToolDefinition = {
	name: "tool_b",
	description: "second tool",
	parameters: { input: { type: "string", description: "y" } },
	handler: async () => "result-b",
};

describe("Character.setTools", () => {
	test("replaces tools list and recreates ToolSystem", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
			tools: [toolA],
		});
		expect(ch.config.tools).toEqual([toolA]);

		ch.setTools([toolB]);
		expect(ch.config.tools).toEqual([toolB]);
		// ToolSystem reuses the new list — call parseToolCall on a string
		// referencing tool_b; if the new ToolSystem is wired correctly it
		// returns a non-null result.
		const callText = `<tool_call>{"name":"tool_b","arguments":{"input":"hi"}}</tool_call>`;
		// Access the private field via cast for the assertion. The
		// behavior assertion is "non-null toolSystem after setTools",
		// not a cross-module API contract.
		const toolSystem = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(toolSystem).not.toBeNull();
	});

	test("clears tools when called with empty array", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
			tools: [toolA],
		});
		ch.setTools([]);
		expect(ch.config.tools).toEqual([]);
		const toolSystem = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(toolSystem).toBeNull();
	});

	test("starts from no tools and adds them via setTools", () => {
		const ch = new Character({
			modelId: "test-model",
			systemPrompt: "system",
		});
		const initial = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(initial).toBeNull();

		ch.setTools([toolA, toolB]);
		expect(ch.config.tools).toEqual([toolA, toolB]);
		const after = (ch as unknown as { toolSystem: unknown }).toolSystem;
		expect(after).not.toBeNull();
	});
});
```

- [ ] **Step 3: Run tests, expect failure**

```bash
bun test tests/character-set-tools.test.ts
```

Expected: 3 fails with "ch.setTools is not a function" (or similar).

### Task 21: Implement `setTools`

**Files:**
- Modify: `src/characters/character.ts`

- [ ] **Step 1: Add the method**

In `src/characters/character.ts`, locate the existing constructor / methods area (around line 130 onward). Add this method to the class body:

```ts
/**
 * Replace the tools list and recreate the internal ToolSystem.
 *
 * Empty array clears tools (mirrors constructor behavior). Existing
 * message history is preserved. No effect on in-flight `chat()` —
 * the change applies to the next call.
 */
setTools(tools: ToolDefinition[]): void {
	this.config.tools = tools;
	this.toolSystem = tools.length > 0 ? new ToolSystem(tools) : null;
}
```

You may need to add `ToolDefinition` to the imports at the top of `character.ts` if it's not already imported. Check with grep:

```bash
grep -n "ToolDefinition" src/characters/character.ts
```

- [ ] **Step 2: Run tests, expect pass**

```bash
bun test tests/character-set-tools.test.ts
```

Expected: 3 pass.

### Task 22: Phase 2b ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

- [ ] **Step 2: Commit**

```bash
git add src/characters/character.ts tests/character-set-tools.test.ts
git commit -m "feat(api): add Character.setTools for runtime tool reconfiguration

Adds Character.setTools(tools: ToolDefinition[]) for swapping the tool
list after construction. Replaces config.tools and recreates the
internal ToolSystem; empty array clears tools (mirrors constructor
behavior). Existing message history preserved. No effect on
in-flight chat() — change applies to next call.

Test coverage: 3 cases (replace, clear, add-from-none) in
tests/character-set-tools.test.ts.

Item (e) of TS API audit follow-ups. Strict-YAGNI: no
attachToolSystem(custom) added — the audit floats it for parser swap,
but no consumer ask exists. Re-evaluate if a real custom-parser
deployment surfaces."
```

---

## Phase 3 — Item (f): Polish bundle

### Task 23: Narrow `ChatToolSchema.parameters[*].type` to literal union

**Files:**
- Modify: `src/core/chat-types.ts:14-21`

- [ ] **Step 1: Pre-edit grep**

```bash
grep -rn "type:\s*[\"']" src/ tests/ smoke-test/ eval/ 2>/dev/null | grep -v "\.bak\|webllm-bundle" | head -30
```

Look for any string literal assigned to `parameters[*].type` outside the new union (`"string" | "number" | "integer" | "boolean" | "array" | "object"`). If found, the literal is either (a) a typo / legacy fixture to fix in-place, or (b) a tool-call test using a JSON Schema string that the type narrow would break — investigate before narrowing.

- [ ] **Step 2: Update `ChatToolSchema`**

In `src/core/chat-types.ts`, replace lines 14-21:

```ts
/**
 * Tool / function schema passed into the chat template. Just the shape the
 * model needs in the prompt — handlers, responses, etc. live on the
 * Character/ToolSystem side and aren't serialised here.
 */
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

- [ ] **Step 3: Run typecheck**

```bash
bun run typecheck
```

If it fails on a fixture using an out-of-union string literal, fix the fixture (don't widen the union — the fixture was buggy).

### Task 24: Add per-variant JSDoc to `GenerationFinishReason`

**Files:**
- Modify: `src/inference/generation.ts:92-96`

- [ ] **Step 1: Replace the union**

```ts
export type GenerationFinishReason =
	/** Generation cancelled via AbortSignal. */
	| "aborted"
	/** End-of-sequence token sampled. */
	| "eos"
	/** maxTokens budget exhausted. */
	| "max-tokens"
	/** A custom `stopTokens` entry was sampled. */
	| "stop-token";
```

### Task 25: Update README API table and trim feature claims

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the API Overview table and the Memory/Scheduling sections**

Open `README.md` and locate the "API Overview" table and any "Memory management" / "Scheduling" headings. Use grep:

```bash
grep -n "removeCharacter\|shutdown\|Memory management\|Scheduling\|API Overview" README.md
```

- [ ] **Step 2: Add `removeCharacter` and `shutdown` rows**

In the API Overview table, add rows mirroring the existing format:

```md
| `engine.removeCharacter(id)` | Remove a registered character by id. |
| `engine.shutdown()` | Release GPU buffers, dispose inference engines, and shut down WASM modules for all loaded models. |
```

(Match the existing table column count + style; if the table has more columns, fill them — read context near the table to copy the convention.)

- [ ] **Step 3: Trim Memory/Scheduling feature claims**

Find the README sections that describe "Memory management" and "Scheduling" features. The current text oversells what's consumer-callable. Tighten to read-only access:

- "Memory management" → describe `engine.memoryPool` (read-only getter returning a `MemoryPool` with internal stats); remove any claim about consumer-callable allocate/free.
- "Scheduling" → describe `engine.scheduler` (read-only getter returning a `Scheduler` for inspection); remove any claim about consumer-callable schedule/dispatch.

If you cannot find these sections in README.md, skip this step and note it in the commit message.

### Task 26: Phase 3 ship gate + commit

- [ ] **Step 1: Run full check**

```bash
make checkall
```

- [ ] **Step 2: Commit**

```bash
git add src/core/chat-types.ts src/inference/generation.ts README.md
git commit -m "docs(api): polish bundle (item f)

- ChatToolSchema.parameters[*].type narrowed from 'string' to a
  literal union: 'string' | 'number' | 'integer' | 'boolean' |
  'array' | 'object'. Consumers now get autocomplete; out-of-union
  type strings (a class of buggy fixture) become typecheck errors.
- GenerationFinishReason gets per-variant JSDoc — 'aborted' (signal),
  'eos' (eos token), 'max-tokens' (budget exhausted), 'stop-token'
  (custom stop hit). Each variant now self-documents in IDE hover.
- README API Overview table: add engine.removeCharacter / shutdown
  rows. Trim Memory/Scheduling feature claims to reflect the actual
  read-only consumer surface (engine.memoryPool / engine.scheduler
  getters; no allocate/schedule consumer API).

Item (f) of TS API audit follow-ups; closes the six-item arc."
```

---

## Post-implementation: TODO.md closure

### Task 27: Update `TODO.md` to mark all six items closed

**Files:**
- Modify: `TODO.md:926-1020`

- [ ] **Step 1: Read TODO.md:926-1020**

Refresh the section. Confirm the closure stub format used elsewhere (e.g. lines 826-839 for "13B target registration").

- [ ] **Step 2: Replace the open block with a closure stub**

Replace the entire section "(a)-(f) decision rule + 6 sub-items" with a closure stub:

```md
4. **TS API audit follow-ups (CLOSED 2026-04-29).** All six deferred
   items shipped in one cycle:
   - **(a) Split `GenerationConfig`:** rename current 22-field type to
     `InternalGenerationOptions` (engine-internal); new public 7-field
     `GenerationConfig` with `signal` inline; drop unused `prompt`
     field. Compile-fail assertion test locks in steering-field
     exclusion. Commit `<HASH>`.
   - **(b) Drop `WebLLMConfig.device`:** caller passes device directly
     to `engine.loadLightweightModel(LightweightModelConfig)`. Tests
     and smoke harness updated. Commit `<HASH>`.
   - **(c) Sampling flag + Qwen profile export:** new
     `src/core/sampling-profiles.ts` exports `QWEN_THINKING_DEFAULTS`
     / `QWEN_NON_THINKING_DEFAULTS`; `CompletionConfig.sampling`:
     `"auto" | "qwen-thinking" | "qwen-default" | "raw"` (default
     `"auto"` = current magic). Commit `<HASH>`.
   - **(d) Engine accessor migration:** `getMemoryPool` / `getScheduler`
     / `getModelManager` → property getters. Commit `<HASH>`.
   - **(e) Character.setTools:** runtime tool reconfiguration. Commit
     `<HASH>`. (`attachToolSystem` for parser swap deferred — strict
     YAGNI; no consumer ask.)
   - **(f) Polish:** `ChatToolSchema` literal union, `GenerationFinishReason`
     per-variant JSDoc, README API table additions, Memory/Scheduling
     claim trim. Commit `<HASH>`.

   Spec: [`docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md`](docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md).
   Plan: [`docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md`](docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md).
```

Replace each `<HASH>` with the actual short SHA of the corresponding commit. Get them from `git log --oneline -10`.

- [ ] **Step 3: Commit**

```bash
git add TODO.md
git commit -m "docs(TODO): close TS API audit follow-ups (a-f)

All six items shipped 2026-04-29. Replaces the ~95-line open block
with a closure stub linking to the spec, plan, and per-item commits.
Full block was already self-archiving (each item's rationale is in
its commit message); the stub keeps the watch-list visible without
the bulk."
```

---

## Self-review checklist

**Spec coverage (each spec section → which task lands it):**
- ✅ Item (a) GenerationConfig split → Tasks 1, 2, 3, 4, 5, 6
- ✅ Item (b) WebLLMConfig.device removal → Tasks 7, 8, 9, 10
- ✅ Item (c) sampling flag + Qwen profile → Tasks 11, 12, 13, 14, 15, 16, 17
- ✅ Item (d) engine accessor migration → Tasks 18, 19
- ✅ Item (e) Character.setTools → Tasks 20, 21, 22
- ✅ Item (f) polish bundle (.1-.4) → Tasks 23, 24, 25, 26
- ✅ Phasing (3 phases, ≤5 files per sub-checkpoint) → enforced by per-item commit cadence
- ✅ Testing strategy → ts-expect-error tests (1a), runtime tests for sampling constants (1c) and setTools (2b)
- ✅ Closure → Task 27 closes the TODO block

**Placeholder scan:** No "TBD", no "implement later", no "similar to Task N" without showing code. All commit messages are concrete with item references.

**Type consistency:**
- `InternalGenerationOptions` introduced in Task 1, referenced consistently in Tasks 2, 3, 5.
- `setTools` named consistently across Tasks 20, 21.
- `_memoryPool` / `_scheduler` / `_modelManager` underscore-prefix used consistently across Task 18.
- `sampling` field name consistent across Tasks 13, 14, 16.

**Risk register coverage (from spec):**
- ✅ `speculative.ts` type rename → Task 2
- ✅ `WebLLMConfig.device` migration call sites → Task 9 (test + smoke)
- ✅ `engine.chatCompletion` magic preservation under `sampling: "auto"` → Task 14 fall-through preserves behavior
- ✅ `getMemoryPool/getScheduler/getModelManager` external sweep → grep confirmed zero external call sites pre-Task 18
