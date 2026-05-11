# Gemma 4 E2B Correctness-First Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inference support for Gemma 4 E2B (Q4_K_M) through 5 staged ship gates: foundation refactor → surface wiring → semantic correctness → SWA → shared-KV+bench+closure.

**Architecture:** Each stage produces a closure report under `eval/reports/gemma-4-stage{N}-*/SUMMARY.md` and passes both build (`make checkall`) and runtime gates. Existing models keep current behavior through Stage 1 by reading optional per-layer arrays when present and falling back to scalar fields when absent. Gemma 4 dispatches through new arch-gated code paths added in Stages 2–5.

**Tech Stack:** TypeScript (project source), Bun (build + test), Emscripten/JSPI WASM (`ggml-webgpu` from patched llama.cpp on branch `webllm-browser-patches`), WebGPU, agentchrome (browser smoke), `make` + bash test harnesses.

**Spec:** `docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md`. Reference the spec for scope decisions, gates, and risk register; do not restate those here.

---

## File structure decisions

This plan modifies these files (precise touch points listed per task):

| File | Role | Touched in stages |
|---|---|---|
| `src/core/types.ts` | `ModelHyperparams`, `ModelArchitecture` | 1, 2 |
| `src/models/model-loader.ts` | GGUF metadata → hyperparams | 1, 2, 3, 5 |
| `src/models/gguf-parser.ts` | Add `getMetaNumberArray` / `getMetaBooleanArray` if missing | 1 |
| `src/inference/chat-template.ts` | `formatGemma4`, detection | 2 |
| `src/inference/model-inference.ts` | Per-layer hp dispatch, PLE, dual RoPE, SWA, shared-KV | 1 (no-op compat), 3, 4, 5 |
| `src/inference/causal-embedder-inference.ts` | Per-layer hp dispatch (no-op compat) | 1 |
| `src/inference/encoder-inference.ts` | Per-layer hp dispatch (no-op compat) | 1 |
| `src/models/kv-cache.ts` | Ref-shared KV layer support | 5 |
| `src/core/engine.ts` | Stop-token registration | 2 |
| `src/core/sampling-profiles.ts` | `GEMMA4_DEFAULTS`, dispatch | 2 |
| `src/persistence/indexeddb-store.ts` | Ref-shared KV serialization | 5 |
| `eval/models.ts` | `gemma-4-e2b-it-q4km` registration | 2 |
| `tests/models/*.test.ts` | New per-layer hp tests, formatter tests, sampler tests | 1, 2 |
| `Makefile` | Stage 1 closure-report target (one-line addition optional) | — |

External assets:
- `smoke-test/models/gemma-4-e2b-it-q4km.gguf` — already symlinked
- `smoke-test/webllm-models.js` — regenerated each stage via `bun build eval/models.ts` (Makefile `smoke-test` target)

---

# Stage 1 — Per-layer hyperparams refactor (foundation)

**Goal:** Add optional per-layer arrays to `ModelHyperparams`. Existing models leave arrays undefined (scalar reads unchanged). Loader populates arrays for Gemma 4 from GGUF.

**Stage 1 ship gate (run all):**

1. `make checkall` — green
2. Smoke regression: TinyLlama / qwen3-0.6b / qwen3-1.7b generate first token unchanged vs current dashboard records
3. Gemma 4 hyperparams probe: loading `gemma-4-e2b-it-q4km.gguf` populates `embeddingHeadLengthPerLayer`, `feedForwardLengthPerLayer`, `ropeDimensionCountPerLayer`, `ropeFreqBasePerLayer`, `slidingWindowPattern` with the values from the GGUF metadata dump

## Task 1.1: Add per-layer hp fields to `ModelHyperparams`

**Files:**
- Modify: `src/core/types.ts:186-224` (extend `ModelHyperparams` interface)

- [ ] **Step 1: Re-read the file to confirm current shape**

Run: `Read src/core/types.ts offset=186 limit=40`

Confirm `ModelHyperparams` has scalars `embeddingHeadLength`, `feedForwardLength`, `ropeFreqBase`.

- [ ] **Step 2: Append new optional per-layer fields**

Edit `src/core/types.ts`. After the existing `alibiMaxBias?: number;` line at the end of `ModelHyperparams`, add:

```typescript
	/**
	 * Per-layer head dimension. When present, dispatch code MUST index
	 * by layer (`embeddingHeadLengthPerLayer[i]`) instead of reading the
	 * scalar `embeddingHeadLength`. Length === `layerCount`. Absent for
	 * uniform architectures (Llama, Mistral, Qwen, Phi-3, etc.). Present
	 * for Gemma 4 where SWA layers use a smaller head_dim than global
	 * layers.
	 */
	embeddingHeadLengthPerLayer?: number[];
	/**
	 * Per-layer FFN intermediate size. When present, dispatch code MUST
	 * index by layer. Length === `layerCount`. Absent for uniform
	 * architectures. Present for Gemma 4 (6144 first 15 layers, 12288
	 * remaining 20).
	 */
	feedForwardLengthPerLayer?: number[];
	/**
	 * Per-layer RoPE dimension count. When present, dispatch code MUST
	 * read this for RoPE; absent → use the legacy global value (or
	 * `embeddingHeadLength` if no legacy value applies).
	 */
	ropeDimensionCountPerLayer?: number[];
	/**
	 * Per-layer RoPE base frequency. Length === `layerCount`. Absent →
	 * use scalar `ropeFreqBase`. Present for Gemma 4 (1e6 for global,
	 * 1e4 for SWA).
	 */
	ropeFreqBasePerLayer?: number[];
	/**
	 * Length === `layerCount`. `true` means the layer uses sliding-window
	 * attention; `false` means global causal attention. Absent → all
	 * layers global (the existing behavior). Present for Gemma 4.
	 */
	slidingWindowPattern?: boolean[];
	/** Sliding-window size (token count). Absent unless `slidingWindowPattern` is. */
	slidingWindowSize?: number;
	/**
	 * Number of trailing layers whose attn_k and attn_v reference the
	 * KV cache of an earlier layer instead of allocating their own.
	 * 0 (or absent) means no sharing. Gemma 4 E2B reports 20 (last 20
	 * of 35 layers share earlier KV).
	 */
	sharedKvLayers?: number;
	/**
	 * Final logit softcap value (`tanh(logits / s) * s`). 0 → no softcap.
	 * Read from GGUF `<arch>.final_logit_softcapping`. Present for
	 * Gemma family models (Gemma 4 E2B reports 30.0).
	 */
	finalLogitSoftcap?: number;
```

- [ ] **Step 3: Run typecheck**

Run: `bun run typecheck` (or `make typecheck` if that's the alias)
Expected: PASS (purely additive optional fields; no existing code reads them yet).

- [ ] **Step 4: Commit**

```bash
git add src/core/types.ts
git commit -m "refactor(types): add per-layer hp fields for Gemma 4 prep

Adds optional embeddingHeadLengthPerLayer, feedForwardLengthPerLayer,
ropeDimensionCountPerLayer, ropeFreqBasePerLayer, slidingWindowPattern,
slidingWindowSize, sharedKvLayers, finalLogitSoftcap to
ModelHyperparams. All optional; existing architectures leave them
undefined and continue reading scalar fields. Gemma 4 populates these
in the next commit."
```

---

## Task 1.2: GGUF array readers — `getMetaNumberArray`, `getMetaBooleanArray`

**Files:**
- Inspect: `src/models/gguf-parser.ts` (or wherever `getMetaNumber` lives)
- Modify: same file — add the two array readers if missing

- [ ] **Step 1: Locate `getMetaNumber` definition**

Run: `grep -n "export function getMetaNumber\b" src/models/*.ts`
Read the file at that line.

- [ ] **Step 2: Check if array readers exist**

Run: `grep -n "getMetaNumberArray\|getMetaBooleanArray" src/models/*.ts`

If both already exist, skip to Step 4.

- [ ] **Step 3: Add the missing readers**

Pattern (mirror the existing `getMetaNumber` shape; the GGUF parser
already supports `ARRAY` value types — see existing
`getMetaStringArray` / `getMetaNumberArray` for token type IDs). Add:

```typescript
export function getMetaNumberArray(
	ctx: GgufContext,
	key: string,
	fallback: number[] = [],
): number[] {
	const field = ctx.fields.get(key);
	if (!field) return fallback;
	// GGUF array of numeric values — the existing parser stores them as
	// number[] already if the GGUF type code is one of the numeric
	// scalar enums (UINT32, INT32, FLOAT32, etc.) wrapped in ARRAY.
	if (!Array.isArray(field.value)) {
		throw new Error(`GGUF key ${key} is not an array`);
	}
	return field.value as number[];
}

export function getMetaBooleanArray(
	ctx: GgufContext,
	key: string,
	fallback: boolean[] = [],
): boolean[] {
	const field = ctx.fields.get(key);
	if (!field) return fallback;
	if (!Array.isArray(field.value)) {
		throw new Error(`GGUF key ${key} is not an array`);
	}
	// GGUF BOOL is encoded as UINT8 in the spec; values are 0 or 1.
	return (field.value as Array<number | boolean>).map((v) =>
		typeof v === "boolean" ? v : v !== 0,
	);
}
```

(The exact shape depends on the existing parser API — `field.value` may
be named differently. Read the existing reader implementations and
mirror their access pattern.)

- [ ] **Step 4: Add minimal unit tests**

**Files:**
- Create: `tests/models/gguf-parser-array-readers.test.ts`

```typescript
import { describe, it, expect } from "bun:test";
import { GgufParser } from "../../src/models/gguf-parser.js";
import { getMetaNumberArray, getMetaBooleanArray }
	from "../../src/models/gguf-parser.js";
import { readFileSync, existsSync } from "node:fs";

const GEMMA4 = "smoke-test/models/gemma-4-e2b-it-q4km.gguf";

describe.skipIf(!existsSync(GEMMA4))(
	"GGUF array readers on Gemma 4 E2B",
	() => {
		it("reads gemma4.feed_forward_length as a 35-element array", async () => {
			const buf = readFileSync(GEMMA4);
			const ctx = await GgufParser.parse(buf.buffer);
			const ffn = getMetaNumberArray(ctx, "gemma4.feed_forward_length");
			expect(ffn).toHaveLength(35);
			expect(ffn[0]).toBe(6144);
			expect(ffn[14]).toBe(6144);
			expect(ffn[15]).toBe(12288);
			expect(ffn[34]).toBe(12288);
		});

		it("reads gemma4.attention.sliding_window_pattern as 35-element bool array", async () => {
			const buf = readFileSync(GEMMA4);
			const ctx = await GgufParser.parse(buf.buffer);
			const pat = getMetaBooleanArray(
				ctx,
				"gemma4.attention.sliding_window_pattern",
			);
			expect(pat).toHaveLength(35);
			// Pattern (T,T,T,T,F) × 7 — index 4 (0-based) is first global
			expect(pat[0]).toBe(true);
			expect(pat[3]).toBe(true);
			expect(pat[4]).toBe(false);
			expect(pat[9]).toBe(false);
		});
	},
);
```

- [ ] **Step 5: Run tests**

Run: `bun test tests/models/gguf-parser-array-readers.test.ts -v`
Expected: 2 PASS (or 2 SKIP if the GGUF symlink is missing on this checkout — that's still acceptable).

- [ ] **Step 6: Commit**

```bash
git add src/models/gguf-parser.ts tests/models/gguf-parser-array-readers.test.ts
git commit -m "feat(gguf): array readers for per-layer hp arrays

Adds getMetaNumberArray and getMetaBooleanArray to read GGUF array-
typed metadata (e.g., gemma4.feed_forward_length is a 35-element
INT32 array; gemma4.attention.sliding_window_pattern is a 35-element
BOOL array). Skip-gated on the gemma-4-e2b-it-q4km.gguf symlink so
the suite still passes on fresh clones without the fixture."
```

---

## Task 1.3: Populate per-layer arrays in `model-loader` for Gemma 4

**Files:**
- Modify: `src/models/model-loader.ts:125-160` (the hyperparams return object)

- [ ] **Step 1: Re-read the function**

Run: `Read src/models/model-loader.ts offset=70 limit=100`

Locate the section building the hyperparams return value.

- [ ] **Step 2: Add Gemma 4 per-layer population**

After the existing `alibiMaxBias` field assignment in the return
object, *before* the closing `}` of the return value, add this
arch-gated block. (The return is a single object literal; the new
fields are conditional, so build the object first then mutate, OR
spread an arch-gated record. The spread pattern is cleaner.)

Find the existing block in the file (around line 125-160) and replace
the bare `return { ... };` with a spread + conditional:

```typescript
		const baseHp: ModelHyperparams = {
			architecture: arch,
			contextLength: getMetaNumber(ctx, `${metaPrefix}.context_length`, 2048),
			embeddingLength,
			headCount,
			headCountKv: getMetaNumber(
				ctx,
				`${metaPrefix}.attention.head_count_kv`,
				headCount,
			),
			layerCount: getMetaNumber(ctx, `${metaPrefix}.block_count`, 32),
			vocabularySize: 0,
			embeddingHeadLength: getMetaNumber(
				ctx,
				`${metaPrefix}.attention.key_length`,
				embeddingLength / headCount,
			),
			feedForwardLength: getMetaNumber(
				ctx,
				`${metaPrefix}.feed_forward_length`,
				11008,
			),
			ropeFreqBase:
				getMetaNumberOptional(ctx, `${metaPrefix}.rope_freq_base`) ??
				getMetaNumberOptional(ctx, `${metaPrefix}.rope.freq_base`) ??
				10000,
			ropeScale: getMetaNumber(ctx, `${metaPrefix}.rope_scale`, 1),
			normEpsilon,
			expertCount: getMetaNumber(ctx, `${metaPrefix}.expert_count`, 0),
			expertUsedCount: getMetaNumber(ctx, `${metaPrefix}.expert_used_count`, 0),
			quantType,
			poolingType,
			causalAttention,
			alibiMaxBias,
		};

		if (arch === "gemma4") {
			const layerCount = baseHp.layerCount;
			const keyLenGlobal = getMetaNumber(
				ctx,
				`${metaPrefix}.attention.key_length`,
				baseHp.embeddingHeadLength,
			);
			const keyLenSwa = getMetaNumber(
				ctx,
				`${metaPrefix}.attention.key_length_swa`,
				keyLenGlobal,
			);
			const ropeDimGlobal = getMetaNumber(
				ctx,
				`${metaPrefix}.rope.dimension_count`,
				keyLenGlobal,
			);
			const ropeDimSwa = getMetaNumber(
				ctx,
				`${metaPrefix}.rope.dimension_count_swa`,
				keyLenSwa,
			);
			const freqBaseGlobal = getMetaNumber(
				ctx,
				`${metaPrefix}.rope.freq_base`,
				1_000_000,
			);
			const freqBaseSwa = getMetaNumber(
				ctx,
				`${metaPrefix}.rope.freq_base_swa`,
				10_000,
			);
			const swaPattern = getMetaBooleanArray(
				ctx,
				`${metaPrefix}.attention.sliding_window_pattern`,
				new Array(layerCount).fill(false),
			);
			const ffnPerLayer = getMetaNumberArray(
				ctx,
				`${metaPrefix}.feed_forward_length`,
				new Array(layerCount).fill(baseHp.feedForwardLength),
			);
			const headPerLayer = swaPattern.map((isSwa) =>
				isSwa ? keyLenSwa : keyLenGlobal,
			);
			const ropeDimPerLayer = swaPattern.map((isSwa) =>
				isSwa ? ropeDimSwa : ropeDimGlobal,
			);
			const ropeFreqBasePerLayer = swaPattern.map((isSwa) =>
				isSwa ? freqBaseSwa : freqBaseGlobal,
			);

			return {
				...baseHp,
				embeddingHeadLengthPerLayer: headPerLayer,
				feedForwardLengthPerLayer: ffnPerLayer,
				ropeDimensionCountPerLayer: ropeDimPerLayer,
				ropeFreqBasePerLayer: ropeFreqBasePerLayer,
				slidingWindowPattern: swaPattern,
				slidingWindowSize: getMetaNumber(
					ctx,
					`${metaPrefix}.attention.sliding_window`,
					512,
				),
				sharedKvLayers: getMetaNumber(
					ctx,
					`${metaPrefix}.attention.shared_kv_layers`,
					0,
				),
				finalLogitSoftcap: getMetaNumberOptional(
					ctx,
					`${metaPrefix}.final_logit_softcapping`,
				),
			};
		}

		return baseHp;
```

(Import `getMetaNumberArray`, `getMetaBooleanArray` from the same module
that exports `getMetaNumber`.)

Note: The existing return statement at lines ~125-160 is replaced
wholesale by the above. Confirm the indentation and trailing commas
match the surrounding file style.

- [ ] **Step 3: Run typecheck**

Run: `bun run typecheck`
Expected: PASS.

- [ ] **Step 4: Add hp populate test**

**Files:**
- Create: `tests/models/model-loader-gemma4-hparams.test.ts`

```typescript
import { describe, it, expect } from "bun:test";
import { ModelLoader } from "../../src/models/model-loader.js";
import { readFileSync, existsSync } from "node:fs";

const GEMMA4 = "smoke-test/models/gemma-4-e2b-it-q4km.gguf";

describe.skipIf(!existsSync(GEMMA4))(
	"ModelLoader populates Gemma 4 per-layer hparams",
	() => {
		it("populates all per-layer arrays from GGUF", async () => {
			const buf = readFileSync(GEMMA4);
			const parsed = await ModelLoader.parse(buf.buffer);
			const hp = parsed.hyperparams;

			expect(hp.architecture).toBe("gemma4");
			expect(hp.layerCount).toBe(35);

			// Pattern (T,T,T,T,F) × 7 → indices 4, 9, 14, 19, 24, 29, 34 = global
			expect(hp.slidingWindowPattern).toBeDefined();
			expect(hp.slidingWindowPattern).toHaveLength(35);
			expect(hp.slidingWindowPattern![4]).toBe(false); // global
			expect(hp.slidingWindowPattern![0]).toBe(true); // SWA

			// head_dim: 512 global, 256 SWA
			expect(hp.embeddingHeadLengthPerLayer).toBeDefined();
			expect(hp.embeddingHeadLengthPerLayer![0]).toBe(256);
			expect(hp.embeddingHeadLengthPerLayer![4]).toBe(512);

			// rope_dim: 512 global, 256 SWA
			expect(hp.ropeDimensionCountPerLayer![0]).toBe(256);
			expect(hp.ropeDimensionCountPerLayer![4]).toBe(512);

			// freq_base: 1e6 global, 1e4 SWA
			expect(hp.ropeFreqBasePerLayer![0]).toBe(10_000);
			expect(hp.ropeFreqBasePerLayer![4]).toBe(1_000_000);

			// FFN: 6144 layers 0-14, 12288 layers 15-34
			expect(hp.feedForwardLengthPerLayer).toBeDefined();
			expect(hp.feedForwardLengthPerLayer![0]).toBe(6144);
			expect(hp.feedForwardLengthPerLayer![14]).toBe(6144);
			expect(hp.feedForwardLengthPerLayer![15]).toBe(12288);
			expect(hp.feedForwardLengthPerLayer![34]).toBe(12288);

			// Other Gemma 4 fields
			expect(hp.slidingWindowSize).toBe(512);
			expect(hp.sharedKvLayers).toBe(20);
			expect(hp.finalLogitSoftcap).toBe(30);
		});
	},
);
```

- [ ] **Step 5: Run the new test**

Run: `bun test tests/models/model-loader-gemma4-hparams.test.ts -v`
Expected: 1 PASS (or 1 SKIP without fixture).

- [ ] **Step 6: Run full test suite**

Run: `make checkall`
Expected: green. **If any existing test fails, STOP** — the spread/conditional return changed a code path the existing tests rely on. Diagnose before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/models/model-loader.ts tests/models/model-loader-gemma4-hparams.test.ts
git commit -m "feat(model-loader): Gemma 4 per-layer hyperparams from GGUF

Reads gemma4.attention.{key_length,key_length_swa,sliding_window_*,
shared_kv_layers}, gemma4.feed_forward_length array,
gemma4.rope.{dimension_count,dimension_count_swa,freq_base,freq_base_swa},
and gemma4.final_logit_softcapping into the new per-layer hp fields.
Existing architectures return only scalar fields (per-layer arrays
undefined) — zero behavioral delta for Llama/Mistral/Qwen/Phi-3."
```

---

## Task 1.4: Verify Stage 1 — runtime regression check

**Goal:** Confirm the diff in model-loader.ts is truly additive — existing models load and decode unchanged.

- [ ] **Step 1: Rebuild smoke bundles**

Run:
```bash
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
bun build eval/models.ts --outfile smoke-test/webllm-models.js --target browser
```

(WASM artifacts unchanged — no need to rebuild them for a pure TS change.)

- [ ] **Step 2: Restart smoke server**

Run: `make smoke-restart`
Expected output: `smoke server running on http://localhost:8031`.

- [ ] **Step 3: Probe TinyLlama (first regression model)**

Run:
```bash
agentchrome --port 63846 tabs list 2>&1 | head -1
# Pick the existing tab id (TAB_ID).
```

Then:
```bash
agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=tinyllama-1.1b-chat-q4_0&v=stage1-1" \
  --tab <TAB_ID> --timeout 60000
```

Wait ~25 s. Then:
```bash
agentchrome --port 63846 page snapshot --tab <TAB_ID>
```

Confirm in the snapshot output: "[7/8] First token: <token>" or equivalent first-generated-token line. Record the token ID emitted.

**Cross-check** against the dashboard records: query the live DB or the smoke run history for `tinyllama-1.1b-chat-q4_0`'s recent `generatedIds[0]` value. The new run must emit the same `generatedIds[0]`.

If the dashboard isn't tracking this directly, use a known-good static expected value from a recent run-complete event. (Tip: the simplest path is to capture *baseline*-from-main and *post-Task-1.3* values into a tiny ad-hoc JSON file under `eval/reports/gemma-4-stage1-per-layer-hp-2026-05-10/regression-check.json` and compare them.)

- [ ] **Step 4: Repeat for qwen3-0.6b-q4f16 and qwen3-1.7b-q4f16**

Same procedure as Step 3 with model IDs `qwen3-0.6b-q4f16` and `qwen3-1.7b-q4f16`. Cache-bust with `&v=stage1-2` and `&v=stage1-3` respectively.

- [ ] **Step 5: Write Stage 1 closure report**

**Files:**
- Create: `eval/reports/gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md`

```markdown
# Gemma 4 Stage 1 — Per-layer hyperparams refactor closure

**Date:** 2026-05-10
**Spec:** docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md
**Plan:** docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md
**Commits:** <list of feat/refactor commits from this stage>

## Build gate
`make checkall` — green. Skip count unchanged (33 baseline).

## Runtime gate (no regression)
| Model | generatedIds[0] pre-refactor | generatedIds[0] post-refactor | Status |
|---|---|---|---|
| tinyllama-1.1b-chat-q4_0 | <T0> | <T0> | OK |
| qwen3-0.6b-q4f16 | <T0> | <T0> | OK |
| qwen3-1.7b-q4f16 | <T0> | <T0> | OK |

## Runtime gate (Gemma 4 hp populate)
Loaded `gemma-4-e2b-it-q4km.gguf` through the loader; per-layer arrays
populated as documented in `tests/models/model-loader-gemma4-hparams.test.ts`.
Generation still fails at `buildQKV` (expected — dispatch code paths
not yet wired to read the new arrays; that's Stages 2–5).

## Follow-ups
- Stage 2 wires `final_logit_softcap` and the `gemma4` arch dispatch.
- Stage 3 reads `embeddingHeadLengthPerLayer` in `buildQKV`.
```

- [ ] **Step 6: Commit Stage 1 closure**

```bash
git add eval/reports/gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md
git commit -m "docs(reports): Gemma 4 Stage 1 per-layer hp closure"
```

---

# Stage 2 — Gemma 4 surface wiring

**Goal:** Add the chat-template, sampler defaults, stop-token registration, architecture union member, registration entry, and `final_logit_softcap` plumbing — everything required to dispatch Gemma 4 through `ModelInference` end-to-end. Generation may still produce semantically-wrong output (Stage 3 fixes that) but the harness must reach the generation loop, run softmax+sampling, and emit a coherent-ASCII token stream that terminates on `<end_of_turn>`.

**Stage 2 ship gate (run all):**

1. `make checkall` — green
2. Browser smoke loads `?model=gemma-4-e2b-it-q4km` and greedy-decodes 5 ASCII tokens on `"The capital of France is"` (semantic correctness not yet required)
3. Multi-turn chat stops cleanly on `<end_of_turn>` (no runaway)

## Task 2.1: Add `"gemma4"` to `ModelArchitecture`

**Files:**
- Modify: `src/core/types.ts:100-114` (`ModelArchitecture` union)

- [ ] **Step 1: Edit the union**

In `src/core/types.ts`, change:
```typescript
	| "phi3"
	| "gemma"
	| "qwen"
```
to:
```typescript
	| "phi3"
	| "gemma"
	| "gemma2"
	| "gemma3"
	| "gemma4"
	| "qwen"
```

`gemma2` and `gemma3` are added for completeness so future probes don't get blocked on type-only edits.

- [ ] **Step 2: Run typecheck**

Run: `bun run typecheck`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/core/types.ts
git commit -m "refactor(types): ModelArchitecture union adds gemma2/gemma3/gemma4

Type-only addition. No dispatch code changes yet — Gemma 4 dispatching
is wired in subsequent commits."
```

---

## Task 2.2: `formatGemma4` chat-template formatter + tests

**Files:**
- Modify: `src/inference/chat-template.ts` (add `formatGemma4`, register in dispatch map, extend `detectChatTemplate`)
- Modify: same file's `ChatTemplateFamily` union or equivalent literal type
- Create: `tests/inference/chat-template-gemma4.test.ts`

- [ ] **Step 1: Inspect existing `formatGemma`**

Run: `Read src/inference/chat-template.ts offset=280 limit=40`

Confirm the shape (uses `<start_of_turn>role\n...<end_of_turn>\n`). For
Gemma 4 chat without tool calls, the same format applies — verified
against the GGUF jinja template's plain-message macro path.

- [ ] **Step 2: Write the failing test first**

**Files:**
- Create: `tests/inference/chat-template-gemma4.test.ts`

```typescript
import { describe, it, expect } from "bun:test";
import {
	formatGemma4,
	detectChatTemplate,
} from "../../src/inference/chat-template.js";

describe("formatGemma4", () => {
	it("emits <start_of_turn>user...<end_of_turn>", () => {
		const out = formatGemma4(
			[{ role: "user", content: "Hello." }],
			/* addGenerationPrompt */ true,
		);
		expect(out).toBe(
			"<start_of_turn>user\nHello.<end_of_turn>\n" +
				"<start_of_turn>model\n",
		);
	});

	it("maps assistant role to 'model'", () => {
		const out = formatGemma4(
			[
				{ role: "user", content: "Hi" },
				{ role: "assistant", content: "Hey" },
			],
			false,
		);
		expect(out).toContain("<start_of_turn>model\nHey<end_of_turn>");
	});

	it("does not append generation prompt when flag is false", () => {
		const out = formatGemma4(
			[{ role: "user", content: "Q" }],
			false,
		);
		expect(out).not.toContain("<start_of_turn>model\n");
		expect(out.endsWith("<end_of_turn>\n")).toBe(true);
	});
});

describe("detectChatTemplate — gemma4", () => {
	// Gemma 4's GGUF chat template starts with the format_parameters macro
	// for tool calls; absent tools the plain message branch uses the same
	// <start_of_turn>...<end_of_turn> markers as Gemma 2.
	it("returns 'gemma4' when template contains '{% macro format_parameters'", () => {
		const tmpl =
			"{%- macro format_parameters(properties, required, filter_keys=false) -%}<start_of_turn>user\n{{ messages[0].content }}<end_of_turn>";
		expect(detectChatTemplate(tmpl)).toBe("gemma4");
	});

	it("still returns 'gemma' for Gemma 2's simpler template", () => {
		const tmpl =
			"<start_of_turn>{{ messages[0].role }}\n{{ messages[0].content }}<end_of_turn>";
		expect(detectChatTemplate(tmpl)).toBe("gemma");
	});
});
```

- [ ] **Step 3: Run tests; verify they fail**

Run: `bun test tests/inference/chat-template-gemma4.test.ts -v`
Expected: 5 FAIL (`formatGemma4 is not exported`).

- [ ] **Step 4: Add `formatGemma4` implementation**

In `src/inference/chat-template.ts`, after the existing `formatGemma`
function (around line 282), add:

```typescript
function formatGemma4(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		const role = msg.role === "assistant" ? "model" : msg.role;
		prompt += `<start_of_turn>${role}\n${msg.content}<end_of_turn>\n`;
	}
	if (addGenerationPrompt) {
		prompt += "<start_of_turn>model\n";
	}
	return prompt;
}
```

(Identical to `formatGemma` body; kept as a separate function so
family-specific divergence can be added later without spilling into
Gemma 2 callers.)

Also export it (find the existing `export` for `formatGemma` and add
`formatGemma4` similarly, e.g. add to the formatter dispatch map):

```typescript
const FORMATTERS = {
	// ...
	gemma: formatGemma,
	gemma4: formatGemma4,
	// ...
};
```

And add `"gemma4"` to the `ChatTemplateFamily` union or wherever the
family literal type is declared (search the file for `"gemma"` to find
the right spot).

Then extend `detectChatTemplate` (around line 25). Add a branch for
the format_parameters macro signature **before** the `<start_of_turn>`
branch (so Gemma 4 templates aren't mis-classified as plain Gemma):

```typescript
if (template.includes("{% macro format_parameters")
	|| template.includes("{%- macro format_parameters")) {
	return "gemma4";
}
if (template.includes("<start_of_turn>")) return "gemma";
```

Finally, export `formatGemma4` and add it to the named-export list at
the bottom of the file (mirror how `formatGemma` is exported).

- [ ] **Step 5: Run tests**

Run: `bun test tests/inference/chat-template-gemma4.test.ts -v`
Expected: 5 PASS.

- [ ] **Step 6: Run full suite**

Run: `make checkall`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/inference/chat-template.ts tests/inference/chat-template-gemma4.test.ts
git commit -m "feat(chat-template): Gemma 4 formatter + family detection

Adds formatGemma4 (same body as formatGemma for plain chat; reserved
for future tool-call divergence per Gemma 4's PEG-parsed format) and
extends detectChatTemplate to return 'gemma4' when the template
contains the format_parameters macro. Plain-text Gemma 2 templates
still resolve to 'gemma' for backward compat."
```

---

## Task 2.3: Gemma 4 stop-token registration in engine.ts

**Files:**
- Modify: `src/core/engine.ts:605` (the existing `else if (tmpl === "gemma")` block)
- Modify: `src/core/engine.ts:1060` (the second matching block)

- [ ] **Step 1: Inspect current Gemma stop-token logic**

Run: `Read src/core/engine.ts offset=600 limit=15` and `Read src/core/engine.ts offset=1055 limit=15`

Confirm both blocks add `<end_of_turn>` (token id 107) as a stop token when `tmpl === "gemma"`.

- [ ] **Step 2: Extend both branches**

Change each occurrence of:
```typescript
} else if (tmpl === "gemma") {
```
to:
```typescript
} else if (tmpl === "gemma" || tmpl === "gemma4") {
```

Both Gemma 2 and Gemma 4 use the same `<end_of_turn>` (id 107)
sentinel. Gemma 4's tokenizer also keeps the older `<eos>` /
`<end_of_text>` (id 1) as a generic stop token; Gemma 2's engine
path already covers this, so no separate Gemma 4 branch is needed
for the basic case.

- [ ] **Step 3: Run checkall**

Run: `make checkall`
Expected: green. (No new tests — this is a one-character widening of two existing branches; coverage comes from runtime smoke in Task 2.7.)

- [ ] **Step 4: Commit**

```bash
git add src/core/engine.ts
git commit -m "feat(engine): widen Gemma stop-token branches to include gemma4

Both engine.ts:605 and :1060 'gemma' template branches accept the
new 'gemma4' family detector value. <end_of_turn> id 107 is the
same sentinel; no per-version divergence yet."
```

---

## Task 2.4: `GEMMA4_DEFAULTS` sampling profile + dispatch

**Files:**
- Modify: `src/core/sampling-profiles.ts` (add `GEMMA4_DEFAULTS`, extend `SamplingMode`, extend `resolveSamplingParams` and any "auto" dispatch heuristic)

- [ ] **Step 1: Add `GEMMA4_DEFAULTS`**

In `src/core/sampling-profiles.ts`, after `MISTRAL_DEFAULTS`, add:

```typescript
/**
 * Gemma 4 default sampling. Google's reference inference pipeline uses
 * `temperature=1.0`, `top_p=0.95`, `top_k=64` for instruction-tuned
 * Gemma 4 checkpoints. For benchmarking the project pins temperature 0
 * via the `--eval-temperature` override; runtime chat callers get the
 * higher-temperature spec defaults unless they explicitly override.
 */
export const GEMMA4_DEFAULTS = Object.freeze({
	temperature: 1.0,
	topK: 64,
	topP: 0.95,
	repetitionPenalty: 1.0,
} as const);
```

- [ ] **Step 2: Extend `SamplingMode` union**

Find the `SamplingMode` definition (around line 65) and add `"gemma4"`:

```typescript
export type SamplingMode =
	| "auto"
	| "qwen-thinking"
	| "qwen-default"
	| "phi3"
	| "mistral"
	| "gemma4"
	| "raw";
```

- [ ] **Step 3: Extend `SamplingResolutionInput` and `resolveSamplingParams`**

Search for `isPhi3` in the same file. Add a parallel `isGemma4` boolean
to `SamplingResolutionInput` (after `isPhi3?` or `isMistralInstruct?`):

```typescript
	/**
	 * True when the loaded model is a Gemma 4 architecture (any size).
	 * Selects the Gemma 4 `"auto"` profile (T=1.0 / top_p=0.95 / top_k=64).
	 */
	isGemma4?: boolean;
```

Then in `resolveSamplingParams` (the dispatch ladder), add a branch
for Gemma 4 before the generic fallback. Look for the existing
Mistral-Instruct branch as the model — add a sibling:

```typescript
		if (input.isGemma4) {
			return { ...GEMMA4_DEFAULTS, ...override };
		}
```

The exact insertion point matches the existing `if (input.isMistralInstruct)`
or `if (input.isPhi3)` style. Make sure the explicit `"gemma4"` mode
short-circuits the `"auto"` ladder too — add a case to the explicit
mode switch.

- [ ] **Step 4: Wire `isGemma4` in the call site**

Search for `isMistralInstruct:` in the codebase (rg/grep). Each call
site must also populate `isGemma4`. Usually one call site in
`engine.ts` or `model-inference.ts`:

```typescript
const samplingInput: SamplingResolutionInput = {
	samplingMode: completion.sampling ?? "auto",
	isQwenChatml: hp.architecture === "qwen3" && /* ... */,
	isPhi3: hp.architecture === "phi3",
	isMistralInstruct: /* existing logic */,
	isGemma4: hp.architecture === "gemma4",
	// ...
};
```

Add the field at every call site.

- [ ] **Step 5: Add unit tests for the dispatch**

**Files:**
- Modify: `tests/sampling-dispatch.test.ts` (the existing file from the
  ts-api-audit 2026-04-30 work)

Add a new `describe` block:

```typescript
describe("Gemma 4 auto dispatch", () => {
	it("returns GEMMA4_DEFAULTS when isGemma4 is true and mode is auto", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: false,
			isPhi3: false,
			isGemma4: true,
		});
		expect(out.temperature).toBe(1.0);
		expect(out.topP).toBe(0.95);
		expect(out.topK).toBe(64);
		expect(out.repetitionPenalty).toBe(1.0);
	});

	it("explicit mode 'gemma4' selects the profile regardless of arch flags", () => {
		const out = resolveSamplingParams({
			samplingMode: "gemma4",
			isQwenChatml: true,
			isPhi3: false,
			isGemma4: false,
		});
		expect(out.temperature).toBe(1.0);
	});

	it("consumer override beats profile defaults", () => {
		const out = resolveSamplingParams({
			samplingMode: "auto",
			isQwenChatml: false,
			isGemma4: true,
			override: { temperature: 0 },
		});
		expect(out.temperature).toBe(0);
		expect(out.topP).toBe(0.95); // unchanged
	});
});
```

- [ ] **Step 6: Run the test**

Run: `bun test tests/sampling-dispatch.test.ts -v`
Expected: existing tests still pass, 3 new PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/sampling-profiles.ts tests/sampling-dispatch.test.ts \
        src/core/engine.ts src/inference/model-inference.ts
git commit -m "feat(sampling): GEMMA4_DEFAULTS profile + auto-dispatch

Adds Gemma 4's reference sampling (T=1.0/top_p=0.95/top_k=64) and
wires isGemma4 into resolveSamplingParams. Existing models unchanged.
Bench pass still overrides T=0 via --eval-temperature per the
greedy-by-default doctrine."
```

(Adjust files actually touched — at minimum `sampling-profiles.ts` +
the test; engine.ts / model-inference.ts only if they're the call
sites that needed the `isGemma4` field.)

---

## Task 2.5: Wire `final_logit_softcap` from GGUF into the dispatch

**Files:**
- Modify: `src/inference/model-inference.ts:1108` (the `0.0, // logit_softcap` constant)
- Confirm: `src/models/model-loader.ts` already reads `finalLogitSoftcap` (Task 1.3 added it)

- [ ] **Step 1: Re-read the existing call site**

Run: `Read src/inference/model-inference.ts offset=1100 limit=15`

Confirm the current literal `0.0` parameter passed as `logit_softcap`.

- [ ] **Step 2: Replace the literal with the hp read**

Change:
```typescript
0.0, // logit_softcap (Gemma; not used by Llama/Qwen/Mistral)
```
to:
```typescript
hp.finalLogitSoftcap ?? 0.0, // logit_softcap (Gemma family; 0 = disabled)
```

Search the file for `// logit_softcap` to ensure all sites (there may
be multiple) are updated consistently.

- [ ] **Step 3: Add a unit test**

**Files:**
- Modify: `tests/models/model-loader-gemma4-hparams.test.ts` (the Task 1.3 test file)

Already covered by the `expect(hp.finalLogitSoftcap).toBe(30);` assertion
in Task 1.3 — no additional unit test needed. The end-to-end correctness
of the softcap path is exercised by the Stage 3 semantic gate (when
softcap is off, Gemma generation diverges; when on, it matches reference).

- [ ] **Step 4: Run checkall**

Run: `make checkall`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/inference/model-inference.ts
git commit -m "feat(inference): plumb final_logit_softcap from hp into softmax call

The logit_softcap parameter has been carried through the wasm bridge
(webgpu-bridge.cpp:232) since the original Gemma 2 work; it was hard-
coded to 0.0 at the dispatch site because no shipping model populated
it. Gemma 4 E2B reports 30.0; Stage 1's model-loader now reads it,
this commit threads it through."
```

---

## Task 2.6: `eval/models.ts` registration entry for Gemma 4 E2B

**Files:**
- Modify: `eval/models.ts` — add a new entry near the existing Gemma 2 registration (around line 571)

- [ ] **Step 1: Find the Gemma 2 entry and the canonical place to insert**

Run: `grep -n "gemma-2-2b-q4f16" eval/models.ts`

Read 5 lines on each side for context.

- [ ] **Step 2: Insert the Gemma 4 entry**

After the closing `},` of the Gemma 2 entry, add:

```typescript
	{
		id: "gemma-4-e2b-it-q4km",
		name: "Gemma 4 E2B Instruct (Q4_K_M)",
		family: "Gemma 4",
		architecture: "gemma4",
		// 2.3B effective (PLE), 5.1B total parameter count. paramsB
		// reports the active-compute size that the rest of the fleet
		// uses for tier comparisons.
		paramsB: 2.3,
		// Q4_K_M weights file is 3.11 GB; round to 3110 MB. The PLE
		// table sits inside the same file; chunked binding dispatch
		// handles it (Stage 3 probe confirms exact tensor sizes).
		vramMB: 3110,
		defaultQuant: "q4km",
		availableQuants: ["q4km"],
		capabilities: {
			toolCalling: false, // deferred — Gemma 4 PEG tool format
			structuredOutput: false,
			vision: false, // mmproj weights ship separately
			embedding: false,
		},
		license: "Apache-2.0",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/google/gemma-4-E2B-it",
		ggufUrl: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
		ggufFilePattern: "Q4_K_M",
	},
```

- [ ] **Step 3: Regenerate the bundle**

Run: `bun build eval/models.ts --outfile smoke-test/webllm-models.js --target browser`

- [ ] **Step 4: Run checkall**

Run: `make checkall`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add eval/models.ts smoke-test/webllm-models.js
git commit -m "feat(models): register Gemma 4 E2B (Q4_K_M) in benchmark fleet

Adds gemma-4-e2b-it-q4km to BENCHMARK_MODELS — 2.3B effective,
3.11 GB Q4_K_M, Apache-2.0 (the licensing shift from earlier Gemma
versions removes commercial-use friction). Tool calling deferred
(Gemma 4 uses a custom PEG-parsed format; project parser doesn't
emit that yet). Bundle regenerated for smoke harness consumption."
```

---

## Task 2.7: Stage 2 smoke probe — coherent ASCII generation gate

**Goal:** Run the canonical smoke probe and confirm: (a) load succeeds, (b) greedy decode emits 5 tokens of coherent ASCII, (c) multi-turn stops cleanly on `<end_of_turn>`.

- [ ] **Step 1: Restart smoke server**

Run: `make smoke-restart`

- [ ] **Step 2: Navigate via agentchrome**

```bash
TAB_ID=$(agentchrome --port 63846 tabs list | head -1 | grep -oE '"id":"[A-F0-9]+"' | head -1 | cut -d'"' -f4)

agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&v=stage2-1" \
  --tab "$TAB_ID" --timeout 60000
```

- [ ] **Step 3: Wait for load completion, snapshot**

Wait ~30 s, then:
```bash
agentchrome --port 63846 page snapshot --tab "$TAB_ID"
```

Look for the lines `[7/8] First token:` and subsequent `[7/8] Token N: ...`. The expected outcome is:
- Steps `[1/8]` through `[6/8]` complete cleanly
- Step `[7/8]` produces at least 5 generated tokens
- Each token is in the ASCII range (no control characters, no Unicode escape sequences indicating garbage tensors)
- Generation stops within ~60 tokens (no runaway)

**Acceptable Stage 2 output examples:**
- ` the` ` city` ` of` ` Paris` `.` (semantically right — bonus!)
- ` a` ` city` ` in` ` Europe` `.` (semantically plausible)
- ` the` ` answer` ` to` ` the` ` question` (drifting but coherent)

**Failing Stage 2 output examples:**
- Garbage Unicode / control chars → tensor corruption
- 60+ tokens with no `<end_of_turn>` → stop-token wire-up broken
- Empty / single-token stop → tokenizer or sampler broken

Stage 2 does **not** require semantic correctness. Stage 3 fixes that.

- [ ] **Step 4: Multi-turn stop probe**

Navigate to the chat page:
```bash
agentchrome --port 63846 navigate \
  "http://localhost:8031/chat.html?model=gemma-4-e2b-it-q4km&v=stage2-2" \
  --tab "$TAB_ID" --timeout 60000
```

Submit two messages via the chat UI (e.g., "Hi" then "How are you?"). Each model turn must terminate cleanly without runaway into the next user prompt.

(If chat.html doesn't accept a `?model=` query param the way real-model.html does, use the dropdown to pick `gemma-4-e2b-it-q4km` — the registration from Task 2.6 makes it appear there.)

- [ ] **Step 5: Capture Stage 2 closure report**

**Files:**
- Create: `eval/reports/gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md`

```markdown
# Gemma 4 Stage 2 — Surface wiring closure

**Date:** 2026-05-10
**Spec:** docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md
**Plan:** docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md
**Commits:** <list>

## Build gate
`make checkall` — green.

## Runtime gates

### Greedy decode, 5 tokens
Prompt: `"The capital of France is"`
Output: `<paste actual tokens>`

Assessment: coherent ASCII / not garbage / not runaway. ✅

### Multi-turn stop
Two chat turns; both terminated on `<end_of_turn>` (id 107) cleanly.
No runaway into the next user prompt. ✅

## Known limitations
- Semantic correctness not yet tested (Stage 3 gate)
- Tool calling unsupported (Gemma 4 PEG format not implemented)
- All layers still using global attention (Stage 4 gate)
- KV cache materializes shared layers (Stage 5 gate)

## Follow-ups
- Stage 3: PLE injection + dual RoPE for semantic correctness
```

- [ ] **Step 6: Commit closure**

```bash
git add eval/reports/gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md
git commit -m "docs(reports): Gemma 4 Stage 2 surface wiring closure"
```

---

# Stage 3 — PLE injection + dual RoPE dispatch

**Goal:** Add Per-Layer Embeddings (PLE) injection at each layer's start and dispatch dual-RoPE (per-layer `rope_dim` + `freq_base`). After this stage, the model should generate semantically meaningful output (the canonical `"The capital of France is" → Paris` gate).

**Stage 3 ship gate (run all):**

1. `make checkall` — green
2. Greedy decode emits `Paris` (or ` Paris`) as the first generated token on `"The capital of France is"`
3. 36-prompt eval ≥40% (greedy @ temp=0, via `make bench-inference PERF_MODEL=gemma-4-e2b-it-q4km --eval-temperature 0`)

## Task 3.1: PLE tensor presence + size probe

**Files:**
- Create: `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md`

- [ ] **Step 1: Inspect the GGUF tensor list**

```bash
cd ~/Repos/llama.cpp/gguf-py
uv run --no-project --with numpy --with pyyaml --with sentencepiece \
python -c "
import sys; sys.path.insert(0, '.')
from gguf import GGUFReader
r = GGUFReader(
    '/Users/probello/Repos/webllm/smoke-test/models/gemma-4-e2b-it-q4km.gguf',
    'r',
)
for t in r.tensors:
    if 'per_layer' in t.name or 'embd' in t.name:
        print(f'{t.name}: shape={list(t.shape)} type={t.tensor_type.name} '
              f'size_bytes={t.n_bytes}')
"
```

- [ ] **Step 2: Capture the probe results**

Record in `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md`:

```markdown
# Stage 3 pre-implementation probe — PLE table sizing

## Tensor list (per_layer / embd hits)
<paste output from Step 1>

## Per-layer slice (one layer's `per_layer_token_embd`):
- shape: [262144, 256] (per spec; confirm via probe)
- ggml_type: <Q4_K | Q8_0 | F16>
- size_bytes per layer: <X>
- total across 35 layers: <Y>

## Binding-cap assessment
Per-binding 128 MiB cap. If a single PLE slice exceeds 128 MiB:
- At Q4_K (4 bits per element): 262144 × 256 × 0.5 = 33.5 MiB ✓ fits
- At Q8_0 (1 byte per element): 262144 × 256 × 1 = 67 MiB ✓ fits
- At F16 (2 bytes per element): 262144 × 256 × 2 = 134 MiB ✗ exceeds

**Decision:** ship PLE in whatever quant GGUF stores it as (no
requant needed); chunked binding dispatch handles per-layer
binding if needed.

## Total memory cost
<Y bytes> ≈ <Z GB> GPU-resident. Adds to the existing 3.11 GB Q4_K_M
weight footprint.
```

- [ ] **Step 3: Commit the probe**

```bash
git add eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md
git commit -m "docs(reports): Stage 3 PLE pre-impl probe — tensor sizing"
```

---

## Task 3.2: Load PLE table in `model-loader`

**Files:**
- Modify: `src/models/model-loader.ts` — extend the tensor-loading section to surface `per_layer_token_embd.weight` (or its concrete name from the probe)
- Modify: `src/core/types.ts` — extend `ModelWeights` (or the equivalent loaded-weight metadata interface) with a `perLayerEmbed?: TensorHandle[]` field (per-layer, since each layer has its own slice)

- [ ] **Step 1: Inspect existing weight loading**

Run: `grep -n "token_embd.weight\|tensorMap\b" src/models/model-loader.ts | head -10`

Find how `token_embd.weight` is loaded into `tensorMap` and exposed.

- [ ] **Step 2: Add per-layer-embed handle to weights**

In `src/core/types.ts`, extend the relevant weights interface (search for `ModelWeights` or `LoadedTensors`):

```typescript
	/**
	 * Per-Layer Embeddings table (Gemma 4 only). Length === `layerCount`.
	 * Each handle is the slice `per_layer_token_embd.weight[layer, :, :]`
	 * shaped `[vocab, ple_dim]` where `ple_dim` is read from GGUF
	 * `gemma4.embedding_length_per_layer_input` (Gemma 4 E2B: 256).
	 * Undefined for architectures without PLE.
	 */
	perLayerEmbed?: import("./tensor-handle.js").TensorHandle[];
```

In `src/models/model-loader.ts`, in the tensor-mapping pass, slice the
GGUF `per_layer_token_embd.weight` tensor per layer. Two paths:
1. **Whole table as one tensor**, then strided views per layer — cheaper.
2. **Per-layer slices upfront** as separate `TensorHandle`s — clearer.

Use path 1 if the existing `opView3d` / `opView2d` helpers exist
(they do — `model-inference.ts` uses them) and the underlying GGUF
storage allows it. Otherwise path 2.

Concretely, mirror how `token_embd.weight` is loaded but extend
`tensorMap` with a `perLayerEmbed` array:

```typescript
		const perLayerEmbedTensor = tensorMap.get("per_layer_token_embd.weight");
		const perLayerEmbed: TensorHandle[] | undefined = perLayerEmbedTensor
			? Array.from({ length: hp.layerCount }, (_, layer) =>
					wasm.opView2d(
						perLayerEmbedTensor,
						/* ne0 */ hp.vocabularySize,
						/* ne1 */ pleDim,
						/* nb1 */ pleDim * elemSize(perLayerEmbedTensor.type),
						/* offset */ layer * hp.vocabularySize * pleDim *
							elemSize(perLayerEmbedTensor.type),
					),
			)
			: undefined;
```

(The exact API names need to match what's actually in `ggml-wasm.ts` —
adapt accordingly; the idea is "one tensor on disk, per-layer view at
TS layer".)

Also read `pleDim` from GGUF:
```typescript
const pleDim = getMetaNumber(
	ctx,
	`${metaPrefix}.embedding_length_per_layer_input`,
	0,
);
```

(0 → no PLE for this arch.)

Expose `perLayerEmbed` on the weights/handle the engine returns.

- [ ] **Step 3: Add a loader test**

**Files:**
- Modify: `tests/models/model-loader-gemma4-hparams.test.ts` (extend the existing file from Task 1.3)

Add a new `it` block inside the same `describe.skipIf`:

```typescript
		it("loads per_layer_token_embd as 35 per-layer slices", async () => {
			const buf = readFileSync(GEMMA4);
			const parsed = await ModelLoader.parse(buf.buffer);
			expect(parsed.weights.perLayerEmbed).toBeDefined();
			expect(parsed.weights.perLayerEmbed).toHaveLength(35);
			// Each slice should have shape [vocab, ple_dim].
			// Slice 0 differs from slice 1 (per-layer embeddings).
			// (Tensor-equality probe done in the dispatch test, not here.)
		});
```

- [ ] **Step 4: Run tests**

Run: `bun test tests/models/model-loader-gemma4-hparams.test.ts -v`
Expected: 1 new PASS (plus existing ones still pass).

- [ ] **Step 5: Commit**

```bash
git add src/models/model-loader.ts src/core/types.ts \
        tests/models/model-loader-gemma4-hparams.test.ts
git commit -m "feat(model-loader): expose Gemma 4 per_layer_token_embd as per-layer slices

Adds perLayerEmbed?: TensorHandle[] to the loaded-weights surface.
Loader strides per_layer_token_embd.weight into layerCount views
[vocab, ple_dim]. Architectures without PLE leave the field
undefined."
```

---

## Task 3.3: Inject PLE into residual stream at each layer start

**Files:**
- Modify: `src/inference/model-inference.ts` — add a PLE-add step inside the per-layer forward loop

- [ ] **Step 1: Locate the per-layer loop**

Run: `grep -n "for.*layer\|layers\[i\]\|hp.layerCount" src/inference/model-inference.ts | head -10`

Find the `forwardSingle` (and `forwardPrefill` if separate) per-layer loop.

- [ ] **Step 2: Add the PLE-add op**

At the start of each layer iteration, after the embedding lookup but
before the attention norm, add (gated on `weights.perLayerEmbed`
existence):

```typescript
		if (this.weights.perLayerEmbed && weights.perLayerEmbed[layer]) {
			// PLE: lookup per_layer_embed[layer, token_id, :] and add into
			// the residual. PLE table is [vocab, ple_dim] per layer;
			// embedded into the residual at offset 0 of the hidden dim.
			const pleSlice = weights.perLayerEmbed[layer];
			const pleLookup = wasm.opGetRows(pleSlice, tokenIds);
			// pleLookup: [ple_dim, n_tokens]
			// Add into the residual's first ple_dim channels:
			const residualHead = wasm.opView2d(
				residual,
				pleDim,
				nTokens,
				/* nb1 */ hp.embeddingLength * elemBytes,
				/* offset */ 0,
			);
			residual = wasm.opAddInplace(residualHead, pleLookup);
			// Note: depending on the GGML graph shape, opAddInplace might
			// need to be a regular opAdd that writes to a new tensor.
			// Match the existing pattern used in the file for other
			// per-layer additions.
		}
```

(The exact GGML op names and view construction must match what's in
`ggml-wasm.ts`. The intent is clear: for each token, look up
`per_layer_embed[layer, token_id, :]` and add it into the residual's
first `ple_dim` channels.)

- [ ] **Step 3: Verify graph builds for Gemma 4**

Run: `bun run typecheck`
Expected: PASS.

Smoke load the model:
```bash
agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&v=stage3-1" \
  --tab "$TAB_ID" --timeout 60000
```

Wait 30 s, snapshot. Confirm `[1-7]/8` steps still complete (load
succeeds even if generation is now wrong because Q reshape still
fails on the per-layer head_dim mismatch — Task 3.4 fixes that).

- [ ] **Step 4: Commit**

```bash
git add src/inference/model-inference.ts
git commit -m "feat(inference): inject Per-Layer Embeddings at each layer start (Gemma 4)

Adds a per-layer PLE-lookup + residual-add gated on weights.perLayerEmbed
being populated. For Gemma 4 E2B, each layer's first 256 channels of
the residual receive an additive per-(layer, token) embedding. Other
architectures (no perLayerEmbed) skip the block."
```

---

## Task 3.4: Dual RoPE dispatch (per-layer dim + freq_base)

**Files:**
- Modify: `src/inference/model-inference.ts` — RoPE call site inside `buildQKV` (or wherever RoPE is applied per-layer)

- [ ] **Step 1: Locate the existing RoPE call**

Run: `grep -n "opRope\|rope_freq_base\|ropeFreqBase\|opRotaryEmbed" src/inference/model-inference.ts | head -10`

Find the RoPE op call inside the per-layer attention build.

- [ ] **Step 2: Switch to per-layer values when present**

The existing call likely passes `hp.ropeFreqBase` as a scalar. Change
to:

```typescript
const ropeFreqBase = hp.ropeFreqBasePerLayer
	? hp.ropeFreqBasePerLayer[layer]
	: hp.ropeFreqBase;
const ropeDimCount = hp.ropeDimensionCountPerLayer
	? hp.ropeDimensionCountPerLayer[layer]
	: undefined; // existing default behavior
const headDim = hp.embeddingHeadLengthPerLayer
	? hp.embeddingHeadLengthPerLayer[layer]
	: hp.embeddingHeadLength;
```

Pass `headDim`, `ropeDimCount`, and `ropeFreqBase` into the RoPE op
+ the Q/K reshape lines. The `headDim` substitution is the critical
fix for the original `buildQKV` reshape3d assertion.

Search the file for **every** `hp.embeddingHeadLength` site within
the per-layer hot loop and replace with the per-layer `headDim` local.
The grep at Task 1.4 listed sites — review each one:

- Lines around 934, 983–1023: replace with per-layer `headDim`.
- Line 2520, 2573, 2576, 2582, 2589, 2602, 2616, 2640, 2652, 2665,
  2675, 2685, 2699, 2705 (the speculative-decode / draft path) —
  check whether these are also per-layer or operate on a different
  abstraction. **For Stage 3, only fix the chat-forward path**;
  defer spec-decode paths to a Stage 6 follow-up if they regress on
  Gemma 4. If they're shared, just substitute `headDim` everywhere.

- [ ] **Step 3: Smoke probe — semantic gate**

Restart smoke server (not strictly necessary, but clears state):
```bash
make smoke-restart
```

Navigate:
```bash
agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&v=stage3-2" \
  --tab "$TAB_ID" --timeout 60000
```

Wait 30 s; snapshot. Look for the generated token sequence on
`"The capital of France is"`.

**Pass:** first generated token is `Paris` or ` Paris` (token id ~9097
or 21181 depending on Gemma 4 tokenizer; check via `encode("Paris")`
in the page log).

**Fail (debugging):**
- Garbage Unicode → tensor still misshapen; re-check per-layer head_dim
  substitution; you may have missed a site.
- Coherent but wrong (e.g., ` a city`) → PLE not injected; check
  Task 3.3 graph integration.
- Crash at attention compute → mask or KV shape mismatch from the
  per-layer head_dim change; verify KV cache is allocated with
  `max(global, swa)` head_dim, not `hp.embeddingHeadLength`.

- [ ] **Step 4: 36-prompt eval gate**

Run:
```bash
make bench-inference PERF_MODEL=gemma-4-e2b-it-q4km \
  EVAL_TEMPERATURE=0 PERF_RUNS=1
```

(If `EVAL_TEMPERATURE` isn't a Make var, use the CLI form:
`bun run eval/bench.ts --model gemma-4-e2b-it-q4km --eval-temperature 0`)

Wait for the run to finish (~5–15 min depending on speed).

**Pass gate:** overall score ≥ 0.40 (40%). Per-dimension breakdown
not required at this stage; loose gate accounts for SWA still being
disabled.

- [ ] **Step 5: Commit + closure**

```bash
git add src/inference/model-inference.ts
git commit -m "feat(inference): Gemma 4 dual-RoPE + per-layer head_dim dispatch

Reads hp.embeddingHeadLengthPerLayer, hp.ropeDimensionCountPerLayer,
hp.ropeFreqBasePerLayer when populated and passes per-layer values
into the attention build. Fixes the buildQKV reshape3d assertion
that crashed the Phase 1 probe. With Task 3.3's PLE injection, this
is the load-bearing combo that lifts Gemma 4 generation from
'garbage Unicode' to semantically meaningful output."
```

Create the Stage 3 closure report:

**Files:**
- Create: `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/SUMMARY.md`

```markdown
# Gemma 4 Stage 3 — PLE injection + dual RoPE closure

**Date:** 2026-05-10
**Probe:** ./PROBE.md
**Commits:** <list>

## Build gate
`make checkall` — green.

## Runtime gates

### Semantic-correctness probe
Prompt: `"The capital of France is"`
Greedy first 5 tokens: `<paste actual>`
First-token gate (`Paris` or ` Paris`): <PASS/FAIL>

### 36-prompt eval
Overall score: <X.XX>
Gate (≥0.40): <PASS/FAIL>
Per-dimension: <summary>

## Known limitations
- All layers still using global causal attention (Stage 4 gate);
  expect SWA-trained behavior to suffer at long context.
- KV cache materializes shared layers (Stage 5 gate); ~3 GB wasted.

## Follow-ups
- Stage 4: real SWA windowing — expected eval lift to ≥60%.
```

```bash
git add eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/SUMMARY.md
git commit -m "docs(reports): Gemma 4 Stage 3 PLE+dualRoPE closure"
```

---

# Stage 4 — Real sliding-window attention

**Goal:** Replace Stage 3's all-global attention fallback with real SWA on the 4-of-5 layers marked SWA in `hp.slidingWindowPattern`. Window size 512.

**Stage 4 ship gate (run all):**

1. `make checkall` — green
2. 36-prompt eval ≥0.60 (Phi-3 closure standard; lift from Stage 3's 0.40)
3. Long-context smoke: generate 1000 tokens with no quality cliff at the 512-token window boundary

## Task 4.1: Pre-implementation probe — windowed-mask shape supported by ggml-webgpu?

- [ ] **Step 1: Inspect the existing mask construction**

Run: `grep -n "opSoftMaxExt\|softMaxExt\|attention_mask\|opCausalMask" src/inference/model-inference.ts | head -10`

Find where the causal mask is built. Determine whether the existing
ggml-webgpu `softmax_with_mask` (or equivalent) op takes a generic
mask tensor or a fused "causal" flag.

- [ ] **Step 2: Construct a synthetic windowed-mask test**

**Files:**
- Create: `tests/inference/swa-mask-shape.test.ts`

```typescript
import { describe, it, expect } from "bun:test";
import { GgmlWasm } from "../../src/inference/ggml-wasm.js";
import { existsSync } from "node:fs";

const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	"gpu" in navigator;

describe.skipIf(!HAS_WEBGPU)("SWA mask shape probe", () => {
	it("computes softmax with a windowed mask without dispatch errors", async () => {
		// Build a minimal graph: [seq_len, seq_len] softmax with a mask
		// that's causal AND windowed to 512 tokens.
		const wasm = await GgmlWasm.create();
		const seqLen = 600;
		const window = 512;
		const mask = new Float32Array(seqLen * seqLen);
		for (let q = 0; q < seqLen; q++) {
			for (let k = 0; k < seqLen; k++) {
				const causal = k <= q;
				const inWindow = q - k < window;
				mask[q * seqLen + k] = causal && inWindow ? 0 : -Infinity;
			}
		}
		const scores = new Float32Array(seqLen * seqLen).fill(0.5);
		// ... build graph, dispatch softmax_with_mask, read back.
		// If this throws or returns NaN/Inf rows for the SWA pattern,
		// the mask shape is unsupported and Stage 4 needs a llama.cpp
		// patch.
	});
});
```

(This test is illustrative; the exact `GgmlWasm` API depends on what's
already exposed. The point of the probe is: can ggml-webgpu accept an
arbitrary [seq_len, seq_len] f32 mask without complaining about its
shape or structure? If yes, no patch needed. If no, patch needed.)

- [ ] **Step 3: Run the probe**

Run: `bun test tests/inference/swa-mask-shape.test.ts -v`

Record the result.

- [ ] **Step 4: Document and decide**

**Files:**
- Create: `eval/reports/gemma-4-stage4-swa-2026-05-10/PROBE.md`

```markdown
# Stage 4 SWA mask shape probe

## Probe result
<output of the test from Step 3>

## Decision
- If passed: Stage 4 implementation needs only TS-layer mask
  construction (no llama.cpp patch).
- If failed: scope a llama.cpp patch in `~/Repos/llama.cpp` on the
  `webllm-browser-patches` branch to extend the softmax-mask op
  to accept a windowed-mask shape. Budget +1 patch on the stack
  (currently 9; cap +2 max for this campaign).
```

Commit:
```bash
git add tests/inference/swa-mask-shape.test.ts \
        eval/reports/gemma-4-stage4-swa-2026-05-10/PROBE.md
git commit -m "feat(test): SWA mask shape probe for Stage 4 decision

Pre-implementation probe to determine whether ggml-webgpu's
softmax-with-mask op can express a 512-token sliding window
without a llama.cpp patch. Result documented in PROBE.md."
```

---

## Task 4.2: SWA mask + per-layer dispatch

**Files:**
- Modify: `src/inference/model-inference.ts` — per-layer mask construction inside the attention block

- [ ] **Step 1: Build per-layer mask**

Inside the per-layer loop, before the softmax-with-mask op, construct
the mask based on `hp.slidingWindowPattern[layer]`:

```typescript
const isSwaLayer =
	hp.slidingWindowPattern?.[layer] === true;
const window = hp.slidingWindowSize ?? Infinity;

// Existing causal mask construction:
const baseMask = this.buildCausalMask(nTokens, kvLen);

// SWA layers narrow the mask to the last `window` tokens of each
// query's attention range:
const mask = isSwaLayer
	? this.applyWindow(baseMask, window)
	: baseMask;
```

Where `applyWindow` zeros out (sets to -Inf) any (q, k) pair where
`q - k >= window`. The implementation depends on whether masks are
constructed in TS and passed to WASM as tensors, or built inside
the WASM graph — match the existing pattern.

- [ ] **Step 2: KV cache window read for SWA layers**

For correctness-first, the KV cache **writes** stay full-size (waste
memory but mechanically simple); only the **read** is windowed via
the mask. This keeps the cache allocator unchanged for Stage 4.

Confirm in the existing code: K/V cache reads use a single mask-multiplied
softmax, so windowing the mask is sufficient — no separate KV-window
indexing needed.

- [ ] **Step 3: Smoke-load + first-token probe**

Restart and reload:
```bash
make smoke-restart
agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&v=stage4-1" \
  --tab "$TAB_ID" --timeout 60000
```

Wait 30 s, snapshot. Confirm load succeeds and a coherent output
emerges. The first-token "Paris" gate from Stage 3 must continue
to hold (SWA windowing applied at short ctx ≤ 512 is identical to
full attention).

- [ ] **Step 4: 36-prompt eval (≥0.60 gate)**

Run:
```bash
bun run eval/bench.ts --model gemma-4-e2b-it-q4km \
  --eval-temperature 0
```

**Pass gate:** overall score ≥ 0.60 (lift from Stage 3's ≥0.40 baseline).

- [ ] **Step 5: Long-context smoke**

Use the smoke harness with a long prompt or generation request:
```bash
agentchrome --port 63846 navigate \
  "http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&v=stage4-2&maxTokens=1000" \
  --tab "$TAB_ID" --timeout 120000
```

(If `?maxTokens=` isn't a supported param, hand-load the chat page
and request 1000-token continuation.)

Look for quality cliff signatures at token ~512: word repetition,
loss of topical coherence, switch to garbage. **Pass:** no such
cliff; coherence consistent across the 0–1000 range.

- [ ] **Step 6: Commit + Stage 4 closure**

```bash
git add src/inference/model-inference.ts
git commit -m "feat(inference): Gemma 4 sliding-window attention per layer

Reads hp.slidingWindowPattern and applies a 512-token window mask to
SWA layers (4 of every 5 in Gemma 4's 5:1 local:global cycle). KV
cache writes stay full-size for Stage 4 (correctness-first); Stage 5
adds ref-sharing for the 20 trailing shared-KV layers."
```

**Files:**
- Create: `eval/reports/gemma-4-stage4-swa-2026-05-10/SUMMARY.md`

```markdown
# Gemma 4 Stage 4 — Real SWA closure

**Date:** 2026-05-10
**Probe:** ./PROBE.md
**Commits:** <list>

## Build gate
`make checkall` — green. Patch stack: <9 or 10 depending on probe outcome>.

## Runtime gates

### Semantic regression
First token on `"The capital of France is"`: `<token>` (must stay = Paris or ` Paris`).

### 36-prompt eval
Stage 3 baseline: <X.XX>
Stage 4: <Y.YY>
Lift: <Z>%
Gate (≥0.60): <PASS/FAIL>

### Long-context smoke
1000-token generation; quality assessment at 0/500/750/1000 token marks.
No quality cliff at the 512-token SWA window boundary: <PASS/FAIL>

## Follow-ups
- Stage 5: ref-shared KV (drop ~3 GB VRAM); bench profile run; closure.
```

```bash
git add eval/reports/gemma-4-stage4-swa-2026-05-10/SUMMARY.md
git commit -m "docs(reports): Gemma 4 Stage 4 SWA closure"
```

---

# Stage 5 — Shared-KV ref-sharing + bench + closure

**Goal:** Wire the last 20-of-35 layers' shared K/V references through the KV-cache allocator. Run `smoke-bench` profile mode. Write the campaign closure report.

**Stage 5 ship gate (run all):**

1. `make checkall` — green
2. VRAM drops ≥2 GB vs Stage 4 baseline (verifies ref-sharing landed)
3. `make smoke-bench PERF_MODEL=gemma-4-e2b-it-q4km PERF_RUNS=3` reports median ≥10 tok/s
4. 36-prompt eval ≥ Stage 4 (target ≥0.60; ideally higher with proper KV semantics)
5. Closure report under `eval/reports/gemma-4-e2b-validation-2026-05-10/`

## Task 5.1: KV-cache allocator ref-shared layer support

**Files:**
- Modify: `src/models/kv-cache.ts`

- [ ] **Step 1: Inspect existing layer allocation**

Run: `Read src/models/kv-cache.ts` (read the full file; it's small enough)

Find the per-layer K/V buffer allocation. Identify where layers are
treated as independent.

- [ ] **Step 2: Add a ref-sharing mode**

Add an optional `sharedKvLayers: number` parameter to the cache
allocator. When > 0, the *last* `sharedKvLayers` layers' K/V handles
point at earlier layers' allocated buffers.

The reference mapping for Gemma 4 E2B (35 layers, sharedKvLayers=20):
- Layers 0–14 allocate their own K/V (15 independent)
- Layers 15–34 share — but which earlier layer? The mapping is
  determined by the GGUF; in the typical Gemma 4 pattern,
  layer 15 shares with layer N where N is the closest earlier
  layer of the same SWA/global type. **Probe this**: read the
  GGUF tensor list and check whether `blk.15.attn_k.weight` is
  present or absent. If absent, layer 15 references the next earlier
  global layer's attn_k (it's a shared-KV layer).

A simpler mapping (and what upstream llama.cpp does per PR #21739):
each shared layer's `attn_k` / `attn_v` tensor is **literally
missing** from the GGUF; the loader detects absence and falls back
to "share with the closest earlier same-type layer". So the cache
allocator doesn't need explicit mapping — it just allocates lazily
when the loader actually has weights for that layer.

```typescript
export class KvCacheAllocator {
	// ... existing fields

	/**
	 * Per-layer K/V handles. `null` for ref-shared layers; consumers
	 * MUST resolve to the closest earlier non-null layer's handles.
	 */
	private kHandles: (TensorHandle | null)[] = [];
	private vHandles: (TensorHandle | null)[] = [];

	allocateLayer(layer: number, isShared: boolean): void {
		if (isShared) {
			this.kHandles[layer] = null;
			this.vHandles[layer] = null;
		} else {
			this.kHandles[layer] = /* allocate K buffer */;
			this.vHandles[layer] = /* allocate V buffer */;
		}
	}

	resolveK(layer: number): TensorHandle {
		for (let i = layer; i >= 0; i--) {
			if (this.kHandles[i]) return this.kHandles[i]!;
		}
		throw new Error(`No K cache available for layer ${layer}`);
	}
	// resolveV similarly
}
```

- [ ] **Step 3: Wire in `model-loader.ts`**

In the per-layer weight loading section, when an `attn_k.weight` /
`attn_v.weight` tensor is absent for a layer, call
`kvCache.allocateLayer(layer, isShared=true)`. Otherwise
`isShared=false`.

- [ ] **Step 4: Wire reads in `model-inference.ts`**

Find every site that does `kvCache.kAtLayer(layer)` (or equivalent)
and replace with `kvCache.resolveK(layer)`. Same for V.

- [ ] **Step 5: Smoke-load + memory probe**

Restart smoke server. Load gemma-4-e2b-it-q4km. Look at the page log
for the VRAM/memory line ("[5/8] KV cache: 4096 slots × 35 layers").

Stage 4 KV bytes (full materialization):
- 35 layers × (256 SWA + 512 global head_dim weighted by pattern)
  × 4096 ctx × KV elem size × 2 (K+V)
- ≈ 35 × ~308 × 4096 × 2 (F16) × 2 ≈ 363 MB
- (Adjust for actual numbers; the spec said "~3 GB wasted" — re-derive
  against the actual cache config.)

Stage 5 KV bytes (ref-shared):
- 15 layers allocate; 20 reuse
- ≈ 15/35 × (Stage 4) ≈ 156 MB
- Drop ≈ 207 MB (much less than the spec's 3 GB estimate, which was
  scoped to broader memory footprint).

**Pass gate (revised):** measurable VRAM drop in the page log after
the change vs before. If the drop is <30 % of total cache size, the
ref-sharing isn't landing — re-verify the loader detection logic.

- [ ] **Step 6: Persistence test**

Run: `bun test tests/persistence/ -v` (or whichever path covers
`engine-conversation-persistence.test.ts`)

Expected: existing tests still pass; new skips (if any) flagged.
**Required:** no new failure modes; the snapshot/load tests that
were already skip-gated stay skip-gated.

- [ ] **Step 7: Commit**

```bash
git add src/models/kv-cache.ts src/models/model-loader.ts \
        src/inference/model-inference.ts
git commit -m "feat(kv-cache): Gemma 4 ref-shared KV layers

Last 20 of 35 layers in Gemma 4 E2B don't have their own attn_k /
attn_v tensors; the cache allocator now skips alloc for those
layers and resolveK/resolveV walks back to the closest earlier
allocated layer. Drops KV-cache memory roughly proportional to
the shared/total ratio."
```

---

## Task 5.2: indexeddb-store ref-shared serialization

**Files:**
- Modify: `src/persistence/indexeddb-store.ts`

- [ ] **Step 1: Audit the snapshot path**

Run: `grep -n "kvCache\|serializeKV\|snapshot" src/persistence/indexeddb-store.ts | head -10`

Confirm the serialization assumes "every layer has K + V tensors".

- [ ] **Step 2: Encode the ref-sharing graph**

Serialize K/V per layer. For shared layers, emit a small "this layer
shares K from layer N, V from layer N" marker instead of duplicating
the buffer.

```typescript
interface SerializedKvLayer {
	kind: "own";
	k: Uint8Array;
	v: Uint8Array;
}
interface SerializedKvShared {
	kind: "shared";
	sourceLayer: number; // earlier layer whose K/V this references
}
```

- [ ] **Step 3: Reconstruct on load**

In the import path, when a `kind: "shared"` marker is encountered,
do not allocate a new buffer; instead let `kvCache.resolveK(layer)`
walk back through the sharing graph.

- [ ] **Step 4: Re-run persistence tests**

Run: `bun test tests/persistence/ -v`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add src/persistence/indexeddb-store.ts
git commit -m "feat(persistence): serialize ref-shared Gemma 4 KV layers

Adds a 'shared' marker so the snapshot/load round-trip preserves the
ref-sharing graph instead of materializing duplicate buffers on each
load."
```

---

## Task 5.3: `smoke-bench` profile pass

- [ ] **Step 1: Pre-bench memory drop verify**

Already covered in Task 5.1 Step 5; just re-confirm before the bench:
the page log on a fresh load shows the post-Task-5.1 KV memory figure.

- [ ] **Step 2: Run smoke-bench**

Run:
```bash
make smoke-bench PERF_MODEL=gemma-4-e2b-it-q4km PERF_RUNS=3
```

**Pass gate:** median ≥ 10 tok/s. Realistic E2B median is probably
20–40 tok/s; 10 is the loose floor that catches catastrophic
regression.

- [ ] **Step 3: Capture results**

Save the bench output to `eval/reports/gemma-4-e2b-validation-2026-05-10/smoke-bench.txt`. Extract the median tok/s number for the closure report.

---

## Task 5.4: Final 36-prompt eval

Run:
```bash
bun run eval/bench.ts --model gemma-4-e2b-it-q4km \
  --eval-temperature 0 --runs 1
```

**Pass gate:** ≥ 0.60 (matches Stage 4 floor; ideally lifts further
since shared-KV correctness is load-bearing for Gemma 4's trained
behavior).

Save the eval JSON to
`eval/reports/gemma-4-e2b-validation-2026-05-10/eval-results.json`.

---

## Task 5.5: Closure report + TODO archival

**Files:**
- Create: `eval/reports/gemma-4-e2b-validation-2026-05-10/SUMMARY.md`
- Modify: `TODO.md` (move the Gemma 4 staging block to a closure stub; archive the full block to `TODO_ARCHIVE.md` per the cadence doctrine)

- [ ] **Step 1: Write the closure report**

Template (mirror the Phi-3 closure at
`eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`):

```markdown
# Gemma 4 E2B (Q4_K_M) inference support — campaign closure

**Date:** 2026-05-10
**Spec:** docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md
**Plan:** docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md

## TL;DR
Gemma 4 E2B ships at <X> tok/s median (3-run smoke-bench profile) and
<Y.YY> 36-prompt eval. Apache-2.0; first Gemma-family model in the
fleet via the GGUF/WASM path. Five-stage correctness-first staging
absorbed seven architectural deltas vs the project's prior causal-LM
dispatch.

## Per-stage commit history
| Stage | Commits | Closure report |
|---|---|---|
| 1 — per-layer hp refactor | <commit list> | gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md |
| 2 — surface wiring | <commit list> | gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md |
| 3 — PLE + dualRoPE | <commit list> | gemma-4-stage3-ple-dualrope-2026-05-10/SUMMARY.md |
| 4 — real SWA | <commit list> | gemma-4-stage4-swa-2026-05-10/SUMMARY.md |
| 5 — shared-KV + bench | <commit list> | (this report) |

## Architecture deltas absorbed
1. Per-layer head_dim (Gemma 4 mixes 512 global / 256 SWA)
2. Per-layer FFN size (6144 → 12288 transition at layer 15)
3. Per-layer RoPE dim + freq_base
4. Sliding-window attention with 5:1 local:global pattern
5. Shared-KV references (last 20 of 35 layers)
6. Per-Layer Embeddings table
7. Final logit softcap = 30.0

## Performance
- Tok/s (3-run median, profile mode): <X>
- Eval (36 prompts, greedy @ T=0): <Y.YY>
- KV cache memory: <Z MB> (down from <W MB> pre-Stage-5)

## Patch stack
<9 + 0 or 9 + 1 or 9 + 2> patches on webllm-browser-patches.
Stage 4 SWA: <no patch needed | patch <hash>>.
Stage 5 KV: <no patch needed | patch <hash>>.

## Follow-ups
- E4B SKU: register + smoke; expected to work without further
  architectural changes since the Gemma 4 dispatch is now in place.
- PLE CPU offload: defer unless GPU residency becomes a constraint.
- Tool calling (Gemma 4 PEG format): out of scope this campaign.
```

- [ ] **Step 2: Update TODO.md**

Per CLAUDE.md cadence doctrine, when a top-level block closes, replace
it with a 4-8-line closure stub. Find the Gemma 4 section added in
this campaign and replace with:

```markdown
### Gemma 4 E2B inference support (CLOSED 2026-05-10; archived from TODO.md)

Closed 2026-05-10 — all 5 stages shipped (per-layer hp refactor →
surface wiring → PLE+dualRoPE → real SWA → shared-KV+bench+closure).
E2B Q4_K_M ships at <X> tok/s / <Y.YY> 36-prompt eval. Closure
report: [`eval/reports/gemma-4-e2b-validation-2026-05-10/SUMMARY.md`](eval/reports/gemma-4-e2b-validation-2026-05-10/SUMMARY.md).
Patch stack <growth>; <new patches if any>. Full block archived to
`TODO_ARCHIVE.md`. E4B SKU follow-on probe: queued under "External-
trigger candidates" pending consumer ask.
```

Then move the original full block to `TODO_ARCHIVE.md`.

- [ ] **Step 3: Commit closure**

```bash
git add eval/reports/gemma-4-e2b-validation-2026-05-10/
git commit -m "docs(reports): Gemma 4 E2B campaign closure"
```

- [ ] **Step 4: Commit TODO archival**

```bash
git add TODO.md TODO_ARCHIVE.md
git commit -m "docs(TODO): archive Gemma 4 E2B campaign closure"
```

---

## Self-review

Plan covers all 5 stages of the spec; each stage has explicit build +
runtime gates that match Section 4 of the spec. Per-task TDD where it
makes sense (chat template, sampler, hp loading); runtime-gate-first
where unit tests can't capture the behavior (SWA correctness, PLE
injection, bench, ref-shared KV memory).

No "TBD" or placeholders; all file paths are concrete; verification
commands are runnable. The few `<TAB_ID>`, `<X>`, `<Y.YY>` placeholders
are intentional — they're values captured at runtime and recorded in
the closure reports.

Type consistency: `embeddingHeadLengthPerLayer`, `feedForwardLengthPerLayer`,
`ropeDimensionCountPerLayer`, `ropeFreqBasePerLayer`, `slidingWindowPattern`,
`slidingWindowSize`, `sharedKvLayers`, `finalLogitSoftcap`, `perLayerEmbed`,
`isGemma4`, `GEMMA4_DEFAULTS`, `formatGemma4` — all introduced in early
tasks, all used consistently in later tasks under those same names.

Risks (spec §6) covered:
- Stage 1 silent regression → Task 1.4's 3-model `generatedIds[0]` cross-check
- Stage 3 PLE binding cap → Task 3.1's PROBE.md sizing pre-impl
- Stage 4 SWA mask shape → Task 4.1's pre-implementation probe + decision tree
- Stage 5 persistence break → Task 5.2's explicit serialization handling
- Eval misfire → first-token semantic gate (Task 3.4 Step 3) backs up the eval

## Execution handoff

Per user CLAUDE.md preference: **Subagent-Driven (this session)** —
chosen without asking.

**Next:** Use `superpowers:subagent-driven-development` to execute
Task 1.1 immediately, then proceed task-by-task with two-stage review
between tasks.
