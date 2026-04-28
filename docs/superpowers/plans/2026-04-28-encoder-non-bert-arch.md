# Encoder non-BERT architecture support — implementation plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v2 (post-Phase-0 rewrite, 2026-04-28).** v1 of this plan and its spec assumed both encoders shared BERT's split-QKV / GeLU / fully-biased structure. Phase 0 GGUF probe (commit `43df996`) showed material divergences: nomic-bert has fused QKV + no biases + SwiGLU FFN; jina-bert-v2 has SwiGLU + mixed biases. v2 splits Phase 2 into 2a (bert + jina-compatible path) and 2b (nomic fused-QKV + RoPE), inverts model registration order (jina first), and adds an extra commit. v1 plan preserved at git ref `4c4cd4c`.

**Goal:** Extend `EncoderInference` to support `nomic-embed-text-v1.5` (`nomic-bert`, 137M, 768-dim, fused-QKV + RoPE + SwiGLU) and `jina-embeddings-v2-base-en` (`jina-bert-v2`, 137M, 768-dim, split-QKV + ALiBi + SwiGLU).

**Architecture:** `opRope`, `opSoftMaxExt(max_bias)`, `opSilu`, `opMul`, `opView3d`, `opMulMat`, `opGelu` are all already in the WASM bindings. `EncoderLayerWeights` becomes nullable across all bias and gate fields. Phase 2a lands the bert + jina-bert-v2 paths and the nullable-bias + SwiGLU machinery; nomic constructor-accepts but throws inside `loadWeights` / `buildGraph` with `"nomic forward not enabled until phase 2b"`. Phase 2b layers fused-QKV (`opMulMat` + 3× `opView3d` slicing per llama.cpp's `build_qkv()`) and RoPE (Q+K) on top.

**Tech Stack:** TypeScript / Bun, patched `llama.cpp` `ggml-webgpu` compiled to WASM (already built), `bun test`, `make checkall`, `make bench-full`, `agentchrome`, `sentence-transformers` via `uv run --no-project` (one-shot reference-vector capture).

**Spec:** `docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md` (commit `bf51912`).

**Phasing:** 6 main commits + 2 out-of-band commits (Phase 0 probe, Phase 3a reference vectors). Each main commit `make checkall` clean and independently revertable. Mirrors §17/§18/§19/§20 plan structure.

---

## Task 0: Commit the revised plan

**Files:** `docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md` (this file).

**Purpose:** Land the plan rewrite as its own commit before further phase commits.

- [ ] **Step 1: Force-add and commit the plan**

```bash
git add -f docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md
git commit -m "$(cat <<'EOF'
docs(plan): revise bucket B plan post-Phase-0 (jina-first phasing)

Plan rewrite to match the post-Phase-0 spec rewrite (commit bf51912).
Phase 2 splits into 2a (bert + jina-bert-v2 path; nomic constructor-
accepts but throws on loadWeights/buildGraph) and 2b (nomic fused-QKV
+ RoPE). Model registration order inverts: jina first (closer to bert),
nomic second. 6 main commits + 2 out-of-band (probe + ref vectors).

v1 plan preserved at git ref 4c4cd4c.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md
EOF
)"
```

---

## Task 1: Phase 0 — GGUF discovery probe ✅ DONE (commit `43df996`)

**Result:** Three spec-tagged unknowns + four major architectural divergences resolved. Triggered the spec rewrite. The Phase 0 commit is the only Phase-N artifact landed before the plan revision. Steps below kept for archival reference.

**Files (committed):**
- `eval/reports/encoder-parity-2026-04-28/probe-gguf.ts`
- `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt`
- `eval/reports/encoder-parity-2026-04-28/inputs.json`

Full discovery results live in the artifact. Key load-bearing facts for the rest of the plan:

- `general.architecture` values: `nomic-bert`, `jina-bert-v2` (matches spec assumption).
- `nomic-bert.rope.freq_base = 1000` (NB: NOT the typical 10000; reads cleanly from GGUF via existing `model-loader.ts:95-98` fallback chain).
- jina-bert-v2 ALiBi metadata key absent; default to 8.0.
- `position_embd.weight` absent in both nomic and jina (matches spec).
- nomic-bert tensors: fused `attn_qkv.weight`, no biases anywhere, SwiGLU FFN (`ffn_gate.weight + ffn_up.weight + ffn_down.weight`).
- jina-bert-v2 tensors: split `attn_{q,k,v}.{weight,bias}` + `attn_output.{weight,bias}`, SwiGLU FFN with `ffn_down.bias` only (no `ffn_up.bias`, no `ffn_gate.bias`).
- Both use `tokenizer.ggml.model = bert` → WordPiece works unchanged.
- ⚠️ **API note:** `parseGgufHeader` is `GgufParser.parse(bytes)` (static method on the class), not a free function. Use this form in any future probe / harness scripts.

---

## Task 2: Phase 1 — Types, loader, RoPE helper (commit 1 of 6)

**Files:**
- Modify: `src/core/types.ts:50-60` (extend `ModelArchitecture` union)
- Modify: `src/core/types.ts:95-118` (add `alibiMaxBias?: number` field; export `ENCODER_ARCHITECTURES` + `isEncoderArchitecture()`)
- Modify: `src/models/model-loader.ts:40-106` (broaden `arch === "bert"` branch to `isEncoderArchitecture(arch)`; read jina ALiBi metadata)
- Modify: `src/inference/model-inference.ts:83-89` (extend `getRopeModeForArchitecture`)
- Modify: `tests/encoder-inference.test.ts` (test #1 helper truth-table; test #2 metadata routing)

**Purpose:** Type-system + loader-side foundation. After this task `make checkall` passes, but no encoder forward changes yet — bert path identical, new arches not yet routable through engine.

- [ ] **Step 1: Write the failing test for `isEncoderArchitecture` helper**

Add to `tests/encoder-inference.test.ts`:

```ts
import { isEncoderArchitecture, ENCODER_ARCHITECTURES } from "../src/core/types.js";

describe("isEncoderArchitecture", () => {
  test("returns true for encoder archs", () => {
    expect(isEncoderArchitecture("bert")).toBe(true);
    expect(isEncoderArchitecture("nomic-bert")).toBe(true);
    expect(isEncoderArchitecture("jina-bert-v2")).toBe(true);
  });
  test("returns false for causal archs", () => {
    for (const a of ["llama", "mistral", "qwen", "qwen2", "qwen3", "phi", "gemma", "mixtral", "deepseek"] as const) {
      expect(isEncoderArchitecture(a)).toBe(false);
    }
  });
  test("ENCODER_ARCHITECTURES tuple matches helper truth-table", () => {
    expect(ENCODER_ARCHITECTURES).toEqual(["bert", "nomic-bert", "jina-bert-v2"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "isEncoderArchitecture"`
Expected: FAIL with "no exported member 'isEncoderArchitecture'".

- [ ] **Step 3: Extend `ModelArchitecture` union and add helper in `src/core/types.ts`**

In `src/core/types.ts:50-60`, replace:

```ts
export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "gemma"
	| "qwen"
	| "qwen2"
	| "qwen3"
	| "mixtral"
	| "deepseek"
	| "bert";
```

With:

```ts
export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "gemma"
	| "qwen"
	| "qwen2"
	| "qwen3"
	| "mixtral"
	| "deepseek"
	| "bert"
	| "nomic-bert"
	| "jina-bert-v2";

/** All architectures handled by EncoderInference (bidirectional, no KV cache). */
export const ENCODER_ARCHITECTURES = ["bert", "nomic-bert", "jina-bert-v2"] as const;

export function isEncoderArchitecture(a: ModelArchitecture): boolean {
	return (ENCODER_ARCHITECTURES as readonly string[]).includes(a);
}
```

- [ ] **Step 4: Add `alibiMaxBias` field to `ModelHyperparams`**

In `src/core/types.ts:95-118`, after the `causalAttention?: boolean;` field (line 117), add:

```ts
	/**
	 * Only jina-bert-v2 populates this; the value is passed straight to
	 * `opSoftMaxExt`'s `max_bias` arg, which causes ggml's softmax to apply
	 * the standard ALiBi linear bias (slopes derived as `2^(-8/n_head * h)`).
	 * Defaults to 8.0 when GGUF metadata omits the key (gaianet mirror).
	 */
	alibiMaxBias?: number;
```

- [ ] **Step 5: Run helper tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "isEncoderArchitecture"`
Expected: PASS (all 3 sub-tests).

- [ ] **Step 6: Write the failing test for loader metadata routing**

Add to `tests/encoder-inference.test.ts`:

```ts
import { ModelLoader } from "../src/models/model-loader.js";
import type { GgufContext } from "../src/models/gguf-types.js";

function fakeCtx(arch: string, extras: Record<string, unknown> = {}): GgufContext {
  const meta = new Map<string, { value: unknown }>();
  meta.set("general.architecture", { value: arch });
  meta.set(`${arch}.embedding_length`, { value: 768 });
  meta.set(`${arch}.block_count`, { value: 12 });
  meta.set(`${arch}.attention.head_count`, { value: 12 });
  meta.set(`${arch}.feed_forward_length`, { value: 3072 });
  meta.set(`${arch}.attention.layer_norm_epsilon`, { value: 1e-12 });
  meta.set(`${arch}.context_length`, { value: 8192 });
  meta.set(`${arch}.attention.causal`, { value: false });
  meta.set(`${arch}.pooling_type`, { value: 1 }); // MEAN
  for (const [k, v] of Object.entries(extras)) meta.set(k, { value: v });
  return { metadata: meta, tensors: [], dataOffset: 0, totalDataSize: 0 } as unknown as GgufContext;
}

describe("ModelLoader.extractHyperparams non-BERT encoder branches", () => {
  test("nomic-bert produces RoPE-ready hyperparams", () => {
    const ctx = fakeCtx("nomic-bert", {
      "nomic-bert.rope.freq_base": 1000.0,
    });
    const hp = (ModelLoader as unknown as { extractHyperparams: (c: unknown) => unknown })
      .extractHyperparams(ctx) as Record<string, unknown>;
    expect(hp.architecture).toBe("nomic-bert");
    expect(hp.causalAttention).toBe(false);
    expect(hp.poolingType).toBe("mean");
    expect(hp.ropeFreqBase).toBe(1000.0);
    expect(hp.normEpsilon).toBeCloseTo(1e-12);
    expect(hp.alibiMaxBias).toBeUndefined();
  });
  test("jina-bert-v2 falls back to alibiMaxBias=8.0 when metadata absent", () => {
    const ctx = fakeCtx("jina-bert-v2");
    const hp = (ModelLoader as unknown as { extractHyperparams: (c: unknown) => unknown })
      .extractHyperparams(ctx) as Record<string, unknown>;
    expect(hp.architecture).toBe("jina-bert-v2");
    expect(hp.causalAttention).toBe(false);
    expect(hp.poolingType).toBe("mean");
    expect(hp.alibiMaxBias).toBe(8.0);
  });
  test("jina-bert-v2 honors alibi_bias_max metadata when present", () => {
    const ctx = fakeCtx("jina-bert-v2", {
      "jina-bert-v2.attention.alibi_bias_max": 16.0,
    });
    const hp = (ModelLoader as unknown as { extractHyperparams: (c: unknown) => unknown })
      .extractHyperparams(ctx) as Record<string, unknown>;
    expect(hp.alibiMaxBias).toBe(16.0);
  });
});
```

(`extractHyperparams` is `private static`. Cast pattern reaches it; if a test convention exists in the repo, match it — grep for `extractHyperparams` in `tests/`.)

- [ ] **Step 7: Run loader test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "extractHyperparams non-BERT"`
Expected: FAIL — `extractHyperparams` returns `architecture: "nomic-bert"` but `causalAttention === undefined`.

- [ ] **Step 8: Broaden `extractHyperparams` in `src/models/model-loader.ts`**

Add the import:

```ts
import { isEncoderArchitecture } from "../core/types.js";
```

Replace `model-loader.ts:56-59`:

```ts
const normEpsilon =
  arch === "bert"
    ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
    : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);
```

with:

```ts
const normEpsilon = isEncoderArchitecture(arch)
  ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
  : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);
```

Replace `model-loader.ts:62-71`:

```ts
let poolingType: ModelHyperparams["poolingType"];
let causalAttention: boolean | undefined;
if (arch === "bert") {
  const pt = getMetaNumberOptional(ctx, `${arch}.pooling_type`) ?? 2;
  poolingType = pt === 1 ? "mean" : "cls";
  causalAttention =
    getMetaBooleanOptional(ctx, `${arch}.attention.causal`) ?? false;
}
```

with:

```ts
let poolingType: ModelHyperparams["poolingType"];
let causalAttention: boolean | undefined;
let alibiMaxBias: number | undefined;
if (isEncoderArchitecture(arch)) {
  const pt = getMetaNumberOptional(ctx, `${arch}.pooling_type`) ?? 2;
  poolingType = pt === 1 ? "mean" : "cls";
  causalAttention =
    getMetaBooleanOptional(ctx, `${arch}.attention.causal`) ?? false;
  if (arch === "jina-bert-v2") {
    alibiMaxBias =
      getMetaNumberOptional(ctx, `${arch}.attention.alibi_bias_max`) ?? 8.0;
  }
}
```

Add `alibiMaxBias,` to the returned object literal alongside `causalAttention,`.

- [ ] **Step 9: Run loader tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "extractHyperparams non-BERT"`
Expected: PASS (3 sub-tests).

- [ ] **Step 10: Extend `getRopeModeForArchitecture` in `src/inference/model-inference.ts`**

Replace `model-inference.ts:83-89`:

```ts
export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}
```

with:

```ts
export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	if (architecture === "nomic-bert") return RopeMode.NORMAL;
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}
```

(If Phase 0 had revealed nomic uses NEOX, the parity probe in Task 7 would catch it; flip then.)

- [ ] **Step 11: Run full ship gate**

Run: `make checkall`
Expected: PASS. Bert path unchanged. The existing `EncoderInference construction` test at line 7 of v1 tests will continue to fail-fast on `architecture: "llama"` because the constructor still hard-asserts on bert in this commit (broadening lands in Task 3) — verify the test's regex `/requires architecture "bert"/` still matches.

- [ ] **Step 12: Commit Phase 1**

```bash
git add src/core/types.ts \
        src/models/model-loader.ts \
        src/inference/model-inference.ts \
        tests/encoder-inference.test.ts
git commit -m "$(cat <<'EOF'
feat(types): widen ModelArchitecture for nomic-bert + jina-bert-v2

Phase 1 of bucket B (encoder non-BERT arch). Type-system + loader-side
foundation:
- ModelArchitecture union grows to 12 (adds nomic-bert, jina-bert-v2)
- ENCODER_ARCHITECTURES + isEncoderArchitecture() helper exported from
  types.ts; centralizes the bert/nomic/jina branch
- alibiMaxBias?: number field on ModelHyperparams (jina-only; falls
  back to 8.0 — gaianet mirror omits the metadata key)
- extractHyperparams: arch === "bert" branches broaden to
  isEncoderArchitecture(arch); jina-only ALiBi metadata read with 8.0
  fallback
- getRopeModeForArchitecture: nomic-bert -> NORMAL (sentence-transformers
  convention; verified in Phase 0 probe)

Existing BERT path unchanged. EncoderInference still hard-asserts on
"bert"; routing change lands in Phase 2a. make checkall green.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §4
EOF
)"
```

---

## Task 3: Phase 2a — Encoder forward (bert + jina-bert-v2) + engine routing (commit 2 of 6)

**Files:**
- Modify: `src/inference/encoder-inference.ts:10-36` (`EncoderLayerWeights` nullable bias/gate fields; `qkvFused` field; `EncoderWeights.positionEmb` nullable)
- Modify: `src/inference/encoder-inference.ts:54-62` (drop `arch === "bert"` hard-assert; broaden to `isEncoderArchitecture`)
- Modify: `src/inference/encoder-inference.ts:85-122` (arch-branched `loadWeights`; new `makeTensorOptional`; jina arm; nomic arm throws stub)
- Modify: `src/inference/encoder-inference.ts:172-240` (arch-branched `buildGraph` Points A/D/E/F; nomic Points B-fused + C throw stub)
- Modify: `src/core/engine.ts:589` (broaden routing to `isEncoderArchitecture`)
- Modify: `tests/encoder-inference.test.ts` (rewrite construction test; loader-routing test; `makeTensorOptional` test; `buildGraph` dispatch tests for bert + jina)

**Purpose:** Encoder forward graph + engine routing accept jina-bert-v2 (split QKV with biases, ALiBi, SwiGLU). bert path stays bit-identical. nomic-bert is constructor-accepted but throws inside `loadWeights` / `buildGraph` until Phase 2b — this lets us land the nullable-bias and SwiGLU machinery in isolation, get its `make checkall` clean, then layer fused-QKV + RoPE on top in Phase 2b.

### Phase 2a high-level approach

The encoder grows three orthogonal pieces of machinery in this commit:

1. **Constructor + type widening.** `isEncoderArchitecture` accepts all three arches; `EncoderLayerWeights` becomes nullable across bias/gate fields; new `qkvFused` field.
2. **Loader arch dispatch.** bert preserves the strict full-bias load. jina-bert-v2 loads split QKV with biases + SwiGLU `ffn_gate.weight` + `ffn_down.bias` only (`ffn_up.bias` absent). nomic-bert reaches `loadWeights` and throws `Error("nomic forward not enabled until phase 2b")`.
3. **`buildGraph` arch branches.** Points A (pos-embedding gate), D (softmax with `alibiMaxBias`), E (attn output bias gated), F (SwiGLU vs GeLU FFN) are all gated. Points B-fused and C (RoPE) explicitly throw the same error message — bert + jina paths are exercisable in isolation.

After Phase 2a: bert bit-identical, jina runs end-to-end, nomic constructor-accepts but throws on `loadWeights`. `make checkall` clean.

### Phase 2a TDD steps

- [ ] **Step 1: Write the failing test for non-bert constructor acceptance**

Replace the existing `describe("EncoderInference construction", …)` block in `tests/encoder-inference.test.ts` (lines 6-28) with:

```ts
describe("EncoderInference construction", () => {
  function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
    return {
      architecture: arch,
      contextLength: 512,
      embeddingLength: 384,
      headCount: 12,
      headCountKv: 12,
      layerCount: 12,
      vocabularySize: 30522,
      embeddingHeadLength: 32,
      feedForwardLength: 1536,
      ropeFreqBase: 10000,
      ropeScale: 1,
      normEpsilon: 1e-12,
      expertCount: 0,
      expertUsedCount: 0,
    };
  }
  test("rejects causal LM hyperparams", () => {
    expect(() => new EncoderInference({} as never, makeHp("llama"))).toThrow(
      /not yet support architecture "llama"/,
    );
  });
  test("accepts bert", () => {
    expect(() => new EncoderInference({} as never, makeHp("bert"))).not.toThrow();
  });
  test("accepts nomic-bert", () => {
    expect(() => new EncoderInference({} as never, makeHp("nomic-bert"))).not.toThrow();
  });
  test("accepts jina-bert-v2", () => {
    expect(() => new EncoderInference({} as never, makeHp("jina-bert-v2"))).not.toThrow();
  });
});
```

- [ ] **Step 2: Run construction tests to verify they fail**

Run: `bun test tests/encoder-inference.test.ts -t "EncoderInference construction"`
Expected: FAIL on `nomic-bert` / `jina-bert-v2` cases (current constructor throws on non-bert), and the regex on `llama` doesn't match (current message says `requires architecture "bert"`).

- [ ] **Step 3: Replace the constructor hard-assert in `src/inference/encoder-inference.ts`**

Add to imports:

```ts
import { ENCODER_ARCHITECTURES, isEncoderArchitecture } from "../core/types.js";
```

Replace lines 54-62:

```ts
constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
  if (hyperparams.architecture !== "bert") {
    throw new Error(
      `EncoderInference requires architecture "bert", got "${hyperparams.architecture}"`,
    );
  }
  this.wasm = wasm;
  this.hp = hyperparams;
}
```

with:

```ts
constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
  if (!isEncoderArchitecture(hyperparams.architecture)) {
    throw new Error(
      `EncoderInference does not yet support architecture "${hyperparams.architecture}"; supported: ${ENCODER_ARCHITECTURES.join(", ")}`,
    );
  }
  this.wasm = wasm;
  this.hp = hyperparams;
}
```

- [ ] **Step 4: Run construction tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "EncoderInference construction"`
Expected: PASS (4 sub-tests).

- [ ] **Step 5: Update `EncoderLayerWeights` and `EncoderWeights` interfaces**

In `src/inference/encoder-inference.ts:10-36`, replace the `EncoderLayerWeights` and `EncoderWeights` interface declarations with:

```ts
interface EncoderLayerWeights {
	// QKV — exactly one path is populated per arch:
	qkvFused: TensorPtr | null;          // nomic-bert: fused single matrix
	qProj: TensorPtr | null;             // bert + jina-bert-v2
	qBias: TensorPtr | null;             // bert + jina-bert-v2
	kProj: TensorPtr | null;
	kBias: TensorPtr | null;
	vProj: TensorPtr | null;
	vBias: TensorPtr | null;

	oProj: TensorPtr;                    // all
	oBias: TensorPtr | null;             // bert + jina-bert-v2 (not nomic)

	attnNormW: TensorPtr;                // post-attn LN gamma
	attnNormB: TensorPtr;                // post-attn LN beta

	ffnGate: TensorPtr | null;           // nomic + jina-bert-v2 (SwiGLU)
	ffnUp: TensorPtr;                    // all
	ffnUpBias: TensorPtr | null;         // bert only
	ffnDown: TensorPtr;                  // all
	ffnDownBias: TensorPtr | null;       // bert + jina-bert-v2 (not nomic)

	ffnNormW: TensorPtr;                 // post-FFN LN gamma
	ffnNormB: TensorPtr;                 // post-FFN LN beta
}

interface EncoderWeights {
	tokEmb: TensorPtr;
	/** Null for nomic-bert / jina-bert-v2 — RoPE / ALiBi handle position. */
	positionEmb: TensorPtr | null;
	tokenTypes: TensorPtr;
	inputNormW: TensorPtr;
	inputNormB: TensorPtr;
	layers: EncoderLayerWeights[];
}
```

- [ ] **Step 6: Add `makeTensorOptional` helper**

In `src/inference/encoder-inference.ts`, immediately after the existing `makeTensor` method (line 161), add:

```ts
private makeTensorOptional(
	tensorMap: Map<string, GgufTensorInfo>,
	name: string,
): TensorPtr | null {
	return tensorMap.has(name) ? this.makeTensor(tensorMap, name) : null;
}
```

- [ ] **Step 7: Replace `loadWeights` body with arch-branched form (jina + bert reachable; nomic throws)**

In `src/inference/encoder-inference.ts:64-139` (the `loadWeights` method body), replace the section from `const tokEmb = this.makeTensor(...)` (line 85) through the closing brace of the for-loop building `layers` (line 112) with:

```ts
const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
const positionEmb = hp.architecture === "bert"
	? this.makeTensor(tensorMap, "position_embd.weight")
	: null;
const tokenTypes = this.makeTensor(tensorMap, "token_types.weight");
const inputNormW = this.makeTensor(tensorMap, "token_embd_norm.weight");
const inputNormB = this.makeTensor(tensorMap, "token_embd_norm.bias");

if (hp.architecture === "nomic-bert") {
	throw new Error(
		"EncoderInference: nomic-bert forward not enabled until Phase 2b",
	);
}

const layers: EncoderLayerWeights[] = [];
for (let i = 0; i < hp.layerCount; i++) {
	const p = (s: string) => `blk.${i}.${s}`;
	layers.push({
		qkvFused: null,
		qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
		qBias: this.makeTensorOptional(tensorMap, p("attn_q.bias")),
		kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
		kBias: this.makeTensorOptional(tensorMap, p("attn_k.bias")),
		vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
		vBias: this.makeTensorOptional(tensorMap, p("attn_v.bias")),
		oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
		oBias: this.makeTensorOptional(tensorMap, p("attn_output.bias")),
		attnNormW: this.makeTensor(tensorMap, p("attn_output_norm.weight")),
		attnNormB: this.makeTensor(tensorMap, p("attn_output_norm.bias")),
		ffnGate: this.makeTensorOptional(tensorMap, p("ffn_gate.weight")),
		ffnUp: this.makeTensor(tensorMap, p("ffn_up.weight")),
		ffnUpBias: this.makeTensorOptional(tensorMap, p("ffn_up.bias")),
		ffnDown: this.makeTensor(tensorMap, p("ffn_down.weight")),
		ffnDownBias: this.makeTensorOptional(tensorMap, p("ffn_down.bias")),
		ffnNormW: this.makeTensor(tensorMap, p("layer_output_norm.weight")),
		ffnNormB: this.makeTensor(tensorMap, p("layer_output_norm.bias")),
	});
}
```

- [ ] **Step 8: Update the FakeWasm test stub**

In `tests/encoder-inference.test.ts`, in the `makeFakeWasm()` factory (around line 35), update the `FakeWasm` interface and the stub body to capture the new ops the encoder will dispatch:

```ts
// In FakeWasm interface:
interface FakeWasm {
  fake: GgmlWasm;
  ops: string[];
  softmaxMaxBias: number[];
}

// In makeFakeWasm(), inside the `stub` literal — verify each isn't already present
// (the v1 fake had `mul`, `mulMat`, `gelu`, `add`, `getrows`, `permute`, `cont`,
// `reshape2d`, `reshape3d`, `softMaxExt`); add or augment whichever are missing:
opSilu: () => { ops.push("silu"); return next++; },
opSoftMaxExt: (_x: TensorPtr, _mask: number, _scale: number, maxBias: number) => {
  ops.push("softmaxext");
  softmaxMaxBias.push(maxBias);
  return next++;
},
opRope: () => { ops.push("rope"); return next++; },
opView3d: () => { ops.push("view3d"); return next++; },

// Initialize before return:
const softmaxMaxBias: number[] = [];
return { fake: stub as unknown as GgmlWasm, ops, softmaxMaxBias };
```

(If `opSoftMaxExt` already exists in the v1 fake, replace the existing stub with the new one that records `maxBias`. Test #4 below depends on this.)

- [ ] **Step 9: Write the failing tests for `loadWeights` arch dispatch (bert, jina, nomic-throws) and `makeTensorOptional`**

Append to `tests/encoder-inference.test.ts`:

```ts
describe("EncoderInference.makeTensorOptional", () => {
  test("returns null when tensor name absent", () => {
    const fake = makeFakeWasm();
    const enc = new EncoderInference(fake.fake, {
      architecture: "bert", contextLength: 512, embeddingLength: 384,
      headCount: 12, headCountKv: 12, layerCount: 1,
      vocabularySize: 30522, embeddingHeadLength: 32,
      feedForwardLength: 1536, ropeFreqBase: 10000, ropeScale: 1,
      normEpsilon: 1e-12, expertCount: 0, expertUsedCount: 0,
    });
    const empty = new Map();
    const result = (enc as unknown as { makeTensorOptional: (m: unknown, n: string) => unknown })
      .makeTensorOptional(empty, "missing.weight");
    expect(result).toBeNull();
  });
});

describe("EncoderInference.loadWeights arch dispatch", () => {
  function dim(name: string): GgufTensorInfo {
    return { name, dimensions: [768, 768], type: 0, offset: 0 } as GgufTensorInfo;
  }
  function makeCtx(names: string[]): GgufContext {
    return {
      metadata: new Map(),
      tensors: names.map(dim),
      dataOffset: 0,
      totalDataSize: 0,
    } as unknown as GgufContext;
  }
  function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
    return {
      architecture: arch, contextLength: 512, embeddingLength: 768,
      headCount: 12, headCountKv: 12, layerCount: 1,
      vocabularySize: 30522, embeddingHeadLength: 64,
      feedForwardLength: 3072, ropeFreqBase: 10000, ropeScale: 1,
      normEpsilon: 1e-12, expertCount: 0, expertUsedCount: 0,
      poolingType: "mean", causalAttention: false,
      ...(arch === "jina-bert-v2" ? { alibiMaxBias: 8.0 } : {}),
    };
  }

  const bertNames = [
    "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
    "token_types.weight", "position_embd.weight",
    "blk.0.attn_q.weight", "blk.0.attn_q.bias",
    "blk.0.attn_k.weight", "blk.0.attn_k.bias",
    "blk.0.attn_v.weight", "blk.0.attn_v.bias",
    "blk.0.attn_output.weight", "blk.0.attn_output.bias",
    "blk.0.attn_output_norm.weight", "blk.0.attn_output_norm.bias",
    "blk.0.ffn_up.weight", "blk.0.ffn_up.bias",
    "blk.0.ffn_down.weight", "blk.0.ffn_down.bias",
    "blk.0.layer_output_norm.weight", "blk.0.layer_output_norm.bias",
  ];
  const jinaNames = [
    "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
    "token_types.weight",
    "blk.0.attn_q.weight", "blk.0.attn_q.bias",
    "blk.0.attn_k.weight", "blk.0.attn_k.bias",
    "blk.0.attn_v.weight", "blk.0.attn_v.bias",
    "blk.0.attn_output.weight", "blk.0.attn_output.bias",
    "blk.0.attn_output_norm.weight", "blk.0.attn_output_norm.bias",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight", "blk.0.ffn_down.bias",
    "blk.0.layer_output_norm.weight", "blk.0.layer_output_norm.bias",
  ];

  test("bert: full-bias path; ffnGate null", () => {
    const fake = makeFakeWasm();
    const enc = new EncoderInference(fake.fake, makeHp("bert"));
    enc.loadWeights(makeCtx(bertNames), new Uint8Array(0));
    const layers = (enc as unknown as { weights: { layers: EncoderLayerWeights[] } }).weights.layers;
    expect(layers[0].qProj).not.toBeNull();
    expect(layers[0].qBias).not.toBeNull();
    expect(layers[0].oBias).not.toBeNull();
    expect(layers[0].ffnUpBias).not.toBeNull();
    expect(layers[0].ffnDownBias).not.toBeNull();
    expect(layers[0].ffnGate).toBeNull();
    expect(layers[0].qkvFused).toBeNull();
  });

  test("jina-bert-v2: split QKV + biases; SwiGLU gate; mixed FFN biases", () => {
    const fake = makeFakeWasm();
    const enc = new EncoderInference(fake.fake, makeHp("jina-bert-v2"));
    enc.loadWeights(makeCtx(jinaNames), new Uint8Array(0));
    const layers = (enc as unknown as { weights: { layers: EncoderLayerWeights[] } }).weights.layers;
    expect(layers[0].qProj).not.toBeNull();
    expect(layers[0].qBias).not.toBeNull();
    expect(layers[0].oBias).not.toBeNull();
    expect(layers[0].ffnGate).not.toBeNull();
    expect(layers[0].ffnUpBias).toBeNull();
    expect(layers[0].ffnDownBias).not.toBeNull();
    expect(layers[0].qkvFused).toBeNull();
  });

  test("nomic-bert: throws (Phase 2b not landed)", () => {
    const fake = makeFakeWasm();
    const enc = new EncoderInference(fake.fake, makeHp("nomic-bert"));
    expect(() => enc.loadWeights(makeCtx([
      "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
      "token_types.weight",
    ]), new Uint8Array(0))).toThrow(/not enabled until Phase 2b/);
  });
});
```

Verify the FakeWasm stub already covers `tensorNbytes` and `uploadToTensorChunked` — those are called by `loadWeights`. If missing, add `tensorNbytes: () => 0`, `uploadToTensorChunked: () => {}`.

- [ ] **Step 10: Run loader tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "loadWeights arch dispatch"`
Expected: PASS (3 sub-tests). Also `makeTensorOptional` PASS.

- [ ] **Step 11: Write the failing test for buildGraph arch dispatch (bert + jina)**

Append to `tests/encoder-inference.test.ts`:

```ts
describe("EncoderInference.buildGraph arch dispatch", () => {
  function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
    return {
      architecture: arch, contextLength: 512, embeddingLength: 384,
      headCount: 12, headCountKv: 12, layerCount: 2,
      vocabularySize: 30522, embeddingHeadLength: 32,
      feedForwardLength: 1536, ropeFreqBase: 10000, ropeScale: 1,
      normEpsilon: 1e-12, expertCount: 0, expertUsedCount: 0,
      poolingType: "mean", causalAttention: false,
      ...(arch === "jina-bert-v2" ? { alibiMaxBias: 8.0 } : {}),
    };
  }
  function buildAndCount(arch: ModelHyperparams["architecture"]): {
    getrows: number; rope: number; silu: number; gelu: number;
    view3d: number; softmax_max_bias: number[];
  } {
    const fake = makeFakeWasm();
    const hp = makeHp(arch);
    const enc = new EncoderInference(fake.fake, hp);
    (enc as unknown as { weights: unknown }).weights = {
      tokEmb: 1, positionEmb: arch === "bert" ? 2 : null, tokenTypes: 3,
      inputNormW: 4, inputNormB: 5,
      layers: Array.from({ length: hp.layerCount }, () => ({
        qkvFused: arch === "nomic-bert" ? 100 : null,
        qProj: arch === "nomic-bert" ? null : 10,
        qBias: (arch === "bert" || arch === "jina-bert-v2") ? 11 : null,
        kProj: arch === "nomic-bert" ? null : 12,
        kBias: (arch === "bert" || arch === "jina-bert-v2") ? 13 : null,
        vProj: arch === "nomic-bert" ? null : 14,
        vBias: (arch === "bert" || arch === "jina-bert-v2") ? 15 : null,
        oProj: 16,
        oBias: (arch === "bert" || arch === "jina-bert-v2") ? 17 : null,
        attnNormW: 18, attnNormB: 19,
        ffnGate: (arch === "nomic-bert" || arch === "jina-bert-v2") ? 20 : null,
        ffnUp: 21,
        ffnUpBias: arch === "bert" ? 22 : null,
        ffnDown: 23,
        ffnDownBias: (arch === "bert" || arch === "jina-bert-v2") ? 24 : null,
        ffnNormW: 25, ffnNormB: 26,
      })),
    };
    (enc as unknown as { buildGraph: (n: number) => unknown }).buildGraph(4);
    return {
      getrows: fake.ops.filter(o => o === "getrows").length,
      rope: fake.ops.filter(o => o === "rope").length,
      silu: fake.ops.filter(o => o === "silu").length,
      gelu: fake.ops.filter(o => o === "gelu").length,
      view3d: fake.ops.filter(o => o === "view3d").length,
      softmax_max_bias: fake.softmaxMaxBias,
    };
  }

  test("bert: pos-embedding + GeLU FFN, no rope, no silu, max_bias=0", () => {
    const r = buildAndCount("bert");
    expect(r.getrows).toBe(3);
    expect(r.rope).toBe(0);
    expect(r.silu).toBe(0);
    expect(r.gelu).toBe(2);
    expect(r.softmax_max_bias).toEqual([0, 0]);
  });

  test("jina-bert-v2: no pos-embedding, SwiGLU FFN, no rope, max_bias=8.0", () => {
    const r = buildAndCount("jina-bert-v2");
    expect(r.getrows).toBe(2);
    expect(r.rope).toBe(0);
    expect(r.silu).toBe(2);
    expect(r.gelu).toBe(0);
    expect(r.softmax_max_bias).toEqual([8.0, 8.0]);
  });
});
```

- [ ] **Step 12: Run buildGraph dispatch tests to verify they fail**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: FAIL — current buildGraph has no SwiGLU branch and always passes max_bias=0.0.

- [ ] **Step 13: Implement arch-branched `buildGraph` (Phase 2a — bert + jina paths)**

Add to imports:

```ts
import { getRopeModeForArchitecture } from "./model-inference.js";
```

Replace the body of `private buildGraph(nTokens: number): TensorPtr` (lines 172-240) with:

```ts
private buildGraph(nTokens: number): TensorPtr {
	if (!this.weights) throw new Error("weights not loaded");
	const { wasm, weights, hp } = this;
	const arch = hp.architecture;

	// Phase 2a: nomic-bert reaches loadWeights and throws there. If somehow
	// it gets here (no loadWeights in test path), fail loudly.
	if (arch === "nomic-bert") {
		throw new Error(
			"EncoderInference: nomic-bert forward not enabled until Phase 2b",
		);
	}

	const usesPosEmbedding = arch === "bert";
	const alibiMaxBias =
		arch === "jina-bert-v2" ? (hp.alibiMaxBias ?? 8.0) : 0.0;

	const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
	const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
	const segTensor = wasm.tensorNew1d(GgmlType.I32, 1);
	this.lastLeaves = { tokenIdsTensor, posTensor, segTensor };

	let x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
	if (usesPosEmbedding) {
		if (!weights.positionEmb) {
			throw new Error("bert path requires positionEmb weight");
		}
		x = wasm.opAdd(x, wasm.opGetRows(weights.positionEmb, posTensor));
	}
	x = wasm.opAdd(x, wasm.opGetRows(weights.tokenTypes, segTensor));
	x = this.layerNorm(x, weights.inputNormW, weights.inputNormB);

	const headDim = hp.embeddingHeadLength;
	const nHeads = hp.headCount;
	const E = hp.embeddingLength;
	const invSqrtHd = 1.0 / Math.sqrt(headDim);

	for (let il = 0; il < hp.layerCount; il++) {
		const lw = weights.layers[il];

		// Point B (split QKV path; fused arrives in Phase 2b)
		if (lw.qkvFused) {
			throw new Error(
				"EncoderInference: fused-QKV path not enabled until Phase 2b",
			);
		}
		if (!lw.qProj || !lw.kProj || !lw.vProj) {
			throw new Error(`split-QKV path requires qProj/kProj/vProj for ${arch}`);
		}
		let q = wasm.opMulMat(lw.qProj, x);
		let k = wasm.opMulMat(lw.kProj, x);
		let v = wasm.opMulMat(lw.vProj, x);
		if (lw.qBias) q = wasm.opAdd(q, lw.qBias);
		if (lw.kBias) k = wasm.opAdd(k, lw.kBias);
		if (lw.vBias) v = wasm.opAdd(v, lw.vBias);
		const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
		const k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
		const v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);

		// Point C (RoPE) — only nomic-bert; not exercised in Phase 2a.

		const qp = wasm.opPermute(q3, 0, 2, 1, 3);
		const kp = wasm.opPermute(k3, 0, 2, 1, 3);
		const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

		// Point D (softmax with optional ALiBi)
		const qk = wasm.opMulMat(kp, qp);
		const aw = wasm.opSoftMaxExt(qk, 0, invSqrtHd, alibiMaxBias);

		// Point E (attention output with optional bias)
		const out = wasm.opMulMat(vp, aw);
		const merged = wasm.opReshape2d(
			wasm.opCont(wasm.opPermute(out, 0, 2, 1, 3)),
			E,
			nTokens,
		);
		let attnProj = wasm.opMulMat(lw.oProj, merged);
		if (lw.oBias) attnProj = wasm.opAdd(attnProj, lw.oBias);

		x = this.layerNorm(wasm.opAdd(x, attnProj), lw.attnNormW, lw.attnNormB);

		// Point F (FFN — GeLU two-layer for bert, SwiGLU for jina/nomic)
		let ffnOut: TensorPtr;
		if (lw.ffnGate) {
			// SwiGLU: silu(gate(x)) * up(x), then down(...)
			const gate = wasm.opMulMat(lw.ffnGate, x);
			let up = wasm.opMulMat(lw.ffnUp, x);
			if (lw.ffnUpBias) up = wasm.opAdd(up, lw.ffnUpBias);
			const mid = wasm.opMul(wasm.opSilu(gate), up);
			ffnOut = wasm.opMulMat(lw.ffnDown, mid);
			if (lw.ffnDownBias) ffnOut = wasm.opAdd(ffnOut, lw.ffnDownBias);
		} else {
			// GeLU two-layer (bert)
			let h = wasm.opMulMat(lw.ffnUp, x);
			if (lw.ffnUpBias) h = wasm.opAdd(h, lw.ffnUpBias);
			h = wasm.opGelu(h);
			ffnOut = wasm.opMulMat(lw.ffnDown, h);
			if (lw.ffnDownBias) ffnOut = wasm.opAdd(ffnOut, lw.ffnDownBias);
		}

		x = this.layerNorm(wasm.opAdd(x, ffnOut), lw.ffnNormW, lw.ffnNormB);
	}

	return x;
}
```

- [ ] **Step 14: Run buildGraph dispatch tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: PASS (2 sub-tests — bert + jina; nomic test added in Phase 2b).

- [ ] **Step 15: Broaden engine routing in `src/core/engine.ts:589`**

Replace:

```ts
const isEncoder = parsed.hyperparams.architecture === "bert";
```

with:

```ts
const isEncoder = isEncoderArchitecture(parsed.hyperparams.architecture);
```

Add to engine.ts imports (merge into the existing `from "./types.js"` import block):

```ts
import { isEncoderArchitecture } from "./types.js";
```

- [ ] **Step 16: Run full ship gate**

Run: `make checkall`
Expected: PASS. Bert path bit-identical (BGE / Arctic-Embed exercise the bert branch in Point F); jina path now exercisable; nomic constructor-accepts but throws on `loadWeights`.

- [ ] **Step 17: Commit Phase 2a**

```bash
git add src/inference/encoder-inference.ts \
        src/core/engine.ts \
        tests/encoder-inference.test.ts
git commit -m "$(cat <<'EOF'
feat(encoder): arch-branched forward graph for jina-bert-v2 + bert

Phase 2a of bucket B. EncoderInference accepts all three encoder arches
at the constructor; bert + jina-bert-v2 are forward-runnable; nomic-bert
constructor-accepts but throws inside loadWeights/buildGraph with
"not enabled until Phase 2b".

- EncoderLayerWeights becomes nullable across bias/gate fields; new
  qkvFused field reserved for nomic
- New makeTensorOptional helper returns null instead of throwing on
  absent tensors
- loadWeights branches per-arch: bert preserves strict load; jina
  loads split QKV with biases + SwiGLU ffn_gate + mixed-bias FFN
  (ffn_up no bias, ffn_down has bias); nomic throws stub
- buildGraph branches at four points:
    Point A: bert-only pos-embedding sum
    Point D: softmax with hp.alibiMaxBias (jina) or 0.0 (bert)
    Point E: attn output bias (gated)
    Point F: SwiGLU (silu(gate)*up + down) for jina, GeLU two-layer for bert
  Points B-fused + C (RoPE) throw stub for nomic (Phase 2b lands them)
- engine.ts:589 routing broadens to isEncoderArchitecture()

bert path bit-identical (BGE + Arctic-Embed bench rows unchanged).
make checkall green.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §3
EOF
)"
```

---

## Task 4: Phase 2b — Encoder forward (nomic-bert fused-QKV + RoPE) (commit 3 of 6)

**Files:**
- Modify: `src/inference/encoder-inference.ts` (`loadWeights` adds nomic-bert arm; `buildGraph` adds Points B-fused + C RoPE)
- Modify: `tests/encoder-inference.test.ts` (loadWeights nomic positive test; buildGraph nomic dispatch test)

**Purpose:** Land the nomic-bert structural deltas on top of Phase 2a's foundation. Fused QKV is a single `opMulMat(attn_qkv.weight, x)` followed by three `opView3d` slices. RoPE applies `opRope` to Q and K.

### Phase 2b high-level approach

Two surgical changes in this commit:

1. **`loadWeights` nomic arm.** Replaces the throw stub. Loads `attn_qkv.weight`, `attn_output.weight`, and the SwiGLU FFN tensors with all biases set to null (nomic has none).
2. **`buildGraph` Points B-fused + C.** Detects `lw.qkvFused !== null` and enters the fused-slicing path; otherwise stays on the split path from Phase 2a. RoPE applies post-reshape to Q3 and K3.

After Phase 2b: all three arches forward-runnable. Bert + jina paths bit-identical with Phase 2a. Nomic path runs end-to-end (modulo Phase 4 model registration). `make checkall` clean.

### Phase 2b TDD steps

- [ ] **Step 1: Replace the nomic-bert throw stub in the existing loadWeights test**

In `tests/encoder-inference.test.ts`, find the `test("nomic-bert: throws (Phase 2b not landed)", …)` block from Phase 2a. Replace it with:

```ts
test("nomic-bert: fused QKV; no biases; SwiGLU gate", () => {
  const fake = makeFakeWasm();
  const enc = new EncoderInference(fake.fake, makeHp("nomic-bert"));
  enc.loadWeights(makeCtx([
    "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
    "token_types.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_output.weight",
    "blk.0.attn_output_norm.weight", "blk.0.attn_output_norm.bias",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "blk.0.layer_output_norm.weight", "blk.0.layer_output_norm.bias",
  ]), new Uint8Array(0));
  const layers = (enc as unknown as { weights: { layers: EncoderLayerWeights[] } }).weights.layers;
  expect(layers[0].qkvFused).not.toBeNull();
  expect(layers[0].qProj).toBeNull();
  expect(layers[0].qBias).toBeNull();
  expect(layers[0].kProj).toBeNull();
  expect(layers[0].kBias).toBeNull();
  expect(layers[0].vProj).toBeNull();
  expect(layers[0].vBias).toBeNull();
  expect(layers[0].oBias).toBeNull();
  expect(layers[0].ffnGate).not.toBeNull();
  expect(layers[0].ffnUpBias).toBeNull();
  expect(layers[0].ffnDownBias).toBeNull();
});
```

- [ ] **Step 2: Run nomic loadWeights test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "loadWeights arch dispatch"`
Expected: FAIL — current nomic arm throws.

- [ ] **Step 3: Replace the nomic throw stub in `loadWeights`**

In `src/inference/encoder-inference.ts`, replace the Phase 2a body — the section starting at `if (hp.architecture === "nomic-bert") { throw …` — with:

```ts
const isFused = hp.architecture === "nomic-bert";
const layers: EncoderLayerWeights[] = [];
for (let i = 0; i < hp.layerCount; i++) {
	const p = (s: string) => `blk.${i}.${s}`;
	layers.push({
		qkvFused: isFused ? this.makeTensor(tensorMap, p("attn_qkv.weight")) : null,
		qProj: isFused ? null : this.makeTensor(tensorMap, p("attn_q.weight")),
		qBias: isFused ? null : this.makeTensorOptional(tensorMap, p("attn_q.bias")),
		kProj: isFused ? null : this.makeTensor(tensorMap, p("attn_k.weight")),
		kBias: isFused ? null : this.makeTensorOptional(tensorMap, p("attn_k.bias")),
		vProj: isFused ? null : this.makeTensor(tensorMap, p("attn_v.weight")),
		vBias: isFused ? null : this.makeTensorOptional(tensorMap, p("attn_v.bias")),
		oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
		oBias: this.makeTensorOptional(tensorMap, p("attn_output.bias")),
		attnNormW: this.makeTensor(tensorMap, p("attn_output_norm.weight")),
		attnNormB: this.makeTensor(tensorMap, p("attn_output_norm.bias")),
		ffnGate: this.makeTensorOptional(tensorMap, p("ffn_gate.weight")),
		ffnUp: this.makeTensor(tensorMap, p("ffn_up.weight")),
		ffnUpBias: this.makeTensorOptional(tensorMap, p("ffn_up.bias")),
		ffnDown: this.makeTensor(tensorMap, p("ffn_down.weight")),
		ffnDownBias: this.makeTensorOptional(tensorMap, p("ffn_down.bias")),
		ffnNormW: this.makeTensor(tensorMap, p("layer_output_norm.weight")),
		ffnNormB: this.makeTensor(tensorMap, p("layer_output_norm.bias")),
	});
}
```

- [ ] **Step 4: Run nomic loadWeights test to verify it passes**

Run: `bun test tests/encoder-inference.test.ts -t "loadWeights arch dispatch"`
Expected: PASS (3 sub-tests: bert, jina, nomic-loads).

- [ ] **Step 5: Add the nomic-bert buildGraph dispatch test**

In `tests/encoder-inference.test.ts`, append to the `describe("EncoderInference.buildGraph arch dispatch", …)` block:

```ts
test("nomic-bert: fused QKV (3 view3d/layer) + RoPE (Q+K/layer) + SwiGLU, max_bias=0", () => {
  const r = buildAndCount("nomic-bert");
  expect(r.getrows).toBe(2);
  expect(r.view3d).toBe(6);    // 2 layers × 3 slices (Q, K, V)
  expect(r.rope).toBe(4);      // 2 layers × (Q + K)
  expect(r.silu).toBe(2);
  expect(r.gelu).toBe(0);
  expect(r.softmax_max_bias).toEqual([0, 0]);
});
```

- [ ] **Step 6: Run nomic buildGraph test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: FAIL — current nomic path throws "fused-QKV path not enabled until Phase 2b".

- [ ] **Step 7: Implement Points B-fused + C in `buildGraph`**

In `src/inference/encoder-inference.ts`, in the `buildGraph` body:

(a) Remove the early nomic-bert throw stub:

```ts
if (arch === "nomic-bert") {
	throw new Error(
		"EncoderInference: nomic-bert forward not enabled until Phase 2b",
	);
}
```

(b) Update the per-arch dispatch line to include `usesRope` and `ropeMode`:

```ts
const usesPosEmbedding = arch === "bert";
const usesRope = arch === "nomic-bert";
const alibiMaxBias =
	arch === "jina-bert-v2" ? (hp.alibiMaxBias ?? 8.0) : 0.0;
const ropeMode = usesRope ? getRopeModeForArchitecture(arch) : 0;
```

(c) Replace the per-layer split-QKV block (introduced in Phase 2a) with the fused/split branching + RoPE. Replace this Phase 2a code:

```ts
if (lw.qkvFused) {
	throw new Error(
		"EncoderInference: fused-QKV path not enabled until Phase 2b",
	);
}
if (!lw.qProj || !lw.kProj || !lw.vProj) {
	throw new Error(`split-QKV path requires qProj/kProj/vProj for ${arch}`);
}
let q = wasm.opMulMat(lw.qProj, x);
let k = wasm.opMulMat(lw.kProj, x);
let v = wasm.opMulMat(lw.vProj, x);
if (lw.qBias) q = wasm.opAdd(q, lw.qBias);
if (lw.kBias) k = wasm.opAdd(k, lw.kBias);
if (lw.vBias) v = wasm.opAdd(v, lw.vBias);
const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
const k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
const v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);

// Point C (RoPE) — only nomic-bert; not exercised in Phase 2a.
```

with:

```ts
let q3: TensorPtr;
let k3: TensorPtr;
let v3: TensorPtr;

if (lw.qkvFused) {
	// Point B (fused QKV): one matmul → 3 view3d slices.
	// Mirrors llama.cpp/src/llama-graph.cpp:1088-1095 (build_qkv).
	const qkv = wasm.opMulMat(lw.qkvFused, x);   // [3*E, nTokens]
	const elemSize = 4;                          // matmul output is fp32
	const headBytes = elemSize * headDim;
	const tokenBytes = elemSize * 3 * E;
	q3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, 0);
	k3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, elemSize * E);
	v3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, elemSize * 2 * E);
} else {
	// Point B (split QKV)
	if (!lw.qProj || !lw.kProj || !lw.vProj) {
		throw new Error(`split-QKV path requires qProj/kProj/vProj for ${arch}`);
	}
	let q = wasm.opMulMat(lw.qProj, x);
	let k = wasm.opMulMat(lw.kProj, x);
	let v = wasm.opMulMat(lw.vProj, x);
	if (lw.qBias) q = wasm.opAdd(q, lw.qBias);
	if (lw.kBias) k = wasm.opAdd(k, lw.kBias);
	if (lw.vBias) v = wasm.opAdd(v, lw.vBias);
	q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
	k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
	v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);
}

// Point C (RoPE) — nomic-bert only
if (usesRope) {
	q3 = wasm.opRope(
		q3, posTensor, headDim, ropeMode, hp.contextLength,
		hp.ropeFreqBase, hp.ropeScale,
		0.0, 1.0, 0.0, 0.0,
	);
	k3 = wasm.opRope(
		k3, posTensor, headDim, ropeMode, hp.contextLength,
		hp.ropeFreqBase, hp.ropeScale,
		0.0, 1.0, 0.0, 0.0,
	);
}
```

(The rest of the per-layer body — Points D/E/F — stays exactly as in Phase 2a.)

- [ ] **Step 8: Run nomic buildGraph test to verify it passes**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: PASS (3 sub-tests: bert, jina, nomic).

- [ ] **Step 9: Run full ship gate**

Run: `make checkall`
Expected: PASS. Bert + jina paths bit-identical with Phase 2a output. Nomic now exercisable end-to-end.

- [ ] **Step 10: Commit Phase 2b**

```bash
git add src/inference/encoder-inference.ts \
        tests/encoder-inference.test.ts
git commit -m "$(cat <<'EOF'
feat(encoder): nomic-bert fused-QKV + RoPE forward path

Phase 2b of bucket B. Lands the nomic-bert structural deltas on top of
Phase 2a's foundation:
- loadWeights nomic-bert arm: loads attn_qkv.weight (fused), no biases,
  SwiGLU FFN; replaces Phase 2a throw stub
- buildGraph Point B fused: one opMulMat(qkvFused, x) + 3 opView3d slices
  for Q/K/V. Byte arithmetic per llama.cpp/src/llama-graph.cpp:1088-1095:
    nb1 = elemSize * headDim
    nb2 = elemSize * 3 * E
    offsets 0, elemSize*E, elemSize*2*E
- buildGraph Point C RoPE: opRope on Q3 and K3 post-reshape, NORMAL mode,
  freq_base from GGUF metadata (= 1000 for nomic-embed-v1.5, NOT 10000).

All three encoder arches forward-runnable. Bert + jina paths bit-
identical with Phase 2a. make checkall green.

No GGUF available end-to-end at this commit — that lands Phase 3
(jina) and Phase 4 (nomic).

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §3
EOF
)"
```

---

## Task 5: Phase 3a — Reference-vector capture (one-shot Python)

**Files:**
- Create: `eval/reports/encoder-parity-2026-04-28/capture-refs.py`
- Create: `eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt`
- Create: `eval/reports/encoder-parity-2026-04-28/nomic-ref.json`
- Create: `eval/reports/encoder-parity-2026-04-28/jina-ref.json`

**Purpose:** Generate the numerical-parity reference vectors from `sentence-transformers`. One-shot off-process work; results checked in. Out-of-band commit (not one of the 6 main commits) so a `git revert` of any phase doesn't take the reference vectors with it.

- [ ] **Step 1: Write the requirements pin**

Create `eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt`:

```
sentence-transformers>=3.0,<4.0
einops>=0.7
```

(`einops` is a transitive dependency of nomic's `trust_remote_code=True` config.)

- [ ] **Step 2: Write the reference-capture script**

Create `eval/reports/encoder-parity-2026-04-28/capture-refs.py`:

```python
"""
One-shot reference-embedding capture for the encoder-parity probe.
Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: nomic-ref.json, jina-ref.json — each a list of {"input": str, "vec": [float, ...]}.
"""
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

models = [
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", {"trust_remote_code": True}),
    ("jina",  "jinaai/jina-embeddings-v2-base-en", {"trust_remote_code": True}),
]

for tag, name, kwargs in models:
    print(f"Loading {name}…", file=sys.stderr)
    m = SentenceTransformer(name, **kwargs)
    vecs = m.encode(inputs, normalize_embeddings=True)
    out = [
        {"input": inputs[i], "vec": [float(x) for x in vecs[i].tolist()]}
        for i in range(len(inputs))
    ]
    out_path = HERE / f"{tag}-ref.json"
    out_path.write_text(json.dumps(out))
    print(f"  wrote {out_path} ({len(out)} vectors, dim={len(out[0]['vec'])})", file=sys.stderr)
```

- [ ] **Step 3: Run the capture script**

```bash
cd eval/reports/encoder-parity-2026-04-28 && \
  uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
```

Expected: stderr shows "Loading nomic-ai/nomic-embed-text-v1.5…" and "wrote nomic-ref.json (5 vectors, dim=768)" / "wrote jina-ref.json (5 vectors, dim=768)". Exit 0.

- [ ] **Step 4: Sanity-check the captured vectors**

```bash
bun -e '
const nomic = JSON.parse(await Bun.file("eval/reports/encoder-parity-2026-04-28/nomic-ref.json").text());
const jina = JSON.parse(await Bun.file("eval/reports/encoder-parity-2026-04-28/jina-ref.json").text());
for (const [tag, refs] of [["nomic", nomic], ["jina", jina]]) {
  console.log(tag, "rows:", refs.length, "dim:", refs[0].vec.length);
  for (const r of refs) {
    const norm = Math.sqrt(r.vec.reduce((a,b) => a + b*b, 0));
    console.log("  norm:", norm.toFixed(6), "input:", JSON.stringify(r.input).slice(0, 60));
  }
}
'
```

Expected: each row's L2 norm is `1.000000 ± 1e-6`. Both files: 5 rows, 768-dim.

- [ ] **Step 5: Commit Phase 3a**

```bash
git add eval/reports/encoder-parity-2026-04-28/capture-refs.py \
        eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt \
        eval/reports/encoder-parity-2026-04-28/nomic-ref.json \
        eval/reports/encoder-parity-2026-04-28/jina-ref.json
git commit -m "$(cat <<'EOF'
test(encoder): pin parity reference vectors from sentence-transformers

Phase 3a of bucket B. One-shot capture of 5 reference embeddings per
model from sentence-transformers (nomic-embed-text-v1.5 +
jina-embeddings-v2-base-en) under the same 5 input strings the encoder
parity harness uses.

Reproducible via:
  uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
under eval/reports/encoder-parity-2026-04-28/.

These vectors gate Phase 3 (jina) and Phase 4 (nomic) — per-row cosine
vs the in-tree implementation must clear 0.999 before each model
registration lands.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 6: Phase 3 — Parity harness + jina registration + parity gate (commit 4 of 6)

**Files:**
- Create: `eval/encoder-parity.ts` (browser-driven via agentchrome, reuses `eval/browser-smoke.ts` helpers)
- Modify: `eval/models.ts` (jina entry, mirroring BGE-base shape at lines 393-410)
- Modify: `eval/smoke-profiles.ts` (jina profile entry, mirroring `bge-large` at lines 273-276)
- Modify: `eval/smoke-profiles.ts:355` (add `"jina-embeddings-v2-base-en"` to `embeddings:` selector array)
- Create: `eval/reports/encoder-parity-2026-04-28/SUMMARY.md`

**Purpose:** Build the parity harness and register jina (the simpler-of-the-two model). Parity gate ≥0.999 cosine on all 5 rows before commit.

- [ ] **Step 1: Write the parity harness (browser-driven via agentchrome)**

Bun has no native WebGPU, so the harness drives the browser smoke page (`smoke-test/real-model.html`) via `agentchrome` — same pattern as `eval/embed-perf.ts`. The smoke page exposes `window.engine` and `window.handleId` after model load (see `smoke-test/real-model-page.js:472`); the harness uses `agentchrome js-exec` to call `engine.embed(handleId, text)` and capture each vector.

Create `eval/encoder-parity.ts`:

```ts
#!/usr/bin/env bun
/**
 * Encoder parity harness. Drives the smoke-test page via agentchrome,
 * calls window.engine.embed(handleId, text) for each fixture input,
 * compares cosine similarity vs sentence-transformers reference.
 * Pass gate: cosine >= 0.999 on every row.
 *
 * Run:
 *   bun eval/encoder-parity.ts <modelId> <ref-file>
 *
 * Requires:
 *   - smoke server up (`make smoke-serve`)
 *   - running agentchrome session
 */
import { readFileSync } from "node:fs";
import {
  agentchrome,
  buildSmokeTestUrl,
  ensureModelDownloaded,
  ensureSmokeServerReachable,
  resolveAgentchromeSession,
} from "./browser-smoke.js";
import { getModelById } from "./models.js";

const COSINE_GATE = 0.999;

const [, , modelId, refPath] = process.argv;
if (!modelId || !refPath) {
  console.error("Usage: bun eval/encoder-parity.ts <modelId> <ref-file>");
  process.exit(2);
}

interface Ref { input: string; vec: number[]; }

const refs: Ref[] = JSON.parse(readFileSync(refPath, "utf8"));
const inputs = refs.map(r => r.input);

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const model = getModelById(modelId);
if (!model) throw new Error(`Unknown modelId "${modelId}" — register it in eval/models.ts first.`);

await ensureSmokeServerReachable();
const { port, tab } = await resolveAgentchromeSession();
await ensureModelDownloaded(model);

const url = buildSmokeTestUrl(modelId, model.contextLength, {
  // Reuse the existing embedPerf path to trigger model load. Once the page
  // exposes window.handleId, the parity harness drives engine.embed via
  // agentchrome js-exec — no smoke-page changes needed.
  extraParams: { embedPerf: "single", embedFixture: "short", embedReps: "1" },
});
await agentchrome(["--port", port, "--tab", tab, "navigate", url]);

// Wait for model load: page sets window.handleId after engine.loadModel
// resolves (real-model-page.js:~470). Poll up to 120s.
const loadDeadline = Date.now() + 120_000;
while (Date.now() < loadDeadline) {
  const ready = await agentchrome([
    "--port", port, "--tab", tab,
    "js-exec", "(() => Boolean(window.handleId && window.engine))()",
  ]);
  if (ready.trim() === "true") break;
  await new Promise(r => setTimeout(r, 500));
}

console.log(`Loaded ${modelId}`);

let pass = 0;
const rows: { idx: number; input: string; cos: number; ok: boolean }[] = [];
for (let i = 0; i < inputs.length; i++) {
  const inputJson = JSON.stringify(inputs[i]);
  const script = `(async () => {
    const text = ${inputJson};
    const v = await window.engine.embed(window.handleId, text);
    return Array.from(v);
  })()`;
  const raw = await agentchrome([
    "--port", port, "--tab", tab, "js-exec", script,
  ]);
  const vec = JSON.parse(raw) as number[];
  if (vec.length !== refs[i].vec.length) {
    throw new Error(
      `dim mismatch row ${i}: got ${vec.length}, ref ${refs[i].vec.length}`,
    );
  }
  const cos = cosine(vec, refs[i].vec);
  const ok = cos >= COSINE_GATE;
  if (ok) pass++;
  rows.push({ idx: i, input: inputs[i], cos, ok });
  console.log(
    `  row ${i}  cos=${cos.toFixed(6)}  ${ok ? "PASS" : "FAIL"}  ${
      JSON.stringify(inputs[i]).slice(0, 60)
    }`,
  );
}

console.log(`\n${pass}/${inputs.length} rows passed (gate >= ${COSINE_GATE})`);
process.exit(pass === inputs.length ? 0 : 1);
```

**Verify these helper exports in `eval/browser-smoke.ts` before running:**
- `agentchrome(args: string[]): Promise<string>` — wraps the `agentchrome` CLI; returns stdout.
- `buildSmokeTestUrl(modelId, ctx, { extraParams })` — already used by `eval/embed-perf.ts:74-79`.
- `ensureModelDownloaded`, `ensureSmokeServerReachable`, `resolveAgentchromeSession`.

If `agentchrome js-exec` is not the actual subcommand (try `agentchrome --help`), substitute the real subcommand (`evaluate`, `runtime evaluate`, etc.). The smoke page's globals are `window.handleId` and `window.engine`; verify by grepping `smoke-test/real-model-page.js`.

- [ ] **Step 2: Register jina in `eval/models.ts`**

Add immediately after the `bge-large-en-v1.5-q0f16` entry (after line 410):

```ts
{
  id: "jina-embeddings-v2-base-en-q0f16",
  name: "Jina Embeddings v2 Base EN",
  family: "Jina Embeddings",
  architecture: "jina-bert-v2",
  paramsB: 0.137,
  vramMB: 320,
  defaultQuant: "q0f16",
  availableQuants: ["q0f16"],
  capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
  license: "Apache-2.0",
  contextLength: 8192,
  tier: "ultrafast",
  requiresShaderF16: false,
  downloadUrl: "https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
  ggufUrl: "https://huggingface.co/gaianet/jina-embeddings-v2-base-en-GGUF",
  ggufFilePattern: "f16",   // matches `jina-embeddings-v2-base-en-f16.gguf`
},
```

(The Phase 0 probe confirmed this mirror works.)

- [ ] **Step 3: Register jina profile in `eval/smoke-profiles.ts`**

Add after the `bge-large` profile (after line 276):

```ts
{
  name: "jina-embeddings-v2-base-en",
  model: "jina-embeddings-v2-base-en-q0f16",
  embedding: true,
},
```

In the embedding-profile selector around lines 350-355, append `"jina-embeddings-v2-base-en"` to the `embeddings:` array (and to whichever sibling array carries the broader embedding fleet listing — verify by grepping for `"bge-large"` in that file).

- [ ] **Step 4: Run `make checkall` to confirm registration types are clean**

Run: `make checkall`
Expected: PASS.

- [ ] **Step 5: Run the parity gate for jina**

Start the smoke server: `make smoke-serve`.

Run: `bun eval/encoder-parity.ts jina-embeddings-v2-base-en-q0f16 eval/reports/encoder-parity-2026-04-28/jina-ref.json`

Expected: 5/5 rows PASS with cosine ≥ 0.999. Exit 0.

If any row fails, diagnose in this order:
1. **`alibiMaxBias` value**: log `hp.alibiMaxBias` at engine init. Default 8.0 should be correct for 12-head models.
2. **SwiGLU operand order**: `silu(gate) * up` is correct; `up * silu(gate)` is also correct (commutative); `silu(up) * gate` is wrong.
3. **`ffn_down.bias` add path**: missing this collapses cosine to ~0.4-0.5.
4. **Pooling**: confirm `hp.poolingType === "mean"`.
5. **ALiBi sign inverted**: if all rows show ~0.7 collapse and shorter inputs pass, escalate to user (ggml-side fix needed).

- [ ] **Step 6: Browser smoke (Gate 4) for jina**

```bash
agentchrome connect --status
agentchrome --port <PORT> tabs list
agentchrome --port <PORT> --tab <TAB_ID> navigate \
  "http://localhost:8031/?v=$(date +%s)&model=jina-embeddings-v2-base-en-q0f16&embed=Hello+world."
```

Expected: page `#log` shows successful download + load + embed; no console errors except benign `adapter_info:`.

- [ ] **Step 7: Write Phase 3 SUMMARY.md (jina row only; nomic row filled in Phase 4)**

Create `eval/reports/encoder-parity-2026-04-28/SUMMARY.md`:

```markdown
# Encoder parity probe — 2026-04-28

## Inputs

5 fixed strings in `inputs.json`. Reference embeddings captured via
sentence-transformers in `capture-refs.py`. Per-row gate: cosine >= 0.999.

## jina-embeddings-v2-base-en (BERT + ALiBi + SwiGLU)

| Row | Input (truncated) | Cosine | Pass |
|----:|---|---:|:---:|
| 0 | Hello world. | <fill> | <Y/N> |
| 1 | The quick brown fox jumps … | <fill> | <Y/N> |
| 2 | Embedding models map text … | <fill> | <Y/N> |
| 3 | Café — naïve façade … | <fill> | <Y/N> |
| 4 | . | <fill> | <Y/N> |

**Result:** <X>/5 rows passed.
**alibiMaxBias:** 8.0 (default; gaianet GGUF mirror omits the metadata key).

## nomic-embed-text-v1.5 (BERT + RoPE + SwiGLU + fused-QKV)

(filled in Phase 4 / Task 7)

## Methodology notes

- Reference vectors normalized via sentence-transformers
  `normalize_embeddings=True`.
- WebLLM-side vectors emerge from `engine.embed()` already
  L2-normalized via `EncoderInference.poolAndNormalize`.
- Cosine computed in TS (`eval/encoder-parity.ts`), F32 accumulator.
- 0.999 gate is a soft floor: BGE shows >0.9999 informally on this
  metric. RoPE / ALiBi / SwiGLU degenerate-alignment failure modes
  typically show as <0.95 (often <0.5) — well below the gate.
```

Fill actual cosine values from the harness output. **The gate must show Y on all rows before committing.**

- [ ] **Step 8: Commit Phase 3**

```bash
git add eval/encoder-parity.ts \
        eval/models.ts \
        eval/smoke-profiles.ts \
        eval/reports/encoder-parity-2026-04-28/SUMMARY.md
git commit -m "$(cat <<'EOF'
feat(eval): register jina-embeddings-v2-base-en + parity probe (5/5 >= 0.999)

Phase 3 of bucket B. Registers Jina Embeddings v2 Base EN (137M,
jina-bert-v2 = BERT + ALiBi + SwiGLU + mixed-bias FFN, 768-dim, 8192
ctx, mean pooling) + ships eval/encoder-parity.ts harness.

Parity gate: 5/5 fixed inputs hit cosine >= 0.999 vs sentence-transformers
reference. alibiMaxBias = 8.0 default (GGUF mirror omits the key); ggml's
opSoftMaxExt(..., max_bias) computes per-head linear bias internally.
SwiGLU FFN: silu(gate) * up + down + down_bias (gate/up have no biases).
Browser smoke (gate 4) green.

First encoder in fleet with ALiBi attention bias. Parity harness
reusable for nomic (lands phase 4) and any future encoder additions.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 7: Phase 4 — Nomic registration + parity gate (commit 5 of 6)

**Files:**
- Modify: `eval/models.ts` (nomic entry)
- Modify: `eval/smoke-profiles.ts` (nomic profile + selector array)
- Modify: `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` (fill nomic table)

**Purpose:** Same pattern as Task 6, for nomic. Smaller commit because the harness already exists.

- [ ] **Step 1: Register nomic in `eval/models.ts`**

Add immediately after the `jina-embeddings-v2-base-en-q0f16` entry:

```ts
{
  id: "nomic-embed-text-v1.5-q0f16",
  name: "Nomic Embed Text v1.5",
  family: "Nomic Embed",
  architecture: "nomic-bert",
  paramsB: 0.137,
  vramMB: 320,
  defaultQuant: "q0f16",
  availableQuants: ["q0f16"],
  capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
  license: "Apache-2.0",
  contextLength: 2048,
  tier: "ultrafast",
  requiresShaderF16: false,
  downloadUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
  ggufUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
  ggufFilePattern: "f16",   // matches `nomic-embed-text-v1.5.f16.gguf`
},
```

- [ ] **Step 2: Register nomic profile in `eval/smoke-profiles.ts`**

Add after the `jina-embeddings-v2-base-en` profile:

```ts
{
  name: "nomic-embed-text-v1.5",
  model: "nomic-embed-text-v1.5-q0f16",
  embedding: true,
},
```

Append `"nomic-embed-text-v1.5"` to the `embeddings:` selector array (and any sibling embedding fleet array, mirroring Task 6).

- [ ] **Step 3: Run `make checkall`**

Run: `make checkall`
Expected: PASS.

- [ ] **Step 4: Run the parity gate for nomic**

Run: `bun eval/encoder-parity.ts nomic-embed-text-v1.5-q0f16 eval/reports/encoder-parity-2026-04-28/nomic-ref.json`

Expected: 5/5 rows PASS with cosine ≥ 0.999.

If any row fails, diagnose in this order:
1. **Fused-QKV byte offsets**: this is the High-likelihood risk per the spec. Cross-check against `~/Repos/llama.cpp/src/llama-graph.cpp:1088-1095` exactly. The arithmetic is `nb1 = elemSize * headDim`, `nb2 = elemSize * 3 * E`, offsets `0`, `elemSize * E`, `elemSize * 2 * E`. Common bugs: confusing `nb1` and `nb2`, using `elemSize * E * N` (wrong — N stride is in nb2 already).
2. **RoPE mode**: try `RopeMode.NEOX` instead of `RopeMode.NORMAL`.
3. **freq_base mismatch**: log `hp.ropeFreqBase`; should be 1000. If 10000 fired, the loader fallback chain didn't read `nomic-bert.rope.freq_base` correctly.
4. **headDim**: should be 64 (12 heads × 64 = 768).
5. **SwiGLU operand order**: same checks as jina.
6. **Pooling**: confirm `hp.poolingType === "mean"`.

- [ ] **Step 5: Browser smoke (Gate 4) for nomic**

Same flow as Task 6 step 6, with `nomic-embed-text-v1.5-q0f16`.

- [ ] **Step 6: Fill the nomic row in `SUMMARY.md`**

Replace the `(filled in Phase 4 / Task 7)` stub in `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` with:

```markdown
## nomic-embed-text-v1.5 (BERT + RoPE + SwiGLU + fused-QKV)

| Row | Input (truncated) | Cosine | Pass |
|----:|---|---:|:---:|
| 0 | Hello world. | <fill> | <Y/N> |
| 1 | The quick brown fox jumps … | <fill> | <Y/N> |
| 2 | Embedding models map text … | <fill> | <Y/N> |
| 3 | Café — naïve façade … | <fill> | <Y/N> |
| 4 | . | <fill> | <Y/N> |

**Result:** <X>/5 rows passed.
**RoPE mode:** NORMAL (verified via Phase 0 + parity probe).
**freq_base:** 1000 (loaded from GGUF metadata; nomic-specific value, not 10000).
**Fused-QKV byte offsets:** nb1=4*headDim, nb2=4*3*E, offsets [0, 4*E, 4*2*E].
```

- [ ] **Step 7: Commit Phase 4**

```bash
git add eval/models.ts \
        eval/smoke-profiles.ts \
        eval/reports/encoder-parity-2026-04-28/SUMMARY.md
git commit -m "$(cat <<'EOF'
feat(eval): register nomic-embed-text-v1.5 + parity probe (5/5 >= 0.999)

Phase 4 of bucket B. Registers Nomic Embed Text v1.5 (137M, nomic-bert
= BERT + RoPE + SwiGLU + fused-QKV + no biases, 768-dim, 2048 ctx,
mean pooling).

Parity gate: 5/5 fixed inputs hit cosine >= 0.999. RoPE NORMAL,
freq_base = 1000 (from GGUF metadata; nomic-specific). Fused-QKV byte
offsets matched llama-graph.cpp:1088-1095 exactly. Browser smoke (gate
4) green.

First encoder in fleet with rotary positional embeddings AND fused-QKV
tensor structure. Bucket B's two-model design fully landed; Phase 5
runs bench-full and closes the bucket.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 8: Phase 5 — Bench-full + dashboard refresh + TODO close (commit 6 of 6)

**Files:**
- Modify: `TODO.md` (transition Bucket B from "queued"/"deferred" to "DONE 2026-MM-DD")

**Purpose:** Run the full eval suite to populate cosine-task and latency rows for both new models, refresh the dashboard, transition TODO.md.

- [ ] **Step 1: Start the live dashboard**

```bash
make dashboard-serve   # port 8033
```

Verify reachable: `curl -sf http://localhost:8033/ > /dev/null && echo OK`.

- [ ] **Step 2: Run `make bench-full`**

```bash
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 make bench-full
```

Expected: each registered profile runs its bench; results post live to the dashboard. Wall time ~30-45 min.

- [ ] **Step 3: Inspect the dashboard's Embeddings section**

Open `http://localhost:8033/`. Confirm 6 rows (arctic-embed-s/m, bge-small/large, jina, nomic). Snapshot via `agentchrome` → save to `eval/reports/encoder-parity-2026-04-28/dashboard-2026-MM-DD.png`.

- [ ] **Step 4: Update TODO.md — transition Bucket B**

Find Bucket B in TODO.md (around line 845). Update the heading from `(deferred)` to `**DONE 2026-MM-DD**` with the closure summary. Fill in commit short hashes, p50 latencies, cosine-task scores from bench-full.

- [ ] **Step 5: Run final ship gate**

Run: `make checkall`
Expected: PASS.

- [ ] **Step 6: Commit Phase 5 (closure)**

```bash
git add TODO.md \
        eval/reports/encoder-parity-2026-04-28/dashboard-2026-MM-DD.png
git commit -m "$(cat <<'EOF'
docs(TODO): close bucket B — jina + nomic encoder expansion landed

Phase 5 of bucket B (final commit). 10/10 parity rows green, dashboard
Embeddings section updated to 6 rows, bench-full populates cosine-task
and latency for both new models.

Bucket B retired. Bucket C (causal-LM-derived embedders) remains
deferred behind a deployment ask; bucket A + B together exercise the
full BERT-family encoder lever portfolio (split QKV, fused QKV, RoPE,
ALiBi, GeLU, SwiGLU, full biases, no biases, mixed biases).

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md
Plan: docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md
EOF
)"
```

- [ ] **Step 7: Stop the dashboard server**

```bash
make dashboard-stop
```

(or kill the bun process directly).

---

## Self-review checklist

After all 9 tasks complete, verify:

- [ ] Plan commit + Phase 0 + 6 phase commits + Phase 3a (8 commits total) present in `git log` since the v2 spec commit (`bf51912`).
- [ ] `make checkall` clean on `main`.
- [ ] Parity SUMMARY.md shows 10/10 PASS.
- [ ] Dashboard live at port 8033 shows 6 embedding rows with non-null cosine + latency.
- [ ] TODO.md's Bucket B entry transitioned to DONE with cycle-closure summary.
- [ ] No `<fill>`, `MM-DD`, `<X>`, or `#####` placeholders remain in any committed artifact.
