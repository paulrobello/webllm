# Encoder non-BERT architecture support — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `EncoderInference` to support `nomic-embed-text-v1.5` (BERT+RoPE, 137M, 768-dim) and `jina-embeddings-v2-base-en` (BERT+ALiBi, 137M, 768-dim) under one design / phased implementation.

**Architecture:** Both `opRope` and `opSoftMaxExt(max_bias)` bindings already exist — net surface ~200 LOC across 7 modified + 1 new file. ALiBi is essentially free (`max_bias > 0` triggers ggml's built-in per-head linear bias inside softmax). RoPE adds two `opRope` calls (Q and K) per layer, paired with skipping the position-embedding sum. `ModelArchitecture` enum widens with two new top-level entries (`"nomic-bert"`, `"jina-bert-v2"`); `isEncoderArchitecture()` helper centralizes the branch.

**Tech Stack:** TypeScript / Bun (orchestration), patched `llama.cpp` `ggml-webgpu` compiled to WASM (already built), `bun test` (unit tests), `make checkall` (fmt + lint + typecheck + test ship gate), `make bench-full` (end-to-end eval), `agentchrome` (browser smoke), `sentence-transformers` via `uv run --no-project` (one-shot reference-vector capture).

**Spec:** `docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`.

**Phasing:** 5 commits, each `make checkall` clean and independently revertable. Mirrors §17/§18/§19/§20 plan structure.

---

## Task 0: Commit the plan (per "always commit before work")

**Files:** `docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md` (this file).

**Purpose:** Land the plan as its own commit before any probe / code work begins, so subsequent `git revert`s of any phase commit don't take the plan with them. Mirrors the `cd5bdd2` / `7d0173c` separate-docs-commit cadence in the project's recent history.

- [ ] **Step 1: Force-add and commit the plan**

(`docs/superpowers/` is gitignored per CLAUDE.md — `git add -f` is required.)

```bash
git add -f docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md
git commit -m "$(cat <<'EOF'
docs(plan): bucket B encoder non-BERT arch implementation plan

7-task / 5-commit plan for landing nomic-embed-text-v1.5 (BERT+RoPE)
and jina-embeddings-v2-base-en (BERT+ALiBi). Mirrors the §17/§18/§19/§20
phased plan structure: Phase 0 GGUF probe + Phase 3a one-shot
sentence-transformers reference capture sit outside the 5 main commits
as separate artifact commits, so a git revert of any phase won't take
the probe data or reference vectors with it.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md
EOF
)"
```

---

## Task 1: Phase 0 — GGUF discovery probe

**Files:**
- Create: `eval/reports/encoder-parity-2026-04-28/probe-gguf.ts`
- Create: `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt`
- Create: `eval/reports/encoder-parity-2026-04-28/inputs.json`

**Purpose:** Resolve the three Phase 0-tagged unknowns from the spec (RoPE freq-base key spelling, ALiBi bias-max key spelling, presence/absence of `position_embd.weight`) **before** code changes land. Probe-first doctrine per CLAUDE.md.

- [ ] **Step 1: Write the input fixture (5 strings used by every gate)**

Create `eval/reports/encoder-parity-2026-04-28/inputs.json`:

```json
[
  "Hello world.",
  "The quick brown fox jumps over the lazy dog.",
  "Embedding models map text into a dense vector space where semantic similarity corresponds to cosine distance, enabling efficient retrieval over large corpora.",
  "Café — naïve façade résumé piñata coöperate. 你好世界. Здравствуй мир.",
  "."
]
```

(The 5th input "." reduces to `[CLS] . [SEP]` after tokenization — the smallest non-empty sequence. The 4th tests Unicode through WordPiece byte fallback.)

- [ ] **Step 2: Write the probe script**

Create `eval/reports/encoder-parity-2026-04-28/probe-gguf.ts`:

```ts
#!/usr/bin/env bun
/**
 * Phase 0 probe — download and inspect both candidate GGUFs.
 * Writes 00-gguf-discovery.txt summarizing metadata keys + tensor list.
 * Run: bun eval/reports/encoder-parity-2026-04-28/probe-gguf.ts
 */
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { parseGgufHeader } from "../../../src/models/gguf-parser.js";

const OUT_DIR = "eval/reports/encoder-parity-2026-04-28";
const CACHE_DIR = join(OUT_DIR, "cache");
mkdirSync(CACHE_DIR, { recursive: true });

interface Candidate {
  label: string;
  url: string;
}

const candidates: Candidate[] = [
  {
    label: "nomic-embed-text-v1.5",
    url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf",
  },
  {
    label: "jina-embeddings-v2-base-en",
    url: "https://huggingface.co/gaianet/jina-embeddings-v2-base-en-GGUF/resolve/main/jina-embeddings-v2-base-en-f16.gguf",
  },
];

const lines: string[] = [];
const log = (s: string) => {
  lines.push(s);
  console.log(s);
};

for (const c of candidates) {
  log(`\n=== ${c.label} ===`);
  log(`URL: ${c.url}`);
  const localPath = join(CACHE_DIR, `${c.label}.gguf`);
  if (!existsSync(localPath)) {
    log(`Downloading…`);
    const res = await fetch(c.url);
    if (!res.ok) {
      log(`FAILED: HTTP ${res.status} ${res.statusText}`);
      continue;
    }
    const buf = await res.arrayBuffer();
    writeFileSync(localPath, new Uint8Array(buf));
    log(`Saved ${buf.byteLength} bytes to ${localPath}`);
  } else {
    log(`Reusing cached ${localPath}`);
  }

  const bytes = new Uint8Array(await Bun.file(localPath).arrayBuffer());
  const ctx = parseGgufHeader(bytes);

  log(`general.architecture = ${JSON.stringify(ctx.metadata.get("general.architecture")?.value)}`);

  const archKeys: string[] = [];
  for (const k of ctx.metadata.keys()) {
    if (k.startsWith("nomic-bert.") || k.startsWith("jina-bert-v2.") ||
        k.startsWith("bert.") || k.startsWith("general.") ||
        k.startsWith("tokenizer.")) {
      archKeys.push(k);
    }
  }
  archKeys.sort();
  log(`Metadata keys (${archKeys.length}):`);
  for (const k of archKeys) {
    const v = ctx.metadata.get(k)?.value;
    const repr = typeof v === "string" || typeof v === "number" || typeof v === "boolean"
      ? String(v) : `<${typeof v}>`;
    log(`  ${k} = ${repr}`);
  }

  const tensorNames = ctx.tensors.map(t => t.name).sort();
  log(`Tensors (${tensorNames.length}):`);
  for (const n of tensorNames.slice(0, 50)) log(`  ${n}`);
  if (tensorNames.length > 50) log(`  … (${tensorNames.length - 50} more)`);

  const hasPosEmb = tensorNames.includes("position_embd.weight");
  log(`HAS position_embd.weight: ${hasPosEmb}`);
}

writeFileSync(join(OUT_DIR, "00-gguf-discovery.txt"), `${lines.join("\n")}\n`);
log(`\nWrote ${OUT_DIR}/00-gguf-discovery.txt`);
```

- [ ] **Step 3: Run the probe and capture output**

Run: `bun eval/reports/encoder-parity-2026-04-28/probe-gguf.ts`

Expected: produces `00-gguf-discovery.txt` containing:
- The exact `general.architecture` value for each model (`"nomic-bert"` and `"jina-bert-v2"` per spec assumptions, or alternatives that the rest of the plan must adapt to).
- The exact key spelling for RoPE freq-base on nomic and ALiBi bias-max on jina.
- A pass/fail line for `position_embd.weight` presence.

If the jina mirror URL fails (Phase 0 risk in the spec), search for an alternative: try `https://huggingface.co/api/models?search=jina-embeddings-v2-base-en+gguf` or convert from HF source via `cd ~/Repos/llama.cpp && python convert_hf_to_gguf.py`. Update the probe script's URL and re-run. Document the fallback path inline in `00-gguf-discovery.txt`.

- [ ] **Step 4: Diff the discovered keys against the spec's assumptions**

Open `00-gguf-discovery.txt` and verify:
- nomic-bert RoPE key matches one of `nomic-bert.rope.freq_base` or `nomic-bert.rope_freq_base`. Loader fallback chain at `src/models/model-loader.ts:95-98` already covers both — record which one is in use.
- jina-bert-v2 ALiBi key matches `jina-bert-v2.attention.alibi_bias_max`. If different (e.g. `.alibi_max_bias`), update the key string in Task 2's loader change.
- `position_embd.weight` is **absent** for both nomic and jina — this is the expected case from the spec; presence-but-unused is also OK (loader skips it).

If any divergence appears, edit Task 2's metadata-key strings inline before proceeding.

- [ ] **Step 5: Commit Phase 0 artifact**

```bash
git add eval/reports/encoder-parity-2026-04-28/probe-gguf.ts \
        eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt \
        eval/reports/encoder-parity-2026-04-28/inputs.json
git commit -m "$(cat <<'EOF'
test(encoder): phase 0 GGUF discovery probe (nomic + jina)

Phase 0 of bucket B. Captures exact metadata keys + tensor lists for
nomic-embed-text-v1.5 and jina-embeddings-v2-base-en, resolving the
three spec-tagged unknowns before any code lands:
- nomic-bert RoPE freq-base key spelling -> see 00-gguf-discovery.txt
- jina-bert-v2 ALiBi bias-max key spelling -> see 00-gguf-discovery.txt
- position_embd.weight presence/absence -> see 00-gguf-discovery.txt

Probe artifacts pinned at eval/reports/encoder-parity-2026-04-28/.
inputs.json is the canonical 5-row fixture used by every parity gate.
EOF
)"
```

---

## Task 2: Phase 1 — Types, loader, RoPE helper (commit 1 of 5)

**Files:**
- Modify: `src/core/types.ts:50-60` (extend `ModelArchitecture` union)
- Modify: `src/core/types.ts:95-118` (add `alibiMaxBias?: number` field; export `ENCODER_ARCHITECTURES` + `isEncoderArchitecture()`)
- Modify: `src/models/model-loader.ts:40-106` (broaden `arch === "bert"` branch to `isEncoderArchitecture(arch)`; read jina ALiBi metadata)
- Modify: `src/inference/model-inference.ts:83-89` (extend `getRopeModeForArchitecture`)
- Modify: `tests/encoder-inference.test.ts` (test #1 metadata routing; test #2 helper truth-table)

**Purpose:** Type-system + loader-side foundation. After this task `make checkall` passes, but no encoder forward changes yet — BERT path identical, new archs not yet routable through engine.

- [ ] **Step 1: Write the failing test for `isEncoderArchitecture` helper**

Add to `tests/encoder-inference.test.ts` (after the existing `describe` block, line 28 or end-of-file):

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
Expected: FAIL with `Module '"../src/core/types.js"' has no exported member 'isEncoderArchitecture'` (or similar import error).

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
	 * Defaults to 8.0 if metadata omits it.
	 */
	alibiMaxBias?: number;
```

- [ ] **Step 5: Run helper tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "isEncoderArchitecture"`
Expected: PASS (all 3 sub-tests).

- [ ] **Step 6: Write the failing test for loader metadata routing**

Add to `tests/encoder-inference.test.ts` (in a new `describe` block):

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
  return {
    metadata: meta,
    tensors: [],
    dataOffset: 0,
    totalDataSize: 0,
  } as unknown as GgufContext;
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
  test("jina-bert-v2 produces ALiBi-ready hyperparams", () => {
    const ctx = fakeCtx("jina-bert-v2", {
      "jina-bert-v2.attention.alibi_bias_max": 8.0,
    });
    const hp = (ModelLoader as unknown as { extractHyperparams: (c: unknown) => unknown })
      .extractHyperparams(ctx) as Record<string, unknown>;
    expect(hp.architecture).toBe("jina-bert-v2");
    expect(hp.causalAttention).toBe(false);
    expect(hp.poolingType).toBe("mean");
    expect(hp.alibiMaxBias).toBe(8.0);
  });
  test("jina-bert-v2 falls back to alibiMaxBias=8.0 when metadata absent", () => {
    const ctx = fakeCtx("jina-bert-v2");
    const hp = (ModelLoader as unknown as { extractHyperparams: (c: unknown) => unknown })
      .extractHyperparams(ctx) as Record<string, unknown>;
    expect(hp.alibiMaxBias).toBe(8.0);
  });
});
```

(`extractHyperparams` is currently `private static`. Tests reach it via the same cast pattern used elsewhere in `tests/` — search for `as unknown as { extractHyperparams` to confirm convention. If no such convention exists, expose it via a thin module-level re-export `export const _extractHyperparams = ModelLoader["extractHyperparams"]` instead — tag the export `@internal`.)

- [ ] **Step 7: Run loader test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "extractHyperparams non-BERT"`
Expected: FAIL — `extractHyperparams` returns `architecture: "nomic-bert"` but `causalAttention === undefined` (the existing `if (arch === "bert")` block doesn't fire).

- [ ] **Step 8: Broaden `extractHyperparams` in `src/models/model-loader.ts`**

In `src/models/model-loader.ts`, **first** add the import:

```ts
import { isEncoderArchitecture } from "../core/types.js";
```

(near the existing imports for `ModelHyperparams`).

Then in `extractHyperparams` (lines 40-106), replace the existing `arch === "bert"` branches:

```ts
// existing (line 56-59):
const normEpsilon =
  arch === "bert"
    ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
    : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);
```

becomes:

```ts
const normEpsilon = isEncoderArchitecture(arch)
  ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
  : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);
```

And:

```ts
// existing (lines 62-71):
let poolingType: ModelHyperparams["poolingType"];
let causalAttention: boolean | undefined;
if (arch === "bert") {
  const pt = getMetaNumberOptional(ctx, `${arch}.pooling_type`) ?? 2;
  poolingType = pt === 1 ? "mean" : "cls";
  causalAttention =
    getMetaBooleanOptional(ctx, `${arch}.attention.causal`) ?? false;
}
```

becomes:

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

Finally in the returned object literal (lines 73-105), add `alibiMaxBias,` after `causalAttention,`:

```ts
return {
  architecture: arch,
  // … existing fields …
  poolingType,
  causalAttention,
  alibiMaxBias,
};
```

- [ ] **Step 9: Run loader tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "extractHyperparams non-BERT"`
Expected: PASS (all 3 sub-tests).

- [ ] **Step 10: Extend `getRopeModeForArchitecture` in `src/inference/model-inference.ts`**

In `src/inference/model-inference.ts:83-89`, replace:

```ts
export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}
```

With:

```ts
export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	// nomic-embed uses NORMAL rotary mode per sentence-transformers convention
	// (verified in Phase 0 probe; bake into the helper rather than a magic
	// site-local constant).
	if (architecture === "nomic-bert") return RopeMode.NORMAL;
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}
```

(If Phase 0 reveals nomic uses NEOX instead — the parity probe in Task 5 will catch the bug; flip to `RopeMode.NEOX` then.)

- [ ] **Step 11: Run full type-check + lint + test gate**

Run: `make checkall`
Expected: PASS (no fmt/lint/typecheck/test failures). The existing `EncoderInference construction` test at the top of the file will start failing because the `architecture: "llama"` constructor case still throws but with a now-broader message — verify the test message regex `/requires architecture "bert"/` still matches the existing error string in `encoder-inference.ts:55-58`. If the message changed in Task 3, update the test regex there.

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
  back to 8.0)
- extractHyperparams: arch === "bert" branches broaden to
  isEncoderArchitecture(arch); jina-only ALiBi metadata read
- getRopeModeForArchitecture: nomic-bert -> NORMAL (sentence-transformers
  convention; verified in Phase 0 probe)

Existing BERT path unchanged. EncoderInference still hard-asserts on
"bert"; routing change lands in Phase 2. make checkall green.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §4
EOF
)"
```

---

## Task 3: Phase 2 — EncoderInference forward + engine routing (commit 2 of 5)

**Files:**
- Modify: `src/inference/encoder-inference.ts:29-36` (`EncoderWeights.positionEmb` → `TensorPtr | null`)
- Modify: `src/inference/encoder-inference.ts:54-62` (drop `arch === "bert"` hard-assert)
- Modify: `src/inference/encoder-inference.ts:85-89` (conditional `position_embd.weight` lookup)
- Modify: `src/inference/encoder-inference.ts:172-240` (arch-branched `buildGraph`)
- Modify: `src/core/engine.ts:589` (broaden routing to `isEncoderArchitecture`)
- Modify: `tests/encoder-inference.test.ts` (rewrite construction test; add buildGraph dispatch test)

**Purpose:** Encoder forward graph + engine routing accept the new arches. After this task, `engine.embed()` can dispatch nomic / jina hyperparams without code path changes — only the `position_embd.weight` lookup is conditional.

- [ ] **Step 1: Write the failing test for nomic-bert / jina-bert-v2 construction**

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
Expected: FAIL on the `nomic-bert` and `jina-bert-v2` cases (current constructor throws), and the error-regex on the `llama` case fails to match (current message says `requires architecture "bert"`, new message will say `does not yet support architecture "llama"`).

- [ ] **Step 3: Replace the constructor hard-assert in `src/inference/encoder-inference.ts`**

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

With:

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

Add to imports at the top of the file:

```ts
import { ENCODER_ARCHITECTURES, isEncoderArchitecture } from "../core/types.js";
```

- [ ] **Step 4: Run construction tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "EncoderInference construction"`
Expected: PASS (all 4 sub-tests).

- [ ] **Step 5: Make `positionEmb` nullable + guard the loader call site**

In `src/inference/encoder-inference.ts:29-36`, replace:

```ts
interface EncoderWeights {
  tokEmb: TensorPtr;
  positionEmb: TensorPtr;
  tokenTypes: TensorPtr;
  inputNormW: TensorPtr;
  inputNormB: TensorPtr;
  layers: EncoderLayerWeights[];
}
```

With:

```ts
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

In `loadWeights` (lines 85-89), replace:

```ts
const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
const positionEmb = this.makeTensor(tensorMap, "position_embd.weight");
const tokenTypes = this.makeTensor(tensorMap, "token_types.weight");
```

With:

```ts
const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
const positionEmb = this.hp.architecture === "bert"
  ? this.makeTensor(tensorMap, "position_embd.weight")
  : null;
const tokenTypes = this.makeTensor(tensorMap, "token_types.weight");
```

(Phase 0 probe confirms `position_embd.weight` is absent for nomic/jina; even if present-but-unused, skipping the lookup is correct — the weight isn't referenced by `buildGraph` for those arches.)

- [ ] **Step 6: Write the failing test for buildGraph arch dispatch**

Append to `tests/encoder-inference.test.ts`:

```ts
describe("EncoderInference.buildGraph arch dispatch", () => {
  function buildAndCount(arch: ModelHyperparams["architecture"]): {
    getrows: number;
    rope: number;
    softmax_max_bias: number[];
  } {
    const fake = makeFakeWasm();
    const hp: ModelHyperparams = {
      architecture: arch,
      contextLength: 512,
      embeddingLength: 384,
      headCount: 12,
      headCountKv: 12,
      layerCount: 2, // 2 layers — enough to confirm per-layer behaviour
      vocabularySize: 30522,
      embeddingHeadLength: 32,
      feedForwardLength: 1536,
      ropeFreqBase: 10000,
      ropeScale: 1,
      normEpsilon: 1e-12,
      expertCount: 0,
      expertUsedCount: 0,
      poolingType: "mean",
      causalAttention: false,
      ...(arch === "jina-bert-v2" ? { alibiMaxBias: 8.0 } : {}),
    };
    const enc = new EncoderInference(fake.fake, hp);
    // Stub weights — buildGraph only reads pointer values; nulls are
    // tolerated by the FakeWasm op stubs.
    (enc as unknown as { weights: unknown }).weights = {
      tokEmb: 1, positionEmb: arch === "bert" ? 2 : null, tokenTypes: 3,
      inputNormW: 4, inputNormB: 5,
      layers: Array.from({ length: hp.layerCount }, () => ({
        qProj: 10, qBias: 11, kProj: 12, kBias: 13, vProj: 14, vBias: 15,
        oProj: 16, oBias: 17, attnNormW: 18, attnNormB: 19,
        ffnUp: 20, ffnUpBias: 21, ffnDown: 22, ffnDownBias: 23,
        ffnNormW: 24, ffnNormB: 25,
      })),
    };
    (enc as unknown as { buildGraph: (n: number) => unknown }).buildGraph(4);
    return {
      getrows: fake.ops.filter(o => o === "getrows").length,
      rope: fake.ops.filter(o => o === "rope").length,
      softmax_max_bias: fake.softmaxMaxBias,
    };
  }

  test("bert path: position-embedding lookup, no rope, max_bias=0", () => {
    const r = buildAndCount("bert");
    // 3 getrows: token, position, segment
    expect(r.getrows).toBe(3);
    expect(r.rope).toBe(0);
    expect(r.softmax_max_bias).toEqual([0, 0]); // 2 layers
  });

  test("nomic-bert path: no position-embedding, rope on Q+K each layer, max_bias=0", () => {
    const r = buildAndCount("nomic-bert");
    // 2 getrows: token, segment (no position embedding)
    expect(r.getrows).toBe(2);
    expect(r.rope).toBe(4); // 2 layers × (Q + K)
    expect(r.softmax_max_bias).toEqual([0, 0]);
  });

  test("jina-bert-v2 path: no position-embedding, no rope, max_bias=8.0", () => {
    const r = buildAndCount("jina-bert-v2");
    expect(r.getrows).toBe(2);
    expect(r.rope).toBe(0);
    expect(r.softmax_max_bias).toEqual([8.0, 8.0]);
  });
});
```

The `FakeWasm` factory (already at `tests/encoder-inference.test.ts:35`) needs two new stubs. Update `makeFakeWasm()`:

```ts
// Inside makeFakeWasm(), inside the `stub` object literal, add:
opRope: () => {
  ops.push("rope");
  return next++;
},
opSoftMaxExt: (_x: TensorPtr, _mask: number, _scale: number, maxBias: number) => {
  ops.push("softmaxext");
  softmaxMaxBias.push(maxBias);
  return next++;
},

// And in the FakeWasm interface above:
interface FakeWasm {
  fake: GgmlWasm;
  ops: string[];
  softmaxMaxBias: number[];
}

// And in the function body:
const softmaxMaxBias: number[] = [];
// (return it alongside ops)
return { fake: stub as unknown as GgmlWasm, ops, softmaxMaxBias };
```

- [ ] **Step 7: Run buildGraph dispatch test to verify it fails**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: FAIL — current `buildGraph` always calls `opGetRows` 3 times (token+pos+seg) regardless of arch, never calls `opRope`, always passes `0.0` to `opSoftMaxExt`. The bert case may already pass; nomic and jina cases will fail.

- [ ] **Step 8: Implement arch-branched `buildGraph` in `src/inference/encoder-inference.ts`**

In `src/inference/encoder-inference.ts:172-240`, replace the body of `buildGraph` with the following. The structural changes are at three points; FFN, post-norm, and pooling are unchanged.

Add these imports at the top of the file (alongside the existing imports):

```ts
import { getRopeModeForArchitecture } from "./model-inference.js";
```

Replace `buildGraph(nTokens: number)`:

```ts
private buildGraph(nTokens: number): TensorPtr {
  if (!this.weights) throw new Error("weights not loaded");
  const { wasm, weights, hp } = this;
  const arch = hp.architecture;
  const usesPosEmbedding = arch === "bert";
  const usesRope = arch === "nomic-bert";
  const alibiMaxBias =
    arch === "jina-bert-v2" ? (hp.alibiMaxBias ?? 8.0) : 0.0;
  const ropeMode = usesRope ? getRopeModeForArchitecture(arch) : 0;

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

    const q = wasm.opAdd(wasm.opMulMat(lw.qProj, x), lw.qBias);
    const k = wasm.opAdd(wasm.opMulMat(lw.kProj, x), lw.kBias);
    const v = wasm.opAdd(wasm.opMulMat(lw.vProj, x), lw.vBias);

    let q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
    let k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
    const v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);

    if (usesRope) {
      // Same arity as the causal-LM call site at model-inference.ts:483-503.
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

    const qp = wasm.opPermute(q3, 0, 2, 1, 3);
    const kp = wasm.opPermute(k3, 0, 2, 1, 3);
    const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

    const qk = wasm.opMulMat(kp, qp);
    const aw = wasm.opSoftMaxExt(qk, 0, invSqrtHd, alibiMaxBias);
    const out = wasm.opMulMat(vp, aw);
    const merged = wasm.opReshape2d(
      wasm.opCont(wasm.opPermute(out, 0, 2, 1, 3)),
      E,
      nTokens,
    );
    const attnProj = wasm.opAdd(wasm.opMulMat(lw.oProj, merged), lw.oBias);

    x = this.layerNorm(wasm.opAdd(x, attnProj), lw.attnNormW, lw.attnNormB);

    let h = wasm.opAdd(wasm.opMulMat(lw.ffnUp, x), lw.ffnUpBias);
    h = wasm.opGelu(h);
    const ffnProj = wasm.opAdd(wasm.opMulMat(lw.ffnDown, h), lw.ffnDownBias);

    x = this.layerNorm(wasm.opAdd(x, ffnProj), lw.ffnNormW, lw.ffnNormB);
  }

  return x;
}
```

- [ ] **Step 9: Run buildGraph tests to verify they pass**

Run: `bun test tests/encoder-inference.test.ts -t "buildGraph arch dispatch"`
Expected: PASS (all 3 sub-tests).

- [ ] **Step 10: Broaden engine routing in `src/core/engine.ts:589`**

In `src/core/engine.ts:589`, replace:

```ts
const isEncoder = parsed.hyperparams.architecture === "bert";
```

With:

```ts
const isEncoder = isEncoderArchitecture(parsed.hyperparams.architecture);
```

Add to the imports at the top of the file:

```ts
import { isEncoderArchitecture } from "./types.js";
```

(Find the existing block of `from "./types.js"` imports and merge the new symbol in. If `engine.ts` already imports a different symbol from `./types.js`, add `isEncoderArchitecture` to that existing import statement.)

- [ ] **Step 11: Run full ship gate**

Run: `make checkall`
Expected: PASS. The BGE / Arctic-Embed test paths still exercise the `bert` branch and produce bit-identical outputs.

- [ ] **Step 12: Commit Phase 2**

```bash
git add src/inference/encoder-inference.ts \
        src/core/engine.ts \
        tests/encoder-inference.test.ts
git commit -m "$(cat <<'EOF'
feat(encoder): arch-branched forward graph for nomic-bert + jina-bert-v2

Phase 2 of bucket B. EncoderInference accepts all three encoder archs:
- Constructor uses isEncoderArchitecture() instead of "bert" hard-assert
- buildGraph branches at three load-bearing points:
    - pre-layer pos-embedding: bert only (skipped for nomic + jina)
    - per-layer Q/K rope: nomic-bert only (NORMAL mode, freq_base from
      GGUF), uses opRope arity matching model-inference.ts:483-503
    - softmax max_bias: jina-bert-v2 only (alibiMaxBias from GGUF, falls
      back to 8.0); ggml computes per-head linear bias internally
- positionEmb in EncoderWeights becomes nullable; loadWeights skips the
  position_embd.weight lookup for nomic + jina (Phase 0 confirmed absence)
- engine.ts:589 routing broadens to isEncoderArchitecture()

BERT path bit-identical post-change; BGE + Arctic-Embed test paths
unchanged. make checkall green. No GGUF available yet for nomic/jina
end-to-end at this commit — that lands in Phase 3.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §3
EOF
)"
```

---

## Task 4: Phase 3a — Reference-vector capture (one-shot Python)

**Files:**
- Create: `eval/reports/encoder-parity-2026-04-28/capture-refs.py`
- Create: `eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt`
- Create: `eval/reports/encoder-parity-2026-04-28/nomic-ref.json`
- Create: `eval/reports/encoder-parity-2026-04-28/jina-ref.json`

**Purpose:** Generate the numerical-parity reference vectors from `sentence-transformers`. One-shot off-process work; results are checked in. The bun harness (Task 5) consumes these files.

- [ ] **Step 1: Write the requirements pin**

Create `eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt`:

```
sentence-transformers>=3.0,<4.0
einops>=0.7
```

(`einops` is a dependency of nomic's `trust_remote_code=True` config.)

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

Run:

```bash
cd eval/reports/encoder-parity-2026-04-28 && \
  uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
```

Expected: stderr shows "Loading nomic-ai/nomic-embed-text-v1.5…" and "Loading jinaai/jina-embeddings-v2-base-en…" and "wrote nomic-ref.json (5 vectors, dim=768)" / "wrote jina-ref.json (5 vectors, dim=768)". Exit code 0.

If `nomic-embed-text-v1.5` requires `trust_remote_code=True` and refuses, set `HF_HUB_DISABLE_PROGRESS_BARS=1 TRUST_REMOTE_CODE=1` in the environment and re-run. If jina insists on a different model name, try `jinaai/jina-embeddings-v2-small-en` to verify the script works, then return to base-en.

- [ ] **Step 4: Sanity-check the captured vectors**

Run:

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

Expected: each row's L2 norm is `1.000000 ± 1e-6` (vectors are normalized). Both files have 5 rows, 768-dim. If a row has norm ≈ 0 or ≠ 1, the capture is malformed — re-run.

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
parity harness uses. capture-refs.py is reproducible — run via
  uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
under eval/reports/encoder-parity-2026-04-28/.

These vectors gate Phase 3 / Phase 4 — per-row cosine vs the in-tree
implementation must clear 0.999 before each model registration lands.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 5: Phase 3 — Parity harness + nomic registration + parity gate (commit 3 of 5)

**Files:**
- Create: `eval/encoder-parity.ts` (parity harness, ~80 LOC)
- Modify: `eval/models.ts` (nomic entry, mirroring the BGE-base shape at lines 393-410)
- Modify: `eval/smoke-profiles.ts` (nomic profile entry, mirroring `bge-large` at lines 273-276)
- Modify: `eval/smoke-profiles.ts:355` (add `"nomic-embed-text-v1.5"` to the `embeddings` selector array)
- Create: `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` (writeup)

**Purpose:** Build the parity harness, register nomic, run the parity gate. Gate must pass (≥0.999 cosine on all 5 rows) before commit.

- [ ] **Step 1: Write the parity harness (browser-driven via agentchrome)**

Bun has no native WebGPU, so the harness drives the browser smoke page (`smoke-test/real-model.html`) via `agentchrome` — same pattern as `eval/embed-perf.ts`. The smoke page already exposes `window.engine` and `window.handleId` after model load (see `smoke-test/real-model-page.js:472`); the harness uses `agentchrome js-exec` to call `engine.embed(handleId, text)` and capture each vector.

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
 * Example:
 *   bun eval/encoder-parity.ts nomic-embed-text-v1.5-q0f16 \
 *     eval/reports/encoder-parity-2026-04-28/nomic-ref.json
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

// Wait for model load: the page sets window.handleId after engine.loadModel
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
  // Pass the input through page-side JSON.parse to handle Unicode safely.
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
- `ensureModelDownloaded`, `ensureSmokeServerReachable`, `resolveAgentchromeSession` — same names as in `eval/embed-perf.ts`.

If `agentchrome` exposed in `browser-smoke.ts` doesn't accept `["--port", ..., "--tab", ..., "js-exec", expr]`, check the actual `agentchrome --help` for the exact subcommand (might be `evaluate`, `runtime evaluate`, or `eval`). Match what `agentchrome` ships with on this machine — the helper signature in `browser-smoke.ts` is the authority. If a `js-exec` helper doesn't exist, add a small wrapper inline in `eval/encoder-parity.ts`:

```ts
async function jsExec(port: string, tab: string, expr: string): Promise<string> {
  return agentchrome(["--port", port, "--tab", tab, "js-exec", expr]);
}
```

(verify the subcommand name from `agentchrome --help` and substitute.)

If the page exposes the engine under a different global than `window.handleId` / `window.engine`, grep `smoke-test/real-model-page.js` for the exact symbol names and adjust the script.

- [ ] **Step 2: Register nomic in `eval/models.ts`**

Add immediately after the `bge-large-en-v1.5-q0f16` entry (after line 410):

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
  contextLength: 8192,
  tier: "ultrafast",
  requiresShaderF16: false,
  downloadUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
  ggufUrl: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
  // File on the mirror is `nomic-embed-text-v1.5.f16.gguf`; substring `f16`
  // is unique within the repo (no fp16/Q4 collision).
  ggufFilePattern: "f16",
},
```

(If Phase 0 probe revealed a different file pattern — e.g. `_fp16` like the BGE entries — adjust `ggufFilePattern` accordingly. The spec note at §2 flags this.)

- [ ] **Step 3: Register nomic profile in `eval/smoke-profiles.ts`**

Add after the `bge-large` profile (after line 276):

```ts
{
  name: "nomic-embed-text-v1.5",
  model: "nomic-embed-text-v1.5-q0f16",
  embedding: true,
},
```

In the embedding-profile selectors block (around line 350-355), append `"nomic-embed-text-v1.5"` to the `embeddings:` array and to whichever array carries the broader embedding fleet listing — verify the exact name by grepping for `"bge-large"` in that file and adding alongside.

- [ ] **Step 4: Run `make checkall` to confirm registration types are clean**

Run: `make checkall`
Expected: PASS. (The harness `eval/encoder-parity.ts` is checked too — it lives in the test/eval glob.)

- [ ] **Step 5: Run the parity gate for nomic**

Run: `bun eval/encoder-parity.ts nomic-embed-text-v1.5-q0f16 eval/reports/encoder-parity-2026-04-28/nomic-ref.json`

Expected: 5/5 rows PASS with cosine ≥ 0.999. Exit code 0.

If any row fails, the parity probe has fired — diagnose in this order (most-likely first):
1. **RoPE mode**: try `RopeMode.NEOX` instead of `RopeMode.NORMAL` in `getRopeModeForArchitecture` (Task 2 step 10). Re-run.
2. **freq_base mismatch**: check the value loaded from GGUF (log `hp.ropeFreqBase` at engine init). Compare against nomic's HF config.
3. **headDim**: log `hp.embeddingHeadLength`. nomic-embed-text-v1.5 has 12 heads × 64 = 768. If GGUF reports `64`, that's correct.
4. **Pooling**: log `hp.poolingType`. Should be `"mean"` for nomic. If `"cls"`, the GGUF `pooling_type` metadata didn't map — check loader.

Each diagnosis is a small fix + re-run. The probe artifact `SUMMARY.md` records which row failed at what cosine — direct signal for the cause.

- [ ] **Step 6: Browser smoke (Gate 4) for nomic**

Start the smoke server if not already running:

```bash
make smoke-serve
```

Then via `agentchrome` (reusing the existing session — see CLAUDE.md):

```bash
agentchrome connect --status   # confirm session
# Reuse existing tab. Cache-bust:
agentchrome --port <PORT> tabs list
agentchrome --port <PORT> --tab <TAB_ID> navigate "http://localhost:8031/?v=$(date +%s)&model=nomic-embed-text-v1.5-q0f16&embed=Hello+world."
```

Expected: page `#log` shows successful download + load + embed; no console errors except the benign `adapter_info:` line. Capture a console snapshot.

- [ ] **Step 7: Write Phase 3 SUMMARY.md**

Create `eval/reports/encoder-parity-2026-04-28/SUMMARY.md`:

```markdown
# Encoder parity probe — 2026-04-28

## Inputs

5 fixed strings in `inputs.json`. Reference embeddings captured via
sentence-transformers in `capture-refs.py`. Per-row gate: cosine >= 0.999.

## nomic-embed-text-v1.5 (BERT + RoPE)

| Row | Input (truncated) | Cosine | Pass |
|----:|---|---:|:---:|
| 0 | Hello world. | <fill> | <Y/N> |
| 1 | The quick brown fox jumps … | <fill> | <Y/N> |
| 2 | Embedding models map text … | <fill> | <Y/N> |
| 3 | Café — naïve façade … | <fill> | <Y/N> |
| 4 | . | <fill> | <Y/N> |

**Result:** <X>/5 rows passed.
**RoPE mode:** NORMAL (verified via Phase 0 + parity probe).
**freq_base:** <value loaded from GGUF>.

## jina-embeddings-v2-base-en (BERT + ALiBi)

(filled in Phase 4 / Task 6)

## Methodology notes

- Reference vectors normalized via sentence-transformers
  `normalize_embeddings=True`.
- WebLLM-side vectors emerge from `engine.embed()` already
  L2-normalized via `EncoderInference.poolAndNormalize`.
- Cosine computed in TS (`eval/encoder-parity.ts`), F32 accumulator.
- 0.999 gate is a soft floor: BGE shows >0.9999 informally on this
  metric. RoPE / ALiBi degenerate-alignment failure modes typically
  show as <0.95 (often <0.5) so the gate is comfortably above noise.
```

Fill in the actual cosine values from the harness output. If any row fell below 0.999 after diagnosis fixes, the failure should not be papered over — the gate **must** show Y on all rows before committing.

- [ ] **Step 8: Commit Phase 3**

```bash
git add eval/encoder-parity.ts \
        eval/models.ts \
        eval/smoke-profiles.ts \
        eval/reports/encoder-parity-2026-04-28/SUMMARY.md
git commit -m "$(cat <<'EOF'
feat(eval): register nomic-embed-text-v1.5 + parity probe (5/5 rows >= 0.999)

Phase 3 of bucket B. Registers Nomic Embed Text v1.5 (137M, BERT+RoPE,
768-dim, 8192 ctx, mean pooling) + ships eval/encoder-parity.ts harness.

Parity gate: 5/5 fixed inputs hit cosine >= 0.999 vs sentence-transformers
reference. RoPE mode confirmed NORMAL (sentence-transformers convention);
freq_base loaded from GGUF metadata. Browser smoke (gate 4) green.

First encoder in fleet with rotary positional embeddings — new dashboard
scaling point at 137M / 768-dim. Parity harness reusable for jina (lands
in phase 4) and any future encoder additions.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 6: Phase 4 — Jina registration + parity gate (commit 4 of 5)

**Files:**
- Modify: `eval/models.ts` (jina entry)
- Modify: `eval/smoke-profiles.ts` (jina profile + selector array)
- Modify: `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` (fill jina table)

**Purpose:** Same pattern as Task 5, for jina. Smaller commit because the harness already exists.

- [ ] **Step 1: Register jina in `eval/models.ts`**

Add immediately after the `nomic-embed-text-v1.5-q0f16` entry:

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
  // Verify in Phase 0 probe: file pattern observed on the mirror.
  ggufFilePattern: "f16",
},
```

(If Phase 0 fell back to a different mirror, update `ggufUrl` and `ggufFilePattern` to match what the probe artifact recorded.)

- [ ] **Step 2: Register jina profile in `eval/smoke-profiles.ts`**

Add after the `nomic-embed-text-v1.5` profile:

```ts
{
  name: "jina-embeddings-v2-base-en",
  model: "jina-embeddings-v2-base-en-q0f16",
  embedding: true,
},
```

Append `"jina-embeddings-v2-base-en"` to the `embeddings:` selector array (and any sibling embedding fleet array, mirroring what Task 5 did for nomic).

- [ ] **Step 3: Run `make checkall`**

Run: `make checkall`
Expected: PASS.

- [ ] **Step 4: Run the parity gate for jina**

Run: `bun eval/encoder-parity.ts jina-embeddings-v2-base-en-q0f16 eval/reports/encoder-parity-2026-04-28/jina-ref.json`

Expected: 5/5 rows PASS with cosine ≥ 0.999.

If any row fails, the most-likely diagnosis order:
1. **`alibiMaxBias` value**: log `hp.alibiMaxBias` at engine init. The default 8.0 is correct for 12-head models; if the GGUF metadata explicitly sets a different value (and the loader read it correctly), the value being passed to `opSoftMaxExt` should match.
2. **Per-head slope sign**: ggml's softmax uses negative slopes (heads with later positions see earlier positions less). If parity collapses to ~0.5 across all rows but shorter inputs pass, ALiBi sign is inverted — this would require a ggml-side fix and escalates to user (do not paper over).
3. **Pooling**: confirm `hp.poolingType === "mean"` (jina v2 uses mean pooling).

- [ ] **Step 5: Browser smoke (Gate 4) for jina**

Same flow as Task 5 step 6, with `jina-embeddings-v2-base-en-q0f16` as the model.

- [ ] **Step 6: Fill the jina row in `SUMMARY.md`**

Replace the `(filled in Phase 4 / Task 6)` stub in `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` with:

```markdown
## jina-embeddings-v2-base-en (BERT + ALiBi)

| Row | Input (truncated) | Cosine | Pass |
|----:|---|---:|:---:|
| 0 | Hello world. | <fill> | <Y/N> |
| 1 | The quick brown fox jumps … | <fill> | <Y/N> |
| 2 | Embedding models map text … | <fill> | <Y/N> |
| 3 | Café — naïve façade … | <fill> | <Y/N> |
| 4 | . | <fill> | <Y/N> |

**Result:** <X>/5 rows passed.
**alibiMaxBias:** <value loaded from GGUF, default 8.0>.
```

- [ ] **Step 7: Commit Phase 4**

```bash
git add eval/models.ts \
        eval/smoke-profiles.ts \
        eval/reports/encoder-parity-2026-04-28/SUMMARY.md
git commit -m "$(cat <<'EOF'
feat(eval): register jina-embeddings-v2-base-en + parity probe (5/5 >= 0.999)

Phase 4 of bucket B. Registers Jina Embeddings v2 Base EN (137M,
BERT+ALiBi, 768-dim, 8192 ctx, mean pooling).

Parity gate: 5/5 fixed inputs hit cosine >= 0.999 vs sentence-transformers
reference. alibiMaxBias loaded from GGUF metadata (default 8.0 if absent);
ggml's opSoftMaxExt(..., max_bias) computes per-head linear bias internally,
no extra weight tensor needed. Browser smoke (gate 4) green.

First encoder in fleet with ALiBi attention bias. Parity harness from
Phase 3 reused unchanged.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md §5
EOF
)"
```

---

## Task 7: Phase 5 — Bench-full + dashboard refresh + TODO close (commit 5 of 5)

**Files:**
- Modify: `TODO.md` (transition Bucket B from "queued"/"deferred" to "DONE 2026-MM-DD")

**Purpose:** Run the full eval suite to populate the cosine-task and latency rows for both new models, refresh the dashboard, transition TODO.md.

- [ ] **Step 1: Start the live dashboard**

Run:

```bash
make dashboard-serve   # port 8033
```

Verify it's reachable: `curl -sf http://localhost:8033/ > /dev/null && echo OK` should print OK.

- [ ] **Step 2: Run `make bench-full`**

Run:

```bash
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 make bench-full
```

Expected: each registered profile (existing fleet + nomic + jina) runs its bench; results post live to the dashboard. Wall time: ~30-45 min for the full run.

If a model OOMs or aborts mid-run, capture the error and stop — do not paper over with `--skip`. Most likely cause for new entries is GGUF download failure (mirror down) or a Phase 0-tagged metadata key mismatch that fell through.

- [ ] **Step 3: Inspect the dashboard's Embeddings section**

Open `http://localhost:8033/` and confirm:
- The Embeddings section now shows 6 rows (arctic-embed-s/m, bge-small/large, nomic, jina).
- Cosine-task scatter plot has 6 points.
- Per-task latency bars include the two new entries.

Snapshot the dashboard via `agentchrome` and save to `eval/reports/encoder-parity-2026-04-28/dashboard-2026-MM-DD.png`.

- [ ] **Step 4: Update TODO.md — transition Bucket B**

Find the "Embedding-model expansion candidates (queued 2026-04-28)" section in `TODO.md` (around line 845). Update Bucket B's heading from:

```markdown
**B. Extend `EncoderInference` to non-BERT arch** (deferred). Two
popular asks both require real engineering on top of A's
infrastructure:
```

To:

```markdown
**B. Extend `EncoderInference` to non-BERT arch** ~~(deferred)~~
**DONE 2026-MM-DD** (commits ##### through #####). Two new encoder
arches landed cleanly under one design / 5 phased commits:
- `nomic-embed-text-v1.5-q0f16` (~137M, 768-dim, BERT+RoPE):
  <p50 short> ms p50 single-text short / <X>% on 8-task cosine eval.
  First encoder in fleet with rotary positional embeddings.
- `jina-embeddings-v2-base-en-q0f16` (~137M, 768-dim, BERT+ALiBi):
  <p50 short> ms p50 single-text short / <X>% on 8-task cosine eval.
  First encoder in fleet with ALiBi attention bias — landed for free
  via `opSoftMaxExt(..., max_bias)` (no new ggml op needed).

Parity probe: 10/10 rows >= 0.999 cosine vs sentence-transformers
reference. Probe artifacts pinned at
`eval/reports/encoder-parity-2026-04-28/`.

Net learning: ALiBi was effectively free (1-line change inside the
softmax call); RoPE was straightforward once the encoder forward
broadened to call `opRope` (already used by causal LM). The arch enum
+ `isEncoderArchitecture()` helper makes the next encoder addition
(non-BERT family) a registration-only change for ALiBi-style and a
single-branch addition for novel positional schemes.
```

(Fill in the `MM-DD`, commit short hashes, p50 latency, cosine-task scores from the bench-full output.)

If Bucket B was a separate top-level entry rather than under "Embedding-model expansion candidates", update whatever the actual heading turns out to be — grep `TODO.md` for "Bucket B" before editing.

- [ ] **Step 5: Run final ship gate**

Run: `make checkall`
Expected: PASS.

- [ ] **Step 6: Commit Phase 5 (closure)**

```bash
git add TODO.md \
        eval/reports/encoder-parity-2026-04-28/dashboard-2026-MM-DD.png
git commit -m "$(cat <<'EOF'
docs(TODO): close bucket B — nomic + jina encoder expansion landed

Phase 5 of bucket B. Closure: 10/10 parity rows green, dashboard
Embeddings section updated to 6 rows, bench-full populates cosine-task
and latency for both new models.

Bucket B retired from the embedding-model expansion queue. Bucket C
(causal-LM-derived embedders, e.g. Qwen3-Embedding) remains deferred
behind a deployment ask; bucket A + B together exercise the full BERT-
family encoder lever portfolio.

Spec: docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md
Plan: docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md
EOF
)"
```

- [ ] **Step 7: Stop the dashboard server**

Run:

```bash
make dashboard-stop
```

(or whichever target the project uses; if no stop target, kill the bun process directly.)

---

## Self-review checklist

After all 8 tasks complete, verify:

- [ ] Plan commit + Phase 0 + 5 phase commits + Phase 3a (7 commits total) present in `git log` since the spec commit.
- [ ] `make checkall` clean on `main`.
- [ ] Parity SUMMARY.md shows 10/10 PASS.
- [ ] Dashboard live at port 8033 shows 6 embedding rows with non-null cosine + latency.
- [ ] TODO.md's Bucket B entry transitioned to DONE with cycle-closure summary, mirroring Bucket A's closure entry shape.
- [ ] No `<fill>`, `MM-DD`, or `<X>` placeholders remain in any committed artifact.
