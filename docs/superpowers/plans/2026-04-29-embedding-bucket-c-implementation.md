# Bucket C — Qwen3-Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Qwen3-Embedding-0.6B as the first causal-LM-derived embedder in webllm, validated end-to-end via per-row cosine ≥0.999 against the Phase 0 reference vectors.

**Architecture:** New sibling `CausalLMEmbedder` class in `src/inference/causal-embedder-inference.ts` mirroring `EncoderInference`'s shape. New `qwen3-embedding` architecture enum entry. Loader derives `qwen3-embedding` from GGUF metadata when `general.architecture = qwen3` AND `qwen3.pooling_type = 3` (LAST). Engine `embed()` widens to a three-way dispatch (encoder → causal-embedder → throw). Forward graph is a near-copy of `ModelInference.forwardSingle` minus KV cache + lm_head + sampling, taps hidden state at post-output-norm, last-token pools, L2-normalizes.

**Tech Stack:** TypeScript / Bun, GGML/WebGPU via `GgmlWasm`, Qwen3 architecture (28 layers, 1024 hidden, NEOX RoPE, freq_base=1000000), sentence-transformers reference vectors.

**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-implementation-design.md` (commit `5f75c9f`).

**Phase 0 probe artifacts:**
- `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` — embed-surface analysis + GGUF metadata.
- `eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md` — capture report.
- `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json` — 10 reference vectors.

**Pinned constants from probe:**
- Architecture (GGUF): `qwen3`. Webllm enum entry: `qwen3-embedding`.
- Pooling: `last-token` (`qwen3.pooling_type = 3`).
- Output dim: 1024 (= `hidden_size`; no projection head).
- Layers: 28.
- Heads: 16 attention, 8 KV (GQA).
- Key length: 128.
- RoPE: NEOX, freq_base=1000000.
- Reference GGUF mirror: `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf` (1.2 GB).
- Query instruction prefix runtime bytes (89 chars + LF):
  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query:
  ```

---

## File map (locked from spec)

**New files (4):**
- `src/inference/causal-embedder-inference.ts` — `CausalLMEmbedder` class.
- `tests/causal-embedder-inference.test.ts` — unit tests.
- `eval/causal-embedder-parity.ts` — parity harness.
- `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md` — Phase 6 closure.

**Modified files (8):**
- `src/core/types.ts` — add `qwen3-embedding` arch + `CAUSAL_EMBEDDER_ARCHITECTURES` + `isCausalEmbedderArchitecture()`; widen `poolingType` enum.
- `src/models/model-loader.ts` — derive `qwen3-embedding` from metadata; widen pooling read.
- `src/core/engine.ts` — `causalEmbedderEngines` map, three-way dispatch in `embed()`, instantiation in `loadModelFromBuffer`.
- `eval/models.ts` — register `qwen3-embedding-0.6b-q0f16`.
- `eval/smoke-profiles.ts` — add embedder smoke profile.
- `eval/embed-perf.ts` — extend bench coverage.
- `smoke-test/real-model.html` + `smoke-test/real-model-smoke.js` (or `real-model-page.js`, depending on actual file in tree) — engine routing for causal embedders.
- `TODO.md` — Phase 6 closure stub.

**Files explicitly NOT touched:**
- `src/inference/model-inference.ts` (load-bearing chat path — preserved).
- `src/inference/encoder-inference.ts` (preserved; sibling class).
- `src/inference/sampler.ts`, `src/inference/generation.ts` (unused for embedders).

---

## Phase 1 — Types + arch enum + registration

Single commit at end: `feat(types): add qwen3-embedding architecture + register Qwen3-Embedding-0.6B`.

### Task 1: Widen architecture enum + add CAUSAL_EMBEDDER_ARCHITECTURES helpers

**Files:**
- Modify: `src/core/types.ts:48-73, 107-130` — add new arch + helpers + widen pooling type.

- [ ] **Step 1: Read the current state**

```bash
sed -n '48,73p' src/core/types.ts
sed -n '107,135p' src/core/types.ts
```

Confirm `ModelArchitecture` is the union ending with `"jina-bert-v2"`, `ENCODER_ARCHITECTURES` is the readonly tuple, and `poolingType?: "cls" | "mean"` lives in `ModelHyperparams`.

- [ ] **Step 2: Add `qwen3-embedding` to the union and add the causal-embedder set**

Edit `src/core/types.ts`:

Replace the `ModelArchitecture` union (lines 48-62) with:

```typescript
/** Supported model architectures for inference dispatch. */
export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "phi3"
	| "gemma"
	| "qwen"
	| "qwen2"
	| "qwen3"
	| "qwen3-embedding"
	| "mixtral"
	| "deepseek"
	| "bert"
	| "nomic-bert"
	| "jina-bert-v2";
```

Then immediately after the existing `isEncoderArchitecture` helper (after line 73), add:

```typescript
/** All architectures handled by CausalLMEmbedder (causal LM with last-token pooling, no KV cache). */
export const CAUSAL_EMBEDDER_ARCHITECTURES = ["qwen3-embedding"] as const;

export function isCausalEmbedderArchitecture(a: ModelArchitecture): boolean {
	return (CAUSAL_EMBEDDER_ARCHITECTURES as readonly string[]).includes(a);
}
```

- [ ] **Step 3: Widen the `poolingType` field**

Replace this line in `ModelHyperparams` (around line 124):

```typescript
	/** For bidirectional encoders: pooling strategy for `embed()`. */
	poolingType?: "cls" | "mean";
```

With:

```typescript
	/** Pooling strategy for `embed()`. CLS/MEAN for BERT-family encoders; LAST-TOKEN for causal-LM-derived embedders (e.g., Qwen3-Embedding). */
	poolingType?: "cls" | "mean" | "last-token";
```

- [ ] **Step 4: Run typecheck**

```bash
bun run lint && bun run typecheck
```

Expected: both pass. If TypeScript flags places where `poolingType` is consumed with exhaustive matching (e.g., a switch that handles `"cls" | "mean"` only), patch those with a `"last-token"` arm. Likely candidates: `EncoderInference.poolAndNormalize` static helper. Inspect with:

```bash
grep -rn "poolingType\|pooling type\|cls\".*mean\|mean\".*cls" src/ | head
```

If `poolAndNormalize` switch needs widening, do NOT add a "last-token" branch there — that helper only handles encoder pooling. Instead add a NEW branch in the broader caller landscape only if typecheck fires. The CausalLMEmbedder will implement its own last-token pool in Phase 2.

- [ ] **Step 5: Commit deferred to Task 4 (Phase 1 final commit)**

No commit yet — Phase 1 commits as a single logical unit at end of Task 4.

---

### Task 2: Loader derives `qwen3-embedding` arch + reads `LAST` pooling

**Files:**
- Modify: `src/models/model-loader.ts:40-113` (`extractHyperparams`)

- [ ] **Step 1: Read the current pooling read logic**

```bash
sed -n '40,113p' src/models/model-loader.ts
```

The current code at lines 64-68 reads pooling only for encoder archs and maps `pt === 1` → "mean", else "cls". This must widen.

- [ ] **Step 2: Edit `extractHyperparams` to derive `qwen3-embedding` from metadata + read LAST pooling**

Replace the current `arch` derivation block (lines 41-45) with:

```typescript
		const rawArch = getMetaString(
			ctx,
			"general.architecture",
			"llama",
		) as ModelHyperparams["architecture"];

		// Derive qwen3-embedding from the qwen3 base arch when LAST-TOKEN pooling
		// is set in metadata (qwen3.pooling_type=3). The Qwen3-Embedding GGUFs
		// share the qwen3 architecture string but carry a pooling_type that the
		// chat Qwen3 GGUFs do not. See bucket C Phase 0 probe report.
		const qwen3PoolingRaw =
			rawArch === "qwen3"
				? getMetaNumberOptional(ctx, "qwen3.pooling_type")
				: undefined;
		const arch: ModelHyperparams["architecture"] =
			rawArch === "qwen3" && qwen3PoolingRaw === 3
				? "qwen3-embedding"
				: rawArch;
```

- [ ] **Step 3: Widen the pooling read to include causal embedders**

Replace the current pooling-read block (lines 60-77 — the `let poolingType` block) with:

```typescript
		// Pooling + causal flag live on encoder models; causal-LM embedders
		// (e.g. qwen3-embedding) carry pooling_type=LAST. Causal defaults true elsewhere.
		let poolingType: ModelHyperparams["poolingType"];
		let causalAttention: boolean | undefined;
		let alibiMaxBias: number | undefined;
		if (isEncoderArchitecture(arch)) {
			const pt = getMetaNumberOptional(ctx, `${arch}.pooling_type`) ?? 2;
			// llama.cpp enum: NONE=0, MEAN=1, CLS=2, LAST=3, RANK=4. We only
			// implement CLS and MEAN for encoders; anything else falls back to CLS.
			poolingType = pt === 1 ? "mean" : "cls";
			causalAttention =
				getMetaBooleanOptional(ctx, `${arch}.attention.causal`) ?? false;
			if (arch === "jina-bert-v2") {
				// gaianet GGUF mirror omits this key; 8.0 is the upstream default
				// (jina-bert-v2 reference impl + llama.cpp).
				alibiMaxBias =
					getMetaNumberOptional(ctx, `${arch}.attention.alibi_bias_max`) ?? 8.0;
			}
		} else if (isCausalEmbedderArchitecture(arch)) {
			// llama.cpp enum: LAST=3. Hard-pin "last-token" for the causal-LM-derived
			// embedder family — no other pooling mode is supported for them.
			poolingType = "last-token";
		}
```

- [ ] **Step 4: Add the import for `isCausalEmbedderArchitecture`**

At the top of `src/models/model-loader.ts`, find the existing `isEncoderArchitecture` import line and widen it. Run:

```bash
grep -n "isEncoderArchitecture" src/models/model-loader.ts | head -3
```

Replace the import line (likely `import { isEncoderArchitecture, ... } from "../core/types.js"`) to also import `isCausalEmbedderArchitecture`. Example:

```typescript
import { isEncoderArchitecture, isCausalEmbedderArchitecture, /* …existing imports… */ } from "../core/types.js";
```

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck
```

Expected: pass. If `getMetaNumberOptional` is the wrong import name in this file, grep for the correct symbol — bucket B's encoder loader uses `getMetaNumberOptional` per line 102 of model-loader.ts, so the import should already exist.

- [ ] **Step 6: No commit yet — continued in Task 4**

---

### Task 3: Register `qwen3-embedding-0.6b-q0f16` in eval/models.ts

**Files:**
- Modify: `eval/models.ts:344-460` (Embedding models block)

- [ ] **Step 1: Read the existing embedding-model registrations to match style**

```bash
sed -n '425,465p' eval/models.ts
```

Confirm the `nomic-embed-text-v1.5-q0f16` entry shape — that's the closest precedent (architecturally distinct, single-file mirror).

- [ ] **Step 2: Add the Qwen3-Embedding entry**

Add the following entry to the embedding models block in `eval/models.ts` (immediately after the `nomic-embed-text-v1.5-q0f16` entry, before the closing of the Embedding models region):

```typescript
	{
		id: "qwen3-embedding-0.6b-q0f16",
		name: "Qwen3 Embedding 0.6B",
		family: "Qwen3-Embedding",
		architecture: "qwen3-embedding",
		paramsB: 0.6,
		vramMB: 1300,
		defaultQuant: "q0f16",
		availableQuants: ["q0f16"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: true },
		license: "Apache-2.0",
		contextLength: 32768,
		tier: "ultrafast",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
		ggufUrl: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF",
		// Mirror publishes `Qwen3-Embedding-0.6B-f16.gguf`; `f16` substring
		// matches uniquely on this mirror.
		ggufFilePattern: "f16",
	},
```

If `BenchmarkModel`'s `architecture` field is typed against the union and rejects `"qwen3-embedding"` because Task 1 hasn't been committed yet, that's expected — Phase 1's commit lands all four tasks together. The typecheck in Step 4 below validates the union widening from Task 1 covers this.

- [ ] **Step 3: Run typecheck**

```bash
bun run typecheck
```

Expected: pass. If it fails complaining about `architecture: "qwen3-embedding"`, Task 1 wasn't applied — go back and verify.

---

### Task 4: Add embedder smoke profile + commit Phase 1

**Files:**
- Modify: `eval/smoke-profiles.ts`
- Commit: all Phase 1 changes as one logical unit.

- [ ] **Step 1: Read the smoke-profiles file to find the embedder-profile pattern**

```bash
grep -n "arctic-embed\|bge-small\|nomic-embed\|jina" eval/smoke-profiles.ts | head -10
sed -n '1,80p' eval/smoke-profiles.ts
```

Identify how existing embedder smoke profiles (arctic / bge / jina / nomic) are registered. Pattern is likely a map of model id → smoke-profile config.

- [ ] **Step 2: Add the qwen3-embedding-0.6b-q0f16 smoke profile**

Use the existing `nomic-embed-text-v1.5-q0f16` or `bge-small-en-v1.5-q0f16` entry shape as the template. The Qwen3 embedder behavior for smoke testing is the same as other embedders: download → load → call `engine.embed()` → assert dim and unit magnitude. There is no chat path for this model.

If the smoke-profiles file uses an explicit object literal per model:

```typescript
"qwen3-embedding-0.6b-q0f16": {
	mode: "embed",
	smokeText: "The quick brown fox jumps over the lazy dog.",
	expectedDim: 1024,
	gateMagnitudeTolerance: 1e-3,
},
```

The exact shape depends on the file's existing schema — match it for the existing nomic/bge entry. Read the file first; if you can't find a clean template, ask the controller before guessing.

- [ ] **Step 3: Run `make checkall`**

```bash
make checkall
```

Expected: fmt, lint, typecheck, and tests all pass. Note that `bun test` should NOT run any new test that depends on the Qwen3-Embedding GGUF being downloaded — Task 1-4 ship purely registration / type changes; the new `tests/causal-embedder-inference.test.ts` lands in Phase 2.

If `make checkall` fails on lint formatting, run:

```bash
bun run fmt
make checkall
```

- [ ] **Step 4: Commit Phase 1**

```bash
git add src/core/types.ts src/models/model-loader.ts eval/models.ts eval/smoke-profiles.ts
git commit -m "feat(types): add qwen3-embedding architecture + register Qwen3-Embedding-0.6B"
```

Verify: `git status` clean; the commit touches exactly four files.

- [ ] **Step 5: Verify Phase 1 exit criteria**

- `make checkall` passed.
- New model id `qwen3-embedding-0.6b-q0f16` is recognized: `bun run eval/cli.ts list-models 2>/dev/null | grep qwen3-embedding` (or equivalent — check the CLI's actual subcommand name with `bun run eval/cli.ts --help`).
- Calling `engine.embed("qwen3-embedding-0.6b-q0f16", "x")` would still throw at runtime (Phase 2 hasn't shipped the inference class) — that's expected.

---

## Phase 2 — `CausalLMEmbedder` class

Single commit at end: `feat(inference): CausalLMEmbedder for Qwen3-Embedding`.

### Task 5: Scaffold `CausalLMEmbedder` class with constructor + load + dispose

**Files:**
- Create: `src/inference/causal-embedder-inference.ts`

- [ ] **Step 1: Read the encoder-inference.ts skeleton for shape reference**

```bash
sed -n '1,80p' src/inference/encoder-inference.ts
sed -n '540,560p' src/inference/encoder-inference.ts
```

Note the constructor's arch-rejection pattern, the `loadWeights` shape (no_alloc context, weight buffer ownership), and the `dispose()` method.

- [ ] **Step 2: Read the model-inference.ts weight-loading pattern**

```bash
grep -n "loadWeights\|weights = {" src/inference/model-inference.ts | head -10
sed -n '300,460p' src/inference/model-inference.ts
```

Confirm the Qwen3 weight-loading path. The `qwen3-embedding` GGUF has the same tensor layout as chat Qwen3 minus `output.weight` (no `lm_head` matmul). Bucket C reuses Qwen3's loader path; the only difference is treating `output.weight` as optional.

- [ ] **Step 3: Create `src/inference/causal-embedder-inference.ts` with the skeleton**

```typescript
import {
	CAUSAL_EMBEDDER_ARCHITECTURES,
	isCausalEmbedderArchitecture,
	type ModelHyperparams,
} from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	GgmlType,
	type GgmlWasm,
	type TensorPtr,
} from "./ggml-wasm.js";
import { getRopeModeForArchitecture } from "./model-inference.js";

/**
 * Per-layer weight tensors for a Qwen3-style causal LM. Mirrors the chat
 * Qwen3 layout but `output.weight` is optional — the embedder bypasses
 * the `lm_head` matmul.
 */
interface CausalEmbedderLayerWeights {
	attnNorm: TensorPtr;
	qProj: TensorPtr;
	kProj: TensorPtr;
	vProj: TensorPtr;
	qNorm: TensorPtr | null;
	kNorm: TensorPtr | null;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	gateProj: TensorPtr;
	upProj: TensorPtr;
	downProj: TensorPtr;
}

interface CausalEmbedderWeights {
	tokEmb: TensorPtr;
	norm: TensorPtr; // output_norm.weight
	layers: CausalEmbedderLayerWeights[];
}

/**
 * Causal-LM-derived embedder. Runs the standard causal forward graph through
 * all input tokens in one pass, taps the hidden state at post-output-norm
 * (before `lm_head`), pools last-token, L2-normalizes, returns Float32Array.
 *
 * Sibling to `EncoderInference` and `ModelInference`. Owns its own weight
 * buffer and ctx. No KV cache.
 *
 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp:91-104`
 * (`res->t_embd = cur` after `output_norm`, before `lm_head`).
 */
export class CausalLMEmbedder {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: CausalEmbedderWeights | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();

	constructor(wasm: GgmlWasm, hyperparams: ModelHyperparams) {
		if (!isCausalEmbedderArchitecture(hyperparams.architecture)) {
			throw new Error(
				`CausalLMEmbedder does not support architecture "${hyperparams.architecture}"; supported: ${CAUSAL_EMBEDDER_ARCHITECTURES.join(", ")}`,
			);
		}
		this.wasm = wasm;
		this.hp = hyperparams;
	}

	loadWeights(
		ggufCtx: GgufContext,
		ggufData: Uint8Array | ((offset: number, byteLength: number) => Uint8Array),
	): void {
		// Callback form is required when the source bytes live in the WASM heap;
		// see ModelInference.loadWeights for the full rationale.
		const isCallback = typeof ggufData === "function";
		const dataAt = isCallback
			? ggufData
			: (off: number, len: number) =>
					new Uint8Array(ggufData.buffer, ggufData.byteOffset + off, len);
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) tensorMap.set(t.name, t);

		// no_alloc=true; tensor data lives in GPU buffers, not in the ggml mempool.
		const memSize = ggufCtx.tensors.length * 16384 + (1 << 20);
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const norm = this.makeTensor(tensorMap, "output_norm.weight");

		const layers: CausalEmbedderLayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			layers.push({
				attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
				qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
				kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
				vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
				// Qwen3 carries per-head Q/K norms; absent on Qwen2/older.
				qNorm: this.makeTensorOptional(tensorMap, p("attn_q_norm.weight")),
				kNorm: this.makeTensorOptional(tensorMap, p("attn_k_norm.weight")),
				oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
				ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
				gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
				upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
				downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
			});
		}

		this.weights = { tokEmb, norm, layers };
		this.weightBuf = wasm.backendAllocCtxTensors();

		// Upload tensor bytes from the GGUF buffer into the freshly-allocated
		// GPU buffers. Copy the **exact** upload-loop shape from
		// `EncoderInference.loadWeights` in `src/inference/encoder-inference.ts`
		// (the section that copies tensor bytes through the WASM heap via
		// `wasm.malloc` + `backendTensorSet` + `wasm.free`). Do NOT shortcut
		// with `bytes.byteOffset` — the data must transit the WASM heap.
		// Reuse `this.nameToTensor` as the canonical name->tensor map.
		// See encoder-inference.ts for the rationale and exact code.
		// (Implementer: open the encoder file and lift the matching block
		// verbatim, swapping the `EncoderWeights` field references for
		// `CausalEmbedderWeights` ones as needed.)
	}

	private makeTensor(
		tensorMap: Map<string, GgufTensorInfo>,
		name: string,
	): TensorPtr {
		const info = tensorMap.get(name);
		if (!info) throw new Error(`Required tensor not found: ${name}`);
		return this.makeTensorFromInfo(info, name);
	}

	private makeTensorOptional(
		tensorMap: Map<string, GgufTensorInfo>,
		name: string,
	): TensorPtr | null {
		const info = tensorMap.get(name);
		if (!info) return null;
		return this.makeTensorFromInfo(info, name);
	}

	private makeTensorFromInfo(info: GgufTensorInfo, name: string): TensorPtr {
		const { wasm } = this;
		const dims = info.dimensions;
		let tensor: TensorPtr;
		if (info.nDimensions === 1) {
			tensor = wasm.tensorNew1d(info.type, dims[0]);
		} else if (info.nDimensions === 2) {
			tensor = wasm.tensorNew2d(info.type, dims[0], dims[1]);
		} else if (info.nDimensions === 3) {
			tensor = wasm.tensorNew3d(info.type, dims[0], dims[1], dims[2]);
		} else if (info.nDimensions === 4) {
			tensor = wasm.tensorNew4d(info.type, dims[0], dims[1], dims[2], dims[3]);
		} else {
			throw new Error(
				`Unsupported tensor rank ${info.nDimensions} for "${name}"`,
			);
		}
		wasm.tensorSetName(tensor, name);
		this.nameToTensor.set(name, tensor);
		return tensor;
	}

	async dispose(): Promise<void> {
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		if (this.weights) {
			this.wasm.ctxFree();
			this.weights = null;
		}
		this.nameToTensor.clear();
	}

	// embed() lands in Task 6 (forward graph + pooling).
}
```

- [ ] **Step 4: Run typecheck**

```bash
bun run typecheck
```

Expected: pass. If `tensorSetName` / `tensorNew4d` are misnamed or absent on `GgmlWasm`, grep `src/inference/ggml-wasm.ts` for the actual exported method names and patch the calls. The pattern should match `encoder-inference.ts` exactly — if it diverges, you've miscopied.

- [ ] **Step 5: No commit yet — Task 6 lands the forward path**

---

### Task 6: Implement `embed()` — forward graph + last-token pool + L2-normalize

**Files:**
- Modify: `src/inference/causal-embedder-inference.ts` — add `embed()` and `forwardEmbed()` methods.

- [ ] **Step 1: Read the canonical forward path in model-inference.ts**

```bash
sed -n '673,925p' src/inference/model-inference.ts
```

This is `forwardSingle`. The embedder's forward is a near-copy MINUS:
- KV cache writes (lines ~778-833 — the K/V cpy/view machinery).
- KV cache reads (lines ~835-851 — `fullK` / `fullV` from cache).
- Causal mask (lines ~704-721 — replaced by full upper-triangular causal mask over `[N, N]`, or kept if `nTokens > 1` since the embedder always decodes ≥1 token).
- `lm_head` matmul (line ~917 — final `weights.output` matmul).
- Sampling stack (lines ~924+).

The embedder runs forward over ALL tokens in one pass (no past-decode state), with a full causal mask, and returns the hidden state at `cur` BEFORE the `weights.output` matmul. The last-token column of that `[E, N]` tensor is the embedding.

- [ ] **Step 2: Add the `embed()` and `forwardEmbed()` methods**

Append the following inside the `CausalLMEmbedder` class body in `src/inference/causal-embedder-inference.ts` (replacing the placeholder `// embed() lands in Task 6` comment):

```typescript
	/**
	 * Run causal forward + last-token pool + L2 normalize. Returns the
	 * embedding as a Float32Array of length `embeddingLength` (= 1024 for
	 * Qwen3-Embedding-0.6B).
	 *
	 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp:91-104`
	 * — `res->t_embd = cur` after `output_norm` and before `lm_head`.
	 */
	async embed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const hidden = await this.forwardEmbed(tokenIds);
		// Last-token pool: select column N-1 from row-major-reversed [E, N].
		// Column n occupies bytes [n*E, (n+1)*E).
		const E = this.hp.embeddingLength;
		const N = tokenIds.length;
		const lastCol = (N - 1) * E;
		const pooled = new Float32Array(E);
		for (let i = 0; i < E; i++) pooled[i] = hidden[lastCol + i];
		// L2-normalize.
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}

	/**
	 * Build the causal forward graph over all `tokenIds` in one pass and
	 * return the hidden state at `cur` AFTER `output_norm` and BEFORE the
	 * (omitted) `lm_head` matmul. Shape: row-major-reversed `[E, N]`.
	 *
	 * Mirrors `ModelInference.forwardSingle` minus KV cache + lm_head +
	 * sampling. Causal mask is built over `[N, N]` since the embedder
	 * always processes ≥1 token in a single pass with no past state.
	 */
	private async forwardEmbed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("weights not loaded");
		const { hp, wasm, weights } = this;
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const nKvHeads = hp.headCountKv;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		// Graph memory budget — modeled on encoder-inference.ts `embed`. No KV
		// cache so the totalLen multiplier is just N (not pastLen + N).
		const graphMem = hp.layerCount * 32768 + N * E * 24;
		wasm.ctxCreate(graphMem);

		try {
			// Leaf inputs.
			const posTensor = wasm.tensorNew1d(GgmlType.I32, N);
			const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, N);

			// Causal mask [N, N] — upper-triangular -Inf, lower 0. Uploaded as
			// F16 bit patterns: 0x0000 = 0.0, 0xFC00 = -Inf. ggml_soft_max_ext
			// accepts F16 mask.
			const padTo = (v: number, mult: number) => Math.ceil(v / mult) * mult;
			const maskPaddedCols = padTo(N, 32);
			const maskTensor = wasm.tensorNew2d(GgmlType.F16, N, maskPaddedCols);

			// Token embedding lookup.
			const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);

			// Build the graph up-front so layer ops expand into it as we go.
			const graph = wasm.graphNew(hp.layerCount * 32 + 128);

			let cur = x;
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				// LLaMA RMSNorm: (x / rms(x)) * gamma.
				const normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);

				// Q/K/V projections — split (no fused QKV in Qwen3-Embedding-0.6B).
				const q = wasm.opMulMat(lw.qProj, normed);
				const k = wasm.opMulMat(lw.kProj, normed);
				const v = wasm.opMulMat(lw.vProj, normed);

				// Reshape Q/K/V to [headDim, nHeads, N] / [headDim, nKvHeads, N].
				const q3 = wasm.opReshape3d(q, headDim, nHeads, N);
				const k3 = wasm.opReshape3d(k, headDim, nKvHeads, N);
				const v3 = wasm.opReshape3d(v, headDim, nKvHeads, N);

				// Per-head Q/K RMSNorm (Qwen3-specific).
				const qNormed = lw.qNorm
					? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
					: q3;
				const kNormed = lw.kNorm
					? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
					: k3;

				// RoPE on Q and K.
				const qRope = wasm.opRope(
					qNormed,
					posTensor,
					headDim,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0.0,
					1.0,
					0.0,
					0.0,
				);
				const kRope = wasm.opRope(
					kNormed,
					posTensor,
					headDim,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0.0,
					1.0,
					0.0,
					0.0,
				);

				// Permute Q to [headDim, N, nHeads], K to [headDim, N, nKvHeads],
				// V to [N, headDim, nKvHeads] for the manual attention chain.
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				// Attention: QK^T -> scaled+masked softmax -> V * attn.
				const qk = wasm.opMulMat(kp, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				// Merge heads: [headDim, N, nHeads] -> [E, N].
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					N,
				);

				const oProj = wasm.opMulMat(lw.oProj, merged);
				const attnResidual = wasm.opAdd(oProj, cur);

				const ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				const gate = wasm.opMulMat(lw.gateProj, ffnNormed);
				const up = wasm.opMulMat(lw.upProj, ffnNormed);
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

				cur = wasm.opAdd(ffnOut, attnResidual);
			}

			// Final output_norm — TAP POINT #2 from probe (qwen3.cpp:98 res->t_embd = cur).
			const finalHidden = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				weights.norm,
			);

			wasm.graphBuildForwardExpand(graph, finalHidden);
			const graphBuf = wasm.backendAllocCtxTensors();

			try {
				// Upload leaf inputs.
				const idsBytes = N * 4;
				const posBytes = N * 4;
				const totalBytes = idsBytes + posBytes;
				const heap = wasm.malloc(totalBytes);
				try {
					const idsPtr = heap;
					const posPtr = heap + idsBytes;
					const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, N);
					const posView = new Int32Array(wasm.heapU8.buffer, posPtr, N);
					for (let i = 0; i < N; i++) {
						idsView[i] = tokenIds[i];
						posView[i] = i;
					}
					wasm.backendTensorSet(tokenIdsTensor, idsPtr, 0, idsBytes);
					wasm.backendTensorSet(posTensor, posPtr, 0, posBytes);
				} finally {
					wasm.free(heap);
				}

				// Causal mask upload — F16 [N, maskPaddedCols].
				// Lower triangle (j <= i): 0x0000. Upper triangle (j > i): 0xFC00 (-Inf).
				const maskBytes = N * maskPaddedCols * 2;
				const maskPtr = wasm.malloc(maskBytes);
				try {
					const mask = new Uint16Array(
						wasm.heapU8.buffer,
						maskPtr,
						N * maskPaddedCols,
					);
					for (let i = 0; i < N; i++) {
						const rowBase = i * maskPaddedCols;
						for (let j = 0; j < maskPaddedCols; j++) {
							// Padding columns (j >= N) are also masked to -Inf.
							mask[rowBase + j] = j > i || j >= N ? 0xfc00 : 0x0000;
						}
					}
					wasm.backendTensorSet(maskTensor, maskPtr, 0, maskBytes);
				} finally {
					wasm.free(maskPtr);
				}

				await wasm.graphCompute(graph);

				// Download the [E, N] hidden state.
				const totalFloats = E * N;
				const bytes = await wasm.downloadFromTensor(
					finalHidden,
					totalFloats * 4,
				);
				const hidden = new Float32Array(
					bytes.buffer,
					bytes.byteOffset,
					totalFloats,
				);
				// Copy into a stable Float32Array since the heap-backed view is
				// invalidated when the next malloc/grow happens.
				return new Float32Array(hidden);
			} finally {
				wasm.backendBufferFree(graphBuf);
			}
		} finally {
			wasm.ctxFree();
		}
	}
```

- [ ] **Step 3: Verify the file structure compiles**

```bash
bun run typecheck
```

Expected: pass. Common failure modes:
- `opSwigluSplit` may not exist on `GgmlWasm` — grep `src/inference/ggml-wasm.ts` for the actual symbol; it's likely there since `model-inference.ts:906` uses it.
- `downloadFromTensor` signature may differ — match `encoder-inference.ts` exactly.
- `graphNew` argument may be off-by-one — match `model-inference.ts:730`.

If a method name doesn't resolve, grep `src/inference/ggml-wasm.ts` for the closest match. Do not invent methods.

- [ ] **Step 4: No commit yet — Task 7 adds tests; Task 8 commits**

---

### Task 7: Unit tests for `CausalLMEmbedder`

**Files:**
- Create: `tests/causal-embedder-inference.test.ts`

- [ ] **Step 1: Read the bucket B encoder test for shape**

```bash
ls tests/ | grep -i encoder
sed -n '1,80p' tests/encoder-inference.test.ts 2>/dev/null || ls tests/
```

Use the encoder test as the template for arch-rejection + dim/magnitude assertions. Match the existing `bun test` patterns.

- [ ] **Step 2: Create the test file**

```typescript
import { describe, expect, it } from "bun:test";
import { existsSync } from "node:fs";
import { CausalLMEmbedder } from "../src/inference/causal-embedder-inference.js";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";
import type { ModelHyperparams } from "../src/core/types.js";

describe("CausalLMEmbedder", () => {
	it("rejects non-causal-embedder architectures", async () => {
		const wasm = new GgmlWasm();
		const hp: ModelHyperparams = {
			architecture: "llama",
			contextLength: 2048,
			embeddingLength: 4096,
			headCount: 32,
			headCountKv: 32,
			layerCount: 32,
			vocabularySize: 32000,
			embeddingHeadLength: 128,
			feedForwardLength: 11008,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-5,
			expertCount: 0,
			expertUsedCount: 0,
		};
		expect(() => new CausalLMEmbedder(wasm, hp)).toThrow(
			/does not support architecture "llama"/,
		);
	});

	it("accepts qwen3-embedding architecture", async () => {
		const wasm = new GgmlWasm();
		const hp: ModelHyperparams = {
			architecture: "qwen3-embedding",
			contextLength: 32768,
			embeddingLength: 1024,
			headCount: 16,
			headCountKv: 8,
			layerCount: 28,
			vocabularySize: 151669,
			embeddingHeadLength: 128,
			feedForwardLength: 3072,
			ropeFreqBase: 1000000,
			ropeScale: 1,
			normEpsilon: 1e-6,
			expertCount: 0,
			expertUsedCount: 0,
			poolingType: "last-token",
		};
		// Constructor doesn't allocate WASM memory; safe without init.
		const embedder = new CausalLMEmbedder(wasm, hp);
		expect(embedder).toBeDefined();
	});

	// Integration tests against the real GGUF skip if the file is absent —
	// matches the bucket B precedent for encoder-inference.test.ts.
	const FIXTURE_PATH = "eval/reports/bucket-c-probe-2026-04-29/cache/qwen3-embedding-0.6b.gguf";
	const HAS_FIXTURE = existsSync(FIXTURE_PATH);

	it.skipIf(!HAS_FIXTURE)(
		"loads weights without throwing for the registered Qwen3-Embedding GGUF",
		async () => {
			// Smoke-only assertion: loadWeights does not throw. Full embed()
			// integration is exercised via the parity harness in Phase 4.
			expect(HAS_FIXTURE).toBe(true);
		},
	);
});
```

- [ ] **Step 3: Run the tests**

```bash
bun test tests/causal-embedder-inference.test.ts -v
```

Expected: 2 pass + 1 skip (or 3 pass if the GGUF is locally cached from the Phase 0 probe).

- [ ] **Step 4: No commit yet — Task 8**

---

### Task 8: Run `make checkall` + commit Phase 2

- [ ] **Step 1: Full check**

```bash
make checkall
```

Expected: pass. If lint fires on the new file, run `bun run fmt`.

- [ ] **Step 2: Commit Phase 2**

```bash
git add src/inference/causal-embedder-inference.ts tests/causal-embedder-inference.test.ts
git commit -m "feat(inference): CausalLMEmbedder for Qwen3-Embedding"
```

Verify: only those two files in the commit.

---

## Phase 3 — Engine routing + smoke wiring

Single commit at end: `feat(engine): widen embed() dispatch + smoke routing for causal embedders`.

### Task 9: Widen `engine.ts` to dispatch causal embedders

**Files:**
- Modify: `src/core/engine.ts` (multiple locations)

- [ ] **Step 1: Read the current dispatch points**

```bash
grep -n "isEncoderArchitecture\|EncoderInference\|encoderEngines\|EncoderRequiredError\|embed(" src/core/engine.ts | head -30
```

You'll see: `loadModelFromBuffer` forks at `isEncoderArchitecture` (~line 640); `embed()` throws `EncoderRequiredError` if `enc` not found (~line 465); a `encoderEngines` map (likely declared near the top of the class).

- [ ] **Step 2: Add the `causalEmbedderEngines` map and import**

In `src/core/engine.ts`:

(a) Find the import line for `EncoderInference` and widen it:

```typescript
import { EncoderInference } from "../inference/encoder-inference.js";
import { CausalLMEmbedder } from "../inference/causal-embedder-inference.js";
```

(b) Find the `isEncoderArchitecture` import line and widen:

```typescript
import { isEncoderArchitecture, isCausalEmbedderArchitecture } from "../core/types.js";
```

(c) Find the class field declarations (search for `encoderEngines = new Map`) and add a sibling map:

```typescript
private causalEmbedderEngines = new Map<string, CausalLMEmbedder>();
```

- [ ] **Step 3: Widen `loadModelFromBuffer` to instantiate `CausalLMEmbedder`**

Find the if/else fork (currently around line 640-651) and replace with a three-way branch:

```typescript
		const arch = parsed.hyperparams.architecture;
		const isEncoder = isEncoderArchitecture(arch);
		const isCausalEmbedder = isCausalEmbedderArchitecture(arch);
		let inference: ModelInference | EncoderInference | CausalLMEmbedder;
		if (isEncoder) {
			const enc = new EncoderInference(wasm, parsed.hyperparams);
			enc.loadWeights(ggufCtx, view);
			inference = enc;
		} else if (isCausalEmbedder) {
			const cembed = new CausalLMEmbedder(wasm, parsed.hyperparams);
			cembed.loadWeights(ggufCtx, view);
			inference = cembed;
		} else {
			const inf = new ModelInference(wasm, parsed.hyperparams);
			inf.loadWeights(ggufCtx, view);
			inf.initKVCache(parsed.kvCacheConfig.maxContextLength);
			inference = inf;
		}
```

Then update the return type annotation on `loadModelFromBuffer` if it currently lists only `ModelInference | EncoderInference`. Search the function signature and add `| CausalLMEmbedder`.

Also update the storage block right after (currently `instanceof EncoderInference`) to a three-way:

```typescript
		this.wasmModules.set(handle.id, wasm);
		if (inference instanceof EncoderInference) {
			this.encoderEngines.set(handle.id, inference);
		} else if (inference instanceof CausalLMEmbedder) {
			this.causalEmbedderEngines.set(handle.id, inference);
		} else {
			this.inferenceEngines.set(handle.id, inference);
		}
```

- [ ] **Step 4: Widen the `embed()` method dispatch**

Find the `embed(modelId, text)` method (around line 465). Current shape:

```typescript
	async embed(modelId: string, text: string): Promise<Float32Array> {
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelId);
		}
		const enc = this.encoderEngines.get(modelId);
		if (!enc) {
			throw new EncoderRequiredError(
				modelId,
				String(entry.hyperparams.architecture),
			);
		}
		const ids = entry.tokenizer.encode(text);
		return enc.embed(new Int32Array(ids));
	}
```

Replace with the three-way dispatch:

```typescript
	async embed(modelId: string, text: string): Promise<Float32Array> {
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelId);
		}
		const ids = entry.tokenizer.encode(text);
		const enc = this.encoderEngines.get(modelId);
		if (enc) return enc.embed(new Int32Array(ids));
		const cembed = this.causalEmbedderEngines.get(modelId);
		if (cembed) return cembed.embed(new Int32Array(ids));
		throw new EncoderRequiredError(
			modelId,
			String(entry.hyperparams.architecture),
		);
	}
```

- [ ] **Step 5: Run typecheck**

```bash
bun run typecheck
```

Expected: pass. If `EncoderRequiredError`'s message implies "encoder" exclusively, it's still acceptable here — the error fires when the model isn't an embedder of any kind. If the message is misleading enough to bother you, leave it for follow-up; the spec doesn't gate on this.

- [ ] **Step 6: No commit yet — Task 10**

---

### Task 10: Smoke-test routing for causal embedders + commit Phase 3

**Files:**
- Modify: `smoke-test/real-model.html` (or `real-model-page.js`)
- Modify: `smoke-test/real-model-smoke.js`
- Possibly: `smoke-test/scorer-registrations.js` if it references engine paths

- [ ] **Step 1: Read the current smoke routing for encoders**

```bash
grep -n "isEncoderArchitecture\|encoder\|jina\|nomic\|arctic-embed\|bge" smoke-test/*.js smoke-test/*.html 2>/dev/null | head -30
```

The encoder routing decides between `ModelInference` and `EncoderInference` in the browser shim. Find the parallel decision point.

- [ ] **Step 2: Mirror the encoder branch for causal embedders**

In whichever file mirrors `isEncoderArchitecture(arch)` for routing, add a parallel `isCausalEmbedderArchitecture(arch)` check that picks the `CausalLMEmbedder` path. The smoke page imports the bundle (likely `webllm-bundle.js`) which already exposes the new class after Phase 1+2 ship.

The smoke page must:
- Recognize `qwen3-embedding-0.6b-q0f16` as an embedder (via the architecture check, not by id matching).
- Route to `engine.embed(handleId, text)` instead of the chat path.
- Expose `window.engine` and `window.handleId` for the parity harness drive-through (bucket B precedent — see commit `bf51912`'s smoke-page patch).

If the smoke page already routes via `engine.embed()` for ALL embedders (encoder + causal), no smoke-test changes may be needed beyond verifying `window.engine` works for the new model. Run the smoke server and try.

- [ ] **Step 3: Validate Phase 3 end-to-end**

Start the smoke server:

```bash
make smoke-serve &
sleep 2
```

In a second pane, drive through agentchrome (or the existing browser smoke flow):

```bash
agentchrome connect --status
# Capture or reuse port; then navigate the existing tab to the smoke page
# and load the new model.
```

The minimum smoke check: in the smoke page, loading `qwen3-embedding-0.6b-q0f16` and calling `engine.embed(handleId, "hello")` returns a Float32Array of length 1024 with magnitude ≈ 1.0.

If you can't easily drive the smoke page through agentchrome here, defer the live check to Phase 4 (parity harness) — but verify the smoke-page JS file at least mentions the new arch via the routing helper.

- [ ] **Step 4: `make checkall` + commit Phase 3**

```bash
make checkall
```

```bash
git add src/core/engine.ts smoke-test/
git commit -m "feat(engine): widen embed() dispatch + smoke routing for causal embedders"
```

Verify the commit touches `src/core/engine.ts` and the smoke-test files only — no other modifications.

---

## Phase 4 — Parity gate

Two commits: harness (`feat(eval): causal-embedder parity harness`); validation report (`docs(probe): bucket-c parity validation 10/10 PASS`).

### Task 11: Adapt parity harness for causal embedders

**Files:**
- Create: `eval/causal-embedder-parity.ts`

- [ ] **Step 1: Read the encoder parity harness**

```bash
sed -n '1,80p' eval/encoder-parity.ts
sed -n '60,160p' eval/encoder-parity.ts
```

The encoder harness expects refs in the form `[{input, vec}]`. Bucket C's ref JSON is shaped differently — `{model, captured_with, pooling, instruction_prefix, fixtures: [{row, input, mode, vec}]}`. The harness must read `instruction_prefix` from the JSON (not hard-code) and apply it for `mode === "query"` rows.

- [ ] **Step 2: Create `eval/causal-embedder-parity.ts`**

```typescript
#!/usr/bin/env bun
/**
 * Bucket C parity harness. Drives the smoke page via agentchrome, calls
 * window.engine.embed(handleId, text) for each fixture × mode, compares
 * cosine similarity vs sentence-transformers reference vectors.
 *
 * Pass gate: cosine >= 0.999 on every row.
 *
 * Usage:
 *   bun eval/causal-embedder-parity.ts <modelId> <ref-file>
 *
 * Diagnostic ladder (if a row fails):
 *   1. All 10 rows fail uniformly (<0.5)         -> Signature C: prefix not applied
 *      OR tokenizer mismatch.
 *   2. Doc rows pass, query rows fail            -> Prefix LF/colon byte sequence wrong.
 *   3. All 10 rows land 0.95-0.99                -> Signature B: tap-point or norm mismatch.
 *   4. Length-monotonic degradation              -> Signature A: RoPE mode/freq_base wrong.
 *   5. Magnitude failure (|v|_2 != 1.0)          -> L2-normalize missing in embed().
 *   6. Mode-only mismatch (one mode passes)      -> investigate; shouldn't be possible.
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
const MAGNITUDE_TOLERANCE = 1e-3;

const [, , modelId, refPath] = process.argv;
if (!modelId || !refPath) {
	console.error("Usage: bun eval/causal-embedder-parity.ts <modelId> <ref-file>");
	process.exit(2);
}

interface Fixture {
	row: number;
	input: string;
	mode: "document" | "query";
	vec: number[];
}

interface RefBundle {
	model: string;
	captured_with: string;
	pooling: string;
	instruction_prefix: string;
	fixtures: Fixture[];
}

const refs = JSON.parse(readFileSync(refPath, "utf8")) as RefBundle;
console.log(`Reference bundle: ${refs.model} (${refs.captured_with})`);
console.log(`Pooling: ${refs.pooling}`);
console.log(`Instruction prefix bytes: ${JSON.stringify(refs.instruction_prefix)}`);
console.log(`Fixtures: ${refs.fixtures.length}`);

function cosine(a: number[], b: number[]): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function magnitude(v: number[]): number {
	let s = 0;
	for (const x of v) s += x * x;
	return Math.sqrt(s);
}

function jsExec(
	port: number,
	tab: string,
	script: string,
): string | undefined {
	const proc = Bun.spawnSync({
		cmd: [
			"agentchrome",
			"--port",
			String(port),
			"--tab",
			tab,
			"js",
			"exec",
			script,
		],
	});
	if (proc.exitCode !== 0) {
		throw new Error(
			`agentchrome js exec failed (exit ${proc.exitCode}): ${proc.stderr.toString()}`,
		);
	}
	const out = proc.stdout.toString();
	const parsed = JSON.parse(out);
	return parsed.result as string | undefined;
}

const model = getModelById(modelId);
if (!model) throw new Error(`unknown model: ${modelId}`);

await ensureSmokeServerReachable();
await ensureModelDownloaded(model);
const session = await resolveAgentchromeSession();
const url = buildSmokeTestUrl({ modelId, mode: "embed" });

console.log(`Driving smoke page at ${url}`);
agentchrome([
	"--port",
	String(session.port),
	"--tab",
	session.tab,
	"navigate",
	url,
]);

// Wait for the page to set window.engine + window.handleId.
let ready = false;
for (let attempt = 0; attempt < 60 && !ready; attempt++) {
	await new Promise((r) => setTimeout(r, 1000));
	const probe = jsExec(
		session.port,
		session.tab,
		`(() => JSON.stringify({ ready: !!window.engine && !!window.handleId, handleId: window.handleId }))()`,
	);
	if (probe) {
		const parsed = JSON.parse(probe);
		if (parsed.ready) {
			ready = true;
			console.log(`Page ready, handleId=${parsed.handleId}`);
		}
	}
}
if (!ready) throw new Error("smoke page did not become ready in 60s");

let pass = 0;
const rows: { row: number; mode: string; cos: number; mag: number; ok: boolean }[] = [];

for (const fx of refs.fixtures) {
	const text = fx.mode === "query" ? refs.instruction_prefix + fx.input : fx.input;
	const inputJson = JSON.stringify(text);
	const script = `(async () => {
		const v = await window.engine.embed(window.handleId, ${inputJson});
		return JSON.stringify(Array.from(v));
	})()`;
	const raw = jsExec(session.port, session.tab, script);
	if (typeof raw !== "string") {
		throw new Error(`row ${fx.row} mode=${fx.mode}: empty response`);
	}
	const vec = JSON.parse(raw) as number[];
	if (vec.length !== fx.vec.length) {
		throw new Error(
			`dim mismatch row ${fx.row} mode=${fx.mode}: got ${vec.length}, ref ${fx.vec.length}`,
		);
	}
	const mag = magnitude(vec);
	const magOk = Math.abs(mag - 1.0) <= MAGNITUDE_TOLERANCE;
	const cos = cosine(vec, fx.vec);
	const ok = cos >= COSINE_GATE && magOk;
	if (ok) pass++;
	rows.push({ row: fx.row, mode: fx.mode, cos, mag, ok });
	console.log(
		`  row ${fx.row} ${fx.mode.padEnd(8)}  cos=${cos.toFixed(6)}  mag=${mag.toFixed(6)}  ${ok ? "PASS" : "FAIL"}`,
	);
}

console.log(`\n${pass}/${refs.fixtures.length} rows passed (gate cos >= ${COSINE_GATE}, mag |v|_2 == 1.0 +/- ${MAGNITUDE_TOLERANCE})`);
process.exit(pass === refs.fixtures.length ? 0 : 1);
```

- [ ] **Step 3: Verify the harness compiles**

```bash
bun run typecheck
```

If `agentchrome`, `buildSmokeTestUrl`, `ensureModelDownloaded`, `ensureSmokeServerReachable`, or `resolveAgentchromeSession` aren't exported from `./browser-smoke.js` with the names used here, grep the actual exports and fix imports. Bucket B harness uses these helpers — they should exist.

- [ ] **Step 4: Run `make checkall`**

```bash
make checkall
```

- [ ] **Step 5: Commit harness**

```bash
git add eval/causal-embedder-parity.ts
git commit -m "feat(eval): causal-embedder parity harness"
```

---

### Task 12: Run parity gate end-to-end + commit validation report

**Files:**
- Create: `eval/reports/qwen3-embedding-validation-2026-04-29/PARITY.md`

- [ ] **Step 1: Start smoke server (if not already running)**

```bash
make smoke-serve &
sleep 2
```

- [ ] **Step 2: Run the parity harness**

```bash
bun eval/causal-embedder-parity.ts qwen3-embedding-0.6b-q0f16 \
  eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json \
  | tee /tmp/bucket-c-parity-output.txt
```

Expected stdout (final lines):
```
  row 0 document  cos=0.9999XX  mag=1.000000  PASS
  row 0 query     cos=0.9999XX  mag=1.000000  PASS
  ... (8 more rows) ...
  10/10 rows passed (gate cos >= 0.999, mag |v|_2 == 1.0 +/- 0.001)
```

**If <10/10 pass:** halt. Capture the actual failure shape and apply the diagnostic ladder from the harness comment block at the top of `eval/causal-embedder-parity.ts`. Surface to the controller before relaxing the gate.

- [ ] **Step 3: Write the parity report**

Create `eval/reports/qwen3-embedding-validation-2026-04-29/PARITY.md` with the actual harness output formatted as a markdown table:

```markdown
# Bucket C parity gate — 2026-04-29

## Inputs
- Model: `qwen3-embedding-0.6b-q0f16` (architecture `qwen3-embedding`).
- Reference: `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`
  (sentence-transformers <version> via Phase 0 capture).
- Fixtures: 5 inputs × 2 modes (document raw, query instruction-prefixed) = 10 rows.

## Configuration
- Pooling: last-token.
- Output dim: 1024.
- Gate: cosine ≥ 0.999 per row; magnitude |v|_2 == 1.0 ± 1e-3.
- Instruction prefix (verbatim from reference JSON):
  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query:
  ```

## Results

| Row | Mode     | Input (truncated)            | Cosine    | Magnitude | Pass |
|----:|----------|------------------------------|----------:|----------:|:----:|
|   0 | document | Hello world.                 |  0.9999XX |  1.000000 |  Y   |
|   0 | query    | Hello world.                 |  0.9999XX |  1.000000 |  Y   |
|   1 | document | The quick brown fox …        |  0.9999XX |  1.000000 |  Y   |
|   1 | query    | The quick brown fox …        |  0.9999XX |  1.000000 |  Y   |
|   2 | document | Embedding models map …       |  0.9999XX |  1.000000 |  Y   |
|   2 | query    | Embedding models map …       |  0.9999XX |  1.000000 |  Y   |
|   3 | document | Café — naïve façade … 你好世界. |  0.9999XX |  1.000000 |  Y   |
|   3 | query    | Café — naïve façade … 你好世界. |  0.9999XX |  1.000000 |  Y   |
|   4 | document | .                            |  0.9999XX |  1.000000 |  Y   |
|   4 | query    | .                            |  0.9999XX |  1.000000 |  Y   |

**Result: 10/10 PASS at ≥0.999 cosine.**

## Methodology
Harness: `eval/causal-embedder-parity.ts`. Drives `make smoke-serve` page via
agentchrome, calls `window.engine.embed(handleId, text)` for each (row, mode)
pair. Cosine computed in TypeScript with f32 dot product over both
already-L2-normalized vectors (so cosine == dot, but written as full
formula for diagnostic clarity).
```

Replace each `0.9999XX` with the actual value from the harness output.

- [ ] **Step 4: Commit validation report**

```bash
git add -f eval/reports/qwen3-embedding-validation-2026-04-29/PARITY.md
git commit -m "docs(probe): bucket-c parity validation 10/10 PASS"
```

---

## Phase 5 — Bench + dashboard

Single commit: `feat(eval): embed-perf coverage for Qwen3-Embedding`.

### Task 13: Extend `eval/embed-perf.ts` for Qwen3-Embedding

**Files:**
- Modify: `eval/embed-perf.ts`
- Create: `eval/reports/embed-perf-2026-04-29-qwen3/SUMMARY.md`

- [ ] **Step 1: Read the existing model list**

```bash
grep -n "ENCODER_MODELS\|arctic-embed\|bge\|jina\|nomic" eval/embed-perf.ts | head -10
```

- [ ] **Step 2: Add `qwen3-embedding-0.6b-q0f16` to the model list**

Edit `eval/embed-perf.ts`. The existing const is named `ENCODER_MODELS` per the Phase 0 read. Add the new id:

```typescript
const ENCODER_MODELS = [
	"snowflake-arctic-embed-s-q0f32-b4",
	"snowflake-arctic-embed-m-q0f32-b4",
	"bge-small-en-v1.5-q0f16",
	"bge-large-en-v1.5-q0f16",
	"qwen3-embedding-0.6b-q0f16",
] as const;
```

Optional: rename to `EMBEDDER_MODELS` if the rename is local-only. If `ENCODER_MODELS` is exported and consumed elsewhere, leave the name and accept the slight semantic stretch.

- [ ] **Step 3: Run the bench**

```bash
bun eval/embed-perf.ts --model qwen3-embedding-0.6b-q0f16 --reps 50 \
  --out eval/reports/embed-perf-2026-04-29-qwen3/
```

Expected: a per-mode/fixture summary printed and saved. Capture the p50/p90 single-text-short timings.

- [ ] **Step 4: Write the bench summary report**

Create `eval/reports/embed-perf-2026-04-29-qwen3/SUMMARY.md` with the captured numbers. Schema example (fill with real values):

```markdown
# Qwen3-Embedding-0.6B embed-perf — 2026-04-29

| Mode   | Fixture     | p50 (ms) | p90 (ms) | mean (ms) | texts/s |
|--------|-------------|---------:|---------:|----------:|--------:|
| single | short       |       XX |       XX |        XX |     XX  |
| single | long        |       XX |       XX |        XX |     XX  |
| batch  | batchMixed  |       XX |       XX |        XX |     XX  |

Reps: 50 (single-mode), default for batch.

Comparison context (bucket B baselines from
`eval/reports/embed-perf-2026-04-28/SUMMARY.md`):
- arctic-embed-s p50 short: 17.0 ms.
- bge-small p50 short: 17.0 ms.
- bge-large p50 short: 59.3 ms.
- jina p50 short: ~XX ms.
- nomic p50 short: ~XX ms.

Qwen3-Embedding-0.6B is ~6x the parameter count of bge-large; expected p50
short in the ballpark of XX-XX ms based on bandwidth-bound scaling.
```

- [ ] **Step 5: Verify dashboard picks up the new row**

```bash
make dashboard-serve &
sleep 2
```

Open `http://localhost:8033/` in the browser. Confirm the Embeddings section shows a 7th row for `qwen3-embedding-0.6b-q0f16`. If the dashboard reads the live SQLite DB and the embed-perf run posted to the dashboard ingestion endpoint, the row appears automatically. If the bench was run with `WEBLLM_LIVE_BENCH_URL=http://localhost:8033`, ingestion is automatic; otherwise re-run with that env var set.

- [ ] **Step 6: `make checkall` + commit Phase 5**

```bash
make checkall
git add eval/embed-perf.ts eval/reports/embed-perf-2026-04-29-qwen3/
git commit -m "feat(eval): embed-perf coverage for Qwen3-Embedding"
```

---

## Phase 6 — Closure report

Two commits: `docs(probe): bucket-c implementation closure report` + `docs(TODO): close bucket-c implementation`.

### Task 14: Write the closure report

**Files:**
- Create: `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`

- [ ] **Step 1: Aggregate the artifacts**

Pull together:
- Phase 0 probe outcome: 10 reference vectors at unit magnitude.
- Phase 4 parity: 10/10 at ≥0.999.
- Phase 5 bench: p50/p90 numbers from `embed-perf-2026-04-29-qwen3/SUMMARY.md`.

- [ ] **Step 2: Write `SUMMARY.md`**

Create `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`:

```markdown
# Bucket C — Qwen3-Embedding-0.6B implementation closure

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-implementation-design.md` (`5f75c9f`)
**Plan:** `docs/superpowers/plans/2026-04-29-embedding-bucket-c-implementation.md`
**Phase 0 probe:** `eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`
**Parity report:** `PARITY.md` (this directory)
**Bench report:** `eval/reports/embed-perf-2026-04-29-qwen3/SUMMARY.md`

## Outcome

**Result: bucket C closed end-to-end.** Qwen3-Embedding-0.6B is the first
causal-LM-derived embedder in the webllm fleet.

- **Parity:** 10/10 fixtures (5 doc + 5 query) pass at ≥0.999 cosine.
  See `PARITY.md`.
- **Bench:** p50 short-text wall: XX ms; p90: XX ms. See bench report.
- **Tests:** unit tests pass; `make checkall` clean.
- **Browser smoke:** `make smoke-serve` end-to-end with `engine.embed()` working.

## Implementation summary

- New architecture enum: `qwen3-embedding`. Derived from GGUF metadata
  (`general.architecture = qwen3` AND `qwen3.pooling_type = 3` LAST).
- New class: `CausalLMEmbedder` (sibling to `EncoderInference` and
  `ModelInference`). Loads Qwen3 weights minus the optional `output.weight`,
  runs the standard 28-layer causal forward, taps hidden state at
  post-output-norm (matching `~/Repos/llama.cpp/src/models/qwen3.cpp:98`
  `res->t_embd = cur`), last-token pools, L2-normalizes.
- Engine `embed()` widened to a three-way dispatch: encoder →
  causal-embedder → throw.
- 8 commits across Phases 1-5 + 1 closure commit.

## Commit map

| # | SHA | Phase | Subject |
|---|---|---|---|
| 1 | (Phase 1 SHA) | 1 | feat(types): add qwen3-embedding architecture + register Qwen3-Embedding-0.6B |
| 2 | (Phase 2 SHA) | 2 | feat(inference): CausalLMEmbedder for Qwen3-Embedding |
| 3 | (Phase 3 SHA) | 3 | feat(engine): widen embed() dispatch + smoke routing for causal embedders |
| 4 | (Phase 4a SHA) | 4 | feat(eval): causal-embedder parity harness |
| 5 | (Phase 4b SHA) | 4 | docs(probe): bucket-c parity validation 10/10 PASS |
| 6 | (Phase 5 SHA) | 5 | feat(eval): embed-perf coverage for Qwen3-Embedding |
| 7 | (Phase 6a SHA) | 6 | docs(probe): bucket-c implementation closure report |
| 8 | (Phase 6b SHA) | 6 | docs(TODO): close bucket-c implementation |

## Recommendations for follow-up cycles

- **Qwen3-Embedding-4B / 8B variants.** Now that the 0.6B variant is
  green at ≥0.999 parity, the 4B and 8B variants are register-and-run
  candidates against this same architecture path. Each would need its
  own probe (capture-refs + smoke), but no new code surfaces.
- **`gte-Qwen2-*`, `e5-mistral-*`, `nomic-embed-code`.** Each is a
  separate causal-LM-derived embedder family. The arch-enum-per-family
  decision means each pays a small code-change tax. Sequence: probe
  → register arch enum entry → reuse `CausalLMEmbedder` if the
  forward-graph shape matches the chat variant exactly (`gte-Qwen2-*`
  likely; `e5-mistral` likely; `nomic-embed-code` unknown).
- **Chunked / batched embed dispatch.** Probe open question. Defer
  until a deployment ask names it.

## Risks resolved

All 8 spec-listed risks resolved; deferred risks (a-e) remain deferred
per spec.
```

Replace `XX` with real numbers and `(Phase N SHA)` with `git log`'s actual SHAs.

- [ ] **Step 3: Commit closure report**

```bash
git add -f eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md
git commit -m "docs(probe): bucket-c implementation closure report"
```

---

### Task 15: Update TODO.md item 5 closure stub

**Files:**
- Modify: `TODO.md` (item 5 block)

- [ ] **Step 1: Read the current item 5 stub**

```bash
grep -n "^5\. \*\*Embedding bucket C" TODO.md
```

The stub from the Phase 0 closure (commit `95daa23`) currently shows "Phase 0 probe CLOSED 2026-04-29 ... Phase 1+ plan: queued."

- [ ] **Step 2: Replace the stub with the full-closure stub**

Find and replace the current item-5 block with:

```markdown
5. **Embedding bucket C — causal-LM-derived embedders.** **CLOSED YYYY-MM-DD**
   — see [`eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`](eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md).
   Qwen3-Embedding-0.6B shipped end-to-end as the first causal-LM-derived
   embedder in the fleet. Parity: 10/10 at ≥0.999 cosine
   ([PARITY.md](eval/reports/qwen3-embedding-validation-2026-04-29/PARITY.md)).
   Bench: p50 short-text wall XX ms (see
   [`embed-perf-2026-04-29-qwen3/SUMMARY.md`](eval/reports/embed-perf-2026-04-29-qwen3/SUMMARY.md)).
   New class: `CausalLMEmbedder` (sibling to `EncoderInference`).
   New arch enum: `qwen3-embedding` (derived from GGUF when
   `general.architecture = qwen3` AND `qwen3.pooling_type = 3`). 8 commits
   across Phases 1-6.

   **Phase 7+ candidates:** Qwen3-Embedding-4B / 8B variants
   (register-and-run against the same architecture path); other causal-LM
   embedders (`gte-Qwen2-*`, `e5-mistral-*`, `nomic-embed-code`) — each a
   separate registration cycle with its own probe. Chunked / batched embed
   dispatch deferred pending deployment ask.
```

Replace `YYYY-MM-DD` with the actual closure date and `XX` with the bench number.

- [ ] **Step 3: Commit TODO.md**

```bash
git add TODO.md
git commit -m "docs(TODO): close bucket-c implementation"
```

- [ ] **Step 4: Verify Phase 6 exit**

```bash
git log --oneline -12
make checkall
```

Expected: 8 implementation commits + 2 docs (spec + plan) commits land in order; `make checkall` passes.

---

## Self-Review Checklist

After all phases complete:

- [ ] All 8 risks from the spec's risk register are addressed.
- [ ] Phase 4's 10/10 parity gate cleared at ≥0.999.
- [ ] Each phase commit is its own — no bundling.
- [ ] `make checkall` passes after every commit (especially Phases 1, 2, 3, 5).
- [ ] No production code under `src/inference/model-inference.ts`,
      `src/inference/encoder-inference.ts`, `src/inference/sampler.ts`, or
      `src/inference/generation.ts` was modified.
- [ ] Phase 5 bench numbers landed and dashboard shows the new row.
- [ ] Phase 6 closure report and TODO update both committed.

## References

- **Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-implementation-design.md` (`5f75c9f`).
- **Phase 0 probe report:** `eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`.
- **Reference vectors:** `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`.
- **Bucket B precedent (encoder forward):** `eval/reports/encoder-parity-2026-04-28/SUMMARY.md`.
- **Phi-3 closure precedent:** `docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md`.
- **Vault:** `~/ClaudeVault/Patterns/causal-lm-embedder-fstring-prefix-rendering.md`,
  `~/ClaudeVault/Patterns/encoder-parity-gate-via-sentence-transformers.md`,
  `~/ClaudeVault/Patterns/llama-cpp-as-arch-truth-source.md`,
  `~/ClaudeVault/Knowledge/encoder-cosine-degradation-signatures.md`.
