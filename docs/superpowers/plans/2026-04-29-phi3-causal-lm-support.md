# Phi-3 Causal LM Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add causal-LM inference support for Phi-3 family GGUFs, gated on `architecture === "phi3"`, with zero impact on the 11 currently-shipping causal models. Validate end-to-end on `phi-3.5-mini-q4f16` (3.82B, 32 layers, hidden 3072, no GQA, intermediate 8192).

**Architecture:** Phi-3 GGUFs ship fused-QKV (`blk.N.attn_qkv.weight` shaped [3·n_embd, n_embd]) and fused gate-up FFN (`blk.N.ffn_up.weight` shaped [2·n_ff, n_embd]) instead of split projections. Implementation mirrors the Bucket B nomic encoder pattern (`src/inference/encoder-inference.ts:263-296`, commit `3982af9`): one matmul + view-slice instead of three separate matmuls. New code path is fully architecture-gated; existing 11 models hit the unchanged split-QKV path.

**Tech Stack:** TypeScript, Bun, the patched `webllm-browser-patches` llama.cpp branch, ggml-webgpu, agentchrome for browser-side smoke + eval.

---

## Pre-flight context (load this once before starting)

**Reference implementation pattern** — the encoder fused-QKV path shipped as part of Bucket B. Read these spans before Phase 2:

- `src/inference/encoder-inference.ts:117-119` — fused tensor declaration
- `src/inference/encoder-inference.ts:263-296` — forward fused matmul + 3-way view3d slice
- `src/inference/encoder-inference.ts:267` — `F32_BYTES` import + `elemSize` derivation

**Files that will change in this plan:**

- Modify `src/core/types.ts` — add `"phi3"` to `ModelArchitecture` union
- Modify `eval/models.ts:268-283` — flip architecture to `"phi3"`, confirm spec, force-add Mistral-Nemo-style comment
- Modify `eval/smoke-profiles.ts` — add `phi-3.5-mini-warm` profile entry
- Modify `src/inference/model-inference.ts` — extend `LayerWeights` with optional fused fields, branch loader on `architecture === "phi3"`, branch forward graph on fused-QKV / fused-gate-up presence, add `outputBias`
- Modify `src/inference/lightweight.ts` (if relevant) — verify it doesn't need symmetric updates
- Test: `tests/phi3-fused-loader.test.ts` (new, unit test for view-offset math)
- Create: `eval/reports/phi-3-validation-2026-04-29/SUMMARY.md` (closure report)

**Files explicitly NOT changing:** WGSL shaders, the WASM bridge in `ggml-wasm.ts`, KV-cache layout, RoPE call sites for non-Phi models, `chat-template.ts` (already has `phi3`).

**Reversibility:** every change is gated on `architecture === "phi3"` or on the `qkvFused !== null` check. Reverting Phase 2 alone reverts the lever; Phase 1 (registration) leaves a known-failing entry that the smoke probe will catch on next run.

---

## Phase 1 — Probe + Register

### Task 1: HEAD-verify the GGUF on bartowski

**Files:**
- (probe only — no file changes)

- [ ] **Step 1: HEAD-probe the canonical Phi-3.5-mini Q4_K_M file**

Run:

```bash
curl -sIL "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf" 2>&1 | grep -iE "^(HTTP|content-length|x-linked-size|location)" | head
```

Expected: `HTTP/2 302` → `HTTP/1.1 200 OK` and `Content-Length` / `x-linked-size` near 2.39 GB (~2,393,232,608 bytes).

- [ ] **Step 2: Inspect the GGUF metadata for architecture string**

Run (from a Bun script — NOT curl):

```bash
bun run -e "
const url='https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf';
const r=await fetch(url,{headers:{Range:'bytes=0-65535'}});
const buf=new Uint8Array(await r.arrayBuffer());
console.log('Magic:',new TextDecoder().decode(buf.slice(0,4)));
const dv=new DataView(buf.buffer);
console.log('Version:',dv.getUint32(4,true));
console.log('TensorCount:',dv.getBigUint64(8,true));
console.log('MetadataKVCount:',dv.getBigUint64(16,true));
// Quick scan: dump strings near the front so we can eyeball arch + tensor names.
const ascii=[...buf.slice(24,16384)].map(b=>(b>=32&&b<127)?String.fromCharCode(b):'.').join('');
console.log(ascii.match(/general\\.[a-z._]+/g)?.slice(0,10));
console.log(ascii.match(/blk\\.0\\.[a-z._]+/g)?.slice(0,15));
"
```

Expected: `Magic: GGUF`, the metadata-key list includes `general.architecture` (value will be `phi3` per llama.cpp arch table), and the layer-0 tensor list includes `blk.0.attn_qkv.weight` and `blk.0.ffn_up.weight` (note: NO `attn_q.weight` / `ffn_gate.weight`).

- [ ] **Step 3: Document the inventory**

If the probe confirms `attn_qkv.weight` + fused `ffn_up.weight`: continue. If the GGUF instead has split QKV (publisher-dependent), STOP — the plan's Path B premise is wrong and we drop back to Path A (loader-only views).

### Task 2: Add `"phi3"` to the architecture union

**Files:**
- Modify: `src/core/types.ts:80-92`

- [ ] **Step 1: Read current state**

Read `src/core/types.ts:80-92` to confirm the union shape.

- [ ] **Step 2: Add the new variant**

Edit `src/core/types.ts` to extend the union:

```typescript
export type ModelArchitecture =
	| "llama"
	| "mistral"
	| "phi"
	| "phi3"
	| "gemma"
	| "qwen"
	| "qwen2"
	| "qwen3"
	| "mixtral"
	| "deepseek"
	| "bert"
	| "nomic-bert"
	| "jina-bert-v2";
```

(Insert `"phi3"` directly after `"phi"`. The pre-existing `"phi"` is for older Phi-1/Phi-2 GGUFs; do not remove it.)

- [ ] **Step 3: Run typecheck to confirm no immediate breakage**

Run: `make typecheck`
Expected: PASS.

- [ ] **Step 4: Commit (no other changes yet)**

```bash
git add src/core/types.ts
git commit -m "types: add phi3 to ModelArchitecture union"
```

### Task 3: Register `phi-3.5-mini-q4km` and `phi-3.5-mini-warm`

**Files:**
- Modify: `eval/models.ts:267-283` (existing `phi-3.5-mini-q4f16` block)
- Modify: `eval/smoke-profiles.ts` (add warm profile after the qwen3-14b entry)

- [ ] **Step 1: Read the existing phi-3.5-mini block**

Read `eval/models.ts:265-285` to see the current registration.

- [ ] **Step 2: Replace the existing entry with the validated `q4km` registration**

Edit `eval/models.ts` to replace the existing `phi-3.5-mini-q4f16` block with:

```typescript
	// Phi-3.5-mini-instruct — 3.82B, fused QKV + fused gate-up FFN.
	// Re-registered 2026-04-29 (Path B: fused-forward, Phi-3-gated).
	// 32 layers, hidden 3072, 32 heads (no GQA), intermediate 8192,
	// vocab 32064. MIT license. Sliding window present in HF config
	// but at sliding_window=262144 (effectively no SWA at our ctx=4096).
	// Architecture string in GGUF is "phi3" (per llama.cpp arch table);
	// the older "phi" entry is preserved for Phi-1 / Phi-2 GGUFs.
	{
		id: "phi-3.5-mini-q4km",
		name: "Phi-3.5 Mini Instruct (Q4_K_M, fused-forward)",
		family: "Phi",
		architecture: "phi3",
		paramsB: 3.82,
		vramMB: 2520,
		defaultQuant: "q4f16_1",
		availableQuants: ["q4f16_1"],
		capabilities: { toolCalling: false, structuredOutput: false, vision: false, embedding: false },
		license: "MIT",
		contextLength: 4096,
		tier: "balanced",
		requiresShaderF16: false,
		downloadUrl: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
		ggufUrl: "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF",
		ggufFilePattern: "Q4_K_M",
	},
```

(Note: changes `architecture` from `"phi"` to `"phi3"`, `id` from `phi-3.5-mini-q4f16` to `phi-3.5-mini-q4km`, `ggufFilePattern` to `"Q4_K_M"` matching the bartowski file; updated `name` + comment.)

- [ ] **Step 3: Add a smoke-profile warm entry**

Edit `eval/smoke-profiles.ts` to add (after the `qwen3-14b-q4ks-warm` block from the prior 13B work):

```typescript
	// ── Phi-3.5 Mini Instruct (2026-04-29 fused-forward) ──
	{
		name: "phi-3.5-mini-warm",
		model: "phi-3.5-mini-q4km",
		temperature: 0.6,
		prompt: DEFAULT_PROMPT,
	},
```

- [ ] **Step 4: Run checkall**

Run: `make checkall`
Expected: PASS — fmt + lint + typecheck + 452 tests, no skips count change.

- [ ] **Step 5: Commit**

```bash
git add eval/models.ts eval/smoke-profiles.ts
git commit -m "feat(eval): re-register phi-3.5-mini-q4km with phi3 architecture

Pre-existing phi-3.5-mini-q4f16 entry was registered with arch='phi'
(stale, Phi-1/Phi-2 dialect) and a non-bartowski downloadUrl; this
re-registers under arch='phi3' matching the llama.cpp GGUF emit and
the bartowski mirror's actual filename pattern. Smoke profile
phi-3.5-mini-warm added.

Loader/forward support follows in the next commit; this entry
will fail the smoke probe until then (architecture-gated path is
not yet wired)."
```

### Task 4: Download the GGUF locally

**Files:**
- Create: `smoke-test/models/phi-3.5-mini-q4km.gguf` (gitignored binary; ~2.39 GB)

- [ ] **Step 1: Pre-flight disk-space check**

Run: `df -h /Users/probello/Repos/webllm/smoke-test/models/`
Expected: `Avail` ≥ 5 GB. If not, ABORT and surface to user (per the 13B work, APFS may be holding TM snapshots — `tmutil listlocalsnapshots /` to inspect).

- [ ] **Step 2: Download the GGUF**

Run:

```bash
cd /Users/probello/Repos/webllm/smoke-test/models && \
  curl -L --fail \
    -o phi-3.5-mini-q4km.gguf \
    https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

Expected: completes with exit 0, filesize ~2,393,232,608 bytes (verify with `ls -la`). At ~30-40 MB/s this is ~60-80s.

- [ ] **Step 3: No commit — these binaries are gitignored**

Confirm via `git status` that the new file is ignored.

---

## Phase 2 — Loader extension (fused tensor support)

### Task 5: Extend `LayerWeights` with optional fused fields

**Files:**
- Modify: `src/inference/model-inference.ts:12-31`

- [ ] **Step 1: Read current LayerWeights interface**

Read `src/inference/model-inference.ts:12-31`.

- [ ] **Step 2: Add the fused fields and gate the existing fields**

Replace the `LayerWeights` interface in `src/inference/model-inference.ts` with:

```typescript
interface LayerWeights {
	attnNorm: TensorPtr;
	// Optional Phi-3 norm bias. RMSNorm + bias add. Null on llama / qwen /
	// mistral GGUFs (which lack norm biases).
	attnNormBias: TensorPtr | null;
	// Phi-3 fused QKV: single [3*n_embd, n_embd] matrix. When non-null,
	// qProj/kProj/vProj are null and the forward graph takes the fused
	// matmul + view-slice path (mirrors encoder-inference.ts:263-296).
	qkvFused: TensorPtr | null;
	qProj: TensorPtr | null;
	kProj: TensorPtr | null;
	vProj: TensorPtr | null;
	// Qwen2 / Qwen2.5 use biased Q/K/V projections. Llama, Qwen3, Mistral,
	// and Phi-3 (whose biases are baked into qkvFused) don't — these stay
	// null and the forward graph skips the add. Without this, qwen2 GGUFs
	// produce garbage (random-token) output because Q/K/V are off by the
	// bias shift.
	qBias: TensorPtr | null;
	kBias: TensorPtr | null;
	vBias: TensorPtr | null;
	qNorm: TensorPtr | null;
	kNorm: TensorPtr | null;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	ffnNormBias: TensorPtr | null;
	// Phi-3 fused gate-up: single [2*n_ff, n_embd] matrix. When non-null,
	// gateProj/upProj are null and the forward graph splits the matmul
	// output into gate/up halves before SwiGLU.
	gateUpFused: TensorPtr | null;
	gateProj: TensorPtr | null;
	upProj: TensorPtr | null;
	downProj: TensorPtr;
}
```

(Adds 4 new fields: `attnNormBias`, `qkvFused`, `ffnNormBias`, `gateUpFused`. Loosens `qProj`/`kProj`/`vProj`/`gateProj`/`upProj` to nullable.)

- [ ] **Step 3: Add `outputBias` to `WeightTensors`**

Replace the `WeightTensors` interface (currently at `src/inference/model-inference.ts:33-38`) with:

```typescript
interface WeightTensors {
	tokEmb: TensorPtr;
	norm: TensorPtr;
	// Optional Phi-3 final-norm bias.
	normBias: TensorPtr | null;
	output: TensorPtr | null;
	// Phi-3 lm_head bias. Null for all other architectures.
	outputBias: TensorPtr | null;
	layers: LayerWeights[];
}
```

- [ ] **Step 4: Typecheck — confirm callsites surface compile errors that we'll fix in Task 6**

Run: `make typecheck`
Expected: errors at the loader (`makeTensor` calls returning `TensorPtr` assigned to nullable fields are fine; the issue surfaces at the forward path where `lw.qProj` is now nullable and used as non-null at line 471 et seq). **Do not fix these yet** — Task 7 is where the forward branch lands. **DO** fix any errors that surface in unrelated files (lightweight.ts, generation.ts, etc.) — those should be safe widenings only.

- [ ] **Step 5: No commit yet — Task 6 brings the loader up to compile**

### Task 6: Branch the loader on `architecture === "phi3"`

**Files:**
- Modify: `src/inference/model-inference.ts:213-251` (current loader body)

- [ ] **Step 1: Read the current loader body**

Read `src/inference/model-inference.ts:210-260`.

- [ ] **Step 2: Replace the loader body with the architecture-branched version**

Find the existing block:

```typescript
		const output = tensorMap.has("output.weight")
			? this.makeTensor(tensorMap, "output.weight")
			: null;

		const layers: LayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			layers.push({
				attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
				qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
				kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
				vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
				qBias: tensorMap.has(p("attn_q.bias"))
					? this.makeTensor(tensorMap, p("attn_q.bias"))
					: null,
				kBias: tensorMap.has(p("attn_k.bias"))
					? this.makeTensor(tensorMap, p("attn_k.bias"))
					: null,
				vBias: tensorMap.has(p("attn_v.bias"))
					? this.makeTensor(tensorMap, p("attn_v.bias"))
					: null,
				qNorm: tensorMap.has(p("attn_q_norm.weight"))
					? this.makeTensor(tensorMap, p("attn_q_norm.weight"))
					: null,
				kNorm: tensorMap.has(p("attn_k_norm.weight"))
					? this.makeTensor(tensorMap, p("attn_k_norm.weight"))
					: null,
				oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
				ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
				gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
				upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
				downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
			});
		}

		this.weights = { tokEmb, norm, output, layers };
```

Replace with:

```typescript
		const output = tensorMap.has("output.weight")
			? this.makeTensor(tensorMap, "output.weight")
			: null;
		const outputBias = tensorMap.has("output.bias")
			? this.makeTensor(tensorMap, "output.bias")
			: null;
		const normBias = tensorMap.has("output_norm.bias")
			? this.makeTensor(tensorMap, "output_norm.bias")
			: null;

		const isPhi3 = hp.architecture === "phi3";
		const layers: LayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			const has = (s: string) => tensorMap.has(p(s));
			const opt = (s: string) =>
				has(s) ? this.makeTensor(tensorMap, p(s)) : null;
			if (isPhi3) {
				// Phi-3: fused QKV + fused gate-up. The forward graph slices
				// the fused outputs via opView3d / opView2d. Per-layer norms
				// may carry an optional bias (RMSNorm + bias).
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: opt("attn_norm.bias"),
					qkvFused: this.makeTensor(tensorMap, p("attn_qkv.weight")),
					qProj: null,
					kProj: null,
					vProj: null,
					qBias: null,
					kBias: null,
					vBias: null,
					qNorm: null,
					kNorm: null,
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: opt("ffn_norm.bias"),
					gateUpFused: this.makeTensor(tensorMap, p("ffn_up.weight")),
					gateProj: null,
					upProj: null,
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
				});
			} else {
				// Default split-QKV / split-gate-up path used by llama / qwen* /
				// mistral / etc. Nothing here moves; only nulls were added for
				// phi3-only fields.
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: null,
					qkvFused: null,
					qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
					kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
					vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
					qBias: opt("attn_q.bias"),
					kBias: opt("attn_k.bias"),
					vBias: opt("attn_v.bias"),
					qNorm: opt("attn_q_norm.weight"),
					kNorm: opt("attn_k_norm.weight"),
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: null,
					gateUpFused: null,
					gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
					upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
				});
			}
		}

		this.weights = { tokEmb, norm, normBias, output, outputBias, layers };
```

- [ ] **Step 3: Run typecheck — only forward-graph callsites should still fail**

Run: `make typecheck`
Expected: errors localized to `forwardSingle()` (lines ~471, ~640, ~653) where nullable `lw.qProj` / `lw.gateProj` / `lw.upProj` are unwrapped without checks. These are fixed in Task 7. No errors anywhere else.

- [ ] **Step 4: Commit (compile-broken commit is OK — Phase 2 lands as one logical unit)**

Hold the commit until Task 7 lands the forward branch (logical unit per project commit cadence: `feat(inference): phi3 fused-QKV + fused-gate-up support`).

### Task 7: Branch the forward graph on fused presence

**Files:**
- Modify: `src/inference/model-inference.ts` (the QKV block ~lines 471-486 and the FFN block ~lines 640-644)

- [ ] **Step 1: Read the current QKV block**

Read `src/inference/model-inference.ts:460-490`.

- [ ] **Step 2: Add the F32_BYTES import**

At the top of `src/inference/model-inference.ts`, change:

```typescript
import {
	type BufferPtr,
	GgmlType,
	type GgmlWasm,
	type GraphComputeProfile,
	RopeMode,
	type TensorPtr,
} from "./ggml-wasm.js";
```

to:

```typescript
import {
	type BufferPtr,
	F32_BYTES,
	GgmlType,
	type GgmlWasm,
	type GraphComputeProfile,
	RopeMode,
	type TensorPtr,
} from "./ggml-wasm.js";
```

- [ ] **Step 3: Replace the QKV split block with the fused-or-split branch**

Find this block in `forwardSingle`:

```typescript
			// LLaMA RMSNorm: (x / rms(x)) * gamma. ggml_rms_norm only does the
			// normalize step — the per-dim gain `attn_norm.weight` must be applied
			// separately. Same for `ffn_norm.weight` and the final `output_norm.weight`.
			const normed = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				lw.attnNorm,
			);

			const qRaw = wasm.opMulMat(lw.qProj, normed);
			const kRaw = wasm.opMulMat(lw.kProj, normed);
			const vRaw = wasm.opMulMat(lw.vProj, normed);
			const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
			const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
			const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;

			const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			const k3 = wasm.opReshape3d(k, headDim, hp.headCountKv, nTokens);
			const v3 = wasm.opReshape3d(v, headDim, hp.headCountKv, nTokens);
			const qReady = lw.qNorm
				? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
				: q3;
			const kReady = lw.kNorm
				? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
				: k3;
```

Replace with:

```typescript
			// LLaMA RMSNorm: (x / rms(x)) * gamma. ggml_rms_norm only does the
			// normalize step — the per-dim gain `attn_norm.weight` must be applied
			// separately. Same for `ffn_norm.weight` and the final `output_norm.weight`.
			let normed = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				lw.attnNorm,
			);
			if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

			let q3: TensorPtr;
			let k3: TensorPtr;
			let v3: TensorPtr;

			if (lw.qkvFused) {
				// Phi-3 fused QKV: one matmul → 3 view3d slices.
				// Mirrors src/inference/encoder-inference.ts:263-296 and
				// llama.cpp/src/llama-graph.cpp:1088-1095 (build_qkv).
				// Phi-3.5-mini has no GQA (n_kv_head == n_head), so all
				// three slices are full-width [E, nTokens]; for any future
				// Phi-3 variant with GQA, the K and V slices would shrink
				// proportionally (kvDim < E).
				const E = hp.embeddingLength;
				const kvDim = headDim * hp.headCountKv;
				const fusedRowDim = E + 2 * kvDim;
				const qkv = wasm.opMulMat(lw.qkvFused, normed); // [fusedRowDim, nTokens]
				const elemSize = F32_BYTES;
				const headBytes = elemSize * headDim;
				const tokenBytes = elemSize * fusedRowDim;
				q3 = wasm.opView3d(
					qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, 0,
				);
				k3 = wasm.opView3d(
					qkv, headDim, hp.headCountKv, nTokens,
					headBytes, tokenBytes, elemSize * E,
				);
				v3 = wasm.opView3d(
					qkv, headDim, hp.headCountKv, nTokens,
					headBytes, tokenBytes, elemSize * (E + kvDim),
				);
			} else {
				if (!lw.qProj || !lw.kProj || !lw.vProj) {
					throw new Error(
						`split-QKV path requires qProj/kProj/vProj for ${hp.architecture}`,
					);
				}
				const qRaw = wasm.opMulMat(lw.qProj, normed);
				const kRaw = wasm.opMulMat(lw.kProj, normed);
				const vRaw = wasm.opMulMat(lw.vProj, normed);
				const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
				const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
				const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;
				q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
				k3 = wasm.opReshape3d(k, headDim, hp.headCountKv, nTokens);
				v3 = wasm.opReshape3d(v, headDim, hp.headCountKv, nTokens);
			}

			const qReady = lw.qNorm
				? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
				: q3;
			const kReady = lw.kNorm
				? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
				: k3;
```

- [ ] **Step 4: Read the current FFN block**

Read `src/inference/model-inference.ts:633-650`.

- [ ] **Step 5: Replace the FFN gate-up block with the fused-or-split branch**

Find:

```typescript
			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			const ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			const gate = wasm.opMulMat(lw.gateProj, ffnNormed);
			const up = wasm.opMulMat(lw.upProj, ffnNormed);
			// Fused silu(gate) * up — single GPU op instead of silu+mul.
			const ffnHidden = wasm.opSwigluSplit(gate, up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
```

Replace with:

```typescript
			const oProj = wasm.opMulMat(lw.oProj, merged);
			const attnResidual = wasm.opAdd(oProj, cur);

			let ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);

			let gate: TensorPtr;
			let up: TensorPtr;
			if (lw.gateUpFused) {
				// Phi-3 fused gate-up: one matmul → 2 view2d slices.
				// llama.cpp's LLM_FFN_SWIGLU mode with gate=NULL splits the
				// fused [2*ffSize, nTokens] output into halves before
				// SwiGLU; see llama-build.cpp build_ffn LLM_FFN_SWIGLU.
				const ffSize = hp.feedForwardLength;
				const fused = wasm.opMulMat(lw.gateUpFused, ffnNormed); // [2*ffSize, nTokens]
				const elemSize = F32_BYTES;
				const tokenBytes = elemSize * 2 * ffSize;
				gate = wasm.opView2d(fused, ffSize, nTokens, tokenBytes, 0);
				up = wasm.opView2d(
					fused, ffSize, nTokens, tokenBytes, elemSize * ffSize,
				);
			} else {
				if (!lw.gateProj || !lw.upProj) {
					throw new Error(
						`split-gate-up path requires gateProj/upProj for ${hp.architecture}`,
					);
				}
				gate = wasm.opMulMat(lw.gateProj, ffnNormed);
				up = wasm.opMulMat(lw.upProj, ffnNormed);
			}
			// Fused silu(gate) * up — single GPU op instead of silu+mul.
			const ffnHidden = wasm.opSwigluSplit(gate, up);
			const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

			cur = wasm.opAdd(ffnOut, attnResidual);
```

- [ ] **Step 6: Read the final-norm + output block**

Read `src/inference/model-inference.ts:649-657`.

- [ ] **Step 7: Apply final-norm bias and lm_head bias**

Find:

```typescript
		const finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		const logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
```

Replace with:

```typescript
		let finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		if (weights.normBias) finalNorm = wasm.opAdd(finalNorm, weights.normBias);
		let logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
		if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);
```

- [ ] **Step 8: Confirm `opView2d` exists**

Run: `grep -n "opView2d\|opView3d" /Users/probello/Repos/webllm/src/inference/ggml-wasm.ts | head`
Expected: both exist as `wasm.opView2d` and `wasm.opView3d` exports. If `opView2d` is missing, the FFN slice falls back to `opView3d` with one of the dims set to 1 (this is a contingency, document if it fires).

- [ ] **Step 9: Run typecheck**

Run: `make typecheck`
Expected: PASS — no remaining errors.

- [ ] **Step 10: Run all tests (no Phi-3 specific tests yet — Task 8 adds those)**

Run: `bun test`
Expected: 452 pass / 11 skip / 0 fail (same as baseline). Any new failure indicates we broke a non-Phi model — ABORT and diagnose.

### Task 8: Unit-test the fused-view offsets

**Files:**
- Create: `tests/phi3-fused-loader.test.ts`

The point of this test is to lock in the offset/stride math for the fused QKV and fused gate-up slices, since a single off-by-element here is the most likely failure mode and is invisible until the model produces gibberish.

- [ ] **Step 1: Write the failing test**

Create `tests/phi3-fused-loader.test.ts`:

```typescript
import { describe, expect, test } from "bun:test";

// These reproduce the exact offset/stride math from
// src/inference/model-inference.ts:forwardSingle's qkvFused branch.
// Phi-3.5-mini values: hidden 3072, headDim 96, nHeads 32, n_kv 32 (no GQA).
//
// The fused matrix is [3*E, E] (when no GQA) or [E + 2*kvDim, E] (with GQA),
// stored row-major. After matmul with x=[E, nTokens] the output is
// [fusedRowDim, nTokens]. Q occupies rows [0, E), K rows [E, E+kvDim),
// V rows [E+kvDim, E+2*kvDim).

describe("phi3 fused QKV view-offset math", () => {
	const F32_BYTES = 4;
	const cases = [
		{ name: "phi-3.5-mini (no GQA)", E: 3072, headDim: 96, nHeads: 32, nKvHeads: 32 },
		{ name: "hypothetical phi3 with GQA 4:1", E: 3072, headDim: 96, nHeads: 32, nKvHeads: 8 },
	];
	for (const c of cases) {
		test(c.name, () => {
			const kvDim = c.headDim * c.nKvHeads;
			const fusedRowDim = c.E + 2 * kvDim;
			const tokenBytes = F32_BYTES * fusedRowDim;
			const qOffset = 0;
			const kOffset = F32_BYTES * c.E;
			const vOffset = F32_BYTES * (c.E + kvDim);
			// Q occupies the first E rows.
			expect(qOffset).toBe(0);
			// K starts immediately after Q.
			expect(kOffset).toBe(F32_BYTES * c.E);
			// V starts immediately after K.
			expect(vOffset).toBe(F32_BYTES * c.E + F32_BYTES * kvDim);
			// All three slices share tokenBytes = F32 * (E + 2*kvDim).
			expect(tokenBytes).toBe(F32_BYTES * (c.E + 2 * kvDim));
			// Verify Q dimensions: [headDim, nHeads, nTokens] over the first E rows.
			expect(c.headDim * c.nHeads).toBe(c.E);
			// Verify K/V dimensions: [headDim, nKvHeads, nTokens].
			expect(c.headDim * c.nKvHeads).toBe(kvDim);
		});
	}
});

describe("phi3 fused gate-up view-offset math", () => {
	const F32_BYTES = 4;
	// Phi-3.5-mini values: ffSize 8192.
	const ffSize = 8192;
	test("gate-up halves are contiguous and equal-size", () => {
		const tokenBytes = F32_BYTES * 2 * ffSize;
		const gateOffset = 0;
		const upOffset = F32_BYTES * ffSize;
		expect(gateOffset).toBe(0);
		expect(upOffset).toBe(F32_BYTES * ffSize);
		expect(tokenBytes).toBe(2 * upOffset);
	});
});
```

- [ ] **Step 2: Run test — confirm it passes**

Run: `bun test tests/phi3-fused-loader.test.ts -v`
Expected: 3 pass / 0 fail. (No code change required; this is locking in invariants.)

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `make checkall`
Expected: 455 pass / 11 skip / 0 fail (the 3 new tests landed on top of the prior 452).

- [ ] **Step 4: Commit Phase 2 (Tasks 5+6+7+8 as one logical unit)**

```bash
git add src/inference/model-inference.ts tests/phi3-fused-loader.test.ts
git commit -m "feat(inference): phi3 fused-QKV + fused-gate-up support

Adds architecture-gated fused-forward path for Phi-3 family GGUFs:
- Loader: when hp.architecture === 'phi3', populate qkvFused +
  gateUpFused instead of split q/k/v/gate/up tensors. Optional
  attn_norm/ffn_norm/output_norm biases and lm_head output_bias
  loaded conditionally.
- Forward: when lw.qkvFused is non-null, do one matmul + 3
  opView3d slices into Q/K/V (mirrors encoder-inference.ts
  nomic path). Same for fused gate-up: matmul + 2 opView2d
  slices before SwiGLU.

Path B per the 2026-04-29 perf tradeoff analysis: ~96 fewer
dispatches/token at 32-layer Phi-3.5-mini scale, ~2% steady-
state vs the loader-only Path A alternative. The other 11
shipping causal models hit the unchanged split path.

tests/phi3-fused-loader.test.ts locks the view-offset math
for both fused branches, including a hypothetical-GQA case
to keep future Phi-3 GQA variants on the same path."
```

---

## Phase 3 — Smoke probe

### Task 9: Browser smoke probe

**Files:**
- (browser-side only — no file edits)

- [ ] **Step 1: Confirm WASM build is current**

Run: `make wasm-build`
Expected: builds wasm32 (`webllm-wasm.{js,wasm}`) and wasm64 (`webllm-wasm-mem64.{js,wasm}`). The Phi-3 model fits well under 4 GiB so wasm32 is sufficient.

- [ ] **Step 2: Confirm smoke server is running**

Run: `curl -sf http://localhost:8031/ -o /dev/null -w "%{http_code}\n"`
Expected: `200`. If not, run `make smoke-serve &` and re-check.

- [ ] **Step 3: List Chrome tabs to find the smoke-test tab**

Run: `agentchrome --port 62847 tabs list 2>&1 | head`
Expected: an existing `WebLLM Real Model Test` tab. Capture its ID.

- [ ] **Step 4: Navigate to the Phi-3 smoke URL**

Run (substitute the captured tab ID):

```bash
agentchrome --port 62847 navigate --tab <TAB_ID> \
  "http://localhost:8031/real-model.html?model=phi-3.5-mini-q4km&ctx=4096&prompt=hi&ingest=off&v=$(date +%s)"
```

(Note: `wasm=mem64` is NOT specified — Phi-3.5-mini is small enough for the wasm32 path.)

- [ ] **Step 5: Poll the page log until terminal state**

Run:

```bash
until agentchrome --port 62847 page snapshot --tab <TAB_ID> 2>/dev/null \
    | grep -qE "\[8/8\]|Fetch failed|stage failed|Generation failed|Error:|exception"; do
  sleep 5
done
```

Use `run_in_background: true` so the runtime auto-notifies on completion. Expected: completion within ~60-90 seconds (no GGUF download — file is already local).

- [ ] **Step 6: Verify the 8 stages all green**

After the wait, run:

```bash
agentchrome --port 62847 page snapshot --tab <TAB_ID> 2>&1 | tail -200
```

Expected:
- `[3/8] GGUF parsed: arch=phi3 emb=3072 heads=32/32 layers=32 vocab=32064 ctx=131072` (or similar — confirms the GGUF advertises `phi3` and our loader accepted it)
- `[4/8] Weights loaded` succeeds (no `attn_q.weight not found` errors)
- `[7/8] Generated N tokens in ... finish=eos` (decode > 0 tok/s, terminated cleanly)
- `[8/8] embed cosine ... ≥0.75 ... ‖v‖=1.00` (embed sanity passes)

- [ ] **Step 7: Verify zero console errors / warnings**

Run:

```bash
agentchrome --port 62847 console read --tab <TAB_ID> --errors-only --limit 30 2>&1
agentchrome --port 62847 console read --tab <TAB_ID> --type warn --limit 30 2>&1
```

Expected: `[]` for both. `adapter_info:` lines are benign per the project workflow doc.

- [ ] **Step 8: Phase 3 gate**

PASS if: all 8 stages green AND decode > 0 tok/s AND embed cosine ≥ 0.75 AND console clean. Any FAIL → diagnose by reading the page snapshot's stage-N error string. The most likely failure modes:
  1. `attn_qkv.weight not found` — the GGUF doesn't actually have fused QKV (probe was wrong) → fall back to Path A
  2. Gibberish output — view-offset math wrong (Task 8 invariants didn't catch it). Diagnose by lowering temp to 0 and comparing layer-0 hidden state against a CPU reference (out of scope for this plan; surface to user).
  3. `Q/K/V dimensions don't match attn_output` — the K and V row counts disagree with `n_kv` × `headDim` (GQA mismatch). Re-check `hp.headCountKv` reads correctly from GGUF.

No commit — Phase 3 is pass/fail validation, not a code-producing phase.

---

## Phase 4 — 36-prompt sanity eval

### Task 10: Browser eval

**Files:**
- (no file edits)

- [ ] **Step 1: Confirm dashboard is up**

Run: `curl -sf http://localhost:8033/health -o /dev/null -w "%{http_code}\n"`
Expected: `200`. If not, run `make dashboard-serve &` and re-check.

- [ ] **Step 2: Run the eval**

Run (foreground, 5-15 min wall-time at expected ~35-50 tok/s × 36 prompts × ~50-200 tokens each):

```bash
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \
  make bench-browser-eval PROFILE=phi-3.5-mini-warm
```

Expected output: `Done: N/36 passing · overall X%`.

- [ ] **Step 3: Phase 4 gate**

PASS if: overall ≥ 60%. Reference points (same suite):
- mistral-7b-instruct-v0.3-q4ks: 24/36 (68%)
- qwen3-1.7b-q4f16 thinking-on: 89%
- qwen3-4b-q4f16: 88-90%
- qwen3-14b-q4ks: 34/36 (94%)

Phi-3.5-mini external benchmarks (Open LLM Leaderboard, MMLU, etc.) place it in the 65-75% range vs Qwen2.5-3B (which lands at 86% on our suite). **Predicted band: 70-80%.** Hard floor: ≥60%.

If the eval falls below floor: diagnose via the dashboard's per-dimension breakdown (`/evals` API). The most informative failure pattern would be a flat-low score across all dimensions (suggesting numerical bug) vs a single-dimension cliff (suggesting tokenizer or chat-template mismatch — `chat-template.ts:316` has the `phi3` formatter; verify it's the right one).

No commit — Phase 4 is validation.

---

## Phase 5 — Smoke-bench tok/s

### Task 11: 3-run profile-mode smoke-bench

**Files:**
- (no file edits)

- [ ] **Step 1: Run smoke-bench**

Run (foreground, ~3-5 min):

```bash
make smoke-bench PERF_MODEL=phi-3.5-mini-q4km PERF_RUNS=3
```

(Note: NO `WASM_VARIANT=mem64` — Phi-3.5-mini fits under 4 GiB so the wasm32 path is used.)

- [ ] **Step 2: Phase 5 gate**

PASS if: 3-run median ≥ 25 tok/s (hard floor). Predicted band 35-50 tok/s based on:
- llama-3.2-3b 58 tok/s (3.21B, GQA 3:1, simpler arch)
- qwen2.5-3b 45 tok/s (3.1B, GQA 8:1)
- Phi-3.5-mini 3.82B + no GQA → expect somewhat slower than qwen2.5-3b due to larger KV reads (no GQA = full K+V dim per layer).

Capture the 3-run output verbatim for the closure report.

If median falls below floor: the fused-forward path may be paying a kernel-shape penalty we didn't predict (matmul shape [3*E, E] × [E, 1] hits a different shader path than three [E, E] × [E, 1] dispatches). Surface to user; the fallback is Path A (loader-only views, no forward changes).

No commit — Phase 5 is measurement.

---

## Phase 6 — Closure

### Task 12: Closure report

**Files:**
- Create: `eval/reports/phi-3-validation-2026-04-29/SUMMARY.md` (gitignored — force-add per project convention)

- [ ] **Step 1: Write the closure report**

Create `eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`. Mirror the structure of `eval/reports/13b-validation-2026-04-29/SUMMARY.md` (which we shipped today as the canonical template). Required sections:

- **Headline** — bullet-list of the gates with pass/fail status and headline numbers
- **Discovery arc** — Phase 1 through Phase 6 in order
- **Eval result** — the X/36 = Y% headline + the comparison table to existing fleet
- **Speed result** — the 3-run smoke-bench table + per-phase decode mean times + backend attribution
- **Working set** — disk size, KV cache size at ctx=4096, total at decode
- **Lever closure** — what's in scope, what's out of scope (no canonical-6 integration this round)
- **Reproduction** — exact commands

Pay particular attention to:
- Capturing the **dispatch-count delta** vs split-QKV (predicted ~96 dispatches/token saved). If the smoke-bench profile-mode run shows `backendDispatchCount`, write down the exact number and compare to the prediction.
- Calling out the architecture-gated reversibility: zero impact on the other 11 causal models (verified in Phase 2's `make checkall` step).

- [ ] **Step 2: Commit closure report + close TODO entries**

```bash
git add -f eval/reports/phi-3-validation-2026-04-29/SUMMARY.md
# Update TODO.md "Deferred" section to mark Phi-3 as CLOSED, then:
git add TODO.md
git commit -m "docs(report): close Phi-3 causal LM support with phi-3.5-mini validation

Path B (fused-forward, Phi-3-gated) shipped end-to-end:
- Phase 4 sanity eval: <X>/36 = <Y>%
- Phase 5 smoke-bench: median <Z> tok/s
- Dispatch count: <N> (predicted 96 fewer than split path)
- Console clean; canonical-6 unaffected (make checkall green
  through Phase 2 commit)."
```

### Task 13: TODO archival

**Files:**
- Modify: `TODO.md` — remove Phi-3 from "Deferred" section
- Modify: `TODO_ARCHIVE.md` — append closure entry

- [ ] **Step 1: Read TODO.md "Deferred" section**

Read `TODO.md` around line 1142 — `### Deferred (out of scope per current ceilings)`.

- [ ] **Step 2: Update TODO.md**

In `TODO.md`, under the `Deferred` section, replace the Wave-1 architectures bullet:

```markdown
- **Wave-1 architectures Gemma 2, Phi 3.** 5+ gaps for Gemma; mostly
  fused-QKV for Phi 3. Re-evaluate if a model in either family lands
  as a real deployment ask.
```

with:

```markdown
- **Wave-1 architecture Gemma 2.** 5+ gaps for Gemma (pre+post norm
  pairs, logit/attn soft-cap, sliding-window, (1+w) RMSNorm).
  Re-evaluate if a Gemma 2 model lands as a real deployment ask.
- ~~**Phi 3**~~ **CLOSED 2026-04-29.** Phi-3.5-mini-instruct-q4km
  shipped via Path B (fused-forward, phi3-gated).
  [`eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`](eval/reports/phi-3-validation-2026-04-29/SUMMARY.md).
```

- [ ] **Step 3: Append the full block to TODO_ARCHIVE.md**

Append (mirroring the 13B archival):

```markdown
---

## Phi-3 causal LM support (closed 2026-04-29; archived from TODO.md)

Closed 2026-04-29 — Phi-3.5-mini-instruct-q4km validated end-to-end
on the wasm32 path with the new architecture-gated fused-QKV +
fused-gate-up forward graph (Path B). Eval <X>/36 = <Y>%, 3-run
smoke-bench median <Z> tok/s. Closure report at
[`eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`](eval/reports/phi-3-validation-2026-04-29/SUMMARY.md).

Implementation plan at
[`docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md`](docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md).
```

- [ ] **Step 4: Commit archival**

```bash
git add TODO.md TODO_ARCHIVE.md
git commit -m "docs(TODO): archive Phi-3 causal LM support block

Closed in commit <PREVIOUS_SHA>. Per TODO archival cadence,
moves the deferral text out of TODO.md into TODO_ARCHIVE.md
and replaces the bullet with a closure stub.

Separate commit from the closure itself per the doctrine
'TODO archival is its own commit ... so a git revert of the
archive is reversible without touching the closures.'"
```

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GGUF reports `general.architecture = "phi"` instead of `"phi3"` | Low-Medium | Phase 1 Task 1 Step 2 inspects the metadata directly. If it's `"phi"`, update the eval/models.ts entry and the loader branch to match. |
| GGUF actually has split QKV (older repackage) | Low | Phase 1 Task 1 Step 2 inspects layer-0 tensor names. Fall back to Path A if so. |
| View-offset math wrong | Low-Medium | Task 8 unit tests lock the math; Phase 3 smoke probe catches gibberish; Phase 4 eval catches subtle correctness. |
| FFN gate-up split inverted (gate vs up swapped) | Medium | If Phase 3 produces gibberish, swap the offsets and re-run. The fix is one-line. |
| `opView2d` doesn't exist for matrix slicing | Low | Task 7 Step 8 verifies. Fallback: use `opView3d` with one dim = 1. |
| Sliding-window attention required (HF config has `sliding_window=262144`) | Very Low | At ctx=4096, 262144 > ctx so no sliding effectively kicks in. If a Phi-3 variant with smaller `sliding_window` ships, this assumption breaks — re-evaluate. |
| Eval gate fails (<60%) | Low | External Phi-3.5-mini benchmarks place it firmly above. Most likely failure mode = chat-template mismatch (chat-template.ts:316 must match the model's expected `<\|user\|>` / `<\|assistant\|>` tokens). |
| Smoke-bench below 25 tok/s | Low | Architectural extrapolation places it 35-50 tok/s. Below 25 would suggest a kernel-shape pessimization on the [3*E, E]-shaped matmul; surface and fall back to Path A. |
| Per-phase regression on the 11 currently-shipping causal models | Very Low | Architecture-gated. Verified in Task 7 Step 10 (full `bun test` post-Phase-2). Any regression there blocks the commit. |

---

## Out of scope for this plan

- Integrating Phi-3.5-mini into the canonical 6 parity sweep (separate decision; the canonical 6 are pinned at TinyLlama / Qwen3-0.6B / Qwen3-1.7B / Mistral-7B / Llama-3.1-8B / Qwen3-8B; Phi-3 sits in a different size band).
- Adding `phi3-thinking-warm` profile (Phi-3.5-mini doesn't have a thinking mode; Qwen3 family is the only one with that switch).
- Phi-3-medium (14B) or Phi-3-MoE registration (separate work item if appetite arises).
- Path B for any other architecture (gemma, deepseek, etc.) — each is its own architecture-gated branch.
- Refactoring `LayerWeights` to be a discriminated union of "split" vs "fused" variants. The current additive-nullable approach is sufficient and keeps the diff small.

---

## Self-Review

**Spec coverage:** every spec requirement maps to a task — fused-QKV (T6/T7), fused-gate-up (T6/T7), outputBias (T6/T7), `phi3` arch flag (T2), eval/models.ts entry (T3), smoke-profiles (T3), smoke probe (T9), 36-prompt eval (T10), smoke-bench (T11), closure report (T12), TODO archival (T13). ✅

**Placeholder scan:** `<X>` / `<Y>` / `<Z>` / `<N>` / `<TAB_ID>` / `<PREVIOUS_SHA>` are intentional substitution markers in commit-message and report templates, not "TBD" placeholders. Every code step has full code; every command step has the exact command. ✅

**Type consistency:** `qkvFused`, `gateUpFused`, `attnNormBias`, `ffnNormBias`, `outputBias`, `normBias` all defined in T5 and used consistently in T6/T7. `LayerWeights` and `WeightTensors` interface names match the file's existing conventions. The `opView3d`/`opView2d` calls match the encoder reference at `encoder-inference.ts:267-296`. ✅

---

## Execution

Per the global preference (CLAUDE.md): execute via `superpowers:subagent-driven-development` in this session immediately after review.
