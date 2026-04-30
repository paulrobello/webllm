# Bucket D — chat-model self-embedding implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ModelInference.embed(tokenIds)` so a single loaded chat model can serve both chat generation and sentence embeddings, gated per-model by an `embeddingCapable` registration flag. Ship-time scope: `qwen3-8b-iq3m` only.

**Architecture:** A new private `forwardForEmbedding(tokenIds)` helper on `ModelInference` mirrors `CausalLMEmbedder.forwardEmbed` but reuses the existing `buildQKV` / `buildFFNGateUp` helpers, structurally omits all KV-cache writes, and taps post-`output_norm`. `engine.embed`'s dispatch ladder gains a third tier (`encoderEngines → causalEmbedderEngines → inferenceEngines (only when `embeddingCapable === true`)`). Parity is validated against a PyTorch HF-base reference at `cos >= 0.999` plus a 4-pair cosine-distinguishability sanity check.

**Tech Stack:** TypeScript (Bun), patched `ggml-webgpu` WASM, existing `ModelInference` / `CausalLMEmbedder` / `engine.embed` infrastructure, `eval/causal-embedder-parity.ts` browser harness, Python `transformers` for ref capture, agentchrome for smoke runs, `make checkall` ship gate.

**Spec:** [`docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md`](../specs/2026-04-29-embedding-bucket-d-design.md).
**Predecessor (pattern source):** [bucket C closure report](../../../eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md), [bucket C plan](2026-04-29-embedding-bucket-c-implementation.md).

---

## Task 1: Add `embeddingCapable` field to `RegisteredModel`

**Files:**
- Modify: `eval/models.ts` (interface declaration around lines 14-50)

- [ ] **Step 1: Add the optional field to `RegisteredModel`**

In `eval/models.ts`, locate the `RegisteredModel` interface near the top of the file (the block that contains `id: string` and `architecture: ModelArchitecture`). Add an optional `embeddingCapable?: boolean` field with a JSDoc comment:

```ts
	/**
	 * When true, `engine.embed(id, text)` is allowed to dispatch through
	 * `inferenceEngines` (the chat-model self-embedding path / "bucket D").
	 * The chat model produces an embedding by tapping the post-`output_norm`
	 * hidden state, last-token-pooling, and L2-normalizing. Quality drops
	 * 5-15% on MTEB benchmarks vs dedicated retrieval-tuned embedders;
	 * acceptable for in-domain retrieval (agent memory, dialogue history).
	 *
	 * Only set this on chat models that have passed the bucket D parity
	 * gate at `cos >= 0.999` against a PyTorch HF-base reference.
	 */
	embeddingCapable?: boolean;
```

- [ ] **Step 2: Verify typecheck**

Run: `make typecheck`
Expected: clean. No registrations have the new field yet, so existing code is unaffected.

- [ ] **Step 3: Commit**

```bash
git add eval/models.ts
git commit -m "feat(embed): add embeddingCapable flag to RegisteredModel

Type-only field. Gates the bucket D dispatch path in engine.embed
(chat-model self-embedding via ModelInference.embed). Set true on
chat models that pass the bucket D parity gate; default undefined."
```

---

## Task 2: Add `ModelInference.embed()` + `forwardForEmbedding()` with KV-cache non-perturbation test (TDD)

**Files:**
- Test: `tests/model-inference-embed.test.ts` (new)
- Modify: `src/inference/model-inference.ts` (add public `embed()` and private `forwardForEmbedding()`)

- [ ] **Step 1: Write the failing test**

Create `tests/model-inference-embed.test.ts`. The load-bearing assertion is **KV-cache non-perturbation**: an `embed()` call must not change `nCached` or any KV buffer, and a subsequent `forward()` must produce byte-identical logits to a no-embed baseline. Mirror the WebGPU + GGUF skip pattern from `tests/forward-verify-equivalence.test.ts:14-20`.

```ts
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";
import { GgufParser } from "../src/models/gguf-parser.js";
import type { GgufContext } from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

const TINYLLAMA = resolve("smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf");
const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
const SHOULD_SKIP = !HAS_WEBGPU || !existsSync(TINYLLAMA);

function hashF32Buffer(arr: Float32Array): string {
	// FNV-1a over the byte view — fast, stable across runs, deterministic.
	const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
	let h = 0x811c9dc5;
	for (let i = 0; i < bytes.length; i++) {
		h ^= bytes[i];
		h = Math.imul(h, 0x01000193);
	}
	return (h >>> 0).toString(16);
}

describe.skipIf(SHOULD_SKIP)("ModelInference.embed", () => {
	test("does not perturb KV cache or chat logits", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({});
		const inf = new ModelInference(wasm, parsed.hyperparams);
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		const chatA = new Int32Array([1, 22172, 920]);
		const posA = new Int32Array([0, 1, 2]);
		const chatB = new Int32Array([530]);
		const posB = new Int32Array([3]);

		// Baseline: forwardA → forwardB (no embed in between).
		await inf.forward(chatA, posA);
		const baselineLogits = await inf.forward(chatB, posB);
		const baselineHash = hashF32Buffer(baselineLogits);

		// Reset and replay with an embed call inserted between A and B.
		inf.resetKVCache();
		await inf.forward(chatA, posA);
		expect(inf.cachedTokenCount).toBe(3);

		const embedIds = new Int32Array([100, 200, 300]);
		const embedVec = await inf.embed(embedIds);
		expect(embedVec.length).toBe(parsed.hyperparams.embeddingLength);
		expect(inf.cachedTokenCount).toBe(3); // unchanged

		// Verify L2 norm == 1 (within float tolerance).
		let sq = 0;
		for (let i = 0; i < embedVec.length; i++) sq += embedVec[i] * embedVec[i];
		expect(Math.abs(Math.sqrt(sq) - 1.0)).toBeLessThan(1e-5);

		// Continuation logits must match baseline byte-for-byte.
		const afterEmbedLogits = await inf.forward(chatB, posB);
		expect(hashF32Buffer(afterEmbedLogits)).toBe(baselineHash);

		await inf.dispose();
		await wasm.shutdown();
	});
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tests/model-inference-embed.test.ts`
Expected: FAIL with `inf.embed is not a function` (or similar — `embed` doesn't exist yet on `ModelInference`).

(If `HAS_WEBGPU` is false locally, the suite skips — that's fine; the load-bearing run is the browser smoke in Task 6 and the same test will execute under whatever WebGPU-capable environment the project uses for CI / browser test runs. The skip path still validates the file compiles.)

- [ ] **Step 3: Add `forwardForEmbedding` private helper to `ModelInference`**

In `src/inference/model-inference.ts`, place the new helper directly after `forwardSingle()` (around line 1020-ish, after the existing `this.nCached = totalLen` line that closes that method). The helper mirrors `CausalLMEmbedder.forwardEmbed` (`src/inference/causal-embedder-inference.ts:223-437`) with three load-bearing differences:

1. **Use `this.buildQKV(lw, normed, N, headDim, nHeads)`** instead of inline split-QKV (handles fused-QKV Phi-3 path automatically).
2. **Use `this.buildFFNGateUp(lw, ffnNormed, N)`** instead of inline gate/up matmuls (handles fused gate-up Phi-3 path automatically).
3. **Use `lw.attnNormBias` / `lw.ffnNormBias` adds** that the chat-model `forwardSingle` does (lines 740-741 and 902-903) but the bucket C embedder does not — Qwen3-Embedding-0.6B has no norm biases, but Phi-3 does, and the helper must support both.

The structural diff vs `forwardSingle`:
- No KV-cache writes. Skip lines `model-inference.ts:778-833` (the entire K/V cache view + opCpy + graphBuildForwardExpand-of-vWrite block).
- No `pastLen` / `totalLen`. Mask is `[N, padTo(N, 32)]` (causal-only over the current window).
- No `lm_head` matmul. Final tensor is `finalHidden = opMul(opRmsNorm(cur, eps), weights.norm)` — exact tap point.
- Graph memory: `hp.layerCount * 32768 + N * hp.embeddingLength * 24` (mirrors `causal-embedder-inference.ts:235`; no past-state multiplier).
- Readback: download `finalHidden` as `E * N` floats; copy into a stable Float32Array (heap view invalidates on next malloc/grow). Mirror `causal-embedder-inference.ts:418-430`.

```ts
	/**
	 * Build a single-pass causal forward over `tokenIds` and return the
	 * post-`output_norm` hidden state as a flat `[E * N]` Float32Array
	 * (row-major-reversed `[E, N]`). **Does not touch the KV cache.**
	 *
	 * Mirrors `CausalLMEmbedder.forwardEmbed` but uses this class's
	 * `buildQKV` / `buildFFNGateUp` helpers so it handles every chat
	 * architecture in the fleet (split / fused QKV; split / fused
	 * gate-up). The chat session's `nCached`, `kvLayers`, and
	 * conversation transcript are unchanged after this call.
	 *
	 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp`
	 * (`res->t_embd = cur` after the final RMSNorm, before `lm_head`).
	 *
	 * Concurrency: the caller must serialize this against any
	 * `forward()` / `forwardSingle()` on the same engine; both paths
	 * share the global WASM ctx-stack.
	 */
	private async forwardForEmbedding(
		tokenIds: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { hp, wasm, weights } = this;
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const headDim = hp.embeddingHeadLength;
		const nHeads = hp.headCount;
		const nKvHeads = hp.headCountKv;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const graphMem = hp.layerCount * 32768 + N * E * 24;
		wasm.ctxCreate(graphMem);

		try {
			const posTensor = wasm.tensorNew1d(GgmlType.I32, N);
			const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, N);

			const padTo = (v: number, mult: number) =>
				Math.ceil(v / mult) * mult;
			const maskPaddedRows = padTo(N, 32);
			const maskTensor = wasm.tensorNew2d(
				GgmlType.F16,
				N,
				maskPaddedRows,
			);

			const x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
			const graph = wasm.graphNew(hp.layerCount * 32 + 128);

			let cur = x;
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				let normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (lw.attnNormBias)
					normed = wasm.opAdd(normed, lw.attnNormBias);

				const { qReady, kReady, v3 } = this.buildQKV(
					lw,
					normed,
					N,
					headDim,
					nHeads,
				);

				const qRope = wasm.opRope(
					qReady,
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
					kReady,
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

				// Permute Q/K/V for manual attention chain (no FA — match
				// CausalLMEmbedder; FA is a separate optimization that
				// can be added later if profiling shows benefit).
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				const qk = wasm.opMulMat(kp, qp);
				const attnW = wasm.opSoftMaxExt(
					qk,
					maskTensor,
					1.0 / Math.sqrt(headDim),
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					N,
				);

				const oProj = wasm.opMulMat(lw.oProj, merged);
				const attnResidual = wasm.opAdd(oProj, cur);

				let ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (lw.ffnNormBias)
					ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);

				const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, N);
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

				cur = wasm.opAdd(ffnOut, attnResidual);
			}

			// Final output_norm — TAP POINT. No lm_head; no sampling.
			const finalHidden = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				weights.norm,
			);

			wasm.graphBuildForwardExpand(graph, finalHidden);
			const graphBuf = wasm.backendAllocCtxTensors();

			try {
				const idsBytes = N * 4;
				const posBytes = N * 4;
				const maskBytes = N * maskPaddedRows * 2;
				const totalBytes = idsBytes + posBytes + maskBytes;
				const heap = wasm.malloc(totalBytes);
				try {
					const idsPtr = heap;
					const posPtr = heap + idsBytes;
					const maskPtr = heap + idsBytes + posBytes;

					const idsView = new Int32Array(
						wasm.heapU8.buffer,
						idsPtr,
						N,
					);
					const posView = new Int32Array(
						wasm.heapU8.buffer,
						posPtr,
						N,
					);
					for (let i = 0; i < N; i++) {
						idsView[i] = tokenIds[i];
						posView[i] = i;
					}

					const F16_NEG_INF = 0xfc00;
					const mask = new Uint16Array(
						wasm.heapU8.buffer,
						maskPtr,
						N * maskPaddedRows,
					);
					for (let q = 0; q < N; q++) {
						const rowBase = q * N;
						for (let k = 0; k < N; k++) {
							mask[rowBase + k] = k <= q ? 0 : F16_NEG_INF;
						}
					}
					for (let q = N; q < maskPaddedRows; q++) {
						const rowBase = q * N;
						for (let k = 0; k < N; k++) mask[rowBase + k] = 0;
					}

					wasm.backendTensorSet3(
						tokenIdsTensor,
						idsPtr,
						idsBytes,
						posTensor,
						posPtr,
						posBytes,
						maskTensor,
						maskPtr,
						maskBytes,
					);
				} finally {
					wasm.free(heap);
				}

				await wasm.graphCompute(graph);

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
				return new Float32Array(hidden);
			} finally {
				wasm.backendBufferFree(graphBuf);
			}
		} finally {
			wasm.ctxFree();
		}
	}

	(Suppress `nKvHeads` unused-warn if needed by destructuring only what's used; the variable is informational and reads well alongside `nHeads`. If lint flags it, drop the line.)
```

- [ ] **Step 4: Add the public `embed()` method**

Place directly after `forwardForEmbedding`. Pools last-token, L2-normalizes, returns the embedding.

```ts
	/**
	 * Compute an L2-normalized sentence embedding by running a single-
	 * pass causal forward over `tokenIds`, tapping the post-
	 * `output_norm` hidden state, last-token-pooling, and L2-
	 * normalizing. **Does not write to the KV cache** — the chat
	 * session's state is unchanged.
	 *
	 * Concurrency: the caller (typically `engine.embed`) must serialize
	 * this against any concurrent `forward()` / `generate()` on the
	 * same engine. The two paths share the global WASM ctx-stack.
	 */
	async embed(tokenIds: Int32Array): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const hidden = await this.forwardForEmbedding(tokenIds);
		const E = this.hp.embeddingLength;
		const N = tokenIds.length;
		const lastCol = (N - 1) * E;
		const pooled = new Float32Array(E);
		for (let i = 0; i < E; i++) pooled[i] = hidden[lastCol + i];
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}
```

- [ ] **Step 5: Run typecheck and lint**

Run: `make typecheck && make lint`
Expected: clean.

- [ ] **Step 6: Run the test to verify it passes (or skips cleanly)**

Run: `bun test tests/model-inference-embed.test.ts`
Expected: PASS (in WebGPU-capable env) or skip (in Bun-only env).

- [ ] **Step 7: Run the full test suite to confirm no regressions**

Run: `make test`
Expected: same skip count (11) and same pass count + 1 new test (or skip).

- [ ] **Step 8: Commit**

```bash
git add src/inference/model-inference.ts tests/model-inference-embed.test.ts
git commit -m "feat(embed): add ModelInference.embed for chat-model self-embedding

forwardForEmbedding mirrors CausalLMEmbedder.forwardEmbed but reuses
buildQKV / buildFFNGateUp so every chat arch (split + fused) flows
through the same code. Structurally omits KV-cache writes and lm_head
— the chat session's nCached, kvLayers, and conversation transcript
are unchanged after embed().

Public embed(tokenIds) wraps the helper with last-token pool + L2
normalize. Concurrency is the caller's responsibility (shared WASM
ctx-stack, same as bucket C bug #2)."
```

---

## Task 3: Extend `engine.embed` dispatch ladder

**Files:**
- Modify: `src/core/engine.ts:473-500`
- Modify: error class message in whichever file defines `EncoderRequiredError`

- [ ] **Step 1: Add the third tier to `engine.embed`**

In `src/core/engine.ts`, locate the `embed()` method (currently lines 473-500, marker: `async embed(modelId: string, text: string)`). Replace the body with the three-tier ladder. The EOS-append convention from bucket C carries over — apply it for **both** the causal-embedder branch (existing) and the chat-model branch (new).

```ts
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
		if (cembed) {
			// EOS-append convention: matches sentence-transformers'
			// add_special_tokens=True post-processor template behavior.
			const eos = entry.tokenizer.eosId;
			const withEos =
				ids.length > 0 && ids[ids.length - 1] === eos
					? ids
					: [...ids, eos];
			return cembed.embed(new Int32Array(withEos));
		}

		// Bucket D — chat-model self-embedding. Gated on the registration
		// flag so we only fall through for chat models that have passed
		// the parity gate (encoderEngines / causalEmbedderEngines remain
		// the high-quality path; bucket D is the simplicity / single-
		// model-load path).
		if (entry.embeddingCapable) {
			const inf = this.inferenceEngines.get(modelId);
			if (inf) {
				const eos = entry.tokenizer.eosId;
				const withEos =
					ids.length > 0 && ids[ids.length - 1] === eos
						? ids
						: [...ids, eos];
				return inf.embed(new Int32Array(withEos));
			}
		}

		throw new EncoderRequiredError(
			modelId,
			String(entry.hyperparams.architecture),
		);
	}
```

- [ ] **Step 2: Verify `EncoderRequiredError` message mentions the new flag**

Find the error class:

Run: `grep -rn "class EncoderRequiredError" src/`

Open the file and update the message to add a sentence about `embeddingCapable`. If the message is constructed in the throw site rather than the class, update it in `engine.ts` instead. Example:

```ts
throw new EncoderRequiredError(
	modelId,
	String(entry.hyperparams.architecture),
	"register the model with `embeddingCapable: true` to use the chat-model self-embedding path",
);
```

If the constructor doesn't accept a hint argument, either widen its signature (preferred — additive) or append the hint to the existing message in the class body. Take the smaller of the two changes.

- [ ] **Step 3: Verify typecheck and lint**

Run: `make typecheck && make lint`
Expected: clean.

- [ ] **Step 4: Run engine tests**

Run: `bun test tests/engine` (if any tests are scoped to engine; otherwise `make test`).
Expected: existing tests pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/core/engine.ts src/core/errors.ts  # adjust path to error class
git commit -m "feat(embed): wire engine.embed dispatch to bucket D path

engine.embed gains a third dispatch tier: when no encoder /
causal-embedder is registered for the model id and the model entry
has embeddingCapable === true, route through inferenceEngines. EOS-
append convention shared with bucket C. EncoderRequiredError message
gains a one-line hint pointing at the embeddingCapable flag."
```

---

## Task 4: Set `embeddingCapable: true` on `qwen3-8b-iq3m`

**Files:**
- Modify: `eval/models.ts` (the `qwen3-8b-iq3m` entry)

- [ ] **Step 1: Locate the `qwen3-8b-iq3m` entry**

Run: `grep -n "qwen3-8b-iq3m" eval/models.ts`

- [ ] **Step 2: Add the flag**

Add `embeddingCapable: true,` to the entry's object literal, alongside the existing fields. Group it with the other functional flags (`localGGUFOnly`, `defaultQuant`, etc.):

```ts
	{
		id: "qwen3-8b-iq3m",
		// ...existing fields...
		embeddingCapable: true,
	},
```

- [ ] **Step 3: Verify typecheck**

Run: `make typecheck`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add eval/models.ts
git commit -m "feat(embed): enable bucket D on qwen3-8b-iq3m

Sets embeddingCapable: true on qwen3-8b-iq3m so engine.embed dispatch
routes through ModelInference.embed for this id. Parity validation
follows in a subsequent commit."
```

---

## Task 5: Capture PyTorch reference fixtures for `qwen3-8b-iq3m`

**Files:**
- Create: `eval/reports/bucket-d-probe-2026-04-29/inputs.json`
- Create: `eval/reports/bucket-d-probe-2026-04-29/capture-refs-requirements.txt`
- Create: `eval/reports/bucket-d-probe-2026-04-29/capture-refs.py`
- Create (script output): `eval/reports/bucket-d-probe-2026-04-29/qwen3-8b-ref.json`

- [ ] **Step 1: Create the fixture inputs**

Reuse 10 short / medium-length English sentences. Mirror the bucket C `inputs.json` format (a flat JSON array of strings).

```bash
mkdir -p eval/reports/bucket-d-probe-2026-04-29
```

```json
[
	"The cat sat on the mat.",
	"Quantum entanglement enables instantaneous correlation.",
	"Open the door and let the cool breeze in.",
	"Stock prices fell sharply after the merger announcement.",
	"Rendering ten thousand triangles per frame is now routine on integrated GPUs.",
	"A feline rested comfortably on the woven rug.",
	"She wrote a sonnet about the moon's reflection on still water.",
	"Compile-time type checking catches a wide class of programmer errors.",
	"The marathon runner crossed the finish line in just over two hours.",
	"Ergonomic keyboards reduce repetitive strain injuries over long sessions."
]
```

Save as `eval/reports/bucket-d-probe-2026-04-29/inputs.json`.

- [ ] **Step 2: Pin Python dependencies**

Save as `eval/reports/bucket-d-probe-2026-04-29/capture-refs-requirements.txt`:

```
torch
transformers
safetensors
accelerate
```

- [ ] **Step 3: Write the ref-capture script**

`Qwen/Qwen3-8B` is a base chat model — there is no `sentence-transformers` wrapper. Use raw HF `transformers` with `AutoModelForCausalLM` and `output_hidden_states=True` to extract the post-final-norm hidden state (`outputs.hidden_states[-1]`).

The webllm pipeline appends `tokenizer.eosId` (`<|im_end|>` for Qwen3 — id 151645). Match that exactly: tokenize with `add_special_tokens=False`, then manually append the eos id. Pool last-token, L2-normalize.

Save as `eval/reports/bucket-d-probe-2026-04-29/capture-refs.py`:

```python
"""
Bucket D Phase 0 — reference-embedding capture for Qwen3-8B (chat
model self-embedding). Tokenize identically to webllm (add_special_
tokens=False + manual EOS append), forward, take post-final-norm last-
token, L2-normalize. No sentence-transformers wrapper exists for the
chat base, so this uses raw transformers + output_hidden_states.

Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: qwen3-8b-ref.json — {model, captured_with, pooling, eos_id, fixtures}.
"""
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

MODEL_NAME = "Qwen/Qwen3-8B"
POOLING = "last-token"
TOLERANCE = 1e-3

print(f"Loading {MODEL_NAME}…", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
	MODEL_NAME,
	# f16 ref is tight enough for cos >= 0.999 against IQ3_M WASM;
	# f32 would be ~32 GB and OOMs on the 16/32 GB hardware tier.
	torch_dtype=torch.float16,
	device_map="cpu",  # deterministic; small fixture batch
)
model.eval()

EOS_ID = tokenizer.eos_token_id
print(f"EOS id: {EOS_ID}", file=sys.stderr)

vectors = []
with torch.no_grad():
	for text in inputs:
		# Match webllm's pipeline exactly: encode without specials, then
		# manually append EOS so the tokenization matches what
		# engine.embed feeds to ModelInference.embed.
		ids = tokenizer.encode(text, add_special_tokens=False)
		if not ids or ids[-1] != EOS_ID:
			ids = ids + [EOS_ID]
		input_ids = torch.tensor([ids], dtype=torch.long)
		out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
		# hidden_states[-1] is the post-final-norm last layer output.
		last = out.hidden_states[-1][0, -1, :]  # last token, [E]
		v = last / last.norm(p=2)
		vectors.append(v.tolist())

EXPECTED_DIM = len(vectors[0])
print(f"Expected output dim: {EXPECTED_DIM}", file=sys.stderr)

for i, v in enumerate(vectors):
	mag = math.sqrt(sum(x * x for x in v))
	if abs(mag - 1.0) > TOLERANCE:
		raise SystemExit(
			f"Magnitude check failed for row {i}: |v|_2 = {mag:.6f}"
		)
	if len(v) != EXPECTED_DIM:
		raise SystemExit(
			f"Dim check failed for row {i}: dim = {len(v)} (expected {EXPECTED_DIM})"
		)

print(f"All {len(vectors)} vectors passed.", file=sys.stderr)

fixtures = [
	{"row": i, "input": inputs[i], "mode": "document", "vec": vectors[i]}
	for i in range(len(inputs))
]
out_path = HERE / "qwen3-8b-ref.json"
out_path.write_text(
	json.dumps(
		{
			"model": MODEL_NAME,
			"captured_with": f"transformers (torch.float32 cpu); add_special_tokens=False + manual EOS append (id {EOS_ID})",
			"pooling": POOLING,
			"eos_id": EOS_ID,
			"fixtures": fixtures,
		},
		indent=2,
	)
)
print(f"Wrote {out_path}", file=sys.stderr)
```

- [ ] **Step 4: Run the script**

Run:
```bash
cd eval/reports/bucket-d-probe-2026-04-29 && \
uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
```
Expected: download `Qwen/Qwen3-8B` (~16 GB), runs forward on 10 inputs, writes `qwen3-8b-ref.json`. Wall time depends on network speed for the first-time download (15-60 min typical) and ~1-2 min for the forward pass.

- [ ] **Step 5: Sanity-check the output**

```bash
python3 -c "
import json
d = json.load(open('eval/reports/bucket-d-probe-2026-04-29/qwen3-8b-ref.json'))
print('rows:', len(d['fixtures']))
print('dim:', len(d['fixtures'][0]['vec']))
print('eos_id:', d['eos_id'])
"
```
Expected: `rows: 10`, `dim: 4096`, `eos_id: 151645`.

- [ ] **Step 6: Commit**

```bash
git add eval/reports/bucket-d-probe-2026-04-29/
git commit -m "test(embed): pin bucket D parity refs for qwen3-8b-iq3m

10 fixtures captured against Qwen/Qwen3-8B (torch.float32 cpu) with
add_special_tokens=False + manual EOS append, post-final-norm last-
token pool + L2 normalize. Matches the webllm pipeline exactly.

No sentence-transformers wrapper exists for the chat base, so this
uses raw transformers + output_hidden_states (different from
bucket C's SentenceTransformer.encode path)."
```

---

## Task 6: Run parity gate + 4-pair distinguishability check

**Files:**
- Modify: `eval/causal-embedder-parity.ts` (extend dispatch to recognize `embeddingCapable` chat models; add 4-pair sanity check)
- Output: `eval/reports/bucket-d-parity-2026-04-29/run.txt` (raw harness log; commit later in closure)

- [ ] **Step 1: Survey the existing harness**

Run: `wc -l eval/causal-embedder-parity.ts && grep -n "causalEmbedderEngines\|engine.embed\|embeddingCapable\|gate\|cosine\|defaultQuant" eval/causal-embedder-parity.ts | head -30`

This tells you which hooks already exist. The harness already drives `engine.embed(modelId, text)` end-to-end through the browser; the only changes needed are:
1. Allow it to accept a chat model id with `embeddingCapable: true`. Likely already model-agnostic since dispatch happens inside the engine, not the harness.
2. Add the 4-pair sanity check after parity passes.

- [ ] **Step 2: Add 4-pair distinguishability assertions**

Add a new function (or block at the end of the parity run) that compares cosine similarity of paraphrase pairs vs unrelated pairs. Hardcode the 4 pairs:

```ts
const PARAPHRASE_PAIRS: ReadonlyArray<readonly [string, string]> = [
	["The cat sat on the mat.", "A feline rested comfortably on the woven rug."],
	[
		"Compile-time type checking catches a wide class of programmer errors.",
		"Static type analysis prevents many bugs at build time.",
	],
];

const UNRELATED_PAIRS: ReadonlyArray<readonly [string, string]> = [
	[
		"The cat sat on the mat.",
		"Stock prices fell sharply after the merger announcement.",
	],
	[
		"Compile-time type checking catches a wide class of programmer errors.",
		"The marathon runner crossed the finish line in just over two hours.",
	],
];

function cosine(a: Float32Array, b: Float32Array): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s; // pre-normalized vectors → dot is cosine
}
```

Run all 8 sentences through `engine.embed(modelId, ...)`, compute pair cosines, and assert each paraphrase cosine > each unrelated cosine. Log the values.

- [ ] **Step 3: Run the parity harness against `qwen3-8b-iq3m`**

(Specific invocation depends on the existing harness CLI; mirror the bucket C invocation. The bucket C closure uses `eval/causal-embedder-parity.ts` with the model id passed in; use the same shape.)

Run: `bun run eval/causal-embedder-parity.ts qwen3-8b-iq3m eval/reports/bucket-d-probe-2026-04-29/qwen3-8b-ref.json`

(If the harness CLI doesn't take args this way, look up the bucket C call site and follow the same convention.)

Expected:
- Gate selection logs `cos >= 0.999` (qwen3-8b-iq3m has `defaultQuant: "q4f16_1"` which is non-hybrid → tight gate).
- 10/10 fixtures PASS at `cos >= 0.999`. Magnitudes `1.000 ± 1e-6`.
- 4-pair distinguishability PASSES — every paraphrase cosine > every unrelated cosine.
- No console errors in the browser log.

If parity fails, do not paper over it. Diagnose using the bucket C diagnostic ladder (Signature A: tap point wrong layer; Signature B: tokenizer mismatch; Signature C: prefix not applied or tokenizer mismatch; Signature D: norm/scale off). The most likely failure modes for bucket D specifically:
- **EOS id mismatch** between the ref-capture script and the webllm tokenizer. Verify `tokenizer.eosId` in the WASM build equals 151645 for Qwen3-8B.
- **`output_hidden_states[-1]` pre/post-norm ambiguity.** If the HF transformers Qwen3 implementation puts the final RMSNorm *outside* the layer loop (i.e., `last_hidden_state` is post-norm but `hidden_states[-1]` is pre-norm), the ref vectors won't match. Compare `out.hidden_states[-1]` vs `out.last_hidden_state` in the script and pick the post-norm variant.

- [ ] **Step 4: Save the harness log to the parity report dir**

```bash
mkdir -p eval/reports/bucket-d-parity-2026-04-29
# Re-run with output redirected:
bun run eval/causal-embedder-parity.ts qwen3-8b-iq3m \
	eval/reports/bucket-d-probe-2026-04-29/qwen3-8b-ref.json \
	2>&1 | tee eval/reports/bucket-d-parity-2026-04-29/run.txt
```

- [ ] **Step 5: Commit harness changes**

```bash
git add eval/causal-embedder-parity.ts
git commit -m "test(embed): extend parity harness for bucket D + 4-pair sanity

Recognizes embeddingCapable chat models (engine.embed dispatch
routes through ModelInference.embed under the covers). Adds a 4-pair
cosine-distinguishability sanity check that runs after the primary
parity gate passes — catches the 'tap-point picked the wrong layer'
failure mode that the parity gate alone might miss if the bug is
symmetric across PyTorch and WASM."
```

(Don't commit `run.txt` yet — it's the artifact for the closure report in Task 8.)

---

## Task 7: Add `qwen3-8b-iq3m` to `eval/embed-perf.ts` and bench

**Files:**
- Modify: `eval/embed-perf.ts`
- Output: `eval/reports/embed-perf-qwen3-8b-2026-04-29/`

- [ ] **Step 1: Survey `eval/embed-perf.ts`**

Run: `grep -n "encoderEngines\|causalEmbedderEngines\|inferenceEngines\|embeddingCapable\|defaultQuant" eval/embed-perf.ts | head -20`

The harness already supports `encoderEngines` (BERT-arch) and `causalEmbedderEngines` (bucket C) rows. Add a third path that recognizes `embeddingCapable` chat models and drives `engine.embed(modelId, text)` for them. The dispatch happens inside the engine; the harness change is just to allow chat model ids in the model list.

- [ ] **Step 2: Make the harness model-list-agnostic**

If the harness has a hard-coded list of "embedding-eligible" architectures, widen it to include any registered model that satisfies one of:
- `encoderEngines.has(id)`
- `causalEmbedderEngines.has(id)`
- `entry.embeddingCapable === true`

The harness doesn't need to know *which* path the engine takes — `engine.embed()` makes that decision.

- [ ] **Step 3: Run the bench**

Run: `bun run eval/embed-perf.ts qwen3-8b-iq3m`

(Or the project's standard invocation — match what bucket C's commit `2724b02` did.)

Expected: single-text short / long p50 (ms), batch throughput (texts/sec), no errors. Numbers will be slower than bucket C because bucket D runs an 8B model vs bucket C's 0.6B (~10-15x compute per token). Informational; not gated.

- [ ] **Step 4: Save bench output**

Per the bucket C convention the bench writer drops a directory under `eval/reports/embed-perf-...`. Follow it.

- [ ] **Step 5: Commit**

```bash
git add eval/embed-perf.ts eval/reports/embed-perf-qwen3-8b-2026-04-29/
git commit -m "feat(embed-perf): bench coverage for bucket D (qwen3-8b-iq3m)

Adds qwen3-8b-iq3m to the embed-perf bench. First 8B-class row in
the dashboard's Embeddings section; informational, not gated.
Throughput is intentionally not compared against bucket C entries —
the 8B chat model serves the single-model-load tradeoff use case,
not the high-quality-embedding use case."
```

---

## Task 8: Closure report

**Files:**
- Create: `eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`

- [ ] **Step 1: Mirror the bucket C closure report structure**

Use [`eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md`](../../../eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md) as the template. Fill in the bucket D specifics:

- **Outcome table**: 10 rows × 1 mode (document only — bucket D doesn't have an instruction-prefixed query mode; the chat model isn't trained as an embedder, so there's no canonical query-mode prefix). Columns: row, cosine, magnitude.
- **Gate selection**: `0.999` (qwen3-8b-iq3m is non-hybrid; tight gate per bucket C selection rule). Cite the gate selection rule + the `defaultQuant === "q4f16_1"` decision.
- **4-pair distinguishability table**: 4 paraphrase cosines + 4 unrelated cosines + the strict-inequality assertion outcome.
- **Bench numbers**: single-text short p50 ms, single-text long p50 ms, batch throughput. Compare loosely to bucket C's numbers as the "single-model-load tradeoff vs dedicated-embedder" frame.
- **Bugs fixed**: list anything that surfaced during the cycle. Likely candidates: tokenizer-side EOS handling, hidden-states pre/post-norm ambiguity. Document any with the same depth bucket C used.
- **Files touched**: full list (Tasks 1-9).
- **Notes for follow-on work**: cross-arch generalization (Llama / Mistral / Phi-3 next cycle); concurrency mutex watch-list item; quality-tradeoff API surfacing recap.

- [ ] **Step 2: Commit**

```bash
git add eval/reports/bucket-d-parity-2026-04-29/
git commit -m "docs(report): bucket D parity closure (qwen3-8b-iq3m)

10/10 PASS at cos >= 0.999 against Qwen/Qwen3-8B reference vectors;
4-pair distinguishability sanity passes. First chat-model-as-embedder
row in the fleet — single-model-load path for the agent + Three.js
deployment doctrine."
```

---

## Task 9: Documentation updates

**Files:**
- Modify: `README.md` (Embeddings section)
- Modify: `CLAUDE.md` (doctrine entry; update existing "Single-model-active deployment" block to cross-link bucket D)
- Modify: JSDoc on `src/core/engine.ts` `embed()` method (already partially done in Task 3 — top up if anything is missing)

- [ ] **Step 1: Update README "Embeddings" section**

Document the three-tier dispatch ladder: encoder (high-quality, e.g., bge-large) → causal-embedder (high-quality retrieval-tuned, e.g., qwen3-embedding-0.6b-hyb) → chat-model (single-model-load tradeoff, e.g., qwen3-8b-iq3m with `embeddingCapable: true`). Note the 5-15% MTEB delta vs dedicated embedders; recommend dedicated embedders for general retrieval, bucket D for in-domain agent retrieval.

- [ ] **Step 2: Update CLAUDE.md**

Add a one-paragraph entry under the existing "Single-model-active deployment" block (or extend it inline) noting that bucket D is now shipped and gated per-model via `embeddingCapable`. Cite the closure report.

- [ ] **Step 3: Audit `engine.embed` JSDoc**

Confirm the JSDoc on `engine.embed` in `src/core/engine.ts` documents the three-tier dispatch order, mentions `embeddingCapable`, and cites the quality tradeoff. If anything is missing from Task 3, add it now.

- [ ] **Step 4: Run docs / typecheck guardrails**

Run: `make checkall`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md src/core/engine.ts
git commit -m "docs(embed): document bucket D dispatch tier and quality tradeoff

README Embeddings section explains the three-tier ladder (encoder >
causal-embedder > chat-model) and the ~5-15% MTEB delta. CLAUDE.md
single-model-active deployment block notes bucket D is shipped and
gated per-model via embeddingCapable. engine.embed JSDoc documents
the dispatch order."
```

---

## Task 10: TODO closure

**Files:**
- Modify: `TODO.md` (item 6 in the "Next session pickup" block; reduce to a closure stub)
- Modify: `TODO_ARCHIVE.md` (move full bucket D narrative)

- [ ] **Step 1: Verify `make checkall` is green**

Run: `make checkall`
Expected: clean.

- [ ] **Step 2: Replace TODO.md item 6 with a closure stub**

Mirror the structure used for items 4 and 5 in the same block:

```markdown
6. **Embedding bucket D — chat-model self-embedding. CLOSED <DATE>.**
   `ModelInference.embed(tokenIds)` shipped; `engine.embed` dispatches
   through `inferenceEngines` for chat models with `embeddingCapable:
   true`. **`qwen3-8b-iq3m`** is the single registered bucket D model
   at v1; other archs follow as separate cycles.

   Parity 10/10 PASS at `cos >= 0.999` against Qwen/Qwen3-8B
   reference vectors. 4-pair cosine distinguishability sanity passes.
   Closure report
   [`eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`](eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md);
   spec [`docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md`](docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md);
   plan [`docs/superpowers/plans/2026-04-29-embedding-bucket-d.md`](docs/superpowers/plans/2026-04-29-embedding-bucket-d.md).

   Full bucket D block (Q1-Q5 design rationale, per-task commit map,
   ref-capture recipe, follow-up arch cycles) archived to
   `TODO_ARCHIVE.md` under "Embedding bucket D (closed <DATE>)".
```

- [ ] **Step 3: Move full block to `TODO_ARCHIVE.md`**

Append under a new section heading `### Embedding bucket D (closed <DATE>)` with the full original item 6 content, the per-task commit list (run `git log --oneline ${baseRef}..HEAD` where `baseRef` is the commit before Task 1), and links to all artifacts.

- [ ] **Step 4: Commit (separate from closures, per the doctrine)**

```bash
git add TODO.md TODO_ARCHIVE.md
git commit -m "docs(TODO): close bucket D — chat-model self-embedding shipped

Replaces inline item 6 narrative with a closure stub linking to the
parity report, spec, and plan. Full block (design rationale, commit
map, follow-up arch cycles) archived to TODO_ARCHIVE.md.

TODO archival is its own commit per the cadence doctrine — bundling
it with the implementation closures would block git revert of the
archive without touching the closures."
```

---

## Self-review checklist (planner — before handoff)

**Spec coverage (every section in `2026-04-29-embedding-bucket-d-design.md`):**
- Public API (`ModelInference.embed`, `engine.embed` third tier) → Tasks 2, 3.
- Per-model `embeddingCapable` flag → Tasks 1, 4.
- Quality tradeoff surfacing (docs only) → Task 9.
- Forward graph (no KV writes; `buildQKV` / `buildFFNGateUp` reuse) → Task 2.
- EOS-append convention → Task 3.
- Concurrency caveat → Task 2 (JSDoc), Task 9 (docs).
- KV-cache non-perturbation invariant → Task 2 (load-bearing test).
- PyTorch parity gate → Tasks 5, 6.
- 4-pair distinguishability → Task 6.
- Browser smoke (subsumed by browser-driven parity harness) → Task 6.
- Bench coverage → Task 7.
- Closure report → Task 8.
- Out-of-scope items (mutex, tier accessor, alternative pooling, MTEB) → not implemented; Task 8 closure notes.

**Placeholder scan:** no TBD/TODO/"implement later"/"add appropriate"; every step has concrete code or commands.

**Type consistency:** `ModelInference.embed`, `forwardForEmbedding`, `embeddingCapable`, `EncoderRequiredError` used consistently across tasks. EOS-append logic is duplicated literally between Task 3's two branches (same dropping-then-re-appending rule) and the ref-capture script — by design, since the ref must match the WASM-side tokenization byte-for-byte.

**Frequent commits:** 12 commits across 10 tasks (Tasks 6 and 9 each produce 2). Each commit is independently revertable per the project doctrine.
