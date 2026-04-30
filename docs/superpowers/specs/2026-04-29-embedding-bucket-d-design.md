# Embedding bucket D — chat-model self-embedding (design)

**Date:** 2026-04-29
**Predecessor:** [bucket C closure report](../../../eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md);
[bucket C implementation design](2026-04-29-embedding-bucket-c-implementation-design.md)
**Status:** brainstormed; ready for implementation plan

## Motivation

The project's load-bearing use case is **agent + Three.js coexistence in
a single tab** (browser-side LLM driving 3D agents alongside a
renderer). On the 16 GB unified-memory floor that hardware tier targets,
loading a chat model **and** a dedicated embedder simultaneously eats
into the WebGPU budget that Three.js + KV cache also draw from.

Bucket D adds a **single-model-load** embedding path: when a user's
retrieval workload is in-domain (agent memory, dialogue history,
semantic search over game state), they can run their already-loaded
chat model in embedding mode — saving a second model load and halving
cold-start. Quality drops 5-15% on MTEB benchmarks vs dedicated
retrieval-tuned embedders (e.g., bucket C's Qwen3-Embedding-0.6B-hyb)
but is "good enough" for in-domain retrieval.

The bucket C cycle proved out the architecture-routing groundwork
(`metaPrefix` split, EOS-append convention, hybrid-tier parity gate,
per-binding 128 MiB cap doctrine). Bucket D generalizes that
infrastructure to chat models, reusing ~70% of `CausalLMEmbedder`'s
forward-pass logic.

## Architecture

### Public API

Add a new instance method to `ModelInference`:

```ts
class ModelInference {
  // ...existing chat-generation surface...

  /**
   * Compute the post-`output_norm` hidden state for `tokenIds`, last-
   * token-pool, L2-normalize, and return the embedding vector. **Does
   * not write to the KV cache.** The chat session's `nCached`,
   * `kvLayers`, and conversation transcript are unchanged after this
   * call.
   *
   * Concurrency: the caller must serialize this call relative to any
   * concurrent `forward()` / `generate()` on the same engine. The two
   * paths share the global WASM ctx-stack; concurrent graph builds
   * race (same root cause as bucket C's parity-harness ↔ embedPerf
   * race).
   */
  async embed(tokenIds: Int32Array): Promise<Float32Array> { ... }
}
```

Extend `engine.embed(modelId, text)`'s dispatch ladder to a third tier:

```
encoderEngines.get(id)         → BERT-arch encoders          (unchanged)
causalEmbedderEngines.get(id)  → bucket C path               (unchanged)
inferenceEngines.get(id)       → bucket D path  (NEW; gated on entry.embeddingCapable)
else                           → throw EncoderRequiredError  (unchanged)
```

`engine.embed`'s return type stays `Promise<Float32Array>`; no caller
changes are required for downstream code.

### Per-model registration flag

Add an optional field to `RegisteredModel`:

```ts
interface RegisteredModel {
  // ...existing fields...
  /** When true, `engine.embed(id, text)` is allowed to dispatch
   *  through `inferenceEngines` for this model. Chat models opt in
   *  declaratively after passing the bucket D parity gate. */
  embeddingCapable?: boolean;
}
```

Ship-time scope: **`qwen3-8b-iq3m` only.** Other registered chat models
(Llama 3.x, Mistral 7B, Phi-3.5) opt in through follow-up cycles, each
with its own PyTorch reference capture, parity run, and closure report.
Each follow-up is independent of the API surface and reuses the same
harness machinery.

### Quality tradeoff surfacing

**Documentation only, no runtime API.** The user already opts into the
quality tradeoff by registering a chat model with `embeddingCapable:
true`. JSDoc on `engine.embed`, the README "Embeddings" section, and a
CLAUDE.md doctrine entry document the tier ladder
(encoder > causal-embedder > chat-model) and the ~5-15% MTEB delta vs
dedicated embedders. No tier accessor, no result-shape change, no
runtime warning.

### Forward graph

`ModelInference.embed(tokenIds)` calls a new private helper
`forwardForEmbedding(tokenIds)` that mirrors `CausalLMEmbedder.
forwardEmbed` (`src/inference/causal-embedder-inference.ts:223+`) but
reuses `ModelInference`'s existing `buildQKV` and `buildFFNGateUp`
helpers so it handles every chat architecture in the fleet:

- **Split-QKV** (Llama / Qwen / Mistral / SmolLM / Hermes): three
  matmuls per layer, optional bias add, reshape.
- **Fused-QKV** (Phi-3): single matmul on `lw.qkvFused`, three
  `opView3d` slices.

The existing `forwardSingle` body uses these helpers; the embedding
forward differs only in:

1. **No KV-cache writes.** The `opCpy` to `kv.k` / `kv.v` views (lines
   `model-inference.ts:778-788+`) is structurally absent. Attention
   reads K and V from the freshly-built `kRope` and `v3` tensors of
   the same forward pass, not from the cache.
2. **No `nCached` advance.** The chat session's `nCached` field is
   never read or written by the embedding forward.
3. **No lm_head matmul.** The tap point is post-`output_norm`,
   pre-`lm_head` — same as bucket C, same as
   `qwen3.cpp:98 res->t_embd = cur`.
4. **No sampling.** The forward returns the hidden state directly;
   `embed()` pools last-token and L2-normalizes inline.
5. **Causal mask over `[N, N]`.** No past-state component (`pastLen ==
   0`); the mask is built once per call.

Graph memory budget: `layerCount * 32768 + N * E * 24`, mirroring
`CausalLMEmbedder.forwardEmbed`. (No past-state multiplier because
there is no past state.)

### EOS-append convention

`engine.embed` appends `tokenizer.eosId` to the encoded ids before
calling `inf.embed(idsWithEos)`, dropping any existing trailing EOS
first to keep the count at one. Same convention as bucket C
(`engine.ts:484-494`). The reference-capture script does the same on
the PyTorch side, so parity holds.

This is a **project convention**, not a model-trained convention —
chat models weren't fine-tuned to interpret EOS as "summarize what
you've seen." We pick it because (a) bucket C's last-token pooling
already targets the EOS position; (b) it gives a deterministic rule
that is independent of the input text; (c) it matches what
sentence-transformers does for any add-special-tokens=True path. The
spec calls this out so future readers don't mistake it for a quality-
optimized choice.

### Concurrency

`ModelInference.embed` and `ModelInference.forward` / `generate` race
on the global WASM ctx-stack if called concurrently — same root cause
as bucket C bug #2 (`webgpu-bridge.cpp` push/pop on a single global
ctx slot). The single-active-conversation deployment doctrine (see
CLAUDE.md) makes this acceptable for v1: callers serialize their own
forward + embed calls.

An internal mutex on the engine that serializes graph builds is a
follow-up watch-list item, not a v1 deliverable. Defer until a real
caller hits the race.

## Components touched

| File | Change | Approx LOC |
|---|---|---:|
| `src/inference/model-inference.ts` | Add `embed(tokenIds)` public method + private `forwardForEmbedding(tokenIds)` helper. | ~180 |
| `src/core/engine.ts` | Extend `embed()` dispatch (lines 473-500) with the third-tier lookup gated on `entry.embeddingCapable`. | ~15 |
| `eval/models.ts` | Add `embeddingCapable?: boolean` to `RegisteredModel`; set on `qwen3-8b-iq3m`. | ~5 |
| `eval/causal-embedder-parity.ts` | Recognize `embeddingCapable` chat models and route through `engine.embed`. | ~20 |
| `eval/reports/bucket-d-probe-2026-04-29/capture-refs.py` | New PyTorch script: load `Qwen/Qwen3-8B`, tokenize identically to webllm, append EOS, forward, tap post-final-norm last-token, L2, dump 10 fixtures. | new (~80) |
| `tests/model-inference-embed.test.ts` | KV-cache non-perturbation unit test. | new (~60) |
| `eval/embed-perf.ts` | Recognize `embeddingCapable` chat models so the bench harness drives `engine.embed` for them. | ~10 |
| `eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md` | Closure report (parity + 4-pair distinguishability + smoke + bench). | new |
| `README.md` / `CLAUDE.md` / JSDoc | Quality-tradeoff documentation. | ~30 |

Total: ~270 LOC of code + ~80 LOC Python ref capture + tests + reports.

## Data flow

```
engine.embed("qwen3-8b-iq3m", "search query")
    │
    ├─ _modelManager.get(id)               → entry (must be loaded)
    ├─ tokenizer.encode(text)               → ids
    ├─ encoderEngines.get(id)               → undefined
    ├─ causalEmbedderEngines.get(id)        → undefined
    ├─ check entry.embeddingCapable         → true
    ├─ inferenceEngines.get(id)             → ModelInference instance
    ├─ append tokenizer.eosId               → idsWithEos
    │
    └─ inf.embed(new Int32Array(idsWithEos))
        │
        ├─ forwardForEmbedding(tokenIds): hidden [E, N]
        │   ├─ wasm.ctxCreate(graphMem)
        │   ├─ leaf inputs: pos, ids, causal mask [N, padTo(N, 32)]
        │   ├─ x = opGetRows(weights.tokEmb, tokenIdsTensor)
        │   ├─ for il in [0, layerCount):
        │   │     normed = RMSNorm(cur) * attnNorm   (+ optional bias)
        │   │     {qReady, kReady, v3} = buildQKV(lw, normed, N, headDim, nHeads)
        │   │     qRope, kRope = opRope(...)
        │   │     # NO opCpy to kv.k / kv.v — load-bearing diff vs forwardSingle
        │   │     qk = opMulMat(kp, qp); attn = softMaxExt(qk, mask, scale)
        │   │     attnOut = opMulMat(vp, attn); merged = reshape2d(...)
        │   │     attnResidual = opAdd(opMulMat(oProj, merged), cur)
        │   │     ffnNormed = RMSNorm(attnResidual) * ffnNorm
        │   │     ffnHidden = buildFFNGateUp(lw, ffnNormed, N)
        │   │     cur = opAdd(opMulMat(downProj, ffnHidden), attnResidual)
        │   ├─ finalHidden = RMSNorm(cur) * weights.norm     ← TAP POINT
        │   ├─ graphBuildForwardExpand(graph, finalHidden)
        │   ├─ backendAllocCtxTensors → upload pos/ids/mask via tensorSet3
        │   ├─ graphCompute (await)
        │   └─ readback finalHidden → Float32Array(E * N)
        │
        ├─ pool: lastCol = (N-1)*E; pooled = hidden[lastCol .. lastCol+E]
        ├─ L2-normalize pooled
        └─ return pooled                  (E = 4096 for Qwen3-8B)
```

**Invariants:**
- `forwardForEmbedding` allocates a fresh ctx per call. No overlap
  with the chat-session ctx; no leak across calls (`try/finally`
  releases the ctx after readback).
- KV-write call sites are **structurally absent** in
  `forwardForEmbedding`, not branched-around. Less surface to
  test, less to break.
- `buildQKV` and `buildFFNGateUp` are reused as-is; their fused/split
  branching means Phi-3 / Llama / Qwen / Mistral all flow through
  the same code.
- `this.nCached`, `this.kvLayers[i].k`, `this.kvLayers[i].v` are
  invariant across an `embed()` call. The KV-cache non-perturbation
  unit test is the load-bearing assertion of this.

## Error handling

| Condition | Behavior |
|---|---|
| `modelId` not registered | `ModelNotFoundError(modelId)` (existing) |
| Model registered but not loaded | `ModelNotLoadedError(modelId)` (existing) |
| No encoder, no causal embedder, `embeddingCapable !== true` | `EncoderRequiredError(modelId, architecture)` — message text gains one sentence: "or register the model with `embeddingCapable: true` if it's a chat model" |
| `embeddingCapable === true` but `inferenceEngines` lookup fails | `EncoderRequiredError` falls through; programmer error in registration; covered by existing tests |
| `tokenIds.length === 0` after EOS-append (impossible — EOS is always appended) | Defensive throw `"embed() received empty input after tokenization"`, mirrors `CausalLMEmbedder.embed` |
| Architecture not yet validated for bucket D | The `embeddingCapable` flag is the gate. We only set it on architectures that have passed the parity gate. |

No new error classes.

## Testing strategy

1. **Unit — KV-cache non-perturbation** (`tests/model-inference-embed.test.ts`).
   - Load tinyllama (smallest chat model; we bypass the dispatch flag
     and call `inf.embed` directly to keep the test independent of
     `eval/models.ts` registration).
   - `forward([t1, t2, t3])` → `logitsA`. Snapshot `nCached === 3` and
     a hash of every `kvLayers[i].k` / `kvLayers[i].v` buffer.
   - `embed([e1, e2])` → returns Float32Array; assert `nCached === 3`
     unchanged, every KV buffer hash unchanged.
   - `forward([t4])` → `logitsB`. Re-run the whole sequence
     `forward([t1..t3]) → forward([t4])` without the `embed` and
     compare `logitsB` byte-for-byte. **Load-bearing assertion** that
     `embed` is structurally orthogonal to chat state.

2. **Parity — PyTorch forward-graph equivalence** (extends
   `eval/causal-embedder-parity.ts`).
   - 10 fixtures from `bucket-d-probe-2026-04-29/capture-refs.py`.
   - Gate: `cos >= 0.999` (qwen3-8b-iq3m is non-hybrid; tight gate
     per bucket C selection rule). Magnitudes `1.000 ± 1e-6`.

3. **Sanity — 4-pair cosine distinguishability** (same harness, after
   parity passes).
   - 4 hardcoded pairs: 2 paraphrase pairs, 2 unrelated pairs.
   - Assert each paraphrase cosine > each unrelated cosine. Catches
     the "tap-point picked the wrong layer" failure mode that A might
     miss if the bug is symmetric across WASM and PyTorch.

4. **Smoke — browser end-to-end** (`make smoke-bench` profile or
   analogous). Verify `engine.embed("qwen3-8b-iq3m", "...")` returns
   a 4096-dim L2-normalized vector with no console errors.

5. **Bench — `eval/embed-perf.ts` row** for `qwen3-8b-iq3m`. Single-
   text short / long p50; batch throughput. First 8B-class row in the
   dashboard's Embeddings section. Informational, not gated.

6. **Existing test suite stays green.** `make checkall` (fmt + lint +
   typecheck + test) is the ship gate per CLAUDE.md.

## Out of scope

- **Other chat archs** (Llama 3.x, Mistral 7B, Phi-3.5). Each is a
  separate cycle: ref capture + parity run + smoke + bench + closure
  report. The API surface is fixed at v1; subsequent cycles only add
  registrations.
- **Internal serialization mutex.** Concurrency is documented;
  caller serializes. Watch-list follow-up; not v1.
- **Tier accessor** (`engine.getEmbeddingTier(modelId)`). Speculative;
  no caller asks for it. Defer.
- **Mean-pool / LM2Vec-style pooling alternatives.** Last-token pool
  with EOS append is the project convention; alternative pooling
  modes (mean, attention-weighted) are a future quality lever, not a
  v1 surface.
- **Internal mutation of `nCached` to "use chat history as
  embedding context".** Bucket D treats `embed(text)` as a fresh
  encoding of `text` alone. Mixing chat history into the embedding
  context is a separate design (and would re-open the KV-cache-
  poisoning question that v1 deliberately avoids).
- **MTEB micro-benchmark in the parity gate.** If we want
  retrieval-quality numbers, that's a separate `eval/reports/`
  artifact, not a ship gate.

## Open questions

None. The five design questions raised in TODO.md item 6 are all
resolved:

| # | Question | Answer |
|---|---|---|
| 1 | KV-cache interaction | A — stateless forward path |
| 2 | Auto-fallback vs opt-in | C — per-model `embeddingCapable` flag |
| 3 | Quality-tradeoff surfacing | A — docs only |
| 4 | Parity gate strategy | A + B — PyTorch forward parity + 4-pair distinguishability |
| 5 | Ship-time chat-model coverage | A — `qwen3-8b-iq3m` only |
