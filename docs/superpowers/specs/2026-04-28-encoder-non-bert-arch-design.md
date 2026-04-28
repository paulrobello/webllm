# Encoder non-BERT architecture support — design

**Date:** 2026-04-28
**Status:** Implementation complete (Phase 5 closed). This file captures the
spec corrected to match what shipped — see "Post-implementation corrections"
note below.
**Tracking:** TODO.md "Embedding-model expansion candidates — Bucket B" (DONE).
**Cycle precedent:** §17/§18/§19/§20 phased plan structure.

> **Revision note (2026-04-28).** A first-pass spec under this filename
> assumed both encoders were paper-BERT clones with split QKV, attention
> biases, and GeLU FFN. The Phase 0 GGUF discovery probe (`43df996`,
> `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt`) showed
> material structural divergences. This document is the post-probe rewrite
> that the implementation actually uses. The original (incorrect) spec
> survives at git ref `8064f80:docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`
> as historical context.

> **Post-implementation corrections (2026-04-28, bucket B follow-up #11).**
> Phase 3 / Phase 4 integration surfaced four spec/reality mismatches; the
> implementation diverged from the post-Phase-0 spec to match llama.cpp
> ground truth. This section is the canonical patch. Truth sources cited
> per item:
>
> 1. **jina-bert-v2 FFN is GeGLU** (`gelu(gate) * up`), not SwiGLU.
>    Truth: `~/Repos/llama.cpp/src/models/bert.cpp:122-130`. The spec's
>    Point F branch was generalized from "SwiGLU only" to per-arch
>    activation selection (silu for nomic, gelu for jina).
> 2. **nomic-bert RoPE mode is NEOX**, not NORMAL. Truth:
>    `~/Repos/llama.cpp/src/llama-model.cpp:9266` (uses
>    `LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_NOMIC_BERT*`).
> 3. **Encoder ALiBi mask is `-|i - j|` populated**, not zero-filled
>    or NULL. Truth: `~/Repos/llama.cpp/src/llama-graph.cpp:411` —
>    `ggml_soft_max_ext` requires a non-NULL mask leaf when `max_bias
>    > 0` (`ggml.c:4012`); ALiBi adds the per-head slope to the
>    `-|i - j|` mask values.
> 4. **WordPiece tokenizer loader falls back to `bos_token_id` /
>    `eos_token_id`** when `cls_token_id` / `mask_token_id` are
>    absent. Nomic GGUF omits the cls/mask metadata; BERT convention
>    is `cls = bos = 101`, `sep = eos = 102`, so the fallback yields
>    the correct token IDs. The spec's "Zero changes" note for the
>    tokenizer was incomplete.
>
> Each correction is now reflected in the relevant section below
> (Goal, Phase 0 findings tables, Per-arch dispatch, Point D softmax,
> Point F FFN, getRopeModeForArchitecture row, Tokenizer section,
> failure-diagnosis notes).

## 0. Goal & non-goals

**Goal.** Extend `EncoderInference` to support two new BERT-family encoder
architectures by adding the structural variants needed for each:

- **`nomic-embed-text-v1.5`** — `nomic-bert` arch, 137M params, 768-dim,
  2048 ctx, mean pooling. **Fused QKV**, **no biases anywhere**, **SwiGLU
  FFN**, **RoPE** (NEOX mode, `freq_base = 1000` from GGUF). First
  encoder in fleet with rotary positional embeddings AND with
  causal-LM-style tensor structure.
- **`jina-embeddings-v2-base-en`** — `jina-bert-v2` arch, 137M params,
  768-dim, 8192 ctx, mean pooling. Split QKV with biases, **mixed-bias
  GeGLU FFN** (`gelu(gate) * up`; gate + up no bias; down has bias),
  **ALiBi** attention (default `max_bias = 8.0` — GGUF mirror omits
  the metadata key). First encoder in fleet with ALiBi attention bias.

Both register and validate under one design / phased implementation, with
**jina shipping first** in Phase 2 (smaller delta from BERT — split QKV
and biases match) and **nomic shipping second** in a new Phase 2b (fused
QKV + bias-less is the bigger structural delta).

**Non-goals.**

- Matryoshka dimension truncation (defer; surfaceable later as
  `embed(text, { dim?: number })`).
- Chinese coverage (`jina-embeddings-v2-base-zh` deferred).
- Causal-LM-derived embedders (Bucket C, untouched).
- Continuous numerical-parity test fixtures (one-time integration probes
  only; no checked-in continuous-CI parity gate).
- `nomic-bert-moe`, `jina-bert-v3`, `modern-bert`, `neo-bert`, `eurobert`,
  `gemma-embedding` (other BERT-family arches in upstream llama.cpp).
  Each has its own structural surprises (MoE FFN, different RoPE
  orientation, etc.) and lands only on a deployment ask.

**Scope ceiling.** All changes are additive and reversible. Existing
`bert` path bit-identical post-change.

## 1. Phase 0 findings (canonical reference)

From `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt` plus
the secondary tensor-list dump:

### Tensor structure per arch

| | bert (existing) | nomic-bert | jina-bert-v2 |
|---|---|---|---|
| Top-level | `token_embd.weight`, `token_embd_norm.{weight,bias}`, `token_types.weight`, `position_embd.weight` | `token_embd.weight`, `token_embd_norm.{weight,bias}`, `token_types.weight` (no `position_embd`) | `token_embd.weight`, `token_embd_norm.{weight,bias}`, `token_types.weight` (no `position_embd`) |
| Per-block QKV | `attn_q.weight`, `attn_q.bias`, `attn_k.weight`, `attn_k.bias`, `attn_v.weight`, `attn_v.bias` | `attn_qkv.weight` (fused, **no bias**) | `attn_q.weight`, `attn_q.bias`, `attn_k.weight`, `attn_k.bias`, `attn_v.weight`, `attn_v.bias` |
| Per-block O | `attn_output.{weight,bias}` | `attn_output.weight` (**no bias**) | `attn_output.{weight,bias}` |
| Post-attn LN | `attn_output_norm.{weight,bias}` | `attn_output_norm.{weight,bias}` | `attn_output_norm.{weight,bias}` |
| Per-block FFN | `ffn_up.{weight,bias}`, `ffn_down.{weight,bias}` (GeLU two-layer) | `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` (**no biases**, **SwiGLU**) | `ffn_gate.weight`, `ffn_up.weight` (no bias), `ffn_down.{weight,bias}` (**GeGLU**, mixed bias) |
| Post-FFN LN | `layer_output_norm.{weight,bias}` | `layer_output_norm.{weight,bias}` | `layer_output_norm.{weight,bias}` |

### Metadata findings

| | bert | nomic-bert | jina-bert-v2 |
|---|---|---|---|
| `general.architecture` | `bert` | `nomic-bert` | `jina-bert-v2` |
| `<arch>.attention.layer_norm_epsilon` | 1e-12 | 1e-12 | 1e-12 |
| `<arch>.attention.causal` | false | false | false |
| `<arch>.pooling_type` | 1 (MEAN) or 2 (CLS) | 1 (MEAN) | 1 (MEAN) |
| `<arch>.context_length` | varies | 2048 | 8192 |
| `<arch>.head_count` | 12 | 12 | 12 |
| `<arch>.embedding_length` | 384/768 | 768 | 768 |
| RoPE `freq_base` | — | `nomic-bert.rope.freq_base = 1000` | — |
| ALiBi `bias_max` | — | — | **absent** — fall back to 8.0 |
| `position_embd.weight` | present | absent | absent |
| `tokenizer.ggml.model` | bert (WordPiece) | bert (WordPiece) | bert (WordPiece) |

### Forward-pass capability matrix

| Axis | bert | nomic-bert | jina-bert-v2 |
|---|---|---|---|
| Position encoding | learned absolute (add to embedding) | RoPE (Q + K, after reshape) | ALiBi (softmax `max_bias`) |
| QKV layout | split projections, each + bias | fused matmul + slice via `view_3d` | split projections, each + bias |
| Attention output bias | yes | no | yes |
| FFN type | GeLU two-layer | SwiGLU (`silu(gate) * up`) | GeGLU (`gelu(gate) * up`) |
| FFN biases | up + down | none | down only |
| LayerNorm structure | post-norm (residual then LN) | post-norm | post-norm |
| Pooling | CLS or MEAN | MEAN | MEAN |
| WordPiece tokenizer | yes | yes | yes |

## 2. Components & files affected

| File | Change | LOC est. |
|---|---|---|
| `src/core/types.ts` | Add `"nomic-bert"`, `"jina-bert-v2"` to `ModelArchitecture` union; add `alibiMaxBias?: number`; export `ENCODER_ARCHITECTURES` + `isEncoderArchitecture()` helper. | +5 |
| `src/models/model-loader.ts` | `extractHyperparams`: broaden `arch === "bert"` branches to `isEncoderArchitecture(arch)`; read jina ALiBi metadata (with 8.0 fallback). | +20 |
| `src/inference/encoder-inference.ts` | Drop `bert` hard-assert. Make all bias and gate fields nullable in `EncoderLayerWeights`. Add `qkvFused` field. Add `makeTensorOptional` for absent-tolerant lookup. Branch `loadWeights` per arch. Branch `buildGraph` at six load-bearing points (pre-layer pos-embedding, fused-vs-split QKV, RoPE, attn-output bias, softmax max_bias, FFN type/bias). | +180 |
| `src/inference/model-inference.ts` | `getRopeModeForArchitecture`: `nomic-bert → NEOX` (per llama.cpp `llama-model.cpp:9266`). | +3 |
| `src/core/engine.ts` | `engine.ts:589` routing → `isEncoderArchitecture(arch)`. | +5 |
| `eval/models.ts` | Two new entries (nomic + jina). | +30 |
| `eval/smoke-profiles.ts` | Two new profile rows + selector array additions. | +20 |
| `eval/encoder-parity.ts` | New parity harness (browser-driven via agentchrome). | +110 |
| `tests/encoder-inference.test.ts` | New tests: helper truth-table, loader routing per arch, optional-tensor loader, buildGraph dispatch per arch (now covers fused-QKV, SwiGLU, bias-presence, RoPE/ALiBi). | +120 |

**No-touch but verified-current:** `src/inference/ggml-wasm.ts` (already
exposes `opRope`, `opSoftMaxExt`, `opMulMat`, `opMul`, `opSilu`,
`opView3d`), `src/inference/tokenizer.ts` (WordPiece works for all three
— `tokenizer.ggml.model = bert` for both new arches), `eval/embed-perf.ts`
(generic over registered embed-profile entries).

**Total estimate:** ~490 LOC across 8 modified + 1 new file (vs original
~200 LOC). The increase is concentrated in `encoder-inference.ts` (+180
vs +60) and tests (+120 vs ~30) — both linear in the number of arch
branches, not architectural complexity.

**New helper in `types.ts`** (unchanged from v1):

```ts
export const ENCODER_ARCHITECTURES = ["bert", "nomic-bert", "jina-bert-v2"] as const;
export function isEncoderArchitecture(a: ModelArchitecture): boolean {
  return (ENCODER_ARCHITECTURES as readonly string[]).includes(a);
}
```

## 3. Forward-pass deltas

The current `buildGraph` (`encoder-inference.ts:172-240`) has six
load-bearing arch-sensitive points after this revision (vs three in v1).

### `EncoderLayerWeights` — nullable shape

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

  attnNormW: TensorPtr;                // all
  attnNormB: TensorPtr;                // all (post-attn LN)

  // FFN — gate is non-null for SwiGLU (nomic) and GeGLU (jina), null
  // for plain GeLU two-layer (bert):
  ffnGate: TensorPtr | null;           // nomic (SwiGLU) + jina-bert-v2 (GeGLU)
  ffnUp: TensorPtr;                    // all
  ffnUpBias: TensorPtr | null;         // bert only
  ffnDown: TensorPtr;                  // all
  ffnDownBias: TensorPtr | null;       // bert + jina-bert-v2 (not nomic)

  ffnNormW: TensorPtr;                 // all
  ffnNormB: TensorPtr;                 // all
}
```

### `EncoderWeights` (top-level)

```ts
interface EncoderWeights {
  tokEmb: TensorPtr;                   // all
  positionEmb: TensorPtr | null;       // bert only
  tokenTypes: TensorPtr;               // all (segment 0 hardcoded)
  inputNormW: TensorPtr;               // all
  inputNormB: TensorPtr;               // all
  layers: EncoderLayerWeights[];
}
```

### Per-arch dispatch (added pre-loop)

```ts
const arch = hp.architecture;
const usesPosEmbedding = arch === "bert";
const usesRope = arch === "nomic-bert";
const alibiMaxBias = arch === "jina-bert-v2" ? (hp.alibiMaxBias ?? 8.0) : 0.0;
const ropeMode = usesRope ? getRopeModeForArchitecture(arch) : 0;
```

### Point A — pre-layer positional addition

```ts
let x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
if (usesPosEmbedding) {
  if (!weights.positionEmb) throw new Error("bert path requires positionEmb");
  x = wasm.opAdd(x, wasm.opGetRows(weights.positionEmb, posTensor));
}
x = wasm.opAdd(x, wasm.opGetRows(weights.tokenTypes, segTensor));
x = this.layerNorm(x, weights.inputNormW, weights.inputNormB);
```

### Point B — Q/K/V production (split or fused)

Mirrors llama.cpp's `build_qkv()` (`src/llama-graph.cpp:1064-1138`).

```ts
let q3: TensorPtr;          // shape: [headDim, nHeads, nTokens]
let k3: TensorPtr;
let v3: TensorPtr;

if (lw.qkvFused) {
  // Fused: one matmul produces [3*E, nTokens]; slice via opView3d into
  // three [headDim, nHeads, nTokens] sub-views. Element size is 4 (the
  // matmul output is fp32 regardless of weight dtype).
  const qkv = wasm.opMulMat(lw.qkvFused, x);   // [3*E, nTokens]
  const elemSize = 4;
  const headBytes = elemSize * headDim;
  const tokenBytes = elemSize * 3 * E;
  // q starts at byte 0, k at byte E*4, v at byte 2*E*4 within each token.
  q3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, 0);
  k3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, elemSize * E);
  v3 = wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, elemSize * 2 * E);
} else {
  // Split: three separate matmuls + biases (BERT/Jina pattern).
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
```

**Note on the fused-QKV view layout:** ggml stores tensors with `nb[]`
strides; `view_3d`'s `nb1` is the byte stride between rows of `ne0`, and
`nb2` is the byte stride between rows of `ne1`. For the fused matmul
output of shape `[3*E, nTokens]` (fp32 contiguous), the per-element row
size of an `[headDim, nHeads, nTokens]` view is `4 * headDim` (nb1) and
the per-token stride is `4 * 3 * E` (nb2). The starting byte offsets
(`0`, `4*E`, `4*2*E`) place the three views over the consecutive Q, K,
V chunks within each token. **This is `nHeads == nKvHeads` only;** if a
fused encoder ever has GQA with `nHeads != nKvHeads`, the K/V slicing
must use the smaller `n_embd_kv` for nb2 — flagged but not in scope.

### Point C — RoPE (nomic-bert only)

```ts
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

`hp.ropeFreqBase` for nomic comes from GGUF metadata directly (= 1000;
the loader's existing fallback chain at `model-loader.ts:95-98` already
reads `${arch}.rope.freq_base`). `ropeMode = NEOX` per
`getRopeModeForArchitecture("nomic-bert")` — matches llama.cpp's
`LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_NOMIC_BERT*`
(`~/Repos/llama.cpp/src/llama-model.cpp:9266`).

### Point D — softmax (with optional ALiBi)

```ts
const qp = wasm.opPermute(q3, 0, 2, 1, 3);
const kp = wasm.opPermute(k3, 0, 2, 1, 3);
const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

const qk = wasm.opMulMat(kp, qp);
// Mask leaf is REQUIRED for jina (max_bias > 0) and OPTIONAL (NULL)
// for bert/nomic. When required, populate with -|i - j| per row/col
// per llama.cpp `llama-graph.cpp:411`; ggml then adds the per-head
// ALiBi slope to those values during softmax.
const maskLeaf = alibiMaxBias > 0 ? this.alibiMaskLeaf(nTokens) : 0;
const aw = wasm.opSoftMaxExt(qk, maskLeaf, invSqrtHd, alibiMaxBias);
```

`alibiMaxBias` is `0.0` for bert/nomic and `hp.alibiMaxBias ?? 8.0` for
jina. ggml's softmax derives per-head linear slopes internally when
`max_bias > 0` (slopes `2^(-8/n_head * h)`), but **a non-NULL mask
tensor is still required** (`ggml.c:4012` asserts `mask != NULL` when
`max_bias > 0`). For an encoder with full bidirectional attention the
mask values are simply the negative-distance pattern `-|i - j|`; the
slope-times-distance addition produces the per-head ALiBi bias. For
bert/nomic (no ALiBi) the mask leaf is omitted (NULL); jina populates
it once at graph build via `alibiMaskLeaf(nTokens)`.

### Point E — attention output (with optional bias)

```ts
const out = wasm.opMulMat(vp, aw);
const merged = wasm.opReshape2d(
  wasm.opCont(wasm.opPermute(out, 0, 2, 1, 3)),
  E, nTokens,
);
let attnProj = wasm.opMulMat(lw.oProj, merged);
if (lw.oBias) attnProj = wasm.opAdd(attnProj, lw.oBias);

x = this.layerNorm(wasm.opAdd(x, attnProj), lw.attnNormW, lw.attnNormB);
```

### Point F — FFN (GeLU two-layer, SwiGLU, or GeGLU)

The gated branch picks the gate activation per-arch — **silu for nomic
(SwiGLU)** and **gelu for jina (GeGLU)** — matching llama.cpp
`bert.cpp:122-130` (jina/jina-bert-v2 uses `ggml_gelu` on the gate
projection):

```ts
const gateAct: "silu" | "gelu" =
  arch === "jina-bert-v2" ? "gelu" : "silu";

let ffnOut: TensorPtr;
if (lw.ffnGate) {
  // SwiGLU (nomic): silu(gate(x)) * up(x).
  // GeGLU  (jina) : gelu(gate(x)) * up(x).
  const gate = wasm.opMulMat(lw.ffnGate, x);
  let up = wasm.opMulMat(lw.ffnUp, x);
  if (lw.ffnUpBias) up = wasm.opAdd(up, lw.ffnUpBias);   // bert never enters this branch
  const activated = gateAct === "gelu" ? wasm.opGelu(gate) : wasm.opSilu(gate);
  const mid = wasm.opMul(activated, up);
  ffnOut = wasm.opMulMat(lw.ffnDown, mid);
  if (lw.ffnDownBias) ffnOut = wasm.opAdd(ffnOut, lw.ffnDownBias);  // jina yes, nomic no
} else {
  // GeLU two-layer (current bert path).
  let h = wasm.opMulMat(lw.ffnUp, x);
  if (lw.ffnUpBias) h = wasm.opAdd(h, lw.ffnUpBias);
  h = wasm.opGelu(h);
  ffnOut = wasm.opMulMat(lw.ffnDown, h);
  if (lw.ffnDownBias) ffnOut = wasm.opAdd(ffnOut, lw.ffnDownBias);
}

x = this.layerNorm(wasm.opAdd(x, ffnOut), lw.ffnNormW, lw.ffnNormB);
```

The bert path is bit-identical — for bert, `ffnGate === null`, both
biases are non-null, and the `else` branch executes exactly as before.

## 4. Loader & metadata

### `extractHyperparams` deltas (unchanged from v1)

```ts
import { isEncoderArchitecture } from "../core/types.js";

// existing arch lookup unchanged
const arch = getMetaString(ctx, "general.architecture", "llama") as
  ModelHyperparams["architecture"];

const normEpsilon = isEncoderArchitecture(arch)
  ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
  : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);

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

`ropeFreqBase` already reads `${arch}.rope_freq_base` / `${arch}.rope.freq_base`
(model-loader.ts:95-98) — works for `nomic-bert.rope.freq_base = 1000`
without modification.

### `EncoderInference.loadWeights` — arch-branched tensor lookup

The existing `makeTensor` throws on absent weights. Add a sibling
`makeTensorOptional` that returns `null` instead:

```ts
private makeTensorOptional(
  tensorMap: Map<string, GgufTensorInfo>,
  name: string,
): TensorPtr | null {
  return tensorMap.has(name) ? this.makeTensor(tensorMap, name) : null;
}
```

Per-arch loadWeights body becomes (replacing current lines 85-122):

```ts
const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
const positionEmb = this.hp.architecture === "bert"
  ? this.makeTensor(tensorMap, "position_embd.weight")
  : null;
const tokenTypes = this.makeTensor(tensorMap, "token_types.weight");
const inputNormW = this.makeTensor(tensorMap, "token_embd_norm.weight");
const inputNormB = this.makeTensor(tensorMap, "token_embd_norm.bias");

const layers: EncoderLayerWeights[] = [];
for (let i = 0; i < hp.layerCount; i++) {
  const p = (s: string) => `blk.${i}.${s}`;
  const arch = hp.architecture;
  const isFused = arch === "nomic-bert";

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

The `isFused` branching is the *only* arch-specific switch in the loader
beyond `position_embd.weight` and the optional-bias gating. Everything
else degrades cleanly via `makeTensorOptional`'s null return on absent
tensors.

### Tokenizer

`buildTokenizerConfig` already routes `tokenizer.ggml.model == "bert"` →
WordPiece. Both nomic and jina v2 export `tokenizer.ggml.model = "bert"`.

**Cls/mask token fallback (added in Phase 4).** Nomic GGUF omits
`tokenizer.ggml.cls_token_id` and `tokenizer.ggml.mask_token_id` (jina
and the existing arctic/bge entries do export them). The WordPiece
loader now falls back to `bos_token_id` / `eos_token_id` when the cls/
mask keys are absent — BERT convention is `cls = bos = 101` and
`sep = eos = 102`, so the fallback yields the correct token IDs for
nomic and is a no-op for any GGUF that exports the canonical keys.

(Jina v2 also sets `tokenizer.ggml.pre = "jina-v2-en"`, a BPE pre-tokenizer
hint. WordPiece tokenization ignores BPE pre-tokenizer settings; the field
is informational.)

## 5. Validation strategy

Four gates per model, each blocking the next. Same shape as v1; gate
thresholds unchanged.

### Gate 1 — Parsing & loading (no metric)

Both models must:
1. Parse the GGUF without errors.
2. Stream into the WASM heap and pass `backendAllocCtxTensors`.
3. Run a single `engine.embed(handleId, "Hello world.")` call without
   throwing and return a vector of the expected dim (768 each).

Fail mode: stops the work; no metric collected.

### Gate 2 — Numerical-parity probe (per model, ≥0.999 cosine)

Fixture pinned at `eval/reports/encoder-parity-2026-04-28/`:

```
inputs.json        # 5 fixed strings (committed in 43df996)
nomic-ref.json     # 5 reference embeddings from sentence-transformers (768-dim each, F32)
jina-ref.json      # 5 reference embeddings from sentence-transformers (768-dim each, F32)
capture-refs.py    # Reference-capture script (one-shot)
SUMMARY.md         # cosine sim per input × model, pass/fail per row, methodology
```

WebLLM-side capture: bun harness `eval/encoder-parity.ts` drives the
browser smoke page via `agentchrome`, calls
`window.engine.embed(window.handleId, text)` for each fixture input,
computes cosine vs the pinned reference per row. **Per-row gate: cosine
≥ 0.999.** Fail diagnoses (most-likely first):

For **nomic** specifically: fused-QKV view offsets, RoPE mode (must be
NEOX — `LLAMA_ROPE_TYPE_NEOX` per `llama-model.cpp:9266`), `freq_base`
(1000 vs 10000 — confirm GGUF reads correctly), SwiGLU vs GeLU FFN
type, missing-bias add path inadvertently firing, missing cls/mask
token fallback to bos/eos.

For **jina** specifically: ALiBi `max_bias` value (8.0 default),
ALiBi mask must be populated with `-|i - j|` (NULL mask trips
`ggml.c:4012` assertion), GeGLU gate activation (jina uses `gelu`,
not `silu` — divergent from nomic's SwiGLU; `bert.cpp:122-130`),
`ffn_down.bias` add path.

### Gate 3 — Cosine-task eval (existing 8-task suite)

After parity probes pass, both models run through `make bench-full`.
Apples-to-apples comparison rows added to the dashboard's Embeddings
section. **No hard accuracy gate** — the cosine-task suite is
informational at this stage.

### Gate 4 — Browser smoke (end-to-end)

Standard `make smoke-serve` + agentchrome flow with each new model
selected. Verifies the engine routing change at `engine.ts:589`
(the `isEncoderArchitecture` broadening) survived end-to-end.

### "Done" definition

- Both models registered in `eval/models.ts` and `eval/smoke-profiles.ts`.
- Both pass Gates 1-4.
- Parity probe artifacts committed under
  `eval/reports/encoder-parity-2026-04-28/`.
- Dashboard Embeddings section shows 6 rows.
- TODO.md "Bucket B" entry transitions from "queued" to "DONE 2026-MM-DD".

## 6. Testing, error handling, rollout & risks

### Unit tests (`tests/encoder-inference.test.ts` extension)

Five new test groups, each focused and isolated:

1. **`isEncoderArchitecture()`** — table-driven truth-table.
2. **Loader metadata routing** — `nomic-bert` (RoPE freq_base = 1000),
   `jina-bert-v2` (alibiMaxBias = 8.0 fallback when key absent), bert
   unchanged.
3. **`makeTensorOptional`** — returns null on absent tensor; populates
   correctly when present.
4. **`buildGraph` arch dispatch** — for each arch, assert the recorded
   `(opGetRows, opRope, opMulMat, opMul, opSilu, opGelu, opSoftMaxExt
   max_bias)` call sequence matches the expected per-arch shape. Three
   sub-tests (one per arch). Mock `GgmlWasm` records all op calls.
5. **`loadWeights` arch dispatch** — given a synthesized GGUF tensor map,
   verify that bert loads `attn_q.weight/bias` and rejects `attn_qkv.weight`,
   nomic loads `attn_qkv.weight` (SwiGLU `ffn_gate.weight`) and skips all
   biases, jina loads split QKV with biases and GeGLU `ffn_gate.weight`
   with no `ffn_up.bias` but with `ffn_down.bias`. Three sub-tests.

`EncoderInference.poolAndNormalize` and `layerNorm` are reused unchanged.
Integration / smoke is covered by Gate 4.

### Error handling (boundary policy)

- Unknown encoder arch in GGUF → constructor throws with supported list.
- Missing required weight (e.g. `attn_qkv.weight` for nomic) → `makeTensor`
  throws as before; `makeTensorOptional` returns null instead.
- `buildGraph` reaches a path requiring a null tensor → throws with a
  diagnostic message naming the missing weight (e.g. `"split-QKV path
  requires qProj/kProj/vProj for ${arch}"`).
- Missing required RoPE / ALiBi metadata → log a warning and use the
  documented fallback (10000 / 8.0). Gate 2 catches semantic bugs.

No defensive code beyond the boundary.

### Rollout phasing (revised — 6 commits within plan, jina first)

- **Phase 0 — Probe** (commit `43df996`, already landed): GGUF discovery
  artifacts.
- **Phase 1 — Types + loader + RoPE helper** (commit 1 of 5): arch enum
  extension, `isEncoderArchitecture`, `extractHyperparams` deltas,
  `getRopeModeForArchitecture` extension. Tests #1, #2.
- **Phase 2a — Encoder forward (jina-bert-v2 first) + engine routing**
  (commit 2 of 5): `EncoderInference` broadens to encoder-arch
  isEncoder check, nullable bias fields, `makeTensorOptional`,
  buildGraph branches Points A/D/E/F (and split-QKV in Point B).
  Tests #3, #4 (jina + bert sub-tests), #5 (bert + jina). Engine
  routing change. **bert path bit-identical.** Excludes nomic-specific
  fused-QKV slicing and RoPE (Point B fused branch + Point C).
- **Phase 2b — Encoder forward (nomic-bert)** (commit 3 of 5): adds
  Point B fused-QKV slicing + Point C RoPE. Tests #4 (nomic sub-test),
  #5 (nomic sub-test).
- **Phase 3a — Reference-vector capture** (out-of-band, separate commit):
  one-shot Python via `uv run --no-project --with-requirements`.
- **Phase 3 — Parity harness + jina registration** (commit 4 of 5):
  `eval/encoder-parity.ts`, `eval/models.ts` jina entry, parity gate.
- **Phase 4 — Nomic registration + parity gate** (commit 5 of 5):
  same shape, jina-experience-applied.
- **Phase 5 — Bench-full + dashboard refresh + TODO close** (commit 6
  of 5 — NB: revised count): `make bench-full`, dashboard screenshot,
  TODO transition.

Note: numbering inverted vs v1 (jina-first / nomic-second instead of
nomic-first / jina-second) because jina shares more with bert and is
the better incremental step. Splitting Phase 2 into 2a + 2b also lets
the bert path's bit-identity be verified before the nomic-only fused
QKV machinery lands.

### Risk register (revised)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Fused-QKV `view_3d` offsets miscomputed | High | Phase 2b parity probe catches; cross-reference against `llama-graph.cpp:1088-1095` exactly. The byte-offset arithmetic is ggml-specific and has bitten upstream contributors. |
| Nomic SwiGLU output diverges from reference | Medium | Parity probe catches; first diagnosis is `silu(gate) * up` order vs `up * silu(gate)` (commutative — same); second is `ffn_gate` and `ffn_up` swapped (asymmetric — produces ~0.5 cosine collapse). |
| Jina ALiBi sign inverted | Low | Gate 2 catches; if true escalates to user (ggml-side ALiBi sign convention is settled but worth verifying). |
| `f_clamp_kqv` (referenced in `build_qkv`) needed | Low | Phase 0 didn't surface a `clamp_kqv` metadata key for either arch; only Falcon-style models use it. We omit the clamp branch entirely. |
| `attn_q_norm` / `attn_k_norm` (referenced at `bert.cpp:44-58`) | Low | Phase 0 tensor list shows neither arch has these. We omit the branch. |
| `attn_norm_2` (referenced at `bert.cpp:91`) | Low | Not present in either arch. Omit. |
| RoPE `freq_base = 1000` instead of 10000 | Resolved | Phase 0 confirmed; loader reads from GGUF correctly. |
| `position_embd.weight` present-but-unused in one of the arches | Resolved | Phase 0 confirmed absent in both. |
| `attn_output.bias` absence breaks bert | Low | bert always has it; `makeTensorOptional` returns null only when truly absent. Bert tests verify this. |
| Existing BERT path bit-non-identical | Medium | Each phase commit runs `make checkall` AND a BGE/Arctic-Embed bench-full row check; if BGE row drifts beyond ±2%, revert and investigate. |

**Reversibility:** any single phase can be reverted via `git revert`
without affecting others.

## 7. Open questions / decisions

**None at the design level.** Phase 0 resolved all spec-tagged unknowns.
The remaining unknowns (fused-QKV byte offsets matching ggml exactly,
SwiGLU operand order, ALiBi sign) are caught by the Gate 2 parity probe
— first failure becomes a one-line fix + re-run.

## 8. Cross-references

- TODO.md → "Embedding-model expansion candidates — Bucket B".
- `docs/superpowers/specs/2026-04-24-encoder-forward-pass-design.md` —
  original BERT-encoder bring-up; this spec is a delta on it.
- `docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md` —
  matching implementation plan.
- `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt` —
  Phase 0 probe artifact (commit `43df996`).
- `~/Repos/llama.cpp/src/models/bert.cpp`,
  `~/Repos/llama.cpp/src/llama-graph.cpp:1064-1138` — canonical
  reference for the encoder forward pass across BERT-family arches.
- CLAUDE.md → "Workflow policies (set 2026-04-28)" — probe-first,
  always-commit-before-work, complexity ≠ time.
- Original (deprecated) v1 spec at `8064f80:docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`.
