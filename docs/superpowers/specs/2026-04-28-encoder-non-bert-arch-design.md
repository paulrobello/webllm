# Encoder non-BERT architecture support — design

**Date:** 2026-04-28
**Status:** Draft, awaiting user spec review.
**Tracking:** TODO.md "Embedding-model expansion candidates — Bucket B".
**Cycle precedent:** §17/§18/§19/§20 phased plan structure.

## 0. Goal & non-goals

**Goal.** Extend `EncoderInference` to support two new BERT-family encoder
architectures by adding RoPE and ALiBi positional-encoding paths to the
existing forward graph:

- **`nomic-embed-text-v1.5`** — BERT + RoPE, 137M params, 768-dim, 8192 ctx,
  mean pooling. First encoder in fleet with rotary positional embeddings.
- **`jina-embeddings-v2-base-en`** — BERT + ALiBi, 137M params, 768-dim,
  8192 ctx, mean pooling. First encoder in fleet with ALiBi attention bias.

Both models register and validate under one design / one PR cycle, with
phased implementation (nomic first since RoPE is the harder change; jina is
a trivial follow-on once RoPE lands and the encoder dispatch broadening is
in place).

**Non-goals.**

- Matryoshka dimension truncation (defer; surfaceable later as
  `embed(text, { dim?: number })`).
- Chinese coverage (`jina-embeddings-v2-base-zh` deferred).
- Causal-LM-derived embedders (Bucket C, untouched).
- Continuous numerical-parity test fixtures (one-time integration probes
  only; no checked-in continuous-CI parity gate).

**Scope ceiling.** All changes are additive and reversible. Existing BERT
path bit-identical post-change (gated by `arch === "bert"` branch in
`buildGraph`).

## 1. Key insight

`opRope` and `opSoftMaxExt(qk, mask, scale, max_bias)` are **already
available** in the WASM bindings. ALiBi is essentially free — `max_bias > 0`
is an existing argument to the existing softmax call, and ggml computes
per-head linear bias internally from this scalar. RoPE adds two `opRope`
calls (Q and K) per layer, paired with dropping the position-embedding
sum at input. Net code surface: ~140 LOC across 7 modified files. No new
ops, no new modules, no WASM rebuild required.

## 2. Components & files affected

| File | Change | LOC est. |
|---|---|---|
| `src/core/types.ts` | Add `"nomic-bert"`, `"jina-bert-v2"` to `ModelArchitecture` union; add `alibiMaxBias?: number` to `ModelHyperparams`; export `ENCODER_ARCHITECTURES` + `isEncoderArchitecture()` helper. | +5 |
| `src/models/model-loader.ts` | `extractHyperparams`: broaden the existing pooling/causal/normEpsilon branches from `arch === "bert"` to `isEncoderArchitecture(arch)`; read jina ALiBi metadata. | +20 |
| `src/inference/encoder-inference.ts` | Drop `arch === "bert"` hard-assert; `loadWeights`: conditional `position_embd.weight` lookup; `buildGraph`: arch-branched positional handling at three points (pre-layer pos-embedding sum, per-layer Q/K, softmax). | +60 |
| `src/inference/model-inference.ts` | `getRopeModeForArchitecture`: add `nomic-bert` → `RopeMode.NORMAL` (sentence-transformers convention; verify in Phase 0). | +3 |
| `src/core/engine.ts` (line 589) | Replace `architecture === "bert"` check with `isEncoderArchitecture(...)`. | +5 |
| `eval/models.ts` | Two new entries: `nomic-embed-text-v1.5` and `jina-embeddings-v2-base-en` with pinned `ggufFilePattern`. | +25 |
| `eval/smoke-profiles.ts` | Two new profile rows mirroring BGE-base shape. | +20 |
| `eval/encoder-parity.ts` | New harness for the per-model numerical-parity probe (see §5). | +60 |

**No-touch but verified-current:** `src/inference/ggml-wasm.ts` (`opRope` +
`opSoftMaxExt` already exposed), `src/inference/tokenizer.ts` (WordPiece
works for both — both inherit BERT-base tokenizer with the same
`[CLS]`/`[SEP]` framing), `eval/embed-perf.ts` (generic over registered
embed-profile entries).

**Total estimate:** ~200 LOC across 7 modified + 1 new file.

**New helper in `types.ts`:**

```ts
export const ENCODER_ARCHITECTURES = ["bert", "nomic-bert", "jina-bert-v2"] as const;
export function isEncoderArchitecture(a: ModelArchitecture): boolean {
  return (ENCODER_ARCHITECTURES as readonly string[]).includes(a);
}
```

## 3. Forward-pass deltas

The current `buildGraph` (`encoder-inference.ts:172-240`) has three
load-bearing arch-sensitive points.

### Point A — pre-layer positional addition (line 187)

```ts
let x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
if (usesPosEmbedding) {                                       // bert only
  x = wasm.opAdd(x, wasm.opGetRows(weights.positionEmb, posTensor));
}
x = wasm.opAdd(x, wasm.opGetRows(weights.tokenTypes, segTensor));
x = this.layerNorm(x, weights.inputNormW, weights.inputNormB);
```

`token_types` and the segment-zero leaf stay for **all three arches**: both
nomic and jina inherit BERT's segment-embedding table (single segment in
practice; ggml expects the weight tensor present).

### Point B — Q/K projections inside the per-layer loop (lines 201-210)

```ts
const q = wasm.opAdd(wasm.opMulMat(lw.qProj, x), lw.qBias);
const k = wasm.opAdd(wasm.opMulMat(lw.kProj, x), lw.kBias);
const v = wasm.opAdd(wasm.opMulMat(lw.vProj, x), lw.vBias);

let q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
let k3 = wasm.opReshape3d(k, headDim, nHeads, nTokens);
const v3 = wasm.opReshape3d(v, headDim, nHeads, nTokens);

if (usesRope) {                                               // nomic-bert only
  q3 = wasm.opRope(q3, posTensor, headDim, ropeMode, hp.contextLength,
                   hp.ropeFreqBase, hp.ropeScale, /* attn_factor */ 1.0,
                   /* beta_fast */ 32.0, /* beta_slow */ 1.0);
  k3 = wasm.opRope(k3, posTensor, headDim, ropeMode, hp.contextLength,
                   hp.ropeFreqBase, hp.ropeScale, 1.0, 32.0, 1.0);
}

const qp = wasm.opPermute(q3, 0, 2, 1, 3);
const kp = wasm.opPermute(k3, 0, 2, 1, 3);
```

Exact `opRope` arity will be matched against the existing causal-LM call
sites in `model-inference.ts:483-503` during implementation.

### Point C — softmax over qk (line 218)

```ts
const aw = wasm.opSoftMaxExt(qk, 0, invSqrtHd, alibiMaxBias);
```

ggml's softmax computes per-head linear bias internally when
`max_bias > 0`. Standard ALiBi slopes (`2^(-8/n_head * h)`) are derived
from `max_bias` automatically — no extra weight tensor needed.

### Per-arch dispatch (added pre-loop)

```ts
const arch = hp.architecture;
const usesPosEmbedding = arch === "bert";
const usesRope = arch === "nomic-bert";
const alibiMaxBias = arch === "jina-bert-v2" ? (hp.alibiMaxBias ?? 8.0) : 0.0;
const ropeMode = usesRope ? getRopeModeForArchitecture(arch) : 0;
```

### `EncoderWeights` interface (line 29) becomes

```ts
interface EncoderWeights {
  tokEmb: TensorPtr;
  positionEmb: TensorPtr | null;   // null for nomic-bert / jina-bert-v2
  tokenTypes: TensorPtr;
  inputNormW: TensorPtr;
  inputNormB: TensorPtr;
  layers: EncoderLayerWeights[];
}
```

`loadWeights` guards the `makeTensor("position_embd.weight")` call site
(currently throws on absence) with `if (usesPosEmbedding)`.

**Total inside `buildGraph`:** ~25 added lines, ~3 modified lines.
Everything else (FFN, post-norm, residuals, pooling) is identical across
all three arches.

## 4. Loader & metadata

`ModelLoader.extractHyperparams` reads metadata under `${arch}.*`
namespacing. With the arch enum extended, the existing pattern works
directly. Verified-by-upstream-source keys (from
`llama.cpp/convert_hf_to_gguf.py` for nomic and jina v2):

| Field | bert | nomic-bert | jina-bert-v2 |
|---|---|---|---|
| `embedding_length` | `bert.embedding_length` | `nomic-bert.embedding_length` | `jina-bert-v2.embedding_length` |
| `block_count` | `bert.block_count` | `nomic-bert.block_count` | `jina-bert-v2.block_count` |
| `attention.head_count` | `bert.attention.head_count` | `nomic-bert.attention.head_count` | `jina-bert-v2.attention.head_count` |
| `attention.layer_norm_epsilon` | `bert.attention.layer_norm_epsilon` | `nomic-bert.attention.layer_norm_epsilon` | `jina-bert-v2.attention.layer_norm_epsilon` |
| `feed_forward_length` | `bert.feed_forward_length` | `nomic-bert.feed_forward_length` | `jina-bert-v2.feed_forward_length` |
| `pooling_type` | `bert.pooling_type` | `nomic-bert.pooling_type` | `jina-bert-v2.pooling_type` |
| `causal_attention` | `bert.attention.causal` | `nomic-bert.attention.causal` | `jina-bert-v2.attention.causal` |
| RoPE freq base | — | `nomic-bert.rope.freq_base` | — |
| ALiBi bias max | — | — | `jina-bert-v2.attention.alibi_bias_max` |

### `extractHyperparams` deltas

```ts
// existing arch lookup unchanged
const arch = getMetaString(ctx, "general.architecture", "llama") as
  ModelHyperparams["architecture"];

// existing normEpsilon branch broadens to "is encoder-style"
const normEpsilon = isEncoderArchitecture(arch)
  ? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
  : getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);

// existing pooling/causal block broadens
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

// ropeFreqBase already reads `${arch}.rope_freq_base` / `${arch}.rope.freq_base`
// (lines 95-98) — works for nomic-bert without modification.
```

**Tokenizer:** `buildTokenizerConfig` already routes
`tokenizer.ggml.model == "bert"` → WordPiece. Both nomic and jina v2
export `tokenizer.ggml.model = "bert"` and use the same `[CLS]`/`[SEP]`
framing as BERT-base. **Zero changes.**

**Risks for Phase 0 to verify:**

- nomic GGUF metadata key for RoPE freq is `nomic-bert.rope.freq_base`
  per upstream convention; if the actual file uses
  `nomic-bert.rope_freq_base` (older format) the existing fallback chain
  at `extractHyperparams` lines 95-98 covers both.
- jina v2's `alibi_bias_max` key spelling is unverified — upstream
  `convert_hf_to_gguf.py` writes `jina-bert-v2.attention.alibi_bias_max`,
  but mirrors may have repacked. The 8.0 fallback is the standard
  `2^(-8/n_head * h)` slope generator value for 12-head models.
- Whether `position_embd.weight` is present-but-unused or absent for
  each model.

All three resolve in Phase 0's GGUF-discovery probe before code lands.

## 5. Validation strategy

Four gates, in order, each blocking the next.

### Gate 1 — Parsing & loading (no metric)

Both models must:

1. Parse the GGUF without errors (loader exits cleanly through the new
   metadata branches).
2. Stream into the WASM heap and pass `backendAllocCtxTensors`.
3. Run a single `engine.embed("Hello world.")` call without throwing
   and return a vector of the expected dim (768 each).

Fail mode: stops the work; no metric collected. Captured as a one-line
probe artifact.

### Gate 2 — Numerical-parity probe (per model, ≥0.999 cosine)

Fixture pinned at `eval/reports/encoder-parity-2026-04-28/`:

```
inputs.json        # 5 fixed strings (short / medium / long / unicode / empty-after-tokenize)
nomic-ref.json     # 5 reference embeddings from sentence-transformers (768-dim each, F32)
jina-ref.json     # 5 reference embeddings from sentence-transformers (768-dim each, F32)
capture-refs.py    # Reference-capture script (one-shot)
SUMMARY.md         # cosine sim per input × model, pass/fail per row, methodology
```

**Reference capture (one-time, off-process):** a short Python script
run in a temp `uv run --no-project` venv invokes `sentence-transformers`
directly (`SentenceTransformer.encode(..., normalize_embeddings=True)`).
Script + venv requirements live alongside the fixture; results are
checked in.

**WebLLM-side capture:** new bun harness `eval/encoder-parity.ts` loads
each model via the public `engine` API, runs the same 5 inputs through
`engine.embed()`, computes cosine vs the pinned reference per row,
writes pass/fail to `SUMMARY.md`.

**Per-row gate:** cosine ≥ 0.999. (BGE shows >0.9999 on this metric in
informal post-hoc comparison; 0.999 is a soft floor for F32/F16
mixed-precision noise.)

Fail mode for any row: implementation is wrong. Diagnoses, in order of
likelihood: RoPE mode (NORMAL vs NEOX), `freq_base` (10000 vs file
value), `headDim`, ALiBi `max_bias` value, or pooling (mean vs cls).
Probe artifact records which row failed at which cosine — direct signal
for the bug.

### Gate 3 — Cosine-task eval (existing 8-task suite)

After parity probes pass, both models run through `make bench-full`.
Apples-to-apples comparison rows added to the dashboard's Embeddings
section:

| Model | dim | params | p50 single-text short | 8-task cosine | New scaling lever |
|---|---:|---:|---:|---:|---|
| arctic-embed-s | 384 | 33M | … | … | (existing) |
| arctic-embed-m | 768 | 109M | … | … | (existing) |
| bge-small | 384 | 33M | 17.0 ms | 91% | (existing) |
| bge-large | 1024 | 335M | 59.3 ms | 89% | (existing) |
| **nomic-embed-text-v1.5** | 768 | 137M | tbd | tbd | **first encoder w/ RoPE** |
| **jina-embeddings-v2-base-en** | 768 | 137M | tbd | tbd | **first encoder w/ ALiBi** |

**No hard accuracy gate** — the cosine-task suite is informational at
this stage (precedent from Bucket A: BGE-large posted 89% and was
accepted). The interesting question is whether either encoder beats
BGE-base equivalents at the same scale, which is signal for future
deployment-ask routing, not pass/fail for landing.

### Gate 4 — Browser smoke (end-to-end)

Standard `make smoke-serve` + agentchrome flow on
`smoke-test/real-model.html` with each new model selected. Verifies the
engine routing change at `engine.ts:589` (the `isEncoderArchitecture`
broadening) survived end-to-end.

### "Done" definition

- Both models registered in `eval/models.ts` and `eval/smoke-profiles.ts`
- Both pass Gates 1-4
- Parity probe artifacts committed under
  `eval/reports/encoder-parity-2026-04-28/`
- Dashboard Embeddings section shows 6 rows
- TODO.md "Bucket B" entry transitions from "queued" to "DONE 2026-MM-DD"
  with cycle-closure summary

## 6. Testing, error handling, rollout & risks

### Unit tests (`tests/encoder-inference.test.ts` extension)

Three new tests, each focused and isolated:

1. **Loader metadata routing** — fake GGUF metadata with
   `general.architecture = "nomic-bert"` produces hyperparams with
   `architecture: "nomic-bert"`, `causalAttention: false`,
   `poolingType: "mean"`, populated `ropeFreqBase`. Same for
   `jina-bert-v2` with `alibiMaxBias` populated.
2. **`isEncoderArchitecture()`** — table-driven: returns true for
   `bert | nomic-bert | jina-bert-v2`, false for
   `llama | qwen3 | mistral | …`.
3. **`buildGraph` arch dispatch** — assert that the position-embedding
   `opGetRows` is called for `bert` only, that `opRope` is called for
   `nomic-bert` only, and that `opSoftMaxExt`'s 4th arg is `0.0` for
   bert/nomic and the configured `alibiMaxBias` for jina. Mock
   `GgmlWasm` records the call sequence; assertion is on the recorded
   call list.

`EncoderInference.poolAndNormalize` and `layerNorm` are reused unchanged
— no new tests. Integration / smoke is covered by Gate 4 (not by
automated tests — smoke requires a real WebGPU adapter and isn't
reachable from `bun test`). Matches BGE precedent.

### Error handling (boundary policy)

- Unknown encoder arch in GGUF → `extractHyperparams` propagates the
  unknown string; `EncoderInference` constructor throws with a clear
  message: `EncoderInference does not yet support architecture
  "<arch>"; supported: bert, nomic-bert, jina-bert-v2`.
- Missing required RoPE / ALiBi metadata → log a warning and use the
  documented fallback (10000 / 8.0). Don't throw; the parity probe
  (Gate 2) catches semantic bugs that fall through.
- Missing `position_embd.weight` for `bert` → unchanged behavior. For
  `nomic-bert` / `jina-bert-v2` the call site is now guarded so absence
  is non-fatal.
- WordPiece tokenizer mismatch → existing `buildTokenizerConfig` already
  throws on unsupported `tokenizer.ggml.model`; no new handling needed.

No defensive code beyond the boundary. Internal arch branches use
`if`/`else if`/`else` exhaustive shape — adding a fourth encoder arch
later forces a compile-time / runtime error if the new branch is missed.

### Rollout phasing (mirrors §17/§18/§19/§20 plan structure)

- **Phase 0 — Probe.** Download both GGUFs, parse, dump tensor names +
  metadata keys. ~30 min wall; produces
  `eval/reports/encoder-parity-2026-04-28/00-gguf-discovery.txt`.
  Confirms exact `rope.freq_base` / `alibi_bias_max` key spelling and
  presence/absence of `position_embd.weight`. Gate: artifacts committed
  before any code lands.
- **Phase 1 — Types + loader + helper** (commit 1, ~50 LOC): arch enum
  extension, `isEncoderArchitecture`, `extractHyperparams` deltas,
  `getRopeModeForArchitecture` extension. Unit tests #1 + #2.
  `make checkall` passes.
- **Phase 2 — Encoder forward (nomic + jina) + engine routing** (commit 2,
  ~80 LOC): `EncoderInference.buildGraph` branching, `loadWeights`
  conditional position-embedding, `engine.ts:589` broadening. Unit
  test #3. `make checkall` passes.
- **Phase 3 — Model registration + parity probe (nomic)** (commit 3):
  `eval/models.ts` + `smoke-profiles.ts` rows for nomic. Capture HF
  reference vectors. Run `eval/encoder-parity.ts`. Gate: 5/5 rows
  ≥0.999. Artifact committed.
- **Phase 4 — Same for jina** (commit 4): same shape. Gate: 5/5 rows
  ≥0.999.
- **Phase 5 — Cosine-task eval + dashboard** (commit 5): `make
  bench-full`, dashboard refresh, screenshot artifact, TODO.md
  transition.

Five commits, each `make checkall` clean, each independently revertable.
The TODO doctrine "always commit before work" is honored at every phase
boundary.

### Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GGUF mirror unavailable for jina (gaianet/...) | Medium | Phase 0 probe fails fast → search alternative mirror or convert from HF source via `convert_hf_to_gguf.py` against local `~/Repos/llama.cpp` |
| nomic uses NEOX instead of NORMAL RoPE mode | Low | Parity probe (Gate 2) catches with one of the failing 5 rows; flip mode and re-run; total cost ~5 min |
| jina ALiBi expects per-head custom slopes (not max_bias-derived) | Low | Gate 2 catches; if true, ALiBi becomes non-trivial (need new ggml op) → escalate to user, do not paper over |
| `position_embd.weight` is present in GGUF but unused for nomic/jina | Low | Loader skips loading it; no semantic effect; harmless extra bytes |
| `convert_hf_to_gguf.py` rope_freq_base key changes between llama.cpp versions | Low | Loader fallback chain at `${arch}.rope.freq_base` → `${arch}.rope_freq_base` → 10000 |
| Existing BERT path bit-non-identical post-change | Low | Phase 1+2 commits each pass `make checkall` AND `bench-full` shows BGE row unchanged within ±2% |
| Test fixture vector serialization corrupts F32 | Low | Use `Buffer.from(f32.buffer).toString('base64')` round-trip; verify on the reference machine before checking in |

**Reversibility:** any single phase can be reverted via `git revert`
without affecting others. The arch enum extension (Phase 1) is the only
widening change; all subsequent phases are additive.

## 7. Open questions / decisions

**None at design level.** Phase 0 will resolve the three Phase 0-tagged
unknowns (RoPE key spelling, ALiBi key spelling, `position_embd.weight`
presence) before any code lands.

## 8. Cross-references

- TODO.md → "Embedding-model expansion candidates — Bucket B" (queued
  2026-04-28, this doc closes the queued status by promoting to
  in-progress).
- `docs/superpowers/specs/2026-04-24-encoder-forward-pass-design.md` —
  original BERT-encoder bring-up; this spec is a delta on it.
- CLAUDE.md → "Workflow policies (set 2026-04-28)" — probe-first,
  always-commit-before-work, complexity ≠ time.
- `~/ClaudeVault/Patterns/probe-first-methodology-validates-architecture-pivots.md`
  — methodology precedent.
