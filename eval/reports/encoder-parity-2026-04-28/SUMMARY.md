# Encoder parity probe — 2026-04-28

## Inputs

5 fixed strings in `inputs.json`. Reference embeddings captured via
sentence-transformers in `capture-refs.py`. Per-row gate: cosine >= 0.999.

## jina-embeddings-v2-base-en (BERT + ALiBi + GeGLU)

| Row | Input (truncated)                                    | Cosine    | Pass |
|----:|------------------------------------------------------|----------:|:----:|
|   0 | Hello world.                                         | 1.000000  |  Y   |
|   1 | The quick brown fox jumps …                          | 1.000000  |  Y   |
|   2 | Embedding models map text …                          | 1.000000  |  Y   |
|   3 | Café — naïve façade résumé piñata coöperate. 你好世界. … | 1.000000  |  Y   |
|   4 | .                                                    | 1.000000  |  Y   |

**Result:** 5/5 rows passed.
**alibiMaxBias:** 8.0 (default; gaianet GGUF mirror omits the metadata key).

## nomic-embed-text-v1.5 (BERT + RoPE + SwiGLU + fused-QKV)

| Row | Input (truncated)                                    | Cosine    | Pass |
|----:|------------------------------------------------------|----------:|:----:|
|   0 | Hello world.                                         | 0.999999  |  Y   |
|   1 | The quick brown fox jumps …                          | 1.000000  |  Y   |
|   2 | Embedding models map text …                          | 0.999999  |  Y   |
|   3 | Café — naïve façade …                                | 1.000000  |  Y   |
|   4 | .                                                    | 1.000000  |  Y   |

**Result:** 5/5 rows passed.
**RoPE mode:** NEOX (split-halves, not interleaved). Per
`llama.cpp/src/llama-model.cpp:9266` (`LLM_ARCH_NOMIC_BERT →
LLAMA_ROPE_TYPE_NEOX`) and the HF config's
`rotary_emb_interleaved: false`. The plan template's "NORMAL" hint was
wrong; cosines under NORMAL stalled at ~0.87-0.94 across all 5 rows.
**freq_base:** 1000 (loaded from GGUF `nomic-bert.rope.freq_base`;
nomic-specific value, not 10000).
**Fused-QKV byte offsets:** nb1=4*headDim, nb2=4*3*E, offsets [0, 4*E, 4*2*E].
**Tokenizer:** WordPiece. The nomic-embed-text-v1.5 GGUF omits
`tokenizer.ggml.cls_token_id` and `tokenizer.ggml.mask_token_id`.
`model-loader.ts` now falls back to `bos_token_id` / `eos_token_id`
for WordPiece tokenizers (BERT-family convention: [CLS]=bos, [SEP]=eos).

## Methodology notes

- Reference vectors normalized via sentence-transformers
  `normalize_embeddings=True`.
- WebLLM-side vectors emerge from `engine.embed()` already
  L2-normalized via `EncoderInference.poolAndNormalize`.
- Cosine computed in TS (`eval/encoder-parity.ts`), F32 accumulator.
- 0.999 gate is a soft floor: BGE shows >0.9999 informally on this
  metric. RoPE / ALiBi / SwiGLU degenerate-alignment failure modes
  typically show as <0.95 (often <0.5) — well below the gate.

## Phase 3 fixes that landed alongside the harness

Four latent bugs surfaced during diagnosis and were fixed before the
gate cleared:

1. **GeGLU vs SwiGLU (load-bearing).** `src/inference/encoder-inference.ts`
   originally called `opSilu(gate)` for both gated archs. Per
   `llama.cpp/src/models/bert.cpp:122-130`, jina-bert-v2 routes through
   `LLM_FFN_GEGLU` (`gelu(gate) * up`), nomic-bert through
   `LLM_FFN_SILU` (`silu(gate) * up`). Activation now branches on
   `arch === "nomic-bert"`. Cosine on jina jumped from ~0.96 → 1.000.

2. **ALiBi mask leaf (Phase 2a latent).** `ggml_soft_max_ext` asserts
   `mask != NULL` whenever `max_bias > 0` (`ggml.c:4012`). Previous
   code passed `null`, aborting on the first ALiBi softmax call.
   Now allocates an `[N, N]` F32 leaf and populates it with
   `mask[i,j] = -|i - j|` per `llama-graph.cpp:411`; ggml's per-head
   slope multiplication on top of that produces the standard ALiBi
   linear bias.

3. **Encoder routing in smoke page (Phase 1 latent).**
   `smoke-test/real-model-page.js` mirrors `isEncoderArchitecture` from
   `src/core/types.ts` so jina-bert-v2 / nomic-bert load through
   `EncoderInference` instead of `ModelInference` (which would crash
   with `Weight 'output_norm.weight' not found`).

4. **Window globals for harness drive-through.**
   `smoke-test/real-model-page.js` exposes `window.engine` +
   `window.handleId` after model load so external eval harnesses
   (encoder-parity, future drivers) can call `engine.embed` via
   `agentchrome js exec` without smoke-page changes.

## Browser smoke (Gate 4)

The parity harness drives `real-model.html` end-to-end (download +
WASM init + WebGPU + weight load + embed) and reports 5/5 PASS for
jina. Console errors: **0**. `adapter_info:` informational logs from
the WebGPU backend present (benign, see CLAUDE.md).
