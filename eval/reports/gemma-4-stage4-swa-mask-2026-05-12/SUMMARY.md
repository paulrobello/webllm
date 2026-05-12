# Stage 4.1 final gate — long-context Gemma 4 SWA parity probe

**Date:** 2026-05-12
**Outcome:** **BLOCKED — per-binding 128 MiB cap on the tap probe AND a separate Gemma 4 + flashAttn=true regression discovered.** No correctness signal obtained; Stage 4.1 remains unverified at long context AND a production-path regression surfaced that wasn't caught by the original short-prompt smoke.

> **⚠ Surprise finding 2026-05-12 (late):** While trying option 1 from
> the original recommendation list (use the standard chat path to
> validate at 1129 tokens), the chat path failed at *short* prompts too
> (the same 46-token "The capital of France is" used in the Stage 4.1
> closure smoke). Reproduced on the committed tip `15b57dd` with all
> in-flight changes stashed. **The regression is `flashAttn=true`-
> specific**: same prompt works fine on `real-model.html` (defaults to
> `flashAttn=false`) and produces `Assistant: Paris` in 0.4s. **The
> "Paris" claim in TODO.md:1160 was on the manual-attention path, not
> the FA path used by `chat.html` + `createConversation`.** See
> "Side-finding" section below.

---

## What we set out to do

Stage 4.1 implementation (commits `b4f6bdf` Phase A + `0739d80` Phase B)
shipped per-layer SWA mask wiring across all four chat-path forwards.
Short-prompt smokes (TinyLlama Q4_0 165.6 tok/s bit-identical
fall-through; Gemma 4 E2B "Paris" output at 46-token prompt) confirmed
the implementation doesn't break non-SWA models and doesn't break
Gemma 4 at short prompts (where window=512 covers all positions and the
banded mask is a no-op).

The Stage 4.1 final gate per `TODO.md:1167-1238` requires a
**long-context parity probe**: tokenize ~1000 tokens, run Gemma 4 E2B
end-to-end via `forwardWithLayerTaps`, capture per-layer hidden states +
final-norm + top-K logits, compare against HuggingFace `transformers`
reference. Gate: SWA-layer cosine ≥ 0.95 at all positions; final-position
argmax matches HF.

## What we tried

| Attempt | N (tokens) | Capture mode | Result | Failure detail |
|---|---|---|---|---|
| 1 | 1129 | full (35 layer taps) | OOM at `backendAllocCtxTensors` | `ggml_tallocr_alloc: node_428 needed 6936576 / available 3919872` |
| 2 | 560 | full (35 layer taps) | OOM at `backendAllocCtxTensors` | `ggml_tallocr_alloc: node_244 needed 3440640 / available 1916928` |
| 3 | 1129 | finalOnly (no per-layer pin) | OOM at `backendAllocCtxTensors` | identical to attempt 1 |

In all three cases the failing allocation size is exactly `N × E × F32`
(= 6.94 MB at N=1129, 3.44 MB at N=560 — both matching `embeddingLength=1536`).
The allocator's available space at failure is also proportional to N
(3.92 MB at N=1129, 1.92 MB at N=560), suggesting the graph allocator
packs into a buffer sized somewhere below the 128 MiB per-binding cap.

Attempt 3 ruled out the per-layer tap pin as the dominant pressure: the
new `skipLayerTaps: true` option (commits-in-flight) removes the 35
explicit `graphBuildForwardExpand(graph, cur)` calls. JS eval against
the loaded page confirmed the new code path was active (`hasNewPath:
true`, `query` contains `finalOnly=1`). Same identical failure point as
attempt 1 → the per-layer cur values were already being lifetime-packed
correctly by the allocator; the bottleneck is elsewhere in the graph
(likely the simultaneously-live intermediates per layer: `normed`, Q/K/V
projections, RoPE outputs, attention weights, FFN gate/up, etc., plus
the pinned `inpPerLayer` PLE tensor at `pleDim × layerCount × N × F32
= 40 MB` at N=1129).

## Why long-context tap parity is infeasible on this hardware

The WebGPU `maxStorageBufferBindingSize` is **128 MiB on Chrome/Apple
regardless of total VRAM** (CLAUDE.md per-binding cap doctrine).
`ggml-webgpu` packs all graph tensors into a single storage buffer
allocation; that buffer can't exceed the binding cap.

For Gemma 4 E2B at long context the graph budget breaks down (approx):
- `inpPerLayer` PLE materialization: `pleDim × layerCount × N × F32`
  - N=1129: 256 × 35 × 1129 × 4 = **40 MB** (pinned via explicit expand —
    each per-block PLE injection slices into it)
  - N=560: 256 × 35 × 560 × 4 = 20 MB
- Per-layer live working set (Q/K/V/attn-out/FFN gate-up/etc.): each
  layer has 8-12 intermediates of varying shapes; conservative estimate
  4-8 × (N × E × F32). At N=1129 that's 28-56 MB peak per layer.
- Embedding output + per-layer `cur` (one or two simultaneously alive):
  6.94 MB at N=1129.

Total peak at N=1129 ≈ 75-100 MB even with optimal packing. We're
hitting the 128 MiB cap with very little headroom.

Smaller N doesn't help if the goal is crossing the 512 SWA window:
- N=560 also OOMs (the per-layer working set still adds up).
- N=400 might fit but doesn't cross the SWA window, so it'd validate
  only the global-attention layers.

## Stage 4.1 status after this probe

**Unchanged from before the probe:** Stage 4.1 implementation landed,
short-prompt smokes green, but the long-context end-to-end gate from
`TODO.md:1167` is **unverified**. The probe revealed an unrelated
infrastructure limit, not an SWA correctness signal.

What we DO have evidence for:
- Phase A unit tests cover `writeCausalMaskF16` correctness (banded
  mask byte layout).
- Phase B per-layer dispatch routes SWA vs global mask tensors per
  layer correctly (verified at short prompts).
- Non-SWA models are bit-identical (TinyLlama Q4_0 smoke).
- Gemma 4 E2B short-prompt output coherent (Paris on 46-token prompt).

What we DON'T have evidence for:
- SWA banded mask values are uploaded correctly at large pastLen.
- SWA layers actually behave like a windowed attention at N > 512.
- The full chain (per-layer mixed-GQA + shared-KV + PLE + RoPE + SWA)
  numerically matches HF at long context.

## Recommended next steps (in cost order)

1. **Validate via standard chat path (no parity, just non-crash).**
   Run `engine.chatCompletion` greedy on the 1129-token prompt. Uses
   `forwardSingle` directly (no tap retention). If it doesn't crash and
   produces coherent continuation, that's *positive but weak* evidence
   the SWA chain works. Cost: 5 min. **Caveat:** depends on whether
   `forwardSingle` at N=1129 single-shot fits the cap; if not, prefill
   tiling kicks in, and the open `injectPerBlockPle` view assertion may
   surface.

2. **`llama-cli` reference comparison at N ≤ 400.** Run our patched
   `llama.cpp` (same code as ggml-webgpu shares) on CPU/Metal for the
   same GGUF + same 400-token prompt + greedy. Capture first generated
   token + top-K. Compare against WebLLM run at the same N where the
   tap probe works. Validates parity at sub-window N — doesn't cross the
   SWA boundary but proves the rest of the chain (incl. global-attn
   layers) matches a llama.cpp reference (same backend math). Cost: 20 min.

3. **Incremental per-layer capture (35 separate forwards).** Wire up
   the `captureTaps` / `lastLayerTaps` scaffolding currently sitting
   unused at `model-inference.ts:421-426`. Run forward N times, each
   time capturing only one specific layer's hidden state via a debug
   readback inside the per-layer loop. 35x slower but bypasses the cap.
   Cost: 2-4 hours TS work + 30 min run time.

4. **Backend patch to split graph tensors across multiple bindings.**
   Modify `ggml-webgpu` to allocate per-layer scratch in separate
   `wgpu::Buffer` objects so per-binding cap doesn't apply to the
   *total* graph budget. Cost: serious patch-stack work; risk to other
   probe paths.

## Side-finding: Gemma 4 + flashAttn=true is broken since Phase B

Discovered while trying to validate the long-context path via standard
chat (option 1 from the recommended next steps). Test matrix:

| Page | Model | `flashAttn` | Prompt | Result |
|---|---|---|---|---|
| `chat.html` | tinyllama-1.1b-chat-q4_0 | true (default) | 46 tok | ✅ 987-char coherent reply |
| `chat.html` | gemma-4-e2b-it-q4km | true (chat-models.js:68) | 46 tok | ❌ `Error: unreachable` |
| `chat.html` | gemma-4-e2b-it-q4km | true | ~500 chars | ❌ `Error: unreachable` |
| `chat.html` | gemma-4-e2b-it-q4km | true | 1129 tok | ❌ `Error: unreachable` |
| `real-model.html` | gemma-4-e2b-it-q4km | false (no `?fa=on`) | 46 tok | ✅ `Assistant: Paris` 10.6 tok/s |

The 46-token chat.html failure reproduces with **all today's in-flight
changes stashed** — i.e., on committed tip `15b57dd`. So this is a
Phase B (commit `0739d80`) regression, not new today. The TODO line 1160
"Paris" smoke that confirmed Phase B closure was run on `real-model.html`
with `flashAttn=false` defaulted — the FA chat path was never actually
exercised on Gemma 4 after Phase B landed.

`flashAttn=true` is a **hard requirement** for `createConversation`
(`src/core/engine.ts:711-713`: `Conversations require FA mode`). So
production Gemma 4 chat is currently broken at every prompt length.

Hypothesis (unverified): the FA shader path either (a) doesn't accept
the banded-mask byte layout that `writeCausalMaskF16` writes when
`swaWindow > 0`, or (b) doesn't tolerate the mixed per-layer head_dim
(256 SWA / 512 global) that Phase B threads through the per-layer
mask dispatch. Phase A unit tests only cover the helper's byte output,
not the FA shader's consumption of it.

Next probe (cheap): set `flashAttn=true` in
`smoke-test/parity-capture-page.js:143` (currently hardcoded `false`)
and re-run the 46-token Gemma 4 capture. If it fails identically, the
bug is in FA + Gemma 4 regardless of the mask, narrowing to the shape
plumbing. If it fails with a different signature, the banded mask is
implicated.

**This finding upgrades the "Stage 4.1 long-context probe blocked"
status to "Stage 4 broke production Gemma 4 in FA mode" — needs to
be fixed before Stage 4.2 / 4.3 / 4.4 can proceed.**

## Files / artifacts

- `hf-ref.json` — HF reference for full 1129-token prompt (1.5 MB)
- `hf-ref-560.json` — HF reference for first 560 tokens (1.5 MB)
- `inputs-560.json` — input IDs for the 560-token slice
- (no WebLLM capture written — all attempts failed at allocation)

In-flight code changes:
- `src/inference/model-inference.ts:2166,2183-2186,2403-2410` —
  `skipLayerTaps` option added to `forwardWithLayerTaps`. Verified
  inert (the per-layer pin wasn't the bottleneck); kept since it's a
  cheap option to have around.
- `smoke-test/parity-capture.html` — cache-buster on inline page-script
  import (fixes a pre-existing module-cache trap).
- `smoke-test/parity-capture-page.js` — `?finalOnly=1` URL param
  threaded through to `forwardWithLayerTaps`.
- `eval/tools/parity-capture/inputs-longctx.json` — replaced degenerate
  126-word "A A A..." placeholder with ~1100-token technical prose.

## What this changes about Stage 4

`TODO.md` line 1186 plan is now:

- [x] Increase `graphMem` multiplier across all four chat-path forwards.
- [x] Thread per-layer `nHeads` / `nHeadsKv` through all four chat-path forwards.
- [ ] Debug and fix `injectPerBlockPle` view assertion. *(unchanged)*
- [ ] Long-context parity verification. **BLOCKED** — needs one of the
  recommended next steps above to proceed.
- [ ] Implement Stage 4.2 (Gemma 2 alternating SWA). *(can proceed
  independently — Gemma 2 has window=4096 and small prompts won't hit
  the cap)*.
- [ ] Commit Stage 4 and update TODO.md for Stage 5.

## FA probe result (2026-05-12 EOS-2)

**Outcome: parity-capture path with `flashAttn=true` SUCCEEDS at N=9.**

Test config:
- Page: `parity-capture.html?model=gemma-4-e2b-it-q4km`
- Edit: `parity-capture-page.js:148` `flashAttn: false` → `flashAttn: true`
- Bundle: rebuilt via `make smoke-test` on tip `65ac040`
- Forward path: `forwardWithLayerTaps`
- Prompt: `"The capital of France is"` → 9 tokenIds
- Result: 35 layers tapped in **0.28s**, embDim=1536, topK=16, no trap

Refined picture:

| Path | Forward | FA | Outcome |
|---|---|---|---|
| `chat.html` | `forwardSingle` + `forwardDecode` | true | ❌ `Error: unreachable` |
| `parity-capture.html` | `forwardWithLayerTaps` | true | ✅ 0.28s, 35 layers captured |
| `real-model.html` | `forwardSingle` (greedy) | false | ✅ `Paris` 10.6 tok/s |

The discriminator-outcome interpretation in TODO.md:1196 was that "Success →
the bug is in chat.html's worker/engine init path specifically." That was
wrong on the codepath dimension — `forwardWithLayerTaps` and
`forwardSingle`/`forwardDecode` are different forward methods even though
both honor `flashAttn` and live in the same `ModelInference` instance.

The actual narrowing is:

1. ✅ FA shader compiles and runs on Gemma 4 at all — `forwardWithLayerTaps`
   exercises the same shader.
2. ✅ Mixed per-layer head_dim (256 SWA / 512 global) is dispatched
   correctly through `buildQKV` — Phase B's threading works in
   `forwardWithLayerTaps`.
3. ❌ Something in `forwardSingle` and/or `forwardDecode` is still wrong
   for Gemma 4 + FA. Plausible candidates:
   - The banded mask byte layout (`writeCausalMaskF16` with `swaWindow > 0`)
     is consumed differently by FA vs the manual-softmax path. The FA
     shader expects a flat 2-D mask of shape `[totalLen, padded_rows]`;
     the helper writes one row per query token. Whether banded vs full
     causal differ in their byte sequence under FA needs a spot-check.
   - `forwardSingle` / `forwardDecode` build the FA call with
     `wasm.opFlashAttn(qp, fullK, fullV, mask, ...)` where `mask` is
     the freshly-written per-layer mask leaf. `forwardWithLayerTaps`
     follows the same pattern but with a separately-allocated mask
     leaf. The wiring is structurally identical; a byte-layout
     mismatch should hit both.
   - More likely: the **graph shape** at the decode step (`pastLen > 0`,
     `nTokens = 1`) is where the FA call breaks. `forwardWithLayerTaps`
     only ever runs prefill-shape (full sequence, no KV cache reuse).
     `forwardDecode` runs `nTokens=1` with `pastLen` past tokens in the
     KV cache. The "unreachable" trap at the very first message means
     prefill, not decode, is failing — so:
   - Most likely candidate: **prefill-batched** shape in `chat.html`
     differs from prefill via `forwardWithLayerTaps`.
     `engine.chatCompletion` invokes prefill through `forwardAllPositions`
     when prefill > 1 token (which a 9-token prompt is). Phase B
     touched `forwardAllPositions` as well, but `forwardAllPositions`
     was not exercised by today's parity-capture probe.

### Next probe (cheap, ~5 min)

Add a `forwardAllPositions` smoke probe with FA=true + Gemma 4 at the
same 9-token prompt. If it traps, the FA bug is in `forwardAllPositions`
(prefill-batched) specifically. If it succeeds, the trap is in
`forwardDecode` (`nTokens=1` decode-shape FA), surfacing later than the
"first message" framing suggested.

Wiring this probe means adding either:

- a third smoke page (`smoke-test/fa-prefill-probe.html`) that
  instantiates `ModelInference` + invokes `forwardAllPositions`
  directly, mirroring `parity-capture-page.js` plumbing; or
- a debug knob in `engine.chatCompletion` that swaps `forwardAllPositions`
  for `forwardWithLayerTaps` — easier but invasive.

The standalone probe page is the cheaper option (~80 lines of JS,
clones the parity-capture harness).

### Status update on the production Gemma 4 chat regression

Production Gemma 4 chat (`chat.html`) is broken since Phase B (`0739d80`,
2026-05-11). The FA forward dispatch through `forwardSingle`/`forwardAllPositions`/
`forwardDecode` is the failing path; `forwardWithLayerTaps` works. This
is a Stage 4 regression, not a new today's change regression — TS
changes committed in this session (`447ff82` per-layer plumbing,
`65ac040` parity-capture) preserve the same failure signature.

## FA forwardAllPositions + forwardSingle discriminator probe (2026-05-12 EOS-3)

New standalone probe page added at `smoke-test/fa-prefill-probe.html` +
`smoke-test/fa-prefill-probe-page.js`. Drives `ModelInference` directly,
calls either `forwardVerify` (→ `forwardAllPositions`) or `forward()` (→
`forwardSingle`) on the same 9-token raw "The capital of France is" prompt.
URL knobs: `?path=verify|forward`, `?ctx=N`, `?fa=off`.

**Both forwards SUCCEED on Gemma 4 + FA at the 9-token raw prompt + ctx=4096.**

| Path | Forward | FA | nTokens | ctxLen | Outcome |
|---|---|---|---|---|---|
| `chat.html` | `forwardSingle` (in conversation pool) | true | 46 (chat-templated) | 4096 | ❌ `Error: unreachable` |
| `fa-prefill-probe.html?path=forward` | `forwardSingle` (via `inf.forward()`) | true | 9 (raw) | 4096 | ✅ 0.294s, argmax=236772 logit=-9.381 |
| `fa-prefill-probe.html?path=verify` | `forwardAllPositions` (via `inf.forwardVerify()`) | true | 9 (raw) | 4096 | ✅ 0.283s, argmax=236772 logit=-9.381 |
| `parity-capture.html` (FA=true edit) | `forwardWithLayerTaps` | true | 9 (raw) | n/a | ✅ 0.28s, 35 layers tapped |
| `real-model.html` | `forwardSingle` (greedy) | false | varies | varies | ✅ `Assistant: Paris` 10.6 tok/s |

**Initial probe with ctx=131072** (GGUF default) failed at `ggml_aligned_malloc:
insufficient memory (attempted to allocate 3842.09 MB)` during `initKVCache`
— the FA F16 KV cache at owningLayers=15 × maxHeadDim=512 × ctxLen=131072 × 2 bytes × 2 (K+V)
= ~3.75 GiB. Clamping `ctxLen` to 4096 (chat.html's clamp via `model.contextLength`)
makes the probe pass. The "Error: unreachable" trap on chat.html is NOT this OOM
— chat.html clamps ctx to 4096 too.

### What this means for the regression hunt

The previous "narrow to `forwardSingle` / `forwardAllPositions` / `forwardDecode`"
framing was too broad. None of the three traps on a raw 9-token prefill with FA=true
+ ctx=4096. The Gemma 4 + FA chat.html trap is therefore caused by one of:

1. **Chat-formatted prompt content.** `formatGemma4` emits `<|turn>` (id 105),
   `<turn|>` (id 106), BOS, and other special tokens; the conversation prefill
   ends up ~46 tokens vs the probe's 9 raw tokens. Special-token IDs near vocab
   bottom may engage a code path (e.g. unused-token row in `tok_emb`, or BF16-
   cast row index) that raw IDs in the 40K–800K range don't.
2. **`forwardDecode` (nTokens=1 + pastLen>0).** The trap might fire at the first
   *decode* step rather than prefill — the chat UI message-rendering wouldn't
   distinguish prefill-fail from first-decode-fail. The "first message" framing
   in TODO.md was inferential, not measured.
3. **Conversation pool / KV-snapshot pre-state.** `chatCompletionWithConversation`
   does `inf.forward(midIds, midPos)` for shared-prefix-aware prefill at line
   936, then `generateTextStream` re-prefills the last token. If the KV-cache
   snapshot path mutates `nCached` to a non-zero value before the first prefill,
   `forwardSingle` runs with `pastLen > 0` — a shape the probe never exercises.
4. **Long-context cap (Phase B SWA mask byte layout).** At chat-template nTokens=46,
   the banded SWA mask shape differs from the trivial all-zeros short-mask. The
   probe doesn't exercise SWA mask (`needsSwaMask` requires either nTokens>1 OR
   pastLen+nTokens>swaWindow, but window=512 and nTokens=9 falls in the
   non-banded regime).

Hypothesis ranking by likelihood: **(4) > (1) > (2) > (3)**. (4) is the only one
that intersects Phase B's structural changes; the others would have produced an
unreachable trap at *any* Phase B model not just Gemma 4.

### Next cheap probe (~10 min)

Extend `fa-prefill-probe-page.js` to:
- Accept `?chat=1` — apply the actual chat-template via `formatGemma4`
  (or copy the same logic) before tokenization.
- Accept `?nTokens=N` — repeat the 9-token prompt to reach an arbitrary
  prefill length, sweeping above and below the SWA window (512).
- If the chat-formatted 46-token prefill traps and the raw 46-token-via-
  repetition prefill doesn't → bug is content-specific (hypothesis 1).
- If both 46-token forms trap and 9-token doesn't → bug is length-specific
  (hypothesis 4 — SWA mask trap below or near boundary).
- If neither traps → bug is in conversation-pool / decode path (hypotheses
  2 or 3); switch to a `chatCompletion`-driven probe.

Once the failing shape is reproduced in the standalone probe, the trap
location bisection is straightforward.
