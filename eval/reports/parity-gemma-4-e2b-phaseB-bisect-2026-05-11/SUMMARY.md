# Gemma 4 E2B — Phase B length×content bisection (2026-05-11)

**Outcome:** Phase A's per-block divergence reproduces across all
prompt shapes longer than Phase 4's 6-token canonical, **with
length as the dominant variable and special-token content
secondary.** The bug is in the per-block forward path and scales
with sequence length; localize next to attention math (SWA layers
diverge first and worst).

## Headline matrix

| Probe                 | Tokens | Special tokens | Final-norm cos | Block 2 cos | Block 11 cos | Top-16 overlap |
|-----------------------|--------|----------------|----------------|-------------|--------------|----------------|
| Phase 4 (canonical)   | 6      | none           | **0.9722 ✓**   | ≥ 0.95      | ~ 0.97       | 13 / 16        |
| **B2 short chat**     | 10     | yes (`<|turn>`)| 0.9055         | 0.9275      | 0.8510       | 10 / 16        |
| **B1 long plain**     | 92     | **none**       | **0.3894**     | **0.6784**  | **0.2396**   | 7 / 16         |
| **Phase A long chat** | 95     | yes (`<|turn>`)| **0.5824**     | **0.6520**  | **0.1538**   | **0 / 16**     |

Comparing B1 (92-token plain) and Phase A (95-token chat) is the
key control:

- Both prompts are ~the same length (92 vs 95 tokens).
- B1 has **no special tokens** in the prompt body. Phase A has
  `<|turn>`/`<turn|>`/`\n`/the system block.
- Per-block cosines track within ~0.1 of each other throughout
  the stack. Final-norm cosine differs by 0.19. Top-16 overlap
  differs by 7 (B1 keeps 7 hits; Phase A drops to 0).

→ Length explains most of the divergence; chat scaffolding
amplifies the tail (last-layer logits) but doesn't create the
catastrophic mid-stack residual drift. **The hypothesis that
special-token handling (`formatGemma4`, BOS/EOS, system block) is
the gating bug is rejected.**

## Where the failure lives in the stack

All three failing probes share a per-block divergence profile:

| Block | Type             | B2 cos | B1 cos | A cos |
|-------|------------------|--------|--------|-------|
| 0–1   | SWA (sliding)    | 0.99   | 0.98   | 0.99→0.95 |
| **2** | SWA              | 0.93   | 0.68   | 0.65  |
| 3–8   | SWA              | 0.95–0.99 | 0.85–0.94 | 0.71–0.90 |
| 9–11  | SWA              | 0.85–0.93 | 0.24–0.67 | 0.15–0.49 |
| 12    | SWA              | 0.81   | 0.62   | 0.35  |
| 13    | SWA (last pre-share-source) | 0.82 | 0.64 | 0.31 |
| 14    | full (kvFromStart) | 0.96 | 0.78   | 0.79  |
| 15–34 | shared-KV layers | 0.92–0.97 | 0.61–0.92 | 0.51–0.91 |

Phase 4 verified the shared-KV ref wiring at L15–34 works
correctly (block-by-block recovery after L14). The new finding
shows the SWA path **upstream** of L14 introduces all the damage
once the prompt is long enough to engage it past a handful of
tokens.

## Probe candidates (Phase C — intra-block taps)

Block 1 is where cosine first drops below 0.99 on the failing
probes. The candidates inside an SWA layer (post-PLE-injection
order from `model-inference.ts:forwardSingle`):

1. **Pre-RMSNorm** input residual — should match if the prior
   block was good.
2. **Q/K/V projections** — three independent matmuls; bug
   localizable by tap.
3. **QK-norm** (Gemma 3+ feature) — applied to Q and K
   independently.
4. **RoPE / RoPE-with-freq_factors** — different per-layer
   freq factors per Phase 5 / Task 3.3k. Plausible at length.
5. **V bare-RMSNorm** (Task 3.3h, Gemma-4-specific).
6. **SWA causal mask** — Gemma SWA layers should mask all
   positions *outside* the 512-window; at length 92/95 the
   window is wider than the seq, so the mask is fully causal
   (same as full attention). If the mask math wrongly applies
   the window even when seq < window, attention reads zero from
   most positions.
7. **Attention numerator / softmax denominator** — sequence-
   length-dependent.
8. **Post-attention RMSNorm + gain.**
9. **FFN (GELU + gate × up)** (Gemma uses GELU, Task 3.3g).
10. **Post-FFW RMSNorm + gain** (Task 3.3d).
11. **Per-block PLE injection** (Tasks 3.3a, 3.3b).
12. **layer_output_scale** (Task 3.3e).

The biggest-yield first taps: (6) SWA mask math and (4) RoPE.
Both can be exercised by capturing block-1 intermediate tensors
and comparing to HF's `attention_mask` / `position_ids` /
post-RoPE Q/K.

## Doctrine

Phase A established the bug exists; Phase B bisected it to length-
dependent SWA layer math. The remaining campaign is intra-block
forensics — best handled in a fresh session with a dedicated
intra-block tap added to `forwardWithLayerTaps`. Stage 4 SWA work
(implementing the windowed mask properly) is now elevated from
"pre-planned long-context fix" to "likely the gating Stage 3 bug
fix" — they may turn out to be the same lever.

Recommend the next session opens with: add intra-block taps at
block 1 (entry → Q/K/V → post-RoPE → post-attn-out), capture on
the same 92-token plain prompt (B1 fixture; no chat-template
noise), localize the first op-level divergence.

## Artifacts

- `eval/reports/parity-gemma-4-e2b-b1-longplain-2026-05-11/`
  (92-token plain control)
- `eval/reports/parity-gemma-4-e2b-b2-shortchat-2026-05-11/`
  (10-token short chat control)
- `eval/reports/parity-gemma-4-e2b-chat-emb001-2026-05-11/`
  (95-token chat-formatted, the actual eval input)
