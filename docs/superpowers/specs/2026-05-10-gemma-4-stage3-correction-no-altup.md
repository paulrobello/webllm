# Gemma 4 E2B Stage 3 — Architecture correction (no AltUp/Laurel)

**Date:** 2026-05-10
**Status:** Supersedes scope claims in `2026-05-10-gemma-4-stage3-gemma3n-architecture-addendum.md`
**Reason:** Direct probe of the Gemma 4 E2B GGUF tensor list (`smoke-test/models/gemma-4-e2b-it-q4km.gguf`) revealed that AltUp and Laurel tensors are **not present** in this model.

## Correction

The previous addendum (commit `179ec18`) claimed Gemma 4 E2B inherits the full Gemma 3N architecture including AltUp (4-stream predict/correct) and Laurel (low-rank parallel residual). That generalization was based on the original Task 3.2 implementer's reading of `~/Repos/llama.cpp/src/models/gemma3n.cpp` without verifying against the actual GGUF.

The Task 3.2a+b+c probe (`95a5c21`) found the complete per-block tensor inventory in this GGUF is:

| Tensor | Present? | Purpose |
|---|---|---|
| `blk.L.attn_{q,k,v,output}.weight` | ✅ | Standard attention QKVO |
| `blk.L.attn_{q,k}_norm.weight` | ✅ | **QK norm** (Gemma 3+ feature, replaces softcap) |
| `blk.L.attn_norm.weight` | ✅ | Pre-attention RMSNorm |
| `blk.L.post_attention_norm.weight` | ✅ | **Post-attention RMSNorm** (Gemma family pre+post norm pattern) |
| `blk.L.ffn_{up,gate,down,norm}.weight` | ✅ | SwiGLU FFN |
| `blk.L.post_ffw_norm.weight` | ✅ | **Post-FFW RMSNorm** |
| `blk.L.{inp_gate,proj,post_norm}.weight` | ✅ | Gated PLE per block (this campaign's load-bearing new path) |
| `blk.L.layer_output_scale.weight` | ✅ | **Per-layer output scaling** (probable Gemma 4 specific) |
| `blk.L.altup_*` | ❌ ABSENT | (Would be needed for Gemma 3N; this model doesn't ship them) |
| `blk.L.laurel_*` | ❌ ABSENT | (Would be needed for Gemma 3N; this model doesn't ship them) |

## Corrected Stage 3 scope

The 9 weights fields added by commit `95a5c21` stay (they're optional + gated on presence). For Gemma 4 E2B, only the per-block PLE fields (`pleInpGate`, `plePerBlockProj`, `plePostNorm`) and per-layer hp fields populate. AltUp/Laurel fields stay undefined for this model — that's the **correct** runtime behavior.

What still needs implementation in Stage 3:

| Component | Original spec | Addendum | This correction |
|---|---|---|---|
| Per-layer head_dim + dual RoPE | ✅ Task 3.4 | ✅ | ✅ Task 3.4 (unblocks `buildQKV`) |
| PLE pre-loop projection chain | (under-spec'd) | ✅ Task 3.3a | ✅ Task 3.3a |
| Per-block gated PLE injection | (under-spec'd) | ✅ Task 3.3b | ✅ Task 3.3b |
| QK norm (`attn_q_norm` / `attn_k_norm`) | (not in spec) | (not in addendum) | ✅ **NEW Task 3.3c** |
| Post-attention norm + post-FFW norm | Mentioned in spec §2 as known | Not in addendum | ✅ **NEW Task 3.3d** |
| `layer_output_scale` per-layer scaling | (not in spec) | (not in addendum) | ✅ **NEW Task 3.3e** |
| AltUp 4-stream | (not in spec) | Required per addendum | ❌ **REMOVED — not in this GGUF** |
| Laurel low-rank residual | (not in spec) | Required per addendum | ❌ **REMOVED — not in this GGUF** |

## Cost reduction

The addendum estimated Stage 3 at 10 tasks. The corrected list is 7 tasks (3.1 + 3.2 + 3.2a/b/c + 3.3a + 3.3b + 3.3c + 3.3d + 3.3e + 3.4 + 3.5 — wait that's 11. Let me recount):

- 3.1 DONE: probe (`c98dc1a`)
- 3.2 DONE: PLE loader (`0c91ce8` + `6c5da48`)
- 3.2a+b+c DONE bundled: per-block tensors loader (`95a5c21`)
- 3.3a: pre-loop PLE projection
- 3.3b: per-block gated PLE injection
- 3.3c: QK norm dispatch (apply attn_q_norm / attn_k_norm to Q / K after projection)
- 3.3d: post-attention norm + post-FFW norm dispatch
- 3.3e: layer_output_scale dispatch
- 3.4: per-layer head_dim + dual RoPE
- 3.5: smoke probe + 40% eval gate + closure

10 tasks total; 3 done, 7 remaining. Same task count as the addendum but with three AltUp/Laurel-replaced-with-norms tasks. Net effort: roughly equivalent (norms are simpler than AltUp but there are more of them).

## Notes for future detection

If a Gemma 4 variant ships with AltUp/Laurel tensors later (e.g., a hypothetical E1B or a future revision), the loader changes from commit `95a5c21` already detect them via tensor presence. No further loader changes needed; an op-chain implementer just needs to wire AltUp / Laurel into the forward graph when `weights.altUpGlobal` is populated.
