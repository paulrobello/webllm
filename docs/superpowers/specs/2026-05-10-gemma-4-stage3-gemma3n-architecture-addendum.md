# Gemma 4 E2B Stage 3 — Gemma 3N architecture addendum

**Date:** 2026-05-10
**Status:** Approved by user 2026-05-10 ("re-spec and continue")
**Supersedes scope of:** Stage 3 only of the original spec (`2026-05-10-gemma-4-e2b-correctness-first-support-design.md`)
**Probe source of truth:** `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md` (with addendum committed in `0c91ce8`)

## What changed

The original Stage 3 scope was "PLE injection + dual RoPE + per-layer head_dim". The Task 3.2 PROBE revealed that Gemma 4 E2B is **architecturally Gemma 3N** (Google's MatFormer/AltUp variant), not a vanilla transformer with PLE bolted on. The vanilla Gemma 4 SKUs (26B A4B MoE, 31B Dense) are out of scope under the 8B parameter ceiling; both shippable SKUs (E2B, E4B) inherit the Gemma 3N architecture.

The canonical llama.cpp reference is `~/Repos/llama.cpp/src/models/gemma3n.cpp` (the file name predates the Gemma 4 rebadge; the implementation is unchanged).

## Architectural components added to Stage 3

Stage 3 now must absorb three additional components, each with its own tensors and per-block ops. Together with the original Stage 3 components (per-layer head_dim, dual RoPE, plain PLE add) they form the full Gemma 3N forward pass.

### A. AltUp (Alternating Updates) — 4-stream hidden-state architecture

- **Per-block tensors:**
  - `blk.L.altup_correct_coef.weight` — corrective coefficients for the non-active streams
  - `blk.L.altup_predict_coef.weight` — predictive coefficients used to project the active stream into the non-active streams before the attention block
- **Global tensors:**
  - `altup_proj.weight` — projects token embeddings into the AltUp 4-stream layout pre-loop
  - `altup_unembd_proj.weight` — projects the AltUp streams back to a single hidden state pre-output
- **Op sequence (per block):**
  1. Predict: project active stream to the 3 non-active streams using `altup_predict_coef`
  2. Run attention + FFN on the active stream as usual
  3. Correct: update the non-active streams using `altup_correct_coef` and the attention output
- **Active stream rotation:** the "active" stream rotates across layers in a documented pattern; the Gemma 3N reference encodes this rotation explicitly

### B. Laurel — low-rank parallel residual

- **Per-block tensors:**
  - `blk.L.laurel_l.weight` — left projection (down to low rank)
  - `blk.L.laurel_r.weight` — right projection (up from low rank back to hidden)
- **Op:** Parallel to the main attention branch, runs `x → laurel_l → laurel_r → +residual`. Low parameter count; equivalent to a lightweight LoRA-style refinement.

### C. Gated-PLE per block (replaces "PLE add" from original Stage 3)

- **Per-block tensors:**
  - `blk.L.inp_gate.weight` — gating projection
  - `blk.L.proj.weight` — final projection up to hidden
  - `blk.L.post_norm.weight` — RMSNorm after the projection
- **Global tensors (already loaded in Task 3.2 commit `6c5da48`):**
  - `per_layer_token_embd.weight` — [8960, 262144] Q5_K lookup
  - `per_layer_model_proj.weight` — [1536, 8960] BF16 projection
  - `per_layer_proj_norm.weight` — [256] F32 norm
- **Op sequence (pre-loop, once):**
  1. `GET_ROWS(per_layer_token_embd, token_ids)` → `[pleDim*L, n_tokens]`, scaled `sqrt(pleDim)`
  2. `MUL_MAT(per_layer_model_proj, inp_batch)` scaled `1/sqrt(hidden)`
  3. `RMS_NORM(per_layer_proj_norm)` on step 2
  4. `ADD(step3, step1)` scaled `1/sqrt(2)` → `inp_per_layer` permuted to `[pleDim, n_tokens, L]`
- **Op sequence (per block L):**
  5. Slice `inp_per_layer[:, :, L]` → `[pleDim, n_tokens]`
  6. `MUL_MAT(inp_gate.weight)` + GELU activation
  7. Multiply step-6 by the slice (step 5) — the gated PLE contribution
  8. `MUL_MAT(proj.weight)` up to hidden dim
  9. `RMS_NORM(post_norm.weight)`
  10. ADD into non-active AltUp streams via the AltUp correct step

## Stage 3 revised task list

Original tasks 3.1, 3.2 are landed. The revised Stage 3 task list:

- **Task 3.1 (DONE, commit `c98dc1a`):** PLE tensor sizing probe
- **Task 3.2 (DONE, commits `0c91ce8` + `6c5da48`):** Expose `per_layer_token_embd` + `per_layer_model_proj` + `per_layer_proj_norm` on the loader's weights surface
- **Task 3.2a (new):** Expose AltUp tensors (`altup_proj`, `altup_unembd_proj` global; `altup_correct_coef`, `altup_predict_coef` per-block) on the loader's weights surface
- **Task 3.2b (new):** Expose Laurel tensors (`laurel_l`, `laurel_r` per-block) on the loader's weights surface
- **Task 3.2c (new):** Expose per-block gated-PLE tensors (`inp_gate`, `proj`, `post_norm` per-block) on the loader's weights surface
- **Task 3.3a (new — replaces original 3.3):** Implement the pre-loop PLE projection chain (steps 1-4 above) producing `inp_per_layer`. Skipped for non-Gemma-3N architectures.
- **Task 3.3b (new):** Implement the per-block gated-PLE op chain (steps 5-9 above) producing a per-block PLE residual contribution. Skipped for non-Gemma-3N architectures.
- **Task 3.3c (new):** Implement AltUp pre-loop projection + per-block predict/correct ops + post-loop unembed projection. Skipped for non-Gemma-3N architectures.
- **Task 3.3d (new):** Implement Laurel per-block low-rank residual. Skipped for non-Gemma-3N architectures.
- **Task 3.4 (unchanged):** Per-layer head_dim + dual RoPE dispatch in `buildQKV`. This is the fix that allows generation to advance past the buildQKV reshape3d assertion.
- **Task 3.5 (revised gate):** Smoke probe — `"The capital of France is"` greedy decode produces `Paris` (or ` Paris`). 36-prompt eval ≥0.40 (loose Stage 3 gate; Stage 4's SWA lift will raise this to ≥0.60). Stage 3 closure report.

## Architecture-detection predicate

Gemma 3N must be detected separately from "plain Gemma 4" so the new code paths only fire for AltUp-using SKUs. The current registration declares `architecture: "gemma4"`. The runtime should gate the new paths on the **presence of AltUp tensors in the loaded weights**, not on the arch string — this approach naturally handles future Gemma 3N variants (different SKU names, same architecture) and lets the dense 26B/31B Gemma 4 SKUs (if ever added later) skip the AltUp code path automatically.

Concrete gate: `if (weights.altupProj && weights.altupUnembdProj)`.

## Scope decisions reconfirmed

- **Correctness-first staging maintained.** Stage 3 still ships full-attention-on-every-layer (SWA windowing is Stage 4); the AltUp architecture itself doesn't conflict with the all-global-attention fallback.
- **Shared-KV still Stage 5.** Each AltUp stream's KV cache stays materialized through Stages 3 + 4.
- **E2B only this campaign.** E4B follow-on probe after Stage 5 closure; same architecture so the additional work is just a registration entry + smoke validation.

## References

- `~/Repos/llama.cpp/src/models/gemma3n.cpp` — canonical forward-pass reference
- `~/Repos/llama.cpp/src/llama-arch.cpp` — confirms `LLM_ARCH_GEMMA4` and `LLM_ARCH_GEMMA3N` are distinct enums but share the gemma3n graph builder
- Task 3.1 probe report: `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md` (with addendum)
- Task 3.2 commits: `0c91ce8` (PROBE addendum), `6c5da48` (PLE tensor loader exposure)
