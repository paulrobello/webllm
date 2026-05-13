# Gemma 4 E2B — Campaign Closure (CLOSED 2026-05-12) ✅

**Campaign:** Gemma 4 E2B IT correctness-first browser inference support.
**Started:** 2026-05-10 (Phase 1 probe + spec).
**Closed:** 2026-05-12 EOS (this report).
**Final eval:** **34 / 48 = 70.8 %** at greedy temp=0 on the 36-prompt
+ 12 semantic-reasoning suite — well above the ≥40 % Stage 3 gate and
above the ≥60 % Phi-3 closure baseline. Bit-identical to Stage 3
closure (zero Stage-4 regression).
**Final speed:** **38.6 tok/s p50** in Pass 2 canonical regime (fresh
headless Chrome / 30s cooldown / 5 runs / profile mode).
**Patch budget consumed:** 0 of +2 allotted (no llama.cpp patches were
required — Gemma 4 wired entirely in TypeScript on top of the
existing patched ggml-webgpu surface).

## Outcome at a glance

| Item                                   | Result |
|---|---|
| Eval ≥40 % gate (Stage 3 closure)      | ✅ 68 % → 70.8 % (clear) |
| Long-context chat coherence (Stage 4.1)| ✅ 2,238-token prompt, fact-correct retrieval, 0 crashes |
| Stage 4 eval re-gate (Stage 4.4)       | ✅ 70.8 % bit-identical to Stage 3 |
| Bench-full integration (Stage 5.2)     | ✅ `gemma-4-e2b-warm` in `SMOKE_PROFILE_SETS.full` |
| Dashboard accuracy×speed dot           | ✅ rendered (22 perf runs + 6 evals in `eval/reports/smoke-runs.db`) |
| Patch budget                           | ✅ 0 of +2 used |
| Ship gate (`make checkall`)            | ✅ 782 pass / 36 skip / 0 fail / 39312 expect() calls |

## §27 / §28 / §32 classification

The campaign-defining root cause was a **§27 free-win retrospectively**:
a single missing entry in `getRopeModeForArchitecture` —
Gemma family uses NEOX-style RoPE (split-halves) per
`llama-model.cpp:2275-2310`, but the project mapped only
`nomic-bert`, `phi3`, and `qwen*` to NEOX and let Gemma fall through
to `RopeMode.NORMAL` (interleaved). Fix was three lines plus a
load-bearing comment block. Eval lifted **9 % → 68 %** (+59 pp) from
that one change. Stages 1–2 and Stage 3 sub-tasks 3.3a–k were all
load-bearing correctness pieces — individually each was correct —
but invisible to the smoke probe until paired with the NEOX
rotation phase.

The Stage 3 closure SUMMARY ranked this as the highest single-line
return-on-effort in the project's perf/correctness history. It
became a **doctrine lesson** (now in CLAUDE.md): *"Demote candidates
are usually plural"* — the Gemma 2 demote SUMMARY enumerated five
candidates; un-demote needed six fixes (Q1.6), three of which
weren't on the original list. Same pattern as Gemma 4 — multiple
correctness pieces had to land before the headline gate moved.

## Per-stage closure highlights

### Stage 1 — Per-layer hyperparams refactor (2026-05-10) ✅

Converted scalar `embeddingHeadLength`, `feedForwardLength`,
`ropeDimensionCount`, `ropeFreqBase` into per-layer arrays. Existing
models replicate the scalar `layerCount` times (zero behavioral
delta); Gemma 4 populates per-layer from GGUF (head_dim varies
across layers per the mixed-GQA pattern).

**Commits:** `84151ad` (types) · `c274600`/`47499e1` (GGUF readers) ·
`05f5238`/`4fe0f71` (loader).
**Gate:** `make checkall` green + 3-model smoke `generatedIds[0]`
match (TinyLlama 143.7 tok/s, Qwen3 0.6B 115.2 tok/s, Qwen3 1.7B
79.3 tok/s — all coherent first token, no console errors).
**Closure:** [`gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md`](../gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md).

### Stage 2 — Surface wiring (2026-05-10) ✅

Chat template (`formatGemma4`), stop-token registration
(`<end_of_turn>`), `GEMMA4_DEFAULTS` sampler,
`final_logit_softcapping=30.0` plumb, `eval/models.ts` registration,
bundle regeneration.

**Commits:** `848333e` · `f779a22` · `ef0bb32` · `fb45c01` ·
`ad741db` · `72ede26`.
**Gate (adjusted):** the plan's "5 coherent ASCII tokens" gate was
structurally unachievable at Stage 2 because the `buildQKV`
reshape3d assertion required Stage 3 Task 3.4. Stage 2 ships
surface-only; runtime gate folded into Stage 3.
**Closure:** [`gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md`](../gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md).

### Stage 3 — Forward-pass correctness (2026-05-11) ✅

The load-bearing stage. Sub-tasks 3.1 + 3.2 + 3.3a-k landed all
correctness pieces individually; the NEOX RoPE fix (folded into the
gemma family branch in `getRopeModeForArchitecture`) unblocked the
headline gate.

**Headline:** Eval **9 % → 68 %** (+59 pp) at greedy temp=0.

| Dimension              | Before | After    | Δ        |
|------------------------|-------:|---------:|---------:|
| instruction-following  | 19.4 % | **91.7 %** | **+72 pp** |
| reasoning              |  0 %   | **83.3 %** | **+83 pp** |
| semantic-reasoning     |  0 %   | **79.8 %** | **+80 pp** |
| tool-calling           | 16.7 % | 16.7 %     | 0 (capability disabled by design — Gemma 4 PEG format unsupported) |
| **Overall**            |  9 %   | **68 %**   | **+59 pp** |

**Architectural pieces shipped:**

- **Gated PLE per block** (`per_layer_token_embd`, `per_layer_model_proj`,
  `per_layer_proj_norm`, `inp_gate`, `proj`, `post_norm`) —
  `buildPreLoopPle` + `injectPerBlockPle` per `gemma3n.cpp:317-371`
  + `gemma4.cpp:328-353`. Commits: `6c5da48`, `95a5c21`, `ba0f90e`,
  `cf56960`.
- **QK norm** — pre-existed generically for Qwen3; fires
  automatically for Gemma 4 via `opt("attn_q_norm.weight")` /
  `attn_k_norm.weight` at `model-inference.ts:399-400`.
- **Pre+post norm pairs** — `postAttentionNorm` / `postFfwNorm`
  loaded via `opt("post_attention_norm.weight")` /
  `opt("post_ffw_norm.weight")`. Ternary-gated `opMul(opRmsNorm(x,
  epsilon), gain)` per `gemma4.cpp:246-249` + `323-326`. Commit:
  `73f77df`.
- **Per-layer output scaling** — `layerOutputScale` applied at end
  of each per-layer iteration per `gemma4.cpp:355-358`. Commit:
  `c4e5659`.
- **Gemma embedding scaling** — `opScale(x, sqrt(embeddingLength))`
  inserted after `opGetRows(weights.tokEmb, ...)` at all four
  forward methods, gated on `arch === "gemma4"` (later widened to
  whole Gemma family in Q1.6). Commit: `63c1a6d`.
- **GELU FFN** — replaced `opSwigluSplit(gate, up)` with
  `opMul(opGelu(gate), up)` for Gemma 4 (later whole Gemma family).
  Commit: `79dd05d`.
- **V bare-RMS-norm inside `buildQKV`** — `opRmsNorm(v3, epsilon)`
  (no gain) before return for Gemma 4. Commit: `a321df6`.
- **Drop final-logit-softcap misuse in FA** — pass 0.0 to FA's
  `logit_softcap` (Gemma 4 has `f_attention_scale = 1.0` with no
  attention softcap per `gemma4.cpp:11`). Commit: `ac8bbe1`.
- **BF16 → F32 cast at weight load** — Pass-1's "benign CPU
  fallback" was actually a correctness-blocking
  CommandBuffer-invalidation bug. `bf16BytesToF32Bytes` helper
  converts at upload. Commits: `d6132ed` + `2591525`.
- **rope_freqs (freq_factors)** — new `op_rope_with_freqs` WASM
  binding + TS wrapper + per-layer plumbing through SWA pattern.
  Commit: `dec6f2d`.
- **Attention softmax scale** — `attnSoftmaxScale(hp, headDim)`
  helper returns 1.0 for `gemma4` (Gemma 4 trains `attn_q_norm` /
  `attn_k_norm` gains to compensate for the missing 1/√d_k).
  Commit pre-3.3l Phase 3.
- **NEOX-RoPE family extension** (the §27 free-win): three lines in
  `getRopeModeForArchitecture` mapping `gemma2` / `gemma3` /
  `gemma4` to `RopeMode.NEOX`. Commits: `be63158` + `c8c8447`.
- **Shared-KV at layers 15-34** — `n_layer_kv_from_start=15`, 20
  shared layers. End-of-stack cosine recovered **0.0420 → 0.9722**;
  top-1 argmax MATCHES HF reference (id 9079 "Paris"); top-16
  overlap 1/16 → 13/16. Phase 4 closure:
  [`parity-gemma-4-e2b-shared-kv-2026-05-11/SUMMARY.md`](../parity-gemma-4-e2b-shared-kv-2026-05-11/SUMMARY.md).
- **Chat-template tokenization audit (Phase 5)** — unsloth GGUF
  vocab stores turn-boundary tokens under non-standard literals
  `<|turn>` (id 105) and `<turn|>` (id 106). `formatGemma4` was
  hard-coding the classical `<start_of_turn>` / `<end_of_turn>`,
  causing BPE to fragment into ~7 untrained pieces per turn.
  Template-sniff substring fix; classical Gemma 2 / 3 unaffected.
  Commits: `d8a0835` + `4fc5993`. Closure:
  [`gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md`](../gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md).

**Closure:** [`gemma-4-stage3-closure-2026-05-11/SUMMARY.md`](../gemma-4-stage3-closure-2026-05-11/SUMMARY.md).

### Stage 4 — Real sliding-window attention (2026-05-11 / 2026-05-12) ✅

Replaced Stage 3's "all-global" fallback with real per-layer banded
SWA on the 4-of-5 SWA-marked layers (window=512). Phase A (the
shared mask-construction primitive) + Phase B (per-method wiring)
landed without a llama.cpp patch — both `opSoftMaxExt` and
`opFlashAttn` already handle the mask as a purely additive
per-element term.

**Sub-stages:**

- **4.0 — Windowed-mask feasibility probe.** ✅ Closed at
  [`gemma-4-stage4-probe-2026-05-11/SUMMARY.md`](../gemma-4-stage4-probe-2026-05-11/SUMMARY.md) —
  no llama.cpp patch needed.
- **4.1 — Per-layer mask construction.** ✅ Phase A `b4f6bdf` +
  Phase B `0739d80`. Long-context closure verified at 2,238 tokens
  (4.4× the SWA window) at
  [`gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md`](../gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md).
- **4.2 — Gemma 2/3 SWA pattern derivation from `swa_period`.** ✅
  Commit `e5454ad`; 4 tests at
  `tests/models/model-loader-gemma2-hparams.test.ts`.
- **4.3 — Long-context regression probe.** PARTIAL ✅ at
  [`gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md`](../gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md) —
  TinyLlama API gate passes (cosine 0.9855); Gemma 4 N=560 blocked
  by the per-binding 128 MiB cap. Functional gate is met by 4.1's
  2,238-token closure; numerical-parity escalation deferred to
  multi-binding scratch allocation if needed by future work.
- **4.4 — Eval re-gate.** ✅ 34/48 = **70.8 %** at
  [`gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md`](../gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md) —
  bit-identical to Stage 3 closure (SWA invisible at sub-window
  prompts; functional long-context covered by 4.1).

**Side-quest closure:** the FA-VEC `prefillTileSize=16` clamp for
Gemma family head_dim > 128 shipped as commit `9ea3bfc` (Stage 4
prerequisite). Pre-existing bug surfaced only when chat.html drove
FA for the first time at Stage 4 validation; not a regression.

### Stage 5 — Bench + closure (2026-05-12) ✅

- **5.1 — Pre-rebase baseline capture.** Pass 1 (3-run / shared-Chrome)
  superseded by Pass 2 (5-run / fresh-Chrome-per-model / 30s cooldown).
  Pass 2 matmul time moved -4.5 % to -8.3 % (faster) across the
  canonical 6 vs the 2026-05-04 baseline — methodology-driven
  cleanup, no regression. New canonical capture regime documented in
  TODO `§Stage 5.1`. Closure:
  [`pre-rebase-baselines-2026-05-12/SUMMARY.md`](../pre-rebase-baselines-2026-05-12/SUMMARY.md).
- **5.2 — Add `gemma-4-e2b-warm` to `SMOKE_PROFILE_SETS.full`.** ✅
  Commit `f8c4c65` (one-line insert after `gemma-2-2b-warm`).
  Dashboard renders Gemma 4 alongside the canonical 6 in
  accuracy×speed scatter.

## Final perf snapshot

Pass 2 canonical regime, profile mode, 5 runs (`PERF_RUNS=5`):

| Metric                  | Value             | Notes |
|---|---|---|
| Decode tok/s (p50)      | **38.6**          | Spread 38.6 % across 5 runs (45.6/44.6/30.7/38.6/36.9) — dispatch-heavy variance |
| Matmul (median, ms)     | **8.19**          | Stable: mean 8.22, median 8.19, p90 8.72 across 65 samples |
| Matmul share            | 47.6 %            | Lower than canonical 6 (50-82 %) — dispatch overhead is the bigger slice |
| Encode (median, ms)     | 4.10              | 23.1 % share; per-dispatch ~3.9 µs |
| Dispatch/token          | **1040**          | Highest in fleet (canonical 6: 450-805) — PLE injection per layer |
| Steps per run           | 65 single-token   | At 14-token generation on "Tell one short joke." with stop-token finish |

**Reading note:** Gemma 4's high run-to-run spread is intrinsic — its
1040-dispatches-per-token nature means per-step micro-stalls accumulate.
Matmul compute itself is stable. Tighter headline median is queued as
a follow-up watch (7-9 runs + trimmed mean would help) but is not
gating any further work.

## Patch budget — 0 of +2 used

The campaign was originally scoped with +2 llama.cpp patches allowed
(Stages 4 and 5). Neither was needed:

- **Stage 4** patch budget — the windowed-mask feasibility probe
  (4.0) found both `opSoftMaxExt` and `opFlashAttn` already accept
  arbitrary additive masks; banded SWA is byte-content-only delta
  from full-causal. Wired entirely in TypeScript.
- **Stage 5** patch budget — perf was non-regressive without any
  llama.cpp change.

The 11-patch effective stack on `webllm-browser-patches` (excluding
the 16 dormant JSEP probes on top of `b54503497`) is unchanged from
pre-campaign.

## Doctrine lessons banked

1. **Demote candidates are usually plural** (Stage 3 closure +
   Q1.6 retrospective). When a model demotes from `full`, the
   demote SUMMARY's enumerated causes are usually a starting set,
   not a complete set. Future demote SUMMARYs should bias toward
   "expect plural" rather than "one of these five".
2. **Soft-cap order is non-trivial.** Naïve `softcap → scale`
   silently corrupts attention. Reference order is
   `scale → softcap → softmax` per `ggml-cpu/ops.cpp:8232-8233`.
   Added to the chat-template / RoPE family audit list for the
   next llama.cpp rebase.
3. **Lookup-table extension audits pay off across the family.**
   Three of the six Q1 fixes were "extend the `gemma4`-only branch
   to the whole Gemma family" or "add gemma2/gemma3 to the NEOX
   list". The audit caught the lot in one cycle. Canonical surfaces
   to audit per major rebase: `getRopeModeForArchitecture`,
   `attnSoftmaxScale`, chat-template detector, `isGemmaFamily`,
   GELU-vs-SwiGLU branch tables.
4. **Pass 1 noise is real.** Yesterday's 3-run shared-Chrome capture
   said "matmul bit-identical"; Pass 2 5-run fresh-Chrome capture
   showed matmul actually moved -4.5 % to -8.3 % uniformly. New
   canonical capture regime: fresh Chrome per model + 30s cooldown
   + 5 runs + headless. All future §32 / §27 / §28 baseline
   captures use this regime.
5. **Cap-probe doctrine sometimes hits a wall the codebase can't
   climb without an upstream patch.** Stage 4.3's per-binding 128
   MiB cap blocked strict parity-numerical gating on Gemma 4 at
   long context; the functional gate (Stage 4.1's 2,238-token
   coherence) was good enough. Multi-binding scratch allocation is
   queued as an out-of-stage opportunity if a future probe needs
   it; not gating shipping.

## Follow-ups (not gating; queued in TODO)

- **Embedding-path SWA support.** `forwardForEmbedding`
  (`model-inference.ts:1907`) was deliberately left on the original
  single-mask path because Gemma 4 isn't registered as an embedder.
  If a Gemma SWA model ever ships as an embedder, mirror the
  Phase B SWA wiring there.
- **`debugLayerOutput` SWA support.** Same situation — debug-only
  path skipped to keep the Phase B diff narrow.
- **Gemma 4 perf tightening** — 7-9-run + trimmed-mean re-capture
  to tighten the 38.6 tok/s headline. Compute is stable; only the
  variance reads loosely.
- **Upstream patch follow-up — bump FA-VEC `ne[1] < 20` ceiling.**
  P2 hygiene: recover prefill throughput on Gemma 4 by letting
  larger tiles use VEC at `q_tile=1`. One-line edit to
  `~/Repos/llama.cpp` `webllm-browser-patches`
  `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp:734`; should
  also be filed upstream. **Quantified cost:** 42.3 s TTFT for
  2,238-token prefill at q_tile=1 today.
- **Strict numerical parity gate for Gemma 4 long-context.**
  Multi-binding scratch allocation in ggml-webgpu would unblock
  Stage 4.3's strict gate. Functional gate already met; numerical
  re-gate is only justified if a probe finds a quality issue at
  long context.
- **E4B SKU.** Same architecture path; expected registration-only
  delta. Queue as a follow-on probe after Gemma 4 E2B has time in
  the canonical fleet.

## Out of scope (campaign-final)

- **E4B SKU** beyond the queued follow-on probe.
- **PLE CPU offload** (memory optimization for E4B+ — would require
  JS↔WASM boundary work).
- **26B A4B MoE and 31B Dense Gemma SKUs** — above the 8B parameter
  ceiling per CLAUDE.md workflow policies.
- **Multimodal Gemma 4** (vision/audio paths) — Gemma 4 E2B base
  ships text-only; multimodal variants exceed the project's
  agent-coexistence VRAM budget.
- **MTP drafter** — deferred behind upstream llama.cpp Discussion
  #22735.

## References — full per-stage closure links

- **Spec:** [`docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md`](../../../docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md)
- **Plan:** `docs/superpowers/plans/2026-05-10-gemma-4-e2b-correctness-first-support.md`
- **Stage 3 correction (authoritative):** [`docs/superpowers/specs/2026-05-10-gemma-4-stage3-correction-no-altup.md`](../../../docs/superpowers/specs/2026-05-10-gemma-4-stage3-correction-no-altup.md)
- **Stage 1:** [`gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md`](../gemma-4-stage1-per-layer-hp-2026-05-10/SUMMARY.md)
- **Stage 2:** [`gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md`](../gemma-4-stage2-surface-wiring-2026-05-10/SUMMARY.md)
- **Stage 3 closure:** [`gemma-4-stage3-closure-2026-05-11/SUMMARY.md`](../gemma-4-stage3-closure-2026-05-11/SUMMARY.md)
- **Phase 5 chat-template:** [`gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md`](../gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md)
- **Parity probes:** `parity-gemma-4-e2b-{2026-05-11, attnscale-fix, b1-longplain, b2-shortchat, chat-emb001, phaseB-bisect, rope-neox-fix, shared-kv, stage3-block0}-2026-05-11/`
- **Stage 4.0 probe:** [`gemma-4-stage4-probe-2026-05-11/SUMMARY.md`](../gemma-4-stage4-probe-2026-05-11/SUMMARY.md)
- **Stage 4.1 long-context:** [`gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md`](../gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md)
- **Stage 4.3 partial:** [`gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md`](../gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md)
- **Stage 4.4 eval re-gate:** [`gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md`](../gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md)
- **Stage 5.1 pre-rebase baselines:** [`pre-rebase-baselines-2026-05-12/SUMMARY.md`](../pre-rebase-baselines-2026-05-12/SUMMARY.md)

**Tip context at closure:**
- webllm tip: `225054e` (`docs(stage5.1+5.2): Pass 2 retake`).
- llama.cpp `webllm-browser-patches` tip: `ebc7c3d82` (16 inert
  JSEP probe commits on top of canonical `b54503497`; JSEP gated
  OFF in the WASM build — effectively `b54503497` for our binary).
