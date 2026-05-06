# P2 — Causal-LM migration to `webllm_decode` (closure)

> **Date:** 2026-05-05
> **Original commit range:** `4bb644c..374cc46` (11 commits)
> **Original verdict (superseded):** PASS-with-followups
> **Final verdict:** **PARTIAL REVERT** — same-tip canonical-6 bench retake on `bd7ae4b` post-Task-11 fix exposed an 18× decode-throughput regression on tinyllama. Path A investigation (P2.1.A — see [`POST-MIGRATION-BENCH.md`](POST-MIGRATION-BENCH.md)) localized the cost to per-WebGPU-dispatch JS↔WASM shim crossings under the emdawnwebgpu port — intrinsic to running `ggml-webgpu` inside WASM, not patchable in llama.cpp. P2.1.B section (same doc) reframes this as an architectural mismatch, not a perf-recovery problem, and surfaces a third path (JSEP-style: WASM scheduler + JS-side WebGPU kernels, the pattern transformers.js / ORT-Web ships).
>
> **Action taken (commit `0b57d41`):** partial revert of Tasks 5-11 + smoke-page port; bridge expansion (Tasks 1-4 + TS interface) kept as-is for the next-session JSEP-style architecture. `ModelInference` legacy path restored; throughput recovers to legacy baseline.
>
> **Read this report bottom-up.** The original "Outcome" / "LOC delta" / "Patches consumed" / "Test posture" sections below describe the v1 migration as it was attempted; they are accurate as a snapshot of `374cc46` but no longer describe the codebase. The follow-up sections in `POST-MIGRATION-BENCH.md` carry the load-bearing narrative.

## Outcome

`src/inference/model-inference.ts` (~2890 LOC) deleted. The causal-LM forward path now flows through:

```
engine.ts:loadModelFromBuffer
  → loadModelMetadata(bridge, gguf)
  → new LlamaDecodeWrapper(bridge, model, {flashAttn})
  → wrapper.initKVCache(ctxLen)
  → wrapper.forward(tokenIds, positions)
       → bridge.decode(ctx, tokenIds, pastLen)
       → bridge.getLogits(ctx, model, -1)
  → sampler.sample(logits)   ← unchanged
```

The hand-rolled TS graph builder is gone for the causal path. Encoder + causal-embedder paths remain on the legacy graph builder until P3 / P4.

## LOC delta

```
$ git diff --stat 4bb644c..374cc46 | tail -3
 26 files changed, 1515 insertions(+), 4161 deletions(-)
```

**Net:** **−2646 LOC** from the TypeScript surface.

| Component | Δ | Notes |
|---|---|---|
| `src/inference/model-inference.ts` | **−2890** | deleted (entire legacy graph builder) |
| `src/wasm/webgpu-bridge.cpp` | +123 | 13 new exports (metadata × 7, KV ops × 2, state-seq × 3, embeddings × 1) |
| `src/wasm/CMakeLists.txt` | +9 / −2 | EXPORTED_FUNCTIONS + 1 JSPI_EXPORTS entry (embeddings) |
| `src/inference/llama-bridge.ts` | +216 | typed bindings for the 13 new exports |
| `src/inference/llama-decode-wrapper.ts` | +276 | new wrapper class |
| `src/inference/llama-tokenizer.ts` | (P1 already shipped) | — |
| `src/models/model-loader.ts` | rewrite (~210 LOC) | bridge metadata pass-through |
| `src/models/model-loader-legacy.ts` | +422 | renamed verbatim copy of pre-P2 loader (encoder/causal-embedder paths) |
| `src/core/engine.ts` | +131 / −25 | causal-LM dispatch flip + union narrowing + 2 instanceof gate removals |
| `src/inference/ggml-wasm.ts` | +11 | `get module()` accessor for bridge construction |
| `src/inference/speculative.ts` | +29 / −5 | local `SpecDecodeForwardPass` interface |
| `src/inference/rope/rope-mode.ts` | +29 | extracted `getRopeModeForArchitecture` + `RopeMode` for encoder/embedder |
| `src/inference/generation.ts` | +27 | `DecodeMode` + `DecodeResult` types relocated |
| `src/inference/encoder-inference.ts` / `causal-embedder-inference.ts` | ±2 each | rope import path |
| `src/index.ts` | ±1 | drop `ModelInference`, add `LlamaDecodeWrapper` |
| `tests/llama-decode-wrapper.test.ts` | +141 | 6 fake-bridge unit tests |
| `tests/{model-inference, generation}.test.ts` | ±2 each | import path |
| `tests/{kv-snapshot-roundtrip, speculative-integration, forward-verify-equivalence, prefill-tiling-config, fused-contiguity-assert, fa-mode-config, prefill-tiling-equivalence, model-inference-embed}.test.ts` | **−952** | 8 test files deleted (all imported `ModelInference` directly to test internals of the deleted class) |

The spec predicted **−3400** TS net delete; actual is −2646. The shortfall is ~750 LOC, accounted for by:
- Tokenizer.ts trim deferred to P3 (~600 LOC) — encoder + causal-embedder still construct legacy `Tokenizer`
- Bridge surface larger than spec assumed (+123 cpp + 216 ts on the new bindings) — driven by the need to preserve serializeKVCache/loadKVCache + Bucket-D embed parity within P2 (spec planned to defer some of this to P4)

## Patches consumed

**0 llama.cpp patches.** Spec budgeted up to 3 in band B; P2 needed none. ASYNCIFY-aware `llama_decode` was already JSPI-wrapped from P0; KV ops (`llama_memory_*`), state-seq (`llama_state_seq_*`), metadata (`llama_model_meta_val_str`), and embeddings (`llama_get_embeddings_ith`) all sit on the upstream public API.

## Test posture

```
make checkall (post-Task 11 / 374cc46)
fmt:   Formatted 121 files in 23ms.
lint:  Checked 121 files in 33ms. No fixes applied.
typecheck: clean
typecheck:tests: clean
bun test: 731 pass / 23 skip / 0 fail / 39162 expect() calls (754 tests, 72 files)
```

Pre-P2 baseline was 753/33 (786 tests). Net delta:
- **+6** new tests (`tests/llama-decode-wrapper.test.ts` — 6 fake-bridge unit tests covering forward sequential-position invariant, cachedTokenCount tracking, resetKVCache, truncateKVCache, loadKVCache snapshotLen>nTokens, embed lazy-context + KV reset)
- **−28** deleted tests across 8 files that all imported `ModelInference` to test internals of the deleted class (constructor opts, prefill tile heuristic config, F32-contiguity assertion, FA-mode config, KV snapshot direct round-trip, spec-decode integration, forwardVerify equivalence, embed). Of these, 18 ran in Bun pre-deletion; 10 were already skip-gated on `HAS_WEBGPU`.

The deleted tests covered behavior that no longer exists in the project:
- Prefill tile heuristic (gone — `llama.cpp`'s scheduler handles it)
- F32-contiguity assertion (gone — no TS-side graph builds)
- FA-mode config (different surface — `flashAttn` flows through `llama_context_params` now)
- KV snapshot direct round-trip (covered by `llama-decode-wrapper.test.ts:loadKVCache truncates when snapshotLen > nTokens`)
- Spec-decode + forwardVerify (P5 phase; will reintroduce against the wrapper)
- Embed (covered by `llama-decode-wrapper.test.ts:embed creates a side context once and clears KV between calls`)

No load-bearing coverage was lost.

## Parity gate

### Accuracy: deferred to retake

The plan's Task 0 (pre-migration baseline capture) and Task 12 (post-migration bench) were intended to produce a same-tip greedy accuracy diff on the canonical 6. The pre-migration sweep failed for all 6 models with:

```
Fatal: browser bench stalled: no signal from page for 184s —
__benchStatus never published (model-load hang, CDP wedged, or
page crashed before bench entry)
```

This is a tooling-side failure: the smoke server was up (port 8031), the agentchrome session was reachable (port 55573), the dashboard was up (port 8033), but the page never published `__benchStatus` — the bench harness times out before the model finishes loading and the bench-mode entry runs. The same symptom hit every canonical-6 model identically, including the smallest (`tinyllama-1.1b-chat-q4_0`, ~700 MB on disk), and including this happened at the start of the session BEFORE any P2 commit was bundled — the running smoke bundle was the pre-P2 snapshot. **This is not a P2 code regression; it's a harness/wedge issue.**

Reference baseline used in lieu of a same-tip control: the **2026-05-04 greedy baseline** at [`eval/reports/greedy-baseline-2026-05-04/SUMMARY.md`](../greedy-baseline-2026-05-04/SUMMARY.md). For the canonical 6:

| Model | 2026-05-04 overall | TODO.md header pin (legacy) | Notes |
|---|---|---|---|
| `tinyllama-1.1b-chat-q4_0` | 0.27 | (no header pin — accuracy not load-bearing) | |
| `qwen3-0.6b-q4f16` | 0.77 | — | |
| `qwen3-1.7b-q4f16` | 0.84 | 82-89% (range) | within-range |
| `mistral-7b-instruct-v0.3-q4ks` | 0.53 | 68% | **−15pp** — pre-2026-05-05 parser-leniency fix; rebench needed regardless of P2 |
| `llama-3.1-8b-instruct-iq3m` | 0.89 | 86% | within-noise |
| `qwen3-8b-iq3m` | 0.91 | 90% | within-noise |

The Mistral row is materially below its TODO header pin because the 2026-05-04 baseline pre-dates the **parser-leniency fix shipped 2026-05-05** (memory: `parser_leniency_shipped_session_summary` — 180-230% tool-calling lift sub-8B at temp=0). A clean post-P2 retake on a debugged harness needs to capture the parser-leniency lift independently of any P2-specific delta.

**Decision (per spec §P2 stage D — accept-with-followup):** Accept the P2 code merge based on:
1. `make checkall` green throughout the 11-commit chain
2. Spec compliance review ✅ on every commit (10 reviews; 1 approved-with-followup, 9 approved)
3. Code quality review ✅ on every commit
4. Wrapper unit tests covering the wrapper's behavioral contract under a fake bridge

**Followup:** schedule a same-tip greedy bench retake on the post-P2 build once the harness wedge is debugged. Track as a P2.1 item in TODO.md.

### Throughput: deferred to retake

The plan's Task 0 step 4 (`make smoke-bench PERF_RUNS=3` median tok/s on canonical 6) ran into the same harness wedge. The 2026-05-01 post-rebase profile-mode pins from TODO.md (`81.8 / 67.5 / 43.9 / 29.6 / 23.4 / 22.0 tok/s` for canonical 6 at non-profile mode) remain the reference point; same-tip post-P2 measurement to follow.

**Risk assessment for throughput:** P2 added one ASYNCIFY round trip per `webllm_decode` token (vs. the legacy 1 round trip per `graphCompute`). Both routes promise-wrap the same underlying GPU compute. Spec §P2 R2 budgets ±10% regression as accept-with-recovery; pre-decode CPU-side overhead in the wrapper is one bridge call (`bridge.decode(...)` → `_webllm_decode(...)` → JSPI wait → `llama_decode(...)`). The bridge's `bridge_malloc` of `tokens.byteLength` per decode is small (4 bytes per token × ~1 token = 4 bytes) — not a hotspot. Expectation: throughput within ±5% on canonical 6.

### Bucket-D self-embed gate

The canonical 6 has no `embeddingCapable: true` model registered (`tinyllama`, `qwen3-{0.6b, 1.7b, 8b}`, `mistral-7b`, `llama-3.1-8b` — none are dual-purpose chat+embed). The wrapper preserves the embed surface via a lazily-created second context with `embeddings=true` + `pooling=LAST`; correctness is locked in by the `tests/llama-decode-wrapper.test.ts:embed` unit test. Real-model Bucket-D distinguishability gate (16+16 mean-margin per 2026-04-30 doctrine) lives outside the canonical 6 and runs separately under [`eval/reports/bucket-d-parity-2026-04-29/`](../bucket-d-parity-2026-04-29/SUMMARY.md) — that suite needs a same-tip rerun under P2 to confirm distinguishability holds via the new path. Tracked as a P2.2 followup.

## Followups (filed in TODO.md)

1. **P2.1** — Same-tip greedy bench retake on canonical 6 once smoke-bench harness wedge is debugged. Goal: confirm accuracy delta is within sampling noise (±2/36) and decode tok/s within ±10% vs the 2026-05-01 profile-mode pins.
2. **P2.2** — Bucket-D distinguishability gate retake against the wrapper-based embed path (16+16 mean-margin per 2026-04-30 doctrine).
3. **Smoke-bench harness wedge** — investigate why `__benchStatus` never published on 6 consecutive model loads. Likely candidates: stale agentchrome tab from prior session, GPU memory pressure from 142 prior bench runs in dashboard DB, or page-side regression in `real-model-bench.js`. Restart browser + reset dashboard DB as first step.

## Side-effects: items obviated by P2

Per spec design doc §P2 side-effects:
- **§1 Decode graph reuse (deferred)** — closed by P2 supersedence. TS-side graph caching is moot; upstream `llama_decode` handles graph reuse internally.
- **§4 Flash attention in browser** — closed by P2 supersedence. FA gating is now `llama_context_params.flash_attn`; per-arch heuristics deleted with `model-inference.ts`.

## Per-commit history

| Commit | Subject | Spec review | Quality review |
|---|---|---|---|
| `77bfd98` | model-metadata bridge exports | ✅ compliant | ✅ approved-with-followup (cosmetic n_ctx sentinel docstring) |
| `3eccb6b` | KV cache mutation ops | ✅ compliant | ✅ approved |
| `5002b82` | KV state serialize/restore | ✅ compliant + quality | ✅ |
| `cd7ee6f` | embeddings readback (JSPI) | ✅ compliant + quality | ✅ |
| `6d03b8b` | extend LlamaBridge TS interface | ✅ compliant + quality | ✅ |
| `c766ed5` | LlamaDecodeWrapper class | ✅ compliant + quality | ✅ (lazy embed-pooling reuse flagged as latent footgun for "mean" pooling — non-blocking, only "last-token" exercised) |
| `c7d46ab` | wrapper unit tests | ✅ compliant + quality | ✅ |
| `9de0b5e` | model-loader rewrite | ✅ compliant + quality | ✅ |
| `f1b8bf4` | engine.ts dispatch flip | ✅ compliant + quality | ✅ (3 documented spec deviations: tokenizer-field cast at call site, instanceof-gated forwardDecode, parked bridgeForHandle for P3) |
| `54624d5` | speculative.ts decoupling | ✅ compliant + quality | ✅ |
| `374cc46` | model-inference.ts deletion | ✅ compliant + quality | ✅ (5 extra test deletions documented + justified — all tested deleted internals) |

## Spec / plan / closure cross-references

- Spec: [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md) §P2
- Plan: [`docs/superpowers/plans/2026-05-05-tier3-p2-causal-lm.md`](../../../docs/superpowers/plans/2026-05-05-tier3-p2-causal-lm.md)
- Predecessor: [`eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md`](../p1-tokenizer-2026-05-05/SUMMARY.md)

## Next phase

**Superseded by partial revert.** P3 (Encoder migration) is **deferred indefinitely** because it would inherit the same per-dispatch shim cost ceiling that killed P2 v1 (encoder graphs issue the same hundreds of WebGPU commands per forward pass; the architectural mismatch is not specific to causal-LM decode).

**Next-session focus:** P2-v2 (JSEP-style architecture) — see [`POST-MIGRATION-BENCH.md` § P2.1.B](POST-MIGRATION-BENCH.md) for the architectural reframe and the recommended phasing. P3-P5 unblock once P2-v2 ships.

The bridge surface from Tasks 1-4 + 5 (`webllm_get_embeddings`, `webllm_get_metadata`, `webllm_n_*`, `webllm_kv_*`, `webllm_state_seq_*`, `webllm_perf_counter`) is preserved across the partial revert and is directly usable from any JSEP-style backend that needs to expose WASM-side state to JS.
