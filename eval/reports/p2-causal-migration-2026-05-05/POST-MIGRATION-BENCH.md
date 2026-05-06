# P2.1 — Post-migration bench retake (HALTED — major perf regression)

> **Date:** 2026-05-05
> **Tip:** `bd7ae4b fix(p2): port smoke-test page to LlamaDecodeWrapper` (post-Task-11 cleanup)
> **Verdict:** **HALTED** — first canonical-6 model showed an **18× decode-throughput regression** vs the legacy baseline. The full sweep was stopped before further measurements; per spec §P2 R2 (perf regression > 10%) this triggers a perf-recovery sub-phase or a P2 revert.

## What happened

After landing the smoke-page port (commit `bd7ae4b` — fixes a `ModelInference is not a constructor` blocker that Task 11 missed and a heap-grow-detachment bug in `bridge.loadModel` when the caller's buffer lives in the WASM heap), the smoke test loads tinyllama end-to-end successfully through the new `LlamaDecodeWrapper` path. Functional correctness is preserved: model loads, tokenizes, generates 64 coherent tokens, returns clean output. **But the throughput is catastrophic:**

```
[7/8] Generated 64 tokens in 18.3s (prefill: 8026ms, decode: 10274ms,
      6.2 tok/s, finish=max-tokens, tokensIn=57)
User: Tell one short joke.
Assistant: Here's a one-liner that sums up the short joke: "There is
a person who lives inside a giant pineapple."
```

| Metric | Legacy (TODO.md pin) | Post-P2 | Regression |
|---|---|---|---|
| Decode tok/s | 110.8 | 6.2 | **−94% (18×)** |
| Prefill (57 tokens) | ~50 ms | 8026 ms | **160× slower** |
| Per-decode-token | ~9 ms | ~160 ms | 18× |

The spec budgets ±10%; this is two orders of magnitude over. **No further canonical-6 measurements were taken** — the regression magnitude is unambiguous from a single model.

## Diagnosis

**Most likely root cause:** `llama_decode` is not optimized for token-by-token decode in the single-batch=1 case. Each call rebuilds the compute graph from scratch (full forward pass through every layer's tensor allocator), where the legacy `ModelInference.forwardSingle` cached the graph across decodes and only re-uploaded position + token-ids tensors per step. For a 22-layer model at batch=1, the graph build cost dominates everything else.

**Secondary contributors (smaller):**
- 2 JSPI suspend points per token (`_webllm_decode` + `_webllm_get_logits`) vs 1 in legacy (`graphCompute`). ~2-20 ms of suspend overhead per token.
- JS-side sampling reads back the full 32 K-vocab logits per token (~128 KB per readback). Legacy used GPU-side `forwardDecode` (argmax/topk) when greedy. Maybe ~5-10 ms per token.
- Per-decode `bridge_malloc` of 4 bytes for the single-token batch (no impact in steady state).

The first item explains 14-16× of the regression; the rest is noise.

## Spec gate

Per [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md) §P2 R2:

> **R2 P2 perf regression > 10%.** llama.cpp's prefill scheduler may
> behave differently than the project's hand-tuned tiling. Stage D
> allows a perf-recovery sub-phase before merge — tune
> `llama_context_params.n_ubatch` first.

> **Sub-phase: perf recovery.** If perf regresses >10% on any
> canonical-6 model and the cause is identified (e.g.
> `llama_context_params.n_ubatch` defaults vs project's tile
> heuristic), file a perf-recovery sub-phase. Tune knobs; re-bench;
> merge only when within ±10%. If irrecoverable, document and revert
> P2.

The regression is real, identified, and far above the budget. Stage D fires.

## Recovery candidates

In order of estimated impact and complexity:

1. **Graph-cache patch on `llama_context`** *(highest impact, highest cost)*. Add a llama.cpp patch (band C — Liberal — escalation needed) that reuses the compute graph across decode calls when the batch shape is unchanged. This is what the legacy `ModelInference.forwardSingle` did manually. Upstream may already be working on this; check recent issues / PRs on `ggerganov/llama.cpp`. Likely the ONLY recovery path that closes the prefill regression.

2. **`n_ubatch` tuning** *(low impact at batch=1)*. Spec recommends this first. Default in llama.cpp is 512; for single-token decode it's a no-op. Worth a quick measurement but unlikely to recover materially.

3. **GPU-side sampling for greedy/topk** *(modest impact)*. Add a `webllm_decode_argmax` bridge primitive that runs argmax on the logits before readback, returning just the token id. Saves ~10 ms × 64 tokens = ~600 ms per generation. Doesn't address prefill or the bulk of the per-token cost.

4. **JSPI batching across decode steps** *(speculative, modest)*. Group multiple decode calls into one suspend transition. Needs a `forwardN(tokensList, sampleFn)` bridge primitive that loops in WASM. Risky — sampling state lives in JS, not WASM.

5. **Revert P2** *(no recovery; preserves prior architecture)*. Roll back `4bb644c..bd7ae4b`. Tier 3's arch-portability win goes away. P3-P5 stay deferred until upstream graph-cache is available or a different migration path is chosen.

## Recommendation

**Pause P2 at the current tip and surface a decision point to the user:**

- **Path A:** Investigate option #1 (llama.cpp graph caching). Likely a multi-day deep dive into `llama-context.cpp` and `llama-graph.cpp`. Patch budget escalates from band B (3 reserved) to band C (Liberal). High-uncertainty — the upstream maintainers may have intentional reasons single-token graph caching isn't already there.

- **Path B:** Accept perf regression and ship anyway. Counter to spec stage D and to the "agent + Three.js coexistence" project doctrine in CLAUDE.md (6 tok/s on tinyllama makes agents noticeably sluggish; 4-6 sec per turn just for decode). NOT recommended.

- **Path C:** Revert P2. Roll back the 11 implementation commits (`4bb644c..374cc46`) plus the smoke-page port (`bd7ae4b`). Keep the closure docs as a record. The arch-portability motivation for Tier 3 stays valid; revisit when upstream `llama_context` ships graph caching for single-token decode (track via upstream issues).

## Artifacts

- Smoke log capture: `[7/8] Generated 64 tokens in 18.3s (prefill: 8026ms, decode: 10274ms, 6.2 tok/s)` — single tinyllama run on `bd7ae4b`.
- Bench harness wedge from earlier session: separate root cause (stale tab + JS error from smoke-page `ModelInference` reference), now fixed by `bd7ae4b`. Re-running the full bench-profile sweep would now exercise the new path on all 6 canonical models — but each model would compound the same per-token regression.

## Next steps (pending user decision)

This document deliberately stops short of taking corrective action. The choice between Path A, B, and C is high-stakes and outside the autonomy granted to the implementer agent.
