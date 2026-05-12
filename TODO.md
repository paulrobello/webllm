# WebLLM Project Status & Roadmap

> **Date:** 2026-04-27
> **Status:** End-to-end browser inference is working for both causal LMs and
> BERT-style encoders (Arctic-Embed). `make bench-full` now drives the
> generative profiles plus the embedding profiles end-to-end into the live
> SQLite-backed dashboard. The dashboard has a dedicated Embeddings section
> (cosine, latency, throughput) and the per-dimension / temperature-sweep /
> Accuracy×Speed charts split Qwen thinking-on vs thinking-off into
> distinct series. The eval suite separates chat-style semantic reasoning
> from true embedding-vector tasks. **The library is now the single source
> of truth for decode**: `engine.chatCompletion` drives [7/8], the
> interactive chat box, and bench mode through the same `Generator.generate`
> that public consumers hit. The parallel decode loop in
> `createSmokeCompletionRunner` is gone. Decode-mode selection is now
> per-step (greedy / topk / full) so steps without active steering state
> stay on the topk fast path even when the config configures steering.
> **Current canonical baselines (post-§32, `bun run eval/perf.ts
> --runs 3` non-profile, captured 2026-04-27/28).** The canonical 6
> are the ship-gate fleet for every rebase + sweep cycle (§27 / §32
> templates):
> - tinyllama-1.1b-chat-q4_0 (Q4_0): **110.8 tok/s decode**
> - qwen3-0.6b-q4f16 (actual: Q8_0): **89.8 tok/s decode**
>   (was ~17 — see `3e5be59`: CPU post-filter top-K replaced the
>   full-vocab readback + JS sampling pipeline that was costing
>   ~76 tok/s on Qwen3's 152K vocab)
> - qwen3-1.7b-q4f16 (actual: Q8_0): **62.2 tok/s decode**
>   (warmup-dominated at 17 tokens; longer 117-token runs land
>   higher steady-state)
> - mistral-7b-instruct-v0.3-q4ks (Q4_K_S): **35.0 tok/s decode**
>   (5-run median; first-run wall-time outlier excluded)
> - llama-3.1-8b-instruct-iq3m (IQ3_M): **27.2 tok/s decode**
>   (5-run median; -6.2% vs §27 baseline 29.0, accepted as final
>   per §32 — H1 "tied-embedding × #22456 aliasing" rejected at §32a)
> - qwen3-8b-iq3m (IQ3_M): **27.2 tok/s decode** (post-§27, +80% on
>   §16's pre-rebase 16.2 from upstream's #22344 fast i-quant
>   mat-vec kernels)
>
> Smaller-fleet pins (wave-1 / arch-survey, 2026-04-26):
> - smollm2-360m-q4f16 (Q4_0): ~106 tok/s decode (within noise of
>   TinyLlama at the same quant despite 3× fewer params — encode
>   overhead dominates at this scale; see §10 wave-1 entry)
> - qwen3-4b-q4f16 (Q4_0): 35.5 tok/s, **highest accuracy in
>   fleet** at 88-90% (§10 wave-1)
>
> Profile-mode pins for the canonical 6 (3-run median, perturbed
> -15 to -28% vs non-profile due to per-dispatch timestamp
> sampling). Refreshed 2026-05-01 post-§27 rebase (mul-mat
> vectorize fix #22578): **81.8 / 67.5 / 43.9 / 29.6 / 23.4 / 22.0
> tok/s**. Pre-rebase 2026-05-01 same-day same-tip control was
> 80.4 / 62.5 / 41.7 / 28.7 / 23.3 / 21.3 — every model improved
> or held; matmul time decreased uniformly -1.4% to -5.0%. See
> `eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md`.
> Earlier 2026-04-28 pre-rebase profile baselines (87.9 / 68.2 /
> 44.0 / 29.7 / 23.5 / 21.8) supersede only on cross-day deltas
> — environmental floor drift -3 to -8% over the 4-day gap, see
> `eval/reports/pre-rebase-baselines-2026-05-01/SUMMARY.md` for
> the drift table.
>
> Bench-full coverage at 1.7B landed (6 profiles · 3 off + 3 thinking)
> with overall accuracy 82–89% and per-profile decode (smoke chat
> regression, oneShot tok/s) 45.9–49.8. Smoke-regression numbers
> are lower than `perf.ts` steady-state due to harness overhead;
> compare against `perf.ts` for engine-throughput claims. Dashboard
> on port 8033 has all 6 dots in the accuracy×speed scatter.
>
> llama.cpp `webllm-browser-patches` rebased onto upstream master
> 2026-04-25 (carries `13d36cf89` FA browser unblock + `dd2914dc8`
> SSM_SCAN/set_rows changes). 10 patches now (added a row_norm
> codegen-stability split). No engine regression; FA path doesn't
> engage on Qwen3-1.7B decode shapes — see Active Step §5 for the
> diagnosis. Smoke page now runs a shader-cache warmup after [6/8]
> engine adoption to keep speed measurements steady-state across
> WASM rebuilds.
>
> Decode hotspot diagnostic landed 2026-04-26 (Active Step §6):
> matmul dequant-stub on both Q8 (Qwen3-1.7B) and Q4 (TinyLlama)
> moved `backendMatmulMs` by less than ±5.5% / ±2.5% with dispatch
> count unchanged — both kernels are **memory-bound, not
> compute-bound**. Follow-up src0-vs-src1 discrimination (Active
> Step §7) ran 2026-04-26: **src0 (weights) dominates**, src1
> (activations) is already L2-cached fine. Stub A moved matmul
> -0.8% Q4 / 0.0% Q8; Stub B moved matmul **-20% Q4 / -40% Q8**
> with decode +5.5% / +45%. `OUTPUTS_PER_WG` sweep (§8) confirmed
> OPW=4 locally optimal — bigger OPW only reduces src1 reads
> (already cached); doesn't address src0. Quant lever (§9) tested
> on Qwen3-1.7B: Q4_0 -11.8% matmul / +0.7% tok/s (in noise),
> Q4_K_M -5.8% matmul / -4% tok/s (regression — K-quant compute
> overhead claws back bandwidth savings). Reverted to Q8 baseline
> for dashboard continuity. **Net characterization: matmul ≈ 33%
> of decode time, bandwidth-bound fraction ≈ 40% of matmul on
> Q8 → theoretical ceiling for any pure-bandwidth lever ≈ 13%
> total decode.** Further matmul-kernel tuning is in diminishing-
> returns territory.
>
> **Pivot 2026-04-26: scope expansion to larger models.** Decode
> kernel tuning has bottomed out at the current model fleet (max
> 1.7B). The active priority is now **exercising the registered-
> but-untested 3B–4B models and registering 7B+ candidates with
> small quants** to characterize how the engine scales. See
> Active Step §10 below for the campaign plan. Subgroup-
> cooperative loading and FA-shape-routing are deferred behind
> the size-campaign work until we see how kernels behave at
> 3B+ scale (memory pressure, KV cache size, dispatch counts may
> reshape the profile in ways that change which lever matters).
>
> **Wave 1 complete (2026-04-26):** 7/10 done · 2 deferred ·
> 1 optional skipped.
> - smollm2-360m-q4f16: 106 tok/s steady-state Q4_0 / 24/36 (62%).
> - qwen2.5-1.5b-q4f16: 84 tok/s / 29/36 (81%). Run uncovered
>   bug #25 — qwen2 attention biases were silently dropped
>   (gibberish output, 4% pre-fix); fix lands `attn_{q,k,v}.bias`
>   loaders + opAdd in all 3 forward branches.
> - smollm2-1.7b-q4f16: 86 tok/s / 27/36 (74%). 24 layers, no
>   GQA, KV 1536 MB (largest). 31% faster than Qwen3-1.7B at
>   same params.
> - qwen2.5-3b-q4f16: 45 tok/s / 32/36 (86%). 36 layers, GQA
>   8:1. Bias path (#25) generalizes cleanly to 3B.
> - llama-3.2-3b-q4f16: 58 tok/s / 27/36 (76%). 28 layers, GQA
>   3:1, KV 896 MB. 29% faster than qwen2.5-3b at same scale.
>   Encode overhead 15.5% — new fleet low.
> - hermes-3-llama-3.2-3b-q4f16: 60 tok/s / 27/36 (74%).
>   Identical arch to base llama-3.2-3b (572 dispatches/token,
>   exact match); tool-calling advantage invisible at warm
>   temp (gate at 0.4). Sanity-check entry confirming the
>   fine-tune doesn't change inference cost profile.
> - **qwen3-4b-q4f16: 35.5 tok/s / 32-33/36 (88-90%).** 36 layers,
>   GQA 4:1, KV 144 MB, 805 dispatches/token (matched §10
>   prediction within 1%). **Highest accuracy in fleet** (90%
>   thinking-on; 88% off — beats prior leader qwen2.5-3b's 86%).
>   Required new GGUF-streaming-into-WASM-heap loader path
>   (Completed 2026-04-26 §11) to fit through Chrome's 2 GiB
>   single-allocation cap.
> - DEFERRED: gemma-2-2b (pre+post norm pairs, logit/attn
>   soft-cap, sliding-window, (1+w) RMSNorm), phi-3.5-mini
>   (fused QKV).
>
> Cross-family speed/accuracy pattern is now confirmed across
> the full 0.6B → 4B span: **Llama family fastest/lower-
> accuracy → Qwen family slower/higher-accuracy**. Speed delta
> tracks dispatch count (layer count + arch-specific extras),
> not param count. Within Qwen, accuracy keeps climbing into
> 4B (qwen3-1.7B 82-89% → qwen3-4B 88-90%).
>
> **Loader / parser refactor 2026-04-26 (Completed §11):**
> smoke loader streams GGUF directly into WASM heap; parser
> API takes `Uint8Array` (sub-view aware); `loadWeights`
> accepts a callback source for HEAPU8-backed bytes;
> `uploadRangeChunked` re-derives source views per-chunk after
> scratch malloc to survive heap-grow detachment;
> `ctxCreate` memSize no longer over-allocates by
> `ggufCtx.totalDataSize` (pre-existing bug — `no_alloc=true`
> means that buffer was never used). Unblocks all wave-2
> 7B+ candidates; reclaims multi-GB headroom across all sizes.
>
> **Wave 2 complete 2026-04-26 (Completed §12, §13, §15, §16):** 4/4 done.
> - **mistral-7b-instruct-v0.3-q4ks**: 34.4 tok/s steady-state
>   / 26/36 (68%). 32 layers, GQA 4:1, KV 1024 MB, 650
>   dispatches/token, matmul 47.0% of graph (~45% of decode).
>   First non-Llama/Qwen entry. Q3_K_M tried first; gibberish
>   output uncovered **bug #28: Q3_K matmul shader has a
>   correctness bug.** Wave-1 never exercised Q3_K (all Q4_0);
>   §9 tested Q4_K_M only. Q4_K_S workaround works.
> - **llama-3.1-8b-instruct-iq3m**: 16.3 tok/s steady-state /
>   31/36 (86%). 32 layers, GQA 4:1, KV 1024 MB, 652
>   dispatches/token (matches Mistral 7B), matmul 71.4% of
>   graph (new fleet high — ~69% of decode). First 8B in
>   fleet. Llama-3.1-8B Q4_K_S exceeds 4 GiB WASM cap; pivoted
>   to IQ3_M (3609 MB) via the IQ-family code path (verified
>   coherent first on Mistral IQ4_XS). 86% accuracy ties
>   qwen2.5-3b, 18 points above Mistral 7B Q4_K_S — quant
>   compute cost halves throughput vs Mistral but model-
>   quality recovers most of the accuracy gap.
> - **mistral-7b-instruct-v0.3-q3km** (§15, bug #28 fix
>   verified under sustained load): 19.7 tok/s steady-state
>   / 26/36 (69%). Same Mistral base as Q4_K_S; 43% slower
>   despite 15% smaller file (3360 vs 3953 MB). K-quant
>   compute overhead defeats bandwidth savings — tied
>   accuracy with Q4_K_S. **Q3_K is not a useful speed
>   lever even when the file fits the cap;** Q4_K_S
>   remains the throughput default at 7B.
> - **qwen3-8b-iq3m** (§16, wave 2 closer): **16.2 tok/s
>   steady-state / 33/36 (90%) off / 33/36 (90%) on.** 36
>   layers, GQA 4:1, KV 1024 MB, **805 dispatches/token
>   (matches qwen3-4b — qwen3 architecture-invariant)**,
>   matmul 66.7% of graph (~65% of decode). **Effectively
>   tied with llama-3.1-8b on speed** despite +23%
>   dispatches — bandwidth-bound matmul flattens dispatch
>   overhead. **Top accuracy in fleet** (ties qwen3-4b
>   thinking-on at 90%); Qwen3 family quality advantage
>   holds at 8B (+4 points over Llama-3.1-8B IQ3_M).
> - **Net wave-2 finding:** at scale the §A subgroup-
>   cooperative-loading ceiling rises sharply
>   (4B Q4_0: ~13% → 7B Q4_K_S: ~18% → 8B IQ3_M: ~26-28%
>   of decode time). The lever's percentage of total
>   decode keeps growing with scale, regardless of
>   family. **Quant compute cost is the load-bearing
>   throughput axis at 7B+; family is a quality knob**
>   (Q4_K_S → IQ3_M halves speed; same-quant family
>   swap moves throughput <1%).
>
> **§A closed 2026-04-26 (Completed §17):** lever 1
> (THREADS_PER_BLOCK 4→2 in `mul_mat_vec.wgsl`) measured
> against the canonical 4-baseline. Only TinyLlama Q4_0
> benefited (sub-trigger -2.9% matmul / +0.6% tok/s,
> noise). The ~26-28% wave-2 ceiling estimate above was
> an upper bound on a lever that turned out to be
> **structurally inapplicable to the 7B+ fleet**: Q4_K_S
> (Mistral) is a K-quant with TPB=16 and a different
> block layout, explicitly excluded from §A's lever-1
> design; IQ3_M (both 8Bs) has no `mul_mat_vec.wgsl`
> path and routes through general `mul_mat.wgsl` instead.
> Levers 2 (vec4-packed loads) and 3 (`d`-scale lifting)
> face the same applicability constraint. Shader
> reverted; no patches landed.
>
> **§C closed 2026-04-26 (Completed §19):** drafter
> speculative decoding measured at K=4 on
> qwen3-8b-iq3m via qwen3-0.6b-q4f16 → 0.20× ratio
> (3.0 vs 15.3 tok/s baseline). Verify-readback (4 ×
> 152K logits = ~2.4 MB/step) plus K full-vocab
> drafter forwards dominate. Output is functionally
> correct (Leviathan sampling preserves the target
> distribution); the lever is just paying readback
> overhead it can't earn back at v1's CPU-side
> verify. Engine routing reverted (`aac7080`); driver,
> sampler helpers, `forwardVerify`, `truncateKVCache`,
> 19 tests, and smoke/Makefile plumbing remain in
> tree behind a "reserved in v1" throw.
>
> **§4 reclosed 2026-04-26 (Completed §20):** FA revisit
> at prefill / long-decode scope re-landed `ggml_flash_attn_ext`
> behind a `flashAttn?: boolean` config gate (default off)
> with F16 K + F16 V cache and a long-prompt harness.
> Branch `feat/fa-revisit-prefill-long-decode` fast-forward
> merged to `main` (top commit `b872b5f`). **6 of
> 32 planned cells captured.** TinyLlama 1.1B Q4_0 covered
> full 4-cell matrix: FA wins everywhere (short-short -6.6%
> TTFT / +4.9% decode; long-short -10.0% TTFT / +16.4%
> decode). Mistral-7B Q4_K_S short-short FA-on -3.3% (still
> over the 3% gate; better than §18's -5.8% thanks to F16 KV).
> Mistral long-short + 8B models blocked at
> `backend_alloc_ctx_tensors` — a generic WebGPU
> max-buffer-binding limit at 32 layers × seq=512 (independent
> of FA mode). **Decision: close §4 again, but keep the gate
> as opt-in infra** rather than fully reverting like §18 —
> the TinyLlama win is real and worth preserving. Next lever
> with meaningful headroom is **§C v2 GPU-resident verify**,
> or **7B+ long-prefill graph-buffer infra** to unblock the
> measurements that §4 still can't reach at scale.
>
> **§4 closed 2026-04-26 (Completed §18):** `ggml_flash_attn_ext`
> integrated into all three attention branches (MLA/GQA/MHA)
> with F16 KV cache and transposed V layout. Measured on the
> canonical 4-baseline: FA engaged on all 4 models (dispatch
> counts -10-13%, matmul -2 to -16%), but the new
> `backendAttentionMs` overhead (1.3-3.3 ms/step) exceeds
> the savings at single-token decode (N=1). Mistral-7B
> regressed -5.8% (blocking — exceeds 3% gate); no model
> gained ≥2%. FA's main wins are prefill (long prompts) and
> longer decode batches (>256 tokens) — neither is exercised
> by the bench-inf gate. Bridge wrappers (`33f10eb`), TS
> bindings (`4692bce`+`d26d736`), and surface test (`068ef84`)
> retained as future-work infrastructure; implementation
> reverted via `git checkout 068ef84 --`. **Next lever
> (post-§20):** §C v2 with GPU-resident verify (skips the
> 2.4 MB / step readback that sank §C v1), or 7B+
> long-prefill graph-buffer infra to unblock the §4
> measurements that the buffer-binding limit prevents
> at scale (see §20 closure for details).
>
> **§C-v2-A closed 2026-04-27 on side branch
> `feat/spec-decode-v2-greedy`:** greedy spec-decode + GPU-
> resident K+1 verify (`forwardVerifyArgmax`, 16 B/step
> readback vs v1's 2.4 MB) measured against the §22.5 ship
> gates. Gate 1 (high-α speedup ≥1.5×): **0.36×** (5.7 vs
> 16.0 tok/s on `qwen3-8b-iq3m` × `qwen3-0.6b-q4f16` K=4).
> Gate 2 (low-α safety ≥0.95× on `creative-low-alpha` /
> 200 tokens): **0.78×** (12.7 vs 16.2 tok/s). AdaptiveGate
> sticky disengage works (no sustained gate-1-style collapse)
> but per-step verify overhead at this scale (drafter 4×~12 ms
> ≈ 48 ms + verify K+1 ≈ 70-80 ms = ~120 ms/step) caps even
> the perfect-accept ceiling at ~33 tok/s vs 16 tok/s baseline
> — and measured α ≈ 0.2-0.25 inverts the trade. Output
> coherent on both gates (finish=stop-token). Branch retained
> as side-branch infra; **do not merge to `main`** —
> resurrection only worth it once a much larger target lands
> (70B+ via MEMORY64) or 7B+ long-prefill graph-buffer work
> cuts per-step verify cost. Next lever on `main`: 7B+ long-
> prefill graph-buffer infrastructure (§20's deferred
> dependency for FA-at-scale).
>
> **Plan files:** `docs/superpowers/plans/2026-04-20-webllm-implementation.md` (Phase 1)

---

## Project Constraints

The workflow policies that gate every change on this project —
**8B model-size ceiling** (revised 2026-04-29 from prior 30B),
**16 GB / 32 GB / 128 GB hardware baseline**, **single-model-active
deployment**, **per-binding 128 MiB cap doctrine** (hybrid quant for
vocab-too-big-for-cap models), **quick-wins override on YAGNI**,
**probe-first default**, **complexity ≠ implementation time**, and
**always commit before work** — live in
[`CLAUDE.md`](CLAUDE.md#workflow-policies-set-2026-04-28) so they
apply to all sessions, not just ones that load TODO.md. Read that
section before starting any new work; entries below cite the policies
(e.g. "deferred under the 8B ceiling") without re-stating them.

**Use case anchor (set 2026-04-29):** the project's load-bearing
deployment is **agent + Three.js coexistence in a single browser
tab** — a small chat model drives 3D agents alongside a renderer.
This anchors the 8B ceiling, the hardware baseline, the single-
model-active doctrine, and the bucket D "chat-model self-embedding"
direction.

---

## Project Milestones

### Completed

- [x] GGUF parser for model metadata + tokenizer
- [x] SPM + BPE tokenizer (encode/decode, ▁ normalization, byte fallback)
- [x] Emscripten WASM build for ggml-webgpu backend
- [x] Full transformer forward pass (embedding, attention, FFN, RMSNorm)
- [x] Multi-template chat formatting (llama2, chatml, gemma, phi3, llama3, mistral-v7, zephyr)
- [x] Auto-prepend default system message for models without one
- [x] Multi-turn chat in browser (KV cache reset + full prompt reformat)
- [x] Sampling (temperature, top-k, top-p, repetition penalty)
- [x] KV cache for incremental decoding
- [x] Decode profiling pipeline (`eval/perf.ts`, `make bench-inference`)
- [x] Performance optimizations (items 2, 3, 5, 7, 8, 9, 11 below)
- [x] GPU-side ARGMAX/TOP_K logits reduction (item 11)
- [x] `make smoke-bench` end-to-end benchmark target with agentchrome
- [x] Semantic-reasoning eval dimension split from true embedding-vector tasks
- [x] Live benchmark dashboard migrated to Chart.js with richer comparison charts
- [x] Model support roadmap documented in `docs/MODEL_SUPPORT.md`
- [x] Public streaming APIs (`generateStream`, `chatCompletion`) wired through `Generator` + `InferenceSession`
- [x] Encoder forward pass for BERT-style embedding models (`WebLLM.embed()`, Arctic-Embed-s/m)
- [x] `make bench-full` exercises arctic-embed profiles end-to-end alongside generative profiles
- [x] Dashboard "Embeddings" section (cosine per task, median latency, throughput)
- [x] Charts that key on `modelId` now split Qwen thinking-on vs thinking-off into distinct series

---

## Cumulative Bug Fix History

The full bug-fix history (28 entries through 2026-04-26) lives in
`TODO_ARCHIVE.md`. Recent fixes (2026-04-27 onward) are documented
inline in their cycle's "Completed on" entry below.


## Debug Tools

`src/inference/model-inference.ts` has instrumented debug helpers. From the browser console on `smoke-test/real-model.html`:

```js
await window.inference.debugReadEmbeddingRow(1);  // BOS

window.inference.resetKVCache();
await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
await window.inference.debugReadKCache(0, 64*4, 0);
await window.inference.debugReadVCache(0, 64*4, 0);

await window.inference.debugReadNormWeight("attn0", 8);

await window.inference.debugLayerOutput(
  22172, 0, "layer_output"
  // or: "pre_attn" | "attn_normed" | "attn_q" | "attn_k" | "attn_v"
  //     | "attn_out" | "post_attn" | "ffn_normed" | "ffn_gate" | "ffn_up"
  //     | "ffn_hidden" | "ffn_out"
);
```

`smoke-test/real-model.html` stashes `window.{inference, tokenizer, parsedModel}` for console use.

---

# Inference Performance Optimizations

Baseline (pre-optimization): ~44 tok/s decode, ~130 ms prefill on TinyLlama 1.1B
Q4_0 via Emscripten WebGPU in-browser.

**Steady-state decode baselines (2026-04-25, non-profile, realistic
sampler):** TinyLlama Q4_0 ~107 tok/s · Qwen3 0.6B thinking-off ~83
tok/s · Qwen3 0.6B thinking-on ~17 tok/s · Qwen3 1.7B thinking-on
~66 tok/s · (Qwen3 1.7B thinking-off measured at ~59 tok/s but on a
17-token run that's warmup-dominated; the thinking-on number is the
clean steady-state at this size). These are the canonical numbers to
compare new perf work against.

**Quant caveat:** model IDs `qwen3-*-q4f16` actually resolve to
`Qwen3-*-Q8_0.gguf` files (610 MB at 0.6B, 1.7 GB at 1.7B) because no
`ggufFilePattern` is pinned and the picker fallback hits Q8 first. So
within-Qwen3 comparisons across sizes are clean (both Q8), but
TinyLlama-Q4 vs Qwen3-Q8 absolute matmul ms cross two variables —
read matmul *fraction of graph time* across families, not absolute ms.
Cleanup item: pin `ggufFilePattern: "Q4_K"` (or similar) on the qwen3
entries in `eval/models.ts` if Q4 comparisons become load-bearing.

**Profile-mode hotspot ranking (2026-04-25, `--profile`).** Captured
through `make smoke-bench PERF_RUNS=3` after fixing three latent
harness bugs (`953c560`) that had been masking trace collection since
the consolidation landed. `--profile` enables ggml-webgpu's detailed
backend timing and perturbs throughput measurably:

| Profile          | Steady-state | Profile-mode | Perturbation |
|------------------|--------------|--------------|--------------|
| TinyLlama Q4_0   | ~107 tok/s   | 76 tok/s     | -29%         |
| Qwen3-0.6 off    | ~83 tok/s    | 55 tok/s     | -34%         |
| Qwen3-0.6 on     | ~17 tok/s    | 14.6 tok/s   | -14%         |
| Qwen3-1.7 off    | ~59 tok/s*   | 41.5 tok/s   | -29%         |
| Qwen3-1.7 on     | ~66 tok/s    | 43.2 tok/s   | -34%         |

\* warmup-dominated 17-token run; trust the thinking-on number for the
clean 1.7B steady-state.

Cite steady-state numbers for throughput claims; cite profile-mode
ratios for hotspot ranking. Mixing the two is what derailed the
"Apr-23 regression" investigation.

Per-step decode breakdown by mode (medians from the runs above):

| Bucket (ms/step)        | TinyLlama topk | Qwen3-0.6 topk | Qwen3-0.6 full | Qwen3-1.7 topk-off | Qwen3-1.7 topk-on |
|-------------------------|----------------|----------------|----------------|--------------------|-------------------|
| graphComputeMs          | 11.67 (88.8%)  | 13.91 (88.8%)  | 22.62 (94.5%)  | 19.10 (93.6%)      | 19.00 (92.8%)     |
| downloadResultMs        |  1.20 ( 9.1%)  |  1.37 ( 8.8%)  |  0.93 ( 3.9%)  |  1.00 ( 4.7%)      |  1.10 ( 5.6%)     |
| build + alloc + upload  |  0.26 ( 2.0%)  |  0.36 ( 2.3%)  |  0.35 ( 1.5%)  |  0.36 ( 1.7%)      |  0.32 ( 1.5%)     |
| **totalMs**             | **13.14**      | **15.66**      | **23.93**      | **20.40**          | **20.60**         |

Backend attribution (% of `graphComputeMs`, profile mode only):

| Field                    | TinyLlama topk | Qwen3-0.6 topk | Qwen3-0.6 full | Qwen3-1.7 topk-off | Qwen3-1.7 topk-on |
|--------------------------|----------------|----------------|----------------|--------------------|-------------------|
| backendMatmulMs          | 3.85 (33.0%)   | 4.05 (29.1%)   | 6.31 (27.9%)   | 6.68 (33.8%)       | 6.88 (34.2%)      |
| backendEncodeOverheadMs  | 2.71 (23.2%)   | 4.07 (29.2%)   | 3.53 (15.6%)   | 4.20 (22.4%)       | 3.90 (19.9%)      |
| backendAttentionMs       | 0.40 ( 3.5%)   | 0.49 ( 3.5%)   | 0.75 ( 3.3%)   | 0.46 ( 2.6%)       | 0.52 ( 2.7%)      |
| backendDispatchCount     | 450/token      | 629/token      | 619/token      | 629/token          | 629/token         |

Headlines:
- **Decode is graph-compute-bound across every profile** (~89–95% of
  step time). Readback is a 4–9% slice; further async-readback work
  has tiny headroom.
- **Dispatch count is architecture-invariant within Qwen3** (629/token
  for both 0.6B and 1.7B topk paths). The earlier "graph-shape reduction
  has more leverage on Qwen3" framing was right about the dispatch
  delta vs TinyLlama (629 vs 450); what we now know is that *encode
  overhead's absolute cost is ~flat* (4.07 → 4.20 ms across the 2.83×
  param jump). Its *fraction* of graph time drops with model size
  (29.2% → 22.4%) — i.e. the encode lever has *diminishing*
  trajectory at scale.
- **Matmul scales sub-linearly with parameter count** within Qwen3:
  4.05 → 6.88 ms (1.7×) for 2.83× params, consistent with bandwidth-
  bound GEMV on Q8 weights. Matmul share grows from 29.1% → 34.2%, so
  matmul kernel work has *growing* trajectory at scale and is now the
  clear lead bucket on Qwen3-1.7B (33–34%) by ~12pp over encode.
- **§2's topk fix holds at scale.** Thinking-on at 1.7B routes 342/348
  steps (98%) through topk; the remaining 6 full-path steps add ~3 ms/
  step but contribute negligibly to wall time. Cost-per-token is at
  near-parity with thinking-off at this size.
- **Decision criteria don't cleanly hit the >40% / >25% / ~30%
  thresholds** in TODO §4. Matmul 33–34% is below the >40% rule;
  encode 20–22% is below the >25% rule; "balanced ~30%" is closest
  but matmul leads by a margin. The recommendation below is a
  trajectory call (encode flat-absolute, matmul still scaling), not
  a threshold match.
- **Consolidation tightened TinyLlama dispatches and matmul share**
  vs the stale 2026-04-22 numbers (489 → 450 dispatches/token, matmul
  share 40.4% → 33.0%). Treat that as a quiet consolidation win, not
  a headline.

Items in rough order of expected impact. Each entry explains the idea, where
the code lives today, the expected win, and the risk/tradeoff.

---

## High impact

### 1. Decode graph reuse (deferred)
- **Where**: `src/inference/model-inference.ts::forward()`
- **Today**: every call to `forward()` calls `ctxCreate`, builds the full
  ~440-op graph, calls `backendAllocCtxTensors`, uploads leaf inputs, runs
  compute, then `ctxFree`s. For a decode step with `nTokens=1` the graph
  shape is identical across steps.
- **Change**: cache the graph for `nTokens=1` decode. Only update:
  - leaf inputs (`posTensor`, `tokenIdsTensor`, mask row for current position)
  - the K/V cache offsets (encoded into the graph nodes via `pastLen * kNb1`
    constants, so this needs either graph reuse with runtime offset or
    rebuilding just the KV views each step)
- **Expected**: 2–5× decode throughput. The JS-side graph construction +
  WASM asyncify round trip is currently a meaningful chunk of per-step cost.
- **Risk**: the ggml graph stores absolute offsets inside view tensors.
  Reusing the graph requires either:
  - Adding a C-side helper that mutates view tensor offsets in place, or
  - Refactoring KV cache layout so writes always go to a fixed slot and the
    "real" position is a permutation applied separately, or
  - Pre-building graphs for every possible past-length (memory hungry)
- **Profile measurement (2026-04-21):**
  - mean total per decode step: 16.75 ms
  - ctxCreate: 0.00 ms (0.0%)
  - buildGraph: 0.21 ms (1.3%)
  - backendAlloc: 0.05 ms (0.3%)
  - uploadLeaves: 0.02 ms (0.1%)
  - graphCompute: 6.92 ms (41.3%)
  - downloadLogits: 9.53 ms (56.9%)
  - teardown: 0.02 ms (0.1%)
  - Non-GPU overhead (ctxCreate + buildGraph + backendAlloc + teardown) = 1.7%.
- **Status**: deferred. Big structural change relative to its current
  expected win at 17.5 ms/step. Revisit if we can measure that graph
  building (not GPU compute) is actually the bottleneck.
- **Phase A skipped:** GPU compute + logits download dominate; moving graph
  build to C can at best claw back ~1.7% and isn't worth the C-side
  maintenance burden.

### 2. Re-enable batched compute passes in the WebGPU backend ✅ DONE
- **Where**: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
- **Fix**: flipped `batch_compute_passes` to `true`. Safe now that
  overlap-only conflict detection (item 3) doesn't schedule inter-dispatch
  CopyBufferToBuffer calls for most ops.
- **Actual gain**: marginal on top of item 3 (most of the 33% came from
  item 3). Still worth keeping for larger models where driver overhead
  of many compute passes would dominate.

### 3. Refine buffer-conflict detection (overlap-only) ✅ DONE
- **Where**: `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
  `__EMSCRIPTEN__` block in `ggml_backend_webgpu_build_multi`.
- **Was**: created a temp GPU buffer for EVERY same-buffer-different-offset
  binding pair, even when the ranges didn't actually overlap. WebGPU's
  validation only rejects overlapping ranges, not disjoint ones.
- **Fix**: added an `overlap` check (`!(a_end <= b_start || b_end <= a_start)`)
  before creating a temp buffer. Most attention/FFN op bindings turned out
  to be disjoint slices of the shared graph buffer.
- **Actual gain**: **+28% decode throughput** (44 → 56 tok/s). The bulk of
  the original overhead was unnecessary `CopyBufferToBuffer` calls.

---

## Medium impact

### 4. Enable flash attention in the browser 🟡 UPSTREAM UNBLOCKED 2026-04-25
- **Where**: `ggml-webgpu.cpp::ggml_backend_webgpu_device_supports_op`
  under `GGML_OP_FLASH_ATTN_EXT` — currently `#ifndef __EMSCRIPTEN__`
  on our patch branch (rebase point `68f1738d5`).
- **Was blocked on**: emdawnwebgpu does NOT expose
  `wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix`; llama.cpp's
  original FA shaders required subgroup-matrix.
- **Unblock**: upstream commit `13d36cf89` ("ggml-webgpu: enable
  FLASH_ATTN_EXT on browser without subgroup matrix", #22199, merged
  2026-04-24) adds a tile / vec FA fallback that does NOT require
  subgroup-matrix. Path forward is now a rebase, not waiting on
  Chrome.
- **Expected gain (decode):** modest. Decode-mode attention is N=1,
  so the [seq, seq] matrix that FA avoids materializing is tiny. The
  per-token win comes from fewer dispatches (likely 3–5 fewer per
  layer, ~100/token on Qwen3) reducing encode overhead by ~0.5–1 ms,
  plus a small `backendAttentionMs` saving. Realistic ceiling at
  decode: 3–5%. Not a 10% lever on its own.
- **Expected gain (prefill):** substantial. FA's main win is for
  long-prompt prefill where attention matrix size scales with seq².
  If prefill latency / TTFT becomes a target, this is the lever.
- **Action**: rebase the patch branch to a point at-or-after
  `13d36cf89`, drop the `#ifndef __EMSCRIPTEN__` guard, re-verify
  smoke + perf, capture the dispatch-count delta on Qwen3-1.7B as
  the headline measurement. Rebase carries non-trivial maintenance
  cost (~9 patches; see `docs/LLAMA_CPP_PATCHES.md`).

### 5. Fused SwiGLU op ✅ DONE
- **Where**: `src/inference/model-inference.ts` FFN section.
- **Actual gain**: +1–2% (58 → ~58.5 tok/s). Modest — FFN compute is
  dominated by the three mul_mats, not by silu/mul. Kept for cleanliness.

### 6. F16 KV cache ❌ NET LOSS AT SHORT CONTEXT
- **Tried**: switched K and V to `F16`. WebGPU backend handled F16×F32
  mul_mat correctly — no correctness regression.
- **Measured**: **-7.7%** decode throughput (55.3 tok/s vs 59.9 baseline).
  F16×F32 mul_mat not as fast as F32×F32 for small matrices, and F32→F16
  conversion on every KV write adds dispatch overhead at short contexts.
- **Reverted**: code stays F32. Reconsider for long-context workloads
  (1000+ tokens) where bandwidth savings on attention reads will overtake
  write-path overhead.

---

## Low impact / polish

### 7. Skip redundant `opCont` calls ✅ DONE
- **Where**: K/V cache writes in `forward()`.
- **Actual gain**: +0–2% on top of item 5. Still worth keeping: fewer
  dispatches, cleaner code.

### 8. Skip mask tensor for nTokens=1 decode ✅ DONE (partial)
- **Where**: `forward()` mask allocation + upload + softmax_ext call.
- **Actual gain**: flat. Still a cleanup. Pre-allocating mask at
  `initKVCache` time deferred.

### 9. Reduce JS↔WASM boundary crossings ✅ DONE
- **Where**: `src/wasm/webgpu-bridge.cpp` + `src/inference/ggml-wasm.ts`
  + `forward()` in `model-inference.ts`.
- **Change**: added `backend_tensor_set3` in C bridge — single bundled WASM
  call for pos + ids + mask instead of 2–3 separate hops.
- **Actual gain**: **+5–7% decode throughput** (55.6 → 58.7 tok/s median).

### 10. Benchmark the current pipeline ✅ DONE
- **Where**: `eval/perf.ts` + `make smoke-bench` + `make bench-inference-save`.
- **Current Task 5 profiled investigation baseline**: 93.5 tok/s on the
  profiled TinyLlama-1.1B Q4_0 browser run (`PERF_RUNS=3`), with median-run
  wall time 2027 ms, `graphComputeMs` mean 9.96 ms, and `downloadResultMs`
  mean 0.62 ms. Read this as a profiling baseline for hotspot ranking, not as
  the new steady-state browser throughput baseline.
- `make smoke-bench` — end-to-end: builds WASM+JS, starts server, launches
  agentchrome (headed), runs 3 perf iterations with `--profile`, cleans up.
  All smoke targets (`smoke-serve`, `smoke-open`, `smoke-run`, `smoke-bench`)
  depend on `smoke-test` for fresh builds.

### 11. GPU-side ARGMAX/TOP_K logits reduction ✅ DONE (negligible gain)
- **Where**: `src/wasm/webgpu-bridge.cpp` (C bridge), `src/inference/ggml-wasm.ts`
  (TS bindings), `src/inference/model-inference.ts::forwardDecode()`,
  `src/inference/generation.ts` (decode loop routing),
  `src/inference/sampler.ts::sampleFromTopK()`,
  `src/core/engine.ts` (wiring), `smoke-test/real-model.html` (both code paths).
- **What**: Added `ggml_argmax` and `ggml_top_k` to the WASM bridge. New
  `forwardDecode()` method builds the same transformer graph but appends
  ARGMAX/TOP_K tail ops, downloading 4 bytes (greedy) or k×8 bytes (topk)
  instead of 128KB (32K×float32) full logits. Generation loop auto-selects
  mode: greedy (temp=0, no penalty), topk (topK>0), or full (fallback).
  Smoke test step 7 and chat handler both use the greedy path.
- **Actual gain**: **+0.5%** (58.7 → 59.0 tok/s). Negligible.
- **Why**: The readback bottleneck is synchronization latency, not data size.
  At the time of measurement, `downloadFromTensor()` still paid queue/map wait
  latency that dominated the ~9.5ms readback regardless of whether the payload
  was 4 bytes or 128KB. Reducing data size only saved the final memcpy
  (~0.01ms for 128KB).
- **Follow-up completed**: the browser stack now has a real request-based async
  readback path:
  - `~/Repos/llama.cpp/ggml/include/ggml-webgpu.h`
  - `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp`
  - `src/wasm/webgpu-bridge.cpp`
  - `src/inference/ggml-wasm.ts::downloadFromTensor()`
  This adds backend `begin / poll / finish / cancel` support, wires it through
  the WASM bridge, and uses heap allocation safely across async boundaries.
- **Current status**: correctness/integration is fixed, browser smoke passes,
  and the latest profiled run shows the completion-driven path has reduced
  `downloadResultMs` to a 0.64 ms mean. Readback is now a smaller slice of
  decode latency than the rest of the step.
- **Infrastructure value**: The ARGMAX/TOP_K bridge functions, `forwardDecode()`,
  generation-loop routing, and the new async readback request API are all useful
  foundations for the next round of latency-hiding work. The
  `sampleFromTopK()` sampler method enables CPU sampling on GPU-reduced
  candidate sets for temperature > 0.

---

## Dashboard & Visualization

Live bench dashboard at `smoke-test/dashboard.*` (served by
`eval/live-server.ts` on port 8033). Each section below is independently
shippable on top of the existing eval/run data — no new bench metadata
needs to be collected.

### 12. Convert existing charts to a proper charting library ✅ DONE
- **Where**: `smoke-test/dashboard.js`, `smoke-test/dashboard.html`,
  `smoke-test/dashboard.css`, `smoke-test/vendor/chart.umd.min.js`.
- **Done**: introduced self-hosted Chart.js 4 UMD and converted the main
  dashboard charts to managed Chart.js instances with dark-theme colors,
  legends, tooltips, and dynamic chart-host sizing.
- **Follow-up**: use `make vendor-refresh` after bumping `chart.js` to refresh
  the vendored browser bundle.

### 13. Accuracy × Speed scatter chart ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderScatterChart`,
  `smoke-test/dashboard.html`.
- **What**: one dot per profile. X = mean tok/s, Y = eval `overall`.
- **Answers**: "which profile should we actually ship?"

### 14. Per-dimension grouped bars per model ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderDimGroupedChart`,
  `smoke-test/dashboard.html`.
- **What**: one row per model (cold profile only); grouped bars for
  `tool-calling / reasoning / instruction-following / semantic-reasoning`.
- **Answers**: "which model do I pick for workload X?"

### 15. Temperature sweep per dimension ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderTempSweepChart`,
  `smoke-test/dashboard-charts.js`, `tests/dashboard-charts.test.ts`.
- **What**: per (model, dimension), cold / warm / hot grouped bars.
- **Answers**: "is temperature hurting me on dimension X?"
- **Regression covered**: the hot bucket now has explicit data + color coverage
  so it cannot disappear silently.

### 16. Thinking on vs off delta (Qwen) ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderThinkingDeltaChart`,
  `smoke-test/dashboard.html`.
- **What**: two-bar pairs per dimension comparing Qwen thinking off/on at
  matched temperature.
- **Answers**: "is thinking worth the extra decode time, and on which
  dimensions?"

### 17. Time-to-first-token (prefill latency) chart ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderTtftChart`,
  `smoke-test/dashboard.html`.
- **What**: horizontal bar chart of `oneShot.prefillMs` per profile.
- **Answers**: "how long until the first token for each profile?"

### 18. Finish reason breakdown ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderFinishChart`,
  `smoke-test/dashboard.html`.
- **What**: stacked horizontal bars showing `eos / max-tokens / stop-token /
  error / unknown` counts by profile.
- **Answers**: "is this profile producing clean completions, or is it
  running off the end?"

### 19. Score over time (regression detection) ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderSeriesChart`,
  `eval/live-db.ts::loadEvalSeries`, `eval/live-server.ts` `/evals/series`.
- **What**: line chart of `overall` across successive eval runs by profile.
- **Answers**: "did a code change regress anything?"

### 20. Quantization comparison (future — requires multi-quant models)
- **Where**: new panel.
- **What**: same model at q4 vs q8 vs f16 — accuracy delta vs speed delta.
- **Blocker**: today every entry in `eval/models.ts` has exactly one
  quant. Needs multi-quant registrations to be meaningful. Deferred.

### 21. Dedicated Embeddings dashboard section ✅ DONE
- **Where**: `smoke-test/dashboard.html` (new section divider + three
  panels), `smoke-test/dashboard.js`
  (`renderEmbeddingCosineChart` / `renderEmbeddingLatencyChart` /
  `renderEmbeddingThroughputChart`),
  `smoke-test/dashboard-charts.js`
  (`buildEmbeddingCosineChartData` / `buildEmbeddingLatencyChartData` /
  `buildEmbeddingThroughputChartData`),
  `tests/dashboard-charts.test.ts`.
- **What**: separate "Embeddings" section at the bottom of the dashboard
  with three panels — per-task cosine similarity, median ms-per-text
  latency, and texts/sec throughput. Per-dimension grouped chart now
  excludes embedding-only evals and drops the `embedding` column so
  generative-model rows aren't cluttered.
- **Answers**: "how fast and how good is each embedding model?"

### 22. Accuracy × Speed model colour key ✅ DONE
- **Where**: `smoke-test/dashboard.js::renderScatterChart`.
- **What**: scatter dots are grouped by `(modelId, thinking)` into one
  dataset per model+mode, each with its own colour from an 8-stop
  palette. The chart's top legend acts as the colour key. Stable colour
  assignment via sorted keys.
- **Answers**: "which dot is which model?" without needing to hover.

---

## Won't-do (for now)

- **Smaller quants (Q2_K / Q3_K)**: quality/speed tradeoff, not a pipeline improvement.
- **Speculative decoding**: requires a drafter model; large project.
- **Custom kernel fusion beyond GLU**: huge effort, marginal win.

---

## Next Steps

### Decode hotspot decision / rebaseline (2026-04-22 Task 5)
- **Current hotspot:** matmul path tuning remains the lead target.
- **Profile-mode hotspot evidence:** `make smoke-bench PERF_RUNS=3
  PERF_MODEL=tinyllama-1.1b-chat-q4_0` reported median **93.5 tok/s**, median
  run **184 ms prefill**, **331 ms decode**, and **2027 ms wall time** for the
  full page completion. Across 90 single-token greedy decode traces,
  `graphComputeMs` averaged **9.96 ms / 91.8%** of step time while
  `downloadResultMs` averaged **0.62 ms / 5.7%**. Because this came from
  `--profile` mode, use it for hotspot ranking and direction, not as a new
  steady-state throughput claim.
- **Backend attribution:** `backendMatmulMs` averaged **4.02 ms / 40.4% of
  graph time**, ahead of `backendEncodeOverheadMs` (**2.81 ms / 28.2%**) and
  `backendAttentionMs` (**0.40 ms / 4.0%**). `backendDispatchCount` stayed at
  **489** per token; that supports the encode/dispatch-overhead suspicion but
  is not itself a timed bottleneck metric.
- **Decision:** keep structural follow-up deferred. If perf work resumes, the
  current profiled traces suggest keeping the next optimization pass narrow and
  targeting matmul first, with encode overhead as the secondary decode-compute
  suspect.
- **Update (2026-04-22):** Matmul follow-up attempt (increase legacy Q outputs per wg) showed no meaningful retained gain and was reverted.
- **Update (2026-04-25):** baseline moved to ~110 tok/s after the GPU TOP_K
  smoke-loop wiring (commit `9156deb`).
- **Update (2026-04-25, library-as-source-of-truth consolidation):** the
  smoke loop has been deleted and replaced with a thin
  `engine.chatCompletion` adapter; `Generator.generate` now picks the
  decode mode dynamically per step. Numbers in this section pre-date
  both 2026-04-25 changes — **all bucket percentages here are stale**.
  Re-profile first; see "Active next steps §1" below.

### Completed on 2026-04-24 / 2026-04-25 / 2026-04-26 — see `TODO_ARCHIVE.md`

The detailed "Completed on" entries from 2026-04-24, 2026-04-25, and
2026-04-26 live in `TODO_ARCHIVE.md`. They cover: encoder forward
pass bring-up, library consolidation (single-source decode), kernel-
tuning campaign §1-§9 (memory-bound matmul characterization),
wave-1 + wave-2 model fleet rollout, §11-§16 closures (Q3_K bug fix,
quant promotions, qwen3-8b registration), plus 28 numbered bug
fixes. The 2026-04-27 cycle (§21-§26) below is the active reference.

### Completed on 2026-04-27 / 2026-04-28 — see `TODO_ARCHIVE.md`

The detailed §21-§32 perf-cycle entries (encoder perf, prefill-tiling,
FA revisit at 7B+, §C-v2-A re-measurement, llama.cpp rebase free-win
sweep, post-rebase regression accept, MEMORY64 cap probe + bridge
sub-probe, prefill-tile heuristic refactor) plus the post-§32
"Resumption checklist" historical notes live in `TODO_ARCHIVE.md`.
Active baselines + status are in the header block above; current
candidates are in the sections that follow.

### Active next steps (post-§31b + squash + housekeeping)

**Status:** all three opt-in probes from the post-§32 list closed
2026-04-28 (§32a / §31b / patch-12 squash); **all three fresh
housekeeping items closed 2026-04-28** (#4 dashboard refresh /
#5 pre-rebase baselines / #6 §32 SUMMARY cross-link); **all three
doc-style next-step candidates closed 2026-04-28** (#7 TODO header
pin refresh / #8 BENCHMARKS tier expansion / #9 CLAUDE.md
doctrine capture); **#10 vault-save closed 2026-04-28** (5 notes
landed under `~/ClaudeVault/Patterns/` + `Knowledge/`; index
rebuilt, MANIFESTs verified — see closure entry in Watch list).
**Algorithmic-perf backlog fully cleared.** All algorithmic levers
at the canonical 4-baseline are exhausted (§17-§29 closed matmul,
FA, drafter, encoder, prefill-tiling, spec-decode families).
Rebase cadence: 2026-05-04 fired (§27 hybrid — maintenance free
win, perf neutral; two local LayerNorm patches subsumed by
upstream `d4b0c22f9`, patch stack 11 → 9). Encoder parity PASS at
cosine 0.76. Cross-day perf vs 2026-05-01 baseline noise on 5/6;
mistral-7b -14% outlier flagged for next-session rerun. Tip:
`fc1f81242` on upstream base `a817a22bc`. Sweep matrix at
[`eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md`](eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md);
same-day pre-rebase control at
[`eval/reports/pre-rebase-baselines-2026-05-04/`](eval/reports/pre-rebase-baselines-2026-05-04/)
(retained as a process lesson — same-day capture was anomalously
cold and overstated the gain by ~30% on small models).
Prior cycle 2026-05-01 was §27 (mul-mat vectorize #22578 +
upscale shader; tip `e29753286`); sweep matrix at
[`eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md`](eval/reports/llama-cpp-rebase-2026-05-01/SUMMARY.md).

### Gemma 4 E2B inference support (in progress 2026-05-10)

**Status:** Phase 1 probe complete; spec written and committed.
Spec: [`docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md`](docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md).
Implementation plan written via `superpowers:writing-plans` after spec
approval; execution via `superpowers:subagent-driven-development`.

**Target:** `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M (3.11 GB), already
symlinked at `smoke-test/models/gemma-4-e2b-it-q4km.gguf`. E4B as
follow-on probe after Stage 5 closure (same arch path; expected
registration-only delta).

**Probe findings (2026-05-10):** GGUF parses + weights load + tokenizer
works; generation fails at `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)`
in `ModelInference.buildQKV`. Seven architectural deltas confirmed vs
current causal-LM dispatch — see spec §2 for full GGUF metadata dump
and §3 for the table of GGUF keys → project impact.

**Scope decisions (committed 2026-05-10):**
- Correctness-first staging (Stages 1–3 collapse Gemma 4's variations
  onto familiar paths; Stages 4–5 lift to production shape)
- Linear 5-stage shape (one gate per stage, mid-implementation
  "Gemma 4 produces correct output" milestone at Stage 3)
- E2B-only scope (E4B deferred to follow-on probe)
- Patch-budget cap +2 max on `webllm-browser-patches` (Stages 4 + 5)

**Stages:**

1. **Stage 1 — Per-layer hyperparams refactor (foundation).** Convert
   scalar `embeddingHeadLength`, `feedForwardLength`, `ropeDimensionCount`,
   `ropeFreqBase` into per-layer arrays. Existing models replicate the
   scalar `layerCount` times (zero behavioral delta). Gemma 4 populates
   per-layer from GGUF. **Gate:** `make checkall` green + 3-model
   `generatedIds[0]` match on TinyLlama / qwen3-0.6b / qwen3-1.7b.
   **Artifact:** `eval/reports/gemma-4-stage1-per-layer-hp-<date>/SUMMARY.md`.

2. **Stage 2 — Gemma 4 surface wiring.** Chat template (`formatGemma4`),
   stop-token registration (`<end_of_turn>`), `GEMMA4_DEFAULTS` sampler,
   `final_logit_softcapping=30.0` wiring, `eval/models.ts` registration,
   bundle regeneration. **Gate:** smoke loads + greedy-decodes 5
   coherent ASCII tokens on `"The capital of France is"`; multi-turn
   stops cleanly on `<end_of_turn>`.
   **Artifact:** `eval/reports/gemma-4-stage2-surface-wiring-<date>/SUMMARY.md`.

3. **Stage 3 — Gemma 4 E2B forward-pass correctness (gated PLE +
   QK norm + post-norms + scaling + dual RoPE).** **CLOSED 2026-05-11
   EOS-12 — 68 % eval, well above the ≥40 % gate.** Closure report:
   [`eval/reports/gemma-4-stage3-closure-2026-05-11/SUMMARY.md`](eval/reports/gemma-4-stage3-closure-2026-05-11/SUMMARY.md).
   Root cause: a single missing entry in
   `getRopeModeForArchitecture` — Gemma family uses NEOX-style
   RoPE (split-halves) per llama-model.cpp:2275-2310, but the
   project mapped only `nomic-bert`, `phi3`, and `qwen*` to NEOX
   and let Gemma fall through to `RopeMode.NORMAL` (interleaved).
   The bug was invisible to Phase 4's 6-token short-completion
   parity probe (rotation phases too small to compound) and
   surfaced only on the chat-formatted Phase A probe + Phase B
   length × content bisection. Fix: three lines plus a load-
   bearing comment block. Per-dimension lift: reasoning
   0 → 83 %, semantic-reasoning 0 → 80 %, instruction-following
   19 → 92 %, overall **9 → 68 %**.

   Sub-tasks (3.3a-l), Phase 4 shared-KV, Phase 5 chat-template:
   all kept as-is; they were correct individually and required
   the NEOX pairing to actually exercise correctly at scale.
   Final scope per
   the 2026-05-10 correction (`docs/superpowers/specs/2026-05-10-gemma-4-stage3-correction-no-altup.md`):
   the unsloth Q4_K_M GGUF has **no AltUp/Laurel tensors** (the previous
   addendum overshot). Real component list:

   - **Gated PLE per block** (load-bearing): pre-loop projection chain
     + per-block gated GELU injection through `inp_gate`/`proj`/`post_norm`
   - **QK norm**: `attn_q_norm` / `attn_k_norm` after Q/K projection
     (Gemma 3+ replaces softcap with this)
   - **Pre+post norm pairs**: `post_attention_norm` + `post_ffw_norm`
     (Gemma family pattern, applied after attn/ffn output)
   - **Per-layer output scaling**: `layer_output_scale.weight` per block
   - **Per-layer head_dim + dual RoPE** (unblocks `buildQKV` reshape3d
     assertion)

   The AltUp/Laurel weights interface slots from commit `95a5c21` stay
   as optional fields, dormant for this GGUF — correct gating behavior
   for any future Gemma 3N variant.

   Spec chain:
   - Base: [`docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md`](docs/superpowers/specs/2026-05-10-gemma-4-e2b-correctness-first-support-design.md)
   - Addendum (superseded): [`docs/superpowers/specs/2026-05-10-gemma-4-stage3-gemma3n-architecture-addendum.md`](docs/superpowers/specs/2026-05-10-gemma-4-stage3-gemma3n-architecture-addendum.md)
   - Correction (authoritative): [`docs/superpowers/specs/2026-05-10-gemma-4-stage3-correction-no-altup.md`](docs/superpowers/specs/2026-05-10-gemma-4-stage3-correction-no-altup.md)

   **Gate:** first generated token on `"The capital of France is"` is
   `Paris` (or ` Paris`); 36-prompt eval ≥40% (loose for Stage 4 to
   lift further).
   **Artifact:** `eval/reports/gemma-4-stage3-ple-dualrope-<date>/SUMMARY.md`
   (PROBE.md + PROBE addendum already landed in commits c98dc1a + 0c91ce8).

   **Stage 3 sub-task progress (2026-05-11, updated end-of-session):**
   - ✅ Task 3.1 (commit `c98dc1a`): PLE pre-impl sizing probe →
     `eval/reports/gemma-4-stage3-ple-dualrope-2026-05-10/PROBE.md`
   - ✅ Task 3.2 (commits `0c91ce8` PROBE addendum, `6c5da48` feat):
     `per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm`
     exposed on loader
   - ✅ Task 3.2a+b+c bundled (commit `95a5c21`): per-block tensors
     loaded — `inp_gate`/`proj`/`post_norm` plus optional AltUp/Laurel
     slots that stay undefined for this GGUF
   - ✅ Task 3.4 (commit `064611d`): per-layer head_dim + per-layer FFN
     dim + dual RoPE dispatch in `model-inference.ts` —
     **unblocks `buildQKV` reshape3d crash**; smoke probe now reaches
     `[7/8]` and emits 64 tokens at 87 tok/s (output `<unused6226>…`
     — garbage, expected without PLE injection)
   - ✅ Task 3.3a (commits `ba0f90e` feat + `7fd0167` docs): pre-loop
     PLE projection chain. `buildPreLoopPle()` helper materializes
     `inpPerLayer` with shape `[pleDim, n_tokens, layerCount]` and
     `graphBuildForwardExpand`-es it in all three forward methods.
     Op sequence per `gemma3n.cpp:317-371` (steps 1-4).
   - ✅ Task 3.3b (commits `cf56960` feat + `6f9db1b` docs): per-block
     gated PLE injection. `injectPerBlockPle()` helper slices
     `inpPerLayer` at slot `il` → `inp_gate` MUL_MAT + GELU → MUL slice
     → `proj` MUL_MAT → RMSNorm with `post_norm` → residual add.
     Op sequence per `gemma4.cpp:328-353`.
   - ✅ Task 3.3c (no commit needed): QK norm was already wired
     generically — `qNorm`/`kNorm` loaded at `model-inference.ts:399-400`
     via `opt("attn_q_norm.weight")` / `opt("attn_k_norm.weight")` and
     applied in `buildQKV:908-913`. Pre-existed for Qwen3 family; fires
     automatically for Gemma 4.
   - ✅ Task 3.3d (commits `73f77df` feat + `ff8965d` docs):
     post-attention norm + post-FFW norm. New `LayerWeights` fields
     `postAttentionNorm` / `postFfwNorm` loaded via
     `opt("post_attention_norm.weight")` / `opt("post_ffw_norm.weight")`.
     Ternary-gated `opMul(opRmsNorm(x, hp.normEpsilon), gain)` applied
     to `oProj` before `attnResidual` add, and to `ffnOut` before final
     residual add. Op sequence per `gemma4.cpp:246-249` + `323-326`.
   - ✅ Task 3.3e (commit `c4e5659`): `layerOutputScale` field +
     `opMul(cur, lw.layerOutputScale)` applied at the end of each
     per-layer iteration, right after PLE injection. Op sequence per
     `gemma4.cpp:355-358`.
   - ✅ Task 3.3f (commit `63c1a6d`): Gemma embedding scale. Inserted
     `wasm.opScale(x, Math.sqrt(hp.embeddingLength))` after each
     `opGetRows(weights.tokEmb, ...)` at all four forward methods,
     gated on `hp.architecture === "gemma4"`. Op sequence per
     `gemma4.cpp:149`.
   - ✅ Task 3.3g (commit `79dd05d`): Gemma GELU FFN activation.
     Replaced `opSwigluSplit(gate, up)` with
     `wasm.opMul(wasm.opGelu(gate), up)` at all four forward methods
     when Gemma 4; SwiGLU retained elsewhere. Op sequence per
     `gemma4.cpp:320` (LLM_FFN_GELU + LLM_FFN_PAR).
   - ✅ Task 3.3h (commit `a321df6`): Gemma V bare-RMS-norm inside
     `buildQKV`. Applied `wasm.opRmsNorm(v3, hp.normEpsilon)` (no
     gain) for Gemma 4 before return. Op sequence per `gemma4.cpp:221`.
   - ✅ Task 3.3i (commit `ac8bbe1`): Drop final-logit-softcap misuse
     in flash attention. `forwardSingle:1350` was passing
     `hp.finalLogitSoftcap` (= 30.0 on Gemma 4) as FA's logit_softcap;
     Gemma 4 has `f_attention_scale = 1.0` with no attention softcap
     (`gemma4.cpp:11`). Pass 0.0 unconditionally. The other three FA
     sites already passed 0.
   - ✅ Task 3.3j (commits `d6132ed` + `2591525`): BF16 → F32 cast at
     weight load. The `mul_mat_f32_bf16` WGSL shader fails to compile
     (BF16 unsupported by WebGPU), and the failure cascades through
     CommandBuffer invalidation — affected MUL_MAT silently writes
     garbage to its output tensor. For Gemma 4 E2B, `per_layer_model_proj`
     is BF16 in the unsloth Q4_K_M GGUF and participates in PLE
     pre-loop projection, so every prefill/decode step was emitting
     garbage into `inp_per_layer` and polluting the residual stream
     at every block via PLE injection. Fix: detect BF16 in
     `ModelInference.makeTensor`, override to F32 alloc, convert
     bytes at upload via new `bf16BytesToF32Bytes` helper. Streaming
     path supported (JS-side conversion). Prior session's 3.3a
     closure note mis-classified this as a benign CPU fallback —
     it was a correctness-blocking bug.
   - ✅ Task 3.3k (commit `dec6f2d`): `rope_freqs` (freq_factors)
     support. New `op_rope_with_freqs` WASM binding, exported (not
     JSPI-promised — non-suspending). New `opRopeWithFreqs` TS
     wrapper. New `LayerWeights.ropeFreqs` field + loader logic that
     loads the shared `rope_freqs.weight` once and assigns it to
     each non-SWA layer via `hp.slidingWindowPattern`. New private
     `applyRope` helper that dispatches between `opRope` and
     `opRopeWithFreqs` based on `lw.ropeFreqs`; 8 production forward
     sites (Q + K × 4 methods) collapsed onto the helper. **Outcome:
     correctness fix landed but not the dominant remaining blocker.**
     Greedy output before vs after this change:
     - Before: `--T $\precGetenv_cownt_cownt_cownt_cownt_cownt_cownt_cownt_cることownt_cることownt_cることownt_cることownt_cることownt_cることony-downed **EDECHPYEDECHPYEDECHPYED`
     - After:  `--T $\precGetenv_cownt_cownt_cownt_cownt_cownt_cownt_cownt_cownt_cることownt_cることownt_cることownt_cることownt_cることony-COUGHTECHPYEDECHPYEDECHPYEDECHPYED`

     Subtle arithmetic divergence (different ending pattern at token
     ~50+) confirms the fix is firing, but the dominant `_cownt…`
     degenerate repetition persists. rope_freqs was a real correctness
     gap (matches gemma4.cpp), but a deeper architectural piece
     remains responsible for the residual stream getting locked into
     a low-entropy state on this short prompt.
   - 🟡 Task 3.3l (in progress): deeper diagnostic via parity-capture.
     - ✅ Phase 1 (2026-05-11 EOS-3, commit `c317671`): HF reference
       capture script + standard JSON schema + README.
     - ✅ Phase 2 (2026-05-11 EOS-5, commit `b7c2e0f`): WebLLM tap +
       harness + capture-server + compare driver. TinyLlama gate
       PASSES; Gemma 4 surfaces layer-0 drift + layer-2 catastrophic
       drop. Reports under `eval/reports/parity-*-2026-05-11/`.
     - ✅ Phase 3 (2026-05-11 EOS-6, commit `78f12e1`): embedding-
       output tap + attention softmax scale bug fix. Gemma 4 layers
       0-14 now cosines 0.88-0.98 (recovered from 0.34-0.97 jitter);
       catastrophic boundary moves to L15 (shared-KV transition).
       See "Phase 3 findings" below.
     - ✅ Phase 4 (2026-05-11 EOS-7): wire shared-KV at layers
       15-34 (Gemma 4 E2B `n_layer_kv_from_start=15`, 20 shared
       layers). End-of-stack cosine recovered **0.0420 → 0.9722**;
       top-1 argmax now MATCHES HF reference (id 9079 "Paris"
       for `"The capital of France is"`); top-16 overlap 1/16 →
       13/16. Reports under
       `eval/reports/parity-gemma-4-e2b-shared-kv-2026-05-11/`
       (see `SUMMARY.md` for the closure write-up). Chat smoke
       still degenerate (`<eos>` × N) — downstream chat-template
       tokenization issue, surfaced separately as Phase 5.
     - ✅ Phase 5 (2026-05-11 EOS-8, commits `d8a0835` feat +
       `4fc5993` docs): chat-template tokenization audit.
       **Root cause:** the unsloth Gemma-4 / Gemma-3N GGUF vocab
       stores the turn-boundary tokens under non-standard literals
       `<|turn>` (id 105) and `<turn|>` (id 106), not the classical
       `<start_of_turn>` / `<end_of_turn>`. `formatGemma4` was
       hard-coding the classical literals, so the SPM tokenizer
       BPE-fragmented each into ~7 unrelated pieces and the model
       received untrained input on every turn boundary. **Fix:**
       template-sniff `<|turn>` substring in `formatGemma4` and
       in the two `addChatStopToken` call sites (engine.ts chat
       and chatCompletion paths); emit the matching literal pair.
       Classical Gemma 2 / 3 templates unaffected. **Outcome:**
       `tokensIn` 75 → 41-46; greedy smoke went from `<eos>` × N
       to coherent English ("Please provide the text you would
       like me to help you with.") with finish=stop-token after
       13 tokens. Browser-console probe confirms
       `tok.encode("<|turn>")=[105]` and `tok.encode("<turn|>")=[106]`.
       Ship gate green (763 pass / 0 fail). Closure report:
       [`eval/reports/gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md`](eval/reports/gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md).
     Candidates in priority order:
     (a) **Intermediate hidden-state comparison** vs HuggingFace
         `transformers` Gemma 4 reference run on the same prompt
         tokens. Tap `forwardForEmbedding` to capture per-layer
         residual stream, save to JSON, compare. The bucket-D ref
         capture pattern (`eval/reports/bucket-d-probe-2026-04-29/
         capture-refs.py`) is the closest precedent. Pinpoints the
         FIRST layer where residual diverges from canonical.
     (b) **Chat template + tokenizer audit.** Verify the BOS token
         is added with the right ID and the `<start_of_turn>` /
         `<end_of_turn>` tokens encode to their correct GGUF IDs.
         Look-bearing if BPE tokenization is off — the model would
         see different tokens than training data.
     (c) **Real SWA implementation** (originally Stage 4). Currently
         falls back to all-global; for prompt+output < 512 tokens
         this shouldn't matter much, but verify by capping output to
         within-window range or implementing SWA.
     (d) **Per-layer KV-cache slot size**. Gemma 4 mixed-head-dim
         means SWA layers store smaller K/V than global. The
         project currently sizes for worst-case (largest) per the
         loader spec. If allocation is wrong, attention reads
         garbage from KV cache.

   **Smoke-gate progression 2026-05-11 (post-3.3j):** All Stage 3 fixes
   landed (3.3a–j). The BF16 cast (3.3j) was THE correctness unblocker:
   `mul_mat_f32_bf16` shader failures were NOT benign one-shot CPU
   fallbacks (as 3.3a's closure note had claimed); they were
   cascading CommandBuffer invalidations writing garbage to
   `per_layer_model_proj` output every layer/token. After 3.3j the
   output transitions from `<unused14><unused11>…` unused-token
   noise to **real-vocab tokens** — `LA_T_cowntहांत_cَour…` at
   temp=1.0, `_cownt_cownt_cownt…_cることownt…` at temp=0. The
   greedy degenerate repetition signature points at a remaining
   load-bearing arch piece: `rope_freqs` (Task 3.3k). The bf16
   device-error log in tab titles is now gone — console verifies
   no shader compile failures during decode.

4. **Stage 4 — Real sliding-window attention.** Replace Stage-3
   "all-global" fallback with real SWA on the 4-of-5 layers marked
   local in the GGUF pattern. Window 512; mask + KV-window logic
   per upstream llama.cpp. May require 1 llama.cpp patch if
   ggml-webgpu's softmax-with-mask path can't express the windowed
   mask. **Gate:** eval lifts ≥60% (Phi-3 closure standard);
   long-context probe (1000-token generation) shows no quality cliff
   at the 512-token window boundary.
   **Artifact:** `eval/reports/gemma-4-stage4-swa-<date>/SUMMARY.md`.

5. **Stage 5 — Shared-KV ref-sharing + bench + closure.** Wire the
   last-20-of-35 layers' shared K/V references through the KV-cache
   allocator (currently materializes duplicate K/V; ~3 GB wasted),
   per-conversation snapshot/load respects the sharing graph,
   `indexeddb-store` serializes the ref-shared layout. **Gates:**
   VRAM drops ≥2 GB vs Stage 4; smoke-bench profile-mode 3-run
   median ≥10 tok/s; eval ≥ Stage 4 (target ≥60% holds). May
   require 1 llama.cpp patch for the cache allocator.
   **Artifact:** `eval/reports/gemma-4-e2b-validation-<date>/SUMMARY.md`
   (Phi-3 closure template; absorbs phases 5–6 of Phi-3).

**Out of scope (this campaign):** E4B SKU; PLE CPU offload; 26B A4B
MoE and 31B Dense SKUs; multimodal (vision/audio); MTP drafter (deferred
behind upstream llama.cpp Discussion #22735).

**Risks:** Stage 1 silent regression (mitigated by 3-model `generatedIds[0]`
gate); Stage 4 SWA mask shape unsupported by ggml-webgpu (mitigated by
synthetic windowed-mask probe before implementation); Stage 5 ref-shared
KV breaks persistence (mitigated by `engine-conversation-persistence`
test surface). Wall-clock risk: 5 sessions plan; partial credit lands
if Stage 4/5 stall (Stages 1–3 alone produce a usable correctness-first
Gemma 4).

#### Resume in fresh session — pickup instructions (updated 2026-05-12 EOS)

**Read this first, then
[`eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md)
for the full diagnostic, repro recipes, and four recommended unblock paths.**

---

##### Where Stage 4 actually stands

- **Stage 4.0 + Stage 4.1 code shipped** as commits `b4f6bdf` (Phase A —
  `writeCausalMaskF16` helper + `uploadLeaves` SWA params + 9 unit tests)
  and `0739d80` (Phase B — per-layer SWA/global mask dispatch across all
  four chat-path forwards).
- **Closure smoke was misleading.** The TODO line below ("Gemma 4 E2B
  Q4_K_M: `Paris` on 'The capital of France is', 57.1 tok/s") was run
  on `real-model.html` which defaults to `flashAttn=false`. The FA path
  used by `chat.html` + `createConversation` was never exercised on
  Gemma 4 after Phase B, and it traps.
- **Long-context end-to-end parity gate** (`forwardWithLayerTaps` on a
  ~1000-token prompt vs HF reference) is **infeasible on Chrome/Apple**
  due to the WebGPU `maxStorageBufferBindingSize=128 MiB` cap. Three
  probe attempts at N=1129 / N=560 / N=1129+`skipLayerTaps` all OOM at
  `backendAllocCtxTensors`.

##### Two open blockers (in priority order)

1. 🔴 **[CRITICAL] Gemma 4 + `flashAttn=true` traps with `Error:
   unreachable` at every prompt length.** Confirmed reproducible on
   committed tip `15b57dd` with all 2026-05-12 in-flight changes
   stashed — so this is a Phase B (`0739d80`) regression, not a today's
   change regression. Production chat is currently broken for Gemma 4:
   `chat-models.js:68` hardcodes `flashAttn: true` because
   `createConversation` (`engine.ts:711-713`) requires FA mode.

   **Repro:**
   ```
   make smoke-serve &
   agentchrome --port <PORT> navigate \
     "http://localhost:8031/chat.html?model=gemma-4-e2b-it-q4km&v=$(date +%s)"
   # Wait for load, type "The capital of France is", click Send.
   # Result: chat-msg.assistant.error rendered with "Error: unreachable".
   ```

   **Cheap discriminator probe** (5-min budget): edit
   `smoke-test/parity-capture-page.js:143` to set `flashAttn: true`
   (currently `false`), rebuild bundle, rerun the 46-token Gemma 4
   capture (the standard parity-capture path). Outcomes:
   - Same OOM signature as before → FA bug is in the existing
     long-context graph, not the mask. Probably need backend split or
     different KV layout.
   - Different / earlier failure → narrows to FA-shader's consumption
     of the banded mask OR mixed-head-dim plumbing in the per-layer
     FA dispatch (`forwardSingle:1610-1640`, `forwardAllPositions`,
     `forwardDecode`).
   - Success → the bug is in `chat.html`'s worker / engine init path
     specifically, not in the FA forward.

   **Comparison evidence:** TinyLlama in chat.html (FA on) works fine;
   Gemma 4 in real-model.html (FA off) returns "Paris" in 0.4s — so
   the bug is FA × Gemma 4 specifically.

2. 🟡 **Long-context tap-based parity blocked by per-binding cap.** See
   [`eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md)
   "Recommended next steps" section for four ordered options. Once
   blocker (1) is fixed, the lowest-cost option becomes viable: drive
   the standard chat path (which uses `forwardSingle` — no per-layer
   tap retention) at 1129 tokens and observe non-crash + coherent
   output as a weak positive signal. Tap-based per-layer cosine is
   only achievable via option 3 (incremental 35-forward capture using
   the `captureTaps` scaffolding) or option 4 (backend patch to split
   graph tensors across multiple WebGPU bindings).

##### In-flight uncommitted work (2026-05-12 EOS-2: cleared)

Workspace clean on tip `01c00db`. The three EOS-1 in-flight TS
chunks landed as commits `447ff82` (per-layer headCount plumbing +
graphMem), `65ac040` (parity-capture cache-buster + skipLayerTaps +
long-context fixture), and `01c00db` (FA discriminator probe
SUMMARY). The `captureTaps`/`lastLayerTaps` scaffolding was deleted
before commit (no producer). The three debris files
(`input_ids_tmp.txt`, `navigate_tmp.py`,
`webllm-divergence-localization.skill`) were deleted.

**Load-bearing comments restored:** `model-loader.ts` had two
load-bearing comment blocks deleted by the prior agent (iSWA remap
rule citing `llama-model.cpp:2007-2014`, and the `finalLogitSoftcap`
/ PLE field comments around `pleDim`). Restored before commit per
the "Surgical Changes" doctrine in `~/.claude/CLAUDE.md`.

**Health check (post-commit, tip `01c00db`):** `make checkall` green
(fmt + lint + typecheck + 772 tests pass / 36 skip / 0 fail).

##### Recommended first actions for the next session

**Pickup state 2026-05-12 EOS-2** — workspace clean on tip `01c00db`.
Three atomic commits landed this session:
- `447ff82` feat(gemma4): per-layer headCount/headCountKv plumbing + graphMem bump
- `65ac040` feat(parity-capture): cache-buster + skipLayerTaps + long-context fixture
- `01c00db` docs(stage4): FA discriminator probe outcome

FA discriminator probe ran: **`forwardWithLayerTaps` + FA=true + Gemma 4
at N=9 SUCCEEDS in 0.28s, 35 layers tapped.** chat.html still traps
`Error: unreachable` at the same prompt. See
[`eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md)
"FA probe result" section for the refined picture and dispatch plan.

**Refined bug location:** FA shader compiles + runs on Gemma 4 fine; the
trap is specifically in `forwardSingle` / `forwardAllPositions` /
`forwardDecode` when invoked via `engine.chatCompletion`. Since trap
fires on the very first message (`forwardAllPositions` prefill, not
`forwardDecode`), the most likely candidate is `forwardAllPositions`.

1. **Next cheap probe (~5 min): `forwardAllPositions` FA=true smoke.**
   Two options:
   - Standalone probe page (`smoke-test/fa-prefill-probe.html`)
     mirroring `parity-capture-page.js` but invoking
     `forwardAllPositions` directly with the 9-token Gemma 4 prompt.
     ~80 lines of JS, no engine wiring. (Recommended.)
   - Invasive: temporary debug knob in `engine.chatCompletion` that
     swaps `forwardAllPositions` for `forwardWithLayerTaps`. Easier
     code-wise but mutates the production path.

   Outcomes:
   - TRAPS → bug is in `forwardAllPositions` (prefill-batched FA).
     Diff `forwardAllPositions` against `forwardWithLayerTaps` in
     `0739d80` to find the offending line.
   - SUCCEEDS → trap is in `forwardDecode` (`nTokens=1` decode-shape
     FA), surfacing later than the "first message" framing suggested.
     Different bisection target.

2. **Once the offending forward is identified, bisect `0739d80` line by
   line** — Phase B's per-layer SWA mask wiring is the only structural
   change to those forwards since they last worked with FA. Likely
   candidates:
   - Per-layer mask leaf allocation site (each forward allocates one,
     unlike pre-Phase-B's single shared mask).
   - Mask byte upload (`writeCausalMaskF16` with `swaWindow > 0`)
     producing a layout the FA shader rejects.
   - Per-layer K/V views feeding FA with a stride that disagrees
     with the per-layer head_dim.

3. **Only after Gemma 4 + FA works again**, return to Stage 4.1
   long-context closure. The cheapest gate is now "drive
   `engine.chatCompletion` greedy on the 1129-token prompt via
   `chat.html`; verify non-crash + coherent output". Per-layer
   parity remains out of reach without the captureTaps producer or
   a backend patch.

##### Smokes verified post-Phase B (2026-05-11, manual-attn path only)
- TinyLlama Q4_0: 165.6 tok/s decode, coherent English, non-SWA
  fall-through bit-identical.
- Gemma 4 E2B Q4_K_M `Paris` smoke: ran on `real-model.html`
  (`flashAttn=false`). **Does NOT cover the FA path** that
  `chat.html` + `createConversation` require.

##### Architectural notes that didn't make it into earlier sessions
- **KV-cache helpers** (`serializeKVCache`, `loadKVCache`, KV size
  estimators at `model-inference.ts:715-970`) still read scalar
  `hp.headCount` / `hp.headCountKv`. Technically wrong for
  Gemma 4 mixed-GQA, but doesn't matter today: these helpers are only
  used by `engine.ts` for in-process serialization (no cross-model
  interop) and per-layer KV slots are already sized correctly at
  `initKVCache` time via `embeddingHeadLengthPerLayer`. Fix is queued
  as a follow-up only if Gemma 4 KV checkpointing ever ships.
- **`injectPerBlockPle` view assertion** under prefill tiling (originally
  surfaced 2026-05-12) was never reproduced after the `graphMem`
  bump made tiling unnecessary for short-prompt Gemma 4. Status
  unchanged from prior sessions: investigate only if it resurfaces.

---

**After Stage 4.1 final gate clears, the queue is:**

1. **Stage 4.2 — Gemma 2 alternating-period SWA derivation.**
   First action: confirmation dump. Load `gemma-2-2b-q4f16` in the
   browser, log `window.parsedModel.hyperparams.slidingWindowPattern`,
   verify it matches the period-2 alternation (`[F,T,F,T,...]`). If
   Q1.4's un-demote work already populated it correctly, this stage
   is a one-line closure note. If not, derive it in
   `model-loader.ts:519-523` from a `swa_period` integer fallback.

2. **Stage 4.3 — Long-context regression probe.** Generate
   1000-token output on a fixed long prompt; measure
   argmax-divergence vs HF reference or a known-good `llama-cli` run.
   Gate: no quality cliff at the 512-token SWA boundary.

3. **Stage 4.4 — Eval re-gate.** `bench-profile
   PROFILES=gemma-4-e2b-warm` — 36-prompt eval ≥ 68 % (must not
   regress Stage 3 closure baseline).

4. **Campaign Q3 (Stage 5) — bench + closure.** Pre-rebase
   baselines on the canonical 6, add Gemma 4 to bench-full, single
   canonical closure SUMMARY.

**Out-of-stage opportunistic work (not gating Stage 4):**
- **Embedding-path SWA support.** `forwardForEmbedding` (line 1907)
  was deliberately left on the original single-mask path because
  Gemma 4 isn't registered as an embedder. If a Gemma SWA model
  ever ships as an embedder, mirror the Phase B SWA wiring there
  too. Tracked here so the gap isn't forgotten.
- **`debugLayerOutput` SWA support.** Same situation — debug-only
  path skipped to keep the Phase B diff narrow. Add SWA wiring if
  the path ever gets used for long-context SWA debugging.

---

**Gemma 4 Stage 3 CLOSED 2026-05-11 EOS-12 at 68 %** (≥40 % gate
cleared, also above the ≥60 % Phi-3 closure baseline). Root cause:
Gemma family was falling through to interleaved RoPE in
`getRopeModeForArchitecture`. Single-line fix at `be63158`; added
gemma2/gemma3 to the NEOX list pre-emptively at `c8c8447`.

Closure report:
[`eval/reports/gemma-4-stage3-closure-2026-05-11/SUMMARY.md`](eval/reports/gemma-4-stage3-closure-2026-05-11/SUMMARY.md).

**Campaign Q1 (Gemma 2 un-demote) CLOSED 2026-05-11 EOS-13 at 60 %.**
Root cause was **plural** (six fixes, three not on the original
demote-SUMMARY candidate list): NEOX-RoPE (pre-Q1 `c8c8447`),
attention + final logit soft-capping with op_tanh binding
(`f2735d5` + `5d1aba4`), JSDoc placement (`bb73d4f`), embed-scale +
GELU FFN extension to whole gemma family + scale-first softcap order
(`31d53a5`), un-demote + closure docs (`dc3304a`). Eval: 92 %
reasoning, 72 % instruction-following, 61 % semantic-reasoning, 17 %
tool-calling@capability=false.

Closure report:
[`eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md`](eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md).

**Doctrine lessons banked EOS-13 (candidates for CLAUDE.md):**

- **Demote candidates are usually plural.** Original Gemma 2
  demote SUMMARY listed five candidate causes with similar weight;
  un-demote needed six fixes, three of which weren't on the
  original list. Future demote SUMMARYs should bias toward
  "expect plural" rather than "one of these five".
- **Soft-cap order is non-trivial.** Naïve order
  (`softcap → scale`) silently corrupts the attention distribution
  because the cap acts on the wrong-magnitude input. Reference
  order (`scale → softcap → softmax`) is recorded in
  `ggml-cpu/ops.cpp:8232-8233` as `scale /= logit_softcap`. The
  WebGPU host (`ggml-webgpu.cpp:1942-1944`) does the same trick.
  Add to the chat-template / RoPE family audit list at the next
  llama.cpp rebase.
- **Lookup-table extension audits pay off across the family.**
  Three of the six Q1 fixes were "extend the `gemma4`-only branch
  to the whole Gemma family" or "add gemma2/gemma3 to the NEOX
  list". The audit caught the lot in one cycle. The
  `getRopeModeForArchitecture` / `attnSoftmaxScale` / chat-template
  detector / `isGemmaFamily` / GELU-vs-SwiGLU branch tables are
  the canonical surface to audit per major rebase.

**Gemma 4 Stage 3 supporting closures** (Phase 5 chat-template fix
+ Task 3.5 probe chain) are documented in the closure SUMMARY
linked above. The Phase 5 standalone report lives at
[`eval/reports/gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md`](eval/reports/gemma-4-stage3-phase5-chat-template-2026-05-11/SUMMARY.md);
the Task 3.5 probe chain artifacts (eval-baseline, default-system,
temp-sweep, stop-token-audit, parity probes A/B1/B2) are under
`eval/reports/gemma-4-stage3-*-2026-05-11/` and
`eval/reports/parity-gemma-4-e2b-*-2026-05-11/`.

────────────────────────────────────────────────────────────────
### Campaign Q1 — Gemma 2 un-demote — **CLOSED 2026-05-11 EOS-13** ✅

`gemma-2-2b-warm` un-demoted back into `SMOKE_PROFILE_SETS.full`
at 60 % overall eval (92 % reasoning / 72 % instruction-following
/ 61 % semantic-reasoning / 17 % tool-calling@capability=false).
Speed: 58.8 tok/s decode. Closure SUMMARY:
[`eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md`](eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md).

**Root cause (six items, plural):** NEOX RoPE (pre-Q1 `c8c8447`),
attention soft-cap (`f2735d5` + `5d1aba4`), final-logit soft-cap
(`5d1aba4`), embed-scale extended to whole gemma family
(`31d53a5`), GELU FFN extended to whole gemma family (`31d53a5`),
scale-first softcap order in manual softmax path (`31d53a5`). The
original demote SUMMARY enumerated five candidates with similar
weight; the actual un-demote needed six fixes, three of which
weren't on the original list — doctrine: *expect plural root
causes for demotes*.

Original Q1 plan (Q1.1-Q1.5) and the surfaced Q1.6 (gemma-family
branch extension + softcap order) below are preserved for
historical context; everything CLOSED.

**Out-of-original-scope Q1.6** surfaced 2026-05-11 EOS-13 when
Q1.4 smoke probe still showed whitespace lock after Q1.1-Q1.3
landed. Adding it to the campaign was the right call (rather
than punting to a separate campaign) since Q1.2-Q1.3 alone
didn't move the smoke probe and the architectural reading was
on-hand.

Full original plan (Q1.1-Q1.5 sub-task specs + pre-work
verification + Gemma 2 GGUF metadata dump + Q1.6 retrospective
+ risk register): archived to
[`TODO_ARCHIVE.md`](TODO_ARCHIVE.md) under "Campaign Q1 —
Gemma 2 un-demote".

────────────────────────────────────────────────────────────────
### Campaign Q2 — Stage 4: real sliding-window attention (queued 2026-05-11 EOS-12)

**Goal:** replace the all-global causal-mask fallback with a real
per-layer SWA windowed mask on Gemma 4 SWA layers (and any other
SWA-using model that registers — Gemma 2, Gemma 3, potentially
some Qwen3 variants).

**Current state:** Gemma 4 SWA layers use full causal attention.
At prompt+output lengths ≤ 512 tokens (the SWA window), this
produces correct math because the window is wider than the
sequence. At longer prompts the SWA layers see all positions
instead of the local 512 window, which over-mixes information
and degrades long-context coherence. The 36-prompt eval doesn't
exercise this (each task < 200 tokens), so Stage 4 was not gating
Stage 3 closure.

**Pre-flight probe (Stage 4.0 — windowed-mask feasibility).**
**CLOSED 2026-05-11 — ✅ no llama.cpp patch needed.** Both
`opSoftMaxExt` (`soft_max.wgsl:184,211`) and `opFlashAttn`
(`flash_attn.wgsl:230-232`) handle the mask as a purely additive
per-element term (`v += slope * mask_val(i)`); `slope=1.0` when
`max_bias=0` (project default). There is no position-driven
masking in the shader. A banded windowed mask is identical in
shape, dtype, and strides to the current causal mask — only the
byte content differs. Closure report:
[`eval/reports/gemma-4-stage4-probe-2026-05-11/SUMMARY.md`](eval/reports/gemma-4-stage4-probe-2026-05-11/SUMMARY.md).

**Stage 4.1 — Per-layer mask construction.** Implementation
**LANDED 2026-05-11** in two commits:

- Phase A (`b4f6bdf`, foundation): extracted
  `writeCausalMaskF16(view, totalLen, nTokens, pastLen,
  maskPaddedCols, swaWindow?)` from `uploadLeaves`; threaded
  optional `swaMaskTensor + swaWindow` through `uploadLeaves`
  with zero-behavior-change defaults; 9 unit tests cover full-
  causal + banded SWA + edge cases.
- Phase B (`0739d80`, per-method wiring): allocates
  `swaMaskTensor` alongside `maskTensor` in `forwardSingle`,
  `forwardWithLayerTaps`, `forwardAllPositions`, `forwardDecode`
  when `hp.slidingWindowPattern?.some(b => b)` AND
  `hp.slidingWindowSize > 0`. Per-layer attention dispatch (FA +
  manual softmax × 4 methods) picks
  `layerMask = isSwaLayer && swaMaskTensor !== 0 ? swaMaskTensor
  : needsMask ? maskTensor : 0`. Non-SWA models see
  `swaMaskTensor = 0` always and stay bit-identical. SWA mask
  needed when `nTokens > 1 || pastLen + nTokens > swaWindow`,
  which covers both prefill and the long-context decode-step
  case where the window no longer covers all past KV.
- Smoke verified post-Phase B: TinyLlama 165.6 tok/s decode
  (vs 168.1 baseline, -1.5% within noise; coherent English);
  Gemma 4 E2B emits "Paris" on "The capital of France is",
  finish=stop-token, 57.1 tok/s; console clean.
- Ship gate: 772 pass / 0 fail / 39285 expect() calls.

**Final gate** (parity probe — 1000-token Gemma 4 prompt, SWA
layers cosine ≥ 0.95 vs HF) is the next step. Phase B is the
load-bearing implementation; the parity probe verifies the
windowed-mask behavior actually fires at >512 token contexts
where SWA matters. The 6-token parity-capture harness from
3.3l-P2 (Phases 3-5) is the closest precedent — extend the
captured prompt length and re-run. **Artifact (pending):**
`eval/reports/gemma-4-stage4-swa-mask-<date>/SUMMARY.md`.

**Stage 4.2 — Gemma 2 alternating-period SWA support.** Gemma 2's
GGUF schema doesn't carry a per-layer SWA boolean array; it ships
`swa_period = 2` (or omits it and defaults to 2) per
`gemma2.cpp:6-8`. At load time, derive a per-layer SWA pattern
from the integer period: `slidingWindowPattern[i] = (i % period
!= 0)` (every period-th layer is global, others are local).
**Gate:** load `gemma-2-2b-q4f16`, dump `hp.slidingWindowPattern`,
confirm it matches the period-2 alternation. This work overlaps
with Q1.4 — if Q1 ships first, this stage may already be done.

**Stage 4.3 — Long-context regression probe.** Generate 1000-token
output on a fixed long prompt with Gemma 4 E2B; measure perplexity
or argmax-divergence vs a reference (HF or a known-good llama.cpp
build). **Gate:** no quality cliff at the 512-token boundary;
generation stays coherent. **Artifact:**
`eval/reports/gemma-4-stage4-longcontext-<date>/SUMMARY.md`.

**Stage 4.4 — Eval re-gate.** Re-run `bench-profile
PROFILES=gemma-4-e2b-warm`. **Gate:** 36-prompt eval at ≥ 68 %
(Stage 3 closure baseline — must not regress). Long-context Q&A
pulled from existing eval suites should improve.

**Out of scope:** SWA-with-FlashAttention (FA's mask path may not
support windowed masks; if so, gate FA off when the mask is
windowed, follow-up campaign for FA + windowed-mask).

**Risks:**
- `ggml-webgpu` mask shader may not accept arbitrary masks (Stage
  4.0 verifies). If patched, +1 to the local patch stack.
- Building two mask tensors instead of one ~doubles per-layer
  mask-allocation cost. Probably negligible (tiny tensors) but
  worth a smoke-bench cross-check.
- Gemma 2 / Gemma 3 patterns may differ subtly from Gemma 4's; the
  Stage 4.2 derivation needs to match each arch's reference.

────────────────────────────────────────────────────────────────
### Campaign Q3 — Stage 5: bench + closure write-up (queued 2026-05-11 EOS-12)

**Goal:** refresh canonical perf baselines now that the NEOX fix
landed, capture Gemma 4 E2B into the dashboard fleet, and close
the Gemma 4 campaign with a single canonical SUMMARY.

**Stage 5.1 — Pre-rebase baseline capture.** Per the §32a doctrine
("Pre-rebase baseline doctrine"), capture profile-mode benches on
the canonical 6 BEFORE any rebase fires. The NEOX fix changed
`getRopeModeForArchitecture` behavior only for Gemma family; the
canonical 6 (`tinyllama-warm`, `qwen3-0.6b-q4f16`,
`qwen3-1.7b-q4f16`, `mistral-7b-instruct-v0.3-q4ks`,
`llama-3.1-8b-instruct-iq3m`, `qwen3-8b-iq3m`) shouldn't move
— but sweep verifies. **Artifact:**
`eval/reports/pre-rebase-baselines-<date>/SUMMARY.md`.

**Stage 5.2 — Add `gemma-4-e2b-warm` to the canonical 6 (now 7).**
Update the `bench-full` profile set to include the Gemma 4 entry;
update the dashboard's headline-baseline table; ensure the
accuracy×speed scatter renders the new dot. **Artifact:**
inline in 5.1's pre-rebase capture or a separate Stage 5.2 entry.

**Stage 5.3 — Closure SUMMARY.** Write a single canonical Gemma 4
campaign closure at
`eval/reports/gemma-4-e2b-validation-<date>/SUMMARY.md`. Folds in:
Stages 1-3 closure highlights (already documented per-stage),
Stage 4 outcome (if landed), Stage 5 perf data, the §27/§28/§32
classification (the NEOX fix is a §27 free-win retrospectively —
single-line code change, eval +59 pp). Cross-link from the
README BENCHMARKS table.

**Gate:** dashboard renders Gemma 4 in the accuracy×speed scatter
at the post-fix score; perf median for `gemma-4-e2b-warm` lands
in the dashboard SQLite; closure SUMMARY merges to main.

**Out of scope:** un-demoting Gemma 2 in this campaign (Q1's job);
Gemma 4 E4B / 9B / 14B SKUs (above the 8B ceiling or out of the
unsloth/E2B canonical path).

**Risks:** low. The NEOX fix is architecture-gated; non-Gemma
models on `bench-full` should be bit-identical (regression
sanity-check is the headline output).

────────────────────────────────────────────────────────────────

**Phase 3 closure (prior EOS) preserved below** for reference:

**Phase 3 findings — Gemma 4 attention softmax scale was wrong:**

The bug: `gemma4.cpp:11` sets `hparams.f_attention_scale = 1.0f`
("Gemma4 uses self.scaling = 1.0, no pre-attn scaling") passed
verbatim as `kq_scale` to `ggml_soft_max_ext` at `llama-graph.cpp:
2033`. Gemma 4's `attn_q_norm` / `attn_k_norm` gains are trained to
compensate for the missing 1/√d_k factor. WebLLM was applying the
default 1/√head_dim across every architecture; for Gemma 4 with
head_dim=256 that's a 16× scaling difference, producing essentially
uniform attention weights and non-uniform drift compounding across
the residual stream.

Fix: new `attnSoftmaxScale(hp, headDim)` helper returns 1.0 for
`gemma4` and 1/√head_dim for everything else; 8 call sites updated
(FA + manual paths in forwardSingle, forwardForEmbedding,
forwardWithLayerTaps, forwardAllPositions, and the 6th debug path).

Parity-capture before vs after (eval/reports/parity-gemma-4-e2b-
attnscale-fix-2026-05-11/REPORT.md):

| Block | Before | After | Δ |
|---|---|---|---|
| embed | 0.9953 | 0.9953 | — (unchanged) |
| L0 | 0.9756 | 0.9600 | -0.016 (slight, accepted) |
| L1 | 0.9685 | 0.9807 | +0.012 |
| L2 | 0.6591 ⚠ | 0.8648 | +0.206 ★ |
| L4 | 0.9375 | 0.9822 | +0.045 |
| L9 | 0.6730 | 0.9603 | +0.287 ★ |
| L11 | 0.3436 | 0.8790 | +0.535 ★ |
| L14 | 0.8445 | 0.9742 | +0.130 |
| **L15** | 0.5572 | **0.6605** | (still bad — Δ -0.31 NEW boundary) |
| L34 | 0.0335 | 0.0420 | (still terrible) |

TinyLlama regression-check after fix: PASS (end-stack cos 0.9855,
top-1 argmax matches HF id 3681). Conditional is architecture-gated.

Gemma 4 chat smoke (greedy temp=0): output transitioned from low-
entropy repetition `_cownt_cownt_cownt…` to high-entropy mixed-vocab
noise `เชพอ'ircleこれから話precise…`. This signature change matches
"layers 0-14 mostly correct, layer 15+ destroys the residual stream".

**Phase 4 plan — shared-KV at layers 15-34:**

The bug: Gemma 4 E2B has `num_kv_shared_layers=20`. Per gemma4.cpp:
208-238, `has_kv(il)` returns false for layers ≥ `n_layer_kv_from_
start` (= `n_layer - shared_kv_layers` = 15). Shared-KV layers DON'T
compute their own K/V from `wk`/`wv` — instead they REUSE the K/V
cache slot from an earlier (full-attention-or-SWA-matching) layer.

WebLLM currently computes fresh K/V from `lw.kProj` / `lw.vProj` at
every layer including 15-34. The GGUF does ship `attn_k.weight` /
`attn_v.weight` at every layer, but at shared-KV layers those weights
should be IGNORED — the attention reads from the cache slot of an
earlier layer.

Reference: llama-kv-cache-iswa.cpp + llama-kv-cache.cpp (look for
`reuse layer %d, is_swa = %d` log line at llama-kv-cache.cpp:249).
The remapping is layer-type-aware: each shared SWA layer reuses the
LAST same-SWA-type pre-share slot; each shared full layer reuses the
LAST full pre-share slot.

For unsloth/gemma-4-E2B-it: SWA pattern is
`[T,T,T,T,F,T,T,T,T,F,T,T,T,T,F]` for layers 0-14. Layers 15-34
follow the pattern `[T,T,T,T,F]×4` and reuse — per llama.cpp's iSWA
remap — the LAST same-type SWA / full slot before the boundary.
Concretely (the pattern is `il_kv_reuse = first_pre_share_match`):
- Layer 14 (full, has own KV at slot 14) — the last full layer before share
- Layer 13 (SWA, has own KV at slot 13) — the last SWA layer before share
- Layer 15 (SWA, share) → reuses slot 13
- Layer 16 (SWA, share) → reuses slot 13
- Layer 17 (SWA, share) → reuses slot 13
- Layer 18 (SWA, share) → reuses slot 13
- Layer 19 (full, share) → reuses slot 14
- Layer 20 (SWA, share) → reuses slot 13
- ...

Confirm this mapping by adding instrumented logs in llama.cpp side
and dumping at load time, or by reading `llama_kv_cache_init` debug
output.

Phase 4 implementation:
1. **Loader change** (`src/models/model-loader.ts`): expose
   `sharedKvLayers` already loaded, plus a derived per-layer
   `kvReuseFromLayer[il]: number | null` mapping. For `il < n_layer
   - sharedKvLayers`: `null` (own K/V). For `il >= ...`: index of
   the last preceding same-type layer with own K/V.
2. **Inference change** (`forwardSingle` + `forwardWithLayerTaps`):
   in the per-layer loop, if `kvReuseFromLayer[il] !== null`, SKIP
   the K/V projection + KV-cache write for this layer. Instead,
   point `fullK` / `fullV` views at the cached slot of layer
   `kvReuseFromLayer[il]`.
3. **Q projection still happens** at every layer (each layer has
   its own `attn_q.weight`).
4. **Attention itself** uses the borrowed K/V plus this-layer's Q.
5. **Memory savings**: shared-KV layers don't allocate cache slots
   (Stage 5's headline 2 GB savings). Phase 4 can either share
   the existing-slot view (cheap) or skip allocation (savings).
   For correctness-first, sharing the view is enough; allocation
   savings can land later.
6. **Validation**: re-run parity capture. Expected: layers 15-34
   cosines recover to the same ~0.93-0.98 regime as 0-14.
7. **Gates**: end-stack cos ≥ 0.95; greedy smoke produces real
   English; ≥40% eval on the 36-prompt suite (Stage 3 closure
   target).

**Risks / open questions:**
- The exact iSWA remap (which layer's KV does layer 15 reuse?)
  must be empirically validated. If the mapping is wrong, the
  parity report will surface it again at L15.
- The `attn_k.weight` / `attn_v.weight` tensors at shared-KV
  layers — should they be deleted from the loader (free memory)
  or just ignored? Ignoring is simpler; deleting saves a few MB.
- Stage 4 (real SWA windowed mask) is still pending. For 6-token
  prompts SWA is invisible; longer prompts will need it.

**Phase 4 recapture quickstart** (same as Phase 3 — only the
WebLLM side needs rebuild):

```bash
# 1. Restart capture-server
RUN_DIR=eval/reports/parity-gemma-4-e2b-shared-kv-$(date +%Y-%m-%d)
mkdir -p "$RUN_DIR"
cp eval/reports/parity-gemma-4-e2b-stage3-block0-2026-05-11/hf-ref.json \
   "$RUN_DIR/hf-ref.json"
lsof -ti:8035 | xargs kill -9 2>/dev/null
bun run eval/tools/parity-capture/capture-server.ts \
  --run-dir "$RUN_DIR" --port 8035 &

# 2. Rebuild bundle after code changes
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser

# 3. Re-run WebLLM capture (HF reference reused)
agentchrome --port 63846 navigate \
  "http://localhost:8031/parity-capture.html?model=gemma-4-e2b-it-q4km&inputIds=2,818,5279,529,7001,563&v=$(date +%s)"

# 4. Compare
uv run --no-project --with numpy python3 \
  eval/tools/parity-capture/compare.py --run-dir "$RUN_DIR"
```

**Phase 2 results (eval/reports/parity-*-2026-05-11/REPORT.md):**

| Run | Layer 0 cos | First sudden Δ ≤ -0.05 | End-stack cos | Top-16 / top-1 |
|---|---|---|---|---|
| TinyLlama Q4_0 | 0.9987 ✓ | none | 0.9855 | 15/16 · argmax MATCH (id 3681) |
| Gemma 4 E2B Q4_K_M | 0.9756 ⚠ | block 2 (−0.31) | 0.1450 | 2/16 · argmax MISS (web 236761 vs ref 9079 "Paris") |

Gemma 4 per-layer trace: 0.9756 / 0.9685 / **0.6591** / 0.8056 / 0.9375
/ 0.9090 / 0.8507 / 0.7925 / 0.8075 / 0.6730 / 0.5649 / 0.3436 / 0.6177
/ 0.5743 / 0.8445 / 0.5572 / … / 0.0335 at layer 34. The jitter (not
monotonic) is unusual — most architectural bugs cause smooth decay.
Possible signatures: per-block aliasing in `inpPerLayer` slicing,
non-deterministic op order in PLE inject, or numerical instability in
the `layer_output_scale` × residual chain.

**SWA pattern (confirmed via `window.parsedModel.hyperparams`):**
local layers 0,1,2,3,5,6,7,8,…,33; global layers 4,9,14,19,24,29,34
(every 5th). Local: head_dim=256, ropeBase=10000, ropeDim=256. Global:
head_dim=512, ropeBase=1e6, ropeDim=512, ropeFreqs.weight present.
FFN dim: 6144 for layers 0-14; 12288 for layers 15-34 (Gemma 4 E2B's
per-layer FFN-dim ladder). PLE / layerOutputScale / postAttnNorm /
postFfwNorm / qNorm / kNorm all present on every layer.

**The layer-2 drop is NOT a local/global transition** — layers 0,1,2,3
are all local SWA with identical hyperparams. So the bug is something
intrinsic to the per-block forward path that compounds non-uniformly.
Layer 0 already at 0.9756 (below 0.99 first-block gate) is itself a
finding: even FP32-vs-Q4 on block 0 should be ≥ 0.99 (precedent:
TinyLlama layer 0 at 0.9987).

**Where to start (Phase 3 plan):** add finer taps inside `forwardWith­
LayerTaps` for block 0 specifically so we can pinpoint the op that
first drifts. Phase 3 candidates:

1. **Embedding-output tap.** HF's `hidden_states[0]` is the embedding-
   table lookup BEFORE block 0 (already captured but discarded by
   `capture-hf-ref.py:114-117` — index `hidden[0]` is dropped).
   Surface `embedding_output_last_token` in both HF + webllm captures;
   compare. If embedding output already drifts, the bug is in
   `opGetRows(tokEmb, ids) + opScale(sqrt(1536))`. If embedding output
   matches HF perfectly, the bug is inside block 0.

2. **Within-block-0 checkpoint ladder.** Add an optional capture mode
   that taps the residual stream at 6 checkpoints inside block 0:
   (a) after `attn_norm + scaled` (pre-QKV); (b) after `qNorm/kNorm`
   on Q,K; (c) after RoPE on Q,K; (d) after attention out_proj +
   post_attention_norm; (e) after first residual add; (f) after FFN +
   post_ffw_norm + second residual add; (g) after PLE inject;
   (h) after layer_output_scale. Capture these for both HF (via
   forward hooks) and webllm.

3. **Compare ATTENTION OUTPUT specifically.** Gemma 3+ uses Q/K norms
   AND no attention softcap (Gemma 4 drops softcap entirely per
   gemma4.cpp:11). The Q/K norm + softmax + V chain has narrow
   surface area; tap before/after each piece. Bug candidates: the
   `qNorm/kNorm` is applied AFTER the per-head reshape rather than
   before (or vice versa); the per-head RMS-norm operates on the
   wrong tensor layout for Gemma's head_dim=256 SWA layers.

4. **Stretch: capture ALL token positions (not just last).** PLE
   injection per-block uses `slot[il]` of `inpPerLayer[pleDim,
   nTokens, layerCount]`. If the slicing is off-by-one or
   transposed, only some token columns get corrupt PLE data, and
   the "last token" tap would see only a partial picture. Capturing
   all token positions reveals per-position divergence shape.

**Where the data lives:** `eval/reports/parity-tinyllama-2026-05-11/`
+ `eval/reports/parity-gemma-4-e2b-2026-05-11/` (both have
`hf-ref.json`, `webllm.json`, `REPORT.md`). HF reference captures
take ~30s each on CPU (re-runnable; they're deterministic at fp32).

**Quickstart (recapture + compare from scratch):**
```bash
# 1. Start capture-server for the run dir
RUN_DIR=eval/reports/parity-gemma-4-e2b-$(date +%Y-%m-%d)
mkdir -p "$RUN_DIR"
bun run eval/tools/parity-capture/capture-server.ts \
  --run-dir "$RUN_DIR" --port 8035 &

# 2. HF reference capture (re-uses hfdownloader cache)
uv run --no-project --with-requirements \
  eval/tools/parity-capture/requirements.txt \
  python eval/tools/parity-capture/capture-hf-ref.py \
  --model unsloth/gemma-4-E2B-it \
  --inputs eval/tools/parity-capture/inputs.json \
  --output "$RUN_DIR/hf-ref.json" --add-bos

# 3. WebLLM capture (use HF's input_token_ids to isolate from tokenizer)
make smoke-serve &
agentchrome --port 63846 navigate \
  "http://localhost:8031/parity-capture.html?model=gemma-4-e2b-it-q4km&inputIds=2,818,5279,529,7001,563&v=$(date +%s)"

# 4. Compare
uv run --no-project --with-requirements \
  eval/tools/parity-capture/requirements.txt \
  python eval/tools/parity-capture/compare.py --run-dir "$RUN_DIR"
```

**Original Phase 2 plan (now CLOSED):** kept below for reference;
the workflow steps are still valid for re-running the captures.
The pickup instructions for Phase 3 are above.

**Phase 2 plan (this is the entire next session's scope):**

1. **TinyLlama parity sanity-check first.** Phase 2 work should land
   on a baseline that we trust. Capture HF reference + webllm
   tapped-forward for TinyLlama on the canonical prompt
   (`The capital of France is`), expect cosine ≥ 0.99 at every layer
   (TinyLlama is a known-good causal LM in the project). This is the
   "the pipeline works" gate before pointing it at Gemma 4.

2. **Run the HF capture** (still Phase 1 work that wasn't executed):
   ```bash
   hfdownloader download unsloth/gemma-4-E2B-it
   hfdownloader download TinyLlama/TinyLlama-1.1B-Chat-v1.0
   for model in unsloth/gemma-4-E2B-it TinyLlama/TinyLlama-1.1B-Chat-v1.0; do
     SLUG=$(echo "$model" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
     RUN_DIR=eval/reports/parity-$SLUG-$(date +%Y-%m-%d)
     mkdir -p "$RUN_DIR"
     uv run --no-project --with-requirements \
       eval/tools/parity-capture/requirements.txt \
       python eval/tools/parity-capture/capture-hf-ref.py \
       --model "$model" \
       --inputs eval/tools/parity-capture/inputs.json \
       --output "$RUN_DIR/hf-ref.json" \
       --add-bos
   done
   ```
   The TinyLlama capture is small (~2GB model, fp32 fast on CPU);
   Gemma 4 is ~5GB, slower but still tractable on CPU. Use `--device
   cuda` or `--device mps` if available.

3. **WebLLM tap instrumentation** in `src/inference/model-inference.ts`:
   - Add a new method `forwardWithLayerTaps(tokenIds, positions, options)`
     that mirrors `forwardSingle` but `graphBuildForwardExpand`s the
     residual stream after each block (per-layer `cur`) and reads each
     back after `graphCompute`. Returns
     `{ perLayerResidual: Float32Array[], finalNormHidden: Float32Array, logitsTop16: { ids: Int32Array, values: Float32Array } }`.
   - Gate the tap behind a constructor flag (`capturePerLayerTaps`) so
     production forwards stay zero-cost.
   - Last-token only: read back row `[nTokens-1, :]` from each tap;
     keeps memory under ~250 KB per capture.

4. **Browser harness** at `smoke-test/parity-capture.html` (mirror
   `real-model.html`'s loader, but instantiate ModelInference with
   `capturePerLayerTaps: true` and call `forwardWithLayerTaps` instead
   of going through chatCompletion). POSTs the resulting JSON to the
   capture server.

5. **Capture server** at `eval/tools/parity-capture/capture-server.ts`:
   - Bun HTTP server, configurable port (default 8035 — register in
     `~/.claude/used_ports.md`).
   - POST `/capture` → writes body to `<run-dir>/webllm.json`.
   - Single-file, no deps beyond Bun's built-in HTTP.

6. **Comparison driver** at `eval/tools/parity-capture/compare.py`:
   - Reads `<run-dir>/hf-ref.json` + `<run-dir>/webllm.json`.
   - Verifies `n_layer` and `n_embd` match.
   - Computes cosine + L2 per layer.
   - Emits `<run-dir>/REPORT.md` with a per-layer table + the
     "first layer below threshold" callout.
   - Optional `--threshold 0.95` arg (default 0.95 for end-of-stack,
     0.99 for first-block sanity).

7. **Run the full pipeline on TinyLlama first**, then Gemma 4. The
   TinyLlama REPORT.md should show cosine ≥ 0.99 at every layer
   (proves the tap + comparison work). The Gemma 4 REPORT.md should
   show a sudden drop at the buggy block — that's the diagnostic.

8. **Fix the localized bug** based on the first divergent layer. The
   op sequence at that block + its inputs (residual from the previous
   block) are the search space. Strip the tap instrumentation if
   it's behind a flag (no production code lives in the tap).

**Required reading before touching code (Phase 2 entry):**
1. `eval/tools/parity-capture/README.md` — workflow + format spec.
2. `eval/tools/parity-capture/capture-hf-ref.py` — already generic;
   no changes needed unless the JSON schema evolves.
3. `src/inference/model-inference.ts:forwardSingle` (around line 1170
   onward in current HEAD) — copy this method's structure into the
   new `forwardWithLayerTaps`. The tap = adding each block's
   `cur` to the graph's forward-expand list before `graphCompute`.
4. `smoke-test/real-model.html` + `smoke-test/real-model-page.js` —
   loader pattern to mirror for the capture harness.
5. `eval/live-server.ts` — example of a Bun HTTP server in this
   project; mirror its style for `capture-server.ts`.

**Why scaffold without running Phase 1 capture yet:** running it in
this session would only produce a half-comparison (no webllm side to
compare against). The capture script is checked in and ready; Phase 2
captures both sides + compares in one cohesive session, which is the
right unit of work.

**Estimated remaining work to ship Gemma 4 E2B:**
- Phase 2 build-out (~1 session — tap + harness + server + compare
  + first run).
- TinyLlama parity proves the pipeline (~30 min within Phase 2).
- Gemma 4 parity → identifies 1-2 architectural bugs (~30 min within
  Phase 2). Bug-fix → re-probe → re-compare loop continues until
  Stage 3 closes.
- Task 3.5 closure report (~30 min after Stage 3 structurally closes).
- Stage 4 (real SWA): 2 sub-tasks; may need 1 llama.cpp patch.
- Stage 5 (shared-KV + bench + closure): 5 sub-tasks; may need
  1 llama.cpp patch.

**Required reading before touching code:**
1. `docs/superpowers/specs/2026-05-11-gemma-4-stage3-embedding-scale-gelu-ffn-addendum.md`
   — the canonical follow-on spec covering 3.3f→3.3k; documents the
   BF16 cascade-corruption diagnosis and the rope_freqs investigation.
2. `eval/reports/bucket-d-probe-2026-04-29/capture-refs.py` — the
   established pattern for capturing HF transformers reference
   hidden states for parity comparison. Adapt for gemma-4-e2b: load
   `unsloth/gemma-4-E2B-it` via `transformers.AutoModelForCausalLM`,
   tokenize the same prompt webllm uses, capture per-layer residual
   stream via forward hooks, save JSON.
3. `src/inference/model-inference.ts:forwardForEmbedding` — tap
   point. Already returns the final hidden state; add a temporary
   intermediate-tap mode that captures EVERY layer's residual.
4. `src/inference/chat-template.ts:303` formatGemma4 — verify the
   exact prompt string passed to tokenizer.encode matches what HF's
   transformers tokenizer produces for the same chat messages.
   Compare token IDs lock-step.

**Task 3.3l — diagnostic plan (option (a) hidden-state comparison):**

1. Write `eval/reports/gemma-4-stage3-tap-points-2026-05-11/capture-refs.py` modeled on the bucket-D pattern. Use `hfdownloader download unsloth/gemma-4-E2B-it` first (per CLAUDE.md HF doctrine), then `uv run --no-project --with transformers ...` to load + tokenize + forward-with-hooks. Save per-layer residual stream as JSON.
2. Add a temporary instrumentation to `model-inference.ts:forwardForEmbedding` that captures per-layer residual after each block (graph tap; readback once at end). Run smoke probe with the same prompt as the python ref.
3. Compute cosine + L2 norm difference per layer. The FIRST layer where webllm diverges meaningfully from HF reference is the load-bearing missing piece.
4. Fix that piece. Re-run hidden-state comparison. Stage 3 closes when end-of-stack residual matches HF reference at cosine ≥ 0.95 (looser than the embedder parity gate since 35 layers compound error).
5. After Stage 3 closes structurally, run greedy smoke + the
   36-prompt eval ≥40% gate (Task 3.5).

**Last verified state (2026-05-11 EOS-7, after this session):**
- Branch `main` HEAD: pending Phase 4 commit. Tree dirty: pending
  shared-KV feat + TODO docs.
- `make checkall`: green (762 pass / 36 skip / 0 fail) post-Phase-4.
- WASM build current: `webllm-wasm.js` + `webllm-wasm.wasm` in
  `smoke-test/` from EOS-4 (no new WASM exports needed in Phase 4 —
  pure TS changes). wasm64 / mem64 targets NOT rebuilt.
- Bundle current: `smoke-test/webllm-bundle.js` rebuilt this session
  with `kvReuseFromLayer` + `buildQOnly` + shared-KV gates in all
  forward methods. Re-run `bun build src/index.ts --outfile
  smoke-test/webllm-bundle.js --target browser` after touching TS
  in Phase 5.
- Patch stack on `~/Repos/llama.cpp` branch `webllm-browser-patches`:
  9 patches (unchanged this session). Phase 4 fix was pure TS.
- **Gemma 4 chat smoke (greedy temp=0) post-Phase-4:** 13 tokens of
  `<eos>` (regressed from EOS-6's mixed-script noise — both were
  wrong; the `<eos>` regression is downstream of Phase 4 and tracks
  a chat-template tokenization issue, not a forward-pass bug).
- **Gemma 4 parity capture (raw input_ids):** end-stack cos 0.9722;
  top-1 argmax id 9079 ("Paris") MATCHES HF reference; top-16
  overlap 13/16. See `parity-gemma-4-e2b-shared-kv-2026-05-11/`.
- **TinyLlama smoke + parity:** unchanged. Phase 4 changes are
  predicate-gated on `hp.kvReuseFromLayer?.[il]`, set only for
  Gemma 4 family.
- agentchrome session on port 63846 active, tab id
  `094440A57C7855615A7AE1070C4FF61D`. `make smoke-serve` running on
  8031. Capture-server on 8035 killed at session end.
- GGUF symlink at `smoke-test/models/gemma-4-e2b-it-q4km.gguf` still
  in place.
- Pinned parity runs (gitignored under `eval/reports/`):
  - `parity-tinyllama-2026-05-11/` — Phase 2 baseline (PASS)
  - `parity-gemma-4-e2b-2026-05-11/` — Phase 2 Gemma 4 FAIL baseline
  - `parity-gemma-4-e2b-stage3-block0-2026-05-11/` — Phase 3 with
    embedding tap (FAIL; pre-fix snapshot)
  - `parity-gemma-4-e2b-attnscale-fix-2026-05-11/` — Phase 3
    post-fix (FAIL only at L15+)
  - `parity-tinyllama-attnscale-regression-2026-05-11/` — Phase 3
    regression-check (PASS)
  - `parity-gemma-4-e2b-shared-kv-2026-05-11/` — **Phase 4 closure
    run** (cosine 0.9722; argmax MATCH)

**Per-task commits this session (most-recent first):**
- `5db5e70` docs(TODO): Stage 3 — 3.3l Phase 3 closed; Phase 4 queued
- `78f12e1` **fix(gemma4): attention softmax scale = 1.0** (not
  1/sqrt(head_dim)) — load-bearing correctness fix; 8 call sites
  updated; embedding-output tap added on both HF + WebLLM sides
- `8975b9b` docs(TODO): Stage 3 — 3.3l Phase 2 closed; Phase 3 plan queued
- `b7c2e0f` **feat(parity-capture): Phase 2** — WebLLM tap + harness
  + server + compare driver
- `e67926b` docs(TODO): Stage 3 — 3.3l Phase 2 plan (from prior session)

Prior session (EOS-4) commits: `c317671` (Phase 1 scaffolding),
`db9ee8d` (3.3k closure + 3.3l queue), `dec6f2d` (3.3k rope_freqs),
`bac18f1` (3.3k spec).

**Workflow to resume (Task 3.3l Phase 4 — wire shared-KV):**

Detailed plan is in the "Phase 4 plan" block above. Quick checklist:

1. Read `~/Repos/llama.cpp/src/llama-kv-cache-iswa.cpp` and
   `llama-kv-cache.cpp:249` (the `reuse layer %d, is_swa = %d`
   log line + surrounding context) to confirm the layer-type-aware
   remap rule. Phase 4 plan above states the conjecture; verify
   before coding.
2. Add `kvReuseFromLayer[il]: number | null` to
   `ModelHyperparams` (or as a separate field on
   `ModelInference`). Compute in `src/models/model-loader.ts`
   from `sharedKvLayers` + `slidingWindowPattern`.
3. In `forwardSingle` and `forwardWithLayerTaps`, gate the K/V
   projection + KV-cache write block on `kvReuseFromLayer[il] ===
   null`. For shared layers, point `fullK` / `fullV` view ops at
   `this.kvLayers[kvReuseFromLayer[il]].{k,v}` instead.
4. Q projection stays per-layer at every layer.
5. Rebuild bundle: `bun build src/index.ts --outfile
   smoke-test/webllm-bundle.js --target browser`.
6. Restart capture-server pointed at a fresh run-dir; copy HF ref
   from `parity-gemma-4-e2b-stage3-block0-2026-05-11/hf-ref.json`
   (no re-capture needed — HF reference is deterministic at fp32).
7. Re-run parity. **Gate:** layers 15-34 cosines recover to
   ≥ 0.85 (target ≥ 0.93 like 0-14, but allow some slack since
   compounding error is unavoidable).
8. If layers 15-34 still drift, the remap rule is wrong → check
   actual llama.cpp KV-cache init logs (add a printf in
   `llama-kv-cache.cpp:249` if needed, rebuild llama.cpp side,
   skip — but we're not running libllama here, so the cleaner
   path is to instrument the WebLLM loader to log which slot it
   thinks layer 15 should reuse).
9. Run Gemma 4 chat smoke (greedy temp=0) — expect coherent English
   output, not mixed-script noise.
10. Run the 36-prompt eval on Gemma 4 (`make bench-*` for the
    targeted profile, or just the smoke bench task ID). **Gate:**
    ≥ 40% (Stage 3 closure target).

**Estimated remaining work to ship Gemma 4 E2B:**
- Task 3.3l Phase 4 (shared-KV) — ~1 session if the remap rule is
  straightforward; possibly 2 sessions if the iSWA cache plumbing
  needs deeper rework.
- Task 3.5 closure report (~30 min after Phase 4 closes structurally).
- Stage 4 (real SWA windowed mask): 2 sub-tasks; may need 1
  llama.cpp patch if ggml-webgpu's softmax mask can't express the
  windowed shape.
- Stage 5 (bench + closure — much of the original Stage 5 KV
  ref-sharing was pulled forward into Phase 4): 1-2 sub-tasks for
  bench + closure report; should be light.
- Total wall-clock budget: 2-3 more focused sessions to ship.

### Tier 3 migration to upstream `llama_decode` (REDIRECTED 2026-05-05)

**Status: P0 + P1 closed. P2 v1 reverted 2026-05-05 (18× decode regression — architectural mismatch).** P2-v2 (JSEP-style architecture) **Phase 1 research probe closed 2026-05-05; Phase 2 spec + plan written 2026-05-05, ready to execute.** P3-P5 deferred behind P2-v2 gate.

**Why redirected:** P2 v1 routed causal-LM through `ggml-webgpu` compiled inside WASM (Dawn / `emdawnwebgpu` port). Per-WebGPU-command JS↔WASM shim crossings under the emdawnwebgpu port dominated decode time. Path A investigation (commits `c8e1dc6` + `fe167aa`) ruled out graph caching, JSPI polling loops, and end-of-graph waits as bottlenecks; the cost is intrinsic to running `ggml-webgpu` inside WASM and not patchable in llama.cpp. The fix is architectural: route WebGPU calls from JS, not from WASM (the JSEP pattern that transformers.js + ORT-Web ships in production). Full reasoning: [`eval/reports/p2-causal-migration-2026-05-05/POST-MIGRATION-BENCH.md`](eval/reports/p2-causal-migration-2026-05-05/POST-MIGRATION-BENCH.md) §P2.1.A + §P2.1.B.

**Tier 3 spec (broader, governs P3-P5):**
[`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md)
(written before the P2-v2 redirect; still load-bearing for P3-P5).

**P2-v2 spec + plan (governs Phase 2 prototype):**
- Spec: [`docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md`](docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md)
- Plan: [`docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md`](docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md)

#### Phase 2 prototype CLOSED 2026-05-06 — Outcome D (chat works, JSEP dormant) confirmed by Task 11/12, then **OUTCOME E** banked by Task 13/14 synthetic probe (routing-layer validated: scheduler routes MUL_MAT to JSEP via `offload_op` when src on host_buf, `runOp = 1`; execution-layer blocked by missing cross-backend tensor-import shim — Phase 3 path firmed to **Option A-prime / full JSEP residency**). **Phase 3 in progress 2026-05-06: Stages 0/1/1.5/2 closed; Stage 3 (Q4_K matmul kernel) queued for fresh session — see "Next session pickup — Phase 3 Stage 3" below.**

- **Spec:** [`docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md`](docs/superpowers/specs/2026-05-05-p2-v2-jsep-prototype-design.md)
- **Plan:** [`docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md`](docs/superpowers/plans/2026-05-05-p2-v2-jsep-prototype.md)
- **Closure summary:** [`eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md`](eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md) (revised post-Task-8)
- **Spike diagnostic:** [`eval/reports/p2-v2-prototype-2026-05-05/SPIKE-RESULTS.md`](eval/reports/p2-v2-prototype-2026-05-05/SPIKE-RESULTS.md)
- **Disposition (post-Task-12):** **PARTIAL-UNBLOCK / OUTCOME D**. Task 11 (`039f448`) added the device-hint approach — `webllm_load_model` enumerates registered backends and, when the JSEP build is detected, passes `params.devices = {webgpu_dev, NULL}` to `llama_model_load_from_file` so libllama's GPU enumeration reduces to `[WebGPU]`. Task 12 spike (`?v=task11-1`) verified the chat completes correctly: greedy 5-token continuation of "The capital of France is" yields `"Paris.\n\n2"` (semantically correct), 12.30 ms/token decode (~81 tok/s), 238 ms prefill, 304 ms model load. Stderr confirms the device-hint fired: `[webllm] JSEP build detected: pinning libllama devices to WebGPU only` + `llama_prepare_model_devices: using device WebGPU (WebGPU) (unknown id) - 4095 MiB free`. **However**, JSEP `runOp` counter delta over the 5-token decode window is **0** (also `alloc/free/write/read/clear/sync` all zero). With weights+KV in `webgpu_buf` (not host_buf, not jsep_buf), the scheduler's `offload_op` path is gated off (`ggml_backend_buffer_is_host(src->buffer)` is false on webgpu_buf). JSEP is registered but structurally dormant during chat. **Conclusion (post-Task-12):** Phase 2's stated architectural goal — JSEP-routed kernels running inside a real chat decode — was not achieved. Phase 3 must own the kernel-coverage push. Two remaining paths: (1) **Option A-prime** — kernel SET_ROWS / GET_ROWS / ROPE / SOFT_MAX / MUL / ADD + matmul dtype permutations (~1k LOC); revert (or invert) the device-hint so libllama enumerates JSEP only and JSEP runs the full graph. (2) **Synthetic offload probe** — bypass libllama; build a tiny ggml graph with weights on host_buf and a MUL_MAT consumer; verify scheduler routes MUL_MAT to JSEP via `offload_op`. ~100 LOC. Doesn't unblock real chat but proves the kernel dispatch path correct in its native habitat — credible smoke for the Phase 2 work that exists.
- **Per-task commits:**
  - Task 0: `91e0396` (JSPI tensor-get-async fix) + `1094351` (pre-prototype baseline)
  - Tasks 1+2+4 amended: llama.cpp `webllm-browser-patches` `48acb658d` (single +1 patch)
  - Task 3: `09ba2d4` (TS runtime scaffold)
  - Task 4: `43390b0` (matmul kernel)
  - Task 5: `04a38cc` (rms_norm kernel)
  - Task 6: `d1a8348f` (engine integration + bundle wiring)
  - Task 7: `0f1973e` (counter wiring + jsep smoke harness + initial closure)
  - Task 7 SUMMARY: `4872307`
  - Task 8 spike + diagnosis: spike commit (`smoke-test/p2-v2-spike.{html,src.ts}`, Makefile bundle wire, SUMMARY.md revised)
  - Task 9 metadata-op allowlist: llama.cpp `7919d1839` (NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE in `supports_op` + graph_compute fast-path)
  - Task 10 supports_buft + offload_op: llama.cpp `49413d8e9` (broaden supports_buft to accept host + WebGPU bufts; add offload_op for MUL_MAT/RMS_NORM)
  - Task 11 device-hint unblock: webllm `039f448` (zero-patch — enumerate registered backends in `webllm_load_model`, pass `params.devices = {webgpu_dev, NULL}` when JSEP also registered)
  - Task 12 gate measurement: spike `?v=task11-1` (no commit — measurement only; recorded in SUMMARY.md "TL;DR (Task 11 + Task 12 update)")
  - Task 13 synthetic offload probe entry-point: webllm `4353594` (zero-patch — `webllm_synthetic_offload_probe` bridge function + spike harness)
  - Task 14 probe gate measurement: spike `?v=task14-3` (no commit — measurement only; routing validated via `jsep.counters.runOp = 1` after offload dispatch; execution failed in `dispatchMatmul` on missing host-buf tensor handle; recorded in SUMMARY.md "TL;DR (Task 13 + Task 14 update)")
  - Task 14 harness fixes: pending commit — cache-bust dynamic `webllm-wasm-jsep.js` import + `await` JSPI-promised probe call + add `webllm_synthetic_offload_probe` to JSPI_EXPORTS in `src/wasm/CMakeLists.txt` + post-throw counter snapshot
- **Patch stack:** llama.cpp `webllm-browser-patches` carries **+3 patches** (`48acb658d` Phase 2; `7919d1839` Task 9 metadata-op allowlist; `49413d8e9` Task 10 supports_buft + offload_op). **Phase 2 budget exhausted; Task 11 was zero-patch.**
- **Build infra unchanged for production path:** `make wasm-build` legacy artifacts unaffected; canonical-6 baseline unchanged. `make wasm-build-jsep` produces `webllm-wasm-jsep.{js,wasm}` + `webllm-bundle-jsep.js` + the new `p2-v2-spike.js`. `make checkall` green post-Task-11.

### Stage 4.36 closed — Phase 3 closed for testable subset (2026-05-08)

Canonical-6 JSEP parity sweep ran on the testable subset of the
canonical-6 (TinyLlama r2=8 from Stage 4.35; qwen3-0.6b r2=2;
qwen3-1.7b r2=2 added this stage). All three match the non-JSEP
`webllm-wasm.js` reference at `generatedIds[0]` and across all 5
greedy-decode tokens; both Qwen3 models predict
`[12095,13,576,6722,315]` (" Paris. The capital of") matching ref.
Probe 21b regression guard re-confirmed `P-21b-clean` (synthetic
shapes, dispatch-agnostic). `make checkall` green: 747 pass, 0 fail.
Closure: [`STAGE-4.36-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md).
Reference manifest: [`canonical6-refs.json`](eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json).

The remaining canonical-6 entries (mistral-7b-q4ks, llama-3.1-8b-iq3m,
qwen3-8b-iq3m) all exceed the wasm32 4 GiB JSEP heap cap
(`src/wasm/CMakeLists.txt:249`) and are deferred. Closure argues
mathematical interpolation: r2=4 (the only un-exercised value) is
structurally identical to the tested r2={2, 8} — the WGSL u32 divide
is parameterized with no branch on the quotient. Re-enablement paths
(wasm-mem64 JSEP build, streaming-loader spike, or accept
interpolation) catalogued in the closure report's "Deferred subset"
section. Path 3 (acceptance) is sufficient for Phase 3 closure;
paths 1/2 stay deferred unless a separate signal motivates 7B+
JSEP testing.

**Re-enablement attempt 2026-05-08 — §31-style cap probe, negative
result.** Spec + plan in `docs/superpowers/{specs,plans}/2026-05-08-
jsep-mem64-build-target*.md` (SUPERSEDED). Phase A Task A1 (CMakeLists.txt
refactor) shipped cleanly and was reverted; Task A2 hit
`static_assert(sizeof(void *) == 4, ...)` at `ggml-jsep.cpp:830` — a
deliberate JSEP wasm32-only guard against pointer truncation in
`EM_ASM_INT` calls. Deeper analysis surfaced a second blocker:
`host_mirror` (`ggml-jsep.cpp:251`) duplicates every weight inside
the wasm heap regardless of cap, so the streaming-loader path also
covers 0 of 3 deferred models. Both lifts (mem64 build flag +
streaming-loader) are blocked at architectural layers we deliberately
built into JSEP during Phases 0-3. Closure report:
[`eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`](eval/reports/jsep-mem64-2026-05-08/SUMMARY.md).
Path 3 (mathematical interpolation acceptance) remains the operative
closure; paths 1/2 stay deferred behind the documented blockers.

Per-stage stubs (Stage 3 + Stage 4.1 → 4.36) archived to
[`TODO_ARCHIVE.md`](TODO_ARCHIVE.md) — see *Phase 3 JSEP causal-LM
decode investigation* section. The archive holds both the per-stage
**briefs** (the pre-stage queue items) and the verbose per-stage
**CLOSED paragraphs** that previously lived in this file under
"Phase 3 progress (2026-05-06)" (moved 2026-05-08 per CLAUDE.md
archival cadence). Per-stage closure reports stay under
`eval/reports/p2-v2-option-a-prime-2026-05-06/`.

### qwen3-0.6b-thinking-cold semantic-reasoning wedge (CLOSED 2026-05-08; archived from TODO.md)

Closed 2026-05-08 — root cause attributed upstream (Dawn/Chrome
WebGPU process pressure: "Device lost! Reason: 1, Message: A valid
external Instance reference no longer exists" surfaces as wasm
`unreachable` via `GGML_ABORT` at `ggml-webgpu.cpp:543`). Probe A
(emb-001 alone, post-warmup) passed — cumulative-state confirmed,
input-specificity ruled out. Wedge rate: ~30% session / ~20%
lifetime, qwen3-0.6b-thinking-cold only (qwen3-1.7b / llama-3.2-1b /
qwen3-8b: 0/11 wedges in DB). **No structural code shipped** — the
cumulative state we control (`engine.resetConversation` per task) is
already exhausted; a real fix needs WebGPU device recreation between
runs (uncertain efficacy, ~37 min wall per fix-validation pass).
**Abort guard `f3cbca9` is the operative mitigation** (verified Run
#2 in the investigation: wedged at task 35-38, guard fired, no row
written to `evals`, bench summary surfaced `[FAIL]` cleanly).
Closure report:
[`eval/reports/qwen3-0.6b-thinking-cold-wedge-2026-05-08/SUMMARY.md`](eval/reports/qwen3-0.6b-thinking-cold-wedge-2026-05-08/SUMMARY.md).
Re-evaluation triggers (rate >50%, second model wedges, upstream
device-lifecycle change, Chrome/Dawn fix lands, consumer report)
catalogued in the closure report. Full block (queued hypothesis
surface, paste-and-go bootstrap, implementation steps, risk
register, branch outcomes) archived to `TODO_ARCHIVE.md` under
"qwen3-0.6b-thinking-cold semantic-reasoning wedge investigation".

### 13B target registration (CLOSED 2026-04-29; archived from TODO.md)

Closed 2026-04-29 — all 6 phases (probe → register → smoke →
36-prompt eval → smoke-bench → report) passed in one session.

- **Eval:** 34/36 = **94%** (gate ≥60%) — **new fleet accuracy
  leader** (was qwen3-4B at 88-90%).
- **Speed:** 3-run smoke-bench median **18.9 tok/s** (gate ≥12,
  predicted band 15-19 — top of band).
- Closure report:
  [`eval/reports/13b-validation-2026-04-29/SUMMARY.md`](eval/reports/13b-validation-2026-04-29/SUMMARY.md).
- Registration commit `a4c8189`; closure commit `c3063fd`.
- Full block (rationale, phasing, risk register, alternates)
  archived to `TODO_ARCHIVE.md`.

### Phi-3 causal LM support (CLOSED 2026-04-29; archived from TODO.md)

Closed 2026-04-29 — all 6 phases passed; first fused-projection causal
LM in the fleet (Path B fused-forward, `architecture === "phi3"`-gated).
**Eval 27/36 = 72%; smoke-bench 31.6 tok/s.** Implementation commit
`8392bca`; bug-fix commits `7915abb` + `7c85a2a`; closure commit
`31612a2`. Plan
[`docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md`](docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md);
closure report
[`eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`](eval/reports/phi-3-validation-2026-04-29/SUMMARY.md).
Full block (eval/speed details, Path A vs Path B note for the next
fused-projection arch) archived to `TODO_ARCHIVE.md` under "Phi-3
causal LM support (closed 2026-04-29)".

### Chat-template family dispatch hardening (CLOSED 2026-05-04; archived from TODO.md)

Three-layer fix shipped 2026-05-04 — sampling profile + chat-stop
registration + template formatter must all agree per family or the
model wanders past end-of-turn. Smoke audit expanded 4 → 19 GGUFs
(commit `da720a6`); engine chat-stop registration widened for
ChatML/Gemma/Mistral families (`c3d8261`); Mistral-Instruct
formatter + `MISTRAL_DEFAULTS` sampling profile auto-dispatch
(`dafe4b4` / `27aacef` / `1f064e9`). Memory note:
`feedback_chat_template_family_dispatch.md`. Full block (per-layer
detail + per-family rationale) archived to `TODO_ARCHIVE.md` under
"Chat-template family dispatch hardening".

### Next-session pickup batch (queued 2026-04-29; CLOSED 2026-05-03; archived from TODO.md)

The 11-item pickup queue active 2026-04-29 → 2026-05-03 — bridging
the §31b/§32 perf-cycle close and the start of the P2-v2 / Phase 3
JSEP work — closed in full. Items shipped: TS API audit follow-ups
(item 4), embedding bucket C (item 5), embedding bucket D (item 6),
Phi-3.5-mini bucket D extension §28 NEGATIVE (item 7), frame-probe
coexistence baseline (item 8), NPC scenario sizing probes 9a-9d
(item 9), dual-mode deployment (item 10), prefix-cache mechanism +
persistence + worker migration (item 11 + sub-#5/#6). Phi-3 closure
follow-ups (item 2): (a)+(b) closed; (c) Path A vs Path B A/B
deferred until next fused-projection arch is queued.

Daily upstream cadence check (item 1) and pre-rebase baseline
freshness (item 3) carry forward in the "Watch list / optional
cadence work" section below — same procedure, fresh location.

Two queued sub-follow-ups carry forward in "External-trigger
candidates":
- **Storage B (GPU-resident KV)** — defer until per-call overhead
  measured against real harness usage and the API has stabilized;
  requires `ggml-webgpu` patches.
- **Concurrent in-flight per conversation** — defer until a
  consumer asks; requires KV cloning at concurrency request time.

Full block (per-item rationale, commit ledger, closure-report links,
deferred follow-up provenance) archived to `TODO_ARCHIVE.md` under
"Next-session pickup batch (queued 2026-04-29; closed 2026-05-03)".

---

**Embedding fleet expansion (CLOSED 2026-04-28; archived from TODO.md).**
Buckets A (BGE small/large via commit `41b27bd`) and B (jina + nomic
non-BERT arch — ALiBi/GeGLU + NEOX-RoPE/fused-QKV/SwiGLU, 5/5 parity
each via 10 commits) shipped. Dashboard now shows 6 embedding rows
covering the full BERT-family lever portfolio. Encoder fixed-cost-
per-dispatch observation pinned 2026-04-28 (~5.2-5.7 µs flat across
450 → 805 dispatch/token range; encoder share scales inversely with
model size — captured in vault note `Knowledge/wasm-webgpu-encoder-
fixed-cost-per-dispatch.md`, promoted to watch list under the 8B
ceiling). Full block (commit ledger, per-bug provenance, scaling
table) archived to `TODO_ARCHIVE.md` under "Embedding-model
expansion campaign (closed 2026-04-28)".


### Bucket B follow-ups (post-closure, 2026-04-28) — CLOSED; archived

Both queued follow-up items (#11 spec accuracy patch; #12 vault-save
bucket B doctrines — 4 notes landed under `~/ClaudeVault/`) closed
2026-04-28. Full block (per-section spec-patch breakdown, four
vault-note descriptions with cross-link map, MANIFEST verification)
archived to `TODO_ARCHIVE.md` under "Bucket B follow-ups (post-closure,
2026-04-28)".

---

### Embedding-model expansion candidates (queued 2026-04-28; CLOSED 2026-04-29; archived from TODO.md)

Three-bucket campaign closed end-to-end. Bucket A (BGE small/large
register-and-run) and Bucket B (jina + nomic non-BERT arch — ALiBi,
GeGLU, NEOX-RoPE, fused QKV, SwiGLU, mixed biases) closed
2026-04-28. Bucket C (Qwen3-Embedding-0.6B-hyb causal-LM-derived)
closed 2026-04-29; Qwen3-Embedding-4B/8B remain register-and-run on
top of bucket C foundation if a future MTEB ask names them. Full
block (per-bucket scope, commit ledger, latent-bug provenance,
encoder-lever portfolio dashboard, post-cycle stretch picks)
archived to `TODO_ARCHIVE.md` under "Embedding-model expansion
campaign (closed 2026-04-28)".

---

### Fresh next-step candidates (2026-04-28) — closed; see `TODO_ARCHIVE.md`

All nine 2026-04-28 closed items (§32a profile-mode rebench, §31b
MAXIMUM_MEMORY upper-bound probe, patch-12 squash, dashboard refresh,
pre-rebase baseline capture, §32 SUMMARY cross-link, TODO header pin
refresh, BENCHMARKS tier expansion, CLAUDE.md doctrine capture)
live in `TODO_ARCHIVE.md`. No fresh candidates queued.


### Watch list / optional cadence work (2026-04-28)

Backlog is fully cleared. Upstream cadence check 2026-04-28: 2 commits
since `f9f33654a` (Nemotron Nano 3 Omni model support; `common/`
reasoning-budget helper) — **neither touches `ggml/src/ggml-webgpu/`**;
no rebase trigger near firing. Below items are honest candidates if
appetite remains; none are forced.

10. ~~**Vault-save the project's reusable doctrines.**~~ **DONE
    2026-04-28** — 5 notes landed:
    - `Patterns/rebase-and-sweep-cycle-template.md` (§27 / §28 / §32
      templates with decision rules)
    - `Patterns/cap-probe-bump-first-doctrine.md` (§31b lesson —
      bump first, characterize second)
    - `Patterns/pre-rebase-baseline-doctrine.md` (§32a lesson —
      same-model pre/post bucket comparison)
    - `Patterns/patch-stack-squash-via-cherry-pick-chain.md`
      (post-§32 patch-12 cleanup; confidence gate = artifact
      byte-identity pre/post)
    - `Knowledge/wasm-webgpu-encoder-fixed-cost-per-dispatch.md`
      (~5.2-5.7 µs/dispatch flat across the canonical 6 fleet)

    Probe-first doctrine generalization deliberately *not* duplicated
    — `Patterns/probe-first-methodology-validates-architecture-pivots.md`
    (2026-04-28, pre-existing) already covers the general doctrine
    via the §31 wasm64 narrative. The 5 new notes cross-link to it.
    Index rebuilt: 3056 notes / 970 tags / 9 MANIFESTs. All 5 notes
    verified in their respective folder MANIFESTs.

**Watch-list items (no action now; flag for next session):**

- **Pre-rebase baseline freshness window.** `eval/reports/pre-rebase-
  baselines-2026-04-28/` matrix is fresh as of 2026-04-28; freshness
  window ~1 month. **Re-capture if no rebase triggers fire by
  ~2026-05-28** — if the matrix ages past freshness without being
  consumed, the §32a process-improvement value evaporates.
  Procedure: `make smoke-bench PERF_MODEL=<m> PERF_RUNS=3` × 6
  canonical models, ~18 min wall. Only do this on the "stale-matrix
  + still-no-rebase-ETA" branch; otherwise let the next rebase
  trigger consume the matrix and start a fresh capture.
- **Upstream cadence check — DAILY.** Procedure: `cd ~/Repos/llama.cpp
  && git fetch origin && git log webllm-browser-patches..origin/master
  --oneline -- ggml/src/ggml-webgpu/ ggml/include/`. **If the result
  is non-empty**, a rebase trigger has fired — apply the §32 procedure
  (rebase, sweep, classify per §27/§28/§32 templates). **If empty**,
  log and skip. Cadence policy set 2026-04-29: run daily even when
  the surface has been quiet, since the cost is ~30s and a missed
  rebase costs much more than catching one promptly. Last fired:
  **2026-05-04** (§27 hybrid — `d4b0c22f9` ggml-webgpu LAYER_NORM
  upstream landing subsumed local patches `72b6d001e` +
  `c775ac26d`; rebased to upstream tip `a817a22bc`, local tip
  `fc1f81242`, patch stack 11 → 9; encoder parity PASS; perf
  neutral vs 2026-05-01 cross-day baseline; mistral-7b -14%
  outlier flagged. Sweep matrix:
  [`eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md`](eval/reports/llama-cpp-rebase-2026-05-04/SUMMARY.md)).
  Last clean run: **2026-05-08** (clean — `master` advanced past
  `d5003b6e4`; the only hit on `ggml/src/ggml-webgpu/` or
  `ggml/include/` since the 2026-05-04 §27-hybrid rebase is
  upstream `a8fd165fe` (CUDA: lower-case PCI bus id, standardize
  for ggml — #22820), a comment-only edit in `ggml-backend.h`
  with no API surface change and no WebGPU touch). Prior clean
  run: 2026-05-05.
- **Test skip count.** Currently 33 (rebaselined 2026-05-03 after
  prefix-cache mechanism + persistence ship), all environmental:
  - `pipeline-cache.test.ts` × 5 (`!indexedDBAvailable` — IndexedDB
    is a browser API, missing in Bun native)
  - `persistence-indexeddb-store.test.ts` × 8 (`!indexedDBAvailable`,
    Task 3.1 of prefix-cache persistence)
  - `engine-conversation-persistence.test.ts` × 10 (`!HAS_WEBGPU ||
    !existsSync(TINYLLAMA)`, Tasks 1.2 + 1.3 — 3 export tests + 3
    import tests + 4 round-trip / fingerprint-mismatch / corrupt-magic
    extensions)
  - `kv-snapshot-roundtrip.test.ts` × 3 (`!HAS_WEBGPU`,
    serializeKVCache / loadKVCache primitives from prefix-cache
    mechanism)
  - `forward-verify-equivalence.test.ts` × 1 describe (`!HAS_WEBGPU
    || !existsSync(TINYLLAMA)`)
  - `prefill-tiling-equivalence.test.ts` × 1 describe (`!HAS_WEBGPU`)
  - `speculative-integration.test.ts` × 1 describe (`!HAS_WEBGPU
    || !existsSync(TINYLLAMA)`)
  - `wordpiece-golden.test.ts` × 1 describe (`!fixturesPresent` —
    opt-in HF golden fixtures)
  - residue × 3 across `causal-embedder-inference`,
    `chat-template-special-tokens`, `model-inference-embed`
    (per-test fixture-presence guards)

  These are correct safety guards, not bugs or side-branch leftovers.
  Watch for *new* skips appearing beyond this 33-count baseline — that
  might indicate an accidental regression. Browser-side smoke tests
  cover the WebGPU + IndexedDB code paths that skip-pass here.
- **Tool-format investigations — CLOSED 2026-05-04.** Both probes
  ran; investigation report at
  [`eval/reports/tool-format-investigation-2026-05-04/SUMMARY.md`](eval/reports/tool-format-investigation-2026-05-04/SUMMARY.md).
  - **Llama-3.x — RECLASSIFIED.** Not a chat-template format issue.
    Llama-3.1-8B already scores 0.98 on the current Qwen3 path — no
    formatter swap would help. Sub-8B Llama emits *structurally-
    shaped but malformed* attempts (Llama-3.2-3B: XML-inside-XML
    wrapper; 1B: header-parroting; Hermes-3-3B: fabricated JSON
    keys). **Right fix is parser-side, not template-side.** Filed
    follow-up: **"Tool-call parser leniency for sub-8B Llama-3
    family emissions"** (see Watch list new entry).
  - **Phi-3 — CLOSED no-fix.** Phi-3.5-mini emits zero structural
    tool-call signal (no `<tool_call>` tags, no JSON, no `name`/
    `arguments` references) — hallucinates results conversationally.
    Phi-3-instruct base is not function-calling-fine-tuned; no
    learned format exists to elicit. Microsoft documentation
    confirms: no Phi-3 function-calling format. **Action:** demote
    tool-calling expectations for Phi-3.5-mini in dashboard /
    per-model docs. Strong on reasoning (1.00) + instruction-
    following (0.76); accept the 0.17 tool-calling floor.
- **Tool-call parser leniency for sub-8B Llama-3 family — CLOSED 2026-05-04.**
  `XML_NESTED_RE` shipped in `src/characters/tool-system.ts`; 5 new
  tests in `tests/tool-system.test.ts` cover positive (Llama-3.2-3B
  JSON-in-XML), negative (pure-XML args graceful no-parse), and
  negative (missing args tag). Tool-calling-dimension re-bench at
  greedy temp=0:
  - **llama-3.2-3b: 0.17 → 0.48** (+0.31 absolute, +186% relative)
  - **llama-3.2-1b: 0.17 → 0.56** (+0.39, +234% — surprise: lifted
    more than 3B; greedy decoding alone removes the parroting noise
    that dominated at temp=0.6)
  - **hermes-3-3b: 0.17 → 0.23** (+0.06; confirms diagnosis —
    mangled JSON not parseable at any layer)
  Closure report appended to
  [`eval/reports/tool-format-investigation-2026-05-04/SUMMARY.md`](eval/reports/tool-format-investigation-2026-05-04/SUMMARY.md);
  re-bench artifacts at
  [`eval/reports/parser-leniency-rebench-2026-05-04/`](eval/reports/parser-leniency-rebench-2026-05-04/).
  Process lesson codified: "greedy temp + lenient parsing is the
  right combination for sub-8B Llama-3 tool-calling" — not just one
  or the other.
- **Encoder parity reference vectors freshness.** `eval/reports/
  encoder-parity-2026-04-28/{jina,nomic}-ref.json` are pinned to
  whatever sentence-transformers / HF model versions resolved on
  2026-04-28. Reproducible via `capture-refs.py` (uv-driven, the
  HF-side weights are content-addressed). **Re-capture if the
  parity gate ever fires a regression after a Phase 2 forward-graph
  change** to confirm the regression is local (not a reference
  drift). Otherwise leave pinned; the gate is a known-good fixture.
- **TS API audit follow-ups — surfaced from final review (2026-04-29).**
  Two of three queued items closed 2026-04-30; the third deferred
  to its own cycle pending a larger fix.
  - **Sampling-dispatch unit test. CLOSED 2026-04-30** (commit
    `b76b546`). Extracted the dispatch ladder from `engine.ts` into
    a pure `resolveSamplingParams()` helper in
    `src/core/sampling-profiles.ts` and unit-tested the full matrix
    (4 modes × `enableThinking` × Qwen-vs-non-Qwen × consumer override
    × per-field precedence × consumer 0 honored as a real override).
    13 new tests in `tests/sampling-dispatch.test.ts`. Engine call
    site reduced to a single named call against the same shape;
    identical behavior.
  - **Tool-schema mirror-drift sentinel. CLOSED 2026-04-30** (commit
    `76acf05`). Promoted the `JsonSchemaParameterType` literal union
    to a single exported type in `core/chat-types.ts`; the two other
    mirrors (`inference/chat-template.ts`'s `ChatTemplateToolSchema`,
    `characters/tool-system.ts`'s `ToolParameter`) now import it.
    Comment-based "lock-step" contract upgraded to a typecheck
    error. Pure type-only refactor — zero runtime cost.
- **`tsconfig.json` widening to enforce test typechecks. CLOSED 2026-04-30**
  (commit `3328f8e`). Added `tsconfig.test.json` (extends main config)
  with `include: ["src/**/*.ts", "tests/**/*.ts"]`,
  `allowImportingTsExtensions: true` (safe under `noEmit`; resolves
  the 6 TS5097 leaks where tests import `.ts` from `eval/`), and
  `noUnusedLocals/Parameters` off (tests declare fixture vars not
  always used). Wired into ship gate via new `typecheck:tests` script
  (`package.json`) and `typecheck-tests` Makefile target; `checkall`
  now runs it after the production typecheck.

  Probe found 101 latent errors (vs watch-list estimate of 5-15);
  driven to 0 file-by-file across 14 fix commits (`4114138` through
  `bb802b6`). Patterns used: per-test cast-through-unknown helpers,
  `smoke-test/*.d.ts` ambient shims for the three JS files tests
  import, and an `EngineInternals` helper in
  `engine-streaming-api.test.ts` that centralizes the unsafe
  private-field access at one boundary instead of 18.

  The load-bearing finding was the **TS2578 audit signal**: 4 unused
  `@ts-expect-error` directives in `tests/generation-config-public.test.ts`
  were silently inactive because the directives sat one line above the
  property TS reports the error on. Fixed in `9890b31`. The public
  `GenerationConfig` type was correct as-is; the test was the bug.
  Convention going forward: place `@ts-expect-error` on the property
  line, not the variable line.

### External-trigger candidates

Five open candidates, all conditional:

- **Prefix-cache Storage B (GPU-resident KV).** Trigger: per-call
  overhead measured against real harness usage + API stable enough
  for a deployment ask. Carried forward from prefix-cache item 11
  follow-up #3 (closed 2026-05-02 batch). Requires `ggml-webgpu`
  patches.

- **Prefix-cache concurrent in-flight per conversation.** Trigger:
  consumer ask. Carried forward from prefix-cache item 11 follow-up
  #4 (closed 2026-05-02 batch). Requires KV cloning at concurrency
  request time.

- ~~**MEMORY64 full bridge migration**~~ → **CLOSED 2026-04-29.**
  All 8 phases shipped. Production wasm64 originally needed a
  build-time bind-group shim patch; on 2026-04-29 the Emdawnwebgpu
  port was bumped to Dawn `v20260423.175430` (post `8d78be5`), the
  patch script was deleted, and Mistral-Nemo Q4_K_S validated
  end-to-end at 26.7 tok/s. Closure report:
  [`eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`](eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md).

- **§D concat-graph batched encoder compute.** Trigger: a real
  batch-encoder-throughput use-case (was non-goal in §21). The
  encoder is dispatch-bound; the only structural lever left is
  amortizing dispatches across a batched concat-graph. §27
  rebase didn't deliver an encoder-side free win.

- **Upstream `ggml-webgpu` rebase + free-win sweep.** Trigger:
  upstream actually moves on the `ggml-webgpu/` surface again.
  Running base is now `a95a11e5b` (post-§28-on-#22504, see closure
  below). Mechanical sweep on the 6-model fleet against the new
  baselines (tinyllama-q4_0 ~107, qwen3-0.6b ~87, qwen3-1.7b ~61,
  mistral-7b-q4ks ~35, llama-3.1-8b-iq3m ~27, qwen3-8b-iq3m ~26).
  Three documented outcome templates: §27 (free-win), §28
  (negative result, lever closed harder), §32 (small regression,
  accepted). Pick the template that matches the data and document.

  **Last rebase 2026-04-30 — #22504 fast-iquant-matmul / #22514
  vulkan-tensor-2d → §28 negative result.** Hypothesis: #22504
  delivers iquant gains analogous to #22344's +80% on
  qwen3-8b-iq3m. Outcome: NO measured gain on either IQ3_M model
  (-0.5 to -0.9 tok/s deltas; same-day same-environment rebase
  delta on tinyllama -1.6%, well within sampling noise). Rebase
  adopted anyway (option value of staying current, §32-style).
  Closure report:
  [`eval/reports/post-rebase-22504-2026-04-30/SUMMARY.md`](eval/reports/post-rebase-22504-2026-04-30/SUMMARY.md).
  Process lesson codified: any cross-day comparison vs a saved
  baseline now requires a same-day same-tip control as the first
  data point in the bench window — see SUMMARY's "Process notes".

### Deferred (out of scope per current ceilings)

- **§C-v2-A resurrection.** Side branch `feat/spec-decode-v2-greedy`
  retains the entire driver, AdaptiveGate, K+1 verify, contract gate,
  and ~30 tests. The only remaining theoretical resurrection path was
  a 70B+ target via MEMORY64 (target/drafter ratio 13× → ~100×); both
  the prior 30B ceiling (2026-04-28) and the current **8B ceiling
  (2026-04-29)** defer this. Tip `4e11d79`. **Do not merge.**
  Re-evaluate only if the model-size ceiling lifts dramatically (no
  current trigger; agent + Three.js use case will not approach 70B).

- **Wave-1 architecture Gemma 2.** 5+ gaps (pre+post norm pairs,
  logit/attn soft-cap, sliding-window, (1+w) RMSNorm). Re-evaluate
  if a model in the family lands as a real deployment ask. Phi-3
  closed 2026-04-29 — see CLOSED stub above.

### Process notes

- For any non-trivial lever pick, the established pattern is:
  invoke `superpowers:writing-plans` with the chosen scope, then
  execute via `superpowers:subagent-driven-development` (per
  global preference). Mirror §17/§18/§19/§20 plan structure:
  explicit phases, measurable gates, measure-and-close decision
  rule.
- Probes (§29, §32a, §31b) are short enough to skip the formal
  plan cycle and execute inline against the existing spec /
  closure reports — match the §29 verify-cost-probe pattern (one
  agentchrome js-exec or one make-target run, one writeup
  commit, done).
- **Cap probes — bump first, characterize second** (lesson from
  §31b). When a measurement hits a cap at a configurable value,
  immediately try bumping the configuration to confirm whether
  the cap is configuration-bound or toolchain/runtime-bound. The
  bump is cheap (1 line + 1 rebuild attempt). §31a missed this
  step and landed a "configured-ceiling-bound, not hardware-
  bound" framing that understated the constraint by one layer
  (the configuration ceiling *was* the toolchain ceiling).
- **Pre-rebase baselines** (lesson from §32a). When a rebase
  cycle plans a `bench` sweep for outcome classification, capture
  profile-mode traces on the same model fleet pre-rebase. Post-
  rebase regressions that classify as "small, accepted" can then
  be diagnosed via same-model pre/post deltas instead of cross-
  model proxies. ~15 min one-time cost; saves a follow-on probe
  when a regression fires.
- **History rewrites for forward fix-ups** (lesson from §32 →
  patch-12-squash). When a rebase produces a semantic conflict
  resolved by a forward-fix-up commit (rather than amending the
  upstream-touching patch in place to avoid history rewriting),
  schedule a squash cleanup pass in a follow-on cycle. The squash
  is mechanically: cherry-pick chain on a temp branch with
  `cherry-pick --no-commit <fixup>` followed by `commit --amend
  --no-edit` to fold the fixup into its target. Confidence gate
  is **WASM byte-identity pre/post squash** — if the artifact
  matches, the squash is semantically a no-op and safe to swap.

#### Archived: How to test §A lever 1 — see `TODO_ARCHIVE.md`

Closed 2026-04-26 — §17. Full pre-change baseline shell snippet,
expected-results map, and reproduction instructions preserved in
the archive.

#### Operational gotchas

- **Cold-shader artifact.** The smoke page does a shader-
  cache warmup after [6/8] engine adoption. If you see
  "1.0 tok/s" after a fresh WASM rebuild, the warmup
  didn't run — investigate before investigating "the
  engine."
- **Bench-profile timeout on 8B+.** If chat-smoke times
  out at 180s, run `make smoke-stop && make smoke-restart`
  to clear stale agentchrome state, then retry. Not a
  regression in the bench harness.
- **`make smoke-bench` bundles `--profile`; `make
  bench-inference` does not.** Use the former for backend
  attribution (perturbs timing); the latter for canonical
  steady-state tok/s. The §16 entry has both for qwen3-8b.

### Historical context, Active next steps, Deferred kernel-tuning targets — see `TODO_ARCHIVE.md`

The Apr-23 smoke-bench regression diagnosis (resolved 2026-04-25 —
sampler-config methodology change, not an engine regression), the
§1-§10 active-next-steps work that drove the wave-1 + wave-2 model
campaigns through 2026-04-26, and the deferred §A-§D kernel-tuning
target portfolio (now all closed via §17 / §19 / §20 / §21 / §24 /
§26) live in `TODO_ARCHIVE.md`. Items 11-12 (3+ binding buffer-
conflict edge case; JSPI feasibility checkpoint) are also archived
there as latent follow-ups, not active work.


## Environment

```bash
source ~/emsdk/emsdk_env.sh
make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
cd smoke-test && python3 -m http.server 8031
# Cache-bust: ?v=$(date +%s)
```

## Local Dependencies

This repo depends on a local patched llama.cpp at `~/Repos/llama.cpp/` on branch
**`webllm-browser-patches`**. Patches (full inventory in `docs/LLAMA_CPP_PATCHES.md`):
1. `ggml: iterative ggml_visit_parents_graph for WASM stack safety`
2. `ggml-webgpu: browser + ASYNCIFY support bundle`
3. `ggml-webgpu: request-based browser readback API`
4. `ggml-webgpu: harden async readback request cleanup`
5. `ggml-webgpu: notify browser async readback completion`
6. `ggml-webgpu: add opt-in browser graph profiling`
7. `ggml-webgpu: specialize browser decode matmul dispatch` (paired with patch 8)
8. `Revert "ggml-webgpu: specialize browser decode matmul dispatch"` (effective no-op vs patch 6)
9. `ggml-webgpu: add GGML_OP_NORM (LayerNorm) support` — load-bearing for the BERT encoder path; without it `engine.embed()` returns bit-identical output for every input.
