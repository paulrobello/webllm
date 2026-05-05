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

### Tier 3 migration to upstream `llama_decode` (NEW DIRECTION 2026-05-05)

**Status:** **P1 UNBLOCKED 2026-05-05 via JSPI pivot (commit
`b4d4b48`).** Dropped `-sASYNCIFY` for `-sJSPI` by flipping
`GGML_WEBGPU_JSPI=ON` in `src/wasm/CMakeLists.txt`. JSPI uses native
`WebAssembly.promising` / `Suspending` and sidesteps the
Asyncify+exceptions+ctors trap that previously fired in
`__wasm_call_ctors` whenever a new wasm export was added. P0 spike
PASSES end-to-end on the JSPI build with all four new P1 exports
(`webllm_tokenize`, `_detokenize`, `_token_bos`, `_token_eos`)
re-applied. Required follow-on changes: `JSPI_EXPORTS` uses bare
wasm export names (no leading `_`); TS bridge methods that call
JSPI-promised exports must `await` (every JSPI-listed export now
returns a Promise even when no actual suspend fires, unlike Asyncify
which only Promised-wrapped on real suspends). Closure +
remaining-work breakdown in
[`eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md`](eval/reports/p1-tokenizer-2026-05-05/SUMMARY.md).
**P1 parity status: PASS — 1000/1000 byte-exact across all 5
vocabs** (spm-llama, llama-bpe, qwen2, qwen3, wordpiece-bert; each
200/200, run per-vocab via `?only=<vocab>` filter). Fixture was
regenerated from canonical `llama_tokenize` via the new
`smoke-test/p1-fixture-regen.html` browser harness +
`/save-parity-fixture` POST endpoint on `eval/smoke-serve.ts`.
`LlamaTokenizer` gained an `encoderOnly` option that flips
`addBos=true` for BERT-family vocabs (BERT's BOS IS `[CLS]`).
**Cross-model load fix:** `webllm_load_model` now uses
`use_mmap=false` and `std::remove()` after load. The wasm32 4 GiB
cap that previously fired on the 2nd sequential model load is
closed; the parity harness now runs all 5 vocabs in a single
page load (1000/1000 byte-exact) instead of needing per-vocab
`?only=` isolation. Reasoning: under Emscripten MEMFS mmap pins
the file Uint8Array for the model lifetime *plus* keeps a mapped
view on the heap, doubling the peak; and successive `fopen("wb")`
truncates didn't recycle MEMFS entry storage so 770 MiB +
1017 MiB + 64 MiB accumulated.
**P2:** encoder migration (BGE / Jina) onto `llama_*` API —
naturally extends the `encoderOnly` path used here for
wordpiece-bert.

**Status:** **P0 CLOSED 2026-05-05 — PASS.** Spec at
[`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md);
P0 plan at
[`docs/superpowers/plans/2026-05-05-tier3-p0-spike.md`](docs/superpowers/plans/2026-05-05-tier3-p0-spike.md);
closure report at
[`eval/reports/p0-spike-2026-05-05/SUMMARY.md`](eval/reports/p0-spike-2026-05-05/SUMMARY.md).
TinyLlama Q4_0 → `webllm_decode` → top-1 = " Paris" id 3681 green.
Patch budget B intact (9 core llama.cpp patches, no new ones added).
Three build-config deltas sufficed: `GGML_CPU=ON` (libllama load
expects a CPU backend device), `WASM_BIGINT=1` on wasm32 (libllama's
WebGPU async-wait i64 timeoutNS arg), `LLAMA_WASM_MEM64=OFF` (avoid
wasm32/64 conflict). Decision: **PROCEED to P1 (tokenizer)**.

**Decision summary.** Migrate the entire forward-pass surface
(causal LM, encoder, embedder, tokenizer) off the hand-rolled TS
graph builder onto upstream `llama_*` API surface. Code at risk of
deletion: `src/inference/model-inference.ts` (~2890 LOC),
`src/inference/encoder-inference.ts` (~565 LOC),
`src/inference/causal-embedder-inference.ts`, the bulk of
`src/inference/tokenizer.ts` (~1000 LOC of WordPiece + BPE +
pre-tokenizer regex). Retained: `engine.ts` orchestration,
`sampler.ts`, `generation.ts`, `chat-template.ts`, `stream-router.ts`,
`speculative.ts` (rewired in P5).

**Motivation.** Future-proof for any architecture llama.cpp
supports (LFM2, Mamba, Qwen3-MoE, GLM, DeepSeek, hybrids) without
TS-side reimplementation, and eliminate the maintenance burden of
mirroring upstream graph builders. Triggered by an LFM2.5-350M
support request 2026-05-05 — investigation revealed LFM2's hybrid
short-conv + GQA layers would require ~500-1000 LOC of new TS
graph code that already exists in upstream `src/models/lfm2.cpp`.
Tier 3 makes LFM2 (and the next N hybrids) free.

**Locked decisions (2026-05-05 brainstorm):**

- **Tier 3** (full migration; not T1 trial, not T2 partial)
- **Stage-aware parity bars** per phase (D): byte-exact tokenizer;
  C-strict (accuracy ±sampling-noise, perf ±10% canonical 6,
  embedder distinguishability gate strict) for causal/encoder/
  embedder; perf-strict for spec-decode/KV; "it works" for spike
  + LFM2.
- **Pragmatic patch budget** (B): up to ~3 new core llama.cpp
  patches with upstream-PR discipline; escalate to Liberal (C) if
  P0 reveals ASYNCIFY blockers deeper than expected. Budgeted
  patches: (1) `llama_decode` ASYNCIFY-aware suspend point,
  (2) `llama_kv_cache_init` heap-pressure-aware sizing for the
  WASM-32 ceiling, (3) reserve 1 for unknowns.
- **Pragmatic tap preservation** (B): Bucket-D self-embed survives
  natively via `llama_get_embeddings_ith` (load-bearing per
  CLAUDE.md); speculative-decode rewired in P5 via two
  `llama_context`s sharing one `llama_model`; `debugLayerOutput`
  deleted (parity tests rewritten against logits / public
  embeddings); per-op timing simplified to `llama_perf_*`; custom
  FA gating + prefill tiling deleted (subsumed by
  `llama_context_params`).
- **Bridge structure** (A3 hybrid): one async `webllm_decode`
  wrapper for the hot decode path (single ASYNCIFY round trip per
  token) + direct cwrap bindings for stateless ops (kv reset,
  logits/embedding readback, batch construction, model load).

**Phase order (A — tokenizer-first, LFM2 last):**

1. **P0 Spike** — TinyLlama or smollm2-360m generates "Paris"
   through `llama_decode`. Validates ABI + ASYNCIFY patch + heap-
   aware load. Parity bar: it works.
2. **P1 Tokenizer** — replace BPE / WordPiece in `tokenizer.ts`
   with `llama_tokenize`. Byte-exact gate on the existing fixture
   set (wordpiece-golden, Mistral trailing-space, chat-template
   stop-token suite).
3. **P2 Causal-LM** — `ModelInference` → `webllm_decode` wrapper.
   Delete `model-inference.ts`. C-strict bar.
4. **P3 Encoder** — `EncoderInference` → `llama_encode` + pooling.
   Delete `encoder-inference.ts`. C-strict + cosine parity vs HF
   reference vectors.
5. **P4 Embedder** — `CausalLMEmbedder` →
   `llama_get_embeddings_ith`. C-strict + Bucket-D distinguishability
   gate strict (16+16 mean-margin per 2026-04-30 doctrine).
6. **P5 Spec-decode + KV** — two-context spec-decode rewire; KV
   cache lifecycle moves under `llama_kv_cache`. Perf-strict
   (this path is perf-load-bearing).
7. **P6 LFM2** — `eval/models.ts` registration + ChatML stop tokens
   + smoke test. Trivial; victory lap.

**Side-effects: deferred items obviated.** Mark closed-by-
superseded once spec lands: §1 "Decode graph reuse (deferred)" —
TS-side optimization moot under Tier 3 (upstream handles graph
caching internally). §4 "Flash attention in browser" — gating
becomes `llama_context_params.flash_attn`; per-arch heuristics
delete with `model-inference.ts`.

**Risk register (high-level — full register lives in spec):**

- **R1 ASYNCIFY blocker.** P0 may reveal `llama_decode`'s compute
  path needs deeper restructuring than a single suspend-point
  patch. Escape valve: escalate to Liberal patch budget (C).
- **R2 P2 perf regression > 10%.** llama.cpp's prefill scheduler
  may behave differently than the project's hand-tuned tiling.
  Stage D allows perf-recovery sub-phase before merge.
- **R3 Tokenizer fixture mismatch in P1.** Edge cases (chatml
  `<|im_end|>` lookup, Mistral trailing-space, WordPiece phantom-
  space) may diverge from `llama_tokenize`. Byte-exact gate
  exposes this on a fixture set, not mid-decode.

**MEMORY64 full bridge migration — CLOSED 2026-04-29.** All 8
phases shipped + all three closure follow-ups landed (Q5_K_M
decode validated, Q5_K canonical-6 row, Emscripten port bumped
past `8d78be5` and shim patch deleted). Production wasm64 ships
end-to-end on Mistral-Nemo Q4_K_S (72% eval / 19.3 tok/s
gate-passing) and Mistral-7B Q5_K_M (26.7 tok/s, kernel-coverage
probe). Closure report at
[`eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`](eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md);
Phase 5 parity sweep with Q5_K addendum at
[`eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md`](eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md);
full migration history archived to `TODO_ARCHIVE.md`.

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

### Chat-template family dispatch hardening (CLOSED 2026-05-04)

Closed 2026-05-04 — three-layer fix for the chat-template family
dispatch (sampling profile + chat-stop registration + template
formatter must all agree, or the model wanders past end-of-turn).
Triggered by interactive smoke-test session debugging Mistral 7B
Instruct v0.3 multi-turn self-dialogue, generalized to a fleet-wide
audit + engine widening.

**Layer 1 smoke (`tests/chat-template-special-tokens.test.ts`):**
expanded from 4 → 19 chat-capable GGUFs with a per-model
`chatStopTokens` audit asserting `tokenizer.getId(stopLiteral)` (the
exact API `engine.ts:addChatStopToken` calls) resolves for every
registered family. 92 tests / 0 fail. Header-only progressive read
(2 → 16 → 64 MiB) avoids `readFileSync` ENOMEM on multi-GB fixtures.
Commit `da720a6`.

**Engine chat-stop registration widened (commit `c3d8261`):**
- Non-Qwen ChatML (Hermes-3 + SmolLM2) — register `<|im_end|>`
  explicitly. `<|endoftext|>` deliberately *not* registered for
  non-Qwen chatml because in SmolLM2's vocab it aliases id 0 (unk/pad
  slot).
- Gemma — register `<end_of_turn>`. Was the only family genuinely
  broken pre-fix: eosId=1 is `<eos>`, not the chat turn-end token at
  id 107.
- Detection now template-string-driven via `detectChatTemplate`, not
  architecture flag, so SmolLM2/Hermes-3 (registered as
  `architecture: "llama"` but chatml-templated) get correct
  registration.

**Mistral-Instruct family handling (commits `dafe4b4` + `27aacef`
+ `1f064e9`):**
- `formatLlama2` differentiates Mistral-Instruct from true Llama-2
  via `<<SYS>>` envelope presence: Llama-2 keeps `<<SYS>>...<</SYS>>`
  + trailing space after `[/INST]`; Mistral merges system into first
  user message + omits trailing space (matches official Jinja).
- `</s>` registered as runtime chat-stop for `llama2`/`mistral-v7`
  template families (Mistral GGUFs sometimes ship wrong eos id).
- `MISTRAL_DEFAULTS` sampling profile (T=0.7 / top-p=0.95, official
  MistralAI rec) added with auto-dispatch when
  `detectChatTemplate(...) === "llama2" && !template.includes("<<SYS>>")`.
  At T=1.0 unconstrained, Mistral skips `</s>` for higher-probability
  prose continuations and fabricates multi-turn dialogue.
- Chat-page UI (`smoke-test/chat-settings.js`) mirrors the new
  defaults via `family =~ /^mistral/i`.

**Memory note:** `feedback_chat_template_family_dispatch.md` —
"chat-template family dispatch needs three signals, not one"
(stop tokens + sampling profile + formatter must all agree;
covers the Mistral-v0.3 three-round debug pattern as the canonical
near-miss).

### Next session pickup (queued 2026-04-29; updated 2026-04-29)

**Status:** algorithmic-perf backlog cleared (§17-§29 + Phi-3
support shipped). TS API audit (a)-(f) closed 2026-04-29; full
narrative archived to `TODO_ARCHIVE.md`. Embedding bucket C
(Qwen3-Embedding-0.6B-hyb) closed 2026-04-29 (item 5). Embedding
bucket D (qwen3-8b-iq3m self-embed) closed 2026-04-30 (item 6);
Phi-3.5-mini bucket D extension §28 NEGATIVE 2026-04-30 (item 7).
**Next session focus queued 2026-05-01: NPC scenario sizing probes
(items 9a-9d) under the probe-first preferred-path doctrine, then
dual-mode (main-thread + worker) deployment support gated on probe
9d (item 10).** First-class frame-probe mode shipped 2026-05-01
(item 8); deterministic ~50ms decode-shape hitch confirmed across
5 sequential calls — that finding is what triggers items 9c
(hitch-warmup) and 9d/10 (worker migration). Probe-first means:
no architecture decision (prefix caching, worker plumbing, batched
prompt scheduling) lands before the matching probe produces data.
Daily cadence check (item 1) still required at session start.

1. **Daily upstream cadence check (REQUIRED, ~30s).** Procedure:
   `cd ~/Repos/llama.cpp && git fetch origin && git log
   webllm-browser-patches..origin/master --oneline --
   ggml/src/ggml-webgpu/ ggml/include/`. **If non-empty:** apply
   §32 procedure (rebase, sweep, classify per §27/§28/§32
   templates). **If empty:** log and skip. Last fired:
   2026-05-04 (§27 hybrid — drop local LayerNorm patches subsumed
   by upstream `d4b0c22f9`; encoder parity PASS, perf neutral vs
   2026-05-01 cross-day baseline; tip `fc1f81242`).

2. **Phi-3 closure follow-ups.**
   - ~~(a) Runtime contiguous-tensor assertion in fused helpers.~~
     **CLOSED 2026-04-29** — commit `dc441ce`. Added
     `assertContiguousF32(wasm, tensor, label)` exported from
     `model-inference.ts`; wired into `buildQKV` / `buildFFNGateUp`
     fused branches gated on `ModelInference.assertFusedContiguity`
     (default true). 8-case helper unit test in
     `tests/fused-contiguity-assert.test.ts`. Cost: ~7 wasm-bridge
     round trips per output × 5 outputs/layer × layerCount (only
     when `qkvFused`/`gateUpFused` non-null — Phi-3 today);
     measured <1% of graph-build wall time on Phi-3.5-mini.

   - ~~(b) Chat-template special-token literal audit.~~ **CLOSED
     2026-04-29** — commit `2d65082`.
     `tests/chat-template-special-tokens.test.ts` audits 13 special-
     token literals across formatPhi3 / formatLlama3 / formatChatml /
     formatLlama2-via-Mistral-v3 (4 + 4 + 2 + 3 = 13 single-token
     assertions + 4 round-trip assertions = 17 tests). Each test
     skips gracefully if its GGUF is missing. Probe verified: the
     documented `<|assistant|?` typo encodes as 6 tokens vs 1 for
     the correct `<|assistant|>` — audit fires on the protected bug
     class. **Deferred:** mistral-v7 (`[SYSTEM_PROMPT]`/`[/]` only
     special in v7-template GGUFs; Mistral-Nemo 6.6 GiB too heavy
     for unit test) and gemma (no Gemma GGUF in fleet). Re-introduce
     coverage if a smaller v7-template or any Gemma GGUF lands.

   - (c) **Path A vs Path B A/B measurement on Phi-3.** Loader-only
     views (Path A) — split the fused tensors at upload time
     into materialized Q/K/V/gate/up tensors — vs the shipped
     Path B fused-forward. Predicted Path B win: ~96 dispatches
     saved per token; observed cost: -6% throughput from opCont
     copies. Without the A/B we don't know if the prediction
     holds in practice. **Informational only**; closure report
     already recommends evaluating Path A first for the next
     fused-projection architecture (Phi-4, Granite). **Skip until
     a next fused architecture is queued.**

3. **Pre-rebase baseline freshness.** Matrix at
   `eval/reports/pre-rebase-baselines-2026-04-28/` is fresh until
   ~2026-05-28 (~1-month window). Re-capture only on the
   "stale-matrix + still-no-rebase-ETA" branch; otherwise let the
   next rebase trigger consume it. See watch list below for the
   procedure.

4. **TS API audit follow-ups (CLOSED 2026-04-29).** Phase 1 audit
   + Phase 2 (a-e) + Phase 3 (a-f) all shipped 2026-04-29; full
   block (commit map, decision log, test-surface delta, follow-ups,
   process notes) archived to `TODO_ARCHIVE.md` under the
   "TS API audit follow-ups (closed 2026-04-29)" heading.
   Net: 14 exports trimmed from public surface, `WebLLMError`
   taxonomy exposed, `GenerationConfig` split, `WebLLMConfig.device`
   removed, `CompletionConfig.sampling` flag added,
   `Character.setTools`, engine accessors migrated to properties,
   `ChatToolSchema` literal union. Spec
   [`docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md`](docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md);
   plan
   [`docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md`](docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md).
   Three orthogonal follow-ups filed in the watch list (sampling-
   dispatch unit test; tool-schema mirror-drift sentinel; tsconfig
   widening to enforce `@ts-expect-error` gates).

5. **Embedding bucket C — causal-LM-derived embedders. CLOSED
   2026-04-29.** Qwen3-Embedding-0.6B-hyb shipped (commits
   `deab38a` BPE tokenizer fix → `e2fa58b` bucket C bundle →
   `2724b02` embed-perf bench coverage). Hybrid GGUF (`token_embd`
   Q4_K = 83 MiB + f16 elsewhere) clears the WebGPU 128 MiB
   per-binding cap that blocked the f16 path. Parity 10/10 at
   `cos >= 0.995` (hybrid-tier gate) — cosines 0.996-0.9996,
   magnitudes 1.000 ± 1e-6. Bench: 77 ms / 114 ms p50 single
   short/long, 10.4 texts/sec batch (n=30, 64-text mixed batch).
   Two cross-cutting bugs surfaced and were fixed: BPE merge
   stale-rank validation (chars-only fallback for words like
   "Instruct"/"Query" outside chat framing) and parity-harness
   ↔ in-page embedPerf race (concurrent forward graphs corrupting
   the WASM ctx-stack). Closure report
   [`eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md`](eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md);
   bench [`eval/reports/embed-perf-qwen3-2026-04-29/`](eval/reports/embed-perf-qwen3-2026-04-29/).
   Full bucket C block (Phase 0 → Phase 6 detail, gate-selection
   rationale, hybrid-quant build recipe, follow-on hooks for
   bucket D) archived to `TODO_ARCHIVE.md` under "Embedding bucket
   C (closed 2026-04-29)". Spec
   [`docs/superpowers/specs/2026-04-29-embedding-bucket-c-implementation-design.md`](docs/superpowers/specs/2026-04-29-embedding-bucket-c-implementation-design.md);
   plan
   [`docs/superpowers/plans/2026-04-29-embedding-bucket-c-implementation.md`](docs/superpowers/plans/2026-04-29-embedding-bucket-c-implementation.md).

6. **Embedding bucket D — chat-model self-embedding. CLOSED 2026-04-30.**
   `ModelInference.embed(tokenIds)` shipped; `engine.embed` dispatches
   through `inferenceEngines` for chat models with `embeddingCapable:
   true`. **`qwen3-8b-iq3m`** is the single registered bucket D model
   at v1; other archs follow as separate cycles.

   Parity 10/10 PASS at `cos >= 0.90` (IQ3_M-calibrated gate; new
   third tier in the gate-by-quant-tier scheme alongside `hyb` 0.995
   and default 0.999). 4-pair cosine distinguishability sanity passes
   with clean margin (min paraphrase 0.918 > max unrelated 0.777).
   Closure report
   [`eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`](eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md);
   spec [`docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md`](docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md);
   plan [`docs/superpowers/plans/2026-04-29-embedding-bucket-d.md`](docs/superpowers/plans/2026-04-29-embedding-bucket-d.md).

   Full bucket D block (Q1-Q5 design rationale, per-task commit map,
   ref-capture recipe, IQ3_M gate calibration history, follow-up arch
   cycles) archived to `TODO_ARCHIVE.md` under "Embedding bucket D
   (closed 2026-04-30)".

7. **Bucket D Phi-3.5-mini extension — §28 NEGATIVE 2026-04-30.**
   Parity 10/10 PASS at `cos >= 0.91`, but distinguishability mean-
   margin gate **FAILS** both pooling modes (last-token −0.006,
   mean-pool −0.027 — paraphrase cosines lower than unrelated).
   Demoted (`embeddingCapable: false`); no follow-on cycle queued.
   Cycle keeper infra: `embeddingPooling` field, 16+16 cross-domain
   pair harness, mean-margin gate `mean(P) − mean(U) ≥ 0.05`. Closure
   report
   [`eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md`](eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md);
   full block (5 codified lessons, retire-path) archived to
   [`TODO_ARCHIVE.md` § "Bucket D Phi-3.5-mini extension — §28
   NEGATIVE"](TODO_ARCHIVE.md).

8. **Frame-probe coexistence baseline — CLOSED 2026-05-01.** `?frameProbe=1`
   mode shipped on the smoke page; multi-call probe on `qwen3-8b-iq3m`
   established 8.3 ms median render-loop, 25 tok/s decode, and the
   **deterministic per-call ~49 ms decode hitch** that triggered probes
   9c (warmup) and 9d (worker). Full block (4-bullet results, scene-
   stress finding, probe-first preferred-path doctrine) archived to
   [`TODO_ARCHIVE.md` § "Frame-probe coexistence + NPC scenario sizing
   probes"](TODO_ARCHIVE.md).

9. **NPC scenario sizing probes — ALL FOUR CLOSED 2026-05-01.**
   Downstream decisions all consumed:
   - **9a (PASS):** prefill 89.7% prefix → KV-cache-per-conversation
     load-bearing → drove item 11 prefix cache (shipped 2026-05-02).
     [`eval/reports/probe-9a-2026-05-01/SUMMARY.md`](eval/reports/probe-9a-2026-05-01/SUMMARY.md)
   - **9b (PARTIAL):** batched/sequential 0.72 wall ratio at N=4 →
     sequential is canonical agent-tick pattern.
     [`eval/reports/probe-9b-2026-05-01/SUMMARY.md`](eval/reports/probe-9b-2026-05-01/SUMMARY.md)
   - **9c (FAIL):** warmup does NOT reduce call-0 decode_max → do
     not bake warmup into engine init.
     [`eval/reports/probe-9c-2026-05-01/SUMMARY.md`](eval/reports/probe-9c-2026-05-01/SUMMARY.md)
   - **9d (PASS, 5.5× hitch reduction):** worker absorbs hitch
     (49.8 → 9.1 ms decode_max) → drove item 10 dual-mode (shipped
     2026-05-02).
     [`eval/reports/probe-9d-2026-05-01/SUMMARY.md`](eval/reports/probe-9d-2026-05-01/SUMMARY.md)

   Full block (per-probe rationale, harness extensions, fixture
   inventory, prefix-cache projection math) archived to
   [`TODO_ARCHIVE.md` § "Frame-probe coexistence + NPC scenario sizing
   probes"](TODO_ARCHIVE.md).

10. **Dual-mode deployment (main-thread + worker) — CLOSED 2026-05-02.**
    `WebLLM.init({ worker: true })` ships; same TS surface both modes;
    worker frame-probe **8.3 ms median, 0 drops** (gate <15 ms);
    cross-mode A/B perf **+15.6% to +34.2% faster** in worker; greedy
    token-identical 5/5. Closure report
    [`eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`](eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md);
    full block (architectural levers A1/A2/Path-A/staging-ptr,
    commit ledger, 6 low-priority follow-ups) archived to
    [`TODO_ARCHIVE.md` § "Dual-mode deployment (main-thread + worker)
    — CLOSED 2026-05-02"](TODO_ARCHIVE.md).

11. **Prefix cache via per-conversation KV snapshots — CLOSED
    2026-05-02 (this phase archived).** Mechanism shipped + validated
    end-to-end: `createConversation` / `disposeConversation` /
    `chatCompletion(conv, ...)` / `forkConversation`, with LRU
    eviction on the pool. Headline wins: **interleaved 84% wall
    savings** (Pattern B tick-2 2702 ms vs A's 15853 ms on
    qwen3-8b-iq3m;
    [report](eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md))
    and **fork 72% per-NPC savings / 17.2 s net at N=4 NPCs**
    ([report](eval/reports/prefix-cache-fork-2026-05-02/SUMMARY.md)).
    Side-finding: engine session-tracker delta-encoding bug fixed in
    `c8d1530` — conv-handle mode is now required for correctness in
    interleaved workloads, not just performance. Full closure detail
    (mechanism, batch-transfer trajectory, strided-read negative
    result, spec follow-ups #1+#2 closures, probe inventory) lives in
    [`TODO_ARCHIVE.md` § "Prefix cache via per-conversation KV
    snapshots"](TODO_ARCHIVE.md).

    **Remaining queued follow-ups (kept visible per watch-list
    cadence; defer until a consumer asks):**
    - **#3 Storage B (GPU-resident KV)** — queued. Requires `ggml-
      webgpu` patches. Defer until per-call overhead is measured
      against real harness usage and the API has stabilized.
    - **#4 Concurrent in-flight per conversation** — queued.
      Requires KV cloning at concurrency request time. Defer.
    - **#5 Persistence across reloads — CLOSED 2026-05-03 (consumer
      ask honored).** Two-tier design: engine primitives
      `exportConversation(conv)` / `importConversation(modelId, blob,
      options?)` ship in core; `IndexedDBConversationStore` ships
      behind the `@paulrobello/webllm/persistence` subpath (apps
      wanting OPFS / server sync / encrypted-at-rest implement their
      own store against the same Uint8Array contract). Five new
      error classes (`IncompatibleConversationError` /
      `CorruptBlobError` / `PersistenceUnavailableError` /
      `PersistenceQuotaError` / `PersistenceIOError`); model-
      fingerprint + tokenizer-hash gate refuses cross-quant or
      cross-tokenizer loads; integer `schemaVersion` (no migrations
      yet); per-method transfer-allowlist on the worker bridge
      (`exportConversation` returns are transferred, not copied).
      Spec
      [`docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`](docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md);
      plan
      [`docs/superpowers/plans/2026-05-03-prefix-cache-persistence.md`](docs/superpowers/plans/2026-05-03-prefix-cache-persistence.md).
    - **#6 Worker migration (item 10) — CLOSED 2026-05-03 (probe
      outcome (a): wiring already correct).** `WebLLMProxy` already
      mirrors all four conv methods (surface-sentinel-enforced);
      worker host reflect-dispatches them; `ConversationHandle` is
      plain data and structured-clones cleanly; `ConversationPool`
      lives engine-side so dual-mode `init({worker:true})` puts it
      worker-resident automatically. Gap was test coverage, not
      behavior. Lifecycle regression test added in
      `tests/webllm-proxy-integration.test.ts` (`describe("WebLLMProxy
      — conversation lifecycle")`) exercising create → chatCompletion(
      conv) → fork → chatCompletion(forked) → dispose → typed-error
      on disposed handle, all through the in-process worker channel.
      Real-Worker boot remains covered by the dual-mode browser smoke
      (`real-model.html?worker=1`).

---

**Embedding-model expansion (2026-04-28).** Buckets A and B both
landed: A via commit `41b27bd` (bge-small + bge-large registered);
B via 10 commits closing with Phase 5 (jina-embeddings-v2-base-en
ALiBi/GeGLU + nomic-embed-text-v1.5 NEOX-RoPE/fused-QKV/SwiGLU,
both at 5/5 reference-vector parity). Dashboard Embeddings section
now shows 6 rows. C (causal-LM-derived) remains deferred behind a
deployment ask. See "Embedding-model expansion candidates" section
below.

**Fresh observation pinned 2026-04-28 from #5 data:** the encoder
overhead (`backendEncodeOverheadMs` per step) is a **fixed
per-dispatch cost of ~5.2-5.7 µs**, remarkably flat across the
entire 450 → 805 dispatch/token range:

| Model            | Dispatches | Encoder (median, ms) | µs/dispatch |
|---|---:|---:|---:|
| tinyllama-q4_0   | 450 | 2.40 | 5.3 |
| qwen3-0.6b-q8    | 629 | 3.30 | 5.2 |
| qwen3-1.7b-q8    | 629 | 3.60 | 5.7 |
| mistral-7b-q4ks  | 650 | 3.60 | 5.5 |
| llama-3.1-8b-iq3m| 652 | 3.40 | 5.2 |
| qwen3-8b-iq3m    | 805 | 4.40 | 5.5 |

Implication: **encoder share scales inversely with model size
because matmul shrinks at small models, not because encoder
grows.** Encoder is 24-30% at tiny models (TinyLlama, Qwen3-0.6B)
where matmul is 33-38%; drops to 9-11% at 7-8B where matmul is
49-58%. **Reducing the per-dispatch encode cost would yield
~26% relative speedup at TinyLlama scale (11.40 → 9.00 ms/step
→ 87.9 → 111.1 tok/s).** Lever wasn't load-bearing under the
prior 30B ceiling and is **promoted to "watch list"** under the
new 8B ceiling (2026-04-29) — agent latency goals make the
~26% headroom at sub-1B and ~10% at 1.7B-8B more interesting
than they were at 30B targets, where absolute headroom is <1
tok/s. Re-evaluate once bucket D (chat-model self-embedding)
ships and the agent-loop end-to-end latency profile is in hand.

---

### Bucket B follow-ups (post-closure, 2026-04-28) — CLOSED; archived

Both queued follow-up items (#11 spec accuracy patch; #12 vault-save
bucket B doctrines — 4 notes landed under `~/ClaudeVault/`) closed
2026-04-28. Full block (per-section spec-patch breakdown, four
vault-note descriptions with cross-link map, MANIFEST verification)
archived to `TODO_ARCHIVE.md` under "Bucket B follow-ups (post-closure,
2026-04-28)".

---

### Embedding-model expansion candidates (queued 2026-04-28)

User-driven scope: extend embedding fleet beyond the two registered
Arctic-Embed entries. Three candidate buckets, in increasing scope.
**Buckets A and B closed 2026-04-28;** C remains deferred behind a
deployment ask.

**A. Register more BERT-arch embedders** ~~(in progress 2026-04-28)~~
**DONE 2026-04-28** (commit `41b27bd`). Confirmed cleanly: the
encoder forward path, WordPiece tokenizer, F16 / F32 dtypes, and
CLS pooling (read from GGUF metadata) all already work for BGE
out of the box — zero code changes outside `eval/models.ts`,
`eval/smoke-profiles.ts`, `eval/embed-perf.ts`.
- `bge-small-en-v1.5-q0f16` (~33M, 384-dim): 17.0 ms p50 single-
  text short / 91% on 8-task cosine eval. Apples-to-apples with
  arctic-embed-s.
- `bge-large-en-v1.5-q0f16` (~335M, 1024-dim): 59.3 ms p50
  single-text short / 89% on 8-task cosine eval. **First 335M
  encoder in fleet** — new scaling point for the dashboard's
  Embeddings section. 3.5× latency for 10× params consistent
  with bandwidth-bound encoder behavior.

GGUF source: `ChristianAzinn/bge-{small,large}-en-v1.5-gguf` mirror
(same publisher as the arctic-embed entries already in tree).
File-name pattern: `_fp16` (matches `bge-{small,large}-en-v1.5_fp16.gguf`
unique within each repo).

**Net learning:** the BERT-arch lever is effectively free for any
future ask — no loader changes were required. Stretch picks
(`bge-base-en-v1.5`, `mxbai-embed-large-v1`, `snowflake-arctic-
embed-l`) are now register-and-run candidates with high confidence.
Add them only if a specific deployment ask names them, or as part
of B/C closure. Otherwise dispatch B or C next based on what
embedder family the next deployment ask names.

**B. Extend `EncoderInference` to non-BERT arch** **DONE 2026-04-28**.
Both `jina-embeddings-v2-base-en` (ALiBi, GeGLU, no FFN biases) and
`nomic-embed-text-v1.5` (NEOX RoPE, fused QKV, SwiGLU, no biases)
landed with 5/5 reference-vector parity each at cosine ≥ 0.999999.
Plan v2 in `docs/superpowers/plans/2026-04-28-encoder-non-bert-arch.md`
guided 5 phases (probe / types / forward / registration / closure);
spec v2 at
`docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`.

Commit ledger (10 commits):

| Phase | SHA | Subject |
|---|---|---|
| Plan v1 | `4c4cd4c` | bucket B plan v1 (preserved as artifact) |
| Spec v2 | `bf51912` | post-Phase-0 spec rewrite |
| Plan v2 | `61b8309` | post-Phase-0 plan rewrite |
| 0 (probe) | `43df996` | GGUF discovery probe |
| 1 (types) | `7a41f79` | ModelArchitecture widening |
| 2a (forward) | `7a18074` | bert + jina forward + engine routing |
| 2b (forward) | `3982af9` | nomic fused-QKV + RoPE |
| 2b (review) | `31d6ac2` | view3d offset coverage + F32_BYTES |
| 3a (refs) | `5e85db8` | sentence-transformers reference vectors |
| 3 (jina) | `d16b5b1` | jina registration + 5/5 parity (1.000000) |
| 4 (nomic) | `709511e` | nomic registration + 5/5 parity |

Latent bugs surfaced and fixed during integration:
1. Phase 1/2a: encoder routing in `smoke-test/real-model-page.js`
   only matched `architecture === "bert"`, so jina + nomic loads
   silently fell through to the causal path. Fixed in Phase 3.
2. Phase 2a: `ggml_soft_max_ext` requires a non-NULL mask when
   `max_bias > 0` (`ggml.c:4012`). Phase 3 added the mask leaf
   populated with `-|i - j|` per `llama-graph.cpp:411`.
3. Spec v2 wrong on jina activation: spec said SwiGLU, llama.cpp
   uses GeGLU (`bert.cpp:122-130`). Fixed in Phase 3.
4. Spec v2 + plan wrong on nomic RoPE mode: said NORMAL, llama.cpp
   uses NEOX (`llama-model.cpp:9266`). Fixed in Phase 4.
5. Phase 1: nomic GGUF omits `tokenizer.ggml.cls_token_id` /
   `mask_token_id`. Phase 4 added bos/eos fallback for WordPiece.

Dashboard now shows **6 embedding rows** (arctic-embed-s,
arctic-embed-m, bge-small-en-v1.5, bge-large-en-v1.5,
jina-embeddings-v2-base-en, nomic-embed-text-v1.5) — the full
BERT-family encoder lever portfolio: split QKV, fused QKV, NEOX
RoPE, ALiBi, GeLU, GeGLU, SwiGLU, full biases, no biases, mixed
biases. Bench-full populated cosine-task and per-text latency for
both new models; parity artifacts at
`eval/reports/encoder-parity-2026-04-28/` (jina-ref.json,
nomic-ref.json, dashboard PNG).

Net learning: the non-BERT encoder lever is now exhausted for the
two named families. Remaining encoder asks are register-and-run on
top of this foundation if they share an arch tag already on file
(`bert` / `jina-bert-v2` / `nomic-bert-moe`); novel arch tags would
re-open Phase 0/1.

**C. Causal-LM-derived embedders (`Qwen3-Embedding-0.6B`)**
**→ promoted 2026-04-29 to active next-session focus; see
"Next session pickup" item 5 above for the probe-first phase
plan and scope.** Original scope summary preserved here for
the embedding-bucket campaign narrative: reuses the existing
causal forward but requires:
- Embed-mode toggle on `ModelInference` (skip sampling, return
  hidden state)
- Last-token (or attention-pooled) pooling head
- L2 normalize + project to embedding-dim if model has a head

Scope: medium (additive). Trigger: an MTEB top-of-leaderboard ask
for >1B embedders, where Qwen3-Embedding-4B and Qwen3-Embedding-8B
become candidates. Highest upside (Qwen3-Embedding tops MTEB at
0.6B-8B as of 2026); also opens a path to other causal-LM
embedders (`gte-Qwen2-*`, `e5-mistral-*`).

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
  Last clean run: **2026-05-05** (clean — `master` advanced
  ~9 commits past `a817a22bc` to `d5003b6e4`, but no
  `ggml/src/ggml-webgpu/` or `ggml/include/` commits since the
  2026-05-04 §27-hybrid rebase).
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

Three open candidates, all conditional:

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
