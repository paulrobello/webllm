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
> Current smoke-bench baselines (browser, realistic sampling, 3-trial
> median on 2026-04-25 after qwen3-1.7B characterization landed):
> - tinyllama-1.1b-chat-q4_0: **~105 tok/s decode**
> - qwen3-0.6b-q4f16 (actual: Q8_0), thinking-off: **~85 tok/s decode**
> - qwen3-0.6b-q4f16 (actual: Q8_0), thinking-on: **~93 tok/s decode**
>   (was ~17 — see `3e5be59`: CPU post-filter top-K replaced the
>   full-vocab readback + JS sampling pipeline that was costing
>   ~76 tok/s on Qwen3's 152K vocab)
> - qwen3-1.7b-q4f16 (actual: Q8_0), thinking-on: **~66 tok/s decode**
>   (clean 117-token run; thinking-off 17-token run measured ~59 but
>   is warmup-dominated)
> - smollm2-360m-q4f16 (Q4_0): **~106 tok/s decode** (within noise of
>   TinyLlama-1.1B at the same quant despite 3× fewer params; encode
>   overhead dominates at this scale, matmul takes a back seat — see
>   Active Step §10 wave-1 entry below)
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

### Completed on 2026-04-27

21. **§D encoder/embedding perf cycle measured + closed.** First dedicated
    perf cycle on the encoder fleet (arctic-embed-s 33M / arctic-embed-m
    109M, both F16). Produced harness infra + diagnostic finding rather than
    a shipped lever; mirrors §17 / §19's measure-and-close pattern. Branch:
    `feat/encoder-perf`. Plan / spec at
    `docs/superpowers/plans/2026-04-27-encoder-perf-pass.md` /
    `docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md`
    (Phase 2.5 closure addendum carries the full lever portfolio).

    **Headline finding (Phase 2.5 diagnostic):** `graphCompute` is **95.6%**
    of `embed()` wall time on arctic-embed-s short. A 33M F16 model has
    ~66 MB of weights — at Apple Silicon's ~200 GB/s memory bandwidth the
    actual compute is <1 ms. The remaining ~31 ms is dispatch /
    kernel-launch overhead: encoder graph has ~390 ops × ~80 µs/dispatch
    ≈ 31 ms (matches measurement exactly). Per-call ctx + graph rebuild
    is <1 ms total; download is ~1 ms; pool is <0.1 ms. **Bottleneck is
    dispatch count, not memory bandwidth or arithmetic.**

    **L1 ctx/graph reuse measured + reverted.** Implemented at `5eb1f73`
    (private graphCache field, ensureGraphCache(N), dispose pops graph
    ctx then weight ctx). Single-text p50 wall ms vs Phase 1 baseline:
    arctic-embed-s short +0.6%, long +2.3%; arctic-embed-m short +2.7%,
    long −9.5%. Three slight regressions plus one bimodal-noise reading
    on m-long (~34 ms cluster + ~38 ms cluster, 50/50 split — not a real
    effect). G1 strict reading: no model dropped ≥10%. Reverted at
    `3a6a366` per gate rule. Cosine 0.76 preserved (G3 part 1) throughout.

    **Lever re-ranking against the Phase 2.5 data:**
    - L1 ctx/graph reuse: targets <1% bucket → measured + reverted.
    - L2 GPU-side pool / readback shrink: targets ~3% bucket → not worth
      shipping for ~1 ms.
    - L3 embedBatch sequential loop: zero amortization on dispatch count
      → no-op on the dominant bucket.
    - L4 concat-graph batched compute: only lever with structural headroom
      (potentially 4-8× via dispatch amortization at K≤8). Was explicitly
      listed as non-goal in the spec; correctness-risky (block-diagonal
      mask up to ~85 MB at K=64 batchMixed; or full 4D padded batch
      refactor of `buildGraph`); deferred to future cycle gated on a real
      use-case for batch encoder throughput.

    **Cycle closes per the spec's stop rule:** "a lever's measured impact
    is in the noise AND nothing else profiles as a hotspot → close early;
    document what was tried." L1's null result + Phase 2.5's
    dispatch-overhead characterization rules out L2 / L3-sequential
    without measurement; L4 is out of scope.

    **What ships on `main` from this cycle:**
    - `eval/embed-perf.ts` harness CLI + `EmbedPerfTrace` /
      `waitForEmbedPerfResult` in `eval/browser-smoke.ts`.
    - `eval/fixtures/embed-prompts.ts` pinned text fixtures (short / long
      / batchMixed).
    - `smoke-test/real-model-page.js` `?embedPerf=<single|batch>&embedReps=<N>&embedFixture=<id>`
      URL-param hooks (causal-LM and encoder branches; default off).
    - `Makefile` `embed-perf` + `embed-perf-baseline` targets.
    - `tests/encoder-cosine-parity.test.ts` G3 baseline guard
      (`eval/reports/embed-perf-baseline-cosine.json`, 0.76 ±0.005).
    - `eval/reports/embed-perf-2026-04-27-baseline/` (Phase 1) +
      `eval/reports/embed-perf-2026-04-27-L1/` (L1 negative result)
      raw measurement logs.

    **What's reverted:** `feat(encoder): L1 same-graph-cache across
    embed() calls` (`5eb1f73` reverted by `3a6a366`).

    **Future-cycle resurrection paths:**
    - **Concat-graph batched compute** (deferred L4). Open if a real
      use-case for batch encoder throughput emerges. Implementation
      options at that point: (a) flat concat + block-diagonal mask at
      K≤8 (4-8× ceiling); (b) padded 4D batch dim (cleaner; full
      `buildGraph` rewrite). Harness from this cycle is ready to measure
      against G2.
    - **Larger encoder registration** (deferred wave-2). If `bge-m3` or
      `gte-large-en-v1.5` lands, single-text p50 may flip from
      dispatch-bound to compute/bandwidth-bound — at which point L1
      (and possibly L2) regain relevance. Re-measure then.
    - **Backend-side dispatch coalescing** in upstream `ggml-webgpu`. If
      that ever lands, addresses the §D bucket for free; re-run this
      cycle's harness on a future llama.cpp rebase to spot it.

    **Net characterization of the encoder fleet at this scale:** the
    encoder embed loop is **dispatch-bound, not compute-bound**. Single-
    text levers are exhausted; the only structural lever is dispatch
    amortization across multiple texts in one graph. For deployments that
    don't need batch encoder throughput, no perf work is justified at
    this scale; current numbers (34-52 ms p50) are the practical floor
    until either a use-case emerges or upstream `ggml-webgpu` improves.

22. **§22 7B+ long-prefill graph-buffer tiling — SHIP GATED.** First
    direct attack on the §20 long-prefill abort; mirrors §17/§18/§19/§20/§21's
    measure-and-close pattern. Branch: `feat/prefill-tiling-22`. Raw
    matrix at `eval/reports/prefill-tiling-2026-04-27/SUMMARY.md`;
    Phase 0 diagnostic at `00-phase0-diagnostic.txt` in the same dir.

    **Headline finding (Phase 0 surprise):** the §20 abort is **not**
    the WebGPU max-buffer-binding cap as that closure hypothesized.
    It is the **host-side ggml graph allocator** at `ggml-alloc.c:82`
    (`ggml_tallocr_alloc: not enough space … node_510 needed 8011776,
    available 475648`). The cap is per-graph-buffer in the CPU-side
    metadata context, not a GPU device limit. Tiling the prefill into
    smaller chunks keeps each graph's working-set under the allocator's
    available budget, which is why it works.

    **Matrix (5 cells, prefill-512 for 7B+, prefill-256 for TinyLlama):**

    | Cell | Model | Prefill | Tile | Prefill (ms) | Decode (tok/s) | Status |
    |------|---|---:|---:|---:|---:|---|
    | 1 | tinyllama-1.1b-chat-q4_0      | 256 |   0 |  289 | 101.9 | works (control) |
    | 2 | tinyllama-1.1b-chat-q4_0      | 256 | 128 |  524 | 106.9 | works — Δ TTFT +81.3%, Δ decode +4.9% |
    | 3 | mistral-7b-instruct-v0.3-q4ks | 512 |   0 |  —   |   —   | aborts (`node_510 needed 8011776, available 475648` — confirms §20) |
    | 4 | mistral-7b-instruct-v0.3-q4ks | 512 | 128 | 4368 |  33.6 | works (unblock; matches §18's 34.5 baseline within noise) |
    | 5 | qwen3-8b-iq3m                 | 512 | 128 | 4518 |  16.2 | works (unblock; matches §18's 15.1 baseline within noise) |

    TinyLlama tile=0 vs tile=128 produced bytewise-identical output
    (sampling-level equivalence holds). All 7B+ visible answers were
    coherent on-topic English.

    **Decision-rule evaluation:**
    - TinyLlama TTFT regression at tile=128: **+81.3%** vs ≤3% gate → **FAIL**.
    - TinyLlama decode at tile=128: +4.9% (improvement) → directionally pass.
    - Mistral-7B unblock at tile=128: **YES** — exact §20 abort signature avoided.
    - Qwen3-8B unblock at tile=128: **YES**.
    - **Decision: ship gated** (default-off, opt-in for 7B+).

    **What ships on `feat/prefill-tiling-22` (default-off plumbing):**
    `prefillTileSize?: number` ctor option on `ModelInference`
    (drafter accepts it too for symmetry; default `0` = legacy
    single-graph prefill — Task 1 `c38fb8f`); tile dispatcher in
    the prefill path that splits into `ceil(N/tile)` sequential
    graph dispatches when the prompt exceeds the tile and an
    equivalence test stub (Task 2 `f281ac3`); smoke-page
    `?prefillTile=N` URL param (Task 3 `2fcc334`); `eval/perf.ts`
    `--prefill-tile <n>` flag (Task 4 `18e1677`) plus a
    placeholder Makefile harness target; Phase 0 diagnostic
    capture under the original §20 hypothesis (Task 0 `8e21036`,
    kept as evidence); 5-cell matrix raw logs (Task 5 `5b5705a`)
    under `eval/reports/prefill-tiling-2026-04-27/`.

    **Why default-off rather than default-on:** the TinyLlama
    +81.3% TTFT regression at tile=128 is real, not noise. Each
    prefill tile is one extra ggml graph build + dispatch +
    post-pass; for a 1.1B model whose prefill-256 already fits in
    a single graph the overhead dominates. The gate keeps the
    small-model fast path untouched while letting 7B+ callers opt
    into the unblock.

    **Future-resurrection paths (not landed; reopen on demand):**
    (a) **Per-model auto-default** — add `recommendedPrefillTile?:
    number` to the model registry and select tile=128 automatically
    for 7B+ entries; cheap follow-on, deliberately deferred per
    this branch's ship-gated scope. (b) **tile=64 fallback** —
    untested but cheap if a future model hits the same
    `ggml_tallocr_alloc` abort at tile=128 (larger embedding-dim
    or layer-count pushing per-tile working-set over budget) before
    reopening the upstream allocator question. (c) **Revisit tile
    size if upstream ggml's graph allocator becomes more
    memory-efficient** — lifts the floor and may let the gate flip
    to default-on without TinyLlama regression; track on the next
    llama.cpp rebase.

    **Interaction with §C-v2-A (side branch).** §22 partially
    alleviates the verify-cost lever for short prefills, but the
    K+1 verify cost on 8B+ at the canonical target/drafter ratio
    was **not** measured here. §C-v2-A resurrection still needs
    long-prefill graph-buffer rework that §22 sidesteps (per-tile
    dispatch overhead) rather than fixes (per-graph allocator
    headroom). Treat §22 as a partial unblock for §C-v2-A, not a
    full resurrection trigger.

23. **§22 default-on auto-tile via per-model registry — LANDED.**
    Cheap follow-on to §22; promotes the 7B+ unblock from opt-in
    to right-by-default while preserving the sub-7B fast path.
    Single commit on `main` (`0c50e03`).

    **What ships:**
    - `eval/models.ts`: new `recommendedPrefillTile?: number`
      field on `BenchmarkModel`. Set to `128` on the five 7B+
      entries (mistral-7b-instruct-v0.3-q4ks / -q3km / -iq4xs,
      llama-3.1-8b-instruct-iq3m, qwen3-8b-iq3m). Sub-7B entries
      leave the field unset.
    - `eval/perf.ts`: when `--prefill-tile` is omitted, falls
      back to `model.recommendedPrefillTile`. Explicit
      `--prefill-tile <n>` (including `0`) still wins.
    - `smoke-test/real-model-page.js`: mirror map
      `RECOMMENDED_PREFILL_TILE` keyed by model id (data
      duplication, not logic — bundle / browser boundary
      precludes importing `eval/models.ts`). When `?prefillTile=`
      is absent, falls back to the map; explicit
      `?prefillTile=N` (including `0`) still wins.
    - `tests/eval-models.test.ts`: 2 registry-shape tests pin
      the contract (all 7B+ entries default to 128; no sub-7B
      entry sets the field). 424 → 426 pass.

    **Behaviour after this change:** `bun run eval/perf.ts
    --model qwen3-8b-iq3m` with no `--prefill-tile` flag
    auto-applies tile=128. Opening
    `?model=mistral-7b-instruct-v0.3-q4ks` in the smoke page
    with no `?prefillTile=` does the same. TinyLlama and other
    sub-7B paths are bit-identical to pre-§23 (no map entry,
    falls through to the existing `0` default). Force-disable
    on a 7B+ model still works via explicit `?prefillTile=0` /
    `--prefill-tile 0` for regression sweeps.

    **Why two maps instead of one source of truth:** the smoke
    page is plain JS loaded as a page module; the model registry
    lives in `eval/models.ts` (TS, harness-side). They sit on
    opposite sides of the bundle / browser boundary.
    Cross-importing would either bundle eval into the browser
    surface or break the harness's Node-only imports. The map in
    `real-model-page.js` is data, not logic, and the registry
    field's docstring + the smoke-page comment both call out the
    mirror requirement. A future cycle could fold the map into
    the smoke bundle if drift becomes a problem.

    **Future-resurrection paths (not landed; reopen on demand):**
    (a) **tile=64 fallback in the map** for any future model
    that hits `ggml_tallocr_alloc` at tile=128 (larger
    embedding-dim or layer-count pushing per-tile working-set
    over budget). (b) **Heuristic-based default in
    `ModelInference`** — derive the recommended tile from
    `hyperparams.layerCount × embeddingLength` rather than from
    a hand-curated list. Cleaner, but defers the "is the
    heuristic right" question until a model trips it; the
    explicit map is fine while the 7B+ fleet is small enough to
    enumerate. (c) **Bundle the map into the smoke bundle** if
    drift between the two registries causes a real bug; the
    cycle's commit message + the doc comments in both files
    are the current guard.

24. **§4 FA revisit at 7B+ long-prefill — CLOSED.** Direct
    follow-on to §22+§23 — re-ran the §20 matrix on the 3 cells §20
    could not capture (Mistral-7B-Q4_K_S, Llama-3.1-8B-IQ3_M,
    Qwen3-8B-IQ3_M × {short-short, long-short, short-long, long-long}
    × FA off/on, 24 cells, 3-trial median) with §23's
    `recommendedPrefillTile=128` auto-default unblocking long-prefill
    on 7B+. Landed on `main` directly; zero `src/` change.

    **TTFT (prefill ms, p50):**

    | Model     | short-short | long-short | short-long | long-long |
    |---|---|---|---|---|
    | mistral-7b-q4ks  | 878 → 847 (-3.5%)   | 4723 → 4865 (+3.0%) | 869 → 865 (-0.5%)  | 5582 → 4569 (-18.1%) |
    | llama-3.1-8b-iq3m | 791 → 770 (-2.7%)  | 4737 → 4716 (-0.4%) | 788 → 781 (-0.9%)  | 4914 → 4555 (-7.3%)  |
    | qwen3-8b-iq3m    | 476 → 493 (+3.6%)   | 4880 → 4877 (-0.1%) | 478 → 475 (-0.6%)  | 6348 → 4871 (-23.3%) |

    **Decode tok/s (p50):**

    | Model     | short-short | long-short | short-long | long-long |
    |---|---|---|---|---|
    | mistral-7b-q4ks  | 33.7 → 32.2 (-4.5%) | 31.1 → 30.9 (-0.6%) | 33.6 → 31.4 (-6.5%) | 30.1 → 30.3 (+0.7%) |
    | llama-3.1-8b-iq3m | 16.7 → 16.6 (-0.6%) | 16.7 → 16.7 (+0.0%) | 16.6 → 16.5 (-0.6%) | 16.5 → 16.5 (+0.0%) |
    | qwen3-8b-iq3m    | 15.5 → 15.2 (-1.9%) | 15.7 → 16.0 (+1.9%) | 15.5 → 14.9 (-3.9%) | 15.7 → 15.9 (+1.3%) |

    **Decision-rule evaluation:**
    - **A. Ship default-on:** *FAIL.* Mistral short-short decode
      regresses -4.5% and Qwen3-8B short-short TTFT regresses +3.6%
      (both >3% gate); zero models gain ≥2% on short-long decode.
    - **B. Ship gated (auto, FA on for `nTokens > 1`):** *FAIL.*
      Long-short TTFT deltas across the three 7B+ models (+3.0%,
      -0.4%, -0.1%) are all ≤5% — zero models meet the gated-ship
      threshold. The seq² avoidance win that helped TinyLlama at
      long-short (-10.0%) does not materialize at 7B+/IQ3_M shape.
    - **C. Close §4 again:** *FIRES (default).*

    **Net characterization:** FA stays behind the manual chain at
    7B+ across the canonical 4-workload matrix at prefill-512 —
    matmul is already 65-70% of decode time at this shape, and FA's
    per-step overhead exceeds the prefill saving on three of four
    workloads. The exception is **long-long TTFT** (Mistral -18.1%,
    Qwen3-8B -23.3%, Llama -7.3%) where the cumulative `pastLen`
    during decode amortizes the seq²-avoidance — but neither §20
    rule clause keys on long-long TTFT, and long-long decode tok/s
    wins are tiny (+0.7% / 0.0% / +1.3%), so this characterization
    flag does not flip the ship decision. It is a useful datapoint
    for future spec-decode / long-context cycles.

    **Files retained as future infra:** unchanged from §20 —
    `flashAttn?: boolean` ctor option, `?fa=on` URL param,
    `--fa <on|off>` perf.ts flag, F16 mask + dual V-cache layout,
    `eval/fixtures/long-prompts.ts` fixtures (prefill-256 / -512 /
    -1024), 5 contract tests at `tests/fa-mode-config.test.ts`.

    **Cycle infrastructure:** new files —
    `eval/reports/fa-revisit-7b-2026-04-27/{run-matrix.sh,
    SUMMARY.md, 01-coherence.txt, *.log}` (24 cell logs +
    matrix-driver script + coherence transcripts). Reuses §20's
    plumbing + §22+§23's auto-tile end-to-end. Zero `src/` change;
    zero new tests. `make checkall` remains 426 / 11 / 0.

    **Plan reference:** `docs/superpowers/plans/2026-04-27-fa-revisit-7b-long-prefill.md`.

    **Next lever with measured headroom:** §C-v2-A resurrection
    is the most promising candidate (§22's tile=128 partially
    alleviates the per-step K+1 verify cost — needs a fresh
    measurement cycle on the side branch under
    `prefillTileSize=128` to settle whether tiled-verify drops
    per-step cost enough to break the K=4 even-α ceiling at
    8B IQ3_M × 0.6B Q8). MEMORY64 for 70B-class targets and §D
    concat-graph batched encoder compute remain conditional on
    use-case; a heuristic-based prefill-tile default in
    `ModelInference` (§23 follow-on) is a nice-to-have when the
    7B+ fleet outgrows hand-curation. All explicitly conditional
    — pick on demand.

    **§26 measured + closed §C-v2-A resurrection.** See §26 below.

26. **§26 §C-v2-A re-measurement under §22 tile=128 — CLOSED.**
    Direct empirical test of §24's parting recommendation. Cherry-
    picked the 4 §22 implementation commits (`c38fb8f`, `f281ac3`,
    `2fcc334`, `18e1677` — skipped `8e21036` Phase-0 diagnostic and
    `5b5705a` Task-5 matrix; skipped §23 registry auto-default for
    variable isolation) onto `feat/spec-decode-v2-greedy`. Added one
    conflict-resolution recipe in `smoke-test/real-model-page.js`
    (drop §22 Task 3's references to `diagnoseAlloc` and `embedPerf`
    blocks not present on the side branch). Re-ran the §C-v2-A
    4-cell gate matrix under explicit `--prefill-tile 128` on both
    target and drafter `ModelInference` ctors. 3 outer trials × 3
    perf.ts internal runs = 9 measurements per cell, 36 total.

    **Matrix (median of three 3-run trial-medians):**

    | Cell | Workload         | Drafter | Decode tok/s p50 | Prefill ms p50 |
    |------|------------------|---------|-----------------:|---------------:|
    | 1    | prefill-256      | —       | 15.8             | 2684           |
    | 2    | creative-low-α   | —       | 15.8             | 1721           |
    | 3    | prefill-256      | K=4     | 6.7              | 3166           |
    | 4    | creative-low-α   | K=4     | 8.5              | 1530           |

    **Gates (decisive failures):**
    - **Gate 1 (speedup ≥1.5×):** 6.7 / 15.8 = **0.42×** — FAIL by 3.6×.
    - **Gate 2 (safety ≥0.95×):** 8.5 / 15.8 = **0.54×** — FAIL by 0.4×.

    **Cross-cycle vs §C-v2-A close (`646320c`, tile=0):** baselines
    drift -1.3% / -2.5% (within ±10% threshold); cell 3 drifts +17.5%
    (5.7 → 6.7, marginal improvement, gate-1 gap to 1.5× is still
    3.6×); cell 4 drifts -33% (12.7 → 8.5, **significant safety
    regression** — most likely later AdaptiveGate fire or less-
    effective post-disengage tail under tile=128 plumbing). The
    cell-4 drift is large enough to flag for any future v2-A
    resurrection cycle.

    **Verdict:** the K+1=5 verify graph is three orders of magnitude
    below the 128-token tile threshold and is never split. tile=128
    therefore cannot affect verify cost on this workload. The
    +17.5% cell-3 improvement is real but irrelevant to the gate;
    no incremental lever (better drafter, tighter K, faster cache)
    closes the 3.6× gap to 1.5×. **§C-v2-A is closed under all
    known levers.**

    **Resurrection paths still open (architectural change required):**
    (a) **Faster K+1 verify** via upstream ggml-webgpu dispatch
    coalescing or fused-graph optimization that drops per-step
    verify cost below ~30 ms — re-measure if upstream lands such an
    improvement. (b) **MEMORY64 → 70B-class target** to shift
    target/drafter param ratio from 13× to ~100× (Leviathan-style
    speculation regime). Multi-day engineering; conditional on a
    concrete 70B+ deployment ask.

    **Side branch retained as archived infra.** `feat/spec-decode-
    v2-greedy` tip moves from `646320c` to **`6b20aad`** with the
    cherry-picks + matrix + SUMMARY. Driver, AdaptiveGate, K+1
    verify, contract gate, ~30 unit/integration tests all preserved.
    **Do not merge to `main`.**

    **Files on `main`:**
    - `docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md` (`b23ccc9`).
    - `docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md` (`f0a682c`).
    - This TODO §26 entry.

    **Files on side branch (`feat/spec-decode-v2-greedy`):**
    - 4 cherry-picked §22 commits (`c38fb8f` → `832379a` after rebase shas).
    - `eval/reports/spec-decode-v2-tile128-2026-04-27/{run-matrix.sh, SUMMARY.md, cell-{1,2,3,4}.log}`.

    **Ship gate stamp:** zero `src/` change on `main`. `make checkall`
    on `main` unchanged from pre-§26 (427 pass / 11 skip / 0 fail).
    Side-branch checkall: 454 / 15 / 0 (post cherry-pick).

    **Plan reference:** `docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md`.
    **Spec reference:** `docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md`.
    **Raw matrix:** `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md` on side branch tip `6b20aad`.

### Resumption checklist (start a fresh session here)

**Wave 1 complete (7/10 done · 2 deferred · 1 optional
skipped). Wave 2 complete: 4/4 done** (mistral-7b-v0.3-q4ks
at 34.4 tok/s / 68% — §12; llama-3.1-8b-iq3m at 16.3 tok/s /
86% — §13; mistral-7b-v0.3-q3km at 19.7 tok/s / 69% — §15;
qwen3-8b-iq3m at 16.2 tok/s / 90% off / 90% on — §16). **Six
levers measured + closed:** §A subgroup-cooperative loading
(§17), §4 FA at N=1 decode (§18), §C v1 drafter spec-decode
(§19), §4 FA at prefill / long-decode (§20), §C v2-A greedy
spec-decode + GPU-resident verify (side branch, 2026-04-27),
**§D encoder/embedding perf pass (§21, 2026-04-27 — closed
on data: encoder embed is dispatch-bound at 95.6% graphCompute
share, single-text levers all <5% headroom)**, and **§22 7B+
long-prefill graph-buffer tiling (2026-04-27 — SHIP GATED,
default-off; tile=128 unblocks Mistral-7B-Q4_K_S at 33.6 tok/s
and Qwen3-8B-IQ3_M at 16.2 tok/s but regresses TinyLlama TTFT
+81.3%; opt-in via `?prefillTile=N` and `--prefill-tile <n>`;
Phase 0 disproved §20's GPU-cap hypothesis — actual failure is
the host-side ggml graph allocator at `ggml-alloc.c:82`)**.
**§23 (2026-04-27 — LANDED) flipped §22's gate to default-on
for 7B+ via per-model registry** (`recommendedPrefillTile`
field, mirrored in the smoke page). Sub-7B paths bit-identical
to pre-§23. **§24 (2026-04-27 — CLOSED) re-ran the §20 FA matrix
on the 3 cells §20 could not capture** (Mistral-7B-Q4_K_S,
Llama-3.1-8B-IQ3_M, Qwen3-8B-IQ3_M × 4 workloads × FA off/on,
24 cells, 3-trial median) under §23's auto-tile=128. Decision
rule landed on C: zero models meet B's ≥5% long-short TTFT
gate (deltas +3.0% / -0.4% / -0.1%) and short-short regressions
exceed A's 3% gate (Mistral decode -4.5%, Qwen3-8B TTFT +3.6%).
FA does win long-long TTFT (-7.3% to -23.3%) but neither §20
rule clause keys on long-long. Gate stays default-off as future
infra; zero `src/` change. With those closures, the remaining
headroom is the deferred concat-graph batched compute lever for
encoders (only opens on a batch-throughput use-case) plus
MEMORY64 for 70B-class targets, and §C-v2-A resurrection (§22
partially alleviates per-step K+1 verify cost — never
re-measured under tile=128 since §22 landed).
**§25 (2026-04-27 — LANDED, 10+ commits) dashboard hygiene +
new visualization cycle.** Five new charts on the main inference
tab (`f8e0ae6` family-coloured accuracy×speed scatter,
`b33f019` quant connector lines, `e4978ae` decode tok/s vs
param-count scatter, `5af0370` per-dim score heatmap, `504c837`
latest-vs-prior delta columns on runs+evals tables); explicit
encoder/BERT filter so the main tab is strictly chat-only
(`02f7872`); three encoder-side analogs under the Embeddings
section (`845b687` cosine×latency scatter, `cf4c49d` param×
throughput scatter, `88f3df5` Δ total ms on embeddings table);
delta polarity fix so lower-is-better metrics (`Δ total ms`)
read green=speedup / red=regression (`620407e`). DB audit
during the cycle confirmed `smoke-runs.db` is clean (29 runs /
30 evals, no purge candidates). A `/models` endpoint
refactor on `eval/live-server.ts` (`11c1626`) drives
`isEncoderModel` / `inferEncoderParamCountM` from the registry
instead of hand-maintained id-prefix maps — eliminates the
latent footgun where registering a new encoder family
(nomic-embed-*, e5-*) would silently leak encoder rows back
onto the main tab. Contract test pinned at `14038e2`. Two
narrower follow-ups remain: (a) `inferModelFamily` still uses
id-prefix matching (registry's `family` field could replace it
but the family-color palette is keyed off inferred labels;
small palette/key audit needed); (b) the encoder-architecture
check still hardcodes `architecture === "bert"` — if a
non-BERT encoder ever lands, update `isEncoderModel` and
`inferEncoderParamCountM`. Ship gate (427/11/0) maintained on
every commit.
**§26 (2026-04-27 — CLOSED, side-branch + 3 main commits) §C-v2-A
re-measurement under §22 tile=128.** Direct empirical test of
§24's parting recommendation. Cherry-picked the 4 §22
implementation commits (`c38fb8f`, `f281ac3`, `2fcc334`,
`18e1677`) onto `feat/spec-decode-v2-greedy`; ran the canonical
4-cell gate matrix at qwen3-8b-iq3m × qwen3-0.6b-q4f16 K=4 with
explicit `--prefill-tile 128` on both target and drafter. Gate 1
(speedup ≥1.5×): **0.42×** — FAIL by 3.6×. Gate 2 (safety ≥0.95×):
**0.54×** — FAIL by 0.4×. The K+1=5 verify graph is three orders
of magnitude below the 128-token tile threshold and is never
split, so tile=128 cannot affect verify cost on this workload —
exactly as the spec hypothesized. Cell 4 drift -33% vs §C-v2-A
close (12.7 → 8.5) is a notable safety regression flagged for any
future v2-A resurrection. **§C-v2-A definitively closed under all
known levers**; resurrection now requires architectural change
(faster K+1 verify via upstream ggml-webgpu dispatch coalescing,
OR MEMORY64 → 70B+ target shifting the target/drafter ratio from
13× to ~100×). Side branch retained as archived infra; do not
merge. Files on main: spec `b23ccc9`, plan `f0a682c`, TODO closure
`e715160`. Files on side branch: cherry-picks + matrix + SUMMARY,
tip `6b20aad`. Zero `src/` change on `main`; checkall remains
427/11/0.

Findings, one bug fix, one upstream rebase, one
quant-promotion, encoder perf characterization, plus a
dashboard hygiene pass from these sessions:

- **Bug #28 (Q3_K shader) FIXED — see §14.** Root cause was
  UB shift-by-32 in `load_u32_at_src{,0}` u32 loader helpers
  (`hi << (32u - shift)` when `shift == 0`), corrupting any
  aligned read through these helpers. Q3_K mul_mat_vec and
  Q3_K get_rows are the user-visible victims; Q4_K_S happened
  to use unaligned reads and was unaffected. Patch 11 on
  `webllm-browser-patches` (`a536df4f4` after rebase, was
  `391c59f39` pre-rebase). Q3_K_M now coherent at 24.4 tok/s
  on Mistral-7B; Q4_K_S regression-safe at 36.0 tok/s.
- **llama.cpp rebased to upstream `434b2a1ff` (2026-04-27).**
  13-commit gap from prior base `78433f606`; zero conflicts;
  all 11 patches replayed cleanly. 3 of the 13 commits touched
  `ggml-webgpu/` (Q1_0 kernel #22374, fast i-quant mat-vec
  kernels #22344, performance-portable register-tile / subgroup
  matmul tuning #22241) — none collided with our patch surface
  (LAYER_NORM via `row_norm.wgsl`, browser ASYNCIFY,
  request-based readback API, profiling, UB shift-by-32 fix).
  Post-rebase verification: WASM build clean (~32 KB binary
  growth from new kernels); checkall 427/11/0; browser smoke on
  TinyLlama Q4_0 reported 120 tok/s decode (above the 105
  steady-state baseline) and encoder cosine 0.76 (matches §21
  pin → patches 9-10 LAYER_NORM healthy); zero console
  errors/warnings. Tip is now `981859864`. Safety branch
  preserved at `webllm-browser-patches-pre-rebase-2026-04-27`.
- **Free-win sweep (2026-04-27, post-rebase):** ran
  `bun run eval/perf.ts --runs 3` on six models against the
  §17 "pre-§A change" bench-inf baseline table. **Headline:
  upstream's fast i-quant mat-vec kernels (#22344) delivered
  a +70-80% throughput win on IQ3_M models — the entire 8B+
  fleet got faster for free.**

  | Model                         | Quant   | §17 base | Post-rebase | Δ |
  |---|---|---:|---:|---:|
  | tinyllama-1.1b-chat-q4_0      | Q4_0    | 105.7   | 110.8       | +4.8% |
  | qwen3-0.6b-q4f16              | Q8_0    | ~85     | 89.8        | +5.6% |
  | qwen3-1.7b-q4f16              | Q8_0    | ~59*    | 62.2*       | +5.4% |
  | mistral-7b-instruct-v0.3-q4ks | Q4_K_S  | 34.5    | 35.8        | +3.8% |
  | **llama-3.1-8b-instruct-iq3m**| **IQ3_M** | **16.8** | **29.0** | **+72.6%** |
  | **qwen3-8b-iq3m**             | **IQ3_M** | **15.1** | **27.2** | **+80.1%** |

  \* qwen3-1.7b numbers are 17-token warmup-dominated runs
  (`Tell one short joke.` elicits short Qwen replies); not the
  117-token clean steady-state from the TODO header.

  **Story confirmed by quant-family pattern:** the i-quant
  jump is isolated to IQ3_M (both 8B models, both families).
  K-quant (Q4_K_S Mistral) is essentially flat (+3.8% ≈ noise),
  consistent with #22344 targeting only the i-quant decompression
  path. Dense quants (Q4_0 / Q8_0) all sit at +4-6%, plausibly
  attributable to upstream's register-tile / subgroup matmul
  tuning (#22241).

  **Implications:**
  - The 8B fleet's effective throughput nearly doubled. §16's
    "16.2 tok/s" baseline for `qwen3-8b-iq3m` is now obsolete;
    canonical bench-inf number is **27.2 tok/s**.
  - **§C-v2-A target/drafter ratio analysis shifts —
    EMPIRICALLY SETTLED 2026-04-28 (§28).** Re-ran §26's 4-cell
    gate matrix on side branch `feat/spec-decode-v2-greedy`
    against the rebuilt WASM. Both gates **worsened**, not
    improved: gate 1 = 9.7 / 28.2 = **0.34×** (was 0.42×);
    gate 2 = 12.7 / 28.4 = **0.45×** (was 0.54×). Target
    baseline scaled +78% (15.8 → 28.2 tok/s) but drafted path
    only +45-49% — the drafter (qwen3-0.6b Q8_0) doesn't use
    the i-quant code path, so K=4 drafter forwards retain their
    pre-rebase cost; only the K+1=5 verify pass got faster, and
    that saving is amortized across 5 tokens. The §26
    "resurrection path (a) — faster K+1 verify" is now
    empirically closed: the rebase delivered exactly that and
    the gates moved *against* the thresholds, not toward them.
    Drafter overhead must scale symmetrically with target
    speedup or the relative ratio worsens. Resurrection paths
    still on the table: (b) MEMORY64 → 70B+ target (unchanged);
    (c, new) a smaller i-quant drafter that also uses #22344 —
    no candidate model registered today. Side branch tip
    `9bdd707` carries the §28 matrix + SUMMARY at
    `eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/`.
    **§C-v2-A remains closed under all known levers.**
  - **§17 / §A reopening:** §A's lever 1 was reverted because
    `MUL_ACC_Q4_0` showed only -2.9% matmul / +0.6% tok/s on
    TinyLlama; the wave-2 7B+ fleet was structurally
    inapplicable (K-quant TPB=16, IQ3_M routes through
    `mul_mat.wgsl` not `mul_mat_vec.wgsl`). With IQ3_M now
    fast, §A remains closed for the wrong reason that already
    closed it (lever shape doesn't apply); no change.
  - **Net characterization update at 8B IQ3_M (post-rebase,
    measured 2026-04-27 via `make smoke-bench
    PERF_MODEL=qwen3-8b-iq3m PERF_RUNS=3`, 60-step trace):**

    | Bucket                  | §17 baseline (profile) | Post-rebase (profile) | Δ |
    |---|---:|---:|---:|
    | tok/s (profile mode)    | 14.3                   | 22.0                  | +54% |
    | graphComputeMs (median) | ~68 ms                 | 42.60 ms              | -37% |
    | backendMatmulMs (median)| 48.04 ms               | 23.07 ms              | **-52%** |
    | backendMatmulMs %graph  | 70.5%                  | 55.0%                 | -15.5 pp |
    | backendEncodeOverheadMs |  ~?                    |  4.50 ms / 10.7%      | — |
    | backendAttentionMs      |  ~?                    |  0.72 ms /  1.7%      | — |
    | backendDispatchCount    | 805/token              | 805/token             | unchanged |

    Dispatch count is bit-identical pre/post — the win is
    pure kernel speedup (#22344 i-quant mat-vec) on the same
    graph, not a graph-shape change. Matmul share dropped
    **15.5 percentage points** but is still the lead bucket
    (55.0% of graph). Encode overhead is now the secondary
    suspect at ~10.7%; attention is negligible (1.7%).

    Profile-mode perturbation also shrank: 27.2 (non-profile)
    → 22.0 (profile) is **-19%** vs the historical -29 to -34%
    on Q4_0/Q8_0 — fewer per-dispatch timestamp samples are
    bottlenecking IQ3_M now that the kernel itself is faster.

    **No new lever exposed.** Matmul still leads at 55% but the
    absolute win remaining (halve again → ~10% step gain) is
    smaller than the levers already closed (§A 0.6%, §18 -5.8%,
    §19 0.20× regress). Encode overhead at 10.7% × ~22 tok/s
    means a hypothetical encode-elimination would max out at
    ~10% gain — also below the 1.5× ship-gate threshold for
    new infrastructure work.

  Free-win sweep duration: ~5 minutes wall (one rebuild +
  smoke-restart per model). Sweep done — no follow-on work
  triggered.
- **llama.cpp rebased to upstream `78433f606` (2026-04-26).**
  6-commit gap from prior base `b760272f1`; zero conflicts.
  None of the 6 commits touch `ggml-webgpu/`, WGSL shaders,
  ASYNCIFY, or the graph-visit code we patched. Upstream
  delta was: backend-meta recurrent state fix (we don't use
  recurrent state); CUDA/CPU/OpenCL backend changes (we
  build none of them); CODEOWNERS update. Safety branch
  preserved at `webllm-browser-patches-pre-rebase-2026-04-26`.
- **Workarounds for the 4 GiB WASM cap:** Q4_K_S works at
  7B (3953 MB Mistral). For 8B+, Q4_K_S exceeds the cap;
  IQ3_M / IQ3_S are the smaller-bandwidth working
  alternatives via the IQ-family code path (3609 MB Llama-
  3.1-8B, ~3252 MB Qwen3-8B IQ3_XXS). MEMORY64 to bump
  the cap to 8 GiB requires the wasm memory64 proposal —
  multi-day engineering effort, not a flag flip.
- **§A subgroup-cooperative loading CLOSED 2026-04-26 (§17).**
  Original subgroup-broadcast premise rejected on inspection
  (kernel already partitions src0 perfectly across threads;
  no redundant loads to coalesce). Lever-1 replacement
  (THREADS_PER_BLOCK 4→2) measured on the 4-baseline harness:
  only TinyLlama Q4_0 benefited (sub-trigger -2.9% matmul /
  +0.6% tok/s — noise). Q4_K_S (Mistral) is a K-quant with
  TPB=16 and a different block layout (structurally excluded
  from §A's design); IQ3_M (both 8Bs) has no `mul_mat_vec.wgsl`
  path and routes through general `mul_mat.wgsl` instead.
  Levers 2 + 3 face the same applicability constraint. Shader
  reverted; no patches landed.
- **§4 Flash Attention enable CLOSED 2026-04-26 (§18).**
  Integrated `ggml_flash_attn_ext` into all three attention
  branches (decode, prefill, debug-checkpoint) with F16 KV
  cache + transposed V layout. Measured on the 4-baseline:
  FA engaged on all 4 (dispatch counts -10-13%, matmul
  -2 to -16%), but the new `backendAttentionMs` overhead
  (1.3-3.3 ms/step) exceeds savings at single-token decode.
  Mistral-7B regressed -5.8% (blocking — exceeds 3% gate);
  no model gained ≥2%. **FA's main wins are prefill (long
  prompts) and longer decode batches (>256 tokens) — neither
  is exercised by the bench-inf gate.** Bridge wrappers,
  TS bindings, surface test retained as future-work
  infrastructure (`33f10eb`, `4692bce`+`d26d736`, `068ef84`);
  implementation reverted via `git checkout 068ef84 --
  src/inference/model-inference.ts smoke-test/real-model-page.js`.
  **A future revisit at long-decode or prefill-TTFT scope
  could ship FA without touching the bridge.** See
  `docs/superpowers/plans/2026-04-26-fa-enable.md` for the
  plan and decision-rule details.
- **Loader / parser refactor (§11):** GGUF streams cleanly
  through the WASM heap; ctxCreate over-allocation fixed.
  Confirmed working at 3.6 GB / 3.95 GB streaming.
- **Dashboard hygiene pass (2026-04-26):** dropped 23 broken-
  era runs and 23 broken-era evals from
  `eval/reports/smoke-runs.db`. Three cohorts purged:
  bug-#28 q3km gibberish (3+3); pre-`9156deb` (Apr-25 16:19Z)
  realistic-sampler ½-speed JS slow path (qwen3-0.6b ×6 +
  llama-3.2-1b ×3 = 9 profile runs ×2 phases = 18+18); pre-
  `38e41c4` (Apr-26 03:50Z) qwen2 missing attention biases
  (qwen2.5-1.5b ×1 = 2+2). Repopulated dashboard by re-running
  11 profiles under the post-fix pipeline (qwen3-0.6b cold/
  warm/hot × off+thinking, llama-3.2-1b cold/warm/hot,
  tinyllama-warm, qwen2.5-1.5b-warm) — all 22 phases passed,
  output coherent. **bench-profile harness numbers run ~70%
  of `perf.ts` smoke-bench steady-state** (TinyLlama 73.6 vs
  105 perf.ts; qwen3-0.6b-warm 62 vs 85; qwen2.5-1.5b 42 vs
  84) — known harness-overhead gap, not a regression. Use
  `perf.ts` for engine-throughput claims; bench-profile for
  cross-task accuracy + dashboard. **TinyLlama 35% accuracy
  is real model weakness** (1.1B base-class generates a
  poem when asked for a joke), not broken pipeline. dashboard
  reload required to see the cleanup (live-server SSE doesn't
  broadcast deletes).

**Next target options (pick one — see "Recommended first move"
below; A/B/C/F/§4-decode/§C-v1/§4-prefill/§C-v2-A/§D/§22 all closed):**

A. ~~Add Qwen3-8B IQ3_M as wave-2 model 4.~~ **Done — §16.**
B. ~~§A subgroup-cooperative loading.~~ **CLOSED 2026-04-26 — §17.**
C. ~~Fix the Q3_K shader (#28).~~ **Done — §14.**
F. ~~Promote or retire the Q3_K_M test entry.~~ **Done — §15.**
§4. ~~Flash Attention enable for decode.~~ **CLOSED 2026-04-26 — §18.**
§C. ~~Drafter-based speculative decoding (v1).~~ **CLOSED 2026-04-26 — §19** (measured 0.20× regression; verify-readback dominates).
§4-prefill. ~~FA revisit at prefill / long-decode scope.~~ **CLOSED 2026-04-26 — §20** (TinyLlama wins everywhere; Mistral short-short -3.3% over gate; 7B+ long-prefill blocked by WebGPU buffer-binding limit, not FA).
§C-v2-A. ~~Greedy spec-decode + GPU-resident verify.~~ **CLOSED 2026-04-27 on side branch `feat/spec-decode-v2-greedy`** (gate 1: 0.36× vs ≥1.5× target; gate 2: 0.78× vs ≥0.95×; per-step verify overhead at 8B IQ3_M target × 0.6B Q8 drafter caps α at ~0.2-0.25, well below the K=4 ceiling needed to break even). Driver, K+1 verify, AdaptiveGate, contract gate, creative-low-alpha fixture, `--draft-length` flag, `forwardVerifyArgmax`, and ~30 unit/integration tests retained on side branch as resurrection-ready infra; **do not merge to `main`**. Resurrection paths: (a) much larger target via MEMORY64 shifts target/drafter ratio from 13× to 100×+, (b) faster K+1 verify via 7B+ long-prefill graph-buffer work cuts per-step verify cost. Measurement detail in side-branch TODO §22-§24; tip `646320c`.
§D. ~~Encoder/embedding perf pass.~~ **CLOSED 2026-04-27 — §21** (L1 ctx/graph reuse measured + reverted; Phase 2.5 diagnostic surfaced 95.6% graphCompute share = ~390 dispatches × ~80 µs each → encoder is dispatch-bound, not memory- or compute-bound at this scale; L2/L3-sequential project to <5% combined; only viable lever — concat-graph batched compute — is a non-goal in §D and deferred until a real batch-encoder-throughput use-case emerges). Harness (`eval/embed-perf.ts` + `?embedPerf=…` smoke URL params + `make embed-perf{,-baseline}`) shipped to main; cosine baseline pinned at 0.76 ±0.005 (`tests/encoder-cosine-parity.test.ts`).
§22. ~~7B+ long-prefill graph-buffer tiling.~~ **CLOSED 2026-04-27 — see Completed §22.** Ship-gated default-off; tile=128 unblocks Mistral-7B-Q4_K_S (33.6 tok/s) and Qwen3-8B-IQ3_M (16.2 tok/s) at prefill-512, both within noise of §18 baselines. TinyLlama tile=128 regresses TTFT +81.3% (extra graph dispatches for single-graph-fit models), so the gate stays default-off. Opt in via `?prefillTile=N` (smoke) or `--prefill-tile <n>` (`eval/perf.ts`); ctor option `prefillTileSize` on `ModelInference`. Phase 0 disproved §20's GPU-cap hypothesis: actual abort is the host-side ggml graph allocator at `ggml-alloc.c:82` (not the WebGPU buffer-binding cap). Branch `feat/prefill-tiling-22` (default-off plumbing only — no `recommendedPrefillTile` registry metadata yet; deferred per ship-gated scope). Raw matrix at `eval/reports/prefill-tiling-2026-04-27/SUMMARY.md`.

D. **Bump `MAXIMUM_MEMORY` (deferred §12, dropped in
   priority).** Confirmed in earlier sessions that 4 GiB
   is the 32-bit WASM hard cap. Going beyond requires
   `-sMEMORY64=1` (changes pointer types throughout the
   bridge, possible asyncify interactions). Multi-day
   engineering. Only worth it for wave-3 12B+ candidates
   that need Q4_K_S+.

E. **Remaining deferred items (in rough priority):**
   - **7B+ long-prefill graph-buffer infrastructure**
     *(blocking the §4 hypothesis at scale — see §20).*
     Mistral-7B and both 8B candidates abort at
     `backend_alloc_ctx_tensors` when building long-prefill
     graphs (32 layers × seq=512 of F32 intermediates exceeds
     WebGPU max-buffer-binding cap), independent of FA mode.
     Bumping the `graphMem` allocator did not help; the abort
     is GPU-side. Without this, FA wins at prefill / long-
     decode at the canonical 4-baseline cannot be measured.
   - **§4 FA revisit at long-decode / prefill-TTFT scope**
     **CLOSED 2026-04-26 at §20** — TinyLlama wins everywhere,
     Mistral short-short -3.3% over gate, 7B+ long-prefill
     blocked by the buffer-binding limit above. Gate retained
     in tree (default-off `flashAttn` constructor option +
     `?fa=on` smoke param + `--fa <on|off>` perf.ts flag);
     no further work until the infra item above unblocks the
     measurement.
   - **§B FA shape-routing** for prefill/TTFT — same
     `flash_attn_get_decisions` code path; blocked on the
     7B+ buffer-binding infra item above.
   - **§D encoder/embedding perf pass.** **CLOSED 2026-04-27 — §21.**
     Single-text levers exhausted (L1 reverted; L2/L3-sequential <5%
     combined headroom). The only structural lever is concat-graph
     batched compute (was §D non-goal); reopen if a batch-encoder-
     throughput use-case emerges. Harness (`eval/embed-perf.ts` +
     smoke `?embedPerf=…`) and cosine pin (0.76 ±0.005) shipped.
   - **§C v2-A greedy spec-decode + GPU-resident verify.**
     **CLOSED 2026-04-27 on side branch
     `feat/spec-decode-v2-greedy`** — measured-and-closed
     pattern. Eliminates v1's 2.4 MB / step readback via
     `forwardVerifyArgmax` (16 B / step), but at qwen3-8b-
     iq3m × qwen3-0.6b-q4f16 K=4 still fails both ship gates
     (0.36× high-α speedup; 0.78× low-α safety). Per-step
     drafter forwards (~48 ms) + target K+1 verify (~70-80
     ms) ≈ 120 ms; even at perfect K=4 accept that's only
     ~33 tok/s vs 16 tok/s baseline (~2×, tight at 1.5× spec
     gate even at α=1). Measured α ≈ 0.2-0.25 inverts the
     trade. Driver, AdaptiveGate, contract gate, K+1 verify,
     ~30 tests, and tooling all retained on side branch.
     Resurrection only worth it if (i) a much larger target
     lands (70B+ via MEMORY64 → target/drafter ratio 100×+),
     or (ii) faster K+1 verify via 7B+ long-prefill graph-
     buffer work below cuts per-step verify cost.
   - **Deferred wave-1 architectures** (Gemma 2, Phi 3) —
     5+ gaps for Gemma; mostly fused-QKV for Phi 3. See
     "Completed on 2026-04-26" §9.

**Net characterization at 8B IQ3_M (post-§16, both
families):** matmul ≈ 65-69% of decode (Llama-3.1 71%,
Qwen3 67% × graph 97% of step). **All single-token decode
kernel-tuning AND algorithmic-amortization levers — including
greedy spec-decode with GPU-resident verify — are now closed
without ship.** §17 ruled out matmul-kernel rework (§A); §18
ruled out FA fusion at N=1 decode; §19 ruled out drafter
speculative decoding at K=4 with full-row verify (verify-
readback dominates); §20 ruled out FA at small-prefill /
long-decode scale on the 7B+ fleet (TinyLlama wins preserved
behind a default-off gate; 7B+ blocked by WebGPU max-buffer-
binding limit at long-prefill); §C-v2-A (side branch, 2026-04-27)
ruled out greedy spec with GPU-resident K+1 verify at the
canonical target/drafter ratio (per-step verify overhead caps
α below the K=4 break-even ceiling); §21 closed §D on a
diagnostic finding (encoder embed is dispatch-bound, single-text
levers <5% headroom; only structural lever — concat-graph batched
compute — is a non-goal until a use-case emerges); **§26 ruled
out §C-v2-A resurrection under §22 tile=128 (gate 1 0.42×, gate 2
0.54× — both fail decisively; verify graph at K+1=5 is three
orders of magnitude below the 128-token tile threshold and is
never split, so tile=128 cannot affect verify cost on this
workload).** **All algorithmic levers at the canonical 4-baseline
are now exhausted.** Remaining headroom is **architectural
infrastructure**: MEMORY64 to bring 70B+ targets into reach
(multi-day, conditional on a deployment ask; would also shift the
§C-v2-A target/drafter ratio from 13× to ~100×, the regime where
Leviathan-style speculation pays off); upstream ggml-webgpu
dispatch coalescing or fused-graph optimization (re-run §21 +
§26 harnesses on every llama.cpp rebase to spot free wins);
§D's deferred concat-graph lever (encoder-side fallback if a
batch-throughput use-case appears).

Boot sequence for a fresh session:

1. **`make checkall`** — confirm 427 pass / 11 skip / 0 fail.
   The §C drafter spec-decoding work added 19 unit + integration
   tests across `tests/sampler.test.ts` (7), `tests/speculative-
   rejection.test.ts` (11), `tests/forward-verify-equivalence.test.ts`
   (Bun-skipped, +6 more), `tests/speculative-integration.test.ts`
   (Bun-skipped, 3), and 1 engagement-gate test. The §20 FA-revisit
   work added 5 tests at `tests/fa-mode-config.test.ts` (413 → 418).
   The §21 §D cycle added 1 test at `tests/encoder-cosine-parity.test.ts`
   (418 → 419). The §22 prefill-tile cycle added 5 unit tests at
   `tests/prefill-tiling-config.test.ts` plus 1 Bun-skipped equivalence
   stub at `tests/prefill-tiling-equivalence.test.ts` (419 → 424;
   skip count 10 → 11). The §23 default-on auto-tile cycle added 2
   registry-shape tests in `tests/eval-models.test.ts` (424 → 426).
   The §24 §4 FA revisit at 7B+ long-prefill cycle added 0 tests
   (closure C — measurement campaign + closure writeup; zero `src/`
   change). **§25 dashboard hygiene + new viz cycle added 1 test**
   (`tests/live-server.test.ts` gained a `/models` endpoint contract
   test pinning shape, sort order, and architecture+paramsB
   coverage; 426 → 427 pass). The WebGPU-gated integration tests
   skip under Bun (no `navigator.gpu`).
2. **`git log --oneline -30`** — top of `main` is the §26
   §C-v2-A re-measurement closure (3 commits on `main`; the bulk
   of the cycle's work landed on the `feat/spec-decode-v2-greedy`
   side branch, which is **archived — do not merge**). Tip is
   `e715160 docs(TODO): §26 — §C-v2-A re-measurement under tile=128
   CLOSED`. Below it: `f0a682c docs(plan): §26 §C-v2-A re-measurement
   under §22 tile=128` → `b23ccc9 docs(spec): §26 §C-v2-A re-
   measurement under §22 tile=128`. Below those, the §25
   dashboard cycle (12 commits): `6622ec7 docs(TODO): refresh
   resumption checklist post-/models refactor` → `14038e2
   test(live-server): add /models endpoint contract test` →
   `11c1626` `/models` endpoint + registry-driven filters →
   `dd59704` §25 docs(TODO) refresh → `620407e` polarity fix →
   `88f3df5` #B5 → `cf4c49d` #B3 → `845b687` #B1 → `02f7872`
   chore: encoder filter on main tab → `504c837` #5 → `5af0370` #4
   → `e4978ae` #3 → `b33f019` #2 → `f8e0ae6` #1. Then
   `85988c8 docs(TODO): §24 — §4 FA revisit at 7B+ long-prefill
   MEASURED + CLOSED` is the §24 closure (single docs/measurement
   commit, zero `src/` change). Below §24: §23
   (§22 default-on auto-tile via `recommendedPrefillTile`) landed
   on `main` on 2026-04-27 as a single commit `0c50e03 feat(eval):
   §22 default-on auto-tile via recommendedPrefillTile`. Below it:
   `1b15f37 docs(TODO): refresh resumption checklist post-§22 merge`.
   Then the §22 fast-forward merge from 2026-04-27: `a73ad88
   docs(TODO): §22 — prefill-tile chunking SHIP GATED`. Below it
   the §22 implementation: `5b5705a` (Task 5 matrix),
   `18e1677` (Task 4 perf flag), `2fcc334` (Task 3 smoke wiring),
   `f281ac3` (Task 2 equivalence stub), `c38fb8f` (Task 1 ctor option
   + dispatcher), `8e21036` (Task 0 Phase 0 diagnostic). Below those:
   `b8eebf8` (post-§21 resumption refresh), `b6a288c docs: generalize
   DOCUMENTATION_STYLE_GUIDE.md`. The §21 block: `5e24913` (§21 §D
   closure), `66bc603` (§D Phase 2.5 diagnostic), `3a6a366` (revert L1
   same-graph-cache — gate failed), `f0d89f1` (Phase 2 L1 measurements),
   `5eb1f73` (L1 implementation, reverted), `c24c628` (Phase 2
   choice spec), `a92ca7e` (Phase 1 baseline), `4c237a3`
   (cosine parity test), `582a3ba` (embed-perf Make targets),
   `d51d2c5` (embed-perf harness CLI), `3315a88` (smoke-page
   embedPerf hook), `4944209` (embed-prompts fixtures),
   `670ba2e` (§D plan), `092248e` (§D design spec),
   `a36ef48` (cosine baseline JSON). Before that:
   `b872b5f docs(TODO): §20 — §4 FA revisit measured + CLOSED`,
   then the §20
   implementation commits: `f1b19ab` (long-prompt fixtures
   + perf.ts flags), `ddc6e39` (smoke `?fa=on` + F16 KV
   fix), `faccb8e` (gated FA in `forwardDecode` /
   `forwardVerify` / `debugLayerOutput`), `4bfa6f4` (gated
   FA in `forward()`), `4138232` (F16 mask),
   `91d8e26` (flashAttn ctor option + dual V-cache).
   Below those: `a3df85d` (post-§19 next-step refresh),
   `9984fa4` (§19 docs), `aac7080` (engine spec-decode
   revert), `1b23ca8` (drafter handle-id fix). Below those
   the §19 implementation commits (`bbd1dff` smoke-page +
   Makefile, `1b6fd72`+`81e3df0` engine routing, `1c2db1b`
   integration test, `87e732a`+`5572bd4`+`efa094c`+
   `dd84729` driver, `183b99f`+`90ecf37`+`cf85756`+
   `9d7c258` rejection sampler, `d7e8605`+`11fe3f7`
   sampler helpers, `3fdd347`+`433252b` model-inference
   primitives) — all retained except the engine routing
   block. Below those: `d680371`/`ffd7276` (§18 §4 FA
   closure), `068ef84`/`d26d736`/`4692bce`/`33f10eb`
   (FA infrastructure that survived), then `bebed0c` (§17
   §A closure) and `c98d0a7` (§16 qwen3-8b register).
   The merged branch `feat/prefill-tiling-22` was already
   deleted at merge time. The §20-era `feat/fa-revisit-prefill-
   long-decode` is also already merged; if it's still in your
   local checkout, `git branch -d` is safe (it points at
   `b872b5f` already on `main`).
3. **`git -C ~/Repos/llama.cpp log --oneline -12 webllm-browser-patches`**
   — confirm the **11-patch stack** is intact and the base
   is upstream `434b2a1ff ggml-webgpu: add Q1_0 support
   (#22374)` (rebased 2026-04-27). Tip is `981859864 ggml-webgpu:
   fix UB shift-by-32 in load_u32_at_src{,0}` — patch 11, the
   bug #28 fix (SHA changed from pre-rebase `a536df4f4` because
   of the rebase replay; same patch content). Safety branches
   `webllm-browser-patches-pre-rebase-2026-04-27` (today) and
   `webllm-browser-patches-pre-rebase-2026-04-26` (prior)
   preserve pre-rebase tips if needed. The 2026-04-26 → 2026-04-27
   delta was 13 upstream commits, 3 of them in `ggml-webgpu/`
   (Q1_0 kernel #22374, fast i-quant mat-vec kernels #22344,
   register-tile / subgroup matmul tuning #22241) — **zero
   conflicts** on rebase. WASM rebuild was clean; checkall held
   at 427/11/0; browser smoke on TinyLlama Q4_0 reported 120
   tok/s decode (above the 105 baseline) and encoder cosine 0.76
   (matches §21 pin); zero console errors/warnings. **§17, §18,
   §19, §20, §21, §22, and §23 added zero patches** — the
   `__EMSCRIPTEN__` guard around FA was already removed in the
   2026-04-25 rebase; §20 re-uses the bridge wrappers from §18
   with no new shader work; §21, §22, and §23 are pure-TS /
   pure-JS work above the bridge with no shader changes.
4. **WASM build state.** `smoke-test/webllm-bundle.js` and
   `smoke-test/webllm-wasm.{js,wasm}` mtimes are 2026-04-27
   ~16:11 (post-§22 — contain the `prefillTileSize` ctor
   option, `forwardSingle()` rename, and `?prefillTile=N` URL
   param wiring on top of §21's `runEmbedPerfHook` and §20's
   gated FA path). **§23 did not rebuild WASM** — the
   `recommendedPrefillTile` field is harness-side TS and the
   smoke-page lookup map is plain JS; neither is bundled into
   `webllm-bundle.js`. The 2026-04-27 ~16:11 artifacts remain
   the canonical build for §22+§23 verification.
   If the artifacts look stale, run:
   `source ~/emsdk/emsdk_env.sh && make wasm-build && bun
   build src/index.ts --outfile smoke-test/webllm-bundle.js
   --target browser && cp src/wasm/build/webllm-wasm.{js,wasm}
   smoke-test/ && make smoke-restart`. Then navigate the smoke
   page to `model=mistral-7b-instruct-v0.3-q3km` — Q3_K_M
   coherent at ≥20 tok/s confirms patch 11 is healthy. To
   confirm §23 auto-default wiring, navigate to
   `model=mistral-7b-instruct-v0.3-q4ks&prompt=<prefill-512>`
   *with no `?prefillTile=` param* — the mode bar should show
   the `tile: 128` pill and prefill should complete (§22
   matrix cell 4 numbers ≈ 33 tok/s). To confirm §23 force-
   disable still works, append `&prefillTile=0` to the same
   URL — the pill disappears and prefill should abort with
   the §22 ggml-alloc signature.
5. **Read for context:** §17 (§A closure), §18 (§4 FA
   closure at N=1 decode), §19 (§C drafter spec-decode
   closure), §20 (§4 FA revisit at prefill / long-decode
   scope closure), §21 (§D encoder perf cycle — diagnostic
   close, no ship), §22 (7B+ long-prefill graph-buffer
   tiling — gated ship, default-off), and §23 (§22 default-on
   flip via `recommendedPrefillTile` registry field — landed
   2026-04-27 as a single commit, `0c50e03`). The first six
   follow the "measure-and-close" pattern; §23 is a thin
   policy-layer follow-on with no measurement campaign.
   §22 is the cleanest recent template for **gated-ship**:
   opt-in plumbing threaded through ctor / URL param / CLI
   flag, default-off keeps the fast-path bit-identical,
   decision rule cited matrix numbers — see
   `docs/superpowers/plans/2026-04-27-prefill-tiling.md` and
   `eval/reports/prefill-tiling-2026-04-27/SUMMARY.md`.
   §23 is the cleanest template for **promoting an opt-in
   gate to default-on without a new measurement** when the
   gating decision can be expressed as registry data. §21 remains the cleanest template
   for **closing on a diagnostic finding** when the bottleneck
   profile invalidates the planned levers — see
   `docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md`
   (Phase 2.5 addendum) and
   `docs/superpowers/plans/2026-04-27-encoder-perf-pass.md`.
   The §20 plan
   at `docs/superpowers/plans/2026-04-26-fa-revisit-long-
   decode.md` and the matrix raw logs at
   `eval/reports/fa-revisit-2026-04-27/` carry the FA
   gate's full contract: F16 K + F16 V cache when
   `flashAttn=true` (else legacy F32 K + dim-swapped V),
   F16 causal mask in all four branches, dual V-write
   layouts in `forward` / `forwardVerify` / `forwardDecode`
   / `debugLayerOutput`. The §C plan at
   `docs/superpowers/plans/2026-04-26-speculative-
   decoding.md` and design at `docs/superpowers/specs/
   2026-04-26-speculative-decoding-design.md` are the
   reference for the v2 lever — driver code at
   `src/inference/speculative.ts` is wired up and tested;
   only the engine dispatch needs unblocking.
6. **Dashboard state check** (optional but useful before
   benching): `sqlite3 eval/reports/smoke-runs.db "SELECT
   COUNT(*) FROM runs; SELECT COUNT(*) FROM evals;"` —
   should return **29 runs / 30 evals** (unchanged through
   §17/§18/§19/§20/§21/§22/§23/§24/§26 — none of the eight
   closures produced new dashboard data, only TODO writeups,
   perf.ts logs, §22's `eval/reports/prefill-tiling-2026-04-27/`
   matrix, and §26's `eval/reports/spec-decode-v2-tile128-2026-04-27/`
   matrix on the side branch).
   The live dashboard SSE counter
   shows higher numbers (~52/53) because it accumulates
   streaming events without DB persistence; both views are
   correct but independent. If the dashboard tab is open
   from a prior session, force-reload — SSE doesn't
   broadcast deletes.
7. **Bridge wrappers retained from §18, now used by §20.**
   `op_flash_attn_ext`, `op_flash_attn_ext_set_prec`,
   `op_flash_attn_ext_add_sinks` exist in
   `src/wasm/webgpu-bridge.cpp` and are exported in
   `src/wasm/CMakeLists.txt`. `opFlashAttn`,
   `opFlashAttnSetPrec`, `opFlashAttnAddSinks` exist on
   the `GgmlWasm` class in `src/inference/ggml-wasm.ts`.
   §20 wired call sites into `model-inference.ts` behind
   `flashAttn=true`; the wrappers are now live (not dead)
   when the gate is enabled. **Do not delete them.**

   **§26 side-branch state** (no impact on `main`): the
   `feat/spec-decode-v2-greedy` branch carries the entire v2-A
   driver, AdaptiveGate, K+1 verify, contract gate, and ~30
   unit/integration tests, plus the four cherry-picked §22
   commits and §26's matrix evidence. Tip is `6b20aad`. **Do
   not merge to `main`** — both gates fail decisively (0.42× /
   0.54×) and the cell-4 -33% safety drift makes the path
   strictly worse than at §C-v2-A close. If a future cycle
   resurrects v2-A, branch fresh from `main` rather than
   reviving this branch (the cherry-picks now live on the
   branch itself; carry-over is straightforward).
8. **§20 FA gate + §22/§23 prefill-tile gate state (both on `main`).**
   `new ModelInference(wasm, hp)` with no `opts` argument is
   bit-identical to pre-§20/§22 behaviour: FA defaults off,
   `prefillTileSize` defaults to `0` at the ctor. **§23 moves
   the per-model auto-default up one layer** — the
   harness (`eval/perf.ts`) and the smoke page now consult
   `recommendedPrefillTile` (registry side) /
   `RECOMMENDED_PREFILL_TILE` (smoke side) to pick the ctor
   arg automatically. The ctor itself is unchanged.
   - **FA path:** pass `{ flashAttn: true }` to the constructor,
     append `?fa=on` to the smoke-page URL, or pass `--fa on` to
     `eval/perf.ts`. No auto-default — FA stays opt-in.
   - **Prefill-tile path (§22+§23):** auto-applies tile=128 on
     all 7B+ entries (mistral-7b q4ks/q3km/iq4xs, llama-3.1-8b-
     iq3m, qwen3-8b-iq3m). Sub-7B paths get tile=0
     (single-graph fast path). Override surface:
     `?prefillTile=N` (smoke), `--prefill-tile <n>`
     (`eval/perf.ts`), or `{ prefillTileSize: <n> }`
     (`ModelInference` ctor). Force-disable via `0`.
     Adding new 7B+ entries: set `recommendedPrefillTile: 128`
     in `eval/models.ts` AND add the `modelId → 128` row to
     `RECOMMENDED_PREFILL_TILE` in
     `smoke-test/real-model-page.js` (the registry test in
     `tests/eval-models.test.ts` enforces the registry side).
   - **`eval/perf.ts`** also accepts
     `--prompt-fixture <prefill-256|prefill-512|prefill-1024>` and
     `--decode-tokens <n>` for the long-prefill / long-decode
     harness; fixtures live in `eval/fixtures/long-prompts.ts`.
   - **Mistral-7B and 8B models** abort at `backend_alloc_ctx_tensors`
     on long-prefill workloads with `prefillTileSize=0` — the
     §22 closure documents the actual failure mechanism (host-side
     ggml graph allocator at `ggml-alloc.c:82`, not the WebGPU
     binding cap as §20 originally hypothesized). Post-§23 the
     auto-default makes this transparent for harness consumers;
     the abort surface only re-emerges if a caller passes
     `?prefillTile=0` / `--prefill-tile 0` explicitly. FA mode
     is orthogonal.

**Recommended first move:** **No obvious next lever — pick
deliberately.** §17 (§A matmul kernel), §18 (FA at N=1 decode),
§19 (§C drafter spec-decode at K=4 with full-row verify), §20
(§4 FA at prefill / long-decode), the side-branch §C-v2-A
(greedy spec-decode + GPU-resident K+1 verify), §21 (§D encoder
perf pass), §22 (7B+ long-prefill graph-buffer tiling), §23
(§22 default-on flip), §24 (§4 FA revisit at 7B+ long-prefill —
closed C: zero models meet the gated-ship long-short TTFT
threshold under tile=128), and **§26 (§C-v2-A re-measurement
under §22 tile=128 — both gates fail decisively, 0.42× / 0.54×;
verify graph at K+1=5 is three orders of magnitude below the
128-token tile threshold and is never split)** have all closed
or landed. §22 + §23 together turn the 7B+ long-prefill unblock
from "opt-in workaround" into "default behaviour for the
registered fleet". §24 measured the FA cells §20 could not
capture and confirmed FA stays behind the manual chain at 7B+
on three of four workloads (long-long TTFT being the lone
exception, but neither §20 rule clause keys on it). §26 retired
the last conditional algorithmic lever from §24's parting
recommendation. The algorithmic levers at the canonical
4-baseline are exhausted; remaining options are deliberate
strategic choices, not obvious wins.

**Candidate next levers (none are forced; pick on need),
in rough priority order:**

1. ~~**§C-v2-A resurrection (conditional).**~~ **CLOSED 2026-04-27 — §26.**
   Re-measured under §22 tile=128: gate 1 = 0.42× (FAIL by 3.6×),
   gate 2 = 0.54× (FAIL by 0.4×). Resurrection now requires
   architectural change (faster K+1 verify via upstream ggml-webgpu
   dispatch coalescing, or 70B-class target via MEMORY64), not an
   incremental lever. Side branch retained as archived infra;
   do not merge.
2. **MEMORY64 for 70B-class targets.** Multi-day engineering
   (pointer-type changes through the bridge, asyncify
   interactions). Only worth it for a concrete 70B+ deployment
   ask. Bonus: a 70B+ target shifts the §C-v2-A ratio from 13×
   to ~100×, the regime where Leviathan-style speculation
   actually pays off — see §26 resurrection path (b).
3. **§D concat-graph batched encoder compute.** Only opens on
   a real batch-encoder-throughput use-case (was non-goal in
   §21).
4. **Heuristic-based prefill-tile default in `ModelInference`.**
   §23 lever (b) — fold the per-model registry into a
   ctor-side derivation from `hyperparams.layerCount ×
   embeddingLength`. Nice-to-have when the registered 7B+
   fleet grows past hand-curation; defer until that pressure
   actually exists.
5. **Upstream ggml-webgpu rebase.** Re-run §21 embed-perf and
   §26 spec-decode harnesses on every llama.cpp rebase to spot
   free wins from upstream dispatch coalescing or kernel
   fusion. Mechanical; trigger on demand.

If none of those align with current priorities, the team
should pick a direction explicitly — there is no obvious
next perf lever waiting to be measured.

**~~Secondary option:~~ §D encoder/embedding perf pass — CLOSED 2026-04-27 (see §21).**
Measured, characterized, closed. Single-text levers (L1 ctx/graph
reuse measured + reverted; L2 readback shrink projected at <3% based
on Phase 2.5 diagnostic; L3 sequential embedBatch projected at 0% on
the dispatch-bound bottleneck) are exhausted; the only structural
lever (concat-graph batched compute) was non-goal in this cycle and
is deferred until a real use-case for batch encoder throughput
emerges. Net characterization: encoder embed is **dispatch-bound
(95.6% of step time is `graphCompute` and ~31 ms of that is ~390
dispatches × ~80 µs)**, not memory-bound or compute-bound. The
harness (`eval/embed-perf.ts` + smoke-page `?embedPerf=…` URL params)
is shipped infra; re-run after any future llama.cpp rebase to spot
free wins from upstream `ggml-webgpu` dispatch coalescing.

**Recommended path (any option):** invoke
`superpowers:writing-plans` with the chosen scope, then
execute via `superpowers:subagent-driven-development` (per
global preference). Mirror §17 / §18 / §19 / §20 plan
structure: explicit phases, measurable gates, and a
measure-and-close decision rule.

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
