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
> sampling — see `eval/reports/pre-rebase-baselines-2026-04-28/
> SUMMARY.md`): 87.9 / 68.2 / 44.0 / 29.7 / 23.5 / 21.8 tok/s.
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

The five workflow policies that gate every change on this project —
**30B model-size ceiling**, **quick-wins override on YAGNI**,
**probe-first default**, **complexity ≠ implementation time**, and
**always commit before work** — moved to
[`CLAUDE.md`](CLAUDE.md#workflow-policies-set-2026-04-28) on 2026-04-28
so they apply to all sessions, not just ones that load TODO.md. Read
that section before starting any new work; entries below cite the
policies (e.g. "deferred under the 30B ceiling") without re-stating
them.

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
Upstream cadence check 2026-04-28: no `ggml-webgpu/` movement → no
rebase trigger near firing.

**MEMORY64 full bridge migration promoted to active 2026-04-28**
(was external-trigger; user-requested as next major work item).
See dedicated section below.

---

### MEMORY64 full bridge migration (active, queued 2026-04-28)

**One-line goal.** Migrate the production WebLLM build from the
4 GiB-cap WASM32 path to the 16 GiB-cap WASM64 path so the engine
can host 13B Q4_K_S (~7.4 GiB) and 30B IQ3_M (~12.8 GiB) targets
within the 30B project ceiling.

**Status (2026-04-28):** Phases 0-4 complete; Phase 5 ran and **HALTED at gate failure**;
Phase 1.5 inline optimization landed; **perf re-bench queued for a quiet host**.

| Phase | Commit(s) | Result |
|---|---|---|
| 0 — audit + punch list | `c2cf1ef` (audit), `fe9c406` (filter fix) | 16 first-party stack/heap callsites mapped at `eval/reports/memory64-migration-2026-04-28/PUNCH-LIST.md` |
| 1 — JS bridge_malloc migration | `65cd0a8` | `is64` probe + `_bridge_malloc` / `_bridge_free` wrappers; +2 fixture tests |
| 1.5 — BigInt FFI coverage gap | `061a93c` | Extended `is64`-aware routing (`big()` / `num()` helpers) to all 38 `void*` / `size_t` boundaries; surfaced when Phase 4 wasm64 smoke aborted at `_ctx_create(1245184)` (Phase 1 only wrapped malloc/free). +1 test. |
| 2 — bridge ABI hardening | `9556cf0` | 11 `int32_t` → `size_t` promotions across `webgpu-bridge.cpp`; wasm32 byte-identical pre/post |
| 3 — GGUF loader BigInt boundary | `80b63d6` | Static analysis 14/14 safe; `eval/reports/memory64-migration-2026-04-28/PHASE-3-VERIFY.md` |
| 4 — dual-binary `make wasm-build` | `2ef3e9a` | `wasm-build-{wasm32,mem64}` sub-targets; `?wasm=mem64` smoke toggle; **15/15 PASS on both wasm32 and wasm64** |
| 4.5 — unsigned-pointer fix | `56272cb` | `>>> 0` in wasm32 paths of `malloc()` and `num()` — fixed `RangeError: offset is out of bounds` for any 7B+ wasm32 model (Emscripten linker omits unsigned coercion for custom-export `void*` returns; Phase 1's `_malloc → _bridge_malloc` switch inherited the bug) |
| 5 — bench parity gates | `49be54c` | **HALT.** Step 1 sanity gate (wasm32-vs-pinned) fails on 5/6 models (4-21% regression); Step 4 main gate (wasm64-vs-current-wasm32) passes 5/6, fails TinyLlama −5.5%. Net wasm64-vs-wasm32 median delta 0%. Wasm size +1.9%. Diagnosis: per-FFI helper-dispatch overhead dominates dispatch-heavy small-model decode. |
| 5.5 — Phase 1.5 inline optimization | `c919efa` | Helpers `big()`/`num()` removed; `is64` branch inlined at every FFI call site (38 methods); upload-loop branches hoisted. checkall + both binaries' smoke tests clean. **Perf claim deferred** — measurement under load avg 7.5+ produced 15% spread (70.6 → 80.9 across consecutive same-binary runs); gate cannot be adjudicated without a clean host. |

**Phase 5 re-bench queued.** Conditions required: load avg < 2.0,
< ~10 Chrome processes, dashboard ingest off. Re-run the canonical 6
sweep against `c919efa` and update PHASE-5-PARITY.md. Then proceed
to Phase 6 (deploy decision) or further investigation depending on
the gate outcome.

**Probe state — what's already established:**

- ✅ ASYNCIFY × MEMORY64 round-trip works (§31 closure;
  `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`). The single
  load-bearing risk axis is retired.
- ✅ BigInt ABI gap closed by `bridge_malloc` / `bridge_free` shims
  in `src/wasm/webgpu-bridge.cpp`; exports added in
  `src/wasm/CMakeLists.txt` (§31a; `SUMMARY-31a.md`).
- ✅ Configured cap = **15 GiB measured** at
  `MAXIMUM_MEMORY=16GB` (§31a Phase 3, 15 × 1 GiB allocations
  succeed; iter 15 returns NULL with overhead reserved).
- ✅ Toolchain cap = **16 GiB hard** — Emscripten 5.0.6 wasm-ld
  rejects `--max-memory > 17179869184` at link time (§31b).
  *Implication:* 30B IQ3_M at seq=2048 lands at ~14.8-15.8 GiB
  working set, which is the toolchain ceiling within margin of
  error. 8B and 13B have substantial headroom.
- ✅ `webllm-wasm-mem64.{js,wasm}` builds cleanly via
  `make mem64-probe`; the CMake conditional block already wires
  `-sMEMORY64=1 -sWASM_BIGINT=1 -sMAXIMUM_MEMORY=16GB`.

**Phasing skeleton** (mirrors §17/§18/§19/§20 phased structure;
detailed plan to be authored via `superpowers:writing-plans` as the
next step):

1. **Phase 0 — audit + scope.** Catalog every `m._malloc` /
   `m._free` call site in `src/inference/`, `src/wasm/`,
   `src/models/`, `smoke-test/`, and `eval/`. Audit
   `webgpu-bridge.cpp` for `int32_t size` / `int32_t offset`
   parameters that cap a single transfer at 2 GiB (spec §6 of
   §31a flagged `tensor_set_data` as one example). Map the
   GGUF streaming loader's JS↔WASM boundary points where byte
   offsets must remain BigInt under wasm64. Output: a punch list
   of touched call sites + signature changes; no code yet.
2. **Phase 1 — JS-side bridge migration.** Replace `_malloc` /
   `_free` with `_bridge_malloc` / `_bridge_free` at every TS
   call site identified in Phase 0. The bridge already returns
   BigInt under wasm64 and Number under wasm32, so each callsite
   needs a small marshaling change to accept either. Existing
   wasm32 build remains green throughout (`make checkall` per
   commit). Reversibility: trivial — bridge calls work under
   both binaries.
3. **Phase 2 — bridge ABI hardening.** Promote `int32_t` size /
   offset params in `webgpu-bridge.cpp` to `size_t` /
   `int64_t` where the transfer can exceed 2 GiB. TS bindings
   updated to pass BigInt. Single-file edit + matching CMake
   header signature update so the linker emits BigInt JS shims.
4. **Phase 3 — GGUF loader BigInt boundary.** Update the GGUF
   streaming loader so byte offsets and chunk sizes stay BigInt
   across the JS→WASM boundary. The `uploadRangeChunked` heap-
   grow detachment fix (already in tree) likely generalizes;
   verify no Number-narrowing slips.
5. **Phase 4 — production MEMORY64 build.** Wire
   `webllm-wasm-mem64` as a first-class target (it currently
   only powers the probe page). Update `make wasm-build` to
   produce both binaries; bundle copy logic in
   `Makefile`/`smoke-test/` updated.
6. **Phase 5 — bench parity gates.** Run `make smoke-bench` +
   `make bench-inference` + `make bench-profile` on the canonical
   6 fleet under the wasm64 binary. **Gate: zero regression
   ≥3% on tok/s for any of the 6 models.** If any model regresses,
   diagnose (likely pointer-overhead in hot paths) before
   proceeding. Pre-rebase baselines at
   `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`
   serve as the wasm32 reference.
7. **Phase 6 — single-vs-dual binary deployment.** Decide
   between (a) ship MEMORY64-only (drops the 4 GiB fast path;
   accepts ~5% pointer overhead across all targets per spec
   §3.1 estimate; halves bundle complexity) or (b) ship both
   `webllm-wasm.{js,wasm}` (wasm32) and `webllm-wasm-mem64.{js,wasm}`
   (wasm64) with deploy-time selection (3.5 → 7 MiB total
   payload; preserves wasm32 fast path for ≤4 GiB models).
   Decision criterion: if Phase 5 shows ≤2% wasm32 vs wasm64
   regression on the canonical 6, pick (a); otherwise (b).
8. **Phase 7 — register a >4 GiB validation target.** Pick a
   13B candidate (e.g. `mistral-13b-instruct-q4ks` ~7.4 GiB,
   or `llama-3.1-13b-iq3m` if available) to exercise the
   actual >4 GiB happy path. **Gate: forward pass coherent
   on a 36-prompt sanity eval; tok/s within architecturally
   expected band for 13B Q4_K_S (likely 18-22 tok/s
   extrapolating from 7B Q4_K_S 35.0 tok/s and 8B IQ3_M
   27.2 tok/s).**

**Out of scope (defer or skip):**

- **Lifting the Emscripten 16 GiB linker cap.** Custom wasm-ld
  patch is multi-day + ongoing maintenance; defer until upstream
  Emscripten lifts it (track on every Emscripten upgrade — see
  Watch list).
- **30B targets beyond seq=2048.** Working set lands at the
  16 GiB toolchain ceiling within margin of error; longer
  contexts require either lower-bit quants (IQ2_XXS / IQ2_S)
  or a wasm-ld bump. **Out of scope unless** a deployment ask
  forces it.
- **>30B targets.** Excluded by the 30B project ceiling
  (CLAUDE.md "Workflow policies"). Don't write infra for
  70B+; cite the ceiling and stop.

**Risk register:**

| Risk | Likelihood | Mitigation |
|---|---|---|
| WASM64 perf regression >3% on a canonical-6 model | Medium | Phase 5 gate catches; Phase 6 dual-binary fallback preserves wasm32 fast path. |
| BigInt-vs-Number TS callsite leak (silently truncates a 64-bit pointer) | Medium | Phase 0 punch list grounds Phase 1 migration; Phase 1 commits run `make checkall` per file. tsc strictness catches type drift. |
| Hidden `int32_t` size in bridge param (Phase 0 misses one) | Low | Phase 5 gate exercises 13B model with >2 GiB single-buffer transfers; a missed param shows up as truncation/garbled output, not silent perf loss. |
| GGUF loader BigInt boundary leak under heap-grow | Low | Phase 3 covers; pre-existing `uploadRangeChunked` fix is the precedent. |
| Bundle size doubles under dual-binary deploy | Low (cost-only) | Phase 6 decision criterion picks single-binary if perf delta ≤2%. |
| 13B target's coherence is broken by a quantization bug we haven't seen | Low | Phase 7 sanity eval catches; quantization correctness was tested at 7B/8B in §15/§16. |
| Toolchain ceiling tightens further on Emscripten upgrade | Low | Watch-list re-probe (cheap) catches at upgrade time; current 16 GiB is the absolute spec ceiling for `--max-memory`. |

**Gates per phase:**

- Phase 0: punch list reviewed; no implementation.
- Phase 1-4: `make checkall` clean per commit; existing
  wasm32 build remains green.
- Phase 5: zero ≥3% regression on canonical 6. Block on
  failure; diagnose before Phase 6.
- Phase 7: 13B target loads and runs a 36-prompt sanity eval
  coherently.

**Probe artifacts (canonical reference):**

- `eval/reports/memory64-probe-2026-04-28/SUMMARY.md` — §31
  parent probe (ASYNCIFY × MEMORY64 retired).
- `eval/reports/memory64-probe-2026-04-28/SUMMARY-31a.md` —
  §31a sub-probe (BigInt bridge + 15 GiB cap).
- `eval/reports/memory64-probe-2026-04-28/SUMMARY-31b.md` —
  §31b cap-bump probe (16 GiB toolchain ceiling).
- `docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md`
  — probe spec (the cap-probe series; **not** the full
  migration spec — that's the next step's writing-plans
  output).

**Next concrete action:** invoke `superpowers:writing-plans` to
author `docs/superpowers/plans/2026-MM-DD-memory64-full-migration.md`
(matching `2026-04-28-encoder-non-bert-arch.md` in shape — phases,
gates, commits per phase, success criteria). Per global
preference, execute via `superpowers:subagent-driven-development`
in this session once the plan is reviewed.

**Execution policy reminders** (from CLAUDE.md):

- 30B model-size ceiling — do not write infra for 70B+ unless
  the ceiling lifts.
- Probe-first default — Phase 0 audit *is* the probe; output is
  a punch list, not code.
- Always commit before work — each phase commit per established
  cadence (`feat(wasm): ...`, `refactor(bridge): ...`, etc.); do
  not bundle phases.
- Complexity ≠ implementation time — score phases on maintenance
  burden / surface area / reversibility, not duration.

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
→ 87.9 → 111.1 tok/s).** Lever isn't load-bearing for the
project's size-30B target ceiling — at large models the
absolute headroom is <1 tok/s — but is the only real
opportunity at sub-1B targets if a "tiny-model" deployment
appears. Captured as a finding rather than a next step.

---

### Bucket B follow-ups (post-closure, 2026-04-28) — CLOSED

Both queued follow-up items closed 2026-04-28.

11. ~~**Spec accuracy patch**~~ **DONE 2026-04-28.** Patched
    `docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`:
    added a top-level "Post-implementation corrections" note enumerating
    all four spec/reality mismatches with their llama.cpp truth-source
    line refs; updated §0 (jina FFN GeGLU; nomic RoPE NEOX), §1 Phase 0
    findings tables, §2 components table (`getRopeModeForArchitecture`
    row), §3 Point D (softmax mask leaf required + `-|i-j|` populate
    semantics), §3 Point F (per-arch gate activation: silu for nomic,
    gelu for jina), §4 Tokenizer (cls/mask → bos/eos fallback for
    nomic-style GGUFs), and §5 failure-diagnosis notes. The "(Open
    questions / decisions: None)" §7 block is unchanged — all four
    corrections are derived from already-shipped code, not from open
    decisions.

12. ~~**Vault-save bucket B doctrines**~~ **DONE 2026-04-28.** All
    four notes landed at `~/ClaudeVault/`:
    - `Patterns/encoder-parity-gate-via-sentence-transformers.md` —
      uv-pinned reference capture + agentchrome browser-side cosine
      ≥0.999 gate harness, reusable for any future encoder addition
      or cross-runtime numerical-parity probe.
    - `Patterns/llama-cpp-as-arch-truth-source.md` — authoritative
      file map (`src/models/<arch>.cpp`, `llama-graph.cpp`,
      `llama-model.cpp`) + 3 worked examples (jina GeGLU, nomic
      NEOX RoPE, ALiBi `-|i-j|` mask) from bucket B.
    - `Knowledge/encoder-cosine-degradation-signatures.md` —
      diagnostic ladder mapping cosine-curve shape to root cause:
      Signature A (monotonic length-degradation = positional bug),
      B (compressed-but-flat near-1.0 = activation/scaling bug),
      C (all-rows-uniform-low = tokenizer/input bug),
      D (single-row-spike = pooling/edge-case bug). Plus cheap
      localization tricks (layer-0 cosine, per-block bisect, op-
      count fingerprint, tokenizer diff).
    - `Debugging/jina-bert-v2-gguf-mirror-omits-alibi-key.md` —
      `gaianet/jina-embeddings-v2-base-en-GGUF` omits
      `attention.alibi_bias_max`; default is 8.0; loader fallback
      handles both mirror cases. Generalizes to any GGUF metadata
      key with a documented default (rope.freq_base, layer_norm_
      epsilon, pooling_type, cls/mask token IDs).

    Index rebuilt: 3089 notes / 986 tags / 9 MANIFESTs. All 4 notes
    verified in `Patterns/MANIFEST.md`, `Knowledge/MANIFEST.md`,
    `Debugging/MANIFEST.md`. Cross-links: the four notes
    cross-reference each other plus the existing
    `encoder-architecture-probe-saved-spec-rewrite` (Phase 0 probe)
    and the four bucket B session-specific debugging notes that
    captured the original incident timeline (`alibi-mask-fix-
    insufficient-for-{encoder,jina-v2}-parity`,
    `jina-bert-v2-encoder-parity-debugging`,
    `jina-vs-nomic-ffn-activation-mismatch`).

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
(deferred). Reuses the existing causal forward but requires:
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
- **Upstream cadence check.** Today: 2 commits, 0 in `ggml-webgpu/`.
  Procedure: `cd ~/Repos/llama.cpp && git fetch origin && git log
  webllm-browser-patches..origin/master --oneline -- ggml/src/
  ggml-webgpu/ ggml/include/`. **If the result is non-empty**, a
  rebase trigger has fired — apply the §32 procedure (rebase, sweep,
  classify per §27/§28/§32 templates). **If empty**, log and skip.
  Reasonable cadence: every 1-2 weeks of active development; ad-hoc
  otherwise.
- **Test skip count.** Currently 11 (all environmental — indexedDB
  unavailable in Bun native, WebGPU unavailable in Bun native,
  WordPiece HF fixtures not downloaded, side-branch spec-decode
  tests). Watch for changes; new skips might indicate accidental
  test regressions.
- **Encoder parity reference vectors freshness.** `eval/reports/
  encoder-parity-2026-04-28/{jina,nomic}-ref.json` are pinned to
  whatever sentence-transformers / HF model versions resolved on
  2026-04-28. Reproducible via `capture-refs.py` (uv-driven, the
  HF-side weights are content-addressed). **Re-capture if the
  parity gate ever fires a regression after a Phase 2 forward-graph
  change** to confirm the regression is local (not a reference
  drift). Otherwise leave pinned; the gate is a known-good fixture.

### External-trigger candidates

Three open candidates, all conditional:

- ~~**MEMORY64 full bridge migration**~~ → **promoted to Active next
  step 2026-04-28** (see "MEMORY64 full bridge migration" block
  above the External-trigger section).

- **§D concat-graph batched encoder compute.** Trigger: a real
  batch-encoder-throughput use-case (was non-goal in §21). The
  encoder is dispatch-bound; the only structural lever left is
  amortizing dispatches across a batched concat-graph. §27
  rebase didn't deliver an encoder-side free win.

- **Upstream `ggml-webgpu` rebase + free-win sweep.** Trigger:
  upstream actually moves on the `ggml-webgpu/` surface again.
  Running base is now `f9f33654a` (post-§32). Mechanical sweep
  on the 6-model fleet against the new baselines (tinyllama-q4_0
  ~107, qwen3-0.6b ~87, qwen3-1.7b ~61, mistral-7b-q4ks ~35,
  llama-3.1-8b-iq3m ~27, qwen3-8b-iq3m ~26). Three documented
  outcome templates: §27 (free-win), §28 (negative result, lever
  closed harder), §32 (small regression, accepted). Pick the
  template that matches the data and document.

### Deferred (out of scope per current ceilings)

- **§C-v2-A resurrection.** Side branch `feat/spec-decode-v2-greedy`
  retains the entire driver, AdaptiveGate, K+1 verify, contract gate,
  and ~30 tests. The only remaining theoretical resurrection path is
  a 70B+ target via MEMORY64 (target/drafter ratio 13× → ~100×), but
  the 30B project ceiling (set 2026-04-28) defers that. Tip
  `4e11d79`. **Do not merge.** Re-evaluate only if the 30B ceiling
  lifts.

- **Wave-1 architectures Gemma 2, Phi 3.** 5+ gaps for Gemma; mostly
  fused-QKV for Phi 3. Re-evaluate if a model in either family lands
  as a real deployment ask.

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
