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

### Phase 3 progress (2026-05-06)

**Stage 0 CLOSED 2026-05-06 — `b640d17` + llama.cpp on `webllm-browser-patches` (no patch — bridge change only).** Inverted device-hint via new `WEBLLM_PIN_TO_JSEP=1` build flag in `src/wasm/CMakeLists.txt` and the `webllm_load_model` block in `src/wasm/webgpu-bridge.cpp`. Spike at `?v=A-prime-stage0` confirmed JSEP hosts weights (455 MiB jsep_buf) + KV cache (11 MiB across 22 layers, all `dev = JSEP`). First scheduler abort: `SET_ROWS` on `cache_k_l0` view in `jsep_buf` — exactly as predicted. Op-ordering hypothesis validated. Closure: [`eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-0-PROBE.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-0-PROBE.md).

**Stage 1 CLOSED 2026-05-06 — `e60a39e` + llama.cpp `d8b80dee2`.** Added `dispatchSetRows` (TS, ~330 LOC) with two WGSL pipelines:
- F32→F16 atomic CAS per cell — required for the transposed V cache (`llama-kv-cache.cpp:1281`: `ggml_reshape_2d(v, 1, ggml_nelements(v))` puts ne[0]=1, adjacent indices share a u32 word).
- F32→F32 plain write.
Both paths support I64 + I32 indices. `supports_op` widened in companion llama.cpp commit. Spike at `?v=A-prime-stage1b` confirmed `sched_reserve` passes (798 nodes / 379 splits / 4.90 ms). Patch stack now +4. Closure: [`STAGE-1-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-1-RESULT.md).

**Stage 1.5 CLOSED 2026-05-06 — `ef5ccac` + llama.cpp `d0075e9a6`.** Two follow-ons:
- Phase 2 bug `dispatchRmsNorm` rewritten to **unary** signature (was nSrc=2 expecting input + weight; ggml `ggml_rms_norm` is unary, the per-channel weight multiply is a separate `GGML_OP_MUL` node). Test fixture updated.
- `supports_buft` narrowed from {jsep_buft, host buft, WebGPU buft} → **jsep_buft only**. Phase 2 Task 10's host-buft acceptance caused the scheduler to dispatch ops with CPU-resident sources to JSEP without inserting CPY-to-jsep_buft, leaving `graph_compute` to deref CPU heap pointers as JSEP handles. The Phase 2 offload_op flow (Outcome E synthetic probe) is now dormant by design — Option A-prime supersedes that path.

Stage 1.5 surfaced a deeper Phase 2 ABI bug: the descriptor's per-tensor "handle" slot (`jsep_tensor_handle(t) = t->data − GGML_JSEP_PTR_BASE`) is actually the **offset within the buffer**, not a buffer handle. The Phase 2 synthetic offload probe never tripped this because each test tensor got its own `ggml_jsep_alloc` (offset 0). Under Option A-prime with a real model loaded, ~6 big JSEP buffers each contain 100+ tensors at distinct offsets; the dispatchers' `dataManager.get(handle)` rightly throws `invalid handle 0`. Closure: [`STAGE-1.5-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-1.5-RESULT.md).

**Stage 3 PARTIALLY CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch).** Q4_K WGSL kernel landed in `src/inference/jsep/ops/matmul.ts` (~110 LOC) replacing the Stage-2 throw. Kernel verified correct in isolation via a hand-crafted single-super-block self-test in the spike harness (delta 4.5e-6 vs CPU reference dequant). 805 Q4_K matmul dispatches per 5 decode tokens + 1 prefill (~134/pass, matches TinyLlama's 22 layers × 6 q4_K matmuls). **But:** all 32000 logits in step 0 are exactly 0.0 (no NaN, no Inf, all finite, min=max=0) → **Outcome C**. The all-zero collapse is upstream of the new kernel — possible loci: CPY ordering between JSEP↔CPU splits, RMS_NORM kernel bug on real-model shapes, GET_ROWS / MUL on CPU, or scheduler not invoking `synchronize` between splits. Stage 3.5 (queued) localizes via RMS_NORM self-test, first-model-matmul dst capture, and first-CPU→JSEP-write byte dump. Side improvements that landed: `jsepRead` / `jsepWrite` / `jsepClear` now flush the encoder batcher before issuing host-roundtrip queue ops (correctness fix for FIFO ordering — does not cure Outcome C but removes a latent race); `tests/jsep-matmul-golden.test.ts` got a Q4_K golden case (skips on Bun, structural reference); `src/index-jsep.ts` re-exports `dispatchMatmul` and a few JSEP enums for spike-harness use. Closure: [`STAGE-3-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-3-RESULT.md). Patch stack: 6 (unchanged); webllm: +1 commit.

**Stage 2 CLOSED 2026-05-06 — `9406496` + llama.cpp `53c66649f`.** Bumped `GGML_JSEP_TENSOR_BLOCK_I32` 18→19 and split the conflated slot into `(buf_handle, offset)`. JS-side `JsepTensorMeta` now exposes both fields; `dispatchMatmul` / `dispatchRmsNorm` / `dispatchSetRows` bind via `{buffer, offset, size: rec.size - offset}` using the buffer handle as the dataManager key. Buffer handle source: `tensor->buffer->context->handle` (safe post-Stage-1.5 since `supports_buft = jsep_buft only`). Spike at `?v=A-prime-stage2` progressed past the "invalid handle 0" wall: model loads end-to-end into JSEP (455 MiB jsep_buf weights + 11 MiB KV across 22 layers, all `dev = JSEP`), `sched_reserve` passes (798 nodes / 379 splits / 5.00 ms), then decode failed at the next missing kernel — **Q4_K matmul** (`matmul.ts:316`: `"matmul Q4_K kernel: deferred to Task 7"`). **Outcome B** per the original Stage 2 outcome table. Closure: [`STAGE-2-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-2-RESULT.md). Patch stack +5 → +6.

**Stage 3.5 CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch).** Root cause localized: WebGPU compute pass synchronization-scope rule. The libllama scheduler packs matmul activation `src1` and matmul output `dst` into a single `jsep_buf` at different byte offsets; WebGPU validates bind-group buffer usage at *buffer granularity* (not sub-range), so binding the same `GPUBuffer` as both read-only-storage and read-write-storage fails encoder.finish() with `"usage (Storage(read-write)|Storage(read-only)) includes writable usage and another usage in the same synchronization scope"`. The dispatch was silently rejected; dst stayed at zero; forward pass collapsed.

**Stage 4 partial — divert pattern landed for matmul + RMS_NORM.** When `dst.bufHandle` aliases any src `bufHandle`, allocate a fresh temp `GPUBuffer`, dispatch into it, then `copyBufferToBuffer` back to `dstRec.buffer` at `dst.offset`. The diverted dispatch lives in its own command-encoder (flush the batcher first) so it can't conflict with batched neighbours. Verified post-fix: 1068/1068 model matmuls divert without validation errors; 270/271 RMS_NORM dispatches divert. **But Outcome A "Paris" decode not yet achieved** — matmul `src1` (the activation feeding attn_q) is still corrupt with the same byte pattern as pre-fix (denormals + 1e+18-scale floats — uninitialized memory pattern). The matmul + RMS_NORM divert fixed those kernels but the **upstream producer** of `src1` (likely SET_ROWS for KV cache writes — `dst === view(src[2])` is definitionally aliased) is still failing silently and leaving the buffer untouched. **Stage 4 is incomplete; SET_ROWS divert (with read-modify-write semantics for partial updates) is required to flip Outcome.** Closure: [`STAGE-3.5-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-3.5-RESULT.md). Per-token decode 24-25 ms (Stage 3 baseline 23 ms; +8% from divert overhead — within noise). Patch stack: 6 (unchanged). RMS_NORM real-shape self-test (cols=2048) added to spike harness — permanent regression check.

**Stage 4.1 CLOSED 2026-05-06 — `0161595` (no llama.cpp patch).** SET_ROWS divert with read-modify-write semantics landed in `src/inference/jsep/ops/set-rows.ts` (~80 LOC). Aliasing rate measurement confirmed the brief's hypothesis exactly: `SETROWS_STATS = {total:264, aliasesSrc0:0, aliasesSrc1:0, aliasesSrc2:264}` — 100% structural alias with src[2] (the destination buffer that dst is a view of, per ggml SET_ROWS semantics). Divert fires for every SET_ROWS call (`SETROWS_DIVERT_FIRES = 264`). **But Outcome A "Paris" decode not achieved** — `LOGIT_STATS_STEP0` still all-zero, `GENERATED_TOKENS = [0,0,0,0,0]`. Per-token decode 23.74 ms vs Stage-3.5 baseline 24.30 ms (within noise — divert overhead invisible). This is **exit criterion (b)** from the Stage 4.1 brief: SET_ROWS aliasing was a real latent bug worth fixing structurally, but it's *not* the load-bearing cause of the Outcome C all-zero collapse. Closure: [`STAGE-4.1-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.1-RESULT.md). Patch stack: 6 (unchanged). Next suspect: CPU-side writeback (`jsepWrite`) for unsupported ops — Stage 4.2 brief below.

**Stage 4.2 CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch; no `src/` changes — diagnostic-only stage).** jsepWrite/jsepRead/jsepRunOp wrappers + pre-/post-prefill GPU buffer dumps + uncapturederror listener landed in `smoke-test/p2-v2-spike.src.ts`. Per-token decode 24.34 ms (Stage-4.1 baseline 23.74 ms; +0.6 ms = within noise; the diagnostic adds first-30-call wraps, no perf impact during decode steady state). Headline findings:
- **Buf 11 starts at all-zeros** post-load (`PREPREFILL_BUF11 = {0:[0,…], 4194304:[0,…], …}`) — the post-prefill canonical NaN (`0x7fc00000`) is *computed*, not stale memory. Stage 3.5's "uninitialized memory pattern" framing was off — the corruption is from a JSEP shader producing NaN, then CPU faithfully copying it through `jsepRead`/`jsepWrite`.
- **GPU_ERR_COUNT = 0** — every dispatch passes WebGPU validation. The Stage-3.5 silent-rejection failure mode is genuinely fixed by the divert pattern.
- **All 30 captured runOps hit divert path** (matmul/RMS_NORM/SET_ROWS, all aliasing buf 11 with src1 or src2). The lm_head (likely the only non-divert in the graph) is past the RUN_MAX=30 capture window.
- **Final logits = exactly zero**, not NaN — strongly suggesting **lm_head's dispatch silently doesn't write to its dst buffer**, leaving it at the post-allocation zero state. This is a *separate* bug from the NaN-cascade in buf 11.

Two distinct downstream bugs surfaced (per closure §"Diagnosis"):
- **Bug A** — JSEP-supported ops compute canonical NaN starting somewhere in the chain. First runOp (RMS_NORM dst=[11+0] src0=[11+0]) operates on a known-valid embedding (jsepRead i=0 retrieves it cleanly). By post-prefill, offset 0 reads NaN. Either RMS_NORM at production shape (rows=6, cols=2048 — untested; selftest covers rows=1) or MUL_MAT at K=2048 (selftest covers K=256) produces NaN.
- **Bug B** — lm_head non-divert dispatch silently doesn't land. Logits stay at the buffer's zero-init state.

Closure: [`STAGE-4.2-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.2-RESULT.md). Patch stack: 6 (unchanged). Stage 4.3 brief below splits into 4.3a (production-shape kernel selftests) and 4.3b (full-graph runOp capture + per-op readback).

**Stage 4.4 CLOSED 2026-05-06 — `<pending>` + llama.cpp `<pending>` (P7 — F1 dual-resident host mirror in ggml-jsep.cpp; patch stack 6 → 7).** F1 implemented as designed: `ggml_backend_jsep_buffer_context` gains `void * host_mirror`; `alloc_buffer` allocates + zero-inits a parallel host-side mirror; `set_tensor` / `memset_tensor` / `clear` apply the operation to BOTH the host mirror AND the GPU buffer; `get_tensor` reads from the mirror only (drops the JS round-trip — `COUNTER_DELTAS.read` 1266 → **0**); **`get_base` returns `host_mirror` instead of the `0x2000` sentinel** (the load-bearing change so CPU-fallback ops dereferencing `tensor->data` land in real host RAM); `jsep_tensor_handle` updated to subtract `host_mirror` (offset value invariant). **PARTIAL OUTCOME A — Bug A FIXED.** `FIRST_NAN_DST_PROBE = null` (was first NaN at i=1), `LOGIT_STATS_STEP0.first8` = `[0.0060, 0.0047, -0.0102, 0.0138, -0.0149, 0.0099, -0.0029, -0.0056]` (was all-zero), `topId/topVal = 593/0.159`, `GENERATED_TOKENS = [593, 5871, 945, 16976, 25487]` (was `[0, 0, 0, 0, 0]` — five distinct non-zero ids), `POSTPREFILL_BUF11` carries real f32 at most offsets (was canonical NaN everywhere). The CPU-fallback per-channel RMSNorm gain (Stage 4.3's smoking-gun op between seq 2 and seq 3) now reads real attention-norm weights, killing the NaN cascade through every downstream op. All four kernel selftests still PASS. `make checkall` green. Per-token decode 23.22 ms (within noise of Stage-4.3 baseline 23.92 ms); F1 dual-write only impacts model-load wall time (134 weight uploads). **But:** decoded text = `"ntiuracinateenes"`, not `"Paris"` — partial flip. **Bug C surfaced (follow-on):** GPU→host writeback gap. JSEP ops write to the GPU buffer; the host mirror stays stale; downstream CPU-fallback ops dereference `tensor->data` (now points into mirror) and read the initial-zero contents, never updated by the GPU. Smoking-gun: `FIRST_ALLZERO_DST_PROBE = {i:3, op:42, dstH:18}` (op 42 = `GGML_OP_SET_ROWS`; handle 18 = KV cache); `COUNTER_DELTAS.read = 0` confirms the scheduler isn't inserting `get_tensor` calls to bridge JSEP→host (because `tensor->data` *is* a valid host pointer post-F1 — just not a *current* one). This is exactly the "cross-backend writes" caveat the Stage 4.4 brief footnoted, in the GPU→host direction (the brief flagged the host→GPU direction; the actual failure mode is the inverse). Closure: [`STAGE-4.4-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.4-RESULT.md). Stage 4.5 brief below queues the writeback fix (H1 unconditional / H2 cpy_tensor / H3 graph-walk pre-pass).

**Stage 4.28 CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (P10 amend — extends Stage 4.20's set_tensor weight-hash allowlist from 2 names to 7; patch stack unchanged at 13).** **Outcome P-15-jsep-bypass CONFIRMED — 5 of 7 layer-0 weights byte-clean end-to-end through JSEP set_tensor → device.queue.writeBuffer; 2 weights (`blk.0.ffn_norm.weight` F32 8 KiB, `blk.0.ffn_down.weight` Q6_K 9.4 MiB) bypass the JSEP `set_tensor` hook entirely (live on a non-JSEP buft, almost certainly CPU/host buft).** Probe 15 extends the C++ allowlist in `ggml_backend_jsep_buffer_set_tensor` (5 new `std::strcmp` clauses for `attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`) and the JS-side `targetNames` + `elemBytes` (added Q6_K = 210/256). **Closes Suspect 1 from Stage 4.27**: `blk.0.attn_output.weight` ref `0xaae061b5` == set_tensor pre-upload `0xaae061b5` == GPU readback `0xaae061b5` (size 2,359,296 bytes) — output-projection weight upload byte-integrity is bit-clean. Same all-pass for `attn_q.weight` (re-confirmed `0xf2f7188c`), `attn_k.weight` (re-confirmed `0x9399f36a`), `ffn_gate.weight` (`0xafdfc33a`, 6.18 MiB Q4_0), `ffn_up.weight` (`0x76f44e42`, 6.18 MiB Q4_0). **Suspect 2 (`ffn_norm.weight` gain-vector mis-load) cannot be tested by the JSEP `set_tensor` hook** — the probe captures zero hits for that tensor name, meaning libllama allocated it to a non-JSEP backend buffer (consistent with Stage 4.27's smoking-gun table showing `ffn_norm-0` running on `backend=CPU`). The brief's risk-register #1 anticipated this exact case; the fallback is a CPU-side `set_tensor` hook OR a `cb_eval` weight-tap via no-op `ggml_view + ggml_dup` schedule. `blk.0.ffn_down.weight` (Q6_K, consistent with Stage 4.22's surprise finding that TinyLlama-1.1b-chat-q4_0.gguf is actually Q4_K projections + Q6_K embeddings) also bypasses JSEP `set_tensor` — its scheduling-allocation routes Q6_K weights to a different backend at module-load time. `GENERATED_TEXT = "inonic boso-"` (unchanged — bug still active, framing now refined to: byte-integrity gap is on the *CPU-buft* side of the load path, not the JSEP side). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Per-token decode 1287 ms (elevated because the spike still carries Probes 13/14 from Stages 4.25/4.26 — none on the production decode path). Files touched: `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` (allowlist 2→7), `smoke-test/p2-v2-spike.src.ts` (`targetNames` 2→7; `elemBytes` adds Q6_K = 210/256; new `[STAGE-4.28]` synthesis block emitting `P-15-{clean,gain,output-proj,ffn,jsep-bypass}` outcome line). Closure: [`STAGE-4.28-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.28-RESULT.md). Raw artifact: [`STAGE-4.28-spike-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.28-spike-output.txt). Stage 4.29 brief below queues Probe 16: CPU-side `set_tensor` hook mirroring the JSEP probe's pattern, gated on the same 7-name allowlist, pushing to `globalThis.__cpuWeightHashLog` so the JS spike can synthesize a unified P-15-* verdict that covers all 7 weights. Closes suspect 2-a (CPU-buft `ffn_norm.weight` byte-integrity) in one probe; if clean, pivots Stage 4.30 to suspect 3 (first8-window blindness on `kqv_out-0`).

**Stage 4.26 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch — `webllm_q4k_q8k_matmul` shim landed in webllm's own `src/wasm/webgpu-bridge.cpp`, mirroring Stage 4.24's `webllm_dequantize_q4_K` pattern; patch stack unchanged at 13).** **Outcome H-4-libllama-imprecise CONFIRMED — `llamaVsF64Max = 4.178e-2`, four orders of magnitude larger than `wgslVsF64Max = 7.94e-6`. libllama is the imprecise side of the cross-module disagreement by a wide margin; webllm's WGSL Q4_K matmul kernel is _more accurate than libllama's CPU reference_ on the same captured Q-projection inputs. The matmul-precision investigation is now closed.** Probe 14 added `webllm_q4k_q8k_matmul(src0_q4k, src1_f32, dst_f32, M, K, N)` C export wrapping `ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float` (= `quantize_row_q8_K` per src1 row) + `ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot` (= `vec_dot_q4_K_q8_K` per output element). The wasm32 build resolves both to the `_generic` (scalar) variants under `GGML_CPU_GENERIC` — same code path libllama would take in any wasm32 deployment. The spike harness mallocs src0/src1/dst, copies the captured `cap.src0Bytes` / `cap.src1Bytes` from Probe 10 onto the WASM heap, calls the shim, and scores libllama's output against an f64 reference computed in JS over the same `src0Dequant` / `src1View` (no `Math.fround`; pure double accumulation). It also re-scores webllm's WGSL captured `cap.dstAfterBytes` against the same f64 oracle for an apples-to-apples comparison. Headline numbers: `llamaVsF64Max = 4.178e-2` at idx 11567, `wgslVsF64Max = 7.94e-6`, `llamaVsWgslMax = 4.178e-2` at the same idx. The 4.178e-2 envelope is dominated by libllama's per-element src1 Q8_K quantization loss (~1/127 ≈ 7.9e-3 per element, ~1.3e-1 envelope after a 2048-K dot product); WGSL doesn't quantize src1, it consumes the raw f32 activations directly, hence its tighter f64 floor. The historical 5.24e-4 first8 disagreement (Stage 4.17 idx 0–2 `Qcur-0`) is consistent with this: 5.24e-4 was the first8 sample, 4.178e-2 is the worst-element across all 12,288 outputs; the per-element distribution sits between the two. **Implication:** The Q-projection matmul is _not_ the bug source for "inonic boso-" — webllm is more accurate than the reference path that decodes correctly. The cascade producing "inonic boso-" must come from a different op in the prefill or decode path. `GENERATED_TEXT = "inonic boso-"` (unchanged — bug still active, framing now decoupled from matmul precision). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Per-token decode 360 ms (elevated vs 23 ms baseline due to one-shot CPU matmul + heap-state in the same prefill cycle; not on the production decode path). Files touched: `src/wasm/webgpu-bridge.cpp` (+ `webllm_q4k_q8k_matmul` shim adjacent to Stage 4.24's dequant shim), `src/wasm/CMakeLists.txt` (+ `_webllm_q4k_q8k_matmul` to EXPORTED_FUNCTIONS; intentionally NOT in JSPI_EXPORTS), `smoke-test/p2-v2-spike.src.ts` (+ Probe 14 sub-block in post-Probe-13 try, gated on `cap.src0Type === GGML_TYPE_Q4_K`; emits `PROBE14_LLAMA_MATMUL_VS_F64` JSON + `[probe14]` verdict line). Closure: [`STAGE-4.26-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.26-RESULT.md). Raw artifact: [`STAGE-4.26-spike-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.26-spike-output.txt). Stage 4.27 brief below pivots to the existing `__stage417Checkpoints` per-layer diff framework — re-run the JSEP spike vs the non-JSEP reference probe on the current code, identify the first checkpoint where the two diverge by a magnitude inconsistent with kernel precision artifacts (≥1e-2 absolute on small-magnitude tensors), and characterize that op as the cascade source.

**Stage 4.25 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch — Kahan path lives entirely in `src/inference/jsep/ops/matmul.ts`; patch stack unchanged at 13).** **Outcome H-3b-structural CONFIRMED — Kahan-corrected accumulation in the WGSL Q4_K MUL_MAT kernel produces bit-identical output to the non-Kahan baseline (`kahanVsBaselineMax = 0` exact, all 8 first-output positions match), and the existing `MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64 = 7.94e-6` proves the WGSL kernel is already accurate to within ~8e-6 of f64 truth — 67× smaller than the 5.24e-4 cross-module disagreement. f32 accumulation precision is ruled out as the dominant error source.** Probe 13 added a `kahan = false` parameter to `buildMatmulShader` (Q4_K branch only — the only quant type the Q-projection codepath uses at production scale on TinyLlama), splicing a Neumaier-Kahan compensated accumulator into the K-loop when `kahan == true`; gated by a one-shot `globalThis.__stage425KahanArm` flag in `dispatchMatmul` matched on `(M=2048, K=2048, N=6, src0=GGML_TYPE_Q4_K)` so only the layer-0 `Qcur-0` dispatch takes the variant kernel. Variant lives under a separate pipeline cache key (`mat-q4_k-f32-f32-2-kahan`) so the production pipeline cache is unchanged for the other ~1935 Q4_K MUL_MAT dispatches in prefill. A confirmation flag (`__stage425KahanFired = true`) was added to disambiguate "Kahan ran, output unchanged" from "gate never fired" — `kahanFired = true` confirmed the gate engaged. Two non-exclusive explanations for the bit-identical output: **(1)** WGSL compiler elision of the compensation math (Naga / Tint can apply algebraic simplification across `(acc + term) - acc - term = 0`; WGSL has no `volatile` / `FP_CONTRACT off`); **(2)** compensation magnitude below ULP at the final `acc + compensation` step (per-add lost-low ~ULP(0.045)≈5.4e-9, summed corrections potentially below ULP at the end). Distinguishing (1) from (2) requires Naga IR disassembly and is disproportionate to the finding — the **structural conclusion is robust either way**: the pre-existing `maxAbsDeltaVsF64 = 7.94e-6` already proves f32 accumulation precision can close at most 1.5% of the 5.24e-4 gap. The remaining 99% must come from libllama-side imprecision, a different src1 input upstream, or a fused-dequant/multiply boundary inside `vec_dot_q4_K_q8_K`. `GENERATED_TEXT = "inonic boso-"` (unchanged — bug still active, framing now structural). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Files touched: `src/inference/jsep/ops/matmul.ts` (`buildMatmulShader` and `buildPipeline` gain optional `kahan` parameter; Q4_K branch wraps in block scope; new dispatch-time gate on `__stage425KahanArm`); `smoke-test/p2-v2-spike.src.ts` (arms `__stage425KahanArm` adjacent to `__probe10Capture`; new `MATMUL_PROBE13_DELTA` JSON + `[probe13]` verdict line; reads `__stage425KahanFired` for explicit confirmation). Closure: [`STAGE-4.25-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.25-RESULT.md). Raw artifact: [`STAGE-4.25-spike-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.25-spike-output.txt). Stage 4.26 brief below queues Probe 14: a `webllm_q4k_q8k_matmul` shim (mirror of Stage 4.24's `webllm_dequantize_q4_K` pattern) that runs libllama's CPU `vec_dot_q4_K_q8_K` on the captured production inputs and computes `llamaVsF64Max`; verdict H-4-libllama-{imprecise|precise|mid} based on whether libllama is the imprecise side (≥1e-4), agrees with f64 truth (≤1e-5), or contributes partially.

**Stage 4.24 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch — `webllm_dequantize_q4_K` shim landed in webllm's own `src/wasm/webgpu-bridge.cpp`, not in the `webllm-browser-patches` series; patch stack unchanged at 13).** **Outcome H-3b CONFIRMED — Q4_K dequant is bit-clean; the 5.24e-4 production Qcur-0 delta is f32 matmul accumulation-order disagreement.** Probe 12 added a `webllm_dequantize_q4_K(src, dst, k)` C export wrapping `ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float` (= `dequantize_row_q4_K` in `ggml-quants.c`), and a post-prefill block in `smoke-test/p2-v2-spike.src.ts` that takes Stage 4.22's `__probe10Capture.result.src0Bytes` (the 2,359,296-byte Q4_K tile for `blk.0.attn_q.weight`) and dequantizes it via two paths: (A) `dequantQ4_KTile` (existing JS port of WGSL `load_q4_K`), (B) `mod._webllm_dequantize_q4_K` (libllama). Element-wise diff over **4,194,304** outputs: `maxAbsDelta = 0` (exact, not single-ULP), `nNaN = 0`, `nInf = 0`, first-8 outputs byte-identical (`PROBE12_DEQUANT_DELTA = {"M":2048,"K":2048,"totalElems":4194304,"maxAbsDelta":0,"maxIdx":-1,"verdict":"H-3b"}`). The WGSL kernel's dequant logic is provably correct against libllama's reference — Stage 4.22's self-consistency check verified the kernel against its own dequant; Probe 12 closes the gap by verifying that JS port against libllama directly. **Hypothesis split for Stage 4.25:** the remaining variable is f32 matmul accumulation order. WGSL kernel reduces 2048-K partial sums via subgroup tree + workgroup horizontal add (4 OUTPUTS_PER_WG × 16-wide subgroup); libllama reduces via SIMD lane-pair adds + horizontal sum (`vec_dot_q4_K_q8_K`'s AVX2 `_mm256_hadd_ps` / NEON `vaddvq_f32` / scalar fallback). f32 reductions of length 2048 with O(0.1) operands disagree on their last 12-13 mantissa bits ≈ O(1e-4) — fits the 5.24e-4 envelope. Stage 4.18's 4.77e-7 WGSL-vs-f32-loop delta confirms the WGSL kernel matches a *chosen* k-major f32 reference, not libllama's reduction order. `GENERATED_TEXT = "inonic boso-"` (unchanged — bug still active, framing localized to matmul accumulation). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Per-token decode 311.60 ms (within noise of Stage 4.22's 879.7 ms run-with-sweep envelope; this run reuses Stage 4.23's spike state without the q4_0 sweep enabled). Files touched: `src/wasm/webgpu-bridge.cpp` (+`webllm_dequantize_q4_K` shim), `src/wasm/CMakeLists.txt` (+ `_webllm_dequantize_q4_K` in EXPORTED_FUNCTIONS; intentionally NOT in JSPI_EXPORTS — synchronous CPU dequant doesn't need promising-wrap overhead), `smoke-test/p2-v2-spike.src.ts` (Probe 12 block in post-Probe-10 try, gated on `cap.src0Type === GGML_TYPE_Q4_K`). Closure: [`STAGE-4.24-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.24-RESULT.md). Raw artifact: [`STAGE-4.24-spike-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.24-spike-output.txt). Stage 4.25 brief below queues a Kahan-summed WGSL accumulator probe gated to `Qcur-0` only — measures whether the 5.24e-4 collapses to ≤1e-5 with f32-Kahan accumulation (H-3b-Kahan ship target) or stays put (structural — Branch C: downstream cascade mitigation).

**Stage 4.23 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; commits the previously-untracked `smoke-test/p2-v2-ref-probe.{html,src.ts,js}` ref-capture harness from Stage 4.17 Probe 7; patch stack unchanged at 13).** **Outcome H-3 — Probe 11 hypothesis (host-mirror writeback gap) misframed; the 5.24e-4 first8 Qcur-0 delta originates from the WGSL Q4_K matmul kernel disagreeing with libllama's CPU Q4_K matmul on the same Q-projection inputs.** Side-by-side diff of the spike's `__stage417Checkpoints` (JSEP build, patch stack 13) and the non-JSEP ref-probe's `__refCheckpoints` (`webllm-wasm.js`, same llama.cpp tip — WebGPU compiled but doesn't engage on TinyLlama's per-token shapes, so Q-proj falls back to libllama's CPU Q4_K dequant + GEMM) for the first 12 nodes of layer-0 prefill: idx=0 `attn_norm-0` (CPU on both) Δ=1e-7 (input bit-clean — rules out src1 staleness); idx=1 `Qcur-0` (jsep_buf vs CPU) Δ=**5.242e-4** (the historical number, exactly reproduced); idx=4 `Vcur-0` (CPU on both — V-proj falls to CPU on the spike too, per Stage 4.18 Probe 8b) Δ=**1e-9** (conclusive: when the spike takes the same code path as the reference, output agrees to numerical precision); idx=6 `Kcur-0` (jsep_buf vs CPU) Δ=**3.376e-4** (corroborates: WGSL Q4_K matmul produces the same scale of disagreement on a different but adjacent Q4_K dispatch); idx=9 `kq-0` Δ=1.19e-2 (Q@K^T amplifies upstream Q+K disagreement). **Why Stage 4.22's writeback-gap framing missed this:** the historical 5.24e-4 traces back to Stage 4.17's 96-checkpoint diff between **two separate WASM modules** (JSEP spike vs non-JSEP ref-probe), not a within-spike host_mirror comparison; Stage 4.22's f32 self-consistency check matched the WGSL kernel against a JS port of its own `load_q4_K` dequant (`dequantQ4_KTile`), so it verified the kernel against itself, not against libllama's `dequantize_row_q4_K`. **Hypothesis split for Stage 4.24:** H-3a (likely) — WGSL `load_q4_K` reconstructs Q4_K super-block scales/mins differently from libllama; H-3b (less likely) — both kernels reconstruct identically but accumulate the 2048-K matmul partial sums in different orders, accumulating ~K × 1e-7 = 2e-4 of f32 rounding. H-3a is the priority and easier to disprove (dequant cross-check on the captured `__probe10Capture.result.src0Bytes` against an EM_ASYNC_JS shim into `ggml_dequantize_row_q4_K`). `GENERATED_TEXT = "inonic boso-"` (unchanged from Stage 4.22 — bug still active, framing now correct). All 6 spike selftests + 5 sweep selftests still PASS at the Stage 4.22 tip. `make checkall` not re-run (no source-code change to library or kernel; only commits the ref-probe files). Closure: [`STAGE-4.23-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.23-RESULT.md). Stage 4.24 brief below queues Probe 12: capture `__probe10Capture.result.src0Bytes`, dequant via `dequantQ4_KTile` (WGSL-equivalent) and via libllama's `ggml_dequantize_row_q4_K` (EM_ASYNC_JS shim into the existing WASM module), diff element-wise; `maxAbsDelta > 1e-5` ⇒ H-3a confirmed (fix WGSL dequant); `≤ 1e-5` ⇒ H-3b (investigate matmul accumulation order).

**Stage 4.22 CLOSED 2026-05-07 — `82147e8` (no llama.cpp patch — pure JS-side spike + matmul.ts probe gate; patch stack 13 unchanged).** **Outcome G-2 CONFIRMED — kernel bit-clean on production inputs.** Probe 10 captured the actual src0 / src1 / dst-after bytes the kernel saw at the first production JSEP MUL_MAT dispatch in TinyLlama prefill (layer-0 Q-projection, `Qcur-0` ne=[2048,6,1,1]) via pre-encoder + kernel-encoder + post-encoder mapAsync staging buffers, then replayed those captured bytes through the same `dispatchMatmul` entry point as a one-off synthetic call. Both the captured production output and the synthetic replay match an f32 element-wise k-major CPU reference to within **4.768e-7** (single ULP at `outputMaxAbs=6.37`); first-8 outputs are bit-identical between captured and synthetic. The dispatch / kernel-execution boundary is exonerated — pipeline cache collisions, bind-group offset mismatches, workgroup count off-by-ones, src0/src1 swaps are mathematically excluded by the bit-identical first-8 outputs. **Surprise finding** — TinyLlama-1.1b-chat-q4_0.gguf actually contains Q4_K projections + Q6_K embeddings (`token_embd.weight` t=12 Q4_K; `blk.0.attn_q.weight` t=12 Q4_K; `blk.0.attn_v.weight` t=14 Q6_K; `output.weight` t=14 Q6_K). The "Q4_0" in the filename is the HuggingFace quant tier label, not the on-disk tensor type — **Stage 4.18's "Q4_0 production-shape sweep" was therefore measuring a different code path from production**. The 312× delta gap that motivated Stage 4.22 was an apples-vs-oranges comparison all along. (Footnote: Stage 4.22's f32 reference was JS-side dequant of captured Q4_K bytes via `dequantQ4_KTile` — a port of WGSL `load_q4_K` — so the kernel was verified against its own dequant logic, not against libllama's; Stage 4.23 closes that gap.) All 6 spike selftests still PASS, all 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Files touched: `src/inference/jsep/ops/matmul.ts` (Probe 10 capture branch in divert path; gated on Q4_0 OR Q4_K src0; auto-disarms after first fire); `smoke-test/p2-v2-spike.src.ts` (`dequantQ4_KTile` port of WGSL `load_q4_K`; generalized `runMatmulQ4_0FromBytes` → `runMatmulFromBytes(src0Type, ...)` covering both Q4_0 and Q4_K; `compareF32Buffers` helper; post-prefill probe10 block). Closure: [`STAGE-4.22-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.22-RESULT.md). Stage 4.23 above re-derived the 5.24e-4 number's provenance and reframed Stage 4.22's writeback-gap follow-on hypothesis (Outcome H-3).

**Stage 4.21 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; pure JS-side spike extension; patch stack unchanged at 13).** **Outcome F-1 CONFIRMED — GPU bytes match `set_tensor`'s pre-upload hash. The entire weight-upload chain (GGUF → set_tensor → `Module.jsepWrite` → `device.queue.writeBuffer` → JSEP `GPUBuffer`) is bit-clean end-to-end.** Probe 9c walks `globalThis.__weightHashLog` after `bridge.loadModel` and, for each entry, encodes `copyBufferToBuffer(bufRec.buffer, entry.offset, staging, 0, entry.size)` + `device.queue.submit` + `await staging.mapAsync(GPUMapMode.READ, 0, entry.size)`, then FNV-1a-32 hashes the staging bytes and compares to `entry.fnv1a_pre`. Results: `blk.0.attn_q.weight` GPU `0xf2f7188c` == set_tensor pre-upload `0xf2f7188c` (size 2,359,296); `blk.0.attn_k.weight` GPU `0x9399f36a` == set_tensor pre-upload `0x9399f36a` (size 294,912). The JSEP weight buffers (handles 36-39 in `LIVE_BUFFERS`, 4×128 MiB) are allocated with `STORAGE | COPY_SRC | COPY_DST` already; no flag widening needed. **Implication:** Stage 4.18's three sub-hypotheses now resolve cleanly: (U-A) attn_norm-0 — refuted by Stage 4.19 (bit-identical); (U-B) Q4_0 weight upload — **refuted end-to-end** by Stage 4.20 + 4.21 combined (every link of the chain is bit-clean from GGUF file all the way to GPU memory); (U-C) WGSL pipeline cache collision / kernel under production conditions — now the only remaining suspect. The 5.24e-4 production Qcur-0 delta vs 1.68e-6 synthetic-sweep delta (312× gap) must originate inside the dispatch / kernel-execution boundary at production conditions in a way the Stage 4.18 standalone sweep didn't reproduce. `GENERATED_TEXT = "inonic boso-"` (gibberish, unchanged — bug still active, just localized). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Probe runs only at model-load time (two ~MiB GPU readbacks); decode-loop perf unaffected. Closure: [`STAGE-4.21-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.21-RESULT.md). Raw artifact: [`STAGE-4.21-spike-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.21-spike-output.txt). Stage 4.22 brief below queues kernel-input capture: a one-shot `copyBufferToBuffer` interception inside the JSEP MUL_MAT dispatch path, fed back into the standalone Stage-4.18 synthetic harness with the captured production src0+src1 bytes. Yes ⇒ Stage 4.18 sweep missed an output-tile boundary case; no ⇒ bug between dispatch site and shader execution (pipeline cache / bind-group / workgroup-count / src0-src1 swap).

**Stage 4.20 CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (P10 — `g_weight_hash_probe` flag + `ggml_jsep_set_weight_hash_probe(int)` toggle export + FNV-1a-32 hash hook in `ggml_backend_jsep_buffer_set_tensor`; patch stack 12 → 13). webllm-side: `_ggml_jsep_set_weight_hash_probe` JSEP-only export gating in `src/wasm/CMakeLists.txt`, plus `GgufParser`-based JS-side reference hash + verdict in `smoke-test/p2-v2-spike.src.ts`.** **Outcome F CONFIRMED — bytes preserved end-to-end into `set_tensor`.** Both layer-0 Q4_0 weight tensors hash bit-exactly: `blk.0.attn_q.weight` C++ pre-upload `0xf2f7188c` == JS-side ref `0xf2f7188c` (size 2,359,296 bytes, [2048,2048] Q4_0); `blk.0.attn_k.weight` C++ pre-upload `0x9399f36a` == JS-side ref `0x9399f36a` (size 294,912 bytes, [256,2048] Q4_0). Sizes match the byte-per-element calculation (`elemCount × 18/32` for Q4_0) on both sides. **Implication:** the GGUF parser → ggml allocator → set_tensor caller chain is bit-clean. The Stage 4.18/4.19 production `Qcur-0` Δ=5.24e-4 / `Kcur-0` Δ=3.38e-4 cannot originate from upstream byte corruption visible to set_tensor — Outcome E is rejected. Stage 4.18's three sub-hypotheses now resolve to: (U-A) attn_norm-0 — refuted by Stage 4.19 (bit-identical); (U-B) Q4_0 weight upload — **partially refuted** by Probe 9b (set_tensor's view matches GGUF; the host→GPU `Module.jsepWrite` → `device.queue.writeBuffer` link is not yet measured); (U-C) WGSL pipeline cache collision — not addressed. All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Probe runs only at model-load time (two extra ~MiB FNV passes); decode-loop perf unaffected. Closure: [`STAGE-4.20-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.20-RESULT.md). Raw artifacts: [`STAGE-4.20-spike-output.json`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.20-spike-output.json). Stage 4.21 brief below queues Probe 9c: GPU-side post-upload mapAsync readback hash to disambiguate Outcome F-1 (kernel re-investigate) from Outcome F-2 (host→GPU corruption).

**Stage 4.19 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; webllm-side `NODE_DUMP_ALLOWLIST` extension only; patch stack unchanged at 12).** **Branch 2 of Stage 4.18's brief CONFIRMED — Q-projection's src1 (`attn_norm-0`) is bit-identical between JSEP and wasm32 sides; src0 (Q4_0 weight bytes) is the suspect.** Probe 9a added `attn_norm-0`, `inp_embd`, and `l_out-0` to the cb_eval allowlist in `src/wasm/webgpu-bridge.cpp::NODE_DUMP_ALLOWLIST`. Spike at `?v=stage4.19a` + ref-probe at `?v=stage4.19a` rebuilt (`make wasm-build-jsep` + `make wasm-build-wasm32` + manual cp + `bun build smoke-test/p2-v2-ref-probe.src.ts`), 108 checkpoints captured each, diff via `STAGE-4.18-diff.py`. **Smoking gun:** idx=0 `attn_norm-0` ne=[2048,6,1,1] backend=CPU max_abs_delta=**0.000000** (RMSNorm runs on CPU on both sides → input embedding bytes bit-identical → output bytes bit-identical). idx=1 `Qcur-0` ne=[2048,6,1,1] backend=jsep_buf max_abs_delta=**0.000524** — the Q-projection MUL_MAT is the **first** JSEP-side op in the prefill chain, and the delta lights up at the very first JSEP dispatch. Combined with Stage 4.18's 1.68e-6 ULP-tight kernel measurement at the same shape, the 312× gap can only originate in src0 = `wq.weight` for layer 0 — i.e., the weight upload path. idx=4-5 `Vcur-0` Δ=0.000000 on CPU (consistent with Stage 4.18 Probe 8b — V-on-CPU). idx=6-8 `Kcur-0` Δ=3.38e-4 on jsep_buf (K-projection MUL_MAT, same scale of error as Q-projection — strong corroboration that both projections suffer the same upload-side defect). idx=9 `kq-0` Δ=1.19e-2 on jsep_buf (Q@K^T amplifies the upstream Δ_Q + Δ_K by accumulation length 64 — not a separate bug). After idx=15 the per-token cascade explodes (≥0.04 by `result_norm`, ≥6 by `result_output`) as expected. **Note on `inp_embd`:** added to allowlist but produced no checkpoint — `inp_embd` is a leaf input tensor (`llama-graph.cpp:1718-1720`: `ggml_new_tensor_2d` + `ggml_set_input`, no producing op), so cb_eval (which fires per-op during graph compute) never fires on it. The actual post-`ggml_get_rows`+select compute output is named `"embd"` (`llama-graph.cpp:1778`); future probes that need to drill upstream of attn_norm-0 should add `embd`. The current `inp_embd` allowlist entry is a documented no-op (kept with explanatory comment in source). All 6 spike selftests + 5 sweep selftests still PASS. `make checkall` not re-run on this stage (instrumentation-only edit, no behavioural change to library or kernel code; smoke equivalence is the load-bearing check). Per-token decode 481 ms (within noise of Stage 4.18's 879.7 ms run-with-sweep envelope; this run did not enable the sweep). Closure: [`STAGE-4.19-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-RESULT.md). Raw artifacts: [`STAGE-4.19-jsep-checkpoints.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-jsep-checkpoints.txt), [`STAGE-4.19-ref-checkpoints.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-ref-checkpoints.txt), [`STAGE-4.19-diff-output.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-diff-output.txt). Stage 4.20 brief below queues Probe 9b: weight-upload byte-hash check on the JSEP `set_tensor` path for layer-0 wq/wk weights, with Outcome E (upload corruption — one-line fix expected) vs Outcome F (upload clean — re-open kernel investigation with production-weight inputs) decision branches.

**Stage 4.18 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; webllm-side Q4_0 sweep selftest + cb_eval `backend=` tag; patch stack unchanged at 12).** **Outcome reframed — kernel-precision claim REFUTED.** Probe 8a (per-shape Q4_0 matmul sweep over all 5 production shapes: q-out-proj [2048,2048,6], k-v-proj [256,2048,6], ffn-gate-up [5632,2048,6], ffn-down [2048,5632,6], lm-head [32000,2048,1]) shows **the JSEP Q4_0 kernel matches both an f64 ground-truth reference and an f32 element-wise loop reference to ≤2.07e-6 / ≤1.25e-6 absolute** at every shape. At the production Q-projection shape (2048,2048,6) specifically, synthetic delta is **1.68e-6** vs the **5.24e-4** observed at Qcur-0 in real prefill — **312× larger in production** than any single matmul kernel can produce. The kernel cannot account for the production delta; it must originate upstream. Probe 8b (extended `node_dump_cb` to log `ggml_backend_buffer_name(t->buffer)`) maps the full Option A-prime split: **JSEP runs only Q-proj, K-proj, Q×K^T, out-proj** (4 ops/layer × 22 layers = 88 JSEP MUL_MAT outputs). V-projection (also Q4_0!), all RMSNorms, RoPE, permute, softmax, V@softmax, all FFN, and lm_head all run on CPU on the JSEP-side spike. **The Vcur-0 Δ=0 anomaly from Stage 4.17 is now fully explained:** V-projection is on CPU on both sides → bit-identical by construction. Same is true for kqv_out-0, ffn_norm-0, etc. (all CPU on both sides). 5 sweep results + 96-line backend-tagged JSEP/REF checkpoint dumps + new `STAGE-4.18-diff.py` (regex updated for `backend=` field) saved. All 6 spike selftests + 5 new sweep selftests PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Per-token decode 879.7 ms (within noise of Stage 4.16's 458 ms — this run includes the 5-shape Q4_0 sweep before the `bridge.decode` loop, but PER_TOKEN_MS is computed only over the decode loop). **Smoking-gun reframing:** the production Qcur-0 5.24e-4 first8 delta must come from one of (U-A) attn_norm-0 input differs between JSEP and CPU sides — most plausible because attn_norm-0 is NOT in the cb_eval allowlist so we can't see whether it's on JSEP or CPU; (U-B) Q4_0 weight bytes upload corruption — JSEP's set_tensor / weight upload path produces bit-different bytes than what GGUF parse landed in heap; (U-C) WGSL pipeline cache key collision producing a subtly different shader. Closure: [`STAGE-4.18-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-RESULT.md). Raw artifacts: [`STAGE-4.18-q4_0-sweep.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-q4_0-sweep.txt), [`STAGE-4.18-jsep-checkpoints-with-backend.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-jsep-checkpoints-with-backend.txt), [`STAGE-4.18-ref-checkpoints-with-backend.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-ref-checkpoints-with-backend.txt), [`STAGE-4.18-diff.py`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-diff.py). Stage 4.19 brief below queues localization of the upstream delta source — extend cb_eval allowlist with `attn_norm-0`/`inp_embd` and add a hash-on-upload check to the JSEP `set_tensor` path so weight-upload corruption shows up immediately.

**Stage 4.17 PARTIAL CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; webllm-side cb_eval bridge hook + non-JSEP ref-probe harness; patch stack unchanged at 12).** **Outcome B (kernel-correctness) CONFIRMED.** Probe 7 added `cb_eval`-based per-node first8 dump in `webgpu-bridge.cpp::node_dump_cb` (gated via new `webllm_enable_node_dump(int)` JS export; allowlist of 11 layer-0 + final tensor names: `Qcur-0`/`Kcur-0`/`Vcur-0`/`kq-0`/`kq_soft_max-0`/`kqv_out-0`/`attn_out-0`/`ffn_norm-0`/`ffn_out-0`/`result_norm`/`result_output`). Both `make wasm-build-wasm32` (production non-JSEP) and `make wasm-build-jsep` pick up the hook from the shared source file. New `smoke-test/p2-v2-ref-probe.{html,src.ts}` (~150 LOC) loads TinyLlama Q4_0 GGUF through `webllm-wasm.js`, runs identical prefill + greedy 5-decode, captures matching checkpoint set on `window.__refCheckpoints`. Reference produces `topId=3681 (" Paris"), topVal=13.04` — ground truth confirmed. JSEP produces `topId=297 ("in"), topVal=10.46`. **Smoking gun (96 checkpoints × prefill + 5 decode):** `Qcur-0` first8 max-abs-Δ = **5.24e-4** at the FIRST compute node (Q4_0 matmul output dim 2048); `Vcur-0` is **bit-identical** (suspect: V projection runs on CPU under Option A-prime scheduler split — Stage 4.13's retracted-but-prescient hypothesis); `Kcur-0` Δ = 3.38e-4 (same kernel as V, same shape, but non-zero — Vcur=0 is anomalous). The first checkpoint to cross the 1e-3 "structural" threshold is **`attn_out-0` at idx 11 (max-abs-Δ = 4.77e-3)** — this is `residual + out_proj × kqv_out_post_permute` where the out-proj matmul has same shape [2048,2048] × [2048,6] as Q-proj at idx 0. RMSNorm of the small-magnitude attn_out amplifies the relative diff: `ffn_norm-0` Δ = 1.83e-1 (38× the absolute attn_out diff because RMSNorm scales by 1/√(mean²+ε) at near-zero magnitude). By `result_norm` (post-22-layer, with layers 1-21 unmonitored by allowlist), absolute Δ = +5.83; logits Δ = +6.61 — flips the argmax token. **No NaN, no Inf, no all-zero pathology** — Stage 4.16's `EM_ASYNC_JS` fix landed correctly; remaining bug is purely numerical compounding precision noise across 22 layers. Per-token decode 474 ms (within noise of Stage 4.16 baseline 458 ms). All 6 selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Closure: [`STAGE-4.17-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-RESULT.md). Raw artifacts: [`STAGE-4.17-jsep-checkpoints.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-jsep-checkpoints.txt), [`STAGE-4.17-ref-checkpoints.txt`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-ref-checkpoints.txt), [`STAGE-4.17-diff.py`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-diff.py). Stage 4.18 brief below queues per-shape Q4_0 matmul self-test sweep (production shapes [2048,2048] / [256,2048] / [2048,5632] / [5632,2048] / [32000,2048]) against a numpy/CPU reference dequant, plus a "is V really on CPU?" backend-trace probe to explain the Vcur=0 anomaly.

**Stage 4.16 PARTIAL CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (P9 — `EM_ASYNC_JS` for `ggml_jsep_read` in `ggml-jsep.cpp:149-163`; patch stack 11 → 12).** **Fifth outcome of Stage 4.15's matrix CONFIRMED — Probe 6 ruled out all four documented sub-hypotheses (mirror-mismatch / offset-mismatch / undersized-read / fires-on-different-node), then a 4-float `mirror_post_h1` peek added to Probe 6 caught H1 fire-and-forget red-handed.** Cross-correlation is exact for Qcur-0: Probe 5 (divert) writes `(h=26, o=4194304, ne=[2048,6])` 49152 bytes valid; Probe 6 (H1) reads `(bctx_handle=26, dst_offset=4194304, dst_size=49152, name="Qcur-0")` — same node, same handle, same offset, same size, same `view_src=null`. Yet `mirror_post_h1[0..4]` *immediately after H1 returns* reads `[0, 0, 0, 0]` for the load-bearing Qcur-0 write. Smoking gun: `ggml_jsep_read` was declared `EM_JS(void, ...)` not `EM_ASYNC_JS`. Under JSPI (`-sJSPI_EXPORTS=...`, no `WebAssembly.Suspending` import), the JS body's returned Promise from `Module.jsepRead` was discarded; the readback ran asynchronously and host_mirror updates landed AFTER the C++ caller had moved on. The legacy comment at the EM_JS site claimed JSPI awaits implicitly — it does not; only `WebAssembly.Suspending`-wrapped imports suspend a `WebAssembly.promising`-wrapped export. `EM_ASYNC_JS` (in `<emscripten/em_js.h>:73`) routes the body through `Asyncify.handleAsync`, which Emscripten 5.0.7 lowers to JSPI's Suspending wrap. Fix is one macro change + an `await`. **PARTIAL Outcome A:** decode flips from stuck-at-confident-wrong (`topId=593/0.159`, `"ntiuhuihnerquant"`) to varied-but-still-wrong (`topId=297/topVal=10.46`, `LOGIT_STATS_STEP0.first8 = [-8.39, -8.11, 1.14, -5.41, -5.62, -4.41, -6.30, -7.71]`, finiteCount=32000, no NaN/Inf, `GENERATED_TOKENS=[297,8927,13601,29877,29899]`, `GENERATED_TEXT="inonic boso-"` — English-letter morphology, not yet "Paris"). H1-inverse (jsepWrite) unaffected — `device.queue.writeBuffer` is sync, no Promise. Per-token decode 458.5 ms (vs Stage 4.15's 107.7 / Stage 4.5's 25.0 ms; ~18× regression — H1 now actually awaits per-runOp; ~1602 readbacks per token). Optimization (dirty-bit, batched readback at slice boundaries, peeled-consumer-only) deferred to Stage 5+. All 6 selftests still PASS. `make checkall` green (747 pass / 36 skip / 0 fail). Closure: [`STAGE-4.16-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.16-RESULT.md). Stage 4.17 brief below queues per-attention-output reference-diff to localize the remaining downstream bug (kernel-correctness at production shape vs cross-backend boundary leak vs CPU-fallback ROPE/SOFT_MAX issue).

**Stage 4.15 PARTIAL CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; webllm-side spike instrumentation only — `src/inference/jsep/ops/matmul.ts` divert path adds gated tempDst + dstRec.buffer readback at `__stage415DivertProbe`; `smoke-test/p2-v2-spike.src.ts` enables the gate before model load; patch stack unchanged at 11).** **Branch 3 of Stage 4.14's disambiguation table CONFIRMED — divert path lands data correctly; H1 GPU→host writeback fails to deposit it in host_mirror at the offset `get_tensor` reads from.** Probe 5 captures `tempDst[0..16)` and `dstRec.buffer[dst.offset..+16)` per divert dispatch (cap 32, deferred mapAsync drain). **Smoking-gun #1 (Branch 1 REJECTED):** i=1 (layer 0 Q proj, dst=[2048,6] @ h26+4194304) records `tempF4 = [-6.26e-5, 1.87e-5, -6.09e-5, -9.48e-5]` — kernel produces valid output at the load-bearing Q-shape (M=2048). i=2 (layer 0 K proj, dst=[256,6] @ h26+4194304) records `tempF4 = [6.85e-5, 1.35e-4, -1.74e-4, 7.41e-5]` — also valid. **Smoking-gun #2 (Branch 2 REJECTED):** for every captured entry, `tempBytes` is byte-exact equal to `dstBytes` (`tempEqDst === true` for all 32). copyBufferToBuffer lands at `dst.offset` in `dstRec.buffer`. **Smoking-gun #3 (Branch 3 CONFIRMED):** Probe 5 i=1 wrote valid Q to GPU(handle=26, offset=4194304); Stage 4.14 `__jsepGetTensorLog` reports `host_mirror[26]+4194304` reads `[0, 0, 0, 0]` for `name = "Qcur-0"` — the data exists on the GPU but `Module.jsepRead` (called from H1 in `ggml-jsep.cpp:779-786`) is not reading it into host_mirror. Cascade corroboration: i=3 (attention scores Q × K_cache, dst=[256,6,32]) records `tempF4 = [0,0,0,0]` — its src1 (Q ROPE'd on CPU) is genuinely zero at GPU kernel time because the CPU read host_mirror's stale zeros, ROPE'd zeros, and H1-inverse synced zeros back to GPU. i=5 (layer 1 Q proj, dst=[2048,6]) also records zero output — layer 0's broken output zeros the residual stream cascading through every subsequent layer. **Stage 4.14's "JSEP MUL_MAT divert produces no host-visible output" framing was right in observable effect but wrong in mechanism — the divert IS host-visible at the GPU level; H1's writeback path is what fails to mirror it.** All three sub-hypotheses for Stage 4.16 to disambiguate (H1-mirror-mismatch / H1-stale-buffer / H1-fires-on-different-node-than-dispatch) are listed in the closure report. Spike replay reproduced (`__stage415DivertLog.length === 32`, `tempEqDst === true`, 6 selftests PASS, `make checkall` green). Per-token decode 107.7 ms (Stage 4.14 baseline 127.92 ms; deferred mapAsyncs land outside decode timing). Closure: [`STAGE-4.15-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.15-RESULT.md). Raw probe data: [`STAGE-4.15-probe5-raw.json`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.15-probe5-raw.json). Stage 4.16 brief below queues per-runOp logging of `(node->buffer->context->handle, jsep_tensor_handle(node), ggml_nbytes(node), tensor_name)` to cross-correlate against Probe 5's `(dst.bufHandle, dst.offset, dst.ne)` and identify which sub-hypothesis fires.

**Stage 4.14 PARTIAL CLOSED 2026-05-07 — `<pending>` + llama.cpp `ddeb2fb6e` (Stage-4.14 Probe 4 instrumentation — `ggml-cpu.cpp` post-compute per-node dst readback + `ggml-jsep.cpp::ggml_backend_jsep_buffer_get_tensor` log; patch stack 10 → 11; expected to revert at Stage 4.15 once structural fix lands).** **Stage 4.13's CPU-D narrative REFUTED; new diagnosis CPU-E (JSEP MUL_MAT divert dispatch produces no host-visible output) CONFIRMED.** Probe 4 captures every CPU op's `dst_addr`/`src0_addr` post-compute and every JSEP `get_tensor` bridge call. **Smoking-gun #1:** zero CPU ops write to addr 99811136 between i=1 (`attn_norm-0` valid) and i=2 (`Kcur-0 (view)` zeros). The `99811136` slot is recycled by ggml's allocator — Kcur-0 isn't a clobbered live value, it's a fresh sched-allocated `CPU#Kcur-0#0` bridge slot that never received its expected payload. **Smoking-gun #2:** ggml-backend-sched correctly invokes JSEP `get_tensor` for every cross-backend Q/K projection (handle 26, distinct offsets per layer). The bridge faithfully reads `host_mirror[h] + offset`. For `Qcur-0/Kcur-0` (both at offset 4194304) and `Qcur-1` (at offset 6295552) and every other Q-projection across all layers, the read returns `[0,0,0,0]`. **Smoking-gun #3:** `Kcur-1` at offset 528384 reads `[-3.09e-6, -1.52e-6, 4.02e-6, 4.66e-6]` — bit-identical to Vcur-0's CPU MUL_MAT output. So the "valid" Kcur-1 read is **stale V-projection data leftover from an earlier set_tensor write**, not a real K projection. **Therefore every JSEP MUL_MAT for Q (every layer) and K (layer 0) produces zero output at host_mirror.** `COUNTER_DELTAS.read == runOp == 1602` confirms H1 fires for every JSEP op, so the post-runOp writeback is executing — yet host_mirror reads zeros. The bug must live inside the divert dispatch path itself: either the kernel writes zeros into tempDst (bind-group / dispatch-dim mismatch on Q-shape M=2048), or copyBufferToBuffer doesn't land at the expected dstRec.buffer offset, or H1's readAsync samples a different buffer than the divert wrote. Prefill+decode reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId/topVal=593/0.159`, 6 selftests PASS, `make checkall` green). Per-token decode 127.92 ms (Stage 4.13 baseline 127.40 ms; instrumentation invisible). Closure: [`STAGE-4.14-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.14-RESULT.md). Stage 4.15 brief below queues per-divert-dispatch readback of tempDst + dstRec.buffer at the just-written offset, with three branch-on-outcome paths (kernel bug / copy bug / readback-vs-buffer mismatch).

**Stage 4.13 PARTIAL CLOSED 2026-05-07 — `<pending>` + llama.cpp `3b0e40d6f` (Stage-4.13 Probe 3 instrumentation — CPU MUL_MAT capture in `ggml-cpu.cpp` + set_tensor `name` + `data_addr` enrichment + `alloc_buffer` host_mirror publication in `ggml-jsep.cpp`; patch stack 9 → 10; expected to revert at Stage 4.14 once structural fix lands).** **All three predicted sub-cases (CPU-A / CPU-B / CPU-C) REJECTED; new diagnosis CPU-D (cross-backend buffer-aliasing) CONFIRMED.** Tensor-name capture in `set_tensor` revealed Stages 4.10/4.11/4.12 had **K and V slot labels swapped**: `(h26, 0)` is the **K cur layer 0** slot, **not** V; `(h26, 528384)` is the **V cur layer 0** slot. Slice 3 SET_ROWS node 0 reads `(h26, 0)` for K cache write; node 4 reads `(h26, 528384)` for V cache write. **Bug is on the K side**: `JSEP#Kcur-0 (view)#0` lands as 6144B zeros at `(h26, 0)` from CPU heap addr 99827008 — the SAME address as i=1's `JSEP#attn_norm-0` write (which contained valid normed×gain). Between i=1 and i=2, scratch 99827008's first 6144B gets zeroed by some intervening CPU op. K projection on JSEP (slice 1/2 MUL_MAT) writes to `(h26, 4194304)` = host_mirror[26]+4194304 = 53658624, NOT to 99827008 where Kcur-0 view points. So Kcur-0's view is mis-aliased (points at the wrong scratch); the JSEP K result never reaches `(h26, 0)`; slice 3 reads zeros for K → K cache layer 0 = zeros → broken attention (`Q×K^T = 0`, uniform softmax) → garbage decode. V side works because Vcur-0's MUL_MAT runs on **CPU** (callIdx=2, dst=108215616, src1=valid normed×gain) and its set_tensor copies the actual MUL_MAT dst to `(h26, 528384)`. The 10 [256,6] CPU MUL_MATs are all Vcur-N projections; no CPU MUL_MAT produces a Kcur-N. The 10 [2048,6] CPU MUL_MATs are output projections (`kqv_out`) whose src1 is `softmax(Q×K^T)V` — observed src1 patterns of `[0,0,0,0]` and denormal-style garbage are **downstream artifacts** of broken K cache, not independent bugs. Prefill+decode reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId=593/0.159`, 6 selftests PASS, `make checkall` green). Per-token decode 127.40 ms (Stage 4.12 baseline 127.40 ms; instrumentation invisible). Closure: [`STAGE-4.13-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.13-RESULT.md). Stage 4.14 brief below queues a tighter localization probe (capture every CPU op writing to addr 99827008 between i=1 and i=2) before committing to Path R (re-aim Kcur-0's view at JSEP K projection result), Path U (force K projection to CPU like V), or Path Q (eliminate the zeroing op).

**Stage 4.12 PARTIAL CLOSED 2026-05-07 — `<pending>` + llama.cpp `b50f92fd3` (Stage-4.12 diagnostic patch — Probe 2 CPU graph_compute instrumentation in `ggml-cpu.cpp` + JSEP residency resolver in `ggml-jsep.cpp` + set_tensor logging on handle 26; patch stack 8 → 9; expected to revert at Stage 4.13 once the structural fix lands).** **Probe 2's three predicted sub-cases all rest on a false premise.** `__cpuGraphLog` (30 calls × 42 nodes) shows **zero** nodes with a jsep-resident `dst`, `src0`, or `src1` — the CPU backend operates entirely on tensors already split off into CPU buft by ggml-backend's scheduler. Cross-backend writes into `jsep_buf` go through `ggml_backend_jsep_buffer_set_tensor`, NOT through cgraph nodes the CPU backend executes. Set_tensor follow-up gives the smoking gun: `__setTensorLog` shows the V SET_ROWS' source and K SET_ROWS' source land via twin 6144-byte `set_tensor` calls — but **V's lands as all-zeros at `(h26, 0)`** while **K's lands as valid f32 at `(h26, 528384)`**. Sequence at `(h26, 0)`: 49152 bytes (input embedding) → 49152 bytes (RMS_NORM result) → **6144 bytes ZEROS** (V projection result), exactly the allocator-coalesced slot Stage 4.11 hypothesized — and the V projection's CPU-side scratch is filled with zeros by the time the scheduler CPYs it back into `jsep_buf`. Three sub-cases for V's CPU producer: (CPU-A) MUL_MAT skipped entirely, scratch passes through; (CPU-B) MUL_MAT runs but reads zero src1/src0; (CPU-C) MUL_MAT routes to JSEP but allocator-coalesces V's dst onto (h26, 0) clobbered by input embedding/RMS_NORM. CPU-C is most likely given the size-49152 → size-49152 → size-6144 sequence at offset 0. Stage 4.11 baseline reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId/topVal=593/0.159`, 6 selftests PASS, `make checkall` green). Per-token decode 129.36 ms (Stage 4.11 baseline 126.04 ms; +2.6% within noise — instrumentation visible during prefill, invisible during decode). Closure: [`STAGE-4.12-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.12-RESULT.md). Stage 4.13 brief below queues the disambiguation probe (CPU-A vs CPU-B vs CPU-C) followed by the structural fix.

**Stage 4.11 CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (no patch — diagnostic instrumentation only in `ggml-jsep.cpp::ggml_backend_jsep_graph_compute`; patch stack unchanged at 8).** **HD' CONFIRMED — and asymmetric.** Probe 1 added ENTRY/EXIT host_mirror snapshots for `h26+0` and `h26+528384` keyed by `s_sliceIdx411` into `globalThis.__interSliceLog` (60 entries = 30 enter + 30 exit; cap shared with Stage 4.10 graph-log). **Smoking gun:** the K side works (`h26+528384` = `[-3.09e-6, -1.52e-6]` at slice 3 enter — populated in time by a CPU CPY+ROPE chain between slice 2 exit and slice 3 enter), but the V side is broken (`h26+0` = `[0, 0]` at slice 3 enter; stays zero through slice 10 exit; turns into K-shaped data — not V — by slice 11 enter, suggesting allocator reuse from a later layer rather than V landing). Slice-0 RMS_NORM legitimately writes to `h26+0` ([-1.30e-3, 1.90e-3]); a CPU op between slices 0 exit and 1 enter overwrites with smaller values; another CPU op between slice 2 exit and slice 3 enter zeros the slot — most plausibly the scheduler reusing the slot for V's tensor allocation, with V's producer not firing in time. Cross-checked `__jsepGraphLog`: JSEP's distinct `dstO` values for `dstH=26` across all 30 slices are `{4194304, 6295552, 528384}` — **offset 0 absent**, so the producer is necessarily a CPU subgraph (or never fires). Stage 4.10's "missing 3rd projection" observation narrows: K's CPU subgraph fires correctly; V's does not. Stage 4.10 baseline reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId/topVal=593/0.159`, 6 selftests PASS, `make checkall` green). Per-token decode 126.04 ms (vs 127.42 baseline; instrumentation invisible). Closure: [`STAGE-4.11-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.11-RESULT.md). Stage 4.12 brief below queues Probe 2 (CPU graph_compute instrumentation) to localize WHERE V's CPU producer writes, plus the structural-fix decision once that data lands.

**Stage 4.10 CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (no patch — diagnostic instrumentation only in `ggml-jsep.cpp::ggml_backend_jsep_graph_compute`; patch stack unchanged at 8).** **HA strict-form REJECTED, refined HA' / HD CONFIRMED.** Per-graph_compute slice composition logged into `globalThis.__jsepGraphLog` (first 30 invocations × per-node `[op, dstH, dstO, src0H, src0O, src0Op, src1H, src1O, src1Op, src0VsOp]`). **Smoking gun:** slice 3 (the FIRST SET_ROWS slice) has `nodes[0] = {op:42, dstH:25, dstO:0, src0H:26, src0O:0, src0Op:0, s0VsOp:0, src1H:26, src1O:524288, src1Op:0}` and `nodes[4] = {op:42, dstH:25, dstO:262144, src0H:26, src0O:528384, src0Op:0, s0VsOp:0, ...}`. Both SET_ROWS' src0 have **`src0Op = GGML_OP_NONE = 0`** AND `s0VsOp = 0` — they are LEAF tensors in JSEP's split-cgraph view, not produced by any op visible to JSEP. The scheduler split treats `h26+0` (V data) and `h26+528384` (K data) as cross-backend boundary inputs; the producer chain lives in CPU subgraphs that should populate `host_mirror[26]+0..6144` via direct `tensor->data → host_mirror` writes (post-F1). HA's strict form is rejected: 3 JSEP graph_compute calls fire BEFORE slice 3 (slices 0/1/2 = RMS_NORM h26+0 → proj A h26+4194304 → proj B h26+4194304), with CPU subgraphs interleaved between them. Pre-SET_ROWS JSEP slices write to h26+0 (RMS_NORM, but content ≠ V/K data) and h26+4194304 (allocator-coalesced projection scratch); **none write the V/K-shaped data the SET_ROWS leaves expect at h26+0 / h26+528384**. HB weakened (s0VsOp=0 rules out view-src indirection); HC unlikely (Stage 4.9 callIdx 2-7 show stable K data implying no spurious clearing). The actual root cause appears to be **HD: cross-backend leaf without producer landing data in time** — the CPU CPY/MUL_MAT chain that should populate the leaves either doesn't fire before slice 3 or writes to a different absolute address. Stage 4.9 baseline reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId/topVal=593/0.159`, 6 selftests PASS, `make checkall` green). Closure: [`STAGE-4.10-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.10-RESULT.md). Stage 4.11 brief below queues two probes in priority order: (1) inter-slice host_mirror snapshot to localize WHEN h26+0 gets populated; (2) CPU graph_compute instrumentation to localize WHICH CPU subgraph produces the data.

**Stage 4.9 CLOSED 2026-05-07 — `<pending>` + llama.cpp `<pending>` (P8 — H1-inverse host→GPU writeback in `ggml_backend_jsep_graph_compute` per-runOp pre-pass; patch stack 7 → 8).** **Outcome C-2 — H1-inverse fires correctly but does not unblock Outcome A.** New `__h1invDiag` capture in `module.jsepWrite` (gated on the distinctive i=3-src0 signature `handle=26, offset=0, size=6144`) records the first 8 bytes of `host_mirror[hostPtr..]` for the eight load-bearing H1-inverse calls across one prefill + 5 decode steps. **Smoking gun:** `callIdx 0` (prefill i=3) and `callIdx 1` (decode-step-1 SET_ROWS) both report `first8F32 = [0, 0, 0, 0, 0, 0, 0, 0]`. `callIdx 2-7` report real F32 K data (`[7.19e-5, -2.05e-5, -3.38e-5, ...]`, magnitudes consistent with post-ROPE K). So at the FIRST two SET_ROWS dispatches, host_mirror[h26+0..6144] **is itself zero** — H1-inverse faithfully syncs zeros to GPU. Stage 4.8's framing ("CPU op chain updates host_mirror but not GPU") was structurally incomplete: in the load-bearing window, the CPU CPY+ROPE chain has **not yet written to host_mirror at the offset SET_ROWS reads from**. `LOGIT_STATS_STEP0` bit-identical to Stage 4.5 (`topId/topVal = 593/0.159`); `GENERATED_TEXT = "ntiuhuihnerquant"` bit-identical to Stage 4.5. Per-token decode 131.80 ms (vs 25.04 baseline; 5.3× regression from per-runOp pre-pass uploading every src tensor including weights). `COUNTER_DELTAS.write = 4404` (was ~134; +4270 H1-inverse fires). All 6 kernel selftests still PASS. `make checkall` green. H1-inverse stays despite not flipping Outcome A — it is the structurally correct symmetric counterpart to Stage 4.5 H1, and `callIdx 2-7` evidence shows the writeback gap is real for all but the first two dispatches. Optimization (dirty-bit tracking, sync only sources whose host_mirror changed) deferred to Phase 3 Stage 5+. Closure: [`STAGE-4.9-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.9-RESULT.md). Stage 4.10 brief below queues localization of the host_mirror-staleness root cause: WHY does CPU CPY+ROPE not write to `host_mirror[h26+0..6144]` before the first 2 SET_ROWS dispatches? Stage 4.9 diagnostic captures (`__h1invDiag`) retained in spike + jsep for Stage 4.10's first probe.

**Stage 4.8 CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; spike + JSEP diagnostic instrumentation only).** Stage 4.7's "Reading R1" framing (i=3 SET_ROWS divert dispatch silently fails) was a misframing. The dispatcher, the kernel, the divert path are all correct. The bug is upstream: at i=3 dispatch time, `src0` (h26o0 — the K-projection-after-ROPE buffer) is **stale on the GPU side**. Step A (eager-warmup probe at engine init, two shapes) did NOT fix it — the bug isn't generic first-call cold-start. Step B (windowed temp-dst + src0 readback inside `dispatchSetRows`, gated on a `globalThis.__stage48DivertHook` flag set just before `bridge.decode()`) captured: pre-kernel temp-dst row 0..5 = all zeros (pre-copy of zero real-dst); **post-kernel temp-dst row 0 = all zeros**, rows 2..5 = sparse `0x8000` (= f16 -0.0) cells exactly where `src0` raw bytes show `00 00 00 80` (= f32 -0.0). The kernel is correct; it reads f32 from src0, packs via `pack2x16float`, atomic-CAS writes f16. The reads return mostly 0.0 (writes 0x0000 = no-op) and sparse -0.0 (writes 0x8000) because `src0`'s GPU region was never populated — i=2 (K-projection MUL_MAT) writes to h26o4194304, but i=3 reads from h26o0. The h26o4194304→h26o0 hop is a CPU-fallback op chain (likely `CPY` + ROPE) that updates host_mirror but **never writes back to the GPU buffer**. Stage 4.5 H1 adds GPU→host writeback after every JSEP runOp; the symmetric **host→GPU** writeback after every CPU-fallback op (flagged but not addressed by the Stage 4.4 brief) is the missing piece. Closure: [`STAGE-4.8-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.8-RESULT.md). False trail ruled out: an earlier Stage 4.8 sentinel-probe variant added awaits BEFORE `runOpOrig` and saw "status=-1" — this turned out to be JSPI **not** preserving the C-stack `desc` array (`ggml-jsep.cpp:409`, declared outside the for loop) across `EM_ASM_INT` Promise-await reentries. Diagnostic-only artifact; fix was to move all probe awaits to AFTER `runOpOrig`. Patch stack: 7 (unchanged). Stage 4.9 brief below queues the host→GPU writeback fix (H1-inverse, mirroring Stage 4.5 H1).

**Stage 4.7 D2-tight CLOSED 2026-05-07 — `<pending>` (no llama.cpp patch; spike instrumentation only).** D2-tight rewrites the `mod.jsepRunOp` wrapper as `async`. After the original dispatcher returns, when the op is one of the first 10 SET_ROWS, the wrapper flushes `runtime.encoderBatcher`, copies 16 bytes (8 F16 cells) from `dst[dstO..+16)` into a staging buffer, awaits `mapAsync(GPUMapMode.READ)`, and stores the result on `setRowsDiagEntry.dstImmediateFirst8U16`. Because `graph_compute` is in `JSPI_EXPORTS` (`src/wasm/CMakeLists.txt:158`), JSPI awaits the returned `Promise<number>` on the wasm side — the wasm-side caller suspends until the readback resolves, so when wasm proceeds to the NEXT jsepRunOp the read above is guaranteed to reflect ONLY this op's output (no later ops have run yet). The `): number =>` return annotation is dropped and the closure is cast through `any` to keep the `JsepModule` interface honest. **Outcome: Reading R1 confirmed.** i=3 (FIRST SET_ROWS, K-cache layer 0, dstO=0, divert) reads `dstImmediateFirst8U16 = [0,0,0,0,0,0,0,0]` — byte-exact identical to its end-of-decode `dstPostFirst8U16`. Every other captured SET_ROWS (i=4, 14, 15, 26, 27, 39, 40, 51, 52) shows non-zero F16 cells with `dstImmediate == dstPost` byte-exact. **R2 is ruled out: the dispatch itself silently failed to land i=3's writes; no later op had a chance to overwrite anything.** All 6 selftests still PASS. Spike chat path unchanged: `GENERATED_TEXT = "ntiuhuihnerquant"`, PER_TOKEN_MS = 25.24 (vs 25.16 D2-lite baseline — within noise; the per-call readback only fires for the first 10 SET_ROWS dispatches and lands during prefill, not decode). `make typecheck` green. Closure: [`STAGE-4.7-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.7-RESULT.md). Stage 4.8 brief below queues localization of the first-call corner case in the SET_ROWS divert dispatcher (suspect surfaces: encoder-batcher first flush, pipeline cache miss for the SET_ROWS variant, bind-group / temp-dst first allocation lifecycle, pre-copy of zero-initialised dst surviving as the post-copy-back result). Patch stack: 7 (unchanged).

**Stage 4.6 D2-lite PARTIALLY CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch; spike instrumentation only).** D2-lite captures src[0] (8 F32), src[1] (8 I64 indices low-half), and dst (8 F16 cells) for the first 10 SET_ROWS dispatches via deferred Promise.then microtask staging copies. **Findings:** src0 is sensible across all 10 (`[-1.067, 0.656, -0.110, -0.110, ...]` for K — note the (2,3)/(4,5) pair-mate identity consistent with ROPE @ position 0, or `[-0.0006, 0.0009, ...]` for V); indices are exactly correct (`[0,1,2,3,4,5]` for K-cache shape `[256,512]`; strided `[0, 512, 1024, 1536, 2048, 2560, 3072, 3584]` for V-cache transpose `[1, 131072]`). **H-indices REJECTED.** **H-source WEAKENED** — no garbage / NaN / Inf values seen. **But i=3 (FIRST SET_ROWS, K-cache layer 0 at dstO=0) shows `dstPostFirst8U16 = [0, 0, 0, 0, 0, 0, 0, 0]`** at end-of-decode while every other captured SET_ROWS shows non-zero F16 cells (i=4, i=14, i=15, i=26, i=27, i=39, i=40, i=51, i=52). Two readings: (R1) i=3's dispatch silently failed — position-0 K cache at layer 0 was never written; (R2) i=3 wrote correctly but a later op overwrote cells 0..7 with zeros (no captured SET_ROWS targets dstO=0..7 of K-cache layer 0; possible jsepClear or divert pre-copy from another op). The end-of-decode readback is 5 decode steps + ~100+ ops removed from i=3, so we can't tell which reading without an immediate post-dispatch capture. Closure: [`STAGE-4.6-D2-LITE-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.6-D2-LITE-RESULT.md). Stage 4.7 brief below queues D2-tight (synchronous readback inside the wrapped jsepRunOp before it returns) to disambiguate R1 vs R2. Patch stack: 7 (unchanged).

**Stage 4.6 D1 PARTIALLY CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch; spike instrumentation only).** D1 V-cache transpose SET_ROWS selftest landed in `smoke-test/p2-v2-spike.src.ts` (~270 LOC). Builds 16 F16 dst cells with sentinel pattern, 4 source rows with F16-exact values, I64 indices `[0, 1, 6, 7]` chosen so cells 0&1 share u32 word 0 and cells 6&7 share u32 word 3 (atomic-CAS race). **Both no-divert AND divert variants PASS bit-exactly** — `maxAbsDeltaTargeted=0`, `maxAbsDeltaUntargeted=0`, no NaN, no Inf; postF16 matches expectedF16 cell-for-cell. The divert path's pre-copy correctly preserves untargeted cells; the F32→F16 atomic CAS at adjacent pair-mates doesn't corrupt either side; I64 indices read correctly (low 32 bits). **Implication: `dispatchSetRows` is exonerated as the source of Stage 4.5's wrong-decode bug.** Stage 4.5's `FIRST_ALLZERO_DST_PROBE` reading was (a) — false positive — because the multi-MB KV cache buffer at offset 0 has only 8 cells in the `first8` window and SET_ROWS targets a sparse subset; cells not in the indices list stay at post-allocation zero. Closure: [`STAGE-4.6-D1-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.6-D1-RESULT.md). Stage 4.6 is not yet fully closed — D2 (per-dispatch CPU-reference diff in production graph context) and D3 (ROPE / SOFT_MAX / attention-masking inspection) remain. Stage 4.6 D2/D3 brief below queues the source-data + indices + attention-consumer hypotheses, in priority order based on the Stage 4.5 token-2-onward divergence pattern (token 0 + 1 deterministic across H1 on/off; tokens 2+ differ — points at a decode-step KV write/read interaction). Patch stack: 7 (unchanged).

**Stage 4.5 CLOSED 2026-05-06 — `<pending>` + llama.cpp `e0fa38928` (P7 follow-on — H1 unconditional GPU→host writeback in `ggml_backend_jsep_graph_compute`; patch stack unchanged at 7 — H1 lives inside the F1 patch as a 12-LOC addition). Bonus: Makefile build-order race fix (cp-before-bundle) so the spike no longer captures a stale wasmImports table when a wasm import gets DCE'd or re-introduced across builds.** **PARTIAL OUTCOME A — H1 fires correctly: `COUNTER_DELTAS.read = 1602` (was 0), matching the runOp count exactly; FIRST_NAN_DST_PROBE still null (Bug A stable). Per-token decode 25.04 ms (was 23.30 — H1 cost is +7%, much smaller than the brief's 50-150ms estimate; likely because most ops have small dst tensors and the per-op jsepRead fixed cost dominates over bytes-read scaling). All 4 kernel selftests still PASS.** **But Outcome A "Paris" decode NOT achieved** — `LOGIT_STATS_STEP0` is bit-exactly identical to Stage 4.4 (`topId/topVal = 593/0.159`); generated tokens 2-4 changed (`[593, 5871, 945, 16976, 25487]` → `[593, 5871, 15669, 15565, 12150]`); decoded text `"ntiuhuihnerquant"` (was `"ntiuracinateenes"`). Both gibberish — H1 affected data flow but not curatively. **`FIRST_ALLZERO_DST_PROBE` still fires on the same op** — `{i:3, op:42 (SET_ROWS), dstH:18 (KV cache), dstO:0}`. Two readings: (a) the KV cache row at offset 0 legitimately contains zeros at the start (false positive), or (b) SET_ROWS is genuinely writing to the wrong offset/cell/layout. Reading (b) is more load-bearing because TinyLlama uses the **transposed V-cache layout** (`llama-kv-cache.cpp:1281`: ne[0]=1, adjacent indices share a u32 word) and Stage 1's `dispatchSetRows` has an F32→F16 atomic-CAS path specifically for this case. If the atomic CAS indexing is off-by-one or the wrong source row is read, V-cache writes go to the wrong place — forward pass produces real-but-wrong tokens, exactly what we see. Closure: [`STAGE-4.5-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.5-RESULT.md). Stage 4.6 brief below queues KV-cache write correctness localization (real-shape SET_ROWS selftest with V-cache transpose layout + per-dispatch CPU-reference diff).

**Stage 4.3 CLOSED 2026-05-06 — `<pending>` (no llama.cpp patch; spike instrumentation only).** Production-shape kernel selftests (4.3a) and full-graph runOp capture (4.3b) landed in `smoke-test/p2-v2-spike.src.ts`. **Stage 4.3a result: all four selftests PASS** — `RMSNORM_MULTIROW_NODIVERT/DIVERT` (rows=6, cols=2048) and `MATMUL_PROD_NODIVERT/DIVERT` (M=64, K=2048, N=6, Q4_K) all match CPU reference (max delta 2e-6 / 3.5e-4 respectively, no NaN/Inf, divert path matches non-divert exactly). **Bug A is NOT in the kernels or divert paths.** Stage 4.3b raised RUN_MAX 30→1700, MAX_LOG 30→3000, and added a unified `evtSeq` interleave so jsepWrite/jsepRead/jsepRunOp ordering is unambiguous. The smoking-gun timeline: `seq 0` jsepWrite buf19@0 = **valid** embedding · `seq 1` jsepRunOp i=0 RMS_NORM (in-place divert) · `seq 2` jsepRead buf19@0 = **valid** normed output `[-0.336, 0.49, ...]` · `seq 3` jsepWrite buf19@0 = **GARBAGE** `[-5e-5, 142, -4.5, -7.4e18, ...]` · `seq 4` jsepRunOp i=1 MUL_MAT consumes garbage src1 · `seq 5` jsepRead reads canonical NaN at the MUL_MAT dst. Between seq 2 (read) and seq 3 (write) is exactly one CPU op — the per-channel `MUL` (gain × normed = `out[r,c] = normed[r,c] * attn_norm.weight[c]`). Implied weight values from the output bytes: `[1.5e-4, 290, 8.96, -7.5e18, ...]` — wildly out of range for an RMSNorm gain (should be ~1.0). **Distribution-of-handle smoking gun:** all 1206 jsepWrites and 1266 jsepReads target handle 19 (the activations buffer); ZERO traffic on weight buffers (handles 14-17, the four 128 MiB weight buckets). The CPU-fallback ops never copy weights from JSEP to host — they dereference `tensor->data` directly, which on JSEP is the sentinel `GGML_JSEP_PTR_BASE = 0x2000` plus a per-tensor offset. ggml-backend treats this as a valid host pointer (because `get_base()` claims it is), so CPU MUL reads `0x2000 + offset` as F32, which is uninitialized wasm-heap RAM. Garbage weights × valid input → garbage activations → garbage MUL_MAT inputs → ±Inf accumulators → NaN dst → cascade through every downstream op. **Bug B (lm_head all-zero) is a downstream symptom of Bug A**, not an independent bug. Per-token decode 23.92 ms (within noise of Stage-4.2 baseline 24.34 ms). Closure: [`STAGE-4.3-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.3-RESULT.md). Patch stack: 6 (unchanged). Q4_K production-shape and multi-row RMS_NORM selftests are now permanent regression checks in the spike. Stage 4.4 brief below queues the F1 dual-resident-weights fix in `ggml-jsep.cpp`.

### Closure-stub discipline — every TODO closure prepares for a fresh session

Every time a stage / phase / probe closes and gets recorded above:

1. **Bank a closure paragraph in-place** (the Phase 3 progress entries above are the canonical style — outcome verdict, headline metrics, smoking-gun observations, links to the closure report under `eval/reports/`, patch-stack delta).
2. **Replace the active brief below with a fresh paste-and-go brief** for the next stage/probe, sized for a cold start: assume the next session has zero conversation context, no idea what files just changed, and no live agentchrome tab. The brief must include:
   - A single-line goal + single-line context (what just shipped, what's next).
   - A **paste-and-go bootstrap block** that verifies state in 30 seconds: working-tree tip, llama.cpp branch + tip, smoke server status, agentchrome session port + reusable tab id (with fallback to `tabs create` if absent), and a baseline replay step that prints expected pre-fix markers (so the operator confirms they're starting from the right place).
   - The implementation steps: files to read first (with line ranges), files to touch (with brief role per file), code sketches where useful.
   - The exit criteria as a checklist that maps onto explicit log markers from the spike / harness output.
   - A "branch on outcome" section listing what to do if the fix flips, partially flips, or doesn't flip.
3. **Collapse the prior brief to a one-line "Earlier Stage X.Y brief — collapsed (full text in closure report)"** pointer so the active surface stays focused. The full brief lives in the `STAGE-X.Y-RESULT.md` closure report under `eval/reports/<probe>/`.
4. **Commit cadence stays:** `docs(reports): Stage X.Y closure — <one-line>` + `docs(TODO): Stage X.Y closed — queue Stage X.Z <one-line>` as separate commits before any implementation work for the next stage starts (per the "Always commit before work" workflow policy in `CLAUDE.md`).

The discipline exists because every closing TODO update is the *handoff packet* for the next session — even if "the next session" is the same operator 10 minutes later. Treat it as if a teammate walking in cold has to pick up where you left off.

### Phase 3 trajectory assessment (set 2026-05-07, post-Stage-4.26)

Read this **before** starting Stage 4.27 to decide whether the
investigation is still load-bearing or scope-reducible.

**The single outstanding bug.** JSEP spike produces `"inonic boso-"`
from `"The capital of France is"`; the non-JSEP reference probe
produces coherent text from the same prompt with the same WGSL
kernels. Only scheduler routing differs.

**Investigation status.**
- **Ruled out (Stages 4.22 → 4.26):** Q4_K dequant correctness
  (H-3b), WGSL accumulation precision (H-3b-structural), Kahan
  recovery (zero impact), libllama matmul precision
  (H-4-libllama-imprecise — libllama is *worse* than webllm).
  Five stages ruled out matmul-kernel and matmul-precision
  hypotheses entirely.
- **Last actual signal we have:** Stage 4.17 checkpoint diff
  (2026-05-07 morning) — `attn_out-0` first ≥1e-3 at idx 11,
  `ffn_norm-0` at 1.83e-1 (idx 12), `result_norm` 5.83 (idx 14),
  `result_output` 6.61 (idx 15). The patch stack has grown
  12 → 13 since with multiple host-mirror writeback fixes
  (Stages 4.4 / 4.5 / 4.16), so that smoking-gun table is
  potentially stale.
- **Genuinely unknown right now:**
  - Whether the cascade is in prefill or decode.
  - Whether the failing op is a matmul, RMSNorm, FA, softmax,
    SET_ROWS, or KV-cache layout.
  - Whether Stages 4.4 / 4.5 / 4.16 host-mirror fixes already
    shifted the picture (the diff may be tighter than 4.17's).
  - Whether the bug is a structural cross-backend issue
    (CPU-fallback writeback gap, wrong buffer offset,
    gain-vector misload) — those are the remaining suspect
    categories now that all numerical hypotheses are dead.

**What Stage 4.27 buys.** A re-captured `__stage417Checkpoints`
diff. **One probe, ~5 minutes of work.** Three branches:

| Outcome | Diagnosis | Estimated remaining work |
|---|---|---|
| Same pattern as Stage 4.17 (`attn_out-0` / `ffn_norm-0` first divergent) | Post-4.4/4.5/4.16 patch stack didn't close the cascade; bug structurally identical | 2-3 more probes to localize the failing op |
| Tighter than Stage 4.17 (no prefill checkpoint reaches 1e-2) | Cascade is in decode (lm_head / KV-cache / RoPE / SET_ROWS); pivot to `LOGIT_STATS_STEP0` diff | 2-3 probes |
| Different first-divergent op (e.g. `Vcur-0` or `Kcur-0` now ≥1e-2) | Patch-stack regression from Stages 4.18-4.26 | Worst case ~6 probes (bisect) |

**Risk assessment.** The trajectory has the hallmarks of a long
tail: 26 sub-stages since Stage 3.5 nominally closed, each ruling
out one hypothesis. The remaining suspect categories are
**structural** (cross-backend writeback, KV layout, gain-vector
load), not numerical, so Stage 4.27 *should* land a sharper
signal — but if it doesn't, the realistic envelope is another
5-10 stages.

**Two paths forward:**

1. **Run Stage 4.27 next session.** Cheap, high information value;
   tells you within ~30 minutes whether you are 2 probes from done
   or 10 probes from done.
2. **Step back and reassess JSEP scope.** The Phase 3 cycle exists
   because the JSEP / Option A-prime path was meant to unlock
   `llama_decode` integration on top of webllm's WGSL kernels.
   The non-JSEP reference path already decodes correctly and is
   not blocked. Open question: **is JSEP load-bearing for the
   project's actual use case** (agent + Three.js, 8B ceiling,
   single-active-model)? If the non-JSEP `webllm-wasm.js` route
   already meets the perf and feature budget on the canonical 6,
   Phase 3 may be a path with no shipping requirement behind it,
   and the entire cycle could be deferred or scoped down. The
   project-level benefit of finishing Phase 3 (= upstream
   `llama_decode` schedules our WGSL kernels for free, instead
   of the hand-rolled forward-pass builder in
   `src/inference/engine.ts`) is real but **not load-bearing**
   for any committed use case. Worth an explicit go / no-go
   decision before sinking another 5-10 stages.

**Recommendation:** run 4.27 first (it's effectively free), then
let the diff data drive the go / no-go conversation.

**Update post-Stage-4.27 (2026-05-07 afternoon).** Stage 4.27 ran and
landed Outcome **A** — the cascade is structurally identical to Stage
4.17, *not* tighter and *not* different. Estimated remaining work:
"2-3 more probes to localize the failing op" per the table above.
Stage 4.28 (Probe 15) is the first of those probes; it tests the
two highest-prior surviving hypotheses (output-projection weight
+ gain-vector byte-integrity) in a single allowlist-extension pass.
Re-evaluate the JSEP go / no-go after Stage 4.28's outcome —
Probe 15 either localizes the bug (closes the cycle in one more
probe) or eliminates suspects 1+2 and forces Stage 4.29 to extend
`node_dump_cb` for full-tensor stats (a structural change to the
checkpoint framework, justifying a fresh trajectory assessment).

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

Per-stage stubs (Stage 3 + Stage 4.1 → 4.36) archived to
[`TODO_ARCHIVE.md`](TODO_ARCHIVE.md) — see *Phase 3 JSEP causal-LM
decode investigation* section. Per-stage closure reports stay under
`eval/reports/p2-v2-option-a-prime-2026-05-06/`.

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
