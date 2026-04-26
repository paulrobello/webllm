# WebLLM Project Status & Roadmap

> **Date:** 2026-04-26
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
> **Wave 2 begun 2026-04-26 (Completed §12):** 1/4 done.
> - **mistral-7b-instruct-v0.3-q4ks**: 34.4 tok/s steady-state
>   / 26/36 (68%). 32 layers, GQA 4:1, KV 1024 MB, 650
>   dispatches/token, matmul 47.0% of graph (~45% of decode —
>   highest in fleet). First non-Llama/Qwen entry. Q3_K_M
>   tried first; gibberish output uncovered **bug #28: Q3_K
>   matmul shader has a correctness bug.** Wave-1 never
>   exercised Q3_K (all Q4_0); §9 tested Q4_K_M only. Q4_K_S
>   workaround works (same K-quant family as Q4_K_M).
>   Practical wave-2 ceiling at the current llama.cpp upstream
>   is 7B Q4_K_S; Llama-3.1-8B / Qwen3-8B exceed the 4 GiB
>   WASM cap at Q4_K_S (~4.5 GB) and route through the broken
>   Q3_K shader at Q3_K_S. Either fix #28 or bump
>   MAXIMUM_MEMORY to 8 GB to admit them.
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

1. Embedding lookup used `opCpy` Q4_0→F32 (unsupported; replaced with `ggml_get_rows`).
2. Leaf input data (`posTensor`, `tokenIdsTensor`, mask) must be written with `backendTensorSet` *after* `backendAllocCtxTensors`.
3. SPM tokenizer: ▁ normalization (encode + decode), code-point iteration, byte-fallback via `<0xHH>` text.
4. KV writes were orphaned by `graph_build_forward_expand` (unreachable from logits) — now explicitly expanded per layer.
5. KV writes ordered BEFORE attention reads in the graph node list.
6. RMSNorm gamma was never multiplied in — now applied at all three norm sites (attn, ffn, final).
7. Custom `GGML_OP_DIAG_MASK_INF` shader broken past head 0; later replaced by `ggml_soft_max_ext` with explicit causal mask tensor.
8. **V cache permute used wrong `ggml_permute` arguments** — silent shape mismatch in subsequent cpy scrambled V values. Fixed `(2, 0, 1, 3)` → `(1, 2, 0, 3)`.
9. WASM build -O1 → -O3 (3.4MB → 1.77MB).
10. Sampling wired in via `Sampler` class (temp / top-k / top-p / repetition penalty).
11. `ggml_soft_max_ext` + `op_get_rows` WASM bindings added.
12. Multi-turn chat garbled output — TinyLlama without a system message interprets Zephyr markers as comparison operators. Fixed by auto-prepending DEFAULT_SYSTEM in `formatChatPrompt`.
13. GPU TOP_K decode path reshaped logits as `[vocab, 1]` before `ggml_get_rows`; ggml gathers along row dimension, so the graph produced `[vocab, topK]` and later failed reshape assertions. Fixed by reshaping logits to `[1, vocab]` before `opGetRows`.
14. Dashboard Temperature sweep hot series produced data but could render invisibly because `CHART_COLORS.red` was missing. Fixed by extracting shared temperature-sweep data construction and defining the hot color as `#f85149`.
15. **Encoder V permute tripped `ggml_mul_mat`'s `is_transposed` assertion** — `permute(v3, 1,2,0,3)` produced the right logical shape `[N, headDim, nHeads]` but left `nb[0] > nb[1]`. Wrapped in `opCont` to match llama.cpp's no-KV-cache BERT path.
16. **BERT WordPiece vocab follows llama.cpp's phantom-space convention** — `convert_hf_to_gguf.py::BertModel.set_vocab` rewrites the HF vocab so word-initial tokens gain a `▁` (U+2581) prefix and `##xyz` continuations have the `##` stripped. Our tokenizer was looking up `happy` / `##ful` (HF style) and missing every entry, producing `[CLS][UNK][SEP]` for every input. Rewrote `wpSubword` and `decodeWordPiece` to match `llm_tokenizer_wpm_session`. HF golden fixture in `tests/wordpiece-golden.test.ts` now guards it.
17. **`ggml-webgpu` silently no-op'd `GGML_OP_NORM`** — only `RMS_NORM` and `L2_NORM` were in `supports_op`, so LayerNorm fell through and the result buffer kept its zero-init contents. Combined with `GGML_CPU=OFF` (no scheduler fallback) every encoder forward produced bit-identical output regardless of input. Added a `LAYER_NORM` variant to `row_norm.wgsl` (Σx + Σx² in one pass), registered the pipeline, and dispatched `GGML_OP_NORM` through `ggml_webgpu_row_norm`. See `docs/LLAMA_CPP_PATCHES.md` patch #9.
18. **Bench-full smoke page hard-coded the causal-LM path at step [4/8]** — `ModelInference.loadWeights` failed on BERT GGUFs with `Weight "output_norm.weight" not found`. Page now branches on `arch === "bert"` and uses `EncoderInference`, skipping KV cache / generation / reference-encoder steps with explanatory pass logs.
19. **HF downloader picked Q4_K_M for arctic-embed** because the MLC-style `q0f32` defaultQuant didn't match any file in the GGUF repo and `q4_k_m` was first in the fallback list. Added a `ggufFilePattern?: string` field to `BenchmarkModel`; arctic-embed pins `"f16"` and the picker checks it ahead of `defaultQuant`.
20. **Tokenizer.encode("") returned `[]` for WORDPIECE** — bypassed the `[CLS] ... [SEP]` framing via an unconditional empty-string short-circuit in `encode()`. WORDPIECE now always frames; other tokenizer types keep returning `[]`.
21. **Score-over-time chart was blank despite a populated DB** — `renderSeriesChart` was defined but never invoked from the `render()` loop, so the panel always showed the bar-empty placeholder. Adding the call to the render loop (between `renderFinishChart` and `renderTable`) fixed it. Also fixed: `seriesLoaded` was sticky after the first fetch, so SSE-delivered evals were invisible to the chart; now reset on every `eval_complete` event. The category x-axis was missing its `labels` array, so even when called the points had nowhere to plot — now built from the sorted union of timestamps.
22. **Dashboard charts keyed on `modelId` collapsed Qwen thinking-on/off** — Temperature sweep, per-dimension grouped, and Accuracy×Speed scatter all shared a key for both Qwen modes; latest-wins silently overwrote one with the other. Group keys now include `thinking`; series labels gain a `" (think)"` suffix when thinking is on so non-thinking-capable models keep their existing labels.
23. **`engine.generateStream` qwen3-chatml wiring diverged from the smoke loop in 4 places** discovered while consolidating onto the library. Effects: (a) `maskedTokensWhileThinking` and `maskedTokensAfterThinkingUntilAnswer` were missing `<|endoftext|>`, so the model could emit it mid-think and either get a stray stop or pollute the chain-of-thought; (b) `maskedTokensAfterAnswerStarts` mistakenly included `<|im_end|>`, which is the chat EOS — the model could not terminate normally during the visible answer and qwen3 thinking-on always ran to `max-tokens`; (c) `<|endoftext|>` wasn't auto-added as a stop token; (d) the smoke loop's first-post-`</think>` leading-whitespace guard (forces `</think>` to be followed by a token starting with whitespace) had no library counterpart. Fixed all four; added `requireLeadingWhitespaceAfterThinking` to `GenerationConfig` for parity, and threaded the seed through `engine.generateStream`'s internal `Sampler` (added `CompletionConfig.seed`) so smoke runs are reproducible through the public API.
24. **`Generator.generate` computed `gpuMode` once, statically, before the decode loop** — `requiresFullLogitsSteering = (any qwen3 mask set configured)` forced `gpuMode = "full"` for the entire run. Once the smoke loop migrated onto the library, qwen3 thinking-off ran at ~17 tok/s on the full path instead of ~83 tok/s on the topk path, even on steps where no steering state was active. Replaced with per-step dynamic selection: `greedy` if sampler is greedy + no penalty; `topk` if `sampler.topK > 0` AND no current steering state (`thinkDepth === 0 && !waitingForVisibleAnswer && !hasVisibleAnswerText`); else `full`. The smoke loop's old code had this dynamic check inline; the library now matches.
25. **Qwen2 / Qwen2.5 attention biases were silently dropped, producing random-token output.** Discovered while running §10 wave-1 model 2 (`qwen2.5-1.5b-q4f16`): the smoke chat regression "passed" structurally but emitted gibberish (`"ña！" szerǃ yaboler...`) and accuracy collapsed to 1/36 = 4%. `eval/models.ts` resolved to `qwen2.5-1.5b-instruct-q4_0.gguf`, which carries `blk.<i>.attn_q.bias`, `attn_k.bias`, `attn_v.bias` tensors that **only the qwen2 architecture uses** (Llama, Qwen3, Mistral, etc. all leave Q/K/V projections unbiased). `ModelInference.loadWeights` only requested the `.weight` tensors, so Q/K/V values were off by a constant shift in every layer, polluting attention scores from the first prefill step. Fix: added `qBias`/`kBias`/`vBias: TensorPtr | null` to `LayerWeights`, conditionally loaded mirroring the existing `qNorm`/`kNorm` pattern (lines 140-145), and wrapped every `opMulMat` of qProj/kProj/vProj with `opAdd(bias)` when present in all three forward branches (prefill, decode, debug-checkpoint). Verified post-fix: same model produces `"Why don't scientists trust atoms? Because they're always splitting up!"`, finish=eos, accuracy 29/36 = **81%**. Dispatch count went from 573 to 657 (+84 = 3 ops × 28 layers, exactly matches the per-layer bias add). Regression coverage is the smoke chat regression itself — a unit-level test would have to mock 15+ wasm methods and only test mechanical wiring; the live bench output is the higher-signal check.
26. **Dashboard "Accuracy & tool-calling" panel listed embedding-only models with empty/zero rows.** `renderEvalDimensions()` and `renderEvalsTable()` in `smoke-test/dashboard.js` iterated over every eval, including embedding evals whose only dimension is `"embedding"`. The result: each arctic-embed run rendered as either a single embedding bar surrounded by null space (cards) or a row whose only dimension chip read `embedding: 1/1 · 100%` (table) — not the panel's intent, and duplicative against the dedicated Embeddings section that already shows cosine + latency + throughput. Same convention already existed in `renderDimGroupedChart()` at line 785 (`if (dims.length === 1 && dims[0] === "embedding") continue`); applied that pattern in `renderEvalDimensions`, `renderEvalsTable`, and the header `eval-count` badge in `renderEvals()` so all three reflect accuracy/tool-calling evals only.
27. **Smoke page silently mis-bucketed `?thinking=1` runs on non-thinking models.** Llama, SmolLM2, Qwen2/2.5, etc. don't have `<think>`/`</think>` token IDs and don't reference `enable_thinking` in their chat templates. With `?thinking=1` set, the smoke page's `thinkingEnabled` flag still flowed through to: (a) `maxTokens 1024` instead of 64 (16× the decode budget for runs that can't terminate via `</think>`), (b) the run's recorded `thinking: "on"` field which polluted the dashboard's thinking-on/off comparison panels with non-thinking runs, and (c) the `assistantText` display path. The engine itself was safe — `isQwenChatml` gating in `engine.ts:240-296` plus `shouldCloseThinkBlock` gating in `chat-template.ts:107` meant no thinking-mask wiring or `<think>` template injection actually ran for non-qwen3 models. But the cosmetic and dashboard-level effects were still wrong, and the mis-routed runs were hard to spot. Fixed by adding `modelSupportsThinking(parsed)` to `smoke-test/real-model-smoke.js` (returns true iff the chat template references both `enable_thinking` and `<think>`, mirroring the engine's gate; encoders short-circuit to false). The smoke page checks this immediately after [2/8] parse and rejects with a clear error message before any GPU/WASM init happens — fail-fast, no wasted work. Verified end-to-end via agentchrome on tinyllama (rejects after [2/8] with the new error) and qwen3-0.6b (still progresses to [7/8] with thinking enabled). Regression test in `tests/real-model-smoke.test.ts` covers Qwen3 (true), Qwen2/Llama/BERT (false), partial-marker templates (false), and missing-field defensiveness.
28. **Q3_K matmul kernel produces gibberish output in ggml-webgpu.** Discovered while bringing up the first wave-2 model (`mistral-7b-instruct-v0.3-q3km`, 3.36 GB Q3_K_M). Loader path streamed the GGUF cleanly through the §11 WASM-heap callback, speed numbers came out clean (profile-mode 21.4 tok/s · steady-state 25.2 tok/s · matmul 26.91 ms / 59.3% of graph · 650 dispatches/token), but assistant text was pure noise from token 1 (`�t2rhtt […]hetttilh […]ttttshttttttttlugusus…`) — same structural symptom as bug #25 (qwen2 missing biases) but Mistral has no biases. Verified non-causes: GGUF metadata reads cleanly (`llama.rope.freq_base = 1000000.0`, vocab 32768, RMS eps 1e-5, file_type 12 = LLAMA_FTYPE_MOSTLY_Q3_K_M); `supports_op` covers `GGML_TYPE_Q3_K` for both MUL_MAT and GET_ROWS in upstream `ggml-webgpu.cpp`; `ggmlTypeSize` table correctly reports `110/256` bytes/elem; `MUL_ACC_Q3_K` shader exists in `mul_mat_vec.wgsl`. Architecture is data-driven (`general.architecture = "llama"`, no Mistral-specific branch needed; chat template detects as `llama2` for [INST]/[/INST]). Repeated the same model at Q4_K_S (3953 MB, same K-quant family that §9 verified works via Q4_K_M): output is fully coherent with correct factual answers — Q4_K_S inference runs at steady-state 34.4 tok/s with 62% faster matmul (16.21 ms / 47.0% of graph) and 26/36 = 68% accuracy on bench-full. **The Q3_K shader has a correctness bug**, not the loader, parser, model arch, tokenizer, or chat template. Wave-1 never exercised this code path (all entries pinned to Q4_0); §9's K-quant test was Q4_K_M only. Q3_K_M skipped as a wave-2 quant; if a future workload needs it, the bug investigation starts in `~/Repos/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl::MUL_ACC_Q3_K` (110-byte super-block, 16-thread cooperative load, scale unpack via `s_shift1`/`s_shift2` masks `0x0F0F0F0Fu` | `0x30303030u`). The Q3_K_M GGUF still on disk at `smoke-test/models/mistral-7b-instruct-v0.3-q3km.gguf` (3.5 GB) can be re-pinned later without re-downloading. **Workaround: pin Q4_K_S or Q4_K_M for 7B+ entries** (Q4_0 is over the 4 GiB WASM cap at 7B+, so K-quants are forced).

---

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

### Completed on 2026-04-24

- Fixed GPU TOP_K row gathering by reshaping logits to `[1, vocab]` before
  `opGetRows`.
- Added `WEBLLM_ASSERTIONS` / `wasm-build-debug` for preserving WASM abort
  diagnostics when needed.
- Added the `semantic-reasoning` eval dimension and moved chat-style semantic
  tasks out of the true embedding-vector track.
- Added cosine-similarity scoring helpers and regression tests for embedding
  vector scoring.
- Captured eval sampler/context params so the dashboard can bucket temperature
  and show run details.
- Migrated live dashboard charts to self-hosted Chart.js and implemented
  accuracy × speed, per-dimension grouped bars, temperature sweep, Qwen
  thinking deltas, TTFT, finish reasons, and score-over-time.
- Fixed the Temperature sweep hot bucket rendering regression with shared
  chart-data tests.
- Documented model support and follow-up roadmap in `docs/MODEL_SUPPORT.md`.
- Wired the public streaming APIs (`WebLLM.generateStream()` and
  `WebLLM.chatCompletion()`) through `Generator` + `InferenceSession`, with
  stop-token handling, abort metadata, Qwen thinking controls, and tests in
  `tests/engine-streaming-api.test.ts`.
- 2026-04-24: Encoder forward pass shipped. `WebLLM.embed(modelId, text)`
  drives a BERT-style bidirectional graph (input embed + post-norm
  attention + GeLU FFN, mean/CLS pooling, L2-normalized output) on the
  Arctic-Embed-s GGUF; smoke step `[8/8]` passes with `cosine('happy',
  'joyful') ≈ 0.77`. Bring-up uncovered three load-bearing fixes:
  V-permute → `opCont` to satisfy `ggml_mul_mat`'s `is_transposed`
  assertion; tokenizer rewritten for llama.cpp's phantom-space BERT
  vocab convention (HF golden fixture in `tests/wordpiece-golden.test.ts`
  guards it); and `GGML_OP_NORM` added to the patched ggml-webgpu
  backend (commit `68f1738d5`, see `docs/LLAMA_CPP_PATCHES.md` patch #9).

### Completed on 2026-04-25

- Wired arctic-embed-s/m profiles into `make bench-full`. New
  `embedding?: boolean` flag on `SmokeProfile`; `bench.ts` skips the
  `chat-smoke` (speed) phase for embedding profiles; `browser-eval.ts`
  auto-restricts embedding-only models to the embedding dimension.
  `eval/models.ts` gains a `ggufFilePattern?: string` field so the HF
  downloader pins the verified F16 GGUF instead of falling through to
  Q4_K_M. `smoke-test/real-model-page.js` branches on `arch === "bert"`
  and routes through `EncoderInference`; `engine.adoptPreloadedModel`
  now accepts either inference type. End-to-end:
  `8/8 tasks passing · overall 93%` for arctic-embed-s.
- HF golden WordPiece fixture: `scripts/extract-bert-vocab.ts` dumps
  the real Arctic-Embed-s vocab (30522 tokens) to JSON;
  `scripts/generate-bert-golden.py` generates HF reference encodings
  for 16 probe strings (single words, multi-word, subword splits,
  casing, accents, punctuation carve-outs, empty string);
  `tests/wordpiece-golden.test.ts` asserts byte-for-byte parity.
- Dashboard restructure:
  - New "Embeddings" section with cosine, median latency, and
    throughput panels.
  - Per-dimension grouped chart now excludes embedding-only evals and
    drops the `embedding` column; renders `null` (not `0`) for
    dimensions a model wasn't scored on.
  - Temperature sweep, per-dimension grouped, and Accuracy×Speed scatter
    now key on `(modelId, thinking)` so Qwen thinking-on and
    thinking-off render as distinct rows / colours.
  - Accuracy × Speed scatter coloured by model with the chart's own top
    legend acting as the colour key.
  - Score over time: wired into the render loop (was defined but never
    called), category x-axis given a labels array built from the sorted
    union of timestamps, `seriesLoaded` cache invalidated on every
    `eval_complete` event.
- TODO.md records an Apr-23 ~50% smoke-bench tok/s drop; bisect attributed
  it to the encoder-innocent commit `5542bef`. See Next Steps §1 for the
  2026-04-25 root-cause finding (sampler-config change in the smoke page,
  not an engine regression).
- Re-baselined item 11 with the `?slowpath=1` URL gate (temporary, not
  committed): full 32 K-vocab readback costs only ~0.1 ms/token over the
  4-byte ARGMAX readback even post-async-readback. The "negligible gain"
  framing for `forwardDecode("greedy")` is correct. The entire ~10 ms/token
  gap between greedy and realistic sampling lives in the JS sampling
  pipeline.
- `perf(smoke): route realistic-sampler decode through GPU TOP_K path`
  (commit `9156deb`). Added a topk middle branch in
  `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` that
  calls `inference.forwardDecode(..., "topk", sampler.topK)` and feeds
  the reduced indices/values into `sampler.sampleFromTopK(...)`. Gated
  to skip when qwen masking/thinking state is active. **Measured impact
  (TinyLlama Q4_0, 3 trials median): 52.9 → 110.7 tok/s (2.1×)**;
  recovers 96% of the way to the greedy upper bound (114.8 tok/s).
  Qwen3 thinking-off also benefits (~76 tok/s); thinking-on routes
  through the unchanged full path (~16.6 tok/s).
- **TODO §2 done — library is now the single source of truth for
  decode** (committed as `6865a2c`). The smoke decode loop
  in `createSmokeCompletionRunner` was a 200-line duplicate of
  `Generator.generate` that silently dropped throughput when the topk
  fast path landed on one side but not the other. Consolidation steps:
  - **Library fixes (Phase 1):** `engine.generateStream` qwen3 wiring
    parity (bug-fix #23 above); `requireLeadingWhitespaceAfterThinking`
    added to `GenerationConfig` and wired through
    `Generator.generate`'s post-`</think>` sampling guard.
  - **Library extension (Phase 2A):** `CompletionConfig.seed` added to
    `src/core/chat-types.ts`; `engine.generateStream` threads it into
    the internal `Sampler({ ... seed })` construction.
  - **Library refactor (Phase 3):** dynamic per-step decode-mode
    selection in `Generator.generate` (bug-fix #24 above).
  - **Smoke-side rewrite (Phase 2B):**
    `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` now
    a ~50-line adapter over `engine.chatCompletion`. Deleted 11 dead
    helpers (`getForbiddenReentryTokens`, `getThinkingTokenIds`, the 3
    qwen `getMaskedTokens*`, `getExtraStopTokenIds`, `maskTokenLogits`,
    `isVisibleTextToken`, `isWhitespaceOnlyTextToken`,
    `tokenStartsWithWhitespace`, `decodeForDebug`).
    `smoke-test/real-model-page.js` constructs the WebLLM engine +
    `adoptPreloadedModel` once after [6/8] (covers both causal-LM and
    encoder paths); reused by [7/8], the interactive chat box, and
    bench mode. `smoke-test/real-model-bench.js` accepts engine +
    handleId from caller (no longer creates its own GPU adapter +
    engine). `smoke-test/real-model-runtime.js` drops manual prompt
    tokenization + KV reset; passes the full message array through.
    `tests/real-model-runtime.test.ts` rewritten for the new signature.
  - **Browser smoke verification (this session, single-run on tab
    `52C698CC3FF17A7A9B85EC5CB5EC67E2`, port 50840):** tinyllama
    106.9 tok/s · 64 tokens · finish=max-tokens; qwen3 thinking-off
    83.4 tok/s · 25 tokens · finish=eos; qwen3 thinking-on 17.3 tok/s
    · 236 tokens · finish=eos; embed cosine=0.76 on all three. Output
    text byte-identical across two tinyllama re-runs (seed=12345).
    Console: no errors. `make checkall`: 390 pass / 5 skip / 0 fail
    across 43 files. Committed as `6865a2c`.
- **Profile harness re-baselined** (commit `953c560`). Three latent
  bugs in `eval/perf.ts` had been silently masking every `--profile`
  run since the consolidation: greedy-only trace filter dropped the
  realistic-sampler topk traces; `buildSmokeTestUrl` API drift swallowed
  `?perf=` and `?profile=1`; and `fetchDecodeTraces` couldn't parse
  agentchrome's `output_file` overflow envelope (>16 KB results).
  Fixed all three, added `--thinking` to perf.ts, then captured the
  fresh medians that now drive the "Inference Performance
  Optimizations" preamble.

### Completed on 2026-04-26

1. **§10 wave 1, model 1: smollm2-360m-q4f16 registered + benched.**
   First entry in the large-model size campaign. Smallest registered
   generative model; ultrafast-tier reference point.
   - **Profile registered:** `smollm2-360m-warm` (temperature 0.6,
     `DEFAULT_PROMPT`); added to `SMOKE_PROFILE_SETS.full`.
   - **Repo fix:** the registered `huggingface-quants/SmolLM2-360M-
     Instruct-GGUF` returns HTTP 401 (gated/missing as of 2026-04-26);
     repointed `eval/models.ts` to `bartowski/SmolLM2-360M-Instruct-GGUF`
     and pinned `ggufFilePattern: "Q4_0"` so the picker doesn't fall
     through to Q4_K_M. Q4_0 keeps the cross-family GEMV comparison
     honest against `tinyllama-1.1b-chat-q4_0`.
   - **Architecture (from GGUF metadata):** llama arch · 32 layers ·
     n_head 15 · n_head_kv 5 (GQA 3:1) · embedding_length 960 ·
     head_dim 64 · context_length 8192 (we run at ctx=4096). File
     size 219.1 MB (Q4_0). KV cache at ctx=4096 ≈ 320 MB
     (`2 × 32 × 5 × 64 × 4096 × 4`).
   - **Speed (3-trial median, `eval/perf.ts`):**
     - **Steady-state 106.2 tok/s** (runs: 106.2 / 103.1 / 106.7)
       — within noise of TinyLlama-1.1B Q4_0 (~107 tok/s) despite
       3× fewer params. The speed gap collapse is consistent with
       **encode-overhead dominating at small scale**: SmolLM2's
       32-layer dispatch count (651/token) eclipses TinyLlama's
       22 layers (450/token) and Qwen3-1.7B's 28 layers (629/token).
     - **Profile-mode 75.6 tok/s** (perturbation -29%, identical
       to TinyLlama Q4_0's perturbation factor).
   - **Profile-mode backend attribution (60-step decode):**
     - `backendMatmulMs`: 3.11 mean / 27.9% of graph
     - `backendEncodeOverheadMs`: **3.70 mean / 33.2% of graph
       — leads matmul. First model in the fleet where this is true**
       (TinyLlama: matmul 33% > encode 28%; Qwen3-1.7B: matmul 34% >
       encode 22%). Implication: at this scale dispatch overhead is
       the bigger lever than matmul tuning, which lines up with the
       §6–§9 "matmul kernel tuning has bottomed out" finding from
       the other end of the size axis.
     - `backendAttentionMs`: 0.49 mean / 4.4%
     - `backendDispatchCount`: **651/token** (highest in fleet).
   - **Smoke chat regression:** PASSED. Output: `"Why did the tomato
     turn red? Because it saw the salad dressing!"` — finish=eos,
     21 tokens, no console errors.
   - **Accuracy (`bench-full --profiles smollm2-360m-warm`):**
     **24/36 passing · overall 62%** — lowest accuracy in the fleet,
     expected at 0.36B. Tool-calling skipped (temp 0.6 > 0.4 gate);
     embedding skipped (model lacks the capability). Dashboard runs
     27 / evals 28 (was 26/27 pre-test). Dot landed in
     accuracy×speed scatter.
   - **Behavioural surprises:** none in correctness — chat template,
     tokenizer, KV/attention all clean. The interesting finding is
     architectural: **dispatch count grows faster than parameters
     across the small-model regime** (360M / 32 layers > 1.1B / 22
     layers), which inverts the encode-vs-matmul fraction split.
     Worth re-checking after wave 1's 1.5B+ entries to see whether
     this is a scale crossover or a SmolLM2-specific design choice
     (32 layers at 0.36B implies an unusually deep+narrow shape:
     embedding_length 960 vs Qwen3-0.6B's 1024 at 28 layers).

2. **§10 wave 1, model 2: qwen2.5-1.5b-q4f16 registered + benched
   (after architectural fix).** First wave-1 model to expose a
   correctness gap.
   - **Profile registered:** `qwen2.5-1.5b-warm` (temperature 0.6,
     `DEFAULT_PROMPT`); added to `SMOKE_PROFILE_SETS.full`.
   - **Repo + quant:** `Qwen/Qwen2.5-1.5B-Instruct-GGUF` mirror is
     open. Pinned `ggufFilePattern: "Q4_0"` to skip the picker's
     Q4_K_M fallback (Q4_K_M was a -4% regression on Qwen3-1.7B per
     §9; Q4_0 also matches TinyLlama and SmolLM2 wave-1 quant for
     clean cross-family GEMV comparison). File 1016.8 MB.
   - **Architecture (qwen2 / GGUF metadata):** 28 layers · n_head 12
     · n_head_kv 2 (GQA 6:1, the most aggressive in fleet) ·
     embedding 1536 · head_dim 128 · ffn 8960 · ctx_max 32768 (we
     run at 4096). KV cache @ ctx=4096 ≈ 224 MB.
   - **First-run finding (broken): qwen2 attention biases were
     silently dropped.** `attn_q.bias` / `attn_k.bias` / `attn_v.bias`
     tensors exist in qwen2 GGUFs but our `ModelInference.loadWeights`
     only requested `.weight`. Result: gibberish output (`"ña！"
     szerǃ yaboler..."`), accuracy 1/36 = 4%. See bug-fix #25 above
     for the full diagnosis and fix.
   - **Post-fix re-bench (after bias support landed):**
     - Output: `"Why don't scientists trust atoms? Because they're
       always splitting up!"` — coherent, finish=eos, 22 tokens
       (was 64-token max-tokens with gibberish pre-fix).
     - Accuracy: **29/36 = 81%** (was 4% pre-fix). Within range of
       Qwen3-1.7B's 82-89% per-profile band; +14 points over
       SmolLM2-360M's 62%, consistent with 4× larger param count.
     - Speed (3-trial median):
       - Steady-state **84.3 tok/s** (runs: 83.9, 84.3, 85.2).
       - Profile-mode **57.6 tok/s** (perturbation -32%).
     - Profile-mode backend attribution (63-step decode):
       - `backendMatmulMs`: 5.53 mean / **40.1% of graph** —
         highest matmul fraction in fleet.
       - `backendEncodeOverheadMs`: 4.30 mean / 31.2% — high but
         second to matmul.
       - `backendAttentionMs`: 0.44 / 3.2%.
       - `backendDispatchCount`: **657/token** (+84 from the
         pre-bias-fix 573, exactly 3 ops × 28 layers — confirms
         every q/k/v bias add lands in the graph).
   - **Architectural finding: qwen3 vs qwen2 dispatch delta is
     almost exactly the cost of Q-norm + K-norm.** Pre-fix qwen2.5
     reported 573 dispatches/token; Qwen3-0.6B/1.7B both report
     629 at the same 28 layers. 629 - 573 = 56 = 2 ops × 28 layers,
     matching Qwen3's distinguishing feature (per-head Q-norm and
     K-norm). After bias support, qwen2.5 reports 657 — 28 more
     than Qwen3 because Q3 has biases too? No: Qwen3 doesn't bias
     Q/K/V (its weights confirm this). 657 - 629 = 28 = the bias
     add we now do for qwen2 (3 adds × 28 layers = 84 total; but
     dispatches per token is 657 - 573 = 84, which adds to a base
     where Qwen3-style q-norm/k-norm aren't done). Net: Qwen2 path
     adds 84 dispatches; Qwen3 path adds 56. Either way, the dispatch
     budget tracks per-layer-extras precisely.
   - **`SMOKE_PROFILE_SETS.full` entry kept** (the 4% accuracy dot
     ingested before the fix is now superseded by the 81% dot from
     the post-fix re-run; dashboard latest-wins handles it).

3. **Dashboard "Accuracy & tool-calling" panel cleanup.** Filtered
   embedding-only evals out of `renderEvalDimensions` (the cards),
   `renderEvalsTable` (the runs list), and the header `eval-count`
   badge in `renderEvals`. Same condition as the existing
   `renderDimGroupedChart` filter at line 785 (`dims.length === 1
   && dims[0] === "embedding"`). Embedding evals continue to render
   in the dedicated Embeddings section (cosine + latency +
   throughput).

4. **Dashboard sort/filter persisted to localStorage.** Sort and
   filter selections were lost on every reload. Added a small
   persistence layer keyed at `webllm-dashboard-filters/v1`:
   `loadPersistedFilters()` restores `sortKey`/`sortDir`,
   `evalSortKey`/`evalSortDir`, `thinkingFilter`, `textFilter`;
   `syncFilterControlsToState()` reflects loaded values into the
   `<select>` and `<input>` after restore; `persistFilters()`
   saves on every mutation site (4 in total). Defensive try/catch
   on both read and write — private-mode browsers / quota errors
   fall back to defaults silently. Verified end-to-end via
   agentchrome: set sortKey=oneShotTokensPerSec/asc + thinking=on
   + text=qwen → reload → all three restore (active sort header
   still shows `.sort-asc`, dropdown still reads "on", search
   input still reads "qwen"). No console errors.

5. **§10 wave 1, model 3: smollm2-1.7b-q4f16 registered + benched.**
   Same scale as Qwen3-1.7B but different family (llama arch)
   for a clean cross-family contrast at the 1.7B mark.
   - **Profile registered:** `smollm2-1.7b-warm` (temperature 0.6,
     `DEFAULT_PROMPT`); added to `SMOKE_PROFILE_SETS.full`.
     Mungert mirror is open; `ggufFilePattern: "Q4_0"` pinned for
     family parity (TinyLlama, SmolLM2-360M, Qwen2.5-1.5B all
     wave-1-pinned to Q4_0).
   - **Architecture (llama / GGUF metadata):** 24 layers (fewest
     among 1.5B+ entries) · n_head 32 · n_head_kv 32 (**no GQA!**
     full multi-head — one of the few in fleet) · embedding 2048
     · head_dim 64 (small/many heads, opposite design from
     Qwen2.5's wide GQA at 128/2). ffn 8192 · ctx_max 8192. File
     size 920.1 MB. **KV cache @ ctx=4096 = 1536 MB** —  by far
     the largest in fleet (vs 320 MB for SmolLM2-360M, 224 MB for
     Qwen2.5-1.5B). Direct consequence of `n_head_kv = 32`.
   - **Speed (3-trial median):**
     - Steady-state **86.3 tok/s** (runs: 86.8 / 86.3 / 83.7) —
       **31% faster than Qwen3-1.7B** (~66 steady) at identical
       1.7B params. Three reasons stack: 24 layers vs 28 (-14%),
       no per-head Q-norm/K-norm (-56 dispatches/token), and Q4_0
       vs Qwen3's Q8_0 (lower bandwidth). Net dispatch count 491
       vs Qwen3's 629 = -22%.
     - Profile-mode 57.7 tok/s (perturbation -33%, in-line with
       the fleet pattern).
   - **Profile-mode backend attribution (48-step decode):**
     - `backendMatmulMs`: 5.18 mean / 35.4% — comparable to
       Qwen2.5-1.5B (40.1%) despite the very different
       attention/KV shape. The 1.5B–1.7B class clusters at
       matmul = 35-40% of graph time.
     - `backendEncodeOverheadMs`: 2.96 mean / **20.2%** —
       **lowest in fleet** (TinyLlama 28%, SmolLM2-360M 33%,
       Qwen2.5-1.5B 31%, Qwen3-1.7B 22%). Few-but-heavy layers
       amortize encode overhead better than many-thin-layers.
     - `backendAttentionMs`: 0.46 mean / 3.2%.
     - `backendDispatchCount`: **491/token** — only +41 over
       TinyLlama's 450 despite 56% more params. The architectural
       win is clear: 24 layers × ~20 dispatches/layer ≈ 480, plus
       a few global ops, matches the 491 observed.
   - **Smoke chat regression:** PASSED. Output: `"Why did the
     bicycle fall over? Because it was tired of being flat!"` —
     finish=eos, 17 tokens.
   - **Accuracy (`bench-full --profiles smollm2-1.7b-warm`):**
     **27/36 passing · overall 74%**. Mid-range: above
     SmolLM2-360M (62%) and below Qwen2.5-1.5B (81%) /
     Qwen3-1.7B (82-89%). Consistent with public SmolLM2
     benchmarks — family trades quality for size/speed.
   - **Wave-1 cross-family pattern emerging:** at 1.5B-1.7B the
     llama/qwen2/qwen3 families cluster as: **SmolLM2 fastest +
     lowest accuracy → Qwen2 mid speed + mid accuracy → Qwen3
     slowest + highest accuracy**. Speed delta tracks dispatch
     count (-14% layers + per-head norm overhead in Qwen3) more
     than parameter count.

6. **§10 wave 1, model 4: qwen2.5-3b-q4f16 registered + benched.**
   First 3B-class entry; stress-tests the qwen2 bias path
   (bug-fix #25) at 2× the 1.5B scale.
   - **Profile registered:** `qwen2.5-3b-warm` (temperature 0.6,
     `DEFAULT_PROMPT`); added to `SMOKE_PROFILE_SETS.full`.
     `Qwen/Qwen2.5-3B-Instruct-GGUF` mirror open;
     `ggufFilePattern: "Q4_0"` pinned (1905.3 MB, well under
     WASM cap).
   - **Architecture (qwen2 / 36 layers):** n_head 16 · n_head_kv 2
     (GQA 8:1, even more aggressive than 1.5B's 6:1) ·
     embedding 2048 · head_dim 128 · ffn 11008 · ctx_max 32768.
     KV cache @ ctx=4096 = 288 MB (only +64 over 1.5B's 224 MB
     thanks to GQA 8:1 keeping n_head_kv flat at 2). +8 layers
     over qwen2.5-1.5b accounts for the dispatch-count delta
     (657 → 841 = +184 = ~23 ops/layer × 8 layers).
   - **Speed (3-trial median):**
     - Steady-state **45.1 tok/s** (runs: 44.8 / 45.7 / 45.1).
       1.87× slower than qwen2.5-1.5b (84.3 tok/s) — linear-ish
       scaling with parameter ratio (2×) at this size class.
     - Profile-mode 32.3 tok/s (perturbation -28%).
   - **Profile-mode backend attribution (39-step decode):**
     - `backendMatmulMs`: 8.91 mean / 34.4% — scales 1.67× from
       qwen2.5-1.5b's 5.32 ms (vs 2× param ratio → sub-linear,
       which is the bandwidth-bound matmul signature §6-§9
       characterized).
     - `backendEncodeOverheadMs`: 5.63 mean / 21.7% (was 3.69 at
       1.5B = +52%, tracks the +29% layer-count increase plus
       per-step overhead growth).
     - `backendAttentionMs`: 0.66 mean / 2.6%.
     - `backendDispatchCount`: **841/token** (highest in fleet
       to date).
   - **Smoke chat regression:** PASSED. Output: `"Why did the
     tomato turn red? Because it saw the salad dressing!"` —
     finish=eos, 14 tokens, no console errors. Same prompt that
     caused gibberish on qwen2.5-1.5b pre-fix; now coherent at
     2× the scale, confirming bug-fix #25 works generally for
     the qwen2 family.
   - **Accuracy (`bench-full --profiles qwen2.5-3b-warm`):**
     **32/36 passing · overall 86%** — within the Qwen3-1.7B
     band (82-89%) and the highest non-qwen3 entry in the fleet.
     +5 points over qwen2.5-1.5b (81%), consistent with the
     2× param scale for an instruction-tuned model.
   - **Architectural finding: matmul-bandwidth fraction holds at
     3B.** §9 characterized matmul as ≈40% of decode time on
     Q8_0 / ≈20% on Q4_0 at the 1.7B scale. At 3B Q4_0,
     matmul = 34.4% of graph time (graph = ~84% of step time)
     ≈ 29% of step time. That's modestly lower than 1.5B's
     ~38% of step (matmul% × graph%). Suggests the bandwidth-
     bound kernel-tuning ceiling (§A subgroup-cooperative
     loading) might still be worth ~10-12% of decode time at
     3B, slightly less than the 1.7B prediction. Re-evaluate
     once 4B (qwen3-4b) lands for the full size sweep.

7. **§10 wave 1, model 5: llama-3.2-3b-q4f16 registered + benched.**
   First non-qwen 3B-class entry; cross-family contrast against
   qwen2.5-3b at the same param scale.
   - **Profile registered:** `llama-3.2-3b-warm` (temperature 0.6,
     `DEFAULT_PROMPT`); added to `SMOKE_PROFILE_SETS.full`.
     Bartowski mirror open. Pinned `ggufFilePattern: "Q4_0."`
     (with trailing dot) to disambiguate against the ARM repack
     variants `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8` — those use a
     SVE/dot-product layout our shader can't decode.
   - **Architecture (llama / 28 layers):** n_head 24 · n_head_kv 8
     (GQA 3:1, much less aggressive than qwen2.5-3b's 8:1) ·
     embedding 3072 (wider than qwen2.5-3b's 2048) · head_dim 128 ·
     ffn 8192 (narrower than qwen2.5-3b's 11008) · ctx_max
     **131072** (32× the 4096 we run at — clear long-context
     headroom). KV cache @ ctx=4096 = **896 MB** — 3.1× larger
     than qwen2.5-3b (288 MB) due to less aggressive GQA, but
     still well under SmolLM2-1.7B's 1536 MB. File 1832.9 MB.
   - **Speed (3-trial median):**
     - Steady-state **58.2 tok/s** (runs: 60.0 / 58.2 / 57.0) —
       **29% faster than qwen2.5-3b** (45.1 tok/s) at the same
       param class. Three architectural differences stack: 28
       layers vs 36 (-22%), no per-projection biases (-84
       dispatches/token vs qwen2 path), wider/shallower vs qwen's
       narrower/deeper.
     - Profile-mode 37.9 tok/s (perturbation -35%, slightly above
       fleet's typical -28 to -33%).
   - **Profile-mode backend attribution (156-step decode):**
     - `backendMatmulMs`: 8.28 mean / 34.9% — almost identical
       to qwen2.5-3b's 8.91 mean / 34.4%. Despite Llama's wider
       hidden (3072 vs 2048) and Qwen2.5's deeper layer count,
       per-step matmul cost converges at the 3B Q4_0 scale.
     - `backendEncodeOverheadMs`: 3.67 mean / **15.5%** —
       **new fleet low** (was smollm2-1.7b's 20.2%). Fewer
       layers + bias-free + GQA 3:1 stacks to the smallest
       per-step encode cost we've seen.
     - `backendAttentionMs`: 0.51 mean / 2.1%.
     - `backendDispatchCount`: **572/token** — 32% lower than
       qwen2.5-3b's 841. The 184-dispatch delta breaks down as:
       28 layers × ~3 fewer ops/layer (no qwen2 biases, no extra
       norm path) = ~84 fewer; plus 8-layer count delta × 23
       ops = ~184. Sub-linear sum.
   - **Smoke chat regression:** PASSED. Output: `"A man walked
     into a library and asked the librarian, 'Do you have any
     books on Pavlov's dogs and Schrödinger's cat?' The
     librarian replied, 'It rings a bell, but I'm not sure if
     it's here or not.'"` — coherent and notably clever, 53
     tokens, finish=eos.
   - **Accuracy (`bench-full --profiles llama-3.2-3b-warm`):**
     **27/36 passing · overall 76%** — 10 points below
     qwen2.5-3b's 86% at the same param scale, mirroring the
     wave-1 cross-family pattern: **Llama family fastest with
     lower accuracy, Qwen family slower with higher accuracy**.
     The pattern is consistent across the 1.5-3B band now
     (smollm2-1.7b 74% vs qwen2.5-1.5b 81%; llama-3.2-3b 76%
     vs qwen2.5-3b 86%).
   - **3B-class speed/accuracy table (Q4_0):**
     | Model              | Layers | Disp/tok | tok/s | Accuracy |
     |--------------------|-------:|---------:|------:|---------:|
     | qwen2.5-3b         |     36 |      841 |  45.1 |     86%  |
     | llama-3.2-3b       |     28 |      572 |  58.2 |     76%  |

8. **§10 wave 1, model 6: hermes-3-llama-3.2-3b-q4f16 registered
   + benched.** Llama-3.2-3B fine-tune by NousResearch with
   tool-calling and structured-output capabilities; same arch
   as base llama-3.2-3b so no new arch work needed.
   - **Profile registered:** `hermes-3-llama-3.2-3b-warm`
     (temp 0.6, `DEFAULT_PROMPT`); added to
     `SMOKE_PROFILE_SETS.full`. Switched ggufUrl to bartowski
     mirror — NousResearch's mirror has only K-quants + Q8_0,
     no Q4_0. Pinned `ggufFilePattern: "Q4_0."` matching the
     base llama-3.2-3b convention.
   - **Architecture (llama / 28 layers):** identical to base
     llama-3.2-3b (n_head 24, n_head_kv 8 GQA 3:1, embedding
     3072, head_dim 128, ffn 8192, KV @ 4096 = 896 MB).
     Confirmed via dispatch count = 572 — exact match with base.
   - **Speed (3-trial median):**
     - Steady-state **60.4 tok/s** (runs: 56.6 / 60.4 / 62.0) —
       within 4% of base llama-3.2-3b's 58.2; the difference is
       run-to-run variance, not arch. Confirms fine-tuning a
       model doesn't change the inference cost profile.
     - Profile-mode 42.8 tok/s (perturbation -29%).
   - **Profile-mode backend attribution (138-step decode):**
     - `backendMatmulMs` 8.13 / 39.0% (vs base 8.28 / 34.9% —
       within noise; the % delta is from a slightly faster
       graph compute on this run).
     - `backendEncodeOverheadMs` 3.75 / 18.0%.
     - `backendAttentionMs` 0.49 / 2.3%.
     - `backendDispatchCount` **572** — exactly matches base
       llama-3.2-3b, as expected for a fine-tune.
   - **Smoke chat regression:** PASSED. Output: `"I told the
     ref I mixed up my shoes. He said, 'I don't care how you
     do it in your bedroom, just do it right!'"` — coherent,
     finish=eos, 47 tokens.
   - **Accuracy (`bench-full --profiles hermes-3-llama-3.2-3b-warm`):**
     **27/36 passing · overall 74%** — 2 points below base
     llama-3.2-3b's 76% at the same profile.  **Hermes's tool-
     calling advantage is invisible at warm temp** because the
     `tool-calling` dimension auto-skips above temp 0.4 (the
     gate on rigid JSON output). To see Hermes's specialized
     contribution, run with a cold profile or
     `--dimension tool-calling` override; deferred as a follow-
     up since cold/temp-sweep profiles were not part of the
     wave-1 campaign scope.
   - **Wave-1 finding:** the 3B band has **two empirically
     interchangeable llama-arch entries** (base + Hermes
     fine-tune) that produce statistically tied throughput on
     identical prompts — useful as a sanity check on the
     bench-full harness, less useful as a fleet diversification
     point. If a future campaign cares about tool-calling, run
     hermes-3 cold separately.

9. **Gemma 2 + Phi 3 deferred from wave 1 — architectural gaps
   identified.** Both families need substantially more
   inference-path work than the qwen2 bias fix did. Documented
   here so future work has a clear scope.

   **Gemma 2 (gemma-2-2b-q4f16) — needs all of:**
   - Pre-norm AND post-norm pairs for both attn and FFN (4 norm
     sites per layer instead of 2); requires `attn_post_norm`
     and `ffn_post_norm` tensors loaded + extra `opMul`/
     `opRmsNorm` calls in all 3 forward branches.
   - Logit soft-capping at output:
     `logits = soft_cap * tanh(logits / soft_cap)`. Requires
     `opTanh` WASM binding (not currently exposed) plus
     `final_logit_softcapping` metadata read.
   - Attention soft-capping inside the attention block,
     applied between Q·K^T and softmax. Same `opTanh` plus
     `attn_logit_softcapping` metadata.
   - **RMSNorm `(1 + weight)` scaling** — Gemma uses
     `weight + 1` while Llama uses just `weight`. Either patch
     in a Gemma-specific RMSNorm path or pre-bake `+1` into
     the loaded gamma tensor.
   - Sliding-window attention alternating with full attention
     (every other layer). Significant complexity; alternating
     attention masks per layer.
   - Bartowski mirror has only Q4_K_M (1.6 GB) + Q8_0
     (2.7 GB), no Q4_0 — would need a separate
     `ggufFilePattern: "Q4_K_M"` pin.

   **Phi 3 (phi-3.5-mini-q4f16) — needs at minimum:**
   - Fused QKV projection (`attn_qkv.weight` instead of
     separate `attn_q/k/v.weight`). Requires either splitting
     the fused tensor at load time or a fused-QKV forward path.
   - Verify FFN structure (Phi3 uses `gate_up.weight` fused
     vs the SwiGLU split llama uses).
   - Verify chat template alignment (we have `phi3` in
     `chat-template.ts:8` but inference path is untested).

   **Per §10 stop-conditions, both deferrals are recorded
   without code changes.** Bench-full was *not* run on either
   model; the architectural gaps are clear enough from tensor
   inventories and llama.cpp Gemma2/Phi3 sources that running
   them blind would just produce garbage and burn 1.6-3.8 GB
   of HF bandwidth per attempt. Adding either family is now a
   concrete future task with the inventory above as the spec.

10. **§10 wave 1, model 7 (final supported): qwen3-4b-q4f16
    registered + benched — wave 1 complete.** Largest model
    in fleet (4.0B, 36 layers, qwen3 arch with q/k norm).
    First model to require the GGUF-streaming-into-WASM-heap
    path (see §11 below). Highest accuracy in fleet at the
    cost of throughput.
    - **Profiles registered:** `qwen3-4b-warm` and
      `qwen3-4b-thinking-warm` (qwen3 family gets both modes
      per 0.6B/1.7B convention) in `eval/smoke-profiles.ts`;
      added to `SMOKE_PROFILE_SETS.full` and `qwen3-sizes`.
    - **Repo + quant:** `Qwen/Qwen3-4B-GGUF` (the official
      mirror) carries only K-quants + Q5/Q6/Q8 (no Q4_0).
      Switched to `unsloth/Qwen3-4B-GGUF` and pinned
      `ggufFilePattern: "Q4_0."` (trailing dot, llama-3.2-3b
      style — defensive against any future ARM-repack
      variants). File 2375.8 MB (Q4_0).
    - **Architecture (qwen3 / GGUF metadata):** 36 layers ·
      n_head 32 · n_head_kv 8 (GQA 4:1) · embedding 2560 ·
      head_dim 128 · ctx_max 40960 (we run at 4096) · vocab
      151936. KV cache @ ctx=4096 ≈ 144 MB
      (`2 × 36 × 8 × 128 × 4096 × 4 / 1048576 = 144`).
    - **Speed (3-trial median, `eval/perf.ts`):**
      - **Steady-state 35.5 tok/s** (runs: 35.3 / 35.5 / 37.9).
        Cleanest 3-trial spread in fleet.
      - **Profile-mode ~32 tok/s** (single trial, perturbation
        ~10% — much smaller fraction than smaller models since
        graph compute dominates more thoroughly here).
    - **Profile-mode backend attribution (18-step decode):**
      - `backendMatmulMs`: 10.54 mean / **35.6% of graph** —
        matmul leads decisively, consistent with the §6–§9
        bandwidth-bound matmul characterization at scale.
      - `backendEncodeOverheadMs`: 4.90 mean / 16.5% of graph
        — encode overhead's *fraction* keeps shrinking as
        scale grows (smollm2-360m 33% → qwen2.5-3b 18% →
        qwen3-4b 16.5%); *absolute* cost stays roughly flat
        across the fleet. Encode-overhead dominance is a
        sub-1B-class concern.
      - `backendAttentionMs`: 0.71 mean / 2.4% of graph
        (lowest fraction in fleet).
      - `backendDispatchCount`: **805/token** — matches the
        TODO §10 prediction ("a 4B model with ~36 layers
        would hit ~810/token"). Within 1% of the projection.
    - **Smoke chat regression:** PASSED. Output (off):
      `"Why don't skeletons fight each other? Because they
      don't have the guts! 😄"` — finish=eos, 19 tokens, no
      console errors. Thinking-on output also coherent with
      `<think>` block (~200 reasoning tokens) → punchline.
    - **Accuracy (`bench-full --profiles qwen3-4b-warm
      qwen3-4b-thinking-warm`):**
      - **Off: 32/36 = 88%.** Highest non-thinking accuracy
        in fleet (qwen2.5-3b held the prior record at 86%).
      - **Thinking: 33/36 = 90%.** Highest overall in fleet.
      Tool-calling skipped (temp 0.6 > 0.4 gate); embedding
      skipped (model lacks the capability). Both dots
      ingested into the live dashboard via SSE; verified
      `qwen3-4b-warm thinking=off overall=0.88 /
      qwen3-4b-thinking-warm thinking=on overall=0.90` via
      `/evals` API.
    - **Cross-family scaling pattern confirmed at 4B.**
      Wave-1 evidence is now consistent across 0.6B → 4B:
      Llama family fastest/lower-accuracy → Qwen family
      slower/higher-accuracy. qwen3-4b at 35 tok/s sits
      ~22% below qwen2.5-3b's 45 tok/s for ~30% more params,
      with comparable dispatch counts (805 vs 841). Speed
      delta tracks **matmul bandwidth** (qwen3-4b hidden=2560
      vs qwen2.5-3b hidden=2048; ~25% more bytes per matmul).
      Accuracy +2-4 points over qwen2.5-3b.
    - **Wave 1 final tally: 7/10 done · 2 deferred (gemma-2-2b,
      phi-3.5-mini per §9) · 1 optional skipped
      (qwen2.5-coder-1.5b — code-gen eval not in scope).
      Wave 1 complete.**

11. **GGUF streaming into WASM heap — unblocks all 4B+ models.**
    Discovered while attempting wave-1 model 7: a JS-side
    `new Uint8Array(N)` allocation caps at ~2 GiB on this
    Chrome (probed: 2000 MB OK, 2147 MB fails with "Array
    buffer allocation failed"). Q4_0 4B is 2266 MB — exceeded
    the cap before we even reached WASM. The previous fetch
    pattern also peaked at 2× file size (chunks-array +
    flat-buffer concat) which would OOM Chrome at ~1 GiB
    files due to memory pressure even when the single
    allocation would fit.
    - **Fix landed (this session):**
      1. **Parser API change.** `GgufParser.parse` and
         `ModelLoader.parseModel` now accept `Uint8Array`
         instead of `ArrayBuffer`. The parser uses
         `new DataView(data.buffer, data.byteOffset, data.
         byteLength)` so a sub-view at non-zero offset works
         correctly — guarded by a new sub-view regression
         test in `tests/gguf-parser.test.ts`.
      2. **`loadWeights` accepts a callback source.**
         `ModelInference.loadWeights` and
         `EncoderInference.loadWeights` accept either a
         `Uint8Array` (existing path) or a
         `(srcOffset, byteLength) => Uint8Array` callback.
         Required because `wasm.ctxCreate` and
         `backendAllocCtxTensors` can grow WASM memory,
         which detaches any pre-existing JS view of HEAPU8;
         the callback re-derives a fresh view from the live
         heap on each access.
      3. **`uploadRangeChunked` added to `GgmlWasm`.** New
         method takes the same callback. The internal 4 MiB
         scratch malloc inside the chunk loop can also
         trigger growth (and detach the source view between
         construction and `set`); `uploadRangeChunked`
         resolves the callback *after* the malloc, once per
         chunk, so the slice is always derived from the
         current HEAPU8.
      4. **Smoke loader streams into WASM heap.** Reordered
         steps: [1/8] WebGPU init → [2/8] Fetch
         (malloc model region in heap, stream chunks via
         `wasm.heapU8.set(chunk, ptr+off)`) → [3/8] Parse →
         [4/8] Load weights via the callback path. After
         loadWeights, `wasm.free(modelPtr)` reclaims the
         staging copy before KV cache + graph buffers
         allocate. View can exceed 2 GiB because views over
         a backing ArrayBuffer ≥ 2 GiB are allowed even
         when allocations aren't.
    - **Second fix: `ctxCreate` memSize was over-allocating.**
      Both `ModelInference.loadWeights` and
      `EncoderInference.loadWeights` were calling
      `wasm.ctxCreate(tensors.length * 16384 + ggufCtx.
      totalDataSize + 1MB)`. But `ctx_create` in
      `webgpu-bridge.cpp` sets `no_alloc=true`, so the
      ggml mempool only holds tensor *metadata* — actual
      tensor data lives in GPU buffers via
      `backendAllocCtxTensors`. Adding `totalDataSize`
      reserved a multi-GB unused buffer. For Q4_0 4B that
      was 2267 MB on top of the 2376 MB model staging copy,
      pushing total WASM allocation past the 4 GB cap.
      Removed `+ ggufCtx.totalDataSize` from both call
      sites; verified no regression on qwen3-0.6b
      (629 dispatches, matmul 3.78 ms — within noise of
      pre-fix). This fix likely also helps headroom on
      wave-2 7B+ entries.
    - **Verification:** all 393 unit tests pass (added 1 for
      sub-view parsing). qwen3-0.6b streams through the new
      path with no regression. qwen3-4b passed end-to-end
      smoke + 2 bench-full profiles. The `loadWeights`
      callback path is wired through to `uploadRangeChunked`
      only when invoked from the smoke loader; the
      `Uint8Array` path (engine.ts, tests, smoke-test/index.html
      synthetic-GGUF flow) still uses the original
      `uploadToTensorChunked` so the existing static-buffer
      callers are unaffected.
    - **What this unblocks:** all wave-2 candidates (7B+ at
      Q3_K_M, ~3 GB; 8B at Q3_K_S, ~3.4 GB) are now within
      the loader's reach. Remaining ceiling is the WASM
      4 GB cap itself, which gates how big a model + KV +
      activation working set can coexist. For an 8B Q3_K_M
      with KV ≈ 256 MB at ctx=4096, total ≈ 3.5 GB
      committed during load — close but possible.

12. **§10 wave 2, model 1: mistral-7b-instruct-v0.3-q4ks
    registered + benched.** First wave-2 entry; first 7B+ in
    fleet; first non-Llama/Qwen family. Two-attempt landing
    that uncovered Q3_K shader bug (#28).
    - **First attempt: Q3_K_M failed.** Pinned `Q3_K_M`
      (3.36 GB) for size headroom under the 4 GiB WASM cap.
      Loader streamed cleanly via §11; speed metrics looked
      normal (profile-mode 21.4 tok/s · steady-state 25.2 tok/s
      · 650 dispatches/token); but smoke chat regression
      "passed" structurally with **pure-noise output from
      token 1** (`�t2rhtt […]hetttilh […]…`). Same symptom
      shape as bug #25 (qwen2 biases) but Mistral has none.
      Triaged: GGUF metadata clean, `supports_op` covers
      Q3_K, `ggmlTypeSize` correct, `MUL_ACC_Q3_K` shader
      exists. Q3_K matmul kernel has a correctness bug —
      see bug #28 above. Wave-1 never exercised Q3_K (all
      Q4_0); §9 verified Q4_K_M only.
    - **Second attempt: Q4_K_S succeeded.** Re-pinned
      `Q4_K_S` (3953 MB, same K-quant family that §9 verified
      via Q4_K_M). Output coherent with correct factual answers
      (chemistry quiz: Al, Fe, Si, S — all correct).
    - **Profile registered:** `mistral-7b-v0.3-warm`
      (temperature 0.6, `DEFAULT_PROMPT`); added to
      `SMOKE_PROFILE_SETS.full`.
    - **Architecture (llama / 32 layers):** n_head 32 ·
      n_head_kv 8 (GQA 4:1) · embedding 4096 (widest in
      fleet) · head_dim 128 · ffn 14336 · ctx_max 32768 (we
      run at 4096) · vocab 32768 · `rope.freq_base = 1000000`
      (Mistral's higher base, vs Llama's 10000). KV cache @
      ctx=4096 = **1024 MB** (4× larger than Llama-3.2-3B's
      896 MB at the same n_head_kv=8 due to twice the layer
      count and embedding width).
    - **Speed (3-trial median):**
      - Steady-state **34.4 tok/s** (runs: 34.6 / 34.3 / 34.4
        — tightest spread in fleet).
      - Profile-mode **28.0 tok/s** (perturbation -19%,
        smaller than wave-1's typical -28% to -35%; graph
        compute dominates so much that profile overhead is a
        smaller relative slice at this scale).
      - Prefill **824 ms** (~10-token prompt + chat template).
    - **Profile-mode backend attribution (189-step decode):**
      - `backendMatmulMs` 16.21 mean / **47.0% of graph** —
        wave-1 ended at "matmul = 33-35% of graph"; at 7B
        Q4_K_S matmul jumps significantly. Combined with
        `graphComputeMs` 95% of step, **matmul is ~45% of
        decode time at 7B Q4_K_S**, vs wave-1's max ~33% at
        4B Q4_0. The §6–§9 bandwidth-bound matmul
        characterization holds qualitatively at scale, but
        the lever's percentage of total decode keeps growing.
      - `backendEncodeOverheadMs` 4.23 / 12.3% — encode
        overhead's *fraction* keeps shrinking (smollm2-360m
        33% → qwen2.5-3b 22% → qwen3-4b 16% → mistral-7b
        12%). Absolute cost stays nearly flat across fleet.
      - `backendAttentionMs` 0.62 / 1.8% (lowest in fleet).
      - `backendDispatchCount` **650/token** — 32 layers ×
        ~20 ops/layer matches; lower than qwen3-4b's 805
        despite +75% params, because Mistral has no
        biases / no per-head Q/K-norms.
    - **Smoke chat regression:** PASSED. Q4_K_S output
      includes coherent jokes (`Why was the math book sad?
      Because it had too many problems.`, `What do you call
      a fake noodle? An impasta!`, etc.) and factually
      correct chemistry-quiz answers in interactive mode.
      Cosmetic note: model emits stray `<</SYS>>` markers
      between turns (Llama-2 separator hallucination — the
      [INST]/[/INST] template detected as `llama2` lacks
      `<<SYS>>` for Mistral, but the model has clearly seen
      training data with both formats). Doesn't affect
      correctness; not a blocker.
    - **Accuracy (`bench --profiles mistral-7b-v0.3-warm`):**
      **26/36 = 68%** — below qwen3-4b's 88%, qwen2.5-3b's
      86%, llama-3.2-3b's 76%. Two factors stack: (a) Q4_K_S
      is more aggressive quantization than Q4_0 with measurable
      quality loss; (b) Mistral-7B-Instruct-v0.3 (Apr 2024)
      isn't as polished as Llama-3.x or Qwen3 — it's a
      first-generation instruct release. Tool-calling skipped
      (warm temp 0.6 > 0.4 gate); embedding skipped (model
      lacks capability).
    - **Lever-ceiling implication for §A subgroup-cooperative
      loading.** §9 measured matmul as ≈40% bandwidth-bound
      on Q8 (Stub B) at 1.7B scale; that's the fraction
      addressable by pure-bandwidth levers. At 4B the §A
      ceiling was ~13% of decode time; at 7B Q4_K_S, with
      matmul = 45% of decode and ~40% of that bandwidth-bound,
      the ceiling rises to ~18% of decode time. **Subgroup-
      cooperative loading becomes more attractive at 7B+
      scale.** Whether it's worth the engineering cost is
      still open until measured against actual workload mix.
    - **Cross-family scaling at 7B (Mistral vs all others):**
      | Model              | Layers | Disp/tok | tok/s | Accuracy | Quant |
      |--------------------|-------:|---------:|------:|---------:|-------|
      | qwen2.5-3b         |     36 |      841 |  45.1 |     86%  | Q4_0  |
      | llama-3.2-3b       |     28 |      572 |  58.2 |     76%  | Q4_0  |
      | qwen3-4b           |     36 |      805 |  35.5 |  88-90%  | Q4_0  |
      | **mistral-7b**     |     32 |      650 |  34.4 |     68%  | Q4_K_S |
      Mistral-7B Q4_K_S sits at qwen3-4b's speed but with
      88% → 68% accuracy. Quant aggressiveness is real
      cost. To get a clean 7B speed/accuracy claim we'd
      need a 7B Q4_0, which doesn't fit the WASM cap;
      Q4_K_M (4170 MB) is also over the cap. Q4_K_S is
      the largest quant that fits.
    - **What this unblocks:** §10 wave-2 has a working
      reference at the 7B mark with the §11 loader and
      Q4_K_S quant. Llama-3.1-8B / Qwen3-8B at Q4_K_S
      (~4500 MB) are over the cap; would need Q3_K_S
      (3494 / 3595 MB) which routes through the broken
      Q3_K kernel. Practical wave-2 ceiling at the
      current llama.cpp upstream is **7B Q4_K_S**.
      Bigger models require either fixing the Q3_K
      shader or bumping `MAXIMUM_MEMORY` to 8 GB
      (deferred §12).

### Resumption checklist (start a fresh session here)

**Wave 1 complete (7/10 done · 2 deferred · 1 optional
skipped).** Wave 2 underway: **1/4 done** (mistral-7b-v0.3-q4ks
at 34.4 tok/s / 68% accuracy — see §12 above). Two new bugs
landed in this session:

- **Bug #28 (Q3_K shader):** Q3_K matmul produces gibberish
  in ggml-webgpu. Wave-1 never exercised it (all Q4_0); §9
  tested only Q4_K_M. Q3_K_M was the smallest viable 7B+
  quant under the WASM cap, but it's broken — Q4_K_S is
  the workaround for 7B (3953 MB, fits with margin) but
  is the largest quant that fits, so 8B+ models would need
  either a Q3_K shader fix or a `MAXIMUM_MEMORY` bump.
- **Loader / parser refactor (§11):** GGUF streams cleanly
  through the WASM heap; ctxCreate over-allocation fixed.
  Both confirmed working at the 7B mark (3.95 GB streamed).

**Next target options (pick one — no implicit ordering):**

A. **Continue wave 2 with another 7B Q4_K_S model** to widen
   the cross-family signal at 7B (Mistral is the only 7B+
   data point so far, and it's hampered by the most
   aggressive quant that fits). Llama-3.1-8B Q4_K_S is
   4475 MB — over the cap. Options narrow to: (a)
   Mistral-Nemo-Instruct (12B, has Q4_K_S? probably too
   large), (b) re-run Mistral with different prompt sets
   for tool-calling / cold-temp signal, (c) verify that
   bartowski Mistral mirror has any other instruction-
   tuned 7B variant that fits. Lower-leverage than B/C/D.

B. **Fix the Q3_K shader (#28) to unblock 8B+ models.**
   Investigation starts in
   `~/Repos/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl::MUL_ACC_Q3_K`.
   If fixed, Llama-3.1-8B / Qwen3-8B at Q3_K_S (3494 /
   3595 MB) become viable and we get clean 8B cross-family
   data. Carries debug effort cost (Q3_K shader is the most
   complex in the legacy/K-quant family). Once fixed, wave-2
   has 4 candidates with the same workflow.

C. **Bump `MAXIMUM_MEMORY` to 8 GB (deferred §12).** Would
   admit Q4_K_M / Q4_K_S of larger models without needing
   Q3_K. Requires verifying browser support for >4 GB WASM
   heaps, which has historically been spotty. Quick to try
   (one CMake flag); slow to validate.

D. **Pick up a deferred kernel-tuning lever.** §A subgroup-
   cooperative loading is now more attractive: at 7B Q4_K_S
   the §A ceiling rose from ~13% (4B Q4_0) to ~18% of decode
   time. Mistral-7B is a fresh perf baseline for testing.

B. **Pick up a deferred kernel-tuning lever (§A subgroup
   loading, §B FA shape-routing).** The 7-model wave-1
   baseline is now a strong regression-detection harness.
   §A's realistic ceiling at the current scale is ~13% of
   decode time (per §6–§9 analysis); §B helps prefill TTFT
   and matters more for longer prompts. §B may also become
   more attractive at 7B+ where K-dimension grows further.

C. **Implement one of the deferred architectures (Gemma 2
   or Phi 3).** §9 in "Completed on 2026-04-26" has the
   feature inventory for both. Gemma 2 is more work
   (5 distinct gaps); Phi 3 is mostly fused-QKV unpacking.

D. **`qwen2.5-coder-1.5b` / Hermes-3 cold-temp follow-ups**
   only matter if a code-gen task or tool-calling comparison
   becomes load-bearing for the dashboard.

**Decode tuning at the current scale remains characterized
as bottomed out** — §6/§7/§8/§9 measured the bandwidth-bound
fraction at ~40% of matmul on Q8 / ~20% on Q4, ceiling for
any pure-bandwidth lever ~13% of decode time. The 4B data
point in §10 #10 is consistent: matmul leads decisively
(35.6%) but the absolute bandwidth budget per matmul keeps
growing with hidden size, so percentage-of-decode for any
matmul lever stays in the 10-13% range. Whether subgroup-
cooperative loading actually pays its 13% ceiling is open.

Boot sequence (any of A-D):

1. `make checkall` — confirm 393 pass / 5 skip / 0 fail
   (was 392 before §11's sub-view regression test).
2. `git -C ~/Repos/llama.cpp log --oneline -10 webllm-browser-patches`
   — confirm the 10-patch stack is intact.
3. Read the "Completed on 2026-04-26" section, especially
   §10 (qwen3-4b headline + cross-family pattern) and §11
   (loader/parser API change — important if your work
   touches `GgufParser`/`ModelLoader`/`loadWeights`).

Verify GGUF mirrors *before* running smoke-bench — wave 1
hit three bad mirrors (smollm2-360m's huggingface-quants
→ 401, hermes-3's NousResearch → no Q4_0, qwen3-4b's
official Qwen mirror → only K-quants). Unsloth and
bartowski have been the reliable fallbacks. Probe via
`curl -s "https://huggingface.co/api/models/<repo>/tree/main"`
filtered through `python3` to list `.gguf` files + sizes.

Old kernel-tuning targets remain available as **deferred** items:

- **§A — subgroup-cooperative `q_packed` loading** (last
  bandwidth lever; realistic ceiling ~13% decode; was the prior
  §10 entry, now demoted).
- **§B — FA shape-routing for prefill/TTFT** (§5 path a;
  higher-impact for long prompts; FA's main win is seq>1).
- **§C — drafter-based speculative decoding** (large project;
  2-3× wall-clock potential).
- **§D — encoder/embedding perf pass** (untouched since §21).

These deferred targets are below the size campaign in priority,
not abandoned — pick them up after wave 1 lands or if the
campaign hits a blocker (e.g., WASM cap forces a quant rewrite
for 7B+).

Note: the smoke page now does a shader-cache warmup after [6/8]
engine adoption. If you see "1.0 tok/s" on the smoke page after
a fresh WASM rebuild, the warmup didn't run — investigate before
investigating "the engine."

### Historical context (for archive — do not action again)

1. **RESOLVED (2026-04-25): Apr-23 smoke-bench "regression" is a benchmark
   methodology change, not an engine regression.** Bisect (TS bundle only;
   WASM and llama.cpp HEAD constant via `make smoke-test` rebuilds):

   | Commit | Sampler used by smoke page | Decode path | tok/s |
   |---|---|---|---|
   | `0548cd4` (last fast point) | `Sampler({ temperature: 0 })` | `forwardDecode` (4-byte readback) | **118.9** |
   | `d111560` (profiling commit) | greedy | `forwardDecode` | **118.5** |
   | `d131cf0` (KV cap commit) | greedy | `forwardDecode` | **119.6** |
   | `5542bef` (qwen stability) | `makeSmokeSampler` (temp 0.7, topK 40, topP 0.95, repPenalty 1.05) | `forward` (full 32K logits) + JS topK/topP | **56.6** (page-reported) |
   | `main` (HEAD a0d5b9a) | same as 5542bef | `forward` + JS topK/topP | **59.1** |
   | `main` + `?temp=0&rep=1` URL override | greedy override | `forwardDecode` | **116.8** |

   **Root cause**: the new smoke decode loop in `5542bef` (and inherited by
   the page-shell extractions through HEAD) only takes the GPU-reduced
   ARGMAX fast path when `sampler.isGreedy && sampler.noPenalty`
   (`smoke-test/real-model-smoke.js` → `getSmokeSamplingConfig`,
   `smoke-test/real-model-page.js` decode loop). The new realistic sampler
   has `temperature 0.7` and `repetitionPenalty 1.05`, so every step falls
   through to `inference.forward()` — full 32K-vocab logits download plus
   JS-side temperature/topK/topP/penalty work — instead of the 4-byte
   greedy readback the old greedy page used.

   **Engine evidence**: the same `main` build serving the same WASM hits
   116.8 tok/s when the URL forces `temp=0&rep=1`. That matches the 118.9
   tok/s baseline at `0548cd4` to within noise. There is no decode-path
   regression in the inference engine, ggml-webgpu, or async readback.

   **Decision (2026-04-25)**: realistic sampling is the new canonical
   smoke-bench baseline. Do not re-baseline against the historical
   `~115` / `~93.5` greedy-path numbers — they measured a different
   workload and are retired as comparison targets. `forwardDecode` (greedy)
   stays available as the upper-bound diagnostic via `?temp=0&rep=1`, but
   smoke-bench tracks the realistic-sampling number going forward.

   **Re-baseline of item 11 (2026-04-25)**, controlled comparison on HEAD
   with TinyLlama-1.1B Q4_0 (3 trials each, medians shown):

   | Scenario | Decode path | tok/s | ms/token |
   |---|---|---|---|
   | A — greedy + `forwardDecode` (4 B readback) | fast | **114.8** | 8.7 |
   | B — greedy + `forward` (128 KB readback, argmax JS) | mid | **115.9** | 8.7 |
   | C — sampled + `forward` + JS topK/topP/penalty | slow | **52.9** | 19.0 |

   A vs B: ~0.1 ms/token. **Item 11's "negligible" framing still holds**
   even post-async-readback — full 32 K-vocab readback costs essentially
   nothing extra over the 4-byte ARGMAX readback. My earlier hedge was
   wrong; do not block on re-baselining item 11 again.

   B vs C: ~10 ms/token. **The entire 2× slowdown is the JS sampling
   pipeline** (`Sampler.sample()` over 32 K floats: temperature scale +
   topK selection + topP normalization, plus `applyRepetitionPenalty`
   over the recent 64-token window). If decode optimization resumes,
   that is the lever — vectorize / partial-sort / GPU-side topK before
   chasing readback or graph-reuse. The temporary `?slowpath=1` URL
   gate used to capture scenario B is not committed; re-add it from
   git history if you want to re-measure.

   **Fix landed (2026-04-25)**: the GPU TOP_K path already existed in
   `Generator.generate` (`src/inference/generation.ts` 3-way branch
   greedy/topk/full) but the smoke decode loop only branched greedy/full,
   so realistic-sampler steps fell through to full-vocab readback + JS
   topK on 32 K floats. Added a topk middle branch in
   `smoke-test/real-model-smoke.js::createSmokeCompletionRunner` that
   calls `inference.forwardDecode(..., "topk", sampler.topK)` and feeds
   the reduced indices/values into `sampler.sampleFromTopK(...)` (which
   already applies repetition penalty + temperature + topP on the
   k-element set). Gated to skip when qwen masking/thinking state is
   active (`thinkDepth > 0`, `waitingForVisibleAnswer`,
   `hasVisibleAnswerText`) so the GPU's pre-mask top-K choice can't
   leak masked tokens. Measured impact (TinyLlama Q4_0, 3 trials):
   ~53 → ~111 tok/s (2.1×). Qwen3 thinking-off also benefits
   (~76 tok/s with coherent output); thinking-on routes through the
   full path unchanged.

2. **DONE (2026-04-25): library is now the single source of truth for
   decode.** See "Completed on 2026-04-25 (cont.)" above and bug-fix
   entries #23 and #24 for the full landed change set.

---

### Active next steps

1. **DONE (2026-04-25): re-profiled decode against the canonical
   pipeline.** Numbers landed in the "Inference Performance
   Optimizations" preamble above. Three latent harness bugs had been
   hiding the entire profile path since the consolidation; fixed in
   `953c560`. The fresh numbers ratify the same broad direction the
   stale 2026-04-22 profile pointed at — graph compute dominates,
   matmul + encode/dispatch overhead are the leads — but with the
   refinement that **encode overhead leads on Qwen3 (29.2% of graph)
   while matmul leads on TinyLlama (33%)**. The biggest single lever
   is still §2 below: getting qwen3 thinking-on off the full path.

2. **DONE (2026-04-25): qwen3 steering routes through topk + CPU mask
   filter** (commit `3e5be59`). Replaced the full-vocab readback /
   JS sampling pipeline with a `K + STEERING_TOPK_BUFFER` GPU TOP_K
   request followed by CPU-side mask filtering inside
   `Generator.generate`. Decision driven by the diagnostic capture
   on 2026-04-25: masked tokens land in top-K of full-vocab logits
   0.31% of the time, top-(K+10) 0.41% — the masks live deep in
   the tail, so CPU post-filter virtually never exhausts the pool
   and GPU-side WGSL masking would be over-engineering. Measured
   impact: **17.3 → 93.0 tok/s, 5.4×** — much larger than the
   ~24 tok/s prediction because the hot bucket was the JS sampler
   over Qwen3's 152K vocab, not the readback. `waitingForVisibleAnswer`
   stays on the full path because its whitespace-guard resampling
   needs full-vocab access (~2 of 236 captured steps). Output
   coherent on smoke step [8/8]; no regression on TinyLlama or
   Qwen3 thinking-off.

3. **Decode graph reuse** (item 1 in "Inference Performance
   Optimizations" preamble) remains deferred. The 2026-04-21 profile
   measured non-GPU overhead (ctxCreate + buildGraph + backendAlloc +
   teardown) at 1.7% of decode-step time — too small to chase. After
   the throughput uplift and the consolidation, that fraction may
   have grown; re-evaluate as part of §1's profile pass before
   committing to the C-side refactor.

4. **DONE (2026-04-25): characterized qwen3-1.7b-q4f16 through the
   consolidated pipeline.** Numbers landed in the "Inference
   Performance Optimizations" preamble above. Headlines:
   - Steady-state thinking-on **66 tok/s** (clean, 117-token run);
     thinking-off **59 tok/s** but on a 17-token warmup-dominated
     run — trust the thinking-on number for the canonical 1.7B rate.
   - Output coherence verified on the smoke page: clean `<think>` →
     answer transition, finish=eos, embed step still passes
     (cosine=0.76).
   - **Hypothesis confirmed**: dispatch count is architecture-invariant
     within Qwen3 (629/token at both 0.6B and 1.7B). Encode overhead's
     *absolute* cost stays nearly flat (4.07 → 4.20 ms) across 2.83×
     param scale; its *fraction* of graph time *drops* (29.2% →
     22.4%). Matmul scales sub-linearly (4.05 → 6.88 ms, 1.7×) but
     its fraction *grows* (29.1% → 34.2%).
   - **§2's topk fix holds at scale.** 342/348 thinking-on steps
     route through topk; the 6 full-path steps add ~3 ms/step but
     contribute negligibly to wall time.
   - **Quant caveat surfaced**: the `qwen3-*-q4f16` model IDs resolve
     to `Qwen3-*-Q8_0.gguf` (no `ggufFilePattern` pinned, picker
     fallback hits Q8 first). Within-Qwen3 comparisons are clean
     (both Q8); TinyLlama-Q4 vs Qwen3-Q8 absolute matmul ms cross
     two variables — read fractions, not absolute ms, across families.
   - **Bench-full coverage landed (2026-04-25, second session pass)**:
     6 1.7B profiles registered in `eval/smoke-profiles.ts` (3 off +
     3 thinking, mirroring 0.6B layout) and added to `full` /
     `llama-vs-qwen` / `thinking-modes` sets plus a new `qwen3-sizes`
     set. `bench.ts --profiles qwen3-1.7b-*` ran clean: 12/12 phases
     passed, no errors. Dashboard ingested all 6 dots.
     - Per-profile speed (oneShot tok/s, smoke chat regression):
       off-cold 48.3 · off-warm 47.2 · off-hot 45.9 · thinking-cold
       49.8 · thinking-warm 48.2 · thinking-hot 47.8.
     - Per-profile accuracy (overall): off-cold 82% · off-warm 87% ·
       off-hot 88% · thinking-cold 83% · thinking-warm 89% ·
       thinking-hot 87%.
     - Per-dimension headlines: tool-calling 65–71% (cold only;
       skipped at warm/hot per gate), reasoning 92–100% (thinking-
       warm hits 100%), instruction-following 100% across all
       profiles, semantic-reasoning 68–72% (flat; not addressed
       by this size bump).
     - Speed regression note: smoke chat regression numbers
       (45.9–49.8 tok/s) are lower than `perf.ts` steady-state
       (~59–66 tok/s); the gap is the chat-regression harness
       overhead (page-load + interactive run path) not engine
       throughput. Use `perf.ts` for engine claims and dashboard
       for cross-profile accuracy×speed tradeoff.

5. **DONE (2026-04-25, third pass): rebased onto upstream master
   carrying `13d36cf89` (FA browser unblock).** Branch is now 10
   commits on top of upstream (added a row_norm codegen-stability
   split as patch 10 before rebasing — `docs/LLAMA_CPP_PATCHES.md`
   updated with new patch count and rebase note). One conflict on
   `ggml-webgpu.cpp` end-of-`graph_compute` resolved cleanly
   (kept our profiling finalization block; upstream restructured
   nearby `WEBGPU_CPU_PROFILE_TOTAL_END` placement). Backup at
   `webllm-browser-patches-pre-fa-rebase`.

   **No regression:** Qwen3-1.7B steady-state 66.8 tok/s thinking-on
   (was 65.8 pre-rebase — within noise). Profile-mode dispatch count
   **629/token unchanged**, `backendAttentionMs` 0.59ms unchanged,
   matmul/encode within noise. Output coherent on smoke step [7/8]
   ("Why don't scientists trust atoms..."), embed step [8/8] passes.

   **FA didn't engage on these decode shapes.** The new upstream
   `ggml_webgpu_flash_attn_get_decisions` returns the
   subgroup-matrix-required path for our N=1 decode shapes (head_dim
   128, GQA 16/8, K=2048+); browser hits the `supports_op = false`
   branch at line 4460 and falls back to the manual attention path.
   The new VEC and TILE paths target different shapes (longer K, or
   prefill seq>1). To actually engage FA, would need to dig into
   `flash_attn_get_decisions` to understand which shape regions
   route to VEC vs TILE vs subgroup-matrix — see §6 path (a) below.

   **Cold-shader artifact discovered + fixed.** The first decode
   after a WASM rebuild reported 1.0 tok/s on the smoke page — that's
   shader compilation, not a regression. Added a warmup pass in
   `smoke-test/real-model-page.js` after [6/8] engine adoption: runs
   a 2-token `chatCompletion` with realistic sampling (temp 0.6,
   topK 40, repPenalty 1.05) so the topk decode pipeline compiles
   here, not on the first measured call (greedy warmup would only
   compile the greedy/full path). Encoder models warm with
   `embed("warmup")` instead. Verified: cold reload reports
   "[6/8] Shader-cache warmup complete in ~290ms" then
   "[7/8] ... 60.5 tok/s" instead of 1.0 tok/s. Warmup result is
   discarded; KV cache is reset automatically by the next
   `chatCompletion` call. `perf.ts`'s 3-trial median already
   absorbs cold-shader on the first trial, so no harness change
   needed there. The warmup runs on every page load including
   interactive use, not just measurement runs — flag-gating on
   `chatSmoke=` / `bench=` URL params is a follow-up if the ~290ms
   load cost matters.

6. **DONE (2026-04-26): matmul dequant-stub diagnostic — both Q8 and
   Q4 GEMV decode are memory-bound, not compute-bound.** Stubbed out
   the dequant arithmetic in `mul_mat_vec.wgsl::MUL_ACC_Q8_0`
   (`f32(get_byte_i32(q_packed, byte_idx)) * 0.0`) and `MUL_ACC_Q4_0`
   (`(f32(q_byte & 0xFu) - 8.0) * 0.0` / `(f32((q_byte >> 4u) & 0xFu)
   - 8.0) * 0.0`) — preserves the `q_packed` and `d` load chain via
   IEEE-754 `* 0.0 ≠ statically 0` so the optimizer can't DCE the
   reads, just zeroes the contribution to `row_sum`. Rebuilt WASM,
   profiled both quants on the consolidated pipeline against
   immediately-prior baselines:

   | Quant / Model              | Baseline matmul | Stub matmul | Delta  | Dispatch (load survived?) |
   |----------------------------|----------------:|------------:|-------:|---------------------------|
   | Q8_0 / Qwen3-1.7B think-on |        6.67 ms  |    7.04 ms  |  +5.5% | 629 → 629 ✅              |
   | Q4_0 / TinyLlama-1.1B chat |        3.76 ms  |    3.67 ms  |  -2.4% | 450 → 450 ✅              |

   Both deltas are within profile-mode noise (compare ±5% noise on
   raw 3-trial medians); the dispatch-count invariant on both
   quants confirms the load chain wasn't optimizer-eliminated.
   **If the kernel were compute-bound, removing the FMA work should
   drop matmul time substantially (e.g. 30–60%). It barely moved on
   either quant — that's the memory-bound signature.** Stubs reverted
   (`git checkout --` on `mul_mat_vec.wgsl`); WASM rebuilt clean;
   TinyLlama steady-state back to **106.2 tok/s** post-revert.

   **Implication for the next lever:** dequant fusion is *not* the
   right target. The activation vector `src1` (called `x_block` in
   the per-quant inner loops) is loaded by every workgroup from
   global memory; cache it in `var<workgroup>` shared memory and
   load each k-stride exactly once per workgroup. See §7 below for
   the design.

   **FA shape-routing investigation (path a)** remains untouched as
   a separate side-quest — defer unless prefill optimization
   becomes a target.

   **Cleanup item** worth landing whenever next touching `eval/models.ts`:
   pin `ggufFilePattern` on the `qwen3-*-q4f16` entries so the file on
   disk matches the model ID, or rename the IDs to `*-q8` to be honest
   about what the picker fetches.

7. **DONE (2026-04-26): src0-vs-src1 discrimination identified
   src0 (quantized weights) as the dominant bandwidth bottleneck
   on both Q4 and Q8 GEMV decode kernels.** The literal-constant
   form of Stub A (`x_block[i] = 1.0`) tripped a WGSL→SPIR-V
   compiler regression on the first attempt — `backendMatmulMs`
   exploded from 3.87 → 139.62 ms with dispatch count unchanged,
   suggesting register spill of `var<private>` array initialization
   to private memory rather than register allocation. The fix was
   to keep the array fill pattern identical but source `x_block[i]`
   from a single broadcast-load (`let x_const = f32(src1[src1_idx_base])`)
   so the optimizer still sees a real memory dependency and
   register-allocates `x_block` normally. Stub B mirrored that
   approach for src0 (`let d_const = f32(load_f16_at_src0(0u));
   let q_const = load_u32_at_src0(0u);`). With both stubs in
   structurally-equivalent broadcast form, results were:

   | Stub | Quant / Model              | Baseline matmul | Stub matmul | Δ matmul | Δ tok/s | Dispatch |
   |------|----------------------------|----------------:|------------:|---------:|--------:|----------|
   | A    | Q4_0 / TinyLlama-1.1B chat |          3.87 ms |      3.84 ms |    -0.8% |   -3.3% | 450 ✅   |
   | A    | Q8_0 / Qwen3-1.7B          |          6.67 ms |      6.67 ms |     0.0% |    n/a* | 629 ✅   |
   | B    | Q4_0 / TinyLlama-1.1B chat |          3.87 ms |      3.09 ms |  **-20%** | **+5.5%** | 450 ✅   |
   | B    | Q8_0 / Qwen3-1.7B          |          6.67 ms |      3.98 ms |  **-40%** | **+45%**  | 629 ✅   |

   \* Qwen3-1.7B captured under thinking-off in profile mode for
   both Stub A baseline (44.8 tok/s) and Stub B (65.2 tok/s) since
   the matmul kernel is identical regardless of thinking mode.

   **Decision per matrix:** B collapses, A barely moves → re-run
   `OUTPUTS_PER_WG` 4 → 8 (or 16). Bigger tiles amortize each
   weight load across more output rows; this is the matching
   structural lever for src0-bandwidth dominance. Q8 is the
   sweeter target since each block carries 32 q-bytes vs Q4's
   16 q-half-bytes — proportionally more bandwidth per dispatch
   to recover.

   **Stubs reverted** (`git -C ~/Repos/llama.cpp checkout --
   ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl`); WASM
   rebuilt clean; TinyLlama steady-state ~77.5 tok/s on the
   resulting 2-trial verification (within profile-mode noise of
   pre-stub baseline 81.9). `git -C ~/Repos/llama.cpp status`
   confirms clean working tree.

8. **DONE (2026-04-26): `OUTPUTS_PER_WG` sweep on
   `WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG` — OPW=4 is
   locally optimal in [2, 8]. The §7 lever was wrong; the
   reasoning that "bigger tiles amortize weight loads" doesn't
   hold under analysis.** Three-trial medians on the same
   profile harness as §7:

   | OPW  | TinyLlama Q4 tok/s | TinyLlama matmul ms | Qwen3-1.7B Q8 tok/s | Qwen3 matmul ms |
   |-----:|-------------------:|--------------------:|--------------------:|----------------:|
   |   2  |             — *    |               — *   |               34.9  |          8.17   |
   |   4  |             81.9  |               3.87  |               44.8  |          6.67   |
   |   8  |             80.7  |               3.81  |               41.0  |          6.83   |

   \* OPW=2 only profiled on Qwen3-1.7B; the trend was clear
   enough not to spend a TinyLlama run (more workgroups → more
   redundant src1 reads, exactly the inverse of the OPW=8
   regression).

   **Why bigger OPW doesn't help (corrected analysis).** Total
   src0 reads = num_wg × src0_per_wg = (m/OPW) × (OPW × num_blocks ×
   bytes_per_block) = m × num_blocks × bytes_per_block — *invariant
   to OPW*. What bigger OPW *does* reduce is **total src1 reads**
   (each WG reads src1 once and reuses it across OPW output rows;
   total src1 reads scale as m/OPW). Stub A in §7 already
   established that src1 isn't the bandwidth bottleneck, so the
   amortization-of-src1 win bigger OPW provides isn't load-bearing.
   On Q8 specifically, OPW=8 made things *worse* (-8.5% tok/s,
   +2.4% matmul ms) — almost certainly per-thread register
   pressure: the inner loop now keeps `acc: array<f32, 8>` plus
   8 × 8 q-bytes' worth of in-flight FMA state instead of 4 × 8.
   OPW=2 made things much worse (-22% tok/s, +22% matmul ms),
   ruling out "more parallelism for latency hiding" as the lever.

   **What this means for the actual src0 bottleneck.** Stub B in
   §7 measured src0 bandwidth as the dominant cost. Real bandwidth
   savings can only come from:

   - **Smaller-bandwidth quantization** (Q4_K ≈ 4.85 bpw vs Q8_0
     8.5 bpw). A 1.7B Q4_K model would have ~57% of Q8_0's weight
     bandwidth per token. Quality cost needs eval validation —
     dashboard-driven A/B against the existing Qwen3-1.7B Q8 dot
     would be the way.
   - **Subgroup intrinsics for cooperative loading** — threads in
     a subgroup share src0 reads via `subgroupBroadcast` /
     `subgroupShuffle`. Could reduce per-thread src0 reads by
     `subgroup_size`. Requires `enable subgroups;` (already in
     mul_mat_vec.wgsl gated on `USE_SUBGROUP_REDUCTION`) and may
     need shader-architecture changes to expose the right access
     pattern. emdawnwebgpu does support subgroups (just not
     subgroup-matrix); §5 covers the latter.
   - **Inner-loop restructure** for better memory coalescing.
     Current Q8 reads `q_packed` at `block_byte_base + 2u + 4u *
     (thread_within_block * 2u + packed_idx)` — packed_idx
     iterates 0..1 inside the row loop, so consecutive threads in
     the same row issue strided 4-byte loads. Switching to
     `vec4<u32>` reads (load all 4 q_packed at once per block per
     row) might hit the L1/L2 line size more efficiently and is
     a smaller change than subgroup-cooperative loading.

   **OPW reverted to 4** in
   `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp:48`.
   `git -C ~/Repos/llama.cpp status` confirms clean working tree.

9. **DONE (2026-04-26): smaller-bandwidth quantization tested via
   `unsloth/Qwen3-1.7B-GGUF` mirror. Q4_0 buys ~12% matmul
   reduction (about 1/3 of Stub B's prediction) for +0.7% tok/s
   (in noise) and 42% smaller download. Q4_K_M is a slight
   regression. Reverted to Q8_0 to keep dashboard baseline
   continuity.** Three-trial profile-mode medians on Qwen3-1.7B
   thinking-off:

   | Quant     | tok/s |  matmul ms | Δ matmul | Δ tok/s | File MB |
   |-----------|------:|-----------:|---------:|--------:|--------:|
   | Q8_0 base |  44.8 |       6.67 |    —     |    —    |    1749 |
   | Q4_K_M    |  43.0 |       6.28 |   -5.8%  |   -4.0% |    1056 |
   | Q4_0      |  45.1 |       5.88 |  -11.8%  |   +0.7% |    1008 |

   **Why Stub B over-predicted.** Stub B replaced *all* src0
   reads with a single broadcast (~1000× bandwidth cut) →
   matmul -40% on Q8 / -20% on Q4. That measures the
   *bandwidth-bound fraction* of matmul kernel time: ~40% on Q8,
   ~20% on Q4. Switching Q8 → Q4_0 only halves bandwidth, so the
   modeled win is 50% × 40% = 20% matmul drop, observed 11.8%
   (~60% of model). The gap is attributable to Q4_0's slightly
   different inner-loop arithmetic (Q4 unpacks two 4-bit nibbles
   per byte vs Q8's single byte), modest cache-pattern shifts,
   and run-to-run variance.

   **Q4_K_M's poor result** is consistent with K-quants being
   compute-heavier per element: 8 sub-blocks per 256-element
   super-block, multiple scales/mins per super-block, more
   metadata reads. The bandwidth savings (Q8 1.06 bpw → Q4_K_M
   0.56 bpw, same as Q4_0) get clawed back by more inner-loop
   arithmetic and metadata fetches. Q4_K is a quality/bandwidth
   trade-off, not a quality/throughput one.

   **Net for the §7 lever investigation:** matmul on Qwen3-1.7B
   is ~33% of decode time, of which ~40% is bandwidth-bound. So
   the *theoretical max* speedup from any pure-bandwidth lever
   is ~13% of decode time. Bigger structural wins (prefill,
   speculative decoding, drafter models) live elsewhere.

   **Side fixes landed:**
   - `src/models/gguf-parser.ts::ggmlTypeSize` was missing Q8_0,
     Q4_K and all K-quants — they fell to `?? 4` which
     over-estimates `totalDataSize` by ~7×. With `no_alloc:true`
     in `ctx_create` (`src/wasm/webgpu-bridge.cpp`), this hadn't
     been load-bearing for Q8_0 in practice, but Q4_K_M would
     have requested a ~6 GB ggml ctx buffer (above 4 GB WASM
     cap) without the fix. Table now covers F32, F16, Q4_0–Q8_K
     legacy + K-quants, I32, BF16.
   - `eval/models.ts` Qwen3-1.7B entry has a maintenance comment
     describing how to swap to unsloth's mirror + ggufFilePattern
     to re-run the quant experiment.

10. **IN PROGRESS (wave 1 underway): large-model test campaign.**
    The current smoke fleet tops out at Qwen3-1.7B. Decode-kernel
    tuning has bottomed out at this scale (§6–§9). The active
    priority is now **scaling the model fleet** — exercise the
    registered-but-unprofiled 3B–4B models, then register and
    test 7B+ candidates with small quants. Whether the
    bandwidth-bound matmul picture from §6–§9 holds at 3B+
    scale is the first question this campaign answers; that
    informs whether the deferred kernel-tuning levers (§A
    subgroup-cooperative loading, §B FA shape-routing) become
    worth chasing again.

    **Wave 1: registered-but-untested models (lowest risk; same
    kernel paths; just need profile registration).** Order by
    increasing size. Each entry is already in `eval/models.ts`;
    none has a smoke profile in `eval/smoke-profiles.ts`.
    - [x] `smollm2-360m-q4f16` (0.36B) — DONE 2026-04-26.
      Steady-state 106 tok/s Q4_0 / profile-mode 75.6 / 651
      dispatches/token / 24/36 accuracy. Encode overhead leads
      matmul. See "Completed on 2026-04-26 §1" above for full
      numbers + the bartowski-mirror repo fix.
    - [x] `qwen2.5-1.5b-q4f16` (1.54B) — DONE 2026-04-26 after
      adding qwen2 bias support (bug-fix #25). Steady-state 84.3
      tok/s / profile-mode 57.6 / 657 dispatches/token / 29/36
      = 81% accuracy. Matmul leads at 40.1% (highest in fleet).
      See "Completed on 2026-04-26 §2" above.
    - [x] `smollm2-1.7b-q4f16` (1.71B) — DONE 2026-04-26.
      Steady-state 86.3 tok/s / profile-mode 57.7 / 491
      dispatches/token / 27/36 = 74% accuracy. 24 layers, no
      GQA (n_head_kv=32), KV cache 1536 MB at ctx=4096 (largest
      in fleet). 31% faster than Qwen3-1.7B at same params. See
      "Completed on 2026-04-26 §5" above.
    - `qwen2.5-coder-1.5b-q4f16` (1.54B) — code-tuned variant;
      mostly interesting if we add a code-generation eval task.
      Same arch as qwen2.5-1.5b (qwen2 with bias support).
    - [-] `gemma-2-2b-q4f16` (2.61B) — DEFERRED 2026-04-26.
      Architectural gap: needs pre+post norm pairs, logit/attn
      soft-cap (new opTanh WASM binding), RMSNorm (1+w) scaling,
      sliding-window attention every other layer. Bench-full
      not run; inventory in "Completed on 2026-04-26 §8" above.
    - [x] `qwen2.5-3b-q4f16` (3.09B) — DONE 2026-04-26.
      Steady-state 45.1 tok/s / profile-mode 32.3 / 841
      dispatches/token (highest in fleet) / 32/36 = 86%
      accuracy. 36 layers (qwen2 with bias support); KV 288 MB
      thanks to GQA 8:1. See "Completed on 2026-04-26 §6" above.
    - [x] `llama-3.2-3b-q4f16` (3.21B) — DONE 2026-04-26.
      Steady-state 58.2 tok/s / profile-mode 37.9 / 572
      dispatches/token / 27/36 = 76% accuracy. 28 layers, GQA
      3:1, KV 896 MB. 29% faster than qwen2.5-3b at same param
      scale. See "Completed on 2026-04-26 §7" above.
    - [x] `hermes-3-llama-3.2-3b-q4f16` (3.21B) — DONE 2026-04-26.
      Steady-state 60.4 tok/s / profile-mode 42.8 / 572
      dispatches/token (matches base) / 27/36 = 74% accuracy.
      Tool-calling advantage invisible at warm temp (gate at
      0.4); rerun cold for that signal. See "Completed on
      2026-04-26 §8" above.
    - [-] `phi-3.5-mini-q4f16` (3.82B) — DEFERRED 2026-04-26.
      Architectural gap: needs fused QKV projection unpacking
      and FFN gate_up split. Inventory in §8 above.
    - [x] `qwen3-4b-q4f16` (4.0B) — DONE 2026-04-26.
      Steady-state 35.5 tok/s / profile-mode 32 / 805
      dispatches/token (matched §10 prediction within 1%) /
      32/36 = 88% off, 33/36 = 90% thinking-on. **Highest
      accuracy in fleet.** 36 layers, GQA 4:1, KV 144 MB.
      Required loader refactor (§11 below) to fit through
      JS 2 GiB allocation cap. See "Completed on 2026-04-26
      §10" above.

    **Wave 2: register 7B+ candidates with small quants if the
    WASM 4 GB cap allows.** Q4_0 7B = ~3.94 GB just for weights —
    sits right at the WASM cap; would need Q3_K_M (~3.4 GB) or
    smaller. **Q3_K shader is broken (bug #28)** so K-quants are
    forced to Q4_K_S/Q4_K_M; only Q4_K_S (3953 MB Mistral) fits.
    Practical wave-2 ceiling at the current llama.cpp upstream is
    7B Q4_K_S. Candidates:
    - [x] `mistral-7b-instruct-v0.3-q4ks` — DONE 2026-04-26.
      Steady-state 34.4 tok/s / profile-mode 28.0 / 650
      dispatches/token / 26/36 = 68% accuracy. Q3_K_M attempt
      first (gibberish — bug #28). Q4_K_S workaround works.
      See "Completed on 2026-04-26 §12" above.
    - Llama-3.1-8B-Instruct — Q4_K_S 4475 MB > WASM cap.
      Q3_K_S 3495 MB fits but routes through broken Q3_K
      shader. Blocked on either bug #28 fix OR
      MAXIMUM_MEMORY bump.
    - Qwen3-7B / Qwen3-8B — Qwen3-7B has no published mirror
      tested yet; Qwen3-8B Q4_K_S 4580 MB > cap, Q3_K_S 3595
      MB blocked on bug #28. Same blocker.
    - Mistral-7B-Instruct-v0.7 — Mistral-v0.7+ uses
      `[SYSTEM_PROMPT]` template (already detected as
      `mistral-v7`), but no Q4_K_S verified < 4 GB cap yet.
    - Gemma-2-9B (probably won't fit; deferred per §9
      architecture gap above for the 2B variant).

    **Per-model action sequence:**
    1. Register a smoke profile in `eval/smoke-profiles.ts` —
       at minimum a `<id>-warm` (temperature 0.6, prompt
       `DEFAULT_PROMPT`); for Qwen3 family also add `-thinking-warm`.
       Add the new name to `SMOKE_PROFILE_SETS.full` (and the
       relevant family set, e.g., `qwen3-sizes` for qwen3-4b).
    2. `make smoke-bench PERF_MODEL=<id> PERF_RUNS=3` — captures
       tok/s, prefill/decode ms, matmul ms, dispatch count.
       Watch for: download success, model loads, [7/8] coherent
       output, no console errors.
    3. `make bench-full --profiles <profile-name>` — lands the
       accuracy×speed dot in the dashboard.
    4. Update this TODO with measured numbers (tok/s, matmul ms,
       dispatch count, dashboard accuracy summary, anything
       surprising).

    **Failure modes to watch for at scale:**
    - **WASM memory exhaustion at load.** ggml ctx_create + tensor
      uploads + KV cache must fit in the 4 GB WASM cap. Q4 4B
      models are ~2.25 GB weights + KV; should fit.  7B Q3_K may
      not. If `_ctx_create` fails or `RangeError: Memory size out
      of bounds`, drop to a smaller quant or investigate
      `MAXIMUM_MEMORY` bump.
    - **GPU buffer allocation failures.** Browsers vary on
      `maxBufferSize` / `maxStorageBufferBindingSize`. Look for
      `Buffer creation failed` in the WebGPU console.
    - **KV cache scaling.** Default `contextLength` in
      `eval/models.ts` is 4096 for most entries. KV cache size =
      `2 × n_layer × n_head × head_dim × seq × 4 bytes`. For
      Llama 3.2-3B at 4K context that's ~128 MB; for an 8B at
      4K it's ~512 MB. Could be the second hardest constraint
      after weights.
    - **Dispatch count growth.** TinyLlama (22 layers) emits 450
      dispatches/token; Qwen3-1.7B (28 layers) 629/token. A 4B
      model with ~36 layers would hit ~810/token; 7B (~32 layers
      typical) ~720/token. Encode-overhead-per-step scales
      linearly with dispatch count.
    - **Matmul `m` dimension scaling.** Most matmuls have
      `m = hidden_size`. Going from 1.7B (hidden 2048) to 4B
      (hidden 2560) to 7B (hidden 4096) increases per-matmul
      bandwidth roughly proportionally. The §6–§9
      bandwidth-bound characterization may or may not hold —
      part of the campaign value is finding out.

    **Output format for each measurement:** add a numbered entry
    under "### Completed on YYYY-MM-DD" with the model id,
    profile name, observed tok/s (steady-state and profile-mode),
    matmul ms / fraction, dispatch count, KV cache size at the
    test prompt length, smoke output verdict, and any
    behavioural surprises (e.g., chat template quirks, BOS
    handling, tool-call format edge cases).

    **Stop conditions / when to pivot back to deferred §A–§D:**
    - All wave 1 models land cleanly with no engine regressions
      and decode behaviour matches §6–§9 predictions → §A
      subgroup loading becomes worth attempting.
    - WASM cap forces a build change (e.g., MAXIMUM_MEMORY
      bump to 8 GB, JSPI investigation §12) → that becomes the
      blocker, address it before continuing the campaign.
    - A model exposes a correctness bug (template, tokenizer,
      arch) → fix in `src/models/` and add a regression test
      before resuming the campaign.

### Deferred kernel-tuning targets (behind §10 in priority)

§A. **Subgroup-cooperative `q_packed` loading in
    `mul_mat_vec.wgsl::MUL_ACC_Q8_0` / `MUL_ACC_Q4_0`.** The
    remaining Stub B remediation (was the prior §10 "NEXT"
    before the size pivot). Threads in a subgroup share their
    loaded `q_packed` values via `subgroupBroadcast` so each
    thread issues fewer src0 reads. Realistic ceiling: ~13% of
    decode time at the current scale (matmul × bandwidth-bound
    fraction). May change with larger models — re-evaluate after
    wave 1 of §10. Risk: the current per-thread access uses
    `thread_within_block` to address q_packed, which already
    partitions src0 across threads — further sharing across
    subgroup boundaries (vs the current within-workgroup
    partitioning) may not buy anything.

    Lower-cost alternative: vec4-packed loads (read 4
    consecutive `q_packed` u32s in one instruction). Smaller
    engineering cost, smaller predicted win.

§B. **FA shape-routing for prefill/TTFT** (§5 path a). Decode
    shape (N=1, head_dim 128, GQA 16:8) doesn't engage FA
    post-rebase. Higher-impact for prefill latency / longer
    prompts; FA's main win is seq>1. Probe
    `flash_attn_get_decisions` for the VEC vs TILE vs
    subgroup-matrix shape regions and adjust the guard. Becomes
    more attractive once larger models (longer K dimension at
    the same context length) land in §10.

§C. **Drafter-based speculative decoding.** Larger project,
    well-trodden territory in the literature. Potential 2–3×
    wall-clock decode for chat-style workloads where the
    drafter is mostly right. Drafter could be one of the §10
    wave 1 small models (smollm2-360m or qwen3-0.6b) paired
    with a 3B+ target.

§D. **Encoder/embedding perf pass.** §21 dashboard section
    shipped but the encoder forward pass hasn't had a perf
    pass. Quick win possible if anyone uses arctic-embed-s/m
    at throughput.

11. The latent 3+ binding buffer-conflict edge case in
    `ggml_backend_webgpu_build_multi` (item 3 in preamble) remains
    untested — no llama op hits it today.

12. **JSPI feasibility checkpoint** remains a follow-up investigation,
   not the next implementation step.
   - **Go/no-go:** no-go for the current milestone; the
     completion-driven readback path is the active baseline.
   - **What would have to change if revisited:** flip the WASM build
     from the current ASYNCIFY setup toward JSPI-related flags in
     `src/wasm/CMakeLists.txt`, replace
     `ggml-wasm.ts::callWithAsyncify()` with direct JSPI-compatible
     async export handling, re-audit Emscripten runtime exports to
     remove Asyncify-specific methods and keep only the JSPI-needed
     surface, assess whether the local `~/Repos/llama.cpp` branch's
     `ggml-webgpu: browser + ASYNCIFY support bundle` needs a
     parallel JSPI patch path, and verify browser support/behavior
     on the actual target matrix before any migration.

---

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
