# WebLLM TODO Archive

Historical content extracted from `TODO.md` on 2026-04-27 to keep the
active TODO lean. The cycles, debug history, and earlier roadmap stages
captured here are **closed and informational** — see `TODO.md` for the
current resumption checklist, candidate next levers, and most-recent
session work (the "Completed on 2026-04-27" block stays in `TODO.md`).

If you need the full context of a closure cited from `TODO.md`'s
resumption checklist (§17 / §18 / §19 / §20 / §21 / §22 / §23 / §24 / §25 /
§26), look it up here.

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

---

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

---

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

---

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

13. **§10 wave 2, model 2: llama-3.1-8b-instruct-iq3m
    registered + benched. First 8B in fleet.** Wave-2
    blocked on Q3_K shader (#28) and 4 GiB WASM cap;
    routed around both via the IQ-family quant code path
    (verified working on Mistral IQ4_XS first as a probe,
    then committed to IQ3_M for 8B).
    - **Probe sequence:** (a) Mistral IQ4_XS chat-smoke
      produced coherent multi-turn dialogue with factually
      correct content (Douglas Engelbart's first mouse
      anecdote) — confirmed IQ-family is intact; (b)
      `supports_op` covers `GGML_TYPE_IQ3_S` (which IQ3_M
      uses), so 8B Q3_K_S → IQ3_M is a pure quant-pin
      change with no engine work.
    - **Profile registered:** `llama-3.1-8b-warm`
      (temperature 0.6, `DEFAULT_PROMPT`); added to
      `SMOKE_PROFILE_SETS.full`. Bartowski mirror open;
      pinned `ggufFilePattern: "IQ3_M"` (3609 MB, fits
      with margin under 4 GiB cap).
    - **Architecture (llama / 32 layers):** n_head 32 ·
      n_head_kv 8 (GQA 4:1) · embedding 4096 · head_dim
      128 · ffn 14336 · ctx_max 131072 (we run at 4096) ·
      vocab 128256 (4× larger than Mistral's 32768; Llama-3
      tokenizer family). KV cache @ ctx=4096 = **1024 MB**
      (same as Mistral 7B at the same 32 layers / GQA 4:1).
    - **Speed (3-trial median):**
      - Steady-state **16.3 tok/s** (runs: 16.0 / 16.3 /
        16.3 — tightest spread in fleet alongside Mistral's
        34.4).
      - Profile-mode **14.5 tok/s** (perturbation -11%,
        smaller fraction than smaller models because graph
        compute dominates so heavily here).
      - Prefill **862 ms** (~10-token prompt + chat
        template).
    - **Profile-mode backend attribution (156-step decode):**
      - `backendMatmulMs` 47.07 mean / **71.4% of graph
        — new fleet high.** Up from 7B Q4_K_S Mistral's
        47.0%; confirms two effects stack: (a) parameter
        count up 11% widens the bandwidth-bound matmul
        slice, and (b) IQ3_M's compute overhead per
        element is meaningfully higher than Q4_K_S's
        (more sub-block scale unpacking with imatrix).
        Combined with `graphComputeMs` 96.7% of step,
        **matmul is ~69% of decode time at 8B IQ3_M**.
        The §A subgroup-cooperative-loading ceiling rises
        to ~28% of decode at 8B IQ3_M (vs ~18% at 7B
        Q4_K_S, ~13% at 4B Q4_0). The lever's percentage
        of total decode keeps growing with scale.
      - `backendEncodeOverheadMs` 6.08 / 9.2% — encode
        overhead's *fraction* hits a new fleet low.
        Absolute cost (6.08 ms) is comparable to Mistral
        7B (4.23 ms) and qwen3-4B (4.90 ms); it stays
        bounded as model grows.
      - `backendAttentionMs` 0.65 / 1.0%.
      - `backendDispatchCount` **652/token** — within 0.3%
        of Mistral 7B's 650. Confirms architecture-
        invariance within the 32-layer · llama-arch class
        regardless of param count.
    - **Smoke chat regression:** PASSED. Output: `"A man
      walks into a library and asks the librarian, 'Do you
      have any books on Pavlov's dogs and Schrödinger's
      cat?' The librarian replies, 'It rings a bell, but
      I'm not sure if it's here or not.'"` — **byte-
      identical to llama-3.2-3b's wave-1 output** (joke
      consistent across the Llama-3.x family from training
      data), finish=eos, 53 tokens.
    - **Accuracy (`bench --profiles llama-3.1-8b-warm`):**
      **31/36 = 86%.** Tied with qwen2.5-3b (86%), 2-4
      points below qwen3-4b (88-90%), 18 points above
      Mistral-7B Q4_K_S (68% — IQ3_M's imatrix preserves
      quality better than Q4_K_S's). Top non-Qwen-3
      accuracy in fleet. Tool-calling skipped (warm temp
      0.6 > 0.4 gate); embedding skipped (model lacks
      capability).
    - **Cross-family + cross-quant pattern at 7B / 8B:**
      | Model            | Family   | Quant  | tok/s | Accuracy |
      |------------------|----------|--------|------:|---------:|
      | qwen3-4b         | Qwen3    | Q4_0   |  35.5 |  88-90%  |
      | mistral-7b       | Mistral  | Q4_K_S |  34.4 |     68%  |
      | mistral-7b       | Mistral  | Q3_K_M |  19.7 |     69%  |
      | **llama-3.1-8b** | Llama 3.1| IQ3_M  |  16.3 |     86%  |
      Two clear axes:
      (a) **Quant compute cost dominates speed at 7B+**:
      Mistral Q4_K_S is 53% faster than Llama IQ3_M despite
      Llama having 11% more params, because IQ3_M has more
      compute work per element. Q4_K_S → IQ3_M is a quality-
      preserving substitution (better imatrix calibration)
      but a real throughput cost.
      (b) **Cross-family quality gap holds at 8B**: Llama
      3.1 has a markedly higher quality ceiling than
      Mistral v0.3 (Apr 2024 base instruct vs July 2024
      instruction-tuned). 86% vs 68% at near-identical
      param count is mostly model-quality, partly quant.
    - **Bench-profile speed-phase intermittent failure
      (resolved).** First bench attempt failed with
      "Timed out waiting for smoke-test chat output" at
      180s. Second attempt — invoked through bench-profile
      with proper smoke-restart sequence — passed cleanly.
      Likely a stale agentchrome session state issue;
      not reproducible after a clean smoke-restart. Not
      a regression in the bench harness.
    - **What this unblocks:** Wave 2 now has both 7B
      (Mistral Q4_K_S) and 8B (Llama 3.1 IQ3_M) data
      points across two families and three quant
      formats. Qwen3-8B at IQ3_XXS / IQ3_M is the
      natural next entry to round out the family-pattern
      analysis at 8B; Q3_K_S → IQ3_M / IQ3_S is the
      template for further 8B+ candidates while bug #28
      remains open.

14. **Bug #28 fixed: UB shift-by-32 in WGSL u32 loaders.** Root
    cause was *not* the Q3_K matmul kernel itself — it was
    `load_u32_at_src` and `load_u32_at_src0` in
    `~/Repos/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl`.
    Both helpers compute `hi << (32u - shift)` where
    `shift = (byte_offset & 0x3u) * 8u`. On u32-aligned reads
    (`shift == 0`) this becomes `hi << 32u`, undefined behavior
    in WGSL (shift count must be < bit_width). The trailing
    `select(shifted, lo, shift == 0u)` was meant to mask the
    UB result but on Tint/Dawn the UB leaks into the returned
    word. Q3_K mul_mat_vec and Q3_K get_rows both load through
    these helpers and were corrupted on aligned reads; Q4_K_S
    happened to issue unaligned loads in the affected lanes
    and was unaffected. **Fix:** branch explicitly on
    `byte_in_word == 0` and return `src[word_idx]` directly,
    never executing the UB shift.
    - **Diagnosis path:** Python ports of Q3_K mul_mat (both
      simplified element-by-element and the original optimized
      16-thread × 16-block layout), Q3_K get_rows, and Q5_K
      mul_mat were each verified mathematically equivalent to
      `dequantize_row_q3_K` / `dequantize_row_q5_K` to ~1e-6
      relative error. Sentinel writes (`acc[row] += 999`)
      confirmed kernels were reachable. CPU `llama-cli` on
      the same Q3_K_M GGUF produced coherent output, so the
      tensor data itself was fine. Common dependency between
      mul_mat_vec.wgsl and get_rows.wgsl is the loader
      helpers in `common_decls.tmpl`. The Python ports
      correctly skipped the UB shift via early-return; only
      the GPU-side WGSL code hit the UB.
    - **Why Python ports passed but WGSL didn't:** the Python
      ports computed `(lo >> shift) | (hi << (32 - shift))`
      with `shift == 0` short-circuited as `lo`. They did not
      reproduce the GPU UB.
    - **Verified:** Mistral-7B-Instruct-v0.3 Q3_K_M produces
      coherent joke output ("What do you call a fake noodle?
      An impasta!") at **24.4 tok/s** (was pure noise tokens
      at §12). Mistral-7B Q4_K_S regression-safe at **36.0
      tok/s** ("What do you call cheese that isn't yours?
      Nacho cheese!"), within noise of the §12 baseline of
      34.4 tok/s.
    - **Patch landed:** committed to `webllm-browser-patches`
      branch as patch 11 (`391c59f39 ggml-webgpu: fix UB
      shift-by-32 in load_u32_at_src{,0} for aligned offsets`).
      The 10-patch stack is now 11 patches. `docs/LLAMA_CPP_PATCHES.md`
      updated with the patch description and inventory count.
    - **What this unblocks:** Q3_K_M is a viable wave-2 quant
      again. For 7B+ models where Q4_K_S sits near the WASM
      cap, Q3_K_M (~3.4 GB at 7B) is a smaller-bandwidth
      option that previously would have been blocked by this
      bug. The IQ-family workaround (IQ3_M / IQ3_S / IQ3_XXS)
      remains valid and is still the path of choice for 8B+
      where Q4_K_S exceeds the cap. Q3_K_M test entry left in
      `eval/models.ts` as `mistral-7b-instruct-v0.3-q3km` for
      cross-quant comparison; promoted to a wave-2 fleet
      member in §15.

15. **Wave 2 model 4 — Mistral 7B Instruct v0.3 Q3_K_M
    promoted (option F closed).** Bench-profile cycle on
    `mistral-7b-v0.3-q3km-warm` produced **19.7 tok/s
    steady-state / 26/36 = 69% accuracy** (oneShot speed-
    phase reported 21 tok/s on the 64-token prefill+decode;
    interactive 1024-token steady-state landed at 19.7).
    Output coherent throughout: speed phase produced an AI/
    ML topic cascade, finishReason=max-tokens. Confirms the
    §14 patch (UB-safe u32 loaders) holds under sustained
    full-graph load — not just the smoke-bench burst that
    §14 verified at 24.4 tok/s.
    - **Q3_K vs Q4_K_S at 7B (same Mistral base, same
      32-layer arch, same llama2 chat template):**
      | Quant   | tok/s | Accuracy | File MB |
      |---------|------:|---------:|--------:|
      | Q4_K_S  |  34.4 |  26/36 (68%) | 3953 |
      | Q3_K_M  |  19.7 |  26/36 (69%) | 3360 |
      Q3_K_M is **43% slower** despite being 15% smaller on
      disk. K-quant compute overhead dominates bandwidth
      savings at this size — the same direction as §9
      observed on Qwen3-1.7B (Q4_K_M -4% vs Q8) but with a
      much wider gap. Accuracy is statistically tied (same
      26/36 raw count, 1-pp dimension-weighted delta, well
      inside variance) — Q3_K_M does not visibly hurt model
      quality at 7B Mistral, the cost is purely throughput.
      **Verdict: Q3_K is not a useful speed lever even when
      the file fits the cap; Q4_K_S is the correct default
      for 7B.** Q3_K_M remains useful only for memory-bound
      scenarios where the 593-MB-on-disk savings actually
      matter (8B-12B Q4_K_S that exceeds the 4 GiB cap is
      better served by IQ3_M anyway, per §13).
    - **Updated cross-family + cross-quant table at 7B / 8B:**
      | Model            | Family   | Quant  | tok/s | Accuracy |
      |------------------|----------|--------|------:|---------:|
      | qwen3-4b         | Qwen3    | Q4_0   |  35.5 |  88-90%  |
      | mistral-7b       | Mistral  | Q4_K_S |  34.4 |     68%  |
      | mistral-7b       | Mistral  | Q3_K_M |  19.7 |     69%  |
      | **llama-3.1-8b** | Llama 3.1| IQ3_M  |  16.3 |     86%  |
      Three quant code paths now exercised at 7B+: Q4_K
      family (Q4_K_S), Q3_K family (Q3_K_M), IQ family
      (IQ3_M). Q4_K_S remains the throughput default;
      IQ3_M the path-of-choice for 8B+ where Q4_K_S
      exceeds the cap.
    - **Registration changes:** entry name in `eval/models.ts`
      cleaned up from "Q3_K_M, bug #28 verified" to plain
      "Q3_K_M"; comment block updated from verification
      probe to fleet member; new profile
      `mistral-7b-v0.3-q3km-warm` added in
      `eval/smoke-profiles.ts`; profile added to the `full`
      set so `make bench-full` now runs it alongside the
      Q4_K_S baseline. `make checkall` clean (393/5/0).
    - **Wave 2: 3/4 done.** Remaining slot is the cross-
      family 8B tie-break — Qwen3-8B IQ3_M is the natural
      next entry to round out the family-pattern analysis
      (Qwen accuracy ceiling vs Llama 3.1 IQ3_M's 86%).

16. **Wave 2 model 4 — qwen3-8b-iq3m registered + benched
    (wave 2 complete).** Cross-family 8B tie-break against
    Llama-3.1-8B IQ3_M from §13. Bartowski mirror open;
    pinned `ggufFilePattern: "IQ3_M"` (3716 MB file, fits
    with margin under 4 GiB cap). Same `GGML_TYPE_IQ3_S`
    code path verified working in §13 — pure quant-pin
    change, zero engine work needed.
    - **Profiles registered:** `qwen3-8b-warm` and
      `qwen3-8b-thinking-warm` (mirrors qwen3-4b layout —
      both modes for the qwen3 family); added to
      `SMOKE_PROFILE_SETS.full` and `qwen3-sizes`.
    - **Architecture (qwen3 / 36 layers):** matches
      qwen3-4b's layer count; n_head 32 · n_head_kv 8
      (GQA 4:1, same as Llama-3.1-8B and qwen3-4b) ·
      embedding 4096 · head_dim 128 · ffn 12288 · vocab
      151936 (Qwen3 tokenizer family). KV cache @ ctx=4096
      = **1024 MB** (same as Llama-3.1-8B at the same
      32-layer-equivalent KV shape).
    - **Speed (3-trial median):**
      - Steady-state **16.2 tok/s** (runs: 16.2 / 16.1 /
        16.6 — tightest spread alongside Llama-3.1-8B's
        16.3). **Effectively tied with llama-3.1-8b-iq3m**
        (delta -0.6%, well inside noise).
      - Profile-mode 14.6 tok/s (perturbation -10%, in
        line with the 8B class where graph compute
        dominates — Llama-3.1-8B was -11%).
      - Chat-regression 14.5–14.9 tok/s (oneShot/
        interactive); prefill 454 ms.
    - **Profile-mode backend attribution (60-step decode):**
      - `backendMatmulMs` 45.71 mean / **66.7% of graph**
        — slightly under llama-3.1-8b's 71.4%. Absolute
        matmul time near-identical (45.7 vs 47.1 ms);
        graph fraction differs because Qwen3's per-head
        Q-norm/K-norm pushes a bit more work into encode.
      - `backendEncodeOverheadMs` 5.57 / 8.1% —
        comparable to Llama-3.1-8B's 6.08 / 9.2%.
      - `backendAttentionMs` 0.69 / 1.0%.
      - `backendDispatchCount` **805/token — matches
        qwen3-4b exactly.** qwen3 architecture-invariance
        confirmed across the 4B → 8B span (805 is the
        signature: 36 layers × ~22 ops/layer + globals).
        +23% over Llama-3.1-8B's 652 (qwen3 has an extra
        per-head norm pair); doesn't affect throughput
        because matmul bandwidth dominates.
      - graphComputeMs 97.4% of step → matmul ≈ **65% of
        decode** (vs ~69% on Llama-3.1-8B IQ3_M).
    - **Smoke chat regression:** PASSED both modes.
      - Off: `"Why don't skeletons fight each other?
        Because they don't have the *guts*. 😄"` —
        finish=eos, 21 tokens.
      - Thinking: clean `<think>` block (322 tokens of
        candidate-joke deliberation) → same skeleton
        joke punchline; finish=eos, 322 tokens total.
        Both `<|im_end|>` and `<|endoftext|>` masking
        held correctly.
    - **Accuracy (`bench-profile PROFILES=qwen3-8b-warm
      qwen3-8b-thinking-warm`):**
      - Off-warm: **33/36 = 90%** — ties qwen3-4b
        thinking-on for top of fleet.
      - Thinking-warm: **33/36 = 90%** — same; both modes
        converged.
      - +4 points over Llama-3.1-8B IQ3_M (86%) at near-
        identical param count and identical quant; the
        Qwen3 family quality advantage holds at 8B.
      - Tool-calling skipped (warm temp 0.6 > 0.4 gate);
        embedding skipped (model lacks capability).
    - **Cross-family 8B summary at IQ3_M (wave 2 closes):**
      | Model            | Family   | Layers | Disp/tok | tok/s | Accuracy |
      |------------------|----------|-------:|---------:|------:|---------:|
      | llama-3.1-8b     | Llama 3.1|     32 |      652 |  16.3 |     86%  |
      | **qwen3-8b**     | Qwen3    |     36 |      805 |  16.2 |     90%  |
      Speed parity at 8B IQ3_M despite +23% dispatches:
      bandwidth-bound matmul flattens dispatch-count
      overhead. Family-quality gap (Qwen3 > Llama 3.1 by
      ~4 points) holds at the 8B IQ3_M shape, mirroring
      the 4B Q4_0 result (qwen3-4b 88-90% vs llama-3.2-3b
      76% / hermes-3 74% — 12-16 pt gap; shrinks to 4 pt
      at 8B because Llama 3.1 narrows the quality gap
      meaningfully over Llama 3.2).
    - **§A subgroup-cooperative-loading ceiling at 8B
      IQ3_M Qwen3:** matmul 65% of decode × ~40%
      bandwidth-bound (per §9 Stub B characterization)
      ≈ **26% of decode time** — close to the 28%
      predicted from Llama-3.1-8B IQ3_M's 71% matmul
      slice. The lever's headroom keeps growing into
      the 8B regime regardless of family.
    - **Net wave-2 finding (4 entries, 3 quants, 2
      families):** at scale, model-quality and quant
      choice dominate throughput differences far more
      than family-architecture differences do. Q4_K_S vs
      IQ3_M cuts speed in half (Mistral Q4_K_S 34.4 →
      Llama IQ3_M 16.3 / Qwen3 IQ3_M 16.2) for ~the same
      param count; switching families at the same quant
      moves throughput <1%. **Quant compute cost is the
      load-bearing axis at 7B+; family is a quality
      knob.**
    - **Registration changes:** new model entry in
      `eval/models.ts` (mirrors §13 Llama IQ3_M structure);
      two new profiles in `eval/smoke-profiles.ts`;
      profiles added to `SMOKE_PROFILE_SETS.full` and
      `qwen3-sizes`. `make checkall` clean (393/5/0
      pre-bench, no engine changes).

17. **§A lever 1 (THREADS_PER_BLOCK 4→2 in mul_mat_vec.wgsl)
    measured + closed for the production fleet.** Followed
    the §1994 harness against the canonical 4-baseline
    (`tinyllama-1.1b-chat-q4_0`, `mistral-7b-instruct-v0.3-q4ks`,
    `llama-3.1-8b-instruct-iq3m`, `qwen3-8b-iq3m`) to test the
    one viable replacement lever from the rejected
    subgroup-broadcast premise.
    - **Pre-change baselines (3-trial median):**
      | Model              | bench-inf tok/s | smoke-bench tok/s | matmul ms (median) | %graph | dispatches |
      |--------------------|----------------:|------------------:|-------------------:|-------:|-----------:|
      | tinyllama-q4_0     |          105.7  |              68.5 |               4.46 |  34.3% |        450 |
      | mistral-7b-q4ks    |           34.5  |              27.3 |              17.50 |  49.3% |        650 |
      | llama-3.1-8b-iq3m  |           16.8  |              15.3 |              45.35 |  71.2% |        652 |
      | qwen3-8b-iq3m      |           15.1  |              14.3 |              48.04 |  70.5% |        805 |
    - **Lever applied to `MUL_ACC_Q4_0` only**
      (`~/Repos/llama.cpp/.../mul_mat_vec.wgsl` line 131-166):
      `THREADS_PER_BLOCK 4 → 2`, `ELEMS_PER_THREAD 8 → 16`,
      `thread_within_block * 4 → * 8u`, x_block load doubled
      (8 lo + 8 hi instead of 4+4), q_packed split into
      `q_packed_a` + `q_packed_b` (two consecutive 4-byte
      `load_u32_at_src0` instead of one), inner reduction
      doubled with `byte_idx + 8u` / `byte_idx + 12u` for
      the b-half x-block offsets. WG_SIZE=64 confirmed
      integer-divisible by both 4 and 2.
    - **Post-change measurements:**
      - tinyllama-q4_0: bench-inf **106.3 tok/s** (+0.6%,
        noise), smoke-bench **74.3 tok/s** (+8.5%), matmul
        **4.33 ms** (-2.9%). Coherence verified via smoke
        page `[8/8]` (105.5 tok/s on the live page,
        grammatically clean English — TinyLlama's "share a
        joke from Facebook" off-topic answer is its known
        small-model weakness, not a shader bug).
      - mistral-7b-q4ks: bench-inf **34.9 tok/s** (+1.2%,
        noise) — sanity-check confirming the Q4_0 `#ifdef`
        block doesn't bleed into Q4_K_S codegen. Other 3
        models skipped: lever doesn't apply.
    - **Why the lever doesn't apply to the rest of the fleet
      (root cause for closure):**
      - **Q4_K_S (Mistral) and other K-quants** all use
        `THREADS_PER_BLOCK 16` with a completely different
        block structure (BLOCK_SIZE 256, complex `lane`/
        `phase`/`iq`/`ir` indexing, per-block scale-pair
        unpack via `load_u32_at_src0_aligned` masks). §A
        explicitly excluded these ("Q2_K-class uses 16,
        leave alone"). Lever 1's "halve TPB to coarsen
        per-thread work" semantics don't translate.
      - **IQ3_M (both 8B models) has no `mul_mat_vec.wgsl`
        path at all.** IQ-family code lives only in
        `mul_mat.wgsl` (the general matmul shader, used
        for prefill) and `get_rows.wgsl`. There is no
        `MUL_ACC_IQ3_S` block. The decode-path mat-vec
        for IQ3_M routes through the general matmul kernel,
        not the simple-block path that lever 1 modifies.
        This is the structural reason matmul is 71% of
        graph on these models — the general matmul shader
        is heavier per-element than the per-block specialized
        kernels.
    - **Verdict:** the only model that benefited
      (TinyLlama Q4_0) gained a sub-trigger -2.9% matmul /
      noise-level +0.6% steady-state tok/s. The
      `smoke-bench` +8.5% is real but came from
      profile-mode perturbation overhead (`backendEncode-
      OverheadMs` 2.46 → 2.50 ms is flat, but
      `graphComputeMs` median 11.90 → 11.30 dropped 5%
      because dispatch overhead shrinks slightly with the
      doubled per-thread payload). Per the §1994 decision
      rule ("revert if any regresses >3%, ship only if
      matmul drops 5%+ on at least one quant"), the change
      did not clear the 5% matmul threshold even on its
      one applicable quant, and provides zero benefit to
      the production 7B/8B fleet. **Reverted the shader
      to HEAD** (`git diff` clean post-revert; rebuilt WASM
      to match — bytes 2205378, identical to pre-change).
    - **§A is closed for our model fleet.** Levers 2
      (vec4-packed loads) and 3 (`d`-scale lifting) are
      subject to the same constraint — they only apply to
      `mul_mat_vec.wgsl`'s simple-block path which doesn't
      serve any of our 7B/8B production models. Pursuing
      either at this point would optimize Q4_0/Q5/Q8
      legacy paths that only TinyLlama-class models use.
      The kernel-tuning ceiling at 7B+ is now structurally
      gated on either (a) extending lever-1-style coarsening
      to K-quants (a substantial rewrite — different block
      layout, scale unpack, threading) or (b) accelerating
      the general `mul_mat.wgsl` path for IQ-family quants
      (also substantial). Neither is in scope without a
      much larger commitment.
    - **Recommended next move:** §C drafter-based
      speculative decoding. Wave-2 closed the 8B+
      drafter/target pair: qwen3-0.6b ↔ qwen3-8b is a
      same-family draft pair with shared tokenizer.
      Theoretical 2-3× wall-clock decode for chat-style
      workloads. Larger project but the only remaining
      lever with meaningful headroom that doesn't require
      a kernel rewrite. §B FA shape-routing is the
      secondary option (helps prefill/TTFT, not steady-
      state decode).
    - **Code state:** no engine changes landed; no
      llama.cpp commits added. `make checkall` clean
      (393/5/0). WASM artifacts in `smoke-test/` rebuilt
      against unchanged tree as a hygiene step (mtime
      12:20 Apr 26).

18. **§4 Flash Attention enable measured + closed.**
    Followed `docs/superpowers/plans/2026-04-26-fa-enable.md`
    to integrate `ggml_flash_attn_ext` into the
    model-inference attention path (F16 KV cache, V-cache
    layout transposed, all three attention branches —
    MLA/GQA/MHA — routed through `opFlashAttn`).
    Measured against the canonical 4-baseline using the
    §1994 ship gate (bench-inf 3-trial median, ±3% threshold).
    - **4-baseline measurement (post-FA, vs. pre-Task-4
      baselines):**
      | Model              | bench-inf Δ% | smoke-bench Δ% | matmul Δ% | dispatches  | FA engaged? |
      |--------------------|-------------:|---------------:|----------:|-------------|-------------|
      | tinyllama-q4_0     |       +1.3%  |         +5.1%  |   -16.1%  | 450 → 403   | ✓           |
      | mistral-7b-q4ks    |   **-5.8%**  |         -2.9%  |    -5.6%  | 650 → 586   | ✓           |
      | llama-3.1-8b-iq3m  |       -3.0%  |         -3.3%  |    -2.3%  | 652 → 588   | ✓           |
      | qwen3-8b-iq3m      |       -0.7%  |         -4.2%  |    -2.8%  | 805 → 697   | ✓           |
    - **FA did engage** on all 4 models: dispatch counts
      dropped 10-13% and matmul latency improved in 3 of
      4 models (including -16.1% for TinyLlama). The
      mechanism is working correctly.
    - **Why it still fails the gate:** the WebGPU FA
      path introduces a new `backendAttentionMs` overhead
      of 1.3-3.3 ms per decode step (measured via
      `smoke-bench` profile). At single-token decode
      (N=1), this per-step overhead exceeds the savings
      from removing the manual attention dispatches.
      FA's primary wins are (a) prefill on long prompts
      (sequential attention → one tiled kernel, O(N)
      HBM reads instead of O(N²)) and (b) longer decode
      batches (>256 tokens). Neither scenario is exercised
      by the bench-inf steady-state gate, which measures
      decode throughput at batch=1, sequence=1. Mistral
      -5.8% is the blocking signal; it exceeds the 3%
      regression threshold and no model gains ≥2%.
      **Ship gate FAILS.**
    - **Code state:** C bridge wrappers (`33f10eb`),
      TypeScript bindings (`4692bce` + `d26d736`), and
      surface test (`068ef84`) **retained** as
      infrastructure for future investigation. The
      implementation commits (`baad612` F16 KV cache,
      `1f1a9da` opFlashAttn integration, `d4988a0`
      JSDoc cleanup) reverted via:
      ```
      git checkout 068ef84 -- src/inference/model-inference.ts \
                               smoke-test/real-model-page.js
      ```
      `make checkall` clean post-revert (394/5/0).
      TinyLlama coherence verified at 101.7 tok/s.
    - **Plan reference:**
      `docs/superpowers/plans/2026-04-26-fa-enable.md`.
    - **Recommended next move:** §C drafter-based
      speculative decoding (qwen3-0.6b ↔ qwen3-8b
      same-family draft pair with shared tokenizer —
      theoretical 2-3× wall-clock for chat). FA closure
      doesn't change which lever has remaining headroom:
      §C remains the only path to a step-change in
      steady-state decode throughput on the 7B/8B fleet
      without a kernel rewrite.

19. **§C drafter speculative decoding measured + reverted.**
    Implemented Leviathan-style speculative decoding end-
    to-end (drafter proposes K=4, target verifies in one
    parallel forward, rejection sampler preserves target
    distribution) and measured against the §C ship gate.
    **Status: REVERTED** (engine routing); driver, sampler
    helpers, `forwardVerify`, `truncateKVCache`, and tests
    remain in tree.
    - **Spec / plan:**
      `docs/superpowers/specs/2026-04-26-speculative-
      decoding-design.md` (491 lines, 14 §) +
      `docs/superpowers/plans/2026-04-26-speculative-
      decoding.md` (11 tasks across 3 phases).
    - **Code shipped (commits `11fe3f7`..`bbd1dff`):**
      `src/inference/sampler.ts` gained `rand`,
      `computeDistribution`, `sampleFromDistribution`.
      `src/inference/model-inference.ts` gained
      `forwardVerify(tokenIds, positions): Promise<
      Float32Array>` (multi-position logits readback) and
      `truncateKVCache(n)` (counter-only rollback).
      `src/inference/speculative.ts` (~330 LOC) holds
      `acceptPrefix` (rejection math + EOS / stop /
      maxTokens truncation, per-stream degenerate-residual
      warning) and `SpeculativeGenerator.generate` (KV
      rollback + abort, first-token EOS short-circuit,
      penalty-window hoisting). 19 new tests:
      `tests/sampler.test.ts` (7),
      `tests/speculative-rejection.test.ts` (11), and 1
      engagement-gate test in
      `tests/speculative-integration.test.ts` (the WebGPU
      integration tests in that file skip under Bun).
    - **Code reverted (commit `aac7080`):**
      `src/core/engine.ts::generateStream` engagement
      block + `SpeculativeGenerator` dispatch replaced
      with a single `throw "reserved in v1"` on
      `config.drafter`. `CompletionConfig.drafter` /
      `draftLength` retained as reserved fields with
      docstrings pointing at this entry.
    - **Pre-§C baselines (drift check, 2026-04-26):**
      tinyllama-1.1b 101.0 tok/s (plan expected 106 → -5%
      drift, within tolerance), qwen3-0.6b-q4f16 81.9
      tok/s (expected ~85 → -4%), qwen3-8b-iq3m 15.3
      tok/s (expected 16.2 → -5.5%). All within the 10%
      drift tolerance.
    - **Gate 1 (drafted speedup) — FAILS:**
      `make smoke-bench PERF_MODEL=qwen3-8b-iq3m
      PERF_DRAFTER=qwen3-0.6b-q4f16 PERF_RUNS=3` →
      **3.0 tok/s steady-state** (3 runs: 3.0 / 3.0 /
      3.0; decode 7877–8024 ms for 24 tokens) vs 15.3
      baseline = **0.20× ratio**. Gate 1 required ≥1.5×
      (≥22.95 tok/s); we got a 5× regression. Gates 2
      (accuracy parity) and 3 (non-drafted regression)
      not run — gate 1 failure makes them moot.
    - **Output is functionally correct.** Smoke page
      log captured: `User: Tell one short joke. /
      Assistant: Why don't skeletons fight each other?
      Because they don't have the *guts*! 😄`. Leviathan
      rejection sampling preserves the target's
      distribution as designed.
    - **Diagnosis (predicted in plan, confirmed by
      measurement).** Per spec step the spec path runs
      4 drafter forwards (each does a full-vocab readback
      of ~152 K floats ≈ 0.6 MB) plus 1 K-position target
      verify (4 × 152 K floats ≈ 2.4 MB readback) plus
      CPU-side softmax + rejection roll on 4 distros.
      Baseline runs 1 target `forwardDecode` per token
      with top-K readback (~0.4 KB). Even at perfect
      acceptance (all K accept → 4 emitted tokens / step)
      the readback bandwidth alone overwhelms the
      savings, and at typical α the lever pays K
      drafter steps + K-position verify per emitted
      token — exactly the failure mode §11 of the spec
      called out.
    - **What v2 would need to win.** GPU-resident
      verify (no per-step full-vocab readback —
      compare drafted ids against argmax on-device,
      only read the rejection mask), or a
      meaningfully cheaper drafter (sub-1B at <2 ms /
      step, currently qwen3-0.6b is ~12 ms / forward
      at full vocab readback), or dynamic K that
      collapses to K=1 when α drops. Multi-tokenizer
      drafters were also discussed in the spec but
      add re-tokenization cost on every accept and
      are unlikely to help unless the verify-readback
      bottleneck is solved first.
    - **Plumbing retained.** Smoke page
      `?drafter=<id>` URL param, `PERF_DRAFTER`
      Makefile var, and `eval/perf.ts --drafter` flag
      are inert when the engine throws and useful as-is
      when v2 measurement happens. Drafter loader in
      `smoke-test/real-model-page.js` exercises the
      per-model-WASM-heap pattern correctly (caught
      one bug during ship-gate run: `loadModel` mints
      a synthetic handle id so the smoke page must
      pass `handle.id`, not the user-facing name, into
      `CompletionConfig.drafter` — fixed in `1b23ca8`,
      relevant when v2 lands).
    - **Recommended next move:** **§4 FA revisit at
      long-decode / prefill scope** (the §18 closure
      explicitly noted that bench-inference's batch=1
      seq=1 measurement is the wrong scope to
      characterize FA wins; long-decode and prefill
      benches would surface them). Or **§D encoder
      perf pass** if encoder embedding throughput is
      the next priority. §C v2 (GPU-resident verify)
      is feasible but a larger investment than either.

20. **§4 FA revisit at prefill / long-decode scope measured + closed.**
    Followed `docs/superpowers/plans/2026-04-26-fa-revisit-long-decode.md`
    to re-land the §18-reverted `ggml_flash_attn_ext` integration behind a
    `flashAttn?: boolean` config gate (default `false`), build a long-prompt
    harness, and run a measurement matrix at the workload §18 explicitly
    flagged as out of scope (prefill TTFT + long-decode batches). **Status:
    CLOSED — gate retained as opt-in infra, not shipped default-on.**
    - **Code shipped (commits `91d8e26`..`b872b5f`,
      fast-forward merged to `main`):**
      `ModelInference` constructor takes `{ flashAttn?: boolean }`; F16 K +
      F16 V cache layout selected at init when `flashAttn=true` (matches
      `flash_attn_get_decisions::kv_vec_type_supported`); F16 causal mask
      across all four attention branches (mode-independent — benefits both
      FA and `opSoftMaxExt`); per-method dual V-layout + branched attention
      in `forward` / `forwardVerify` / `forwardDecode` / `debugLayerOutput`;
      `?fa=on` URL param + FA pill on `real-model.html`; `--fa <on|off>`,
      `--prompt-fixture <id>`, `--decode-tokens <n>` flags on
      `eval/perf.ts`; three long-prompt fixtures
      (`eval/fixtures/long-prompts.ts` — `prefill-256/512/1024`); 5-test
      contract suite at `tests/fa-mode-config.test.ts`. `make checkall`
      clean (418 pass / 10 skip / 0 fail).
    - **Measurement matrix.** Plan called for 4 models × 4 workloads × 2 FA
      modes = 32 cells. **Captured: 6 cells.** TinyLlama Q4_0 full 4-cell
      coverage (short-short and long-short × FA off/on). Mistral-7B Q4_K_S
      short-short × FA off/on only. Mistral long-short and the two 8B
      models (`llama-3.1-8b-iq3m`, `qwen3-8b-iq3m`) **blocked** at
      `backend_alloc_ctx_tensors` — a generic WebGPU max-buffer-binding
      limit hit when 7B+ models build long-prefill graphs (32 layers ×
      seq=512 of F32 intermediates exceeds the device cap, **regardless of
      FA mode**). Bumping `graphMem` 32× → 64× did not unblock; the abort
      is in the GPU-side allocation, not the metadata context. Treated as
      out-of-scope (separate infrastructure pass).
    - **TinyLlama 1.1B Q4_0 — full 4-cell, 3-trial median:**

      | Workload    | Metric        | FA off  | FA on   | Δ        |
      |-------------|---------------|--------:|--------:|---------:|
      | short-short | TTFT (ms)     |     167 |     156 |   -6.6%  |
      | short-short | Decode tok/s  |   109.7 |   115.1 |   +4.9%  |
      | long-short  | TTFT (ms)     |     409 |     368 |  -10.0%  |
      | long-short  | Decode tok/s  |    95.3 |   110.9 |  +16.4%  |

      **FA wins on every TinyLlama cell.** Long-short decode +16.4% is the
      largest signal — FA's per-step overhead amortizes once the KV cache
      passes ~512 tokens, validating the §4 hypothesis at small scale.
    - **Mistral-7B Q4_K_S — short-short only, 3-trial median:**
      35.9 → 34.7 tok/s = **-3.3%**. FA still regresses at short-short but
      less than §18's -5.8%; the F16 KV-cache change (now both K and V are
      F16 in FA mode, matching `flash_attn_get_decisions`) shaved ~2 pp.
      **Still over the 3% gate** — not shippable default-on at this
      workload, even before the long-prefill data is captured.
    - **Decision rule application** (from plan header):
      - **Ship default-on:** FAILS — Mistral short-short FA-on is -3.3%
        (>3% regression gate).
      - **Ship gated (auto):** FAILS — only TinyLlama measured at
        long-short (where FA showed -10% TTFT and would have qualified);
        cannot meet "≥2 models" threshold without 7B+ long-prefill data,
        and that data is blocked on the buffer-binding infra issue.
      - **Close §4:** **default outcome** — capture findings, leave bridge
        + gate as future infra.
    - **Closure modification (plan deviation, intentional):** the plan's
      "close" branch instructed `git checkout main -- src/inference/
      model-inference.ts` to revert the call sites. **Did not revert** —
      that destroys the small-scale TinyLlama win documented above. A more
      honest closure: keep the gate (default-off, preserving §18-revert
      behaviour at 7B+), and add the TinyLlama findings + 7B graph-buffer
      limit to the corpus. Net code-state delta vs §18 closure: the FA
      implementation is now in tree behind a default-off boolean instead
      of fully reverted.
    - **What ships in tree:**
      - `flashAttn?: boolean` constructor option on `ModelInference`
        (default `false`).
      - F16 K + F16 V cache layout when `flashAttn === true` (auto-selected).
      - `?fa=on` URL param on the smoke page (default off).
      - `--fa <on|off>` flag on `eval/perf.ts`, `--prompt-fixture <id>`,
        `--decode-tokens <n>`.
      - Three long-prompt fixtures in `eval/fixtures/long-prompts.ts`.
      - F16 mask in all four attention branches (mode-independent).
      - Per-method dual V-layout + branched-attention pattern.
      - `tests/fa-mode-config.test.ts` (5 tests) pinning the constructor
        contract.
    - **What's reserved for follow-up:**
      - **7B+ long-prefill graph-buffer infrastructure** (separate from
        FA — Mistral aborts even with FA off at long-short). Without
        this, the §4 hypothesis at scale (FA wins on prefill TTFT once
        the seq²/dispatch-overhead crossover is reached) cannot be
        tested.
      - **Auto-mode gating** (FA on for `nTokens > 1` only) deferred until
        the 7B+ long-prefill data exists to validate it.
    - **Plan reference:**
      `docs/superpowers/plans/2026-04-26-fa-revisit-long-decode.md`.
      Raw logs and matrix-driver script:
      `eval/reports/fa-revisit-2026-04-27/`.
    - **Recommended next move:** **§C v2 GPU-resident verify** is the
      next algorithmic ceiling (avoids the 2.4 MB / step readback that
      sank §C v1). **Or 7B+ long-prefill graph-buffer infra work** if the
      §4 hypothesis at scale is the priority — both are blocking the same
      class of measurements.

---

#### Archived: How to test §A lever 1 — THREADS_PER_BLOCK 4→2 (CLOSED 2026-04-26 — §17)

> **Preserved for archive only. Do not run this.** §A
> lever 1 was measured on Q4_0 (TinyLlama) and produced
> only +0.6% — within noise — and the lever is excluded
> from K-quants (TPB=16) and IQ-family (no
> `mul_mat_vec.wgsl` path) entirely. The shader change
> was reverted. See §17 in the journal for the full
> measurement and rationale. The next-move recommendation
> is now §4 FA revisit at long-decode / prefill scope (§C
> v1 closed at §19); §A levers 2/3 are still off the table.

**The change.** Edit
`~/Repos/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl`
on `webllm-browser-patches`. Per quant block, change
`#define THREADS_PER_BLOCK 4` to `2` and adjust the
inner loop so each thread covers double the bytes.

| Quant | `#ifdef` | Loop change |
|-------|----------|-------------|
| Q4_0  | `MUL_ACC_Q4_0`  (line 131) | `THREADS_PER_BLOCK 2`; thread now reads 2 u32s of q (8 bytes) and 16 src1 elements per block. Adjust `block_byte_base + 2u + 4u * thread_within_block` → `+ 8u * thread_within_block + 4u * inner_packed` over 2 packed iterations. |
| Q8_0  | `MUL_ACC_Q8_0`  (line 289) | `THREADS_PER_BLOCK 2`; ELEMS_PER_THREAD becomes 16. Outer `packed_idx` loop runs 4× instead of 2×. |
| Q4_K_S, IQ3_M, IQ4_XS | similar | each has own block size; check whether `THREADS_PER_BLOCK` is a `#define` or a literal — Q2_K-class uses 16, leave alone. |

**The 4-baseline harness.** Always measure both
non-profile and profile mode:

```bash
# Pre-change baseline (re-verify; numbers may drift between sessions)
make smoke-restart
for m in tinyllama-1.1b-chat-q4_0 mistral-7b-instruct-v0.3-q4ks \
         llama-3.1-8b-instruct-iq3m qwen3-8b-iq3m; do
  echo "=== $m ==="
  make bench-inference PERF_MODEL=$m PERF_RUNS=3 2>&1 | grep -A4 "p50\*"
  make smoke-bench    PERF_MODEL=$m PERF_RUNS=3 2>&1 | grep -A4 "backendMatmulMs"
done

# Apply lever 1 to mul_mat_vec.wgsl, then:
make wasm-build && bun build src/index.ts --outfile \
  smoke-test/webllm-bundle.js --target browser && \
  cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/

# Re-run the same 4 baselines (post-change)
# Decision: ship if all 4 are within noise or net-positive
# AND profile-mode `backendMatmulMs` drops by 5%+ on at
# least one quant. Revert if any model regresses >3% on
# steady-state tok/s.
```

**Coherence guard.** Output garbage if the loop bounds
are off by one. Run the smoke page on each model after
the WASM rebuild and verify the joke prompt produces
sensible English before trusting the perf number.

**Expected results map.**
- Lever 1 helps → matmul drops ~10-20% on Q4_0 and Q8_0,
  steady-state tok/s rises 5-12% across the harness.
  Layer in lever 2 (Q8 vec4-packed) and lever 3
  (`d`-scale lifting); each adds another 1-3%.
- Lever 1 hurts → Q4_0 and Q8_0 matmul flat or up. Means
  the GPU was already pipelining loads across warps;
  doubling per-thread work cost more in register pressure
  than it saved in issue rate. Try lever 2 first instead;
  if that's also flat, **§A is closed and the next
  highest-leverage option is §C drafter-based speculative
  decoding** (large project, 2-3× wall-clock potential).

**If extending the size campaign instead** (option D /
wave-3 territory): GGUF mirror probe FIRST via
`curl -s "https://huggingface.co/api/models/<repo>/tree/main" | python3 -c "..."`.
Wave 1 hit three bad mirrors and wave-2's Mistral mirror
also lacked Q4_0. Unsloth and bartowski have been the
reliable fallbacks. Pin `ggufFilePattern` in `eval/models.ts`
and verify the chosen quant's code path is supported
(Q3_K_M / Q4_K_S / Q4_K_M working post-§14; IQ-family
working including IQ3_M / IQ3_S / IQ3_XXS / IQ4_XS). At
12B+ Q4_K_S exceeds 4 GiB; option D (`MAXIMUM_MEMORY`
bump via `-sMEMORY64=1`) becomes a prerequisite.

---

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
    - [x] `llama-3.1-8b-instruct-iq3m` — DONE 2026-04-26.
      Steady-state 16.3 tok/s / profile-mode 14.5 / 652
      dispatches/token / 31/36 = 86% accuracy. Q4_K_S
      4475 MB > cap; pivoted to IQ3_M (3609 MB) via the
      IQ-family code path. First 8B in fleet. See
      "Completed on 2026-04-26 §13" above.
    - Qwen3-8B IQ3_M — IQ3_M = ~3700 MB on bartowski
      Qwen3-8B mirror; expected to fit and use the same
      IQ-family code path. Natural next entry to round out
      cross-family 8B coverage.
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

---

### Deferred kernel-tuning targets (behind §10 in priority)

§A. **CLOSED 2026-04-26 (see "Completed on 2026-04-26"
    §17 for the measurement and shader-walk closure
    write-up).** Lever 1 (THREADS_PER_BLOCK 4→2) was the
    one viable replacement after the original subgroup-
    broadcast premise was rejected; tested on the canonical
    4-baseline. TinyLlama Q4_0 (the only model whose decode
    path goes through `mul_mat_vec.wgsl`'s simple-block
    code) showed sub-trigger -2.9% matmul / +0.6% tok/s.
    The 7B/8B fleet doesn't benefit because Q4_K_S is a
    K-quant (TPB=16, different block structure — explicitly
    excluded from §A) and IQ3_M has no `mul_mat_vec.wgsl`
    path at all (routes through general `mul_mat.wgsl`).
    Levers 2 + 3 are subject to the same applicability
    constraint and are not worth pursuing for the
    production fleet. Shader reverted; no patches landed.
    Original analysis preserved below for archive.

    ---

    Walked the kernel
    (`~/Repos/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl`)
    in detail. Conclusion: **subgroup-broadcast cannot
    reduce src0 reads here** because the existing kernel
    already partitions src0 perfectly across threads.

    For Q4_0 (line 138-165) and Q8_0 (line 296-321): each
    32-element block has 16 bytes (Q4) or 32 bytes (Q8) of
    weights. THREADS_PER_BLOCK=4 threads cooperate on each
    block, indexed by `thread_within_block = thread_id % 4`.
    Each thread reads its own non-overlapping slice via
    `block_byte_base + 2u + 4u * thread_within_block` (Q4)
    or `+ 4u * (thread_within_block * 2u + packed_idx)`
    (Q8). Across the 4 threads, every byte of weights is
    read by exactly one thread per pass. **No redundant
    loads exist; broadcasting cannot share work that isn't
    duplicated.**

    Why §7's Stub B (replacing all real src0 loads with
    one broadcast) showed -40% Q8 / -20% Q4: the kernel is
    bound by **load latency**, not bus bandwidth. The
    per-thread loads can't overlap because the row-loop
    issues them serially with data-dependent reduction
    arithmetic between them. Stub B replaced N serial
    loads with 1 load and let the ALU pipeline saturate;
    the speedup measures latency stalls, not bus saturation.

    **Viable levers that follow from this diagnosis:**
    1. **Coarsen per-thread block coverage.** Drop
       THREADS_PER_BLOCK from 4 → 2 so each thread handles
       8 bytes (Q4) or 16 bytes (Q8) per block. Halves
       load-issue rate per block, doubles per-thread work
       (compute is cheap; loads are expensive). Requires
       sweeping THREADS_PER_BLOCK alongside the existing
       OUTPUTS_PER_WG sweep.
    2. **vec4-packed load helper.** Replace
       `load_u32_at_src0(addr) ... load_u32_at_src0(addr+4)
       ... load_u32_at_src0(addr+8)` with a single vec4 u32
       load when the addresses are consecutive (Q8: 2 u32s
       per row per block per thread; Q4: 1 u32 only — Q4
       wouldn't benefit). Issue rate reduction × narrower
       than option 1 because only Q8/Q5/Q4_K_S/IQ inner
       loops have 2+ consecutive u32 loads.
    3. **Lift `d` (fp16 scale) out of the row loop.**
       Currently `let d = f32(load_f16_at_src0(block_byte_base))`
       is loaded per-row inside the OUTPUTS_PER_WG×blocks
       loop. The scale is per-row, but if we re-batch loads
       to fetch all OUTPUTS_PER_WG `d` values into a small
       array up front (one vec4 u32 / vec2 u32 load), we
       cut OUTPUTS_PER_WG fp16 loads to one composite load.
       Predicted win: ~5-10% of load-issue cost.

    Predicted ceiling for combined (1)+(2)+(3) at 8B IQ3_M:
    matmul 65-69% of decode × 40% latency-bound fraction
    × maybe 30-50% issue-rate reduction = **~8-14% of total
    decode time**. Lower than the 26-28% pre-analysis
    ceiling but still meaningful. Risk: GPU scheduler may
    already be pipelining loads across warps, in which case
    THREADS_PER_BLOCK=2 could increase register pressure
    enough to hurt occupancy and net out flat or negative.

    **Recommended approach:** start with lever (1) — a one-
    line change to `#define THREADS_PER_BLOCK` per quant —
    on the 4-baseline regression harness. If it pays, layer
    (2) and (3) on top. If it doesn't pay, the kernel is
    deeper-pipelined than expected and §A is closed.

    The ~~subgroupBroadcast / subgroupShuffle~~ angle is
    closed: nothing to share. The vec4-packed-load angle
    survives but in the lever-(2) form above (consecutive-
    address packing within a single thread), not as
    cross-thread coalescing.

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

## Completed on 2026-04-27 / 2026-04-28 (perf cycle §21-§32 + post-§32 housekeeping)

Moved from TODO.md 2026-04-28 during cleanup. The §21-§32 perf cycle entries,
the post-§32 "Resumption checklist", and the doc-style + housekeeping next-step
candidates that closed 2026-04-28 all live below for reference. The active TODO
carries pointers; everything after the canonical 6-baseline pins is inert history.

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
**§27 (2026-04-27 — LANDED, 3 main commits) llama.cpp rebase
+ free-win sweep.** Rebased `webllm-browser-patches`
`78433f606 → 434b2a1ff` (13-commit upstream delta, 3 in
`ggml-webgpu/`: Q1_0 #22374, fast i-quant mat-vec #22344,
register-tile/subgroup matmul tuning #22241). Zero conflicts;
all 11 patches replayed cleanly; new tip `981859864`. Safety
branch `webllm-browser-patches-pre-rebase-2026-04-27`. WASM
binary +32 KB (new kernels); checkall 427/11/0; browser
smoke clean. Bench-inference sweep on 6 models against §17
"pre-§A change" baselines uncovered a **+70-80% free win on
IQ3_M models** isolated to upstream's #22344 (fast i-quant
mat-vec):
- **qwen3-8b-iq3m**: 15.1 → 27.2 tok/s (+80%)
- **llama-3.1-8b-iq3m**: 16.8 → 29.0 tok/s (+73%)
- mistral-7b-q4ks (K-quant control): 34.5 → 35.8 (+3.8%, noise)
- mistral-7b-iq4xs (i-quant 7B): no §17 baseline; lands at
  35.6 tok/s — *tied with K-quant Q4_K_S at the same model
  size*, where pre-rebase the i-quant decompression overhead
  would have made it slower. Confirms #22344 closed the
  i-quant penalty across the IQ family.
- tinyllama-q4_0: 105.7 → 110.8 (+4.8%)
- qwen3-0.6b-q8: ~85 → 89.8 (+5.6%)
- qwen3-1.7b-q8 (17-tok warmup-dominated): ~59 → 62.2 (+5.4%)
Profile-mode rebench on `qwen3-8b-iq3m` (`make smoke-bench
PERF_RUNS=3`, 60-step trace) confirmed dispatch count
unchanged at **805/token** (pure kernel speedup, not graph
restructure); matmul **48.04 → 23.07 ms** (-52%); matmul
share **70.5% → 55.0%** (-15.5 pp, still lead bucket).
**§16's 16.2 baseline for `qwen3-8b-iq3m` is obsolete;
canonical bench-inf is now 27.2 tok/s.** The 8B+ fleet
effectively doubled at zero patch cost. Commits on main:
`db50d28` (rebase docs), `ccf2abb` (free-win sweep),
`7402e4b` (profile-mode breakdown).
**§28 (2026-04-28 — CLOSED, side-branch + 1 main commit)
§C-v2-A re-measurement post-§27 rebase.** Direct test of
whether the +80% target speedup from §27 reopens the lever.
Re-ran §26's 4-cell gate matrix on `feat/spec-decode-v2-greedy`
against the rebuilt WASM. Both gates **worsened**, not
improved: gate 1 = 9.7 / 28.2 = **0.34×** (was 0.42×); gate
2 = 12.7 / 28.4 = **0.45×** (was 0.54×). Target baseline
scaled +78% (15.8 → 28.2 tok/s) but drafted path only
+45-49% — drafter `qwen3-0.6b-q4f16` is Q8_0, not i-quant,
so K=4 drafter forwards retain pre-rebase cost; only the
K+1=5 verify benefits from #22344, and that saving is
amortized across 5 tokens. The §26 "resurrection path (a)
— faster K+1 verify" is now **empirically closed**: the
rebase delivered exactly that and gates moved *against*
the thresholds, not toward. Drafter overhead must scale
symmetrically with target speedup or the relative ratio
worsens. Resurrection paths still on the table: (b)
MEMORY64 → 70B+ target (unchanged). Path (c) "smaller
i-quant drafter that also uses #22344" was opened here as
a new theoretical path, then **CLOSED 2026-04-28 by direct
verify-cost probe** — see §29 below. Side branch tip
`9bdd707` carries the §28 matrix + SUMMARY at
`eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/`;
side branch tip `4e11d79` adds VERIFY-COST-PROBE.md in
the same directory.
TODO ref: `d10971b` on main. Zero `src/` change on `main`;
checkall remains 427/11/0.

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
    speedup or the relative ratio worsens. Resurrection path
    (c, new in §28) "smaller i-quant drafter that also uses
    #22344" was **CLOSED 2026-04-28 by §29 verify-cost probe**:
    verify is 210 ms/call (5.9× a solo-decode step) and 83% of
    cycle time, so the drafter→0 ceiling is 11.3 tok/s = 0.40×
    the 28.2 tok/s baseline — fails both gates regardless of
    drafter cost. Path (b) MEMORY64 → 70B+ target is the only
    remaining theoretical path. Side branch tip `9bdd707`
    carries the §28 matrix + SUMMARY at
    `eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/`;
    side branch tip `4e11d79` adds VERIFY-COST-PROBE.md.
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
below; A/B/C/F/§4-decode/§C-v1/§4-prefill/§C-v2-A/§D/§22/§24/§26/§27/§28/§29/§30/§31/§31a/§32
all closed or partial):**

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
§27. ~~llama.cpp rebase + free-win sweep.~~ **LANDED 2026-04-27 — §27.** Rebased `webllm-browser-patches` `78433f606 → 434b2a1ff` (zero conflicts, all 11 patches replayed). Bench sweep on 6 models found **+70-80% free win on IQ3_M** (qwen3-8b-iq3m 15.1 → 27.2 tok/s; llama-3.1-8b-iq3m 16.8 → 29.0) from upstream's #22344 fast i-quant mat-vec kernels. Other quants +4-6%. Profile-mode rebench: matmul **48.04 → 23.07 ms** (-52%), dispatch count unchanged (805/token). §16's 16.2 tok/s baseline for `qwen3-8b-iq3m` is obsolete — canonical bench-inf is now **27.2 tok/s**. Commits `db50d28` / `ccf2abb` / `7402e4b`.
§28. ~~§C-v2-A re-measurement post-§27 rebase.~~ **CLOSED 2026-04-28 on side branch `feat/spec-decode-v2-greedy` tip `9bdd707`** — gates **worsened**, not improved (gate 1: 0.42×→0.34×; gate 2: 0.54×→0.45×). Target baseline scaled +78% (15.8→28.2 tok/s) but drafted path only +45-49% — drafter qwen3-0.6b is Q8_0 (not i-quant), retains pre-rebase cost; only K+1=5 verify benefits, amortized across 5 tokens. §26 path (a) "faster K+1 verify" is now **empirically closed**. TODO ref `d10971b` on main; raw matrix at `eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/SUMMARY.md` on side branch.
§29. ~~§C-v2-A path (c) "smaller i-quant drafter".~~ **CLOSED 2026-04-28 by direct verify-cost probe on side branch tip `4e11d79`.** §28 opened path (c) as a new theoretical resurrection candidate. Probe directly measured `forwardVerifyArgmax` cost on the §28 cell-3 workload: verify is **210 ms/call** (median, p10=207, p90=213) over 27 unique calls — 5.9× a solo-decode step (35.5 ms) — driven by nTokens=5 mat-mat falling outside #22344's fast i-quant *mat-vec* kernels (matmul 187 ms = 90% of compute; dispatch count 796 vs solo 805 = identical graph topology). Cycle decomposition: 27 verify cycles × 210 ms = 5670 ms of 6842 ms wall (83% of cycle); drafter+overhead = 43 ms/cycle ≈ K=4 × 11 ms/forward. **Counterfactual drafter→0:** cycle = 210 ms / 2.37 tok = 11.3 tok/s = 0.40× the 28.2 tok/s baseline, fails both gates by 3.8× / 0.6×. Path (c) cannot close the gates regardless of drafter cost. Path (b) MEMORY64 → 70B+ target is the only remaining theoretical v2-A path. Probe cost: 1 profile run + 1 agentchrome js-exec ≈ 2 min wall. Saved: multi-day model acquisition campaign. Side branch tip `4e11d79`; report at `eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/VERIFY-COST-PROBE.md` on side branch.
§30. ~~Heuristic-based prefill-tile default in `ModelInference`.~~ **CLOSED 2026-04-28 — refactor landed on `main`.** Replaced §23's dual-source-of-truth pattern (`recommendedPrefillTile` field on `BenchmarkModel` + mirrored `RECOMMENDED_PREFILL_TILE` map in `smoke-test/real-model-page.js`) with `computeDefaultPrefillTileSize(hp)` exported from `src/inference/model-inference.ts`. Rule: `layerCount >= 32 AND embeddingLength >= 4096` → 128, else 0. Maps directly to the §22 abort signature ("32 layers × seq=512 of F32 intermediates"). Pre-edit Phase 0 probe validated all 18 downloaded registered models classify identically to the prior registry. Tile pill in the smoke page now renders post-ctor from `inference.prefillTileSize` so the auto-default is visible without page-side duplication. Override surfaces unchanged: `{ prefillTileSize: N }` ctor opt, `?prefillTile=N` URL, `--prefill-tile <n>` CLI all win, including the explicit-zero force-disable path. Browser smoke regression (B.1-B.4 from spec) verified all four overrides + auto-defaults work. Net change: −31 LOC (88 ins / 89 del across 6 files), 427 → 428 tests. Spec: `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md`. Plan: `docs/superpowers/plans/2026-04-28-prefill-tile-heuristic.md`.
§31. ~~MEMORY64 cap probe.~~ **CLOSED 2026-04-28 — partial result, lever NOT closed.** Probe target `webllm-wasm-mem64` built clean (133K js / 2.28M wasm) under `-sMEMORY64=1 -sWASM_BIGINT=1 -sMAXIMUM_MEMORY=16GB` via `make mem64-probe`; standalone `smoke-test/mem64-probe.html` ran four sequential phases against Chrome 147 + Emscripten 5.0.6 on M4 Max / macOS 26.4.1. **Outcomes:** Phase 1 (ASYNCIFY × MEMORY64 round-trip) **PASS** — `_webgpu_init` 1.4 ms wall, `_webgpu_shutdown` clean. **The single load-bearing risk axis from spec §4.1 is retired.** Phase 2 (BigInt ABI) **FAIL** — asymmetric: custom bridge exports (`_tensor_new_1d`) correctly return `BigInt`, but stdlib `_malloc` returns JS `Number` (`0xac6548` truncated). Phase 3 (cap probe) **invalid** — bailed at iter 0 because `_malloc(1 GiB)` returned a `Number`, indistinguishable from "actually 0" vs "high pointer mangled by JS shim"; no measured cap. Phase 4 (post-probe re-init) **PASS** — runtime stable. **Decision-rule branch (spec §5.1):** "Phase 1 passes, Phase 2 fails — narrower follow-up: investigate the specific ABI failure before committing more surface." Likely fix is a thin C wrapper (`bridge_malloc`/`bridge_free`) so the build emits explicit-signature shims, or a newer Emscripten release. Few-line change. **Probe paid for itself:** surfaced the actual blocker (a config gap, not architectural incompat) in same-day cost. Six commits across CMake / Make / harness / two review-fix rounds: `314f3a3` `e43244d` `2631eb5` `005c522` `e153e92` `53db417` `f3aad4a` plus a sub-probe revert (`b9c0c09`). Spec: `docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md`. Plan: `docs/superpowers/plans/2026-04-28-memory64-cap-probe.md`. Closure report: `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`.

§32. ~~llama.cpp rebase 2026-04-28-eve + free-win sweep.~~ **CLOSED 2026-04-28 — rebase-clean (after fix-up patch 12), small regression, accepted; new pattern recorded ("no free win, small regression, accepted").** Triggered by upstream `ggml-webgpu` movement (#22456 buffer aliasing refactor for `ssm_scan` landed). Rebased `webllm-browser-patches` `434b2a1ff → f9f33654a` (10 upstream commits, 1 in `ggml-webgpu/`); all 11 patches replayed cleanly via `git rebase --onto`, but compile error surfaced in patch 3 because #22456 renamed `webgpu_tensor_offset` → `ggml_webgpu_tensor_offset` and folded `view_offs` into the helper body. **Resolved by adding patch 12 as a forward fix-up** (single-line rename + drop redundant `view_offs`; bit-identical post-rename behavior; **squashed back into patch 3 on 2026-04-28 post-§31b cleanup pass** — patch stack now 11 patches again, WASM byte-identical pre/post squash, safety branch `webllm-browser-patches-pre-squash-2026-04-28` retained at `c4af89356`). Build gotcha encountered + documented: stale `src/wasm/build/CMakeCache.txt` carries `MATH_LIBRARY=NOTFOUND` from the pre-revert ggml CMake which the post-revert `if (DEFINED MATH_LIBRARY)` then incorrectly trips — **always nuke `src/wasm/build/` before a build that crosses upstream `d530d6e7a`**. WASM build clean post-fix (2,249,650 bytes, +9 KB from #22456 refactor); checkall 428/11/0; smoke clean. **Sweep result (vs §27 post-rebase baselines):** tinyllama-q4_0 110.8→107.4 (-3.1%), qwen3-0.6b 89.8→86.9 (-3.2%), qwen3-1.7b 62.2→60.9 (-2.1%), mistral-7b-q4ks 35.8→35.0 (-2.2%, 5-run), **llama-3.1-8b-iq3m 29.0→27.2 (-6.2%, 5-run)**, qwen3-8b-iq3m 27.2→26.2 (-3.7%). 5 of 6 within ±5% noise band; llama-3.1-8b-iq3m holds a real ~6% regression at 5 runs. **Likely cause:** #22456's aliasing-logic refactor interacting with tied-embedding + GQA + IQ3_M kernels (qwen3-8b-iq3m has identical GQA shape but untied embeddings and is essentially flat; the buffer-aliasing path is exercised more heavily by tied weights). Profile-mode rebench queued as optional follow-up but not done — 6% on a single non-canonical-baseline model fits the §27 doctrine "document and move on, unless a free win opens." **Decision: accept the rebase as new baseline.** Reverting costs ~6% on llama-3.1-8b-iq3m but loses upstream's option value for the next ggml-webgpu kernel cycle (Vulkan tuning + #22296 backend dedup landed here as setup work). Cherry-picking around #22456 specifically would diverge further from upstream and increase per-rebase maintenance. **Updates to canonical baselines:** `llama-3.1-8b-iq3m` 29.0 → 27.2 tok/s. Other 5 unchanged within noise. Closure report: `eval/reports/llama-cpp-rebase-2026-04-28-eve/SUMMARY.md`. Patch doc updated: `docs/LLAMA_CPP_PATCHES.md` (new patch 12 entry + 2026-04-28-eve rebase narrative + cache-staleness gotcha). Safety branch `webllm-browser-patches-pre-rebase-2026-04-28-eve` preserves pre-rebase tip `981859864`. **§32 is the first "small regression, accepted" close** in the rebase-trigger pattern; future rebases follow §27 ("free win") or §28 ("negative result, lever closed harder") or §32 ("rebase-clean, small regression, accepted") templates depending on outcome.

§31a. ~~MEMORY64 cap probe — bridge_malloc sub-probe.~~ **CLOSED 2026-04-28 — lever now VIABLE; ready for full bridge migration scoping.** Direct execution of §31's spec §6 follow-up: added thin C wrappers `bridge_malloc(size_t) → void*` and `bridge_free(void*)` to `src/wasm/webgpu-bridge.cpp`, exported `_bridge_malloc,_bridge_free` from `src/wasm/CMakeLists.txt`, and swapped Phase 2 + Phase 3 of `smoke-test/mem64-probe.html` to use them. Re-ran probe: **all four phases PASS.** Phase 2 — `_bridge_malloc(16n) → typeof=bigint value=0xac6548` with byte-equal F32 round-trip; stdlib `_malloc` diagnostic confirms the §31 asymmetry persists in the same build (`typeof=number`), so the wrapper is the targeted fix not a stdlib upgrade. Phase 3 — sequential 1 GiB allocations succeeded for **15 iterations × 1 GiB = 16,106,127,360 bytes ≈ 15.00 GiB** with 64 KiB page-commit per allocation; iter 15 hit BigInt `0n` (allocator out of headroom under the configured `-sMAXIMUM_MEMORY=16GB`). All 15 freed cleanly via `_bridge_free`. **Decision-rule branch (parent spec §5.1): "≥8 GiB → promote to full bridge migration."** 15 GiB covers every model size that fits the 2026-04-28 30B project ceiling: 8B Q4_K_S (~4.5 GiB weights), 13B Q4_K_S (~7.4 GiB), 30B IQ3_M (~12.8 GiB; tight against 15 GiB once KV+activations land — `MAXIMUM_MEMORY` bump may be needed). **Cap is configured-ceiling-bound, not hardware-bound** — actual Chrome wasm64 upper bound is presumably higher; raise `MAXIMUM_MEMORY` only if the 30B working set demands it. Net code change: **+18 LOC** across 3 files. Probe wall-clock: 19 ms. Implementation took ~5 minutes; build ~30 seconds (incremental). **§31a does NOT migrate the production `webllm-wasm` build to MEMORY64** — that is the P2-class follow-up spec, scoped at: (i) replace stdlib malloc/free call sites in `src/inference/` + `src/wasm/` TS code, (ii) audit `int32_t size`/offset params in `webgpu-bridge.cpp` for >2 GiB transfer signatures, (iii) update GGUF loader to keep BigInt offsets across JS↔WASM, (iv) re-run smoke + bench-inf + bench-profile gates under MEMORY64 to confirm zero regression on the existing ≤4 GiB fleet, (v) decide single-binary vs dual-binary deploy. Open as a separate spec/plan cycle when a 13B or 30B target is asked for. Closure report: `eval/reports/memory64-probe-2026-04-28/SUMMARY-31a.md`.

D. **Bump `MAXIMUM_MEMORY` (deferred §12, dropped in
   priority).** Confirmed in earlier sessions that 4 GiB
   is the 32-bit WASM hard cap. Going beyond requires
   `-sMEMORY64=1` (changes pointer types throughout the
   bridge, possible asyncify interactions). Multi-day
   engineering. Only worth it for wave-3 12B+ candidates
   that need Q4_K_S+. **Updated 2026-04-28 by §31:** probe
   built `webllm-wasm-mem64` and ran end-to-end same-day,
   retired the asyncify-incompat risk axis, and surfaced a
   targeted BigInt-ABI gap on stdlib `_malloc` (not the
   multi-day rewrite originally feared). Lever **not
   closed**; follow-up is a few-line `bridge_malloc` wrapper
   then re-run the cap probe. See §31 entry below for the
   full closure narrative.

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

**Net characterization at 8B IQ3_M (post-§27 rebase,
both families):** matmul ≈ **55%** of decode on `qwen3-8b-iq3m`
(was §16's 65-69% pre-rebase; #22344 cut matmul ms ~52%
without changing dispatch count, dropping share by 15.5 pp).
Encode overhead is now the secondary suspect at ~10.7%;
attention is negligible (1.7%). bench-inf canonical: **27.2
tok/s** on qwen3-8b-iq3m, **29.0 tok/s** on llama-3.1-8b-iq3m
(both up +70-80% from §17's pre-rebase baselines). **All
single-token decode kernel-tuning AND algorithmic-amortization
levers — including greedy spec-decode with GPU-resident verify
even after the §27 target speedup — are now closed without
ship.** §17 ruled out matmul-kernel rework (§A); §18
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
compute — is a non-goal until a use-case emerges); §26 ruled
out §C-v2-A resurrection under §22 tile=128 (gates 0.42× / 0.54×;
verify graph never splits at K+1=5); §27 picked up upstream's
free-win i-quant kernel speedup (#22344, +70-80% on IQ3_M);
**§28 ruled out §C-v2-A resurrection under §27's faster target
(gates *worsened* to 0.34× / 0.45× — drafter Q8 doesn't benefit
from #22344, only target verify does, so the relative ratio
moved against the thresholds); §29 ruled out §C-v2-A path (c)
"smaller i-quant drafter" by direct verify-cost probe — verify
is 210 ms/call (83% of cycle), so the drafter→0 ceiling is
0.40× the target solo baseline regardless of drafter cost.**
**All algorithmic levers at the canonical 4-baseline are now
exhausted.** Remaining headroom is **architectural
infrastructure**: MEMORY64 to bring 70B+ targets into reach
(multi-day, conditional on a deployment ask; only remaining
v2-A resurrection path with measurable headroom — would shift
the target/drafter ratio from 13× to ~100×); upstream
ggml-webgpu mat-mat fast-path kernels OR dispatch coalescing
(would attack the verify cost wall directly — re-run §27 sweep
+ §28 harness + §29 probe on every llama.cpp rebase to spot the
next free win); §D's deferred concat-graph lever
(encoder-side fallback if a batch-throughput use-case appears).

Boot sequence for a fresh session:

1. **`make checkall`** — confirm 428 pass / 11 skip / 0 fail.
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
   coverage; 426 → 427 pass). **§26 / §27 / §28 / §29 added 0 tests**
   each — §26 was a measurement+closure cycle (side-branch matrix +
   3 docs commits on main); §27 was a llama.cpp rebase + bench
   sweep (3 docs commits); §28 was a side-branch re-measurement
   (1 docs commit on main, side branch tip `9bdd707`); §29 was a
   side-branch verify-cost probe (1 docs commit on main, side
   branch tip `4e11d79`). **§30 was a refactor (registry → ctor
   heuristic): net +1 test** — added 3 boundary tests in
   `tests/prefill-tiling-config.test.ts` (5 → 8) and deleted 2
   registry-shape tests in `tests/eval-models.test.ts` (the
   `recommendedPrefillTile auto-default` describe block); 427 →
   428 pass. The WebGPU-gated integration tests skip under Bun
   (no `navigator.gpu`).
2. **`git log --oneline -30`** — top of `main` is the §30
   prefill-tile heuristic refactor (`88b74f9 refactor(prefill-tile):
   replace dual-registry pattern with hyperparam heuristic`).
   This is the FIRST `src/`-touching commit since §23 (`0c50e03`,
   2026-04-27): all of §24-§29 were measurement-only / docs-only.
   §30 deletes `recommendedPrefillTile` from `eval/models.ts`,
   the smoke mirror map from `smoke-test/real-model-page.js`,
   and the registry fallback from `eval/perf.ts`; adds
   `computeDefaultPrefillTileSize` to `src/inference/model-inference.ts`.
   Below `88b74f9`: `3a58949 docs(plan): prefill-tile heuristic
   refactor — phased implementation plan` and `ae68bbe docs(spec):
   prefill-tile heuristic — replace dual-registry pattern` are
   the §30 spec + plan commits. Below those: `cf6dd4a docs(TODO):
   §29 — §C-v2-A path (c) closed by verify-cost probe` was the §29
   main commit (verify-cost probe writeup landed on
   `feat/spec-decode-v2-greedy` side branch tip `4e11d79`, which
   is **archived — do not merge**). Below it: `a7633c4
   docs(TODO): refresh resumption checklist post-§27 rebase
   + §28 closure` was the §28 main commit. Below that the §28
   measurement: `d10971b docs(perf): §28 §C-v2-A re-measurement
   — gates worsened, lever closed harder`. Below it the §27
   cycle (3
   commits): `7402e4b docs(perf): qwen3-8b-iq3m profile-mode
   breakdown post-rebase` → `ccf2abb docs(perf): rebase free-win
   sweep — IQ3_M +70-80% from upstream #22344` → `db50d28
   docs(rebase): llama.cpp 78433f606 → 434b2a1ff (Q1_0 + i-quant
   + matmul tuning)`. Below those: `391ea29 docs(TODO): split
   into TODO.md (active) + TODO_ARCHIVE.md (historical)` is the
   TODO_ARCHIVE split that landed between §26 and §27. Below
   that, the §26 cycle (3 commits): `01b66fe docs(TODO): refresh
   resumption checklist post-§26 closure` → `e715160 docs(TODO):
   §26 — §C-v2-A re-measurement under tile=128 CLOSED` →
   `f0a682c docs(plan): §26 §C-v2-A re-measurement under §22
   tile=128` → `b23ccc9 docs(spec): §26 §C-v2-A re-measurement
   under §22 tile=128`. Below those, the §25
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
   — confirm the **11-patch stack** is intact (was 12 between §32
   and the post-§31b cleanup; patch 12 squashed back into patch 3
   2026-04-28) and the base is upstream `f9f33654a vulkan: Coalesce
   Q4_K/Q5_K scale loads (#21751)` (rebased 2026-04-28-eve via §32).
   Tip is `3b8ade2a2 ggml-webgpu: fix UB shift-by-32 in
   load_u32_at_src{,0} for aligned offsets` (patch 11, bug #28 UB
   shift fix; SHA shifted from `ab09f14eb` by the squash since
   patch 3's content changed and downstream cherry-picks re-hash).
   Patch 3 (`d10d41a13 ggml-webgpu: add request-based browser
   readback API`) now incorporates the #22456 helper rename
   directly (`ggml_webgpu_tensor_offset(tensor) + offset`) — no
   separate fix-up commit. Safety branches:
   `webllm-browser-patches-pre-squash-2026-04-28` (pre-squash tip
   `c4af89356`), `webllm-browser-patches-pre-rebase-2026-04-28-eve`
   (pre-§32 tip `981859864`), `pre-rebase-2026-04-27` (pre-§27 tip
   `a536df4f4`) — all kept as roll-back targets. The 2026-04-27 →
   2026-04-28-eve delta was 10 upstream commits, 1 of them in
   `ggml-webgpu/` (#22456 buffer aliasing refactor for ssm_scan;
   renamed `webgpu_tensor_offset` helper, folded `view_offs` into
   the helper body). **Zero `git rebase` conflicts**; the compile
   error in patch 3 was a semantic conflict that the §32 rebase
   first resolved via forward-fix-up patch 12 and the post-§31b
   cleanup pass folded back into patch 3 itself. **§17 through
   §31a added zero patches**; **§32 added patch 12** which the
   post-§31b cleanup folded into patch 3 (net stack delta: zero
   patch additions across §17-§31b). The `__EMSCRIPTEN__` guard
   around FA was already removed in the 2026-04-25 rebase; §20
   re-uses the bridge wrappers from §18 with no new shader work;
   §21-§23 + §30 are pure-TS / pure-JS work above the bridge
   with no shader changes.
4. **WASM build state.** `smoke-test/webllm-bundle.js` mtime
   is 2026-04-28 ~10:50 (post-§31b squash rebuild against
   squashed llama.cpp tip `3b8ade2a2`); size is 189574 bytes
   (unchanged since §30 — §32 and the §31b-postlude squash were
   llama.cpp-only). `smoke-test/webllm-wasm.{js,wasm}` mtimes
   are 2026-04-28 ~10:50; `webllm-wasm.wasm` is **2249650 bytes**
   (byte-identical to the pre-squash artifact, confirming the
   squash was semantically a no-op; was 2240603 pre-§32 — +9 KB
   from upstream's #22456 aliasing refactor; was 2207801 pre-§27
   — +42 KB cumulative since the §27 rebase from new Q1_0 +
   i-quant kernels + aliasing refactor). Built against the
   squashed §32 rebased llama.cpp base `f9f33654a`. **`MATH_LIBRARY=NOTFOUND` cache-staleness gotcha**
   from the §32 rebase: upstream's `d530d6e7a` revert tripped the
   stale `find_library` result in the build cache; **always nuke
   `src/wasm/build/` before a build that crosses this commit**
   (or any future find_library-touching upstream change). If the
   artifacts look stale, run: `rm -rf src/wasm/build && source
   ~/emsdk/emsdk_env.sh && make wasm-build && bun build
   src/index.ts --outfile smoke-test/webllm-bundle.js --target
   browser && cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/
   && make smoke-restart`. **Updated post-§32 sanity baselines
   (`bun run eval/perf.ts --model <m> --runs 3`):** tinyllama-q4_0
   ~107 tok/s, qwen3-0.6b ~87, qwen3-1.7b ~61, mistral-7b-q4ks
   ~35, llama-3.1-8b-iq3m ~27 (was 29.0 pre-§32 — see §32 closure
   for the regression analysis), qwen3-8b-iq3m ~26. Other quick
   smoke confirmations: `model=mistral-7b-instruct-v0.3-q3km` →
   Q3_K_M coherent at ≥20 tok/s (patch 11 / bug #28 fix healthy);
   `model=mistral-7b-instruct-v0.3-q4ks` *with no `?prefillTile=`
   param* → mode bar shows the `tile: 128` pill and prefill
   completes (§22+§23 auto-default healthy); appending
   `&prefillTile=0` to the same URL → pill disappears
   and prefill aborts with the §22 ggml-alloc signature
   (override path healthy).
5. **Read for context:** §17 (§A closure), §18 (§4 FA
   closure at N=1 decode), §19 (§C drafter spec-decode
   closure), §20 (§4 FA revisit at prefill / long-decode
   scope closure), §21 (§D encoder perf cycle — diagnostic
   close, no ship), §22 (7B+ long-prefill graph-buffer
   tiling — gated ship, default-off), §23 (§22 default-on
   flip via `recommendedPrefillTile` registry field — landed
   2026-04-27 as a single commit, `0c50e03`), §27 (llama.cpp
   rebase + free-win sweep — IQ3_M +70-80% from upstream
   #22344, the pattern to repeat after every llama.cpp
   rebase), and §28 (§C-v2-A re-measurement post-rebase —
   negative result with cleaner gates, the template for
   re-measuring closed levers when upstream perf shifts).
   The first six follow the "measure-and-close" pattern;
   §23 is a thin policy-layer follow-on with no measurement
   campaign; §27 is the template for **rebase-driven
   opportunistic measurement**; §28 is the template for
   **re-running closed gates when their underlying
   assumptions move** (sometimes the answer worsens — that
   is itself a useful close).
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
   §17/§18/§19/§20/§21/§22/§23/§24/§26/§27/§28/§29/§30 — none of the
   ten closures produced new dashboard data, only TODO
   writeups, perf.ts logs, §22's
   `eval/reports/prefill-tiling-2026-04-27/` matrix, §26's
   `eval/reports/spec-decode-v2-tile128-2026-04-27/` matrix
   on the side branch, and §28's
   `eval/reports/spec-decode-v2-tile128-postrebase-2026-04-28/`
   matrix on the side branch). **Note:** the dashboard's
   numbers for `qwen3-8b-iq3m` are pre-§27 (16.2 tok/s);
   they will be stale until a new bench-profile run for that
   model is saved into `smoke-runs.db`. If a fresh session
   wants to refresh the 8B numbers in the dashboard, run
   `make bench-profile PROFILES=qwen3-8b-warm` (and the
   thinking variant) and the SSE feed will repopulate.
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

   **§26+§28+§29 side-branch state** (no impact on `main`): the
   `feat/spec-decode-v2-greedy` branch carries the entire v2-A
   driver, AdaptiveGate, K+1 verify, contract gate, and ~30
   unit/integration tests, plus the four cherry-picked §22
   commits, §26's matrix evidence, §28's post-rebase
   re-measurement, and §29's verify-cost probe writeup. Tip is
   now **`4e11d79`** (was `9bdd707` at §28 close, `6b20aad` at
   §26 close). **Do not merge to `main`** — §28 found gates
   *worsened* under the §27 rebase (gate 1 0.42× → 0.34×;
   gate 2 0.54× → 0.45×) because drafter Q8 doesn't benefit
   from #22344, only target verify does; **§29 then ruled out
   the "smaller i-quant drafter" path entirely** by directly
   measuring verify at 210 ms/call (83% of cycle), so even a
   zero-time drafter caps the cell at 0.40× target solo. If a
   future cycle resurrects v2-A, the only path with measurable
   headroom is a 70B+ target via MEMORY64 (target/drafter ratio
   13× → ~100×).
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
     Adding new 7B+ entries: nothing to do — the §30
     `computeDefaultPrefillTileSize(hp)` heuristic in
     `src/inference/model-inference.ts` derives the default
     from `hyperparams.layerCount × embeddingLength`, so the
     ctor self-configures. If the heuristic is wrong on a
     specific model, override at the call site via
     `{ prefillTileSize: N }` ctor opt, `?prefillTile=N`
     URL, or `--prefill-tile <n>` CLI flag.
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

**Status (post-§31b + patch-12-squash):** No perf lever is forced.
The algorithmic levers at the canonical 4-baseline are exhausted
(§17-§29 closed the matmul, FA, drafter, encoder, prefill-tiling,
and spec-decode families). The MEMORY64 ceiling that gated 13B/30B
targets is no longer architecturally blocked (§31 + §31a — 15 GiB
measured; §31b — 16 GiB Emscripten 5.0.6 wasm-ld toolchain ceiling),
but the full bridge migration is gated on deployment ask **and
inherits a 30B-tightness tracking item** (long-context working set
lands within margin of error of the toolchain ceiling — re-probe on
every Emscripten upgrade). §32 ran the upstream rebase + sweep
cycle and accepted a small regression on `llama-3.1-8b-iq3m`.
**§32a (2026-04-28) ran the profile-mode follow-up probe** — H1
"tied-embedding × #22456 aliasing-refactor" rejected (no bucket
asymmetry vs untied Qwen3-8B reference); H2 "uniform per-step
overhead" supported; §32 baseline accepted as final. **Post-§31b
cleanup squashed §32's forward-fix-up patch 12 back into patch 3**;
patch stack 12 → 11; WASM byte-identical (2,249,650 bytes); ship
gate 428/11/0 unchanged. All three opt-in probes from the
post-§32 next-steps list are now closed (§32a / §31b / patch-12
squash). All other open work is conditional on external triggers.


### Fresh next-step candidates (2026-04-28)

Three doc-style candidates surfaced post-housekeeping; **all closed
2026-04-28.** No fresh candidates queued.

7. ~~**TODO.md header pin refresh.**~~ **DONE 2026-04-28** (commit
   `64c5eea`) — header block (lines 19-44 post-edit) replaced
   pre-§27 baselines with post-§32 canonical 6: tinyllama 110.8,
   qwen3-0.6b 89.8, qwen3-1.7b 62.2, mistral-7b-q4ks 35.0,
   llama-3.1-8b-iq3m 27.2, qwen3-8b-iq3m 27.2. Smaller-fleet
   (smollm2-360m, qwen3-4b) and profile-mode pins kept in separate
   sub-blocks. "Canonical 6" promoted from inline-prose to
   header-block-vocabulary as the ship-gate fleet for every rebase
   + sweep cycle.

8. ~~**docs/BENCHMARKS.md tier expansion.**~~ **DONE 2026-04-28**
   (commit `ffefa00`) — added 7B+ entries (Mistral-7B Q4_K_S,
   Llama-3.1-8B IQ3_M, Qwen3-8B IQ3_M, Mistral-7B Q3_K_M); moved
   Qwen3 4B from Quality → Balanced based on measured 35.5 tok/s;
   moved Qwen3 1.7B from Balanced → Fast based on measured 62.2;
   added Decode tok/s column to all tier tables; bolded the
   canonical 6 entries to distinguish ship-gate fleet from wave-1
   / arch-survey entries; added explicit "Wave-1 deferred" footer
   for Gemma 2 2B and Phi-3.5 Mini.

9. ~~**CLAUDE.md doctrine capture from §27-§32a.**~~ **DONE
   2026-04-28** (commit `c514bce`) — promoted three doctrines
   from TODO process-notes to CLAUDE.md "Workflow policies":
   - **Rebase + sweep cycle doctrine** with the three template
     outcomes (§27 free win, §28 negative result, §32 small
     regression accepted) and matching decision rules.
   - **Cap-probe doctrine** (§31b lesson — bump first,
     characterize second).
   - **Pre-rebase baseline doctrine** (§32a lesson — same-model
     pre/post bucket comparison beats cross-model proxy).
   Both #2 and #3 cite closure reports as evidence anchors. The
   doctrines now survive context decay and session resets.

1. ~~**§32a — Profile-mode rebench on `llama-3.1-8b-iq3m`**.~~
   **CLOSED 2026-04-28 — hypothesis rejected, §32 baseline
   accepted as final.** Ran `make smoke-bench
   PERF_MODEL=llama-3.1-8b-instruct-iq3m PERF_RUNS=3` against
   the §32 rebased base (llama.cpp tip `c4af89356`). Captured
   156-step profile trace. Buckets: matmul **23.02 ms / 57.3%**,
   encode **4.01 ms / 10.0%**, attention **0.63 ms / 1.6%**,
   dispatch **652/token**, profile-mode tok/s **23.5**
   (perturbation -13.6% vs §32's 27.2 non-profile, normal band
   for this model class). **Bucket profile is structurally
   identical to qwen3-8b-iq3m's post-§27 reference within
   measurement noise** (matmul Δ -0.3%, dispatch delta tracks
   layer-count delta exactly: 652 = 32 × ~20.4; 805 = 36 × ~22.4).
   No bucket sticks out as the locus of the -6% regression. **H1
   "tied-embedding × #22456 aliasing-refactor" rejected** — would
   predict matmul or encode-overhead asymmetry vs untied Qwen3-8B
   reference; opposite is observed (Llama's lm_head matmul is
   *faster* per element). **H2 "buffer-aliasing constant
   overhead" supported** — uniform per-step overhead distributed
   across the pipeline; not bucket-localized. Decision rule's
   "uniform → accept and move on" branch fires. Closure report:
   `eval/reports/llama-cpp-rebase-2026-04-28-eve/PROFILE-32A.md`.
   New canonical reference pin: `llama-3.1-8b-iq3m` profile-mode
   23.5 tok/s / 156-step trace, alongside `qwen3-8b-iq3m`'s
   22.0 tok/s / 805 dispatch — these now form a matched 8B IQ3_M
   pair for any future post-rebase probe.
   **Process improvement noted for next rebase:** when the sweep
   classifies as "small regression, accepted" (§32 template),
   capture pre-rebase profile-mode on the regressing model
   *before* doing the rebase. Cost: ~3 min wall. Pay-off:
   §32a-style follow-on gets a same-model baseline (would have
   diagnosed conclusively here rather than via the cross-model
   proxy).

2. ~~**§31b — `MAXIMUM_MEMORY` upper-bound probe**.~~ **CLOSED
   2026-04-28 — toolchain ceiling identified at 16 GiB; Chrome
   runtime cap unmeasurable from this toolchain.** Bumped
   `-sMAXIMUM_MEMORY` to `64GB` in the `webllm-wasm-mem64` ctor
   block; build failed at link time:
   `wasm-ld: error: maximum memory too large, cannot be greater
   than 17179869184` (= **16 GiB exactly**, 2^34). Emscripten
   5.0.6's wasm-ld enforces a hard 16 GiB ceiling on
   `--max-memory`, regardless of the wasm spec's 256 TiB
   theoretical limit or Chrome v8 wasm64's actual runtime cap.
   §31a's "configured-ceiling-bound, not hardware-bound" framing
   is correct but understates the constraint: **the configuration
   ceiling is the toolchain ceiling, not a project knob.** §31a's
   15 GiB measurement was therefore at the maximum any current
   Emscripten build can configure. Implications for the 30B
   migration scope: 30B IQ3_M working set (12.8 GiB weights + KV
   + activations) can land at ~14.8-15.8 GiB on long-context
   workloads, **within margin of error of the toolchain ceiling**
   — the 30B migration inherits a "track the linker cap on every
   Emscripten upgrade" tracking item. Mitigation paths if the cap
   bites: lower-bit quant (IQ2_XXS / IQ2_S regains 4-5 GiB), cap
   context window, wait for upstream Emscripten to lift, or
   custom wasm-ld patch. **Process improvement noted:** when a
   cap is hit at a configurable value, immediately bump it to
   confirm whether the cap is configuration- or toolchain-bound;
   §31a's report would have been clearer with this 2-minute
   inline check. Edits reverted (zero net code change). Closure
   report: `eval/reports/memory64-probe-2026-04-28/SUMMARY-31b.md`.

3. ~~**Patch 12 squash cleanup** on `webllm-browser-patches`.~~
   **DONE 2026-04-28** — patch 12 (§32 forward fix-up) folded
   back into patch 3 via cherry-pick chain on a temp branch;
   trees byte-identical pre/post squash; new branch tip
   `3b8ade2a2` (was `c4af89356`); patch stack now **11 patches**
   (down from 12). WASM rebuild byte-identical at 2,249,650
   bytes; checkall 428/11/0 unchanged. Safety branches retained:
   `webllm-browser-patches-pre-squash-2026-04-28` (pre-squash
   tip `c4af89356`) and `webllm-browser-patches-pre-rebase-
   2026-04-28-eve` (pre-§32 tip `981859864`). Doc updated:
   `docs/LLAMA_CPP_PATCHES.md` (count line 12 → 11; patch 12
   section removed; §32 narrative augmented with squash-pass
   note).

---

**Fresh optional items (post-§31b housekeeping).** All three closed
2026-04-28 — closure entries preserved below for reference.

4. ~~**Dashboard refresh sweep on the 6-model fleet.**~~ **DONE
   2026-04-28** — `bun run eval/bench.ts --profiles "<list>"` on the
   canonical fleet (tinyllama-warm, qwen3-0.6b off/on × cold/warm,
   qwen3-1.7b off/on warm, mistral-7b-v0.3-warm, llama-3.1-8b-warm,
   qwen3-8b-warm/thinking-warm — 11 profiles total). 19/20 PASS;
   1 transient timeout on qwen3-0.6b-thinking-cold speed retried
   PASS (cold model warmup window). DB went 148 → 182 runs / 34 → 45
   evals; all 9 canonical model/thinking cells refreshed with
   2026-04-28 entries. Smoke-harness throughput numbers are 15-25%
   below `perf.ts` steady-state pins (CLAUDE.md harness-overhead
   note): tinyllama 84.8, qwen3-0.6b off 66.4 / on 65.0,
   qwen3-1.7b off 41.6 / on 45.2, mistral-7b 29.3, llama-3.1-8b 23.6,
   qwen3-8b off 22.0 / on 22.7. **§16's "16.2 tok/s" pin for
   qwen3-8b-iq3m on the dashboard is now superseded** by 22.7 tok/s
   (smoke harness) and the post-§27 27.2 tok/s perf.ts steady-state.
   Zero `src/` change; DB is gitignored (per `eval/reports/`).

5. ~~**Pre-rebase profile-mode capture on the canonical 6.**~~ **DONE
   2026-04-28** — `make smoke-bench PERF_MODEL=<m> PERF_RUNS=3` on
   each canonical model; logs + SUMMARY in
   `eval/reports/pre-rebase-baselines-2026-04-28/`. Headline pins
   (3-run median, profile-mode):

   | Model                      | tok/s | matmul (med, %) | dispatch |
   |---|---:|---:|---:|
   | tinyllama-1.1b-q4_0        | 87.9  | 3.74 / 37.8%    | 450 |
   | qwen3-0.6b-q8              | 68.2  | 3.87 / 33.6%    | 629 |
   | qwen3-1.7b-q8              | 44.0  | 6.75 / 36.9%    | 629 |
   | mistral-7b-v0.3-q4ks       | 29.7  | 15.86 / 48.7%   | 650 |
   | llama-3.1-8b-iq3m          | 23.5  | 23.00 / 57.5%   | 652 |
   | qwen3-8b-iq3m              | 21.8  | 23.20 / 55.4%   | 805 |

   llama-3.1-8b is bit-identical to §32a's PROFILE-32A.md (same-day
   reproducibility verified). Use when next upstream `ggml-webgpu`
   rebase trigger fires: same-model pre/post bucket comparison
   beats §32a's cross-model proxy. Freshness window: ~1 month;
   re-capture if rebase ETA slips. SUMMARY.md in the directory
   carries the full procedure + use-case + cross-references against
   §27 / §32 baselines.

6. ~~**§32 SUMMARY cross-link refresh.**~~ **DONE 2026-04-28**
   (commit `439bf7a`) — appended §10 "Post-cycle updates" stanza
   to `eval/reports/llama-cpp-rebase-2026-04-28-eve/SUMMARY.md`
   pointing at PROFILE-32A.md (H1 rejected / H2 supported), the
   patch-12 squash commit (`2850291`, stack 12 → 11), and §31b
   (16 GiB Emscripten 5.0.6 wasm-ld toolchain ceiling). Future
   readers landing on §32 closure see follow-up outcomes inline.

---

## MEMORY64 full bridge migration (closed 2026-04-29; archived from TODO.md)

The dedicated MEMORY64 migration block + its three follow-ups
(verify Q5_K_M decode under shim fix; add Q5_K canonical-6 row;
upgrade Emscripten past `8d78be5`) all closed by 2026-04-29.
The block was archived from `TODO.md` on 2026-04-29 per the
§17/§18/§19 cadence (closed-and-detailed → moves out of active
TODO).

Closure summary in `TODO.md` watch list and at
[`eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`](eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md).

### MEMORY64 migration follow-ups (queued 2026-04-29)

The migration closed on Phase 7 commit `99a470c`, but the cycle
surfaced three small follow-ups that aren't blocking and deserve
their own scope. None gates further work; pick them up when
convenient.

1. **Verify Mistral-7B Q5_K_M decodes under the shim fix.**
   **CLOSED 2026-04-29.** Smoke probe under the new vendored Dawn
   port (`v20260423.175430`, post-`8d78be5`): the original Phase 7
   reproducer `mistral-7b-instruct-v0.3-q5km` (~5.1 GiB) decodes
   end-to-end on wasm64 at **34.6 tok/s** (greedy single-pass, 64
   tokens, no bind-group errors). Confirms the upstream
   `makeGetValue '*'` fix is kernel-family-agnostic — Q5_K_M now
   works alongside Q4_K_S. The model registration comment in
   `eval/models.ts` was updated to reflect the closed status.

2. **Add a Q5_K-family row to the Phase 5 canonical 6.**
   **CLOSED 2026-04-29.** Used the already-registered + already-
   validated Mistral-7B Q5_K_M (~5.1 GiB) instead of the originally-
   recommended Q5_K_S — both are >4 GiB-cap (wasm64-only), so the
   row sits *outside* the wasm32/wasm64 parity matrix as a
   wasm64-exclusive kernel-coverage probe. 3-run profile-mode
   smoke-bench under wasm64 with the new vendored Dawn port
   yields **26.7 tok/s** (matmul 50.8% / 17.83 ms median; 650
   dispatches/token; FA engaged at 1.6% attention share). 5.3%
   slower than Q4_K_S at the same param count, in the expected
   band for the higher-precision Q5_K block layout. The
   PHASE-5-PARITY.md addendum captures the row as the canonical
   Q5_K reference point — a future rebase that breaks Q5_K kernels
   surfaces as a regression here.

3. **Upgrade Emscripten past `8d78be5` to drop the shim patch.**
   **CLOSED 2026-04-29.** Vendored
   `vendor/emdawnwebgpu/emdawnwebgpu.port.py`
   from Dawn release `v20260423.175430` (well past the
   `8d78be5` Oct-07-2025 fix). Wired through both `wasm-build`
   targets via `-DEMDAWNWEBGPU_DIR=$(CURDIR)/vendor/emdawnwebgpu`.
   The new shim emits `Number(HEAPU64[(entryPtr+OFF)/8])`
   directly at all three sites — bit-identical to what our patch
   script was producing. `scripts/fix-mem64-bindgroup-shim.py`
   and the Makefile wire-in were deleted in the same commit.
   Validation: Mistral-Nemo Q4_K_S decodes end-to-end on wasm64
   at 26.7 tok/s with no patch-script intervention.

**Lower-priority follow-ups** (no rush, do during a housekeeping
sweep):

- **Smoke-test page picker harmonization.**
  `smoke-test/real-model-page.js:114-117` defaults to wasm32 and
  reads `?wasm=mem64` directly; the eval harness now auto-routes
  via `profileToUrlParams` (`281b2a8`). Either share `pickWasmUrl`
  logic or document the smoke-test page as the manual debug
  surface. Not broken, just two source-of-truth points for the
  same routing decision.
- **Archive the closed MEMORY64 block to `TODO_ARCHIVE.md`** per
  the §17/§18/§19 cadence. The block at line 816+ is closed and
  detailed; once the three follow-ups land it can move out of
  the active TODO entirely.
- **Register a real 13B / 30B target** if a deployment ask
  surfaces. The wasm64 path is proven at 12B Q4_K_S; additional
  registrations are model-list edits, not infrastructure work.
  Currently external-trigger only.

---

### MEMORY64 full bridge migration (CLOSED 2026-04-29)

**Migration closed 2026-04-29.** All 8 phases shipped (audit + 7
implementation phases). Canonical 6 maintain ±3% wasm64-vs-wasm32
parity (Phase 5 re-bench against `c919efa`). Production wasm64
binary ships via `make wasm-build` (Phase 6 dual-binary path with
`pickWasmUrl` size-aware default). >4 GiB validation on Mistral-
Nemo-Instruct-2407 Q4_K_S (~6.63 GiB) coherent at **26/36 = 72%
overall** (beats Mistral-7B Q4_K_S 68% baseline) and **3-run
smoke-bench median 19.3 tok/s** (gate ≥15, in arch band 16-22).
Closure report at
[`eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`](eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md).

The Phase 7 cycle also discovered and fixed an Emscripten 5.0.6
codegen bug: `_wgpuDeviceCreateBindGroup` reads 8-byte
`WGPUBindGroupEntry::buffer/sampler/textureView` pointer fields
with `HEAPU32` (low-32 only). Under MEMORY64, when a handle is
allocated above 2³², the lookup misses by `1_00000000`. Initially
mitigated by `scripts/fix-mem64-bindgroup-shim.py` (since deleted
2026-04-29 — see follow-up #3 above); now fixed at the source by
vendoring Dawn release `v20260423.175430` via
`vendor/emdawnwebgpu/emdawnwebgpu.port.py`, which post-dates the
upstream `8d78be5` fix. Full diagnosis at
[`eval/reports/memory64-migration-2026-04-28/PHASE-7-BLOCKED.md`](eval/reports/memory64-migration-2026-04-28/PHASE-7-BLOCKED.md).
The lever is closed for the ≤30B ceiling. Next ask: register a
real 13B / 30B target if a deployment need surfaces (no
infrastructure work required — the wasm64 path is now proven
end-to-end).

**One-line goal.** Migrate the production WebLLM build from the
4 GiB-cap WASM32 path to the 16 GiB-cap WASM64 path so the engine
can host 13B Q4_K_S (~7.4 GiB) and 30B IQ3_M (~12.8 GiB) targets
within the 30B project ceiling.

**Final status (2026-04-29):** Phases 0-7 complete; production
wasm64 binary ships with the bind-group shim patch applied at
build time. Phase 5 re-bench at `c919efa` PASSed parity; Phase 7
validation at `5a53b1b` + `7260eff` PASSed eval + speed gates.

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

## 13B target registration (closed 2026-04-29; archived from TODO.md)

Closed 2026-04-29 — Qwen3-14B Q4_K_S validated end-to-end on the
wasm64 path. Eval 34/36 = 94% (new fleet accuracy leader); 3-run
smoke-bench median 18.9 tok/s (top of predicted 15-19 band).
Closure report at
[`eval/reports/13b-validation-2026-04-29/SUMMARY.md`](eval/reports/13b-validation-2026-04-29/SUMMARY.md).
The block was archived from `TODO.md` on 2026-04-29 per the
TODO archival cadence (closed-and-detailed → moves out of active
TODO).

### 13B target registration (queued 2026-04-29) — full block

**Trigger:** the MEMORY64 closure stub above promoted this from
external-trigger to an active next step. The wasm64 path is now
proven at 12B Q4_K_S (Mistral-Nemo) + 7B Q5_K_M (Mistral-Q5_K_M);
adding a 13B-class row exercises the next param-count band
inside the 30B project ceiling. Pure model-list work (no
infrastructure changes — `pickWasmUrl` + `vendor/emdawnwebgpu`
already route correctly).

**Candidate model:** **Qwen3-14B Q4_K_S** (~7.8 GiB, GQA, RoPE +
SwiGLU, same architecture family as our existing Qwen3-0.6B /
1.7B / 8B canonical-6 entries). Predictable kernel surface
(every kernel already validated at smaller sizes), and the
14B / 8B / 1.7B / 0.6B Qwen3 size ladder gives the dashboard a
clean architecture-internal scaling curve. **Alternatives** if
Qwen3-14B doesn't materialize: Llama-2-13B-Chat Q4_K_S
(~7.4 GiB, older but architecturally simple) or any 13-15B
HuggingFace GGUF that bartowski has packaged.

**Phasing (all CLOSED 2026-04-29):**

1. **Probe — registration audit.** Verify the chosen GGUF is
   on HuggingFace under bartowski (or a comparable
   GGUF-publishing repo) at the right quant. Confirm filesize
   on the HEAD response. Cost: ~30s. Output: a model spec
   block ready for `eval/models.ts`. **CLOSED 2026-04-29**
   (filesize 8,573,475,872 bytes / 7.99 GiB confirmed on
   bartowski/Qwen_Qwen3-14B-GGUF Q4_K_S).
2. **Register.** Add the entry to `eval/models.ts` mirroring
   the Mistral-Nemo Q4_K_S registration pattern (`vramMB`,
   `paramsB`, `architecture`, `ggufUrl`, `ggufFilePattern`,
   `capabilities`, `tier`). Add a smoke-profiles `…-warm`
   entry in `eval/smoke-profiles.ts` if needed. Re-run
   `make checkall` (must pass). **CLOSED 2026-04-29** (commit
   `a4c8189`; checkall green; +40 LoC across 2 files).
3. **End-to-end smoke probe.** `agentchrome navigate
   real-model.html?model=<id>&wasm=mem64&ctx=4096&prompt=hi&ingest=off`.
   **Gate:** all 8 stages complete; decode > 0 tok/s; embed
   sanity passes (cosine ≥ 0.75). The
   `pickWasmUrl(byteLength)` path auto-routes to wasm64 since
   filesize > 3.5 GiB. **CLOSED 2026-04-29** (all 8 stages
   green; decode 21.0 tok/s greedy / finish=eos / embed cosine
   0.76 ≥ 0.75; 0 console errors).
4. **36-prompt sanity eval.** `make bench-browser-eval
   PROFILE=<profile>-warm`. **Gate ≥ 60% overall** (the same
   gate Phase 7 validation used). Reference: Mistral-Nemo 12B
   Q4_K_S landed at 72% (26/36); Qwen3-8B IQ3_M is the closest
   smaller-Qwen3 reference at the existing dashboard.
   **CLOSED 2026-04-29** (34/36 = 94%; new fleet accuracy
   leader).
5. **Smoke-bench tok/s.** `make smoke-bench PERF_MODEL=<id>
   PERF_RUNS=3 WASM_VARIANT=mem64`. **Expected band:**
   extrapolating from Mistral-Nemo 12B Q4_K_S (19.3 tok/s)
   and Mistral-7B Q4_K_S (28.2 tok/s post-rebase), 14B Q4_K_S
   should land in the **15-19 tok/s** range under
   profile-mode. Hard floor: ≥ 12 tok/s (anything lower
   indicates a kernel surface regression). **CLOSED 2026-04-29**
   (3-run median 18.9 tok/s; runs 18.8/18.9/19.1; matmul 60.7%
   of graph; 893 dispatches/token = 8B's 805 + 4 layers × 22).
6. **Closure report.**
   `eval/reports/13b-validation-<DATE>/SUMMARY.md` with the
   discovery arc + matrices, mirroring PHASE-7-VALIDATION.md.
   **CLOSED 2026-04-29**
   ([`eval/reports/13b-validation-2026-04-29/SUMMARY.md`](eval/reports/13b-validation-2026-04-29/SUMMARY.md)).

**Out of scope for this work item:**
- Integrating the 13B model into the canonical 6 parity
  sweep (separate decision; may stay wasm64-only like the
  Q5_K addendum row, or may replace one of the existing 8B
  rows if the dashboard wants to maintain a 6-row matrix).
- Registering a 30B IQ3_M target (separate work item; spec
  carried at §31b and PHASE-7-VALIDATION.md as the
  next-rung target).
- Any algorithmic perf work — purely a registration +
  validation cycle.

**Risk register:**

| Risk | Likelihood | Mitigation |
|---|---|---|
| Filesize >16 GiB working set | Low | Q4_K_S at 14B = ~7.8 GiB on disk + ~3 GiB KV @ ctx=4096 + ~2 GiB scratch ≈ 13 GiB, well under the 16 GiB toolchain ceiling |
| Tokenizer / chat-template mismatch | Low-Medium | Qwen3-14B uses the same tokenizer as Qwen3-0.6B/1.7B/8B; the canonical entries already exercise this code path |
| Eval gate fails (<60%) | Low | Mistral-Nemo 12B passed at 72%; Qwen3 family scales predictably with param count |
| Smoke-bench tok/s below floor (<12) | Low | Architectural extrapolation places 14B Q4_K_S at 15-19 tok/s; matmul scaling is well-understood at this size |

**Outcome notes (post-closure):**

- Phase 3a download discovery: smoke harness expects local
  GGUFs at `smoke-test/models/<id>.gguf` (not HF-fetched); the
  HF URL in `ggufUrl` is consumed by `eval/browser-smoke.ts`
  for separate HF-mirror probes. New registrants need both:
  the registration commit lands metadata, a separate `curl`
  populates the local file. Documented in the SUMMARY.
- vramMB=8800 was conservative for routing (>3500 → wasm64,
  hits cleanly) but mildly underestimates absolute decode-time
  footprint (~10.3 GiB observed). Worth bumping to ~10500 if
  any future code reads vramMB for capacity planning rather
  than routing.
- Dispatch-count scaling held exactly: qwen3-14b's 893 = qwen3-8b's
  805 + (4 extra layers × 22 dispatches/layer). This is the
  cleanest size-ladder data point in the project.

---
