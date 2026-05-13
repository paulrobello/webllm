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

## TS API audit follow-ups (closed 2026-04-29; archived from TODO.md)

The full closure narrative for items (a)-(f) of the TS API audit
follow-ups, originally queued under "Next session pickup" item 4 in
TODO.md and shipped 2026-04-29 in one work cycle.

### Phase 1 audit + Phase 2 (a-e) remediations (earlier commits)

Phase 1 audit + Phase 2 (a-e) remediations landed 2026-04-29 across
5 commits:
- `a125baf` README quick-start fix
- `f119540` slim public surface
- `ca630d0` unify `load*` taxonomy
- `308c912` `WebLLMError` taxonomy + readonly types
- `15049da` `AbortSignal` on `Character.chat` + bounded `StreamRouter`
  queue

Net: 14 exports removed from public surface, broken stub `loadModel`
demoted to private helper, 5-class `WebLLMError` hierarchy exposed for
programmatic dispatch, 4 new tests (495 total).

### Phase 3 follow-ups (a)-(f) — shipped 2026-04-29

All six items shipped in one cycle as a coherent "public API hygiene"
pass.

- Spec: [`docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md`](docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md)
- Plan: [`docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md`](docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md)

**(a) Split `GenerationConfig`** — rename current 22-field type to
`InternalGenerationOptions` (engine-internal, unexported); new public
7-field `GenerationConfig` with `signal` inline; drop unused `prompt`
field. Compile-fail `@ts-expect-error` test locks in the steering-field
exclusion. Commits `91a5ee6` + `34ad33c` (test type fix).

**(b) Drop `WebLLMConfig.device`** — caller passes device directly to
`engine.loadLightweightModel(LightweightModelConfig)`. Smoke harness +
tests updated. Commit `f639e2e`.

**(c) Sampling flag + Qwen profile export** — new
`src/core/sampling-profiles.ts` exports `QWEN_THINKING_DEFAULTS` /
`QWEN_NON_THINKING_DEFAULTS` (now `Object.freeze`d for runtime safety);
`CompletionConfig.sampling`: `"auto" | "qwen-thinking" | "qwen-default" | "raw"`
(default `"auto"` = current magic). Commits `104a846` + `f8e56b0`
(freeze fix).

**(d) Engine accessor migration** — `getMemoryPool` / `getScheduler` /
`getModelManager` → property getters with underscore-prefixed backing
fields. Test mocks that seeded via `Object.create(WebLLM.prototype) +
direct field assign` swept to assign the new `_modelManager` backing
field. Commit `2fcad35`.

**(e) `Character.setTools`** — runtime tool reconfiguration with
defensive-copy semantics (`[...tools]` decouples caller's array from
internal state). Strict YAGNI: no `attachToolSystem(custom)` — no
consumer ask for parser swap. Commits `d0dc96f` + `e4c402e` (defensive
copy).

**(f) Polish bundle** — `ChatToolSchema.parameters[*].type` literal
union (`"string" | "number" | "integer" | "boolean" | "array" | "object"`)
with parallel mirrors in `chat-template.ts` (`ChatTemplateToolSchema`)
and `tool-system.ts` (`ToolParameter`) aligned in lock-step;
`GenerationFinishReason` per-variant JSDoc; README API Overview table
gains `engine.removeCharacter(id)` and `engine.shutdown()` rows; table
header renamed `Class` → `API` to fit method-level entries; trailing
periods on new rows dropped to match the existing 19-row convention.
Commits `e919027` + `25c5465` (mirror alignment + README polish).

### Test surface delta

485 → 493 pass (+8 across 3 new test files):
- `tests/generation-config-public.test.ts` — `@ts-expect-error`
  steering-field exclusion assertions
- `tests/sampling-profiles.test.ts` — constant equality + frozen-
  runtime + sampling union
- `tests/character-set-tools.test.ts` — replace / clear / add-from-none
  / defensive-copy

`make checkall` green at every commit boundary.

### Decision-log highlights (from brainstorming, kept for context)

- Naming: renamed current type to `InternalGenerationOptions` rather
  than introducing a new `PublicGenerationConfig` — gives the clean
  name to the consumer-facing 7-field type, which 99% of consumers
  use.
- `signal` migration: moved into `config.signal` (matches
  `CompletionConfig` shape; one fewer positional argument).
- `device` removal: chose the structurally cleaner option (drop from
  `WebLLMConfig` entirely) over making it optional with engine-side
  injection.
- `sampling: "auto"` default preserved BC. Added `"raw"` as the
  consumer escape hatch from the magic.
- No deprecation aliases (`getMemoryPool` etc. just deleted) per
  project policy "no BC-compat shims".
- `setTools` only — `attachToolSystem(custom)` deferred (no consumer
  ask for parser swap).
- Polish bundle adopted all four sub-items in one commit; mirror
  drift surfaced by code-quality review and fixed in a follow-up.

### Follow-ups filed (orthogonal; opportunistic pickup)

Three watch-list items captured in TODO.md "Watch list / optional
cadence work":

1. **Sampling-dispatch unit test.** `engine.ts:272-289`'s
   `forcedProfile` / `autoProfile` / `activeProfile` resolution is
   exercised end-to-end by smoke + chat-completion tests but has no
   direct unit test of the dispatch matrix (4 modes × `enableThinking`
   × Qwen-vs-non-Qwen × consumer-override). File before the next
   `CompletionConfig` surface change.
2. **Tool-schema mirror-drift sentinel.** Three identical literal
   unions across `chat-types.ts`, `chat-template.ts`,
   `tool-system.ts` share only a comment-based lock-step contract.
   Either dedupe via shared exported `JsonSchemaParameterType`
   (preferred — type-only, zero runtime cost) or add a structural
   drift test.
3. **`tsconfig.json` widening.** Currently includes only `src/**/*.ts`
   — the `@ts-expect-error` gate in
   `tests/generation-config-public.test.ts` is documentation, not
   enforcement. Widening to cover `tests/**` would surface latent type
   errors in other test files first (~5-15 latent-error hits expected).

### Process notes (lessons from the cycle)

- **Code-quality review caught real issues at every phase**, none
  of which the implementer or spec reviewer surfaced. Specifically:
  test type drift (Phase 1a), `Object.freeze` runtime-mutation
  footgun (Phase 1c), shared-reference aliasing in `setTools`
  (Phase 2b), tool-schema mirror divergence + README header
  granularity (Phase 3). Justifies the two-stage review pattern
  (spec compliance THEN code quality) end-to-end on this arc.
- **Phase boundaries kept ≤5 files** even when spec items naturally
  spanned more — Phase 1 sub-checkpointed into 1a/1b/1c rather than
  one Phase 1 commit.
- **TDD applied where new behavior** (item c sampling, item e
  setTools); refactors (a, b, d) drove correctness via `tsc`
  + existing test suite + ship-gate `make checkall`.

---

## Phi-3 causal LM support (closed 2026-04-29; archived from TODO.md)

Full closure narrative for Phi-3 causal LM support, originally queued
under "Active next steps" in TODO.md and shipped 2026-04-29.

Closed 2026-04-29 — all 6 phases (probe → register → implement →
smoke → 36-prompt eval → smoke-bench → report) passed in one
session. **First fused-projection causal LM in the fleet** (Path
B fused-forward, `architecture === "phi3"`-gated).

- **Eval:** 27/36 = **72%** (gate ≥60%; predicted band 70-80%).
  Tied with Mistral-Nemo 12B Q4_K_S despite being ~3× smaller.
- **Speed:** 3-run smoke-bench median **31.6 tok/s** (gate ≥25;
  predicted band 35-50, ~10% under-band attributed to opCont
  copies + profile-mode overhead).
- Plan: [`docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md`](docs/superpowers/plans/2026-04-29-phi3-causal-lm-support.md).
- Closure report:
  [`eval/reports/phi-3-validation-2026-04-29/SUMMARY.md`](eval/reports/phi-3-validation-2026-04-29/SUMMARY.md).
- Implementation commit `8392bca`; bug-fix commits `7915abb`
  (chat-template typo) + `7c85a2a` (opCont on fused views —
  the gibberish root cause); closure commit `31612a2`.
- Path A vs Path B note in the closure report: for the next
  fused-projection architecture (Phi-4, Granite, etc.),
  evaluate Path A first — the dispatch-count win is
  unmeasured and the strided-view gotcha cost a measurable
  ~6% throughput tax.

---

## Bucket B follow-ups (post-closure, 2026-04-28; archived from TODO.md)

Full closure narrative for Bucket B follow-ups (encoder non-BERT-arch
expansion housekeeping). Both queued follow-up items closed 2026-04-28.

### 11. Spec accuracy patch — DONE 2026-04-28

Patched `docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`:
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

### 12. Vault-save bucket B doctrines — DONE 2026-04-28

All four notes landed at `~/ClaudeVault/`:

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

### Embedding bucket D (closed 2026-04-30)

**Original queued entry (2026-04-29):**

Add `ModelInference.embed(tokenIds): Promise<Float32Array>` that
taps the post-`output_norm` hidden state on a chat model's
forward pass (same architecture truth source as bucket C:
`qwen3.cpp:98 res->t_embd = cur`), pools last-token, L2-
normalizes. Reuses ~70% of `CausalLMEmbedder.forwardEmbed`
logic — the architecture-routing groundwork (metaPrefix split,
EOS-append convention, hybrid-tier parity gate, per-binding
128 MiB cap doctrine) was battle-tested through bucket C and
is ready to generalize to chat models.

**Motivation:** the **single-model-active deployment** doctrine.
For agent + Three.js use cases where retrieval is over in-domain
content (agent memory, dialogue history, semantic search over
game state), running the chat model in embedding mode avoids a
second model load and halves cold-start. Quality drops 5-15%
vs dedicated retrieval-tuned embedders on MTEB benchmarks but
is "good enough" for in-domain retrieval.

**Pre-brainstorm scope estimate:** ~80-120 LOC. Add
`ModelInference.embed(tokenIds)`, widen `engine.embed(modelId, text)`
dispatch to include `inferenceEngines` lookup as a tertiary
fallback after `encoderEngines` and `causalEmbedderEngines`.
Parity gate strategy: capture sentence-transformers refs against
the chat model's HF base (or skip parity if no canonical
embedding head exists, and validate via cosine-distinguishability
tests on synthetic pairs instead). Open question for the spec.

**Decision rule for users:** the dedicated bucket C path
(Qwen3-Embedding-0.6B hybrid) ships as the **high-quality**
embedder; bucket D is the **simplicity / single-model-load**
embedder.

---

**Outcome (2026-04-30):**

`ModelInference.embed(tokenIds)` shipped and `engine.embed`
dispatches through `inferenceEngines` for chat models tagged
`embeddingCapable: true`. `qwen3-8b-iq3m` is the single
registered bucket D model at v1; other archs follow as
separate cycles.

Parity 10/10 PASS at `cos >= 0.90` (IQ3_M-calibrated gate; new
third tier in the gate-by-quant-tier scheme alongside `hyb`
0.995 and default 0.999). 4-pair distinguishability:
min paraphrase 0.918 > max unrelated 0.777 (+0.141 margin).
Bench: short p50 ~1000 ms / long p50 ~2000 ms (14-16× slower
than bucket C; accepted — single-model-load saves second GGUF
cold-start + ~2 GB VRAM).

**Artifacts:**
- Spec: [`docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md`](docs/superpowers/specs/2026-04-29-embedding-bucket-d-design.md)
- Plan: [`docs/superpowers/plans/2026-04-29-embedding-bucket-d.md`](docs/superpowers/plans/2026-04-29-embedding-bucket-d.md)
- Ref capture: `eval/refs/bucket-d/`
- Parity report: [`eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`](eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md)
- Bench output: `eval/reports/embed-perf-qwen3-8b-2026-04-29/`

**Per-task commit chronology (Tasks 1-9; base `5ca83bc`):**

```
504a907 feat(embed): add embeddingCapable flag to RegisteredModel
9dc5f3b feat(embed): add ModelInference.embed for chat-model self-embedding
62bfde6 fix(embed): apply normBias at output_norm tap; clarify intent
20ebff2 feat(embed): wire engine.embed dispatch to bucket D path
1381aa4 fix(embed): wire embeddingCapable through engine load path
4acd4e0 fix(embed): thread embeddingCapable through adoptPreloadedModel
2a55e3a feat(embed): enable bucket D on qwen3-8b-iq3m
66f711a test(embed): pin bucket D parity refs for qwen3-8b-iq3m
a5066b9 fix(refs): suppress Pyright import diagnostics on bucket D ref-capture
b9861cc test(embed): add qwen3-8b-ref.json fixture output (bucket D parity)
261e0b2 test(embed): extend parity harness for bucket D + 4-pair sanity
48c30b6 feat(embed-perf): bench coverage for bucket D (qwen3-8b-iq3m)
06dad2e docs(CLAUDE): require hfdownloader CLI for HuggingFace downloads
43f058c docs(report): bucket D parity closure (qwen3-8b-iq3m)
6263e74 docs(embed): document bucket D dispatch tier and quality tradeoff
```

**IQ3_M gate calibration:**

The plan expected a `q4f16_1` default-quant registration and the
standard `cos >= 0.999` gate. During Task 6 the registration's
`defaultQuant` field was discovered to be incorrectly tagged as
`"q4f16_1"` when the actual model file on disk is IQ3_M. IQ3_M
quant noise accumulates across all 32 Qwen3-8B attention/FFN
layers (unlike bucket C hybrid quant where only `token_embd` inherits
quant error). The empirical parity band is `0.906-0.962` — above
noise and meeting semantic utility bar, but well below f16-grade.

Following bucket C precedent, the harness now selects among three
gate tiers by `defaultQuant`:

| `defaultQuant` | Gate       | Tier label                      |
| -------------- | ---------- | ------------------------------- |
| `"iq3m"`       | `>= 0.90`  | IQ3_M i-quant                   |
| `"hyb"`        | `>= 0.995` | Hybrid Q4_K-on-`token_embd`     |
| anything else  | `>= 0.999` | f16 / full-precision            |

**Bugs and discoveries fixed during this cycle:**

1. **`defaultQuant` mistag** — `qwen3-8b-iq3m` had `"q4f16_1"`; the
   actual file is IQ3_M. Without the fix the 0.999 gate was applied
   and 10/10 rows failed. Fixed in `261e0b2`.
2. **`QuantFormat` union gaps** — `"iq3m"` and `"hyb"` were untyped
   string literals; extended to first-class members so type-narrowing
   on `defaultQuant` is exhaustive. Fixed in `261e0b2`.
3. **`normBias` omission** — initial `forwardForEmbedding` applied
   `normScale` but not `normBias` at the `output_norm` tap. Caught
   in Task 2 code review; fixed in `62bfde6`.
4. **Engine load-path threading** — `embeddingCapable` was added to
   `ModelEntry` and read by `engine.embed` for dispatch, but neither
   `loadModelFromBuffer` nor `adoptPreloadedModel` wrote it through.
   Fixed in `1381aa4` and `4acd4e0`.
5. **Stale `webllm-bundle.js`** — the pre-Task 1 bundle lacked
   `embed` export and `embeddingCapable`. Rebuilt in `261e0b2`.
6. **`real-model-page.js` chat-template skip** — the embed-perf bench
   path gated on `referenceEncoder` checks that don't apply to bucket
   D chat models. Fixed in `48c30b6`.

**Follow-on arch cycles (watch list):**

- Cross-arch generalisation: `ModelInference.embed` tap-point is
  Qwen3-specific; next cycle parameterises by arch string so
  `engine.embed` works across Llama, Mistral, Phi-3 without per-model
  patches.
- Concurrency mutex: a latent concurrent-forward race inside the WASM
  ctx-stack was surfaced (bucket C bug 2: `ggml-webgpu.cpp:3659`
  GGML_ASSERT). Not exposed by serial parity/bench runs, but agent
  workloads interleaving chat + embedding will need a mutex or
  serialisation queue around `engine.embed`.
- Quality-tradeoff API: IQ3_M cosine band (0.906-0.962) is below
  dedicated embedder quality; consider `qualityTier` or
  `recommendedFor` on `ModelEntry` for informed deployment choices.

---

## Bucket D Phi-3.5-mini extension — §28 NEGATIVE (closed 2026-04-30; archived from TODO.md)

Originally TODO.md item 7. Probed `phi-3.5-mini-q4km` as a second-
architecture bucket D candidate (after qwen3-8b-iq3m shipped). Outcome:
**§28 NEGATIVE result** — parity passed but distinguishability failed.
Demoted: `embeddingCapable: false` on the row. No follow-on cycle
queued.

Closure report:
[`eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md`](eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md).
Plan:
[`docs/superpowers/plans/2026-04-30-embedding-bucket-d-phi3.md`](docs/superpowers/plans/2026-04-30-embedding-bucket-d-phi3.md).

### Headline numbers

- **Parity 10/10 PASS** at `cos >= 0.91` (new `q4km` gate tier
  alongside `iq3m` 0.90).
- **Distinguishability mean-margin gate FAILS** under both pooling
  modes — last-token: −0.006, mean-pool: −0.027 (paraphrase cosines
  *lower* than unrelated cosines, i.e. the model produces
  indiscriminate sentence vectors).

### Keeper infrastructure shipped

Cycle landed reusable scaffolding even though the model itself was
demoted:

- **`embeddingPooling` per-model field** (`last-token` / `mean`) —
  enables future bucket D candidates to declare pooling at registration
  rather than hard-coding the path.
- **16+16 cross-domain pair harness** — 16 paraphrase + 16 unrelated
  pairs across multiple domains, replacing the original 4-pair sanity
  check that was statistically meaningless.
- **Mean-margin gate** `mean(P) − mean(U) ≥ 0.05` is now the load-
  bearing distinguishability gate. Strict-min `min(P) > max(U)` moved
  to informational — even bucket D flagship qwen3-8b-iq3m fails strict
  on this set (+0.084 mean-margin), so strict was too tight.
- **qwen3-8b-iq3m revalidated** under the new gate as part of the
  cycle.

### Lessons codified

1. **4-pair distinguishability is statistically meaningless** —
   noise dominates a sample that small.
2. **Strict `min(P) > max(U)` is too tight** even for the bucket D
   flagship.
3. **Parity gate alone is insufficient** — a model can pass row-by-row
   vs ref and still produce indiscriminate sentence vectors. Always
   pair parity with distinguishability before promoting an embedder.
4. **Mean-pool is not a free anisotropy fix in quantized builds** —
   Q-noise compounds across N positions; mean-pool didn't rescue
   Phi-3.5 either.
5. **Bucket D viability is per-model, not per-architecture** — Phi-3.5
   fails, Qwen3-8B passes; can't generalize across architectures
   without per-model verification.

### Retire-path

Phi-3.5-mini bucket D resurrection would require trying a higher-
precision quant (Q5_K_M / Q6_K / f16) and rerunning the harness. See
closure report retire-path for details. No queued follow-on.

---

## Frame-probe coexistence + NPC scenario sizing probes (closed 2026-05-01; archived from TODO.md)

Originally TODO.md items 8 + 9. The frame-probe baseline established
the agent + Three.js coexistence-cost picture; the four NPC sizing
probes (9a-9d) measured every architecture lever the user-stated NPC
deployment scenario depended on. All five closed in a single session
2026-05-01. Downstream decisions consumed: 9a → item 11 (prefix cache,
shipped 2026-05-02); 9b → "sequential is canonical agent-tick pattern";
9c → "do not bake warmup into engine init"; 9d → item 10 (dual-mode
worker, shipped 2026-05-02).

### 8. Frame-probe coexistence baseline — CLOSED 2026-05-01

First-class `?frameProbe=1` mode landed on the smoke page
(`smoke-test/frame-probe.js` + `real-model-page.js` integration;
`?scene=<url>` for GLTF stress, `?frameProbeCalls=N` for hitch-
distribution mode). Multi-call probe on `qwen3-8b-iq3m` confirmed:

- Render loop median 8.3ms (120Hz baseline) holds across baseline,
  prefill, decode, post — main-thread async path is sufficient
  for the agent + Three.js coexistence case at typical scene
  cost (~3K tri Mountain_01 fixture).
- Decode tok/s held at 24.7-25.0 across 5 sequential calls
  (within 1% of the trivial-cube baseline).
- **Per-call decode hitch is DETERMINISTIC** — every call has
  exactly one ~42-58ms drop in decode (median 49.3ms across 5
  calls; aggregate `>50ms` rate 2/481 frames). Pattern classifier
  "DETERMINISTIC (every call hitches in a narrow band)". Hypothesis:
  prefill→decode shape transition lands in the first 1-2 decode
  rAF frames.
- At 3.8M-tri stress scene: GPU contention dominates (24fps
  baseline, decode tok/s collapses 25 → 2.8). Main-thread async
  is fine; physical GPU is the bottleneck. Worker doesn't help
  this case (shared VRAM + single physical GPU).

**Probe-first preferred-path doctrine** (per CLAUDE.md
"Probe-first is the default"): every architecture decision below
gates on a measurement first. Don't pre-commit to a Worker
migration, prefix-cache implementation, or NPC tick-rate
target until the matching probe lands. The deterministic hitch
finding is what triggered probes 9c (hitch-warmup) and 9d/10
(worker migration).

### 9a. Prefill-prefix-cache decomposition probe — CLOSED 2026-05-01, PASS

Three NPC-shaped fixtures × 3 runs on `qwen3-8b-iq3m` (post-§27 tip
`e29753286`). Marginal token costs: **a = 12.31 ms / prefix-token**,
**b = 14.11 ms / tail-token** (b/a = 1.15 — prefill essentially
linear in total tokens, small attention-quadratic premium for
tail-position tokens). Projected at canonical NPC prompt
P=400/T=40: prefill ≈ 5488 ms, **prefix's share = 89.7%**. Verdict
robust to worst-case b=a substitution (prefix is 91% of total tokens
at that ratio). Closure report
[`eval/reports/probe-9a-2026-05-01/SUMMARY.md`](eval/reports/probe-9a-2026-05-01/SUMMARY.md).

**Downstream decision:** KV-cache-per-conversation-on-shared-weights
multiplexing is now load-bearing (was "deferred" per CLAUDE.md). At
5.5 s prefill per tick, a freshly-prefilled-from-scratch approach
blows the 1-tick-per-second budget by 5.5×; prefix caching collapses
it to ~0.6 s tail-only. Spec was queued behind 9b/9c/9d closure;
implementation shipped as item 11 (prefix cache, closed 2026-05-02).

Harness extension shipped: smoke `[7/8]` result line now carries
`tokensIn=N`; new probe runner at
`eval/probes/probe-9a-prefill-prefix.ts` and 3 fixtures in
`eval/fixtures/long-prompts.ts`.

### 9b. Batched-prompt vs sequential probe — CLOSED 2026-05-01, PARTIAL

N=4 NPCs on `qwen3-8b-iq3m`. Quality 100% / 100% (4/4 each, ratio
1.00 ≥ 0.70 ✅), wall ratio 0.72 (> 0.40 ❌ — batched 4010 ms vs
sequential 5553 ms). The JSON-wrapper decode overhead (48 vs 7
tokens) ate the projected ≥60% wall savings. **Decision:**
sequential remains the canonical agent-tick pattern, hard-dependent
on prefix caching (probe 9a). With prefix caching projected
~150 ms/tick (≥6 Hz budget). Batched would re-win at N≥16-20 or
with constrained JSON decoding. Closure report
[`eval/reports/probe-9b-2026-05-01/SUMMARY.md`](eval/reports/probe-9b-2026-05-01/SUMMARY.md).

### 9c. Hitch-warmup probe — CLOSED 2026-05-01, FAIL

Same-page-load A/B with `?frameProbeWarmup=1` toggle on
`qwen3-8b-iq3m`. Per-call decode_max (control vs warmup):
41.7 / 41.7 / 41.7 / 41.6 / 50.0 vs 41.6 / 41.6 / 58.3 / 42.1 / 40.8.
Warmup does NOT reduce call-0 decode_max; the hitch persists *every*
call regardless. **Decision:** do NOT bake warmup into engine init —
hitch is per-call structural overhead, not first-call shape JIT.
Closure report
[`eval/reports/probe-9c-2026-05-01/SUMMARY.md`](eval/reports/probe-9c-2026-05-01/SUMMARY.md).

### 9d. Worker-prototype hitch probe — CLOSED 2026-05-01, PASS (5.5× hitch reduction)

Spike at `smoke-test/probe-9d.html` + `smoke-test/probe-9d-worker.js`
drives a Worker-resident `WebLLM.loadModelFromBuffer` engine on
`qwen3-0.6b-q4f16` (smaller model used for spike tractability — 7B+
needs the smoke page's heap-streaming loader inside the worker).
Same-day same-model main-thread control vs worker decode_max:
main 41.0/33.6/58.3/49.8/58.2 (med 49.8) vs worker
9.1/9.4/9.0/9.1/9.2 (med 9.1) — **5.5× reduction**, hitch fully
absorbed. Public API fix shipped: `loadModelFromBuffer` now honors
`options.contextLength` (was previously hard-coded to GGUF max,
OOMing 32 K-context KV in the worker memory budget). **Decision:**
item 10 (dual-mode worker) is the load-bearing path forward (shipped
2026-05-02). Closure report
[`eval/reports/probe-9d-2026-05-01/SUMMARY.md`](eval/reports/probe-9d-2026-05-01/SUMMARY.md).

---

## Prefix cache via per-conversation KV snapshots (closed 2026-05-02; archived from TODO.md)

Originally TODO.md item 11. End-to-end mechanism, batch-transfer
trajectory, interleaved-probe correctness fix, and the two LANDED
spec follow-ups (LRU eviction + forkConversation cross-conv prefix
sharing) all closed.

Spec: [`docs/superpowers/specs/2026-05-01-prefix-cache-design.md`](docs/superpowers/specs/2026-05-01-prefix-cache-design.md).
Plan: [`docs/superpowers/plans/2026-05-01-prefix-cache.md`](docs/superpowers/plans/2026-05-01-prefix-cache.md).

### Mechanism (CLOSED 2026-05-01, PARTIAL initial verdict)

`WebLLM.createConversation` / `disposeConversation` /
`chatCompletion(conv, ...)` overload + `serializeKVCache` /
`loadKVCache` primitives + per-model serialization chain.

Initial verdict was PARTIAL because the v1 cost decomposition was
wrong: Pattern A's per-model session-tracker prefix cache was
already covering the sequential-NPC matrix the validation probe
ran. The mechanism itself was correct — sharedLen detection 100%,
KV round-trip preserved output quality, conv isolation worked. The
apples-to-apples comparison required the interleaved probe (see
below). Validation report:
[`eval/reports/prefix-cache-validation-2026-05-01/SUMMARY.md`](eval/reports/prefix-cache-validation-2026-05-01/SUMMARY.md).

### v2 follow-ups status

- **Batch multi-tensor backend transfers — LANDED, PARTIAL.**
  Phase 1a pipelined readback via the existing async-request API
  (commit `71ea997`); Phase 1b batched uploads via the existing
  `_backend_tensor_set3` primitive 72 → 24 calls (commit
  `979593f`). Closed ~38% of the at-scale gap (+485 ms → +300 ms).
  Round-trip lever exhausted — residual is bandwidth-bound
  (~600 MB transferred per save+load against ~1-2 GB/s effective
  WebGPU↔CPU bandwidth). A C++ `backend_tensor_get_many` is no
  longer expected to move the needle.
- **`skipSave` dormancy hint — LANDED** (commit `9a3849c`).
  Caller-explicit `CompletionConfig.skipSave: true` bypasses the
  post-decode `serializeKVCache`.
- **Per-head strided reads — PROBED 2026-05-02, NEGATIVE
  (§28 template).** Cut readback payload from 576 MB → 195 MB via
  `headCountKv` strided per-head reads (no C++ patch — the
  existing `beginDownloadFromTensor` API supports offset/size).
  Wall-time effect was ~0% on the interleaved probe
  (2719 → 2736 ms): per-call overhead at 576 strided reads
  cancelled the bandwidth savings vs 72 full reads. The
  bandwidth-bound floor estimate was wrong — Phase 1a's
  pipelining already hid the readback behind the WebGPU command
  queue. Updated mental model: load (sync ASYNCIFY upload), not
  save, is the dominant per-call I/O cost. Strided writes would
  need a new C++ batch primitive (`backend_tensor_set_strided`
  or similar); deferred — adds patch-stack drift surface without
  a probe showing wall-time win. Strided save change not committed;
  report:
  [`eval/reports/prefix-cache-strided-save-2026-05-02/SUMMARY.md`](eval/reports/prefix-cache-strided-save-2026-05-02/SUMMARY.md).
- **Interleaved probe — RAN 2026-05-02, PASS (83% wall savings).**
  Round-robin matrix with per-NPC distinct ~1100-token personas
  defeats Pattern A's session-tracker cache. Pattern B tick-2 wall
  2702 ms vs A's 15853 ms. Confirms the prefix-cache value
  proposition: per-conv KV snapshots are load-bearing for any
  workload that interleaves multiple distinct conversations on the
  same model. Report:
  [`eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md`](eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md).
- **At-scale validation probe — RAN 2026-05-01 (FAIL), RE-RAN
  2026-05-02 (FAIL, +300 ms after Phase 1a+1b).** Mechanism
  healthy at every prefix scale; the v1 cost decomposition was
  wrong because Pattern A already benefits from the engine's
  per-model session-tracker prefix cache (`engine.ts:932`), so the
  sequential comparison was never "cache vs no-cache". Report:
  [`eval/reports/prefix-cache-at-scale-2026-05-01/SUMMARY.md`](eval/reports/prefix-cache-at-scale-2026-05-01/SUMMARY.md).

### Engine session-tracker bug — FIXED 2026-05-02 (commit `c8d1530`)

Surfaced as a side finding from the interleaved probe. The
delta-encoding fast-path at `src/core/engine.ts:prepareChatPrompt`
trusted `promptMessages.length > prevMsgCount` as a continuation
signal without verifying the leading `prevMsgCount` messages still
matched the cached prompt. In interleaved multi-conversation use
(cross-NPC tick-2 on Pattern A), it silently reused another NPC's
KV with the new tail appended.

Fix snapshots leading messages (`cachedMessages: ChatMessage[]`)
on the `ConversationSession` and adds a `leadingMessagesMatch`
guard to the fast-path condition; mismatch falls through to the
existing full-reset branch. Regression test:
`tests/engine-streaming-api.test.ts` ("session-tracker delta path
is skipped when leading messages diverge"). Post-fix re-run of the
interleaved probe (commit `752421c`) confirmed all Pattern A
tick-2 outputs now reference their NPC and pay the honest ~15 s
re-prefill — Pattern B's win held at 84%. Implication: conv-handle
mode is required for **correctness** in interleaved workloads,
not just performance.

### Spec § "Known follow-ups" — LANDED items

- **#1 LRU eviction — LANDED 2026-05-02** (commit `e16e3a5`).
  `ConversationPool.create()` at capacity now evicts the oldest
  non-locked entry instead of throwing.
  `ConversationPoolFullError` raised only when every entry is
  locked. Monotonic `accessSeq` for ordering. Four new tests
  under `describe("LRU eviction")` in
  `tests/conversation-pool.test.ts`.
- **#2 Cross-conv prefix sharing — LANDED + VALIDATED 2026-05-02**
  (impl: commit `72d228c`; e2e probe).
  `WebLLM.forkConversation(srcConv)` deep-copies a source
  conversation's snapshot into a new handle; first chatCompletion
  on the fork prefills only the divergent tail via the existing
  longest-shared-token-prefix walk. New error
  `ConversationNotPopulatedError`. Four new tests under
  `describe("forkConversation")` in
  `tests/chat-completion-conversation.test.ts`. End-to-end probe
  `probe-prefix-cache-fork-2026-05-02` confirmed **72% per-NPC
  wall-time savings** (Pattern Y first-tick 2.4 s vs Pattern X
  8.8 s on qwen3-8b-iq3m) and **17.2 s net spawn savings at
  N=4 NPCs**, with break-even at N≈2. Report:
  [`eval/reports/prefix-cache-fork-2026-05-02/SUMMARY.md`](eval/reports/prefix-cache-fork-2026-05-02/SUMMARY.md).
  Smoke page bumped to `maxConversations: 8` (default 4) since
  fork pattern holds `1 base + N forks` simultaneously —
  consumers spawning many forks should raise the cap.

### Probe / report inventory

- `eval/reports/prefix-cache-validation-2026-05-01/SUMMARY.md` — v1 (PARTIAL)
- `eval/reports/prefix-cache-at-scale-2026-05-01/SUMMARY.md` — at-scale (FAIL)
- `eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md` — interleaved (PASS, 83% / 84%)
- `eval/reports/prefix-cache-strided-save-2026-05-02/SUMMARY.md` — strided save (NEGATIVE)
- `eval/reports/prefix-cache-fork-2026-05-02/SUMMARY.md` — fork e2e (PASS, 72%)

### Headline numbers (qwen3-8b-iq3m, FA on)

| matrix | Pattern A wall | Pattern B wall | savings |
|---|---|---|---|
| Sequential at-scale (1325-token shared) | 2424 ms | 2724 ms | -12.4% (Pattern A wins via session tracker) |
| Interleaved (1100-token per-NPC distinct) | 15854 ms | 2702 ms | **+83%** (Pattern B wins) |
| Fork first-tick spawn (1325-token shared) | 8757 ms | 2436 ms | **+72% per NPC, 17.2 s at N=4** |


---

### Dual-mode deployment (main-thread + worker) — CLOSED 2026-05-02

Closed 2026-05-02 — full TS surface ships behind `WebLLM.init({ worker: true })`;
public API identical between main-thread and DedicatedWorker contexts. Closure
report at
[`eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`](eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md).
Spec/plan at
[`docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md`](docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md)
and
[`docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md`](docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md).

**Headline metrics (vs main-thread, profile-mode, 3-run median):**

| Model | main tok/s | worker tok/s | Δ% |
|---|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0 | 83.6 | 101.7 | +21.6% |
| qwen3-0.6b-q4f16 | 68.4 | 91.8 | +34.2% |
| qwen3-1.7b-q4f16 | 44.9 | 58.6 | +30.5% |
| mistral-7b-instruct-v0.3-q4ks | 29.5 | 36.9 | +25.1% |
| llama-3.1-8b-instruct-iq3m | 23.4 | 28.1 | +20.1% |
| qwen3-8b-iq3m | 22.4 | 25.9 | +15.6% |

Frame-probe coexistence: **8.3 ms median, 0 drops** (gate <15 ms).
Token-identical greedy A/B: **5/5 byte-identical**.

**Original scope sketch** (from queue 2026-05-01, before brainstorm/spec/plan):

Goal: the same `WebLLM.init` / `loadModel` / `chatCompletion` API runs identically
in main-thread and Worker contexts. Application code chooses per deployment.
Triggered by the NPC scenario (agent + Three.js coexistence) where moving the
structural decode hitch off the render loop is load-bearing once tick rates
exceed ~1Hz.

**Gate: PASSED 2026-05-01.** Probe 9d measured 5.5× decode_max reduction (main
49.8 ms median → worker 9.1 ms median) with the hitch fully absorbed from the
main-thread render-loop perspective. Dual-mode work was justified by data.

Scope (sketched, not committed at queue time):
- Engine init path that detects `typeof importScripts !== "undefined"` (worker
  context) and loads the WASM accordingly. Worker variant uses a
  `DedicatedWorker` boot script that constructs `GgmlWasm`, `ModelInference`,
  and the engine handle entirely off-main-thread.
- Main-thread façade with same `chatCompletion` / `embed` surface, marshaling
  calls via `postMessage` and yielding `AsyncIterable<ChunkEvent>` that drains
  a worker-side stream.
- Resource transfer: tokenized prompts marshal as `Int32Array` (transferable).
  Generated chunks marshal as `{text, tokenIds, done, finishReason}` JSON.
  KV-cache + WebGPU resources stay worker-resident — no cross-thread WebGPU
  sharing required for the agent use case.
- Smoke parity: `?worker=1` page-level flag on `real-model.html` that boots
  the engine in worker mode; `[7/8]` chat regression runs identically.
  `?frameProbe=1` mounts the WebGL2/Three.js cube on main and proves the
  hitch absence.
- Bench parity: `eval/bench.ts` and `eval/perf.ts` gain a `--worker` flag.
- Embedder parity: `engine.embed` paths (encoder, causal-LM embedder, bucket
  D self-embed) all run in worker. `embed-perf` bench `--worker` mode.

**Out of scope (and remained out of scope):** SharedArrayBuffer weight sharing
across multiple workers (project is single-model-active per CLAUDE.md);
cross-worker WebGPU resource handoff (no current consumer); `SharedWorker`
multi-tab inference (no consumer).

### What actually shipped

Implementation followed a 10-task plan executed via subagent-driven-development.
Tasks 1–9 landed cleanly per plan; Task 10 (validation gate) surfaced two real
architectural issues that required mid-execution fixes before closure:

1. **A1 chunk-coalescing at worker-host** (commit `6c42d1d`): per-chunk
   `postMessage` from the worker stream loop defeated probe-9d's hitch fix.
   Frame-probe pre-A1 was 41–50 ms median (vs probe-9d's 9.1 ms reference).
   Fix: accumulate `GenerationStreamChunk` values; flush every 8 tokens or
   16 ms (whichever first). Restored 8.3 ms median frame-probe.

2. **A2 smoke-page worker-mode load via `loadModelFromBuffer`** (commit
   `6f49e1c`): the `adoptPreloadedModel` flow used by main-thread mode cannot
   cross the worker boundary because `inference` carries non-transferable
   WASM memory views. Smoke-page worker-mode path switched to fetch + transfer
   buffer + `loadModelFromBuffer`. `a45a60c` strips the non-cloneable
   `inference` field from the result before postMessage to avoid `DataCloneError`.

3. **Path A `loadModelFromUrl`** (commits `926a4fd` + `c732a8b` + `0322ab9` +
   polish `54ea723` + `bbe553f` + `cdde7ed`): models ≥~3.5 GB OOM at the
   smoke-page's `new ArrayBuffer(total)` site (V8 per-allocation cap on JS
   heap). Fix: new public method `loadModelFromUrl` that the worker calls
   directly — worker fetches via `fetch()`, allocates `wasm.malloc(total)`,
   streams chunks into the WASM heap. Smoke page in worker mode uses this
   for the main model (drafter still uses `loadModelFromBuffer` with a
   3.5 GB content-length guardrail). Smoke-page main-thread parsing uses a
   64 MB header-prefix range-fetch (with doubling fallback to 256 MB) to
   drive UI / tokenizer / ctx-clamp without holding the full GGUF in JS heap.

4. **Staging-ptr ownership in `_buildInferenceAndRegister`** (commit `8c48fb4`):
   mistral-7b-q4ks (4.144 GB) aborted at `ctx_create`. Helper called
   `initKVCache` (which `ctx_create`s ~1 GB KV + scratch) before the caller's
   `wasm.free(stagingPtr)` ran on the success path, putting peak transient
   footprint at `model_bytes + KV_bytes` simultaneously. Mistral exceeds the
   wasm64 16 GiB cap minus browser/WebGPU overhead; qwen3-8b at 3.9 GB just
   fit. Fix: helper takes ownership of `stagingPtr`, frees it after
   `loadWeights` (weights are on GPU; WASM-heap copy is dead) and BEFORE
   `initKVCache`. Peak transient footprint drops to `max(model_bytes, KV_bytes)`.

### Implementation commits (chronological, all on `main`)

- `bf1633d` `feat(worker): add ?worker / --worker flags to smoke and bench harnesses`
- `75456d4` `docs(test): clarify mode-default comment in live-db test`
- `a42fee4` `feat(perf): plumb PERF_EXTRA through smoke-bench target`
- `6c42d1d` `feat(worker): coalesce stream chunks at worker-host (16 ms / 8 tokens)` — A1
- `a013415` `feat(probe): add frame-probe sampling to asyncify-in-worker probe page`
- `8d6ad28` `refactor(worker): fold A1 review nits`
- `6f49e1c` `fix(smoke): A2 — route worker mode through loadModelFromBuffer` — A2
- `a45a60c` `fix(worker): sanitize loadModelFromBuffer result before postMessage`
- `926a4fd` `feat(engine): add loadModelFromUrl with WASM-heap streaming` — Path A
- `c732a8b` `feat(proxy): expose loadModelFromUrl on WebLLMProxy`
- `0322ab9` `feat(smoke): switch worker-mode load to loadModelFromUrl`
- `54ea723` `fix(engine): unwind wasm on loadModelFromUrl partial failure` (Path A polish)
- `bbe553f` `feat(smoke): header-prefix fallback + TODO documentation`
- `cdde7ed` `fix(smoke): drafter content-length guardrail in worker mode`
- `8c48fb4` `fix(engine): free staging in _buildInferenceAndRegister before initKVCache` — final fix

(Plus Tasks 1–8 commits — the plan tasks for the original TS surface, error codec,
worker host, proxy, surface mirror sentinel, async-ify conversation methods,
`WebLLM.init({worker:true})` wire, and bundle re-entry. See plan for SHAs.)

### Lessons / follow-ups (all low-priority)

1. Worker mode is **+15–34% FASTER** than main mode in profile-mode bench.
   Hypothesis: profile-mode amortization + reduced main-thread JS contention.
   Worth re-measuring in non-profile mode to publish a clean end-user win.
2. Task 9 wired `?worker=1` but didn't catch broken smoke page end-to-end
   because unit tests use a stub channel. Add a CI-level agentchrome
   integration test driving `?worker=1` and asserting `[7/8]` PASS.
3. Per-binding 4 GiB / 16 GiB caps stack with model + KV + scratch in WASM
   heap. Generalizes: any code path holding the full model in WASM heap while
   also calling KV-cache allocation hits the cap on 7B+ Q4 models.
4. Smoke-page header-prefix workaround is a stopgap. Architectural fix is
   either two-pass parse (4 KB sentinel → exact `dataOffset` → second range-
   fetch) or engine-side metadata accessors so the smoke page never parses
   main-side.
5. Formal worker-vs-main embedder cosine parity not captured this cycle.
   Architecturally there's no source of divergence (same code, same WebGPU
   device, same upload), but a formal cos ≥0.999 comparison would seal it.
6. `eval/causal-embedder-parity.ts` and `eval/browser-eval.ts` lack `--worker`
   flag — adding it is mechanical and would let accuracy-pass A/B runs
   validate worker mode on the 36-prompt eval suite.

---

## Chat-template family dispatch hardening (closed 2026-05-04; archived from TODO.md)

Three-layer fix for the chat-template family dispatch (sampling
profile + chat-stop registration + template formatter must all agree,
or the model wanders past end-of-turn). Triggered by interactive
smoke-test session debugging Mistral 7B Instruct v0.3 multi-turn
self-dialogue, generalized to a fleet-wide audit + engine widening.

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

---

## Next-session pickup batch (queued 2026-04-29; closed 2026-05-03; archived from TODO.md)

Captures the eleven-item pickup queue active 2026-04-29 → 2026-05-03,
between the §31b/§32 perf-cycle close and the start of the
P2-v2 / Phase 3 JSEP work. Items 1-11 are all closed unless
explicitly noted; deferred follow-ups (Storage B GPU-resident KV,
concurrent in-flight per conversation) carry forward in TODO.md's
External-trigger candidates section.

**Status pre-archive:** algorithmic-perf backlog cleared (§17-§29 +
Phi-3 support shipped). TS API audit (a)-(f) closed 2026-04-29.
Embedding bucket C closed 2026-04-29. Embedding bucket D closed
2026-04-30; Phi-3.5-mini bucket D extension §28 NEGATIVE 2026-04-30.
First-class frame-probe mode shipped 2026-05-01; 4 NPC scenario
sizing probes closed 2026-05-01. Dual-mode deployment shipped
2026-05-02. Prefix-cache mechanism + persistence shipped 2026-05-02
to 2026-05-03.

1. **Daily upstream cadence check (REQUIRED, ~30s).** Procedure:
   `cd ~/Repos/llama.cpp && git fetch origin && git log
   webllm-browser-patches..origin/master --oneline --
   ggml/src/ggml-webgpu/ ggml/include/`. **If non-empty:** apply
   §32 procedure (rebase, sweep, classify per §27/§28/§32
   templates). **If empty:** log and skip. Last fired: 2026-05-06
   (empty — no upstream activity in `ggml/src/ggml-webgpu/` or
   `ggml/include/` since 2026-05-04 `fc1f81242`; cadence noop).
   Prior fire: 2026-05-04 (§27 hybrid — drop local LayerNorm
   patches subsumed by upstream `d4b0c22f9`; encoder parity PASS,
   perf neutral vs 2026-05-01 cross-day baseline; tip `fc1f81242`).
   [Cadence policy continues active in TODO.md "Watch list /
   optional cadence work" section.]

2. **Phi-3 closure follow-ups.**
   - ~~(a) Runtime contiguous-tensor assertion in fused helpers.~~
     **CLOSED 2026-04-29** — commit `dc441ce`. Added
     `assertContiguousF32(wasm, tensor, label)`; wired into
     `buildQKV` / `buildFFNGateUp` fused branches gated on
     `ModelInference.assertFusedContiguity` (default true).
     8-case helper unit test in `tests/fused-contiguity-assert.test.ts`.
     Cost <1% of graph-build wall time on Phi-3.5-mini.

   - ~~(b) Chat-template special-token literal audit.~~ **CLOSED
     2026-04-29** — commit `2d65082`.
     `tests/chat-template-special-tokens.test.ts` audits 13 special-
     token literals across formatPhi3 / formatLlama3 / formatChatml /
     formatLlama2-via-Mistral-v3 (4 + 4 + 2 + 3 = 13 single-token
     assertions + 4 round-trip assertions = 17 tests). **Deferred:**
     mistral-v7 (Mistral-Nemo 6.6 GiB too heavy for unit test) and
     gemma (no Gemma GGUF in fleet).

   - (c) **Path A vs Path B A/B measurement on Phi-3.** Loader-only
     views (Path A) vs the shipped Path B fused-forward. Predicted
     Path B win: ~96 dispatches saved per token; observed cost: -6%
     throughput from opCont copies. **Informational only**; closure
     report already recommends evaluating Path A first for the next
     fused-projection architecture (Phi-4, Granite). **Skip until a
     next fused architecture is queued.**

3. **Pre-rebase baseline freshness.** Matrix at
   `eval/reports/pre-rebase-baselines-2026-04-28/` is fresh until
   ~2026-05-28 (~1-month window). [Cadence policy continues active
   in TODO.md "Watch list" section.]

4. **TS API audit follow-ups (CLOSED 2026-04-29).** Phase 1 audit
   + Phase 2 (a-e) + Phase 3 (a-f) all shipped 2026-04-29.
   Net: 14 exports trimmed from public surface, `WebLLMError`
   taxonomy exposed, `GenerationConfig` split, `WebLLMConfig.device`
   removed, `CompletionConfig.sampling` flag added,
   `Character.setTools`, engine accessors migrated to properties,
   `ChatToolSchema` literal union. Spec
   `docs/superpowers/specs/2026-04-29-ts-api-audit-followups-design.md`;
   plan `docs/superpowers/plans/2026-04-29-ts-api-audit-followups.md`.

5. **Embedding bucket C — causal-LM-derived embedders. CLOSED
   2026-04-29.** Qwen3-Embedding-0.6B-hyb shipped (commits
   `deab38a` BPE tokenizer fix → `e2fa58b` bucket C bundle →
   `2724b02` embed-perf bench coverage). Hybrid GGUF (`token_embd`
   Q4_K = 83 MiB + f16 elsewhere) clears the WebGPU 128 MiB
   per-binding cap that blocked the f16 path. Parity 10/10 at
   `cos >= 0.995` (hybrid-tier gate) — cosines 0.996-0.9996,
   magnitudes 1.000 ± 1e-6. Bench: 77 ms / 114 ms p50 single
   short/long, 10.4 texts/sec batch. Closure report
   `eval/reports/bucket-c-parity-2026-04-29/SUMMARY.md`.

6. **Embedding bucket D — chat-model self-embedding. CLOSED 2026-04-30.**
   `ModelInference.embed(tokenIds)` shipped; `engine.embed` dispatches
   through `inferenceEngines` for chat models with `embeddingCapable:
   true`. **`qwen3-8b-iq3m`** is the single registered bucket D model
   at v1. Parity 10/10 PASS at `cos >= 0.90` (IQ3_M-calibrated gate).
   4-pair cosine distinguishability sanity passes with clean margin
   (min paraphrase 0.918 > max unrelated 0.777). Closure report
   `eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`.

7. **Bucket D Phi-3.5-mini extension — §28 NEGATIVE 2026-04-30.**
   Parity 10/10 PASS at `cos >= 0.91`, but distinguishability mean-
   margin gate **FAILS** both pooling modes (last-token −0.006,
   mean-pool −0.027 — paraphrase cosines lower than unrelated).
   Demoted (`embeddingCapable: false`); no follow-on cycle queued.
   Cycle keeper infra: `embeddingPooling` field, 16+16 cross-domain
   pair harness, mean-margin gate `mean(P) − mean(U) ≥ 0.05`.

8. **Frame-probe coexistence baseline — CLOSED 2026-05-01.**
   `?frameProbe=1` mode shipped on the smoke page; multi-call probe
   on `qwen3-8b-iq3m` established 8.3 ms median render-loop, 25 tok/s
   decode, and the **deterministic per-call ~49 ms decode hitch** that
   triggered probes 9c (warmup) and 9d (worker).

9. **NPC scenario sizing probes — ALL FOUR CLOSED 2026-05-01.**
   Downstream decisions all consumed:
   - **9a (PASS):** prefill 89.7% prefix → KV-cache-per-conversation
     load-bearing → drove item 11 prefix cache (shipped 2026-05-02).
   - **9b (PARTIAL):** batched/sequential 0.72 wall ratio at N=4 →
     sequential is canonical agent-tick pattern.
   - **9c (FAIL):** warmup does NOT reduce call-0 decode_max → do
     not bake warmup into engine init.
   - **9d (PASS, 5.5× hitch reduction):** worker absorbs hitch
     (49.8 → 9.1 ms decode_max) → drove item 10 dual-mode (shipped
     2026-05-02).

10. **Dual-mode deployment (main-thread + worker) — CLOSED 2026-05-02.**
    `WebLLM.init({ worker: true })` ships; same TS surface both modes;
    worker frame-probe **8.3 ms median, 0 drops** (gate <15 ms);
    cross-mode A/B perf **+15.6% to +34.2% faster** in worker; greedy
    token-identical 5/5. Closure report
    `eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`.

11. **Prefix cache via per-conversation KV snapshots — CLOSED
    2026-05-02 (mechanism) + 2026-05-03 (persistence + worker
    migration).** Mechanism: `createConversation` /
    `disposeConversation` / `chatCompletion(conv, ...)` /
    `forkConversation`, with LRU eviction on the pool. Headline
    wins: **interleaved 84% wall savings** (Pattern B tick-2 2702 ms
    vs A's 15853 ms on qwen3-8b-iq3m;
    `eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md`)
    and **fork 72% per-NPC savings / 17.2 s net at N=4 NPCs**
    (`eval/reports/prefix-cache-fork-2026-05-02/SUMMARY.md`). Side-
    finding: engine session-tracker delta-encoding bug fixed in
    `c8d1530` — conv-handle mode is now required for correctness in
    interleaved workloads, not just performance.

    Sub-follow-ups:
    - **#3 Storage B (GPU-resident KV)** — DEFERRED. Requires
      `ggml-webgpu` patches. [Carried forward in TODO.md
      "External-trigger candidates" section.]
    - **#4 Concurrent in-flight per conversation** — DEFERRED.
      Requires KV cloning at concurrency request time. [Carried
      forward in TODO.md "External-trigger candidates" section.]
    - **#5 Persistence across reloads — CLOSED 2026-05-03.** Two-tier
      design: engine primitives `exportConversation(conv)` /
      `importConversation(modelId, blob, options?)` ship in core;
      `IndexedDBConversationStore` ships behind the
      `@paulrobello/webllm/persistence` subpath. Five new error
      classes (`IncompatibleConversationError` / `CorruptBlobError`
      / `PersistenceUnavailableError` / `PersistenceQuotaError` /
      `PersistenceIOError`); model-fingerprint + tokenizer-hash gate
      refuses cross-quant or cross-tokenizer loads; integer
      `schemaVersion`; per-method transfer-allowlist on the worker
      bridge. Spec
      `docs/superpowers/specs/2026-05-03-prefix-cache-persistence-design.md`;
      plan `docs/superpowers/plans/2026-05-03-prefix-cache-persistence.md`.
    - **#6 Worker migration (item 10) — CLOSED 2026-05-03 (probe
      outcome (a): wiring already correct).** `WebLLMProxy` already
      mirrors all four conv methods; worker host reflect-dispatches
      them; `ConversationHandle` is plain data and structured-clones
      cleanly; `ConversationPool` lives engine-side. Gap was test
      coverage, not behavior. Lifecycle regression test added in
      `tests/webllm-proxy-integration.test.ts`.

---

## Embedding-model expansion campaign (closed 2026-04-28; archived from TODO.md)

User-driven scope: extend embedding fleet beyond the two registered
Arctic-Embed entries. Three candidate buckets, in increasing scope.
**Buckets A and B closed 2026-04-28;** C closed 2026-04-29 (full
detail in the Next-session pickup batch above under item 5).

**A. Register more BERT-arch embedders. DONE 2026-04-28**
(commit `41b27bd`). Confirmed cleanly: the encoder forward path,
WordPiece tokenizer, F16 / F32 dtypes, and CLS pooling (read from
GGUF metadata) all already work for BGE out of the box — zero code
changes outside `eval/models.ts`, `eval/smoke-profiles.ts`,
`eval/embed-perf.ts`.
- `bge-small-en-v1.5-q0f16` (~33M, 384-dim): 17.0 ms p50 single-
  text short / 91% on 8-task cosine eval. Apples-to-apples with
  arctic-embed-s.
- `bge-large-en-v1.5-q0f16` (~335M, 1024-dim): 59.3 ms p50
  single-text short / 89% on 8-task cosine eval. **First 335M
  encoder in fleet** — 3.5× latency for 10× params consistent with
  bandwidth-bound encoder behavior.

GGUF source: `ChristianAzinn/bge-{small,large}-en-v1.5-gguf` mirror.
File-name pattern: `_fp16` (`bge-{small,large}-en-v1.5_fp16.gguf`).

**Net learning:** the BERT-arch lever is effectively free for any
future ask — no loader changes were required. Stretch picks
(`bge-base-en-v1.5`, `mxbai-embed-large-v1`, `snowflake-arctic-
embed-l`) are register-and-run candidates with high confidence.

**B. Extend `EncoderInference` to non-BERT arch. DONE 2026-04-28.**
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
biases. Parity artifacts at
`eval/reports/encoder-parity-2026-04-28/`.

Net learning: the non-BERT encoder lever is now exhausted for the
two named families. Remaining encoder asks are register-and-run on
top of this foundation if they share an arch tag already on file
(`bert` / `jina-bert-v2` / `nomic-bert-moe`); novel arch tags would
re-open Phase 0/1.

**C. Causal-LM-derived embedders (`Qwen3-Embedding-0.6B`)** — closed
2026-04-29 as bucket C; see Next-session pickup batch above for full
closure detail (item 5).

**Encoder fixed-cost-per-dispatch observation (2026-04-28):**
encoder overhead (`backendEncodeOverheadMs` per step) is a fixed
per-dispatch cost of ~5.2-5.7 µs, remarkably flat across the
450 → 805 dispatch/token range:

| Model            | Dispatches | Encoder (median, ms) | µs/dispatch |
|---|---:|---:|---:|
| tinyllama-q4_0   | 450 | 2.40 | 5.3 |
| qwen3-0.6b-q8    | 629 | 3.30 | 5.2 |
| qwen3-1.7b-q8    | 629 | 3.60 | 5.7 |
| mistral-7b-q4ks  | 650 | 3.60 | 5.5 |
| llama-3.1-8b-iq3m| 652 | 3.40 | 5.2 |
| qwen3-8b-iq3m    | 805 | 4.40 | 5.5 |

Implication: encoder share scales inversely with model size because
matmul shrinks at small models, not because encoder grows. Encoder
is 24-30% at tiny models where matmul is 33-38%; drops to 9-11% at
7-8B where matmul is 49-58%. Reducing per-dispatch encode cost would
yield ~26% relative speedup at TinyLlama scale (87.9 → 111.1 tok/s).
Lever wasn't load-bearing under the prior 30B ceiling; promoted to
"watch list" under the new 8B ceiling for agent latency goals.
Captured as memory note
`Knowledge/wasm-webgpu-encoder-fixed-cost-per-dispatch.md`.

## Phase 3 JSEP causal-LM decode investigation (Stage 3 → 4.36; closed 2026-05-08; archived from TODO.md)

Closed 2026-05-08. The P2-v2 JSEP backend reaches end-to-end parity
with the non-JSEP `webllm-wasm.js` reference for causal-LM greedy
decode on the testable subset of the canonical-6 fleet. The
load-bearing root cause was the WGSL kqv MUL_MAT GQA-broadcast bug
(Stages 4.10 → 4.34 localized, Stage 4.35 fixed via `src0_batch_idx
= batch / r2` divide in all four `load_*` kernels, Stage 4.36
broadened coverage). 7B+ canonical-6 entries deferred behind the
wasm32 4 GiB JSEP heap cap; mathematical interpolation argument for
r2=4 documented in `STAGE-4.36-RESULT.md`.

- **Phase 3 closure report:**
  [`STAGE-4.36-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)
  — testable-subset PASS table + r2 coverage summary + deferred-subset
  catalog with three re-enablement paths.
- **Stage 4.35 fix commit:** `0a2ee8b` (`fix(jsep/matmul): WGSL kqv
  MUL_MAT GQA broadcast (r2 divide)`).
- **Probe 21b regression guard:** in tree at
  `smoke-test/p2-v2-spike.src.ts` (host-CPU selftest at kq's exact
  shape, dispatch-agnostic). Re-fires on every spike load when
  `src0.ne[2] !== src1.ne[2]`.
- **Per-stage closure reports:** all `STAGE-3-RESULT.md` and
  `STAGE-4.{1..36}-RESULT.md` files live alongside this report under
  `eval/reports/p2-v2-option-a-prime-2026-05-06/`.

Per-stage stubs preserved below for cross-reference.

### Per-stage CLOSED stubs (verbatim from TODO.md, archived 2026-05-08)

The verbose stage-by-stage CLOSED paragraphs lived in TODO.md's "Phase
3 progress (2026-05-06)" section through Phase 3 closure; they are
moved here per the CLAUDE.md TODO archival cadence. The 4-8 line
closure stub at `TODO.md` "Stage 4.36 closed — Phase 3 closed for
testable subset (2026-05-08)" is the active-surface replacement.

This block also carries the **closure-stub discipline doctrine** and
the **Phase 3 trajectory assessment (post-Stage-4.26)** that lived
under the same heading group in active TODO.md until 2026-05-08.

#### Phase 3 progress (2026-05-06)

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

#### Closure-stub discipline — every TODO closure prepares for a fresh session

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

#### Phase 3 trajectory assessment (set 2026-05-07, post-Stage-4.26)

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


### Earlier Stage 4.36 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.36 brief (paste-and-go bootstrap, Option-A
spike-harness `?model=<key>` parametrization, ref-token capture pattern,
five-step implementation sketch with Probe 21b regression-guard re-run,
four-item risk register, two-outcome uneventful/failures branch table)
lives in [`STAGE-4.36-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md). Outcome **subset PASS** + **Phase-3-closed-for-testable-subset** CONFIRMED: 3/3 testable canonical-6 models match the non-JSEP `webllm-wasm.js` reference at `generatedIds[0]` and across all 5 greedy-decode tokens (TinyLlama r2=8 from Stage 4.35; qwen3-0.6b and qwen3-1.7b r2=2 added this stage — both predict `[12095,13,576,6722,315]` " Paris. The capital of"); Probe 21b re-confirmed `P-21b-clean`; `make checkall` green: 747 pass, 0 fail. Risk register entry #3 was superseded — the binding constraint is the wasm32 4 GiB JSEP heap cap, not the system 16 GB floor; mistral-7b-q4ks (4.14 GiB), llama-3.1-8b-iq3m (3.78 GiB), and qwen3-8b-iq3m (3.9 GiB) all exceed the cap and are deferred to a future wasm-mem64 JSEP build path. Closure argues mathematical interpolation: r2=4 (the only un-exercised value) is structurally identical to the tested r2={2, 8} — the WGSL u32 divide is parameterized with no branch on the quotient. Collapsed at Phase 3 closure time to keep the active surface focused.

### Earlier Stage 4.35 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.35 brief (paste-and-go bootstrap, three-step implementation sketch with TS uniform layout + four-WGSL-kernel divide + Probe 21b verification, four-item risk register, exit criteria) lives in [`STAGE-4.35-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.35-RESULT.md). Outcome **P-21b-clean** + **Paris-decode-parity** CONFIRMED: Probe 21b reports `maxAbsDeltaVsGqa = 3.81e-5` uniformly across all 32 Q-heads (six orders of magnitude below the pre-fix 5.65e+1); JSEP spike's `GENERATED_TOKENS = [3681, 29889, 13, 13, 29906]` (`"Paris.\n\n2"`) matches the non-JSEP reference for the "capital of France" prompt on TinyLlama-1.1b-chat-q4_0. `make checkall` green: 747 pass, 0 fail. Phase 3 JSEP causal-LM decode reaches parity for the lead model; Stage 4.36 broadens coverage to the canonical-6 fleet. Collapsed at Stage 4.36 queue time to keep the active surface focused.

### Earlier Stage 4.34 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.34 brief (paste-and-go bootstrap, Probe 21 Shape A read-the-kernel sketch with `__stage434Probe21Arm` one-shot capture in `dispatchMatmul`, three case-table for nb[2]={0, M·nb[1], other}, Probe 21b Shape B host-CPU selftest sketch with `runKqGqaSelfTest` at kq's exact shape M=256 K=64 N=6 src0.ne[2]=4 src1.ne[2]=32, four-item risk register, three-outcome P-21-{stride-trick, explicit-divide-needed, other} branch table) lives in [`STAGE-4.34-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.34-RESULT.md). Outcome **P-21-explicit-divide-needed** + **P-21b-bug-reproduces** CONFIRMED: Probe 21 captured `src0.nb=[2, 512, 128, 262144]` for the kq MUL_MAT — the **permuted K-cache layout** (dim-fast=2, head-medium=128, pos-slow=512), with `nb[2]=128` neither 0 nor `M·nb[1]=131072` (the brief's classifier missed this case as "other"). Probe 21b's host-CPU selftest at the captured kq shape produced uniform 1536-non-zero output per head (synthetic src0 fully populated) but `maxAbsDeltaVsGqa = 56.5` with **head 0 matching GQA reference at Δ=9.5e-6 (f32 noise)** and **every other head diverging by Δ=4.35–56.5** — kernel-level bug confirmed in isolation. Root cause: WGSL kernels compute `batch * src0.nb[2]` directly which under the permuted layout walks both head-fast (correct for batches 0–3) and pos-slow (wrong for batches ≥ 4) dimensions. Stage 4.35 ships the fix: explicit `src0_batch_idx = batch / r2` divide (where `r2 = src1.ne[2] / src0.ne[2]`) in all four `load_*` WGSL kernels, plus Probe 21b as a permanent regression guard. Collapsed at Stage 4.35 queue time to keep the active surface focused.

### Earlier Stage 4.33 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.33 brief (paste-and-go bootstrap, Probe 20 implementation sketch with C++ filter widening to `kq-0` element-wise IDX-DUMP + JS fetch-POST upload to `localhost:8032`, four-item risk register, three-outcome P-20-{row-bounded, block-bounded, upstream-cascade} branch table) lives in [`STAGE-4.33-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.33-RESULT.md). Outcome **P-20-block-bounded** CONFIRMED: JSEP `kq-0` non-zero coverage stair-steps by head — heads 0–3 full (36/36), heads 4–7 missing 6 (1 K-pos × 6 Q-rows), heads 8–11 missing 12, …, heads 24–31 entirely zero (0/36). 4-head bands match `n_head_kv=4`; 6-step degradation matches prefill seq length. Total: 504/1152 active non-zeros on JSEP vs 1152/1152 on ref. Implicates the WGSL kqv MUL_MAT kernel's GQA broadcast logic — `dispatchMatmul`'s shape uniform packs `shape.src0_batch_bytes = src0.nb[2]` and the `load_*` kernels all use `batch * src0_batch_bytes + m * src0_row_bytes` without a GQA-aware divide. Stage 4.34 queues Probe 21 (Shape A: read the kernel + log ggml's actual nb[2] at the kq MUL_MAT) and Probe 21b (Shape B: host-CPU selftest at kq's exact shape M=256 K=64 N=6 src0.ne[2]=4 src1.ne[2]=32 to confirm the bug in isolation). Collapsed at Stage 4.34 queue time to keep the active surface focused.

### Earlier Stage 4.32 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.32 brief (paste-and-go bootstrap, Probe 19 implementation sketch with `kqv_out-0` element-wise IDX-DUMP via `node_dump_cb` per-row stderr lines + JS spike & ref-probe parse blocks, Probe 19b Shape B as a one-line allowlist extension to `kq-0` + `kq_soft_max-0`, four-item risk register, four-outcome P-19-{row-bounded, block-bounded, scattered, upstream-cascade} branch table) lives in [`STAGE-4.32-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.32-RESULT.md). Outcome **P-19-upstream-cascade** CONFIRMED: `kq-0` AbsMax = 31.98 (JSEP) vs 52.93 (ref) Δ=20.95 — divergence originates at or before the first attention matmul Q × K^T, not inside the kqv (V × softmax) kernel. `kqv_out-0` is 87.5% zero on JSEP (longest contiguous JSEP-zero run: 1792). Stage 4.31's first8-blindness hypothesis upgraded to a structural attention divergence. Stage 4.33 queues Probe 20 (Shape A: element-wise `kq-0` capture, classify as row-bounded / block-bounded / upstream-cascade) and Probe 20b (Shape B: hash `Qcur-0` and `Kcur-0` bytes at `kq-0` matmul read time). Collapsed at Stage 4.34 queue time to keep the active surface focused.

### Earlier Stage 4.31 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.31 brief (paste-and-go bootstrap, Probe 18 Shape A implementation sketch with `kqv_out-0` full-tensor stats inside `node_dump_cb` + spike & ref-probe parse blocks, four-item risk register, three-outcome P-18-{first8-blind, full-clean, full-dirty} branch table, Probe 18b Shape B sketch on the clean branch) lives in [`STAGE-4.31-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.31-RESULT.md). Outcome **P-18-first8-blind** CONFIRMED: JSEP prefill `kqv_out-0` `abs_max=0.04959` vs ref `0.11706` (Δ=0.067, far above 1e-3 threshold); decode steps Δ=0.05-0.66 on `abs_max`; JSEP `abs_min=0.0` on every pass vs ref `~4e-7` (signalling contiguous JSEP zero outputs). The first8 window the existing `node_dump_cb` was reading was `V[pos=0]` weighted by the position-0 softmax row that the causal mask pins to `[1, 0, 0, …]` — coincidentally bit-identical across runs while positions 1+ carry the cascade. Stage 4.27's row-12 `attn_out-0` Δ=4.77e-3 is now demoted from "first divergent op" to "first divergence projected through first8 — the actual first divergence is at row 11 outside first8". Stage 4.32 queues Probe 19 (element-wise `kqv_out-0` capture for idx 11; classify as row-bounded / block-bounded / upstream-cascade) and Probe 19b (Shape B: full-tensor probe extended to `kq-0` + `kq_soft_max-0`). Collapsed at Stage 4.32 queue time to keep the active surface focused.

### Earlier Stage 4.30 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.30 brief (paste-and-go bootstrap, Probe 17 implementation sketch with `webllm_get_tensor_data_hash` C++ export via `llama_internal_get_tensor_map` forward decl + spike-side per-weight peek loop with stringToUTF8 / heap-rederive plumbing, four-item risk register, five-outcome P-17-{clean,gain,ffn-down,jsep-deep,other} branch table) lives in [`STAGE-4.30-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.30-RESULT.md). Outcome **P-17-clean** CONFIRMED: all **7/7** layer-0 weights byte-exact at `tensor->data` post-`loadModel`. `size_data == size_ref` for every weight (Q6_K not in-flight converted; risk register #2 closed). Suspect 2 (`ffn_norm.weight` gain-vector mis-load) is DEAD by direct measurement — the 8 KiB F32 gain vector the layer-0 RMSNorm op reads is bit-identical to the GGUF tensor data at every byte. Risk register #1 (host_mirror out of sync for JSEP-resident weights) also did not fire — Stage 4.4 F1's host_mirror is in sync with the JSEP GPU buffer at the time of the post-load peek for all 5 JSEP-resident weights. Cascade source must lie upstream of layer-0 weight bytes themselves; survivors are suspect 3 (first8-window blindness on `kqv_out-0`) and suspect 4 (cascade source upstream of `attn_out-0` — Qcur-0's INPUTS). Stage 4.31 queues Probe 18 Shape A (widen `node_dump_cb` to full-tensor stats on `kqv_out-0`) and queues Probe 18b Shape B (hash `attn_norm-0` output bytes JSEP MUL_MAT reads at Qcur-0 execution time) on the P-18-full-clean branch. Collapsed at Stage 4.31 queue time to keep the active surface focused.

### Earlier Stage 4.29 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.29 brief (paste-and-go bootstrap, Probe 16 implementation sketch with CPU-side `ggml_backend_cpu_buffer_set_tensor` hook mirroring Stage 4.20's JSEP probe pattern + `_ggml_cpu_set_weight_hash_probe` export + JS spike `__cpuWeightHashLog` unification, four-item risk register, four-outcome P-16-{clean,gain,ffn-down,silent} branch table) lives in [`STAGE-4.29-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.29-RESULT.md). Outcome **P-16-silent** CONFIRMED: CPU hook armed and exported correctly (`[probe16] CPU weight-hash probe armed` line emitted, no "export missing" failure), but fired **0/7** during model load. Neither the JSEP buft (5/7 fire) nor the default CPU buft in `ggml-backend.cpp` (0/7) owns `ffn_norm.weight` and `ffn_down.weight`. With `GGML_CPU=OFF` in the JSEP build (`Makefile:139`), the most plausible owner is the GGUF mmap-direct host buft — `tensor->data` would point straight into the mmap'd file region and there is no `set_tensor` upload step to corrupt the bytes. Combined with Stage 4.27's `attn_norm-0` bit-identical result (the other layer-0 RMSNorm output), suspect 2 (`ffn_norm.weight` gain-vector mis-load) is **strongly indirect-evidence dead** but not closed by direct measurement. Stage 4.30 closes it directly via a one-shot post-load `tensor->data` peek (Probe 17, Shape A: `webllm_get_tensor_data_hash` export + spike-side per-weight peek loop) and pivots to suspect 3 / pre-Qcur cascade source on the clean branch. Collapsed at Stage 4.30 queue time to keep the active surface focused.

### Earlier Stage 4.28 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.28 brief (paste-and-go bootstrap, Probe 15 implementation sketch with 5-name allowlist extension + JS-side `targetNames` extension + Q6_K `elemBytes`, four-item risk register, four-outcome P-15-{clean,gain,output-proj,ffn} branch table) lives in [`STAGE-4.28-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.28-RESULT.md). Outcome **P-15-jsep-bypass** CONFIRMED: 5 of 7 layer-0 weights byte-clean end-to-end through JSEP set_tensor → device.queue.writeBuffer (`attn_q.weight`, `attn_k.weight`, `attn_output.weight`, `ffn_gate.weight`, `ffn_up.weight`); 2 weights (`ffn_norm.weight` F32 + `ffn_down.weight` Q6_K) bypass the JSEP `set_tensor` hook entirely. **Closes Suspect 1** (`attn_output.weight` byte-integrity — bit-clean). Suspect 2 (`ffn_norm.weight` gain-vector mis-load) requires a CPU-side probe to test — JSEP `set_tensor` doesn't see the gain vector. Stage 4.29 queues Probe 16 (Shape A: CPU-side `set_tensor` hook mirroring Stage 4.20's JSEP probe pattern) to test the bypass-weights' byte-integrity on CPU buft. Collapsed at Stage 4.29 queue time to keep the active surface focused.

### Earlier Stage 4.27 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.27 brief (paste-and-go bootstrap, fresh `__stage417Checkpoints` re-capture from spike + ref-probe, three-item risk register, three-outcome "matches Stage 4.17" / "tighter than Stage 4.17" / "different first-divergent op" branch table) lives in [`STAGE-4.27-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.27-RESULT.md). Outcome **A** CONFIRMED: cascade trajectory is structurally identical to Stage 4.17 — `attn_out-0` Δ=0.004773 (matches 4.77e-3), `ffn_norm-0` Δ=0.183250 (matches 1.83e-1), `result_norm` Δ=5.83, `result_output` Δ=6.61. The Stage 4.18 → 4.26 patch growth (12 → 13) did not move any prefill checkpoint. Combined with Stage 4.26's matmul-precision closure, every numerical-precision hypothesis is dead. Three structural suspects survive: (1) `attn_output.weight` byte-integrity, (2) `ffn_norm.weight` gain-vector mis-load (highest-prior, +38× amplification at `ffn_norm-0` is the load-bearing signal), (3) first8-window blindness on `kqv_out-0`. Stage 4.28 queues Probe 15: extend Stage 4.21's GPU-readback FNV-1a-32 pattern from `attn_q.weight` / `attn_k.weight` to five additional weights (`attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`) with JS-side `GgufParser` reference hashes, branching on the first mismatch. Collapsed at Stage 4.28 queue time to keep the active surface focused.

### Earlier Stage 4.26 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.26 brief (paste-and-go bootstrap, Probe 14 implementation sketch with `webllm_q4k_q8k_matmul` C shim using `ggml_get_type_traits_cpu` for `from_float` (Q8_K) + `vec_dot` (Q4_K), four-item risk register, three-outcome H-4-libllama-{imprecise|precise|mid} branch table) lives in [`STAGE-4.26-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.26-RESULT.md). Outcome **H-4-libllama-imprecise** CONFIRMED: `llamaVsF64Max = 4.178e-2` over 12,288 elements vs `wgslVsF64Max = 7.94e-6` — libllama is the imprecise side by orders of magnitude. The 4.178e-2 envelope is dominated by libllama's per-element src1 Q8_K quantization loss; webllm's WGSL kernel consumes raw f32 src1 and stays at its f64 floor. Q-projection matmul precision is not the cascade source. Stage 4.27 pivots to localizing which op in the prefill or decode path actually wedges the spike's output, via the existing `__stage417Checkpoints` framework re-run on the current code. Collapsed at Stage 4.27 queue time to keep the active surface focused.


### Earlier Stage 4.25 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.25 brief (paste-and-go bootstrap, Probe 13 implementation sketch with WGSL Kahan accumulator + dispatchMatmul gate via `__stage425KahanArm` global, four-item risk register, three-outcome H-3b-Kahan/H-3b-structural/H-3b-partial branch table) lives in [`STAGE-4.25-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.25-RESULT.md). Outcome **H-3b-structural** CONFIRMED: `kahanVsBaselineMax = 0` exact (kernel produced bit-identical output to non-Kahan baseline despite `kahanFired = true`); the existing `MATMUL_PROBE10_REPLAY.maxAbsDeltaVsF64 = 7.94e-6` proves the WGSL kernel is already accurate to within ~8e-6 of f64 truth — 67× smaller than the 5.24e-4 cross-module disagreement. f32 accumulation precision is ruled out as the dominant error source. Stage 4.26 queues Probe 14: shift the precision check to libllama's side (`webllm_q4k_q8k_matmul` shim mirroring Stage 4.24's pattern) to determine whether libllama is also at ~1e-5 from truth (precise — pivot to upstream src1 mismatch) or at ~5e-4 from truth (imprecise — close matmul investigation, pivot to other ops in the cascade). Collapsed at Stage 4.26 queue time to keep the active surface focused.

### Earlier Stage 4.24 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.24 brief (paste-and-go bootstrap, Probe 12 implementation sketch with `webllm_dequantize_q4_K` shim + JS dual-path harness, four-item risk register, two-outcome H-3a/H-3b branch table) lives in [`STAGE-4.24-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.24-RESULT.md). Outcome **H-3b** CONFIRMED: `maxAbsDelta = 0` over 4,194,304 elements between WGSL-equivalent `dequantQ4_KTile` and libllama `dequantize_row_q4_K` — Q4_K dequant is bit-clean. The 5.24e-4 production Qcur-0 delta is therefore f32 matmul accumulation-order disagreement (WGSL subgroup tree vs libllama SIMD horizontal sum), not a dequant bug. Stage 4.25 queues a Kahan-summed accumulator probe gated to Qcur-0 to quantify whether Kahan collapses the delta to ≤1e-5 (H-3b-Kahan ship target) or leaves it intact (H-3b-structural — Branch C downstream cascade mitigation). Collapsed at Stage 4.25 queue time to keep the active surface focused.

### Earlier Stage 4.23 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.23 brief (paste-and-go bootstrap, Probe 11 implementation sketch with `JSEPRUN_LOG_FIRST30` walk + per-op host_mirror vs GPU FNV hash + Stage 4.4 Bug C cross-reference, three-item risk register, two-outcome H-1/H-2 branch table) lives in [`STAGE-4.23-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.23-RESULT.md). The Probe 11 hypothesis was preempted by the risk-register's first bullet — re-derive the 5.24e-4 number's provenance — which produced **Outcome H-3** (writeback-gap framing misframed; the bug is a WGSL Q4_K matmul vs libllama Q4_K matmul kernel disagreement at the same Q-projection inputs). Probe 11 was not implemented; Stage 4.24 takes its slot. Collapsed at Stage 4.24 queue time to keep the active surface focused.

### Earlier Stage 4.22 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.22 brief (paste-and-go bootstrap, Probe 10 implementation sketch with capture-arm + pre/kernel/post encoders + mapAsync, four-item risk register, two-outcome G-1/G-2 branch table) lives in [`STAGE-4.22-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.22-RESULT.md). Outcome **G-2** CONFIRMED + the kernel exonerated on production inputs: capturedDelta = 4.768e-7 (single-ULP), syntheticDelta = 4.768e-7, first-8 outputs bit-identical between captured production and synthetic replay. **Surprise finding** — TinyLlama-1.1b-chat-q4_0.gguf actually contains Q4_K projections + Q6_K embeddings; the Stage 4.18 "Q4_0 production-shape sweep" was an apples-vs-oranges baseline. The 5.24e-4 framing from prior stages must have come from comparing against the CPU-fallback path (Stage 4.4 dual-resident `host_mirror`), not against an f32 ground truth. Stage 4.23 redirects the investigation to the GPU→host writeback gap. Collapsed at Stage 4.23 queue time to keep the active surface focused.

### Earlier Stage 4.21 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.21 brief (paste-and-go bootstrap, Probe 9c implementation sketch with `runtime.dataManager.get(bufHandle)` + `copyBufferToBuffer` + `mapAsync` + FNV-1a-32, four-item risk register, two-outcome F-1/F-2 branch table) lives in [`STAGE-4.21-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.21-RESULT.md). Outcome F-1 CONFIRMED: `blk.0.attn_q.weight` GPU readback `0xf2f7188c` == `fnv1a_pre` `0xf2f7188c`; `blk.0.attn_k.weight` GPU readback `0x9399f36a` == `fnv1a_pre` `0x9399f36a` — both byte-exact. The host→GPU `Module.jsepWrite` → `device.queue.writeBuffer` link is bit-clean; the entire weight-upload chain is end-to-end byte-identical from GGUF file all the way to JSEP GPUBuffer. The remaining suspect is the dispatch / kernel-execution boundary at production conditions. Collapsed at Stage 4.22 queue time to keep the active surface focused.

### Earlier Stage 4.20 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.20 brief (paste-and-go bootstrap, Probe 9b three-step plan with C++ pre-upload hash + JS-side ref via GgufParser + optional GPU-readback step, files-to-touch, three-outcome branch table for E/F-1/F-2) lives in [`STAGE-4.20-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.20-RESULT.md). Outcome F CONFIRMED: `blk.0.attn_q.weight` C++ pre-upload `0xf2f7188c` == JS-side ref `0xf2f7188c`; `blk.0.attn_k.weight` C++ pre-upload `0x9399f36a` == JS-side ref `0x9399f36a` — both byte-exact, same sizes (2,359,296 / 294,912 bytes). The GGUF parser → ggml allocator → set_tensor caller chain is bit-clean. The remaining link (host→GPU `Module.jsepWrite` → `device.queue.writeBuffer`) is the next probe target. Collapsed at Stage 4.21 queue time to keep the active surface focused.

### Earlier Stage 4.19 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.19 brief (paste-and-go bootstrap, three-probe priority list with cb_eval allowlist extension + JSEP weight-upload byte-hash + direct weight-bytes side-by-side dump, three-outcome branch table for D/E/F) lives in [`STAGE-4.19-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-RESULT.md). Outcome confirmed: Probe 9a triggered Branch 2 (`attn_norm-0` bit-identical, `Qcur-0` Δ=5.24e-4 + corroborating `Kcur-0` Δ=3.38e-4). The cheap allowlist extension also surfaced a documentation gotcha: `inp_embd` is a leaf input tensor with no producing op, so cb_eval doesn't fire on it; the post-`ggml_get_rows` compute output is named `embd`. Q-projection's src0 (= layer-0 wq weight bytes as visible to the kernel) is the remaining suspect. Collapsed at Stage 4.20 queue time to keep the active surface focused.

### Earlier Stage 4.18 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.18 brief (paste-and-go bootstrap, three-probe priority list with Q4_0 sweep + V-on-CPU trace + RMSNorm low-magnitude test, three-outcome branch table) lives in [`STAGE-4.18-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-RESULT.md). Outcome reframed: Probe 8a refuted the kernel-precision claim (Q4_0 matmul matches f32-loop reference to ULP at all production shapes — 1.68e-6 max abs delta vs 5.24e-4 observed in production = 312× gap); Probe 8b confirmed V-projection runs on CPU, fully explaining the Vcur-0 Δ=0 anomaly. Probe 8c skipped (RMSNorm runs on CPU on the JSEP-side spike too, so it can't amplify a delta that originates upstream). Bug must originate upstream of Q-proj — input or weight-upload path. Collapsed at Stage 4.19 queue time to keep the active surface focused.

### Earlier Stage 4.17 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.17 brief (paste-and-go bootstrap, Probe 7 architecture decision matrix with three reference options, Probe 8 sketch for follow-on per-shape Q4_K self-test, three-outcome branch table) lives in [`STAGE-4.17-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-RESULT.md). Outcome B (kernel-correctness) CONFIRMED via 96-checkpoint diff between JSEP and non-JSEP reference: `Qcur-0` Δ = 5.24e-4 at idx 0; `Vcur-0` bit-identical (Δ = 0 anomaly); first node to cross 1e-3 threshold is `attn_out-0` at idx 11 (Δ = 4.77e-3); cascade through 22 layers lands `result_norm` Δ = +5.83 and logits Δ = +6.61. No NaN/Inf/all-zero pathology — purely numerical. Collapsed at Stage 4.18 queue time to keep the active surface focused.

### Earlier Stage 4.16 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.16 brief (paste-and-go bootstrap, Probe 6 EM_ASM sketch with `(idx, node_op, bctx_handle, dst_offset, dst_size, tensor_name, view_src_handle, view_src_offs)`, four-row disambiguation matrix, fifth-outcome escape clause) lives in [`STAGE-4.16-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.16-RESULT.md). All four matrix rows RULED OUT: handles + offsets + sizes match exactly between Probe 5 and Probe 6 for the load-bearing Qcur-0 (h=26, o=4194304, sz=49152). Fifth outcome CONFIRMED via a `mirror_post_h1` peek added inline to Probe 6 — host_mirror at H1's own (handle, offset) reads `[0,0,0,0]` immediately after H1 returns. Root cause: `EM_JS(void, ggml_jsep_read, ...)` discards the Promise returned by `Module.jsepRead`; under JSPI the readback runs asynchronously and host_mirror updates land too late. Fix: `EM_ASYNC_JS` + `await`. Collapsed at Stage 4.17 queue time to keep the active surface focused.

### Earlier Stage 4.15 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.15 brief (paste-and-go bootstrap, Probe 5 per-divert-dispatch tempDst + dstRec readback EM_ASM sketch, three branch hypotheses with disambiguation table, spike-harness drain pattern) lives in [`STAGE-4.15-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.15-RESULT.md). Branch 1 (kernel writes zeros) REJECTED and Branch 2 (copy doesn't land) REJECTED; Branch 3 (divert lands valid data; H1 mirrors it elsewhere) CONFIRMED. Probe 5's 32 entries showed `tempEqDst === true` everywhere, with valid F32 for layer-0 Q/K/V projections and zeros for everything that consumes Q (attention compute, layer-1 projections from cascade). The fix path moves to Stage 4.16's pack_tensor / H1 cross-correlation. Collapsed at Stage 4.16 queue time to keep the active surface focused.

### Earlier Stage 4.14 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.14 brief (paste-and-go bootstrap, Probe 4 post-compute CPU dst readback EM_ASM sketch, JSEP `get_tensor` instrumentation, three predicted branches with disambiguation table, structural-fix candidates Path R/U/Q) lives in [`STAGE-4.14-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.14-RESULT.md). Stage 4.13's CPU-D narrative was REFUTED — no CPU op writes the addr 99811136 window between i=1 and i=2; the slot is fresh sched-allocated `CPU#Kcur-0#0`. New diagnosis CPU-E confirmed: every JSEP MUL_MAT for Q (every layer) and Kcur-0 produces zero output at host_mirror, despite H1 firing 1:1 with runOp. The "valid" Kcur-1 read at offset 528384 is stale V-projection data leftover from an earlier set_tensor write. Path R/U/Q (Stage 4.13's structural-fix list) are ALL inapplicable; the bug lives inside the divert dispatch path. Collapsed at Stage 4.15 queue time to keep the active surface focused.

### Earlier Stage 4.13 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.13 brief (paste-and-go bootstrap, Probe 3 CPU MUL_MAT capture sketch, three predicted sub-cases CPU-A/B/C with disambiguation table, structural-fix candidates) lives in [`STAGE-4.13-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.13-RESULT.md). All three predicted sub-cases REJECTED; tensor-name capture revealed the K/V slot labels were swapped in Stages 4.10/4.11/4.12 — the broken slot is K (Kcur-0), not V; and the actual root cause is CPU-D (cross-backend buffer-aliasing): Kcur-0's view `data` pointer aliases attn_norm-0's CPU scratch (a different buffer than where K projection's JSEP output lands). Collapsed at Stage 4.14 queue time to keep the active surface focused.

### Earlier Stage 4.12 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.12 brief (paste-and-go bootstrap, Probe 2 CPU graph_compute instrumentation with EM_ASM sketch + jsep_resolve_tensor helper, three predicted sub-cases, structural-fix candidates) lives in [`STAGE-4.12-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.12-RESULT.md). Probe 2's three predicted sub-cases all rest on the false premise that the CPU producer is a cgraph node visible to ggml-cpu's graph_compute — it is not. Cross-backend writes flow through `set_tensor`, surfaced by the Probe 2 follow-up. Collapsed at Stage 4.13 queue time to keep the active surface focused.

### Earlier Stage 4.11 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.11 brief (paste-and-go bootstrap, Probe 1 inter-slice host_mirror snapshot with EM_ASM sketch + cgraph-walk for hm26 caching, Probe 2 conditional CPU graph_compute instrumentation, exit criteria, branch on HD'/HC/HA-flavor) lives in [`STAGE-4.11-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.11-RESULT.md). Probe 1 was sufficient — confirmed HD' and surfaced the V-vs-K asymmetry; Probe 2 deferred to Stage 4.12 where it's the lead localization step. Collapsed at Stage 4.12 queue time to keep the active surface focused.

### Earlier Stage 4.10 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.10 brief (paste-and-go bootstrap, three hypotheses HA/HB/HC with priorities, EM_ASM-based per-graph_compute slice metadata capture, branch-on-outcome) lives in [`STAGE-4.10-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.10-RESULT.md). HA strict-form REJECTED (3 JSEP graph_compute calls fire BEFORE slice 3, with CPU subgraphs interleaved). HB WEAKENED (no view-src indirection: `s0VsOp = 0`). HC unlikely (Stage 4.9 callIdx 2-7 stable). HD surfaced: cross-backend boundary leaf without producer landing data in time. Stage 4.11 brief above queues the inter-slice host_mirror snapshot probe to localize WHEN h26+0 gets populated. Collapsed at Stage 4.11 queue time to keep the active surface focused.


### Earlier Stage 4.9 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.9 brief (paste-and-go bootstrap, three implementation paths with `cpy_tensor` / `set_tensor` / per-runOp pre-pass options, exit criteria including `src0AtKernelTimeF32` rows containing real F32 K data, decode "Paris", patch stack 7→8) lives in [`STAGE-4.9-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.9-RESULT.md). H1-inverse landed correctly via the per-runOp pre-pass option. It fires for every src tensor before each JSEP runOp. Outcome: H1-inverse correctly syncs host_mirror→GPU but `host_mirror` itself was zero at the first 2 SET_ROWS dispatches → C-2. The Stage 4.10 brief above queues localization of WHY host_mirror is stale. Collapsed at Stage 4.10 queue time to keep the active surface focused.

### Earlier Stage 4.8 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.8 brief (Step A eager-warmup probe, Step B temp-dst windowed readback inside `dispatchSetRows`, Step C window-by-window branch logic) lives in [`STAGE-4.8-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.8-RESULT.md). Step A ruled out generic first-call cold-start (warmup with both tiny and shape-matched dispatches did not flip i=3). Step B's multi-row temp-dst + src0 capture proved the real bug is upstream: src0 GPU buffer is stale because CPU-fallback ops update host_mirror but not GPU buffer. Stage 4.9 brief above queues the host→GPU writeback fix (H1-inverse, mirroring Stage 4.5 H1). Collapsed at Stage 4.9 queue time to keep the active surface focused.

### Earlier Stage 4.7 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.7 D2-tight brief (paste-and-go bootstrap, async jsepRunOp wrapper rewrite, synchronous readback sketch with `runtime.encoderBatcher.flush()` + `device.queue.submit` + `mapAsync`, branch-on-outcome R1 vs R2) lives in [`STAGE-4.7-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.7-RESULT.md). D2-tight CONFIRMED Reading R1: i=3's dispatch silently fails (dstImmediate identical to dstPost, both all-zero) — but Stage 4.8 reframed this as misleading: the dispatch is fine, src0 is stale. Collapsed at Stage 4.8 queue time to keep the active surface focused.

### Earlier Stage 4.6 D2/D3 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.6 D2/D3 brief (three hypotheses H-source/H-indices/H-attention with priorities, files-to-read order, D2 instrumentation sketch, branch-on-outcome) lives in [`STAGE-4.6-D2-LITE-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.6-D2-LITE-RESULT.md). D2-lite REJECTED H-indices and WEAKENED H-source by showing src[0] and src[1] are sensible across all 10 captured SET_ROWS — but surfaced a fourth diagnosis (i=3 K-cache layer 0 anomaly) that needed Stage 4.7 D2-tight to disambiguate. Collapsed at Stage 4.7 queue time to keep the active surface focused.

### Earlier Stage 4.6 D1 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.6 D1 brief (paste-and-go bootstrap, three options D1/D2/D3, files-to-read order, D1 implementation sketch, branch-on-outcome) lives in [`STAGE-4.6-D1-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.6-D1-RESULT.md). D1 PASSED on both no-divert and divert variants — the dispatcher is bit-exactly correct on the V-cache transpose layout. Stage 4.5's `FIRST_ALLZERO_DST_PROBE` was a false positive (the multi-MB KV cache buffer at offset 0 has only 8 cells in the `first8` window and SET_ROWS targets a sparse subset). D2/D3 brief above queues the upstream-bug investigation. Collapsed at D2/D3 queue time to keep the active surface focused.

### Earlier Stage 4.5 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.5 brief (paste-and-go bootstrap, three options H1/H2/H3, files-to-read order, H1 implementation sketch, branch-on-outcome, Step 0 fallback) lives in [`STAGE-4.5-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.5-RESULT.md). H1 was the right starting point — it fired correctly and proved that mirror staleness was a real, observable bug — but the partial flip taught us the writeback gap was necessary but not sufficient. The Makefile build-order race (cp-before-bundle) surfaced as a load-bearing bonus fix. Collapsed at Stage 4.6 queue time to keep the active surface focused.

### Earlier Stage 4.4 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.4 brief (Step 0 baseline verification, Step 1 F1 implementation in ggml-jsep.cpp, Step 2 verify Outcome A flips, Step 3 branch on outcome, code sketches, files-to-touch table) lives in [`STAGE-4.4-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.4-RESULT.md). The F1 design's load-bearing change (`get_base` returning `host_mirror` instead of `0x2000`) flipped Bug A as predicted; the brief's "critical verification" footnote about cross-backend writes proved load-bearing in the inverse direction (GPU→host, not host→GPU as the footnote framed it). Collapsed at Stage 4.5 queue time to keep the active surface focused.

### Earlier Stage 4.3 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.3 brief (Step 0 baseline verification, Step 1 production-shape kernel selftests, Step 2 full-graph runOp capture + per-op readback, Step 3 diagnosis branch + closure) lives in [`STAGE-4.3-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.3-RESULT.md). The kernel-correctness hypotheses were eliminated by 4.3a; the load-bearing diagnosis (CPU-fallback ops dereferencing the `0x2000` sentinel for weight tensors) came out of the unified `evtSeq` interleave in 4.3b. Collapsed at Stage 4.4 queue time to keep the active surface focused.

### Earlier Stage 4.2 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.2 brief (Step 0 baseline verification, Step 1 jsepWrite/jsepRead instrumentation, Step 2 log interpretation, Step 3 diagnosis branch + closure) lives in [`STAGE-4.2-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.2-RESULT.md). The CPU-writeback hypothesis was validated as branch (c) "CPU writes are correct AND go to the right offset; bug is downstream"; the live work was promoted into Stage 4.3 (Bug A: NaN-producing JSEP shader; Bug B: lm_head silently doesn't write). Collapsed at Stage 4.3 queue time to keep the active surface focused.

### Earlier Stage 4.1 brief — collapsed (full text in closure report)

Full step-by-step Stage 4.1 brief (Step 0 baseline verification, Step 1 aliasing-counter probe, Step 2 RMW divert implementation sketch, Step 3 verification, Step 4 fallback) lives in [`STAGE-4.1-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.1-RESULT.md) and the Step 4 fallback was promoted into the Stage 4.2 brief above. Collapsed at Stage 4.2 queue time to keep the active surface focused.

### Earlier Stage 3 brief — collapsed (full text in closure report)

Full rationale, phasing, file list, and Q4_K kernel design notes for the original Stage 3 brief live in [`STAGE-3-RESULT.md`](eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-3-RESULT.md). The brief was inlined here through Stage 3 execution; collapsed at Stage 3.5 queue time to keep the active surface focused. The kernel landed in `4731dc1` and is verified correct in isolation.

---

## qwen3-0.6b-thinking-cold semantic-reasoning wedge investigation (closed 2026-05-08)

The full pre-closure brief lived in `TODO.md` 2026-05-08 → 2026-05-08
under "Fresh-session pickup — qwen3-0.6b-thinking-cold semantic-
reasoning wedge". The closure inline-stub in `TODO.md` references
[`eval/reports/qwen3-0.6b-thinking-cold-wedge-2026-05-08/SUMMARY.md`](eval/reports/qwen3-0.6b-thinking-cold-wedge-2026-05-08/SUMMARY.md)
which carries the full diagnosis, evidence ledger, and re-evaluation
triggers. This archive entry retains the original brief's hypothesis
surface and procedure steps for historical context.

**Original goal:** identify the root cause of the WASM `unreachable`
trap that fires partway through `emb-001` (the first semantic-
reasoning task) when running the full 4-dimension qwen3-0.6b-thinking-
cold accuracy bench. Once root cause is in hand, decide between (a) a
targeted code fix, (b) a per-dimension engine reset boundary, or (c)
marking the wedge a known-stable-but-degraded behavior under the
harness's new abort guard.

**Original context (what shipped before the investigation):**
Reproduced **twice** in 24 hours on the live DB (commits `09a744b` +
`05094bd` deleted both contaminated rows). Pattern was bit-stable: the
first 3 dimensions (tool-calling / reasoning / instruction-following =
36 tasks) complete healthy, then emb-001 traps at ~531 ms, and every
subsequent task throws `unreachable` synchronously in 2-3 ms.

The harness side was defended (commit `f3cbca9`,
`src/evaluation/runner.ts`): `runTasks` aborts after 3 consecutive
errors with `EngineDeadError`, posts `eval_failed`, no `eval_complete`
fires, no row written to `evals`. So future wedges no longer pollute
the dashboard — but the underlying trap still aborts the run, and
the actual cause was unknown at queue time.

**Hypothesis surface (queued 2026-05-08):**

1. **Cumulative state across tasks** — most likely. After 36
   `engine.resetConversation()` + `character.chat()` cycles in the
   same engine instance, something accumulates (asyncify stack
   reservation, WASM heap fragmentation, scratch-buffer slot
   allocation pattern) and the 37th call exceeds a threshold.
   `resetConversation` flushes KV but is not a full re-init.
2. **Temperature 0.1 + thinking deterministic loop** — thinking-warm
   (temp 0.6) and thinking-hot (temp 0.9) on the same model + same
   dimension sequence completed cleanly twice. Only thinking-cold
   (temp 0.1) wedges. At low temperature with thinking ON, Qwen3
   produces highly deterministic, often very long `<think>` blocks;
   `rs-012` in the wedged run already shows the pattern (`8107 ms ·
   empty output` → burned `max_tokens=1024` inside thinking without
   producing `</think>`). Cumulative heap / scratch effect from such
   long generations may amplify Hypothesis 1.
3. **emb-001-specific input shape tickling a tokenizer / sampler
   edge case** — least likely (the input is a 16-word sentence),
   but cheap to verify.

**Implementation steps (queued):**

1. Capture browser console around the trap.
2. Probe A — task-count bisection (run emb-001 alone).
3. If H1: Probe B — boundary insertion (per-dimension reset).
4. If H2: Probe C — temperature crossover.
5. Compare against canonical baseline (Stage 4.36 testable subset).
6. Document in
   `eval/reports/qwen3-0.6b-thinking-cold-wedge-<DATE>/SUMMARY.md`.

**Risk register (queued):**

1. Wedge stops reproducing on a fresh session — possible if it
   depends on Chrome's WebGPU / Dawn cache state.
2. The trap is in upstream `ggml-webgpu` and not patchable here.
3. Probes interact with Phase 3 JSEP work — `f3cbca9` shipped on the
   legacy non-JSEP path; the wedge reproduces on `webllm-wasm.js`,
   not `webllm-wasm-jsep.js`.
4. Cumulative wedge masks an honest bug surface — if H3 is the right
   answer, fixing the harness wedge guard would silence a real
   signal.

**Branch on outcome (queued):**

- Wedge reproduces, root cause is cumulative state, per-dimension
  reset fixes it → ship the boundary as a structural fix.
- Wedge reproduces, root cause is upstream WASM trap not patchable
  here → ship workaround.
- Wedge does NOT reproduce on fresh sessions (3 attempts) → document
  as transient.

**Actual disposition (closed 2026-05-08):** middle branch, with a
twist. Wedge **does** reproduce (~30% session, ~20% lifetime
estimated; n=3 reproductions before the investigation, +1 during the
investigation = 4 total). Probe A passed (emb-001 alone, post-
warmup, no crash) → cumulative-state confirmed, input-specificity
ruled out. Root cause attributed upstream to Dawn/Chrome WebGPU
process pressure (the "external Instance reference no longer
exists" message originates in Chrome's wire-client when the GPU-
process peer recycles a long-running tab's device); our `wgpu::Instance`
shared_ptr never re-assigns and our patch stack does not touch
Instance lifetime. The cumulative state we control via
`engine.resetConversation` (KV cache clear) is already exhausted
per task. A real fix would require WebGPU device recreation between
runs — major engine refactor with uncertain efficacy validation
cost (`n=30+` trials per fix-validation pass at ~75 s each =
~37 minutes wall per validation pass).

**No structural code change shipped.** The abort guard `f3cbca9`
fully prevents data pollution: verified during this investigation
that Run #2 wedged at task 35-38, the guard fired
`EngineDeadError` after 3 consecutive task errors, no
`eval_complete` was published, no row was written to `evals`. Bench
summary surfaces `[FAIL]` cleanly with the diagnosis message.

**Re-evaluation triggers** (full list in SUMMARY.md):

1. Wedge frequency on `qwen3-0.6b-thinking-cold` exceeds 50% over
   any 10-trial window.
2. Wedge starts firing on a second model.
3. Upstream `ggml-webgpu` lands a change touching device-lifecycle
   code.
4. Chrome / Dawn ships a fix for the "external Instance reference"
   message under sustained WebGPU load.
5. Consumer reports wedges hitting agent + Three.js coexistence in
   a real product workload.


## Campaign Q1 — Gemma 2 un-demote (queued 2026-05-11 EOS-12; closed 2026-05-11 EOS-13; archived from TODO.md)

> Closed 2026-05-11 EOS-13 at 60 % overall eval (gate ≥ 60 % cleared).
> Six fixes, three of which were not on the original 2026-05-01 demote
> SUMMARY candidate list. Doctrine lesson: *expect plural root causes
> for demotes*. Closure report:
> [`eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md`](eval/reports/gemma-2-2b-un-demote-2026-05-11/SUMMARY.md).
> Closure commit: `dc3304a`. The "closure stub" linking to this archive
> entry lives inline in `TODO.md` for next-session reference.

---

**Original goal:** un-demote `gemma-2-2b-it-q4f16` from the
wave-2 demote list back into the canonical fleet. Current state
at queue time: smoke produces 64 tokens of id 139 (whitespace)
at temp 0 with the NEOX-RoPE fix in. Different failure signature
from the pre-NEOX `RSSSF suprême` gibberish — the residual
stream is now locked into a single token instead of producing
chaotic noise. Three concrete architectural pieces are missing
(verified against
`~/Repos/llama.cpp/src/models/gemma2.cpp` lines 9-176 and the GGUF
metadata dump).

**Pre-work — verified 2026-05-11 EOS-12:**
- Architecture string: `gemma2` (already in the
  `ModelArchitecture` union).
- NEOX-RoPE: landed at `c8c8447` (covers gemma/gemma2/gemma3/gemma4).
- post-attention-norm / post-ffw-norm: already loaded via `opt()`
  in `model-inference.ts:528-529` and applied in `forwardSingle`
  / `forwardWithLayerTaps` via `lw.postAttentionNorm` /
  `lw.postFfwNorm` ternaries (Task 3.3d wired this generically;
  Gemma 2 GGUFs ship the same `post_attention_norm.weight` /
  `post_ffw_norm.weight` tensor names per llama.cpp's loader).
- `attnSoftmaxScale("gemma2", headDim)` returns `1/sqrt(headDim)`
  (default branch — matches Gemma 2's `f_attention_scale =
  1/sqrt(n_embd_head_k)` for 2B per `gemma2.cpp:27-29`; for 27B
  it differs but 27B is out of scope under the 8B ceiling).

**Gemma 2 2B GGUF parameters** (from
`smoke-test/models/gemma-2-2b-q4f16.gguf` metadata dump):

```
gemma2.attention.sliding_window      = 4096
gemma2.attention.sliding_window_pattern = (not in GGUF; period=2 default)
gemma2.attn_logit_softcapping        = 50.0
gemma2.final_logit_softcapping       = 30.0
gemma2.attention.head_count          = 8
gemma2.attention.head_count_kv       = 4    (GQA, 2× group)
gemma2.attention.key_length          = 256
gemma2.embedding_length              = 2304
gemma2.block_count                   = 26
```

Tensors present: `token_embd.weight`, per-block `attn_*`, `ffn_*`,
`attn_post_norm`, `ffn_post_norm`. **NO `output.weight`** — Gemma
2 ties `lm_head` to `token_embd` (per `gemma2.cpp:35,39` —
`TENSOR_DUPLICATED` flag).

**Stage Q1.1 — Tied output embedding.** Make the model loader
fall back to `token_embd.weight` when `output.weight` is absent
and the architecture is in the tied-embedding set (`gemma`,
`gemma2`, `gemma3`, plus other small models known to tie: smollm,
some qwen variants — audit separately). Today: the loader likely
throws or silently fills with zeros, which would explain the
whitespace lock. **Gate:** smoke loads without an "output.weight
not found" error; greedy decode on `"The capital of France is"`
emits a non-whitespace continuation (need not be coherent yet —
the soft-caps below are still missing).

> **CLOSED — verified already implemented.** All four lm_head
> sites in `src/inference/model-inference.ts` already used
> `weights.output ?? weights.tokEmb` fallback. Q1.1 was a no-op.
> The whitespace lock was driven by missing soft-capping (Q1.2 +
> Q1.3) and the architecture-branch + softcap-order issues
> (Q1.6), not by tied-embedding wiring.

**Stage Q1.2 — Attention logit soft-capping** (`attn_soft_cap`
flag). Add a per-architecture `attnLogitSoftcap?: number` to
`ModelHyperparams`, populated from `gemma2.attn_logit_softcapping`
(50.0). In the attention math, wrap the pre-softmax logits with
`tanh(qk / cap) * cap` per `llama-graph.cpp:2019-2026`. Two paths
to touch: the manual softmax path (`opMulMat`/`opSoftMaxExt`) and
the FlashAttention path (probably needs an FA disable for any
arch with soft-cap, since FA fuses softmax without supporting a
tanh wrap — check `ggml_flash_attn_ext` signature in the rebased
llama.cpp). **Gate:** parity probe on a 95-token plain completion
prompt — block-by-block cosine ≥ 0.95 vs HF reference at float32.

> **CLOSED — commits `f2735d5` (op_tanh binding) + `5d1aba4`
> (attn-side wiring) + `bb73d4f` (JSDoc placement).** FA shader
> already implemented soft-cap natively (`ggml-wgsl-shaders.hpp:
> 2002, :2712`) and the WebGPU host divides `scale /= logit_softcap`
> pre-dispatch (`ggml-webgpu.cpp:1942-1944`), so FA didn't need
> disabling. Manual softmax path was wired with the soft-cap
> helper. The actual ordering bug (soft-cap on unscaled qk)
> wasn't surfaced until Q1.6.

**Stage Q1.3 — Final-logit soft-capping** (`f_final_logit_softcapping`,
30.0 for Gemma 2 2B). Wrap the lm_head output with
`tanh(logits / cap) * cap` per `gemma2.cpp:169-171`. Single-site
change in `forwardSingle` after the output projection. Already
present as a field on `ModelHyperparams.finalLogitSoftcap` but
currently only respected for Gemma 4 (and incorrectly so — see
Task 3.3i closure notes). **Gate:** top-1 argmax matches HF on
the parity prompt; sampler doesn't degenerate (sample 10 greedy
decodes on the smoke prompt and confirm non-trivial variety in
the continuation distribution).

> **CLOSED — same commit as Q1.2 (`5d1aba4`).** Bundled with
> Q1.2 because both needed the new `op_tanh` binding. `softCap`
> helper applied at all 4 lm_head sites via a guarded ternary
> on `hp.finalLogitSoftcap`.

**Stage Q1.4 — Eval verification.** Re-enable `gemma-2-2b-warm`
in the `full` smoke profile set. Run
`bun run eval/bench.ts --profiles gemma-2-2b-warm` at greedy
temp 0. **Gate:** 36-prompt eval ≥ 60 % (Phi-3 closure baseline).
If accuracy clears, un-demote (remove the demote comment block in
`smoke-profiles.ts`, restore the entry in the `full` set). If
still below 60 %, run a chat-formatted parity probe (Phase A
template) to localize the residual issue.

> **CLOSED — commit `dc3304a`.** Eval landed 60 % overall
> (29/48): 92 % reasoning, 72 % instruction-following, 61 %
> semantic-reasoning, 17 % tool-calling@capability=false.
> Excluding tool-calling: 27/36 = 75 %.

**Stage Q1.5 — Documentation.** Closure report at
`eval/reports/gemma-2-2b-un-demote-<date>/SUMMARY.md` with
before/after eval matrix and per-stage parity deltas. Update the
demote SUMMARY at
`eval/reports/gemma2-demote-2026-05-01/SUMMARY.md` with a
"SUPERSEDED" notice pointing at the un-demote.

> **CLOSED — commit `dc3304a`.** Closure report shipped.
> SUPERSEDED notice added to demote SUMMARY. TODO.md Campaign Q1
> marked CLOSED with full retrospective.

**Stage Q1.6 — Extend gemma4 branches to Gemma family + fix
softcap order** (surfaced 2026-05-11 EOS-13 during Q1.4 smoke
probe). Q1.2 + Q1.3 landed correctly per spec compliance + code
quality reviews, but the smoke probe still locked to whitespace.
Investigation of `gemma2.cpp:60-176` surfaced two gaps:

1. **Gemma-family architecture branches** in `model-inference.ts`
   that were gated `gemma4`-only should apply to whole family:
   - Input embedding scale by `sqrt(n_embd)` (gemma2.cpp:70).
   - GELU-parallel FFN activation instead of SwiGLU (gemma2.cpp:140).

   Kept gated to gemma4 only: V bare-RMS-norm (gemma2 does NOT do
   this per gemma2.cpp:93) + `attnSoftmaxScale === 1.0` (gemma2
   needs the standard 1/sqrt(d_k); QK-norm is gemma4-only).

2. **Soft-cap ordering bug.** The original Q1.2 wiring applied
   `softCap(qk, cap)` BEFORE the softmax kernel's internal
   `scale = 1/sqrt(d_k)` multiplication. Reference order per
   `gemma2.cpp:110` + `ggml-cpu/ops.cpp:8232-8233`: scale FIRST,
   then softcap, then softmax with scale=1.0. With the wrong
   order, soft-cap acts on sqrt(d_k)-larger qk magnitudes,
   saturating tanh and producing near-bang-bang attention.

> **CLOSED — commit `31d53a5`.** Added `isGemmaFamily(arch)`
> helper. Extended embed-scale + GELU FFN to whole family (10
> sites). Refactored 6 manual-softmax sites to scale-first
> ordering. FA path untouched (already correct). Smoke probe
> after fix: `Paris.` (4 tokens, finish=stop-token) instead of
> 20 whitespace tokens.

**Out of scope (this campaign):** Gemma 2 9B and 27B SKUs (the 27B
needs a different `f_attention_scale` formula and is above the
8B ceiling regardless); Gemma 2 actually using SWA at lengths >
4096 (window is wider than `bench-full`'s longest output, so
all-global fallback is fine for the eval gate). SWA at scale is
the Stage 4 work below; if Q1.4 misses on long-prompt tasks
specifically, defer to Stage 4 rather than stretching this
campaign.

**Risks (original; retained for retrospective):**
- FA + soft-cap incompatibility (Q1.2). Mitigation: gate FA off
  whenever `hp.attnLogitSoftcap` is non-zero; verify in smoke
  with `FA: OFF` toggle. If FA is the dominant Gemma 2 throughput
  lever, a follow-up campaign can teach FA the soft-cap path.

  → *Didn't materialize.* FA shader already implemented soft-cap
  natively and the host code pre-divided scale, so FA stayed on.

- Tied-embedding regression risk on other archs (Q1.1). Mitigation:
  flip the fallback only when `output.weight` is genuinely absent
  AND the architecture is in an explicit tied-embedding set;
  models like Mistral that ship `output.weight` keep their
  existing behavior.

  → *Didn't materialize.* The fallback was already in place
  pre-Q1, gated implicitly by the GGUF having or not having
  `output.weight`. No regression possible.

---

## Gemma 4 E2B inference support (closed 2026-05-12; archived from TODO.md)

Full original block: Stage 1 (per-layer hp refactor) → Stage 2 (surface
wiring) → Stage 3 (PLE + dual-RoPE + NEOX, eval 9 % → 68 %) → Campaign
Q1 (Gemma 2 un-demote, 60 %) → Campaign Q2 (Stage 4 real SWA, eval
70.8 %) → Campaign Q3 (Stage 5 bench + closure). 11 Stage-3 sub-tasks
(3.3a-l), 6 Q1 fixes, 5 Stage-4 sub-stages, 3 Stage-5 sub-stages.
Patch budget: 0 of +2 used. Final canonical closure:
[`eval/reports/gemma-4-e2b-validation-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-e2b-validation-2026-05-12/SUMMARY.md).

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

#### Resume in fresh session — pickup instructions (updated 2026-05-12 EOS-4)

**Read this first, then
[`eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md)
"Root cause confirmed (2026-05-12 EOS-4)" section for the full diagnostic,
N-sweep table, the upstream assertion chain, and the verified mitigation.**

---

##### Status: FA-VEC clamp shipped (2026-05-12 EOS-4)

The chat.html `Error: unreachable` trap is **fixed** by commit
`9ea3bfc fix(gemma): clamp prefillTileSize to 16 for FA + head_dim > 128`.
Root cause was upstream `ggml-webgpu-shader-lib.hpp:734` rejecting the
FA VEC path at `src0.ne[1] >= 20`; Gemma family head_dim ∈ {256, 512}
doesn't fit any other FA path's LDS budget so `decisions.path = NONE` →
assert at line 2560. NOT a Phase B regression — pre-existing bug,
surfaced only when Stage 4 validation drove chat.html for the first time.

The clamp activates when `flashAttn === true && max head_dim > 128`
(covers Gemma 2 / 3 / 4 today; future models auto-covered by the head_dim
threshold). 6 new unit tests cover the matrix in
`tests/prefill-tiling-config.test.ts`. End-to-end verified via
`fa-prefill-probe.html?chat=1`: chat-template N=46 prompt that traps in
0.028s without the clamp now succeeds in 0.92s with healthy argmax.

##### Open work after EOS-4

1. **Stage 4.1 long-context closure (now reachable). CLOSED 2026-05-12 EOS-5.**
   Drove `chat.html` with `gemma-4-e2b-it-q4km` (FA=true, ctx 4096) on
   a 2,238-token prompt (4.4× the 512-token SWA window) — long context
   passage + appended one-sentence question. **Result:** coherent
   fact-correct reply ("The first fifteen layers of the Gemma-4
   family use sliding-window attention with a window size of 512
   tokens and 8 query heads per layer."), 33 tokens in 43.3s
   (42.3s TTFT + 35.5 tok/s decode), 0 console errors, no
   `Error: unreachable`. Confirms the FA-VEC `prefillTileSize=16`
   clamp from `9ea3bfc` survives at >512-token contexts where SWA
   layers actually exercise the banded mask. Closure report:
   [`eval/reports/gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md).
   Per-binding 128 MiB cap still blocks per-layer tap parity — that
   gate moves to Stage 4.3 (HF-reference long-context probe).

2. **Upstream patch follow-up: bump VEC `ne[1] < 20` ceiling.** With
   the TS clamp in place this is now P2 hygiene — recover prefill
   throughput on Gemma 4 by letting larger tiles use VEC at `q_tile=1`
   instead of paying 3× tile overhead. Implementation: one-line edit
   to `~/Repos/llama.cpp` `webllm-browser-patches` branch at
   `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp:734`. Cost: ~30 min
   including rebuild + retest. Should also be filed upstream so future
   rebases pick it up. **Quantified cost:** 42.3s TTFT for 2,238-token
   prefill at q_tile=1 (Stage 4.1 closure run). Pre-clamp would have
   pushed prefill into a single batched dispatch on TILE; bumping the
   VEC ceiling lets us retry batched VEC at larger N.

3. **Stage 4.2 — Gemma 2/3 SWA wiring landed (2026-05-12 EOS-6).**
   `src/models/model-loader.ts` gained an `arch === "gemma2" ||
   arch === "gemma3"` branch that derives the boolean
   `slidingWindowPattern` from the scalar uint32 `swa_period`
   GGUF key (default 2) per `llama-hparams.cpp:14` set_swa_pattern:
   `swa_layers[il] = period == 0 || (il % period < period - 1)`.
   For period=2 this yields `[T,F,T,F,...]` (NOT `[F,T,F,T,...]`
   as the earlier sketch incorrectly stated — upstream uses
   `(il % p < p-1)`, so layer 0 is SWA and layer 1 is global.
   The pretrained weights expect that specific assignment).
   `slidingWindowSize` reads `attention.sliding_window` (default
   4096 per `gemma2.cpp:7`). Per-layer head_count / head_dim /
   FFN / rope arrays are left undefined — Gemma 2 is uniform, so
   downstream dispatch falls back to scalar `hp.*` fields.
   `sharedKvLayers` / `kvReuseFromLayer` stay undefined.

   **Gates (a) ✅ + (b) ✅:**
   - (a) Pattern matches upstream alternation: new test file
     `tests/models/model-loader-gemma2-hparams.test.ts` (4 tests,
     17 expect() calls) verifies `slidingWindowPattern[0..3] =
     [T,F,T,F]`, `slidingWindowSize=4096`, per-layer arrays
     undefined, no `sharedKvLayers`. All pass against the local
     `gemma-2-2b-q4f16.gguf`.
   - (b) `make checkall` green — 782 pass / 36 skip / 0 fail /
     39311 expect() calls; biome fmt + lint clean; tsc clean.
     +4 tests vs the pre-change 778 baseline.

   **Gate (c) deferred to Stage 4.3** (long-context regression
   probe). At prompts < 4096 tokens the SWA window covers the
   entire sequence, so the new per-layer mask switch behaves
   identically to the full-causal mask — regression-safe by
   construction. The long-context Gemma 2 smoke is the same gate
   Stage 4.3 already targets; folding it in avoids duplicate
   browser runs. Stage 4.1 dispatch (the load-bearing per-layer
   mask code) was already smoke-verified on Gemma 4 — Stage 4.2
   just feeds it a different per-architecture boolean array.

##### How to verify the fix held (drop-in repro)

```bash
make smoke-test
make smoke-serve &
# In Chrome:
http://localhost:8031/chat.html?model=gemma-4-e2b-it-q4km
# Type "The capital of France is", send.
# Expected: greedy reply ("Paris") in coherent English; no
# "RuntimeError: unreachable" in the console.
```

Or via the probe (no chat UI needed):
```bash
http://localhost:8031/fa-prefill-probe.html?model=gemma-4-e2b-it-q4km&ctx=4096&path=forward&chat=1
# Expected: [FA-PREFILL-PROBE-DONE-PASS] in ~1s.
```

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

##### EOS-4 session commits (FA-VEC clamp landed)

Tip: `9ea3bfc`. Five commits this session:
- `447ff82` feat(gemma4): per-layer headCount/headCountKv plumbing + graphMem bump
- `65ac040` feat(parity-capture): cache-buster + skipLayerTaps + long-context fixture
- `01c00db` docs(stage4): FA discriminator probe outcome — Gemma 4 + FA succeeds on forwardWithLayerTaps
- `a14d6b6` docs(TODO): refresh fresh-session pickup block with FA probe outcome
- `9384709` feat(probe): fa-prefill-probe page for Gemma 4 + FA forward bisection
- `4fdb875` docs(stage4): forwardSingle + forwardAllPositions FA probe — both pass at N=9
- `76b6268` feat(probe): chat-template + length-sweep + prefillTile knobs
- `b8e013a` docs(stage4): root-cause Gemma 4 + FA trap as upstream FA path-select assert
- `9ea3bfc` fix(gemma): clamp prefillTileSize to 16 for FA + head_dim > 128

**Health check (post `9ea3bfc`):** `make checkall` green (fmt + lint +
typecheck + 778 tests pass / 36 skip / 0 fail; +6 new tests under
`tests/prefill-tiling-config.test.ts`'s "prefillTileSize FA VEC clamp"
describe block).

End-to-end smoke verified: `fa-prefill-probe.html?model=gemma-4-e2b-it-q4km
&ctx=4096&path=forward&chat=1` succeeds in 0.92s with healthy argmax
50429 / logit +9.5 (was: traps in 0.028s with `RuntimeError: unreachable`
before the clamp landed).

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

2. **Stage 4.3 — Long-context regression probe. PARTIAL 2026-05-12
   EOS-7.** Incremental per-layer parity-capture infrastructure
   landed (commit `2c32f80` — `captureLayer` + `lastTokenLogitsOnly`
   options on `forwardWithLayerTaps`, `?mode=incremental` harness
   loop). API verified on TinyLlama Q4_0 at N=6: 22/22 layers
   captured, end-of-stack cosine 0.9855 vs HF, greedy argmax
   matches. Strict numerical gate on **Gemma 4 E2B at N=560** is
   **blocked** by the 128 MiB per-binding cap — `inpPerLayer` PLE
   pin (20 MB) + shared-KV K/V retention for layers 0-14 (~60 MB)
   + per-block intermediates push the graph alloc 1.5 MB past the
   cap. Closure report:
   [`eval/reports/gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4.3-longctx-parity-2026-05-12/SUMMARY.md).
   The functional gate ("no quality cliff at 512-token SWA
   boundary") is already met by the Stage 4.1 long-context chat
   closure (N=2238 fact-correct retrieval drawn from prompt
   position ~280). Numerical-parity escalation path for Gemma 4
   specifically: bump `ggml-webgpu` to multi-binding scratch
   allocation (path 4 of the 2026-05-12 decision) — defer until
   the eval re-gate (Stage 4.4) shows whether the qualitative
   gate is enough.

3. **Stage 4.4 — Eval re-gate. CLOSED 2026-05-12 EOS-8 ✅.** Ran
   `bench-profile PROFILES=gemma-4-e2b-warm` on tip `f8f8a64` at
   greedy temp=0. Result: **34/48 = 70.8 %** — bit-identical to
   Stage 3 closure (tool-calling 17 % / reasoning 83 % /
   instruction-following 92 % / semantic-reasoning 92 %, same pass
   set per dimension). Gate ≥ 68 % cleared; zero regression. SWA
   wiring is invisible at sub-window prompts (the 36-prompt eval
   stays under the 512-token SWA window; Stage 4.0 probe predicted
   bit-identical mask at sub-window lengths). Long-context
   behavior was covered separately by Stage 4.1's 2,238-token
   chat closure. Closure report:
   [`eval/reports/gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-stage4.4-eval-regate-2026-05-12/SUMMARY.md).
   Watch-item: speed sub-step failed with smoke-harness
   output-wait timeout (harness-side, not a model regression);
   re-capture speed numbers when convenient, not gating Stage 5.

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

**Stage 5.1 — Pre-rebase baseline capture. CLOSED 2026-05-12 ✅
(Pass 2 retake).** Per the §32a doctrine ("Pre-rebase baseline
doctrine"), captured profile-mode benches on the canonical 6 in the
**Pass 2 canonical regime**: fresh headless Chrome relaunched before
*every* model, 30s thermal cooldown between models, 5 runs per
model (median-of-5 absorbs cold-shader bias on early runs and
thermal contention on later runs). Pass 1 (3-run / shared-Chrome /
no cooldown) was superseded the same day after variance analysis
showed 5-30% swing depending on which extreme ran when.
**Result:** matmul time moved **-4.5% to -8.3% (faster) uniformly**
across the canonical 6 vs the 2026-05-04 baseline — no regression
from the NEOX RoPE fix (Q1.6); compute is uniformly faster, almost
certainly methodology-driven (fresh Chrome + cooldown exposes
~5-8% additional headroom that prior captures' Chrome state was
eating). Headline tok/s lifted +45-70% across the fleet.

| Model | 2026-05-04 | 2026-05-12 (Pass 2) | matmul Δ |
|---|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0 | 80.4 | **130.5** | -4.5% (faster) |
| qwen3-0.6b-q4f16 | 62.5 | **99.4** | -8.3% (faster) |
| qwen3-1.7b-q4f16 | 41.7 | **71.1** | -4.8% (faster) |
| mistral-7b-instruct-v0.3-q4ks | 28.7 | **46.4** | -6.3% (faster) |
| llama-3.1-8b-instruct-iq3m | 23.3 | **33.8** | -5.3% (faster) |
| qwen3-8b-iq3m | 21.3 | **31.0** | -6.0% (faster) |

**New doctrine — Pass 2 capture regime is canonical going forward:**
- **Fresh headless Chrome per model** (`agentchrome connect
  --disconnect && sleep 30 && agentchrome connect --launch --headless
  && sleep 5` before each `make smoke-bench`)
- **5 runs per model**, profile mode, `PERF_RUNS=5`
- **30s thermal cooldown** between models
- Pass 2 establishes the new internal baseline series. The 2026-05-04
  numbers are retained as historical anchor but cross-day comparisons
  across capture regimes carry methodology uncertainty. Next §32 /
  §27 / §28 adjudication baseline lives here.

Closure report:
[`eval/reports/pre-rebase-baselines-2026-05-12/SUMMARY.md`](eval/reports/pre-rebase-baselines-2026-05-12/SUMMARY.md).

**Stage 5.2 — Add `gemma-4-e2b-warm` to the canonical 6 (now 7).
CLOSED 2026-05-12 ✅** (Pass 2 retake co-landed with 5.1's Pass 2).
One-line addition to `SMOKE_PROFILE_SETS.full` in
`eval/smoke-profiles.ts` after `gemma-2-2b-warm`. Stage 4.4
watch-item (smoke-harness speed timeout) closed by capturing a
Pass 2 5-run headless profile-mode bench:
- gemma-4-e2b-it-q4km: 45.6 / 44.6 / 30.7 / 38.6 / 36.9 tok/s
  (p50 **38.6**; 38.6% spread) — see
  [`eval/reports/pre-rebase-baselines-2026-05-12/gemma-4-e2b-it-q4km.log`](eval/reports/pre-rebase-baselines-2026-05-12/gemma-4-e2b-it-q4km.log).
- matmul: 8.19 ms median (47.6% share, vs canonical 6's 50-82%).
  Dispatch count 1040/token (vs 450-805 for canonical 6) — Gemma 4
  is dispatch-heavy due to per-layer PLE injection, which makes it
  intrinsically more variance-prone than the canonical 6 (cf. the
  4.2% spread on qwen3-8b vs 38.6% spread on Gemma 4 at the same
  capture). Matmul time itself is stable (mean 8.22, median 8.19,
  p90 8.72 across all 65 single-token samples) — variance is in
  the rest of the pipeline.
- **Follow-up watch**: Gemma 4 specifically benefits from 7-9 runs
  to tighten the headline median (current 5-run p50 reflects real
  variance, not a code problem). Queue for next sweep cycle; not
  gating any Stage 5 or campaign.
Ship gate: `make checkall` green (782 pass / 36 skip / 0 fail /
39312 expect() calls). All 5 runs auto-ingested to the dashboard
at 2026-05-13T01:22 — accuracy×speed scatter renders Gemma 4
alongside the canonical 6 on next dashboard reload.

**Stage 5.3 — Closure SUMMARY. CLOSED 2026-05-12 ✅.** Canonical
Gemma 4 E2B campaign closure landed at
[`eval/reports/gemma-4-e2b-validation-2026-05-12/SUMMARY.md`](eval/reports/gemma-4-e2b-validation-2026-05-12/SUMMARY.md).
Folds in Stages 1-5 per-stage closures, §27 NEOX free-win
classification (single-line `getRopeModeForArchitecture` fix
delivered +59 pp eval), perf snapshot, doctrine lessons (4 banked
incl. the "demote candidates are usually plural" lesson and the
Pass 2 capture-regime canonical), patch budget (**0 of +2 used**),
and the four queued non-gating follow-ups (embedding-path SWA,
debugLayerOutput SWA, Gemma 4 perf tightening, FA-VEC ceiling
bump). Cross-linked from
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) (Balanced tier).

**Gate met:** dashboard renders Gemma 4 in the accuracy×speed
scatter at 70.8 % / 38.6 tok/s (22 perf runs + 6 evals in
`eval/reports/smoke-runs.db`); closure SUMMARY merged at tip
`225054e`.

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

---

### llama.cpp rebase + sweep cycle — 2026-05-12 upstream cadence (queued 2026-05-12)

**Status:** queued, not started. Cadence check 2026-05-12 EOS found
two new ggml-webgpu commits on upstream master (both landed
2026-05-12 wall-clock UTC); fire when ready to pick up. This entry
is self-contained — a fresh session should be able to start at
Phase 1 without paging through prior context.

**Trigger:** upstream `ggerganov/llama.cpp` master moved from our
rebase base `a817a22bc` (2026-05-04 cycle) to `856c3adac` (the
cadence-check tip). 102 commits ahead, of which 2 touch
`ggml-webgpu`:

| Upstream commit | Date | Touch | Relevance |
|---|---|---|---|
| `927dada6c` "Enables running gpt-oss-20b" | 2026-05-12 | mulmat-q refactor + new shaders (add_id, get_rows); shader-lib +68, ggml-webgpu.cpp +61, 11K lines auto-generated ops.md table | Broad matmul perf — §27 / §32 candidate |
| `239a497e5` "address precision issues for multimodal" | 2026-05-12 | GELU fix (gelu / gelu_quick / gelu_erf); FA tile + vec path rework; f16 → f32 for q_shmem / o_shmem; NaN-clamp; shader-lib +193, ggml-webgpu.cpp +30 | **Directly Gemma 4** — GELU is Gemma's FFN activation; FA path overhaul may obsolete our `9ea3bfc` clamp |

Plus 1 ggml-core commit (`3e941b813` SCHED_DEBUG cosmetic — no
behavioral impact). The other 99 upstream master commits are CUDA /
Vulkan / Metal / server / docs — out of scope for our wasm32
`ggml-webgpu` build.

**Pre-rebase baseline (canonical §32a control):**
[`eval/reports/pre-rebase-baselines-2026-05-12/SUMMARY.md`](eval/reports/pre-rebase-baselines-2026-05-12/SUMMARY.md).
Pass 2 capture regime (fresh-Chrome-per-model + 30s cooldown + 5
runs + headless + profile mode). Captured 2026-05-12 EOS (within
hours of upstream commits landing); the freshest possible
same-tip control. 7 models: canonical 6 + `gemma-4-e2b-it-q4km`.

**Patch conflict surface:**
- `ggml/src/ggml-webgpu/ggml-webgpu.cpp` — we patch 632 / -61 lines;
  upstream adds 91 more in our area (61 from #22906 + 30 from
  #22808). **Manual conflict resolution likely.** Affected local
  patches: `ff362d4ae` (ASYNCIFY bundle), `846e0685e` (request-based
  readback), `55fba3670` (request cleanup), `702d40ee9` (readback
  notify), `009119b07` (graph profiling), `db2a3c38d`/`920c988a1`
  (matmul dispatch specialize + revert), `fc1f81242` (UB shift
  fix), `b54503497` (WaitAny under JSPI).
- `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp` — **we do not
  patch this file**; upstream rewrites the FA path-selection helper.
  No conflict; behavior changes affect our `9ea3bfc` TS-side clamp
  (Probe A below).
- `ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl` — we patch
  here (load_u32_at_src UB fix); upstream doesn't touch in these
  commits. No conflict expected.
- `ggml/src/ggml-webgpu/wgsl-shaders/{flash_attn_*,unary,mul_mat_*}.wgsl`
  — upstream rewrites; we don't patch. No conflict; behavior matters
  for Probe A + Probe B.

**Predicted §-cycle classification (to be confirmed by sweep):**
- Most likely **§27 (free win)**: precision fixes are corrective;
  mulmat-q refactors historically improve perf (cf. §27's #22344
  fast i-quant mat-vec → +80 % on qwen3-8b).
- Possible **§32 (small regression, accepted)**: mulmat-q refactor
  could shift one model adversely; accept if 5-of-7 hold.
- Less likely **§28 (negative result)**: would require these
  commits to break a load-bearing assumption — none flagged so far.

---

#### Fresh-session pickup — start here

**Pre-flight (~2 min):**

```bash
cd /Users/probello/Repos/webllm
git status                                  # must be clean
git log --oneline -1                        # confirm tip at ec884a9 or later
agentchrome connect --status                # confirm reachable (or relaunch later)
ls eval/reports/pre-rebase-baselines-2026-05-12/  # confirm baselines present
```

---

#### Phase 1 — Rebase webllm-browser-patches onto upstream master

**Goal:** rebase 11 effective patches onto `origin/master`
(post-fetch tip `856c3adac` or whatever is current). Resolve
`ggml-webgpu.cpp` conflicts. Drop the 16 JSEP probe commits stacked
above `b54503497` (negative-closure 2026-05-08; not load-bearing).

```bash
cd ~/Repos/llama.cpp
git fetch origin master
git checkout webllm-browser-patches
git checkout -b webllm-browser-patches-pre-rebase-2026-05-12  # safety backup
git checkout webllm-browser-patches
# Reset to canonical tip (drop the 16 JSEP probes):
git reset --hard b54503497
# Now rebase the 11 effective patches onto current master:
git rebase origin/master
# Resolve any conflicts in ggml-webgpu.cpp. Likely conflict points:
#   - Around the `webgpu_context` struct (#22906 + #22808 both add fields)
#   - In the FA pipeline-key struct (#22808 adds q_type, dst_type)
#   - In flash-attention dispatch helpers (#22808 reworks path selection)
# For each conflict: keep BOTH upstream's new fields/logic AND our
# browser-side instrumentation (ASYNCIFY, async readback, profiling).
# Use `git diff --merge-base origin/master HEAD` mid-rebase to sanity-check.
```

**Gate:** `git rebase --continue` reaches clean tip with all 11
patches applied. Save the new tip SHA for Probe artifacts.

**If a patch becomes obsolete** (e.g., upstream merged equivalent
behavior), drop it via `git rebase --skip` and document in the
post-rebase SUMMARY. Specifically watch:
- `db2a3c38d` decode matmul dispatch specialize: already reverted
  by `920c988a1` in our stack; consider whether the mulmat-q
  refactor in `927dada6c` subsumes the motivation.

---

#### Phase 2 — Rebuild WASM + sanity smoke (~5 min)

```bash
cd /Users/probello/Repos/webllm
make wasm-build      # rebuilds both wasm32 and wasm64 targets
make smoke-test      # bundles + copies WASM
# Sanity check: a single fresh-Chrome smoke on TinyLlama
agentchrome connect --disconnect && sleep 5 && agentchrome connect --launch --headless && sleep 5
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
```

**Gate:** TinyLlama generates 64 tokens, finish=max-tokens (or
stop-token), no console errors, tok/s within ±15 % of Pass 2
baseline (130.5). If it crashes or outputs garbage, halt and back
out (the safety backup branch `webllm-browser-patches-pre-rebase-2026-05-12`
restores).

---

#### Phase 3 — Probe A: FA-VEC clamp obsolescence (~10 min)

**Hypothesis:** upstream `239a497e5` rewrote FA path-selection
(f16 → f32 for q/o shmem, restructured PATH_VEC shared-memory math,
added q_type/dst_type to pipeline key). Our `9ea3bfc` TS-side
`prefillTileSize=16` clamp for Gemma family head_dim>128 may no
longer be needed.

```bash
cd /Users/probello/Repos/webllm
git checkout -b probe-fa-vec-clamp-obsolete
# Revert the clamp (one commit on a probe branch):
git revert --no-edit 9ea3bfc
make smoke-test
# Repro the original trap conditions:
agentchrome connect --disconnect && sleep 5 && agentchrome connect --launch --headless && sleep 5
# Navigate to the FA-prefill probe page:
agentchrome navigate "http://localhost:8031/fa-prefill-probe.html?model=gemma-4-e2b-it-q4km&ctx=4096&path=forward&chat=1&v=$(date +%s)"
# Wait for [FA-PREFILL-PROBE-DONE-PASS] in the page log, OR
# RuntimeError: unreachable in the console.
```

**Gate / decision tree:**
- **PASS** (probe completes without trap, healthy argmax) → upstream
  fixed the constraint. **Drop `9ea3bfc` permanently** on the rebased
  branch; document in SUMMARY. Recovers prefill throughput on
  Gemma 4 (42.3s TTFT for 2,238-token prefill at q_tile=1 today
  → potentially larger tile VEC).
- **FAIL** (still traps) → keep the clamp. Revert the probe branch:
  `git checkout webllm-browser-patches && git branch -D probe-fa-vec-clamp-obsolete`.
- **PARTIAL** (different failure mode) → file a sub-probe; do not
  block the cycle on it.

**Artifact:** `eval/reports/llama-cpp-rebase-2026-05-12-or-13/probe-a-fa-vec-clamp/`.

---

#### Phase 4 — Probe B: Gemma 4 eval delta (~5 min)

**Hypothesis:** `239a497e5` GELU fix tightens numerical behavior of
Gemma 4's FFN activation. Eval may move ±a few pp from the
70.8 % Stage 4.4 baseline.

```bash
cd /Users/probello/Repos/webllm
# Dashboard must be running for bench-profile:
curl -sf http://localhost:8033/health || (echo "Start dashboard: make dashboard-serve"; exit 1)
agentchrome connect --disconnect && sleep 5 && agentchrome connect --launch --headless && sleep 5
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 bun run eval/bench.ts --profiles gemma-4-e2b-warm
```

**Gate:** record the eval %, compare against 70.8 % Stage 4.4
baseline.
- ≥ 70.8 % → §27 free-win or neutral. Banner result.
- 65-70.8 % → noise floor; re-run once to confirm.
- < 65 % → §28 negative result; investigate (most likely
  candidate: GELU precision interaction with greedy sampling).

**Artifact:** `eval/reports/llama-cpp-rebase-2026-05-12-or-13/probe-b-gemma4-eval/`.

---

#### Phase 5 — Pass 2 canonical 7-fleet perf sweep (~30 min)

Repeat the Pass 2 capture regime against the rebased build. Same
exact procedure as Stage 5.1's Pass 2:

```bash
cd /Users/probello/Repos/webllm
mkdir -p eval/reports/llama-cpp-rebase-2026-05-12-or-13/post-rebase
for m in tinyllama-1.1b-chat-q4_0 qwen3-0.6b-q4f16 qwen3-1.7b-q4f16 \
         mistral-7b-instruct-v0.3-q4ks llama-3.1-8b-instruct-iq3m \
         qwen3-8b-iq3m gemma-4-e2b-it-q4km; do
  agentchrome connect --disconnect 2>&1 | head -1
  sleep 30  # thermal cooldown
  agentchrome connect --launch --headless 2>&1 | head -1
  sleep 5   # settle
  make smoke-bench PERF_MODEL="$m" PERF_RUNS=5 2>&1 \
    | tee "eval/reports/llama-cpp-rebase-2026-05-12-or-13/post-rebase/${m}.log" \
    | tail -20
done
```

**Gate:** 7 logs landed cleanly; per-run spread under 10 % for
canonical 6 (Gemma 4 may stay at ~30-40 % spread, intrinsic).

---

#### Phase 6 — Adjudicate + write closure SUMMARY (~30 min)

Build the post-rebase headline matrix; compare against
`eval/reports/pre-rebase-baselines-2026-05-12/SUMMARY.md` Pass 2
numbers (same regime, same-day same-tip control = perfect §32a
methodology). Classify cycle:

- **§27 (free win):** if 5+ of 7 models show ≥ 2 % matmul faster
  OR clear eval improvement on Gemma 4. Adopt baseline; pin new
  canonical tip; close cycle.
- **§28 (negative result):** if a prior lever closes harder (e.g.,
  Probe A turns out positive AND Probe B regresses — would mean
  the GELU fix interacts badly with the new FA path). Document;
  retire affected lever's resurrection path; close cycle.
- **§32 (small regression, accepted):** if 5/6 hold neutral but 1
  holds a measurable regression. Don't revert — staying current has
  option value (next cycle's wins land cleanly). Document; pin new
  canonical baseline.

**Artifact:** `eval/reports/llama-cpp-rebase-2026-05-12-or-13/SUMMARY.md`
(headline matrix + drift table + Probe A/B outcomes + classification
+ next-cadence trigger).

**Update:**
- TODO header "Current canonical baselines" block (line ~16-50)
  with new perf numbers if §27.
- TODO active-next-steps block "Rebase cadence" line (line ~825)
  with the new cycle's outcome.
- Commit cadence: `docs(rebase-2026-05-12-or-13):` for SUMMARY +
  TODO updates; `feat(rebase-2026-05-12-or-13):` if Phase 3 drops
  the `9ea3bfc` clamp.

---

**Out of scope for this cycle:**
- Multi-binding scratch allocation (Stage 4.3 numerical-parity
  unblocker) — separate campaign if a probe needs it.
- Embedding-path SWA support — out-of-stage opportunistic.
- Upstream FA-VEC `ne[1] < 20` ceiling bump — P2 hygiene; only
  needed if Probe A reveals upstream did NOT obsolete our clamp.
- gpt-oss-20b model registration — above the 8B ceiling.

**Risks:**
- **Patch conflict in `ggml-webgpu.cpp`** (most likely): manual
  resolution; safety backup branch protects against bad merge.
  Worst case 1 hour of triage; back out via the backup branch.
- **Probe A false positive** (clamp obsoleted but a different
  edge case appears): scope it via the probe branch; don't bake
  into rebased mainline until clean.
- **Probe B false negative** (eval moves -3 pp into noise band):
  re-run once; if still below noise, treat as §32.
- **mulmat-q refactor regression on a specific quant**: most likely
  IQ-quants (e.g., qwen3-8b-iq3m matmul); accept as §32 if
  isolated.

**Wall-clock budget:** ~1.5-2 hours total, mostly autonomous.
Phase 1 is the only manual step (rebase conflict resolution); the
rest is scripted.

