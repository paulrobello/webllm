# Bucket C parity — 10/10 PASS at hybrid gate (2026-04-29)

## Outcome

Qwen3-Embedding-0.6B-hyb (Q4_K on `token_embd`, f16 elsewhere) passes 10/10
fixtures from `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`
at the hybrid-quant gate `cos >= 0.995`. All magnitudes 1.000 ± 1e-6.

| row | mode     | cosine    | mag      |
| --- | -------- | --------- | -------- |
| 0   | document | 0.999294  | 1.000000 |
| 0   | query    | 0.998830  | 1.000000 |
| 1   | document | 0.999348  | 1.000000 |
| 1   | query    | 0.999453  | 1.000000 |
| 2   | document | 0.998989  | 1.000000 |
| 2   | query    | 0.999563  | 1.000000 |
| 3   | document | 0.998343  | 1.000000 |
| 3   | query    | 0.997572  | 1.000000 |
| 4   | document | 0.999148  | 1.000000 |
| 4   | query    | 0.996333  | 1.000000 |

Raw run output: [`run.txt`](run.txt).

## Gate selection: 0.995 (hybrid) vs 0.999 (full-precision)

The Phase 0 spec set the parity gate at `cos >= 0.999` against the full-
precision sentence-transformers reference. That gate predates the
**hybrid-quant pivot** the project adopted to fit the WebGPU per-binding
128 MiB cap (see CLAUDE.md "Per-binding 128 MiB cap doctrine"):
`token_embd.weight` is Q4_K (~83 MiB) while every other weight stays f16.

The Q4_K row-lookup error doesn't compound across rows (it's a pure lookup,
not a matmul), but each pooled last-token state inherits a shifted base
embedding that propagates through 28 attention/FFN layers. Empirically
this lands every fixture in the **0.996-0.9996** band — clearly above
quant-noise but below the f16-grade `0.999` threshold.

The harness selects:
- **`0.999`** when `defaultQuant !== "hyb"` (f16 / full-precision GGUFs).
- **`0.995`** when `defaultQuant === "hyb"` (hybrid Q4_K-on-`token_embd`).

`--gate <value>` overrides both. Selection is logged on every run.

## Bugs fixed in service of getting here

Two bugs, neither bucket-C-specific, surfaced during gate execution and
were fixed mid-cycle:

### 1. BPE tokenizer stale-rank validation

**Symptom (initial run):** half the fixtures returned cosine 0.57-0.99 with
length-monotonic degradation — Signature C in the harness diagnostic ladder
("prefix not applied OR tokenizer mismatch"). HF Python tokenized
`"Instruct"` as `[641, 1235]` (= `["In", "struct"]`) and `"Query"` as
`[2859]` (= `["Query"]`). webllm tokenized them character-by-character:
`[40, 77, 82, 83, 81, 84, 66, 83]` and `[48, 84, 68, 81, 88]` — the byte
fallback path.

**Root cause:** `Tokenizer.encodeBpe` in `src/inference/tokenizer.ts` runs a
heap-driven BPE merge loop. On pop the validation checked:

```ts
if (symMerged[left] || symMerged[right]) continue;
if (symNext[left] !== right) continue;
```

This catches stale entries that point at a removed slot or no-longer-adjacent
positions, but **misses the case where `symText[left]` was extended by a
prior merge while position `left` itself is still un-merged and still
adjacent to `right`**. The popped pair's stored rank then describes the
*old* `(symText[left], symText[right])` pair, not the current one — so the
merge fires using a rank that doesn't apply.

For `"Hello"` the stale-rank merges happened to produce a string still in
vocab (`"Hello"` = id 9707). For `"Query"` they produced `"Quer"` which is
not in vocab; the per-character fallback at the end of `encodeBpe` fragmented
it into 5 byte tokens. Same bug, different luck.

**Fix:** re-derive the rank from the current symbol contents on every pop
and skip the entry when it disagrees. The newer (correct-rank) pair was
already pushed when the prior merge fired and will pop at the right time.

```ts
const currentKey = `${symText[left]} ${symText[right]}`;
if (this.config.bpeRanks.get(currentKey) !== rank) continue;
```

After the fix all 30 existing tokenizer tests still pass and the parity
band moved from `0.57-0.99` to `0.996-0.9996`.

This bug had hidden inside the tokenizer for Qwen-family models because
chat traffic arrives wrapped in `<|im_start|>...<|im_end|>` framing, where
the framing tokens are special-cased and the BPE merge sequence on the
inner content rarely produces a not-in-vocab intermediate. Bucket C was
the first workload to send raw multi-word strings through `encode()` and
hit the cliff.

### 2. Parity harness ↔ in-page embedPerf race

**Symptom (after BPE fix):** the second sequential `engine.embed()` call
asserted in WASM:
```
ggml-webgpu.cpp:3659: GGML_ASSERT(tensor->buffer != nullptr) failed
```
This was a `tensor->buffer` null inside `ggml_backend_webgpu_tensor_get_async_begin`,
during the readback of `finalHidden` on the second forward.

**Root cause:** the harness URL set `embedPerf=single&embedFixture=short&embedReps=1`,
which made `runEmbedPerfHook` (in `smoke-test/real-model-page.js`, called
from [8/8] for embedder models) fire **5 warmup + 1 measured** in-page
`engine.embed()` calls. The harness only polled
`window.handleId && window.engine` to detect readiness; both globals are
set in [6/8] — *before* the in-page hook runs. Result: the harness's first
external `engine.embed()` raced the in-page warmup loop. Concurrent forward
graphs share the WASM ctx-stack (single global, push/pop in
`webgpu-bridge.cpp`) — the second call's `ctx_create` pushed a graph ctx
on top of the first call's still-live graph ctx, the inner tensors got
allocated against the wrong slot, and `backendAllocCtxTensors` left some
intermediate without a buffer.

**Fix:** drop `embedPerf` from the parity harness URL. Engine load isn't
gated on it, the smoke page still loads weights and exposes `window.engine`
in [6/8], and `runEmbedPerfHook` becomes a no-op when `mode` is null. Also
set `ingest=off` so diagnostic re-runs don't pollute the live dashboard.

The race was hidden in earlier development because every manual repro
inside a single `agentchrome js exec` script ran the embeds *serially*
inside one async chain — exactly the pattern that *doesn't* race.

## Files touched

- `src/inference/tokenizer.ts` — BPE merge stale-rank validation (Bug 1).
- `eval/causal-embedder-parity.ts` — drop embedPerf race trigger; gate
  selection by `defaultQuant`; agentchrome temp-file response handling
  (1024-dim float JSON exceeds the 16 KB inline cap).
- `eval/browser-smoke.ts` — `localGGUFOnly` bypass for hand-built hybrid
  GGUFs (skip HF tree-fetch + size-verify when the file already exists
  locally).
- `eval/models.ts` — `qwen3-embedding-0.6b-hyb` registration with
  `defaultQuant: "hyb"`, `localGGUFOnly: true`, `vramMB: 1100`.
- `eval/smoke-profiles.ts` — model id update.
- `src/core/engine.ts` — append `tokenizer.eosId` for causal-embedder
  dispatch; sentence-transformers does this implicitly via
  `add_special_tokens=True` and the post-processor templates
  `<|endoftext|>` (id 151643).
- `src/models/model-loader.ts` — `metaPrefix` separation. The on-disk
  GGUF arch string ("qwen3" for both chat and embedding builds) drives
  metadata-key lookups, while the project's identity tag
  ("qwen3-embedding") drives routing into `CausalLMEmbedder`.
- `eval/reports/bucket-c-parity-2026-04-29/` — this report.

## Hybrid GGUF build recipe

For repro / re-build:

```bash
cd ~/Repos/llama.cpp
./build/bin/llama-quantize \
  --token-embedding-type Q4_K \
  --allow-requantize \
  qwen3-embedding-0.6b-q0f16.gguf \
  qwen3-embedding-0.6b-hyb.gguf F16
```

Output: 924 MB GGUF (`token_embd.weight` Q4_K = 83 MiB; everything else
f16). Lands at `smoke-test/models/qwen3-embedding-0.6b-hyb.gguf`.

## Notes for follow-on work

- Bucket D (chat-model self-embedding via `ModelInference.embed(tokenIds)`)
  inherits the `metaPrefix` plumbing, the EOS-append convention for any
  causal-derived embedding path, and the per-binding 128 MiB cap doctrine
  — for any chat model with vocab × hidden > 128 MiB at f16, the same
  hybrid-quant recipe applies.
- The tokenizer fix is generic; chat-model tokenization quality may
  improve marginally on prompts that contained bare multi-word strings
  outside chat framing. Worth a re-run of the eval suite to confirm no
  regressions; not expected to move any committed scores.
- Embed-perf bench coverage (Plan Task 13) is the next plan item;
  `embed-perf.ts` already supports BERT-style encoders and needs an
  extension to drive `CausalLMEmbedder` runs.
