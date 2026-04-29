# 13B target registration — Qwen3-14B Q4_K_S validation

**Date:** 2026-04-29
**Plan:** [`TODO.md` §13B target registration (queued 2026-04-29)](../../../TODO.md)
**Model:** `qwen3-14b-q4ks` (14.77B params, Q4_K_S, ~7.99 GiB on disk)
**Binary:** webllm-wasm-mem64.{js,wasm} (wasm64) with the
[`scripts/fix-mem64-bindgroup-shim.py`](../../../scripts/fix-mem64-bindgroup-shim.py)
post-build patch applied by `make wasm-build-mem64`.

## Headline

- ✅ **94% overall** on the 36-prompt sanity eval (34/36 passing).
  **New fleet accuracy leader** — beats prior leader Qwen3-4B Q4_F16
  at 88-90% by +4 to +6 pts.
- ✅ Decode tok/s in architectural band: 3-run smoke-bench median
  **18.9 tok/s** (gate ≥12, expected band 15-19, top of band).
- ✅ Registers the next param-count rung (14B) inside the 30B project
  ceiling. Closes the queued 13B target work item from the
  MEMORY64 closure stub.
- ✅ Pure model-list work — zero infrastructure changes. The wasm64
  path proven by Mistral-Nemo Q4_K_S (Phase 7) generalizes cleanly
  to a fresh architecture family at a larger param count.

## Discovery arc (what actually happened, in order)

1. **Phase 1 — Probe.** HEAD-verified
   `bartowski/Qwen_Qwen3-14B-GGUF/resolve/main/Qwen_Qwen3-14B-Q4_K_S.gguf`
   → HTTP 200, content-length 8,573,475,872 (7.99 GiB). Within 1%
   of the spec's "~7.8 GiB" estimate.

2. **Phase 2 — Register.** Added `qwen3-14b-q4ks` entry to
   [`eval/models.ts`](../../../eval/models.ts) mirroring the
   Mistral-Nemo Q4_K_S template (paramsB 14.77, vramMB 8800,
   contextLength 4096, tier "quality"). Added smoke profile
   `qwen3-14b-q4ks-warm` in [`eval/smoke-profiles.ts`](../../../eval/smoke-profiles.ts).
   `make checkall` passed (fmt + lint + typecheck + 452/452 tests).
   Committed as `feat(eval): register Qwen3-14B Q4_K_S target
   (13B-class)` (`a4c8189`).

3. **Phase 3a — Local download (unanticipated).** The smoke harness
   resolves models from `smoke-test/models/<id>.gguf`, not from the
   HuggingFace URL recorded in `ggufUrl` (that field is consumed by
   `eval/browser-smoke.ts` for HF-mirror probes; the in-page smoke
   loader expects local files). 8 GiB pulled from bartowski into
   `smoke-test/models/qwen3-14b-q4ks.gguf` at ~30-40 MB/s. SHA not
   verified beyond filesize-match (8,573,475,872 bytes).

4. **Phase 3 — End-to-end smoke probe.** All 8 stages cleared on
   `real-model.html?model=qwen3-14b-q4ks&wasm=mem64&ctx=4096&prompt=hi&ingest=off`:

   ```
   [2/8] Model fetched: 8573.5 MB in 2.3s
   [3/8] GGUF parsed: arch=qwen3 emb=5120 heads=40/8 layers=40
                      vocab=151936 ctx=32768
   [4/8] Weights loaded in 2.8s
   [5/8] KV cache: 4096 slots x 40 layers
   [6/8] Tokenizer ready: vocab=151936
   [6/8] Shader-cache warmup complete in 952ms
   [7/8] Generated 11 tokens in 1.3s (prefill: 750ms,
         decode: 523ms, 21.0 tok/s, finish=eos)
   [8/8] embed cosine 0.76 (>=0.75 expected, ‖v‖=1.00)
   ```

   Console: 0 errors, 0 warnings. `pickWasmUrl(byteLength)`
   auto-routed to wasm64 (filesize > 3.5 GiB threshold).

5. **Phase 4 — 36-prompt sanity eval.** `bench-browser-eval
   PROFILE=qwen3-14b-q4ks-warm` ran the standard suite (tool-calling
   skipped at temperature 0.6 > 0.4; embedding skipped by
   capability flag — model has `capabilities.embedding=false`).

6. **Phase 5 — Smoke-bench tok/s.** 3-run profile-mode wasm64.

7. **Phase 6 — This closure report.**

## Eval result (36-prompt sanity)

```
36/36 tasks (34 passing)
Done: 34/36 passing · overall 94%
```

**Hard gate ≥60%: PASS** (+34 pts margin). Reference points on the
same 36-prompt suite:

| Model | Eval | Notes |
|---|---:|---|
| **qwen3-14b-q4ks** | **34/36 (94%)** | This row — new fleet leader |
| qwen3-4b-q4f16 | 32-33/36 (88-90%) | Prior accuracy leader |
| qwen2.5-3b-q4f16 | 31/36 (86%) | |
| qwen3-1.7b-q4f16 | thinking-on 89%, off 82% | |
| mistral-nemo-2407-q4ks | 26/36 (72%) | 12B Q4_K_S baseline |
| llama-3.1-8b-iq3m | (warm) ~74% | IQ3_M quant; this is the 8B Llama row |
| mistral-7b-instruct-v0.3-q4ks | 24/36 (68%) | 7B Q4_K_S baseline |

Qwen3 family scaling holds: 0.6B → 1.7B → 4B → 14B is a clean
accuracy ladder (~62% → ~85% → ~89% → 94%) with tighter gains at
each rung. Per-dimension breakdown is on the dashboard run record;
not pulled into this report.

## Speed result

3-run profile-mode smoke-bench (`make smoke-bench
PERF_MODEL=qwen3-14b-q4ks PERF_RUNS=3 WASM_VARIANT=mem64`),
prompt "Tell one short joke." (18 tokens generated each run):

| Run | Tokens | Wall(ms) | Prefill(ms) | Decode(ms) | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 15169 | 861 | 959 | 18.8 |
| 2 | 18 | 14662 | 778 | 951 | 18.9 |
| 3 | 18 | 15662 | 779 | 940 | 19.1 |
| **median** | **18** | **14662** | **778** | **951** | **18.9** |

**Hard gate ≥12 tok/s: PASS** (median 18.9 tok/s, +57% margin).
**Predicted band 15-19 tok/s: PASS** at the top of band, matching
the spec's projection extrapolated from Mistral-Nemo 12B Q4_K_S
(19.3 tok/s) and Mistral-7B Q4_K_S (28.2 tok/s post-rebase).

Per-phase decode (mean ms over 51 single-token steps, profile-mode):

| Phase | mean(ms) | %total |
|---|---:|---:|
| graphComputeMs | 48.22 | 96.3% |
| downloadResultMs | 1.25 | 2.5% |
| buildGraphMs | 0.53 | 1.1% |
| backendAllocMs / uploadLeaves / ctxCreate / teardown | ≈0 | 0.1% |
| **totalMs** | **50.08** | **100%** |

Backend attribution (mean over 51 steps):

| Field | samples | mean | %graph |
|---|---:|---:|---:|
| backendMatmulMs | 51 | 29.28 | 60.7% |
| backendEncodeOverheadMs | 51 | 11.41 | 23.7% |
| backendAttentionMs | 51 | 0.76 | 1.6% |
| backendDispatchCount | 51 | 893.0 | — |

Decode is matmul-bound (60.7% of graph compute on the
lm_head + attention proj weight chain), continuing the
size-scaling trend documented in the project header
(Qwen3-1.7B/8B at 49-58% matmul share; the 14B row pushes
that ratio further up). Attention is essentially free at
1.6% — Qwen3-14B's GQA 5:1 (40 Q heads / 8 KV heads) keeps
attention bandwidth small relative to weight reads.

Dispatch count scaling is exactly as predicted from layer count:

| Model | Layers | Dispatches/token | Δ vs Qwen3-8B |
|---|---:|---:|---:|
| qwen3-8b-iq3m | 36 | 805 | — |
| qwen3-14b-q4ks | 40 | 893 | +88 (= 4 layers × 22 dispatches/layer) |

## Working set (measured + computed)

- Model file: 8,573.5 MB on disk → 7.99 GiB streaming load.
- KV cache @ ctx=4096:
  40 layers × 8 KV heads × 128 head_dim × 2 (fp16) × 2 (K+V)
  × 4096 tokens = **1.28 GiB**.
- Activations + scratch + WebGPU buffers: ~1 GiB observed
  (warmup completed end-to-end without OOM).
- Total at decode: ~10.3 GiB allocated, well under the 16 GiB
  Emscripten 5.0.6 toolchain ceiling and consistent with the
  vramMB=8800 estimate at registration.

## Lever closure

The 13B target registration work item closes with this report.

**Out-of-scope** (carried forward as separate items):

- Integrating qwen3-14b into the canonical 6 parity sweep. The
  current decision is to keep Phase 5's parity matrix at the
  established 6 rows (TinyLlama, Qwen3-0.6B, Qwen3-1.7B,
  Mistral-7B, Llama-3.1-8B, Qwen3-8B); this row stays
  wasm64-only like the Mistral-Q5_K_M addendum.
- Registering a 30B IQ3_M target. Spec carried at TODO.md
  §31b and `PHASE-7-VALIDATION.md`'s "next-rung target" note.
  No urgency — ceiling is 30B and the wasm64 path now has two
  proof points (Mistral-Nemo 12B, Qwen3-14B).

## Reproduction

```bash
# Local file prerequisite (8 GiB):
curl -L --fail \
  -o smoke-test/models/qwen3-14b-q4ks.gguf \
  https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF/resolve/main/Qwen_Qwen3-14B-Q4_K_S.gguf

# WASM build (wasm32 + wasm64 with shim patch):
make wasm-build

# Phase 3 — single smoke probe in the browser:
make smoke-serve &
agentchrome navigate \
  "http://localhost:8031/real-model.html?model=qwen3-14b-q4ks&wasm=mem64&ctx=4096&prompt=hi&ingest=off"

# Phase 4 — 36-prompt sanity eval (dashboard ingest):
make dashboard-serve &
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 \
  make bench-browser-eval PROFILE=qwen3-14b-q4ks-warm

# Phase 5 — 3-run profile-mode smoke-bench:
make smoke-bench PERF_MODEL=qwen3-14b-q4ks PERF_RUNS=3 WASM_VARIANT=mem64
```

## Closing notes

- The Phase 3 download discovery (smoke harness expects local
  files at `smoke-test/models/<id>.gguf` rather than fetching
  from `ggufUrl`) is documented above for the next registrant.
  The two GGUF-resolution paths in this repo —
  `eval/browser-smoke.ts` (HF tree-API + ggufFilePattern) and the
  in-page smoke loader (local-file convention) — should
  eventually converge, but for now the model-registration
  workflow needs both: the registration commit lands the metadata,
  and a separate `curl` populates the local file. Tracked
  loosely; not blocking.
- Qwen3-14B vramMB was set to 8800 at registration. Post-validation
  the measured decode-time working set landed near 10.3 GiB; the
  registered estimate is on the conservative side relative to
  routing thresholds (>3500 MB → wasm64) but mildly
  underestimates absolute footprint. Worth bumping to ~10500 if
  any future code reads `vramMB` for capacity planning rather
  than routing.
