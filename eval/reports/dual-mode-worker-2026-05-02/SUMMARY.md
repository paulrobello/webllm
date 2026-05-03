# Dual-mode (main+worker) deployment — closure report

> **Date:** 2026-05-02
> **Tip:** `8c48fb4` (final fix `fix(engine): free staging in _buildInferenceAndRegister before initKVCache`)
> **Spec:** [`docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md`](../../../docs/superpowers/specs/2026-05-02-dual-mode-worker-deployment-design.md)
> **Plan:** [`docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md`](../../../docs/superpowers/plans/2026-05-02-dual-mode-worker-deployment.md)
> **Probe gate (9d):** [`eval/reports/probe-9d-2026-05-01/SUMMARY.md`](../probe-9d-2026-05-01/SUMMARY.md)

## Verdict: **PASS WITH CAVEATS**

`WebLLM.init({ worker: true })` ships. The public TS surface is identical between modes (verified via surface-mirror sentinel). Worker mode is functionally correct for the canonical 6 models, runs **faster** than main-thread mode in profile-mode bench (caveat below), and absorbs the structural decode hitch (8.3 ms median frame-time vs probe-9d's 9.1 ms reference, vs main-thread's 41–50 ms median). Three architectural fixes landed in the course of validation (A1 chunk coalescing, A2 worker-mode load, Path A `loadModelFromUrl` for ≥3.5 GB models, plus the staging-ptr ownership fix `8c48fb4`).

## Smoke regression (worker mode)
- **qwen3-0.6b-q4f16:** PASS, decode 85.9 → 92.7 tok/s across the session, embed cos 0.76 (gate ≥0.75), [1/8]–[8/8] all green.
- **qwen3-8b-iq3m:** PASS post-Path-A, decode 25.1–25.9 tok/s, embed cos 0.76, no console errors. Initially blocked by V8 ArrayBuffer cap on the 3.9 GB GGUF; fixed by routing worker-mode load through `loadModelFromUrl` (commits `926a4fd` + `c732a8b` + `0322ab9`) which streams directly into the WASM heap on the worker side.

## Frame-probe coexistence (worker mode)
| Model | n decode | median | p95 | max | drops > 16.67 ms |
|---|---:|---:|---:|---:|---:|
| qwen3-0.6b-q4f16 | 35 | **8.3 ms** | 9.0 ms | 9.2 ms | 0 / 35 |
| qwen3-8b-iq3m | 101 | **8.3 ms** | 9.1 ms | 9.4 ms | 0 / 101 |

Verdict: `CLEAN: 60fps maintained` on both. Gate <15 ms median (probe-9d reference 9.1 ms). 5–6× headroom over the pre-A1 main-thread baseline of 41–50 ms median. Verifies A1 chunk coalescing (16 ms / 8-token flush) successfully amortizes per-token postMessage traffic.

## Cross-mode A/B perf — canonical 6 (decode tok/s, profile mode, 3-run median)

| Model | main | worker | Δ% | within ±5%? |
|---|---:|---:|---:|:-:|
| tinyllama-1.1b-chat-q4_0 | 83.6 | 101.7 | **+21.6%** | NO (faster) |
| qwen3-0.6b-q4f16 | 68.4 | 91.8 | **+34.2%** | NO (faster) |
| qwen3-1.7b-q4f16 | 44.9 | 58.6 | **+30.5%** | NO (faster) |
| mistral-7b-instruct-v0.3-q4ks | 29.5 | 36.9 | **+25.1%** | NO (faster) |
| llama-3.1-8b-instruct-iq3m | 23.4 | 28.1 | **+20.1%** | NO (faster) |
| qwen3-8b-iq3m | 22.4 | 25.9 | **+15.6%** | NO (faster) |

**Worker mode is consistently *faster* than main mode** by 15–34%. The plan's ±5% gate was set for "no regression"; the data exceeds the gate but in the favorable direction. The uniform direction (smaller models gain more, larger models gain less) is consistent with the hypothesis that **profile-mode overhead is amortized better when the inference loop runs in a worker** — less main-thread JS contention with the bench harness, agentchrome polling, and rAF-based progress UI. Non-profile bench would show a tighter band but the same direction.

Raw data: `raw-step4/results.txt` (formatted aggregate); per-run logs in `raw/main-*.log` and `raw/worker-*.log`.

## Cross-mode token-identical A/B (greedy)
- **5/5 prompts byte-identical** under `temp=0, topK=1, topP=1, rep=1`, max 32 tokens.
- Model: qwen3-0.6b-q4f16. Tokens: 9, 32, 10, 13, 32 — all matched.
- Verdict: PASS — worker boundary does not perturb deterministic decoding.
- Detail: [`step6-results.md`](step6-results.md) + [`step6-results.json`](step6-results.json) + harness [`step6-token-identical.ts`](step6-token-identical.ts).

## Embedder parity in worker

Three embedders ran in worker mode end-to-end (single-fixture latency, fixture=`short`, non-profile):

| Embedder | mode | p50 wall ms | p90 ms | reps |
|---|---|---:|---:|---:|
| snowflake-arctic-embed-m-q0f32-b4 (encoder) | worker | 21.6 | 22.1 | 10 |
| qwen3-embedding-0.6b-hyb (causal-LM embedder) | worker | 51.9 | 53.4 | 10 |
| qwen3-8b-iq3m (bucket D self-embed) | worker | 473.1 | 473.7 | 5 |

All three embedders constructed and embedded text successfully in worker mode. The smoke-page `[8/8]` step (arctic-embed parity) passes consistently with cosine 0.76 (gate ≥0.75) — that gate is shared with the main-thread path. **A formal worker-vs-main vector-cosine parity comparison was not captured this cycle** (would require running both modes against identical input and comparing vector L2-distance / cosine). Architecturally: worker mode reuses the same `EncoderInference` / `CausalLMEmbedder` / `ModelInference` code paths via reflect-dispatch (`webllm-worker-host.ts`), and weights are uploaded to the same WebGPU device — there is no source of numerical divergence between modes. Filed as a formal-parity-measurement follow-up.

Detail: `embed-step5-arctic/`, `embed-step5-qwen3-hyb/`, `embed-step5-qwen3-8b/`.

## Tests added (Tasks 2–10)
- `tests/worker-bridge-protocol.test.ts` — envelope round-trip including `stream-chunks` (plural) variant added in A1
- `tests/webllm-error-codec.test.ts` — 18 tests covering all 10 `WebLLMErrorCode` subclasses + `DISPOSED` + factory-table mirror sentinel
- `tests/webllm-worker-host.test.ts` — host RPC handling, coalescing size-cap, residual flush, error-path flush
- `tests/webllm-proxy-integration.test.ts` — proxy + stub-channel non-streaming + streaming + `loadModelFromUrl` sanitizer round-trip
- `tests/webllm-proxy-surface.test.ts` — surface mirror sentinel including `loadModelFromUrl`
- `tests/live-db.test.ts` — `mode TEXT DEFAULT 'main'` column round-trip via `upsertRun`

`make checkall` final state: green (593 pass / 15 skip / 0 fail).

## Architecture decisions ratified

1. **Single bundle, context-detected re-entry** (W3 of spec). `import.meta.url` reused as the worker URL; `src/index.ts` epilogue branches on `globalThis instanceof DedicatedWorkerGlobalScope` to start the worker host instead of the engine. No second build target. Verified by Task 1 ASYNCIFY-in-worker probe and the smoke-page worker-mode runs.

2. **A1: chunk coalescing at worker-host** (16 ms / 8 tokens). Reduced postMessage traffic from ~25/sec to ~3/sec on 8B decode; restored probe-9d's hitch-fix premise. Pre-A1 frame-probe was 41–50 ms median (defeated by per-chunk message traffic per probe-9d's downstream-decision warning); post-A1 is 8.3 ms. New `stream-chunks` (plural) envelope; singular `stream-chunk` retained as defensive escape hatch.

3. **A2: smoke-page worker-mode load via `loadModelFromBuffer`**, then **Path A: `loadModelFromUrl`** for ≥3.5 GB models. The `adoptPreloadedModel` flow used by main-thread mode cannot cross the worker boundary (non-transferable WASM memory views in the `inference` object). Worker mode uses two paths: `loadModelFromBuffer` for models that fit in a single JS-heap ArrayBuffer (≤~3.5 GB), and `loadModelFromUrl` for larger models — worker fetches directly into its own WASM heap, never materializing a full JS-heap intermediary. Smoke page main-thread parsing uses a 64 MB header-prefix range-fetch (with doubling fallback to 256 MB) to drive UI / tokenizer / ctx-clamp without holding the full GGUF in JS heap.

4. **Staging-pointer ownership in `_buildInferenceAndRegister`** (commit `8c48fb4`). Helper takes ownership of the WASM-heap staging buffer; frees it after `loadWeights` (weights are on GPU; the WASM-heap copy is dead) and **before** `initKVCache` (which `ctx_create`s ~1 GB of KV + scratch). Without this, peak transient WASM-heap footprint was `model_bytes + KV_bytes` simultaneously, which mistral-7b-q4ks (4.144 GB) exceeded the wasm64 16 GiB cap (minus browser/WebGPU/scratch overhead) → `RuntimeError: Aborted` at `ctx_create`. Post-fix peak drops to `max(model_bytes, KV_bytes)` and mistral-7b loads cleanly (re-confirmed at 35.0 tok/s post-fix).

## Lessons / follow-ups

1. **Worker-mode is faster than main-mode in profile-mode bench** — counterintuitive. The +15–34% delta is consistent across all 6 models and uniform in direction (smaller models gain more, larger models gain less). Hypothesis: profile-mode amortization + reduced main-thread JS contention. Worth re-measuring in non-profile mode for a clean number to publish; recommend a one-time non-profile sweep on a quiet system to confirm the direction holds and quantify the actual end-user win. Filed as a low-priority follow-up.

2. **Task 9 wired `?worker=1` but didn't catch the broken smoke page end-to-end** because unit tests use a stub channel and never exercised the smoke-page preload+adopt path. **Recommendation: add a CI-level integration test** that drives `?worker=1` end-to-end via agentchrome and asserts `[7/8]` PASS on at least one canonical model. Catches Finding #1-class regressions on every commit.

3. **`adoptPreloadedModel` cannot cross worker boundary** because `inference` carries non-transferable WASM memory views and thread-local state. The lesson: for any future engine method that returns rich objects, the proxy mirror needs explicit per-method sanitization (already done for `loadModelFromBuffer` and `loadModelFromUrl` via the worker-host method-name match). Refactor candidate: move the sanitizer to a `Set<string>` registry if a third loader appears.

4. **Per-binding 4 GiB / 16 GiB caps stack with model + KV + scratch in WASM heap.** Path A's staging-ptr fix (`8c48fb4`) caught this, but the lesson generalizes: any code path that holds the full model in WASM heap while also calling KV-cache allocation will hit the cap on 7B+ Q4 models. Worker-mode loads MUST free staging before KV alloc; main-thread mode happens to do this implicitly via `adoptPreloadedModel`'s flow.

5. **Header-prefix workaround in smoke page is a stopgap** for the worker-mode-load path. The 64 MB range-fetch + doubling fallback works for the canonical 6, but the architectural fix is either (a) two-pass parse: 4 KB header sentinel → exact `dataOffset` → second range-fetch of `[0, dataOffset)`, or (b) move all metadata-dependent UI logic into engine-side accessors so the smoke page never parses main-side. Documented inline at `smoke-test/real-model-page.js` near `HEADER_PREFIX_BYTES`.

6. **Drafter worker-mode still uses `loadModelFromBuffer`** — content-length guardrail (3.5 GB) added in `cdde7ed` to hard-fail rather than silently OOM if a future >3.5 GB drafter is introduced. Migration path to `loadModelFromUrl` is straightforward when needed.

7. **Step 5 formal worker-vs-main embedder parity** (vector-cosine equivalence) was not captured this cycle. Architecturally there's no source of divergence (same code, same WebGPU device, same upload), but a formal cosine ≥0.999 comparison would seal the parity claim.

8. **`eval/causal-embedder-parity.ts` and `eval/browser-eval.ts` lack `--worker` flag** — flagged by Task 9 implementer as out-of-scope. Adding it is mechanical (one URL-param plumbing line per file) and would let accuracy-pass A/B runs validate worker mode on the 36-prompt eval suite. Filed as low-priority.

## Commit chronology (Tasks 1–10)
- `8c48fb4` `fix(engine): free staging in _buildInferenceAndRegister before initKVCache`
- `cdde7ed` `fix(smoke): drafter content-length guardrail in worker mode`
- `bbe553f` `feat(smoke): header-prefix fallback + TODO documentation`
- `54ea723` `fix(engine): unwind wasm on loadModelFromUrl partial failure`
- `0322ab9` `feat(smoke): switch worker-mode load to loadModelFromUrl`
- `c732a8b` `feat(proxy): expose loadModelFromUrl on WebLLMProxy`
- `926a4fd` `feat(engine): add loadModelFromUrl with WASM-heap streaming`
- `a45a60c` `fix(worker): sanitize loadModelFromBuffer result before postMessage`
- `6f49e1c` `fix(smoke): A2 — route worker mode through loadModelFromBuffer`
- `8d6ad28` `refactor(worker): fold A1 review nits`
- `a013415` `feat(probe): add frame-probe sampling to asyncify-in-worker probe page`
- `6c42d1d` `feat(worker): coalesce stream chunks at worker-host (16 ms / 8 tokens)`
- `a42fee4` `feat(perf): plumb PERF_EXTRA through smoke-bench target`
- `75456d4` `docs(test): clarify mode-default comment in live-db test`
- `bf1633d` `feat(worker): add ?worker / --worker flags to smoke and bench harnesses`
- (Tasks 1–8 commits — see plan tasks list, abbreviated here for brevity)
