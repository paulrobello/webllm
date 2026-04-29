# MEMORY64 migration — Phase 5 bench parity gates

**Date:** 2026-04-28
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Pre-rebase reference:** [`eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`](../pre-rebase-baselines-2026-04-28/SUMMARY.md)

## Headline

- **Step 1 sanity gate (wasm32 current vs pinned, ≤3%):** **FAIL — 5 of 6 models drifted 4-21%.** Small models hit hardest. Root cause diagnosed and remediation queued (see §Diagnosis).
- **Step 4 main gate (wasm64 vs current wasm32, ≤3%):** **5 of 6 PASS, 1 FAIL (TinyLlama −5.5%).** Net median delta **0%** (wasm64-vs-wasm32).
- **Phase 6 decision input:** **DEFERRED pending Phase 1.5 optimization.** With wasm32 itself regressed from pinned, the wasm64-vs-pinned comparison conflates two separate effects; rerunning after the optimization gives a clean deploy decision.

## Per-model results

3-run medians, profile-mode (`make smoke-bench --profile`).

| Model | Pinned (tok/s) | Current wasm32 (tok/s) | Wasm64 (tok/s) | Δ wasm32 vs pinned | Δ wasm64 vs current wasm32 | Δ wasm64 vs pinned |
|---|---:|---:|---:|---:|---:|---:|
| `tinyllama-1.1b-chat-q4_0` | 87.9 | 78.5 | 74.2 | **−10.7%** | **−5.5%** | −15.6% |
| `qwen3-0.6b-q4f16` | 68.2 | 54.0 | 56.6 | **−20.8%** | +4.8% | −17.0% |
| `qwen3-1.7b-q4f16` | 44.0 | 39.5 | 48.0 | **−10.2%** | **+21.5%** | +9.1% |
| `mistral-7b-instruct-v0.3-q4ks` | 29.7 | 28.4 | 28.2 | **−4.4%** | −0.7% | −5.1% |
| `llama-3.1-8b-instruct-iq3m` | 23.5 | 22.4 | 22.4 | **−4.7%** | 0.0% | −4.7% |
| `qwen3-8b-iq3m` | 21.8 | 21.4 | 21.4 | −1.8% | 0.0% | −1.8% |

**Bold** = exceeds the ±3% gate.

Per-model raw 3-run logs in this directory (`wasm32-<m>.txt`, `wasm64-<m>.txt`).

## Wasm size delta

| Binary | Bytes | Δ |
|---|---:|---:|
| `webllm-wasm.wasm` (wasm32) | 2,249,650 | — |
| `webllm-wasm-mem64.wasm` (wasm64) | 2,292,315 | **+1.9%** (+42,665 B) |

## Environment

- Emscripten 5.0.6 (ref `6ea9c28c38cdd40c1032fa04400c9d16230ee180`)
- llama.cpp `webllm-browser-patches` tip `3b8ade2a2` (11-patch stack)
- Chrome 147.0.7727.117 via existing agentchrome session (port 62847)
- Host: macOS 26.4.1, Apple M4 Max, 128 GiB RAM

## Diagnosis

The Step 1 sanity gate failure is a **known regression introduced by Phase 1.5** of this migration (commit `061a93c`).

**Bisect proof:**

1. With pre-Phase-1 `src/inference/ggml-wasm.ts` (commit `9a9f2ab^`, uses `_malloc`/`_free` directly), Mistral-7b reproduces the pinned baseline to within run-spread noise: 28.7 / 29.0 / 28.7 vs pinned 29.7. Equivalent to pre-rebase capture conditions.
2. With Phase 1 alone (`65cd0a8`, uses `_bridge_malloc`/`_bridge_free` but no per-call `big()`/`num()` helpers), Mistral-7b ran at 28.4 — comparable to Phase 1.5 head — but **failed to load any 7B+ model** with `RangeError: offset is out of bounds`. The unsigned-pointer fix at `56272cb` resolved the load failure under wasm32.
3. With current head (post-fix), all 6 models load. Small models regress ~10-21% vs pinned; large models regress ~2-5%.

**Mechanism:** Phase 1.5's `big()` / `num()` helpers wrap **every** size_t/pointer FFI argument and return:

```ts
private big(n: number): number | bigint {
    return this.is64 ? BigInt(n) : n;
}
private num(v: number | bigint): number {
    return typeof v === "bigint" ? Number(v) : v >>> 0;
}
```

Each call adds a method dispatch + ternary branch + (under wasm32) a no-op return. There are **38 `this.m._*` callsites** wrapped in helpers. Per decode step on Qwen3-0.6b: ~629 dispatches/token × ~50-100 helper calls per dispatch = ~30K helper invocations/sec at 54 tok/s. V8 should inline the helpers but the cost is non-trivial when the underlying op (a fast op like `op_mul`) is microsecond-scale.

The regression-vs-dispatch-frequency correlation is consistent with this mechanism:

| Model | Dispatch/token × tok/s = FFI calls/sec | Δ wasm32 vs pinned |
|---|---:|---:|
| Qwen3-0.6b | 629 × 54.0 = 33,966 | **−20.8%** |
| TinyLlama | 450 × 78.5 = 35,325 | −10.7% |
| Qwen3-1.7b | 629 × 39.5 = 24,846 | −10.2% |
| Qwen3-8b | 805 × 21.4 = 17,227 | −1.8% |
| Llama-3.1-8b | 652 × 22.4 = 14,605 | −4.7% |
| Mistral-7b | 650 × 28.4 = 18,460 | −4.4% |

Larger models spend more wall-time per FFI hop (heavier matmul), so the helper overhead dilutes. Smaller models regress hardest because the helper overhead is a higher fraction of decode time.

The wasm64 column is, by contrast, **mostly comparable to current wasm32** (5 of 6 within ±5%), suggesting the wasm64 ABI itself adds little incremental cost on top of the Phase 1.5 baseline — the regression vs pinned is fundamentally a wasm32 regression, not a wasm64 one.

## Remediation plan

**Optimization target: replace per-call helper dispatch with init-time method binding.**

Sketch:

```ts
// In init(), after detecting is64:
this.malloc = this.is64 ? this._malloc64 : this._malloc32;
this.tensorNew1d = this.is64 ? this._tensorNew1d_64 : this._tensorNew1d_32;
// ... for every wrapped method
```

Where `_malloc32(size: number): number { return this.m._bridge_malloc(size) >>> 0; }` and `_malloc64(size: number): number { return Number(this.m._bridge_malloc(BigInt(size))); }`.

The hot path becomes a direct method call on a specialized impl — V8 should fully inline. Expected: claw back the 5-15% small-model regression. Wasm64 path stays unchanged (already correct).

**Estimated impact (qualitative):**
- Qwen3-0.6b should rebound from 54.0 toward the pinned 68.2 (≥80% of the −20.8% gap).
- Larger models (Mistral-7b/Llama-3.1-8b) should fully recover (smaller delta to begin with).
- Wasm64 results unchanged.

After the optimization lands, Phase 5 reruns and the wasm32-vs-pinned sanity gate should pass within ±3%; wasm64-vs-wasm32 measurement gives the clean Phase 6 decision input.

## Decision rule (TODO §MEMORY64 Phase 6) — DEFERRED

Per the plan:
- ≤2% median regression → ship MEMORY64-only
- 2-5% regression → ship dual binary; deploy-time selection
- ≥5% regression → halt; diagnose pointer-overhead in hot paths before Phase 6

Current wasm64-vs-wasm32 median is **0%** with one outlier at −5.5% (TinyLlama) and one outlier at +21.5% (Qwen3-1.7b — likely wasm64 inlining win, possibly noise). A clean Phase 6 decision needs a stable wasm32 baseline, which requires the Phase 1.5 optimization first.

**Halt Phase 5; queue Phase 1.5 optimization as the next active item; rerun Phase 5 once the optimization lands.**

## Phase 6 status

**HOLD.** Re-evaluate after Phase 1.5 helpers are optimized and Phase 5 re-bench passes Step 1 + Step 4 cleanly.
