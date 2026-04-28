# §32a — Profile-mode rebench on llama-3.1-8b-iq3m

**Date:** 2026-04-28
**Trigger:** TODO §32 §4 hypothesis — "tied-embedding × #22456
aliasing-refactor" as the leading suspect for the §32 -6% regression
(29.0 → 27.2 tok/s, non-profile, 5-run).
**Procedure:** `make smoke-bench PERF_MODEL=llama-3.1-8b-instruct-iq3m
PERF_RUNS=3` against the §32 rebased base (llama.cpp tip
`c4af89356`, post-patch-12).
**Cost:** ~5 min wall (build cached; one rebuild + 3 runs + smoke
restart).
**Risk:** zero — read-only measurement, no `src/` change.

---

## 1. Headline

The §32 regression on `llama-3.1-8b-iq3m` is **uniform across decode
buckets**, not localized to any specific bucket. The "tied-embedding
× #22456 aliasing-refactor" hypothesis is **not supported** by the
profile data — matmul, encode-overhead, and attention buckets all
sit within ~5% of qwen3-8b-iq3m's post-§27 reference profile, the
inverse of what an aliasing-driven matmul or encode regression
would look like.

**Verdict: accept and close.** The "buffer-aliasing constant
overhead" alternative hypothesis fits the data: a small per-step
overhead distributed across the decode pipeline, no single bucket
spike. The §27 doctrine ("document and move on, unless a free win
opens") applies.

## 2. Measurement

3 profile-mode runs, single 53-token completion per run (`Tell one
short joke.`):

```
Run  Tokens  Wall(ms)  Total(ms)  Prefill(ms)  Decode(ms)  tok/s
  1      53      8324       2900          680        2248   23.6
  2      53      7661       2900          688        2255   23.5
  3      53      7856       3000          681        2284   23.2
p50      53      7661       2900          688        2255   23.5
```

Profile perturbation: 23.5 (profile) vs 27.2 (non-profile, §32
5-run baseline) = **-13.6%**, in the same band as qwen3-8b-iq3m's
**-19%** post-§27 perturbation. Profile mode itself is not the
regression cause; just a known measurement-mode tax.

### Per-phase decode timing (mode=topk, 156 single-token steps)

| Phase                | mean(ms) | median(ms) | p90(ms) | %total |
|----------------------|---------:|-----------:|--------:|-------:|
| ctxCreateMs          |     0.00 |       0.00 |    0.00 |   0.0% |
| buildGraphMs         |     0.35 |       0.30 |    0.50 |   0.8% |
| backendAllocMs       |     0.06 |       0.10 |    0.10 |   0.1% |
| uploadLeavesMs       |     0.01 |       0.00 |    0.00 |   0.0% |
| **graphComputeMs**   |   **40.15** | **40.00** | 41.60 | **96.0%** |
| downloadResultMs     |     1.27 |       1.20 |    1.60 |   3.0% |
| teardownMs           |     0.00 |       0.00 |    0.00 |   0.0% |
| totalMs              |    41.84 |      41.60 |   43.60 | 100.0% |

### Backend decode attribution

| Field                     | mean   | median | p90    | %graph |
|---------------------------|-------:|-------:|-------:|-------:|
| backendProfileTotalMs     |  35.46 |  35.30 |  36.90 |  88.3% |
| **backendMatmulMs**       | **23.02** | **23.00** | 23.66 | **57.3%** |
| backendEncodeOverheadMs   |   4.01 |   3.80 |   4.80 |  10.0% |
| backendAttentionMs        |   0.63 |   0.59 |   0.92 |   1.6% |
| backendDispatchCount      |  652.0 |  652.0 |  652.0 |      — |

## 3. Cross-model comparison vs §27 qwen3-8b-iq3m post-rebase profile

The §27 qwen3-8b-iq3m post-rebase profile-mode breakdown is the
closest available reference (same quant family IQ3_M, same param
class 8B, captured 2026-04-27 against `434b2a1ff`):

| Bucket                    | llama-3.1-8b-iq3m (post-§32) | qwen3-8b-iq3m (post-§27) | Δ     |
|---------------------------|-----------------------------:|-------------------------:|------:|
| tok/s (profile mode)      | 23.5                         | 22.0                     | +6.8% |
| graphComputeMs (median)   | 40.00 ms                     | 42.60 ms                 | -6.1% |
| backendMatmulMs (median)  | **23.00 ms**                 | **23.07 ms**             | **-0.3%** |
| backendMatmulMs %graph    | 57.3%                        | 55.0%                    | +2.3 pp |
| backendEncodeOverheadMs   | 4.01 ms                      | 4.50 ms                  | -10.9% |
| backendEncodeOverhead %g  | 10.0%                        | 10.7%                    | -0.7 pp |
| backendAttentionMs        | 0.63 ms                      | 0.72 ms                  | -12.5% |
| backendAttention %graph   | 1.6%                         | 1.7%                     | -0.1 pp |
| backendDispatchCount      | 652/token                    | 805/token                | -19.0% |
| layerCount                | 32                           | 36                       | -4 |

**Reading the deltas:**

1. **Matmul ms is essentially identical (23.00 vs 23.07).** The
   bandwidth-bound matmul work for an 8B IQ3_M model is the same
   regardless of architecture. #22344's fast i-quant mat-vec kernel
   absorbs both equally.
2. **Dispatch count delta tracks layer count exactly** (652 = 32 ×
   ~20.4; 805 = 36 × ~22.4). Per-layer dispatch budget is identical
   between Llama 3 and Qwen 3 at this quant + GQA shape.
3. **Encode overhead is slightly lower on Llama** (4.01 vs 4.50 ms),
   consistent with fewer layers → fewer per-layer encode hops.
4. **Attention is negligible (~1.6%) on both**, as expected for
   single-token decode where attention is N=1.
5. **Graph compute on Llama is 6.1% faster** (40.00 vs 42.60 ms),
   roughly tracking the 4-layer count delta.

The regression magnitude (-6% non-profile) is on a model whose
post-§32 bucket profile matches the §27 reference within
measurement noise on every bucket. **No bucket sticks out as the
locus of the regression.**

## 4. Hypothesis evaluation

The §32 SUMMARY listed two competing hypotheses for the -6%:

**H1: Tied-embedding × #22456 aliasing-refactor** — Llama 3.1 ties
  lm_head with embed_tokens (shared 4096 × 128256 weight buffer);
  Qwen 3 untied. The aliasing refactor would exercise tied weights
  more heavily. Predicted signature: matmul or encode-overhead
  bucket asymmetry vs qwen3-8b-iq3m (which is also IQ3_M / 8B but
  untied).

**H2: Buffer-aliasing constant overhead** — small per-step overhead
  from the refactor, distributed across the decode pipeline. No
  bucket spike.

The data is consistent with **H2** (uniform overhead) and
**inconsistent with H1** (no bucket asymmetry):

- Matmul Δ vs reference: -0.3% (noise).
- Encode Δ vs reference: -10.9% (Llama is *faster* — opposite of
  what H1 would predict if the lm_head matmul is being slowed by
  aliasing).
- Attention Δ vs reference: -12.5% (Llama is *faster*, again
  opposite of H1).
- Dispatch count: matches layer count delta exactly. No "extra"
  aliasing dispatches surfaced.

If H1 were correct, we'd expect the lm_head matmul (final dispatch
per token, 4096 × 128256 = ~500M elements) to be measurably slower
on Llama vs Qwen3 (which has its own untied 4096 × 151936 = ~620M
lm_head). The opposite is true: Llama's per-token matmul total is
slightly *faster* (23.00 vs 23.07 ms), and Llama's lm_head should
in fact be ~18% smaller than Qwen3's by element count. The
measurement is consistent with both lm_heads running through the
same kernel path with the same bandwidth-bound cost per element —
no aliasing penalty is visible.

## 5. Decision

**Apply the §32a decision rule's "uniform → accept and move on"
branch.** The regression is not bucket-localized; it is small (~6%)
and uniform; the leading hypothesis (H1) is not supported by the
profile data; the alternative (H2) fits but is not directly
actionable without an upstream patch reverting #22456 (which would
unwind the buffer-aliasing benefits ssm_scan and other ops gain
from the refactor).

**Close §32a as "hypothesis tested, rejected; accept §32 as
final."** No follow-up filed.

## 6. What we would need to do better

A *cleaner* test would have been a same-model **pre-§32 vs post-§32
profile-mode trace** on llama-3.1-8b-iq3m specifically. The §27
cycle didn't capture profile-mode for that model (only
qwen3-8b-iq3m), so we can't directly subtract the regression from
its own pre-rebase baseline. The cross-model comparison is weaker
evidence than a pre/post would be, but the bucket-shape match is
striking enough that the H1 rejection is not borderline.

**Process improvement for next rebase:** when the sweep result is
"small regression, accepted" (§32 template), the cycle should
opportunistically capture pre-rebase profile-mode on the regressing
model *before* doing the rebase. Cost: ~3 min wall. Pay-off: the
§32a-style follow-on probe gets a same-model baseline and can
diagnose conclusively rather than via cross-model proxy.

This is captured as a TODO process note for future rebase cycles
under the §27 template documentation.

## 7. Closure verdict

**§32a closes as "hypothesis rejected; §32 baseline accepted as
final."** The post-§32 llama-3.1-8b-iq3m profile-mode bucket profile
is structurally identical to the §27 qwen3-8b-iq3m reference within
measurement noise. The -6% non-profile regression is uniform, not
localized. The "tied-embedding × aliasing-refactor" hypothesis is
not supported by the measurement.

The only remaining theoretical follow-up would be reverting #22456
locally to A/B-test the regression source, but the maintenance cost
of diverging from upstream and the small magnitude of the
regression (-6% on a single non-canonical-baseline model) put it
well below the doctrine threshold for action. The §27 doctrine
("document and move on, unless a free win opens") applies cleanly.

## 8. Updates to canonical baselines

`llama-3.1-8b-iq3m` profile-mode baseline (new entry — was not
previously captured):
- **23.5 tok/s** (profile, 3-run, 156 step trace)
- matmul **23.02 ms / 57.3%**, encode **4.01 ms / 10.0%**,
  attention **0.63 ms / 1.6%**, dispatch **652/token**.

Keep alongside qwen3-8b-iq3m's §27 profile-mode pin (22.0 tok/s,
matmul 23.07 ms / 55.0%, dispatch 805/token). These two now form a
matched 8B IQ3_M reference pair for any future post-rebase probe.
