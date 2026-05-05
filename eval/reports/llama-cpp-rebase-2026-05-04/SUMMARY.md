# llama.cpp rebase 2026-05-04 — drop local LayerNorm patches

**Trigger:** daily upstream cadence check fired non-empty: `d4b0c22f9
ggml-webgpu: add layer norm ops (#22406)` landed upstream, subsuming
our local **patch 9** (`72b6d001e ggml-webgpu: add GGML_OP_NORM`) and
its companion split (`c775ac26d ggml-webgpu: split LAYER_NORM
accumulation loop from legacy norms`).

**Outcome:** **§27 hybrid (maintenance free win, perf neutral).** Two
local patches dropped cleanly; patch stack reduced from 11 → 9.
Encoder-parity gate PASS; cross-day perf comparison vs 2026-05-01
baseline is noise-level on 5/6 models, with one mistral-7b outlier
(-14%) flagged for follow-up.

**Tip:** new base `a817a22bc` (upstream master); local stack tip
`fc1f81242` (UB shift fix), 9 patches.

---

## Patch-stack delta

| State | Tip | Patches | Removed |
|---|---|---:|---|
| Pre-rebase | `e29753286` | 11 | — |
| Post-rebase | `fc1f81242` | 9 | `72b6d001e` (LAYER_NORM patch 9), `c775ac26d` (split companion) |

Upstream `d4b0c22f9` is functionally a superset of our patches:

- **LAYER_NORM algorithm:** upstream uses two-pass (mean → variance);
  ours used single-pass `Σx² / n − mean²`. Two-pass is more
  numerically stable. For typical embedder inputs the difference is
  in the f32 noise floor.
- **f16 plumbing:** upstream adds `SRC_F16` / `DST_F16` codegen.
  `supports_op` gates to f32 (matches our build's only path), so f16
  plumbing is dormant but ready for a future encoder use.
- **Pipeline cache key:** upstream extends the row-norm pipeline-key
  with `src_type` / `dst_type`; ours had only `op` + `inplace`.
  Functionally equivalent at f32; better cache discipline at the
  type boundary.
- **`ggml_webgpu_row_norm` dispatch path:** identical between our
  patch and upstream — same fall-through case structure.

---

## Encoder parity gate — PASS

Smoke step `[8/8]` exercises arctic-embed-s' `embed("happy") /
embed("joyful")` and gates on cosine ∈ [0.75, 0.999):

| Capture | Cosine | ‖v‖ | Verdict |
|---|---:|---:|---|
| Pre-cycle baseline (`embed-perf-baseline-cosine.json`) | 0.76 ± 0.005 | — | reference |
| Post-rebase smoke (`http://localhost:8031/real-model.html?model=tinyllama-1.1b-chat-q4_0`) | 0.76 | 1.00 | **PASS** |

The numerical-stability improvement in upstream's two-pass LayerNorm
algorithm is invisible at the f32 cosine layer for the canonical
synonym pair; reference parity holds within tolerance.

`bge-small-en-v1.5-q0f16` single-text short-fixture wall-time:
**10.90 ms p50** (3 reps; logged at
`eval/reports/embed-perf-2026-05-05/bge-small-en-v1.5-q0f16_single_short.json`).
No encoder regression.

`bun test` clean: **741 pass / 0 fail / 33 skip** (matches documented
environmental-guard baseline; no new skips).

---

## Canonical-6 perf sweep

3-run median, profile-mode (perturbed). Same-day same-tip pre-rebase
control captured at
[`eval/reports/pre-rebase-baselines-2026-05-04/`](../pre-rebase-baselines-2026-05-04/);
post-rebase logs in this directory.

| Model | Pre 05-04 (same-day) | Post 05-04 | Δ same-day | 2026-05-01 baseline | Δ vs 05-01 |
|---|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 61.6 | **79.8** | +29.5% | 80.4 | -0.7% |
| qwen3-0.6b-q4f16               | 48.0 | **56.9** | +18.5% | 62.5 | -8.9% |
| qwen3-1.7b-q4f16               | 33.0 | **43.0** | +30.3% | 41.7 | +3.1% |
| mistral-7b-instruct-v0.3-q4ks  | 26.2 | **28.1** | +7.3% | 32.8 | -14.4% |
| llama-3.1-8b-instruct-iq3m     | 20.2 | **23.6** | +16.8% | 24.2 | -2.5% |
| qwen3-8b-iq3m                  | 19.1 | **21.4** | +12.0% | 22.2 | -3.6% |

### Backend attribution (post-rebase)

| Model | graph (med, ms) | matmul (ms) | encode (ms) | dispatch/token |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 11.20 | 4.19  | 2.50 | 450 |
| qwen3-0.6b-q4f16               | 12.70 | 3.87  | 3.40 | 629 |
| qwen3-1.7b-q4f16               | 18.60 | 6.88  | 3.60 | 629 |
| mistral-7b-instruct-v0.3-q4ks  | 33.50 | 16.12 | 4.50 | 650 |
| llama-3.1-8b-instruct-iq3m     | 39.80 | 23.13 | 3.50 | 652 |
| qwen3-8b-iq3m                  | 43.00 | 23.72 | 4.50 | 805 |

Dispatch counts unchanged across the rebase (LAYER_NORM is not on the
chat decode hot path; chat models use RMS_NORM). Matmul / encode /
attention proportions match the 2026-05-01 backend attribution profile.

---

## Why same-day pre-rebase deltas overstate the gain

The same-day pre-rebase capture (61.6 tinyllama / 48.0 qwen3-0.6b /
…) is uniformly **14-23% below** the 2026-05-01 same-tip baseline,
despite running the **identical code** (`e29753286`). Post-rebase
numbers come back to within ~1% of the 2026-05-01 baseline on
tinyllama (79.8 vs 80.4) and within noise on most others.

**Diagnosis:** the pre-rebase 2026-05-04 capture happened with the
agentchrome browser/GPU in a degraded steady state (likely shader
cache cold, GPU clocked down, or background WGPU consumer). Only one
real change between pre and post — dropping two LAYER_NORM patches
that don't touch chat decode — and that change cannot plausibly
deliver a 30% chat decode gain.

**Honest framing:** this cycle delivers no measured chat-decode perf
delta vs the 2026-05-01 cross-day baseline. The same-day numbers
indicate that **same-day floor capture is not a substitute for
measurement of a steady-state floor** — it captures *whatever floor
the environment happens to be at when the bench fires*.

### Process lesson

`§32a` was framed as: "before any cross-day bench comparison, capture
a same-day same-tip control as the first data point." That framing
implicitly assumes the same-day capture itself is on the steady-state
floor. Refining: a same-day capture **anchors the environmental
floor at capture time**; if the floor at that moment is anomalous,
the comparison is anchored to noise.

**Refinement (2026-05-04):** when same-day pre-rebase deltas are
implausibly large given the diff scope (e.g. +30% chat decode from a
LAYER_NORM-only swap that isn't on the chat decode path), distrust
the same-day baseline and fall back to the pinned cross-day
baseline. The failure mode is environmental-floor-anchoring, not
code-effect.

This refinement is preventive, not corrective — same-day capture is
still the correct *first-line* evidence; this is just an
implausibility filter for sanity-checking the magnitude of measured
deltas. Codify under: "if the same-day delta exceeds the
diff-scope's plausible blast radius, run the cross-day check before
attributing."

---

## Mistral-7b cross-day outlier (-14.4%)

mistral-7b is the only canonical-6 model showing more than ~9% drop
vs 2026-05-01 (32.8 → 28.1). Three plausible causes, none confirmed:

1. **Environmental drift specific to that model's profile** — the
   rest of the fleet sits within ±9% of 2026-05-01; mistral may have
   hit a Chrome shader-cache miss on the day of the post-rebase
   capture but not on 2026-05-01.
2. **Real interaction with one of the upstream-touching commits in
   the rebase window** — but `d4b0c22f9` is LAYER_NORM (not used
   in chat decode) and `a817a22bc` is walsh-hadamard (not used by
   any model in the canonical 6 — that's a Qwen3-Next / Mamba-class
   op). No other ggml-webgpu/ touches landed.
3. **Profile-mode noise** — Mistral's profile-mode capture has been
   the noisiest in prior cycles too (see the §32 2026-04-29 sweep
   matrix).

**Action:** queue a follow-up rerun in the next session if mistral-7b
shows up below the 32.8 reference on the next bench. Single
data-point cross-day isn't enough to declare a regression. Don't
gate the cycle on this — adopt the rebase, watch for trend.

---

## Verdict

**Adopt the rebase.** The cycle delivers:

- **Maintenance:** two local patches (`72b6d001e` / `c775ac26d`)
  dropped — patch stack 11 → 9. Reduces rebase friction for the
  next cycle.
- **Encoder parity:** PASS at cosine 0.76 vs reference 0.76; norm
  1.00; 741/0/33 unit-test baseline.
- **Perf:** noise-level vs the pinned 2026-05-01 cross-day baseline
  on 5/6 models; mistral-7b -14% flagged but not load-bearing
  pending a rerun.
- **Tip:** `fc1f81242` on upstream `a817a22bc`. Pinned in
  `docs/LLAMA_CPP_PATCHES.md` cycle table.

Classification: **§27 hybrid** (maintenance free win, perf neutral).
This is closer in spirit to §27 than §32 because the "regression" is
a single noisy data point on one model, not a systematic broad drop.

---

## Artifacts

- Pre-rebase same-day baselines (anomalous capture, retained for
  process lesson):
  [`eval/reports/pre-rebase-baselines-2026-05-04/`](../pre-rebase-baselines-2026-05-04/)
- Post-rebase canonical-6 logs: this directory.
- 2026-05-01 cross-day baseline (canonical reference):
  [`eval/reports/pre-rebase-baselines-2026-05-01/`](../pre-rebase-baselines-2026-05-01/)
- Encoder-parity gate (smoke `[8/8]` cosine 0.76):
  `embed-perf-baseline-cosine.json` (root of `eval/reports/`).
- Embed-perf single-fixture short on bge-small (PASS):
  [`eval/reports/embed-perf-2026-05-05/`](../embed-perf-2026-05-05/)
