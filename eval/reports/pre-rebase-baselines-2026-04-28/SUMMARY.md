# Pre-rebase profile-mode baselines — canonical 6 (2026-04-28)

**Captured:** 2026-04-28 (post-§32 / post-§32a / post-patch-12-squash)
**Build:** `webllm-browser-patches` tip `3b8ade2a2` (11-patch stack);
WASM 2,249,650 bytes; ship gate 428/11/0.
**Procedure:** `make smoke-bench PERF_MODEL=<m> PERF_RUNS=3` (profile mode).
**Trigger:** TODO §32a process-improvement note — capture pre-rebase
profile-mode for the canonical fleet *before* the next rebase fires,
so a §32a-style follow-on probe can diagnose any regression via same-
model pre/post deltas instead of the cross-model proxy that §32a had
to use.

---

## 1. Headline matrix

3-run median, profile-mode (perturbed). All medians are column-wise
medians of the 3 runs.

| Model                          | Quant   | tok/s | Steps | graph (med, ms) | matmul (med, ms / %) | encode (med, ms / %) | dispatch/token |
|---|---|---:|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | Q4_0    | 87.9  | 189   | 10.20           | 3.74 / 37.8%         | 2.40 / 24.0%         | 450 |
| qwen3-0.6b-q4f16               | Q8_0    | 68.2  | 72    | 11.20           | 3.87 / 33.6%         | 3.30 / 30.1%         | 629 |
| qwen3-1.7b-q4f16               | Q8_0    | 44.0  | 48    | 17.80           | 6.75 / 36.9%         | 3.60 / 19.6%         | 629 |
| mistral-7b-instruct-v0.3-q4ks  | Q4_K_S  | 29.7  | 189   | 32.70           | 15.86 / 48.7%        | 3.60 / 11.4%         | 650 |
| llama-3.1-8b-instruct-iq3m     | IQ3_M   | 23.5  | 156   | 39.80           | 23.00 / 57.5%        | 3.40 / 8.9%          | 652 |
| qwen3-8b-iq3m                  | IQ3_M   | 21.8  | 60    | 42.40           | 23.20 / 55.4%        | 4.40 / 10.7%         | 805 |

Per-model raw logs in this directory (`<model-id>.log`).

## 2. Cross-references against existing baselines

- **llama-3.1-8b-instruct-iq3m: 23.5 tok/s** — bit-identical to
  §32a's PROFILE-32A.md (same day, same WASM build); the captures
  reproduced cleanly. matmul 23.00 ms / 57.5% vs §32a's 23.02 / 57.3%
  is single-run noise.
- **qwen3-8b-iq3m: 21.8 tok/s** — §27 closure recorded 22.0 tok/s
  profile-mode for this model (post-rebase, same WASM family). Within
  noise band. matmul share 55.4% matches §27's 55.0%.
- **mistral-7b-q4ks: 29.7 tok/s profile-mode** — no prior profile-mode
  pin in TODO/§32 SUMMARY (the §32 baseline was non-profile 35.0).
  Profile-mode perturbation: 35.0 → 29.7 = -15%, normal band.
- **TinyLlama Q4_0: 87.9 tok/s profile-mode** — §32 SUMMARY non-profile
  claim 107.4 → profile 87.9 = -18% perturbation. Normal band for
  the 450-dispatch decode shape.
- **qwen3-0.6b Q8: 68.2 tok/s profile-mode** — §32 non-profile 86.9 →
  profile 68.2 = -22% perturbation. Larger because Qwen3 is dispatch-
  heavier per token (629 vs 450).
- **qwen3-1.7b Q8: 44.0 tok/s profile-mode** — §32 non-profile 60.9 →
  profile 44.0 = -28% perturbation. Same reason as 0.6B; dispatch
  count 629 / decode-bound steps 48 means each profile timestamp
  sample weighs more per second of wall.

## 3. Bucket profile observations

- **Encoder share scales inversely with model size.** Tiny (24% of
  graph) > Qwen3-0.6B (30%) > Qwen3-1.7B (20%) > 7-8B (9-11%). At 8B
  IQ3_M, encoder is a distant secondary cost behind matmul.
- **Matmul share scales with size (and quant compute).** Q4_0 / Q8_0
  models sit at 33-38%; Q4_K_S Mistral 49%; IQ3_M 8B both at 55-58%.
  This pin is consistent with the post-§27 finding that #22344's
  i-quant decompression speedup left matmul still as the lead bucket
  on IQ3_M.
- **Dispatch/token is architecture-invariant within a family.**
  qwen3-0.6b and qwen3-1.7b both at 629/token. qwen3-8b at 805/token
  (matches §16 and §27 pins). llama-3.1-8b at 652. Mistral-7B at 650.
  TinyLlama at 450. The §10 prediction that dispatch count tracks
  layer count + arch-specific extras holds across the fleet.
- **graphCompute share consistently 89-96% of total step time.**
  Other phases (buildGraph, backendAlloc, uploadLeaves, downloadResult,
  teardown) collectively under 11%. No headroom hidden in those
  buckets.

## 4. Use case

When the *next* upstream `ggml-webgpu` rebase trigger fires:

1. Run the new `bun run eval/perf.ts --runs 3` non-profile sweep on
   the same 6 models (the §27 / §32 sweep template).
2. **Classify the outcome** per §27 (free win) / §28 (negative result,
   close lever harder) / §32 (small regression, accept) templates.
3. **If small-regression, accepted (§32 template)**, immediately run
   `make smoke-bench PERF_MODEL=<regressing> PERF_RUNS=3` for
   profile-mode follow-on. Compare the new bucket profile against the
   matching pre-rebase row from this matrix. **Same-model
   diagnosis** beats §32a's cross-model proxy: any bucket that moved
   asymmetrically is the locus of the regression.
4. If the post-rebase bucket profile is structurally identical
   (matmul Δ <1%, encode Δ <1%, dispatch unchanged) → uniform per-step
   overhead; accept and move on (§32a's H2 outcome).
5. If a bucket moved asymmetrically → focused diagnostic; that
   bucket's locus is the §32a-style follow-on lever candidate.

## 5. Freshness window

Pre-rebase baselines decay if either:

- The fleet drifts (model selection changes, new entry added) — these
  pins lock to the canonical 6 only.
- Several weeks pass without a rebase — bench-mode noise is small but
  ambient measurement environment can drift.

If the rebase trigger fires within ~1 month, this matrix is fresh.
After ~1 month with no rebase ETA, re-capture before the next sweep.
The capture cost is ~3 min wall per model (~18 min for the canonical
6); the pay-off only accrues if a rebase trigger actually fires.

## 6. Provenance

Each per-model log file in this directory contains the full
`make smoke-bench` output for that model: 3 runs of 64 generated
tokens each (or ~17-322 for thinking-on profiles), with per-phase
decode timing, backend decode attribution, and dispatch counts. The
median row is the basis for the headline matrix above.

llama-3.1-8b-instruct-iq3m's log is the §32a re-capture under
identical procedure; reproducibility verified against
PROFILE-32A.md.
