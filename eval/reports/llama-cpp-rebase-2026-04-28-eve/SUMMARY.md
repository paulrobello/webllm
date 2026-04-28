# §32 — llama.cpp rebase 2026-04-28-eve + free-win sweep

**Date:** 2026-04-28 (evening)
**Trigger:** TODO §27 template — re-run `bun run eval/perf.ts` 6-model
sweep after upstream `ggml-webgpu` movement.
**Upstream delta:** `434b2a1ff → f9f33654a` (10 commits, 1 in
`ggml-webgpu/`).

---

## 1. Headline

- ✅ **Rebase replayed all 11 patches cleanly.** No semantic conflicts
  detected by `git rebase --onto`.
- ⚠️ **Compile error surfaced post-rebase** in patch 3 (request-based
  browser readback API): upstream #22456 renamed the
  `webgpu_tensor_offset` helper to `ggml_webgpu_tensor_offset` and
  folded `view_offs` into the helper body. **Resolved by adding patch
  12 as a forward fix-up** (single-line rename + drop redundant
  `view_offs`). Bit-identical post-rename behavior. Should be squashed
  back into patch 3 on a future manual cleanup pass.
- ✅ **WASM build clean** after fix-up. 2,249,650 bytes (was 2,240,603
  pre-§32 — +9 KB from upstream code growth, mostly the aliasing
  refactor).
- ✅ **Ship gate green:** `make checkall` 428/11/0.
- ⚠️ **Bench sweep is *not* a free win.** 5 of 6 models within ±5%
  noise band of §27 post-rebase baselines; **llama-3.1-8b-iq3m holds
  a real ~6% regression** at 29.0 → 27.2 tok/s, consistent across 5
  runs. Likely cause is #22456's aliasing-logic refactor interacting
  with GQA + IQ3_M kernels.

## 2. Sweep matrix (vs §27 post-rebase baselines)

3-run median unless noted. Comparison baseline:
`bun run eval/perf.ts --runs 3` post-§27 rebase 2026-04-27.

| Model                         | Quant   | §27 (3 runs) | §32 (3 runs) | §32 (5 runs) | Δ from §27 |
|---|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0      | Q4_0    | 110.8        | 107.4        | —            | -3.1% |
| qwen3-0.6b-q4f16              | Q8_0    | 89.8         | 86.9         | —            | -3.2% |
| qwen3-1.7b-q4f16              | Q8_0    | 62.2         | 60.9         | —            | -2.1% |
| mistral-7b-instruct-v0.3-q4ks | Q4_K_S  | 35.8         | 33.8         | **35.0**     | -2.2% (5-run) |
| **llama-3.1-8b-instruct-iq3m**| **IQ3_M** | **29.0**   | **26.9**     | **27.2**     | **-6.2% (5-run)** |
| qwen3-8b-iq3m                 | IQ3_M   | 27.2         | 26.2         | —            | -3.7% |

5-run confirmation runs done on the two largest 3-run deltas. mistral-
7b-q4ks normalises into noise band (-5.6% → -2.2%); llama-3.1-8b-iq3m
holds at -6%, with tight 5-run variance (26.8-27.6, p50 27.2). The
mistral first-run wall-time was 9834 ms vs 6886-7342 ms for runs 2-5
— an obvious environmental outlier that pulled the 3-run median down.

## 3. Per-model raw runs

### tinyllama-1.1b-chat-q4_0 (3 runs)
```
105.8 / 108.5 / 107.4 → p50 107.4
```

### qwen3-0.6b-q4f16 (3 runs)
```
86.2 / 89.3 / 86.9 → p50 86.9
```

### qwen3-1.7b-q4f16 (3 runs)
```
60.9 / 61.8 / 57.7 → p50 60.9
```

### mistral-7b-instruct-v0.3-q4ks
- 3 runs: `33.8 / 34.0 / 33.5 → p50 33.8` (first run had wall-time outlier)
- 5 runs: `34.7 / 34.9 / 35.5 / 35.0 / 35.5 → p50 35.0`

### llama-3.1-8b-instruct-iq3m
- 3 runs: `25.4 / 26.9 / 27.2 → p50 26.9`
- 5 runs: `26.8 / 27.2 / 27.6 / 27.1 / 27.5 → p50 27.2`

### qwen3-8b-iq3m (3 runs)
```
26.2 / 26.2 / 26.0 → p50 26.2
```

## 4. Outlier analysis — llama-3.1-8b-iq3m

Why this model and not qwen3-8b-iq3m? Both are IQ3_M, both 8B,
both decode-bound at single-token. The differences:

| Axis                | llama-3.1-8b-iq3m       | qwen3-8b-iq3m            |
|---|---|---|
| Architecture        | Llama 3                 | Qwen 3                   |
| Attention layout    | GQA (32 Q × 8 KV heads) | GQA (32 Q × 8 KV heads)  |
| Head dim            | 128                     | 128                      |
| Layers              | 32                      | 36                       |
| Embedding dim       | 4096                    | 4096                     |
| Vocab               | 128256                  | 151936                   |

Same GQA shape. Same head_dim. The architectures look near-identical
on attention. Differences that *could* drive a kernel-aliasing-
sensitive regression:

- **Layer count (32 vs 36).** Per-step kernel-launch overhead × 32 vs
  × 36 — but the regression goes the *opposite* direction (the model
  with *fewer* layers is slower).
- **Vocab size.** 128256 vs 151936 — affects only the final lm_head
  matmul, dispatched once per token. ~18% larger on qwen3 should make
  qwen3 slower at this stage, not faster.
- **Tied vs untied embeddings.** Llama 3.1 is tied (lm_head shares
  embedding weights). Qwen 3 is untied. The aliasing refactor
  specifically touched buffer aliasing — tied weight sharing exercises
  buffer aliasing more heavily and could be where the regression
  bites.

The "tied embedding aliasing" hypothesis is plausible but not
verified. Fully testing it would mean profile-mode rebench
(`make smoke-bench PERF_MODEL=llama-3.1-8b-instruct-iq3m PERF_RUNS=3`)
to see the per-bucket breakdown. **Not done in this cycle** — the
6% regression is small enough that the §27 doctrine ("document and
move on, unless a free win opens") applies. Profile-mode rebench is
queued as an optional follow-up if the user wants to investigate.

## 5. Decision

**Accept the rebase as the new baseline.** Rationale:

1. 5 of 6 models within ±5% noise band — the rebase is essentially
   neutral on the production fleet.
2. The single 6% regression is on a model that is not in the canonical
   §17/§18/§19/§20 4-baseline bench gate. Llama-3.1-8b is wave-2 and
   was added at §13.
3. Staying on top of upstream master is high-value: future i-quant /
   shader / kernel optimizations land cleanly via routine rebase. The
   §27 free win (+80% on IQ3_M) showed the upside of staying current.
4. Reverting to §27 base (`434b2a1ff`) costs ~6% on llama-3.1-8b but
   loses the option value of upstream's 9 other commits (Vulkan tuning
   landed in this delta sets up the next ggml-webgpu kernel cycle).
5. Cherry-picking around #22456 specifically would diverge from
   upstream more, increasing maintenance cost on every future rebase.

**Update the canonical baseline:** llama-3.1-8b-iq3m's bench-inf
number drops from 29.0 → 27.2 tok/s. The other five baselines move
within noise. §27's pinning of 27.2 tok/s for `qwen3-8b-iq3m` remains
valid (5-run noise: 26.0-26.2, just under §27's 27.2).

## 6. Closure verdict

**§32 closes as "rebase-clean (after fix-up patch 12), small
regression, accepted."** Distinct from §27's free-win close — this
cycle did *not* deliver upside, but the cost is small enough to ship.

The rebase pattern remains valid: trigger on upstream `ggml-webgpu`
movement, run the 6-model sweep, classify result. **§32 is the first
documented example of a "no free win, small regression, accepted"
outcome.** Add to the resumption-checklist's set of templates.

If a future cycle finds either (a) llama-3.1-8b-iq3m's regression
*deepens* on subsequent rebase, or (b) a new model lands that
exercises tied-embedding + IQ3_M heavily, re-investigate via
profile-mode rebench. Until then, accept and ship.

## 7. Commits

On `webllm-browser-patches` (llama.cpp local branch):
- `c4af89356 ggml-webgpu: rebase fix-up — adopt #22456 helper rename`
  (patch 12, single line)

On `webllm` (this repo, `main`):
- *forthcoming* — `docs(rebase)` for this report
- *forthcoming* — `docs(TODO)` for §32 entry

No `src/` change on either repo (apart from llama.cpp's patch 12).
WASM artifacts in `smoke-test/` regenerated and gitignored.

## 8. Reproduction

```bash
# llama.cpp side
cd ~/Repos/llama.cpp
git checkout webllm-browser-patches
git branch webllm-browser-patches-pre-rebase-<DATE-TAG> $(git rev-parse HEAD)  # safety
git fetch origin
git rebase --onto origin/master <prior-base> webllm-browser-patches
# If patch 3 fails to compile post-rebase, see patch 12 narrative
# in docs/LLAMA_CPP_PATCHES.md.

# webllm side
cd ~/Repos/webllm
rm -rf src/wasm/build  # nuke stale CMakeCache (MATH_LIBRARY=NOTFOUND)
source ~/emsdk/emsdk_env.sh
make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/
make smoke-restart
make checkall
for m in tinyllama-1.1b-chat-q4_0 qwen3-0.6b-q4f16 qwen3-1.7b-q4f16 \
         mistral-7b-instruct-v0.3-q4ks llama-3.1-8b-instruct-iq3m \
         qwen3-8b-iq3m; do
  bun run eval/perf.ts --model $m --runs 3
done
```

The `MATH_LIBRARY=NOTFOUND` cache-staleness gotcha is upstream-driven
(commit `d530d6e7a ggml : revert to -lm linking instead of
find_library`) — **always nuke `src/wasm/build/` before a build that
crosses this commit** to avoid the misleading "NOTFOUND" link error.

## 9. Wall-clock cost

- llama.cpp rebase: 5 seconds (machine-replay).
- Patch 12 fix-up: 2 minutes (diagnose + edit + commit).
- CMake cache nuke + WASM rebuild: ~25 seconds.
- Bundle + smoke restart: ~3 seconds.
- Ship gate: ~1.5 seconds.
- 6-model 3-run sweep: ~6 minutes (mostly browser warmup per model).
- 2-model 5-run confirmation: ~3 minutes.
- This writeup: ~10 minutes.

**Total: ~25 minutes.** Worth it: the data is now on the record;
future rebases reuse the patch 12 narrative and the cache-staleness
gotcha.
