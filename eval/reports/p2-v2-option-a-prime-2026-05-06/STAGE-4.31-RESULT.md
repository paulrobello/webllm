# Stage 4.31 — Probe 18 Shape A: full-tensor stats on `kqv_out-0`

**Date:** 2026-05-07
**Probe:** Widen `node_dump_cb` (`src/wasm/webgpu-bridge.cpp`) so that
when `tensor->name == "kqv_out-0"` it walks the FULL tensor (not just
the first 8 elements) and emits a second stderr line —
`[CHECKPOINT-FULL idx=N name=kqv_out-0 n_elements=N finite=F
mean=… abs_max=… abs_min=… nan=… inf=…]` — alongside the existing
`[CHECKPOINT idx=N …]` first-8 line. Both `webllm-wasm-jsep.{js,wasm}`
(JSEP spike) and `webllm-wasm.{js,wasm}` (non-JSEP reference probe)
compile from the same translation unit, so both runs pick up the new
emission. Spike + ref-probe each parse the lines into
`globalThis.__stage431Stats`; the deltas are computed offline by
comparing the two snapshots.
**Outcome:** **P-18-first8-blind CONFIRMED.** Suspect 3
(first8-window blindness on `kqv_out-0`) is real and load-bearing —
the prefill `kqv_out-0` full-tensor `abs_max` differs by **0.067**
between JSEP and reference (>>>>1e-3 threshold), and every decode
step diverges by 0.05–0.66 on `abs_max`. The first-8 window the
existing `node_dump_cb` was reporting was `V[pos=0]` (causal mask
pins the position-0 softmax row to `[1, 0, 0, …]`), so it
coincidentally agreed across runs. Positions 1+ — the exact rows
the existing instrumentation was blind to — carry the cascade.

## TL;DR

Side-by-side `kqv_out-0` full-tensor stats over the prefill (idx 11,
12 288 elements = `[2048, 6, 1, 1]`) and the 5 decode steps (idx
29, 47, 65, 83, 101 — each 2048 elements = `[2048, 1, 1, 1]`):

| idx |       n |   jsep mean |    ref mean | jsep abs_max | ref abs_max | jsep abs_min |     ref abs_min |
| --- | ------: | ----------: | ----------: | -----------: | ----------: | -----------: | --------------: |
| 11  | 12 288  | -6.762e-05  | -5.178e-04  |   **0.04959** |  **0.11706** |          0.0 |       4.47e-07  |
| 29  |  2 048  | -8.728e-04  | -2.215e-04  |   **0.69582** |  **0.05177** |          0.0 |       4.59e-07  |
| 47  |  2 048  | -1.028e-03  | -4.450e-04  |   **0.62272** |  **0.07719** |          0.0 |       1.25e-06  |
| 65  |  2 048  | -4.248e-04  | -6.865e-04  |   **0.75973** |  **0.17834** |          0.0 |       2.68e-07  |
| 83  |  3.339e-04 (jsep) / -6.390e-04 (ref) |             |             |   **0.91935** |  **0.26765** |          0.0 |       4.35e-07  |
| 101 |  7.055e-04 (jsep) / -7.124e-04 (ref) |             |             |   **0.85919** |  **0.11417** |          0.0 |       1.17e-07  |

`[STAGE-4.31] OUTCOME: P-18-first8-blind (full-tensor abs_max delta
exceeds 1e-3 on every kqv_out-0 forward pass; suspect 3 first8-window
blindness on kqv_out-0 CONFIRMED; pivot Stage 4.32 to idx-by-idx
divergence localization within kqv_out-0)`

## What this surfaces

1. **The first8 window was hiding the bug.** Stage 4.27 row 11
   reported `kqv_out-0` `max_abs_delta=0.000000` between JSEP and
   reference. That Δ was over indices 0..7 only, which — for a
   prefill `[2048, 6, 1, 1]` tensor laid out so first8 hits column 0
   row 0..7 — corresponds to `V[pos=0]` weighted by the position-0
   softmax row that the causal mask pins to `[1, 0, 0, 0, 0, 0]`.
   Both backends reproduce `V[pos=0]` faithfully *for position 0*
   because there's no cross-position contribution (mask zeros it
   out). The divergence lives at positions 1..5, which the existing
   first8-only readback never touched. Full-tensor stats expose it
   immediately.
2. **JSEP's `kqv_out-0` carries exact zeros.** `abs_min = 0.0` on
   *every* prefill + decode pass on the JSEP path. Reference's
   `abs_min` is ~1e-7, i.e. all elements are non-zero (as expected
   from a 2048-wide dot-product over real activations). With
   12 288 elements the chance of hitting an exact 0.0 in real f32
   output is essentially zero — this is the dispositive signature
   of partial-zero output: at least one block of indices is being
   skipped or written from an uninitialized region.
3. **Decode `abs_max` blow-up.** JSEP `kqv_out-0` decode-step
   `abs_max` lands at 0.62–0.92 vs reference 0.05–0.27. The bug
   isn't a small numerical drift — JSEP is producing dramatically
   different magnitudes. Combined with the abs_min=0 signal, the
   most plausible structural reading is that one or more
   sub-tile/sub-row outputs are being *dropped* (left as zero) while
   adjacent outputs are being *overcounted* (single tile receives
   what should have been spread across multiple tiles). That fits
   a tile-bounds / dispatch-coverage bug in the JSEP MUL_MAT (or
   FA-equivalent path) that produces `kqv_out-0` from V × softmax.
4. **The cascade source for `attn_out-0` Δ=4.77e-3 is now
   localized one op upstream.** `attn_out-0 = out_proj × kqv_out`.
   With `kqv_out-0` itself diverging by 0.05–0.92 in `abs_max`, no
   amount of output-projection precision can recover; the input is
   wrong. Stage 4.27's smoking-gun row 12 (`attn_out-0`
   max_abs_delta=4.77e-3) is the *first8-visible projection* of a
   much larger upstream divergence. The bug is in the Q×K^T →
   softmax → V → kqv_out chain, not in `out_proj`.

## What this rules out vs leaves open

**Closed (not viable as the cascade source):**
- ❌ Output-projection (`attn_output.weight`) byte-integrity gap —
  Stage 4.28 (set_tensor probe) + Stage 4.30 (post-load
  `tensor->data` peek) both byte-clean.
- ❌ `ffn_norm.weight` gain-vector mis-load — Stage 4.30 byte-clean.
- ❌ Q4_K matmul precision (Stage 4.18 / 4.22 / 4.24 / 4.25 / 4.26)
  — synthetic-replay Δ ≤ 7.94e-6, libllama imprecision ≤ 4.18e-2,
  WGSL is the more accurate side.
- ❌ first8-window blindness as a "doesn't matter" cosmetic — the
  full-tensor delta is real and large; the cascade we've been
  attributing to `attn_out-0` actually starts upstream at
  `kqv_out-0` (or earlier in the attention sub-graph).

**Open (Stage 4.32 candidates):**
1. **Tile-bounds / dispatch-coverage bug in JSEP MUL_MAT for the
   V × softmax matmul.** The abs_min=0 + abs_max blow-up signature
   matches a "one workgroup dispatch covered fewer rows than the
   tensor needs" regression. Stage 4.32 narrows the window: capture
   `kqv_out-0` element-by-element bytes from both runs, compute
   per-index delta, and find the contiguous range(s) where JSEP
   output is zero or wrong. That tells us whether the bug is
   row-bounded (per-row coverage gap), column-bounded (per-column
   coverage gap), or per-block (e.g. every K-th index).
2. **Upstream divergence at `kq_soft_max-0` or `kq-0`.** Stage 4.27
   row 9 captured `kq-0` first8 Δ=0.0119 (already non-trivial) —
   but again first8 only. The same first8-window-blindness logic
   applies: if `kq-0` (Q × K^T) carries a bigger full-tensor
   divergence, the softmax + V mat-mul faithfully amplifies it into
   `kqv_out-0`. Stage 4.32 should *also* expand the full-tensor
   probe to `kq-0` and `kq_soft_max-0` to disambiguate "kqv kernel
   bug" vs "Q×K^T kernel bug propagating downstream".

The two open suspects are **not mutually exclusive** — the Stage
4.27 first8 numbers (Q×K^T 0.0119, kqv_out V[pos=0] 0) were both
underestimates. Stage 4.32 should widen the same probe over the
whole attention sub-graph and stop *only* where the full-tensor
delta is genuinely ≤1e-5.

## What was implemented

### C++ (`src/wasm/webgpu-bridge.cpp`)

```cpp
#include <cmath>
// ... existing includes ...

// Inside node_dump_cb, after the existing first-8 fprintf:
if (std::strcmp(t->name, "kqv_out-0") == 0 && ggml_is_contiguous(t)) {
    int total = (int) ggml_nelements(t);
    double sum_v = 0.0, abs_max = 0.0, abs_min = 0.0;
    bool abs_min_set = false;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < total; ++i) {
        float x = ggml_get_f32_1d(t, i);
        if (std::isnan(x)) { nan_count++; continue; }
        if (std::isinf(x)) { inf_count++; continue; }
        sum_v += (double) x;
        double ax = std::fabs((double) x);
        if (ax > abs_max) abs_max = ax;
        if (!abs_min_set || ax < abs_min) {
            abs_min = ax;
            abs_min_set = true;
        }
    }
    int finite = total - nan_count - inf_count;
    double mean = finite > 0 ? sum_v / (double) finite : 0.0;
    if (!abs_min_set) abs_min = 0.0;
    fprintf(stderr,
            "[CHECKPOINT-FULL idx=%d name=%s n_elements=%d finite=%d "
            "mean=%.9g abs_max=%.9g abs_min=%.9g nan=%d inf=%d]\n",
            g_node_dump_idx, t->name, total, finite,
            mean, abs_max, abs_min, nan_count, inf_count);
}
```

`g_node_dump_idx` is incremented immediately after; the value
captured here matches the `[CHECKPOINT idx=…]` line on the same
forward pass. The probe is gated by `ggml_is_contiguous(t)` (Stage
4.27's table confirms contig=1 for every `kqv_out-0` capture, so
the gate never short-circuits in practice but defends against
view-only entries from future allowlist extensions).

### TS spike (`smoke-test/p2-v2-spike.src.ts`)

After the existing Stage 4.17 `__stage417Checkpoints` block, parse
the new `[CHECKPOINT-FULL …]` lines into typed records and expose
them on `window.__stage431Stats`:

```ts
const stage431Pat =
  /\[CHECKPOINT-FULL idx=(\d+) name=(\S+) n_elements=(\d+) finite=(\d+) mean=(\S+) abs_max=(\S+) abs_min=(\S+) nan=(\d+) inf=(\d+)\]/;
// ... iterate __stderrLines, push parsed records, log each line.
(window as any).__stage431Stats = stage431Stats;
```

### TS ref-probe (`smoke-test/p2-v2-ref-probe.src.ts`)

Same parse + expose block on the non-JSEP page. Reference run lands
its own `__stage431Stats` separate from the spike's.

### Build / run

```bash
make wasm-build-jsep            # rebuild + re-bundle JSEP spike
make wasm-build-wasm32          # rebuild non-JSEP wasm
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/
bun build smoke-test/p2-v2-ref-probe.src.ts \
   --outfile smoke-test/p2-v2-ref-probe.js --target browser

# Run JSEP spike:
#   http://localhost:8031/p2-v2-spike.html?v=stage4.31-probe18&ingest=off
# Run non-JSEP reference:
#   http://localhost:8031/p2-v2-ref-probe.html?v=stage4.31-probe18

# Capture both runs' __stage431Stats via agentchrome js exec.
```

`make checkall` green (747 pass / 36 skip / 0 fail) — the C++
probe and TS parse blocks have no production impact.

## Captured artifacts

- [`STAGE-4.31-jsep-stats.json`](STAGE-4.31-jsep-stats.json) —
  JSEP spike run. `generatedText = "inonic boso-"` (gibberish,
  unchanged from Stage 4.30); `generatedIds = [297, 8927, 13601,
  29877, 29899]`; per-token decode 322 ms; prefill 621 ms;
  model load 380 ms.
- [`STAGE-4.31-ref-stats.json`](STAGE-4.31-ref-stats.json) —
  non-JSEP reference run. `generatedIds = [3681, 29889, 13, 13,
  29906]` (token 3681 = `" Paris"`, the canonical correct
  continuation of "The capital of France is"); per-token decode
  156 ms; prefill 2469 ms (CPU fallback paths in the legacy
  build); model load 808 ms.

`generatedText` divergence between the two runs is the same as
every prior Stage 4.x: the JSEP path generates gibberish; the
non-JSEP path generates Paris. This run was a measurement, not a
fix attempt — the bug is not yet closed.

## Patch stack delta

**llama.cpp `webllm-browser-patches`:** unchanged at 14
(`ebc7c3d82` Stage 4.29 CPU-side hook still HEAD). Stage 4.31 is
webllm-only.

**webllm working tree:**
- `src/wasm/webgpu-bridge.cpp`: +44/-1 (`<cmath>` include +
  `kqv_out-0` full-tensor probe block inside `node_dump_cb`).
- `smoke-test/p2-v2-spike.src.ts`: +44/-0 (Stage 4.31 parse +
  expose block).
- `smoke-test/p2-v2-ref-probe.src.ts`: +43/-0 (same parse +
  expose block).

## Stage 4.32 — queued probe (P-18-first8-blind branch)

**Goal.** Localize *where in `kqv_out-0`* the JSEP and reference
runs disagree. Aggregate stats are sufficient to *detect* the
divergence; element-wise comparison is required to *characterize*
it (per-row gap vs per-block gap vs per-column gap, contiguous zero
range vs scattered exact-zero indices, etc.). The signature decides
whether the bug is in the kqv MUL_MAT kernel itself, in the
upstream Q×K^T → softmax chain that feeds it, or in the SET_ROWS /
KV-cache plumbing for non-position-0 columns.

### Stage 4.32 paste-and-go bootstrap

```bash
cd /Users/probello/Repos/webllm
git log --oneline -5
#   → <Stage 4.31 TODO closure commit>     docs(TODO): Stage 4.31 closed — queue Stage 4.32 idx-by-idx kqv_out-0 diff
#   → <Stage 4.31 reports commit>          docs(reports): Stage 4.31 closure — Outcome P-18-first8-blind
#   → <Stage 4.31 spike commit>            feat(spike): Stage 4.31 Probe 18 Shape A — kqv_out-0 full-tensor stats
#   → cf134e8                              docs(TODO): Stage 4.30 closed — queue Stage 4.31 widen node_dump_cb for kqv_out-0
#   → 52242a7                              docs(reports): Stage 4.30 closure — Outcome P-17-clean

( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → ebc7c3d82   webllm-browser-patches   (patch stack 14 — unchanged from Stage 4.29-4.31)

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
REF_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-ref-probe.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
[ -n "$REF_TAB" ] || REF_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-ref-probe.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
```

### Stage 4.32 implementation sketch (Probe 19 — element-wise `kqv_out-0` capture)

1. **C++ side (`src/wasm/webgpu-bridge.cpp:node_dump_cb`):**
   add an optional element-wise capture mode armed via a new
   `webllm_arm_kqv_capture(int max_idx_count)` export. When armed
   and `tensor->name == "kqv_out-0"`, allocate a JS-owned
   `Float32Array` via `EM_ASM` and copy `ggml_get_f32_1d(t, i)` for
   `i = 0 .. min(n_elements, max_idx_count)` into it. Push to
   `globalThis.__stage432KqvCapture[<idx>]` keyed by the
   `node_dump_cb` idx (so prefill = 11, decode-1 = 29, etc.).
   Disarm after first capture (or after `n_elements` rows).

   **Cheaper alternative if memory pressure is a concern**: emit a
   `[CHECKPOINT-IDX-DUMP idx=N name=kqv_out-0 row=R first16=[…]]`
   stderr line per row R (12288 / 2048 = 6 rows for prefill, 1 row
   per decode); total ~30 KB stderr per run, well within
   `__stderrLines` budget. Parse offline via Python.

2. **JS spike + ref side:** run both pages, capture
   `__stage432KqvCapture` (or the per-row dump line list), pickle
   to `STAGE-4.32-jsep-kqv.json` / `STAGE-4.32-ref-kqv.json`.

3. **Diff script (`STAGE-4.32-diff.py`):** load both captures,
   compute per-index `|jsep[i] - ref[i]|` for each forward pass,
   find:
   - the first index where the delta exceeds 1e-3 (the *real*
     first-divergent index, replacing first8's idx 0),
   - the longest contiguous span where JSEP is exactly 0 (= the
     "uninitialized region" the abs_min=0 signal predicted),
   - the per-row max-abs-delta (does the bug stripe rows or
     columns?).

4. **Run** at `?v=stage4.32-probe19` on both pages.

### Probe 19b implementation sketch (Shape B: extend full-tensor probe to `kq-0` + `kq_soft_max-0`)

(Run *before* Probe 19 if Shape A's element-wise dump is heavier
than expected — Shape B is a one-line allowlist extension that
tells us whether the upstream Q×K^T also carries first8-window
blindness.)

1. **C++ side:** in `node_dump_cb`, change the
   `std::strcmp(t->name, "kqv_out-0") == 0` gate to a 3-name match
   (`kq-0`, `kq_soft_max-0`, `kqv_out-0`). Same emission format,
   same `[CHECKPOINT-FULL …]` stderr line.
2. **TS side:** no change required — `__stage431Stats` parser
   already handles arbitrary names.
3. **Diff:** compare aggregate stats across the three nodes. If
   `kq-0` already shows a >1e-3 delta on `abs_max`, the cascade
   originates at Q×K^T and `kqv_out-0` is faithfully amplifying.
   If only `kqv_out-0` diverges, the bug is in V × softmax (the
   later mat-mul on the chain).

### Files to read first

- [`STAGE-4.31-RESULT.md`](STAGE-4.31-RESULT.md) — this report.
  P-18-first8-blind classification + JSEP / ref aggregate-stat
  side-by-side.
- [`STAGE-4.27-RESULT.md`](STAGE-4.27-RESULT.md) — the smoking-gun
  cascade table (row 9 `kq-0` first8 Δ=0.0119, row 11 `kqv_out-0`
  first8 Δ=0.0; row 12 `attn_out-0` Δ=4.77e-3 first ≥1e-3
  checkpoint). Stage 4.31 has just demoted that row-12 number from
  "first divergence point" to "first divergence projected through
  first8 — the actual first divergence is at row 11 outside first8".
- `src/wasm/webgpu-bridge.cpp:node_dump_cb` — current callback site.
  Stage 4.32 either extends the kqv-only probe to per-element
  capture, or extends the name allowlist to add `kq-0` /
  `kq_soft_max-0`.
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:compute_op` —
  the JSEP backend's MUL_MAT dispatch site. If Stage 4.32 narrows
  the divergence to the V × softmax matmul specifically, this is
  the next read.

### Risk register

1. **`kqv_out-0` element-wise capture is large.** 12 288 × 4 B =
   48 KiB for the prefill, plus 5 × 8 KiB = 40 KiB across decode —
   88 KiB total per run, 176 KiB across both runs. That fits in
   `globalThis` without trouble, but `agentchrome js exec` JSON
   serialisation can be slow on multi-MiB payloads. Mitigation:
   use the per-row stderr dump alternative (6 + 5 = 11 lines per
   run), parse offline with Python.
2. **Decode-step alignment between JSEP and ref runs.** Stage
   4.31 already shows the two runs generate different tokens at
   step 0 (JSEP → 297, ref → 3681). Decode steps 1+ are NOT
   directly comparable on the JSEP side because they're feeding
   the model different inputs. Stage 4.32's first-divergence
   localization should focus on **idx 11 (prefill)** — the only
   forward pass where both runs receive the same input — and
   accept that decode-step divergences are downstream of the
   prefill-step divergence already. (The decode `kqv_out-0`
   divergences in this report are *consequence*, not cause.)
3. **Reference path drift across patch-stack growth.** Stage 4.27
   risk #3 carries forward: patch-stack 13 → 14 (Stage 4.29's
   CPU-side hook) does not affect the reference build's
   correctness, so the ref `__stage431Stats` should be the
   authoritative truth. Sanity-check by re-running Stage 4.27's
   first-divergent-op cascade on the current build before trusting
   Stage 4.32 deltas.
4. **`abs_min = 0.0` could be a false positive.** With f32
   activations and 12 288 elements, the chance of a real element
   landing at exactly 0.0 is non-zero. The dispositive measure is
   the *count* of exact zeros, not the *presence* of one. Stage
   4.32's element-wise capture should include a per-pass
   zero-count + zero-index-list to distinguish "one accidental
   zero" from "a contiguous block of zeros".

### Exit criteria — Stage 4.32 closes when documented in `STAGE-4.32-RESULT.md`

- Element-wise `kqv_out-0` (prefill, idx=11) deltas computed for
  both runs; first-divergent-index identified; structural pattern
  classified (contiguous zero-range, striped, scattered, etc.).
- One of:
  - **P-19-row-bounded**: divergence aligns to row boundaries
    (every 2048-th index OR every 6-th index). MUL_MAT tile-bounds
    bug. Stage 4.33 instruments the kqv MUL_MAT dispatch in
    `ggml-jsep.cpp:compute_op` to capture the workgroup grid
    size + dst write coverage.
  - **P-19-block-bounded**: divergence aligns to some other
    block size (16, 32, 64). Probably a sub-warp / subgroup
    coverage bug. Stage 4.33 inspects the WGSL kernel's
    `@workgroup_size` and write predicates.
  - **P-19-upstream-cascade**: `kqv_out-0` divergence pattern
    matches what V × (softmax(Q×K^T)) would produce if `kq-0` or
    `kq_soft_max-0` were corrupt, not from a kqv kernel bug.
    Stage 4.33 = Probe 19b (extend full-tensor probe to `kq-0`
    and `kq_soft_max-0`); the kqv kernel exonerates.
- Stage 4.33 paste-and-go brief queued for the chosen branch.

### Branch on Probe 19 outcome

- **P-19-row-bounded** (suspect: kqv MUL_MAT tile-bounds bug):
  Stage 4.33 captures the WGSL dispatch + grid + bind-group
  layout for the `kqv_out-0` MUL_MAT specifically.
- **P-19-block-bounded** (suspect: WGSL workgroup coverage):
  Stage 4.33 inspects `src/inference/jsep/ops/matmul.ts` for
  the V × softmax shape's pipeline selection logic.
- **P-19-upstream-cascade** (suspect: Q × K^T or softmax):
  Stage 4.33 = Probe 19b, extending the full-tensor probe up
  the attention sub-graph.
