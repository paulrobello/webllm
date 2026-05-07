# Stage 4.27 — Cascade-source localization via re-captured `__stage417Checkpoints`

**Status:** CLOSED 2026-05-07. **Outcome A confirmed:** the prefill cascade
trajectory is **structurally identical** to Stage 4.17 — the Stage 4.18 → 4.26
patch stack growth (12 → 13) did **not** shift any of the smoking-gun
checkpoints. The bug is exactly where Stage 4.17 located it: small upstream
matmul deltas at `Qcur-0` / `Kcur-0` (~3-5e-4) compound through `attn_out-0`
(4.77e-3) and explode at `ffn_norm-0` (0.183), then drift through the
remaining 21 unmonitored layers to land at ~6 magnitude in `result_norm` /
`result_output`. Stage 4.26 has since ruled out matmul-precision as the
cascade source, so Stage 4.28 must pivot from "what matmul is imprecise"
to "what *structural* difference between scheduler routings makes the spike
diverge". Three suspect categories survive Stage 4.26's matmul-precision
closure: (i) `ffn_norm.weight` gain-vector mis-load on the JSEP host_mirror,
(ii) output-projection `attn_output.weight` byte-integrity (Stages 4.20/4.21
only verified `attn_q.weight` + `attn_k.weight`), (iii) first8-window
blindness on `kqv_out-0` hiding an upstream FA / softmax / SET_ROWS
divergence.

**Patch stack:** 13 (unchanged from Stage 4.26 closure).
**webllm tip:** `3d8853e` (`docs(TODO): add Phase 3 trajectory assessment before Stage 4.27 brief`).
**llama.cpp tip:** `ef89f9314` on `webllm-browser-patches`.
**WASM build:** unchanged (no rebuild needed).

## Headline

| Metric | Stage 4.17 (2026-05-07 morning) | Stage 4.27 (2026-05-07 afternoon) |
|---|---:|---:|
| First ≥1e-3 prefill checkpoint | `attn_out-0` 4.77e-3 | `attn_out-0` **4.773e-3** ✓ |
| First ≥1e-1 prefill checkpoint | `ffn_norm-0` 1.83e-1 | `ffn_norm-0` **0.183250** ✓ |
| `result_norm` cumulative Δ | 5.83 | **5.826660** ✓ |
| `result_output` (logits) cumulative Δ | 6.61 | **6.607800** ✓ |
| Generated text on JSEP path | `"inonic boso-"` | `"inonic boso-"` ✓ (unchanged) |

The post-Stage-4.4/4.5/4.16 host-mirror writeback fixes did **not** close the
cascade. The bug is **not** a writeback-completion gap — it is structurally
upstream of `attn_out-0` and amplified by the RMSNorm at `ffn_norm-0`.

## Smoking-gun table (prefill forward pass, 6-token prompt)

```
idx name              ne                    jsep_be   max_abs_delta   jsep[0]    ref[0]
  0 attn_norm-0       [2048, 6, 1, 1]       CPU       0.000000     0.001405   0.001405  ← bit-identical
  1 Qcur-0            [2048, 6, 1, 1]       jsep_buf  0.000524    -0.016189  -0.016286  ← Stage 4.19 number
  2 Qcur-0            [64, 32, 6, 1]        jsep_buf  0.000524    -0.016189  -0.016286
  3 Qcur-0            [64, 32, 6, 1]        CPU       0.000524    -0.016189  -0.016286  ← host_mirror == GPU readback
  4 Vcur-0            [256, 6, 1, 1]        CPU       0.000000    -0.000799  -0.000799  ← V matmul bit-identical
  5 Vcur-0            [64, 4, 6, 1]         CPU       0.000000    -0.000799  -0.000799
  6 Kcur-0            [256, 6, 1, 1]        jsep_buf  0.000338     0.017701   0.017835
  7 Kcur-0            [64, 4, 6, 1]         jsep_buf  0.000338     0.017701   0.017835
  8 Kcur-0            [64, 4, 6, 1]         CPU       0.000338     0.017701   0.017835  ← host_mirror == GPU readback
  9 kq-0              [256, 6, 32, 1]       jsep_buf  0.011940    -0.005891  -0.005892  ← Q×K^T amplifies upstream Δ via 256-wide dot
 10 kq_soft_max-0     [256, 6, 32, 1]       CPU       0.000000     1.000000   1.000000  ← FIRST8-WINDOW BLINDNESS (causal mask)
 11 kqv_out-0         [2048, 6, 1, 1]       CPU       0.000000    -0.000800  -0.000800  ← FIRST8-WINDOW BLINDNESS (V[pos=0] only)
 12 attn_out-0        [2048, 6, 1, 1]       jsep_buf  0.004773    -0.000087  -0.003757  ← FIRST NON-TRIVIAL Δ (out_proj × kqv)
 13 ffn_norm-0        [2048, 6, 1, 1]       CPU       0.183250    -0.023293  -0.070527  ← +38× JUMP (RMSNorm + gain)
 14 ffn_out-0         [2048, 6, 1, 1]       CPU       0.042198     0.000945   0.001552
 15 l_out-0           [2048, 6, 1, 1]       CPU       0.040474    -0.000442  -0.003505
 16 result_norm       [2048, 1, 1, 1]       CPU       5.826660    -3.567110   2.259550  ← post-22-layer cumulative drift
 17 result_output     [32000, 1, 1, 1]      CPU       6.607800    -8.385740  -7.266420  ← top-1 flips
```

(Decode steps 1-5 — idx 18 onward — show massive divergences but those are
downstream of the prefill cascade and therefore not informative for
localizing the cascade source.)

## Cross-reference against Stage 4.17 + Stage 4.26 findings

- **Stage 4.17 (matmul-precision hypothesis, partial closed):** queued
  Stage 4.18 against the Q4_0 matmul kernel. Stage 4.18 ran a Q4_0
  production-shape sweep and produced `maxAbsDelta = 4.768e-7` (single-ULP)
  on synthetic-replay shapes — the WGSL Q4_0 kernel is bit-clean at
  isolated shapes. Stage 4.22 (Probe 10) re-ran the sweep on captured
  *production* bytes and again produced `maxAbsDelta = 4.768e-7`,
  exonerating the Q4_K matmul on TinyLlama (which is *actually* Q4_K +
  Q6_K despite the `Q4_0` filename).
- **Stage 4.26 (libllama precision shim):** confirmed libllama's CPU
  `quantize_row_q8_K` → `vec_dot_q4_K_q8_K` produces `4.178e-2` from f64
  truth on captured production Q-projection inputs vs webllm's WGSL kernel
  at `7.94e-6` from truth. **webllm is more accurate than the reference
  path that decodes correctly.** Q-projection matmul precision is therefore
  not the cascade source — the 5.24e-4 spike-vs-ref Δ is the difference
  between two equally-valid (and *equally-accurate*) f32 matmul accumulation
  orders, not a precision bug in the WGSL kernel.
- **Net:** Stages 4.18-4.26 systematically eliminated all matmul-precision
  hypotheses for both Q-projection (idx 1) and the upstream of `attn_out-0`
  (idx 12). The Stage 4.27 diff confirms those eliminations did not move
  the needle: the cascade trajectory is identical, so something *other*
  than matmul precision is producing the 4.77e-3 jump at `attn_out-0` and
  the 38× amplification at `ffn_norm-0`.

## Surviving suspect categories

The remaining viable hypotheses for what produces `attn_out-0` Δ=4.77e-3
and `ffn_norm-0` Δ=0.183 — given matmul precision is dead — are:

1. **Output-projection (`attn_output.weight`) byte-integrity gap.**
   Stages 4.20 / 4.21 verified the GGUF parser → ggml allocator →
   `set_tensor` → `Module.jsepWrite` → `device.queue.writeBuffer` chain
   is byte-exact for `blk.0.attn_q.weight` and `blk.0.attn_k.weight`.
   `blk.0.attn_output.weight` was **never directly probed**. If the
   output-projection weight is mis-uploaded for any reason (offset bug
   specific to the `attn_output` tensor name pattern, allocator slot
   collision, etc.), the Δ would surface exactly here — at `attn_out-0`,
   independent of upstream `Qcur` / `Kcur` health.
2. **`ffn_norm.weight` gain-vector mis-load.** The +38× jump from
   `attn_out-0` Δ=4.77e-3 to `ffn_norm-0` Δ=0.183 is unusually large for
   a pure RMSNorm (which only divides by `sqrt(mean(x²)+ε)` and multiplies
   by per-channel gain). Stage 4.3 already identified the per-channel
   gain MUL as a load-bearing CPU-fallback step that, post-Stage-4.4,
   reads from `host_mirror`. If the gain vector in the JSEP path's
   `host_mirror` does not match the GGUF weight bytes (allocator
   double-write, missing `set_tensor` hook fire, type-converter mis-step),
   the amplification is mathematically natural. **This is the single
   most likely structural failure given the cascade shape.**
3. **First8-window blindness on `kqv_out-0`.** Both Stage 4.18 and
   Stage 4.27 see `kqv_out-0` first8 = `V[pos=0]` because the causal
   mask pins position-0's softmax to `[1, 0, 0, …]`. The Δ over
   *positions 1-5* is unmonitored; it could be ≥1e-3 and we would not
   see it. If `kqv_out-0` actually carries a Δ ≥1e-3 outside first8,
   the cascade source is upstream of `attn_out-0` (most likely the
   FA / scaled mat-vec for non-position-0 columns or the SET_ROWS that
   writes K/V into the KV cache).

Suspects already eliminated by Stages 4.4 / 4.5 / 4.20 / 4.21 / 4.22 /
4.24 / 4.25 / 4.26:

- ❌ `attn_q.weight` upload byte-integrity (Stage 4.20 / 4.21).
- ❌ `attn_k.weight` upload byte-integrity (Stage 4.20 / 4.21).
- ❌ Q4_0 matmul kernel precision at production shapes (Stage 4.18 / 4.22).
- ❌ Q4_K dequant correctness vs libllama (Stage 4.24 — bit-clean over
  4,194,304 elements).
- ❌ WGSL accumulation precision (Stage 4.25 — Kahan accumulator produced
  bit-identical output to baseline; webllm at 7.94e-6 from f64 truth).
- ❌ libllama Q4_K×Q8_K matmul precision (Stage 4.26 — libllama is the
  imprecise side at 4.178e-2 from f64 truth).
- ❌ Canonical NaN cascade in CPU-fallback (Stage 4.4).
- ❌ GPU→host writeback completeness (Stage 4.5).

## Stage 4.28 — queued probe

**Target:** `ffn_norm.weight` (per-layer RMSNorm gain vector, layer 0)
byte-integrity on the JSEP path.

**Hypothesis (highest-prior).** The 38× amplification at `ffn_norm-0` is
not an RMSNorm normalization quirk (`attn_out-0` first8 magnitudes are
~1e-4 — small but not pathological) but a gain-vector mis-load: the
`ffn_norm.weight` (per-channel scalar multiplied after `rms_norm`) on the
JSEP host_mirror disagrees with the GGUF weight bytes that the reference
path consumes.

**Probe 15 sketch.**
1. Extend the cb_eval allowlist in `src/wasm/webgpu-bridge.cpp::node_dump_cb`
   to fire on the *weights* `blk.0.ffn_norm.weight`, `blk.0.attn_output.weight`,
   `blk.0.ffn_gate.weight`, `blk.0.ffn_up.weight`, `blk.0.ffn_down.weight`
   — capturing first8 + FNV-1a-32 hash of the full byte range.
2. Add a JS-side reference hash from `GgufParser` (Stages 4.20 / 4.21
   already establish this pattern) for the same five weights.
3. Run the spike + ref harnesses with the extended allowlist; collect
   `STAGE-4.28-jsep-weight-hashes.txt` + `STAGE-4.28-ref-weight-hashes.txt`.
4. Diff against the JS-side reference. Three outcomes:
   - **All five weights byte-exact:** rules out gain-vector / output-proj
     weight mis-load. Suspects (1) and (2) close. Pivot Stage 4.29 to
     suspect (3) — extend `node_dump_cb` to capture max-abs over the
     full tensor (or sample positions 1-5 of `kqv_out-0`) to defeat
     first8-window blindness on `kqv_out-0`.
   - **`ffn_norm.weight` hash mismatch:** Outcome confirmed. Cascade
     source localized to gain-vector load. Stage 4.29 captures the
     buggy upload byte trajectory and traces back to the responsible
     `set_tensor` / `host_mirror` fork.
   - **`attn_output.weight` hash mismatch:** suspect (1) confirmed.
     Output-projection weight is mis-uploaded. Stage 4.29 traces the
     specific upload path that diverges from the bit-clean
     `attn_q.weight` / `attn_k.weight` path.

**Cost estimate:** ~1 probe (~30-45 min) — the cb_eval allowlist
extension is mechanical, JS-side ref hashes already established in
Stages 4.20 / 4.21, no llama.cpp patches needed.

**Risk register.**
- The `node_dump_cb` callback fires on *compute outputs*, not on weight
  tensors directly. Need to either:
  - (a) Hook the weight FNV at `ggml_backend_jsep_set_tensor` time
    instead — fits Stage 4.20's pattern.
  - (b) Add a synthetic `cb_eval` fire on weight tensors at first
    `llama_decode` entry by wrapping each weight in a no-op `ggml_view`
    that gets scheduled — heavier, only if (a) is blocked.
- If all five weights pass byte-exact, the analysis falls back to
  RMSNorm *kernel* precision (gain MUL + epsilon + sqrt rounding) —
  but Stage 4.5 already verified the CPU-fallback host_mirror reads
  from the right buffer post-RMSNorm, so this is a tail risk.

## Files captured

- `STAGE-4.27-jsep-checkpoints.txt` (108 entries; spike, JSEP path,
  patch stack 13).
- `STAGE-4.27-ref-checkpoints.txt` (108 entries; non-JSEP reference
  via `webllm-wasm.js`).
- `STAGE-4.27-diff.txt` (output of `STAGE-4.18-diff.py` — the
  `STAGE-4.17-diff.py` does not parse the post-Stage-4.18 `backend=…`
  field).

## Branch decision per the Stage 4.27 brief

Per the brief's exit-criteria branch table:

> **Diff matches Stage 4.17 pattern** (first ≥1e-3 at `attn_out-0`,
> ≥1e-1 at `ffn_norm-0`): the post-Stage-4.4/4.5/4.16 patch stack didn't
> close the cascade; the bug is structurally identical to what Stage 4.17
> captured. Stage 4.28 deep-dives `attn_out-0` (output projection matmul)
> — re-use Probe 14 on its captured bytes.

The brief's recommended Stage 4.28 follow-up was "deep-dive `attn_out-0`
output-projection matmul" — but Stage 4.26 has *since* ruled out matmul
precision as the cascade source. Adjust Stage 4.28's framing accordingly:
keep `attn_out-0` as the deep-dive target, but pivot from a precision
probe (Probe 14 pattern) to a **byte-integrity probe** on the output-
projection weight + the gain vector, mirroring Stages 4.20 / 4.21's
`attn_q.weight` / `attn_k.weight` pattern. The gain-vector mis-load
hypothesis (suspect 2) is the single most likely structural failure
given the cascade shape (38× amplification at `ffn_norm-0` is the
load-bearing signal), so it gets primary billing in the probe.
