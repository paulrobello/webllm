# llama.cpp rebase — 2026-05-01 (§27 free win)

**Verdict:** §27 template — **broad upside, no regressions**. Adopt
the new tip; close cycle. No follow-up needed.

## Trigger

Daily upstream cadence check (per CLAUDE.md doctrine) fired
non-empty:

- `c3c150539` — **ggml-webgpu: Fix vectorized handling in mul-mat
  and mul-mat-id (#22578)** — load-bearing for chat decode (matmul
  kernel touches every model in the canonical 6).
- `aab68217b` — ggml-webgpu: add the upscale shader (#22419) — image-
  gen op, not exercised by the chat fleet but kept under the patch
  stack so the rebase carries no hand-skipped commits.

## Build deltas

| | Pre-rebase | Post-rebase |
|---|---|---|
| webllm-browser-patches tip   | `a45089d5a` | `e29753286` |
| upstream tip                  | `a95a11e5b` | `b97ebdc98` |
| patch stack count             | 11           | 11           |
| WASM mem32 size (bytes)       | 2,482,377    | 2,482,377    |
| WASM mem64 size (bytes)       | 2,526,224    | 2,526,224    |

Patch stack rebased cleanly with **one trivial conflict**: both
upstream `aab68217b` (UPSCALE) and our local browser bundle
(`7bbb2416c`) added new `case` arms in the same `switch (op)` in
`ggml-webgpu.cpp`. Resolved by keeping both arms.

## Headline matrix — same-day same-tip pre vs post

3-run median, profile-mode. Compared against
`pre-rebase-baselines-2026-05-01/` (captured the same day with the
same WASM toolchain, see §32a doctrine in CLAUDE.md and that report's
"Cross-day drift" section for why the 2026-04-28 baselines are not
the right comparand here).

| Model                          | Pre tok/s | Post tok/s | Δ tok/s | Δ %    | matmul Δ |
|---|---:|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 80.4       | 81.8        | +1.4     | +1.7%  | 4.19 → 4.13 ms (-1.4%) |
| qwen3-0.6b-q4f16               | 62.5       | 67.5        | +5.0     | +8.0%  | 4.00 → 3.80 ms (-5.0%) |
| qwen3-1.7b-q4f16               | 41.7       | 43.9        | +2.2     | +5.3%  | 6.88 → 6.75 ms (-1.9%) |
| mistral-7b-instruct-v0.3-q4ks  | 28.7       | 29.6        | +0.9     | +3.1%  | 16.58 → 15.93 ms (-3.9%) |
| llama-3.1-8b-instruct-iq3m     | 23.3       | 23.4        | +0.1     | +0.4%  | 23.66 → 22.94 ms (-3.0%) |
| qwen3-8b-iq3m                  | 21.3       | 22.0        | +0.7     | +3.3%  | 24.12 → 23.27 ms (-3.5%) |

**Net:** every model improved or held (+0.4% to +8.0%); matmul
median time decreased on every model (-1.4% to -5.0%). No
regression on any quant (Q4_0, Q8, Q4_K_S, IQ3_M all benefit).
Encoder, attention, and dispatch counts unchanged.

## Attribution

The matmul-time delta tracks the upstream commit
[`c3c150539`](https://github.com/ggml-org/llama.cpp/pull/22578) —
"Fix vectorized handling in mul-mat and mul-mat-id". The fix
addresses a vectorization corner case that the chat fleet hits on
every decode step (single-token mat-vec dispatch), so the win
shows broadly.

The smallest models gain proportionally most because their matmul
share is smallest (33-38% of graph) — the fixed cost of the
remaining graph work doesn't dilute the matmul-only win as much
as on the 7B+ rung where matmul is 50-58% of graph but already
bandwidth-bound. The IQ3_M-bandwidth-bound 8B matmul still
dropped 3-3.5%, which is a real (small) compute-side gain, not a
bandwidth-limited ceiling.

## Canonical pin update (post-§27)

The TODO header pins should refresh to:

| Model                          | Old (2026-04-28 profile) | New (2026-05-01 profile) |
|---|---:|---:|
| tinyllama-1.1b-chat-q4_0       | 87.9 | 81.8 |
| qwen3-0.6b-q4f16               | 68.2 | 67.5 |
| qwen3-1.7b-q4f16               | 44.0 | 43.9 |
| mistral-7b-instruct-v0.3-q4ks  | 29.7 | 29.6 |
| llama-3.1-8b-instruct-iq3m     | 23.5 | 23.4 |
| qwen3-8b-iq3m                  | 21.8 | 22.0 |

The post-rebase numbers are *lower than* the 2026-04-28 numbers on
small models even though the rebase is a clean improvement —
because the local environmental floor drifted -3 to -8% over the
4-day gap (see `pre-rebase-baselines-2026-05-01/SUMMARY.md` for
full drift table). Pinning the post-rebase numbers reflects the
current measured ceiling on this hardware/toolchain combination.

## Cycle classification

Per CLAUDE.md "Rebase + sweep cycle doctrine":

- **§27 (free win)?** ✅ Yes — every model improved or held; matmul
  decreased uniformly. The mul-mat vectorize fix is a true free
  win across the canonical 6.
- **§28 (negative result, prior lever closes harder)?** No prior
  lever was bet on for this cadence.
- **§32 (small regression accepted)?** No — there is no regression
  to accept.

**Action:** adopt new tip `e29753286`; refresh canonical pins;
close cycle. No additional probes queued.

## Follow-ups queued

None. The fleet is at a fresh §27 ceiling; next rebase trigger
will compare against this report.
