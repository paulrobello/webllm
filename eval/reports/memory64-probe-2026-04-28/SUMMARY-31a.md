# §31a — MEMORY64 cap probe (sub-probe)

**Date:** 2026-04-28
**Parent probe:** [`SUMMARY.md`](./SUMMARY.md) (§31, 2026-04-28)
**Spec follow-up scope:** [§6 of `2026-04-28-memory64-cap-probe-design.md`](../../../docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md) — "narrow follow-up sub-probe to investigate the BigInt ABI gap before committing more surface."

---

## 1. Headline

- ✅ **Phase 1 PASS.** ASYNCIFY × MEMORY64 round-trip continues to work
  (no regression from the parent probe; init wall = 1.6 ms).
- ✅ **Phase 2 PASS.** `_bridge_malloc(16n)` returns **BigInt**
  (`0xac6548`); 4×F32 heap round-trip is byte-equal. Stdlib `_malloc`
  diagnostic confirms the §31 asymmetry persists in the same build —
  `_malloc(16n) → typeof=number value=0xac6548`. The bridge wrapper is
  the targeted fix.
- ✅ **Phase 3 PASS.** Sequential 1 GiB `_bridge_malloc` loop allocated
  **15 × 1 GiB = 16,106,127,360 bytes ≈ 15.00 GiB** with per-allocation
  page commit (64 KiB fill at offset 0). Iteration 15 returned BigInt
  `0n` (NULL — allocator hit the configured `-sMAXIMUM_MEMORY=16GB`
  ceiling with overhead reserved). All 15 allocations were freed
  cleanly via `_bridge_free`.
- ✅ **Phase 4 PASS.** Post-probe `_webgpu_init` / `_webgpu_shutdown`
  succeeded — runtime is healthy after exercising 15 GiB of heap.

## 2. Decision-rule branch (parent spec §5.1)

> **"≥8 GiB → promote to full bridge migration."**

Cap = 15 GiB measured, comfortably above the 8 GiB threshold.

The ceiling is `MAXIMUM_MEMORY=16GB`, not the hardware/Chrome wasm64
limit — the loop simply ran out of configured headroom. Raising
`MAXIMUM_MEMORY` would let us probe the actual upper bound, but the
measured 15 GiB already covers every model size that fits the
**2026-04-28 30B project ceiling**:

| Target              | Quant     | Approx. weights | Fits 15 GiB? |
|---------------------|-----------|----------------:|-------------:|
| 8B (current fleet)  | Q4_K_S    | ~4.5 GiB        | ✅ ample     |
| 13B                 | Q4_K_S    | ~7.4 GiB        | ✅           |
| 30B                 | IQ3_M     | ~12.8 GiB       | ✅ tight but fits |

(Add KV cache + activations on top — for 30B IQ3_M with seq=2048 KV the
working set is ~14 GiB, very close to the ceiling. A `MAXIMUM_MEMORY`
bump may still be needed before that target is comfortable. 8B and 13B
have substantial margin.)

## 3. Probe result blob

```json
{
  "phase1": "ok",
  "phase2": "ok",
  "phase3_status": "ok",
  "phase3_cap_bytes": 16106127360,
  "phase3_iterations": 15,
  "phase4": "ok",
  "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
  "module_bytes": 16777216,
  "init_wall_ms": 1.6,
  "started_at": "2026-04-28T16:39:10.344Z",
  "finished_at": "2026-04-28T16:39:10.363Z"
}
```

Wall-clock: 19 ms total (initialised 16 MiB heap → grew to 15 GiB →
freed → re-init).

## 4. Visible #log (verbatim)

```
Loading webllm-wasm-mem64.js…
Module loaded. Initial heap: 16777216 bytes
--- Phase 1: ASYNCIFY × MEMORY64 round-trip ---
_webgpu_init OK in 1.6ms
_webgpu_shutdown OK
PHASE 1 PASS
--- Phase 2: BigInt ABI smoke (bridge_malloc) ---
_ctx_create returned idx=0
_tensor_new_1d → typeof=bigint value=0xac65a0
_malloc(16n) [stdlib, diagnostic] → typeof=number value=0xac6548
_bridge_malloc(16n) → typeof=bigint value=0xac6548
Heap round-trip OK (4×F32 byte-equal)
PHASE 2 PASS
--- Phase 3: Cap probe (sequential 1 GiB bridge_malloc) ---
Iter 0: bridge_malloc(1 GiB) OK, total committed=1.00 GiB
Iter 1: bridge_malloc(1 GiB) OK, total committed=2.00 GiB
...
Iter 14: bridge_malloc(1 GiB) OK, total committed=15.00 GiB
Iter 15: _bridge_malloc returned 0 / wrong type (typeof=bigint)
PHASE 3 cap = 15.00 GiB across 15 iters
Freed 15/15 cap-probe allocations
--- Phase 4: Post-cap-probe shutdown health ---
_webgpu_init (post-probe) OK
_webgpu_shutdown (post-probe) OK
PHASE 4 PASS
=== HARNESS COMPLETE ===
```

(Iter 15's "wrong type (typeof=bigint)" log is the harness's combined
type+zero check — `0n` is BigInt zero, which the type guard correctly
treats as allocation failure. Tighten the message in a future polish
pass; not load-bearing.)

## 5. Implementation delta

Three files changed, all minimal:

- `src/wasm/webgpu-bridge.cpp` — added `bridge_malloc(size_t)` and
  `bridge_free(void*)` thin shims around stdlib `malloc`/`free`. The
  Emscripten linker now sees explicit-signature exports and emits
  correct BigInt-marshaling JS shims.
- `src/wasm/CMakeLists.txt` — added `_bridge_malloc,_bridge_free` to
  `EXPORTED_FUNCTIONS` (applied to both `webllm-wasm` and
  `webllm-wasm-mem64` since they share the list).
- `smoke-test/mem64-probe.html` — Phase 2 + Phase 3 now call
  `_bridge_malloc` / `_bridge_free`; kept a one-shot diagnostic call
  to stdlib `_malloc` in Phase 2 to confirm the asymmetry persists.

Net code change: **+18 LOC** across the three files.

## 6. Lever status update

§31's "**NOT closed**" status flips to **"viable, ready for full bridge
migration scoping"**. The original §31 lever bullets:

- ASYNCIFY axis: cleared in §31 — *unchanged*.
- BigInt ABI: ~~targeted gap, fixable with a few-line bridge change~~
  → **closed by §31a — `bridge_malloc` / `bridge_free` work; the gap
  is stdlib-specific, not architectural.**
- Cap measurement: ~~deferred to follow-up sub-probe~~ → **§31a measured
  15 GiB at `MAXIMUM_MEMORY=16GB`** (run actual upper bound when
  `MAXIMUM_MEMORY` raised, if needed).

## 7. What §31a does NOT do

This sub-probe **does not** migrate the production `webllm-wasm` build
to MEMORY64. That migration is a larger scope — it requires:

- Replacing every `m._malloc` / `m._free` call site in `src/inference/`
  and `src/wasm/` TS code with `m._bridge_malloc` / `m._bridge_free`
  (or accepting the stdlib asymmetry and fixing JS-side callers to
  cope with `Number` returns under wasm64).
- Auditing all bridge call sites for `int32_t` size/offset parameters
  that need promotion to `int64_t` / `size_t` for >4 GiB working sets.
  (Spot inspection of `webgpu-bridge.cpp` already turned up
  `tensor_set_data(void*, const void*, int32_t size)` — `int32_t` size
  caps a single transfer at 2 GiB; not a problem for individual model
  weights but the signature is fragile.)
- Updating the model loader's GGUF streaming path so byte offsets and
  buffer sizes stay BigInt across the JS→WASM boundary.
- Re-running the full smoke + bench-inference + bench-profile gates
  under MEMORY64 to confirm zero perf regression for the existing
  ≤4 GiB fleet.
- Decision on whether to ship a single MEMORY64 binary (drops the
  4 GiB-cap fast path; potential pointer overhead) or to ship two
  binaries (3.5 → 7 MiB total bundle payload; deploy-time selection).

This is the **"P2-class follow-up spec"** the original §31 spec §3.1
gestured at. **Not in scope for §31a.** Open it as a separate
spec/plan cycle when a deployment ask demands a 13B or 30B target.

## 8. Reproduction

```bash
make mem64-probe
# Open the printed cache-busted URL in the existing agentchrome tab.
agentchrome --port <PORT> js exec --tab <TAB_ID> \
  --code 'JSON.stringify(window.__memory64ProbeResult)'
```

Expected result blob fields: `phase1=ok`, `phase2=ok`,
`phase3_status=ok`, `phase3_cap_bytes ≈ 16e9`, `phase3_iterations=15`,
`phase4=ok`. Wall-clock <50 ms.

## 9. Probe economics

- Implementation: 5 minutes (3-file edit + Make rebuild).
- Build: ~30 seconds (incremental; ggml archives cached from §31's run).
- Run: 19 ms wall on the page; <5 seconds end-to-end including
  agentchrome navigation + result capture.
- Saved: a multi-day "should we even attempt MEMORY64?" speculation
  cycle that the parent spec §3 originally framed as the worst-case
  path. The two-step probe pattern (parent + sub-probe) cost
  ~6 hours total wall and produced a measured cap value plus a
  retired risk axis.
