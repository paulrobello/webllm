# §31b — `MAXIMUM_MEMORY` upper-bound probe

**Date:** 2026-04-28
**Parent probes:** [`SUMMARY.md`](./SUMMARY.md) (§31), [`SUMMARY-31a.md`](./SUMMARY-31a.md) (§31a, 15 GiB measured at `MAXIMUM_MEMORY=16GB`).
**Goal:** find Chrome's actual wasm64 runtime cap above the §31a configured-ceiling-bound 15 GiB.
**Procedure:** bump `-sMAXIMUM_MEMORY` from `16GB` → `64GB` in the `webllm-wasm-mem64` ctor block of `src/wasm/CMakeLists.txt`; bump `mem64-probe.html` Phase 3 loop from `i < 16` → `i < 64`; rebuild.

---

## 1. Headline

**Negative result — Chrome's runtime cap remains unmeasurable from this toolchain.** The probe failed at link time with a hard cap enforced by Emscripten 5.0.6's `wasm-ld`:

```
wasm-ld: error: maximum memory too large, cannot be greater than 17179869184
```

`17179869184` bytes = **16 GiB exactly** = 2^34 = 2^18 wasm pages (64 KiB each). The linker rejects any `--max-memory` greater than 16 GiB at link time, regardless of the runtime wasm64 memory model's theoretical 256 TiB ceiling (per the wasm spec, page count is u32 → 2^32 × 64 KiB).

§31a's 15 GiB measured cap was therefore already at the **configured ceiling**, with the remaining 1 GiB reserved for allocator overhead. The "configured-ceiling-bound, not hardware-bound" framing in §31a is correct: but the configuration ceiling is **not a project knob**, it's an Emscripten toolchain ceiling. Bumping it requires a different toolchain or a patched wasm-ld.

## 2. Build error (full)

```
[100%] Linking CXX executable webllm-wasm-mem64.js
wasm-ld: error: maximum memory too large, cannot be greater than 17179869184
em++: error: '/Users/probello/emsdk/upstream/bin/wasm-ld -o webllm-wasm-mem64.wasm
  ... --max-memory=68719476736 --initial-memory=16777216 ...' failed (returned 1)
```

`--max-memory=68719476736` = 64 GiB = the value Emscripten passed through from `-sMAXIMUM_MEMORY=64GB`. Linker rejects.

Emscripten / wasm-ld version: **5.0.6** (`6ea9c28c38cdd40c1032fa04400c9d16230ee180`). Probe-binary configuration: `-sMEMORY64=1 -sWASM_BIGINT=1` (otherwise stock).

## 3. Implications for the §31a "15 GiB cap" framing

§31a §2 wrote: *"The ceiling is `MAXIMUM_MEMORY=16GB`, not the hardware/Chrome wasm64 limit — the loop simply ran out of configured headroom."* That framing is technically accurate but understates the constraint: **the 16 GiB ceiling is not an arbitrary configuration choice, it is the maximum value the toolchain accepts.**

Updated framing:

| Layer                | Cap               | Adjustable? |
|----------------------|-------------------|-------------|
| Wasm spec (memory64) | 256 TiB (2^48 B)  | n/a         |
| Chrome v8 wasm64     | unknown (≥15 GiB) | not in our toolchain |
| Emscripten 5.0.6 wasm-ld `--max-memory` | **16 GiB** | **patch wasm-ld or wait for upstream lift** |
| §31a configured `MAXIMUM_MEMORY` | 16 GiB | ⬆ ceiling |
| §31a measured allocator cap | 15 GiB | (1 GiB overhead reserved) |

The bottom three layers are now characterized; the top two remain opaque from this toolchain.

## 4. Implications for the 30B project ceiling

The 30B IQ3_M scenario is **tighter than §31a suggested**:

| Component                         | Approx. size         |
|-----------------------------------|---------------------:|
| 30B IQ3_M weights                 | 12.8 GiB             |
| KV cache @ seq=2048, ~64 layers   | ~1.5–2.0 GiB         |
| Activations + scratch             | ~0.5–1.0 GiB         |
| **Total working set**             | **~14.8–15.8 GiB**   |
| Configured ceiling (§31a)         | 15 GiB               |
| Toolchain ceiling (§31b)          | 16 GiB               |

A long-context 30B IQ3_M deployment would land at **the toolchain ceiling within margin of error**. There is no headroom to bump `MAXIMUM_MEMORY` higher without lifting the linker constraint.

Mitigations, ranked:

1. **Lower-bit quant.** IQ2_XXS / IQ2_S (~10–11 GiB at 30B) regains 4–5 GiB of headroom. Quality cost depends on quant; 30B IQ2_XXS may still beat 8B IQ3_M on most evals.
2. **Cap context window.** Truncating to seq=1024 / 512 cuts KV cache in half / quarter. Fits within the toolchain ceiling but limits the deployment's prompt-handling capacity.
3. **Wait for Emscripten to lift the linker cap.** Upstream change (single-line in `wasm-ld`); no estimated date. Probe again on the next Emscripten upgrade.
4. **Custom wasm-ld patch.** Multi-day effort: rebuild llvm-wasm-ld with the ceiling bumped, distribute via CI. High maintenance burden; not recommended absent a deployment ask.

## 5. Decision

**§31b closes as "lever still viable, ceiling now characterized as toolchain-bound."** §31a's "≥8 GiB → promote to full bridge migration" decision still fires for any target up to ~13B Q4_K_S. For 30B IQ3_M targets, the migration scope inherits a **toolchain-ceiling tracking item**: re-probe the linker cap on every Emscripten upgrade (cheap — just rebuild and read the linker error message).

The MEMORY64 ceiling probe series (§31 → §31a → §31b) is now fully characterized:

- ✅ ASYNCIFY × MEMORY64 round-trip works (§31).
- ✅ BigInt ABI gap fixed via `bridge_malloc`/`bridge_free` (§31a).
- ✅ Configured ceiling = 15 GiB measured (§31a) at the toolchain max.
- ✅ Toolchain ceiling = 16 GiB (§31b, this probe).
- ❓ Chrome v8 wasm64 runtime cap — **unmeasurable from Emscripten 5.0.6**; presumably ≥16 GiB since 15 GiB allocates cleanly through the v8 path.

## 6. Process improvement noted

§31a's report framed the cap as "configured-ceiling-bound, not hardware-bound" without testing whether the configuration was bumpable. §31b's failure-mode probe took 2 minutes (one CMake edit + one rebuild attempt) and would have been a useful inline check during §31a. **Pattern for future cap probes:** when a cap is hit at a configurable value, immediately try bumping that value to confirm whether the cap is configuration- or toolchain-bound. The cost is cheap and avoids landing a misleading "the ceiling is just our config" framing.

## 7. Reproduction

```bash
# 1. Bump src/wasm/CMakeLists.txt (mem64 conditional block, line ~98)
sed -i '' 's/-sMAXIMUM_MEMORY=16GB/-sMAXIMUM_MEMORY=64GB/' src/wasm/CMakeLists.txt
# 2. Nuke the build cache (find_library staleness — see §32 gotcha)
rm -rf src/wasm/build-mem64
# 3. Attempt build; observe the linker error
make mem64-probe 2>&1 | grep -E "maximum memory|max-memory"
# Expected: wasm-ld: error: maximum memory too large, cannot be greater than 17179869184

# 4. Restore
git checkout src/wasm/CMakeLists.txt
```

## 8. Probe economics

- Implementation: 30 seconds (two single-line edits: `MAXIMUM_MEMORY=64GB` in CMakeLists.txt, `i < 64` in mem64-probe.html).
- Build attempt: ~25 seconds before linker failure surfaced.
- Total wall: <2 minutes.
- Saved: a multi-day "is the runtime cap really 16 GiB?" investigation; surfaced a tooling constraint that would otherwise have bitten the 30B migration scope cold.

Net code change in this probe: zero (edits reverted). New artifact: this report.
