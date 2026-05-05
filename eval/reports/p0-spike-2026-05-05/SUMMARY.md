# Tier 3 P0 Spike — Closure Report

**Date:** 2026-05-05
**Spec:** [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-05-tier3-p0-spike.md`](../../../docs/superpowers/plans/2026-05-05-tier3-p0-spike.md)

## Outcome

**✅ PASS** — TinyLlama Q4_0 (actually Q4_K_M per loader output) decodes
the prompt `"The capital of France is"` through the new `webllm_decode`
bridge wrapping upstream `llama_decode`, producing top-1 token id 3681
(` Paris`) with logit 13.043. Decision: **PROCEED to P1 (tokenizer)**.

Smoke trace from the page output:

```
[1/6] Initializing WASM module...
[2/6] Initializing WebGPU backend...
[3/6] Fetching TinyLlama Q4_0 GGUF from /models/tinyllama-1.1b-chat-q4_0.gguf...
     loaded 637.8 MiB
[4/6] Loading model + creating context...
     model loaded in 171 ms; vocab = 32000
[5/6] Decoding prompt (6 tokens)...
     llama_decode status = 0 (899 ms)
[6/6] Reading logits + argmax...
     top-1 token id = 3681 (logit 13.043)
     expected " Paris" id 3681
PASS — top-1 matches " Paris"
```

## Build deltas

Bridge surface (`src/wasm/webgpu-bridge.cpp`): **+~70 LOC C++** for seven
new exports across Tasks 2-5:
- `webllm_load_model` / `webllm_free_model` (commit `20118a3`)
- `webllm_create_context` / `webllm_free_context` (commit `7522404` —
  with one upstream-API drift adaptation: `flash_attn` → `flash_attn_type`
  enum)
- `webllm_decode` / `webllm_get_logits` (commit `9ffe15b`)
- `webllm_n_vocab` (commit `aa73757`)

Build config (`src/wasm/CMakeLists.txt`): **+~30 LOC CMake** across
Tasks 1 + 7:
- Linked `libllama` via `add_subdirectory(LLAMA_CPP_DIR EXCLUDE_FROM_ALL)`
  with `LLAMA_WASM_MEM64=OFF` + `LLAMA_BUILD_*=OFF` flags (commit `f92791f`)
- Flipped `GGML_CPU=ON` to satisfy libllama's "no CPU backend found"
  check at `llama-model.cpp:852` (commit `e91c398`)
- Added `-sWASM_BIGINT=1` to the wasm32 build to handle libllama's
  i64 WebGPU async-wait timeoutNS arg (commit `e91c398`)

TypeScript bindings (`src/inference/llama-bridge.ts`): **+~190 LOC TS**
across Tasks 6 + 7:
- Initial cwrap'd bindings + async wrapper (commit `2d9eaf2`)
- Runtime ABI probe (`is64`) for wasm32 vs wasm64 pointer ABI; mirrors
  `GgmlWasm.init()` pattern (commit `45a5b78`)

Smoke harness (`smoke-test/p0-spike.{html,src.ts}`): **+130 LOC** total
(commit `574bcd9`). Bundled to `smoke-test/p0-spike.js` via:
```
bun build smoke-test/p0-spike.src.ts --outfile smoke-test/p0-spike.js \
  --target browser
```

Fixture (`eval/reports/p0-spike-2026-05-05/PROMPT-FIXTURE.md`): captured
hardcoded prompt token IDs from host `llama-cli` so P0 doesn't depend on
`llama_tokenize` (commit `fa7f44a`).

**WASM binary size:** 4,798,459 bytes (~4.6 MB), from 2,507,744 (~2.4 MB)
pre-Task-1. The +2.2 MB increase is libllama + GGML_CPU's static archive
contributions. Not a P0 concern; revisit during P5 if binary size
becomes a perf-recovery lever.

## Patches applied to ~/Repos/llama.cpp

**None.** Patch budget B is intact at 9 patches. The plan's anticipated
ASYNCIFY-aware `llama_decode` patch and heap-aware `llama_kv_cache_init`
patch turned out to be **not needed** — both async-wait and KV alloc
paths worked unmodified once the build config (GGML_CPU=ON +
WASM_BIGINT=1) was correct.

## ASYNCIFY discovery

- **Auto-marked: yes.** Emscripten's `-sASYNCIFY` build setting (already
  on at `src/wasm/CMakeLists.txt:101,125`) transitively marks
  `llama_decode` as suspendable because it calls
  `ggml_backend_sched_graph_compute_async` → WebGPU readback (which has
  a JS-async import). No `-sASYNCIFY_ADD` list needed.
- **Manual `-sASYNCIFY_ADD`: no.**
- **Source patches: no.**

## KV init discovery

- **Default `n_ctx`: 512 worked unmodified.** No heap-pressure issues on
  TinyLlama at this context size. KV cache footprint at n_ctx=512 with
  TinyLlama's 22 layers × head_dim=64 × n_kv_heads=4 × f16 is tiny —
  ~14 MiB. The 4 GiB wasm32 heap is far from the cap.
- **Heap-aware patch: not needed for P0.** Will revisit at P4 (embedder)
  or P5 (large-context spec-decode) if 8B-class models hit the limit.

## Three discovery branches in the plan vs actual

| Plan branch | Predicted | Actual |
|---|---|---|
| **A — clean PASS** | green smoke + benign console | hit, but only after fixing 3 build-config deltas |
| **B — ASYNCIFY error** | required `-sASYNCIFY_ADD` or source patch | **not hit** (auto-marking sufficed) |
| **C — KV init OOM / alloc fail** | required heap-pressure source patch | **not hit** (defaults worked) |

Three **unanticipated** failure modes surfaced and were resolved without
patches:

1. **wasm64 vs wasm32 ABI mismatch.** Initial Task 6 typed everything
   `bigint`. wasm32 build returns JS Number for pointers. Fix: runtime
   `is64` probe (matching legacy `GgmlWasm.init()` pattern).
2. **Missing CPU backend.** libllama's loader requires
   `ggml_backend_dev_by_type(CPU)` to exist when `use_extra_bufts` is
   enabled. webllm previously had `GGML_CPU=OFF`. Fix: flip to ON.
3. **WASM_BIGINT missing on wasm32.** WebGPU async-wait helpers take i64
   timeoutNS args. wasm32 build had `-sWASM_BIGINT` only on the mem64
   target. Fix: add it to wasm32 too.

## Decision

**✅ PROCEED to P1 (tokenizer migration).** All blockers resolved with
build-config changes; no llama.cpp patches consumed; no ASYNCIFY or KV
risks materialized. Tier 3 viability validated.

The actual fixes were significantly cleaner than the plan anticipated —
the spec's risk register R1 (ASYNCIFY blocker, escape valve to Liberal
patch budget) and R4 (KV cache allocator interaction) can be downgraded
in P1's risk register since neither path had to be exercised.

## Header-block update for TODO.md

Append to the "Tier 3 migration to upstream `llama_decode`" section:

```
**P0 (Spike) — CLOSED 2026-05-05**: PASS. TinyLlama → webllm_decode →
" Paris" green; patch budget intact (still 9 core llama.cpp patches).
Three build-config deltas were sufficient: `GGML_CPU=ON`,
`WASM_BIGINT=1` on wasm32, and `LLAMA_WASM_MEM64=OFF` to avoid wasm32/64
conflict. No source patches needed. Closure report at
[`eval/reports/p0-spike-2026-05-05/SUMMARY.md`](eval/reports/p0-spike-2026-05-05/SUMMARY.md).
Decision: PROCEED to P1 (tokenizer).
```

## Commit chain (this P0 cycle)

| Commit | Subject | Phase |
|---|---|---|
| `49ada70` | docs(TODO): add Tier 3 llama_decode migration as new direction | brainstorm |
| `60bcb59` | docs(spec): Tier 3 migration to upstream llama_decode | spec |
| `2c870ae` | docs(plan): Tier 3 P0 spike implementation plan | plan |
| `fa7f44a` | docs(p0): record TinyLlama Paris-prompt fixture | T0 |
| `f92791f` | build(wasm): link libllama into webllm-wasm + webllm-wasm-mem64 | T1 |
| `20118a3` | feat(wasm): add webllm_load_model / webllm_free_model | T2 |
| `7522404` | feat(wasm): add webllm_create_context / webllm_free_context | T3 |
| `9ffe15b` | feat(wasm): add webllm_decode + webllm_get_logits hot-path bridge | T4 |
| `aa73757` | feat(wasm): add webllm_n_vocab metadata getter | T5 |
| `2d9eaf2` | feat(ts): add llama-bridge.ts cwrap'd bindings + async webllm_decode | T6 |
| `e91c398` | build(wasm): GGML_CPU=ON + WASM_BIGINT=1 for Tier 3 libllama path | T7 |
| `45a5b78` | fix(ts): runtime ABI probe in llama-bridge.ts (wasm32 vs wasm64) | T7 |
| `574bcd9` | feat(smoke): P0 spike harness — TinyLlama → llama_decode → " Paris" | T7 |

## Open questions resolved by P0

From the spec's §Open questions list:

- **OQ1 `llama_model_load_from_buffer`.** Resolved via Emscripten MEMFS
  round-trip: bridge writes buffer to `/tmp/webllm-model.gguf` via
  stdio, calls `llama_model_load_from_file` against that virtual path.
  Adds ~640 MiB MEMFS allocation that persists for the model's lifetime
  — flag for P2 hardening (one-line `std::remove(path)` after load is
  sufficient; called out by the Task 2 code-quality reviewer).
- **OQ4 Greedy bench reproducibility.** Implicitly resolved — greedy
  argmax over `llama_get_logits()` produced the canonical " Paris"
  result, matching host `llama-cli --temp 0` exactly. Determinism
  confirmed.

OQ2 (embedder + flash_attn) and OQ3 (spec-decode KV-share) remain
open — they belong to P3-P5.

## Performance notes (informational, not perf-bar)

| Phase | Time | Notes |
|---|---|---|
| Module load (`createModule()`) | ~unmeasured | one-shot |
| `webgpu_init` | <100ms (estimated) | unchanged from legacy |
| GGUF fetch | ~200 ms | local server, 637.8 MiB |
| `webllm_load_model` (MEMFS write + libllama parse) | **171 ms** | warm filesystem |
| `webllm_decode` (6-token prefill) | **899 ms** | cold, includes shader compile |

The 899 ms decode is **cold-cache** — first decode includes WebGPU
shader compilation. Production / steady-state numbers are not in scope
for P0; P2's stage-D parity bar will set them against canonical-6
baselines in TODO.md.

## What's next (P1)

Per the spec's phase plan: replace `tokenizer.ts` BPE/WordPiece encoders
with `llama_tokenize` calls. Add `webllm_tokenize` / `webllm_detokenize`
bridge exports. Byte-exact 200-prompt fixture parity gate before merge.
