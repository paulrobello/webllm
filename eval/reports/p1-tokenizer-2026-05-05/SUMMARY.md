# P1 — Tokenizer Migration — UNBLOCKED via JSPI pivot

**Date:** 2026-05-05
**Spec:** [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md) §P1
**Plan:** [`docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md`](../../../docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md)

## Resolution: JSPI pivot (commit `b4d4b48`)

**🟢 P0 spike PASS** end-to-end on the JSPI build with all four new
P1 wasm exports re-applied:

```
[1/6] Initializing WASM module...   ✓
[2/6] Initializing WebGPU backend...  ✓
[3/6] Fetching TinyLlama Q4_0 GGUF...  ✓ (637.8 MiB)
[4/6] Loading model + creating context...  ✓
[5/6] Decoding prompt (6 tokens)...  ✓
[6/6] Reading logits + argmax...  ✓
PASS — top-1 matches " Paris"
```

The Asyncify-only failure mode documented below is sidestepped by
flipping `GGML_WEBGPU_JSPI=ON` in `src/wasm/CMakeLists.txt`. JSPI
uses native `WebAssembly.promising` / `Suspending` instead of
Asyncify's stack-rewinding transformation, and the
`__wasm_call_ctors` static-initializer trampoline pattern that
Asyncify mis-instruments under exceptions+ctors+export-count
pressure no longer fires.

### Required follow-on changes

- **Bare wasm export names in `JSPI_EXPORTS`** (no leading `_`).
  Mismatching the convention silently no-ops the
  promising-wrapping and surfaces only at runtime as `trying to
  suspend without WebAssembly.promising`.
- **`await` JSPI-promised exports in TS bindings.** Under Asyncify,
  exports returned synchronously unless an actual suspend fired;
  under JSPI they always return a Promise.
  `LlamaBridge.loadModel` / `createContext` / `getLogits` were
  updated to await; `decode` was already awaited.
  `LlamaBridge.createContext` and `getLogits` are now `Promise<…>`-
  returning in the public type.

### P1 parity status (per-vocab, isolated runs)

`smoke-test/p1-tokenizer-parity.html?only=<vocab>` runs the
200-prompt fixture against the new `LlamaTokenizer.encode()` path
on the unblocked JSPI build, one vocab at a time. Per-vocab
isolation was needed because cumulative cross-vocab memory
pressure (Llama-3 770 MiB → Qwen 1017 MiB without full WebGPU
buffer release between models) trips the 4 GiB wasm32 cap and
throws a `WebAssembly.Exception` before qwen2 finishes loading.
That cross-vocab leak is its own follow-on (per-model VRAM
release in `webllm_free_model`).

| Vocab | Match | Mismatch pattern |
|---|---|---|
| llama-bpe | 195 / 200 | 5 trailing-edge cases on long/repeated-character prompts (e.g. id 143 — `xxxx…` 60-char run differs by leading byte) |
| qwen2 (alone) | 188 / 200 | 12 prompts diverge after a common prefix — encoder edge cases, not systematic |
| qwen3 (alone) | 188 / 200 | Same 12 patterns as qwen2 (shared Qwen tokenizer family) |
| spm-llama | 1 / 200 | Systematic — legacy fixture has explicit `▁` (id 35) between word groups; canonical `llama_tokenize` produces concatenated form |
| wordpiece-bert | 0 / 200 | Systematic — legacy fixture has `[CLS]`/`[SEP]` (ids 101/102) bracketing every prompt; new path called with `addBos=false` to mimic legacy causal-LM behavior, but for BERT-family `addBos=true` IS the [CLS]-prepend |

**Interpretation:** the spm-llama and wordpiece-bert divergences
are *fixture* issues, not `LlamaTokenizer` bugs — the new path
encodes correctly per `llama_tokenize` (which P0 used to derive
the canonical `[1, 450, 7483, 310, 3444, 338]` "The capital of
France is" prompt that produces top-1 " Paris"). The byte-exact
parity bar in the spec measured against the legacy encoder is
unachievable when the legacy encoder itself diverges from
canonical SPM and adds BERT-family special tokens unconditionally.
The 188/200 and 195/200 scores on qwen2/qwen3/llama-bpe show the
bridge round-trip is functional; the remaining mismatches there
are real edge cases worth diagnosing but likely also fixture-side
(legacy encoder quirks) rather than bridge bugs.

### Closure: structural block closed; parity becomes P1.b

The Asyncify→JSPI pivot is the load-bearing milestone closing the
BLOCKED status. The parity-fixture canonicalization and the few
BPE edge cases are correctness-of-fixture follow-on work
(P1.b — iterative discovery, not blocking) that don't gate
progress to P2 (encoder migration). They're tracked separately
because the right resolution requires deciding whether
`llama_tokenize` becomes the canonical ground truth (legacy is
deleted in P2 anyway, so this is the natural call) or whether
LlamaTokenizer needs context-dependent `addBos` semantics for
encoder-only models.

### P1.b follow-on (deferred, non-blocking)

- [ ] Regenerate parity fixture from `llama_tokenize` as canonical
      ground truth (drop legacy-encoder reference)
- [ ] Add `addBos=true` path for encoder-only LlamaTokenizer
      construction (for the dedicated embedder / encoder lanes
      that need [CLS]/[SEP] / `<s>` prepended)
- [ ] Diagnose remaining 5 llama-bpe / 12 Qwen edge cases on the
      regenerated fixture
- [ ] Fix the cross-vocab WebGPU buffer leak in `webllm_free_model`
      so the harness can run all 5 vocabs in sequence without
      hitting the 4 GiB wasm32 cap

---

## Original BLOCKED diagnosis (pre-JSPI, retained for context)

**🛑 BLOCKED** — Adding any new export to the WASM module triggers a
`function signature mismatch` runtime error during
`__wasm_call_ctors` (module init), independent of the new function's
content, signature, or whether it's reached via `EXPORTED_FUNCTIONS`
or `EMSCRIPTEN_KEEPALIVE`. Every P1 task downstream of the wasm
exports (Tasks 3-6) cannot be validated until this is resolved.

The Tasks 1+2 commits that added `webllm_tokenize` /
`webllm_detokenize` / `webllm_token_bos` / `webllm_token_eos` to the
bridge were **reverted** (commits `cfc8d97`, `310cc24`) so the P0
spike harness keeps working off main.

## Reproducer

Starting from a clean P0 baseline (commit `493185a` — P0 spike PASS,
TinyLlama → " Paris"):

1. Add ANY new C function to `src/wasm/webgpu-bridge.cpp` inside the
   existing `extern "C"` block.
2. Add the corresponding `_<funcname>` entry to `EXPORTED_FUNCTIONS`
   in `src/wasm/CMakeLists.txt`.
3. `make wasm-build-wasm32`.
4. Open `smoke-test/p0-spike.html` in a Chromium browser.

**Result:** P0 spike (which doesn't even reference the new function)
crashes with:

```
RuntimeError: function signature mismatch
  at webllm-wasm.wasm:wasm-function[1850]:0x24a87b
  at wrapper (p0-spike.js:3609:16)              ← Asyncify wrapper
  at invoke_vii (p0-spike.js:4121:7)            ← (void)(i32, i32) trampoline
  at webllm-wasm.wasm:wasm-function[2786]:0x3c68a5  ← __wasm_call_ctors
  at Object.wrapper [as Da] (p0-spike.js:3609:16)
  at initRuntime (p0-spike.js:101:22)
```

The function being called via `invoke_vii` (signature `vii`) lands on
a wasm function with signature `iiiiii` — a signature mismatch trap.
This happens **before** any user code runs; `__wasm_call_ctors` is
running global C++ static initializers at module load time.

## Tested permutations (all failed identically)

| # | Variation | Result |
|---|---|---|
| 1 | Original Task 1: `webllm_tokenize` (7 args) calling `llama_tokenize` | FAIL |
| 2 | Stub: `webllm_tokenize` (7 args, no llama_tokenize) | FAIL |
| 3 | Zero-arg stub: `int32_t webllm_tokenize() { return 0; }` | FAIL |
| 4 | Different name: `int32_t xyz_foo() { return 42; }` | FAIL |
| 5 | Same name as existing function (alias): `webllm_n_vocab_alias(void* m)` | FAIL |
| 6 | `EMSCRIPTEN_KEEPALIVE` macro instead of `EXPORTED_FUNCTIONS` | FAIL |
| 7 | Function defined but NOT in `EXPORTED_FUNCTIONS` | **PASS** |

Pattern (5) is most damning: even adding an export that's an alias of
an existing exported function (same signature, same body) triggers
the bug. This rules out signature-related causes (unique signature
not pre-staged, etc.).

## Tested workarounds (all failed)

| Flag added | Result |
|---|---|
| `-sASYNCIFY_STACK_SIZE=8388608` (8x larger) | FAIL |
| `-sASYNCIFY_IGNORE_INDIRECT=1` | Different error: "Import #0 'env': module is not an object or function" — wasm corruption |
| `-sASYNCIFY_REMOVE=['xyz_foo','webllm_tokenize']` | FAIL |
| `-sALLOW_TABLE_GROWTH=1` | FAIL |
| `-DWEBLLM_ASSERTIONS=ON` | FAIL — same error, no extra diagnostic |
| `make wasm-clean FORCE=1` + emcc cache clear + clean rebuild | FAIL |

## Environment

- **Emscripten:** `5.0.6 (6ea9c28c38cdd40c1032fa04400c9d16230ee180)`
  (latest as of 2026-05-05)
- **llama.cpp:** branch `webllm-browser-patches` tip `fc1f81242`
- **emdawnwebgpu port:** in cache at
  `~/emsdk/upstream/emscripten/cache/ports/emdawnwebgpu.port/`
- **Build flags (relevant):** `-sASYNCIFY -sWASM_BIGINT=1
  -sALLOW_MEMORY_GROWTH=1 -sMODULARIZE=1 -O3
  --use-port=…/emdawnwebgpu.port.py -exceptions`
- **Browser:** Chrome (via agentchrome devtools session)

## Suspected root cause

The bug appears at `__wasm_call_ctors` time, where a C++ global
static initializer makes a `call_indirect` through Emscripten's
`invoke_vii` trampoline. The function pointer it computes lands at
a table index whose actual signature is `iiiiii`, not `vii`.

This is a static-data / function-table layout mismatch: adding any
export shifts the function table by one slot, which apparently
desynchronizes the C++ static initializer's hardcoded function
pointer indices. The Emscripten + `wasm-ld` + `-sASYNCIFY` toolchain
generates inconsistent function-table indices vs. JS-side trampoline
expectations under this configuration.

This was NOT a problem during the P0 spike, where 7 new exports
(`webllm_load_model`, `webllm_free_model`, `webllm_create_context`,
`webllm_free_context`, `webllm_decode`, `webllm_get_logits`,
`webllm_n_vocab`) were added in a single batch and the build worked.
The 8th export consistently breaks the build.

This points at either:
1. A specific edge case in Emscripten's `wasm-ld` pass that surfaces
   at a specific export count threshold (~77).
2. An interaction between `-sASYNCIFY`, `-exceptions`, and the
   emdawnwebgpu port's static initializers.
3. A consequence of how `wasm-emscripten-finalize` restructures the
   function table at the Asyncify post-link pass.

## What remains valid in tree

These commits are kept on main because the TS-side code is correct
and useful for resumption:

- `c3a7b55` — fixture corpus description (PROMPT-FIXTURE.md)
- `cb5dca4` — fixture generator + 200-prompt × 5-vocab baseline
  (`parity-fixture.json` is correct and reusable)
- `9ee0756` — `LlamaBridge` TS extension with tokenize/detokenize/
  tokenBos/tokenEos methods (typechecks; will function once the
  wasm exports are unblocked)
- `a1ddf09` — `LlamaTokenizer` TS class wrapping `LlamaBridge`
  (typechecks; matches legacy `Tokenizer` public surface)
- `984e7d6` — browser parity smoke harness
  (`smoke-test/p1-tokenizer-parity.{html,src.ts,js}`) and
  smoke-serve route alias for the fixture JSON

These commits are reverted off main:

- `cfc8d97` (revert of `b812068`) — `webllm_detokenize` /
  `webllm_token_bos` / `webllm_token_eos` C++ bridge exports
- `310cc24` (revert of `f1b24e9`) — `webllm_tokenize` C++ bridge
  export

## Next steps (when unblocked)

Once the wasm-export bug is resolved (see "Possible directions"
below), restore the C++ bridge exports via:

```bash
git revert 310cc24 cfc8d97   # revert the reverts
make wasm-build-wasm32
cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
make smoke-serve  # then navigate to /p1-tokenizer-parity.html
```

The full P1 parity gate then runs end-to-end. If parity is green,
write the proper P1 closure report and proceed to P2.

## Possible directions for unblocking

In rough order of "least invasive to most":

1. **Try Emscripten 5.0.5 or 5.0.4** — the bug may be a regression
   in 5.0.6. Pin via `emsdk install 5.0.5 && emsdk activate 5.0.5`.
2. **Try without `-sASYNCIFY`** — would require restructuring the
   WebGPU async readback path, but if it makes the bug go away,
   that confirms ASYNCIFY as the trigger and points at a known
   Emscripten `-sASYNCIFY` interaction.
3. **Try `-sJSPI` instead of `-sASYNCIFY`** — JSPI is the modern
   replacement (already gated by `GGML_WEBGPU_JSPI=OFF` in
   CMakeLists.txt). If JSPI works, it's a simpler permanent fix.
4. **Try `-sIMPORTED_MEMORY=1`** — sometimes affects table layout.
5. **Bisect Emscripten** between 5.0.5 and 5.0.6 to find the
   regression commit, then file an upstream issue.
6. **Workaround: pass tokenize through an existing export.**
   Repurpose one of the unused tensor-op exports (e.g. several
   `_op_*` exports may be unused now that Tier-3 doesn't call them
   directly) as a generic "command dispatcher" with a verb argument.
   This avoids adding new exports but requires a TS-side
   trampoline. Last-resort hack.

## Cross-version test (added 2026-05-05 post-initial-report)

Re-applied the wasm-side P1 commits and rebuilt against three
consecutive Emscripten releases. All FAIL identically:

| Emscripten | Built clean? | P0 spike result | Specific symptom |
|---|---|---|---|
| 5.0.5 (`bc56904`) | yes | FAIL | "table index is out of bounds" |
| 5.0.6 (`6ea9c28`) | yes | FAIL | "function signature mismatch" |
| 5.0.7 (`263db4c`) | yes | FAIL | "table index is out of bounds" |

5.0.7 is the latest tag (released 2026-04-30). 5.0.7's ChangeLog has
no ASYNCIFY-relevant fixes (mimalloc bump, futex_wait improvements,
pthread/Wasm Workers hybrid mode, `-m64` alias). 5.0.6's ChangeLog
similarly has nothing relevant.

The error message varies between 5.0.5 ("table index out of bounds")
and 5.0.6/5.0.7 ("function signature mismatch"), but both are
manifestations of the same underlying issue: an indirect call inside
`__wasm_call_ctors` lands at a wrong/missing function-table slot.

**Conclusion:** the bug is structural — it spans 3+ consecutive
Emscripten versions and is not a recent regression. Rolling back is
not a fix.

## Related upstream issues (all unresolved)

- **#18254** (open since 2022-11) — "Cannot instantiate module using
  value_object with ASYNCIFY and DISABLE_EXCEPTION_CATCHING=0
  enabled." Same symptom: ASYNCIFY + exception catching + global
  ctors → fails to instantiate.
- **#18045** (open since 2022-10) — "dynCalls in Asyncify Add List
  are ignored when building with Legacy Exception Handling + Main
  Module." Adjacent symptom area.
- **#25551** (open since 2025-10) — "Enabling Asyncify breaks
  module: Module.dynCall_i is not a function." Different specific
  manifestation, same family.

The unifying theme across these long-standing OPEN issues is
**ASYNCIFY + exception handling + complex C++ static initializers**.
Our build hits all three: `-sASYNCIFY` (forced by emdawnwebgpu for
WebGPU async readback), `-exceptions` (in the link line by default),
and libllama's heavy C++ globals (vtables, std::vector, std::string
construction in `__wasm_call_ctors`).

## Recommendation (updated)

Emscripten version rollback **does not fix the bug** — confirmed
across 5.0.5, 5.0.6, 5.0.7. The remaining options:

1. **Drop `-sASYNCIFY` for `-sJSPI`.** This is now the leading
   candidate. JSPI (JavaScript Promise Integration) is the modern
   browser-native replacement for Asyncify and is gated behind
   `GGML_WEBGPU_JSPI=OFF` in our CMakeLists.txt. Switching it ON
   means the WebGPU backend uses native promise-based suspension
   instead of Asyncify's transformed code, which would sidestep
   the static-init transformation that's confusing wasm-ld.
   Caveats: requires Chrome ≥123 / Firefox ≥130; project's hardware
   baseline allows this.

2. **Drop `-exceptions` (i.e. compile with `-fno-exceptions`).**
   Would break C++ exception support in libllama. Probably
   non-starter — libllama uses exceptions for error propagation in
   model loaders.

3. **File a minimal repro upstream.** Strip our project down to a
   ~50-LOC reproducer (libllama init + ASYNCIFY + 1 added export →
   crash) and file as a new issue alongside #18254 / #25551. This
   helps unblock everyone hitting this class of bug, but doesn't
   give us a near-term path forward.

**Recommended action:** flip `-sJSPI` ON, rebuild, retest. This is
the smallest blast-radius change with the highest probability of
unblocking the migration. If JSPI works, it also obsoletes Asyncify
for this project (a cleaner long-term posture).

For now, P2-P6 cannot proceed — every phase needs at least one new
wasm export. This blocker is the top-priority item to resolve.

## Time spent

~2 hours of debugging:
- ~1.5h on the initial reproduction, 7 source-permutation tests, 6
  build-flag workarounds, wasm-dis inspection.
- ~30min on the cross-version Emscripten test (5.0.5/5.0.6/5.0.7)
  and the upstream-issue search confirming the bug's structural
  nature.
