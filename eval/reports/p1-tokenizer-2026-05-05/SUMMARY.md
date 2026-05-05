# P1 — Tokenizer Migration — BLOCKED

**Date:** 2026-05-05
**Spec:** [`docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md`](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md) §P1
**Plan:** [`docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md`](../../../docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md)

## Outcome

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

## Recommendation

The blocker is a real Emscripten + ASYNCIFY interaction issue that's
not specific to webllm's code. **Option 1 (rollback Emscripten to
5.0.5)** is the cheapest test and most likely to unblock — try that
first. If it still fails, **Option 2/3 (drop ASYNCIFY for JSPI)** is
the next move, but it has wider blast radius and should be filed
as its own design discussion.

For now, the rest of the Tier 3 migration (P2-P6) cannot proceed —
every phase needs at least one new wasm export. This blocker is the
top-priority item to resolve.

## Time spent

~1.5 hours of debugging across the bridge build, ASYNCIFY flags,
emcc cache clear, signature variations, and wasm-dis inspection.
Aborted after Option 7 (ALLOW_TABLE_GROWTH) failed.
