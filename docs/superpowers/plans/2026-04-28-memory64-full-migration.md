# MEMORY64 full bridge migration — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v1 (2026-04-28).** Derived from the §MEMORY64 phasing skeleton committed
> to `TODO.md` 2026-04-28. The probe series (§31 / §31a / §31b) already
> retired ASYNCIFY × MEMORY64, fixed the BigInt ABI gap via
> `bridge_malloc` / `bridge_free`, and characterized the toolchain ceiling
> at 16 GiB. This plan executes the production migration on top of those
> probes.

**Goal:** Migrate the production WebLLM build from the 4 GiB-cap WASM32 path
to the 16 GiB-cap WASM64 path so the engine can host 13B Q4_K_S (~7.4 GiB)
and 30B IQ3_M (~12.8 GiB) targets within the 30B project ceiling.

**Architecture:** Two parallel CMake build dirs (already wired by §31a):
`src/wasm/build/` produces `webllm-wasm.{js,wasm}` (wasm32, 4 GiB cap) and
`src/wasm/build-mem64/` produces `webllm-wasm-mem64.{js,wasm}` (wasm64,
16 GiB cap). Both binaries already export `_bridge_malloc` / `_bridge_free`.
This plan (a) migrates every JS-side `_malloc`/`_free` call to the bridge
wrappers (so the same TS code works under both binaries), (b) hardens
`webgpu-bridge.cpp` `int32_t size/offset` params to `size_t` for >2 GiB
single-buffer transfer safety, (c) extends `make wasm-build` to produce
both binaries, (d) gates the migration on a zero-regression bench parity
sweep across the canonical 6 fleet, and (e) registers one >4 GiB
validation target.

**Tech Stack:** TypeScript / Bun, patched `llama.cpp` `ggml-webgpu` compiled
to WASM (wasm32 + wasm64), Emscripten 5.0.6, `bun test`, `make checkall`,
`make smoke-bench`, `make bench-inference`, `agentchrome`.

**Spec context:** No new spec — this plan is the execution document for the
§31 / §31a / §31b probe series. Cap probe spec at
`docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md`. Closure
reports at `eval/reports/memory64-probe-2026-04-28/{SUMMARY,SUMMARY-31a,SUMMARY-31b}.md`.

**Phasing:** 1 plan-commit + 8 phase commits (Phase 0 audit + Phases 1-7
implementation). Each phase commit is `make checkall` clean and
independently revertable. Mirrors §17/§18/§19/§20 plan structure.

**Pre-rebase baselines:** `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`
serves as the wasm32 reference for Phase 5 bench-parity gates.

---

## Task 0: Commit this plan

**Files:** `docs/superpowers/plans/2026-04-28-memory64-full-migration.md` (this file).

**Purpose:** Land the plan as its own commit before any phase work, per the
project doctrine "Always commit before work" (CLAUDE.md). The
`docs/superpowers/` directory is gitignored — force-add is required.

- [ ] **Step 1: Force-add and commit the plan**

```bash
git add -f docs/superpowers/plans/2026-04-28-memory64-full-migration.md
git commit -m "$(cat <<'EOF'
docs(plan): MEMORY64 full bridge migration plan

Execution document for the §31/§31a/§31b probe series. 1 plan-commit
+ 8 phase commits: Phase 0 (audit/punch list, docs-only), Phase 1
(JS bridge_malloc migration), Phase 2 (bridge size_t hardening),
Phase 3 (GGUF loader BigInt boundary), Phase 4 (dual-binary
make wasm-build), Phase 5 (bench parity gates on canonical 6),
Phase 6 (single-vs-dual binary decision), Phase 7 (>4 GiB
validation target).

Probe artifacts:
- eval/reports/memory64-probe-2026-04-28/SUMMARY.md (§31)
- eval/reports/memory64-probe-2026-04-28/SUMMARY-31a.md (§31a, 15 GiB cap)
- eval/reports/memory64-probe-2026-04-28/SUMMARY-31b.md (§31b, 16 GiB toolchain ceiling)
EOF
)"
```

- [ ] **Step 2: Verify the commit landed**

Run: `git log --oneline -1`
Expected: starts with `docs(plan): MEMORY64 full bridge migration plan`.

---

## Task 1: Phase 0 — Call-site audit + punch list (commit 1 of 8 — docs only)

**Files:**
- Create: `eval/reports/memory64-migration-2026-04-28/PUNCH-LIST.md`
- Create: `eval/reports/memory64-migration-2026-04-28/audit-grep.sh`

**Purpose:** Probe-first default (CLAUDE.md). Catalog every `_malloc` /
`_free` call site, every `int32_t size/offset` bridge param, and every
GGUF-loader heap boundary that needs BigInt awareness — *before* writing
any migration code. Output: a punch list keyed to file:line with the
required signature change for each entry. No source code edits in this
task.

**Audit scope (from §MEMORY64 phasing skeleton, TODO.md):**
- `src/inference/` — TS heap allocator wrappers + decode/forward call sites.
- `src/wasm/webgpu-bridge.cpp` — bridge param signatures.
- `src/models/` — GGUF parser/loader (no current malloc usage; document confirmation).
- `smoke-test/` — page-side GGUF streaming loader (`real-model-page.js`).
- `eval/` — bench harnesses (probe-only `mem64-probe.html` already migrated by §31a; verify).

- [ ] **Step 1: Create the audit grep script**

`eval/reports/memory64-migration-2026-04-28/audit-grep.sh`:

```bash
#!/usr/bin/env bash
# MEMORY64 migration call-site audit. Run from repo root. Output is the
# raw evidence behind PUNCH-LIST.md — re-run after each phase commit to
# confirm no callers regressed back to stdlib _malloc/_free.
set -euo pipefail

REPORT_DIR="$(dirname "$0")"
cd "$(dirname "$0")/../../.."

echo "=== JS-side _malloc / _free / wasm.malloc / wasm.free ===" >"$REPORT_DIR/audit-js.txt"
grep -rn '_malloc\b\|_free\b\|wasm\.malloc\|wasm\.free\|m\._malloc\|m\._free' \
  --include='*.ts' --include='*.js' --include='*.html' \
  src/ smoke-test/ eval/ tests/ 2>/dev/null \
  | grep -v 'build/\|build-mem64/\|node_modules\|webllm-bundle\|webllm-wasm' \
  >>"$REPORT_DIR/audit-js.txt" || true

echo "=== JS-side stack allocator (stackAlloc/Save/Restore) ===" >"$REPORT_DIR/audit-stack.txt"
grep -rn 'stackAlloc\|stackSave\|stackRestore' \
  --include='*.ts' --include='*.js' --include='*.html' \
  src/ smoke-test/ tests/ 2>/dev/null \
  | grep -v 'node_modules' \
  >>"$REPORT_DIR/audit-stack.txt" || true

echo "=== Bridge int32_t size/offset params ===" >"$REPORT_DIR/audit-bridge.txt"
grep -n 'int32_t.*\(size\|offset\)\|int32_t mem_size' \
  src/wasm/webgpu-bridge.cpp \
  >>"$REPORT_DIR/audit-bridge.txt" || true

echo "=== GGUF loader heap boundaries ===" >"$REPORT_DIR/audit-gguf.txt"
grep -rn 'modelPtr\|wasm\.malloc\|HEAPU8\|heapU8\.buffer' \
  --include='*.js' --include='*.ts' \
  src/models/ smoke-test/ 2>/dev/null \
  | grep -v 'webllm-bundle\|webllm-wasm' \
  >>"$REPORT_DIR/audit-gguf.txt" || true

echo "Audit complete. See:"
ls -1 "$REPORT_DIR"/audit-*.txt
```

- [ ] **Step 2: Run the audit script**

```bash
chmod +x eval/reports/memory64-migration-2026-04-28/audit-grep.sh
eval/reports/memory64-migration-2026-04-28/audit-grep.sh
```

Expected: four `audit-*.txt` files in
`eval/reports/memory64-migration-2026-04-28/`.

- [ ] **Step 3: Author the punch list**

Write `eval/reports/memory64-migration-2026-04-28/PUNCH-LIST.md` with this
structure (the bullet entries below are the canonical baseline as of
2026-04-28; cross-check each one against the `audit-*.txt` outputs and
add anything new the script surfaces):

```markdown
# MEMORY64 migration — call-site punch list

**Date:** 2026-04-28
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Audit script:** [`audit-grep.sh`](./audit-grep.sh) — re-run after each phase to confirm no regression.

## Phase 1 targets (JS-side `_malloc` / `_free` migration)

Replace each `m._malloc` / `m._free` call (or wrapper around it) with
`m._bridge_malloc` / `m._bridge_free`. The two custom exports already
live in both binaries (see `src/wasm/CMakeLists.txt:51`). Wrappers must
normalize BigInt return values from wasm64 to `number` because no single
call site allocates >2 GiB (largest tensor at 30B IQ3_M ≈ 850 MB).

| File:line | Caller | Notes |
|---|---|---|
| `src/inference/ggml-wasm.ts:257` | `GgmlWasm.malloc()` | Single source of truth — every other call routes through here. |
| `src/inference/ggml-wasm.ts:261` | `GgmlWasm.free()` | Mirrors `.malloc()`. |
| `src/inference/ggml-wasm.ts:373,378` | `uploadToTensor` | Internal use of `this.malloc/this.free` — covered by Phase 1 if the wrappers are migrated. |
| `src/inference/ggml-wasm.ts:389,398` | `uploadToTensorChunked` | Same. |
| `src/inference/ggml-wasm.ts:415,424` | `uploadRangeChunked` | **Load-bearing for GGUF streaming.** Heap-grow detachment is already handled. |
| `src/inference/ggml-wasm.ts:434,457` | `beginDownloadFromTensor` | Async readback heap allocation. |
| `src/inference/encoder-inference.ts:476,503,512,522` | encoder forward + parity probe | Each routes through `wasm.malloc/free`. |
| `src/inference/model-inference.ts:675,723,1039,1084,1614,1659,2024,2029` | decode/forward heap scratch | All use `wasm.malloc/free`. |
| `smoke-test/real-model-page.js:306,333,364,386,438` | GGUF streaming loader | **Phase 3 dependency.** The `wasm.malloc(total)` at line 306 allocates the entire model file; for 13B Q4_K_S that's ~7.4 GiB — exceeds 2^31. Phase 1 routes the call through `_bridge_malloc`; Phase 3 confirms the BigInt size argument flows correctly. |
| `smoke-test/mem64-probe.html` (probe only) | already on `bridge_malloc` per §31a | confirm no regression. |

## Phase 2 targets (bridge ABI hardening — `int32_t` → `size_t`)

| File:line | Function | Change | Why |
|---|---|---|---|
| `src/wasm/webgpu-bridge.cpp:59` | `ctx_create(int32_t mem_size)` | `→ size_t` | mem_size already cast to `size_t` internally; signature should match. ggml metadata budget at 30B is ~1 MB — no functional cap, but consistent ABI. |
| `src/wasm/webgpu-bridge.cpp:129` | `tensor_set_data(void*, const void*, int32_t size)` | `→ size_t` | Single-tensor uploads <2 GiB at 30B (largest = embedding ≈ 850 MB); promotion is conservative safety only. |
| `src/wasm/webgpu-bridge.cpp:133` | `tensor_get_data(void*, void*, int32_t size)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:187,192` | `op_view_2d/3d(... int32_t offset)` | `→ size_t` | View offsets within graph allocator buffer; theoretical >2 GiB at very-long prefill but not exercised at typical seq=2048. |
| `src/wasm/webgpu-bridge.cpp:296` | `backend_tensor_set(... int32_t offset, int32_t size)` | `→ size_t` | Tensor-buffer offset+size pairs. |
| `src/wasm/webgpu-bridge.cpp:303-310` | `backend_tensor_set3(... int32_t sz1/sz2/sz3)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:313` | `backend_tensor_get(... int32_t offset, int32_t size)` | `→ size_t` | Same. |
| `src/wasm/webgpu-bridge.cpp:317-321` | `backend_tensor_get_async_begin(... int32_t offset, int32_t size)` | `→ size_t` | Body already casts to `size_t`. |
| `src/wasm/webgpu-bridge.cpp:328-329` | `backend_tensor_get_async_finish(... int32_t size)` | `→ size_t` | Body already casts to `size_t`. |
| `src/wasm/webgpu-bridge.cpp:272-273` | `graph_new(int32_t size)` | `→ size_t` | Body already casts to `size_t`; `size` is graph-node count, never near 2 GiB. |

**Net effect:** zero functional change at the ≤30B ceiling (no individual
transfer exceeds 2 GiB), but signatures stop pretending to cap at 2 GiB.
Required for the linker to emit BigInt-marshaling JS shims under wasm64
when those bindings cross >2 GiB single buffers in a future scope
expansion (e.g. concat-batched encoder graph buffers).

## Phase 3 targets (GGUF loader BigInt boundary)

| File:line | Concern | Required change |
|---|---|---|
| `smoke-test/real-model-page.js:306` | `wasm.malloc(total)` where `total` can be >4 GiB | After Phase 1, the JS wrapper calls `_bridge_malloc(BigInt(total))`. Phase 3 verifies that no intermediate `Number` narrowing occurs anywhere in the call chain when `total > 2^31`. |
| `smoke-test/real-model-page.js:318` | `wasm.heapU8.set(value, modelPtr + received)` | `modelPtr` is normalized to `number` (safe — Number can represent up to 2^53). `received` is `number`. The `set` call is safe so long as `modelPtr + received < 2^53`. For 13B at 7.4 GiB, `modelPtr` is ~10^10 well under 2^53 ≈ 9×10^15. |
| `smoke-test/real-model-page.js:341` | `new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len)` | Same — `modelPtr + off < 2^53`. |
| `src/inference/ggml-wasm.ts:415-424` | `uploadRangeChunked` heap-grow detachment | Pre-existing fix re-derives `dataAt(off, end - off)` per chunk after `malloc`. Verify it survives BigInt return values from `_bridge_malloc`. |

**No source-code change required at the GGUF parser layer
(`src/models/gguf-parser.ts`).** It operates on `Uint8Array` views — the
heap pointer never crosses its API boundary.

## Phase 7 target (>4 GiB validation candidate)

| Candidate | Approx. size | Why |
|---|---:|---|
| `mistral-7b-instruct-v0.3-q5km` | ~5.0 GiB | First step above the wasm32 4 GiB cap; same arch as the canonical 7B Q4_K_S baseline → tightest pre/post comparison. |
| `mistral-13b-instruct-q4ks` | ~7.4 GiB | Plan-target 13B; depends on a coherent Q4_K_S-quant 13B GGUF being available on HF. |
| `llama-3.1-13b-instruct-iq3m` | ~5.4 GiB | Alternative 13B if Mistral-13B Q4_K_S is unavailable. |

Phase 7 picks one based on availability and registers it via
`eval/models.ts` + `eval/smoke-profiles.ts`.
```

- [ ] **Step 4: Commit Phase 0**

```bash
git add -f eval/reports/memory64-migration-2026-04-28/
git commit -m "$(cat <<'EOF'
docs(plan): MEMORY64 migration Phase 0 — call-site audit + punch list

Phase 0 of the MEMORY64 full migration plan. Probe-first default:
audit every _malloc/_free call site, every int32_t size/offset
bridge param, and every GGUF loader heap boundary before writing
migration code. Output: a punch list keyed to file:line with the
required signature change for each entry.

Punch list scope:
- Phase 1: JS-side _malloc/_free → _bridge_malloc/_bridge_free (~12
  primary callsites, all routed through GgmlWasm.malloc/.free at
  src/inference/ggml-wasm.ts:257,261).
- Phase 2: bridge int32_t → size_t for size/offset params on
  ~10 functions (ctx_create, tensor_set_data, tensor_get_data,
  op_view_2d/3d, backend_tensor_set/set3/get, async_begin/finish,
  graph_new). Conservative ABI hardening; no callsite at the ≤30B
  ceiling exceeds 2 GiB on a single transfer.
- Phase 3: smoke-test/real-model-page.js GGUF loader needs BigInt
  size argument when wasm.malloc(total) for total > 2^31 (e.g.
  13B Q4_K_S 7.4 GiB).
- Phase 7: mistral-7b-q5km / mistral-13b-q4ks / llama-3.1-13b-iq3m
  shortlisted as >4 GiB validation targets.

No source code modified in this commit — punch list grounds Phase 1.
EOF
)"
```

- [ ] **Step 5: Verify Phase 0 commit**

Run: `git log --oneline -1`
Expected: starts with `docs(plan): MEMORY64 migration Phase 0`.

---

## Task 2: Phase 1 — JS-side `_malloc` / `_free` migration to `_bridge_malloc` / `_bridge_free` (commit 2 of 8)

**Files:**
- Modify: `src/inference/ggml-wasm.ts:254-262` (the `malloc` / `free` wrappers — single source of truth)
- Modify: `src/inference/ggml-wasm.ts:196-236` (`init()` — detect wasm32 vs wasm64 once)
- Add field: `src/inference/ggml-wasm.ts` (private `is64: boolean`)
- Test: `tests/ggml-wasm.test.ts` (extend existing tests + add bridge-routing test)

**Purpose:** Switch every JS-side allocator call from stdlib
`_malloc` / `_free` to the custom-export `_bridge_malloc` / `_bridge_free`.
After this task the same TS code works under both wasm32 and wasm64
binaries — wrappers detect ABI shape at `init()` and normalize BigInt
return values to `number` (safe because no single allocation exceeds 2^53
bytes even at 30B IQ3_M scale; largest tensor ≈ 850 MB).

**Why route through wrappers, not call `_bridge_malloc` directly at every
call site:** §31a established the asymmetry (`_malloc` returns Number,
`_bridge_malloc` returns BigInt under wasm64). Putting the normalization
in `GgmlWasm.malloc/.free` means the rest of the codebase keeps its
`number` typing — no callsite-by-callsite BigInt churn.

- [ ] **Step 1: Read the existing test file to match conventions**

Run: `bun test tests/ggml-wasm.test.ts --reporter=verbose`
Expected: PASS (current baseline). If any pre-existing failure, halt and
investigate before starting Phase 1.

- [ ] **Step 2: Write the failing test for bridge-routing**

Add to `tests/ggml-wasm.test.ts`:

```ts
import { describe, expect, test } from "bun:test";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";

describe("GgmlWasm.malloc/free routes through bridge_malloc/bridge_free", () => {
  test("malloc calls _bridge_malloc, not _malloc", () => {
    let bridgeCalled = 0;
    let stdlibCalled = 0;
    const fakeModule = {
      _bridge_malloc: (size: number) => {
        bridgeCalled++;
        return 0xac0000 + size;
      },
      _bridge_free: (_ptr: number) => {},
      _malloc: (_size: number) => {
        stdlibCalled++;
        return 0;
      },
      _free: (_ptr: number) => {},
      HEAPU8: new Uint8Array(64),
      HEAPF32: new Float32Array(16),
    };
    const wasm = new GgmlWasm();
    // biome-ignore lint/suspicious/noExplicitAny: test injection
    (wasm as any).m = fakeModule;
    // biome-ignore lint/suspicious/noExplicitAny: test injection
    (wasm as any).is64 = false;

    const ptr = wasm.malloc(16);
    expect(ptr).toBe(0xac0010);
    wasm.free(ptr);

    expect(bridgeCalled).toBe(1);
    expect(stdlibCalled).toBe(0);
  });

  test("malloc normalizes BigInt return value to number under wasm64", () => {
    const fakeModule = {
      _bridge_malloc: (size: bigint) => {
        expect(typeof size).toBe("bigint");
        return BigInt(0xac0000) + size;
      },
      _bridge_free: (_ptr: bigint) => {},
      HEAPU8: new Uint8Array(64),
      HEAPF32: new Float32Array(16),
    };
    const wasm = new GgmlWasm();
    // biome-ignore lint/suspicious/noExplicitAny: test injection
    (wasm as any).m = fakeModule;
    // biome-ignore lint/suspicious/noExplicitAny: test injection
    (wasm as any).is64 = true;

    const ptr = wasm.malloc(16);
    expect(typeof ptr).toBe("number");
    expect(ptr).toBe(0xac0010);
    wasm.free(ptr);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test tests/ggml-wasm.test.ts -t "routes through bridge_malloc"`
Expected: FAIL — `wasm.malloc` currently calls `m._malloc`, so
`bridgeCalled === 0` and `stdlibCalled === 1`.

- [ ] **Step 4: Add the `is64` field and detection in `init()`**

In `src/inference/ggml-wasm.ts`, after the existing private fields
(around line 100, after `private graphComputeQueue`), add:

```ts
	/**
	 * Set during `init()`. `true` if the loaded module was built with
	 * `-sMEMORY64=1 -sWASM_BIGINT=1` — pointer-returning custom exports
	 * (`_bridge_malloc`, `_tensor_new_*`, etc.) return BigInt values
	 * that the wrappers narrow to `number` because no single allocation
	 * in this codebase exceeds 2^53 bytes (largest tensor at 30B IQ3_M
	 * ≈ 850 MB; full 13B Q4_K_S model file ≈ 7.4 GiB ≪ 2^53). See the
	 * MEMORY64 migration plan for the cap analysis.
	 */
	private is64 = false;
```

In `init()` (around line 228, after `this.installAsyncTensorGetNotifier()`
and before the `_webgpu_init` call), add detection:

```ts
		// Detect wasm32 vs wasm64 ABI shape: under MEMORY64 + WASM_BIGINT,
		// custom-export pointer returns are BigInt; under wasm32 they're
		// Number. A zero-byte _bridge_malloc returns the heap pointer to
		// an empty allocation; both binaries support it.
		const probe = this.m._bridge_malloc(this.m.HEAPU8 ? 0n : 0);
		this.is64 = typeof probe === "bigint";
		this.m._bridge_free(probe);
```

(Note: the conditional `0n : 0` is defensive for the probe call itself —
under wasm64 `_bridge_malloc` requires a BigInt arg, while wasm32
accepts Number. The `HEAPU8` check is just a non-null sentinel; once the
probe completes, `this.is64` is the load-bearing flag.)

Replace the `malloc` / `free` wrappers (lines 254-262):

```ts
	malloc(size: number): number {
		if (this.is64) {
			const ptr = this.m._bridge_malloc(BigInt(size));
			return Number(ptr);
		}
		return this.m._bridge_malloc(size);
	}

	free(ptr: number): void {
		if (this.is64) {
			this.m._bridge_free(BigInt(ptr));
			return;
		}
		this.m._bridge_free(ptr);
	}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test tests/ggml-wasm.test.ts -t "routes through bridge_malloc"`
Expected: PASS (both sub-tests).

- [ ] **Step 6: Run the full GgmlWasm test suite to confirm no regression**

Run: `bun test tests/ggml-wasm.test.ts`
Expected: all tests PASS, including the existing baseline ones that
exercise the wrappers indirectly via `uploadToTensor` etc. If the
existing tests reference `_malloc` directly (the original baseline did
at line 10/48), update them to also stub `_bridge_malloc` /
`_bridge_free`.

- [ ] **Step 7: Run full check**

Run: `make checkall`
Expected: 0 failures. (Test count changes by +2 for the new sub-tests.)

- [ ] **Step 8: Smoke-bench parity probe under wasm32 (zero-regression confirmation)**

Phase 1 is a pure-routing change under wasm32 — `_bridge_malloc` and
`_malloc` go to the same `std::malloc`. Confirm no regression on the
canonical TinyLlama target before committing:

```bash
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
```

Expected: median tok/s within ±3% of the post-§32 baseline (110.8 tok/s
per `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`).

- [ ] **Step 9: Commit Phase 1**

```bash
git add src/inference/ggml-wasm.ts tests/ggml-wasm.test.ts
git commit -m "$(cat <<'EOF'
feat(wasm): route GgmlWasm.malloc/.free through bridge_malloc/_free

Phase 1 of the MEMORY64 full migration. Switch the GgmlWasm.malloc /
.free wrappers from stdlib _malloc / _free to the custom-export
_bridge_malloc / _bridge_free. Pre-existing under both wasm32 and
wasm64 binaries (added by §31a; src/wasm/CMakeLists.txt:51).

Detect wasm32 vs wasm64 ABI shape once during init() via a zero-byte
probe call; under wasm64 the wrappers convert size→BigInt at the
boundary and narrow the BigInt return to Number (safe because no
single allocation exceeds 2^53 bytes even at 30B IQ3_M scale).

Wasm32 behavior is bit-identical pre/post (both names route to the
same std::malloc). Wasm64 behavior is the precondition for Phase 4's
production dual-binary build.

Verified:
- bun test tests/ggml-wasm.test.ts (2 new + existing baseline pass)
- make checkall clean
- make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
  within ±3% of 110.8 tok/s baseline.
EOF
)"
```

- [ ] **Step 10: Verify commit**

Run: `git log --oneline -2`
Expected: top is `feat(wasm): route GgmlWasm.malloc/.free through bridge_malloc/_free`.

---

## Task 3: Phase 2 — Bridge ABI hardening (`int32_t` → `size_t`) (commit 3 of 8)

**Files:**
- Modify: `src/wasm/webgpu-bridge.cpp` — promote `int32_t size` / `int32_t offset` params on the functions enumerated in PUNCH-LIST.md.
- Modify: `src/inference/ggml-wasm.ts` — corresponding TS bindings continue to accept `number` (Emscripten auto-marshals number→i64 under wasm64 with `WASM_BIGINT` for `size_t` params).

**Purpose:** Promote bridge param signatures so the linker emits
BigInt-marshaling JS shims under wasm64 when these bindings cross >2 GiB
single buffers in a future scope expansion. **At the ≤30B ceiling no
single transfer exceeds 2 GiB**, so this is conservative ABI hardening,
not a load-bearing functional change.

**No TS callsite changes required.** Emscripten's `WASM_BIGINT` runtime
auto-marshals JS `number` → wasm `i64` for `size_t` params at the FFI
boundary. The wrappers in `ggml-wasm.ts` keep their `: number` types.

- [ ] **Step 1: Apply the bridge signature changes**

In `src/wasm/webgpu-bridge.cpp`, make the following edits (file:line refs
match the punch list):

Replace line 59:

```cpp
int32_t ctx_create(int32_t mem_size) {
```

with:

```cpp
int32_t ctx_create(size_t mem_size) {
```

Replace lines 129-135:

```cpp
void tensor_set_data(void* tensor, const void* data, int32_t size) {
    memcpy(((struct ggml_tensor*)tensor)->data, data, size);
}

void tensor_get_data(void* tensor, void* out, int32_t size) {
    memcpy(out, ((struct ggml_tensor*)tensor)->data, size);
}
```

with:

```cpp
void tensor_set_data(void* tensor, const void* data, size_t size) {
    memcpy(((struct ggml_tensor*)tensor)->data, data, size);
}

void tensor_get_data(void* tensor, void* out, size_t size) {
    memcpy(out, ((struct ggml_tensor*)tensor)->data, size);
}
```

Replace lines 187-194:

```cpp
void* op_view_2d(void* x, int32_t ne0, int32_t ne1, int32_t nb1, int32_t offset) {
    return ggml_view_2d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, nb1, offset);
}

void* op_view_3d(void* x, int32_t ne0, int32_t ne1, int32_t ne2,
                 int32_t nb1, int32_t nb2, int32_t offset) {
    return ggml_view_3d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, ne2, nb1, nb2, offset);
}
```

with:

```cpp
void* op_view_2d(void* x, int32_t ne0, int32_t ne1, int32_t nb1, size_t offset) {
    return ggml_view_2d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, nb1, offset);
}

void* op_view_3d(void* x, int32_t ne0, int32_t ne1, int32_t ne2,
                 int32_t nb1, int32_t nb2, size_t offset) {
    return ggml_view_3d(current_ctx(), (struct ggml_tensor*)x, ne0, ne1, ne2, nb1, nb2, offset);
}
```

Replace line 272-274:

```cpp
void* graph_new(int32_t size) {
    return ggml_new_graph_custom(current_ctx(), (size_t)size, false);
}
```

with:

```cpp
void* graph_new(size_t size) {
    return ggml_new_graph_custom(current_ctx(), size, false);
}
```

Replace line 296:

```cpp
void backend_tensor_set(void* tensor, const void* data, int32_t offset, int32_t size) {
    ggml_backend_tensor_set((struct ggml_tensor*)tensor, data, offset, size);
}
```

with:

```cpp
void backend_tensor_set(void* tensor, const void* data, size_t offset, size_t size) {
    ggml_backend_tensor_set((struct ggml_tensor*)tensor, data, offset, size);
}
```

Replace lines 303-311:

```cpp
void backend_tensor_set3(
    void* t1, const void* d1, int32_t sz1,
    void* t2, const void* d2, int32_t sz2,
    void* t3, const void* d3, int32_t sz3
) {
    if (t1) ggml_backend_tensor_set((struct ggml_tensor*)t1, d1, 0, sz1);
    if (t2) ggml_backend_tensor_set((struct ggml_tensor*)t2, d2, 0, sz2);
    if (t3) ggml_backend_tensor_set((struct ggml_tensor*)t3, d3, 0, sz3);
}
```

with:

```cpp
void backend_tensor_set3(
    void* t1, const void* d1, size_t sz1,
    void* t2, const void* d2, size_t sz2,
    void* t3, const void* d3, size_t sz3
) {
    if (t1) ggml_backend_tensor_set((struct ggml_tensor*)t1, d1, 0, sz1);
    if (t2) ggml_backend_tensor_set((struct ggml_tensor*)t2, d2, 0, sz2);
    if (t3) ggml_backend_tensor_set((struct ggml_tensor*)t3, d3, 0, sz3);
}
```

Replace line 313:

```cpp
void backend_tensor_get(void* tensor, void* out, int32_t offset, int32_t size) {
    ggml_backend_tensor_get((struct ggml_tensor*)tensor, out, offset, size);
}
```

with:

```cpp
void backend_tensor_get(void* tensor, void* out, size_t offset, size_t size) {
    ggml_backend_tensor_get((struct ggml_tensor*)tensor, out, offset, size);
}
```

Replace lines 317-322:

```cpp
int32_t backend_tensor_get_async_begin(void* tensor, int32_t offset, int32_t size) {
    return ggml_backend_webgpu_tensor_get_async_begin(
        (const struct ggml_tensor*)tensor,
        (size_t)offset,
        (size_t)size);
}
```

with:

```cpp
int32_t backend_tensor_get_async_begin(void* tensor, size_t offset, size_t size) {
    return ggml_backend_webgpu_tensor_get_async_begin(
        (const struct ggml_tensor*)tensor,
        offset,
        size);
}
```

Replace lines 328-330:

```cpp
void backend_tensor_get_async_finish(int32_t request_id, void* out, int32_t size) {
    ggml_backend_webgpu_tensor_get_async_finish(request_id, out, (size_t)size);
}
```

with:

```cpp
void backend_tensor_get_async_finish(int32_t request_id, void* out, size_t size) {
    ggml_backend_webgpu_tensor_get_async_finish(request_id, out, size);
}
```

- [ ] **Step 2: Rebuild WASM (wasm32 path — production binary)**

```bash
make wasm-clean
make wasm-build
```

Expected: build completes; `src/wasm/build/webllm-wasm.wasm` regenerated.

- [ ] **Step 3: Rebuild WASM (wasm64 probe binary — confirm dual-binary still links)**

```bash
make mem64-probe
```

Expected: build completes (the probe Make target also rebuilds the
mem64 binary; URL is printed at the end). The `int32_t → size_t`
promotion is the change wasm64 actually wants — the linker should now
emit BigInt JS shims for these params. No need to navigate the URL in
this step; just confirm the build link succeeds.

- [ ] **Step 4: Run check + smoke-bench parity under wasm32**

```bash
make checkall
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
```

Expected:
- `make checkall` PASS.
- `make smoke-bench` median tok/s within ±3% of the 110.8 baseline.
  Phase 2 is a pure type-promotion at the ABI boundary — under wasm32
  `size_t` is already 32-bit so the binary is bit-identical at the
  amalgamated `int32_t`-cast layer. Any drift is noise.

- [ ] **Step 5: Smoke check — interactive chat box**

Open the smoke page in the existing agentchrome tab; send one short
prompt; confirm a coherent reply lands. This catches any silent ABI
drift the `make checkall` test suite might miss (the existing test
suite covers the bridge surface but with stub modules; this exercises
the real WASM build end-to-end).

```bash
agentchrome connect --status
# Reuse existing session/tab per CLAUDE.md.
agentchrome --port <PORT> tabs list
agentchrome --port <PORT> navigate "http://localhost:8031/?v=$(date +%s)" --tab <TAB_ID>
# Watch the smoke page complete steps 1-8; type "Hi, what is 2+2?" in the chat
# box; confirm a coherent reply.
```

Expected: chat reply lands within ~10s; no console errors.

- [ ] **Step 6: Commit Phase 2**

```bash
git add src/wasm/webgpu-bridge.cpp
git commit -m "$(cat <<'EOF'
refactor(bridge): promote int32_t size/offset params to size_t

Phase 2 of the MEMORY64 full migration. Promote bridge param
signatures so the Emscripten linker emits BigInt-marshaling JS
shims under wasm64 for the >2 GiB-capable transfer paths.

Functions touched (matches PUNCH-LIST.md): ctx_create,
tensor_set_data, tensor_get_data, op_view_2d, op_view_3d,
graph_new, backend_tensor_set, backend_tensor_set3,
backend_tensor_get, backend_tensor_get_async_begin,
backend_tensor_get_async_finish.

At the ≤30B project ceiling no single transfer exceeds 2 GiB
(largest tensor ≈ 850 MB at 30B IQ3_M; full 13B Q4_K_S model
file ≈ 7.4 GiB but uploaded chunked at 4 MiB). This is
conservative ABI hardening — not a functional change at the
current size band.

No TS callsite change required; Emscripten WASM_BIGINT runtime
auto-marshals number → i64 for size_t params.

Verified:
- make wasm-build (wasm32) clean
- make mem64-probe (wasm64 link) clean
- make checkall clean
- smoke-bench tinyllama within ±3% of 110.8 baseline
- agentchrome smoke chat: coherent reply
EOF
)"
```

---

## Task 4: Phase 3 — GGUF loader BigInt boundary verification (commit 4 of 8)

**Files:**
- Modify (if needed): `smoke-test/real-model-page.js` — `wasm.malloc(total)` call site at line 306, plus the `set` / Uint8Array view callers at lines 318, 341, 364, 386, 438.

**Purpose:** Verify the GGUF streaming loader handles the BigInt boundary
correctly when `total > 2^31`. Phase 1 already routes `wasm.malloc(total)`
through `_bridge_malloc(BigInt(total))` under wasm64; Phase 3 confirms
no `Number` narrowing leak occurs anywhere in the loader's call chain
that would silently truncate a large model file allocation.

**Expected outcome: zero source-code change.** Phase 1's wrapper
normalizes `_bridge_malloc`'s BigInt return to `number`, which is safe
because:
- The single largest allocation on the loader path is the full GGUF
  file (~7.4 GiB for 13B Q4_K_S, ~12.8 GiB for 30B IQ3_M).
- All these values are well under `Number.MAX_SAFE_INTEGER` ≈ 9×10^15
  (2^53). Pointer arithmetic (`modelPtr + received` at line 318,
  `modelPtr + off` at line 341) stays exact in JS Number.
- `Uint8Array(buffer, byteOffset, length)` accepts `number` for both
  `byteOffset` and `length`.

If Phase 3 surfaces a leak, it's a Phase 1 wrapper bug, not a Phase 3
loader bug. The verification gate below catches it.

- [ ] **Step 1: Audit the loader for any `Number(...)` or bitwise op that could narrow**

Run: `grep -n 'Number\|>>\|>>>\|| 0\|| 0)' smoke-test/real-model-page.js`
Expected: no `Number(...)` calls, no `>>>` (unsigned right shift)
truncating to 32-bit, no `| 0` integer-cast on heap-pointer-derived
values. If any are found, that's a leak that needs a fix in this task.

- [ ] **Step 2: Static analysis — confirm `wasm.malloc(total)` lands in `number` typing**

Read `smoke-test/real-model-page.js:306-308`:

```js
modelPtr = wasm.malloc(total);
if (!modelPtr) throw new Error(`wasm malloc(${total}) returned null`);
modelByteLength = total;
```

`total` comes from `Number(resp.headers.get("content-length") || 0)` at
line 300 — already a `number`. `wasm.malloc(number)` returns `number`
under both binaries (per Phase 1's wrapper contract). `modelPtr` is
typed `number` implicitly. ✅

- [ ] **Step 3: Confirm pointer arithmetic stays exact**

Read `smoke-test/real-model-page.js:318` and `:341`:

```js
wasm.heapU8.set(value, modelPtr + received);
// and:
new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);
```

Both operations require `modelPtr + received < 2^53` and
`modelPtr + off < 2^53` to avoid silent precision loss. For 30B IQ3_M
at ~12.8 GiB, `modelPtr` is at most `2^32 + slop ≈ 5×10^9`; `received`
is at most `total ≤ 14×10^9`. Sum ≈ 2×10^10 ≪ 2^53. ✅

- [ ] **Step 4: Confirm `uploadRangeChunked` still re-derives heap views per chunk**

Re-read `src/inference/ggml-wasm.ts:409-426`:

```ts
uploadRangeChunked(
    tensor: TensorPtr,
    dataAt: (srcOffset: number, byteLength: number) => Uint8Array,
    byteLength: number,
    chunkSize = 4 * 1024 * 1024,
): void {
    const ptr = this.malloc(Math.min(chunkSize, byteLength));
    try {
        for (let off = 0; off < byteLength; off += chunkSize) {
            const end = Math.min(off + chunkSize, byteLength);
            const slice = dataAt(off, end - off);
            this.heapU8.set(slice, ptr);
            this.m._backend_tensor_set(tensor, ptr, off, slice.byteLength);
        }
    } finally {
        this.free(ptr);
    }
}
```

The `dataAt` callback is invoked *after* `this.malloc(...)`, which
means any heap growth from that malloc cannot detach a stale source
view between derivation and `set`. ✅ Pre-existing fix; no change.

- [ ] **Step 5: Run agentchrome large-model smoke (>2 GiB, ≤4 GiB on wasm32)**

Pick the largest currently-supported model that fits the wasm32 cap as a
sanity check that Phase 1+2 didn't regress the loader path. Mistral-7B
Q4_K_S at 3953 MB is the largest in-fleet wasm32 candidate.

```bash
make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q4ks PERF_RUNS=3
```

Expected: median tok/s within ±3% of the post-§32 baseline (35.0 tok/s).
This exercises the GGUF streaming loader on a 4 GiB-class model under
wasm32 — the largest scale Phase 3 is verifiable at *before* Phase 4
ships the wasm64 production binary.

- [ ] **Step 6: Run check**

```bash
make checkall
```

Expected: 0 failures.

- [ ] **Step 7: Commit Phase 3 (zero source change is the expected outcome)**

If Steps 1-4 surfaced no leak (the expected path), commit a docs-only
verification note:

```bash
git add -f eval/reports/memory64-migration-2026-04-28/PHASE-3-VERIFY.md
git commit -m "$(cat <<'EOF'
docs(plan): MEMORY64 migration Phase 3 — GGUF loader verification

Phase 3 of the MEMORY64 full migration. Static + dynamic
verification that smoke-test/real-model-page.js handles the
BigInt boundary correctly when wasm.malloc(total) crosses
2^31 (e.g. 13B Q4_K_S at 7.4 GiB).

Findings:
- wasm.malloc/_free wrappers (Phase 1) normalize BigInt → number
  at the ABI boundary; loader code keeps `number` typing.
- Pointer arithmetic (modelPtr + received, modelPtr + off) stays
  exact for any value ≤ 2^53 ≈ 9×10^15. At the 30B IQ3_M
  ceiling (~12.8 GiB) the maximum operand sum is ~2×10^10,
  well under the safe-integer cap.
- uploadRangeChunked re-derives the heap view per chunk after
  malloc — the pre-existing heap-grow detachment fix carries
  through unchanged.

Wasm32 mistral-7b-q4ks smoke-bench: within ±3% of the 35.0
baseline. No source-code change required.

The full validation (real >4 GiB load on the wasm64 binary)
fires in Phase 7; Phase 3 is the static gate that says the
loader contract is correct under both binaries.
EOF
)"
```

Where `PHASE-3-VERIFY.md` is a brief 30-50 line note containing the
findings above plus the smoke-bench output.

If Steps 1-4 *did* surface a leak, fix it in `smoke-test/real-model-page.js`,
re-run Steps 5-6, and commit the fix instead with subject
`fix(loader): close BigInt narrowing leak at <site>`.

---

## Task 5: Phase 4 — Production MEMORY64 build (commit 5 of 8)

**Files:**
- Modify: `Makefile` — extend `make wasm-build` to produce both binaries; extend `smoke-test` target to copy both binaries; add `wasm-build-mem64-only` for fast iteration.
- Modify: `smoke-test/real-model.html` (or `real-model-page.js`) — add a query-string toggle (`?wasm=mem64`) to load `webllm-wasm-mem64.js` instead of `webllm-wasm.js`. Default off (preserves current behavior).
- Modify: `tsconfig.json` — already excludes `src/wasm/build`; verify `src/wasm/build-mem64` is also excluded (already done by §31a's amendment).

**Purpose:** Wire the wasm64 binary into the production build pipeline so
both `webllm-wasm.{js,wasm}` and `webllm-wasm-mem64.{js,wasm}` are
emitted by `make wasm-build` and copied into `smoke-test/` by
`make smoke-test`. Add a smoke-page toggle so Phase 5 can A/B the two
binaries on the same Chrome session without rebuilds.

**Defer Phase 6's "single vs dual binary" decision until after Phase 5
measurement.** This phase produces the dual-binary pipeline; Phase 6
applies the decision criterion to the measurement data.

- [ ] **Step 1: Extend `make wasm-build` to produce both binaries**

Replace the existing `wasm-build` target in `Makefile` (lines 75-95):

```make
wasm-build: wasm-build-wasm32 wasm-build-mem64 ## Build both wasm32 (4 GiB cap) and wasm64 (16 GiB cap) binaries

wasm-build-wasm32: ## Build only the wasm32 production binary (current default)
	cd src/wasm && mkdir -p build && cd build && \
	source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CPU=OFF \
		-DGGML_BLAS=OFF \
		-DGGML_METAL=OFF \
		-DGGML_ACCELERATE=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_LLAMAFILE=OFF \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BACKEND_DL=OFF \
		-DWEBLLM_ASSERTIONS=$(WEBLLM_ASSERTIONS) && \
	cmake --build . --config Release -j

wasm-build-mem64: ## Build only the wasm64 (MEMORY64) production binary
	cd src/wasm && mkdir -p build-mem64 && cd build-mem64 && \
	source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CPU=OFF \
		-DGGML_BLAS=OFF \
		-DGGML_METAL=OFF \
		-DGGML_ACCELERATE=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_LLAMAFILE=OFF \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BACKEND_DL=OFF \
		-DWEBLLM_BUILD_MEM64=ON \
		-DCMAKE_C_FLAGS="-sMEMORY64=1" \
		-DCMAKE_CXX_FLAGS="-sMEMORY64=1" \
		-DWEBLLM_ASSERTIONS=$(WEBLLM_ASSERTIONS) && \
	cmake --build . --target webllm-wasm-mem64 --config Release -j
```

The two new targets share their flags with `mem64-probe` (lines
118-141) but separate the build step from the probe-page-restart step.

- [ ] **Step 2: Extend `make smoke-test` to copy both binaries**

Replace the `smoke-test` target (lines 114-116):

```make
smoke-test: wasm-build ## Bundle + copy WASM artifacts into smoke-test/
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
	cp src/wasm/build-mem64/webllm-wasm-mem64.js src/wasm/build-mem64/webllm-wasm-mem64.wasm smoke-test/
```

- [ ] **Step 3: Refactor `mem64-probe` target to reuse `wasm-build-mem64`**

Replace the existing `mem64-probe` target (lines 118-150) with:

```make
mem64-probe: wasm-build-mem64 ## Build wasm64 mem64 binary, copy to smoke-test/, restart smoke server, print probe URL
	cp src/wasm/build-mem64/webllm-wasm-mem64.js src/wasm/build-mem64/webllm-wasm-mem64.wasm smoke-test/
	@lsof -ti:$(SMOKE_PORT) | xargs kill -9 2>/dev/null || true
	@bun run eval/smoke-serve.ts --port $(SMOKE_PORT) >/dev/null 2>&1 &
	@sleep 1
	@echo ""
	@echo "MEMORY64 cap probe ready. Open in your existing agentchrome tab:"
	@echo "  http://localhost:$(SMOKE_PORT)/mem64-probe.html?v=$$(date +%s)"
	@echo ""
	@echo "Result blob will be at window.__memory64ProbeResult after the harness completes."
```

- [ ] **Step 4: Add the smoke-page wasm-binary toggle**

In `smoke-test/real-model-page.js`, locate the WASM module URL
(currently hardcoded as `webllm-wasm.js`). Replace the bare reference
with a query-string-driven selection. The exact line varies — `grep -n
"webllm-wasm.js" smoke-test/real-model-page.js` finds it; the typical
pattern is in a `WebLLM.create({ wasmUrl: ... })` call or a direct
`new URL("webllm-wasm.js", ...)`.

Replace whatever the existing bare reference is with:

```js
const params = new URLSearchParams(window.location.search);
const wasmVariant = params.get("wasm") === "mem64" ? "webllm-wasm-mem64.js" : "webllm-wasm.js";
const wasmUrl = new URL(wasmVariant, window.location.href).toString();
// ... pass wasmUrl into WebLLM.create / engine.init
```

(If the existing code already uses `new URL("webllm-wasm.js", ...)` the
diff is just the variant-selection bit before the `new URL` call.)

- [ ] **Step 5: Build both binaries and verify outputs**

```bash
make wasm-clean && rm -rf src/wasm/build-mem64
make wasm-build
ls -la src/wasm/build/webllm-wasm.{js,wasm} src/wasm/build-mem64/webllm-wasm-mem64.{js,wasm}
```

Expected: all four files exist, each 1-5 MiB. Note the wasm size delta
between wasm32 and wasm64 (recorded in the Phase 5 summary).

- [ ] **Step 6: Bundle copy into smoke-test/**

```bash
make smoke-test
ls -la smoke-test/webllm-wasm{,-mem64}.{js,wasm}
```

Expected: all four files copied.

- [ ] **Step 7: Smoke verify both binaries load**

```bash
make smoke-restart
agentchrome --port <PORT> navigate "http://localhost:8031/?v=$(date +%s)" --tab <TAB_ID>
# Confirm steps 1-8 pass (wasm32 default).
agentchrome --port <PORT> navigate "http://localhost:8031/?v=$(date +%s)&wasm=mem64" --tab <TAB_ID>
# Confirm steps 1-8 pass (wasm64 binary).
```

Expected: both runs complete steps 1-8; chat box returns coherent
replies on both. The wasm64 run will be slightly slower (Phase 5
quantifies it) but functional.

- [ ] **Step 8: Run check**

```bash
make checkall
```

Expected: 0 failures.

- [ ] **Step 9: Commit Phase 4**

```bash
git add Makefile smoke-test/real-model-page.js
git commit -m "$(cat <<'EOF'
feat(build): MEMORY64 dual-binary production build

Phase 4 of the MEMORY64 full migration. make wasm-build now
produces both webllm-wasm.{js,wasm} (wasm32, 4 GiB cap, current
default) and webllm-wasm-mem64.{js,wasm} (wasm64, 16 GiB cap).
make smoke-test copies both into smoke-test/.

Smoke page accepts a ?wasm=mem64 query string to load the
wasm64 binary; default behavior is unchanged (loads wasm32).
This is the substrate for the Phase 5 bench-parity sweep on
the canonical 6 fleet — same Chrome session, same TS bundle,
just a query-string A/B.

Refactor: mem64-probe target now depends on wasm-build-mem64
(shared build step) instead of duplicating the cmake configure
inline.

Verified:
- make wasm-build emits all four artifacts.
- make smoke-test copies both into smoke-test/.
- agentchrome smoke (wasm32 default + ?wasm=mem64) both
  complete steps 1-8 with coherent chat replies.
- make checkall clean.
EOF
)"
```

---

## Task 6: Phase 5 — Bench parity gates on canonical 6 fleet (commit 6 of 8 — measurement)

**Files:**
- Create: `eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md`
- Append to: `eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md` per-model speed-run JSON snippets.

**Purpose:** Run `make smoke-bench` on each of the canonical 6 models
under both binaries (wasm32 default + `?wasm=mem64`); compare against
the pre-rebase baselines. **Gate: zero regression ≥3% on tok/s for any
of the 6 models.** Block on failure; diagnose before Phase 6.

**Why measurement before deploy decision (Phase 6):** the entire dual-
vs-single-binary deployment criterion (TODO §MEMORY64 Phase 6) hinges
on this delta. A separate measurement-only commit makes the data
revertable without unwinding the deploy decision and lets Phase 6 cite
this commit's report directly.

**Canonical 6 fleet** (per CLAUDE.md and pre-rebase baselines):

| Model | Wasm32 baseline (tok/s) |
|---|---:|
| `tinyllama-1.1b-chat-q4_0` | 110.8 |
| `qwen3-0.6b-q4f16` | 89.8 |
| `qwen3-1.7b-q4f16` | 62.2 |
| `mistral-7b-instruct-v0.3-q4ks` | 35.0 |
| `llama-3.1-8b-instruct-iq3m` | 27.2 |
| `qwen3-8b-iq3m` | 27.2 |

(Source: `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`,
3-run median, non-profile mode.)

- [ ] **Step 1: Capture wasm32 baselines (sanity — should match pinned)**

Phase 1+2+4 may have drifted the wasm32 binary by noise; re-baseline
the canonical 6 under the current code first so Phase 5 compares
apples-to-apples.

```bash
for m in tinyllama-1.1b-chat-q4_0 qwen3-0.6b-q4f16 qwen3-1.7b-q4f16 \
         mistral-7b-instruct-v0.3-q4ks llama-3.1-8b-instruct-iq3m \
         qwen3-8b-iq3m; do
  make smoke-bench PERF_MODEL=$m PERF_RUNS=3 \
    | tee eval/reports/memory64-migration-2026-04-28/wasm32-$m.txt
done
```

Expected: each model within ±3% of its pinned baseline above. **If any
drifts >3% under wasm32, Phase 5 halts and the regression is diagnosed
against Phases 1-4 before continuing.** Likely culprits would be the
Phase 1 wrapper or Phase 2 ABI promotion; revert those one at a time
to bisect.

- [ ] **Step 2: Capture wasm64 numbers**

Run the same fleet under the wasm64 binary by setting an env var or
extending the `make smoke-bench` target. The simplest path is a
new opt-in env that flows through to `eval/perf.ts`:

In `Makefile`, locate `bench-inference` / `smoke-bench` and add
`WEBLLM_WASM_VARIANT` propagation:

```make
smoke-bench: smoke-restart ## End-to-end inference benchmark (auto-launches agentchrome if needed)
	@echo "=== smoke-bench: $(PERF_MODEL)$(if $(PERF_DRAFTER), drafter=$(PERF_DRAFTER)), $(PERF_RUNS) runs ==="
	WEBLLM_WASM_VARIANT=$(WASM_VARIANT) bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS) --profile $(if $(PERF_DRAFTER),--drafter $(PERF_DRAFTER))
```

In `eval/perf.ts`, locate the URL it navigates to (typical pattern:
`http://localhost:8031/...?v=...&model=...`); append `&wasm=mem64`
when `WEBLLM_WASM_VARIANT === "mem64"`.

(Exact line in `eval/perf.ts` depends on its current shape. Use
`grep -n 'localhost\|navigate\|smoke-test' eval/perf.ts` to locate.)

Then run:

```bash
for m in tinyllama-1.1b-chat-q4_0 qwen3-0.6b-q4f16 qwen3-1.7b-q4f16 \
         mistral-7b-instruct-v0.3-q4ks llama-3.1-8b-instruct-iq3m \
         qwen3-8b-iq3m; do
  make smoke-bench PERF_MODEL=$m PERF_RUNS=3 WASM_VARIANT=mem64 \
    | tee eval/reports/memory64-migration-2026-04-28/wasm64-$m.txt
done
```

- [ ] **Step 3: Build the parity table**

Author `eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md`:

```markdown
# MEMORY64 migration — Phase 5 bench parity gates

**Date:** <YYYY-MM-DD>
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Pre-rebase reference:** [`eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md`](../pre-rebase-baselines-2026-04-28/SUMMARY.md)

## Headline

- **Gate (zero ≥3% regression on canonical 6 wasm64 vs wasm32):** PASS / FAIL.
- **Net delta range:** wasm64 is X% to Y% (median Z%) of wasm32 baseline.
- **Decision input for Phase 6:** [single-binary | dual-binary] per the
  TODO criterion (≤2% median regression → single).

## Per-model results

| Model | Wasm32 (current, tok/s) | Wasm64 (tok/s) | Δ vs wasm32 | Δ vs pinned baseline |
|---|---:|---:|---:|---:|
| tinyllama-1.1b-chat-q4_0 | <X> | <Y> | <±%> | <±%> |
| qwen3-0.6b-q4f16 | <X> | <Y> | <±%> | <±%> |
| qwen3-1.7b-q4f16 | <X> | <Y> | <±%> | <±%> |
| mistral-7b-instruct-v0.3-q4ks | <X> | <Y> | <±%> | <±%> |
| llama-3.1-8b-instruct-iq3m | <X> | <Y> | <±%> | <±%> |
| qwen3-8b-iq3m | <X> | <Y> | <±%> | <±%> |

## Wasm size delta

| Binary | Bytes |
|---|---:|
| webllm-wasm.wasm (wasm32) | <bytes> |
| webllm-wasm-mem64.wasm (wasm64) | <bytes> |
| Δ | <±%> |

## Environment

- Emscripten 5.0.6 (ref `6ea9c28c38cdd40c1032fa04400c9d16230ee180`).
- Chrome <version> via existing agentchrome session.
- Host: macOS 26.4.1, Apple M4 Max, 128 GiB RAM.

## Decision rule (TODO §MEMORY64 Phase 6)

- ≤2% median regression → ship MEMORY64-only (drops 4 GiB fast path; halves bundle complexity).
- 2-5% regression → ship dual binary; deploy-time selection.
- ≥5% regression → halt; diagnose pointer-overhead in hot paths before Phase 6.

## Per-model speed-run JSONs

(Captured to wasm32-<model>.txt and wasm64-<model>.txt; summary numbers
extracted from each run's `[smoke-bench summary]` block.)
```

Fill in the placeholders from Steps 1-2 outputs.

- [ ] **Step 4: Apply the gate**

If any model's wasm64 result regresses ≥3% vs its wasm32 number from
Step 1, **halt Phase 5 and Phase 6.** Diagnose:
- Common culprit: BigInt boxing overhead in hot paths (e.g. the FFI
  wrappers calling `BigInt(size)` per malloc). Profile with
  `make smoke-bench --profile` and check `backendEncodeOverheadMs`.
- Fix or revert the offending phase before continuing.

If all 6 are within ±3%, proceed to Step 5.

- [ ] **Step 5: Commit Phase 5 measurement**

```bash
git add -f eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md \
           eval/reports/memory64-migration-2026-04-28/wasm32-*.txt \
           eval/reports/memory64-migration-2026-04-28/wasm64-*.txt
git commit -m "$(cat <<'EOF'
docs(plan): MEMORY64 migration Phase 5 — bench parity gates

Phase 5 of the MEMORY64 full migration. Measurement-only commit:
canonical 6 fleet on wasm32 (current binary, post-Phases 1-4) and
wasm64 (new binary). Pre-rebase baselines at
eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md serve as
the wasm32 reference.

Gate: zero ≥3% regression wasm64 vs wasm32. Result: <PASS|FAIL>
with median delta <X%>.

Per-model wasm32 vs wasm64:
- tinyllama-1.1b-q4_0: <X> vs <Y> tok/s (<±%>)
- qwen3-0.6b-q4f16: <X> vs <Y> tok/s (<±%>)
- qwen3-1.7b-q4f16: <X> vs <Y> tok/s (<±%>)
- mistral-7b-q4ks: <X> vs <Y> tok/s (<±%>)
- llama-3.1-8b-iq3m: <X> vs <Y> tok/s (<±%>)
- qwen3-8b-iq3m: <X> vs <Y> tok/s (<±%>)

Decision input for Phase 6: <single-binary | dual-binary> per the
TODO criterion (≤2% median regression → single).
EOF
)"
```

---

## Task 7: Phase 6 — Single-vs-dual binary deployment decision (commit 7 of 8)

**Files:**
- Modify (one of two paths, depending on Phase 5 outcome):
  - **Path A — single-binary (≤2% regression):** delete `webllm-wasm.{js,wasm}` from the production pipeline; rename the mem64 build dir to be the canonical one; update `Makefile` `smoke-test` target to copy only the wasm64 artifacts; remove the `?wasm=mem64` toggle (mem64 is the default).
  - **Path B — dual-binary (>2% regression):** keep both binaries shipping; add deploy-time runtime selection that picks wasm64 if the configured model file exceeds 3.5 GiB (10% under the wasm32 cap), else wasm32. The selection lives in the public engine API — `WebLLM.create({ wasmUrl: ... })` keeps an explicit override.

**Purpose:** Apply the decision criterion from TODO §MEMORY64 Phase 6 to
the Phase 5 measurement data. This phase is the only one where the
correct edits depend on a measurement; both paths are pre-spec'd here so
the chosen one is mechanical.

- [ ] **Step 1: Read the Phase 5 result and pick the path**

Open `eval/reports/memory64-migration-2026-04-28/PHASE-5-PARITY.md`.

If the median regression on the canonical 6 is ≤2%, **Path A** (single-
binary). Otherwise **Path B** (dual-binary).

If 2-5% but a deployment ask names a 13B+ target with an active
performance budget, **Path B** is also acceptable (preserves wasm32 fast
path for ≤4 GiB models). Document the choice in the commit message.

- [ ] **Step 2 (Path A): Collapse to single-binary**

In `Makefile`, replace the `wasm-build` / `smoke-test` / `mem64-probe`
targets to reflect the wasm64 binary as canonical:

```make
wasm-build: ## Build the production wasm64 binary
	cd src/wasm && mkdir -p build && cd build && \
	source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CPU=OFF \
		-DGGML_BLAS=OFF \
		-DGGML_METAL=OFF \
		-DGGML_ACCELERATE=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_LLAMAFILE=OFF \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BACKEND_DL=OFF \
		-DWEBLLM_BUILD_MEM64=ON \
		-DCMAKE_C_FLAGS="-sMEMORY64=1" \
		-DCMAKE_CXX_FLAGS="-sMEMORY64=1" \
		-DWEBLLM_ASSERTIONS=$(WEBLLM_ASSERTIONS) && \
	cmake --build . --target webllm-wasm-mem64 --config Release -j

smoke-test: wasm-build
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm-mem64.js smoke-test/webllm-wasm.js
	cp src/wasm/build/webllm-wasm-mem64.wasm smoke-test/webllm-wasm.wasm
```

(The wasm64 binary is renamed at copy-time to `webllm-wasm.{js,wasm}`
so existing TS code that imports the bundle keeps working.)

Delete the `wasm-build-wasm32` and `wasm-build-mem64` targets (no
longer needed). Delete the `?wasm=mem64` toggle from
`smoke-test/real-model-page.js` (mem64 is the default).

- [ ] **Step 2 (Path B): Wire deploy-time selection**

Keep the dual-binary `Makefile` from Phase 4. In `src/core/engine.ts`
(or wherever `WebLLM.create` resolves the WASM URL), add a model-size-
aware default:

```ts
function pickWasmUrl(modelByteLength: number, override?: string): string {
	if (override) return override;
	const FOUR_GIB_MARGIN = 3.5 * 1024 * 1024 * 1024;
	return modelByteLength > FOUR_GIB_MARGIN
		? "webllm-wasm-mem64.js"
		: "webllm-wasm.js";
}
```

Wire it through the public API: when a model's `content-length` (or
`model.fileSize` from the registry) exceeds 3.5 GiB, default to the
mem64 binary; otherwise use wasm32. Document the override option in
the public `WebLLM.create` JSDoc.

- [ ] **Step 3: Run check + canonical-6 smoke under the chosen path**

```bash
make wasm-clean && rm -rf src/wasm/build-mem64
make wasm-build
make smoke-test
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=3
make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q4ks PERF_RUNS=3
make checkall
```

Expected: tok/s within ±3% of the Phase 5 numbers for whichever binary
the canonical 6 each route through under the new default. `make
checkall` clean.

- [ ] **Step 4: Commit Phase 6**

```bash
git add Makefile smoke-test/real-model-page.js src/core/engine.ts
git commit -m "$(cat <<'EOF'
feat(deploy): MEMORY64 deployment — <single-binary|dual-binary>

Phase 6 of the MEMORY64 full migration. Apply the decision rule
(TODO §MEMORY64 Phase 6) to the Phase 5 measurement data:

Phase 5 median regression (wasm64 vs wasm32): <X%>.
Path chosen: <Path A: single-binary | Path B: dual-binary>.

Path A: webllm-wasm-mem64.{js,wasm} is renamed to
webllm-wasm.{js,wasm} at copy-time; wasm32 build is dropped.
Existing imports (`./webllm-wasm.js`) keep working transparently.
Bundle complexity halves; ~5% pointer overhead is the price.

Path B: both binaries ship; engine.ts picks wasm64 when the model
file exceeds 3.5 GiB (10% under the wasm32 cap). Wasm32 fast
path preserved for ≤4 GiB targets; total bundle payload doubles
from 3.5 → 7 MiB. Public API gains a wasmUrl override.

Verified: make checkall clean; canonical models smoke-bench
within ±3% of Phase 5 numbers.
EOF
)"
```

---

## Task 8: Phase 7 — Register a >4 GiB validation target (commit 8 of 8)

**Files:**
- Modify: `eval/models.ts` — add the chosen 13B (or 7B-Q5_K_M) model entry.
- Modify: `eval/smoke-profiles.ts` — add a profile pinned to the new model at warm temp / off-thinking.
- Create: `eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`

**Purpose:** Register one model whose file exceeds the wasm32 4 GiB cap;
load it under the wasm64 binary; run a 36-prompt sanity eval. **Gate:
forward pass coherent on the eval; tok/s within architecturally
expected band.** Closes the migration with a real >4 GiB happy path.

**Candidate ranking** (from PUNCH-LIST.md):
1. `mistral-7b-instruct-v0.3-q5km` (~5.0 GiB) — first step above the
   cap; tightest pre/post comparison vs the canonical mistral-7B-Q4_K_S.
2. `mistral-13b-instruct-q4ks` (~7.4 GiB) — plan-target 13B; check HF
   for a coherent Q4_K_S GGUF (e.g. `TheBloke/Mistral-13B-Instruct-v0.1-GGUF`
   or a maintained mirror).
3. `llama-3.1-13b-instruct-iq3m` (~5.4 GiB) — alternative if Mistral
   13B Q4_K_S unavailable.

**Default pick:** **Mistral-7B Q5_K_M.** Smallest >4 GiB model that
exercises the path; reuses the Mistral-7B Q4_K_S smoke baseline for
direct quality comparison; lower download cost for repeat verification
than 13B. If a deployment ask names 13B specifically, swap to candidate
2 or 3 inside this same task.

- [ ] **Step 1: Verify the chosen model exists on HF**

```bash
# Mistral-7B Q5_K_M (default pick): TheBloke/Mistral-7B-Instruct-v0.3-GGUF
# (mirror that already publishes Q4_K_S — verify Q5_K_M is in the same repo)
curl -sI "https://huggingface.co/<repo>/resolve/main/<filename>" | head -1
```

Expected: `HTTP/2 200` (or 302 redirect to a 200). If 404, fall through
to candidate 2 or 3 and re-run.

- [ ] **Step 2: Register the model in `eval/models.ts`**

Read the current shape of `eval/models.ts` for an existing 7B/8B entry
(e.g. `mistral-7b-instruct-v0.3-q4ks`) and clone it. The exact entry
shape varies; the relevant fields are typically `id`, `repoId`,
`ggufFilePattern`, `displayName`, `family`, `size`, `quant`, and
maybe `ggufFileSize` for the deploy-time selection in Phase 6 Path B.

Add an entry like (adapt to actual schema):

```ts
{
  id: "mistral-7b-instruct-v0.3-q5km",
  repoId: "<resolved repo>",
  ggufFilePattern: "Q5_K_M",
  displayName: "Mistral-7B-Instruct-v0.3 (Q5_K_M, >4 GiB validation)",
  family: "mistral",
  size: "7B",
  quant: "Q5_K_M",
  fileSizeBytes: 5_140_000_000, // approximate; actual value populated post-fetch
},
```

- [ ] **Step 3: Add a smoke profile**

In `eval/smoke-profiles.ts`, register a profile mirroring the existing
`mistral-7b-instruct-v0.3-q4ks` entry but pinned to the new model at
warm temperature / no thinking gate.

- [ ] **Step 4: Run the 36-prompt sanity eval**

The benchmark fleet uses 36-prompt evals across 4 dimensions. Use the
existing `bench-profile` target:

```bash
make dashboard-serve &
sleep 2
make bench-profile PROFILES=mistral-7b-instruct-v0.3-q5km
```

Expected: 36/36 prompts run; coherent outputs (eval `overall` ≥ 60%
for an instruction-tuned 7B Q5_K_M; the canonical Q4_K_S landed at
68%, Q5_K_M should be at-or-better).

- [ ] **Step 5: Capture decode tok/s**

```bash
make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q5km PERF_RUNS=3
```

Expected: tok/s within the architecturally expected band. For Mistral-
7B Q5_K_M: extrapolating from Q4_K_S 35.0 tok/s and the Q4_K_S → Q5_K_M
quant-compute-cost ratio (Q5_K_M is ~25% larger and ~10-15% slower
in our matmul-bound regime), expect roughly 28-31 tok/s. **Hard gate:
≥18 tok/s.**

If 13B was registered instead, the architectural band is 18-22 tok/s
(extrapolating from 7B Q4_K_S 35.0 and 8B IQ3_M 27.2; 13B Q4_K_S is
roughly 1.7× the 7B param count at the same quant family). **Hard gate:
≥15 tok/s.**

- [ ] **Step 6: Author the validation report**

`eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md`:

```markdown
# MEMORY64 migration — Phase 7 >4 GiB validation

**Date:** <YYYY-MM-DD>
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md)
**Model:** <chosen-id> (~<file-size> GiB)
**Binary:** webllm-wasm-mem64.{js,wasm} (wasm64)

## Headline

- ✅ / ❌ Forward pass coherent on 36-prompt sanity eval (overall <X%>).
- ✅ / ❌ Decode tok/s within architectural band (<measured> vs gate <floor>).

## Eval result

| Dimension | Score |
|---|---:|
| tool-calling | <X/9> |
| reasoning | <X/9> |
| instruction-following | <X/9> |
| semantic-reasoning | <X/9> |
| **overall** | **<X/36>** |

## Speed result

3-run median decode: <X> tok/s.
Architectural band (extrapolated from in-fleet Mistral-7B Q4_K_S 35.0
tok/s): 28-31 tok/s. Hard gate: ≥18 tok/s.

## Working set

- Model file: <X> GiB.
- KV cache @ default seq=2048: ~<X> GiB.
- Activations + scratch: ~<X> GiB.
- Total: ~<X> GiB (vs 16 GiB toolchain ceiling).

## Lever closure

The MEMORY64 full migration plan closes with this commit. The wasm64
binary now ships in production; the canonical 6 maintain ±3% parity
(per Phase 5); a >4 GiB validation target (Mistral-7B Q5_K_M) loads,
runs the sanity eval coherently, and decodes within band.

Migration scope is complete for the ≤30B project ceiling. Next ask:
register a real 13B / 30B target if a deployment need surfaces (no
infrastructure work required — this is purely a model-registration
follow-up).

## Reproduction

```bash
make wasm-build
make smoke-test
make dashboard-serve &
make bench-profile PROFILES=mistral-7b-instruct-v0.3-q5km
make smoke-bench PERF_MODEL=mistral-7b-instruct-v0.3-q5km PERF_RUNS=3
```
```

- [ ] **Step 7: Refresh dashboard + run check**

```bash
make import-reports
make checkall
```

Expected:
- Dashboard refreshes with the new model row in Accuracy×Speed scatter
  and per-dimension grouped bars.
- `make checkall` clean.

- [ ] **Step 8: Update TODO.md to close the migration**

In `TODO.md`, in the "MEMORY64 full bridge migration" block (added
2026-04-28), prepend a closure line like the existing §17/§18/§19/§20
patterns:

```markdown
**Migration closed YYYY-MM-DD.** All 8 phases shipped (audit + 7
implementation phases). Canonical 6 parity holds (±3%); >4 GiB
validation on Mistral-7B Q5_K_M coherent at <eval%> / <X> tok/s.
Closure report at `eval/reports/memory64-migration-2026-04-28/
PHASE-7-VALIDATION.md`. Production wasm64 binary ships via `make
wasm-build`. The lever is closed for the ≤30B ceiling.
```

Move the entire migration block under "Completed on YYYY-MM-DD" or to
`TODO_ARCHIVE.md` per the established cadence.

- [ ] **Step 9: Commit Phase 7**

```bash
git add eval/models.ts eval/smoke-profiles.ts TODO.md
git add -f eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md
git commit -m "$(cat <<'EOF'
feat(eval): register mistral-7b-q5km — >4 GiB MEMORY64 validation

Phase 7 (final) of the MEMORY64 full migration. Register Mistral-7B-
Instruct-v0.3 Q5_K_M (~5.0 GiB) as the >4 GiB validation target;
exercises the wasm64 binary's actual >2^32 happy path.

Sanity eval: <X/36> coherent (gate: instruction-tuned 7B Q5_K_M
should hit ≥60%; canonical Q4_K_S baseline = 68%). Decode tok/s:
<X> (gate: ≥18 in arch band 28-31).

Migration closed: all 8 phases shipped (audit + 7 implementation
phases). Canonical 6 parity holds within ±3% (per Phase 5);
production wasm64 binary ships via make wasm-build (per Phase 6
<single|dual> path). Working set on the new model fits the
toolchain ceiling with margin.

Closure report:
eval/reports/memory64-migration-2026-04-28/PHASE-7-VALIDATION.md

The lever is closed for the ≤30B ceiling. Next ask: registration
of a real 13B / 30B target if a deployment need surfaces (no
infra work required).
EOF
)"
```

---

## Self-review checklist

After all 8 phase commits land, verify:

- [ ] **Spec coverage.** Each phase from TODO §MEMORY64 (Phase 0-7)
      maps to a Task in this plan. ✅
- [ ] **Probe-first default.** Phase 0 produces a punch list, not code. ✅
- [ ] **Always commit before work.** Task 0 commits the plan; each
      Task N commits its phase before the next phase starts. ✅
- [ ] **30B ceiling.** No infrastructure for >30B targets. Phase 7's
      validation target is at most 13B. ✅
- [ ] **Canonical-6 parity gate.** Phase 5 explicitly blocks Phase 6 if
      any model regresses ≥3%. ✅
- [ ] **Reversibility.** Each phase is a separate commit; revert via
      `git revert <sha>` undoes only that phase. ✅
- [ ] **No placeholders.** Every step has actual content (commands,
      code, expected output, file paths). ✅

## Out of scope

The following were intentionally deferred (out-of-scope per project
policies in CLAUDE.md):

- **Lifting the Emscripten 16 GiB linker cap.** Custom wasm-ld patch
  is multi-day + ongoing maintenance; defer until upstream Emscripten
  lifts it. Watch-list re-probe on each Emscripten upgrade.
- **30B targets beyond seq=2048.** Working set lands at the 16 GiB
  toolchain ceiling within margin of error. Out of scope unless a
  deployment ask forces it (mitigations: lower-bit quant or context-
  window cap).
- **70B+ targets.** Excluded by the 30B project ceiling.

## Risk register

(Carried from TODO §MEMORY64; pinned here for plan-execution reference.)

| Risk | Likelihood | Mitigation |
|---|---|---|
| WASM64 perf regression >3% on a canonical-6 model | Medium | Phase 5 gate catches; Phase 6 dual-binary fallback preserves wasm32 fast path. |
| BigInt-vs-Number TS callsite leak | Medium | Phase 0 punch list grounds Phase 1; Phase 1 commits run `make checkall`. tsc strictness catches type drift. |
| Hidden `int32_t` size in bridge param missed by Phase 0 | Low | Phase 7 sanity eval exercises 13B/Q5_K_M with >2 GiB single-buffer transfers; missed param shows up as truncation/garbled output. |
| GGUF loader BigInt boundary leak under heap-grow | Low | Phase 3 covers; pre-existing `uploadRangeChunked` fix is the precedent. |
| Bundle size doubles under dual-binary deploy | Low (cost-only) | Phase 6 decision criterion picks single-binary if perf delta ≤2%. |
| 13B target's coherence broken by an unseen quantization bug | Low | Phase 7 sanity eval catches; quantization correctness was tested at 7B/8B in §15/§16. |
| Toolchain ceiling tightens further on Emscripten upgrade | Low | Watch-list re-probe (cheap) catches at upgrade time; current 16 GiB is the absolute spec ceiling for `--max-memory`. |

## Probe artifacts (pinned references)

- `eval/reports/memory64-probe-2026-04-28/SUMMARY.md` — §31 parent probe
  (ASYNCIFY × MEMORY64 retired).
- `eval/reports/memory64-probe-2026-04-28/SUMMARY-31a.md` — §31a sub-probe
  (BigInt bridge + 15 GiB cap).
- `eval/reports/memory64-probe-2026-04-28/SUMMARY-31b.md` — §31b cap-bump
  probe (16 GiB toolchain ceiling).
- `eval/reports/pre-rebase-baselines-2026-04-28/SUMMARY.md` — wasm32
  reference for Phase 5.
- `docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md` —
  cap-probe spec (this plan supersedes its §6 "follow-up sub-probe"
  pointer with the 8 implementation phases).
